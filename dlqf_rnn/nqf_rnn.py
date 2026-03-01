import torch
import torch.nn as nn
import numpy as np
from nqf_rnn_config import NQFRNNConfig
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────
# 1. Neural Quantile Function (NQF)
#    - 단조증가 보장: weight를 제곱해서 항상 양수
#    - 입력: h_t (LSTM hidden state) + α (분위수 레벨)
#    - 출력: α 분위수에 해당하는 값
# ─────────────────────────────────────────
class NQF(nn.Module):
    def __init__(self, config: NQFRNNConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.latent_dim = config.latent_dim 
        self.num_layers = config.num_layers_nqf

        # (b, l + 1) -> (b, h) -> (b, 1)
        layers = [nn.Linear(self.hidden_dim + 1, self.latent_dim)]

        layers += [nn.Linear(self.latent_dim, self.latent_dim)
                    for _ in range(self.num_layers - 1)]

        layers += [nn.Linear(self.latent_dim, 1)]

        self.layers = nn.ModuleList(layers)
                                            
        self.activation = nn.Tanh()

    def forward(self, h: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        h     : (..., latent_dim)
        alpha : (...,) — 0~1 사이 확률값
        반환  : (..., 1) — 해당 분위수 값
        """
        alpha = alpha.unsqueeze(-1)           # (..., 1)
        x = torch.cat([h, alpha], dim=-1)     # (..., hidden_dim + 1)

        for i, layer in enumerate(self.layers):
            # weight^2 -> monotonisity
            weight_sq = layer.weight ** 2
            x = torch.nn.functional.linear(x, weight_sq, layer.bias)
            if i < len(self.layers) - 1:
                x = self.activation(x)

        return x  # (..., 1)


class NQFLSTM(nn.Module):
    def __init__(self, config: NQFRNNConfig):
        super().__init__()
        self.config = config

        self.input_dim = config.input_dim

        self.hidden_dim = config.hidden_dim
        self.latent_dim = config.latent_dim 

        self.num_layers_lstm = config.num_layers_lstm
        self.num_layers_nqf = config.num_layers_nqf

        self.dropout = config.dropout


        self.lstm = nn.LSTM(
            input_size=self.input_dim + 1,   # 공변량 + 이전 관측값(z_{t-1})
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers_lstm,
            batch_first=True,
            dropout=self.dropout
        )

        # forget gate bias = 1.0 (3.1.1 — alleviating vanishing gradient)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
                # forget gate: hidden_dim ~ 2*hidden_dim 구간
                n = param.shape[0]
                param.data[n // 4: n // 2].fill_(1.0)

    def forward(self, z: torch.Tensor, x: torch.Tensor):
        """
        z : (B, T) — 스케일된 관측값
        x : (B, T, D) — 공변량
        반환: h (B, T, hidden_dim)
        """
        # z를 한 스텝 뒤로 밀어서 입력으로 사용 (autoregressive)
        z_input = torch.roll(z, 1, dims=1)
        z_input[:, 0] = 0.0              # 첫 스텝은 0으로 초기화 (z_{i,0} = 0)
        lstm_input = torch.cat([z_input.unsqueeze(-1), x], dim=-1)  # (B, T, D+1)
        h, _ = self.lstm(lstm_input)     # (B, T, hidden_dim)
        
        return h        


# ─────────────────────────────────────────
# 2. NQF-RNN 전체 모델
#    - Encoder: LSTM
#    - Decoder: NQF
# ─────────────────────────────────────────
class NQFRNN(nn.Module):
    def __init__(self, config: NQFRNNConfig):
        super().__init__()

        self.config = config
        self.nqf = NQF(config)
        self.encode = NQFLSTM(config)

   

    def forward(self, z: torch.Tensor, x: torch.Tensor, alpha: torch.Tensor):
        """
        z     : (B, T)
        x     : (B, T, D)
        alpha : (M,) — 분위수 레벨들
        반환  : (M, B, T0) — 각 분위수별 예측값
        """
        h = self.encode(z, x)            # (B, T, hidden_dim)
        t0 = z.shape[1] // 2            # conditioning length (간단히 절반으로)
        h_pred = h[:, t0:, :]           # (B, T0, hidden_dim)

        B, T0, Dh = h_pred.shape
        M = alpha.shape[0]

        # 텐서 확장 — 논문 Algorithm 1 line 19~25
        h_rep = h_pred.unsqueeze(0).expand(M, B, T0, Dh)          # (M, B, T0, Dh)
        h_rep = h_rep.reshape(M * B, T0, Dh)                       # (M*B, T0, Dh)
        alpha_rep = alpha.view(M, 1, 1).expand(M, B, T0)           # (M, B, T0)
        alpha_rep = alpha_rep.reshape(M * B, T0)                    # (M*B, T0)

        q = self.nqf(h_rep, alpha_rep)  # (M*B, T0, 1)
        q = q.squeeze(-1).reshape(M, B, T0)  # (M, B, T0)
        return q


# ─────────────────────────────────────────
# 3. CRPS Loss (사다리꼴 근사법)
#    - 논문 Algorithm 1 line 27~28
# ─────────────────────────────────────────
def pinball_loss(y: torch.Tensor, q: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """
    y     : (B, T0) — 실제 관측값
    q     : (M, B, T0) — 예측 분위수
    alpha : (M,) — 분위수 레벨
    반환  : scalar
    """
    alpha = alpha.view(-1, 1, 1)                   # (M, 1, 1)
    y_exp = y.unsqueeze(0).expand_as(q)            # (M, B, T0)
    error = y_exp - q
    loss = torch.where(error >= 0, alpha * error, (alpha - 1) * error)
    return loss  # (M, B, T0)


def crps_loss(y: torch.Tensor, q: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """
    사다리꼴 근사법으로 CRPS 계산 (논문 Eq. 3.19)
    y     : (B, T0)
    q     : (M, B, T0)
    alpha : (M,)
    """
    pb = pinball_loss(y, q, alpha)  # (M, B, T0)

    # 사다리꼴 규칙: (pb[m] + pb[m+1]) / 2 * (1/M)
    trap = 0.5 * (pb[:-1] + pb[1:])  # (M-1, B, T0)
    loss = trap.mean() * 2            # ×2는 CRPS 정의의 상수
    return loss


# ─────────────────────────────────────────
# 4. 스케일링 유틸 (논문 Algorithm 1 line 4~6)
# ─────────────────────────────────────────
def compute_scale(y: torch.Tensor, t0: int) -> torch.Tensor:
    """v_i = 1 + mean(y_{1:t0})"""
    return 1 + y[:, :t0].mean(dim=1, keepdim=True)  # (B, 1)


# ─────────────────────────────────────────
# 5. 학습 루프
# ─────────────────────────────────────────
def train_nqf_rnn(
    model: NQFRNN,
    y: torch.Tensor,      # (N, T) — 원시 관측값
    x: torch.Tensor,      # (N, T, D) — 공변량
    t0: int,              # conditioning length
    M: int = 100,         # 분위수 분할 수
    n_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 분위수 레벨 설정 (0 ~ M-1)/M
    alpha = torch.linspace(0, 1, M + 1)[:-1].to(device)  # (M,)

    dataset = TensorDataset(y, x)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        for y_batch, x_batch in loader:
            y_batch = y_batch.to(device)
            x_batch = x_batch.to(device)

            # 스케일링
            v = compute_scale(y_batch, t0)          # (B, 1)
            z_batch = y_batch / v                   # (B, T)

            # forward
            q = model(z_batch, x_batch, alpha)      # (M, B, T0)
            z_target = z_batch[:, t0:]              # (B, T0)

            # CRPS loss
            loss = crps_loss(z_target, q, alpha)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg = epoch_loss / len(loader)
            print(f"Epoch [{epoch+1}/{n_epochs}] | CRPS Loss: {avg:.6f}")

    return model


# ─────────────────────────────────────────
# 6. 간단한 동작 확인 (Sanity Check)
# ─────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 더미 데이터 생성
    N, T, D = 200, 40, 3   # 샘플 수, 시계열 길이, 공변량 차원
    t0 = 20                 # conditioning length

    y_dummy = torch.randn(N, T) * 0.5 + torch.sin(
        torch.linspace(0, 4 * np.pi, T).unsqueeze(0).expand(N, -1)
    )
    x_dummy = torch.randn(N, T, D)

    # 모델 초기화
    model = NQFRNN(
        input_dim=D,
        hidden_dim=64,
        n_lstm_layers=2,
        n_nqf_layers=2,
        nqf_inner_dim=64,
        dropout=0.1,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 학습
    trained_model = train_nqf_rnn(
        model=model,
        y=y_dummy,
        x=x_dummy,
        t0=t0,
        M=100,
        n_epochs=30,
        batch_size=32,
        lr=1e-3,
        device=device,
    )

    # 예측 (inference)
    trained_model.eval()
    with torch.no_grad():
        y_test = y_dummy[:5].to(device)
        x_test = x_dummy[:5].to(device)

        v = compute_scale(y_test, t0)
        z_test = y_test / v

        alpha_test = torch.tensor([0.1, 0.5, 0.9]).to(device)
        q_pred = trained_model(z_test, x_test, alpha_test)  # (3, 5, T0)
        q_pred_rescaled = q_pred * v.squeeze(-1).unsqueeze(0)  # 역스케일링

    print(f"\n예측 분위수 shape: {q_pred_rescaled.shape}")
    print(f"10th percentile 평균: {q_pred_rescaled[0].mean():.4f}")
    print(f"50th percentile 평균: {q_pred_rescaled[1].mean():.4f}")
    print(f"90th percentile 평균: {q_pred_rescaled[2].mean():.4f}")
    print("\n✅ NQF-RNN 구현 완료!")
