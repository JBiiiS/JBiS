import torch
import torch.nn as nn
import numpy as np

from dlqf_rnn_config import DLQFRNNConfig 

# ─────────────────────────────────────────
# 1. BiLSTM Encoder
#    3장 LSTM과의 차이:
#    - bidirectional=True → output이 hidden_dim * 2
#    - z 입력 없음 (단발성 예측이라 autoregressive 불필요)
#    - 마지막 시점 hidden만 반환
# ─────────────────────────────────────────
class BiLSTMEncoder(nn.Module):
    def __init__(self, config: DLQFRNNConfig):
        super().__init__()
        self.config = config

        self.bilstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers_lstm,
            batch_first=True,
            dropout=config.dropout if config.num_layers_lstm > 1 else 0.0,
            bidirectional=True,
            device=config.device     
        )

        # forget gate bias = 1.0 (alleviating vanishing gradient)
        for name, param in self.bilstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
                n = param.shape[0]
                param.data[n // 4: n // 2].fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x      : (B, T, input_dim)
        반환   : (B, hidden_dim * 2) — 마지막 시점 hidden state
        
        ★ 3장과의 차이:
        3장은 (B, T, hidden_dim) 전체 반환 → 각 시점마다 분위수 예측
        4장은 (B, hidden_dim*2) 마지막만 반환 → 내일 분포 하나만 예측
        """
        lstm_output, _ = self.bilstm(x)       # (B, T, hidden_dim * 2)
        h_last = lstm_output[:, -1, :]        # (B, hidden_dim * 2)
        # Because the bidirection is activated, h_t is composed of alternating h_t of each direction such as forward -> backward -> forward -> ..., so that h_t[-1] isn't consolidated form, but only h_t of final backward lstm, while h[:, -1, :] contains automatically concatenated shape([b, 2h])

        return h_last


# ─────────────────────────────────────────
# 2. NQF (Neural Quantile Function)
#    3장이랑 동일 — 단조증가 보장: weight 제곱
#    입력 차원만 hidden_dim*2로 변경
# ─────────────────────────────────────────
class NQF(nn.Module):
    def __init__(self, config: DLQFRNNConfig):
        super().__init__()

        # BiLSTM output이 hidden_dim * 2
        in_dim = config.hidden_dim * 2

        layers = [nn.Linear(in_dim + 1, config.latent_dim)]
        layers += [nn.Linear(config.latent_dim, config.latent_dim)
                   for _ in range(config.num_layers_nqf - 1)]
        layers += [nn.Linear(config.latent_dim, 1)]
        self.layers = nn.ModuleList(layers).to(config.device)

        self.activation = nn.Tanh()

    def forward(self, h: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        h     : (B, hidden_dim * 2)
        alpha : (B,)
        반환  : (B, 1)
        """
        x = torch.cat([h, alpha.unsqueeze(-1)], dim=-1)

        for i, layer in enumerate(self.layers):
            weight_sq = layer.weight ** 2   # 단조증가 보장
            x = torch.nn.functional.linear(x, weight_sq, layer.bias)
            if i < len(self.layers) - 1:
                x = self.activation(x)

        return x  # (B, 1)


# ─────────────────────────────────────────
# 3. DLQF 전체 모델
#    Encoder(BiLSTM) + Decoder(NQF)
# ─────────────────────────────────────────
class DLQFRNN(nn.Module):
    def __init__(self, config: DLQFRNNConfig):
        super().__init__()
        self.config = config

        self.encoder = BiLSTMEncoder(config).to(config.device)
        self.nqf = NQF(config).to(config.device)

    def forward(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        x     : (B, T, input_dim)
        alpha : (M,) — 분위수 레벨 [1/M, 2/M, ..., 1]
        반환  : (B, M) — 내일 |r|의 분위수 78개
        """
        h = self.encoder(x)     # (B, hidden_dim * 2)
        B = h.shape[0]
        M = alpha.shape[0]

        # h: (B, Dh) → (B*M, Dh)
        h_rep = h.unsqueeze(1).expand(B, M, -1).reshape(B * M, -1)
        # alpha: (M,) → (B*M,)
        a_rep = alpha.unsqueeze(0).expand(B, M).reshape(B * M)

        r_q = self.nqf(h_rep, a_rep)           # (B*M, 1)
        r_q = r_q.squeeze(-1).reshape(B, M)    # (B, M)
        return r_q

    def estimate_rv(self, x: torch.Tensor) -> torch.Tensor:
        """
        분위수 78개 뽑아서 RV 추정
        논문 4.2.3: RV_hat = Σ r²_{q,i} / scale_factor²

        x      : (B, T, input_dim)
        반환   : (B,)
        """
        M = self.config.total_quantile
        sf = self.config.scale_factor
        alpha = torch.linspace(1 / M, 1, M).to(x.device)

        r_q = self.forward(x, alpha)                   # (B, M)
        rv_hat = (r_q ** 2).sum(dim=1) / (sf ** 2)    # (B,)
        return rv_hat


# ─────────────────────────────────────────
# 4. L2 Distance Loss (Algorithm 2)
#    두 경험적 분포 사이의 L2 distance
#    = Wasserstein / Energy distance 계열
# ─────────────────────────────────────────
def l2_distance_loss(
    r_true: torch.Tensor,   # (B, M) — 실제 |r| 정렬값
    r_pred: torch.Tensor,   # (B, M) — 예측 분위수 정렬값
) -> torch.Tensor:
    """
    Algorithm 2 벡터화 구현.
    실제 M개 + 예측 M개를 합쳐 2M개로 정렬 후
    두 CDF 곡선 사이 넓이(L2 norm) 계산.
    """
    B, M = r_true.shape

    # 실제 + 예측 합쳐서 정렬
    combined = torch.cat([r_true, r_pred], dim=1)       # (B, 2M)
    combined_sorted, _ = combined.sort(dim=1)

    # 인접 구간 길이
    delta = combined_sorted[:, 1:] - combined_sorted[:, :-1]   # (B, 2M-1)

    # 각 구간 왼쪽 끝에서 두 CDF 값 계산
    x_mid = combined_sorted[:, :-1]                             # (B, 2M-1)
    r_true_s, _ = r_true.sort(dim=1)
    r_pred_s, _ = r_pred.sort(dim=1)

    # (B, 2M-1, M) 브로드캐스팅
    f_true = (r_true_s.unsqueeze(1) <= x_mid.unsqueeze(2)).float().sum(dim=2) / M
    f_pred = (r_pred_s.unsqueeze(1) <= x_mid.unsqueeze(2)).float().sum(dim=2) / M

    diff_sq = (f_true - f_pred) ** 2                    # (B, 2M-1)
    loss = (delta * diff_sq).sum(dim=1).sqrt()          # (B,)
    return loss.mean()


# ─────────────────────────────────────────
# 5. 학습 루프 (Algorithm 3)
# ─────────────────────────────────────────
def train_dlqf(
    model: DLQFRNN,
    X: torch.Tensor,    # (N, T, input_dim)
    Y: torch.Tensor,    # (N, M) — 정렬된 실제 |r| × scale_factor
    config: DLQFRNNConfig,
    device: str = 'cuda'
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    M = config.total_quantile
    alpha = torch.linspace(1 / M, 1, M).to(device)

    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True
    )

    # Algorithm 3: 초기화 안정성 체크
    print("초기화 조건 확인 중...")
    for attempt in range(10):
        for layer in model.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        model.eval()
        with torch.no_grad():
            x_s = X[:1].to(device)
            y_s = Y[:1].to(device)
            r_q = model(x_s, alpha)
            loss_val = l2_distance_loss(y_s, r_q).item()
            std_val = r_q.std().item()

        print(f"  시도 {attempt+1}: loss={loss_val:.4f}, std={std_val:.6f}")
        if loss_val <= 1e2 and std_val >= 1e-3:
            print("  ✅ 초기화 조건 통과\n")
            break

    # 본격 학습
    for epoch in range(config.n_epochs):
        model.train()
        epoch_loss = 0.0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            r_q = model(x_batch, alpha)
            loss = l2_distance_loss(y_batch, r_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{config.n_epochs}] | L2 Loss: {epoch_loss/len(loader):.6f}")

    return model


# ─────────────────────────────────────────
# 6. Sanity Check
# ─────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    cfg = DLQFRNNConfig()

    # 더미 데이터
    N = 400
    X = torch.randn(N, cfg.input_len, cfg.input_dim)
    Y = torch.abs(torch.randn(N, cfg.total_quantile) * 0.5)
    Y, _ = Y.sort(dim=1)   # 정렬 필수

    print(f"X: {X.shape} | Y: {Y.shape}\n")

    model = DLQFRNN(cfg)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    trained = train_dlqf(model, X, Y, cfg, device=device)

    # RV 추정
    trained.eval()
    with torch.no_grad():
        rv = trained.estimate_rv(X[:5].to(device))
    print(f"\n추정 RV (5개): {rv.tolist()}")
    print("\n✅ DLQF 완료!")