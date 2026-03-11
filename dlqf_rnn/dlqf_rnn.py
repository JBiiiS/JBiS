import torch
import torch.nn as nn
import numpy as np

from .dlqf_rnn_config import DLQFRNNConfig 

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
        # Because the bidirection is activated, h_t is composed of alternating h_t of each direction such as forward -> backward -> forward -> ..., so that h_t[-1] isn't consolidated form, but only h_t of final backward lstm, while h[:, -1, :] contains  concatenated shape([b, 2h]) automatically

        return h_last


# ─────────────────────────────────────────
# 2. NQF (Neural Quantile Function)
#    3장이랑 동일 — 단조증가 보장: weight 제곱
#    입력 차원만 hidden_dim*2로 변경
# ─────────────────────────────────────────
class NQF(nn.Module):
    def __init__(self, config: DLQFRNNConfig):
        super().__init__()

        self.config = config

        # BiLSTM output size =  hidden_dim * 2
        in_dim = config.hidden_dim * 2

        layers = [nn.Linear(in_dim + 1, config.latent_dim_1)]
        layers += [nn.Linear(config.latent_dim_1, config.latent_dim_2)]
        layers += [nn.Linear(config.latent_dim_2, config.latent_dim_3)]
        layers += [nn.Linear(config.latent_dim_3, 1)]
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
            weight_sq = layer.weight ** 2   # Monotonicity preserved throughout
            x = torch.nn.functional.linear(x, weight_sq, layer.bias)
            
            if i < len(self.layers) - 1:
                x = self.activation(x)  # Tanh for intermediate layers
            # No activation on final layer—output is log-space
        return x 


# ─────────────────────────────────────────
# 3. DLQF 전체 모델
#    Encoder(BiLSTM) + Decoder(NQF)
# ─────────────────────────────────────────
class DLQFRNN(nn.Module):
    def __init__(self, config: DLQFRNNConfig):
        super().__init__()
        self.config = config


        self.use_learnable_exp = config.use_learnable_exp
        self.use_non_learnable_exp = config.use_non_learnable_exp


        self.encoder = BiLSTMEncoder(config).to(config.device)
        self.nqf = NQF(config).to(config.device)

        
        self.correction_head = nn.Linear(config.hidden_dim * 2, 1)

        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

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
        h_detached = h.detach()

        gamma = self.correction_head(h_detached).squeeze(-1)


        return r_q, gamma

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

        r_q, gamma = self.forward(x, alpha)                   # (B, M)
        rv_base = (r_q ** 2).sum(dim=1) / (sf ** 2)    # (B,)
        rv_hat = rv_base + gamma

        return rv_hat, rv_base, gamma


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
# 5. MSE Loss (Mean Squared Error)
#    실현변동성의 제곱 오차
# ────────────────────────────────────────
def mse_loss(
    rv_true: torch.Tensor,   # (B,) — 실제 실현변동성
    rv_pred: torch.Tensor,   # (B,) — 예측 실현변동성
) -> torch.Tensor:
    """
    Mean Squared Error Loss for Realized Volatility
    
    rv_true : (B,) — real rv
    rv_pred : (B,) — pred rv
    return  : scalar — MSE loss
    """
    mse = torch.mean((rv_true - rv_pred) ** 2)

    return mse


# ─────────────────────────────────────────
# 6. QLIKE Loss (Quasi-Likelihood)
#    고변동성 과소예측에 더 큰 페널티
# ─────────────────────────────────────────
def qlike_loss(
    rv_true: torch.Tensor,   # (B,) — 실제 실현변동성
    rv_pred: torch.Tensor,   # (B,) — 예측 실현변동성
) -> torch.Tensor:
    """
    Quasi-Likelihood Loss (QLIKE)
    
    고변동성(RV_true가 클 때)이 과소예측(RV_pred < RV_true)될 때
    지수적으로 더 큰 페널티를 부여하는 손실함수.
    
    QLIKE = (1/T) * Σ [RV_t / RV_hat_t - log(RV_t / RV_hat_t) - 1]
    
    이는 RV_pred < RV_true인 경우 (under-prediction)에 매우 큰 값이 되므로,
    리스크 관리 관점에서 고변동성을 놓치지 않도록 강제함.
    
    rv_true : (B,) — 실제 실현변동성
    rv_pred : (B,) — 예측 실현변동성
    반환    : scalar — QLIKE loss
    """
    # 수치 안정성: 매우 작은 값 피하기
    epsilon = 1e-8
    rv_pred_safe = torch.clamp(rv_pred, min=epsilon)
    rv_true_safe = torch.clamp(rv_true, min=epsilon)
    
    # QLIKE = RV_t / RV_hat_t - log(RV_t / RV_hat_t) - 1
    ratio = rv_true_safe / rv_pred_safe
    qlike = torch.mean(ratio - torch.log(ratio) - 1.0)
    
    return qlike
