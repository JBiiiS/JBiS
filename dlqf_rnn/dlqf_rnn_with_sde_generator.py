import torch
import torch.nn as nn
import torchsde # type: ignore
import torch.nn.functional as F
from torchsde import BrownianInterval #type:ignore
from .dlqf_rnn_with_sde_config import DLQFRNNWithSDEConfig

class LipSwish(torch.nn.Module):
    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)



#=============================================================================
# [1] Drift Network  μ_θ(t, z)
# =============================================================================

class SDEDrift(nn.Module):
    """
    Defines the deterministic part of the SDE:
        dz = μ_θ(t, z) dt  +  σ_θ(t, z) dW

    * Modified to include Time-Gated Drift Dilation.
    * Removed nn.Tanh() to allow for explosive momentum at the tail.
    """

    def __init__(self, config: DLQFRNNWithSDEConfig):
        super().__init__()
        self.gamma_init = config.gamma_init
        self.alpha_init = config.alpha_init
        self.net = nn.Sequential(
            nn.Linear(config.lstm_hidden_dim * 2 + 1, config.sde_hidden_dim),  # +1 for time
            LipSwish(),
            nn.Linear(config.sde_hidden_dim, config.sde_hidden_dim),
            LipSwish()
        )

        self.linear = nn.Linear(config.sde_hidden_dim, config.lstm_hidden_dim * 2)

        # -------------------------------------------------------------------
        # [Time-Gated Dilation Parameters]
        # 꼬리 구간(t -> 1)에서 드리프트를 기하급수적으로 폭발시킬 제어 변수들.
        # 학습을 통해 모델이 스스로 증폭률과 곡률을 찾게 만듭니다.
        # -------------------------------------------------------------------
        self.raw_gamma = nn.Parameter(torch.tensor(0.5))  # 증폭률 (Scale)
        self.raw_alpha = nn.Parameter(torch.tensor(1.0))  # 곡률 (Power)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, val=0)
                
        nn.init.xavier_uniform_(self.linear.weight) 
        nn.init.constant_(self.linear.bias, val=0)
        
        # self.final_activation = nn.Tanh() <-- 치명적인 에러 원인. 완벽히 삭제합니다.
            
    def forward(self, t, z):
        """
        Args:
            t : 0-dim scalar tensor — current time (passed by torchsde)
            z : (B, lstm_hidden_dim * 2)
        Returns:
            (B, lstm_hidden_dim * 2) — drift vector
        """
        t_batch = torch.full((z.size(0), 1), float(t), device=z.device, dtype=z.dtype)
        tz = torch.cat([t_batch, z], dim=-1)   # (B, 1 + lstm_hidden_dim * 2)
        
        out_1 = self.net(tz)
        mu_base = self.linear(out_1)           # (B, lstm_hidden_dim * 2) - 기본 모멘텀

        # -------------------------------------------------------------------
        # [Apply Time-Gated Drift Dilation]
        # -------------------------------------------------------------------
        # parameter가 음수가 되지 않도록 Softplus로 양수 보장
        gamma = F.softplus(self.raw_gamma)
        alpha = F.softplus(self.raw_alpha)
        
        # Dilation Gate 수식: 1 + gamma * t^alpha
        # t는 0에서 1 사이의 값이므로, alpha가 클수록 t=1에 도달할 때만 문이 급격히 열립니다.
        dilation_gate = 1.0 + gamma * (t_batch ** alpha)
        
        # 기본 모멘텀에 팽창 게이트를 Element-wise 곱연산 (SwiGLU의 효과를 안전하게 구현)
        out = mu_base * dilation_gate
        
        return out
# =============================================================================
# [2] Diffusion Network  σ_θ(t, z)
# =============================================================================

class SDEDiffusion(nn.Module):
    """
    Defines the stochastic part of the SDE.

    Using noise_type='general' in torchsde, so g() must return
    shape (B, lstm_hidden_dim * 2, noise_dim) — a matrix per sample.

    tractable and prevents the lstm_hidden_dim * 2 path from diverging.

    Input  : (t, z)  →  [scalar 0-dim tensor, (B, lstm_hidden_dim * 2)]
    Output : (B, lstm_hidden_dim * 2, noise_dim), calculating noise fitting to the noise dim automatically


    """

    def __init__(self, config: DLQFRNNWithSDEConfig):
        super().__init__()
        self.lstm_hidden_dim = config.lstm_hidden_dim
        self.noise_dim = config.noise_dim

        self.net = nn.Sequential(
            nn.Linear(config.lstm_hidden_dim * 2 + 1, config.sde_hidden_dim),
            LipSwish(),
            nn.Linear(config.sde_hidden_dim, config.sde_hidden_dim),
            LipSwish()
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, val=0)

        self.linear =  nn.Linear(config.sde_hidden_dim, config.lstm_hidden_dim * 2 * config.noise_dim)
        self.final_activation = nn.Sigmoid() 

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, t, z):
        """
        Args:
            t : 0-dim scalar tensor — current time (passed by torchsde)
            z : (B, lstm_hidden_dim * 2)
        Returns:
            (B, lstm_hidden_dim * 2, noise_dim)

        """
        t_batch = torch.full((z.size(0), 1), float(t), device=z.device, dtype=z.dtype)
        tz = torch.cat([t_batch, z], dim=-1)
        out_1 = self.net(tz) 
        out_2 = self.linear(out_1)
        out = self.final_activation(out_2)

        out = out.view(z.size(0), self.lstm_hidden_dim * 2, self.noise_dim)                                             # (B, lstm_hidden_dim * 2 * noise_dim)
        return out


# =============================================================================
# [3] Internal SDE wrapper for torchsde
# =============================================================================

class _SDE(nn.Module):
    """
    Thin wrapper consumed by torchsde.sdeint.

    torchsde requires an object with:
        .noise_type : 'general'  →  g returns (B, lstm_hidden_dim * 2, noise_dim)
        .f(t, y)    : drift,     t is a 0-dim scalar tensor when it is taken as input
        .g(t, y)    : diffusion, t is a 0-dim scalar tensor when it is taken as input
    """

    
    def __init__(self, drift: SDEDrift, diffusion: SDEDiffusion, config: DLQFRNNWithSDEConfig):

        super().__init__()
        self.config = config

        self.noise_type = config.noise_type
        self.sde_type = config.sde_type

        self.drift     = drift
        self.diffusion = diffusion

    def f(self, t, y):
        return self.drift(t, y)

    def g(self, t, y):
        return self.diffusion(t, y)
    


# =============================================================================
# [4] Readout  ζ_θ(z) : latent → observation space
# =============================================================================

class SDEReadout(nn.Module):
    """
    Maps the latent SDE trajectory z(t) to the observable return space x(t).
    Applied point-wise at every time step.

    Architecture mirrors NeuralODEDecoder in neural_ode_back.py.

    Input  : (B, sde_times, lstm_hidden_dim * 2)
    Output : (B, sde_times, output_dim)
    """
    def __init__(self, config: DLQFRNNWithSDEConfig):
        super().__init__()
        self.use_learnable_exp = config.use_learnable_exp
        self.use_non_learnable_exp = config.use_non_learnable_exp
        self.exp_deno = config.exp_deno_init

        if self.use_learnable_exp:
            self.log_deno = nn.Parameter(torch.tensor(float(config.exp_deno_init)).log())

        self.net = nn.Sequential(
            nn.Linear(config.lstm_hidden_dim * 2, config.sde_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.sde_hidden_dim, config.output_dim),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, val=0.0)

    def forward(self, z):
        raw_out = self.net(z).squeeze(-1)

        if self.use_learnable_exp:
            pos_out = torch.exp(raw_out) / self.log_deno.exp()
        elif self.use_non_learnable_exp:
            pos_out = torch.exp(raw_out) / self.exp_deno
        else:
            pos_out = F.softplus(raw_out)

        out = torch.cumsum(pos_out, dim=1)
        return out


# =============================================================================
# [5] SDEGenerator — full Neural SDE generator
# =============================================================================


class SDEGenerator(nn.Module):
    def __init__(self, config: DLQFRNNWithSDEConfig):
        super().__init__()
        self.config = config
        self.drift     = SDEDrift(config)
        self.diffusion = SDEDiffusion(config)
        self._sde      = _SDE(self.drift, self.diffusion, self.config)
        self.readout   = SDEReadout(config)  

    def forward(self, z0):
        ts = self.config.sde_times

        z_path = torchsde.sdeint(
            self._sde, y0=z0, ts=ts,
            dt=ts[1] - ts[0],
            method=self.config.sde_method,
        ) 
        z_path = z_path.permute(1, 0, 2)

        x_hat = self.readout(z_path)

        return x_hat
    

'''

# 0305 0128 new version
#=============================================================================
# [4] Readout  ζ_θ(z) : latent → observation space
# =============================================================================

class SDEReadout(nn.Module):
    """
    Maps the latent trajectory to the observable return space x(t).
    
    * Modified to accept a dynamic `in_dim` so it can be reused for both 
      the stochastic residual (z_path) and the deterministic base (z0 + t).
    """
    def __init__(self, config: DLQFRNNWithSDEConfig, in_dim: int = None):
        super().__init__()
        
        # 입력 차원이 명시되지 않으면 기본값(lstm_hidden_dim * 2)을 사용
        if in_dim is None:
            in_dim = config.lstm_hidden_dim * 2
            
        self.net = nn.Sequential(
            nn.Linear(in_dim, config.sde_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.sde_hidden_dim, config.output_dim),
            nn.Softplus()
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, val=0.0)

    def forward(self, z):
        return self.net(z).squeeze(-1)


# =============================================================================
# [5] SDEGenerator — full Neural SDE generator (with Temporal Skip Connection)
# =============================================================================

class SDEGenerator(nn.Module):
    def __init__(self, config: DLQFRNNWithSDEConfig):
        super().__init__()
        self.config = config
        self.drift     = SDEDrift(config)
        self.diffusion = SDEDiffusion(config)
        self._sde      = _SDE(self.drift, self.diffusion, self.config)
        
        # 1. 잔차용 Readout (입력: SDE 잠재 상태)
        self.readout   = SDEReadout(config, in_dim=config.lstm_hidden_dim * 2)
        
        # 2. 베이스라인용 Readout (입력: LSTM 잠재 상태 + 시간 차원 1개 추가)
        self.base_readout = SDEReadout(config, in_dim=config.lstm_hidden_dim * 2 + 1)  

    def forward(self, z0):
        ts = self.config.sde_times

        # --- 1. 확률적 잔차 경로 (Stochastic Residual Pathway) ---
        z_path = torchsde.sdeint(
            self._sde, y0=z0, ts=ts,
            dt=ts[1] - ts[0],
            method=self.config.sde_method,
        )
        z_path = z_path.permute(1, 0, 2)  # (B, M, lstm_hidden_dim*2)
        residual = self.readout(z_path)   # (B, M)

        # --- 2. 결정론적 베이스라인 경로 (Deterministic Base Pathway) ---
        B, M = z0.size(0), ts.size(0)
        
        # z0를 모든 시간 스텝 M에 대해 확장: (B, M, lstm_hidden_dim*2)
        z0_expanded = z0.unsqueeze(1).expand(-1, M, -1)
        
        # ts 텐서를 (B, M, 1) 형태로 확장 (디바이스 일치 필수)
        t_expanded = ts.view(1, M, 1).expand(B, -1, -1).to(z0.device)
        
        # z0와 t를 마지막 차원을 기준으로 결합: (B, M, lstm_hidden_dim*2 + 1)
        base_input = torch.cat([z0_expanded, t_expanded], dim=-1)
        
        # 시간에 따라 형태가 변하는 베이스라인 곡선 생성
        base = self.base_readout(base_input)  # (B, M)

        # --- 3. 최종 산출물 병합 ---
        x_hat = base + residual

        return x_hat






'''

