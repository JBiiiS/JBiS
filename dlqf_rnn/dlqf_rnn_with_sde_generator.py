import torch
import torch.nn as nn
import torchsde # type: ignore
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

    Time t is concatenated to z so the drift can depend explicitly on time
    (non-autonomous system), which is important for financial path dynamics.

    Input  : (t, z)  →  [scalar 0-dim tensor, (B, lstm_hidden_dim*2]
    Output : (B, lstm_hidden_dim*2)
    """

    def __init__(self, config: DLQFRNNWithSDEConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.lstm_hidden_dim * 2 + 1, config.sde_hidden_dim),  # +1 for time
            LipSwish(),
            nn.Linear(config.sde_hidden_dim, config.sde_hidden_dim),
            LipSwish()
        )

        self.linear = nn.Linear(config.sde_hidden_dim, config.lstm_hidden_dim * 2)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, val=0)
                
        nn.init.xavier_uniform_(self.linear.weight) 
        nn.init.constant_(self.linear.bias, val=0)
            
    def forward(self, t, z):
        """
        Args:
            t : 0-dim scalar tensor — current time (passed by torchsde)
            z : (B, lstm_hidden_dim * 2)
        Returns:
            (B, lstm_hidden_dim * 2) — drift vector
            
        """
        # When a solver proceeds, the calculation is operated with discrete t node, not a full tensor. So, at each node, t is a 0-dim scalar and we can add each scalar into input; broadcast to (B, 1) via multiplication

        t_batch = torch.full((z.size(0), 1), float(t), device=z.device, dtype=z.dtype)
        tz = torch.cat([t_batch, z], dim=-1)   # (B, 1 + lstm_hiddem_dim * 2)
        out_1 = self.net(tz)
        out = self.linear(out_1)
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
        self.final_activation = nn.Softplus() 

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
        self.softplus_deno = config.softplus_deno
        self.net = nn.Sequential(
            nn.Linear(config.lstm_hidden_dim * 2, config.sde_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.sde_hidden_dim, config.output_dim),
            nn.Softplus()
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, val=0.0)
        

    def forward(self, z):
        """
        Args:
            z : (B, sde_times, lstm_hidden_dim*2)
        Returns:
            x : (B, sde_times)
        """

        raw_out = self.net(z).squeeze(-1)
        out = torch.cumsum(raw_out, dim=1) / self.softplus_deno

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

