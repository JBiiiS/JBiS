import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from .dlqf_rnn_with_ode_config import DLQFRNNWithODEConfig

class LipSwish(nn.Module):
    def forward(self, x):
        return 0.909 * F.silu(x)


# =============================================================================
# [1] ODE Function  f_θ(t, z)
# =============================================================================

class ODEFunc(nn.Module):
    def __init__(self, config: DLQFRNNWithODEConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.lstm_hidden_dim * 2 + 1, config.ode_hidden_dim),
            LipSwish(),
            nn.Linear(config.ode_hidden_dim, config.ode_hidden_dim),
            LipSwish()
        )
        self.linear = nn.Linear(config.ode_hidden_dim, config.lstm_hidden_dim * 2)

        self.raw_gamma = nn.Parameter(torch.tensor(config.gamma_init))
        self.raw_alpha = nn.Parameter(torch.tensor(config.alpha_init))

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, val=0)

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, val=0)

    def forward(self, t, z):
        t_batch = torch.full((z.size(0), 1), float(t), device=z.device, dtype=z.dtype)
        tz = torch.cat([t_batch, z], dim=-1)

        out_1 = self.net(tz)
        mu_base = self.linear(out_1)

        gamma = F.softplus(self.raw_gamma)
        alpha = F.softplus(self.raw_alpha)
        dilation_gate = 1.0 + gamma * (t_batch ** alpha)

        return mu_base * dilation_gate


# =============================================================================
# [2] Readout  ζ_θ(z) : latent → observation space
# =============================================================================

class ODEReadout(nn.Module):
    def __init__(self, config: DLQFRNNWithODEConfig):
        super().__init__()
        self.use_learnable_exp = config.use_learnable_exp
        self.use_non_learnable_exp = config.use_non_learnable_exp
        self.exp_deno = config.exp_deno_init

        if self.use_learnable_exp:
            self.log_deno = nn.Parameter(torch.tensor(float(config.exp_deno_init)).log())

        self.net = nn.Sequential(
            nn.Linear(config.lstm_hidden_dim * 2, config.ode_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.ode_hidden_dim, config.output_dim),
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
# [3] ODEGenerator — full Neural ODE generator
# =============================================================================

class ODEGenerator(nn.Module):
    def __init__(self, config: DLQFRNNWithODEConfig):
        super().__init__()
        self.config = config
        self.ode_func = ODEFunc(config)
        self.readout = ODEReadout(config)

    def forward(self, z0):
        ts = self.config.ode_times

        z_path = odeint(
            self.ode_func,
            z0,
            ts,
            method='dopri5',
        )  # (ode_times, B, lstm_hidden_dim*2)

        z_path = z_path.permute(1, 0, 2)  # (B, ode_times, lstm_hidden_dim*2)
        x_hat = self.readout(z_path)       # (B, ode_times)

        return x_hat