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
            LipSwish(),
            nn.Linear(config.sde_hidden_dim, config.lstm_hidden_dim * 2),
            nn.Tanh()
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, val=0)
            
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
        out = self.net(tz)
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
            LipSwish(),
            nn.Linear(config.sde_hidden_dim, config.lstm_hidden_dim * 2 * config.noise_dim),
            nn.Tanh()
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, val=0)

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
        out = self.net(tz)  
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
        self.linear = nn.Linear(config.lstm_hidden_dim * 2, config.output_dim)

        nn.init.normal_(self.linear.weight, mean=0, std=1.0/(config.lstm_hidden_dim * 2) **0.5)
        nn.init.constant_(self.linear.bias, 0)

        

    def forward(self, z):
        """
        Args:
            z : (B, sde_times, lstm_hidden_dim*2)
        Returns:
            x : (B, sde_times)
        """
        return self.linear(z).squeeze(-1)


# =============================================================================
# [5] SDEGenerator — full Neural SDE generator
# =============================================================================

class SDEGenerator(nn.Module):


    def __init__(self, config: DLQFRNNWithSDEConfig):
        super().__init__()
        self.config     = config
        self.lstm_hidden_dim= config.lstm_hidden_dim

        self.drift     = SDEDrift(config)
        self.diffusion = SDEDiffusion(config)
        self._sde      = _SDE(self.drift, self.diffusion, self.config)
        self.readout   = SDEReadout(config)


    def forward(self, z0):
       
        ts = self.config.sde_times                

        # torchsde.sdeint returns (sde_times, B, lstm_hidden_dim*2)
        z_path = torchsde.sdeint_adjoint(
            self._sde,
            y0     = z0,
            ts     = ts,
            dt = ts[1] - ts[0],
            method = self.config.sde_method,
            #adjoint_method = self.config.adjoint_method
        )                                                      # (sde_times, B, lstm_hidden_dim*2)

        z_path = z_path.permute(1, 0, 2)                      # (B, sde_times, lstm_hidden_dim*2)
        x_hat = self.readout(z_path)                     # (B, sde_times = M)

        return x_hat
