import torch
import torch.nn as nn
import torch.nn.utils as weight_norm_utils
import torchsde
from torchsde import BrownianInterval
from neural_sde_gan.neural_sde_config import NeuralSDEConfig

# =============================================================================
# [0] Kidger's LipSwish Activation
# =============================================================================
class LipSwish(nn.Module):
    """
    f(x) = 0.909 * x * sigmoid(x)
    Used by Kidger et al. to maintain Lipschitz continuity while avoiding vanishing gradients.
    """
    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)

# =============================================================================
# [1] Generating Noise
# =============================================================================
class NoiseEmbedder(nn.Module):
    def __init__(self, config: NeuralSDEConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.init_noise_dim, config.latent_dim)
            # Kidger does not use an activation here; it is a linear projection of the initial Gaussian.
        )

    def forward(self, initial_noise):
        return self.net(initial_noise)

# =============================================================================
# [2] Drift Network  μ_θ(t, z)
# =============================================================================
class SDEDrift(nn.Module):
    def __init__(self, config: NeuralSDEConfig):
        super().__init__()
        # Apply Weight Normalization and LipSwish, exactly as in the paper
        self.net = nn.Sequential(
            weight_norm_utils.weight_norm(nn.Linear(config.latent_dim + 1, config.sde_hidden_dim)),
            LipSwish(),
            weight_norm_utils.weight_norm(nn.Linear(config.sde_hidden_dim, config.sde_hidden_dim)),
            LipSwish(),
            weight_norm_utils.weight_norm(nn.Linear(config.sde_hidden_dim, config.latent_dim))
        )

    def forward(self, t, z):
        t_batch = torch.full((z.size(0), 1), float(t), device=z.device, dtype=z.dtype)
        tz = torch.cat([t_batch, z], dim=-1)
        # Removed explicit residual connection and Tanh
        return self.net(tz)

# =============================================================================
# [3] Diffusion Network  σ_θ(t, z)
# =============================================================================
class SDEDiffusion(nn.Module):
    def __init__(self, config: NeuralSDEConfig):
        super().__init__()
        self.latent_dim = config.latent_dim
        self.noise_dim  = config.noise_dim

        self.net = nn.Sequential(
            weight_norm_utils.weight_norm(nn.Linear(config.latent_dim + 1, config.sde_hidden_dim)),
            LipSwish(),
            weight_norm_utils.weight_norm(nn.Linear(config.sde_hidden_dim, config.sde_hidden_dim)),
            LipSwish(),
            weight_norm_utils.weight_norm(nn.Linear(config.sde_hidden_dim, config.latent_dim * config.noise_dim))
        )

    def forward(self, t, z):
        t_batch = torch.full((z.size(0), 1), float(t), device=z.device, dtype=z.dtype)
        tz = torch.cat([t_batch, z], dim=-1)
        out = self.net(tz)
        # Removed explicit residual connection and Tanh
        return out.view(z.size(0), self.latent_dim, self.noise_dim)

# =============================================================================
# [4] Internal SDE wrapper for torchsde
# =============================================================================
class _SDE(nn.Module):
    noise_type = 'general'
    # Paper explicitly requires stratonovich for the reversible_heun solver
    sde_type   = 'stratonovich' 

    def __init__(self, drift: SDEDrift, diffusion: SDEDiffusion):
        super().__init__()
        self.drift     = drift
        self.diffusion = diffusion

    def f(self, t, y):
        return self.drift(t, y)

    def g(self, t, y):
        return self.diffusion(t, y)

# =============================================================================
# [5] Readout  ζ_θ(z) : latent → observation space
# =============================================================================
class SDEReadout(nn.Module):
    def __init__(self, config: NeuralSDEConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.latent_dim, config.sde_hidden_dim),
            LipSwish(),
            nn.Linear(config.sde_hidden_dim, config.output_dim)
        )

    def forward(self, z):
        return self.net(z)

# =============================================================================
# [6] SDEGenerator 
# =============================================================================
class SDEGenerator(nn.Module):
    # ... (__init__ remains the same) ...
    def __init__(self, config: NeuralSDEConfig):
        super().__init__()
        self.config         = config
        self.init_noise_dim = config.init_noise_dim
        self.latent_dim     = config.latent_dim

        self.embedder  = NoiseEmbedder(config)
        self.drift     = SDEDrift(config)
        self.diffusion = SDEDiffusion(config)
        self._sde      = _SDE(self.drift, self.diffusion)
        self.readout   = SDEReadout(config)

        self.register_buffer('real_indices', torch.arange(0, config.steps) * config.multiplier_of_times)

    def forward(self):
        ts = self.config.sde_times_full
        batch_size = self.config.batch_size

        init_noise = torch.randn(batch_size, self.init_noise_dim, device=ts.device)
        z0 = self.embedder(init_noise)

        # BrownianInterval is required for reversible solvers to map exact noise paths
        bm = BrownianInterval(
            t0=ts[0], t1=ts[-1], size=(batch_size, self.config.noise_dim),
            device=self.config.device, levy_area_approximation='none'
        )

        # Kidger's specific solver configuration: reversible_heun
        z_path = torchsde.sdeint_adjoint(
            self._sde,
            y0=z0,
            ts=ts,
            bm=bm,
            method='reversible_heun',
            adjoint_method='adjoint_reversible_heun'
        )

        z_path = z_path.permute(1, 0, 2)
        x_hat_full = self.readout(z_path)
        x_hat_matched = x_hat_full[:, self.real_indices, :]

        return x_hat_full, x_hat_matched