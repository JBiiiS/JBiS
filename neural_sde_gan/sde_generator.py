import torch
import torch.nn as nn
import torchsde # type: ignore
from neural_sde_gan.neural_sde_config import NeuralSDEConfig


# [0} Generating Noise
class NoiseEmbedder(nn.Module):
    def __init__(self, config:NeuralSDEConfig):
        super().__init__()


        self.net = nn.Sequential(
            nn.Linear(config.init_noise_dim, config.latent_dim)
        )

    def forward(self, initial_noise):
        z0 = self.net(initial_noise)
        return z0

#=============================================================================
# [1] Drift Network  μ_θ(t, z)
# =============================================================================

class SDEDrift(nn.Module):
    """
    Defines the deterministic part of the SDE:
        dz = μ_θ(t, z) dt  +  σ_θ(t, z) dW

    Time t is concatenated to z so the drift can depend explicitly on time
    (non-autonomous system), which is important for financial path dynamics.

    Input  : (t, z)  →  [scalar 0-dim tensor, (B, latent_dim)]
    Output : (B, latent_dim)
    """

    def __init__(self, config: NeuralSDEConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.latent_dim + 1, config.sde_hidden_dim),  # +1 for time
            nn.Softplus(),
            nn.Linear(config.sde_hidden_dim, config.sde_hidden_dim),
            nn.Softplus(),
            nn.Linear(config.sde_hidden_dim, config.latent_dim),
            nn.Tanh()
        )
        # Small weight init for numerical stability (same convention as ODEFunc)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, z):
        """
        Args:
            t : 0-dim scalar tensor — current time (passed by torchsde)
            z : (B, latent_dim)
        Returns:
            (B, latent_dim) — drift vector
        """
        # When a solver proceeds, the calculation is operated with discrete t node, not a full tensor. So, at each node, t is a 0-dim scalar and we can add each scalar into input; broadcast to (B, 1) via multiplication

        t_batch = torch.full((z.size(0), 1), float(t), device=z.device, dtype=z.dtype)
        tz = torch.cat([t_batch, z], dim=-1)   # (B, 1 + latent_dim)
        return self.net(tz)


# =============================================================================
# [2] Diffusion Network  σ_θ(t, z)
# =============================================================================

class SDEDiffusion(nn.Module):
    """
    Defines the stochastic part of the SDE.

    Using noise_type='general' in torchsde, so g() must return
    shape (B, latent_dim, noise_dim) — a matrix per sample.

    Low-rank structure (noise_dim << latent_dim) keeps the diffusion
    tractable and prevents the latent path from diverging.

    Input  : (t, z)  →  [scalar 0-dim tensor, (B, latent_dim)]
    Output : (B, latent_dim, noise_dim), calculating noise fitting to the noise dim automatically
    """

    def __init__(self, config: NeuralSDEConfig):
        super().__init__()
        self.latent_dim = config.latent_dim
        self.noise_dim  = config.noise_dim

        self.net = nn.Sequential(
            nn.Linear(config.latent_dim + 1, config.sde_hidden_dim),
            nn.Softplus(),
            nn.Linear(config.sde_hidden_dim, config.sde_hidden_dim),
            nn.Softplus(),
            nn.Linear(config.sde_hidden_dim, config.latent_dim * config.noise_dim),
            nn.Tanh()
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, z):
        """
        Args:
            t : 0-dim scalar tensor — current time (passed by torchsde)
            z : (B, latent_dim)
        Returns:
            (B, latent_dim, noise_dim)
        """
        t_batch = torch.full((z.size(0), 1), float(t), device=z.device, dtype=z.dtype)
        tz = torch.cat([t_batch, z], dim=-1)
        out = self.net(tz)                                              # (B, latent_dim * noise_dim)
        return out.view(z.size(0), self.latent_dim, self.noise_dim)    # (B, latent_dim, noise_dim)


# =============================================================================
# [3] Internal SDE wrapper for torchsde
# =============================================================================

class _SDE(nn.Module):
    """
    Thin wrapper consumed by torchsde.sdeint.

    torchsde requires an object with:
        .noise_type : 'general'  →  g returns (B, latent_dim, noise_dim)
        .sde_type   : 'ito' / 'stranovich'
        .f(t, y)    : drift,     t is a 0-dim scalar tensor when it is taken as input
        .g(t, y)    : diffusion, t is a 0-dim scalar tensor when it is taken as input
    """
    noise_type = NeuralSDEConfig.noise_type # Default: Diagonal
    # Presume the residual noise is independent to each other
    sde_type   = NeuralSDEConfig.sde_type # For Satisfying Martingale Property

    def __init__(self, drift: SDEDrift, diffusion: SDEDiffusion):
        super().__init__()
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

    Input  : (B, T_fine, latent_dim)
    Output : (B, T_fine, output_dim)
    """

    def __init__(self, config: NeuralSDEConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.latent_dim, config.sde_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.sde_hidden_dim, config.output_dim)
        )

    def forward(self, z):
        """
        Args:
            z : (B, T_fine, latent_dim)
        Returns:
            x : (B, T_fine, output_dim)
        """
        return self.net(z)


# =============================================================================
# [5] SDEGenerator — full Neural SDE generator
# =============================================================================

class SDEGenerator(nn.Module):
    """
    Full Neural SDE generator (Kidger et al., 2021).

        z0       ~ N(0, I)                              [pure noise init]
        z(t)      = sdeint(μ_θ, σ_θ, z0, ts)           [latent SDE path]
        x_hat(t)  = ζ_θ(z(t))                          [readout to return space]

    Returns two tensors (mirrors NeuralODEDecoder convention):
        x_hat_full    : (B, T_fine, output_dim)  — full fine-grained path
        x_hat_matched : (B, steps,  output_dim)  — subsampled at real-data timesteps

    The SDE integrates on the fine grid sde_times_full for numerical accuracy.
    Before passing to the discriminator, x_hat is subsampled to sde_times_matched
    so that real and fake paths share the same temporal resolution. Without this,
    the discriminator would trivially detect the difference in grid density
    (jagged vs. smooth spline artefact) instead of learning the true distribution.

    Reference:
        Kidger et al. (2021) "Neural SDEs as Infinite-Dimensional GANs"
        https://arxiv.org/abs/2102.03657
    """

    def __init__(self, config: NeuralSDEConfig):
        super().__init__()
        self.config     = config
        self.init_noise_dim = config.init_noise_dim
        self.latent_dim = config.latent_dim


        self.embedder = NoiseEmbedder(config)
        self.drift     = SDEDrift(config)
        self.diffusion = SDEDiffusion(config)
        self._sde      = _SDE(self.drift, self.diffusion)
        self.readout   = SDEReadout(config)

        # Indices into the fine grid that correspond to real-data timesteps.
        # Same construction as NeuralODEDecoder: every multiplier_of_times-th node.
        self.register_buffer(
            'real_indices',
            torch.arange(0, config.steps) * config.multiplier_of_times,  # (steps,)
        )

    def forward(self):
        """
        Generate a batch of synthetic log-return paths.
        Batch size is taken from config.batch_size.

        Returns:
            x_hat_full    : (B, T_fine, output_dim) — full fine-grained path
            x_hat_matched : (B, steps,  output_dim) — subsampled at real-data timesteps
                            → feed this to the discriminator alongside real data
        """
        ts         = self.config.sde_times_full                # (T_fine,)
        batch_size = self.config.batch_size

        # init_noise ~ N(0, I)  — pure noise
        init_noise = torch.randn(batch_size, self.init_noise_dim, device=ts.device)


        # z0 : (b, latent_dim)
        z0 = self.embedder(init_noise)



        # torchsde.sdeint returns (T_fine, B, latent_dim)
        z_path = torchsde.sdeint_adjoint(
            self._sde,
            y0     = z0,
            ts     = ts,
            method = self.config.sde_method,
        )                                                      # (T_fine, B, latent_dim)

        z_path = z_path.permute(1, 0, 2)                      # (B, T_fine, latent_dim)
        x_hat_full = self.readout(z_path)                     # (B, T_fine, output_dim)

        # Subsample to real-data timesteps so both sides of the discriminator
        # share the same temporal resolution.
        x_hat_matched = x_hat_full[:, self.real_indices, :]   # (B, steps, output_dim)

        return x_hat_full, x_hat_matched
