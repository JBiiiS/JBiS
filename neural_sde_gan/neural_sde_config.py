import torch
from dataclasses import dataclass
from base.base_config import BaseConfig


@dataclass
class NeuralSDEConfig(BaseConfig):
    # ----------------------------------
    # [4] Generator: Neural SDE Settings
    # ----------------------------------
    # How many nodes to augment physical time
    multiplier_of_times : int = 5

    # [Noise Embedding State]
    init_noise_dim: int = 40

    # [Latent SDE State]
    # Dimension of z_t, the hidden state of the SDE.
    # The SDE evolves this latent vector over time.
    latent_dim: int = 96

    # [Brownian Motion]
    # Number of independent Brownian motion channels driving the SDE.
    # noise_dim << latent_dim is typical (low-rank diffusion).
    noise_dim: int = 4

    # [Drift / Diffusion Networks]
    # Hidden layer width for μ_θ(t, z) and σ_θ(t, z).
    sde_hidden_dim: int = 64

    # [Readout / Decoder]
    # Maps latent state z_t → observation x_t.
    # Must match the number of assets in the real data.
    # Set equal to num_assets if modeling full return vector,
    # or a smaller value if modeling a single asset at a time.
    output_dim: int = 100       # default matches BaseConfig.num_assets

    # ----------------------------------
    # [5] Discriminator: Neural CDE Settings
    # ----------------------------------

    # Hidden state dimension of the CDE: h_t ∈ R^{cde_hidden_dim}.
    # The CDE vector field maps (t, h_t) → R^{cde_hidden_dim × output_dim}.
    cde_hidden_dim: int = 128

    # ----------------------------------
    # [6] SDE Solver Settings
    # ----------------------------------
    noise_type: str = 'general'
    sde_type: str = 'stratonovich'
    # Numerical integration method passed to torchsde.sdeint.
    # 'euler'     : Euler-Maruyama  (fast, first-order, good for training)
    # 'milstein'  : Milstein        (second-order, slightly more accurate)
    # 'srk'       : Stochastic RK   (higher-order, slower)
    sde_method: str = 'midpoint'

    # ----------------------------------
    # [7] WGAN-GP Training Settings
    # ----------------------------------

    # Number of discriminator (critic) update steps per generator step.
    # Standard WGAN-GP uses 5 for D.
    epoch_for_D: int = 5
    epoch_for_G: int = 1

    # Gradient penalty coefficient λ in:
    # L_D = E[D(fake)] - E[D(real)] + gp_lambda * GP
    gp_lambda: float = 10.0

    def __post_init__(self):
        # Inherit dt = T / steps from BaseConfig
        super().__post_init__()

        # Time grid shared by both SDE solver and CDE solver.
        # Shape: (steps,)  — one node per real data timestep.
        self.sde_times_full = torch.linspace(0, self.T, (self.multiplier_of_times * (self.steps-1) + 1)).to(self.device)

        self.sde_times_matched = torch.linspace(0, self.T, self.steps).to(self.device)