import torch
from dataclasses import dataclass
from .dlqf_rnn_config import DLQFRNNConfig


@dataclass
class DLQFRNNWithSDEConfig(DLQFRNNConfig):
    # ----------------------------------
    # Neural SDE Settings
    # ----------------------------------


    # [Brownian Motion]
    # Number of independent Brownian motion channels driving the SDE.
    noise_dim: int = 3

    # [Drift / Diffusion Networks]
    # Hidden layer width for μ_θ(t, z) and σ_θ(t, z).
    sde_hidden_dim: int = 16

    # [Readout / Decoder]
    # Maps latent state z_t → observation x_t.
    # Must match the number of assets in the real data.
    # Set equal to num_assets if modeling full return vector,
    # or a smaller value if modeling a single asset at a time.
    output_dim: int = 1       # default matches BaseConfig.num_assets

    # ----------------------------------
    # SDE Solver Settings
    # ----------------------------------
    noise_type: str = 'general'
    sde_type: str = 'stratonovich'
    # Numerical integration method passed to torchsde.sdeint.
    # 'euler'     : Euler-Maruyama  (fast, first-order, good for training)
    # 'milstein'  : Milstein        (second-order, slightly more accurate)
    # 'srk'       : Stochastic RK   (higher-order, slower)
    sde_method: str = 'adversible_heun'
    adjoint_method: str = 'adjoint_reversible_heun'


    def __post_init__(self):
        super().__post_init__()
        self.lstm_hidden_dim: int = self.hidden_dim
        # Time grid shared by both SDE solver 
    
        self.sde_times = torch.linspace(0, 1, self.total_quantile)