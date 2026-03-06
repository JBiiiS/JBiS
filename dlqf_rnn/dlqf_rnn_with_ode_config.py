import torch
from dataclasses import dataclass
from .dlqf_rnn_config import DLQFRNNConfig


@dataclass
class DLQFRNNWithODEConfig(DLQFRNNConfig):
    # ----------------------------------
    # Neural SDE Settings
    # ----------------------------------


    # [Brownian Motion]
    # Number of independent Brownian motion channels driving the SDE.
    noise_dim: int = 3

    # [Drift / Diffusion Networks]
    # Hidden layer width for μ_θ(t, z) and σ_θ(t, z).
    ode_hidden_dim: int = 16

    exp_deno_init: float = 8.5
    softplus_deno: float = 100.0
    
    gamma_init: float = 0.5
    alpha_init: float = 1.0

    output_dim: int = 1       # default matches BaseConfig.num_assets




    def __post_init__(self):
        super().__post_init__()
        self.lstm_hidden_dim: int = self.hidden_dim
        # Time grid shared by both SDE solver 
    
        self.ode_times = torch.linspace(0, 1, self.total_quantile).to(self.device)


        