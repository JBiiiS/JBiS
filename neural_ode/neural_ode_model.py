import torch
import torch.nn as nn

# Standard library for Neural ODEs (Auto-differentiable solver)
from torchdiffeq import odeint

from neural_ode.neural_ode_config import NeuralODEConfig

class NeuralODE(nn.Module):
    """
    [The Full Model]
    Integrates: Encoder -> Sampler -> ODE Solver -> Decoder
    """
    def __init__(self, config: NeuralODEConfig, encoder: nn.Module, sampler: nn.Module, ode_function: nn.Module, decoder: nn.Module):
        super().__init__()
        self.config = config
        
        # 1. Components
        self.encoder = encoder(config)
        self.sampler = sampler()
        self.ode_func = ode_function(config)
        self.decoder = decoder(config)

        

    def forward(self, x):
        """
        Args:
            x: Input stock data [Batch, Time, Dim]
            time_steps: Physical time points to solve for [Time] (e.g., 0, 1/365, ...)
        """
        # [LOGICAL FIX] Reverse the input time.
        # The RNN reads x_29, x_28, ..., x_0.
        # Its final state h_n will correspond to the "state at t=0".
        x_reversed = torch.flip(x, [1])

        # ----------------------------------------
        # Phase 1: Encode & Sample (The "Start")
        # ----------------------------------------
        # mu, logvar: [Batch, latent_dim]
        mu, logvar = self.encoder(x_reversed)
        
        # z0: [Batch, latent_dim] (Noisy initial state)
        z0 = self.sampler(mu, logvar)

        # ----------------------------------------
        # Phase 2: Solve ODE (The "Evolution")
        # ----------------------------------------
        # This is where the magic happens.
        # We integrate 'ode_func' from t[0] to t[end] starting at z0.
        # Output z_traj: [Time, Batch, latent_dim]
        z_traj = odeint(
            self.ode_func, 
            z0, 
            self.config.ode_times, 
            method=self.config.solver_method,
            atol=self.config.atol,
            rtol=self.config.rtol
        )

        # Permute to [Batch, Time, Latent] to match standard format
        z_traj = z_traj.permute(1, 0, 2)

        # ----------------------------------------
        # Phase 3: Decode (The "Observation")
        # ----------------------------------------
        predicted_x = self.decoder(z_traj)

        return predicted_x, mu, logvar