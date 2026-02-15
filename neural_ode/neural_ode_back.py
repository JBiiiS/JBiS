import torch
import torch.nn as nn
from neural_ode_config import NeuralODEConfig

class ODEFunc(nn.Module):
    """
    [The Physics Engine]
    This defines the derivative function: dz/dt = f(z, t).
    Instead of a fixed mathematical formula (like Black-Scholes), 
    we use a neural network to learn the dynamics.
    """
    def __init__(self, config: NeuralODEConfig):
        super().__init__()
        
        # A simple MLP to model complex non-linear dynamics.
        # Structure: z -> Hidden -> Hidden -> dz/dt
        self.net = nn.Sequential(
            nn.Linear(config.latent_dim, config.ode_hidden_dim),
            nn.Tanh(),  # Tanh is generally smoother/stable for ODEs than ReLU
            nn.Linear(config.ode_hidden_dim, config.ode_hidden_dim),
            nn.Tanh(),
            nn.Linear(config.ode_hidden_dim, config.latent_dim)
        )

        # Initialize weights to be small (helps stability in ODEs)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, z):
        """
        Args:
            t: Current time (scalar). Needed by the solver, even if not used directly.
            z: Current latent state [Batch, latent_dim]
        Returns:
            dz_dt: The derivative (gradient) [Batch, latent_dim]
        """
        # (Optional) If dynamics depend on explicit time t, concatenate it here.
        # For autonomous systems (time-invariant laws), we just use z.
        return self.net(z)

class NeuralODEDecoder(nn.Module):
    """
    [Decoder]
    Projects the latent state trajectory z(t) back to the observation space x(t).
    
    Mathematically:
        x_hat_t = Network(z_t)
        
    This network is applied point-wise to every time step.
    Since z_t contains the 'physics' and 'trend', the Decoder extracts
    the specific asset prices from that abstract representation.
    """
    def __init__(self, config: NeuralODEConfig):
        super().__init__()
        
        # We use a small MLP to map Latent Dim -> Input Dim (Stock Price)
        # Architecture: Linear -> ReLU -> Linear
        
        self.net = nn.Sequential(
            # Layer 1: Expand from Latent Space to Hidden Dimension
            nn.Linear(config.latent_dim, config.rec_hidden_dim),
            nn.ReLU(),
            
            # Layer 2: Project down to Original Input Dimension (e.g., 1 for single stock)
            nn.Linear(config.rec_hidden_dim, config.input_dim)
        )

    def forward(self, z_traj):
        """
        Args:
            z_traj: The latent trajectory from ODE Solver.
                    Shape: [Batch, Time, Latent_Dim]
                    
        Returns:
            x_hat: The reconstructed stock prices.
                   Shape: [Batch, Time, Input_Dim]
        """
        # PyTorch's nn.Linear applies to the LAST dimension automatically.
        # So it works perfectly on [Batch, Time, Dim] without any reshaping loops.
        x_hat = self.net(z_traj)
        
        return x_hat













