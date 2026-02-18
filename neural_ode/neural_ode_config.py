import torch
import numpy as np
from dataclasses import dataclass, field
from base.base_config import BaseConfig  # Assuming your previous code is in base_config.py

@dataclass
class NeuralODEConfig(BaseConfig):
    # ----------------------------------
    # [4] Model Architecture (Latent VAE)
    # ----------------------------------
    
    # [Encoder]
    # The dimension of the hidden state in the RNN/GRU encoder.
    # This processes the historical path (t=0 to T) to extract features.
    input_dim: int = 1
    rec_hidden_dim: int = 128  
    num_layers: int = 1
    
    # [Latent Space]
    # The dimension of z0 (the bottleneck). 
    # This determines how much "information" is compressed into the initial state.
    # A smaller dim forces the model to learn more general features (regularization).
    latent_dim: int = 20       

    # [ODE Function / Vector Field]
    # The dimension of the neural network layers inside the ODE function f(z, t).
    # This defines the complexity of the "derivative" or "trend".
    ode_hidden_dim: int = 128  
    
    # ----------------------------------
    # [5] ODE Solver Settings (Numerical Integration)
    # ----------------------------------
    
    # The numerical method to solve the ODE.
    # 'dopri5': Runge-Kutta 4(5) (Adaptive step size, standard for Neural ODEs).
    # 'rk4': Fixed step Runge-Kutta 4 (Faster, but less accurate).
    # 'euler': Euler method (Fastest, least accurate, good for SDEs).
    solver_method: str = 'dopri5' 

    # Absolute and Relative tolerances for adaptive solvers (like dopri5).
    # Lower values = higher accuracy but slower computation.
    atol: float = 1e-5
    rtol: float = 1e-5

    # How many nodes to generate
    multiplier_of_times : int = 5

    # ----------------------------------
    # [6] Loss & Regularization
    # ----------------------------------
    
    # The weight (beta) for the KL Divergence term in the loss function.
    # Loss = MSE + (kl_coeff * KLD)
    # If this is 0, it becomes a standard Autoencoder (prone to overfitting).
    kld_coeff: float = 0.01     

    def __post_init__(self):
        # Call the parent's post_init to calculate 'dt'
        super().__post_init__()

        self.ode_times = torch.linspace(0, self.T, (self.multiplier_of_times * (self.steps-1) + 1)).to(self.device)
        
        # Validation logic (optional)
        if self.solver_method == 'dopri5' and self.steps < 5:
            print("Warning: 'dopri5' works best with continuous time. "
                  f"Your step count ({self.steps}) is low.")