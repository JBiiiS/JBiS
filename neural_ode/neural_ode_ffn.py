import torch
import torch.nn as nn
import numpy as np

# Assuming the config object 'cfg' is passed from your main code
from neural_ode_config import NeuralODEConfig

class NeuralODEEncoder(nn.Module):
    """
    [Encoder]
    Reads the observed time-series data and produces the 
    parameters (mean, log_variance) of the latent distribution q(z0|x).
    """
    def __init__(self, config: NeuralODEConfig):
        super().__init__()
        
        # 1. Recurrent Layer (GRU) to process time-series
        # We use GRU because it's faster and often more stable than LSTM for ODEs.
        self.gru = nn.GRU(
            input_size=config.input_dim, 
            hidden_size=config.rec_hidden_dim, 
            num_layers=config.num_layers, 
            batch_first=True
        )
        
        # 2. Linear Layers to predict Mean (mu) and Log-Variance (logvar)
        # We transform the final hidden state of GRU into these parameters.
        self.mu_head = nn.Linear(config.rec_hidden_dim, config.latent_dim)
        self.logvar_head = nn.Linear(config.rec_hidden_dim, config.latent_dim)

    def forward(self, x):
        # x shape: [Batch, Time, Dim] (e.g., [64, 30, 1])
        
        
        # Run GRU
        # out: [Batch, Time, Hidden]
        # h_n: [1, Batch, Hidden] (The final summary vector)
        _, h_n = self.gru(x)

        # Preparing for when hidden layers are more than 2    
        h_n = h_n[-1:,:,:]
        
        # Squeeze the layer dimension: [Batch, Hidden]
        h_n = h_n.squeeze(0)
        
        # Predict the statistical parameters of z0
        mu = self.mu_head(h_n)
        logvar = self.logvar_head(h_n)
        
        return mu, logvar


class LatentSampler(nn.Module):
    """
    [Sampler]
    Performs the 'Reparameterization Trick'.
    This is where the Gaussian Noise enters the model.
    """
    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar):
        """
        Args:
            mu: Mean of the latent Gaussian [Batch, latent_dim]
            logvar: Log-Variance of the latent Gaussian [Batch, latent_dim]
        Returns:
            z0: Sampled initial state [Batch, latent_dim]
        """
        
        # 1. Calculate Standard Deviation
        # exp(0.5 * log_var) == sqrt(var) == std
        std = torch.exp(0.5 * logvar)
        
        # -----------------------------------------------------------
        # [CRITICAL STEP] The Noise Injection (Reparameterization)
        # -----------------------------------------------------------
        # eps ~ N(0, I)
        # This 'torch.randn_like' is the "Normal Distribution" you asked about.
        # It generates pure noise.
        eps = torch.randn_like(std) 
        
        # z0 = mu + sigma * epsilon
        # We scale the noise by sigma and shift it by mu.
        z0 = mu + eps * std
        
        
        return z0