import torch
import numpy as np
from dataclasses import dataclass
from base.base_config import BaseConfig 

@dataclass
class NQFRNNConfig(BaseConfig):
    num_assets: int = 3

    input_dim: int = 4

    num_layers_lstm: int = 3
    num_layers_nqf: int = 3


    hidden_dim: int = 64 #lstm  dimension
    latent_dim: int = 128 #linear dimension
    
