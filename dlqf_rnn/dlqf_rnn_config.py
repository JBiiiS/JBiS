import torch
import numpy as np
from dataclasses import dataclass, field
from base.base_config import BaseConfig 

@dataclass
class DLQFRNNConfig(BaseConfig):
    num_assets: int = 1

    input_dim: int = 4 # [log return, RV, volume, ID]

    num_layers_lstm: int = 3
    num_layers_nqf: int = 3


    hidden_dim: int = 64 #lstm  dimension
    latent_dim: int = 128 #linear dimension


    total_quantile: int = 78   # 하루 5분봉 개수 (390분 / 5분)
    input_len: int = 66
    scale_factor: int = 100  # |r| × 100 (numerical underflow 방지, 논문 4.2.3)
    
