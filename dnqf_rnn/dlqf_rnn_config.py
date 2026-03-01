import torch
import numpy as np
from dataclasses import dataclass
from base.base_config import BaseConfig 

@dataclass
class DLQFRNNConfig(BaseConfig):
    num_assets: int = 3

    input_dim: int = 4 # [log return, RV, volume, ID]

    num_layers_lstm: int = 3
    num_layers_nqf: int = 3


    hidden_dim: int = 64 #lstm  dimension
    latent_dim: int = 128 #linear dimension


    total_quntile: int = 78   # 하루 5분봉 개수 (390분 / 5분)
    batch_size = 66        # 입력 길이 (과거 약 3개월)
    input_len: int = 100  # |r| × 100 (numerical underflow 방지, 논문 4.2.3)
    
