import torch
import torch.nn as nn
import numpy as np

from .dlqf_rnn_with_ode_config import DLQFRNNWithODEConfig
from .dlqf_rnn_with_ode_model import ODEGenerator


class BiLSTMEncoder(nn.Module):
    def __init__(self, config: DLQFRNNWithODEConfig):
        super().__init__()
        self.config = config

        self.bilstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers_lstm,
            batch_first=True,
            dropout=config.dropout if config.num_layers_lstm > 1 else 0.0,
            bidirectional=True,
            device=config.device
        )

        for name, param in self.bilstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
                n = param.shape[0]
                param.data[n // 4: n // 2].fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_output, _ = self.bilstm(x)
        h_last = lstm_output[:, -1, :]
        return h_last


class DLQFRNNWithODE(nn.Module):
    def __init__(self, config: DLQFRNNWithODEConfig):
        super().__init__()
        self.config = config
        self.encoder = BiLSTMEncoder(config).to(config.device)
        self.ODEGenerator = ODEGenerator(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        r_q = self.ODEGenerator(h)
        return r_q

    def estimate_rv(self, x: torch.Tensor) -> torch.Tensor:
        sf = self.config.scale_factor
        r_q = self.forward(x)
        rv_x = (r_q ** 2).sum(dim=1) / (sf ** 2)
        return rv_x


def l2_distance_loss(r_true, r_pred):
    B, M = r_true.shape
    combined = torch.cat([r_true, r_pred], dim=1)
    combined_sorted, _ = combined.sort(dim=1)
    delta = combined_sorted[:, 1:] - combined_sorted[:, :-1]
    x_mid = combined_sorted[:, :-1]
    r_true_s, _ = r_true.sort(dim=1)
    r_pred_s, _ = r_pred.sort(dim=1)
    f_true = (r_true_s.unsqueeze(1) <= x_mid.unsqueeze(2)).float().sum(dim=2) / M
    f_pred = (r_pred_s.unsqueeze(1) <= x_mid.unsqueeze(2)).float().sum(dim=2) / M
    diff_sq = (f_true - f_pred) ** 2
    loss = (delta * diff_sq).sum(dim=1).sqrt()
    return loss.mean()


def mse_loss(rv_true, rv_pred):
    return torch.mean((rv_true - rv_pred) ** 2)


def qlike_loss(rv_true, rv_pred):
    epsilon = 1e-8
    rv_pred_safe = torch.clamp(rv_pred, min=epsilon)
    rv_true_safe = torch.clamp(rv_true, min=epsilon)
    ratio = rv_true_safe / rv_pred_safe
    return torch.mean(ratio - torch.log(ratio) - 1.0)