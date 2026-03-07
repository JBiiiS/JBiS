import torch
import torch.nn as nn
import torchcde #type:ignore
from .dlqf_rnn_with_ode_config import DLQFRNNWithODEConfig


class LipSwish(torch.nn.Module):
    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)



# =============================================================================
# [1] CDE Vector Field  f_φ(t, h)
# =============================================================================

class CDEFunc(nn.Module):
    """
    The vector field of the Neural CDE discriminator:
        dh = f_φ(t, h) dX(t)

    f_φ maps (t, h) to a matrix of shape (cde_hidden_dim, output_dim),
    so that dh = f_φ(t, h) @ dX(t) is a well-defined vector in R^cde_hidden_dim.

    Time t is concatenated to h for non-autonomous dynamics,
    using the same torch.full pattern as SDEDrift / SDEDiffusion.

    Input  : (t, h)  →  [0-dim scalar tensor, (B, cde_hidden_dim)]
    Output : (B, cde_hidden_dim, output_dim)
    """

    def __init__(self, config: DLQFRNNWithODEConfig):
        super().__init__()
        self.cde_hidden_dim = config.ode_hidden_dim
        self.output_dim     = config.output_dim

        self.net = nn.Sequential(
            nn.Linear(self.cde_hidden_dim + 1, self.cde_hidden_dim),  # +1 for time
            LipSwish(),
            nn.Linear(self.cde_hidden_dim, self.cde_hidden_dim),
            LipSwish(),
            # Output: one row per hidden unit, one column per input channel
            nn.Linear(self.cde_hidden_dim, self.cde_hidden_dim * self.output_dim),
            nn.Tanh()
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, h):
        """
        Args:
            t : 0-dim scalar tensor — current time (passed by torchcde)
            h : (B, cde_hidden_dim) — current CDE hidden state
        Returns:
            (B, cde_hidden_dim, output_dim) — matrix field
        """
        # Same pattern as SDEDrift: broadcast 0-dim scalar t to (B, 1)
        t_batch = torch.full((h.size(0), 1), float(t), device=h.device, dtype=h.dtype)
        th  = torch.cat([t_batch, h], dim=-1)               # (B, 1 + cde_hidden_dim)
        out = self.net(th)                                   # (B, cde_hidden_dim * output_dim)
        return out.view(h.size(0), self.cde_hidden_dim, self.output_dim)
    
    # =============================================================================
# [2] CDEDiscriminator — full Neural CDE discriminator
# =============================================================================

class CDEDiscriminator(nn.Module):
    """
    Neural CDE discriminator (Kidger et al., 2021b).
    """

    def __init__(self, config:DLQFRNNWithODEConfig):
        super().__init__()
        self.config = config
        self.cde_hidden_dim = config.ode_hidden_dim
        self.cde_func  = CDEFunc(config)

        # Initial hidden state — linear projection of the first observation
        self.h0_linear = nn.Sequential(
            nn.Linear(config.output_dim, self.cde_hidden_dim),
            LipSwish(),
            nn.Linear(self.cde_hidden_dim, self.cde_hidden_dim),
            LipSwish(),
            nn.Linear(self.cde_hidden_dim, self.cde_hidden_dim)
        )

        for m in self.h0_linear.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, val=0)

        # Final readout — no activation
        self.readout   = nn.Linear(self.cde_hidden_dim, 1)
        for m in self.readout.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, val=0)
   

    def forward(self, x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x     : (B, steps, output_dim) — real or subsampled generated path
            times : (steps,)               — sde_times_matched for both real and fake
        """
        
        # ------------------------------------------------------------------
        # [1] Build continuous interpolation from discrete path
        # ------------------------------------------------------------------
        # Linear interpolation is preferred by Kidger et al. for the CDE Discriminator 
        # when using the torchsde backend.
        coeffs = torchcde.linear_interpolation_coeffs(x, t=times)
        interpolated_x = torchcde.LinearInterpolation(coeffs, t=times)

        # ------------------------------------------------------------------
        # [2] Initial hidden state from first observation
        # -------------------------------------------------
        x0 = interpolated_x.evaluate(interpolated_x.interval[0]) 
        h0 = self.h0_linear(x0)                # (B, cde_hidden_dim)

        # ------------------------------------------------------------------
        # [3] Integrate CDE using the 'torchsde' backend and 'reversible_heun'
        # ------------------------------------------------------------------
        
        h = torchcde.cdeint(
            X       = interpolated_x,
            func    = self.cde_func,
            z0      = h0,
            t       = interpolated_x.interval,
            method  = 'dopri5',    
            adjoint = True,        
            atol    = 1e-4,      # Extracted as a top-level explicit argument
            rtol    = 1e-4       # Extracted as a top-level explicit argument
        )

        # ------------------------------------------------------------------
        # [4] Readout from final hidden state h(T)
        # ------------------------------------------------------------------
        h_T   = h[:, -1, :]                            # (B, cde_hidden_dim)
        score = self.readout(h_T)                      # (B, 1)
        
        return score

