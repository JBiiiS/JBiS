import torch
import torch.nn as nn
import torchcde #type:ignore
from neural_sde_gan.neural_sde_config import NeuralSDEConfig


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

    def __init__(self, config: NeuralSDEConfig):
        super().__init__()
        self.cde_hidden_dim = config.cde_hidden_dim
        self.output_dim     = config.output_dim

        self.net = nn.Sequential(
            nn.Linear(config.cde_hidden_dim + 1, config.cde_hidden_dim),  # +1 for time
            LipSwish(),
            nn.Linear(config.cde_hidden_dim, config.cde_hidden_dim),
            LipSwish(),
            # Output: one row per hidden unit, one column per input channel
            nn.Linear(config.cde_hidden_dim, config.cde_hidden_dim * config.output_dim),
            nn.Tanh()
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
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
    Neural CDE discriminator (Kidger et al., 2021).

    Takes a path X (real or generated) and returns an unconstrained scalar
    score — used directly as the WGAN critic value.

    Pipeline:
        X  (B, steps, output_dim)
          ↓  torchcde.natural_cubic_spline_coeffs   [continuous spline]
          ↓  h0 = h0_linear(X[:, 0, :])             [init hidden from first obs]
          ↓  torchcde.cdeint : dh = f_φ(t,h) dX(t) [integrate along path]
          ↓  score = readout(h_T)                   [(B, 1), no activation]

    Both real and fake inputs are always (B, steps, output_dim) on sde_times_matched.
    The generator subsamples x_hat_full to x_hat_matched before calling this
    discriminator, ensuring both sides share the same temporal resolution.
    Without that alignment, the discriminator would trivially exploit the
    difference in grid density instead of learning the true distributional gap.

    No activation on the readout: the critic must stay unconstrained for WGAN.

    Reference:
        Kidger et al. (2021) "Neural SDEs as Infinite-Dimensional GANs"
        https://arxiv.org/abs/2102.03657
    """

    def __init__(self, config: NeuralSDEConfig):
        super().__init__()
        self.config = config

        self.cde_func  = CDEFunc(config)

        # Initial hidden state — linear projection of the first observation
        self.h0_linear = nn.Sequential(nn.Linear(config.output_dim, config.cde_hidden_dim),
                                       LipSwish(),
                                       nn.Linear(config.cde_hidden_dim, config.cde_hidden_dim),
                                       LipSwish(),
                                       nn.Linear(config.cde_hidden_dim, config.cde_hidden_dim)
        )
        for m in self.h0_linear.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, val=0)

        # Final readout — no activation
        self.readout   = nn.Linear(config.cde_hidden_dim, 1)

    def forward(self, x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x     : (B, steps, output_dim) — real or subsampled generated path
                    generated path must be x_hat_matched (not x_hat_full)
            times : (steps,)               — sde_times_matched for both real and fake
        Returns:
            score : (B, 1) — unconstrained critic score
        """
        # ------------------------------------------------------------------
        # [1] Build continuous cubic spline from discrete path
        #     coeffs encodes X as a piecewise cubic polynomial over 'times'.
        #     After this step, the raw grid size T no longer matters.
        # ------------------------------------------------------------------

        coeffs = torchcde.linear_interpolation_coeffs(x, t=times)
        intepolated_x = torchcde.LinearInterpolation(coeffs, t=times)

        '''
        coeffs = torchcde.natural_cubic_spline_coeffs(x, t=times)
        spline = torchcde.CubicSpline(coeffs)
        '''
        # ------------------------------------------------------------------
        # [2] Initial hidden state from first observation
        # ------------------------------------------------------------------
        h0 = self.h0_linear(x[:, 0, :])                # (B, cde_hidden_dim)

        # ------------------------------------------------------------------
        # [3] Integrate CDE:  dh = f_φ(t, h) dX(t)
        #     We only need h at the final time T, so we pass [t_start, t_end].
        #     Output shape: (B, 2, cde_hidden_dim)
        # ------------------------------------------------------------------
        t_eval = times[[0, -1]]                         # (2,)  →  [0, T]
        h = torchcde.cdeint(
            X       = intepolated_x,
            func    = self.cde_func,
            z0      = h0,
            t       = t_eval,
            method  = 'rk4',
            adjoint = True,
            adjoint_method = self.config.adjoint_method
        )                              
        # (B, 2, cde_hidden_dim)

        # ------------------------------------------------------------------
        # [4] Readout from final hidden state h(T)
        # ------------------------------------------------------------------
        h_T   = h[:, -1, :]                            # (B, cde_hidden_dim)
        score = self.readout(h_T)                       # (B, 1)
        return score
