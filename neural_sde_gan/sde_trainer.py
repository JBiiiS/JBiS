import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from neural_sde_gan.neural_sde_config    import NeuralSDEConfig
from neural_sde_gan.neural_sde_gan_model import NeuralSDEGAN


# =============================================================================
# [1] Gradient Penalty
# =============================================================================

def _gradient_penalty(model: NeuralSDEGAN, real_x: torch.Tensor, fake_x: torch.Tensor) -> torch.Tensor:
    """
    WGAN-GP gradient penalty (Gulrajani et al., 2017).

    Interpolates between real and fake paths, scores the interpolation,
    then penalises the critic gradient norm for deviating from 1:
        GP = E[ (||∇_x D(x_interp)||_2 - 1)^2 ]

    Args:
        model  : NeuralSDEGAN — provides model.discriminate()
        real_x : (B, steps, output_dim)
        fake_x : (B, steps, output_dim)  ← must be x_hat_matched
    Returns:
        gp     : scalar — gradient penalty term
    """
    B = real_x.size(0)

    # Random interpolation coefficient α ~ Uniform(0, 1) per sample
    alpha = torch.rand(B, 1, 1, device=real_x.device)          # (B, 1, 1)
    interp = (alpha * real_x + (1 - alpha) * fake_x).requires_grad_(True)

    interp_score = model.discriminate(interp)                   # (B, 1)

    # Compute ∇_interp D(interp)
    gradients = torch.autograd.grad(
        outputs    = interp_score.sum(),
        inputs     = interp,
        create_graph = True,
    )[0]                                                        # (B, steps, output_dim)

    # L2 norm over (steps, output_dim), then (||·||_2 - 1)^2
    gp = ((gradients.norm(2, dim=[1, 2]) - 1) ** 2).mean()
    return gp


# =============================================================================
# [2] Single Discriminator Step
# =============================================================================

def _step_D(
    model   : NeuralSDEGAN,
    real_x  : torch.Tensor,
    opt_D   : torch.optim.Optimizer,
    config  : NeuralSDEConfig,
) -> float:
    """
    One discriminator (critic) update step.

    L_D = E[D(fake)] - E[D(real)] + gp_lambda * GP

    The discriminator maximises E[D(real)] - E[D(fake)],
    which equals minimising L_D as written.

    Args:
        model  : NeuralSDEGAN
        real_x : (B, steps, output_dim) — one batch of real log-returns
        opt_D  : discriminator optimiser
        config : NeuralSDEConfig
    Returns:
        d_loss : float
    """
    model.discriminator.train()
    model.generator.eval()
    opt_D.zero_grad()

    # Generate fake paths — detach so G gradients are not computed here
    with torch.no_grad():
        _, fake_x = model.generate()                            # (B, steps, output_dim)

    real_score = model.discriminate(real_x)                     # (B, 1)
    fake_score = model.discriminate(fake_x)                     # (B, 1)
    gp         = _gradient_penalty(model, real_x, fake_x)

    d_loss = fake_score.mean() - real_score.mean() + config.gp_lambda * gp
    d_loss.backward()
    opt_D.step()

    return d_loss.item(), gp.item()


# =============================================================================
# [3] Single Generator Step
# =============================================================================

def _step_G(
    model  : NeuralSDEGAN,
    opt_G  : torch.optim.Optimizer,
) -> float:
    """
    One generator update step.

    L_G = -E[D(fake)]

    The generator minimises L_G, i.e. tries to maximise the critic score
    on its generated paths.

    Args:
        model : NeuralSDEGAN
        opt_G : generator optimiser
    Returns:
        g_loss : float
    """
    model.generator.train()
    model.discriminator.eval()
    opt_G.zero_grad()

    _, fake_x  = model.generate()                               # (B, steps, output_dim)
    fake_score = model.discriminate(fake_x)                     # (B, 1)

    g_loss = -fake_score.mean()
    g_loss.backward()
    opt_G.step()

    return g_loss.item()


# =============================================================================
# [4] Main Training Loop
# =============================================================================

def train(
    model      : NeuralSDEGAN,
    dataloader : DataLoader,
    config     : NeuralSDEConfig,
) -> dict:
    """
    Full WGAN-GP training loop for Neural SDE GAN.

    Each batch alternates between:
        - time_D_per_batch discriminator steps  (critic update)
        - time_G_per_batch generator steps      (generator update)

    Separate optimisers are used for G and D so their parameters
    are never updated together, which is required for adversarial training.

    Args:
        model      : NeuralSDEGAN
        dataloader : yields (B, steps, output_dim) real log-return batches
        config     : NeuralSDEConfig

    Returns:
        history : dict with keys 'd_loss' and 'g_loss' (lists of per-epoch means)
    """
    # ------------------------------------------------------------------
    # Separate optimisers for generator and discriminator
    # ------------------------------------------------------------------
    opt_G = torch.optim.Adam(
        model.generator.parameters(),
        lr           = config.learning_rate,
        weight_decay = config.weight_decay,
        betas        = (0.5, 0.9),              # WGAN-GP recommended betas
    )
    opt_D = torch.optim.Adam(
        model.discriminator.parameters(),
        lr           = config.learning_rate,
        weight_decay = config.weight_decay,
        betas        = (0.5, 0.9),
    )

    history = {'d_loss': [], 'g_loss': []}

    for epoch in range(1, config.num_epochs + 1):

        epoch_d_losses = []
        epoch_g_losses = []

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.num_epochs}", leave=False)

        for real_x in pbar:
            # real_x : (B, steps, output_dim)
            real_x = real_x.to(config.device)

            # --------------------------------------------------------------
            # Phase D : update discriminator time_D_per_batch times
            # --------------------------------------------------------------
            for _ in range(config.time_D_per_batch):
                d_loss = _step_D(model, real_x, opt_D, config)
                epoch_d_losses.append(d_loss)

            # --------------------------------------------------------------
            # Phase G : update generator time_G_per_batch times
            # --------------------------------------------------------------
            for _ in range(config.time_G_per_batch):
                g_loss = _step_G(model, opt_G)
                epoch_g_losses.append(g_loss)

            pbar.set_postfix(d_loss=f"{d_loss:.4f}", g_loss=f"{g_loss:.4f}")

        mean_d = sum(epoch_d_losses) / len(epoch_d_losses)
        mean_g = sum(epoch_g_losses) / len(epoch_g_losses)
        history['d_loss'].append(mean_d)
        history['g_loss'].append(mean_g)

        print(f"Epoch {epoch:>4d} | D loss: {mean_d:.4f} | G loss: {mean_g:.4f}")

    return history
