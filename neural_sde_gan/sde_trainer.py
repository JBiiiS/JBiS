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
    grad_norm  = gradients.norm(2, dim=[1, 2])
    gp = ((grad_norm - 1) ** 2).mean()
    '''
    n_elements = gradients.shape[1] * gradients.shape[2]   # steps * output_dim
    grad_norm  = gradients.norm(2, dim=[1, 2]) / (n_elements ** 0.5)
    gp = ((grad_norm - 1) ** 2).mean()
    '''
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

    return d_loss.item(), gp


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

def _step_wo_gp(model : NeuralSDEGAN, real_x : torch.Tensor, opt_d : torch.optim.Optimizer, opt_g :  torch.optim.Optimizer):
    model.train()
    opt_d.zero_grad()
    opt_g.zero_grad()

    _, fake_x = model.generate()
    fake_score = model.discriminate(fake_x)   

    real_score = model.discriminate(real_x)
    
    loss = fake_score.mean() - real_score.mean()

    loss.backward()

    for params in model.generator.parameters():
        if params.grad is not None:
            params.grad *= -1

    opt_d.step()
    opt_g.step()
    opt_d.zero_grad()
    opt_g.zero_grad()

    with torch.no_grad():
        for module in model.discriminator.modules():
            if isinstance(module, nn.Linear):
                # constrainting L1 norm <= 1
                lim = 1.0 / module.out_features 
                module.weight.clamp_(-lim, lim)

    return fake_score.mean().item(), real_score.mean().item()

