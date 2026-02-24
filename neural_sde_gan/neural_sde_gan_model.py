import torch
import torch.nn as nn

from neural_sde_gan.neural_sde_config   import NeuralSDEConfig
from neural_sde_gan.sde_generator       import SDEGenerator
from neural_sde_gan.cde_discriminator   import CDEDiscriminator


class NeuralSDEGAN(nn.Module):
    """
    [The Full Model]
    Integrates: SDEGenerator  ↔  CDEDiscriminator

    Generator     : z0 ~ N(0,I)  →  SDE  →  x_hat_full, x_hat_matched
    Discriminator : x (real or x_hat_matched)  →  scalar WGAN critic score

    Component classes are passed as arguments and instantiated here,
    following the same convention as NeuralODE in neural_ode_model.py.

    batch_size is read from config.batch_size (defined in BaseConfig),
    so generate() and forward() require no size arguments.

    Training alternates between:
        Phase D — update discriminator (critic) config.time_D_per_batch times
        Phase G — update generator              config.time_G_per_batch times
    Optimizer steps are handled separately in sde_trainer.py.
    """

    def __init__(self, config: NeuralSDEConfig, generator, discriminator):
        super().__init__()
        self.config = config

        # Instantiate from passed classes (same pattern as NeuralODE)
        self.generator     = generator(config)
        self.discriminator = discriminator(config)

    # ------------------------------------------------------------------
    # [1] Generate
    # ------------------------------------------------------------------
    def generate(self):
        """
        Sample a batch of synthetic log-return paths from the Neural SDE.
        Batch size is taken from config.batch_size.

        Returns:
            x_hat_full    : (B, T_fine, output_dim) — full fine-grained path
            x_hat_matched : (B, steps,  output_dim) — subsampled at real-data timesteps
                            → pass this to discriminate()
        """
        return self.generator()

    # ------------------------------------------------------------------
    # [2] Discriminate
    # ------------------------------------------------------------------
    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Score a batch of paths (real or x_hat_matched) via the Neural CDE critic.
        Both real and fake must be (B, steps, output_dim) on sde_times_matched.

        Args:
            x     : (B, steps, output_dim)
        Returns:
            score : (B, 1) — unconstrained WGAN critic score
        """
        times = self.config.sde_times_matched
        return self.discriminator(x, times)

    # ------------------------------------------------------------------
    # [3] Forward  (= generate, for API completeness)
    # ------------------------------------------------------------------
    def forward(self):
        """
        Convenience wrapper — equivalent to generate().
        Returns (x_hat_full, x_hat_matched).
        """
        return self.generate()
