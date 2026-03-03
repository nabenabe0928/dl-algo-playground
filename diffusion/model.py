from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Encodes diffusion timestep t into a vector using sinusoidal positional encoding.

    Same idea as positional encoding in transformers, but applied to time steps.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (batch_size,) integer timesteps --> (batch_size, embed_dim)
        half_dim = self.embed_dim // 2
        log_freq = math.log(10000.0) / (half_dim - 1)
        freq = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -log_freq)
        emb = t[:, None].float() * freq[None, :]
        # Concatenate sin and cos to form the full embedding
        return torch.cat([emb.sin(), emb.cos()], dim=-1)  # (batch_size, embed_dim)


class ResidualBlock(nn.Module):
    """Convolutional residual block conditioned on timestep embedding.

    Each block: GroupNorm -> SiLU -> Conv -> (add time embedding) -> GroupNorm -> SiLU -> Conv
    A skip connection adds the input to the output (with a 1x1 conv if channels change).
    """

    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        # Project time embedding to match channel dimension
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        # 1x1 conv for skip connection when channel dimensions differ
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, in_channels, H, W), t_emb: (batch_size, time_dim)
        h = self.block1(x)
        # Broadcast time embedding: (batch_size, out_channels) --> (batch_size, out_channels, 1, 1)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.skip(x)  # (batch_size, out_channels, H, W)


class SimpleUNet(nn.Module):
    """A minimal U-Net for noise prediction, conditioned on diffusion timestep.

    Architecture (for 32x32 input):
        Encoder: 32x32 -> 16x16 -> 8x8
        Bottleneck: 8x8
        Decoder: 8x8 -> 16x16 -> 32x32

    Skip connections link encoder and decoder at matching resolutions.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        time_dim: int = 256,
    ) -> None:
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        channels = [base_channels * m for m in channel_mults]  # e.g. [64, 128, 256]

        # Initial projection: (batch_size, 3, 32, 32) --> (batch_size, 64, 32, 32)
        self.conv_in = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        # --- Encoder ---
        # Each level: ResidualBlock + Downsample (2x)
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.encoder_blocks.append(ResidualBlock(channels[i], channels[i], time_dim))
            self.downsamples.append(
                nn.Conv2d(channels[i], channels[i + 1], 4, stride=2, padding=1)
            )

        # --- Bottleneck ---
        self.bottleneck = ResidualBlock(channels[-1], channels[-1], time_dim)

        # --- Decoder ---
        # Each level: Upsample (2x) + ResidualBlock (input has skip connection, so 2x channels)
        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for i in reversed(range(len(channels) - 1)):
            self.upsamples.append(
                nn.ConvTranspose2d(channels[i + 1], channels[i], 4, stride=2, padding=1)
            )
            # Input channels doubled due to skip connection from encoder
            self.decoder_blocks.append(ResidualBlock(channels[i] * 2, channels[i], time_dim))

        # Final projection: (batch_size, 64, 32, 32) --> (batch_size, 3, 32, 32)
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], in_channels, kernel_size=3, padding=1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        # Zero-initialize the final conv so the model starts by predicting zero noise
        final_conv = self.conv_out[-1]
        assert isinstance(final_conv, nn.Conv2d)
        nn.init.zeros_(final_conv.weight)
        assert final_conv.bias is not None
        nn.init.zeros_(final_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 3, 32, 32), t: (batch_size,) integer timesteps
        t_emb = self.time_embed(t)  # --> (batch_size, time_dim)
        x = self.conv_in(x)  # --> (batch_size, 64, 32, 32)

        # Encoder: store intermediate features for skip connections
        skips: list[torch.Tensor] = []
        for block, down in zip(self.encoder_blocks, self.downsamples):
            x = block(x, t_emb)
            skips.append(x)
            x = down(x)
        # e.g. skips = [(B, 64, 32, 32), (B, 128, 16, 16)], x = (B, 256, 8, 8)

        x = self.bottleneck(x, t_emb)

        # Decoder: upsample and concatenate with skip connections
        for up, block in zip(self.upsamples, self.decoder_blocks):
            x = up(x)
            x = torch.cat([x, skips.pop()], dim=1)  # Double channels via skip
            x = block(x, t_emb)

        return self.conv_out(x)  # --> (batch_size, 3, 32, 32) predicted noise


class GaussianDiffusion:
    """Implements the forward and reverse processes of DDPM.

    Forward process:  q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
    Reverse process:  p(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2 * I)

    The model learns to predict the noise epsilon added at each step.
    """

    def __init__(self, n_timesteps: int = 1000, device: torch.device | None = None) -> None:
        self.n_timesteps = n_timesteps
        self.device = device or torch.device("cpu")

        # Linear noise schedule: beta goes from 1e-4 to 0.02
        self.betas = torch.linspace(1e-4, 0.02, n_timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        # alpha_bar_t = product of alphas from 1 to t (cumulative product)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward process: add noise to x_0 to get x_t.

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        alpha_bar = self.alpha_bars[t]
        # Reshape for broadcasting: (batch_size,) --> (batch_size, 1, 1, 1)
        alpha_bar = alpha_bar[:, None, None, None]

        x_t = alpha_bar.sqrt() * x_0 + (1.0 - alpha_bar).sqrt() * noise
        return x_t, noise

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """Reverse process: denoise x_t by one step to get x_{t-1}.

        mu_theta = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * model(x_t, t))
        """
        batch_size = x_t.shape[0]
        t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

        # Predict noise using the model
        predicted_noise = model(x_t, t_batch)

        alpha = self.alphas[t]
        alpha_bar = self.alpha_bars[t]
        beta = self.betas[t]

        # Compute the mean of p(x_{t-1} | x_t)
        mean = (1.0 / alpha.sqrt()) * (x_t - (beta / (1.0 - alpha_bar).sqrt()) * predicted_noise)

        if t > 0:
            # Add noise for all steps except the final one (t=0)
            noise = torch.randn_like(x_t)
            mean = mean + beta.sqrt() * noise

        return mean

    @torch.no_grad()
    def sample(self, model: nn.Module, shape: tuple[int, ...]) -> torch.Tensor:
        """Generate images by running the full reverse process from x_T ~ N(0, I) to x_0."""
        model.eval()
        # Start from pure Gaussian noise
        x = torch.randn(shape, device=self.device)

        for t in reversed(range(self.n_timesteps)):
            x = self.p_sample(model, x, t)

        # Clamp to valid pixel range
        return x.clamp(-1.0, 1.0)
