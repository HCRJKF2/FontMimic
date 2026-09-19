"""Style-and-letter conditional GAN modules for 256x256 glyph generation."""

from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(32, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class StyleAffine(nn.Module):
    """Apply FiLM modulation predicted from an external style embedding."""

    def __init__(self, style_embedding_dim: int, channels: int) -> None:
        super().__init__()
        self.style_embedding_dim = int(style_embedding_dim)
        self.channels = int(channels)
        self.affine = nn.Linear(self.style_embedding_dim, self.channels * 2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Identity modulation at initialization: scale=0 and bias=0.
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(
            self, features: torch.Tensor, style_embeddings: torch.Tensor
    ) -> torch.Tensor:
        if style_embeddings.ndim != 2 or style_embeddings.shape != (
                features.shape[0],
                self.style_embedding_dim,
        ):
            raise ValueError(
                "style_embeddings must have shape "
                f"[B, {self.style_embedding_dim}], got {tuple(style_embeddings.shape)}"
            )
        scale, bias = self.affine(style_embeddings).chunk(2, dim=1)
        scale = scale.view(features.shape[0], self.channels, 1, 1)
        bias = bias.view(features.shape[0], self.channels, 1, 1)
        return features * (1.0 + scale) + bias


class PosPredictor(nn.Module):
    """Predict a valid normalized square position from style and letter.

    The three outputs are ``(side, center_x, center_y)`` relative to the original
    image size. The center coordinates are constrained by the predicted side, so
    the corresponding square always remains inside the image.
    """

    def __init__(
            self,
            style_embedding_dim: int,
            letter_embedding_dim: int,
            hidden_dims: Sequence[int] = (256, 256, 128, 128),
            dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.style_embedding_dim = int(style_embedding_dim)
        self.letter_embedding_dim = int(letter_embedding_dim)
        hidden_dims = tuple(int(dim) for dim in hidden_dims)

        input_dim = self.style_embedding_dim + self.letter_embedding_dim
        layers: list[nn.Module] = []
        for output_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, output_dim), nn.LayerNorm(output_dim), nn.SiLU()))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = output_dim
        layers.append(nn.Linear(input_dim, 3))
        self.network = nn.Sequential(*layers)

    def forward(
            self,
            style_embeddings: torch.Tensor,
            letter_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        raw_position = self.network(
            torch.cat((style_embeddings, letter_embeddings), dim=1)
        )
        side = torch.sigmoid(raw_position[:, :1])
        center_unit = torch.sigmoid(raw_position[:, 1:])
        centers = side * 0.5 + center_unit * (1.0 - side)
        return torch.cat((side, centers), dim=1)


class GeneratorBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            style_embedding_dim: int | None = None,
            up_scale: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = _group_norm(in_channels)
        self.norm2 = _group_norm(out_channels)
        self.style_affine1 = (
            StyleAffine(style_embedding_dim, in_channels)
            if style_embedding_dim is not None
            else None
        )
        self.style_affine2 = (
            StyleAffine(style_embedding_dim, out_channels)
            if style_embedding_dim is not None
            else None
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.scale_factor = 2 if up_scale else 1

    def forward(
            self, x: torch.Tensor, style_embeddings: torch.Tensor | None = None
    ) -> torch.Tensor:
        residual = self.skip(F.interpolate(x, scale_factor=self.scale_factor, mode="nearest"))
        x = self.norm1(x)
        if self.style_affine1 is not None:
            if style_embeddings is None:
                raise ValueError("style_embeddings are required by style-affine blocks")
            x = self.style_affine1(x, style_embeddings)
        x = F.interpolate(F.silu(x, inplace=True), scale_factor=self.scale_factor, mode="nearest")
        x = self.conv1(x)
        x = self.norm2(x)
        if self.style_affine2 is not None:
            if style_embeddings is None:
                raise ValueError("style_embeddings are required by style-affine blocks")
            x = self.style_affine2(x, style_embeddings)
        x = self.conv2(F.silu(x, inplace=True))
        return x + residual


class ConditionalGenerator(nn.Module):
    """Map noise, an external style embedding and a letter id to a glyph.

    Style is produced from a reference image by a frozen FontStyleViT. The 52
    letter embeddings remain generator-owned, learnable parameters. When
    use_style_affine is enabled, every normalized activation is additionally
    modulated by style-dependent per-channel scale and bias.

    When ``predicte_pos_args`` is a mapping, a :class:`PosPredictor` is created.
    In that mode ``forward`` returns ``(image, position)``; otherwise it retains
    the legacy image-only return value.
    """

    def __init__(
            self,
            num_letters: int = 52,
            latent_dim: int = 128,
            style_embedding_dim: int = 128,
            letter_embedding_dim: int = 64,
            channels: Sequence[int] = (512, 512, 256, 128, 64, 32, 16),
            ups: Sequence[bool] = (True, True, True, True, True, True),
            image_channels: int = 3,
            use_style_affine: bool = True,
            letter_affine_layers: Sequence[int] = (),
            predict_pos_args: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.style_embedding_dim = style_embedding_dim
        self.use_style_affine = use_style_affine
        self.letter_affine_layers = letter_affine_layers

        self.letter_embedding = nn.Embedding(num_letters, letter_embedding_dim)
        self.input = nn.Linear(
            self.latent_dim + style_embedding_dim + letter_embedding_dim,
            channels[0] * 16,
        )
        self.blocks = nn.ModuleList(
            GeneratorBlock(
                in_channels,
                out_channels,
                (style_embedding_dim + letter_embedding_dim
                 if idx in self.letter_affine_layers
                 else style_embedding_dim)
                if use_style_affine else None,
                up,
            )
            for idx, (in_channels, out_channels, up) in enumerate(zip(channels[:-1], channels[1:], ups))
        )
        self.output_norm = _group_norm(channels[-1])
        self.output_style_affine = (
            StyleAffine(style_embedding_dim, channels[-1])
            if use_style_affine
            else None
        )
        self.to_rgb = nn.Conv2d(channels[-1], image_channels, 3, padding=1)

        self.pos_predictor = (
            PosPredictor(
                style_embedding_dim=style_embedding_dim,
                letter_embedding_dim=letter_embedding_dim,
                **dict(predict_pos_args),
            )
            if predict_pos_args is not None
            else None
        )

        # self.apply initializes Linear layers normally. Restore identity
        # initialization specifically for all style modulation layers.
        self.apply(self._init_weights)
        for module in self.modules():
            if isinstance(module, StyleAffine):
                module.reset_parameters()
        nn.init.orthogonal_(self.letter_embedding.weight)
        self.letter_embedding.requires_grad_(False)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(
            self,
            noise: torch.Tensor,
            style_embeddings: torch.Tensor,
            letter_ids: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if style_embeddings.ndim != 2:
            raise ValueError(
                f"style_embeddings must have shape [B, D], got {tuple(style_embeddings.shape)}"
            )
        letter_embeddings = self.letter_embedding(letter_ids)
        condition = torch.cat(
            (noise, style_embeddings, letter_embeddings), dim=1
        )
        x = self.input(condition).view(noise.shape[0], -1, 4, 4)
        for idx, block in enumerate(self.blocks):
            cur_style_embedding = style_embeddings
            if idx in self.letter_affine_layers:
                cur_style_embedding = torch.cat([style_embeddings, letter_embeddings], dim=1)
            x = block(x, cur_style_embedding)
        x = self.output_norm(x)
        if self.output_style_affine is not None:
            x = self.output_style_affine(x, style_embeddings)
        image = torch.tanh(self.to_rgb(F.silu(x, inplace=True)))
        if self.pos_predictor is None:
            return image
        positions = self.pos_predictor(style_embeddings, letter_embeddings)
        return image, positions


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        self.conv2 = spectral_norm(
            nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1)
        )
        self.skip = spectral_norm(nn.Conv2d(in_channels, out_channels, 1, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.avg_pool2d(self.skip(x), 2)
        x = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
        x = self.conv2(x)
        return F.leaky_relu(x + residual, 0.2, inplace=True)


class ConditionalDiscriminator(nn.Module):
    """Predict real/fake and the generator-owned letter embedding.

    Style is supervised through the frozen FontStyleViT rather than a second,
    discriminator-owned style embedding space.
    """

    def __init__(
            self,
            channels: Sequence[int] = (16, 32, 64, 128, 256, 512, 512),
            image_channels: int = 3,
    ) -> None:
        super().__init__()
        # if len(channels) != 7:
        #     raise ValueError(
        #         "Discriminator channels must contain 7 values for 256x256 -> 4x4"
        #     )
        self.from_rgb = spectral_norm(
            nn.Conv2d(image_channels, channels[0], 3, padding=1)
        )
        self.blocks = nn.ModuleList(
            DiscriminatorBlock(a, b) for a, b in zip(channels[:-1], channels[1:])
        )
        self.final = spectral_norm(nn.Conv2d(channels[-1], channels[-1], 4))
        self.unconditional = spectral_norm(nn.Linear(channels[-1], 1))

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.leaky_relu(self.from_rgb(image), 0.2, inplace=True)
        for block in self.blocks:
            x = block(x)
        features = F.leaky_relu(self.final(x), 0.2, inplace=True).flatten(1)
        score = self.unconditional(features).squeeze(1)
        return score
