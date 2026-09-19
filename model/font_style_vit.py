"""Variable-length Vision Transformer for 128-D font-style embeddings."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep_probability = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_probability + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x * random_tensor.floor() / keep_probability


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(self.activation(self.fc1(x)))
        return self.dropout2(self.fc2(x))


class TransformerBlock(nn.Module):
    """Pre-norm ViT block that explicitly zeros padded query positions."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path: float = 0.0,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=attention_dropout,
            bias=qkv_bias,
            batch_first=True,
        )
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, round(dim * mlp_ratio), dropout)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop_path1(attended)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)


class VariableResolutionViT(nn.Module):
    """ViT backbone for compact, end-padded patch sequences.

    The learned 2-D position map is interpolated separately for each original
    patch grid and then end-padded to the batch token length. This preserves the
    correct positions for mixed aspect ratios without constructing a shared
    image canvas.
    """

    def __init__(
        self,
        patch_dim: int = 256,
        embed_dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        base_grid_size: Sequence[int] = (32, 32),
    ) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if len(base_grid_size) != 2 or min(base_grid_size) <= 0:
            raise ValueError("base_grid_size must contain two positive values")
        self.patch_dim = int(patch_dim)
        self.embed_dim = int(embed_dim)
        self.base_grid_size = (int(base_grid_size[0]), int(base_grid_size[1]))

        self.patch_projection = nn.Linear(self.patch_dim, self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.cls_position = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.patch_position = nn.Parameter(
            torch.zeros(1, self.embed_dim, self.base_grid_size[0], self.base_grid_size[1])
        )
        drop_path_rates = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            TransformerBlock(
                dim=self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
                drop_path=drop_path_rates[index],
                qkv_bias=qkv_bias,
            )
            for index in range(depth)
        )
        self.norm = nn.LayerNorm(self.embed_dim)
        self.apply(self._initialize_module)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.cls_position, std=0.02)
        nn.init.trunc_normal_(self.patch_position, std=0.02)

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _interpolated_positions(
        self,
        grid_sizes: torch.Tensor,
        token_count: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        rows = []
        position_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        for grid_height, grid_width in grid_sizes.detach().cpu().tolist():
            grid = (int(grid_height), int(grid_width))
            count = grid[0] * grid[1]
            if count > token_count:
                raise ValueError(f"Grid {grid} needs {count} patches, batch has {token_count}")
            if grid not in position_cache:
                position_map = F.interpolate(
                    self.patch_position.float(),
                    size=grid,
                    mode="bicubic",
                    align_corners=False,
                )
                position_cache[grid] = position_map.flatten(2).transpose(1, 2).squeeze(0)
            row = position_cache[grid].to(dtype=dtype)
            rows.append(F.pad(row, (0, 0, 0, token_count - count)))
        return torch.stack(rows)

    def forward(
        self,
        patches: torch.Tensor,
        patch_padding_mask: torch.Tensor,
        grid_sizes: torch.Tensor,
        ibot_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if patches.ndim != 3 or patches.shape[-1] != self.patch_dim:
            raise ValueError(
                f"Expected patches [B, N, {self.patch_dim}], got {tuple(patches.shape)}"
            )
        if patch_padding_mask.shape != patches.shape[:2]:
            raise ValueError("patch_padding_mask must have shape [B, N]")
        if grid_sizes.shape != (patches.shape[0], 2):
            raise ValueError("grid_sizes must have shape [B, 2]")

        patch_tokens = self.patch_projection(patches)
        if ibot_mask is not None:
            if ibot_mask.shape != patch_padding_mask.shape:
                raise ValueError("ibot_mask must have shape [B, N]")
            effective_mask = ibot_mask & ~patch_padding_mask
            mask_tokens = self.mask_token.expand(patches.shape[0], patches.shape[1], -1)
            patch_tokens = torch.where(effective_mask.unsqueeze(-1), mask_tokens, patch_tokens)
        patch_tokens = patch_tokens.masked_fill(patch_padding_mask.unsqueeze(-1), 0.0)

        cls_tokens = self.cls_token.expand(patches.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, patch_tokens), dim=1)
        patch_positions = self._interpolated_positions(
            grid_sizes, patches.shape[1], patch_tokens.dtype
        )
        cls_positions = self.cls_position.expand(patches.shape[0], -1, -1)
        tokens = tokens + torch.cat((cls_positions, patch_positions), dim=1)
        key_padding_mask = F.pad(patch_padding_mask, (1, 0), value=False)
        for block in self.blocks:
            tokens = block(tokens, key_padding_mask)
        tokens = self.norm(tokens)
        patch_features = tokens[:, 1:].masked_fill(patch_padding_mask.unsqueeze(-1), 0.0)
        return tokens[:, 0], patch_features


class ProjectionHead(nn.Module):
    """DINO-style MLP followed by a bias-free prototype layer."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 1024,
        bottleneck_dim: int = 256,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        dimensions = [input_dim] + [hidden_dim] * max(0, num_layers - 1) + [bottleneck_dim]
        layers = []
        for index in range(len(dimensions) - 1):
            layers.append(nn.Linear(dimensions[index], dimensions[index + 1]))
            if index < len(dimensions) - 2:
                layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.Linear(bottleneck_dim, output_dim, bias=False)
        self.apply(VariableResolutionViT._initialize_module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, eps=1e-6)
        return self.last_layer(x)


class FontStyleViT(nn.Module):
    """Student/teacher network used by MoCo, DINO and iBOT objectives."""

    def __init__(
        self,
        patch_size: int = 16,
        input_channels: int = 1,
        embed_dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        base_grid_size: Sequence[int] = (32, 32),
        style_dim: int = 128,
        head_hidden_dim: int = 1024,
        head_bottleneck_dim: int = 256,
        head_num_layers: int = 3,
        dino_out_dim: int = 4096,
        ibot_out_dim: int = 4096,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.style_dim = int(style_dim)
        patch_dim = input_channels * self.patch_size * self.patch_size
        self.backbone = VariableResolutionViT(
            patch_dim=patch_dim,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            dropout=dropout,
            attention_dropout=attention_dropout,
            drop_path_rate=drop_path_rate,
            base_grid_size=base_grid_size,
        )
        self.style_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, self.style_dim),
        )
        self.dino_head = ProjectionHead(
            embed_dim,
            dino_out_dim,
            head_hidden_dim,
            head_bottleneck_dim,
            head_num_layers,
        )
        self.ibot_head = ProjectionHead(
            embed_dim,
            ibot_out_dim,
            head_hidden_dim,
            head_bottleneck_dim,
            head_num_layers,
        )
        self.style_head.apply(VariableResolutionViT._initialize_module)

    def encode_view(
        self,
        view: Mapping[str, Any],
        use_ibot_mask: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Encode a view without running the self-supervised prototype heads."""
        cls_feature, patch_features = self.backbone(
            patches=view["patches"],
            patch_padding_mask=view["patch_padding_mask"],
            grid_sizes=view["grid_sizes"],
            ibot_mask=view.get("ibot_mask") if use_ibot_mask else None,
        )
        style_embedding = F.normalize(self.style_head(cls_feature), dim=-1, eps=1e-6)
        return {
            "style_embedding": style_embedding,
            "cls_feature": cls_feature,
            "patch_features": patch_features,
        }

    def forward_view(
        self,
        view: Mapping[str, Any],
        use_ibot_mask: bool,
    ) -> Dict[str, torch.Tensor]:
        result = self.encode_view(view, use_ibot_mask=use_ibot_mask)
        result["dino_logits"] = self.dino_head(result["cls_feature"])
        result["ibot_logits"] = self.ibot_head(result["patch_features"])
        return result

    def forward(self, view: Mapping[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        result = self.encode_view(view)
        return result["cls_feature"], result["style_embedding"]

    def cancel_last_layer_gradients(self) -> None:
        """Freeze unstable prototype layers during the first training epochs."""
        for layer in (self.dino_head.last_layer, self.ibot_head.last_layer):
            for parameter in layer.parameters():
                parameter.grad = None
