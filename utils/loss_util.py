import torch
from torch.nn import functional as F


def foreground_l1(
    prediction: torch.Tensor, target: torch.Tensor, foreground_weight: float
) -> torch.Tensor:
    """L1 loss that gives dark glyph pixels more weight than white background."""
    foreground = ((1.0 - target.float().mean(1, keepdim=True)) / 2.0).clamp(0, 1)
    weights = 1.0 + foreground_weight * foreground
    error = (prediction.float() - target.float()).abs()
    return (weights * error).sum() / weights.expand_as(error).sum()


def gradient_edge_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    operator: str = "scharr",
) -> torch.Tensor:
    """Match horizontal and vertical image gradients with Sobel or Scharr filters."""
    operator = operator.lower()
    if operator == "sobel":
        kernel_values = ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))
        normalization = 8.0
    elif operator == "scharr":
        kernel_values = ((-3.0, 0.0, 3.0), (-10.0, 0.0, 10.0), (-3.0, 0.0, 3.0))
        normalization = 32.0
    else:
        raise ValueError(f"Unknown edge operator {operator!r}; expected 'sobel' or 'scharr'")

    def to_grayscale(images: torch.Tensor) -> torch.Tensor:
        if images.shape[1] == 1:
            return images
        return images.float().mean(dim=1, keepdim=True)

    kernel_x = prediction.new_tensor(kernel_values, dtype=torch.float32).view(1, 1, 3, 3)
    kernel_x = kernel_x / normalization
    kernel_y = kernel_x.transpose(-1, -2)

    def gradients(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        grayscale = to_grayscale(images)
        padded = F.pad(grayscale, (1, 1, 1, 1), mode="replicate")
        return F.conv2d(padded, kernel_x), F.conv2d(padded, kernel_y)

    prediction_x, prediction_y = gradients(prediction)
    target_x, target_y = gradients(target)
    return 0.5 * (F.l1_loss(prediction_x, target_x) + F.l1_loss(prediction_y, target_y))


def style_embedding_loss(
    prediction: torch.Tensor, target: torch.Tensor, kind: str
) -> torch.Tensor:
    cosine = 1.0 - F.cosine_similarity(
        prediction.float(), target.float(), dim=1
    ).mean()
    if kind.lower() == "cosine":
        return cosine
    if kind.lower() in {"mse", "l2"}:
        return F.mse_loss(prediction.float(), target.float())
    if kind.lower() in {"cosine_mse", "cosine+mse"}:
        return cosine + F.mse_loss(prediction.float(), target.float())
    raise ValueError(f"Unknown style loss type: {kind}")


def letter_embedding_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(
        prediction.float(), target.float(), dim=1
    ).mean()


class PerceptualLoss:
    """Content and Gram-style losses over frozen FontStyleViT patch tokens."""

    def __init__(
        self,
        content_kind: str = "l1",
        style_kind: str = "l1",
        content_enabled: bool = True,
        style_enabled: bool = True,
    ) -> None:
        super().__init__()
        supported = {"l1", "mse", "l2", "smooth_l1", "huber"}
        self.content_kind = content_kind.lower()
        self.style_kind = style_kind.lower()
        if self.content_kind not in supported:
            raise ValueError(f"Unknown perceptual content loss: {content_kind}")
        if self.style_kind not in supported:
            raise ValueError(f"Unknown perceptual style loss: {style_kind}")
        self.content_enabled = bool(content_enabled)
        self.style_enabled = bool(style_enabled)

    @staticmethod
    def _elementwise_distance(
        prediction: torch.Tensor, target: torch.Tensor, kind: str
    ) -> torch.Tensor:
        if kind == "l1":
            return (prediction - target).abs()
        if kind in {"mse", "l2"}:
            return (prediction - target).square()
        if kind in {"smooth_l1", "huber"}:
            return F.smooth_l1_loss(prediction, target, reduction="none")
        raise ValueError(f"Elementwise distance does not support {kind!r}")

    @staticmethod
    def _gram_matrix(
        features: torch.Tensor, patch_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        valid = (~patch_padding_mask).unsqueeze(-1)
        masked = features.float() * valid
        valid_count = valid.sum(dim=1, keepdim=True).clamp_min(1)
        return torch.bmm(masked.transpose(1, 2) / valid_count, masked)

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        patch_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(content_loss, style_loss)`` for aligned patch tokens."""
        if patch_padding_mask.shape != prediction.shape[:2]:
            raise ValueError("patch_padding_mask must have shape [B, N]")
        target = target.detach().float()
        prediction = prediction.float()
        valid = ~patch_padding_mask
        if not bool(valid.any()):
            raise ValueError("Perceptual loss received no valid patch tokens")

        content_loss = prediction.new_zeros(())
        if self.content_enabled:
            token_loss = self._elementwise_distance(
                prediction, target, self.content_kind
            ).mean(dim=-1)
            content_loss = token_loss.masked_select(valid).mean()

        style_loss = prediction.new_zeros(())
        if self.style_enabled:
            prediction_gram = self._gram_matrix(prediction, patch_padding_mask)
            target_gram = self._gram_matrix(target, patch_padding_mask)
            style_loss = self._elementwise_distance(
                prediction_gram, target_gram, self.style_kind
            ).mean()
        return content_loss, style_loss


def create_perceptual_loss(
    loss_cfg, perceptual_weight: float, perceptual_style_weight: float
) -> PerceptualLoss:
    return PerceptualLoss(
        content_kind=str(loss_cfg.get("perceptual_content_type", "l1")),
        style_kind=str(loss_cfg.get("perceptual_style_type", "l1")),
        content_enabled=perceptual_weight > 0.0,
        style_enabled=perceptual_style_weight > 0.0,
    )
