"""Evaluate FontStyleViT consistency by randomly splitting input images.

Each input image is split into left/right or top/bottom parts with a seeded,
non-equal random ratio. The script extracts a style embedding for every part,
compares cosine similarities within and across images, and saves charts plus
machine-readable results.

Example:
    CUDA_VISIBLE_DEVICES=1 python tests/evaluate_font_style_similarity.py \
        --checkpoint outputs/font_style_vit_4/checkpoints/latest.pt \
        --images ./data/fonts_dataset/test_data \
        --output-dir outputs/font_style_similarity \
        --weights teacher
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
import yaml
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.font_style_vit import FontStyleViT  # noqa: E402


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Image files, directories, or glob expressions.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "train_font_style.yaml",
        help="Fallback model/preprocessing config when it is absent from the checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "font_style_similarity",
    )
    parser.add_argument(
        "--weights",
        choices=("auto", "teacher", "student"),
        default="auto",
        help="For a training checkpoint, auto prefers EMA teacher weights.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--split-ratio-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(0.35, 0.65),
    )
    parser.add_argument(
        "--max-long-side",
        type=int,
        default=None,
        help="Override dataset.global_view.max_long_side.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Override dataset.global_view.max_pixels.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return dict(data) if isinstance(data, Mapping) else {}


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was added.
        return torch.load(path, map_location="cpu")


def select_state_dict(
    checkpoint: Any,
    requested_weights: str,
) -> Tuple[Mapping[str, torch.Tensor], str]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a state dict or a mapping containing model weights")

    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint, "state_dict"

    if "model" in checkpoint and isinstance(checkpoint["model"], Mapping):
        return checkpoint["model"], "model"

    if requested_weights == "teacher":
        candidates = ("teacher",)
    elif requested_weights == "student":
        candidates = ("student",)
    else:
        candidates = ("teacher", "student", "state_dict")
    for name in candidates:
        state = checkpoint.get(name)
        if isinstance(state, Mapping):
            return state, name
    raise KeyError(
        f"Could not find {requested_weights!r} weights; available keys: "
        f"{sorted(str(key) for key in checkpoint.keys())}"
    )


def clean_state_dict(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        name = str(key)
        for prefix in ("module.", "_orig_mod."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        cleaned[name] = value
    return cleaned


def resolve_model_config(
    checkpoint: Mapping[str, Any],
    external_config: Mapping[str, Any],
) -> Dict[str, Any]:
    candidates = [
        checkpoint.get("model_config"),
        checkpoint.get("config", {}).get("model")
        if isinstance(checkpoint.get("config"), Mapping)
        else None,
        external_config.get("model"),
    ]
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            return dict(candidate)
    raise KeyError("Model config was not found in either checkpoint or --config")


def resolve_dataset_config(
    checkpoint: Mapping[str, Any],
    external_config: Mapping[str, Any],
) -> Dict[str, Any]:
    embedded = checkpoint.get("config")
    if isinstance(embedded, Mapping) and isinstance(embedded.get("dataset"), Mapping):
        return dict(embedded["dataset"])
    dataset = external_config.get("dataset")
    return dict(dataset) if isinstance(dataset, Mapping) else {}


def load_model(
    checkpoint_path: Path,
    config_path: Path,
    requested_weights: str,
    device: torch.device,
) -> Tuple[FontStyleViT, Dict[str, Any], Dict[str, Any], str]:
    checkpoint = torch_load(checkpoint_path)
    external_config = load_yaml(config_path)
    checkpoint_mapping = checkpoint if isinstance(checkpoint, Mapping) else {}
    model_config = resolve_model_config(checkpoint_mapping, external_config)
    dataset_config = resolve_dataset_config(checkpoint_mapping, external_config)
    state, state_name = select_state_dict(checkpoint, requested_weights)

    model = FontStyleViT(**model_config)
    missing, unexpected = model.load_state_dict(clean_state_dict(state), strict=False)
    critical_missing = [
        name for name in missing if name.startswith("backbone.") or name.startswith("style_head.")
    ]
    if critical_missing:
        raise RuntimeError(f"Checkpoint is missing inference parameters: {critical_missing}")
    if unexpected:
        print(f"Warning: ignored {len(unexpected)} unexpected checkpoint keys")
    model.to(device).eval()
    return model, model_config, dataset_config, state_name


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def collect_image_paths(values: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for value in values:
        path = Path(value).expanduser()
        if path.is_dir():
            paths.extend(
                item for item in sorted(path.rglob("*")) if item.suffix.lower() in IMAGE_EXTENSIONS
            )
        elif path.is_file():
            paths.append(path)
        else:
            paths.extend(
                Path(item) for item in sorted(glob.glob(value, recursive=True))
                if Path(item).is_file() and Path(item).suffix.lower() in IMAGE_EXTENSIONS
            )

    unique: List[Path] = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved.suffix.lower() not in IMAGE_EXTENSIONS or resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    if len(unique) < 2:
        raise ValueError(f"At least two readable images are required, found {len(unique)}")
    return unique


def open_grayscale(path: Path) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba)
        return image.convert("L")


def limit_resolution(
    image: Image.Image,
    patch_size: int,
    max_long_side: int,
    max_pixels: int,
) -> Image.Image:
    """Match the dataset's patch-aligned resolution limiting behavior."""
    width, height = image.size
    scale = min(
        1.0,
        max_long_side / max(width, height),
        math.sqrt(max_pixels / max(1, width * height)),
    )
    width = max(patch_size, int(width * scale))
    height = max(patch_size, int(height * scale))
    width = max(patch_size, round(width / patch_size) * patch_size)
    height = max(patch_size, round(height / patch_size) * patch_size)
    aligned_max = max(patch_size, max_long_side // patch_size * patch_size)
    width, height = min(width, aligned_max), min(height, aligned_max)
    while width * height > max_pixels:
        if width >= height and width > patch_size:
            width -= patch_size
        elif height > patch_size:
            height -= patch_size
        else:
            break
    return image.resize((width, height), Image.Resampling.LANCZOS)


def choose_non_equal_cut(length: int, ratio: float, minimum: int) -> int:
    if length < 2 * minimum:
        raise ValueError(f"Split dimension {length} is smaller than two {minimum}px patches")
    cut = max(minimum, min(length - minimum, round(length * ratio)))
    if cut * 2 == length:
        cut = cut + 1 if cut + 1 <= length - minimum else cut - 1
    return cut


def split_image(
    image: Image.Image,
    rng: random.Random,
    ratio_range: Tuple[float, float],
    patch_size: int,
) -> Tuple[Image.Image, Image.Image, Dict[str, Any]]:
    candidates = []
    if image.width >= 2 * patch_size:
        candidates.append("left_right")
    if image.height >= 2 * patch_size:
        candidates.append("top_bottom")
    if not candidates:
        raise ValueError(
            f"Image size {image.size} is too small to create two patch-sized regions"
        )

    orientation = rng.choice(candidates)
    requested_ratio = rng.uniform(*ratio_range)
    if orientation == "left_right":
        cut = choose_non_equal_cut(image.width, requested_ratio, patch_size)
        first = image.crop((0, 0, cut, image.height))
        second = image.crop((cut, 0, image.width, image.height))
        actual_ratio = cut / image.width
    else:
        cut = choose_non_equal_cut(image.height, requested_ratio, patch_size)
        first = image.crop((0, 0, image.width, cut))
        second = image.crop((0, cut, image.width, image.height))
        actual_ratio = cut / image.height
    return first, second, {
        "orientation": orientation,
        "cut": cut,
        "requested_ratio": requested_ratio,
        "actual_ratio": actual_ratio,
    }


def image_to_tensor(image: Image.Image, mean: float, std: float) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return (torch.from_numpy(array).unsqueeze(0) - mean) / std


def patchify(image: torch.Tensor, patch_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    if image.ndim != 3 or image.shape[0] != 1:
        raise ValueError(f"Expected grayscale [1, H, W], got {tuple(image.shape)}")
    if image.shape[-2] % patch_size or image.shape[-1] % patch_size:
        raise ValueError("Preprocessed image dimensions must be divisible by patch_size")
    grid_h = image.shape[-2] // patch_size
    grid_w = image.shape[-1] // patch_size
    patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).reshape(grid_h * grid_w, -1)
    return patches.contiguous(), (grid_h, grid_w)


def collate_images(images: Sequence[torch.Tensor], patch_size: int) -> Dict[str, torch.Tensor]:
    patch_data = [patchify(image, patch_size) for image in images]
    counts = [patches.shape[0] for patches, _ in patch_data]
    max_tokens = max(counts)
    patch_dim = patch_data[0][0].shape[1]
    padded = torch.zeros(len(images), max_tokens, patch_dim, dtype=torch.float32)
    padding_mask = torch.ones(len(images), max_tokens, dtype=torch.bool)
    for index, ((patches, _), count) in enumerate(zip(patch_data, counts)):
        padded[index, :count] = patches
        padding_mask[index, :count] = False
    return {
        "patches": padded,
        "patch_padding_mask": padding_mask,
        "grid_sizes": torch.tensor([grid for _, grid in patch_data], dtype=torch.long),
    }


def move_view(view: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {name: tensor.to(device, non_blocking=True) for name, tensor in view.items()}


@torch.inference_mode()
def extract_embeddings(
    model: FontStyleViT,
    images: Sequence[torch.Tensor],
    patch_size: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    embeddings = []
    for start in range(0, len(images), batch_size):
        view = collate_images(images[start : start + batch_size], patch_size)
        output = model.forward_view(move_view(view, device), use_ibot_mask=False)
        embeddings.append(output["style_embedding"].cpu())
    return F.normalize(torch.cat(embeddings, dim=0).float(), dim=-1)


def save_similarity_csv(path: Path, labels: Sequence[str], matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["part"] + list(labels))
        for label, row in zip(labels, matrix):
            writer.writerow([label] + [f"{value:.8f}" for value in row])


def save_embeddings_csv(
    path: Path,
    labels: Sequence[str],
    image_indices: Sequence[int],
    embeddings: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["part", "image_index"] + [f"embedding_{i:03d}" for i in range(embeddings.shape[1])])
        for label, image_index, embedding in zip(labels, image_indices, embeddings):
            writer.writerow([label, image_index] + [f"{value:.8f}" for value in embedding])


def plot_split_previews(records: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    rows = len(records)
    figure, axes = plt.subplots(rows, 3, figsize=(13, max(3.2, 3.2 * rows)), squeeze=False)
    for row, record in enumerate(records):
        original = record["original"]
        first, second = record["parts"]
        split = record["split"]
        axes[row, 0].imshow(original, cmap="gray", vmin=0, vmax=255)
        if split["orientation"] == "left_right":
            axes[row, 0].axvline(split["cut"], color="red", linewidth=2)
            names = ("A · left", "B · right")
        else:
            axes[row, 0].axhline(split["cut"], color="red", linewidth=2)
            names = ("A · top", "B · bottom")
        axes[row, 0].set_title(
            f"{record['display_name']} · {split['orientation']} · "
            f"{split['actual_ratio']:.1%}/{1.0 - split['actual_ratio']:.1%}"
        )
        axes[row, 1].imshow(first, cmap="gray", vmin=0, vmax=255)
        axes[row, 1].set_title(f"{names[0]} · {first.width}×{first.height}")
        axes[row, 2].imshow(second, cmap="gray", vmin=0, vmax=255)
        axes[row, 2].set_title(f"{names[1]} · {second.width}×{second.height}")
        for axis in axes[row]:
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_similarity_heatmap(
    matrix: np.ndarray,
    labels: Sequence[str],
    output_path: Path,
) -> None:
    count = len(labels)
    size = max(8.0, min(18.0, 4.0 + count * 0.62))
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axis.set_xticks(np.arange(count), labels=labels, rotation=55, ha="right")
    axis.set_yticks(np.arange(count), labels=labels)
    axis.set_title("Cosine similarity of split-image style embeddings")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("cosine similarity")

    if count <= 20:
        for row in range(count):
            for column in range(count):
                value = matrix[row, column]
                color = "white" if abs(value) > 0.55 else "black"
                axis.text(column, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=7)
    for image_index in range(count // 2):
        axis.add_patch(
            Rectangle(
                (2 * image_index - 0.5, 2 * image_index - 0.5),
                2,
                2,
                fill=False,
                edgecolor="lime",
                linewidth=1.8,
            )
        )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def calculate_metrics(
    matrix: np.ndarray,
    image_indices: Sequence[int],
    image_names: Sequence[str],
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    intra = np.asarray([matrix[2 * index, 2 * index + 1] for index in range(len(image_names))])
    inter_values = []
    for left in range(matrix.shape[0]):
        for right in range(left + 1, matrix.shape[0]):
            if image_indices[left] != image_indices[right]:
                inter_values.append(matrix[left, right])
    inter = np.asarray(inter_values, dtype=np.float32)

    nearest_correct = 0
    for index in range(matrix.shape[0]):
        candidates = matrix[index].copy()
        candidates[index] = -np.inf
        nearest = int(np.argmax(candidates))
        nearest_correct += int(image_indices[nearest] == image_indices[index])
    retrieval_accuracy = nearest_correct / matrix.shape[0]

    metrics = {
        "same_image_half_similarity": {
            name: float(value) for name, value in zip(image_names, intra)
        },
        "same_image_mean": float(intra.mean()),
        "same_image_std": float(intra.std()),
        "cross_image_mean": float(inter.mean()),
        "cross_image_std": float(inter.std()),
        "cross_image_min": float(inter.min()),
        "cross_image_max": float(inter.max()),
        "mean_similarity_margin": float(intra.mean() - inter.mean()),
        "part_retrieval_top1_accuracy": float(retrieval_accuracy),
        "part_count": int(matrix.shape[0]),
        "cross_image_pair_count": int(inter.size),
    }
    return metrics, intra, inter


def plot_similarity_summary(
    intra: np.ndarray,
    inter: np.ndarray,
    image_names: Sequence[str],
    metrics: Mapping[str, Any],
    output_path: Path,
    seed: int,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    positions = np.arange(len(image_names))
    axes[0].bar(positions, intra, color="#4C78A8")
    axes[0].axhline(float(inter.mean()), color="#E45756", linestyle="--", label="cross-image mean")
    axes[0].set_xticks(positions, image_names, rotation=35, ha="right")
    axes[0].set_ylim(-1.0, 1.0)
    axes[0].set_ylabel("cosine similarity")
    axes[0].set_title("Similarity between two parts of each image")
    axes[0].legend()
    for position, value in zip(positions, intra):
        axes[0].text(position, min(0.96, value + 0.035), f"{value:.3f}", ha="center", fontsize=8)

    boxplot_data = [intra, inter]
    boxplot_labels = ["same image", "different images"]
    try:
        box = axes[1].boxplot(
            boxplot_data,
            tick_labels=boxplot_labels,
            patch_artist=True,
            showmeans=True,
        )
    except TypeError as error:
        # Matplotlib < 3.9 used ``labels`` instead of ``tick_labels``.
        if "tick_labels" not in str(error):
            raise
        box = axes[1].boxplot(
            boxplot_data,
            labels=boxplot_labels,
            patch_artist=True,
            showmeans=True,
        )
    for patch, color in zip(box["boxes"], ("#72B7B2", "#F58518")):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    jitter_rng = np.random.default_rng(seed)
    for x, values, color in ((1, intra, "#2A6F6B"), (2, inter, "#A34F00")):
        sampled = values
        if len(sampled) > 500:
            sampled = jitter_rng.choice(sampled, size=500, replace=False)
        jitter = jitter_rng.normal(0.0, 0.035, size=len(sampled))
        axes[1].scatter(np.full(len(sampled), x) + jitter, sampled, s=12, alpha=0.35, color=color)
    axes[1].set_ylim(-1.0, 1.0)
    axes[1].set_ylabel("cosine similarity")
    axes[1].set_title(
        f"Distributions · margin={metrics['mean_similarity_margin']:.3f} · "
        f"retrieval@1={metrics['part_retrieval_top1_accuracy']:.1%}"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    ratio_min, ratio_max = (float(value) for value in args.split_ratio_range)
    if not 0.0 < ratio_min <= ratio_max < 1.0:
        raise ValueError("--split-ratio-range must satisfy 0 < MIN <= MAX < 1")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    device = choose_device(args.device)
    image_paths = collect_image_paths(args.images)
    model, model_config, dataset_config, state_name = load_model(
        args.checkpoint.resolve(), args.config.resolve(), args.weights, device
    )
    patch_size = int(model_config.get("patch_size", 16))
    global_profile = dataset_config.get("global_view", {})
    max_long_side = int(
        args.max_long_side
        or global_profile.get("max_long_side", dataset_config.get("max_long_side", 512))
    )
    max_pixels = int(
        args.max_pixels
        or global_profile.get("max_pixels", dataset_config.get("max_pixels", 262_144))
    )
    mean = float(dataset_config.get("mean", 0.5))
    std = float(dataset_config.get("std", 0.5))
    if std <= 0 or max_long_side < patch_size or max_pixels < patch_size**2:
        raise ValueError("Invalid preprocessing values in dataset config")

    rng = random.Random(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    tensors: List[torch.Tensor] = []
    labels: List[str] = []
    image_indices: List[int] = []
    image_names: List[str] = []

    for image_index, path in enumerate(image_paths):
        display_name = f"{image_index:02d}_{path.stem}"
        image_names.append(display_name)
        original = limit_resolution(
            open_grayscale(path), patch_size, max_long_side, max_pixels
        )
        first, second, split = split_image(
            original, rng, (ratio_min, ratio_max), patch_size
        )
        prepared_parts = tuple(
            limit_resolution(part, patch_size, max_long_side, max_pixels)
            for part in (first, second)
        )
        records.append(
            {
                "path": path,
                "display_name": display_name,
                "original": original,
                "parts": prepared_parts,
                "split": split,
            }
        )
        for part_name, part in zip(("A", "B"), prepared_parts):
            tensors.append(image_to_tensor(part, mean, std))
            labels.append(f"{display_name}/{part_name}")
            image_indices.append(image_index)

    embeddings_tensor = extract_embeddings(
        model, tensors, patch_size, args.batch_size, device
    )
    embeddings = embeddings_tensor.numpy()
    similarity = embeddings @ embeddings.T
    metrics, intra, inter = calculate_metrics(similarity, image_indices, image_names)
    metrics.update(
        {
            "checkpoint": str(args.checkpoint.resolve()),
            "weights": state_name,
            "device": str(device),
            "seed": args.seed,
            "split_ratio_range": [ratio_min, ratio_max],
            "mean": mean,
            "std": std,
            "patch_size": patch_size,
            "max_long_side": max_long_side,
            "max_pixels": max_pixels,
            "images": [
                {
                    "index": index,
                    "path": str(record["path"]),
                    "split": record["split"],
                    "part_sizes": [list(part.size) for part in record["parts"]],
                }
                for index, record in enumerate(records)
            ],
        }
    )

    plot_split_previews(records, output_dir / "split_previews.png")
    plot_similarity_heatmap(similarity, labels, output_dir / "similarity_heatmap.png")
    plot_similarity_summary(
        intra,
        inter,
        image_names,
        metrics,
        output_dir / "similarity_summary.png",
        args.seed,
    )
    save_similarity_csv(output_dir / "similarity_matrix.csv", labels, similarity)
    save_embeddings_csv(output_dir / "embeddings.csv", labels, image_indices, embeddings)
    np.save(output_dir / "embeddings.npy", embeddings)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print(f"Loaded {state_name} weights from: {args.checkpoint.resolve()}")
    print(f"Same-image mean similarity:  {metrics['same_image_mean']:.6f}")
    print(f"Cross-image mean similarity: {metrics['cross_image_mean']:.6f}")
    print(f"Mean similarity margin:      {metrics['mean_similarity_margin']:.6f}")
    print(f"Part retrieval top-1:        {metrics['part_retrieval_top1_accuracy']:.2%}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
