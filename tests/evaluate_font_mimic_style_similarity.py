"""Compare reference embeddings with real and GAN-generated glyph grids.

Examples:
    python tests/evaluate_font_mimic_style_similarity.py --device cuda:0
    python tests/evaluate_font_mimic_style_similarity.py --samples 128
    python tests/evaluate_font_mimic_style_similarity.py --gan-checkpoint outputs/font_cgan/font_generator.pt
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.font_mimic_dataset import FontMimicCollator, FontMimicDataset  # noqa: E402
from model import ConditionalGenerator  # noqa: E402
from train import (  # noqa: E402
    generated_grid_style_view,
    load_frozen_style_encoder,
    make_grouped_grid,
    move_style_view,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "train.yaml",
    )
    parser.add_argument("--font-root", type=Path, default=None)
    parser.add_argument(
        "--style-checkpoint",
        type=Path,
        default=None,
        help="Optional override for model.style_encoder.checkpoint.",
    )
    parser.add_argument(
        "--gan-checkpoint",
        type=Path,
        default=None,
        help="Defaults to train.output_dir/checkpoints/latest.pt.",
    )
    parser.add_argument(
        "--gan-weights",
        choices=("auto", "generator_ema", "generator"),
        default="auto",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "font_mimic_style_similarity",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=64,
        help="Number of samples; use 0 for the complete dataset.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--heatmap-limit", type=int, default=100)
    parser.add_argument(
        "--save-grids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save real-left/generated-right PNG comparisons.",
    )
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def resolve_path(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    config = copy.deepcopy(dict(loaded))
    if not isinstance(config.get("dataset"), Mapping):
        raise ValueError("Config must contain a dataset mapping")
    if not isinstance(config.get("model"), Mapping):
        raise ValueError("Config must contain a model mapping")

    if args.font_root is not None:
        config["dataset"]["font_root_path"] = str(args.font_root.expanduser().resolve())
    else:
        config["dataset"]["font_root_path"] = str(
            resolve_path(config["dataset"]["font_root_path"])
        )

    style_options = config["model"].get("style_encoder")
    if not isinstance(style_options, Mapping):
        raise ValueError("Config must contain model.style_encoder")
    if args.style_checkpoint is not None:
        style_options["checkpoint"] = str(args.style_checkpoint.expanduser().resolve())
    else:
        style_options["checkpoint"] = str(resolve_path(style_options["checkpoint"]))
    return config


def resolve_gan_checkpoint(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> Path:
    if args.gan_checkpoint is not None:
        return args.gan_checkpoint.expanduser().resolve()
    output_dir = config.get("train", {}).get("output_dir", "outputs/font_cgan")
    return resolve_path(output_dir) / "checkpoints" / "latest.pt"


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_generator(
    checkpoint_path: Path,
    fallback_config: Mapping[str, Any],
    requested_weights: str,
    device: torch.device,
) -> tuple[ConditionalGenerator, str]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"GAN checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch_load(checkpoint_path)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("GAN checkpoint must contain a mapping")

    generator_config = checkpoint.get("generator_config")
    embedded_config = checkpoint.get("config")
    if generator_config is None and isinstance(embedded_config, Mapping):
        model_config = embedded_config.get("model", {})
        if isinstance(model_config, Mapping):
            generator_config = model_config.get("generator")
    if generator_config is None:
        generator_config = fallback_config["model"]["generator"]
    if not isinstance(generator_config, Mapping):
        raise ValueError("Could not resolve generator architecture")

    candidates = (
        ("generator_ema", "generator")
        if requested_weights == "auto"
        else (requested_weights,)
    )
    state_key = next(
        (key for key in candidates if isinstance(checkpoint.get(key), Mapping)),
        None,
    )
    if state_key is None:
        raise KeyError(f"GAN checkpoint contains none of {candidates}")

    generator = ConditionalGenerator(**copy.deepcopy(dict(generator_config)))
    generator.load_state_dict(checkpoint[state_key], strict=True)
    generator.to(device).eval().requires_grad_(False)
    return generator, state_key


def selected_indices(
    dataset_length: int, start_index: int, sample_count: int
) -> list[int]:
    if not 0 <= start_index < dataset_length:
        raise ValueError(f"--start-index must be in [0, {dataset_length - 1}]")
    if sample_count < 0:
        raise ValueError("--samples cannot be negative")
    available = dataset_length - start_index
    count = available if sample_count == 0 else min(sample_count, available)
    return list(range(start_index, start_index + count))


def safe_filename(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._")
    return value or "font"


def plot_similarity_matrix(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    x_label: str,
    output_path: Path,
    y_label: str = "style reference"
) -> None:
    count = matrix.shape[0]
    size = max(7.0, min(18.0, 5.0 + count * 0.18))
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    if count <= 40:
        axis.set_xticks(np.arange(count), labels, rotation=70, ha="right", fontsize=7)
        axis.set_yticks(np.arange(count), labels, fontsize=7)
    else:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_matched_comparison(
    real_values: np.ndarray,
    generated_values: np.ndarray,
    different_values: np.ndarray,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    bins = np.linspace(-1.0, 1.0, 51)
    axes[0].hist(
        real_values, bins=bins, alpha=0.7, density=True, label="reference vs real grid"
    )
    axes[0].hist(
        generated_values,
        bins=bins,
        alpha=0.65,
        density=True,
        label="reference vs generated grid",
    )
    if different_values.size:
        axes[0].hist(
            different_values,
            bins=bins,
            alpha=0.35,
            density=True,
            label="reference vs different-font real grid",
        )
    axes[0].axvline(real_values.mean(), color="tab:blue", linestyle="--")
    axes[0].axvline(generated_values.mean(), color="tab:orange", linestyle="--")
    axes[0].set_xlabel("cosine similarity")
    axes[0].set_ylabel("density")
    axes[0].set_title("Matched-pair similarity distributions")
    axes[0].legend()

    axes[1].scatter(real_values, generated_values, alpha=0.65, s=28)
    axes[1].plot((-1, 1), (-1, 1), color="black", linestyle="--", linewidth=1)
    axes[1].set_xlim(-1, 1)
    axes[1].set_ylim(-1, 1)
    axes[1].set_xlabel("reference vs real-grid cosine")
    axes[1].set_ylabel("reference vs generated-grid cosine")
    axes[1].set_title("Per-sample real/generated comparison")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_internal_distributions(
    reference_same: np.ndarray,
    reference_different: np.ndarray,
    real_same: np.ndarray,
    real_different: np.ndarray,
    output_path: Path,
) -> None:
    """Plot same-font and different-font internal pair distributions."""
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    bins = np.linspace(-1.0, 1.0, 51)
    panels = (
        (
            axes[0],
            reference_same,
            reference_different,
            "Reference/reference internal similarity",
        ),
        (
            axes[1],
            real_same,
            real_different,
            "Real-grid/real-grid internal similarity",
        ),
    )
    for axis, same_values, different_values, title in panels:
        if same_values.size:
            axis.hist(
                same_values,
                bins=bins,
                alpha=0.7,
                density=True,
                label=f"same font (n={same_values.size})",
            )
            axis.axvline(
                float(same_values.mean()), color="tab:blue", linestyle="--"
            )
        if different_values.size:
            axis.hist(
                different_values,
                bins=bins,
                alpha=0.55,
                density=True,
                label=f"different fonts (n={different_values.size})",
            )
            axis.axvline(
                float(different_values.mean()),
                color="tab:orange",
                linestyle="--",
            )
        axis.set_xlabel("cosine similarity (diagonal excluded)")
        axis.set_ylabel("density")
        axis.set_title(title)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "dataset_index",
        "style_id",
        "font_name",
        "letters",
        "real_cosine_similarity",
        "real_cosine_loss",
        "generated_cosine_similarity",
        "generated_cosine_loss",
        "generated_minus_real_similarity",
        "grid_comparison_path",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def evaluate(
    config: Mapping[str, Any],
    dataset: FontMimicDataset,
    indices: list[int],
    batch_size: int,
    device: torch.device,
    gan_checkpoint: Path,
    gan_weights: str,
    grid_output_dir: Path | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]], str]:
    collator = FontMimicCollator(**copy.deepcopy(config.get("collator", {})))
    style_encoder, _, _ = load_frozen_style_encoder(
        config["model"]["style_encoder"], device
    )
    generator, generator_state_key = load_generator(
        gan_checkpoint, config, gan_weights, device
    )

    patch_size = style_encoder.patch_size
    grid_mean = dataset.reference_dataset.mean
    grid_std = dataset.reference_dataset.std
    grid_max_side = dataset.reference_dataset.max_long_side
    grid_max_pixels = dataset.reference_dataset.max_pixels
    grid_columns = int(config.get("grid_columns", 4))
    reference_embeddings: list[torch.Tensor] = []
    real_embeddings: list[torch.Tensor] = []
    generated_embeddings: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []

    if grid_output_dir is not None:
        grid_output_dir.mkdir(parents=True, exist_ok=True)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        items = [dataset[index] for index in batch_indices]
        batch = collator(items)
        reference_view = move_style_view(batch["style_view"], device, False)
        target_groups = batch["target_images"].to(device)
        letter_groups = batch["letter_ids"].to(device)
        font_batch, letters_per_font = letter_groups.shape

        reference_style = style_encoder(reference_view).float()
        real_view = generated_grid_style_view(
            target_groups,
            patch_size,
            grid_mean,
            grid_std,
            grid_max_side,
            grid_max_pixels,
            grid_columns,
        )
        real_style = style_encoder(real_view).float()

        flat_style = reference_style.repeat_interleave(letters_per_font, dim=0)
        noise = torch.randn(
            font_batch * letters_per_font,
            generator.latent_dim,
            device=device,
        )
        generated_groups = generator(
            noise, flat_style, letter_groups.flatten()
        ).reshape_as(target_groups)
        generated_view = generated_grid_style_view(
            generated_groups,
            patch_size,
            grid_mean,
            grid_std,
            grid_max_side,
            grid_max_pixels,
            grid_columns,
        )
        generated_style = style_encoder(generated_view).float()

        real_cosine = F.cosine_similarity(reference_style, real_style, dim=1)
        generated_cosine = F.cosine_similarity(
            reference_style, generated_style, dim=1
        )
        reference_embeddings.append(reference_style.cpu())
        real_embeddings.append(real_style.cpu())
        generated_embeddings.append(generated_style.cpu())

        comparison_paths = [""] * font_batch
        if grid_output_dir is not None:
            real_grids = make_grouped_grid(target_groups, grid_columns)
            inferred_grids = make_grouped_grid(generated_groups, grid_columns)
            comparisons = torch.cat((real_grids, inferred_grids), dim=-1).cpu()
            for offset, comparison in enumerate(comparisons):
                item = items[offset]
                filename = (
                    f"dataset_{batch_indices[offset]:06d}_"
                    f"style_{int(item['style_id']):04d}_"
                    f"{safe_filename(Path(item['font_name']).stem)}.png"
                )
                path = grid_output_dir / filename
                save_image(
                    comparison,
                    path,
                    normalize=True,
                    value_range=(-1, 1),
                )
                comparison_paths[offset] = str(path)

        for offset, (real_value, generated_value) in enumerate(
            zip(real_cosine.cpu().tolist(), generated_cosine.cpu().tolist())
        ):
            item = items[offset]
            rows.append(
                {
                    "dataset_index": batch_indices[offset],
                    "style_id": int(item["style_id"]),
                    "font_name": str(item["font_name"]),
                    "letters": "".join(item["letters"]),
                    "real_cosine_similarity": float(real_value),
                    "real_cosine_loss": float(1.0 - real_value),
                    "generated_cosine_similarity": float(generated_value),
                    "generated_cosine_loss": float(1.0 - generated_value),
                    "generated_minus_real_similarity": float(
                        generated_value - real_value
                    ),
                    "grid_comparison_path": comparison_paths[offset],
                }
            )
        print(
            f"Processed {min(start + batch_size, len(indices))}/{len(indices)}",
            flush=True,
        )

    return (
        torch.cat(reference_embeddings),
        torch.cat(real_embeddings),
        torch.cat(generated_embeddings),
        rows,
        generator_state_key,
    )


def matrix_metrics(
    matrix: np.ndarray, style_ids: np.ndarray
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    matched = np.diag(matrix)
    same_style = style_ids[:, None] == style_ids[None, :]
    diagonal = np.eye(len(style_ids), dtype=bool)
    different = matrix[~same_style]
    same_other = matrix[same_style & ~diagonal]
    nearest_reference = matrix.argmax(axis=0)
    metrics = {
        "matched_cosine_mean": float(matched.mean()),
        "matched_cosine_std": float(matched.std()),
        "matched_cosine_min": float(matched.min()),
        "matched_cosine_max": float(matched.max()),
        "matched_cosine_loss_mean": float((1.0 - matched).mean()),
        "different_font_cosine_mean": (
            float(different.mean()) if different.size else None
        ),
        "same_font_other_cosine_mean": (
            float(same_other.mean()) if same_other.size else None
        ),
        "reference_retrieval_top1_accuracy": float(
            np.mean(style_ids[nearest_reference] == style_ids)
        ),
    }
    return metrics, matched, different


def internal_matrix_metrics(
    matrix: np.ndarray, style_ids: np.ndarray
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Summarize unique off-diagonal pairs in a square similarity matrix."""
    upper_triangle = np.triu(
        np.ones(matrix.shape, dtype=bool), k=1
    )
    same_style = style_ids[:, None] == style_ids[None, :]
    all_pairs = matrix[upper_triangle]
    same_pairs = matrix[upper_triangle & same_style]
    different_pairs = matrix[upper_triangle & ~same_style]

    def statistics(values: np.ndarray) -> dict[str, Any]:
        if not values.size:
            return {
                "pair_count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
        return {
            "pair_count": int(values.size),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    same_stats = statistics(same_pairs)
    different_stats = statistics(different_pairs)
    margin = (
        float(same_pairs.mean() - different_pairs.mean())
        if same_pairs.size and different_pairs.size
        else None
    )
    metrics = {
        "all_off_diagonal": statistics(all_pairs),
        "same_font": same_stats,
        "different_font": different_stats,
        "same_minus_different_mean": margin,
    }
    return metrics, all_pairs, same_pairs, different_pairs


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.heatmap_limit <= 0:
        raise ValueError("--heatmap-limit must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = load_config(args)
    dataset = FontMimicDataset(copy.deepcopy(config["dataset"]))
    indices = selected_indices(len(dataset), args.start_index, args.samples)
    device = choose_device(args.device)
    gan_checkpoint = resolve_gan_checkpoint(args, config)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_output_dir = output_dir / "grid_comparisons" if args.save_grids else None

    references, real_grids, generated_grids, rows, generator_state_key = evaluate(
        config,
        dataset,
        indices,
        args.batch_size,
        device,
        gan_checkpoint,
        args.gan_weights,
        grid_output_dir,
    )
    references = F.normalize(references.float(), dim=1)
    real_grids = F.normalize(real_grids.float(), dim=1)
    generated_grids = F.normalize(generated_grids.float(), dim=1)
    real_matrix = (references @ real_grids.T).numpy()
    generated_matrix = (references @ generated_grids.T).numpy()
    reference_internal_matrix = (references @ references.T).numpy()
    real_internal_matrix = (real_grids @ real_grids.T).numpy()
    style_ids = np.asarray([row["style_id"] for row in rows])
    real_metrics, real_matched, different_real = matrix_metrics(
        real_matrix, style_ids
    )
    generated_metrics, generated_matched, _ = matrix_metrics(
        generated_matrix, style_ids
    )
    (
        reference_internal_metrics,
        _,
        reference_same,
        reference_different,
    ) = internal_matrix_metrics(reference_internal_matrix, style_ids)
    (
        real_internal_metrics,
        _,
        real_same,
        real_different,
    ) = internal_matrix_metrics(real_internal_matrix, style_ids)
    delta = generated_matched - real_matched
    summary = {
        "sample_count": len(rows),
        "font_count": int(len(np.unique(style_ids))),
        "device": str(device),
        "seed": args.seed,
        "style_checkpoint": str(config["model"]["style_encoder"]["checkpoint"]),
        "gan_checkpoint": str(gan_checkpoint),
        "gan_weights": generator_state_key,
        "real_grid": real_metrics,
        "generated_grid": generated_metrics,
        "reference_internal": reference_internal_metrics,
        "real_grid_internal": real_internal_metrics,
        "comparison": {
            "generated_minus_real_similarity_mean": float(delta.mean()),
            "generated_minus_real_similarity_std": float(delta.std()),
            "generated_better_fraction": float(np.mean(delta > 0)),
        },
    }

    save_rows(output_dir / "matched_pairs.csv", rows)
    np.save(output_dir / "reference_embeddings.npy", references.numpy())
    np.save(output_dir / "real_grid_embeddings.npy", real_grids.numpy())
    np.save(output_dir / "generated_grid_embeddings.npy", generated_grids.numpy())
    np.save(output_dir / "real_similarity_matrix.npy", real_matrix)
    np.save(output_dir / "generated_similarity_matrix.npy", generated_matrix)
    np.save(
        output_dir / "reference_internal_similarity_matrix.npy",
        reference_internal_matrix,
    )
    np.save(
        output_dir / "real_grid_internal_similarity_matrix.npy",
        real_internal_matrix,
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    plot_matched_comparison(
        real_matched,
        generated_matched,
        different_real,
        output_dir / "matched_similarity_comparison.png",
    )
    plot_internal_distributions(
        reference_same,
        reference_different,
        real_same,
        real_different,
        output_dir / "internal_similarity_distributions.png",
    )
    heatmap_count = min(len(rows), args.heatmap_limit)
    labels = [
        f"{row['dataset_index']}:{Path(row['font_name']).stem}"
        for row in rows[:heatmap_count]
    ]
    plot_similarity_matrix(
        reference_internal_matrix[:heatmap_count, :heatmap_count],
        labels,
        "Reference/reference internal embedding similarity",
        "style reference",
        output_dir / "reference_internal_similarity_heatmap.png",
    )
    plot_similarity_matrix(
        real_internal_matrix[:heatmap_count, :heatmap_count],
        labels,
        "Real-grid/real-grid internal embedding similarity",
        "real glyph grid",
        output_dir / "real_grid_internal_similarity_heatmap.png",
        y_label="real glyph grid",
    )
    plot_similarity_matrix(
        real_matrix[:heatmap_count, :heatmap_count],
        labels,
        "Reference vs. real glyph-grid embeddings",
        "real glyph grid",
        output_dir / "real_similarity_heatmap.png",
    )
    plot_similarity_matrix(
        generated_matrix[:heatmap_count, :heatmap_count],
        labels,
        "Reference vs. GAN-generated glyph-grid embeddings",
        "generated glyph grid",
        output_dir / "generated_similarity_heatmap.png",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
