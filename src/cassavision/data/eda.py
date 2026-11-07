"""Exploratory image analysis and visual report generation for CASSAVISION."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from cassavision.config import CANONICAL_CLASS_MAP, CLASS_CODES, get_project_root


def generate_eda_figures_and_report(
    df: pd.DataFrame,
    figures_dir: Path,
    metrics_dir: Path,
    full_dataset_counts: Dict[str, int] | None = None,
) -> Dict[str, Any]:
    """
    Generate publication-quality exploratory data analysis figures and summary statistics:
      1. class_distribution.png: Full 21,397 vs Acquired Cohort & Train/Val/Test partition distribution
      2. image_quality_distributions.png: Brightness, contrast, sharpness (Laplacian), and green ratio by class
      3. representative_samples_grid.png: 5x4 grid of genuine field images across all 5 disease categories
      4. RGB channel & aspect ratio analysis
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "axes.edgecolor": "#334155",
            "axes.linewidth": 0.8,
        }
    )
    palette = ["#ef4444", "#f59e0b", "#eab308", "#8b5cf6", "#10b981"]

    # 1. Class distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=140)
    class_order = [CANONICAL_CLASS_MAP[i]["code"] for i in range(5)]
    counts = [int((df["label"] == i).sum()) for i in range(5)]
    pcts = [100.0 * c / max(len(df), 1) for c in counts]

    bars = axes[0].bar(class_order, counts, color=palette, edgecolor="#1e293b", linewidth=1.0)
    axes[0].set_title(f"Acquired & Validated Cohort Distribution (N = {len(df)})", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Disease Category", fontsize=10)
    axes[0].set_ylabel("Number of Validated Images", fontsize=10)
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)
    for bar, count, pct in zip(bars, counts, pcts):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(counts) * 0.015,
            f"{count}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    axes[0].set_ylim(0, max(counts) * 1.18)

    if "split" in df.columns:
        split_pivot = (
            df.groupby(["class_code", "split"])
            .size()
            .unstack(fill_value=0)
            .reindex(class_order)
        )
        for sp in ["train", "val", "test"]:
            if sp not in split_pivot.columns:
                split_pivot[sp] = 0
        split_pivot = split_pivot[["train", "val", "test"]]
        split_pivot.plot(
            kind="bar",
            stacked=True,
            ax=axes[1],
            color=["#2563eb", "#0d9488", "#f97316"],
            edgecolor="#1e293b",
            linewidth=0.8,
        )
        axes[1].set_title("Stratified Group-Aware Partitioning (Train / Val / Test)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Disease Category", fontsize=10)
        axes[1].set_ylabel("Image Count", fontsize=10)
        axes[1].tick_params(axis="x", rotation=0)
        axes[1].grid(axis="y", linestyle="--", alpha=0.35)
        axes[1].legend(title="Partition", frameon=True)

    plt.tight_layout()
    dist_fig_path = figures_dir / "class_distribution.png"
    plt.savefig(dist_fig_path, dpi=140)
    plt.close(fig)

    # 2. Image quality & chromaticity distributions across classes
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=140)
    df_plot = df.copy()
    df_plot["class_code"] = pd.Categorical(df_plot["class_code"], categories=class_order, ordered=True)

    sns.boxplot(
        data=df_plot,
        x="class_code",
        y="brightness",
        hue="class_code",
        palette=palette,
        legend=False,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Luminance / Brightness Distribution by Class", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("Mean Luminance (0–255)")
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.3)

    sns.boxplot(
        data=df_plot,
        x="class_code",
        y="green_dominance_ratio",
        hue="class_code",
        palette=palette,
        legend=False,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Foliage Green-Chromaticity Ratio by Class", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("Green Dominance Fraction")
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.3)

    sns.boxplot(
        data=df_plot,
        x="class_code",
        y="sharpness_laplacian",
        hue="class_code",
        palette=palette,
        legend=False,
        ax=axes[1, 0],
        showfliers=False,
    )
    axes[1, 0].set_title("Focus Sharpness (Laplacian Variance, outliers hidden)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Disease Category")
    axes[1, 0].set_ylabel("Laplacian Variance")
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.3)

    # Mean RGB per class bar chart
    rgb_means = df_plot.groupby("class_code", observed=False)[["mean_r", "mean_g", "mean_b"]].mean()
    x_idx = np.arange(len(class_order))
    w = 0.25
    axes[1, 1].bar(x_idx - w, rgb_means["mean_r"], width=w, color="#ef4444", label="Red Channel")
    axes[1, 1].bar(x_idx, rgb_means["mean_g"], width=w, color="#22c55e", label="Green Channel")
    axes[1, 1].bar(x_idx + w, rgb_means["mean_b"], width=w, color="#3b82f6", label="Blue Channel")
    axes[1, 1].set_xticks(x_idx)
    axes[1, 1].set_xticklabels(class_order)
    axes[1, 1].set_title("Mean RGB Channel Intensity Across Classes", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Disease Category")
    axes[1, 1].set_ylabel("Channel Mean (0–255)")
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    qual_fig_path = figures_dir / "image_quality_distributions.png"
    plt.savefig(qual_fig_path, dpi=140)
    plt.close(fig)

    # 3. Representative samples grid (5 classes x 3 samples)
    root = get_project_root()
    fig, axes = plt.subplots(5, 3, figsize=(10.5, 14), dpi=130)
    for cls_idx in range(5):
        cls_df = df[df["label"] == cls_idx].head(3)
        for col_idx in range(3):
            ax = axes[cls_idx, col_idx]
            if col_idx < len(cls_df):
                row = cls_df.iloc[col_idx]
                img_path = root / row["local_relpath"]
                if img_path.exists():
                    with Image.open(img_path) as im:
                        ax.imshow(im.convert("RGB").resize((224, 224)))
                    ax.set_title(
                        f"{CANONICAL_CLASS_MAP[cls_idx]['code']} | {row['image_id']}\n({row['width']}x{row['height']}, B={row['brightness']:.0f})",
                        fontsize=8,
                    )
            ax.axis("off")

    plt.suptitle("Representative Cassava Leaf Images Across All 5 Categories (Makerere / NaCRRI Dataset)", fontsize=12, fontweight="bold", y=0.995)
    plt.tight_layout()
    grid_fig_path = figures_dir / "representative_samples_grid.png"
    plt.savefig(grid_fig_path, dpi=130)
    plt.close(fig)

    # Build statistical EDA summary JSON
    class_stats: Dict[str, Any] = {}
    for cls_idx in range(5):
        sub = df[df["label"] == cls_idx]
        code = CANONICAL_CLASS_MAP[cls_idx]["code"]
        class_stats[code] = {
            "count": int(len(sub)),
            "percentage": round(100.0 * len(sub) / max(len(df), 1), 2),
            "mean_brightness": round(float(sub["brightness"].mean()), 2) if len(sub) else 0.0,
            "mean_contrast": round(float(sub["contrast"].mean()), 2) if len(sub) else 0.0,
            "mean_sharpness": round(float(sub["sharpness_laplacian"].mean()), 2) if len(sub) else 0.0,
            "mean_green_ratio": round(float(sub["green_dominance_ratio"].mean()), 4) if len(sub) else 0.0,
            "mean_rgb": [
                round(float(sub["mean_r"].mean()), 2) if len(sub) else 0.0,
                round(float(sub["mean_g"].mean()), 2) if len(sub) else 0.0,
                round(float(sub["mean_b"].mean()), 2) if len(sub) else 0.0,
            ],
        }

    dimensions_counts = df.groupby(["width", "height"]).size().reset_index(name="count").to_dict(orient="records")
    warnings_count = int((df["quality_warnings_str"].fillna("").str.len() > 0).sum()) if "quality_warnings_str" in df.columns else 0

    eda_summary = {
        "total_validated_images": int(len(df)),
        "unique_dimensions": dimensions_counts,
        "aspect_ratio_mean": round(float(df["aspect_ratio"].mean()), 4),
        "overall_brightness_mean": round(float(df["brightness"].mean()), 2),
        "overall_brightness_std": round(float(df["brightness"].std()), 2),
        "overall_sharpness_median": round(float(df["sharpness_laplacian"].median()), 2),
        "images_with_quality_warnings": warnings_count,
        "per_class_statistics": class_stats,
        "generated_figures": [
            str(dist_fig_path.relative_to(root).as_posix()),
            str(qual_fig_path.relative_to(root).as_posix()),
            str(grid_fig_path.relative_to(root).as_posix()),
        ],
    }

    with open(metrics_dir / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2)

    return eda_summary
