"""Evaluation metrics, Expected Calibration Error (ECE), temperature scaling, and error analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from cassavision.config import CANONICAL_CLASS_MAP, CLASS_CODES


def compute_ece(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Compute Expected Calibration Error (ECE) and bin-level reliability statistics.
    probs: shape (N, K) predicted probability matrix
    y_true: shape (N,) integer ground-truth labels
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins_data: List[Dict[str, Any]] = []
    n_total = max(len(y_true), 1)

    for b_idx in range(n_bins):
        low, high = bin_boundaries[b_idx], bin_boundaries[b_idx + 1]
        if b_idx == 0:
            in_bin = (confidences >= low) & (confidences <= high)
        else:
            in_bin = (confidences > low) & (confidences <= high)

        count = int(np.sum(in_bin))
        if count > 0:
            avg_conf = float(np.mean(confidences[in_bin]))
            avg_acc = float(np.mean(accuracies[in_bin]))
            gap = abs(avg_acc - avg_conf)
            ece += (count / float(n_total)) * gap
        else:
            avg_conf = float((low + high) / 2.0)
            avg_acc = 0.0
            gap = 0.0

        bins_data.append(
            {
                "bin_index": b_idx,
                "bin_lower": round(float(low), 2),
                "bin_upper": round(float(high), 2),
                "count": count,
                "mean_confidence": round(avg_conf, 4),
                "empirical_accuracy": round(avg_acc, 4),
                "calibration_gap": round(gap, 4),
            }
        )

    return float(ece), bins_data


def fit_temperature_scaling(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
) -> float:
    """
    Fit optimal post-hoc temperature parameter T > 0 on validation logits by minimizing
    Negative Log-Likelihood (cross-entropy) via L-BFGS-B.
    """
    logits = np.asarray(val_logits, dtype=np.float64)
    labels = np.asarray(val_labels, dtype=np.int64)

    def _nll_loss(t_arr: np.ndarray) -> float:
        t = max(float(t_arr[0]), 0.05)
        scaled = logits / t
        scaled = scaled - np.max(scaled, axis=1, keepdims=True)
        exp_s = np.exp(scaled)
        probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
        chosen = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
        return float(-np.mean(np.log(chosen)))

    res = minimize(_nll_loss, x0=np.array([1.0]), bounds=[(0.1, 10.0)], method="L-BFGS-B")
    return round(float(res.x[0]), 4)


def apply_temperature(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Apply temperature scaling T to logits and return calibrated probability distribution."""
    t = max(float(temperature), 0.05)
    scaled = np.asarray(logits, dtype=np.float64) / t
    return softmax(scaled, axis=-1)


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: Optional[np.ndarray] = None,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute complete classification metrics across all 5 cassava disease categories:
      - Overall Accuracy
      - Macro Precision, Macro Recall, Macro F1
      - Weighted Precision, Weighted Recall, Weighted F1
      - Per-class Precision, Recall, F1, Support, False Positive Rate, False Negative Rate
      - 5x5 Confusion Matrix (raw counts and row-normalized percentages)
      - Expected Calibration Error (ECE) & confidence summary if probabilities provided
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    labels_list = list(range(5))

    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_list, average="macro", zero_division=0
    )
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_list, average="weighted", zero_division=0
    )
    p_cls, r_cls, f1_cls, sup_cls = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_list, average=None, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels_list)
    row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_norm = cm.astype(np.float64) / row_sums

    per_class: Dict[str, Dict[str, Any]] = {}
    total_samples = max(len(y_true), 1)
    for i in labels_list:
        code = CANONICAL_CLASS_MAP[i]["code"]
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(cm.sum() - (tp + fn + fp))
        fpr = fp / float(max(fp + tn, 1))
        fnr = fn / float(max(fn + tp, 1))

        per_class[code] = {
            "class_index": i,
            "class_code": code,
            "class_name": CANONICAL_CLASS_MAP[i]["name"],
            "precision": round(float(p_cls[i]), 4),
            "recall": round(float(r_cls[i]), 4),
            "f1": round(float(f1_cls[i]), 4),
            "support": int(sup_cls[i]),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "false_positive_rate": round(float(fpr), 4),
            "false_negative_rate": round(float(fnr), 4),
        }

    # Identify top confused class pairs (off-diagonal entries in confusion matrix)
    confused_pairs: List[Dict[str, Any]] = []
    for i in labels_list:
        for j in labels_list:
            if i != j and cm[i, j] > 0:
                confused_pairs.append(
                    {
                        "true_class": CANONICAL_CLASS_MAP[i]["code"],
                        "predicted_class": CANONICAL_CLASS_MAP[j]["code"],
                        "count": int(cm[i, j]),
                        "rate_within_true_class": round(float(cm_norm[i, j]), 4),
                    }
                )
    confused_pairs.sort(key=lambda x: x["count"], reverse=True)

    result: Dict[str, Any] = {
        "total_samples": int(len(y_true)),
        "accuracy": round(acc, 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_precision": round(float(p_weight), 4),
        "weighted_recall": round(float(r_weight), 4),
        "weighted_f1": round(float(f1_weight), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": np.round(cm_norm, 4).tolist(),
        "top_confused_pairs": confused_pairs[:10],
    }

    if probs is not None:
        ece_val, ece_bins = compute_ece(probs, y_true, n_bins=n_bins)
        confidences = np.max(probs, axis=1)
        correct_mask = y_pred == y_true
        result["ece"] = round(ece_val, 4)
        result["reliability_bins"] = ece_bins
        result["mean_confidence_overall"] = round(float(np.mean(confidences)), 4)
        result["mean_confidence_correct"] = (
            round(float(np.mean(confidences[correct_mask])), 4) if np.any(correct_mask) else 0.0
        )
        result["mean_confidence_incorrect"] = (
            round(float(np.mean(confidences[~correct_mask])), 4) if np.any(~correct_mask) else 0.0
        )

    return result


def plot_evaluation_figures(
    metrics: Dict[str, Any],
    model_comparison: List[Dict[str, Any]],
    histories: Dict[str, List[Dict[str, Any]]],
    figures_dir: Path,
) -> Dict[str, str]:
    """Generate confusion matrix, reliability diagram, learning curves, and model comparison charts."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: Dict[str, str] = {}

    # 1. Confusion Matrix (Counts + Row-Normalized %)
    cm = np.array(metrics["confusion_matrix"])
    cm_norm = np.array(metrics["confusion_matrix_normalized"]) * 100.0
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), dpi=140)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_CODES,
        yticklabels=CLASS_CODES,
        ax=axes[0],
        cbar=False,
    )
    axes[0].set_title("Test Set Confusion Matrix (Sample Counts)", fontsize=11.5, fontweight="bold")
    axes[0].set_xlabel("Predicted Category")
    axes[0].set_ylabel("True Category")

    annot_pct = np.array([[f"{val:.1f}%" for val in row] for row in cm_norm])
    sns.heatmap(
        cm_norm,
        annot=annot_pct,
        fmt="",
        cmap="YlGnBu",
        xticklabels=CLASS_CODES,
        yticklabels=CLASS_CODES,
        ax=axes[1],
        vmin=0,
        vmax=100,
    )
    axes[1].set_title("Row-Normalized Recall Matrix (%)", fontsize=11.5, fontweight="bold")
    axes[1].set_xlabel("Predicted Category")
    axes[1].set_ylabel("True Category")

    plt.tight_layout()
    cm_path = figures_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=140)
    plt.close(fig)
    saved_paths["confusion_matrix"] = str(cm_path)

    # 2. Per-class Precision, Recall, F1 bar chart
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=140)
    pc = metrics["per_class"]
    x = np.arange(len(CLASS_CODES))
    w = 0.25
    prec = [pc[c]["precision"] for c in CLASS_CODES]
    rec = [pc[c]["recall"] for c in CLASS_CODES]
    f1s = [pc[c]["f1"] for c in CLASS_CODES]

    ax.bar(x - w, prec, width=w, label="Precision", color="#3b82f6", edgecolor="#1e293b")
    ax.bar(x, rec, width=w, label="Recall", color="#10b981", edgecolor="#1e293b")
    ax.bar(x + w, f1s, width=w, label="F1-Score", color="#8b5cf6", edgecolor="#1e293b")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={pc[c]['support']})" for c in CLASS_CODES])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score (0.0 – 1.0)")
    ax.set_title(
        f"Test Set Per-Class Performance (Macro F1 = {metrics['macro_f1']:.3f}, Weighted F1 = {metrics['weighted_f1']:.3f})",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    pc_path = figures_dir / "per_class_metrics.png"
    plt.savefig(pc_path, dpi=140)
    plt.close(fig)
    saved_paths["per_class_metrics"] = str(pc_path)

    # 3. Reliability Diagram & Calibration Analysis
    if "reliability_bins" in metrics:
        bins = metrics["reliability_bins"]
        fig, ax = plt.subplots(figsize=(6.8, 5.2), dpi=140)
        centers = [(b["bin_lower"] + b["bin_upper"]) / 2.0 for b in bins]
        accs = [b["empirical_accuracy"] for b in bins]
        confs = [b["mean_confidence"] for b in bins]
        counts = [b["count"] for b in bins]

        ax.plot([0, 1], [0, 1], "--", color="#64748b", label="Perfect Calibration (Diagonal)")
        ax.bar(
            centers,
            accs,
            width=0.085,
            alpha=0.75,
            color="#0ea5e9",
            edgecolor="#0369a1",
            label="Empirical Accuracy",
        )
        ax.plot(
            [c for c, cnt in zip(confs, counts) if cnt > 0],
            [a for a, cnt in zip(accs, counts) if cnt > 0],
            "o-",
            color="#ef4444",
            linewidth=2,
            label="Calibration Curve",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Predicted Probability / Confidence")
        ax.set_ylabel("Empirical Test Accuracy")
        ax.set_title(f"Reliability Diagram (Calibrated ECE = {metrics.get('ece', 0.0):.4f})", fontsize=11.5, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="upper left")
        plt.tight_layout()
        rel_path = figures_dir / "reliability_diagram.png"
        plt.savefig(rel_path, dpi=140)
        plt.close(fig)
        saved_paths["reliability_diagram"] = str(rel_path)

    # 4. Learning Curves (Train vs Validation Loss & Macro F1 / Accuracy)
    if histories:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), dpi=140)
        for arch_name, hist in histories.items():
            if not hist:
                continue
            epochs = [h["epoch"] for h in hist]
            val_f1 = [h["val_macro_f1"] for h in hist]
            train_loss = [h["train_loss"] for h in hist]
            val_loss = [h["val_loss"] for h in hist]
            axes[0].plot(epochs, train_loss, "--", alpha=0.7, label=f"{arch_name} (train)")
            axes[0].plot(epochs, val_loss, "-", linewidth=2, label=f"{arch_name} (val)")
            axes[1].plot(epochs, val_f1, "o-", linewidth=2, label=f"{arch_name}")

        axes[0].set_title("Training & Validation Loss Progression", fontsize=11.5, fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Cross-Entropy Loss")
        axes[0].grid(True, linestyle="--", alpha=0.35)
        axes[0].legend(fontsize=8.5)

        axes[1].set_title("Validation Macro F1 Progression Across Architectures", fontsize=11.5, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Validation Macro F1")
        axes[1].grid(True, linestyle="--", alpha=0.35)
        axes[1].legend(fontsize=9)

        plt.tight_layout()
        lc_path = figures_dir / "learning_curves.png"
        plt.savefig(lc_path, dpi=140)
        plt.close(fig)
        saved_paths["learning_curves"] = str(lc_path)

    # 5. Model Comparison Chart (Accuracy, Macro F1, Weighted F1 vs Latency)
    if model_comparison:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), dpi=140)
        names = [m["model_name"] for m in model_comparison]
        accs = [m["test_accuracy"] for m in model_comparison]
        mf1s = [m["test_macro_f1"] for m in model_comparison]
        wf1s = [m["test_weighted_f1"] for m in model_comparison]
        latencies = [m["inference_latency_ms"] for m in model_comparison]
        sizes = [max(m.get("model_size_mb", 1.0), 0.5) for m in model_comparison]

        x = np.arange(len(names))
        w = 0.25
        axes[0].bar(x - w, accs, width=w, label="Test Accuracy", color="#0ea5e9")
        axes[0].bar(x, mf1s, width=w, label="Test Macro F1", color="#8b5cf6")
        axes[0].bar(x + w, wf1s, width=w, label="Test Weighted F1", color="#10b981")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, rotation=15, ha="right")
        axes[0].set_ylim(0, 1.05)
        axes[0].set_ylabel("Score")
        axes[0].set_title("Independent Test Performance Across Candidate Models", fontsize=11, fontweight="bold")
        axes[0].grid(axis="y", linestyle="--", alpha=0.35)
        axes[0].legend()

        scatter = axes[1].scatter(
            latencies,
            mf1s,
            s=[s * 18 + 80 for s in sizes],
            c=range(len(names)),
            cmap="viridis",
            alpha=0.85,
            edgecolors="#0f172a",
            linewidth=1.5,
        )
        for name, lat, mf1, sz in zip(names, latencies, mf1s, sizes):
            axes[1].annotate(
                f"{name}\n({sz:.1f} MB)",
                (lat, mf1),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8.5,
                fontweight="bold",
            )
        axes[1].set_xlabel("CPU Single-Image Inference Latency (ms)")
        axes[1].set_ylabel("Test Macro F1 Score")
        axes[1].set_title("Efficiency vs. Macro F1 Trade-Off (Bubble Size = Model MB)", fontsize=11, fontweight="bold")
        axes[1].grid(True, linestyle="--", alpha=0.35)

        plt.tight_layout()
        mc_path = figures_dir / "model_comparison.png"
        plt.savefig(mc_path, dpi=140)
        plt.close(fig)
        saved_paths["model_comparison"] = str(mc_path)

    return saved_paths
