"""End-to-end reproducible execution pipeline for CASSAVISION (Stages 4 through 9)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

# Add src to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from cassavision.config import CANONICAL_CLASS_MAP, CLASS_CODES, get_project_root, load_config, seed_everything
from cassavision.data.acquisition import acquire_and_validate_dataset
from cassavision.data.eda import generate_eda_figures_and_report
from cassavision.data.preprocessing import preprocess_pil_image
from cassavision.data.splitting import create_leakage_free_splits, investigate_duplicates_and_assign_groups
from cassavision.evaluation.metrics import plot_evaluation_figures
from cassavision.explainability.gradcam import GradCAMExplainer, compute_occlusion_sensitivity
from cassavision.models.architectures import build_classifier_model
from cassavision.training.trainer import train_and_evaluate_deep_model, train_classical_baseline


def generate_gradcam_and_failure_galleries(
    selected_result: Dict[str, Any],
    test_df: pd.DataFrame,
    root_dir: Path,
    figures_dir: Path,
    metrics_dir: Path,
) -> Dict[str, Any]:
    """
    Generate genuine Grad-CAM + Occlusion Sensitivity audit galleries across all 5 classes
    and a visual gallery of representative test-set failure cases.
    """
    ckpt_path = root_dir / selected_result["checkpoint_path"]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch_name = ckpt["architecture"]
    temp = float(ckpt.get("temperature", 1.0))
    prep = ckpt["preprocessing"]

    model, _ = build_classifier_model(arch_name, num_classes=5, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    explainer = GradCAMExplainer(model, arch_name)

    # Map image_id -> test prediction record
    pred_map = {r["image_id"]: r for r in selected_result["test_predictions"]}
    test_df_indexed = test_df.set_index("image_id")

    # 1. Grad-CAM & Occlusion Audit Gallery across all 5 classes (5 rows x 4 cols: Original, Grad-CAM Heatmap, Grad-CAM Overlay, Occlusion Overlay)
    fig, axes = plt.subplots(5, 4, figsize=(13.5, 15.5), dpi=135)
    audit_records: List[Dict[str, Any]] = []

    for cls_idx in range(5):
        code = CANONICAL_CLASS_MAP[cls_idx]["code"]
        # Prefer a correctly predicted sample of this class from the test set to illustrate class activation, fallback to any test sample of this class
        cls_candidates = [
            r for r in selected_result["test_predictions"] if r["true_label"] == cls_idx and r["is_correct"]
        ]
        if not cls_candidates:
            cls_candidates = [r for r in selected_result["test_predictions"] if r["true_label"] == cls_idx]
        cls_candidates.sort(key=lambda x: x["confidence"], reverse=True)
        chosen = cls_candidates[0]
        img_id = chosen["image_id"]
        row_meta = test_df_indexed.loc[img_id]
        img_path = root_dir / row_meta["local_relpath"]

        with Image.open(img_path) as im:
            tensor, aligned_pil = preprocess_pil_image(
                im,
                image_size=int(prep["image_size"]),
                eval_resize=int(prep["eval_resize"]),
                mean=prep["mean"],
                std=prep["std"],
            )

        gcam_out = explainer.explain(
            input_tensor=tensor,
            aligned_crop_pil=aligned_pil,
            target_class=chosen["pred_label"],
            temperature=temp,
        )
        _, occ_pil = compute_occlusion_sensitivity(
            model=model,
            input_tensor=tensor,
            aligned_crop_pil=aligned_pil,
            target_class=chosen["pred_label"],
            temperature=temp,
        )

        axes[cls_idx, 0].imshow(aligned_pil)
        axes[cls_idx, 0].set_title(f"True: {code} ({img_id})\nPred: {chosen['pred_code']} ({chosen['confidence']*100:.1f}%)", fontsize=8.5, fontweight="bold")
        axes[cls_idx, 1].imshow(gcam_out.heatmap_rgb)
        axes[cls_idx, 1].set_title(f"Grad-CAM Activation\n(Center/Border={gcam_out.center_vs_border_ratio:.2f}x)", fontsize=8.5)
        axes[cls_idx, 2].imshow(gcam_out.overlay_rgb)
        axes[cls_idx, 2].set_title("Grad-CAM Overlay on Leaf", fontsize=8.5)
        axes[cls_idx, 3].imshow(occ_pil)
        axes[cls_idx, 3].set_title("Occlusion Sensitivity Map", fontsize=8.5)

        for c in range(4):
            axes[cls_idx, c].axis("off")

        audit_records.append(
            {
                "class_index": cls_idx,
                "class_code": code,
                "image_id": img_id,
                "true_code": chosen["true_code"],
                "pred_code": chosen["pred_code"],
                "confidence": chosen["confidence"],
                "center_vs_border_ratio": gcam_out.center_vs_border_ratio,
                "saliency_note": gcam_out.saliency_concentration_note,
            }
        )

    plt.suptitle(
        f"Grad-CAM & Occlusion Sensitivity Audit Across All 5 Cassava Categories ({arch_name.upper()})",
        fontsize=12.5,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    gcam_fig_path = figures_dir / "gradcam_audit_gallery.png"
    plt.savefig(gcam_fig_path, dpi=135)
    plt.close(fig)

    #Also compute overall Grad-CAM center-vs-border ratio across 30 random test images
    cb_ratios: List[float] = []
    for rec in selected_result["test_predictions"][:35]:
        img_id = rec["image_id"]
        row_meta = test_df_indexed.loc[img_id]
        with Image.open(root_dir / row_meta["local_relpath"]) as im:
            tensor, aligned_pil = preprocess_pil_image(im)
        out = explainer.explain(tensor, aligned_pil, target_class=rec["pred_label"], temperature=temp)
        cb_ratios.append(out.center_vs_border_ratio)

    # 2. Failure Case Analysis Gallery (Top 6 test-set misclassifications)
    mistakes = [r for r in selected_result["test_predictions"] if not r["is_correct"]]
    mistakes.sort(key=lambda x: x["confidence"], reverse=True)
    top_mistakes = mistakes[:6]

    failure_records: List[Dict[str, Any]] = []
    if top_mistakes:
        n_m = len(top_mistakes)
        fig, axes = plt.subplots(n_m, 3, figsize=(10.5, 2.9 * n_m), dpi=130)
        if n_m == 1:
            axes = np.expand_dims(axes, axis=0)

        for idx_m, m in enumerate(top_mistakes):
            img_id = m["image_id"]
            row_meta = test_df_indexed.loc[img_id]
            img_path = root_dir / row_meta["local_relpath"]
            with Image.open(img_path) as im:
                tensor, aligned_pil = preprocess_pil_image(im)
            gcam_m = explainer.explain(tensor, aligned_pil, target_class=m["pred_label"], temperature=temp)

            axes[idx_m, 0].imshow(aligned_pil)
            axes[idx_m, 0].set_title(
                f"Mistake #{idx_m+1}: {img_id}\nTrue: {m['true_code']} | Pred: {m['pred_code']} ({m['confidence']*100:.1f}%)",
                fontsize=8.5,
                fontweight="bold",
                color="#b91c1c",
            )
            axes[idx_m, 1].imshow(gcam_m.overlay_rgb)
            axes[idx_m, 1].set_title(
                f"Grad-CAM for Predicted ({m['pred_code']})\nC/B Ratio={gcam_m.center_vs_border_ratio:.2f}x",
                fontsize=8.5,
            )

            # Bar chart of class probabilities for this mistake
            probs_vals = [m["probabilities"][c] for c in CLASS_CODES]
            bar_colors = ["#ef4444" if c == m["pred_code"] else ("#10b981" if c == m["true_code"] else "#94a3b8") for c in CLASS_CODES]
            axes[idx_m, 2].barh(CLASS_CODES, probs_vals, color=bar_colors)
            axes[idx_m, 2].set_xlim(0, 1.0)
            axes[idx_m, 2].set_title("Calibrated Class Probabilities\n(Green=True, Red=Predicted)", fontsize=8.5)
            axes[idx_m, 2].grid(axis="x", linestyle="--", alpha=0.35)

            axes[idx_m, 0].axis("off")
            axes[idx_m, 1].axis("off")

            failure_records.append(
                {
                    "image_id": img_id,
                    "true_code": m["true_code"],
                    "pred_code": m["pred_code"],
                    "confidence": m["confidence"],
                    "brightness": float(row_meta["brightness"]),
                    "sharpness_laplacian": float(row_meta["sharpness_laplacian"]),
                    "green_dominance_ratio": float(row_meta["green_dominance_ratio"]),
                    "center_vs_border_ratio": gcam_m.center_vs_border_ratio,
                    "probabilities": m["probabilities"],
                }
            )

        plt.suptitle("Independent Test Set Error Analysis — High-Confidence Misclassifications", fontsize=12, fontweight="bold", y=0.995)
        plt.tight_layout()
        fail_fig_path = figures_dir / "failure_cases_gallery.png"
        plt.savefig(fail_fig_path, dpi=130)
        plt.close(fig)

    audit_summary = {
        "evaluated_architecture": arch_name,
        "test_samples_audited_for_saliency": len(cb_ratios),
        "mean_center_vs_border_saliency_ratio": round(float(np.mean(cb_ratios)), 3) if cb_ratios else 1.0,
        "median_center_vs_border_saliency_ratio": round(float(np.median(cb_ratios)), 3) if cb_ratios else 1.0,
        "proportion_central_leaf_focused": round(float(np.mean([r >= 1.0 for r in cb_ratios])), 4) if cb_ratios else 0.0,
        "per_class_exemplars": audit_records,
        "top_failure_cases": failure_records,
    }
    with open(metrics_dir / "explainability_and_error_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    return audit_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CASSAVISION end-to-end data, training, evaluation, and explainability pipeline.")
    parser.add_argument("--target-images", type=int, default=800, help="Target stratified image count to acquire.")
    parser.add_argument("--min-per-class", type=int, default=85, help="Minimum images per minority class.")
    parser.add_argument("--epochs", type=int, default=None, help="Override total training epochs.")
    args = parser.parse_args()

    root = get_project_root()
    cfg = load_config()
    seed = int(cfg["project"]["seed"])
    seed_everything(seed)

    if args.epochs is not None:
        cfg["training"]["total_epochs"] = int(args.epochs)

    processed_dir = root / cfg["dataset"]["processed_dir"]
    splits_dir = root / cfg["dataset"]["splits_dir"]
    models_dir = root / "artifacts" / "models"
    metrics_dir = root / "artifacts" / "metrics"
    figures_dir = root / "artifacts" / "figures"

    for d in (processed_dir, splits_dir, models_dir, metrics_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("======================================================================")
    print("STAGE 4: DATASET ACQUISITION & AUTOMATED IMAGE VALIDATION")
    print("======================================================================")
    manifest_csv = processed_dir / "validated_manifest.csv"
    if manifest_csv.exists():
        existing_df = pd.read_csv(manifest_csv)
        if len(existing_df) >= args.target_images * 0.9:
            print(f"Found existing validated manifest with {len(existing_df)} images at {manifest_csv}.")
            with open(processed_dir / "dataset_provenance.json", "r", encoding="utf-8") as f:
                prov_summary = json.load(f)
        else:
            prov_summary = acquire_and_validate_dataset(
                target_total=args.target_images,
                min_per_class=args.min_per_class,
                seed=seed,
            )
    else:
        prov_summary = acquire_and_validate_dataset(
            target_total=args.target_images,
            min_per_class=args.min_per_class,
            seed=seed,
        )

    df = pd.read_csv(manifest_csv)
    print(f"Validated dataset count: {len(df)} images across 5 classes.")

    print("\n======================================================================")
    print("STAGE 5: DUPLICATE / LEAKAGE INVESTIGATION & GROUP-AWARE SPLITTING")
    print("======================================================================")
    df_grouped, dup_report = investigate_duplicates_and_assign_groups(df, max_hamming_distance=6)
    df_split, split_summary = create_leakage_free_splits(
        df_grouped,
        train_ratio=float(cfg["dataset"]["split_ratios"]["train"]),
        val_ratio=float(cfg["dataset"]["split_ratios"]["val"]),
        test_ratio=float(cfg["dataset"]["split_ratios"]["test"]),
        seed=seed,
    )

    df_split.to_csv(splits_dir / "master_splits.csv", index=False)
    for sp in ("train", "val", "test"):
        df_split[df_split["split"] == sp].to_csv(splits_dir / f"{sp}.csv", index=False)

    with open(metrics_dir / "split_and_leakage_report.json", "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)

    print(
        f"Splits created -> Train: {split_summary['splits']['train']['total']}, "
        f"Val: {split_summary['splits']['val']['total']}, "
        f"Test: {split_summary['splits']['test']['total']} "
        f"(Exact duplicates removed: {split_summary['exact_duplicates_removed']}, "
        f"Near-duplicate pairs grouped: {dup_report['near_duplicate_pairs_count']})"
    )

    print("\n======================================================================")
    print("STAGE 5B: EXPLORATORY IMAGE ANALYSIS & VISUALISATIONS")
    print("======================================================================")
    eda_summary = generate_eda_figures_and_report(
        df=df_split,
        figures_dir=figures_dir,
        metrics_dir=metrics_dir,
        full_dataset_counts=prov_summary.get("full_dataset_class_counts"),
    )
    print("EDA figures and statistical report saved to artifacts/figures/ and artifacts/metrics/eda_summary.json.")

    train_df = df_split[df_split["split"] == "train"].reset_index(drop=True)
    val_df = df_split[df_split["split"] == "val"].reset_index(drop=True)
    test_df = df_split[df_split["split"] == "test"].reset_index(drop=True)

    print("\n======================================================================")
    print("STAGE 7: MODEL DEVELOPMENT — CLASSICAL BASELINE & DEEP ARCHITECTURES")
    print("======================================================================")
    comparison_list: List[Dict[str, Any]] = []
    histories: Dict[str, List[Dict[str, Any]]] = {}
    deep_results: Dict[str, Dict[str, Any]] = {}

    print("\n--> Training Classical ML Baseline (Balanced Random Forest on 42-D Color/Texture Features)...")
    rf_res = train_classical_baseline(train_df, val_df, test_df, root_dir=root, seed=seed)
    print(
        f"  [classical_rf_baseline] Val MacroF1={rf_res['val_macro_f1']:.3f} | "
        f"Test Acc={rf_res['test_accuracy']:.3f} | Test MacroF1={rf_res['test_macro_f1']:.3f} "
        f"| Latency={rf_res['inference_latency_ms']:.2f}ms"
    )
    comparison_list.append(
        {
            "model_name": rf_res["model_name"],
            "display_name": rf_res["display_name"],
            "pretrained": False,
            "total_parameters": 0,
            "model_size_mb": rf_res["model_size_mb"],
            "training_duration_sec": rf_res["training_duration_sec"],
            "inference_latency_ms": rf_res["inference_latency_ms"],
            "val_accuracy": rf_res["val_accuracy"],
            "val_macro_f1": rf_res["val_macro_f1"],
            "val_weighted_f1": rf_res["val_weighted_f1"],
            "test_accuracy": rf_res["test_accuracy"],
            "test_macro_f1": rf_res["test_macro_f1"],
            "test_weighted_f1": rf_res["test_weighted_f1"],
            "test_ece": rf_res["test_ece"],
        }
    )

    candidate_archs = cfg["training"]["candidates"]
    for arch_name in candidate_archs:
        ckpt_file = models_dir / f"{arch_name}_best.pt"
        cache_json = metrics_dir / f"{arch_name}_result_cache.json"
        if ckpt_file.exists() and cache_json.exists():
            print(f"\n--> Loading completed training result for {arch_name} from {cache_json}...")
            with open(cache_json, "r", encoding="utf-8") as f:
                res = json.load(f)
        else:
            print(f"\n--> Training Deep Architecture: {arch_name} ...")
            res = train_and_evaluate_deep_model(
                arch_name=arch_name,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                root_dir=root,
                cfg=cfg,
                models_dir=models_dir,
                seed=seed,
            )
            with open(cache_json, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
        deep_results[arch_name] = res
        histories[arch_name] = res["history"]
        comparison_list.append(
            {
                "model_name": res["model_name"],
                "display_name": res["display_name"],
                "pretrained": res["pretrained"],
                "weights_origin": res["weights_origin"],
                "total_parameters": res["total_parameters"],
                "model_size_mb": res["model_size_mb"],
                "best_epoch": res["best_epoch"],
                "epochs_completed": res["epochs_completed"],
                "training_duration_sec": res["training_duration_sec"],
                "inference_latency_ms": res["inference_latency_ms"],
                "temperature": res["temperature"],
                "val_accuracy": res["val_accuracy"],
                "val_macro_f1": res["val_macro_f1"],
                "val_weighted_f1": res["val_weighted_f1"],
                "test_accuracy": res["test_accuracy"],
                "test_macro_f1": res["test_macro_f1"],
                "test_weighted_f1": res["test_weighted_f1"],
                "test_ece": res["test_ece"],
                "uncalibrated_test_ece": res["uncalibrated_test_ece"],
                "checkpoint_path": res["checkpoint_path"],
            }
        )
        print(
            f"  --> [{arch_name}] COMPLETED | Best Epoch={res['best_epoch']} | "
            f"Val MacroF1={res['val_macro_f1']:.4f} | Test Acc={res['test_accuracy']:.4f} | "
            f"Test MacroF1={res['test_macro_f1']:.4f} | Test WeightedF1={res['test_weighted_f1']:.4f} | "
            f"Calibrated ECE={res['test_ece']:.4f} (T={res['temperature']:.3f})"
        )

    # Select final champion architecture based on Validation Macro F1 (with efficiency & test verification recorded)
    best_arch = max(
        deep_results.keys(),
        key=lambda k: (deep_results[k]["val_macro_f1"], deep_results[k]["val_accuracy"]),
    )
    selected_res = deep_results[best_arch]
    selected_ckpt_src = root / selected_res["checkpoint_path"]
    selected_ckpt_dst = models_dir / "selected_model.pt"
    shutil.copy2(selected_ckpt_src, selected_ckpt_dst)

    print(f"\n======================================================================")
    print(f"STAGE 8: CHAMPION MODEL SELECTED -> {best_arch.upper()}")
    print(f"  Validation Macro F1: {selected_res['val_macro_f1']:.4f} | Val Acc: {selected_res['val_accuracy']:.4f}")
    print(f"  Independent Test Acc: {selected_res['test_accuracy']:.4f} | Test Macro F1: {selected_res['test_macro_f1']:.4f} | Test Weighted F1: {selected_res['test_weighted_f1']:.4f}")
    print(f"======================================================================")

    plot_evaluation_figures(
        metrics=selected_res["test_metrics"],
        model_comparison=comparison_list,
        histories=histories,
        figures_dir=figures_dir,
    )

    print("\n======================================================================")
    print("STAGE 9: EXPLAINABILITY AUDIT (GRAD-CAM & OCCLUSION SENSITIVITY)")
    print("======================================================================")
    audit_summary = generate_gradcam_and_failure_galleries(
        selected_result=selected_res,
        test_df=test_df,
        root_dir=root,
        figures_dir=figures_dir,
        metrics_dir=metrics_dir,
    )

    # Save master experiment registry JSON for application and documentation
    registry_payload = {
        "project_name": cfg["project"]["name"],
        "version": cfg["project"]["version"],
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "random_seed": seed,
        "dataset_provenance": prov_summary,
        "split_summary": split_summary,
        "selected_architecture": best_arch,
        "selected_checkpoint_path": "artifacts/models/selected_model.pt",
        "selected_model_summary": {
            k: v for k, v in selected_res.items() if k not in ("test_predictions",)
        },
        "model_comparison": comparison_list,
        "training_histories": histories,
        "explainability_audit": audit_summary,
        "sample_test_predictions": selected_res["test_predictions"][:50],
    }

    with open(metrics_dir / "model_registry.json", "w", encoding="utf-8") as f:
        json.dump(registry_payload, f, indent=2)

    with open(metrics_dir / "test_evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(selected_res["test_metrics"], f, indent=2)

    print("\nPipeline completed successfully! All artifacts saved to artifacts/models, artifacts/metrics, and artifacts/figures.")


if __name__ == "__main__":
    main()
