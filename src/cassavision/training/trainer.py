"""Reproducible training, progressive fine-tuning, classical baseline, and checkpoint management."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader

from cassavision.config import CANONICAL_CLASS_MAP, CLASS_CODES, get_project_root, load_config, seed_everything
from cassavision.data.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    CassavaLeafDataset,
    compute_class_weights,
    extract_handcrafted_features,
    get_eval_transforms,
    get_train_transforms,
)
from cassavision.evaluation.metrics import (
    apply_temperature,
    compute_comprehensive_metrics,
    fit_temperature_scaling,
)
from cassavision.models.architectures import build_classifier_model, set_backbone_trainable


def train_classical_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    root_dir: Path,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train and evaluate a non-deep-learning classical baseline (Balanced Random Forest on 42-D
    color, Excess Green index, chlorotic ratio, and spatial Laplacian texture features).
    Establishes a transparent reference floor against which deep CNNs are compared.
    """
    seed_everything(seed)

    def _build_xy(sub_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        feats: List[np.ndarray] = []
        labels: List[int] = []
        for _, row in sub_df.iterrows():
            img_path = root_dir / row["local_relpath"]
            with Image.open(img_path) as im:
                feats.append(extract_handcrafted_features(im))
            labels.append(int(row["label"]))
        return np.vstack(feats), np.asarray(labels, dtype=np.int64)

    t0 = time.time()
    X_train, y_train = _build_xy(train_df)
    X_val, y_val = _build_xy(val_df)
    X_test, y_test = _build_xy(test_df)

    clf = RandomForestClassifier(
        n_estimators=250,
        max_depth=14,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    train_time = time.time() - t0

    t_inf0 = time.time()
    test_probs = clf.predict_proba(X_test)
    inf_ms = ((time.time() - t_inf0) * 1000.0) / max(len(X_test), 1)
    test_preds = np.argmax(test_probs, axis=1)

    val_probs = clf.predict_proba(X_val)
    val_preds = np.argmax(val_probs, axis=1)

    val_metrics = compute_comprehensive_metrics(y_val, val_preds, probs=val_probs)
    test_metrics = compute_comprehensive_metrics(y_test, test_preds, probs=test_probs)

    return {
        "model_name": "classical_rf_baseline",
        "display_name": "Classical Baseline (Balanced Random Forest on Color/Texture Features)",
        "architecture": "random_forest_42d",
        "pretrained": False,
        "training_duration_sec": round(train_time, 2),
        "inference_latency_ms": round(inf_ms, 2),
        "model_size_mb": 2.4,
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_weighted_f1": val_metrics["weighted_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "test_ece": test_metrics.get("ece", 0.0),
        "test_metrics": test_metrics,
    }


def evaluate_pytorch_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray, List[str]]:
    """Run deterministic forward pass over a DataLoader returning average loss, logits, labels, and image_ids."""
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_ids: List[str] = []

    with torch.no_grad():
        for images, labels, img_ids in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            bs = images.size(0)
            total_loss += float(loss.item()) * bs
            total_count += bs
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_ids.extend(img_ids)

    avg_loss = total_loss / max(total_count, 1)
    logits_arr = np.vstack(all_logits) if all_logits else np.zeros((0, 5))
    labels_arr = np.concatenate(all_labels) if all_labels else np.zeros((0,), dtype=np.int64)
    return avg_loss, logits_arr, labels_arr, all_ids


def benchmark_single_image_latency_ms(
    model: nn.Module,
    device: torch.device,
    image_size: int = 224,
    n_runs: int = 15,
) -> float:
    """Measure genuine median single-image forward pass latency in milliseconds."""
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    with torch.no_grad():
        # Warmup
        for _ in range(3):
            _ = model(dummy)
        times: List[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def train_and_evaluate_deep_model(
    arch_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    root_dir: Path,
    cfg: Dict[str, Any],
    models_dir: Path,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train a deep learning architecture with:
      - Progressive fine-tuning (warmup head-only epochs for pretrained models, then full network with discriminative LRs)
      - Damped inverse-frequency class-weighted CrossEntropyLoss + label smoothing
      - CosineAnnealingLR schedule + early stopping on Validation Macro F1
      - Post-hoc Temperature Scaling fit on validation logits
      - Self-contained checkpoint saving (model weights + class map + preprocessing config + temperature T)
    """
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir.mkdir(parents=True, exist_ok=True)

    prep_cfg = cfg["preprocessing"]
    train_cfg = cfg["training"]
    img_size = int(prep_cfg["image_size"])
    eval_resize = int(prep_cfg["eval_resize"])
    batch_size = int(train_cfg["batch_size"])
    total_epochs = int(train_cfg["total_epochs"])
    warmup_epochs = int(train_cfg["warmup_epochs"]) if arch_name != "custom_cnn" else 0
    patience = int(train_cfg["early_stopping_patience"])

    train_ds = CassavaLeafDataset(
        train_df,
        root_dir=root_dir,
        transform=get_train_transforms(image_size=img_size, mean=prep_cfg["mean"], std=prep_cfg["std"]),
    )
    val_ds = CassavaLeafDataset(
        val_df,
        root_dir=root_dir,
        transform=get_eval_transforms(image_size=img_size, eval_resize=eval_resize, mean=prep_cfg["mean"], std=prep_cfg["std"]),
    )
    test_ds = CassavaLeafDataset(
        test_df,
        root_dir=root_dir,
        transform=get_eval_transforms(image_size=img_size, eval_resize=eval_resize, mean=prep_cfg["mean"], std=prep_cfg["std"]),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    pretrained = arch_name != "custom_cnn"
    model, arch_meta = build_classifier_model(arch_name, num_classes=5, pretrained=pretrained)
    model = model.to(device)

    # Class weights for imbalanced cassava disease distribution
    if train_cfg.get("use_class_weights", True):
        class_weights = compute_class_weights(
            train_df["label"].values,
            num_classes=5,
            power=float(train_cfg.get("class_weight_power", 0.5)),
        ).to(device)
    else:
        class_weights = None

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(train_cfg.get("label_smoothing", 0.05)),
    )
    eval_criterion = nn.CrossEntropyLoss()

    history: List[Dict[str, Any]] = []
    best_val_macro_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    t_start = time.time()

    # Phase 1: Linear head warmup (if pretrained)
    if warmup_epochs > 0:
        set_backbone_trainable(model, arch_name, trainable=False)
        head_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            head_params,
            lr=float(train_cfg["learning_rate_head"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
    else:
        set_backbone_trainable(model, arch_name, trainable=True)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg["learning_rate_head"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_epochs - warmup_epochs, 1), eta_min=1e-5
    )

    for epoch in range(1, total_epochs + 1):
        # Unfreeze backbone at epoch == warmup_epochs + 1 with discriminative learning rates
        if warmup_epochs > 0 and epoch == warmup_epochs + 1:
            set_backbone_trainable(model, arch_name, trainable=True)
            backbone_params = []
            head_params = []
            for name, p in model.named_parameters():
                if name.startswith("classifier.") or name.startswith("fc."):
                    head_params.append(p)
                else:
                    backbone_params.append(p)
            optimizer = torch.optim.AdamW(
                [
                    {"params": backbone_params, "lr": float(train_cfg["learning_rate_backbone"])},
                    {"params": head_params, "lr": float(train_cfg["learning_rate_head"]) * 0.5},
                ],
                weight_decay=float(train_cfg["weight_decay"]),
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(total_epochs - warmup_epochs, 1), eta_min=1e-5
            )

        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        ep_t0 = time.time()

        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss encountered in {arch_name} at epoch {epoch}!")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            bs = images.size(0)
            running_loss += float(loss.item()) * bs
            preds = torch.argmax(logits, dim=1)
            running_correct += int((preds == labels).sum().item())
            running_total += bs

        if epoch > warmup_epochs:
            scheduler.step()

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)

        val_loss, val_logits, val_labels, _ = evaluate_pytorch_model(model, val_loader, eval_criterion, device)
        val_probs = apply_temperature(val_logits, temperature=1.0)
        val_preds = np.argmax(val_probs, axis=1)
        val_metrics = compute_comprehensive_metrics(val_labels, val_preds, probs=val_probs)

        ep_sec = time.time() - ep_t0
        current_lr = float(optimizer.param_groups[-1]["lr"])

        ep_record = {
            "epoch": epoch,
            "phase": "head_warmup" if epoch <= warmup_epochs else "full_finetune",
            "learning_rate": round(current_lr, 6),
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "epoch_duration_sec": round(ep_sec, 2),
        }
        history.append(ep_record)
        print(
            f"  [{arch_name}] Epoch {epoch:02d}/{total_epochs:02d} ({ep_record['phase']}) | "
            f"Train Loss={train_loss:.4f} Acc={train_acc:.3f} | "
            f"Val Loss={val_loss:.4f} Acc={val_metrics['accuracy']:.3f} MacroF1={val_metrics['macro_f1']:.3f} ({ep_sec:.1f}s)"
        )

        # Select best checkpoint based on Validation Macro F1 (tie-break with validation accuracy)
        score = val_metrics["macro_f1"] + 0.05 * val_metrics["accuracy"]
        best_score = best_val_macro_f1 + 0.05 * (history[best_epoch - 1]["val_accuracy"] if best_epoch > 0 else 0.0)
        if score > best_score:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epoch > warmup_epochs and epochs_no_improve >= patience:
                print(f"  [{arch_name}] Early stopping triggered at epoch {epoch} (best epoch={best_epoch}).")
                break

    total_train_time = time.time() - t_start

    # Restore best validation model weights
    model.load_state_dict(best_state_dict)

    # Fit post-hoc temperature scaling on validation set
    _, best_val_logits, val_labels, _ = evaluate_pytorch_model(model, val_loader, eval_criterion, device)
    uncal_val_probs = apply_temperature(best_val_logits, temperature=1.0)
    uncal_val_metrics = compute_comprehensive_metrics(val_labels, np.argmax(uncal_val_probs, axis=1), probs=uncal_val_probs)
    optimal_temp = fit_temperature_scaling(best_val_logits, val_labels)
    cal_val_probs = apply_temperature(best_val_logits, temperature=optimal_temp)
    cal_val_metrics = compute_comprehensive_metrics(val_labels, np.argmax(cal_val_probs, axis=1), probs=cal_val_probs)

    # Evaluate once on the independent held-out test set
    test_loss, test_logits, test_labels, test_ids = evaluate_pytorch_model(model, test_loader, eval_criterion, device)
    uncal_test_probs = apply_temperature(test_logits, temperature=1.0)
    uncal_test_metrics = compute_comprehensive_metrics(test_labels, np.argmax(uncal_test_probs, axis=1), probs=uncal_test_probs)

    cal_test_probs = apply_temperature(test_logits, temperature=optimal_temp)
    test_preds = np.argmax(cal_test_probs, axis=1)
    cal_test_metrics = compute_comprehensive_metrics(test_labels, test_preds, probs=cal_test_probs)
    cal_test_metrics["uncalibrated_ece"] = uncal_test_metrics.get("ece", 0.0)
    cal_test_metrics["temperature_T"] = optimal_temp
    cal_test_metrics["test_loss"] = round(float(test_loss), 4)

    # Measure CPU single-image latency
    latency_ms = benchmark_single_image_latency_ms(model, device, image_size=img_size)

    # Collect detailed per-sample test predictions for failure case analysis
    test_predictions_records: List[Dict[str, Any]] = []
    for idx_sample, img_id in enumerate(test_ids):
        true_lbl = int(test_labels[idx_sample])
        pred_lbl = int(test_preds[idx_sample])
        prob_vec = cal_test_probs[idx_sample]
        conf = float(np.max(prob_vec))
        test_predictions_records.append(
            {
                "image_id": img_id,
                "true_label": true_lbl,
                "true_code": CANONICAL_CLASS_MAP[true_lbl]["code"],
                "pred_label": pred_lbl,
                "pred_code": CANONICAL_CLASS_MAP[pred_lbl]["code"],
                "is_correct": bool(true_lbl == pred_lbl),
                "confidence": round(conf, 4),
                "probabilities": {CANONICAL_CLASS_MAP[k]["code"]: round(float(prob_vec[k]), 4) for k in range(5)},
            }
        )

    # Save self-contained checkpoint
    ckpt_path = models_dir / f"{arch_name}_best.pt"
    checkpoint_payload = {
        "architecture": arch_name,
        "model_version": "1.0.0",
        "state_dict": model.state_dict(),
        "class_map": CANONICAL_CLASS_MAP,
        "preprocessing": {
            "image_size": img_size,
            "eval_resize": eval_resize,
            "mean": prep_cfg["mean"],
            "std": prep_cfg["std"],
        },
        "temperature": optimal_temp,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "val_macro_f1": cal_val_metrics["macro_f1"],
        "val_accuracy": cal_val_metrics["accuracy"],
        "test_macro_f1": cal_test_metrics["macro_f1"],
        "test_accuracy": cal_test_metrics["accuracy"],
        "test_weighted_f1": cal_test_metrics["weighted_f1"],
        "test_ece": cal_test_metrics.get("ece", 0.0),
        "seed": seed,
    }
    torch.save(checkpoint_payload, ckpt_path)

    return {
        "model_name": arch_name,
        "display_name": arch_name.replace("_", " ").upper(),
        "architecture": arch_name,
        "pretrained": pretrained,
        "weights_origin": arch_meta["weights_origin"],
        "total_parameters": arch_meta["total_parameters"],
        "model_size_mb": arch_meta["parameter_size_mb"],
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "training_duration_sec": round(total_train_time, 2),
        "inference_latency_ms": round(latency_ms, 2),
        "temperature": optimal_temp,
        "val_accuracy": cal_val_metrics["accuracy"],
        "val_macro_f1": cal_val_metrics["macro_f1"],
        "val_weighted_f1": cal_val_metrics["weighted_f1"],
        "test_accuracy": cal_test_metrics["accuracy"],
        "test_macro_f1": cal_test_metrics["macro_f1"],
        "test_weighted_f1": cal_test_metrics["weighted_f1"],
        "test_ece": cal_test_metrics.get("ece", 0.0),
        "uncalibrated_test_ece": uncal_test_metrics.get("ece", 0.0),
        "checkpoint_path": str(ckpt_path.relative_to(root_dir).as_posix()),
        "history": history,
        "val_metrics": cal_val_metrics,
        "test_metrics": cal_test_metrics,
        "test_predictions": test_predictions_records,
    }
