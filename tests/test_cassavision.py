"""Comprehensive unit and integration test suite for CASSAVISION."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from cassavision.config import CANONICAL_CLASS_MAP, CLASS_CODES, CLASS_NAMES, load_config, seed_everything
from cassavision.data.preprocessing import (
    CassavaLeafDataset,
    compute_class_weights,
    extract_handcrafted_features,
    get_eval_transforms,
    get_train_transforms,
    preprocess_pil_image,
)
from cassavision.data.splitting import create_leakage_free_splits, investigate_duplicates_and_assign_groups
from cassavision.data.validation import (
    compute_dhash,
    hamming_distance_hex,
    validate_image_bytes,
    validate_image_file,
)
from cassavision.evaluation.metrics import (
    apply_temperature,
    compute_comprehensive_metrics,
    compute_ece,
    fit_temperature_scaling,
)
from cassavision.explainability.gradcam import GradCAMExplainer, compute_occlusion_sensitivity
from cassavision.inference.predictor import CassavaInferenceEngine
from cassavision.models.architectures import build_classifier_model, set_backbone_trainable


def _create_synthetic_leaf_bytes(
    width: int = 300,
    height: int = 240,
    mode: str = "RGB",
    fmt: str = "JPEG",
    green_dominant: bool = True,
) -> bytes:
    """Create a synthetic image buffer for fast, deterministic unit testing."""
    rng = np.random.default_rng(123)
    if mode == "L":
        arr = rng.integers(60, 180, size=(height, width), dtype=np.uint8)
    elif mode == "RGBA":
        arr = np.zeros((height, width, 4), dtype=np.uint8)
        arr[:, :, 0] = 50
        arr[:, :, 1] = 170 if green_dominant else 40
        arr[:, :, 2] = 45
        arr[:, :, 3] = 200
    else:
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        if green_dominant:
            arr[:, :, 0] = rng.integers(40, 90, size=(height, width), dtype=np.uint8)
            arr[:, :, 1] = rng.integers(120, 210, size=(height, width), dtype=np.uint8)
            arr[:, :, 2] = rng.integers(20, 70, size=(height, width), dtype=np.uint8)
        else:
            # Bright blue/red non-leaf pattern
            arr[:, :, 0] = 220
            arr[:, :, 1] = 20
            arr[:, :, 2] = 220
    img = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_canonical_class_mapping_and_config():
    """Verify explicit 5-class mapping consistency across configuration and constants."""
    cfg = load_config()
    assert len(CANONICAL_CLASS_MAP) == 5
    assert CLASS_CODES == ["CBB", "CBSD", "CGM", "CMD", "Healthy"]
    assert len(CLASS_NAMES) == 5
    for idx in range(5):
        assert CANONICAL_CLASS_MAP[idx]["code"] == cfg["dataset"]["classes"][idx]["code"]
        assert CANONICAL_CLASS_MAP[idx]["name"] == cfg["dataset"]["classes"][idx]["name"]


def test_image_validation_valid_and_edge_cases(tmp_path: Path):
    """Test valid RGB, RGBA transparency, grayscale, empty, unsupported extension, too small, and corrupted files."""
    # 1. Valid RGB leaf image
    rgb_bytes = _create_synthetic_leaf_bytes(320, 240, mode="RGB", fmt="JPEG", green_dominant=True)
    res, pil_im = validate_image_bytes(rgb_bytes, filename="leaf.jpg")
    assert res.is_valid is True
    assert pil_im is not None
    assert pil_im.mode == "RGB"
    assert res.width == 320 and res.height == 240
    assert res.green_dominance_ratio > 0.5

    # 2. RGBA image with transparency
    rgba_bytes = _create_synthetic_leaf_bytes(128, 128, mode="RGBA", fmt="PNG")
    res_rgba, pil_rgba = validate_image_bytes(rgba_bytes, filename="leaf_rgba.png")
    assert res_rgba.is_valid is True
    assert res_rgba.has_transparency is True
    assert pil_rgba.mode == "RGB"
    assert any("alpha/transparency" in w for w in res_rgba.quality_warnings)

    # 3. Grayscale image
    gray_bytes = _create_synthetic_leaf_bytes(128, 128, mode="L", fmt="PNG")
    res_gray, pil_gray = validate_image_bytes(gray_bytes, filename="leaf_gray.png")
    assert res_gray.is_valid is True
    assert res_gray.is_grayscale is True
    assert pil_gray.mode == "RGB"
    assert any("Grayscale" in w for w in res_gray.quality_warnings)

    # 4. Empty file (0 bytes)
    res_empty, pil_empty = validate_image_bytes(b"", filename="empty.jpg")
    assert res_empty.is_valid is False
    assert pil_empty is None
    assert "Empty image file" in (res_empty.error_message or "")

    # 5. Unsupported file extension
    res_ext, _ = validate_image_bytes(rgb_bytes, filename="document.pdf")
    assert res_ext.is_valid is False
    assert "Unsupported file extension" in (res_ext.error_message or "")

    # 6. Corrupted / random garbage bytes
    res_corrupt, _ = validate_image_bytes(b"not_an_image_at_all_1234567890", filename="corrupt.jpg")
    assert res_corrupt.is_valid is False
    assert "Corrupted or unreadable" in (res_corrupt.error_message or "")

    # 7. Below minimum dimensions
    tiny_bytes = _create_synthetic_leaf_bytes(16, 16, mode="RGB", fmt="PNG")
    res_tiny, _ = validate_image_bytes(tiny_bytes, filename="tiny.png", min_dimension=32)
    assert res_tiny.is_valid is False
    assert "below minimum" in (res_tiny.error_message or "")

    # 8. Missing file on disk
    res_missing = validate_image_file(tmp_path / "nonexistent.jpg")
    assert res_missing.is_valid is False
    assert "does not exist" in (res_missing.error_message or "")


def test_dhash_and_leakage_free_splitting():
    """Verify exact & near-duplicate detection and zero cluster/MD5/image overlap across train/val/test."""
    rows = []
    rng = np.random.default_rng(42)
    class_counts = {0: 20, 1: 25, 2: 25, 3: 60, 4: 30}
    idx = 0
    for cls_idx, count in class_counts.items():
        for c in range(count):
            idx += 1
            rows.append(
                {
                    "image_id": f"img_{idx:04d}.jpg",
                    "label": cls_idx,
                    "class_code": CANONICAL_CLASS_MAP[cls_idx]["code"],
                    "md5_hash": f"md5_{idx:04d}",
                    "dhash": f"{idx * 987654321 % (1 << 64):016x}",
                }
            )
    # Inject an exact duplicate and a near-duplicate (1-bit Hamming distance)
    rows.append(
        {
            "image_id": "img_exact_dup.jpg",
            "label": rows[0]["label"],
            "class_code": rows[0]["class_code"],
            "md5_hash": rows[0]["md5_hash"],
            "dhash": rows[0]["dhash"],
        }
    )
    near_hash_int = int(rows[1]["dhash"], 16) ^ 1
    rows.append(
        {
            "image_id": "img_near_dup.jpg",
            "label": rows[1]["label"],
            "class_code": rows[1]["class_code"],
            "md5_hash": "md5_near_unique",
            "dhash": f"{near_hash_int:016x}",
        }
    )

    df = pd.DataFrame(rows)
    df_grouped, dup_rep = investigate_duplicates_and_assign_groups(df, max_hamming_distance=6)
    assert dup_rep["exact_duplicate_pairs_count"] >= 1
    assert dup_rep["near_duplicate_pairs_count"] >= 1

    df_split, summary = create_leakage_free_splits(df_grouped, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
    assert summary["exact_duplicates_removed"] == 1
    assert summary["leakage_verification_passed"] is True

    # Ensure img_0002.jpg and img_near_dup.jpg were placed in the exact same split
    sp_orig = df_split.loc[df_split["image_id"] == "img_0002.jpg", "split"].iloc[0]
    sp_near = df_split.loc[df_split["image_id"] == "img_near_dup.jpg", "split"].iloc[0]
    assert sp_orig == sp_near


def test_preprocessing_and_class_weights():
    """Verify training/eval transforms, tensor dimensions, and damped class weights."""
    raw_bytes = _create_synthetic_leaf_bytes(400, 300)
    _, pil_im = validate_image_bytes(raw_bytes, "sample.jpg")
    assert pil_im is not None

    tensor, aligned_crop = preprocess_pil_image(pil_im, image_size=224, eval_resize=256)
    assert tensor.shape == (1, 3, 224, 224)
    assert aligned_crop.size == (224, 224)
    assert torch.isfinite(tensor).all()

    train_tf = get_train_transforms(224)
    aug_tensor = train_tf(pil_im)
    assert aug_tensor.shape == (3, 224, 224)

    feats = extract_handcrafted_features(pil_im)
    assert feats.shape == (42,)
    assert np.isfinite(feats).all()

    # Imbalanced labels (CMD majority = 3, CBB minority = 0)
    labels = [0] * 10 + [1] * 20 + [2] * 20 + [3] * 100 + [4] * 25
    weights = compute_class_weights(labels, num_classes=5, power=0.5)
    assert weights.shape == (5,)
    assert float(weights[0]) > float(weights[3])  # Minority class gets higher weight than majority class
    assert abs(float(weights.mean()) - 1.0) < 1e-5


@pytest.mark.parametrize("arch_name", ["custom_cnn", "mobilenet_v2", "mobilenet_v3_small", "resnet18"])
def test_model_architectures_and_gradcam(arch_name: str):
    """Verify model construction, output dimensions, backbone freezing, and Grad-CAM heatmaps."""
    model, meta = build_classifier_model(arch_name, num_classes=5, pretrained=False)
    assert meta["num_classes"] == 5

    # Test freezing and unfreezing
    set_backbone_trainable(model, arch_name, trainable=False)
    trainable_frozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    set_backbone_trainable(model, arch_name, trainable=True)
    trainable_unfrozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if arch_name != "custom_cnn":
        assert trainable_frozen < trainable_unfrozen

    x = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, 5)

    # Verify Grad-CAM on single image
    raw_bytes = _create_synthetic_leaf_bytes(256, 256)
    _, pil_im = validate_image_bytes(raw_bytes, "cam_test.jpg")
    tensor, aligned_pil = preprocess_pil_image(pil_im)
    explainer = GradCAMExplainer(model, arch_name)
    exp_out = explainer.explain(tensor, aligned_pil, target_class=3)
    assert exp_out.heatmap_2d.shape == (224, 224)
    assert 0.0 <= float(exp_out.heatmap_2d.min()) <= float(exp_out.heatmap_2d.max()) <= 1.0 + 1e-5
    assert exp_out.overlay_rgb.size == (224, 224)


def test_evaluation_metrics_and_temperature_scaling():
    """Verify macro/weighted F1, confusion matrix, ECE, and post-hoc temperature scaling."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 5, size=100)
    # Create overconfident synthetic logits
    logits = rng.normal(0, 2.5, size=(100, 5))
    for i, lbl in enumerate(y_true):
        if i < 70:
            logits[i, lbl] += 4.5

    uncal_probs = apply_temperature(logits, temperature=1.0)
    opt_t = fit_temperature_scaling(logits, y_true)
    cal_probs = apply_temperature(logits, temperature=opt_t)

    y_pred = np.argmax(cal_probs, axis=1)
    metrics = compute_comprehensive_metrics(y_true, y_pred, probs=cal_probs)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert 0.0 <= metrics["weighted_f1"] <= 1.0
    assert len(metrics["confusion_matrix"]) == 5
    assert len(metrics["per_class"]) == 5
    assert "ece" in metrics


def test_inference_engine_and_error_states(tmp_path: Path):
    """Test CassavaInferenceEngine end-to-end with a temporary checkpoint, valid leaf, OOD non-leaf, and invalid file."""
    model, _ = build_classifier_model("custom_cnn", num_classes=5, pretrained=False)
    ckpt_path = tmp_path / "temp_model.pt"
    torch.save(
        {
            "architecture": "custom_cnn",
            "model_version": "1.0.0-test",
            "state_dict": model.state_dict(),
            "class_map": CANONICAL_CLASS_MAP,
            "preprocessing": {
                "image_size": 224,
                "eval_resize": 256,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "temperature": 1.15,
        },
        ckpt_path,
    )

    engine = CassavaInferenceEngine(checkpoint_path=ckpt_path)

    # 1. Valid green leaf image
    leaf_bytes = _create_synthetic_leaf_bytes(256, 256, green_dominant=True)
    resp = engine.predict_from_bytes(leaf_bytes, filename="leaf.jpg", compute_gradcam=True, compute_occlusion=True)
    assert resp.is_valid_image is True
    assert resp.predicted_code in CLASS_CODES
    assert abs(sum(resp.class_probabilities.values()) - 1.0) < 1e-3
    assert resp.gradcam is not None
    assert resp.occlusion_overlay_pil is not None

    # 2. Non-leaf OOD image (magenta/blue with 0% green foliage)
    ood_bytes = _create_synthetic_leaf_bytes(256, 256, green_dominant=False)
    resp_ood = engine.predict_from_bytes(ood_bytes, filename="magenta_car.jpg", compute_gradcam=False)
    assert resp_ood.is_valid_image is True
    assert resp_ood.is_ood_flagged is True
    assert resp_ood.decision_status == "OUT_OF_DISTRIBUTION_WARNING"

    # 3. Corrupted file bytes
    resp_bad = engine.predict_from_bytes(b"corrupted_bytes", filename="bad.jpg")
    assert resp_bad.is_valid_image is False
    assert resp_bad.decision_status == "REJECTED_INVALID_FILE"

    # 4. Missing checkpoint raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        CassavaInferenceEngine(checkpoint_path=tmp_path / "does_not_exist.pt")
