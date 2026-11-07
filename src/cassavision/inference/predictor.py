"""Production inference engine, model checkpoint loader, uncertainty policy, and OOD screening."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from cassavision.config import CANONICAL_CLASS_MAP, CLASS_CODES, get_project_root, load_config
from cassavision.data.preprocessing import IMAGENET_MEAN, IMAGENET_STD, preprocess_pil_image
from cassavision.data.validation import ImageValidationResult, validate_image_bytes
from cassavision.explainability.gradcam import ExplainabilityOutput, GradCAMExplainer, compute_occlusion_sensitivity
from cassavision.models.architectures import build_classifier_model


@dataclass
class PredictionResponse:
    """Complete structured output from the CASSAVISION inference and explainability pipeline."""

    is_valid_image: bool
    validation: Dict[str, Any]
    predicted_index: int
    predicted_code: str
    predicted_name: str
    confidence: float
    uncalibrated_confidence: float
    predictive_entropy: float
    max_possible_entropy: float
    normalized_entropy: float
    temperature: float
    is_uncertain: bool
    is_ood_flagged: bool
    uncertainty_reasons: List[str]
    decision_status: str
    decision_summary: str
    class_probabilities: Dict[str, float]
    uncalibrated_probabilities: Dict[str, float]
    class_details: Dict[str, Any]
    inference_latency_ms: float
    model_architecture: str
    model_version: str
    aligned_crop_pil: Optional[Image.Image] = None
    gradcam: Optional[ExplainabilityOutput] = None
    occlusion_overlay_pil: Optional[Image.Image] = None

    def to_serializable_dict(self) -> Dict[str, Any]:
        d = {
            k: v
            for k, v in self.__dict__.items()
            if k not in ("aligned_crop_pil", "gradcam", "occlusion_overlay_pil")
        }
        if self.gradcam is not None:
            d["gradcam_summary"] = {
                "target_class_code": self.gradcam.target_class_code,
                "target_probability": self.gradcam.target_probability,
                "center_vs_border_ratio": self.gradcam.center_vs_border_ratio,
                "saliency_concentration_note": self.gradcam.saliency_concentration_note,
            }
        return d


class CassavaInferenceEngine:
    """
    Self-contained inference and explainability service for CASSAVISION.
    Loads a trained PyTorch checkpoint along with its verified class mapping, preprocessing parameters,
    and post-hoc calibration temperature $T$.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        config_path: Optional[str | Path] = None,
        device: Optional[str] = None,
    ) -> None:
        self.root = get_project_root()
        self.cfg = load_config(config_path)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        if checkpoint_path is None:
            checkpoint_path = self._resolve_best_checkpoint()
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Trained model checkpoint not found at '{checkpoint_path}'. "
                f"Run the training pipeline first (`python scripts/run_pipeline.py`)."
            )

        self.checkpoint_path = checkpoint_path
        # Load checkpoint safely on target device
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        if not isinstance(ckpt, dict) or "state_dict" not in ckpt or "architecture" not in ckpt:
            raise ValueError(f"Invalid or corrupted checkpoint structure in {self.checkpoint_path}")

        self.arch_name: str = str(ckpt["architecture"])
        self.model_version: str = str(ckpt.get("model_version", "1.0.0"))
        self.temperature: float = float(ckpt.get("temperature", 1.0))
        self.prep_cfg: Dict[str, Any] = ckpt.get("preprocessing", self.cfg["preprocessing"])
        self.class_map: Dict[int, Dict[str, str]] = {
            int(k): v for k, v in ckpt.get("class_map", CANONICAL_CLASS_MAP).items()
        }

        self.model, self.arch_meta = build_classifier_model(
            self.arch_name,
            num_classes=len(self.class_map),
            pretrained=False,
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.explainer = GradCAMExplainer(self.model, self.arch_name)

        eval_cfg = self.cfg.get("evaluation", {})
        self.confidence_threshold = float(eval_cfg.get("confidence_threshold", 0.55))
        self.entropy_threshold = float(eval_cfg.get("entropy_threshold", 1.35))
        self.ood_green_min = float(eval_cfg.get("ood_green_ratio_min", 0.05))
        self.ood_blur_min = float(eval_cfg.get("ood_blur_laplacian_min", 15.0))

    def _resolve_best_checkpoint(self) -> Path:
        """Locate the selected champion checkpoint (`artifacts/models/selected_model.pt` or registry entry)."""
        models_dir = self.root / "artifacts" / "models"
        selected_pt = models_dir / "selected_model.pt"
        if selected_pt.exists():
            return selected_pt
        registry_json = self.root / "artifacts" / "metrics" / "model_registry.json"
        if registry_json.exists():
            with open(registry_json, "r", encoding="utf-8") as f:
                reg = json.load(f)
            rel_ckpt = reg.get("selected_checkpoint_path")
            if rel_ckpt and (self.root / rel_ckpt).exists():
                return self.root / rel_ckpt
        # Fallback to any *_best.pt
        candidates = sorted(models_dir.glob("*_best.pt"))
        if candidates:
            return candidates[0]
        return selected_pt

    def predict_from_bytes(
        self,
        raw_bytes: bytes,
        filename: str = "uploaded_leaf.jpg",
        compute_gradcam: bool = True,
        compute_occlusion: bool = False,
        target_explain_class: Optional[int] = None,
    ) -> PredictionResponse:
        """
        Validate image bytes, preprocess deterministically, run calibrated inference,
        evaluate uncertainty & OOD guardrails, and optionally compute Grad-CAM & Occlusion maps.
        """
        val_res, rgb_img = validate_image_bytes(raw_bytes, filename=filename)
        if not val_res.is_valid or rgb_img is None:
            return PredictionResponse(
                is_valid_image=False,
                validation=val_res.to_dict(),
                predicted_index=-1,
                predicted_code="INVALID",
                predicted_name="Invalid Image Input",
                confidence=0.0,
                uncalibrated_confidence=0.0,
                predictive_entropy=0.0,
                max_possible_entropy=float(np.log(5.0)),
                normalized_entropy=1.0,
                temperature=self.temperature,
                is_uncertain=True,
                is_ood_flagged=True,
                uncertainty_reasons=[val_res.error_message or "Image validation failed."],
                decision_status="REJECTED_INVALID_FILE",
                decision_summary=f"Input file rejected during validation: {val_res.error_message}",
                class_probabilities={},
                uncalibrated_probabilities={},
                class_details={},
                inference_latency_ms=0.0,
                model_architecture=self.arch_name,
                model_version=self.model_version,
            )

        tensor, aligned_crop_pil = preprocess_pil_image(
            rgb_img,
            image_size=int(self.prep_cfg["image_size"]),
            eval_resize=int(self.prep_cfg["eval_resize"]),
            mean=self.prep_cfg["mean"],
            std=self.prep_cfg["std"],
        )
        tensor = tensor.to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.model(tensor)
            uncal_probs_t = F.softmax(logits, dim=-1)[0]
            cal_probs_t = F.softmax(logits / max(self.temperature, 0.05), dim=-1)[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        uncal_probs = uncal_probs_t.cpu().numpy()
        cal_probs = cal_probs_t.cpu().numpy()

        if not np.isfinite(cal_probs).all() or len(cal_probs) != len(self.class_map):
            raise RuntimeError("Model produced invalid or non-finite output probabilities.")

        pred_idx = int(np.argmax(cal_probs))
        conf = float(cal_probs[pred_idx])
        uncal_conf = float(uncal_probs[pred_idx])

        # Shannon predictive entropy H(p) = -sum p_i log(p_i)
        clipped_p = np.clip(cal_probs, 1e-12, 1.0)
        entropy = float(-np.sum(clipped_p * np.log(clipped_p)))
        max_entropy = float(np.log(len(self.class_map)))
        norm_entropy = entropy / max_entropy

        # Evaluate uncertainty & OOD screening criteria
        reasons: List[str] = []
        is_ood = False
        if val_res.green_dominance_ratio < self.ood_green_min and not val_res.is_grayscale:
            is_ood = True
            reasons.append(
                f"Low foliage chromaticity ({val_res.green_dominance_ratio*100:.1f}% green/yellow-green pixels < {self.ood_green_min*100:.0f}% threshold): image may not be a cassava leaf (possible out-of-distribution input)."
            )
        if val_res.sharpness_laplacian < self.ood_blur_min:
            reasons.append(
                f"Severe optical/motion blur detected (Laplacian variance {val_res.sharpness_laplacian:.1f} < {self.ood_blur_min:.1f}); fine lesion textures may be obscured."
            )
        if val_res.is_grayscale:
            reasons.append("Grayscale image lacks RGB chlorosis/mosaic color cues required for reliable differentiation.")

        is_uncertain = is_ood or (conf < self.confidence_threshold) or (entropy > self.entropy_threshold)
        if conf < self.confidence_threshold:
            reasons.append(
                f"Calibrated top-1 confidence ({conf*100:.1f}%) is below the decision reliability threshold ({self.confidence_threshold*100:.0f}%)."
            )
        if entropy > self.entropy_threshold:
            reasons.append(
                f"High predictive entropy ({entropy:.2f} nats / {norm_entropy*100:.0f}% of maximum uniform uncertainty); probability mass is dispersed across multiple classes."
            )

        cls_info = self.class_map[pred_idx]
        if is_ood:
            status = "OUT_OF_DISTRIBUTION_WARNING"
            summary = (
                f"CAUTION — Input flagged by OOD / image-quality guardrails. While the highest softmax score corresponds to "
                f"{cls_info['name']} ({conf*100:.1f}%), the image exhibits non-foliage color distribution or severe quality degradation."
            )
        elif is_uncertain:
            status = "LOW_CONFIDENCE_UNCERTAIN"
            summary = (
                f"UNCERTAIN PREDICTION — Tentative top class is {cls_info['name']} ({conf*100:.1f}% calibrated confidence), "
                f"but model uncertainty exceeds safe diagnostic thresholds. Capture a clearer, well-lit close-up leaf image or consult an agricultural extension officer."
            )
        else:
            status = "CONFIDENT_PREDICTION"
            summary = (
                f"Model predicts {cls_info['name']} with {conf*100:.1f}% calibrated confidence "
                f"(entropy = {entropy:.2f} nats). Review the Grad-CAM attribution map to verify that activation aligns with leaf lamina symptoms."
            )

        gradcam_out: Optional[ExplainabilityOutput] = None
        if compute_gradcam:
            exp_cls = target_explain_class if target_explain_class is not None else pred_idx
            gradcam_out = self.explainer.explain(
                input_tensor=tensor,
                aligned_crop_pil=aligned_crop_pil,
                target_class=exp_cls,
                temperature=self.temperature,
            )

        occ_pil: Optional[Image.Image] = None
        if compute_occlusion:
            exp_cls = target_explain_class if target_explain_class is not None else pred_idx
            _, occ_pil = compute_occlusion_sensitivity(
                model=self.model,
                input_tensor=tensor,
                aligned_crop_pil=aligned_crop_pil,
                target_class=exp_cls,
                temperature=self.temperature,
            )

        class_probs_dict = {
            self.class_map[i]["code"]: round(float(cal_probs[i]), 4) for i in range(len(self.class_map))
        }
        uncal_probs_dict = {
            self.class_map[i]["code"]: round(float(uncal_probs[i]), 4) for i in range(len(self.class_map))
        }

        return PredictionResponse(
            is_valid_image=True,
            validation=val_res.to_dict(),
            predicted_index=pred_idx,
            predicted_code=cls_info["code"],
            predicted_name=cls_info["name"],
            confidence=round(conf, 4),
            uncalibrated_confidence=round(uncal_conf, 4),
            predictive_entropy=round(entropy, 4),
            max_possible_entropy=round(max_entropy, 4),
            normalized_entropy=round(norm_entropy, 4),
            temperature=round(self.temperature, 4),
            is_uncertain=is_uncertain,
            is_ood_flagged=is_ood,
            uncertainty_reasons=reasons,
            decision_status=status,
            decision_summary=summary,
            class_probabilities=class_probs_dict,
            uncalibrated_probabilities=uncal_probs_dict,
            class_details=cls_info,
            inference_latency_ms=round(latency_ms, 2),
            model_architecture=self.arch_name,
            model_version=self.model_version,
            aligned_crop_pil=aligned_crop_pil,
            gradcam=gradcam_out,
            occlusion_overlay_pil=occ_pil,
        )
