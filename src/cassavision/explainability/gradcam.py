"""Grad-CAM and Occlusion Sensitivity explainability for CASSAVISION."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from cassavision.config import CANONICAL_CLASS_MAP, CLASS_CODES
from cassavision.models.architectures import get_gradcam_target_layer


@dataclass
class ExplainabilityOutput:
    """Structured output from Grad-CAM visual explanation."""

    target_class_index: int
    target_class_code: str
    target_class_name: str
    target_probability: float
    heatmap_2d: np.ndarray  # (H, W) normalized in [0, 1]
    heatmap_rgb: Image.Image  # (H, W, 3) colorized jet/magma PIL image
    overlay_rgb: Image.Image  # (H, W, 3) blended overlay on aligned leaf crop
    center_vs_border_ratio: float
    saliency_concentration_note: str
    methodology_disclaimer: str


class GradCAMExplainer:
    """
    Gradient-weighted Class Activation Mapping (Selvaraju et al., ICCV 2017).
    Registers forward and backward hooks on the final convolutional block of the model,
    computes channel-wise importance weights alpha_k^c = GlobalAvgPool(dY^c / dA^k),
    and applies ReLU(sum_k alpha_k^c * A^k) upsampled to the input image dimensions.
    """

    def __init__(self, model: nn.Module, arch_name: str) -> None:
        self.model = model
        self.arch_name = arch_name
        self.target_layer = get_gradcam_target_layer(model, arch_name)
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

    def _save_activation(self, module: nn.Module, inp: Any, out: torch.Tensor) -> None:
        self._activations = out.detach()
        if out.requires_grad:
            out.register_hook(self._save_tensor_gradient)

    def _save_tensor_gradient(self, grad: torch.Tensor) -> None:
        self._gradients = grad.detach()

    def explain(
        self,
        input_tensor: torch.Tensor,
        aligned_crop_pil: Image.Image,
        target_class: Optional[int] = None,
        temperature: float = 1.0,
        alpha_blend: float = 0.45,
        colormap_name: str = "jet",
    ) -> ExplainabilityOutput:
        """
        Generate Grad-CAM heatmap and overlay for input_tensor of shape (1, 3, H, W).
        """
        self.model.eval()
        self._activations = None
        self._gradients = None

        f_hook = self.target_layer.register_forward_hook(self._save_activation)

        try:
            x = input_tensor.clone().detach().requires_grad_(True)
            self.model.zero_grad(set_to_none=True)
            logits = self.model(x)  # (1, K)

            scaled_logits = logits / max(float(temperature), 0.05)
            probs = F.softmax(scaled_logits, dim=-1)[0]

            if target_class is None:
                cls_idx = int(torch.argmax(probs).item())
            else:
                cls_idx = int(target_class)

            target_prob = float(probs[cls_idx].item())
            score = logits[0, cls_idx]
            score.backward()

            if self._activations is None or self._gradients is None:
                raise RuntimeError("Grad-CAM hooks failed to capture activations or gradients.")

            # activations: (1, C, H_f, W_f), gradients: (1, C, H_f, W_f)
            acts = self._activations[0]  # (C, H_f, W_f)
            grads = self._gradients[0]  # (C, H_f, W_f)

            weights = torch.mean(grads, dim=(1, 2), keepdim=True)  # (C, 1, 1)
            cam = torch.sum(weights * acts, dim=0)  # (H_f, W_f)
            cam = F.relu(cam)

            # Upsample to aligned_crop_pil size (W, H)
            w_img, h_img = aligned_crop_pil.size
            cam_up = F.interpolate(
                cam.unsqueeze(0).unsqueeze(0),
                size=(h_img, w_img),
                mode="bilinear",
                align_corners=False,
            )[0, 0]

            cam_np = cam_up.cpu().numpy()
            cam_min, cam_max = float(cam_np.min()), float(cam_np.max())
            if cam_max - cam_min > 1e-8:
                heatmap_2d = (cam_np - cam_min) / (cam_max - cam_min)
            else:
                heatmap_2d = np.zeros_like(cam_np, dtype=np.float32)

        finally:
            f_hook.remove()

        # Colorize heatmap
        cmap = matplotlib.colormaps[colormap_name]
        heatmap_rgba = cmap(heatmap_2d)  # (H, W, 4) in [0, 1]
        heatmap_uint8 = (heatmap_rgba[:, :, :3] * 255.0).astype(np.uint8)
        heatmap_pil = Image.fromarray(heatmap_uint8)

        # Blend overlay with original aligned crop
        orig_arr = np.asarray(aligned_crop_pil.convert("RGB"), dtype=np.float32)
        heat_arr = heatmap_uint8.astype(np.float32)
        # Weight blend intensity slightly more where activation is higher so non-activated regions remain clear
        spatial_alpha = alpha_blend * (0.35 + 0.65 * heatmap_2d[:, :, None])
        overlay_arr = np.clip((1.0 - spatial_alpha) * orig_arr + spatial_alpha * heat_arr, 0, 255).astype(np.uint8)
        overlay_pil = Image.fromarray(overlay_arr)

        # Audit spatial focus: center 70% box vs outer border 15% margin
        my, mx = max(int(h_img * 0.15), 1), max(int(w_img * 0.15), 1)
        center_mask = np.zeros_like(heatmap_2d, dtype=bool)
        center_mask[my : h_img - my, mx : w_img - mx] = True
        center_mean = float(heatmap_2d[center_mask].mean()) if np.any(center_mask) else 0.0
        border_mean = float(heatmap_2d[~center_mask].mean()) if np.any(~center_mask) else 1e-6
        cb_ratio = center_mean / max(border_mean, 1e-4)

        if cb_ratio >= 1.25:
            conc_note = (
                f"Primary spatial activation is concentrated within the central leaf region "
                f"(center-to-border saliency ratio = {cb_ratio:.2f}x)."
            )
        elif cb_ratio >= 0.85:
            conc_note = (
                f"Spatial activation spans both central leaf tissue and peripheral regions "
                f"(center-to-border ratio = {cb_ratio:.2f}x)."
            )
        else:
            conc_note = (
                f"Notice: High activation near image borders/background (center-to-border ratio = {cb_ratio:.2f}x). "
                f"Inspect whether leaf edges or background clutter influenced the score."
            )

        disclaimer = (
            "Grad-CAM highlights spatial regions of the final convolutional feature map whose gradients positively "
            "contribute to the predicted class logit. It is a correlative post-hoc visual interpretability tool and "
            "does NOT prove causal biological understanding or replace laboratory pathology."
        )

        return ExplainabilityOutput(
            target_class_index=cls_idx,
            target_class_code=CANONICAL_CLASS_MAP[cls_idx]["code"],
            target_class_name=CANONICAL_CLASS_MAP[cls_idx]["name"],
            target_probability=round(target_prob, 4),
            heatmap_2d=heatmap_2d,
            heatmap_rgb=heatmap_pil,
            overlay_rgb=overlay_pil,
            center_vs_border_ratio=round(cb_ratio, 3),
            saliency_concentration_note=conc_note,
            methodology_disclaimer=disclaimer,
        )


def compute_occlusion_sensitivity(
    model: nn.Module,
    input_tensor: torch.Tensor,
    aligned_crop_pil: Image.Image,
    target_class: int,
    patch_size: int = 32,
    stride: int = 24,
    temperature: float = 1.0,
) -> Tuple[np.ndarray, Image.Image]:
    """
    Perturbation-based Occlusion Sensitivity map (Zeiler & Fergus, 2014).
    Slides a neutral gray occluding patch across the 224x224 tensor and measures the drop in
    predicted probability for `target_class`. Serves as an independent perturbation check alongside Grad-CAM.
    """
    model.eval()
    _, _, h, w = input_tensor.shape
    with torch.no_grad():
        base_logits = model(input_tensor) / max(float(temperature), 0.05)
        base_prob = float(F.softmax(base_logits, dim=-1)[0, target_class].item())

        ys = list(range(0, h - patch_size + 1, stride))
        xs = list(range(0, w - patch_size + 1, stride))
        occluded_batch: List[torch.Tensor] = []
        coords: List[Tuple[int, int]] = []

        for y0 in ys:
            for x0 in xs:
                occ = input_tensor[0].clone()
                occ[:, y0 : y0 + patch_size, x0 : x0 + patch_size] = 0.0  # ImageNet mean in normalized space
                occluded_batch.append(occ)
                coords.append((y0, x0))

        batch_tensor = torch.stack(occluded_batch, dim=0)
        occ_logits = model(batch_tensor) / max(float(temperature), 0.05)
        occ_probs = F.softmax(occ_logits, dim=-1)[:, target_class].cpu().numpy()

    sens_grid = np.zeros((h, w), dtype=np.float32)
    counts_grid = np.zeros((h, w), dtype=np.float32)

    for (y0, x0), p_occ in zip(coords, occ_probs):
        drop = max(0.0, base_prob - float(p_occ))
        sens_grid[y0 : y0 + patch_size, x0 : x0 + patch_size] += drop
        counts_grid[y0 : y0 + patch_size, x0 : x0 + patch_size] += 1.0

    sens_grid = sens_grid / np.maximum(counts_grid, 1.0)
    s_min, s_max = float(sens_grid.min()), float(sens_grid.max())
    if s_max - s_min > 1e-8:
        sens_norm = (sens_grid - s_min) / (s_max - s_min)
    else:
        sens_norm = np.zeros_like(sens_grid)

    cmap = matplotlib.colormaps["magma"]
    heat_uint8 = (cmap(sens_norm)[:, :, :3] * 255.0).astype(np.uint8)
    orig_arr = np.asarray(aligned_crop_pil.convert("RGB"), dtype=np.float32)
    overlay_arr = np.clip(0.55 * orig_arr + 0.45 * heat_uint8.astype(np.float32), 0, 255).astype(np.uint8)
    return sens_norm, Image.fromarray(overlay_arr)
