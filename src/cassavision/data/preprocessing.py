"""Image preprocessing, biologically justified augmentation, PyTorch Dataset, and class imbalance handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from cassavision.config import get_project_root, load_config

IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]


def get_train_transforms(
    image_size: int = 224,
    mean: List[float] = IMAGENET_MEAN,
    std: List[float] = IMAGENET_STD,
) -> transforms.Compose:
    """
    Build biologically justified training-time augmentations:
    - RandomResizedCrop(224, scale=(0.75, 1.0)): Simulates camera-to-leaf distance variation without cropping out entire leaflets.
    - RandomHorizontalFlip(p=0.5) & RandomVerticalFlip(p=0.2): Cassava leaves are photographed at arbitrary orientations in farmers' fields.
    - RandomRotation(degrees=20): Models natural stem and petiole tilt angles.
    - Moderate ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02): Accounts for tropical sunlight/shade variance
      while strictly limiting hue shifts (<=0.02) so healthy green foliage is NEVER artificially shifted into yellow chlorotic mosaic (CMD/CBSD).
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.75, 1.0),
                ratio=(0.85, 1.15),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=20, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def get_eval_transforms(
    image_size: int = 224,
    eval_resize: int = 256,
    mean: List[float] = IMAGENET_MEAN,
    std: List[float] = IMAGENET_STD,
) -> transforms.Compose:
    """
    Deterministic inference/validation/test preprocessing:
    - Resize shorter side to 256, CenterCrop to 224x224, convert to float tensor [0,1], and normalize with ImageNet statistics.
    - Strictly free of random training augmentations.
    """
    return transforms.Compose(
        [
            transforms.Resize(eval_resize, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def preprocess_pil_image(
    img: Image.Image,
    image_size: int = 224,
    eval_resize: int = 256,
    mean: List[float] = IMAGENET_MEAN,
    std: List[float] = IMAGENET_STD,
) -> Tuple[torch.Tensor, Image.Image]:
    """
    Preprocess a single PIL Image for inference & Grad-CAM explainability.
    Returns:
      - input_tensor: shape (1, 3, image_size, image_size)
      - aligned_crop_pil: the exact 224x224 RGB PIL crop seen by the network (for accurate Grad-CAM overlay)
    """
    rgb_img = img.convert("RGB")
    crop_transform = transforms.Compose(
        [
            transforms.Resize(eval_resize, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop((image_size, image_size)),
        ]
    )
    aligned_crop_pil = crop_transform(rgb_img)
    tensor_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    tensor = tensor_transform(aligned_crop_pil).unsqueeze(0)
    return tensor, aligned_crop_pil


def compute_class_weights(
    labels: List[int] | np.ndarray,
    num_classes: int = 5,
    power: float = 0.5,
) -> torch.Tensor:
    """
    Compute damped inverse-frequency class weights:
    w_c = (N / (K * n_c)) ** power, normalized so mean(w) == 1.0.
    Using power=0.5 (square-root inverse frequency) prevents extreme gradient spikes on minority classes
    while boosting recall on CBB, CBSD, CGM, and Healthy relative to majority CMD.
    """
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    total = float(counts.sum())
    raw_weights = (total / (num_classes * counts)) ** power
    norm_weights = raw_weights / raw_weights.mean()
    return torch.tensor(norm_weights, dtype=torch.float32)


def extract_handcrafted_features(img: Image.Image, size: int = 128) -> np.ndarray:
    """
    Extract a 42-dimensional interpretable color, vegetation-index, and spatial texture feature vector
    for the non-deep-learning classical ML baseline (Random Forest / HistGradientBoosting):
      - 3 RGB means + 3 RGB stds + 3 RGB skewness proxies (9)
      - 3 HSV means + 3 HSV stds (6)
      - 16-bin Hue/Green-Yellow ratio histogram (16)
      - Excess Green (ExG = 2G - R - B) mean, std, 10th, 50th, 90th percentiles (5)
      - Chlorotic Yellow-to-Green index ((R+G)/(2*G+1)) mean & std (2)
      - Spatial gradient / Laplacian texture energy (mean, std, max, 90th pct) (4)
    Total = 42 features.
    """
    rgb = img.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    rgb_mean = arr.mean(axis=(0, 1))
    rgb_std = arr.std(axis=(0, 1))
    rgb_q75_q25 = np.percentile(arr, 75, axis=(0, 1)) - np.percentile(arr, 25, axis=(0, 1))

    hsv = np.asarray(rgb.convert("HSV"), dtype=np.float32) / 255.0
    hsv_mean = hsv.mean(axis=(0, 1))
    hsv_std = hsv.std(axis=(0, 1))

    # 16-bin histogram of Hue & Green-Red difference
    gr_diff = (g - r).flatten()
    hist_gr, _ = np.histogram(gr_diff, bins=16, range=(-0.5, 0.5), density=True)
    hist_gr = hist_gr * 0.0625  # normalize bin sum

    # Excess Green Index
    exg = (2.0 * g - r - b).flatten()
    exg_stats = np.array(
        [
            exg.mean(),
            exg.std(),
            np.percentile(exg, 10),
            np.percentile(exg, 50),
            np.percentile(exg, 90),
        ],
        dtype=np.float32,
    )

    # Chlorotic Ratio
    chlor = ((r + g) / (2.0 * g + 0.05)).flatten()
    chlor_stats = np.array([chlor.mean(), chlor.std()], dtype=np.float32)

    # Texture / spatial gradient energy
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gx = np.abs(gray[:, 1:] - gray[:, :-1])
    gy = np.abs(gray[1:, :] - gray[:-1, :])
    lap = np.abs(
        gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * gray[1:-1, 1:-1]
    )
    tex_stats = np.array(
        [gx.mean() + gy.mean(), lap.mean(), lap.std(), np.percentile(lap, 90)],
        dtype=np.float32,
    )

    feat = np.concatenate(
        [rgb_mean, rgb_std, rgb_q75_q25, hsv_mean, hsv_std, hist_gr.astype(np.float32), exg_stats, chlor_stats, tex_stats]
    )
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)


class CassavaLeafDataset(Dataset):
    """PyTorch Dataset for validated Cassava Leaf Disease images."""

    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: Optional[Path] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir or get_project_root()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        rel_or_abs = Path(row["local_relpath"]) if "local_relpath" in row else Path(row["local_path"])
        img_path = rel_or_abs if rel_or_abs.is_absolute() else (self.root_dir / rel_or_abs)

        with Image.open(img_path) as img:
            rgb_img = img.convert("RGB")

        if self.transform is not None:
            tensor = self.transform(rgb_img)
        else:
            tensor = transforms.ToTensor()(rgb_img)

        label = int(row["label"])
        image_id = str(row["image_id"])
        return tensor, label, image_id
