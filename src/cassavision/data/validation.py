"""Automated image validation, perceptual hashing, quality diagnostics, and OOD screening."""

from __future__ import annotations

import hashlib
import io
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class ImageValidationResult:
    """Structured validation result for an image file or uploaded buffer."""

    filepath: str
    is_valid: bool
    error_message: Optional[str] = None
    format: Optional[str] = None
    mode: Optional[str] = None
    width: int = 0
    height: int = 0
    aspect_ratio: float = 0.0
    file_size_bytes: int = 0
    md5_hash: Optional[str] = None
    dhash: Optional[str] = None
    mean_r: float = 0.0
    mean_g: float = 0.0
    mean_b: float = 0.0
    std_r: float = 0.0
    std_g: float = 0.0
    std_b: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    sharpness_laplacian: float = 0.0
    green_dominance_ratio: float = 0.0
    has_transparency: bool = False
    is_grayscale: bool = False
    quality_warnings: List[str] = None

    def __post_init__(self) -> None:
        if self.quality_warnings is None:
            self.quality_warnings = []

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_dhash(image: Image.Image, hash_size: int = 8) -> str:
    """Compute difference perceptual hash (dHash) as a 64-bit hex string for near-duplicate detection."""
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(gray, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bit_string = "".join("1" if b else "0" for b in diff.flatten())
    return f"{int(bit_string, 2):016x}"


def hamming_distance_hex(hash1: str, hash2: str) -> int:
    """Compute bit-level Hamming distance between two 64-bit hexadecimal dHash strings."""
    if not hash1 or not hash2 or len(hash1) != len(hash2):
        return 64
    xor_val = int(hash1, 16) ^ int(hash2, 16)
    return bin(xor_val).count("1")


def compute_sharpness_laplacian(gray_arr: np.ndarray) -> float:
    """Compute variance of the discrete 2D Laplacian operator as a focus/blur metric."""
    # Standard 3x3 Laplacian kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    g = gray_arr.astype(np.float32)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = (
        g[:-2, 1:-1]
        + g[2:, 1:-1]
        + g[1:-1, :-2]
        + g[1:-1, 2:]
        - 4.0 * g[1:-1, 1:-1]
    )
    return float(np.var(lap))


def compute_green_dominance_ratio(rgb_arr: np.ndarray) -> float:
    """Estimate fraction of pixels exhibiting vegetation/foliage chromaticity (G > R*0.85 and G > B*1.05)."""
    r = rgb_arr[:, :, 0].astype(np.float32)
    g = rgb_arr[:, :, 1].astype(np.float32)
    b = rgb_arr[:, :, 2].astype(np.float32)
    # Excess Green Index (ExG = 2G - R - B) or relative vegetation mask including chlorotic yellow-green
    veg_mask = (g > 25.0) & (g >= r * 0.75) & (g >= b * 1.05) & ((2.0 * g - r - b) > -15.0)
    return float(np.mean(veg_mask))


def validate_image_bytes(
    raw_bytes: bytes,
    filename: str = "uploaded_image.jpg",
    min_dimension: int = 32,
    max_dimension: int = 10000,
) -> Tuple[ImageValidationResult, Optional[Image.Image]]:
    """Validate raw image bytes, returning structured diagnostics and a clean RGB PIL Image if valid."""
    ext = Path(filename).suffix.lower()
    if ext and ext not in SUPPORTED_EXTENSIONS:
        return (
            ImageValidationResult(
                filepath=filename,
                is_valid=False,
                error_message=f"Unsupported file extension '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
                file_size_bytes=len(raw_bytes),
            ),
            None,
        )

    if not raw_bytes or len(raw_bytes) == 0:
        return (
            ImageValidationResult(
                filepath=filename,
                is_valid=False,
                error_message="Empty image file (0 bytes).",
                file_size_bytes=0,
            ),
            None,
        )

    md5_hash = hashlib.md5(raw_bytes).hexdigest()

    try:
        with Image.open(io.BytesIO(raw_bytes)) as img_probe:
            img_probe.verify()

        img = Image.open(io.BytesIO(raw_bytes))
        img.load()
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        return (
            ImageValidationResult(
                filepath=filename,
                is_valid=False,
                error_message=f"Corrupted or unreadable image file: {exc}",
                file_size_bytes=len(raw_bytes),
                md5_hash=md5_hash,
            ),
            None,
        )

    fmt = img.format or ext.lstrip(".").upper()
    mode = img.mode
    width, height = img.size

    if width < min_dimension or height < min_dimension:
        return (
            ImageValidationResult(
                filepath=filename,
                is_valid=False,
                error_message=f"Image dimensions ({width}x{height}) below minimum ({min_dimension}x{min_dimension}).",
                format=fmt,
                mode=mode,
                width=width,
                height=height,
                file_size_bytes=len(raw_bytes),
                md5_hash=md5_hash,
            ),
            None,
        )

    if width > max_dimension or height > max_dimension:
        return (
            ImageValidationResult(
                filepath=filename,
                is_valid=False,
                error_message=f"Image dimensions ({width}x{height}) exceed maximum ({max_dimension}x{max_dimension}).",
                format=fmt,
                mode=mode,
                width=width,
                height=height,
                file_size_bytes=len(raw_bytes),
                md5_hash=md5_hash,
            ),
            None,
        )

    has_transparency = mode in ("RGBA", "LA") or (mode == "P" and "transparency" in img.info)
    is_grayscale = mode in ("L", "1", "I", "F")

    warnings_list: List[str] = []
    if has_transparency:
        warnings_list.append("Image contains alpha/transparency channel; composited onto neutral black background.")
    if is_grayscale:
        warnings_list.append("Grayscale image detected; leaf chlorosis and mosaic color cues may be degraded.")

    # Convert safely to 3-channel RGB
    if mode == "RGBA":
        background = Image.new("RGB", img.size, (0, 0, 0))
        background.paste(img, mask=img.split()[3])
        rgb_img = background
    else:
        rgb_img = img.convert("RGB")

    # Compute pixel statistics on downsampled copy if very large for fast validation
    stat_img = rgb_img if (width <= 512 and height <= 512) else rgb_img.resize((256, 256), Image.Resampling.BILINEAR)
    arr = np.asarray(stat_img, dtype=np.float32)

    if not np.isfinite(arr).all():
        return (
            ImageValidationResult(
                filepath=filename,
                is_valid=False,
                error_message="Image contains non-finite (NaN/Inf) pixel values.",
                format=fmt,
                mode=mode,
                width=width,
                height=height,
                file_size_bytes=len(raw_bytes),
                md5_hash=md5_hash,
            ),
            None,
        )

    mean_rgb = arr.mean(axis=(0, 1))
    std_rgb = arr.std(axis=(0, 1))
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    brightness = float(gray.mean())
    contrast = float(gray.std())
    sharpness = compute_sharpness_laplacian(gray)
    green_ratio = compute_green_dominance_ratio(arr)
    dhash_str = compute_dhash(rgb_img)

    if brightness < 25.0:
        warnings_list.append(f"Severely underexposed image (brightness={brightness:.1f}/255).")
    elif brightness > 235.0:
        warnings_list.append(f"Severely overexposed image (brightness={brightness:.1f}/255).")

    if contrast < 12.0:
        warnings_list.append(f"Extremely low contrast (std={contrast:.1f}); leaf texture details may be indistinct.")

    if sharpness < 15.0:
        warnings_list.append(f"Low sharpness / motion or focus blur detected (Laplacian var={sharpness:.1f}).")

    if green_ratio < 0.05 and not is_grayscale:
        warnings_list.append(
            f"Very low foliage chromaticity ({green_ratio*100:.1f}% green/yellow-green pixels); image may be out-of-distribution (non-leaf)."
        )

    result = ImageValidationResult(
        filepath=filename,
        is_valid=True,
        error_message=None,
        format=fmt,
        mode=mode,
        width=width,
        height=height,
        aspect_ratio=round(width / float(height), 4),
        file_size_bytes=len(raw_bytes),
        md5_hash=md5_hash,
        dhash=dhash_str,
        mean_r=round(float(mean_rgb[0]), 2),
        mean_g=round(float(mean_rgb[1]), 2),
        mean_b=round(float(mean_rgb[2]), 2),
        std_r=round(float(std_rgb[0]), 2),
        std_g=round(float(std_rgb[1]), 2),
        std_b=round(float(std_rgb[2]), 2),
        brightness=round(brightness, 2),
        contrast=round(contrast, 2),
        sharpness_laplacian=round(sharpness, 2),
        green_dominance_ratio=round(green_ratio, 4),
        has_transparency=has_transparency,
        is_grayscale=is_grayscale,
        quality_warnings=warnings_list,
    )
    return result, rgb_img


def validate_image_file(
    filepath: str | Path,
    min_dimension: int = 32,
    max_dimension: int = 10000,
) -> ImageValidationResult:
    """Validate an image file on disk and return its ImageValidationResult."""
    path = Path(filepath)
    if not path.exists():
        return ImageValidationResult(
            filepath=str(path),
            is_valid=False,
            error_message=f"File does not exist: {path}",
        )
    if path.is_dir():
        return ImageValidationResult(
            filepath=str(path),
            is_valid=False,
            error_message=f"Path is a directory, not an image file: {path}",
        )
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        return ImageValidationResult(
            filepath=str(path),
            is_valid=False,
            error_message=f"OS error reading file: {exc}",
        )
    res, _ = validate_image_bytes(raw_bytes, filename=str(path), min_dimension=min_dimension, max_dimension=max_dimension)
    return res
