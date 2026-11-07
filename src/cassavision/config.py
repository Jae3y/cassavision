"""Configuration loading, class mapping, and reproducible path management for CASSAVISION."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

# Explicit canonical class mapping matching the Makerere / Kaggle Cassava Leaf Disease Dataset
CANONICAL_CLASS_MAP: Dict[int, Dict[str, str]] = {
    0: {
        "index": 0,
        "code": "CBB",
        "name": "Cassava Bacterial Blight (CBB)",
        "short_name": "Bacterial Blight",
        "pathogen": "Xanthomonas axonopodis pv. manihotis",
        "symptoms": "Angular water-soaked leaf spots, brown necrotic lesions with yellow halos, leaf blighting and wilting.",
        "agronomic_note": "Bacterial infection often spread via rain splash, infected stem cuttings, or farm tools.",
    },
    1: {
        "index": 1,
        "code": "CBSD",
        "name": "Cassava Brown Streak Disease (CBSD)",
        "short_name": "Brown Streak Disease",
        "pathogen": "Cassava brown streak virus (CBSV / UCBSV, Ipomovirus)",
        "symptoms": "Feathery chlorosis along secondary and tertiary veins, yellow blotches on mature lower leaves.",
        "agronomic_note": "Viral disease transmitted by whiteflies (Bemisia tabaci) and infected cuttings; can cause root necrosis.",
    },
    2: {
        "index": 2,
        "code": "CGM",
        "name": "Cassava Green Mottle (CGM)",
        "short_name": "Green Mottle",
        "pathogen": "Cassava green mottle virus (CGMV, Nepovirus)",
        "symptoms": "Faint to distinct green mottling, yellow-green mosaic patterns, mild leaf distortion on young foliage.",
        "agronomic_note": "Viral infection causing irregular green-yellow mottling and crinkling on emerging leaves.",
    },
    3: {
        "index": 3,
        "code": "CMD",
        "name": "Cassava Mosaic Disease (CMD)",
        "short_name": "Mosaic Disease",
        "pathogen": "Cassava mosaic begomoviruses (ACMV / EACMV)",
        "symptoms": "Severe chlorotic mosaic patches, pale yellow-green discoloration, leaf curling, distortion, stunted leaflets.",
        "agronomic_note": "Most prevalent cassava viral disease across Sub-Saharan Africa; transmitted by Bemisia tabaci and cuttings.",
    },
    4: {
        "index": 4,
        "code": "Healthy",
        "name": "Healthy",
        "short_name": "Healthy Foliage",
        "pathogen": "None (Uninfected leaf)",
        "symptoms": "Uniform deep-green palmate lobes, smooth leaf surfaces, intact margins without chlorosis or necrosis.",
        "agronomic_note": "No visible foliar symptoms of CBB, CBSD, CGM, or CMD detected in the inspected leaf region.",
    },
}

CLASS_NAMES: List[str] = [CANONICAL_CLASS_MAP[i]["name"] for i in range(5)]
CLASS_CODES: List[str] = [CANONICAL_CLASS_MAP[i]["code"] for i in range(5)]
CODE_TO_INDEX: Dict[str, int] = {CANONICAL_CLASS_MAP[i]["code"]: i for i in range(5)}
NAME_TO_INDEX: Dict[str, int] = {CANONICAL_CLASS_MAP[i]["name"]: i for i in range(5)}


def get_project_root() -> Path:
    """Resolve the root directory of the CASSAVISION project."""
    env_root = os.environ.get("CASSAVISION_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load YAML configuration file with fallback to configs/default.yaml."""
    root = get_project_root()
    if config_path is None:
        config_path = root / "configs" / "default.yaml"
    else:
        config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML configuration structure in {config_path}")
    return cfg


def seed_everything(seed: int = 42) -> None:
    """Set deterministic random seeds across Python, NumPy, and PyTorch (CPU & CUDA)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
