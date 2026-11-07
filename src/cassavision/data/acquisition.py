"""Reproducible dataset acquisition pipeline for Makerere / NaCRRI Cassava Leaf Disease Dataset."""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

from cassavision.config import CANONICAL_CLASS_MAP, get_project_root, load_config, seed_everything
from cassavision.data.validation import validate_image_file

logger = logging.getLogger(__name__)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def fetch_official_metadata(repo_id: str, raw_dir: Path) -> Tuple[Path, Path]:
    """Download the official label_num_to_disease_map.json and train.csv from the dataset repository."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    label_map_dest = raw_dir / "label_num_to_disease_map.json"
    train_csv_dest = raw_dir / "full_train_21397.csv"

    if not label_map_dest.exists():
        cached_map = hf_hub_download(repo_id=repo_id, filename="label_num_to_disease_map.json", repo_type="dataset")
        shutil.copy2(cached_map, label_map_dest)

    if not train_csv_dest.exists():
        cached_csv = hf_hub_download(repo_id=repo_id, filename="train.csv", repo_type="dataset")
        shutil.copy2(cached_csv, train_csv_dest)

    # Verify label mapping matches CANONICAL_CLASS_MAP
    with open(label_map_dest, "r", encoding="utf-8") as f:
        loaded_map = json.load(f)

    for k, v in CANONICAL_CLASS_MAP.items():
        if loaded_map.get(str(k)) != v["name"]:
            raise ValueError(
                f"Label mapping mismatch for index {k}: expected '{v['name']}', got '{loaded_map.get(str(k))}'"
            )

    return label_map_dest, train_csv_dest


def resolve_remote_file_index(repo_id: str, raw_dir: Path) -> Dict[str, str]:
    """Build or load a mapping of image_id -> remote path in the Hugging Face repository."""
    cache_index_path = raw_dir / "remote_file_index.json"
    if cache_index_path.exists():
        with open(cache_index_path, "r", encoding="utf-8") as f:
            return json.load(f)

    api = HfApi()
    info = api.dataset_info(repo_id)
    mapping: Dict[str, str] = {}
    for sibling in info.siblings:
        rfile = sibling.rfilename
        if rfile.startswith("train_images/") and rfile.endswith(".jpg"):
            img_id = Path(rfile).name
            mapping[img_id] = rfile

    with open(cache_index_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f)
    return mapping


def select_stratified_cohort(
    full_csv_path: Path,
    remote_index: Dict[str, str],
    target_total: int = 800,
    min_per_class: int = 85,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Select a reproducible, stratified cohort of genuine cassava leaf images preserving class structure."""
    rng = np.random.default_rng(seed)
    by_class: Dict[int, List[str]] = defaultdict(list)

    with open(full_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row["image_id"]
            label = int(row["label"])
            if img_id in remote_index:
                by_class[label].append(img_id)

    total_full = sum(len(v) for v in by_class.values())
    # Calculate allocation ensuring at least min_per_class for minority classes (e.g. CBB)
    # while preserving the realistic dominance of CMD so class imbalance handling is genuinely tested.
    allocations: Dict[int, int] = {}
    remaining = target_total
    for cls_idx in sorted(by_class.keys()):
        natural_count = int(round(target_total * (len(by_class[cls_idx]) / total_full)))
        allocations[cls_idx] = max(min_per_class, natural_count)

    # Adjust largest class (CMD, label 3) so sum equals target_total exactly
    diff = sum(allocations.values()) - target_total
    if diff != 0 and 3 in allocations:
        allocations[3] = max(min_per_class, allocations[3] - diff)

    selected: List[Dict[str, Any]] = []
    for cls_idx in sorted(by_class.keys()):
        candidates = sorted(by_class[cls_idx])
        n_pick = min(len(candidates), allocations[cls_idx])
        chosen_indices = rng.choice(len(candidates), size=n_pick, replace=False)
        for idx in sorted(chosen_indices):
            img_id = candidates[int(idx)]
            selected.append(
                {
                    "image_id": img_id,
                    "label": cls_idx,
                    "class_code": CANONICAL_CLASS_MAP[cls_idx]["code"],
                    "class_name": CANONICAL_CLASS_MAP[cls_idx]["name"],
                    "remote_path": remote_index[img_id],
                }
            )
    return selected


def _download_single_image(
    repo_id: str,
    item: Dict[str, Any],
    images_dir: Path,
    max_retries: int = 3,
) -> Optional[Dict[str, Any]]:
    """Download and copy a single image into raw/images/<class_code>/<image_id>."""
    cls_dir = images_dir / item["class_code"]
    cls_dir.mkdir(parents=True, exist_ok=True)
    dest_path = cls_dir / item["image_id"]

    if dest_path.exists() and dest_path.stat().st_size > 1024:
        item_out = dict(item)
        item_out["local_path"] = str(dest_path)
        return item_out

    for attempt in range(max_retries):
        try:
            cached_path = hf_hub_download(
                repo_id=repo_id,
                filename=item["remote_path"],
                repo_type="dataset",
            )
            shutil.copy2(cached_path, dest_path)
            item_out = dict(item)
            item_out["local_path"] = str(dest_path)
            return item_out
        except Exception as exc:
            if attempt == max_retries - 1:
                logger.warning("Failed to download %s after %d attempts: %s", item["image_id"], max_retries, exc)
                return None
            time.sleep(1.0 * (attempt + 1))
    return None


def acquire_and_validate_dataset(
    target_total: int = 800,
    min_per_class: int = 85,
    max_workers: int = 12,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run full dataset acquisition, image validation, and provenance manifest generation."""
    seed_everything(seed)
    root = get_project_root()
    cfg = load_config()
    repo_id = cfg["dataset"]["hf_repo_id"]
    raw_dir = root / cfg["dataset"]["raw_dir"]
    images_dir = root / cfg["dataset"]["images_dir"]
    processed_dir = root / cfg["dataset"]["processed_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Fetching official metadata from {repo_id}...")
    label_map_path, full_csv_path = fetch_official_metadata(repo_id, raw_dir)

    print("[2/4] Resolving remote file index for 21,397 competition images...")
    remote_index = resolve_remote_file_index(repo_id, raw_dir)

    print(f"[3/4] Selecting stratified cohort (target={target_total}, min_per_class={min_per_class}, seed={seed})...")
    cohort = select_stratified_cohort(
        full_csv_path=full_csv_path,
        remote_index=remote_index,
        target_total=target_total,
        min_per_class=min_per_class,
        seed=seed,
    )

    print(f"[4/4] Downloading & validating {len(cohort)} genuine field images ({max_workers} threads)...")
    t0 = time.time()
    downloaded_items: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_download_single_image, repo_id, item, images_dir): item
            for item in cohort
        }
        completed = 0
        for fut in as_completed(futures):
            completed += 1
            res = fut.result()
            if res is not None:
                downloaded_items.append(res)
            if completed % 50 == 0 or completed == len(cohort):
                elapsed = time.time() - t0
                print(f"  Progress: {completed}/{len(cohort)} ({len(downloaded_items)} valid) in {elapsed:.1f}s")

    # Sort deterministically by image_id
    downloaded_items.sort(key=lambda x: x["image_id"])

    # Perform automated image validation across all downloaded images
    validated_records: List[Dict[str, Any]] = []
    corrupted_records: List[Dict[str, Any]] = []

    for item in downloaded_items:
        val_res = validate_image_file(item["local_path"])
        rec = {
            **item,
            "local_relpath": str(Path(item["local_path"]).relative_to(root).as_posix()),
            **val_res.to_dict(),
        }
        if val_res.is_valid:
            validated_records.append(rec)
        else:
            corrupted_records.append(rec)

    # Save validated metadata CSV and provenance JSON
    manifest_csv_path = processed_dir / "validated_manifest.csv"
    if validated_records:
        fieldnames = [k for k in validated_records[0].keys() if k != "quality_warnings"] + ["quality_warnings_str"]
        with open(manifest_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in validated_records:
                row_dict = {k: v for k, v in r.items() if k != "quality_warnings"}
                row_dict["quality_warnings_str"] = "; ".join(r.get("quality_warnings", []))
                writer.writerow(row_dict)

    full_counts = Counter()
    with open(full_csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            full_counts[int(row["label"])] += 1

    acquired_counts = Counter(r["label"] for r in validated_records)
    summary = {
        "dataset_name": cfg["dataset"]["name"],
        "source_repository": repo_id,
        "original_competition_url": "https://www.kaggle.com/competitions/cassava-leaf-disease-classification",
        "origin_institutions": [
            "Makerere University AI Lab (Kampala, Uganda)",
            "National Crops Resources Research Institute (NaCRRI)",
        ],
        "full_dataset_total_images": sum(full_counts.values()),
        "full_dataset_class_counts": {
            CANONICAL_CLASS_MAP[k]["code"]: full_counts[k] for k in range(5)
        },
        "acquired_total_images": len(validated_records),
        "corrupted_or_rejected_images": len(corrupted_records),
        "acquired_class_counts": {
            CANONICAL_CLASS_MAP[k]["code"]: acquired_counts[k] for k in range(5)
        },
        "acquired_class_ByIndex": {str(k): acquired_counts[k] for k in range(5)},
        "random_seed": seed,
        "manifest_csv": str(manifest_csv_path.relative_to(root).as_posix()),
    }

    provenance_path = processed_dir / "dataset_provenance.json"
    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
