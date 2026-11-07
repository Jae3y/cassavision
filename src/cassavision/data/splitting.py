"""Duplicate/near-duplicate cluster detection and leakage-free stratified dataset splitting."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from cassavision.config import CANONICAL_CLASS_MAP, get_project_root, load_config, seed_everything
from cassavision.data.validation import hamming_distance_hex


class UnionFind:
    """Disjoint-set data structure for grouping exact and near-duplicate images."""

    def __init__(self, elements: List[str]) -> None:
        self.parent = {e: e for e in elements}

    def find(self, x: str) -> str:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            # Deterministic canonical root (lexicographically smaller image_id)
            if rx < ry:
                self.parent[ry] = rx
            else:
                self.parent[rx] = ry


def investigate_duplicates_and_assign_groups(
    df: pd.DataFrame,
    max_hamming_distance: int = 6,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Detect exact (MD5) and near-duplicate (dHash Hamming <= max_hamming_distance) images.
    Assigns a `cluster_id` column so that related/near-duplicate images are never split across
    train, validation, and test partitions.
    """
    df = df.copy()
    image_ids = df["image_id"].astype(str).tolist()
    md5s = df["md5_hash"].astype(str).tolist()
    dhashes = df["dhash"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    uf = UnionFind(image_ids)
    exact_duplicate_pairs: List[Dict[str, Any]] = []
    near_duplicate_pairs: List[Dict[str, Any]] = []

    n = len(image_ids)
    for i in range(n):
        for j in range(i + 1, n):
            if md5s[i] == md5s[j]:
                uf.union(image_ids[i], image_ids[j])
                exact_duplicate_pairs.append(
                    {
                        "image_a": image_ids[i],
                        "image_b": image_ids[j],
                        "label_a": labels[i],
                        "label_b": labels[j],
                        "same_label": labels[i] == labels[j],
                        "md5": md5s[i],
                    }
                )
            else:
                dist = hamming_distance_hex(dhashes[i], dhashes[j])
                if dist <= max_hamming_distance:
                    uf.union(image_ids[i], image_ids[j])
                    near_duplicate_pairs.append(
                        {
                            "image_a": image_ids[i],
                            "image_b": image_ids[j],
                            "label_a": labels[i],
                            "label_b": labels[j],
                            "same_label": labels[i] == labels[j],
                            "hamming_distance": dist,
                        }
                    )

    clusters = [uf.find(img_id) for img_id in image_ids]
    df["cluster_id"] = clusters

    cluster_sizes = Counter(clusters)
    multi_member_clusters = {k: v for k, v in cluster_sizes.items() if v > 1}

    report = {
        "total_images_inspected": n,
        "exact_duplicate_pairs_count": len(exact_duplicate_pairs),
        "near_duplicate_pairs_count": len(near_duplicate_pairs),
        "hamming_threshold": max_hamming_distance,
        "unique_clusters_count": len(cluster_sizes),
        "multi_image_clusters_count": len(multi_member_clusters),
        "cross_label_near_duplicates": sum(1 for p in near_duplicate_pairs if not p["same_label"]),
        "exact_duplicate_pairs": exact_duplicate_pairs[:25],
        "near_duplicate_pairs": near_duplicate_pairs[:25],
    }
    return df, report


def create_leakage_free_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Create group-aware stratified splits (train / val / test) ensuring zero cluster_id or image_id
    overlap across partitions while preserving class distributions.
    """
    seed_everything(seed)
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-5:
        raise ValueError("Split ratios must sum to 1.0")

    df = df.copy().reset_index(drop=True)
    if "cluster_id" not in df.columns:
        df, dup_report = investigate_duplicates_and_assign_groups(df)
    else:
        dup_report = {}

    # Deduplicate exact MD5 duplicates first (keep first occurrence)
    initial_len = len(df)
    df = df.drop_duplicates(subset=["md5_hash"], keep="first").reset_index(drop=True)
    exact_dropped = initial_len - len(df)

    # Group by cluster_id so near-duplicates move together into the same partition
    cluster_df = (
        df.groupby("cluster_id", as_index=False)
        .agg({"label": "first", "image_id": "count"})
        .rename(columns={"image_id": "cluster_size"})
    )

    clusters = cluster_df["cluster_id"].astype(str).tolist()
    cluster_labels = cluster_df["label"].astype(int).tolist()

    # First split: Train vs (Val + Test)
    temp_ratio = val_ratio + test_ratio
    train_clusters, temp_clusters, _, temp_labels = train_test_split(
        clusters,
        cluster_labels,
        test_size=temp_ratio,
        stratify=cluster_labels,
        random_state=seed,
    )

    # Second split: Val vs Test from temp
    relative_test_ratio = test_ratio / temp_ratio
    val_clusters, test_clusters = train_test_split(
        temp_clusters,
        test_size=relative_test_ratio,
        stratify=temp_labels,
        random_state=seed + 1,
    )

    train_set: Set[str] = set(train_clusters)
    val_set: Set[str] = set(val_clusters)
    test_set: Set[str] = set(test_clusters)

    # Verify strict disjointness of clusters
    assert len(train_set.intersection(val_set)) == 0, "Cluster leakage between train and val!"
    assert len(train_set.intersection(test_set)) == 0, "Cluster leakage between train and test!"
    assert len(val_set.intersection(test_set)) == 0, "Cluster leakage between val and test!"

    def _assign_split(cid: str) -> str:
        if cid in train_set:
            return "train"
        if cid in val_set:
            return "val"
        return "test"

    df["split"] = df["cluster_id"].map(_assign_split)

    # Verify zero image_id or md5_hash overlap
    for col in ("image_id", "md5_hash", "cluster_id"):
        tr_vals = set(df[df["split"] == "train"][col])
        va_vals = set(df[df["split"] == "val"][col])
        te_vals = set(df[df["split"] == "test"][col])
        if tr_vals & va_vals or tr_vals & te_vals or va_vals & te_vals:
            raise RuntimeError(f"Leakage detected on column {col}!")

    split_counts: Dict[str, Dict[str, int]] = {}
    for sp in ("train", "val", "test"):
        sub = df[df["split"] == sp]
        counts = Counter(sub["label"].astype(int))
        split_counts[sp] = {
            "total": len(sub),
            "by_class_code": {CANONICAL_CLASS_MAP[k]["code"]: counts[k] for k in range(5)},
            "by_class_index": {str(k): counts[k] for k in range(5)},
        }

    split_summary = {
        "random_seed": seed,
        "exact_duplicates_removed": exact_dropped,
        "leakage_verification_passed": True,
        "zero_image_overlap": True,
        "zero_md5_overlap": True,
        "zero_cluster_overlap": True,
        "splits": split_counts,
        "duplicate_investigation": dup_report,
    }
    return df, split_summary
