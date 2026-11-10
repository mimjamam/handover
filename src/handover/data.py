"""Load, clean and split the support-ticket data.

Source: Tobi-Bueck/customer-support-tickets on Hugging Face (CC BY-NC 4.0).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

DATASET_ID = "Tobi-Bueck/customer-support-tickets"
RAW_PATH = Path("data/tickets.parquet")
SEED = 42

# The dataset mixes two things: 10 company departments (German and English)
# and 42 topic categories such as "Sports" (German only, about 300 each).
# Only the departments are a routing problem, so only they are kept.
DEPARTMENTS = [
    "Technical Support",
    "Product Support",
    "Customer Service",
    "IT Support",
    "Billing and Payments",
    "Returns and Exchanges",
    "Service Outages and Maintenance",
    "Sales and Pre-Sales",
    "Human Resources",
    "General Inquiry",
]

# Share of the data in each split. Each has one job:
#   train        fit the classifier
#   calibration  fit the probability calibration and the abstention threshold
#   validation   choose settings (never used for final numbers)
#   test         final numbers, touched once
SPLITS = {"train": 0.60, "calibration": 0.15, "validation": 0.10, "test": 0.15}


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw tickets, downloading them on first use."""
    if not path.exists():
        from datasets import load_dataset

        path.parent.mkdir(parents=True, exist_ok=True)
        load_dataset(DATASET_ID, split="train").to_pandas().to_parquet(path)
    return pd.read_parquet(path)


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep department tickets, build one text field, drop duplicates."""
    df = raw[raw["queue"].isin(DEPARTMENTS) & raw["body"].notna()].copy()
    subject = df["subject"].fillna("").str.strip()
    body = df["body"].str.strip()
    df["text"] = (subject + "\n" + body).str.strip()
    df = df[df["text"].str.len() >= 10]

    # Identical texts must not land in both train and test, so they are
    # removed. If one text carries two different labels, all copies go:
    # the label cannot be trusted.
    df["key"] = df["text"].map(_normalise)
    labels_per_text = df.groupby("key")["queue"].transform("nunique")
    df = df[labels_per_text == 1].drop_duplicates("key")

    df = df.rename(columns={"queue": "label"})
    return df[["text", "label", "language", "priority"]].reset_index(drop=True)


def _bucket(text: str) -> float:
    """Stable number in [0, 1) from the text, so splits never change."""
    digest = hashlib.sha256(f"{SEED}:{text}".encode()).hexdigest()
    return int(digest[:8], 16) / 0x100000000


def split(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `split` column. Assignment depends only on the ticket text."""
    edges, total = [], 0.0
    for name, share in SPLITS.items():
        total += share
        edges.append((total, name))

    def assign(text: str) -> str:
        b = _bucket(text)
        for edge, name in edges:
            if b < edge:
                return name
        return edges[-1][1]

    out = df.copy()
    out["split"] = out["text"].map(assign)
    return out


def load() -> pd.DataFrame:
    """Cleaned tickets with a `split` column."""
    return split(clean(load_raw()))
