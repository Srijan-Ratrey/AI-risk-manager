"""Loading and splitting IEEE-CIS.

Two things in here carry the whole project's credibility:

1.  The split is TEMPORAL, never random. Fraud is non-stationary; a random
    split lets the model see the future and inflates every number.

2.  Validation is split in two. The calibrator is fitted on val-A and the
    decision threshold is tuned on val-B. Doing both on the same rows makes
    the threshold inherit the calibrator's overfit.

Dataset facts worth knowing before reading this file:
  - TransactionDT is a timedelta in SECONDS from an unstated reference, not
    a calendar timestamp. Min is 86400; the train file spans ~182 days.
  - train_identity covers only ~24% of transactions, so device/browser
    features are null for three quarters of rows. That is real signal (the
    missingness itself is informative), not something to impute away.
  - TransactionAmt is USD. We re-denominate to INR via costs.yaml.
  - There is no merchant id, no card id, and no payment-method column.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CACHE = DATA_DIR / "merged.parquet"
SPLIT_FILE = DATA_DIR / "splits.json"

TARGET = "isFraud"
TIME_COL = "TransactionDT"
AMOUNT_COL = "TransactionAmt"
ID_COL = "TransactionID"

# Not features: the label, the id, and the raw clock. TransactionDT is
# excluded deliberately -- a tree will happily memorise the time axis and
# then collapse on the out-of-time test window.
NON_FEATURES = {TARGET, ID_COL, TIME_COL}

SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class Splits:
    """Row-position boundaries of a time-ordered frame."""

    train_end: int
    val_a_end: int
    val_b_end: int
    n_rows: int

    @property
    def train(self) -> slice:
        return slice(0, self.train_end)

    @property
    def val_a(self) -> slice:
        """Calibration fold."""
        return slice(self.train_end, self.val_a_end)

    @property
    def val_b(self) -> slice:
        """Threshold-selection fold. Later than val-A, so it sits closest to
        the test window the frozen threshold will actually face."""
        return slice(self.val_a_end, self.val_b_end)

    @property
    def test(self) -> slice:
        """Out-of-time. Touched once, at the end."""
        return slice(self.val_b_end, self.n_rows)

    def save(self, path: Path = SPLIT_FILE) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path = SPLIT_FILE) -> "Splits":
        return cls(**json.loads(Path(path).read_text()))


def load_raw(data_dir: Path = DATA_DIR, use_cache: bool = True) -> pd.DataFrame:
    """Merge transaction + identity, downcast, sort by time.

    A naive float64 load of train_transaction is ~2-3 GB; downcasting the
    339 V-columns to float32 roughly halves it.
    """
    if use_cache and CACHE.exists():
        return pd.read_parquet(CACHE)

    transactions = pd.read_csv(data_dir / "train_transaction.csv")
    identity = pd.read_csv(data_dir / "train_identity.csv")

    df = transactions.merge(identity, on=ID_COL, how="left")
    del transactions, identity

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include=["int64"]).columns:
        if col not in {ID_COL, TARGET}:
            df[col] = pd.to_numeric(df[col], downcast="integer")

    df = df.sort_values(TIME_COL, kind="mergesort").reset_index(drop=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE, index=False)
    return df


def make_splits(
    df: pd.DataFrame,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
) -> Splits:
    """Time-ordered 60 / 10 / 10 / 20 split.

    Assumes `df` is already sorted by TransactionDT. Boundaries are row
    positions, so they are stable and can be frozen to disk.
    """
    if not df[TIME_COL].is_monotonic_increasing:
        raise ValueError("frame must be sorted by TransactionDT before splitting")

    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    val_a_end = train_end + (val_end - train_end) // 2
    return Splits(train_end=train_end, val_a_end=val_a_end, val_b_end=val_end, n_rows=n)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Object columns -> pandas category, which LightGBM handles natively.

    Categories are built over the FULL frame on purpose: a category mapping
    is not fitted knowledge about the label, and letting the encoding differ
    between train and test would silently corrupt inference. No label or
    aggregate statistic crosses the split boundary anywhere in this file.
    """
    features = df[feature_columns(df)].copy()
    for col in features.select_dtypes(include=["object"]).columns:
        features[col] = features[col].astype("category")
    return features


def amounts_inr(df: pd.DataFrame, usd_to_inr: float) -> np.ndarray:
    """TransactionAmt is USD; the cost model speaks INR."""
    return df[AMOUNT_COL].to_numpy(dtype=np.float64) * usd_to_inr


def describe_splits(df: pd.DataFrame, splits: Splits) -> pd.DataFrame:
    """Per-split row counts, fraud rate and time window.

    The fraud-rate column is the cheap drift check the plan asks for: if the
    base rate moves materially between train and test, say so.
    """
    rows = []
    for name in ("train", "val_a", "val_b", "test"):
        part = df.iloc[getattr(splits, name)]
        rows.append(
            {
                "split": name,
                "rows": len(part),
                "fraud_rate": float(part[TARGET].mean()),
                "day_start": float(part[TIME_COL].min() / SECONDS_PER_DAY),
                "day_end": float(part[TIME_COL].max() / SECONDS_PER_DAY),
                "mean_amount_usd": float(part[AMOUNT_COL].mean()),
            }
        )
    return pd.DataFrame(rows)


def assert_no_temporal_leakage(df: pd.DataFrame, splits: Splits) -> None:
    """Every split's time window must strictly follow the previous one."""
    bounds = [
        (name, df.iloc[getattr(splits, name)][TIME_COL])
        for name in ("train", "val_a", "val_b", "test")
    ]
    for (earlier_name, earlier), (later_name, later) in zip(bounds, bounds[1:]):
        if not earlier.max() <= later.min():
            raise AssertionError(
                f"temporal leakage: {earlier_name} overlaps {later_name}"
            )


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from src.costs import CostModel

    frame = load_raw()
    split = make_splits(frame)
    assert_no_temporal_leakage(frame, split)
    split.save()

    model = CostModel.load()
    print(f"rows={len(frame):,}  columns={frame.shape[1]}")
    print(f"identity coverage: {frame['DeviceType'].notna().mean():.1%}")
    print(f"overall fraud rate: {frame[TARGET].mean():.4%}")
    print(f"never-fraud accuracy: {1 - frame[TARGET].mean():.4%}")
    print()
    print(describe_splits(frame, split).to_string(index=False))
    print(f"\nsplit boundaries saved to {SPLIT_FILE}")
