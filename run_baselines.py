"""Run the baseline ladder and write it to reports/.

This is the hook of the whole submission: what does "free" look like, in
both ranking quality and rupees, before any model exists.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.baselines import build_ladder, score_ladder
from src.costs import CostModel
from src.data import (
    amounts_inr,
    assert_no_temporal_leakage,
    describe_splits,
    load_raw,
    make_splits,
)

REPORTS = Path(__file__).resolve().parent / "reports"


def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    costs = CostModel.load()
    df = load_raw()
    splits = make_splits(df)
    assert_no_temporal_leakage(df, splits)
    amounts = amounts_inr(df, costs.usd_to_inr)

    y_test = df["isFraud"].to_numpy().astype(bool)[splits.test]
    amounts_test = amounts[splits.test]

    print("=== splits ===")
    print(describe_splits(df, splits).to_string(index=False))

    rungs, chosen = build_ladder(df, splits, costs, amounts)
    table = score_ladder(rungs, y_test, amounts_test, costs)

    print("\n=== baseline ladder (out-of-time test window) ===")
    print(f"cost model version {costs.version}, {costs.usd_to_inr:.0f} INR/USD")
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    print("\n=== tuned on val-B ===")
    for key, value in chosen.items():
        print(f"  {key}: {value}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS / "baseline_ladder.csv", index=False)
    print(f"\nwritten to {REPORTS / 'baseline_ladder.csv'}")


if __name__ == "__main__":
    main()
