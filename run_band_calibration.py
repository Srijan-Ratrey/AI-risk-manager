"""Does per-amount-band calibration rescue the amount-dependent threshold?

The global run produced a null result for t*(a): -Rs 1,830/10k with a CI
spanning zero. The per-segment table suggests why -- PR-AUC falls from 0.62
below $25 to 0.12 above $1k, while cost per 10k rises from Rs 0.8M to
Rs 37.5M over the same range. The amount-dependent rule leans hardest on
probabilities exactly where they are least trustworthy.

If that diagnosis is right, calibrating WITHIN amount bands should sharpen
t*(a) where it matters. This script tests it honestly: bands are defined a
priori, the per-band isotonic fits are on val-A only, the threshold is tuned
on val-B, and test is scored once.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.costs import (
    CostModel, bootstrap_cost_difference, cost_optimal_threshold,
    decisions_amount_dependent, decisions_global,
)
from src.data import amounts_inr, load_raw, make_splits, prepare_features
from src.evaluate import expected_calibration_error, pr_auc

BAND_EDGES = [0, 25, 100, 250, 1000, np.inf]
BAND_NAMES = ["<$25", "$25-100", "$100-250", "$250-1k", ">$1k"]


def bands_of(amount_usd: np.ndarray) -> np.ndarray:
    return np.asarray(pd.cut(amount_usd, BAND_EDGES, labels=BAND_NAMES))


def main() -> None:
    pd.set_option("display.width", 200)
    costs = CostModel.load()
    df = load_raw()
    splits = make_splits(df)
    y = df["isFraud"].to_numpy().astype(bool)
    amounts = amounts_inr(df, costs.usd_to_inr)
    usd = df["TransactionAmt"].to_numpy(dtype=np.float64)
    x = prepare_features(df)

    with open("models/fitted.pkl", "rb") as fh:
        fitted = pickle.load(fh)["fitted"]

    raw = {k: fitted.raw_score(x.iloc[getattr(splits, k)]) for k in ("val_a", "val_b", "test")}
    band = {k: bands_of(usd[getattr(splits, k)]) for k in ("val_a", "val_b", "test")}
    truth = {k: y[getattr(splits, k)] for k in ("val_a", "val_b", "test")}

    # --- diagnosis: is the globally-calibrated model miscalibrated per band?
    global_prob = {k: fitted.calibrator.predict(raw[k]) for k in raw}
    print("=== diagnosis: calibration quality within amount bands (test) ===")
    rows = []
    for name in BAND_NAMES:
        mask = band["test"] == name
        rows.append({
            "band": name,
            "rows": int(mask.sum()),
            "fraud_rate": float(truth["test"][mask].mean()),
            "mean_predicted": float(global_prob["test"][mask].mean()),
            "ece_global_calib": expected_calibration_error(truth["test"][mask], global_prob["test"][mask]),
            "pr_auc": pr_auc(truth["test"][mask], global_prob["test"][mask]),
        })
    diag = pd.DataFrame(rows)
    print(diag.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    # --- per-band isotonic, fitted on val-A only -------------------------
    calibrators = {}
    for name in BAND_NAMES:
        mask = band["val_a"] == name
        if mask.sum() < 500 or truth["val_a"][mask].sum() < 20:
            continue  # too thin to calibrate; fall back to the global fit
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        iso.fit(raw["val_a"][mask], truth["val_a"][mask])
        calibrators[name] = iso
    print(f"\nper-band calibrators fitted: {sorted(calibrators)}")

    def banded_prob(split: str) -> np.ndarray:
        out = fitted.calibrator.predict(raw[split]).copy()
        for name, iso in calibrators.items():
            mask = band[split] == name
            out[mask] = iso.predict(raw[split][mask])
        return out

    banded = {k: banded_prob(k) for k in raw}

    print("\n=== ECE per band: global vs per-band calibration (test) ===")
    rows = []
    for name in BAND_NAMES:
        mask = band["test"] == name
        rows.append({
            "band": name,
            "ece_global": expected_calibration_error(truth["test"][mask], global_prob["test"][mask]),
            "ece_perband": expected_calibration_error(truth["test"][mask], banded["test"][mask]),
        })
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:,.5f}"))
    print(f"\noverall ECE global  {expected_calibration_error(truth['test'], global_prob['test']):.5f}")
    print(f"overall ECE per-band {expected_calibration_error(truth['test'], banded['test']):.5f}")

    # --- does it change the money? ---------------------------------------
    grid = np.linspace(0.0, 1.0, 1001)
    a_val, a_test = amounts[splits.val_b], amounts[splits.test]

    results = {}
    for label, prob in (("global calib", global_prob), ("per-band calib", banded)):
        t = cost_optimal_threshold(truth["val_b"], prob["val_b"], a_val, costs, grid)
        blocked_global = decisions_global(prob["test"], t)
        blocked_adaptive = decisions_amount_dependent(prob["test"], a_test, costs)
        results[label] = {
            "threshold": t,
            "global_cost": costs.cost_per_10k(truth["test"], blocked_global, a_test),
            "adaptive_cost": costs.cost_per_10k(truth["test"], blocked_adaptive, a_test),
            "ci": bootstrap_cost_difference(
                truth["test"], a_test, blocked_global, blocked_adaptive, costs, n_boot=1000
            ),
        }

    print("\n=== amount-dependent t*(a) vs best global threshold ===")
    for label, r in results.items():
        ci = r["ci"]
        flag = "  <-- CI CROSSES ZERO" if ci["crosses_zero"] else "  <-- SIGNIFICANT"
        print(f"\n{label}  (global threshold {r['threshold']:.4f})")
        print(f"  global   Rs {r['global_cost']:,.0f}/10k")
        print(f"  adaptive Rs {r['adaptive_cost']:,.0f}/10k")
        print(f"  saving   Rs {ci['point_estimate']:,.0f}/10k  "
              f"[95% CI {ci['ci_low']:,.0f} to {ci['ci_high']:,.0f}]{flag}")

    pd.DataFrame(rows).to_csv("reports/band_calibration.csv", index=False)
    diag.to_csv("reports/band_diagnosis.csv", index=False)


if __name__ == "__main__":
    main()
