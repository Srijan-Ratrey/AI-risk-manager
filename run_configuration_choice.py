"""Pick the deployed configuration honestly, then report it once on test.

The per-band calibration run produced a tempting number: adaptive t*(a)
under per-band calibration beats a global threshold under per-band
calibration by Rs 297,083/10k. That comparison is unfair and we are not
going to use it. Per-band calibration makes the GLOBAL-threshold policy
worse, so quoting that figure means measuring against a baseline we
deliberately handicapped -- the exact move this project exists to criticise.

The honest protocol: score all four (calibration x policy) configurations
on val-B, choose the winner there, and report its realised cost on test.
The full 2x2 is printed on test as well, for transparency rather than for
selection.

Robustness: per-band isotonic on a thin band overfits (the >$1k band has
only 36 fraud cases in val-A), so we re-run the whole thing under a strict
minimum-positives rule to see whether the result survives.
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.costs import (
    CostModel, bootstrap_cost_difference, cost_optimal_threshold,
    decisions_amount_dependent, decisions_global,
)
from src.data import amounts_inr, load_raw, make_splits, prepare_features

BAND_EDGES = [0, 25, 100, 250, 1000, np.inf]
BAND_NAMES = ["<$25", "$25-100", "$100-250", "$250-1k", ">$1k"]
GRID = np.linspace(0.0, 1.0, 1001)


def build(costs, min_positives: int):
    df = load_raw()
    splits = make_splits(df)
    y = df["isFraud"].to_numpy().astype(bool)
    amounts = amounts_inr(df, costs.usd_to_inr)
    usd = df["TransactionAmt"].to_numpy(dtype=np.float64)
    x = prepare_features(df)

    with open("models/fitted.pkl", "rb") as fh:
        fitted = pickle.load(fh)["fitted"]

    parts = ("val_a", "val_b", "test")
    raw = {k: fitted.raw_score(x.iloc[getattr(splits, k)]) for k in parts}
    band = {k: np.asarray(pd.cut(usd[getattr(splits, k)], BAND_EDGES, labels=BAND_NAMES)) for k in parts}
    truth = {k: y[getattr(splits, k)] for k in parts}
    amt = {k: amounts[getattr(splits, k)] for k in parts}

    calibrators, skipped = {}, []
    for name in BAND_NAMES:
        mask = band["val_a"] == name
        positives = int(truth["val_a"][mask].sum())
        if mask.sum() < 500 or positives < min_positives:
            skipped.append((name, positives))
            continue
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        iso.fit(raw["val_a"][mask], truth["val_a"][mask])
        calibrators[name] = iso

    def banded(split):
        out = fitted.calibrator.predict(raw[split]).copy()
        for name, iso in calibrators.items():
            mask = band[split] == name
            out[mask] = iso.predict(raw[split][mask])
        return out

    probs = {
        "global": {k: fitted.calibrator.predict(raw[k]) for k in parts},
        "per-band": {k: banded(k) for k in parts},
    }
    return truth, amt, probs, calibrators, skipped


def evaluate(costs, truth, amt, probs):
    """Cost of every configuration on val-B (for selection) and test (for reporting)."""
    cells = []
    for calib, prob in probs.items():
        threshold = cost_optimal_threshold(truth["val_b"], prob["val_b"], amt["val_b"], costs, GRID)
        for policy in ("global", "adaptive"):
            def blocked(split):
                return (decisions_global(prob[split], threshold) if policy == "global"
                        else decisions_amount_dependent(prob[split], amt[split], costs))
            cells.append({
                "calibration": calib,
                "policy": policy,
                "threshold": threshold if policy == "global" else np.nan,
                "val_b_cost": costs.cost_per_10k(truth["val_b"], blocked("val_b"), amt["val_b"]),
                "test_cost": costs.cost_per_10k(truth["test"], blocked("test"), amt["test"]),
                "_blocked_test": blocked("test"),
            })
    return cells


def report(costs, label, min_positives):
    print(f"\n{'=' * 72}\n{label}  (min positives per band = {min_positives})\n{'=' * 72}")
    truth, amt, probs, calibrators, skipped = build(costs, min_positives)
    print(f"per-band calibrators fitted: {sorted(calibrators)}")
    if skipped:
        print(f"skipped as too thin (band, val-A positives): {skipped}")

    cells = evaluate(costs, truth, amt, probs)
    table = pd.DataFrame(cells).drop(columns=["_blocked_test"])
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    # Selection happens on val-B. Test is only read afterwards.
    chosen = min(cells, key=lambda c: c["val_b_cost"])
    naive = next(c for c in cells if c["calibration"] == "global" and c["policy"] == "global")

    print(f"\nselected on val-B : {chosen['calibration']} calibration + {chosen['policy']} policy")
    print(f"naive default     : global calibration + global policy")

    # Positive = the refinement chosen on val-B costs MORE on test than the
    # naive default, i.e. the validation improvement did not generalise.
    ci = bootstrap_cost_difference(
        truth["test"], amt["test"],
        chosen["_blocked_test"], naive["_blocked_test"],
        costs, n_boot=1000,
    )
    val_delta = naive["val_b_cost"] - chosen["val_b_cost"]
    print(f"\nval-B : selected beat the default by Rs {val_delta:,.0f}/10k")
    print(f"test  : selected cost Rs {chosen['test_cost']:,.0f}/10k "
          f"vs default Rs {naive['test_cost']:,.0f}/10k")
    verdict = ("no significant difference" if ci["crosses_zero"]
               else ("the refinement is WORSE on test" if ci["point_estimate"] > 0
                     else "the refinement is better on test"))
    print(f"        difference Rs {ci['point_estimate']:,.0f}/10k "
          f"[95% CI {ci['ci_low']:,.0f} to {ci['ci_high']:,.0f}] -> {verdict}")
    if val_delta > 0 and ci["point_estimate"] > 0 and not ci["crosses_zero"]:
        print("        => validation gain did not generalise. This is why test is held out.")
    return {"label": label, "min_positives": min_positives,
            "chosen": f"{chosen['calibration']}+{chosen['policy']}",
            "val_b_gain_vs_default": val_delta,
            "chosen_test_cost": chosen["test_cost"],
            "default_test_cost": naive["test_cost"], "ci": ci}


def main() -> None:
    pd.set_option("display.width", 200)
    costs = CostModel.load()
    out = [
        report(costs, "permissive per-band calibration", min_positives=20),
        report(costs, "strict per-band calibration", min_positives=200),
    ]
    with open("reports/configuration_choice.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print("\nwritten to reports/configuration_choice.json")


if __name__ == "__main__":
    main()
