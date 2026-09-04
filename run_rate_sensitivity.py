"""How much does each cost assumption actually matter?

Every rupee figure in this project rests on constants chosen by hand:
Rs 1,000 chargeback fee, 12% margin, 5% churn, Rs 3,000 LTV, and an
88 INR/USD conversion. Reporting a headline that depends on four numbers
nobody measured, without saying how much they move it, would be exactly the
kind of unexamined claim the rest of the project argues against.

So: vary each constant one at a time, re-derive the operating point, and
report what changes.

PROTOCOL. Each variant re-derives the cost-optimal threshold on val-B and
reports realised cost on test at that frozen threshold, exactly as the main
pipeline does. Nothing is re-tuned on test. The model is loaded, never
retrained -- the constants affect the decision policy, not the classifier.

The F1-optimal threshold is computed once and reused because F1 does not
consult the cost model at all. That is precisely why the headline gap moves:
one side of the comparison is anchored and the other is not.

A NOTE ON PRECISION. Thresholds come off a 0.001 grid (see sweep_thresholds).
Where this script reports a threshold as unchanged, it means unchanged to
that resolution -- not exact mathematical invariance.
"""

from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.costs import (
    CostModel,
    bootstrap_cost_difference,
    cost_optimal_threshold,
    decisions_global,
)
from src.data import load_raw, make_splits, prepare_features

REPORTS = Path("reports")
GRID = np.linspace(0.0, 1.0, 1001)
RATES = (70.0, 80.0, 88.0, 94.0, 100.0)

# Constants to sweep at +/- 50%. `usd_to_inr` is handled separately because
# it rescales the amounts themselves rather than a cost coefficient.
#
# `chargeback_amount_multiplier` is included for completeness even though its
# +50% case (1.5) is not physically meaningful -- you cannot lose more than
# the transaction value, and the flat fees are already separate terms. Read
# that row as a stress test, not a scenario. The -50% case is real: it is
# partial recovery, e.g. goods returned or a dispute won.
CONSTANTS = (
    "chargeback_amount_multiplier",
    "chargeback_fee_inr",
    "ops_handling_inr",
    "margin_rate",
    "churn_probability",
    "customer_ltv_inr",
    "support_contact_inr",
)


def load_scored():
    """Probabilities and labels for val-B and test. No retraining."""
    df = load_raw()
    splits = make_splits(df)
    x = prepare_features(df)
    with open("models/fitted.pkl", "rb") as fh:
        fitted = pickle.load(fh)["fitted"]

    y = df["isFraud"].to_numpy().astype(bool)
    usd = df["TransactionAmt"].to_numpy(dtype=np.float64)
    return {
        "prob_val": fitted.calibrator.predict(fitted.raw_score(x.iloc[splits.val_b])),
        "prob_test": fitted.calibrator.predict(fitted.raw_score(x.iloc[splits.test])),
        "y_val": y[splits.val_b],
        "y_test": y[splits.test],
        "usd_val": usd[splits.val_b],
        "usd_test": usd[splits.test],
    }


def evaluate(model: CostModel, d: dict, t_f1: float) -> dict:
    """Re-derive the operating point under `model` and score it on test."""
    amounts_val = d["usd_val"] * model.usd_to_inr
    amounts_test = d["usd_test"] * model.usd_to_inr

    t_cost = cost_optimal_threshold(d["y_val"], d["prob_val"], amounts_val, model, GRID)

    def cost_at(threshold: float) -> float:
        blocked = decisions_global(d["prob_test"], threshold)
        return model.cost_per_10k(d["y_test"], blocked, amounts_test)

    at_zero, at_inf = model.threshold_limits()
    return {
        "t_cost": t_cost,
        "t_star_low": at_inf,
        "t_star_high": at_zero,
        "cost_optimal_per_10k": cost_at(t_cost),
        "f1_optimal_per_10k": cost_at(t_f1),
        "headline_per_10k": cost_at(t_f1) - cost_at(t_cost),
    }


def main() -> None:
    pd.set_option("display.width", 220)
    base = CostModel.load()
    d = load_scored()

    # F1 never consults the cost model, so this is fixed across every variant.
    t_f1 = float(GRID[int(np.argmax([f1_score(d["y_val"], d["prob_val"] >= t) for t in GRID]))])
    baseline = evaluate(base, d, t_f1)

    print(f"cost model version {base.version}   shipped rate {base.usd_to_inr:.0f} INR/USD")
    print(f"F1-optimal threshold {t_f1:.4f}  (cost-independent, fixed across all variants)")
    print(f"baseline: threshold {baseline['t_cost']:.4f}, "
          f"Rs {baseline['cost_optimal_per_10k']:,.0f}/10k, "
          f"headline Rs {baseline['headline_per_10k']:,.0f}/10k")

    # ---- exchange rate --------------------------------------------------
    rows = []
    for rate in RATES:
        result = evaluate(replace(base, usd_to_inr=rate), d, t_f1)
        rows.append({
            "usd_to_inr": rate,
            "t_cost": result["t_cost"],
            "t_star_range": f"{result['t_star_low']:.3f}-{result['t_star_high']:.3f}",
            "cost_optimal_per_10k": result["cost_optimal_per_10k"],
            "headline_per_10k": result["headline_per_10k"],
            "shipped": "<-- shipped" if rate == base.usd_to_inr else "",
        })
    rate_table = pd.DataFrame(rows)

    print("\n=== exchange rate ===")
    print("The shipped 88 is a DECLARED ASSUMPTION, not a spot rate. Today's actual")
    print("rate is ~94; the transactions themselves are from 2019, when it was ~70.")
    print(rate_table.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    unique_thresholds = rate_table["t_cost"].nunique()
    print(f"\ndistinct operating points across rates 70-100: {unique_thresholds}")
    if unique_thresholds == 1:
        print("=> the decision is unchanged at every rate (to the 0.001 grid resolution).")
        print("   Only magnitudes scale. t*(a)'s limits do not move at all, because they")
        print("   are fixed_fp/(fixed_fp+fixed_fn) and margin/(margin+multiplier) --")
        print("   neither contains the rate. It only relocates where each transaction")
        print("   sits along the curve.")

    # ---- the other cost constants ---------------------------------------
    rows = []
    for name in CONSTANTS:
        for label, factor in (("-50%", 0.5), ("baseline", 1.0), ("+50%", 1.5)):
            value = getattr(base, name) * factor
            result = evaluate(replace(base, **{name: value}), d, t_f1)
            rows.append({
                "constant": name,
                "variant": label,
                "value": value,
                "t_cost": result["t_cost"],
                "t_star_range": f"{result['t_star_low']:.3f}-{result['t_star_high']:.3f}",
                "cost_optimal_per_10k": result["cost_optimal_per_10k"],
                "headline_per_10k": result["headline_per_10k"],
            })
    cost_table = pd.DataFrame(rows)

    print("\n=== every other cost constant, at +/- 50% ===")
    print(cost_table.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    moved = (
        cost_table[cost_table.variant != "baseline"]
        .assign(shift=lambda t: (t.t_cost - baseline["t_cost"]).abs())
        .query("shift > 0.0005")
    )
    print("\n=== which assumptions actually move the operating point? ===")
    if moved.empty:
        print("  none -- the threshold survives every +/-50% perturbation")
    else:
        for _, row in moved.iterrows():
            print(f"  {row['constant']:24s} {row['variant']:9s} "
                  f"threshold {baseline['t_cost']:.3f} -> {row['t_cost']:.3f}")

    # ---- where does the headline stop being real? -----------------------
    # Two constants move the operating point, and they are the two that set
    # the FN:FP cost RATIO -- raising the margin makes a false positive
    # dearer, and recovering part of a chargeback makes a false negative
    # cheaper. Both push the cost-optimal threshold up toward the F1-optimal
    # one, which is exactly what closes the gap. A point estimate cannot say
    # whether the gap has actually vanished, so bootstrap each level.
    rows = []
    for name, values in (
        ("margin_rate", (0.06, 0.12, 0.18)),
        ("chargeback_amount_multiplier", (0.5, 1.0, 1.5)),
    ):
        for value in values:
            model = replace(base, **{name: value})
            amounts_val = d["usd_val"] * model.usd_to_inr
            amounts_test = d["usd_test"] * model.usd_to_inr
            t_cost = cost_optimal_threshold(
                d["y_val"], d["prob_val"], amounts_val, model, GRID
            )
            ci = bootstrap_cost_difference(
                d["y_test"], amounts_test,
                decisions_global(d["prob_test"], t_f1),
                decisions_global(d["prob_test"], t_cost),
                model, n_boot=1000,
            )
            rows.append({
                "constant": name,
                "value": value,
                "fn_fp_ratio_at_1k": float(model.fn_cost(1000.0) / model.fp_cost(1000.0)),
                "t_cost": t_cost,
                "headline_per_10k": ci["point_estimate"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "significant": not ci["crosses_zero"],
            })
    margin_table = pd.DataFrame(rows)

    print("\n=== where does the headline stop being real? ===")
    # Per-column formatting: one float_format would round the values and
    # thresholds to 0 while the rupee columns need no decimals.
    shown = margin_table.assign(
        value=lambda t: t.value.map("{:.2f}".format),
        fn_fp_ratio_at_1k=lambda t: t.fn_fp_ratio_at_1k.map("{:.1f}x".format),
        t_cost=lambda t: t.t_cost.map("{:.3f}".format),
        headline_per_10k=lambda t: t.headline_per_10k.map("{:,.0f}".format),
        ci_low=lambda t: t.ci_low.map("{:,.0f}".format),
        ci_high=lambda t: t.ci_high.map("{:,.0f}".format),
    )
    print(shown.to_string(index=False))
    print("\nOne mechanism, two knobs. The headline exists only while a missed")
    print("fraud costs SUBSTANTIALLY more than a false block. Raise the margin, or")
    print("recover part of the chargeback, and that ratio narrows: the cost-optimal")
    print("threshold rises toward the F1-optimal one and the gap between them stops")
    print("being distinguishable from zero.")
    print("\nSo this is a claim about merchants with THIN MARGINS and")
    print("UNRECOVERABLE chargebacks -- a boundary condition rather than a failure,")
    print("but one that has to be stated rather than averaged away.")

    REPORTS.mkdir(exist_ok=True)
    rate_table.drop(columns=["shipped"]).to_csv(REPORTS / "rate_sensitivity.csv", index=False)
    cost_table.to_csv(REPORTS / "cost_sensitivity.csv", index=False)
    margin_table.to_csv(REPORTS / "margin_sensitivity.csv", index=False)
    print(f"\nwritten to {REPORTS/'rate_sensitivity.csv'}, {REPORTS/'cost_sensitivity.csv'} "
          f"and {REPORTS/'margin_sensitivity.csv'}")


if __name__ == "__main__":
    main()
