"""Train the model and produce every number the submission claims.

Protocol, enforced in this order and never rearranged:
  1. fit on train (early stopping on the tail of train)
  2. calibrate on val-A
  3. choose thresholds on val-B, then FREEZE them
  4. touch test once, reporting realised cost at the frozen thresholds

Step 4 is the one most entries get wrong. Choosing the threshold on test and
then reporting the cost on test is an oracle number, and it inflates exactly
the figure this project is built to argue about. We report the frozen-
threshold cost as the headline and the test-oracle cost alongside it, so the
size of that gap is visible rather than hidden.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.costs import (
    CostModel,
    bootstrap_cost_difference,
    cost_optimal_threshold,
    decisions_amount_dependent,
    decisions_global,
    derive_review_band,
    sweep_thresholds,
)
from src.data import (
    amounts_inr,
    assert_no_temporal_leakage,
    describe_splits,
    feature_columns,
    load_raw,
    make_splits,
    prepare_features,
)
from src.evaluate import (
    expected_calibration_error,
    pr_auc,
    precision_recall_at,
    recall_at_fpr,
    reliability_curve,
    roc_auc,
)
from src.model import train

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"


def best_threshold_by(objective, y, scores, grid) -> float:
    """Threshold maximising an arbitrary objective, for the comparison points."""
    values = [objective(y, scores >= t) for t in grid]
    return float(grid[int(np.argmax(values))])


def main() -> None:
    pd.set_option("display.width", 220)
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    costs = CostModel.load()
    df = load_raw()
    splits = make_splits(df)
    assert_no_temporal_leakage(df, splits)

    y = df["isFraud"].to_numpy().astype(bool)
    amounts = amounts_inr(df, costs.usd_to_inr)
    x = prepare_features(df)
    names = feature_columns(df)

    print("=== splits ===")
    print(describe_splits(df, splits).to_string(index=False))

    print("\n=== training ===")
    fitted = train(x, y, splits)
    print(f"best_iteration = {fitted.best_iteration}")

    y_val_b, y_test = y[splits.val_b], y[splits.test]
    amounts_val_b, amounts_test = amounts[splits.val_b], amounts[splits.test]

    raw_val_b = fitted.raw_score(x.iloc[splits.val_b])
    raw_test = fitted.raw_score(x.iloc[splits.test])
    prob_val_b = fitted.calibrator.predict(raw_val_b)
    prob_test = fitted.calibrator.predict(raw_test)

    # ---- ranking quality -------------------------------------------------
    results: dict[str, object] = {
        "cost_model_version": costs.version,
        "usd_to_inr": costs.usd_to_inr,
        "best_iteration": fitted.best_iteration,
        "test_pr_auc": pr_auc(y_test, raw_test),
        "test_pr_auc_calibrated": pr_auc(y_test, prob_test),
        "test_roc_auc": roc_auc(y_test, raw_test),
        "test_recall_at_0.5pct_fpr": recall_at_fpr(y_test, raw_test, 0.005),
        "test_accuracy_never_fraud": float(1 - y_test.mean()),
        "test_distinct_probabilities": int(len(np.unique(prob_test))),
    }
    print("\n=== ranking (test, out-of-time) ===")
    print(f"PR-AUC raw          {results['test_pr_auc']:.4f}   <- ranking quality")
    print(
        f"PR-AUC calibrated   {results['test_pr_auc_calibrated']:.4f}   "
        f"<- what the policy consumes"
    )
    # Isotonic is a monotone step function, so it maps many raw scores onto
    # one probability. Those ties cost ranking resolution -- a real price paid
    # for probabilities the cost model can actually use.
    print(
        f"  ties cost {results['test_pr_auc'] - results['test_pr_auc_calibrated']:.4f} PR-AUC; "
        f"{len(np.unique(prob_test))} distinct probabilities over {len(prob_test):,} rows"
    )
    print(f"ROC-AUC             {results['test_roc_auc']:.4f}   (for leaderboard comparison only)")
    print(f"recall @ 0.5% FPR   {results['test_recall_at_0.5pct_fpr']:.4f}")
    if results["test_pr_auc"] > 0.75:
        print("!! PR-AUC above the 0.75 leakage alarm -- investigate before trusting this")

    # ---- calibration -----------------------------------------------------
    # LightGBM's binary objective already emits [0,1] scores, so the "before"
    # curve is the raw output as-is. It is a probability in range only -- the
    # whole point of the next two lines is that being in range is not the
    # same as being calibrated.
    raw_as_prob = raw_test
    results["test_ece_before"] = expected_calibration_error(y_test, raw_as_prob)
    results["test_ece_after"] = expected_calibration_error(y_test, prob_test)
    print("\n=== calibration (test) ===")
    print(f"ECE before isotonic  {results['test_ece_before']:.5f}")
    print(f"ECE after  isotonic  {results['test_ece_after']:.5f}")

    for label, probs in (("before", raw_as_prob), ("after", prob_test)):
        curve = reliability_curve(y_test, probs)
        pd.DataFrame(curve).to_csv(REPORTS / f"reliability_{label}.csv", index=False)

    # ---- thresholds, all chosen on val-B ---------------------------------
    grid = np.linspace(0.0, 1.0, 1001)
    t_cost = cost_optimal_threshold(y_val_b, prob_val_b, amounts_val_b, costs, grid)
    t_f1 = best_threshold_by(f1_score, y_val_b, prob_val_b, grid)
    t_acc = best_threshold_by(
        lambda a, b: float((a == b).mean()), y_val_b, prob_val_b, grid
    )
    print("\n=== thresholds chosen on val-B, then frozen ===")
    print(f"cost-optimal      {t_cost:.4f}")
    print(f"F1-optimal        {t_f1:.4f}")
    print(f"accuracy-optimal  {t_acc:.4f}")

    # ---- realised cost on test at the frozen thresholds ------------------
    rows = []
    for label, t in (
        ("cost-optimal", t_cost),
        ("F1-optimal", t_f1),
        ("accuracy-optimal", t_acc),
    ):
        blocked = decisions_global(prob_test, t)
        precision, recall = precision_recall_at(y_test, blocked)
        rows.append(
            {
                "policy": f"global @ {label}",
                "threshold": t,
                "precision": precision,
                "recall": recall,
                "block_rate": float(blocked.mean()),
                "cost_per_10k_inr": costs.cost_per_10k(y_test, blocked, amounts_test),
            }
        )

    # The contribution: a per-transaction threshold t*(a) instead of one number.
    blocked_adaptive = decisions_amount_dependent(prob_test, amounts_test, costs)
    precision, recall = precision_recall_at(y_test, blocked_adaptive)
    rows.append(
        {
            "policy": "amount-dependent t*(a)",
            "threshold": np.nan,
            "precision": precision,
            "recall": recall,
            "block_rate": float(blocked_adaptive.mean()),
            "cost_per_10k_inr": costs.cost_per_10k(y_test, blocked_adaptive, amounts_test),
        }
    )
    policy_table = pd.DataFrame(rows)
    print("\n=== realised cost on test (thresholds frozen from val-B) ===")
    print(policy_table.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    policy_table.to_csv(REPORTS / "policies.csv", index=False)

    # Oracle bound: the best any single global threshold could have done on
    # test if we had been allowed to peek. The gap to the frozen number is
    # the honest cost of not being able to see the future.
    t_oracle = cost_optimal_threshold(y_test, prob_test, amounts_test, costs, grid)
    oracle_cost = costs.cost_per_10k(y_test, decisions_global(prob_test, t_oracle), amounts_test)
    frozen_cost = float(policy_table.loc[0, "cost_per_10k_inr"])
    results["test_oracle_threshold"] = t_oracle
    results["test_oracle_cost_per_10k"] = oracle_cost
    results["frozen_minus_oracle_per_10k"] = frozen_cost - oracle_cost
    print(f"\ntest-oracle threshold {t_oracle:.4f} -> Rs {oracle_cost:,.0f}/10k")
    print(f"price of freezing on val-B: Rs {frozen_cost - oracle_cost:,.0f}/10k")

    # ---- the headline ----------------------------------------------------
    f1_cost = float(policy_table.loc[1, "cost_per_10k_inr"])
    adaptive_cost = float(policy_table.loc[3, "cost_per_10k_inr"])
    results["headline_f1_minus_cost_per_10k"] = f1_cost - frozen_cost
    results["headline_global_minus_adaptive_per_10k"] = frozen_cost - adaptive_cost

    ci_f1 = bootstrap_cost_difference(
        y_test, amounts_test,
        decisions_global(prob_test, t_f1),
        decisions_global(prob_test, t_cost),
        costs, n_boot=1000,
    )
    ci_adaptive = bootstrap_cost_difference(
        y_test, amounts_test,
        decisions_global(prob_test, t_cost),
        blocked_adaptive,
        costs, n_boot=1000,
    )
    results["ci_f1_vs_cost"] = ci_f1
    results["ci_global_vs_adaptive"] = ci_adaptive

    print("\n=== headline ===")
    print(
        f"Optimising F1 instead of cost: Rs {ci_f1['point_estimate']:,.0f}/10k txns "
        f"[95% CI {ci_f1['ci_low']:,.0f} to {ci_f1['ci_high']:,.0f}]"
        + ("  <-- CI CROSSES ZERO" if ci_f1["crosses_zero"] else "")
    )
    print(
        f"Amount-dependent vs best global: Rs {ci_adaptive['point_estimate']:,.0f}/10k txns "
        f"[95% CI {ci_adaptive['ci_low']:,.0f} to {ci_adaptive['ci_high']:,.0f}]"
        + ("  <-- CI CROSSES ZERO" if ci_adaptive["crosses_zero"] else "")
    )

    # ---- three-way policy and review rate --------------------------------
    # The band is derived on val-B and frozen, exactly like the thresholds
    # above. Deriving it on test and then reporting the test review rate
    # would be the same oracle mistake the threshold protocol avoids.
    band_low, band_high = derive_review_band(prob_val_b, t_cost, costs.max_review_rate)
    reviewed_val = (prob_val_b >= band_low) & (prob_val_b < band_high)
    reviewed = (prob_test >= band_low) & (prob_test < band_high)

    results["review_band"] = [band_low, band_high]
    results["review_rate_val_b"] = float(reviewed_val.mean())
    results["review_rate"] = float(reviewed.mean())
    results["fraud_share_in_review_band"] = float(y_test[reviewed].mean())

    print("\n=== three-way policy (band frozen from val-B) ===")
    print(f"review band       [{band_low:.4f}, {band_high:.4f})")
    print(f"review rate val-B {reviewed_val.mean():.2%}  (ceiling {costs.max_review_rate:.0%})")
    print(f"review rate test  {reviewed.mean():.2%}  <- realised at the frozen band")
    print(f"fraud in band     {y_test[reviewed].mean():.2%}  vs {y_test.mean():.2%} overall")

    # ---- cost curve ------------------------------------------------------
    curve = sweep_thresholds(y_test, prob_test, amounts_test, costs, grid)
    pd.DataFrame(curve).to_csv(REPORTS / "cost_curve.csv", index=False)

    # ---- per-segment breakdown -------------------------------------------
    # No merchant id and no payment-method column exist in IEEE-CIS, so we
    # segment on what is actually there plus amount bands.
    blocked_test = decisions_global(prob_test, t_cost)
    test_df = df.iloc[splits.test]
    segments = []
    amount_band = pd.cut(
        test_df["TransactionAmt"], [0, 25, 100, 250, 1000, np.inf],
        labels=["<$25", "$25-100", "$100-250", "$250-1k", ">$1k"],
    )
    for column, values in (
        ("ProductCD", test_df["ProductCD"]),
        ("card4", test_df["card4"]),
        ("card6", test_df["card6"]),
        ("amount_band", amount_band),
    ):
        for level in values.dropna().unique():
            mask = (values == level).to_numpy()
            if mask.sum() < 200:
                continue
            precision, recall = precision_recall_at(y_test[mask], blocked_test[mask])
            segments.append({
                "dimension": column,
                "segment": str(level),
                "rows": int(mask.sum()),
                "fraud_rate": float(y_test[mask].mean()),
                "pr_auc": pr_auc(y_test[mask], prob_test[mask]) if y_test[mask].any() else np.nan,
                "precision": precision,
                "recall": recall,
                "cost_per_10k_inr": costs.cost_per_10k(y_test[mask], blocked_test[mask], amounts_test[mask]),
            })
    segment_table = pd.DataFrame(segments)
    print("\n=== per-segment (test, at the frozen cost-optimal threshold) ===")
    print(segment_table.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    segment_table.to_csv(REPORTS / "segments.csv", index=False)

    # ---- persist ---------------------------------------------------------
    with open(MODELS / "fitted.pkl", "wb") as fh:
        pickle.dump({"fitted": fitted, "feature_names": names}, fh)
    (MODELS / "thresholds.json").write_text(json.dumps({
        "cost_optimal": t_cost, "f1_optimal": t_f1, "accuracy_optimal": t_acc,
        "review_band": results["review_band"],
        "cost_model_version": costs.version,
    }, indent=2))
    (REPORTS / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nartifacts -> {MODELS}, reports -> {REPORTS}")


if __name__ == "__main__":
    main()
