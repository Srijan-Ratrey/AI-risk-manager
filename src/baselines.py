"""The baseline ladder.

Before any model, establish what "free" looks like. Every rung is scored on
the same out-of-time test window, under the same protocol the model will
face: threshold tuned on val-B, then frozen and applied to test.

Reporting each rung in BOTH PR-AUC and rupees is the point. A rung can look
weak on PR-AUC and still be hard to beat on money, because money weights
errors by transaction value and ranking metrics do not.

One honest caveat, stated here and in the README: plan.md originally wanted
a velocity rule (">N transactions per card per hour") as the domain-rule
rung. IEEE-CIS has no card identifier, no device id and no IP, so that rule
is not computable. We substitute the strongest single C-column -- Vesta's
own pre-computed counting features ("how many addresses are associated with
this payment card", exact definitions masked). That is still a one-feature
count rule, so it plays the same role, but the counting was done by the data
provider rather than by us.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.costs import CostModel, cost_optimal_threshold, decisions_global
from src.evaluate import pr_auc, precision_recall_at, recall_at_fpr

SEED = 42
AMOUNT = "TransactionAmt"

# Five cheap, always-populated features for the linear rung.
LINEAR_FEATURES = [AMOUNT, "C1", "C13", "C14", "D15"]


@dataclass
class Rung:
    name: str
    uses: str
    test_scores: np.ndarray | None  # None where the rung does not rank
    test_blocked: np.ndarray


def _tune_and_apply(
    scores_val: np.ndarray,
    y_val: np.ndarray,
    amounts_val: np.ndarray,
    scores_test: np.ndarray,
    model: CostModel,
) -> tuple[np.ndarray, float]:
    """Pick the cost-minimising cut on val-B, freeze it, apply to test.

    Scores need not be probabilities here -- for the rule rungs they are raw
    feature values -- so we sweep over the observed score range rather than
    over [0, 1].
    """
    grid = np.unique(np.quantile(scores_val, np.linspace(0, 1, 501)))
    threshold = cost_optimal_threshold(y_val, scores_val, amounts_val, model, grid=grid)
    return decisions_global(scores_test, threshold), float(threshold)


def build_ladder(
    df: pd.DataFrame,
    splits,
    model: CostModel,
    amounts: np.ndarray,
) -> tuple[list[Rung], dict[str, float]]:
    """Construct every rung below the model."""
    y = df["isFraud"].to_numpy().astype(bool)
    y_val, y_test = y[splits.val_b], y[splits.test]
    amounts_val = amounts[splits.val_b]
    n_test = len(y_test)
    rng = np.random.default_rng(SEED)

    rungs: list[Rung] = []
    chosen: dict[str, float] = {}

    # 1. Predict "never fraud". Uses nothing. This is the accuracy trap.
    rungs.append(
        Rung("Predict never fraud", "nothing", None, np.zeros(n_test, dtype=bool))
    )

    # 2. Block everything. The opposite extreme, and a real cost-model bound.
    rungs.append(
        Rung("Block everything", "nothing", None, np.ones(n_test, dtype=bool))
    )

    # 3. Random at the base rate.
    base_rate = float(y[splits.train].mean())
    random_scores = rng.uniform(size=n_test)
    rungs.append(
        Rung(
            "Random at base rate",
            "nothing",
            random_scores,
            random_scores < base_rate,
        )
    )

    # 4. Single rule on transaction amount.
    amount_val = df[AMOUNT].iloc[splits.val_b].to_numpy(dtype=np.float64)
    amount_test = df[AMOUNT].iloc[splits.test].to_numpy(dtype=np.float64)
    blocked, threshold = _tune_and_apply(
        amount_val, y_val, amounts_val, amount_test, model
    )
    chosen["amount_rule_threshold_usd"] = threshold
    rungs.append(Rung("Amount > threshold", "one feature", amount_test, blocked))

    # 5. Best single C-column count rule, selected on val-B.
    c_columns = [c for c in df.columns if c.startswith("C") and c[1:].isdigit()]
    best_col, best_ap = None, -1.0
    for col in c_columns:
        values = df[col].iloc[splits.val_b].fillna(0).to_numpy(dtype=np.float64)
        ap = pr_auc(y_val, values)
        if ap > best_ap:
            best_col, best_ap = col, ap
    c_val = df[best_col].iloc[splits.val_b].fillna(0).to_numpy(dtype=np.float64)
    c_test = df[best_col].iloc[splits.test].fillna(0).to_numpy(dtype=np.float64)
    blocked, threshold = _tune_and_apply(c_val, y_val, amounts_val, c_test, model)
    chosen["count_rule_column"] = best_col
    chosen["count_rule_threshold"] = threshold
    rungs.append(
        Rung(f"Count rule ({best_col} > t)", "hand-written domain rule", c_test, blocked)
    )

    # 6. Logistic regression on five features, fitted on train only.
    linear = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=SEED),
    )
    x = df[LINEAR_FEATURES]
    linear.fit(x.iloc[splits.train], y[splits.train])
    lr_val = linear.predict_proba(x.iloc[splits.val_b])[:, 1]
    lr_test = linear.predict_proba(x.iloc[splits.test])[:, 1]
    blocked, threshold = _tune_and_apply(lr_val, y_val, amounts_val, lr_test, model)
    chosen["logistic_threshold"] = threshold
    rungs.append(
        Rung("Logistic regression (5 features)", "cheap linear model", lr_test, blocked)
    )

    return rungs, chosen


def score_ladder(
    rungs: list[Rung],
    y_test: np.ndarray,
    amounts_test: np.ndarray,
    model: CostModel,
) -> pd.DataFrame:
    """One table: ranking quality and money, side by side."""
    rows = []
    for rung in rungs:
        precision, recall = precision_recall_at(y_test, rung.test_blocked)
        rows.append(
            {
                "baseline": rung.name,
                "uses": rung.uses,
                "PR-AUC": pr_auc(y_test, rung.test_scores)
                if rung.test_scores is not None
                else np.nan,
                "recall@0.5%FPR": recall_at_fpr(y_test, rung.test_scores)
                if rung.test_scores is not None
                else np.nan,
                "precision": precision,
                "recall": recall,
                "block_rate": float(np.mean(rung.test_blocked)),
                "cost_per_10k_inr": model.cost_per_10k(
                    y_test, rung.test_blocked, amounts_test
                ),
            }
        )
    return pd.DataFrame(rows)
