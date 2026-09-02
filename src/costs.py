"""Rupee cost model and cost-optimal thresholding.

This is the heart of the project. A fraud model's job is not to be accurate,
it is to lose the merchant the least money -- and those are different
objectives with different optimal thresholds.

Two results matter here:

1.  With costs that do not depend on the transaction, the cost-optimal
    threshold has a closed form: t* = c_FP / (c_FP + c_FN).

2.  Our costs DO depend on the transaction. The false-negative cost scales
    with the order value (you eat the chargeback) while the false-positive
    cost scales with the margin. So no single threshold is optimal, and the
    optimal rule is per-transaction:

        block  <=>  p * c_FN(a)  >  (1 - p) * c_FP(a)
               <=>  p            >  c_FP(a) / (c_FP(a) + c_FN(a))
               <=>  p            >  t*(a)

    t*(a) falls as the amount rises -- block high-value orders on weaker
    evidence. That is the project's headline contribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "costs.yaml"


@dataclass(frozen=True)
class CostModel:
    """Cost assumptions, loaded from costs.yaml.

    Amounts passed to every method are in INR. Use `usd_to_inr` to convert
    raw IEEE-CIS TransactionAmt before calling anything here.
    """

    version: str
    usd_to_inr: float
    chargeback_amount_multiplier: float
    chargeback_fee_inr: float
    ops_handling_inr: float
    margin_rate: float
    churn_probability: float
    customer_ltv_inr: float
    support_contact_inr: float
    review_cost_inr: float
    max_review_rate: float

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CONFIG) -> "CostModel":
        cfg = yaml.safe_load(Path(path).read_text())
        fn, fp, rv = cfg["false_negative"], cfg["false_positive"], cfg["review"]
        return cls(
            version=cfg["version"],
            usd_to_inr=float(cfg["usd_to_inr"]),
            chargeback_amount_multiplier=float(fn["chargeback_amount_multiplier"]),
            chargeback_fee_inr=float(fn["chargeback_fee_inr"]),
            ops_handling_inr=float(fn["ops_handling_inr"]),
            margin_rate=float(fp["margin_rate"]),
            churn_probability=float(fp["churn_probability"]),
            customer_ltv_inr=float(fp["customer_ltv_inr"]),
            support_contact_inr=float(fp["support_contact_inr"]),
            review_cost_inr=float(rv["cost_per_review_inr"]),
            max_review_rate=float(rv["max_review_rate"]),
        )

    # -- per-transaction error costs -------------------------------------

    def fn_cost(self, amount_inr: np.ndarray | float) -> np.ndarray:
        """Cost of approving a fraudulent transaction."""
        amount = np.asarray(amount_inr, dtype=np.float64)
        return (
            self.chargeback_amount_multiplier * amount
            + self.chargeback_fee_inr
            + self.ops_handling_inr
        )

    def fp_cost(self, amount_inr: np.ndarray | float) -> np.ndarray:
        """Cost of blocking a legitimate transaction.

        Lost margin, not lost revenue -- plus expected churn and support.
        """
        amount = np.asarray(amount_inr, dtype=np.float64)
        return (
            self.margin_rate * amount
            + self.churn_probability * self.customer_ltv_inr
            + self.support_contact_inr
        )

    # -- thresholds -------------------------------------------------------

    def optimal_threshold(self, amount_inr: np.ndarray | float) -> np.ndarray:
        """The amount-dependent cost-optimal threshold t*(a).

        Block when the calibrated probability exceeds this. Requires the
        probability to be genuinely calibrated -- an uncalibrated 0.9 makes
        this arithmetic meaningless.

        See `threshold_limits` for which direction the curve runs and why
        that depends on the cost constants rather than being universal.
        """
        fp, fn = self.fp_cost(amount_inr), self.fn_cost(amount_inr)
        return fp / (fp + fn)

    def threshold_limits(self) -> tuple[float, float]:
        """(t* as amount -> 0, t* as amount -> infinity).

        The low-amount limit is driven by the flat costs; the high-amount
        limit tends to margin_rate / (margin_rate + chargeback_multiplier).

        Which way the curve runs is NOT universal -- it depends on the cost
        structure. Differentiating t*(a) shows it is decreasing exactly when

            fixed_fp  >  margin_rate * fixed_fn

        where fixed_fp = churn*LTV + support and fixed_fn = fee + ops. For
        the shipped config that is 250 > 0.12 * 1200 = 144, so t* falls with
        amount. Raise the flat chargeback fee enough and the inequality
        flips, and it is then optimal to be *more* permissive on large
        orders. Worth stating plainly rather than presenting "block big
        orders on weaker evidence" as a law of fraud.
        """
        fixed_fp = self.churn_probability * self.customer_ltv_inr + self.support_contact_inr
        fixed_fn = self.chargeback_fee_inr + self.ops_handling_inr
        at_zero = fixed_fp / (fixed_fp + fixed_fn)
        at_inf = self.margin_rate / (self.margin_rate + self.chargeback_amount_multiplier)
        return float(at_zero), float(at_inf)

    # -- realised cost ----------------------------------------------------

    def total_cost(
        self,
        y_true: np.ndarray,
        blocked: np.ndarray,
        amount_inr: np.ndarray,
        reviewed: np.ndarray | None = None,
    ) -> float:
        """Total rupee cost of a set of decisions.

        A blocked fraud costs nothing (that is the win). An approved legit
        transaction costs nothing. We only pay for the two mistakes -- plus
        analyst time for anything routed to manual review.
        """
        y_true = np.asarray(y_true).astype(bool)
        blocked = np.asarray(blocked).astype(bool)
        amount = np.asarray(amount_inr, dtype=np.float64)

        false_negatives = y_true & ~blocked
        false_positives = ~y_true & blocked

        cost = float(
            self.fn_cost(amount[false_negatives]).sum()
            + self.fp_cost(amount[false_positives]).sum()
        )
        if reviewed is not None:
            cost += float(np.asarray(reviewed).astype(bool).sum() * self.review_cost_inr)
        return cost

    def cost_per_10k(
        self,
        y_true: np.ndarray,
        blocked: np.ndarray,
        amount_inr: np.ndarray,
        reviewed: np.ndarray | None = None,
    ) -> float:
        """Total cost normalised to 10,000 transactions -- the headline unit."""
        n = len(np.asarray(y_true))
        if n == 0:
            return 0.0
        return self.total_cost(y_true, blocked, amount_inr, reviewed) / n * 10_000


# -- policies ------------------------------------------------------------


def decisions_global(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Block using a single global threshold. What everyone else builds."""
    return np.asarray(probabilities) >= threshold


def decisions_amount_dependent(
    probabilities: np.ndarray, amount_inr: np.ndarray, model: CostModel
) -> np.ndarray:
    """Block using the per-transaction threshold t*(a). Our contribution."""
    return np.asarray(probabilities) >= model.optimal_threshold(amount_inr)


# -- threshold sweep and uncertainty --------------------------------------


def sweep_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    amount_inr: np.ndarray,
    model: CostModel,
    grid: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Cost curve: total cost per 10k transactions at every threshold.

    Returns the raw arrays so the caller can plot the curve and mark the
    accuracy-optimal, F1-optimal and cost-optimal points on it.
    """
    if grid is None:
        grid = np.linspace(0.0, 1.0, 1001)

    y_true = np.asarray(y_true).astype(bool)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    amount = np.asarray(amount_inr, dtype=np.float64)

    costs = np.empty(len(grid), dtype=np.float64)
    for i, t in enumerate(grid):
        costs[i] = model.cost_per_10k(y_true, probabilities >= t, amount)

    return {"threshold": grid, "cost_per_10k": costs}


def cost_optimal_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    amount_inr: np.ndarray,
    model: CostModel,
    grid: np.ndarray | None = None,
) -> float:
    """The threshold minimising realised cost on the given data.

    IMPORTANT: pick this on validation, then report realised cost on test at
    the frozen value. Picking it on test and reporting the same test cost is
    an oracle number -- exactly the inflated claim this project exists to
    argue against.
    """
    swept = sweep_thresholds(y_true, probabilities, amount_inr, model, grid)
    return float(swept["threshold"][int(np.argmin(swept["cost_per_10k"]))])


def bootstrap_cost_difference(
    y_true: np.ndarray,
    amount_inr: np.ndarray,
    blocked_a: np.ndarray,
    blocked_b: np.ndarray,
    model: CostModel,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """95% CI on (cost of policy A) - (cost of policy B), per 10k transactions.

    A headline rupee figure with no error bar is the most attackable number
    in a submission like this. If the interval crosses zero, that IS the
    finding and it gets reported as such.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(bool)
    amount = np.asarray(amount_inr, dtype=np.float64)
    blocked_a = np.asarray(blocked_a).astype(bool)
    blocked_b = np.asarray(blocked_b).astype(bool)

    n = len(y_true)
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = model.cost_per_10k(
            y_true[idx], blocked_a[idx], amount[idx]
        ) - model.cost_per_10k(y_true[idx], blocked_b[idx], amount[idx])

    point = model.cost_per_10k(y_true, blocked_a, amount) - model.cost_per_10k(
        y_true, blocked_b, amount
    )
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "point_estimate": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "crosses_zero": bool(lo <= 0.0 <= hi),
    }
