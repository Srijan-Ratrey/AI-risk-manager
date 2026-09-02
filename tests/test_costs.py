"""Tests for the money math.

The cost model is the one part of this project where a sign error would be
invisible in the output and would invalidate every headline number. These
tests are cheap and they pin down the arithmetic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.costs import (  # noqa: E402
    CostModel,
    bootstrap_cost_difference,
    cost_optimal_threshold,
    decisions_amount_dependent,
    decisions_global,
)


@pytest.fixture
def model() -> CostModel:
    return CostModel.load()


def test_config_loads_with_expected_shape(model: CostModel) -> None:
    assert model.version
    assert 0.0 < model.margin_rate < 1.0
    assert 0.0 <= model.churn_probability <= 1.0
    assert model.usd_to_inr > 0


def test_worked_examples_from_the_plan(model: CostModel) -> None:
    """The three rows of the headline table must reproduce exactly."""
    for amount, want_fn, want_fp, want_t in [
        (500.0, 1_700.0, 310.0, 0.1542),
        (5_000.0, 6_200.0, 850.0, 0.1206),
        (50_000.0, 51_200.0, 6_250.0, 0.1088),
    ]:
        assert model.fn_cost(amount) == pytest.approx(want_fn)
        assert model.fp_cost(amount) == pytest.approx(want_fp)
        assert model.optimal_threshold(amount) == pytest.approx(want_t, abs=1e-4)


def test_threshold_falls_as_amount_rises(model: CostModel) -> None:
    """The core claim: block high-value orders on weaker evidence."""
    amounts = np.array([100.0, 1_000.0, 10_000.0, 100_000.0])
    thresholds = model.optimal_threshold(amounts)
    assert np.all(np.diff(thresholds) < 0), "t*(a) must be strictly decreasing"


def test_threshold_limits_bracket_the_curve(model: CostModel) -> None:
    at_zero, at_inf = model.threshold_limits()
    assert at_inf < at_zero, "the curve descends from the small-amount limit"
    # margin / (margin + 1) for a full-value chargeback
    assert at_inf == pytest.approx(0.12 / 1.12, abs=1e-6)
    assert model.optimal_threshold(1e-6) == pytest.approx(at_zero, abs=1e-4)
    assert model.optimal_threshold(1e12) == pytest.approx(at_inf, abs=1e-4)


def test_swept_optimum_matches_closed_form_at_constant_costs() -> None:
    """With amount-independent costs, t* = c_FP / (c_FP + c_FN).

    This is the verification that the sweep is minimising what we think it
    is minimising. We neutralise the amount-dependent terms so the closed
    form applies, then check the numerically swept optimum lands on it.
    """
    flat = CostModel(
        version="test-flat",
        usd_to_inr=1.0,
        chargeback_amount_multiplier=0.0,  # FN cost is now a flat 2000
        chargeback_fee_inr=2_000.0,
        ops_handling_inr=0.0,
        margin_rate=0.0,  # FP cost is now a flat 500
        churn_probability=0.0,
        customer_ltv_inr=0.0,
        support_contact_inr=500.0,
        review_cost_inr=0.0,
        max_review_rate=1.0,
    )
    expected = 500.0 / (500.0 + 2_000.0)  # = 0.2
    assert flat.optimal_threshold(1.0) == pytest.approx(expected)

    # Build data whose calibrated score IS the true fraud probability, so the
    # empirical optimum should coincide with the theoretical one.
    rng = np.random.default_rng(0)
    n = 200_000
    probabilities = rng.uniform(0.0, 1.0, size=n)
    y_true = rng.uniform(0.0, 1.0, size=n) < probabilities
    amounts = np.full(n, 1.0)

    swept = cost_optimal_threshold(y_true, probabilities, amounts, flat)
    assert swept == pytest.approx(expected, abs=0.02)


def test_perfect_blocking_of_fraud_costs_nothing(model: CostModel) -> None:
    y_true = np.array([True, False, True, False])
    amounts = np.array([1_000.0, 2_000.0, 3_000.0, 4_000.0])
    assert model.total_cost(y_true, blocked=y_true, amount_inr=amounts) == 0.0


def test_approving_everything_costs_the_fraud(model: CostModel) -> None:
    y_true = np.array([True, False, True])
    amounts = np.array([1_000.0, 2_000.0, 3_000.0])
    blocked = np.zeros(3, dtype=bool)
    # Only the two frauds cost anything.
    expected = float(model.fn_cost(np.array([1_000.0, 3_000.0])).sum())
    assert model.total_cost(y_true, blocked, amounts) == pytest.approx(expected)


def test_blocking_everything_costs_the_legit_margin(model: CostModel) -> None:
    y_true = np.array([True, False, False])
    amounts = np.array([1_000.0, 2_000.0, 3_000.0])
    blocked = np.ones(3, dtype=bool)
    expected = float(model.fp_cost(np.array([2_000.0, 3_000.0])).sum())
    assert model.total_cost(y_true, blocked, amounts) == pytest.approx(expected)


def test_review_cost_is_charged_per_reviewed_row(model: CostModel) -> None:
    y_true = np.zeros(10, dtype=bool)
    amounts = np.full(10, 1_000.0)
    blocked = np.zeros(10, dtype=bool)
    reviewed = np.array([True] * 4 + [False] * 6)
    assert model.total_cost(y_true, blocked, amounts, reviewed) == pytest.approx(
        4 * model.review_cost_inr
    )


def test_cost_per_10k_normalises(model: CostModel) -> None:
    y_true = np.array([True] * 100)
    amounts = np.full(100, 1_000.0)
    blocked = np.zeros(100, dtype=bool)
    total = model.total_cost(y_true, blocked, amounts)
    assert model.cost_per_10k(y_true, blocked, amounts) == pytest.approx(total * 100)


def test_adaptive_rule_is_pointwise_bayes_optimal(model: CostModel) -> None:
    """No sampling involved: check the decision rule against expected cost directly.

    For each transaction the rule must pick whichever action has the lower
    expected cost -- block costs (1-p)*c_FP, approve costs p*c_FN.
    """
    rng = np.random.default_rng(11)
    probabilities = rng.uniform(0.0, 1.0, size=10_000)
    amounts = 10 ** rng.uniform(1.0, 6.0, size=10_000)

    blocked = decisions_amount_dependent(probabilities, amounts, model)
    cost_if_block = (1 - probabilities) * model.fp_cost(amounts)
    cost_if_approve = probabilities * model.fn_cost(amounts)

    assert np.all(blocked == (cost_if_approve >= cost_if_block))


def _simulate(seed: int, n: int, amount_lo: float, amount_hi: float):
    rng = np.random.default_rng(seed)
    probabilities = rng.beta(0.5, 8.0, size=n)  # realistic skew toward 0
    y_true = rng.uniform(0.0, 1.0, size=n) < probabilities
    amounts = 10 ** rng.uniform(amount_lo, amount_hi, size=n)
    return y_true, probabilities, amounts


def test_adaptive_beats_global_out_of_sample(model: CostModel) -> None:
    """The headline claim, compared honestly.

    The global threshold is tuned on a validation half and both policies are
    then scored on a held-out half. Tuning the global threshold on the SAME
    data it is scored on makes it an oracle fit to that sample's noise, and
    it can then appear to win -- which is exactly the inflated-baseline trap
    this project argues against.

    With the shipped cost constants t*(a) only spans ~0.11-0.17, so the
    honest expectation here is a small win, not a dramatic one.
    """
    y_true, probabilities, amounts = _simulate(seed=7, n=400_000, amount_lo=2.0, amount_hi=5.0)
    half = len(y_true) // 2
    val, test = slice(None, half), slice(half, None)

    tuned_t = cost_optimal_threshold(
        y_true[val], probabilities[val], amounts[val], model
    )
    global_cost = model.cost_per_10k(
        y_true[test], decisions_global(probabilities[test], tuned_t), amounts[test]
    )
    adaptive_cost = model.cost_per_10k(
        y_true[test],
        decisions_amount_dependent(probabilities[test], amounts[test], model),
        amounts[test],
    )
    assert adaptive_cost < global_cost


def test_adaptive_margin_grows_when_costs_are_more_amount_sensitive() -> None:
    """The mechanism has teeth when the cost structure actually varies with value.

    A large flat chargeback fee makes t*(a) span a much wider band, and the
    advantage of a per-transaction threshold grows accordingly.
    """
    steep = CostModel(
        version="test-steep",
        usd_to_inr=1.0,
        chargeback_amount_multiplier=1.0,
        chargeback_fee_inr=50_000.0,  # dominates at low amounts
        ops_handling_inr=0.0,
        margin_rate=0.30,
        churn_probability=0.0,
        customer_ltv_inr=0.0,
        support_contact_inr=10.0,
        review_cost_inr=0.0,
        max_review_rate=1.0,
    )
    lo, hi = steep.threshold_limits()
    # Direction is not the point here, span is. This fixture happens to make
    # t*(a) RISE with amount (see CostModel.threshold_limits on why).
    assert abs(lo - hi) > 0.2, "this fixture should span a wide threshold band"

    y_true, probabilities, amounts = _simulate(
        seed=13, n=400_000, amount_lo=1.0, amount_hi=6.0
    )
    half = len(y_true) // 2
    val, test = slice(None, half), slice(half, None)

    tuned_t = cost_optimal_threshold(
        y_true[val], probabilities[val], amounts[val], steep
    )
    global_cost = steep.cost_per_10k(
        y_true[test], decisions_global(probabilities[test], tuned_t), amounts[test]
    )
    adaptive_cost = steep.cost_per_10k(
        y_true[test],
        decisions_amount_dependent(probabilities[test], amounts[test], steep),
        amounts[test],
    )
    assert adaptive_cost < global_cost * 0.98, "expect a clear, not marginal, win"


def test_bootstrap_reports_an_interval_around_the_point_estimate(
    model: CostModel,
) -> None:
    rng = np.random.default_rng(3)
    n = 5_000
    probabilities = rng.beta(0.5, 8.0, size=n)
    y_true = rng.uniform(0.0, 1.0, size=n) < probabilities
    amounts = 10 ** rng.uniform(2.0, 5.0, size=n)

    result = bootstrap_cost_difference(
        y_true,
        amounts,
        decisions_global(probabilities, 0.5),
        decisions_amount_dependent(probabilities, amounts, model),
        model,
        n_boot=200,
    )
    assert result["ci_low"] <= result["point_estimate"] <= result["ci_high"]
    assert isinstance(result["crosses_zero"], bool)
