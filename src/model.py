"""GBDT training, calibration, and the cost-optimal policy.

Design decisions worth defending:

WHY GRADIENT-BOOSTED TREES, NOT DEEP LEARNING. On tabular fraud data GBDTs
beat neural nets, train in minutes, and give SHAP explanations for free.
Reaching for a transformer here would be a red flag, not a strength.

WHY NO CLASS WEIGHTING. plan.md originally said to use `scale_pos_weight`
*instead of* SMOTE, on the grounds that resampling destroys probability
calibration. The first half is right and the second half is a mistake:
reweighting shifts the predicted base rate too, so it breaks calibration
just as surely as resampling does. The real rule is that ANY prior-shifting
technique needs recalibration afterwards. Since the whole rupee argument
depends on probabilities that mean what they say, we train unweighted with
enough trees and let isotonic regression do the work.

FOLD DISCIPLINE. Four disjoint, time-ordered jobs, one each:
    train-fit   -> tree fitting
    train-es    -> early stopping (carved from the tail of train, so that
                   val-A is not used for both model selection AND calibration)
    val-A       -> isotonic calibration
    val-B       -> decision threshold
    test        -> reported once, never tuned on
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

SEED = 42

PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "num_leaves": 128,
    "min_child_samples": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "verbosity": -1,
    "seed": SEED,
    "num_threads": 0,
    # Deliberately absent: scale_pos_weight / is_unbalance. See module docstring.
}


@dataclass
class Fitted:
    booster: lgb.Booster
    calibrator: IsotonicRegression
    best_iteration: int
    early_stop_fraction: float

    def raw_score(self, x: pd.DataFrame) -> np.ndarray:
        """Uncalibrated model output."""
        return self.booster.predict(x, num_iteration=self.best_iteration)

    def probability(self, x: pd.DataFrame) -> np.ndarray:
        """Calibrated probability -- the only thing safe to feed the cost model."""
        return self.calibrator.predict(self.raw_score(x))


def train(
    x: pd.DataFrame,
    y: np.ndarray,
    splits,
    early_stop_fraction: float = 0.10,
    num_boost_round: int = 3000,
    stopping_rounds: int = 100,
) -> Fitted:
    """Fit the booster, then calibrate it on val-A."""
    train_slice = splits.train
    n_train = train_slice.stop - train_slice.start
    n_es = int(n_train * early_stop_fraction)
    fit_end = train_slice.stop - n_es

    x_fit, y_fit = x.iloc[train_slice.start : fit_end], y[train_slice.start : fit_end]
    x_es, y_es = x.iloc[fit_end : train_slice.stop], y[fit_end : train_slice.stop]

    booster = lgb.train(
        PARAMS,
        lgb.Dataset(x_fit, label=y_fit),
        num_boost_round=num_boost_round,
        valid_sets=[lgb.Dataset(x_es, label=y_es)],
        valid_names=["early_stop"],
        callbacks=[
            lgb.early_stopping(stopping_rounds, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )

    # Isotonic on val-A only. Clipped to (0,1) so downstream log/plot code is safe.
    raw_val_a = booster.predict(x.iloc[splits.val_a], num_iteration=booster.best_iteration)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
    calibrator.fit(raw_val_a, y[splits.val_a])

    return Fitted(
        booster=booster,
        calibrator=calibrator,
        best_iteration=booster.best_iteration,
        early_stop_fraction=early_stop_fraction,
    )


def top_reason_codes(
    booster: lgb.Booster,
    x_row: pd.DataFrame,
    feature_names: list[str],
    n: int = 3,
) -> list[tuple[str, float]]:
    """Top-n SHAP contributors for one transaction, most positive first.

    LightGBM computes exact tree SHAP via pred_contrib, so no separate
    explainer object is needed on the serving path.
    """
    contributions = booster.predict(x_row, pred_contrib=True, num_iteration=booster.best_iteration)
    values = np.asarray(contributions)[0][:-1]  # last entry is the base value
    order = np.argsort(values)[::-1][:n]
    return [(feature_names[i], float(values[i])) for i in order]
