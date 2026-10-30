"""Checks for the all-coordinate ridge Cox survival probe."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ridge_cox_lbfgs import RidgeCoxLBFGS


def test_ridge_cox_high_dimension_with_tied_events():
    rng = np.random.default_rng(9)
    features = rng.normal(size=(80, 120))
    baseline_risk = features[:, 0] - 0.7 * features[:, 1]
    durations = np.maximum(1, np.round(rng.exponential(10, 80) / np.exp(baseline_risk / 2)))
    events = rng.binomial(1, 0.8, 80)
    model = RidgeCoxLBFGS(0.1).fit(features, durations, events)
    risks = model.predict_log_partial_hazard(features).to_numpy()
    survival = model.predict_survival_function(features, [1, 5, 10]).to_numpy()
    assert model.coef_.shape == (120,)
    assert np.isfinite(risks).all()
    assert survival.shape == (3, 80)
    assert np.all((survival >= 0) & (survival <= 1))
    assert np.all(np.diff(survival, axis=0) <= 0)
    assert np.linalg.norm(model.coef_) > 0
