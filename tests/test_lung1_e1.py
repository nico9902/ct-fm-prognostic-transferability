"""Small checks for the locked frozen-Cox experiment."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_lung1_frozen_cox_e1 as e1


def test_ipcw_metrics_without_censoring():
    train_time = np.array([1., 2., 3., 4.])
    train_event = np.ones(4, dtype=int)
    test_time = np.array([1., 2., 3.])
    test_event = np.ones(3, dtype=int)
    assert e1.uno_c(train_time, train_event, test_time, test_event,
                    np.array([3., 2., 1.]), tau=4) == 1.0
    assert e1.uno_c(train_time, train_event, test_time, test_event,
                    np.array([1., 2., 3.]), tau=4) == 0.0
    assert e1.cumulative_auc(train_time, train_event, test_time, test_event,
                             np.array([3., 2., 1.]), horizon=2.5) == 1.0
    assert e1.brier(train_time, train_event, test_time, test_event,
                    np.array([0., 0., 1.]), horizon=2.5) == 0.0
    # With an event/censor tie, the event is comparable to the censored case.
    tied_time = np.array([1., 1., 2.])
    tied_event = np.array([1, 0, 1])
    assert e1.uno_c(train_time, train_event, tied_time, tied_event,
                    np.array([3., 1., 2.]), tau=3) == 1.0
    reverse_km = e1.censor_survival(tied_time, tied_event)
    assert reverse_km(1.) == 0.5


def test_locked_folds_and_train_only_cox(tmp_path):
    rng = np.random.default_rng(17)
    n = 60
    data = pd.DataFrame({
        "patient_id": [f"P{i:03d}" for i in range(n)],
        "ct_series_uid": [f"UID{i}" for i in range(n)],
        "overall_stage": ["I" if i % 2 else "IIIb" for i in range(n)],
        "os_event": [0 if i % 5 == 0 else 1 for i in range(n)],
        "os_time_days": [1400. if i % 5 == 0 else float(100 + 11 * i) for i in range(n)],
    })
    folds_path = tmp_path / "folds.csv"
    folds = e1.locked_folds(data, folds_path)
    assert len(folds) == n * e1.REPETITIONS
    assert not folds.duplicated(["repetition", "patient_id"]).any()
    assert e1.locked_folds(data, folds_path).equals(folds)
    feature = rng.normal(size=(n, 24))
    fitted = e1.fit_cox(feature[:45], data.os_time_days.to_numpy()[:45],
                        data.os_event.to_numpy()[:45], 0.1, 8)
    risk, survival = e1.predict(fitted, feature[45:], e1.GRID)
    assert risk.shape == (15,) and survival.shape == (15, len(e1.GRID))
    assert np.isfinite(risk).all() and np.isfinite(survival).all()
    assert np.all((survival >= 0) & (survival <= 1))


def test_nested_run_outputs_one_prediction_per_patient(tmp_path):
    rng = np.random.default_rng(29)
    n = 60
    data = pd.DataFrame({
        "patient_id": [f"P{i:03d}" for i in range(n)],
        "ct_series_uid": [f"UID{i}" for i in range(n)],
        "overall_stage": ["I" if i % 2 else "IIIb" for i in range(n)],
        "os_event": [0 if i % 5 == 0 else 1 for i in range(n)],
        "os_time_days": [1400. if i % 5 == 0 else float(100 + 11 * i) for i in range(n)],
    })
    original = e1.REPETITIONS, e1.OUTER_FOLDS, e1.INNER_FOLDS, e1.ALPHAS
    try:
        e1.REPETITIONS, e1.OUTER_FOLDS, e1.INNER_FOLDS, e1.ALPHAS = 1, 2, 2, (0.1,)
        folds = e1.expected_folds(data)
        inner = e1.inner_assignments(data, folds)
        features = {"synthetic": rng.normal(size=(n, 12))}
        result = e1.run(data, folds, inner, features, tmp_path, components=6, bootstrap=3)
        predictions = pd.read_csv(tmp_path / "out_of_fold_predictions.csv")
        assert len(predictions) == n
        assert predictions.patient_id.nunique() == n
        assert 0 <= result["synthetic"]["uno_c_36m"] <= 1
    finally:
        e1.REPETITIONS, e1.OUTER_FOLDS, e1.INNER_FOLDS, e1.ALPHAS = original
