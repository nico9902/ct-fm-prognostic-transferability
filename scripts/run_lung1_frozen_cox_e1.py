#!/usr/bin/env python3
"""E1: repeated nested-CV linear Cox probes on frozen LUNG1 embeddings.

Input for each model is a CSV with patient_id, ct_series_uid, embedding_path,
and one finite 1-D .npy vector per patient. Run --prepare-only before the
images/embeddings finish to lock the outer and inner fold assignments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.exceptions import ConvergenceError
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "data/derived/lung1/cohort_with_gtv_volume.csv"
FOLDS = ROOT / "configs/lung1_e1_folds.csv"
OUTPUT = ROOT / "outputs/lung1_e1_frozen_cox"
HORIZONS = (365.25, 730.5, 1095.75)
GRID = np.linspace(91.3125, 1095.75, 12)
ALPHAS = (0.01, 0.1, 1.0)
REPETITIONS = 5
OUTER_FOLDS = 5
INNER_FOLDS = 3
SEED = 20261001


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cohort(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, dtype={"patient_id": str, "ct_series_uid": str})
    data = data.loc[data.region_status.eq("ready")].sort_values("patient_id").reset_index(drop=True)
    if data.patient_id.duplicated().any() or data.empty:
        raise ValueError("Eligible cohort must have unique patient IDs")
    data["os_time_days"] = pd.to_numeric(data.os_time_days, errors="raise")
    data["os_event"] = pd.to_numeric(data.os_event, errors="raise").astype(int)
    if not np.isfinite(data.os_time_days).all() or (data.os_time_days <= 0).any():
        raise ValueError("OS times must be positive and finite")
    if not data.os_event.isin((0, 1)).all() or data.os_event.nunique() != 2:
        raise ValueError("OS event must be binary with both outcomes present")
    return data


def strata(data: pd.DataFrame) -> np.ndarray:
    # Combine sparse stage classes so all event/stage strata support five folds.
    early = data.overall_stage.astype(str).str.upper().isin(("I", "II"))
    return (data.os_event.astype(str) + "_" + np.where(early, "early", "late")).to_numpy()


def expected_folds(data: pd.DataFrame) -> pd.DataFrame:
    labels = strata(data)
    if min(np.unique(labels, return_counts=True)[1]) < OUTER_FOLDS:
        raise ValueError("Event/stage strata are too small for five outer folds")
    records = []
    for repetition in range(REPETITIONS):
        splitter = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=SEED + repetition)
        for fold, (_, test) in enumerate(splitter.split(data, labels)):
            for index in test:
                records.append((repetition, fold, data.patient_id.iloc[index]))
    return pd.DataFrame(records, columns=["repetition", "fold", "patient_id"]).sort_values(
        ["repetition", "patient_id"]).reset_index(drop=True)


def locked_folds(data: pd.DataFrame, path: Path) -> pd.DataFrame:
    expected = expected_folds(data)
    if path.exists():
        saved = pd.read_csv(path, dtype={"patient_id": str}).sort_values(
            ["repetition", "patient_id"]).reset_index(drop=True)
        if not saved.equals(expected):
            raise ValueError(f"Locked folds differ from this cohort or protocol: {path}")
        return saved
    path.parent.mkdir(parents=True, exist_ok=True)
    expected.to_csv(path, index=False)
    return expected


def inner_assignments(data: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    records = []
    for repetition in range(REPETITIONS):
        for fold in range(OUTER_FOLDS):
            test_ids = set(folds.loc[(folds.repetition == repetition) & (folds.fold == fold), "patient_id"])
            train = data.loc[~data.patient_id.isin(test_ids)].reset_index(drop=True)
            labels = strata(train)
            splitter = StratifiedKFold(INNER_FOLDS, shuffle=True,
                                      random_state=SEED + 100 + repetition * OUTER_FOLDS + fold)
            for inner_fold, (_, validation) in enumerate(splitter.split(train, labels)):
                for index in validation:
                    records.append((repetition, fold, inner_fold, train.patient_id.iloc[index]))
    return pd.DataFrame(records, columns=["repetition", "fold", "inner_fold", "patient_id"])


def load_embeddings(spec: str, data: pd.DataFrame) -> tuple[str, np.ndarray, str, str]:
    if "=" not in spec:
        raise ValueError("--embeddings must be MODEL=MANIFEST.csv")
    name, filename = spec.split("=", 1)
    path = Path(filename).resolve()
    if not name or not path.is_file():
        raise ValueError(f"Missing embedding manifest: {spec}")
    manifest = pd.read_csv(path, dtype=str)
    required = {"patient_id", "ct_series_uid", "embedding_path"}
    if not required.issubset(manifest):
        raise ValueError(f"{path} is missing {sorted(required - set(manifest))}")
    if manifest.patient_id.duplicated().any() or set(manifest.patient_id) != set(data.patient_id):
        raise ValueError(f"{name}: embedding IDs must match the full locked cohort")
    manifest = manifest.set_index("patient_id").loc[data.patient_id]
    if not np.array_equal(manifest.ct_series_uid.to_numpy(), data.ct_series_uid.to_numpy()):
        raise ValueError(f"{name}: CT series UIDs differ from the cohort")
    arrays = []
    dimension = None
    for pid, entry in manifest.iterrows():
        feature_path = Path(entry.embedding_path)
        if not feature_path.is_absolute():
            feature_path = path.parent / feature_path
        if not feature_path.is_file():
            raise FileNotFoundError(f"{name}/{pid}: {feature_path}")
        vector = np.load(feature_path, allow_pickle=False)
        if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError(f"{name}/{pid}: expected a finite 1-D embedding")
        if dimension is None:
            dimension = vector.size
        if vector.size != dimension:
            raise ValueError(f"{name}/{pid}: embedding dimension changed")
        arrays.append(vector.astype(np.float64, copy=False))
    matrix = np.stack(arrays)
    return name, matrix, sha256(path), hashlib.sha256(matrix.tobytes()).hexdigest()


def censor_survival(times: np.ndarray, events: np.ndarray):
    """Training-only reverse Kaplan-Meier G(t), events preceding censor ties."""
    unique = np.unique(times)
    at_risk = np.array([(times >= t).sum() - ((times == t) & (events == 1)).sum()
                        for t in unique], dtype=float)
    censored = np.array([((times == t) & (events == 0)).sum() for t in unique], dtype=float)
    survival = np.cumprod(1.0 - np.divide(censored, at_risk,
                                          out=np.zeros_like(censored), where=at_risk > 0))

    def evaluate(query):
        q = np.asarray(query, dtype=float)
        index = np.searchsorted(unique, q, side="right") - 1
        return np.where(index >= 0, survival[np.maximum(index, 0)], 1.0)

    return evaluate


def uno_c(train_time, train_event, test_time, test_event, risk, tau=HORIZONS[-1]):
    g = censor_survival(train_time, train_event)
    cases = (test_event == 1) & (test_time <= tau)
    comparable = ((test_time[:, None] < test_time[None, :]) |
                  ((test_time[:, None] == test_time[None, :]) & (test_event[None, :] == 0)))
    earlier = comparable & cases[:, None]
    weights = 1.0 / np.maximum(g(test_time), 1e-12) ** 2
    if np.any(earlier & (g(test_time)[:, None] <= 0.05)):
        return np.nan
    denominator = np.sum(earlier * weights[:, None])
    if denominator == 0:
        return np.nan
    difference = risk[:, None] - risk[None, :]
    comparison = (difference > 1e-8).astype(float)
    comparison += 0.5 * (np.abs(difference) <= 1e-8)
    return float(np.sum(earlier * comparison * weights[:, None]) / denominator)


def cumulative_auc(train_time, train_event, test_time, test_event, risk, horizon):
    g = censor_survival(train_time, train_event)
    cases = (test_event == 1) & (test_time <= horizon)
    controls = test_time > horizon
    if not cases.any() or not controls.any() or np.min(g(test_time[cases])) <= 0.05:
        return np.nan
    weights = 1.0 / g(test_time[cases])
    difference = risk[cases, None] - risk[None, controls]
    comparison = (difference > 1e-8).astype(float)
    comparison += 0.5 * (np.abs(difference) <= 1e-8)
    return float(np.average(comparison.mean(axis=1), weights=weights))


def brier(train_time, train_event, test_time, test_event, survival, horizon):
    g = censor_survival(train_time, train_event)
    died = (test_event == 1) & (test_time <= horizon)
    alive = test_time > horizon
    if g(horizon) <= 0.05 or (died.any() and np.min(g(test_time[died])) <= 0.05):
        return np.nan
    term = np.zeros(len(test_time), dtype=float)
    term[died] = survival[died] ** 2 / g(test_time[died])
    term[alive] = (1 - survival[alive]) ** 2 / g(horizon)
    return float(term.mean())


def prepare_pcs(features, components):
    scaler = StandardScaler().fit(features)
    transformed = scaler.transform(features)
    if components == 0:
        pca = None
        columns = [f"feature_{i:04d}" for i in range(features.shape[1])]
        x = pd.DataFrame(transformed, columns=columns)
    else:
        pca = PCA(n_components=min(components, len(features) - 2, features.shape[1]),
                  svd_solver="randomized", random_state=SEED).fit(transformed)
        columns = [f"pc_{i:02d}" for i in range(pca.n_components_)]
        x = pd.DataFrame(pca.transform(transformed), columns=columns)
    return scaler, pca, columns, x


COX_DEVICE = "cpu"


def fit_pcs(x, times, events, penalizer):
    if x.columns[0].startswith("feature_"):
        from ridge_cox_lbfgs import RidgeCoxLBFGS
        return RidgeCoxLBFGS(penalizer, device=COX_DEVICE).fit(x.to_numpy(), times, events)
    x = x.copy()
    x["time"] = times
    x["event"] = events
    model = CoxPHFitter(penalizer=penalizer, l1_ratio=0.0)
    model.fit(x, duration_col="time", event_col="event", show_progress=False)
    return model


def fit_cox(features, times, events, penalizer, components):
    scaler, pca, columns, x = prepare_pcs(features, components)
    model = fit_pcs(x, times, events, penalizer)
    return scaler, pca, model, columns


def predict(fitted, features, horizons):
    scaler, pca, model, columns = fitted
    transformed = scaler.transform(features)
    x = pd.DataFrame(pca.transform(transformed) if pca is not None else transformed,
                     columns=columns)
    risk = model.predict_log_partial_hazard(x).to_numpy().reshape(-1)
    survival = model.predict_survival_function(x, times=np.asarray(horizons)).to_numpy().T
    if not np.isfinite(risk).all() or not np.isfinite(survival).all():
        raise ValueError("Cox model returned non-finite predictions")
    return risk, survival


def select_penalty(features, data, inner: pd.DataFrame, components):
    prepared = []
    for inner_fold in range(INNER_FOLDS):
        validation = data.patient_id.isin(inner.loc[inner.inner_fold == inner_fold, "patient_id"]).to_numpy()
        train = ~validation
        t_train = data.os_time_days.to_numpy()[train]
        e_train = data.os_event.to_numpy()[train]
        scaler, pca, columns, x_train = prepare_pcs(features[train], components)
        transformed = scaler.transform(features[validation])
        x_validation = pd.DataFrame(pca.transform(transformed) if pca is not None else transformed,
                                    columns=columns)
        prepared.append((x_train, x_validation, t_train, e_train,
                         data.os_time_days.to_numpy()[validation], data.os_event.to_numpy()[validation]))
    scores = []
    for alpha in ALPHAS:
        fold_scores = []
        for x_train, x_validation, t_train, e_train, t_validation, e_validation in prepared:
            model = fit_pcs(x_train, t_train, e_train, alpha)
            risk = model.predict_log_partial_hazard(x_validation).to_numpy().reshape(-1)
            score = uno_c(t_train, e_train, t_validation, e_validation, risk)
            if not np.isfinite(score):
                raise ValueError("Inner validation has insufficient IPCW support")
            fold_scores.append(score)
        scores.append(float(np.mean(fold_scores)))
    return ALPHAS[int(np.argmax(scores))], scores


def fold_metrics(train_time, train_event, test_time, test_event, risk, survival):
    result = {"uno_c_36m": uno_c(train_time, train_event, test_time, test_event, risk)}
    for horizon in HORIZONS:
        column = int(np.argmin(abs(GRID - horizon)))
        result[f"auc_{int(round(horizon / 365.25))}y"] = cumulative_auc(
            train_time, train_event, test_time, test_event, risk, horizon)
        result[f"brier_{int(round(horizon / 365.25))}y"] = brier(
            train_time, train_event, test_time, test_event, survival[:, column], horizon)
    scores = [brier(train_time, train_event, test_time, test_event, survival[:, i], horizon)
              for i, horizon in enumerate(GRID)]
    result["ibs_3m_36m"] = float(np.trapezoid(scores, GRID) / (GRID[-1] - GRID[0])) if np.isfinite(scores).all() else np.nan
    km = KaplanMeierFitter().fit(test_time, event_observed=test_event)
    column_2y = int(np.argmin(abs(GRID - HORIZONS[1])))
    result["calibration_in_large_2y"] = float(survival[:, column_2y].mean() - km.predict(HORIZONS[1]))
    if np.std(risk) > 1e-10:
        calibration_data = pd.DataFrame({"risk": risk, "time": test_time, "event": test_event})
        try:
            calibration = CoxPHFitter().fit(calibration_data, duration_col="time", event_col="event")
            result["calibration_slope"] = float(calibration.params_["risk"])
        except ConvergenceError:
            result["calibration_slope"] = np.nan
    else:
        result["calibration_slope"] = np.nan
    return result


def run(data, folds, inners, embeddings, output, components, bootstrap):
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "out_of_fold_predictions.csv"
    metric_path = output / "fold_metrics.csv"
    predictions = pd.read_csv(prediction_path).to_dict("records") if prediction_path.exists() else []
    metrics = pd.read_csv(metric_path).to_dict("records") if metric_path.exists() else []
    if bool(predictions) != bool(metrics):
        raise ValueError("Both prediction and metric files are required to resume")
    completed = {(row["repetition"], row["fold"], row["model"]) for row in metrics}
    if len(completed) != len(metrics):
        raise ValueError("Duplicate fitted folds in saved metrics")
    contexts = {}
    for repetition in range(REPETITIONS):
        for fold in range(OUTER_FOLDS):
            test_ids = set(folds.loc[(folds.repetition == repetition) & (folds.fold == fold), "patient_id"])
            test = data.patient_id.isin(test_ids).to_numpy()
            train = ~test
            train_data = data.loc[train].reset_index(drop=True)
            inner = inners.loc[(inners.repetition == repetition) & (inners.fold == fold)]
            train_time = train_data.os_time_days.to_numpy()
            train_event = train_data.os_event.to_numpy()
            test_time = data.os_time_days.to_numpy()[test]
            test_event = data.os_event.to_numpy()[test]
            contexts[(repetition, fold)] = (train_time, train_event)
            for name, features in embeddings.items():
                if (repetition, fold, name) in completed:
                    continue
                alpha, inner_scores = select_penalty(features[train], train_data, inner, components)
                fitted = fit_cox(features[train], train_time, train_event, alpha, components)
                risk, survival = predict(fitted, features[test], GRID)
                score = fold_metrics(train_time, train_event, test_time, test_event, risk, survival)
                if not np.isfinite(score["uno_c_36m"]):
                    raise ValueError(f"{name} repeat={repetition} fold={fold}: undefined Uno C")
                metrics.append({"model": name, "repetition": repetition, "fold": fold,
                                "alpha": alpha, "inner_uno_c": max(inner_scores),
                                "n_train": int(train.sum()), "n_test": int(test.sum()),
                                "events_train": int(train_event.sum()), "events_test": int(test_event.sum()),
                                **score})
                for i, (_, patient) in enumerate(data.loc[test].iterrows()):
                    record = {"model": name, "repetition": repetition, "fold": fold,
                              "patient_id": patient.patient_id, "os_time_days": test_time[i],
                              "os_event": test_event[i], "risk": risk[i]}
                    record.update({f"survival_{j:02d}": survival[i, j] for j in range(len(GRID))})
                    predictions.append(record)
                print(f"{name}: repetition {repetition+1}/{REPETITIONS}, fold {fold+1}/{OUTER_FOLDS}, "
                      f"alpha={alpha}, Uno C={score['uno_c_36m']:.3f}", flush=True)
                pd.DataFrame(predictions).to_csv(prediction_path, index=False)
                pd.DataFrame(metrics).to_csv(metric_path, index=False)
    pred = pd.DataFrame(predictions)
    metric = pd.DataFrame(metrics)
    expected = OUTER_FOLDS * REPETITIONS * len(embeddings)
    if len(metric) != expected or len(pred) != REPETITIONS * len(data) * len(embeddings):
        raise ValueError("Incomplete outer-fold predictions")
    summary = {name: {column: float(rows[column].mean()) for column in
                      ("uno_c_36m", "auc_1y", "auc_2y", "auc_3y", "ibs_3m_36m",
                       "calibration_in_large_2y", "calibration_slope")}
               for name, rows in metric.groupby("model")}
    if bootstrap:
        rng = np.random.default_rng(SEED + 999)
        draws = {name: [] for name in embeddings}
        pairs = {(a, b): [] for i, a in enumerate(embeddings) for b in list(embeddings)[i+1:]}
        groups = {(rep, fold, name): group.reset_index(drop=True) for (rep, fold, name), group in
                  pred.groupby(["repetition", "fold", "model"])}
        for _ in range(bootstrap):
            values = {name: [] for name in embeddings}
            for repetition in range(REPETITIONS):
                for fold in range(OUTER_FOLDS):
                    train_time, train_event = contexts[(repetition, fold)]
                    first = groups[(repetition, fold, next(iter(embeddings)))]
                    indices = rng.integers(0, len(first), len(first))
                    for name in embeddings:
                        group = groups[(repetition, fold, name)]
                        if not group.patient_id.equals(first.patient_id):
                            raise ValueError("Paired bootstrap patient ordering differs")
                        selected = group.iloc[indices]
                        values[name].append(uno_c(train_time, train_event,
                            selected.os_time_days.to_numpy(), selected.os_event.to_numpy(),
                            selected.risk.to_numpy()))
            means = {name: float(np.nanmean(values[name])) for name in embeddings}
            for name, value in means.items():
                draws[name].append(value)
            for pair in pairs:
                pairs[pair].append(means[pair[0]] - means[pair[1]])
        for name in embeddings:
            summary[name]["uno_c_36m_ci95"] = np.quantile(draws[name], [0.025, 0.975]).tolist()
        summary["paired_uno_c_differences"] = {
            f"{a}_minus_{b}": {"estimate": summary[a]["uno_c_36m"] - summary[b]["uno_c_36m"],
                                "ci95": np.quantile(values, [0.025, 0.975]).tolist()}
            for (a, b), values in pairs.items()}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=COHORT)
    parser.add_argument("--folds", type=Path, default=FOLDS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--embeddings", action="append", default=[], metavar="MODEL=MANIFEST.csv")
    parser.add_argument("--components", type=int, default=32,
                        help="PCA components; 0 keeps every standardized embedding feature")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--cox-device", default="cpu", help="Device for full-embedding ridge Cox optimization")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.components < 0 or args.bootstrap < 0:
        parser.error("components and bootstrap must be nonnegative")
    if args.components == 0:
        global ALPHAS, COX_DEVICE
        ALPHAS = (0.01, 0.1, 1.0, 10.0)
        COX_DEVICE = args.cox_device
    data = cohort(args.cohort)
    folds = locked_folds(data, args.folds)
    inners = inner_assignments(data, folds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inners.to_csv(args.output_dir / "inner_folds.csv", index=False)
    setup = {"cohort_sha256": sha256(args.cohort), "folds_sha256": sha256(args.folds),
             "n_patients": len(data), "n_events": int(data.os_event.sum()),
             "repetitions": REPETITIONS, "outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS,
             "seed": SEED, "components": args.components, "penalizers": list(ALPHAS),
             "horizons_days": list(HORIZONS), "brier_grid_days": GRID.tolist(),
             "bootstrap": args.bootstrap, "embedding_manifests": {}}
    if args.components == 0:
        setup["solver"] = "full_embedding_l2_cox_efron_lbfgs_breslow_baseline"
    protocol = args.output_dir / "protocol.json"
    previous = json.loads(protocol.read_text()) if protocol.exists() else None
    if previous is not None:
        for key in ("cohort_sha256", "folds_sha256", "components", "penalizers", "horizons_days", "brier_grid_days", "bootstrap", "solver"):
            if previous.get(key) != setup.get(key):
                raise ValueError(f"Existing protocol differs at {key}; use a new output directory")
    if args.prepare_only:
        if previous is None:
            protocol.write_text(json.dumps(setup, indent=2) + "\n")
        print(f"Locked {len(data)} patients and {len(folds)} outer assignments in {args.folds}")
        return
    if not args.embeddings:
        parser.error("At least one --embeddings MODEL=MANIFEST.csv is required")
    features = {}
    for spec in args.embeddings:
        name, matrix, manifest_checksum, matrix_checksum = load_embeddings(spec, data)
        if name in features:
            raise ValueError(f"Duplicate model: {name}")
        features[name] = matrix
        setup["embedding_manifests"][name] = {"sha256": manifest_checksum,
                                              "matrix_sha256": matrix_checksum,
                                              "dimension": matrix.shape[1]}
    if previous is not None and previous["embedding_manifests"] and previous["embedding_manifests"] != setup["embedding_manifests"]:
        raise ValueError("Existing output uses different embedding manifests; use a new output directory")
    protocol.write_text(json.dumps(setup, indent=2) + "\n")
    print(json.dumps(run(data, folds, inners, features, args.output_dir, args.components, args.bootstrap), indent=2))


if __name__ == "__main__":
    main()
