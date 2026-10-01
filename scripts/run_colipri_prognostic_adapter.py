#!/usr/bin/env python3
"""GPU-only supervised adapter battery, with frozen representation re-probing."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

from run_colipri_preserving_adapter import (ROOT, digest, fit_stage, fit_probe,
                                           save_json, status, SCOPES)
from run_colipri_latent_world_model import build_pairs, probe_metrics, seed_all
from lung_jepa.models.prognostic_adapter import PrognosticAdapter, masked_prognostic_loss

TASKS = [f'primary_future_cancer_{h}' for h in ['1y', '2y', '3y']]


def scored_rows(pairs: pd.DataFrame, probabilities: np.ndarray, model: str,
                seed: int, representation: str) -> list[dict]:
    rows = []
    for split in ['train', 'val', 'test']:
        for scope in SCOPES:
            for j, task in enumerate(TASKS):
                mask = pairs.split.eq(split) & pairs[task].notna()
                if scope != 'pooled':
                    mask &= pairs.pair_type.eq(scope)
                y = pairs.loc[mask, task].to_numpy(float)
                metrics = probe_metrics(y, probabilities[mask, j]) if len(np.unique(y)) == 2 else {
                    k: float('nan') for k in ['auroc', 'auprc', 'mcc', 'f1']}
                rows.append(dict(model=model, wm_seed=seed, probe_seed=-1,
                                 representation=representation, split=split, scope=scope,
                                 task=task, n=len(y), positives=int(y.sum()), **metrics))
    return rows


def audit_cohort(pairs: pd.DataFrame, out: Path) -> np.ndarray:
    """Round control uses TRAIN prevalences only; no test outcome fitting."""
    counts, predicted = [], np.zeros((len(pairs), 3), dtype=np.float32)
    for scope in SCOPES:
        for split in ['train', 'val', 'test']:
            sub = pairs[pairs.split.eq(split)]
            if scope != 'pooled':
                sub = sub[sub.pair_type.eq(scope)]
            for task in TASKS:
                y = sub[task]
                counts.append(dict(scope=scope, split=split, task=task, pairs=len(sub),
                                   patients=sub.patient_id.nunique(), positives=int(y.eq(1).sum()),
                                   negatives=int(y.eq(0).sum()), missing=int(y.isna().sum())))
    for scope in ['T0_T1', 'T1_T2']:
        tr = pairs.split.eq('train') & pairs.pair_type.eq(scope)
        predicted[pairs.pair_type.eq(scope)] = pairs.loc[tr, TASKS].mean().to_numpy()
    pd.DataFrame(counts).to_csv(out/'class_distribution.csv', index=False)
    lags = []
    for split in ['train', 'val', 'test']:
        for scope in ['T0_T1', 'T1_T2']:
            sub = pairs[pairs.split.eq(split) & pairs.pair_type.eq(scope) &
                        pairs.primary_future_cancer_1y.eq(1)]
            lags.append(dict(split=split, scope=scope, positives=len(sub),
                             **{f'within_{d}_days': int(sub.primary_cancer_time_days.le(d).sum())
                                for d in [30, 90, 180, 365]}))
    pd.DataFrame(lags).to_csv(out/'diagnosis_lags.csv', index=False)
    return predicted


def bootstrap_heads(pairs: pd.DataFrame, exports: dict[str, list[Path]],
                    selected: str, cfg: dict, out: Path) -> None:
    """Secondary seed-ensemble score analysis, patient-clustered paired bootstrap.

    No embeddings are averaged. Primary tables retain every training/probe seed.
    These intervals describe the fixed test cohort, not model selection uncertainty.
    """
    probabilities = {m: np.mean([np.load(p)['probability'] for p in paths], axis=0)
                     for m, paths in exports.items() if m in [selected, 'linear', 'mlp']}
    rows = []
    for comparator in ['linear', 'mlp']:
        for scope in SCOPES:
            for j, task in enumerate(TASKS):
                mask = pairs.split.eq('test') & pairs[task].notna()
                if scope != 'pooled':
                    mask &= pairs.pair_type.eq(scope)
                sub = pairs[mask]
                y = sub[task].to_numpy()
                a, b = probabilities[selected][mask, j], probabilities[comparator][mask, j]
                patients = sub.patient_id.to_numpy()
                clusters = [np.flatnonzero(patients == pid) for pid in np.unique(patients)]
                rng = np.random.default_rng(1927)
                draws = []
                for _ in range(cfg['bootstrap_samples']):
                    idx = np.concatenate([clusters[i] for i in rng.integers(len(clusters), size=len(clusters))])
                    if len(np.unique(y[idx])) < 2:
                        continue
                    draws.append([fn(y[idx], a[idx])-fn(y[idx], b[idx])
                                  for fn in [roc_auc_score, average_precision_score]])
                for k, (metric, fn) in enumerate([('auroc', roc_auc_score), ('auprc', average_precision_score)]):
                    ci = np.quantile(np.asarray(draws)[:, k], [.025, .975]) if draws else [np.nan, np.nan]
                    rows.append(dict(model=selected, comparator=comparator, scope=scope, task=task,
                                     metric=metric, gain=fn(y, a)-fn(y, b), ci_low=ci[0], ci_high=ci[1],
                                     valid_draws=len(draws), requested_draws=cfg['bootstrap_samples']))
    pd.DataFrame(rows).to_csv(out/'secondary_ensemble_patient_bootstrap.csv', index=False)


def report(out: Path, selected: str) -> None:
    d = pd.read_csv(out/'metrics.csv')
    d.groupby(['model', 'representation', 'split', 'scope', 'task'])[
        ['auroc', 'auprc', 'mcc', 'f1']].agg(['mean', 'std', 'count']).to_csv(out/'summary.csv')
    tables = []
    for split in ['val', 'test']:
        for scope in SCOPES:
            t = d[d.split.eq(split) & d.scope.eq(scope)].pivot_table(
                index=['model', 'representation'], columns='task', values=['auroc', 'auprc'])
            tables.append(f'## {split}: {scope}\n\n```text\n{t.round(4).to_string()}\n```\n')
    text = ('# Supervised prognostic adapter: completed results\n\n'
            f'Validation-selected joint adapter: **{selected}**. Selection uses mean best validation '
            'masked class-weighted BCE across training seeds; no test selection.\n\n'
            'COLIPRI frozen; current scan only; post-t+1 endpoints; no forecasting in this phase. '
            'Direct heads and fresh frozen-feature probes are separate analyses. '
            'No round, scan interval or future feature enters any image model. '
            'Round-only probabilities are estimated on training patients. '
            'All subgroup results use pooled-trained models.\n\n'
            'Primary metrics retain individual seeds. Seed SD is not a patient confidence interval. '
            'Secondary bootstrap uses seed-averaged head probabilities, resamples patients jointly '
            'across their pairs, and skips single-class samples. The very rare T0→T1 1y endpoint '
            'is exploratory. Test has already been used in prior research; this is not a new '
            'untouched external validation. Weighted BCE probabilities are not calibrated risks.\n\n'
            'Full MCC/F1 and per-seed metrics: metrics.csv / summary.csv. '
            'Clinical predictive advantage must be judged against both current-only baselines '
            'and the round-only control, not reconstruction alone.\n\n' + '\n'.join(tables))
    (out/'RESULTS.md').write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/model/colipri_prognostic_adapter.yaml')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.smoke:
        cfg.update(seeds=[0], probe_seeds=[0], epochs=2, patience=2, probe_epochs=2,
                   probe_patience=2, bootstrap_samples=10,
                   output_dir='outputs/colipri_prognostic_adapter_smoke')
        cfg['variants'] = [v for v in cfg['variants'] if v['name'] in ['linear', 'mlp', 'adapter_joint_p1']]
        cfg['wandb_group'] += '-smoke'
    if args.output_dir:
        cfg['output_dir'] = str(args.output_dir)
    out = Path(cfg['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required; no silent CPU fallback')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    sources = [Path(__file__), ROOT/'src/lung_jepa/models/prognostic_adapter.py',
               ROOT/'src/lung_jepa/models/temporal_adapter.py',
               ROOT/'scripts/run_colipri_preserving_adapter.py',
               ROOT/'scripts/run_colipri_latent_world_model.py', args.config]
    paths = argparse.Namespace(**{k: Path(cfg[k]) for k in
                                 ['embedding_cache', 'prognostic_labels', 'diagnostic_manifest', 'patient_splits']})
    fingerprint = {str(p): digest(p) for p in sources + [paths.patient_splits, paths.prognostic_labels,
                    Path(cfg['reference_pairs']), paths.embedding_cache/'embedding_metadata.csv',
                    paths.embedding_cache/'embeddings.npy']}
    for filename, data in [('config.json', cfg), ('fingerprint.json', fingerprint)]:
        if (out/filename).exists():
            assert json.loads((out/filename).read_text()) == data, f'Resume mismatch: {filename}'
        else:
            save_json(out/filename, data)
    if (out/'COMPLETE.json').exists():
        print('Already complete', flush=True)
        return
    pairs, h = build_pairs(paths)
    reference = pd.read_csv(cfg['reference_pairs'], dtype={'patient_id': str})
    keys = ['patient_id', 'pair_type', 'idx_t', 'idx_t1', 'split'] + TASKS
    pd.testing.assert_frame_equal(pairs[keys], reference[keys])
    assert pairs.groupby('patient_id').split.nunique().max() == 1
    pair_ids = (pairs.patient_id.astype(str) + ':' + pairs.pair_type).to_numpy(dtype=str)
    assert len(np.unique(pair_ids)) == len(pairs)
    pairs.to_csv(out/'pairs_manifest.csv', index=False)
    train, val = pairs.split.eq('train').to_numpy(), pairs.split.eq('val').to_numpy()
    # Each current training scan enters scaling once, no future/test feature fitting.
    indices = np.unique(pairs.loc[train, 'idx_t'].to_numpy())
    scaler = StandardScaler().fit(h[indices])
    x = scaler.transform(h[pairs.idx_t]).astype(np.float32)
    y = pairs[TASKS].to_numpy(dtype=np.float32)
    pos = np.nansum(y[train], axis=0)
    neg = np.sum(y[train] == 0, axis=0)
    assert (pos > 0).all() and (neg > 0).all()
    weights = (neg/pos).tolist()
    np.savez(out/'scalers.npz', mean=scaler.mean_, scale=scaler.scale_, train_scan_indices=indices,
             pos_weights=weights)
    save_json(out/'provenance.json', dict(pid=os.getpid(), gpu=torch.cuda.get_device_name(),
              cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), pairs=len(pairs),
              patients=pairs.patient_id.nunique(), pos_weights=weights))
    round_probs = audit_cohort(pairs, out)
    rows = scored_rows(pairs, round_probs, 'round_only', -1, 'training_prevalence')
    np.savez(out/'round_probabilities.npz', probability=round_probs, pair_ids=pair_ids)
    exports, validation = {}, []
    for v in cfg['variants']:
        exports[v['name']] = []
        for seed in cfg['seeds']:
            seed_all(seed)
            work = out/'runs'/f"{v['name']}__seed{seed}"
            status(out, stage='training', model=v['name'], seed=seed)
            model = PrognosticAdapter(**{k: value for k, value in v.items() if k != 'name'},
                hidden_dim=cfg['hidden_dim'], decoder_hidden=cfg['decoder_hidden'], gamma=cfg['gamma'],
                pos_weights=weights, lr=cfg['lr'])
            # Smoke output is CPU; GPU training below is always a W&B-logged background job.
            if args.smoke:
                batch = (torch.from_numpy(x[:4]), torch.from_numpy(y[:4]))
                z, logits = model(batch[0])
                print({'input':list(batch[0].shape), 'z':list(z.shape), 'logits':list(logits.shape),
                       'losses':{k:float(t.detach()) for k,t in model.losses(*batch).items()}}, flush=True)
            model = fit_stage(model, TensorDataset(torch.from_numpy(x[train]), torch.from_numpy(y[train])),
                TensorDataset(torch.from_numpy(x[val]), torch.from_numpy(y[val])), cfg,
                work, f"{v['name']}/seed{seed}", cfg['epochs'], 'val/prognosis')
            done = json.loads((work/'done.json').read_text())
            validation.append(dict(model=v['name'], seed=seed, val_loss=done['best_validation']))
            model.eval().requires_grad_(False).cuda()
            latent, probs, logits_parts, reconstructed = [], [], [], []
            with torch.inference_mode():
                for chunk in np.array_split(x, max(1, int(np.ceil(len(x)/256)))):
                    z, logits = model(torch.tensor(chunk, device='cuda'))
                    latent.append(z.cpu().numpy())
                    probs.append(logits.sigmoid().cpu().numpy())
                    logits_parts.append(logits.cpu().numpy())
                    if model.decoder is not None:
                        reconstructed.append(model.decoder(z).cpu().numpy())
            z, prob = np.concatenate(latent), np.concatenate(probs)
            assert np.isfinite(z).all() and np.isfinite(prob).all()
            export = work/'representations.npz'
            logits_array = np.concatenate(logits_parts)
            np.savez(export, current=z, probability=prob, logits=logits_array, pair_ids=pair_ids)
            diagnostics = {}
            reconstruction = np.concatenate(reconstructed) if reconstructed else None
            for split in ['train', 'val', 'test']:
                mask = pairs.split.eq(split).to_numpy()
                diagnostics[split] = dict(
                    prognosis_loss=float(masked_prognostic_loss(torch.from_numpy(logits_array[mask]),
                        torch.from_numpy(y[mask]), torch.tensor(weights))),
                    displacement_mse=float(np.mean((z[mask]-x[mask])**2)),
                    reconstruction_mse=(float(np.mean((reconstruction[mask]-x[mask])**2))
                                        if reconstruction is not None else None),
                    mean_feature_std=float(z[mask].std(0).mean()))
            save_json(work/'representation_diagnostics.json', diagnostics)
            exports[v['name']].append(export)
            rows.extend(scored_rows(pairs, prob, v['name'], seed, 'trained_head'))
            del model
    vf = pd.DataFrame(validation)
    vf.to_csv(out/'validation_selection.csv', index=False)
    eligible = vf[vf.model.str.startswith('adapter_joint')].groupby('model').val_loss.mean()
    selected = str(eligible.idxmin())
    save_json(out/'selection.json', dict(selected=selected, criterion='mean validation weighted masked BCE',
                                       candidates=eligible.to_dict()))
    pd.DataFrame(rows).to_csv(out/'metrics.csv', index=False)
    status(out, stage='frozen_probes', selected=selected, completed_probes=0)
    run = wandb.init(project=cfg['wandb_project'], group=cfg['wandb_group'],
                     name='evaluation-'+out.name, dir=str(out), config=cfg)
    features = [('native_current', -1, h[pairs.idx_t]), ('native_future_oracle', -1, h[pairs.idx_t1])]
    for name, paths_list in exports.items():
        if name == 'linear':  # Its representation is exactly standardized native input.
            continue
        features.extend((name, seed, np.load(p)['current']) for seed, p in zip(cfg['seeds'], paths_list))
    completed = 0
    total = len(features)*len(TASKS)*len(cfg['probe_seeds'])
    for name, seed, features_array in features:
        for task in TASKS:
            for probe_seed in cfg['probe_seeds']:
                result = fit_probe(features_array, pairs, task, probe_seed, cfg,
                                   out/'probes'/name/f'seed{seed}'/task/f'probe{probe_seed}')
                rows.extend(dict(model=name, wm_seed=seed, representation='frozen_linear_probe', **r)
                            for r in result['rows'])
                completed += 1
                run.log(dict(completed_probes=completed, total_probes=total,
                             model=name, task=task, wm_seed=seed, probe_seed=probe_seed))
            pd.DataFrame(rows).to_csv(out/'metrics.csv', index=False)
            status(out, stage='frozen_probes', model=name, seed=seed,
                   completed_probes=completed, total_probes=total)
    status(out, stage='patient_bootstrap')
    bootstrap_heads(pairs, exports, selected, cfg, out)
    report(out, selected)
    run.log({'metrics': wandb.Table(dataframe=pd.DataFrame(rows)),
             'selected_adapter': selected,
             'bootstrap': wandb.Table(dataframe=pd.read_csv(out/'secondary_ensemble_patient_bootstrap.csv'))})
    run.finish()
    save_json(out/'COMPLETE.json', dict(selected=selected, completed_probes=completed,
                                      training_runs=len(validation)))
    status(out, stage='complete', completed_probes=completed, training_runs=len(validation))


if __name__ == '__main__':
    main()
