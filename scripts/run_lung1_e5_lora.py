#!/usr/bin/env python3
"""Exploratory E5 LoRA: separate LUNG1 training, external prediction and evaluation."""
import argparse
import json
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch

from run_lung1_e2_encoder_bf16_lightning import (
    ROOT, COHORT, CHECKPOINTS, GRID, SETTINGS, AdaptedRisk, FoldModel,
    cohort, load_inputs, load_encoder, risk_all, breslow_survival, sha256, write_json, precision_options,
)
from run_lung1_frozen_cox_e1 import uno_c

EXTERNAL = ROOT / "data/derived/radiogenomics/cohort_with_gtv_volume.csv"
CACHE = ROOT / "outputs/radiogenomics_lora_inputs"
OUTPUT = ROOT / "outputs/lung1_e5_lora"
PRECISION = "bf16-mixed"


def protocol(model):
    return dict(experiment="E5-exploratory-LoRA", model=model, lr=1e-3, epochs=20,
                seed=20261005, precision=PRECISION, training="all_LUNG1",
                checkpoint_sha256=sha256(CHECKPOINTS[model]), cohort_sha256=sha256(COHORT),
                settings={**SETTINGS['lora'], 'lr': 1e-3},
                implementation_sha256=sha256(Path(__file__)),
                training_implementation_sha256=sha256(ROOT / 'scripts/run_lung1_e2_encoder_bf16_lightning.py'))


def checked_frozen(output, model):
    manifest = json.loads((output / 'FROZEN.json').read_text())
    if manifest['protocol'] != protocol(model):
        raise ValueError('Training protocol changed')
    for name, digest in manifest['files'].items():
        if sha256(output / name) != digest:
            raise ValueError(f'Frozen file changed: {name}')
    return manifest


def train(output, model):
    if (output / 'FROZEN.json').exists():
        checked_frozen(output, model)
        print(f'{model}: already frozen')
        return
    data = cohort(COHORT)
    paths, provenance = load_inputs(model, data)
    time = data.os_time_days.to_numpy(float)
    event = data.os_event.to_numpy(int)
    pl.seed_everything(20261005, workers=True)
    SETTINGS['lora'] = {**SETTINGS['lora'], 'lr': 1e-3}
    encoder, _, _ = load_encoder(model, torch.device('cpu'), CHECKPOINTS[model])
    net = AdaptedRisk(model, encoder, 'lora')
    module = FoldModel(net, 'lora', paths, time, event, [], np.array([]), np.array([]),
                       [], np.array([]), np.array([]))
    trainer = pl.Trainer(accelerator='gpu', devices=1, **precision_options(PRECISION), max_epochs=20,
                         logger=False, enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, limit_val_batches=0, num_sanity_val_steps=0)
    trainer.fit(module)
    module.to('cuda').eval()
    with torch.autocast('cuda', dtype=torch.bfloat16 if PRECISION == 'bf16-mixed' else torch.float16,
                        enabled=PRECISION != '32-true'):
        risk = risk_all(module.net, paths, torch.device('cuda'))
    if not np.isfinite(risk).all():
        raise ValueError('Nonfinite training risks')
    torch.save({'trainable_state': {n: p.detach().cpu() for n, p in net.named_parameters()
                                   if p.requires_grad}}, output / 'adapter.pt')
    np.savez(output / 'baseline.npz', time=time, event=event, risk=risk)
    write_json(output / 'FROZEN.json', {'protocol': protocol(model), 'inputs': provenance,
               'train_patient_ids': data.patient_id.tolist(),
               'files': {name: sha256(output / name) for name in ('adapter.pt', 'baseline.npz')}})


def predict(output, model):
    checked_frozen(output, model)
    # Outcome columns are removed before any external inference.
    data = pd.read_csv(EXTERNAL, dtype={'patient_id': str, 'ct_series_uid': str}).drop(
        columns=['os_time_days', 'os_event'])
    data = data.loc[data.global_status.eq('ready')].sort_values('patient_id').reset_index(drop=True)
    folder = CACHE / model / 'original_ct'
    manifest_path = folder / 'input_manifest.csv'
    manifest = pd.read_csv(manifest_path, dtype=str)
    cache_protocol = json.loads((folder / 'protocol.json').read_text())
    if (cache_protocol['cohort_sha256'] != sha256(EXTERNAL)
            or cache_protocol.get('clamp_min_hu') != -1024
            or cache_protocol.get('status_column') != 'global_status'
            or cache_protocol['checkpoint_sha256'] != sha256(CHECKPOINTS[model])):
        raise ValueError('External input provenance mismatch')
    if manifest.patient_id.duplicated().any() or set(manifest.patient_id) != set(data.patient_id):
        raise ValueError('External cache coverage mismatch')
    manifest = manifest.set_index('patient_id').loc[data.patient_id]
    if not np.array_equal(manifest.ct_series_uid, data.ct_series_uid):
        raise ValueError('External CT series mismatch')
    provenance = {'frozen_sha256': sha256(output / 'FROZEN.json'),
                  'cohort_sha256': sha256(EXTERNAL), 'inputs_sha256': sha256(manifest_path),
                  'input_protocol_sha256': sha256(folder / 'protocol.json')}
    if (output / 'PREDICTED.json').exists():
        old = json.loads((output / 'PREDICTED.json').read_text())
        if old != {**provenance, 'predictions_sha256': sha256(output / 'predictions.csv')}:
            raise ValueError('Existing prediction provenance differs')
        return
    pl.seed_everything(20261005, workers=True)
    encoder, _, _ = load_encoder(model, torch.device('cpu'), CHECKPOINTS[model])
    net = AdaptedRisk(model, encoder, 'lora')
    state = torch.load(output / 'adapter.pt', map_location='cpu', weights_only=True)['trainable_state']
    if set(state) != {n for n, p in net.named_parameters() if p.requires_grad}:
        raise ValueError('Adapter parameter coverage mismatch')
    with torch.no_grad():
        for name, parameter in net.named_parameters():
            if name in state:
                parameter.copy_(state[name])
    net.to('cuda').eval()
    with torch.autocast('cuda', dtype=torch.bfloat16 if PRECISION == 'bf16-mixed' else torch.float16,
                        enabled=PRECISION != '32-true'):
        risk = risk_all(net, [folder / p for p in manifest.input_path], torch.device('cuda'))
    baseline = np.load(output / 'baseline.npz', allow_pickle=False)
    survival = breslow_survival(baseline['time'], baseline['event'], baseline['risk'], risk)
    if not np.isfinite(risk).all() or not np.isfinite(survival).all():
        raise ValueError('Nonfinite external predictions')
    result = pd.DataFrame({'patient_id': data.patient_id, 'risk': risk})
    for j in range(len(GRID)):
        result[f'survival_{j:02d}'] = survival[:, j]
    result.to_csv(output / 'predictions.csv', index=False)
    write_json(output / 'PREDICTED.json', {**provenance,
               'predictions_sha256': sha256(output / 'predictions.csv')})


def evaluate(output, model):
    checked_frozen(output, model)
    stamp = json.loads((output / 'PREDICTED.json').read_text())
    if (stamp['predictions_sha256'] != sha256(output / 'predictions.csv')
            or stamp['frozen_sha256'] != sha256(output / 'FROZEN.json')
            or stamp['cohort_sha256'] != sha256(EXTERNAL)):
        raise ValueError('Prediction/cohort/frozen hashes changed')
    data = pd.read_csv(EXTERNAL, dtype={'patient_id': str})
    predicted = pd.read_csv(output / 'predictions.csv', dtype={'patient_id': str})
    data = data.loc[data.global_status.eq('ready')].merge(predicted, on='patient_id', validate='one_to_one')
    frozen_path = ROOT / 'outputs/phase4_external/predictions.csv'
    frozen = pd.read_csv(frozen_path, dtype={'patient_id': str})
    results = {}
    for population, selected in [('primary', data.region_status.eq('ready')),
                                  ('secondary', np.ones(len(data), dtype=bool))]:
        part = data.loc[selected].copy()
        comparison = frozen.loc[(frozen.model == model) & (frozen.population == population)]
        part = part.merge(comparison[['patient_id', 'risk']].rename(columns={'risk': 'frozen_risk'}),
                          on='patient_id', validate='one_to_one')
        if len(part) != int(np.sum(selected)):
            raise ValueError('Missing paired frozen predictions')
        t, e = part.os_time_days.to_numpy(float), part.os_event.to_numpy(int)
        r, f = part.risk.to_numpy(float), part.frozen_risk.to_numpy(float)
        def score(ix, risk):
            return float(uno_c(t[ix], e[ix], t[ix], e[ix], risk[ix], 1095.75))
        ix = np.arange(len(part))
        observed, reference = score(ix, r), score(ix, f)
        rng = np.random.default_rng(20261023)
        draws = []
        for _ in range(2000):
            sample = rng.integers(0, len(part), len(part))
            a, b = score(sample, r), score(sample, f)
            if np.isfinite(a) and np.isfinite(b):
                draws.append((a, a-b))
        if len(draws) < 1900:
            raise ValueError('Too few valid bootstrap replicates')
        results[population] = {'patients': len(part), 'events': int(e.sum()),
            'uno_c_36m': observed, 'frozen_uno_c_36m': reference,
            'difference': observed-reference,
            'ci95': np.percentile(np.array(draws)[:, 0], [2.5, 97.5]).tolist(),
            'difference_ci95': np.percentile(np.array(draws)[:, 1], [2.5, 97.5]).tolist(),
            'valid_bootstrap': len(draws)}
    write_json(output / 'summary.json', {'model': model, 'exploratory': True,
        'censoring': 'external_population', 'bootstrap_seed': 20261023,
        'frozen_predictions_sha256': sha256(frozen_path), 'populations': results})
    (output / 'COMPLETE').write_text('Evaluated exploratory E5\n')


def main():
    global PRECISION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision', choices=('bf16-mixed', '16-mixed', '32-true'), default='bf16-mixed')
    parser.add_argument('--model', choices=('colipri', 'tangerine'), required=True)
    parser.add_argument('--stage', choices=('train', 'predict', 'evaluate'), required=True)
    parser.add_argument('--output-root', type=Path, default=OUTPUT)
    args = parser.parse_args()
    PRECISION = args.precision
    output = args.output_root / args.model
    output.mkdir(parents=True, exist_ok=True)
    if args.stage != 'evaluate':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required')
        if PRECISION == 'bf16-mixed' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 unsupported; choose FP16 or FP32')
    globals()[args.stage](output, args.model)


if __name__ == '__main__':
    main()
