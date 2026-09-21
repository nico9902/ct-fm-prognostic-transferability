#!/usr/bin/env python3
"""Re-export existing checkpoints in manifest order and evaluate frozen probes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from run_colipri_temporal_adapter_ablation import enrich_previous, latent_metrics
from run_colipri_latent_world_model import build_pairs, probe_metrics, seed_all
from lung_jepa.data.colipri_temporal import COLIPRITemporalDataset
from lung_jepa.models.temporal_module import TemporalFutureStateModule


def fit_probe(xs: dict, ys: dict, seed: int) -> tuple[dict, np.ndarray]:
    """Matched linear/BCE/AdamW mini-batch protocol for every representation."""
    seed_all(seed)
    scaler = StandardScaler().fit(xs['train'])
    x = {s: torch.tensor(scaler.transform(v), dtype=torch.float32, device='cuda') for s,v in xs.items()}
    y = {s: torch.tensor(v, dtype=torch.float32, device='cuda') for s,v in ys.items()}
    clf = torch.nn.Linear(x['train'].shape[1], 1, device='cuda')
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=((1-y['train']).sum()/y['train'].sum()).reshape(1))
    opt = torch.optim.AdamW(clf.parameters(), lr=.001, weight_decay=.0001)
    best, stale, state = float('inf'), 0, None
    for epoch in range(100):
        # Historical COLIPRI probe resets the batch permutation seed each epoch.
        order = torch.randperm(len(y['train']), generator=torch.Generator().manual_seed(seed)).to('cuda')
        for idx in order.split(128):
            opt.zero_grad(set_to_none=True)
            loss_fn(clf(x['train'][idx]).squeeze(1), y['train'][idx]).backward()
            opt.step()
        with torch.no_grad():
            vl = float(loss_fn(clf(x['val']).squeeze(1), y['val']))
        if vl < best-1e-5:
            best, stale = vl, 0
            state = {k:v.detach().clone() for k,v in clf.state_dict().items()}
        else:
            stale += 1
        if stale >= 15:
            break
    clf.load_state_dict(state)
    with torch.no_grad():
        prob = clf(x['test']).squeeze(1).sigmoid().cpu().numpy()
    return {**probe_metrics(ys['test'], prob), 'epochs':epoch+1, 'val_loss':best}, prob


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/colipri_temporal_adapter_ablation'))
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; CPU fallback is disabled')
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    paths = argparse.Namespace(embedding_cache=Path('outputs/colipri_decodability/mean/cache'),
        prognostic_labels=Path('data/nlst/metadata/nlst/prognostic_labels/pilot_1000_balanced/nlst_scan_prognostic_labels.csv'),
        diagnostic_manifest=Path('outputs/colipri_decodability/experiment_scan_manifest.csv'),
        patient_splits=Path('outputs/colipri_decodability/patient_splits.csv'))
    pairs, h = build_pairs(paths)
    original = pd.read_csv(a.root/'pairs_manifest.csv', dtype={'patient_id':str})
    keys = ['patient_id','pair_type','idx_t','idx_t1','split']
    pd.testing.assert_frame_equal(pairs[keys].reset_index(drop=True), original[keys].reset_index(drop=True))
    assert pairs.groupby('patient_id').split.nunique().max() == 1
    pairs = enrich_previous(pairs, paths.embedding_cache)
    pairs.to_csv(a.output/'pairs_manifest.csv', index=False)
    tables = {s:pairs[pairs.split.eq(s)].reset_index(drop=True) for s in ['train','val','test']}
    ckpts = sorted((a.root/'checkpoints').glob('*/*/best.ckpt'))
    assert len(ckpts) == 30, len(ckpts)
    run = wandb.init(project='nlst-colipri-temporal-adapter', group='corrected-evaluation',
        name=a.output.name, dir=str(a.output), config={'checkpoints':30,'probe_seeds':list(range(5)),
        'probe':'Linear BCE AdamW lr=.001 wd=.0001 batch=128 epochs=100 patience=15',
        'device':torch.cuda.get_device_name(), 'source':str(a.root), 'training':False})
    runs = {'native': {s:{'current':h[t.idx_t], 'actual_future':h[t.idx_t1]} for s,t in tables.items()}}
    audit, latent = [], []
    (a.output/'predictions').mkdir(exist_ok=True)
    for ckpt in ckpts:
        model = TemporalFutureStateModule.load_from_checkpoint(str(ckpt), map_location='cpu').eval().cuda()
        model.requires_grad_(False)
        name = ckpt.parent.parent.name+'__'+ckpt.parent.name
        audit.append({'run':name, 'checkpoint':str(ckpt), 'sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                      'lambda_sigreg':float(model.hparams.lambda_sigreg)})
        runs[name], saved = {}, {}
        for split,t in tables.items():
            loader = DataLoader(COLIPRITemporalDataset(t,h), batch_size=256, shuffle=False)
            arrays, ids = [], []
            with torch.inference_mode():
                for batch in loader:
                    ids.extend(batch['patient_id'])
                    batch = {k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
                    arrays.append(tuple(v.cpu().numpy() for v in model(batch)))
            assert ids == t.patient_id.astype(str).tolist()
            pred,target,current = [np.concatenate([b[i] for b in arrays]) for i in range(3)]
            if model.adapter is None:
                np.testing.assert_array_equal(current,h[t.idx_t])
                np.testing.assert_array_equal(target,h[t.idx_t1])
            runs[name][split] = {'current':current,'predicted_future':pred,'actual_future':target,
                                'predicted_delta':pred-current,'concat':np.concatenate([current,pred],1)}
            saved.update({f'{split}_{k}':v for k,v in runs[name][split].items()})
            saved[f'{split}_pair_ids'] = (t.patient_id.astype(str)+':'+t.pair_type).to_numpy(dtype=str)
            for rep,value in [('identity',current),('predicted_future',pred)]:
                latent.append({'run':name,'split':split,'representation':rep,**latent_metrics(value,target)})
        np.savez(a.output/'predictions'/f'{name}.npz', **saved)
        del model
        print('exported and verified',name,flush=True)
    pd.DataFrame(audit).to_csv(a.output/'checkpoint_audit.csv',index=False)
    pd.DataFrame(latent).to_csv(a.output/'latent_metrics.csv',index=False)
    run.config.update({'checkpoint_sigreg_weights':sorted(set(x['lambda_sigreg'] for x in audit))})
    rows = []
    (a.output/'probe_predictions').mkdir(exist_ok=True)
    for name,features in runs.items():
        for rep in features['train']:
            for horizon in ['1y','2y','3y']:
                label = 'primary_future_cancer_'+horizon
                valid = {s:t[label].notna().to_numpy() for s,t in tables.items()}
                xs = {s:features[s][rep][valid[s]] for s in tables}
                ys = {s:tables[s].loc[valid[s],label].to_numpy(dtype=int) for s in tables}
                assert all(len(np.unique(y))==2 for y in ys.values())
                for seed in range(5):
                    result,prob = fit_probe(xs,ys,seed)
                    row = {'run':name,'model':name.split('__')[0],'representation':rep,'horizon':horizon,
                           'probe_seed':seed,'n_test':len(ys['test']),**result}
                    rows.append(row)
                    test = tables['test'].loc[valid['test'],keys].copy()
                    test['label'],test['probability'] = ys['test'],prob
                    test.to_csv(a.output/'probe_predictions'/f'{name}_{rep}_{horizon}_{seed}.csv',index=False)
                    run.log({'completed_probes':len(rows),'auroc':result['auroc'],'auprc':result['auprc']})
                pd.DataFrame(rows).to_csv(a.output/'prognostic_per_seed.csv',index=False)
                print(f'completed={len(rows)} {name} {rep} {horizon}',flush=True)
    frame = pd.DataFrame(rows)
    summary = frame.groupby(['model','representation','horizon'])[['auroc','auprc','mcc','f1']].agg(['mean','std'])
    summary.to_csv(a.output/'prognostic_summary.csv')
    run.log({'summary':wandb.Table(dataframe=frame)})
    run.finish()
    (a.output/'COMPLETE.json').write_text(json.dumps({'probes':len(rows),'checkpoints':len(audit)}))
    print('Completed corrected evaluation',flush=True)


if __name__ == '__main__':
    main()
