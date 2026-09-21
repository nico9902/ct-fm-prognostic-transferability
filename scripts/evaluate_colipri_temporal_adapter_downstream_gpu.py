#!/usr/bin/env python3
"""GPU-native downstream prognosis evaluation for saved temporal-adapter runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score

from run_colipri_latent_world_model import HORIZONS, build_pairs


def args():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('outputs/colipri_temporal_adapter_ablation'))
    p.add_argument('--embedding-cache', type=Path, default=Path('outputs/colipri_decodability/mean/cache'))
    p.add_argument('--prognostic-labels', type=Path, default=Path('data/nlst/metadata/nlst/prognostic_labels/pilot_1000_balanced/nlst_scan_prognostic_labels.csv'))
    p.add_argument('--diagnostic-manifest', type=Path, default=Path('outputs/colipri_decodability/experiment_scan_manifest.csv'))
    p.add_argument('--patient-splits', type=Path, default=Path('outputs/colipri_decodability/patient_splits.csv'))
    p.add_argument('--device', default='cuda'); p.add_argument('--epochs', type=int, default=100); p.add_argument('--patience', type=int, default=15)
    p.add_argument('--probe-seeds', nargs='+', type=int, default=[0,1,2,3,4]); p.add_argument('--wandb-project', default='nlst-colipri-temporal-adapter'); p.add_argument('--no-wandb', action='store_true')
    return p.parse_args()


def metric(y, score):
    pred = (score >= .5).astype(int); tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return {'auroc': float(roc_auc_score(y, score)), 'auprc': float(average_precision_score(y, score)), 'mcc': float(matthews_corrcoef(y, pred)), 'f1': float(f1_score(y, pred, zero_division=0)), 'specificity': float(tn/(tn+fp)) if tn+fp else np.nan}


def fit_probe(xtr, ytr, xval, yval, xte, yte, seed, device, epochs, patience):
    torch.manual_seed(seed); xtr = torch.as_tensor(xtr, dtype=torch.float32, device=device); xval = torch.as_tensor(xval, dtype=torch.float32, device=device); xte = torch.as_tensor(xte, dtype=torch.float32, device=device)
    ytr = torch.as_tensor(ytr, dtype=torch.float32, device=device); yval = torch.as_tensor(yval, dtype=torch.float32, device=device)
    mean, std = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True).clamp_min(1e-6); xtr, xval, xte = [((x-mean)/std) for x in [xtr, xval, xte]]
    pos, neg = ytr.sum().clamp_min(1), (1-ytr).sum().clamp_min(1); loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=(neg/pos).reshape(1)); clf = torch.nn.Linear(xtr.shape[1], 1, device=device); opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    best, state, stale = float('inf'), None, 0
    for _ in range(epochs):
        clf.train(); opt.zero_grad(set_to_none=True); loss = loss_fn(clf(xtr).squeeze(1), ytr); loss.backward(); opt.step()
        clf.eval(); val_loss = float(loss_fn(clf(xval).squeeze(1), yval).item())
        if val_loss < best - 1e-5: best, state, stale = val_loss, {k:v.detach().clone() for k,v in clf.state_dict().items()}, 0
        else: stale += 1
        if stale >= patience: break
    clf.load_state_dict(state); clf.eval()
    with torch.inference_mode(): score = torch.sigmoid(clf(xte).squeeze(1)).cpu().numpy()
    return metric(yte, score)


def main():
    a = args(); root = a.root; device = torch.device(a.device if torch.cuda.is_available() else 'cpu'); root.mkdir(exist_ok=True)
    pair_args = argparse.Namespace(embedding_cache=a.embedding_cache, prognostic_labels=a.prognostic_labels, diagnostic_manifest=a.diagnostic_manifest, patient_splits=a.patient_splits)
    pairs, embeddings = build_pairs(pair_args); split_tables = {s:pairs[pairs.split.eq(s)].reset_index(drop=True) for s in ['train','val','test']}; split_idx = {s:pairs.split.eq(s).to_numpy() for s in split_tables}
    npz_files = sorted((root/'predictions').glob('*.npz')); runs = {}
    for f in npz_files:
        name = f.stem; data=np.load(f); runs[name] = {s:{'z_t':data[f'{s}_state'], 'predicted_future':data[f'{s}_pred']} for s in ['train','val','test']}
    z = np.load(a.embedding_cache/'embeddings.npy', mmap_mode='r'); identity_name='identity__seed0'; runs[identity_name]={}
    for s in ['train','val','test']:
        table=split_tables[s]; src=np.asarray(z[table.idx_t.to_numpy()], dtype=np.float32); runs[identity_name][s]={'z_t':src,'predicted_future':src}
    rows=[]
    for run_name, features in runs.items():
        for feature in ['z_t','predicted_future','predicted_delta','concat']:
            for horizon in HORIZONS:
                label=f'primary_future_cancer_{horizon}'; ys={s:pd.to_numeric(split_tables[s][label],errors='coerce').to_numpy(float) for s in split_tables}; masks={s:np.isfinite(ys[s]) for s in ys}
                if any(masks[s].sum()==0 for s in masks) or len(np.unique(ys['train'][masks['train']]))<2: continue
                xs={s:features[s]['predicted_future']-features[s]['z_t'] if feature=='predicted_delta' else (np.concatenate([features[s]['z_t'],features[s]['predicted_future']],1) if feature=='concat' else features[s][feature]) for s in split_tables}
                for seed in a.probe_seeds:
                    result=fit_probe(xs['train'][masks['train']],ys['train'][masks['train']].astype(int),xs['val'][masks['val']],ys['val'][masks['val']].astype(int),xs['test'][masks['test']],ys['test'][masks['test']].astype(int),seed,device,a.epochs,a.patience)
                    rows.append({'run':run_name,'feature':feature,'label':label,'probe_seed':seed,**result})
                    if len(rows)%20==0: print(f'completed probes={len(rows)} device={device}',flush=True)
    out=pd.DataFrame(rows); out.to_csv(root/'downstream_prognostic_metrics.csv',index=False); out.groupby(['run','feature','label'])[['auroc','auprc','mcc','f1']].agg(['mean','std']).to_csv(root/'downstream_prognostic_summary.csv')
    if not a.no_wandb:
        import wandb
        run=wandb.init(project=a.wandb_project, name='gpu-downstream-summary', group='temporal-adapter-ablation', config={'device':str(device),'probe_seeds':a.probe_seeds}); run.log({'downstream/n_rows':len(out)}); run.finish()
    print(f'Done: {root}/downstream_prognostic_summary.csv',flush=True)


if __name__=='__main__': main()
