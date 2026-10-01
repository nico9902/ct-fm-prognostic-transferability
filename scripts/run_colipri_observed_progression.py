#!/usr/bin/env python3
"""Evaluate observed T0->T1 COLIPRI progression for prognosis after T2."""
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

from lung_jepa.models.observed_progression import ObservedProgressionPrognosticator
from lung_jepa.models.prognostic_adapter import masked_prognostic_loss
from run_colipri_latent_world_model import build_pairs, probe_metrics, seed_all
from run_colipri_preserving_adapter import ROOT, digest, fit_stage, save_json, status

TASKS = [f'primary_future_cancer_{h}' for h in ['1y', '2y', '3y']]


def build_triples(pairs: pd.DataFrame) -> pd.DataFrame:
    first = pairs[pairs.pair_type.eq('T0_T1')][
        ['patient_id', 'idx_t', 'idx_t1', 'dt_days', 'split']].rename(
        columns={'idx_t':'idx_t0', 'idx_t1':'idx_t1', 'dt_days':'dt01'})
    second = pairs[pairs.pair_type.eq('T1_T2')][
        ['patient_id', 'idx_t', 'idx_t1', 'dt_days', 'split',
         'primary_cancer_time_days'] + TASKS].rename(
        columns={'idx_t':'idx_t1_check', 'idx_t1':'idx_t2', 'dt_days':'dt12'})
    triples = first.merge(second, on=['patient_id', 'split'], validate='one_to_one')
    if not (triples.idx_t1 == triples.idx_t1_check).all():
        raise ValueError('T1 embedding mismatch between pair manifests')
    triples = triples.drop(columns='idx_t1_check').sort_values(['split', 'patient_id']).reset_index(drop=True)
    if triples.patient_id.duplicated().any() or triples.groupby('patient_id').split.nunique().max() != 1:
        raise ValueError('Expected exactly one patient-level sample')
    return triples


def cohort_audit(triples: pd.DataFrame, out: Path) -> None:
    rows = []
    for split in ['train', 'val', 'test', 'all']:
        sub = triples if split == 'all' else triples[triples.split.eq(split)]
        for task in TASKS:
            y = sub[task]
            rows.append(dict(split=split, task=task, patients=len(sub), valid=int(y.notna().sum()),
                             positives=int(y.eq(1).sum()), negatives=int(y.eq(0).sum()),
                             missing=int(y.isna().sum()), positive_fraction=float(y.mean())))
    pd.DataFrame(rows).to_csv(out/'class_distribution.csv', index=False)
    triples.groupby('split')[['dt01', 'dt12']].agg(['count','mean','std','min','median','max']).to_csv(
        out/'interval_distribution.csv')
    lags = []
    for split in ['train', 'val', 'test']:
        sub = triples[triples.split.eq(split) & triples.primary_future_cancer_1y.eq(1)]
        lags.append(dict(split=split, positives=len(sub),
                         **{f'within_{d}_days':int(sub.primary_cancer_time_days.le(d).sum())
                            for d in [30,90,180,365]}))
    pd.DataFrame(lags).to_csv(out/'diagnosis_lags.csv', index=False)


def evaluate(triples: pd.DataFrame, model_name: str, seed: int,
             logits: np.ndarray, weights: list[float]) -> list[dict]:
    rows = []
    probabilities = 1/(1+np.exp(-logits))
    for split in ['train', 'val', 'test']:
        split_mask = triples.split.eq(split).to_numpy()
        macro_loss = float(masked_prognostic_loss(torch.from_numpy(logits[split_mask]),
                           torch.from_numpy(triples.loc[split_mask,TASKS].to_numpy(np.float32)),
                           torch.tensor(weights)))
        for j, task in enumerate(TASKS):
            mask = split_mask & triples[task].notna().to_numpy()
            y = triples.loc[mask,task].to_numpy(float)
            rows.append(dict(model=model_name, seed=seed, split=split, task=task,
                             n=len(y), positives=int(y.sum()), macro_weighted_bce=macro_loss,
                             **probe_metrics(y, probabilities[mask,j])))
    return rows


def paired_bootstrap(triples: pd.DataFrame, predictions: dict[str,list[Path]],
                     cfg: dict, out: Path) -> None:
    ensembles = {name:np.mean([np.load(path)['probability'] for path in paths],axis=0)
                 for name,paths in predictions.items()}
    comparisons = [('history_gated','current_time_mlp'),
                   ('history_linear','current_time_linear'),
                   ('history_gated','current_linear')]
    rows = []
    test = triples.split.eq('test').to_numpy()
    for candidate, comparator in comparisons:
        if candidate not in ensembles or comparator not in ensembles:
            continue
        for j, task in enumerate(TASKS):
            mask = test & triples[task].notna().to_numpy()
            y = triples.loc[mask,task].to_numpy(int)
            a,b = ensembles[candidate][mask,j],ensembles[comparator][mask,j]
            rng = np.random.default_rng(20260922)
            draws = np.empty((cfg['bootstrap_samples'],2),dtype=float)
            for i in range(cfg['bootstrap_samples']):
                indices = rng.integers(len(y),size=len(y))
                if len(np.unique(y[indices])) < 2:
                    draws[i] = np.nan
                else:
                    draws[i] = [roc_auc_score(y[indices],a[indices])-roc_auc_score(y[indices],b[indices]),
                                average_precision_score(y[indices],a[indices])-average_precision_score(y[indices],b[indices])]
            for k,(metric,fn) in enumerate([('auroc',roc_auc_score),('auprc',average_precision_score)]):
                valid = draws[:,k][np.isfinite(draws[:,k])]
                low,high = np.quantile(valid,[.025,.975])
                rows.append(dict(candidate=candidate,comparator=comparator,task=task,metric=metric,
                                 gain=fn(y,a)-fn(y,b),ci_low=low,ci_high=high,
                                 valid_draws=len(valid),requested_draws=cfg['bootstrap_samples']))
    pd.DataFrame(rows).to_csv(out/'paired_patient_bootstrap.csv',index=False)


def write_report(out: Path) -> None:
    d = pd.read_csv(out/'metrics.csv')
    summary = d.groupby(['model','split','task'])[
        ['auroc','auprc','mcc','f1','recall','specificity','macro_weighted_bce']].agg(['mean','std','count'])
    summary.to_csv(out/'summary.csv')
    sections=[]
    for split in ['val','test']:
        table=d[d.split.eq(split)].pivot_table(index='model',columns='task',values=['auroc','auprc','mcc','f1'])
        sections.append(f'## {split}\n\n```text\n{table.round(4).to_string()}\n```')
    bootstrap=pd.read_csv(out/'paired_patient_bootstrap.csv')
    text = ('# Observed COLIPRI progression prognosis: completed results\n\n'
            'Inputs are frozen T0/T1 COLIPRI embeddings; labels are cancer after T2. '
            'One row per patient, identical split for every model. No test-based model selection. '
            'The predeclared primary comparison is history_gated versus current_time_mlp.\n\n'
            'Seed SD is not a patient confidence interval. Bootstrap uses seed-ensemble probabilities '
            'as a secondary paired patient analysis. Weighted-BCE outputs are not calibrated risk. '
            'This enriched development cohort and previously inspected test set are not external validation.\n\n'
            + '\n\n'.join(sections) + '\n\n## Paired patient bootstrap\n\n```text\n'
            + bootstrap.round(4).to_string(index=False) + '\n```\n')
    (out/'RESULTS.md').write_text(text)


def main() -> None:
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/model/colipri_observed_progression.yaml')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    cfg=yaml.safe_load(args.config.read_text())
    if args.smoke:
        cfg.update(output_dir='outputs/colipri_observed_progression_prognosis_smoke',seeds=[0],
                   models=['current_time_mlp','history_linear','history_gated'],epochs=2,patience=2,
                   bootstrap_samples=20,wandb_group=cfg['wandb_group']+'-smoke')
    if args.output_dir: cfg['output_dir']=str(args.output_dir)
    out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available(): raise RuntimeError('GPU required; no silent CPU fallback')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    paths=argparse.Namespace(**{key:Path(cfg[key]) for key in
        ['embedding_cache','prognostic_labels','diagnostic_manifest','patient_splits']})
    sources=[Path(__file__),ROOT/'src/lung_jepa/models/observed_progression.py',
             ROOT/'src/lung_jepa/models/prognostic_adapter.py',args.config]
    fingerprint={str(path):digest(path) for path in sources+[paths.embedding_cache/'embeddings.npy',
        paths.embedding_cache/'embedding_metadata.csv',paths.prognostic_labels,paths.patient_splits,
        Path(cfg['reference_pairs'])]}
    for filename,value in [('config.json',cfg),('fingerprint.json',fingerprint)]:
        if (out/filename).exists(): assert json.loads((out/filename).read_text())==value,f'Resume mismatch: {filename}'
        else: save_json(out/filename,value)
    if (out/'COMPLETE.json').exists(): print('Already complete',flush=True);return
    pairs,h=build_pairs(paths)
    reference=pd.read_csv(cfg['reference_pairs'],dtype={'patient_id':str})
    keys=['patient_id','pair_type','idx_t','idx_t1','split']+TASKS
    pd.testing.assert_frame_equal(pairs[keys],reference[keys])
    triples=build_triples(pairs);triples.to_csv(out/'triple_manifest.csv',index=False);cohort_audit(triples,out)
    train=triples.split.eq('train').to_numpy();val=triples.split.eq('val').to_numpy()
    raw_current=np.asarray(h[triples.idx_t1],np.float32)
    raw_delta=np.asarray(h[triples.idx_t1]-h[triples.idx_t0],np.float32)
    raw_oracle=np.asarray(h[triples.idx_t2],np.float32)
    current_scaler=StandardScaler().fit(raw_current[train]);delta_scaler=StandardScaler().fit(raw_delta[train])
    oracle_scaler=StandardScaler().fit(raw_oracle[train]);time_scaler=StandardScaler().fit(triples.loc[train,['dt01']])
    current=current_scaler.transform(raw_current).astype(np.float32)
    delta=delta_scaler.transform(raw_delta).astype(np.float32)
    oracle=oracle_scaler.transform(raw_oracle).astype(np.float32)
    time=time_scaler.transform(triples[['dt01']]).astype(np.float32)
    labels=triples[TASKS].to_numpy(np.float32)
    positives=np.nansum(labels[train],axis=0);negatives=np.sum(labels[train]==0,axis=0)
    weights=(negatives/positives).tolist()
    np.savez(out/'scalers.npz',current_mean=current_scaler.mean_,current_scale=current_scaler.scale_,
        delta_mean=delta_scaler.mean_,delta_scale=delta_scaler.scale_,oracle_mean=oracle_scaler.mean_,
        oracle_scale=oracle_scaler.scale_,time_mean=time_scaler.mean_,time_scale=time_scaler.scale_,
        pos_weights=weights)
    save_json(out/'provenance.json',dict(pid=os.getpid(),gpu=torch.cuda.get_device_name(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),patients=len(triples),
        split_counts=triples.split.value_counts().to_dict(),pos_weights=weights,
        model_parameters={name:sum(p.numel() for p in ObservedProgressionPrognosticator(
            name,hidden_dim=cfg['hidden_dim'],time_dim=cfg['time_dim']).parameters()) for name in cfg['models']}))
    rows=[];predictions={name:[] for name in cfg['models']}
    # Native Unicode avoids object arrays and unsafe allow_pickle on reload.
    ids=np.asarray(triples.patient_id.astype(str).tolist(),dtype='U')
    for model_name in cfg['models']:
        model_current=oracle if model_name=='oracle_t2_linear' else current
        for seed in cfg['seeds']:
            seed_all(seed);work=out/'runs'/f'{model_name}__seed{seed}'
            status(out,stage='training',model=model_name,seed=seed)
            model=ObservedProgressionPrognosticator(model_name,hidden_dim=cfg['hidden_dim'],
                time_dim=cfg['time_dim'],dropout=cfg['dropout'],pos_weights=weights,lr=cfg['lr'])
            if args.smoke:
                with torch.no_grad():
                    logits,fused=model(torch.from_numpy(model_current[:4]),torch.from_numpy(delta[:4]),torch.from_numpy(time[:4]))
                print(dict(model=model_name,current=list(model_current[:4].shape),delta=list(delta[:4].shape),
                           time=list(time[:4].shape),logits=list(logits.shape),
                           fused=None if fused is None else list(fused.shape)),flush=True)
            arrays=[model_current,delta,time,labels]
            datasets={split:TensorDataset(*(torch.from_numpy(array[mask]) for array in arrays))
                      for split,mask in [('train',train),('val',val)]}
            model=fit_stage(model,datasets['train'],datasets['val'],cfg,work,
                            f'{model_name}/seed{seed}',cfg['epochs'],'val/prognosis')
            model.eval().requires_grad_(False).cuda();parts=[]
            with torch.inference_mode():
                for start in range(0,len(triples),256):
                    parts.append(model(torch.tensor(model_current[start:start+256],device='cuda'),
                        torch.tensor(delta[start:start+256],device='cuda'),
                        torch.tensor(time[start:start+256],device='cuda'))[0].cpu().numpy())
            logits=np.concatenate(parts);probability=1/(1+np.exp(-logits))
            assert logits.shape==(len(triples),3) and np.isfinite(logits).all()
            export=work/'predictions.npz';np.savez(export,logits=logits,probability=probability,patient_ids=ids)
            predictions[model_name].append(export);rows.extend(evaluate(triples,model_name,seed,logits,weights))
            pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
            del model
    status(out,stage='bootstrap',training_runs=sum(len(v) for v in predictions.values()))
    paired_bootstrap(triples,predictions,cfg,out);write_report(out)
    run=wandb.init(project=cfg['wandb_project'],group=cfg['wandb_group'],name='summary-'+out.name,
                   dir=str(out),config=cfg)
    run.log({'metrics':wandb.Table(dataframe=pd.DataFrame(rows)),
             'bootstrap':wandb.Table(dataframe=pd.read_csv(out/'paired_patient_bootstrap.csv'))})
    run.finish()
    save_json(out/'COMPLETE.json',dict(training_runs=sum(len(v) for v in predictions.values()),
                                      bootstrap_samples=cfg['bootstrap_samples']))
    status(out,stage='complete',training_runs=sum(len(v) for v in predictions.values()))


if __name__=='__main__': main()
