#!/usr/bin/env python3
"""Prognosis from observed native COLIPRI token changes, without a Transformer."""
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

from lung_jepa.models.prognostic_adapter import masked_prognostic_loss
from lung_jepa.models.spatial_progression import SpatialProgressionPrognosticator
from run_colipri_latent_world_model import build_pairs, probe_metrics, seed_all
from run_colipri_observed_progression import TASKS, build_triples, cohort_audit
from run_colipri_preserving_adapter import ROOT, digest, fit_stage, save_json, status


def scale_tokens(raw: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    scaler=StandardScaler().fit(raw[train].reshape(-1,raw.shape[-1]))
    scaled=scaler.transform(raw.reshape(-1,raw.shape[-1])).reshape(raw.shape).astype(np.float32)
    return scaled,scaler


def evaluate(triples: pd.DataFrame, model: str, seed: int, logits: np.ndarray,
             weights: list[float]) -> list[dict]:
    probability=1/(1+np.exp(-logits));rows=[]
    for split in ['train','val','test']:
        sm=triples.split.eq(split).to_numpy()
        macro=float(masked_prognostic_loss(torch.from_numpy(logits[sm]),
            torch.from_numpy(triples.loc[sm,TASKS].to_numpy(np.float32)),torch.tensor(weights)))
        for j,task in enumerate(TASKS):
            mask=sm & triples[task].notna().to_numpy();y=triples.loc[mask,task].to_numpy(float)
            rows.append(dict(model=model,seed=seed,split=split,task=task,n=len(y),
                positives=int(y.sum()),macro_weighted_bce=macro,**probe_metrics(y,probability[mask,j])))
    return rows


def bootstrap(triples: pd.DataFrame, exports: dict[str,list[Path]], cfg: dict, out: Path) -> None:
    ensemble={name:np.mean([np.load(path,allow_pickle=False)['probability'] for path in paths],axis=0)
              for name,paths in exports.items()}
    comparisons=[('token_gated_history','token_gated_control'),
                 ('token_attention_t1','mean_linear'),
                 ('token_gated_history','token_attention_t1'),
                 ('oracle_token_attention','oracle_mean_linear')]
    rows=[];test=triples.split.eq('test').to_numpy()
    for candidate,comparator in comparisons:
        if candidate not in ensemble or comparator not in ensemble: continue
        for j,task in enumerate(TASKS):
            mask=test & triples[task].notna().to_numpy();y=triples.loc[mask,task].to_numpy(int)
            a,b=ensemble[candidate][mask,j],ensemble[comparator][mask,j]
            rng=np.random.default_rng(20260922);draws=[]
            for _ in range(cfg['bootstrap_samples']):
                idx=rng.integers(len(y),size=len(y))
                if len(np.unique(y[idx]))==2:
                    draws.append([roc_auc_score(y[idx],a[idx])-roc_auc_score(y[idx],b[idx]),
                                  average_precision_score(y[idx],a[idx])-average_precision_score(y[idx],b[idx])])
            draws=np.asarray(draws)
            for k,(metric,fn) in enumerate([('auroc',roc_auc_score),('auprc',average_precision_score)]):
                low,high=np.quantile(draws[:,k],[.025,.975])
                rows.append(dict(candidate=candidate,comparator=comparator,task=task,metric=metric,
                    gain=fn(y,a)-fn(y,b),ci_low=low,ci_high=high,valid_draws=len(draws),
                    requested_draws=cfg['bootstrap_samples']))
    pd.DataFrame(rows).to_csv(out/'paired_patient_bootstrap.csv',index=False)


def report(out: Path) -> None:
    d=pd.read_csv(out/'metrics.csv')
    d.groupby(['model','split','task'])[['auroc','auprc','mcc','f1','recall','specificity',
        'macro_weighted_bce']].agg(['mean','std','count']).to_csv(out/'summary.csv')
    sections=[]
    for split in ['val','test']:
        table=d[d.split.eq(split)].pivot_table(index='model',columns='task',values=['auroc','auprc','mcc','f1'])
        sections.append(f'## {split}\n\n```text\n{table.round(4).to_string()}\n```')
    boot=pd.read_csv(out/'paired_patient_bootstrap.csv')
    text=('# Spatial observed COLIPRI progression: completed results\n\n'
          'Frozen native [64,864] T0/T1 tokens; post-T2 cancer labels; no Transformer and no '
          'future-state prediction. Primary comparison: token_gated_history versus the exact '
          'capacity-matched token_gated_control with patient-specific deltas zeroed.\n\n'
          'Five training seeds are primary. Bootstrap is a secondary paired test-set analysis '
          'of seed-ensemble probabilities. This enriched, previously inspected development cohort '
          'is not external validation; weighted-BCE probabilities are not calibrated risks.\n\n'
          +'\n\n'.join(sections)+'\n\n## Paired patient bootstrap\n\n```text\n'
          +boot.round(4).to_string(index=False)+'\n```\n')
    (out/'RESULTS.md').write_text(text)


def pooling_audit(tokens: np.ndarray, embeddings: np.ndarray) -> dict:
    mse=[];cos=[];maximum=[]
    for start in range(0,len(tokens),128):
        pooled=np.asarray(tokens[start:start+128]).mean(1);target=embeddings[start:start+128]
        mse.extend(np.mean((pooled-target)**2,axis=1));maximum.extend(np.max(np.abs(pooled-target),axis=1))
        cos.extend(np.sum(pooled*target,axis=1)/(np.linalg.norm(pooled,axis=1)*np.linalg.norm(target,axis=1)+1e-12))
    return dict(mse_mean=float(np.mean(mse)),mse_max=float(np.max(mse)),cosine_mean=float(np.mean(cos)),
                cosine_min=float(np.min(cos)),max_abs=float(np.max(maximum)))


def main() -> None:
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/model/colipri_spatial_observed_progression.yaml')
    parser.add_argument('--output-dir',type=Path);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();cfg=yaml.safe_load(args.config.read_text())
    if args.smoke:
        cfg.update(output_dir='outputs/colipri_spatial_observed_progression_smoke',seeds=[0],
            models=['mean_linear','token_gated_control','token_gated_history'],epochs=2,patience=2,
            bootstrap_samples=20,wandb_group=cfg['wandb_group']+'-smoke')
    if args.output_dir:cfg['output_dir']=str(args.output_dir)
    out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError('GPU required; no silent CPU fallback')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    paths=argparse.Namespace(**{key:Path(cfg[key]) for key in
        ['embedding_cache','prognostic_labels','diagnostic_manifest','patient_splits']})
    sources=[Path(__file__),ROOT/'src/lung_jepa/models/spatial_progression.py',
        ROOT/'src/lung_jepa/models/prognostic_adapter.py',args.config]
    input_paths=[Path(cfg['token_cache']),Path(cfg['pooling_report']),Path(cfg['reference_pairs']),
        Path(cfg['reference_triples']),paths.embedding_cache/'embeddings.npy',
        paths.embedding_cache/'embedding_metadata.csv',paths.prognostic_labels,paths.patient_splits]
    fingerprint={str(path):digest(path) for path in sources+input_paths}
    for filename,value in [('config.json',cfg),('fingerprint.json',fingerprint)]:
        if (out/filename).exists():assert json.loads((out/filename).read_text())==value,f'Resume mismatch: {filename}'
        else:save_json(out/filename,value)
    if (out/'COMPLETE.json').exists():print('Already complete',flush=True);return
    pairs,embeddings=build_pairs(paths);reference=pd.read_csv(cfg['reference_pairs'],dtype={'patient_id':str})
    pd.testing.assert_frame_equal(pairs[['patient_id','pair_type','idx_t','idx_t1','split']+TASKS],
                                  reference[['patient_id','pair_type','idx_t','idx_t1','split']+TASKS])
    triples=build_triples(pairs);expected=pd.read_csv(cfg['reference_triples'],dtype={'patient_id':str})
    pd.testing.assert_frame_equal(triples,expected);triples.to_csv(out/'triple_manifest.csv',index=False);cohort_audit(triples,out)
    tokens=np.load(cfg['token_cache'],mmap_mode='r')
    if tokens.shape!=(len(embeddings),64,864):raise ValueError(f'Unexpected token cache: {tokens.shape}')
    audit=pooling_audit(tokens,embeddings);save_json(out/'pooling_audit.json',audit)
    if audit['mse_max']>1e-12 or audit['cosine_min']<.99999:raise ValueError(f'Pooling mismatch: {audit}')
    train=triples.split.eq('train').to_numpy();val=triples.split.eq('val').to_numpy()
    raw_state=np.asarray(tokens[triples.idx_t1.to_numpy()],np.float32);state,state_scaler=scale_tokens(raw_state,train);del raw_state
    raw_delta=np.asarray(tokens[triples.idx_t1.to_numpy()]-tokens[triples.idx_t0.to_numpy()],np.float32)
    delta,delta_scaler=scale_tokens(raw_delta,train);del raw_delta
    raw_oracle=np.asarray(tokens[triples.idx_t2.to_numpy()],np.float32);oracle,oracle_scaler=scale_tokens(raw_oracle,train);del raw_oracle
    time_scaler=StandardScaler().fit(triples.loc[train,['dt01']]);time=time_scaler.transform(triples[['dt01']]).astype(np.float32)
    labels=triples[TASKS].to_numpy(np.float32);positive=np.nansum(labels[train],axis=0);negative=np.sum(labels[train]==0,axis=0)
    weights=(negative/positive).tolist()
    np.savez(out/'scalers.npz',state_mean=state_scaler.mean_,state_scale=state_scaler.scale_,
        delta_mean=delta_scaler.mean_,delta_scale=delta_scaler.scale_,oracle_mean=oracle_scaler.mean_,
        oracle_scale=oracle_scaler.scale_,time_mean=time_scaler.mean_,time_scale=time_scaler.scale_,pos_weights=weights)
    save_json(out/'provenance.json',dict(pid=os.getpid(),gpu=torch.cuda.get_device_name(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),patients=len(triples),
        split_counts=triples.split.value_counts().to_dict(),pos_weights=weights,
        model_parameters={name:sum(p.numel() for p in SpatialProgressionPrognosticator(name,
            hidden_dim=cfg['hidden_dim'],time_dim=cfg['time_dim']).parameters()) for name in cfg['models']}))
    ids=np.asarray(triples.patient_id.astype(str).tolist(),dtype='U');rows=[];exports={name:[] for name in cfg['models']}
    for model_name in cfg['models']:
        model_state=oracle if model_name.startswith('oracle_') else state
        for seed in cfg['seeds']:
            seed_all(seed);work=out/'runs'/f'{model_name}__seed{seed}';status(out,stage='training',model=model_name,seed=seed)
            model=SpatialProgressionPrognosticator(model_name,hidden_dim=cfg['hidden_dim'],time_dim=cfg['time_dim'],
                dropout=cfg['dropout'],pos_weights=weights,lr=cfg['lr'])
            if args.smoke:
                with torch.no_grad():result=model(torch.from_numpy(model_state[:2]),torch.from_numpy(delta[:2]),torch.from_numpy(time[:2]))
                print(dict(model=model_name,state=list(model_state[:2].shape),delta=list(delta[:2].shape),
                    time=list(time[:2].shape),logits=list(result[0].shape),attention=None if result[1] is None else list(result[1].shape),
                    gate=None if result[2] is None else list(result[2].shape)),flush=True)
            arrays=[model_state,delta,time,labels]
            datasets={split:TensorDataset(*(torch.from_numpy(array[mask]) for array in arrays))
                      for split,mask in [('train',train),('val',val)]}
            model=fit_stage(model,datasets['train'],datasets['val'],cfg,work,f'{model_name}/seed{seed}',
                            cfg['epochs'],'val/prognosis')
            model.eval().requires_grad_(False).cuda();logits_parts=[];attention_parts=[];gate_parts=[]
            with torch.inference_mode():
                for start in range(0,len(triples),64):
                    result=model(torch.tensor(model_state[start:start+64],device='cuda'),
                        torch.tensor(delta[start:start+64],device='cuda'),torch.tensor(time[start:start+64],device='cuda'))
                    logits_parts.append(result[0].cpu().numpy())
                    if result[1] is not None:attention_parts.append(result[1].cpu().numpy())
                    if result[2] is not None:gate_parts.append(result[2].cpu().numpy())
            logits=np.concatenate(logits_parts);probability=1/(1+np.exp(-logits))
            kwargs=dict(logits=logits,probability=probability,patient_ids=ids)
            if attention_parts:kwargs['attention']=np.concatenate(attention_parts)
            if gate_parts:kwargs['gate']=np.concatenate(gate_parts)
            export=work/'predictions.npz';np.savez(export,**kwargs);exports[model_name].append(export)
            rows.extend(evaluate(triples,model_name,seed,logits,weights));pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);del model
    status(out,stage='bootstrap',training_runs=sum(map(len,exports.values())))
    bootstrap(triples,exports,cfg,out);report(out)
    run=wandb.init(project=cfg['wandb_project'],group=cfg['wandb_group'],name='summary-'+out.name,dir=str(out),config=cfg)
    run.log({'metrics':wandb.Table(dataframe=pd.DataFrame(rows)),
             'bootstrap':wandb.Table(dataframe=pd.read_csv(out/'paired_patient_bootstrap.csv'))})
    run.finish();save_json(out/'COMPLETE.json',dict(training_runs=sum(map(len,exports.values())),
        bootstrap_samples=cfg['bootstrap_samples']))
    status(out,stage='complete',training_runs=sum(map(len,exports.values())))


if __name__=='__main__':main()
