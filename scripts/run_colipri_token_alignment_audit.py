#!/usr/bin/env python3
"""Audit longitudinal correspondence of the frozen 4x4x4 COLIPRI token grid."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
import yaml
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler

from run_colipri_latent_world_model import build_pairs
from run_colipri_preserving_adapter import ROOT, digest, save_json, status

COORDS=np.stack(np.unravel_index(np.arange(64),(4,4,4)),axis=1)
CALIBRATION=(COORDS.sum(1)%2)==0
SHIFTS=np.asarray([(a,b,c) for a in [-1,0,1] for b in [-1,0,1] for c in [-1,0,1]])


def shift_mapping(shift: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    target=COORDS+shift
    valid=np.all((target>=0)&(target<4),axis=1)
    indices=np.ravel_multi_index(target[valid].T,(4,4,4))
    return valid,indices


def analyze_similarity(similarity: np.ndarray) -> dict[str,float]:
    """Analyze one [64,64] source-to-future cosine matrix."""
    diagonal=np.diag(similarity);best=similarity.argmax(1);best_values=similarity[np.arange(64),best]
    sorted_values=np.partition(similarity,-2,axis=1)[:,-2:]
    displacement=COORDS[best]-COORDS
    local_mask=np.max(np.abs(COORDS[:,None,:]-COORDS[None,:,:]),axis=2)<=1
    local=np.where(local_mask,similarity,-np.inf).max(1)
    reverse=similarity.argmax(0)
    mutual=reverse[best]==np.arange(64)
    source,target=linear_sum_assignment(-similarity)
    hungarian_displacement=COORDS[target]-COORDS[source]

    # Select one grid translation on parity-even tokens and test on held-out parity-odd tokens.
    calibration_scores=[]
    for shift in SHIFTS:
        valid,mapped=shift_mapping(shift);source_indices=np.flatnonzero(valid)
        keep=CALIBRATION[source_indices]
        calibration_scores.append(similarity[source_indices[keep],mapped[keep]].mean())
    selected=SHIFTS[int(np.argmax(calibration_scores))]
    valid,mapped=shift_mapping(selected);source_indices=np.flatnonzero(valid)
    keep=~CALIBRATION[source_indices];source_eval=source_indices[keep];target_eval=mapped[keep]
    translated=similarity[source_eval,target_eval];identity=similarity[source_eval,source_eval]
    return {
        'diagonal_cosine':float(diagonal.mean()),'diagonal_cosine_min':float(diagonal.min()),
        'global_best_cosine':float(best_values.mean()),'global_gain':float((best_values-diagonal).mean()),
        'local_best_cosine':float(local.mean()),'local_gain':float((local-diagonal).mean()),
        'same_token_top1_fraction':float(np.mean(best==np.arange(64))),
        'mutual_nn_fraction':float(mutual.mean()),
        'top1_margin':float((sorted_values[:,-1]-sorted_values[:,-2]).mean()),
        'top1_displacement_l1':float(np.abs(displacement).sum(1).mean()),
        'top1_displacement_l2':float(np.linalg.norm(displacement,axis=1).mean()),
        'top1_abs_d0':float(np.abs(displacement[:,0]).mean()),
        'top1_abs_d1':float(np.abs(displacement[:,1]).mean()),
        'top1_abs_d2':float(np.abs(displacement[:,2]).mean()),
        'hungarian_cosine':float(similarity[source,target].mean()),
        'hungarian_fixed_fraction':float(np.mean(target==source)),
        'hungarian_displacement_l1':float(np.abs(hungarian_displacement).sum(1).mean()),
        'translation_d0':int(selected[0]),'translation_d1':int(selected[1]),'translation_d2':int(selected[2]),
        'translation_is_zero':float(np.all(selected==0)),
        'translation_eval_cosine':float(translated.mean()),
        'translation_identity_cosine':float(identity.mean()),
        'translation_eval_gain':float((translated-identity).mean()),
        'translation_eval_tokens':int(len(translated)),
    }


def random_derangements(pairs: pd.DataFrame, repeats: int, seed: int) -> list[np.ndarray]:
    rng=np.random.default_rng(seed);results=[]
    groups=[indices.to_numpy() for _,indices in pairs.groupby(['split','pair_type']).groups.items()]
    for _ in range(repeats):
        mapping=np.empty(len(pairs),dtype=int)
        for group in groups:
            order=rng.permutation(group);shift=int(rng.integers(1,len(order)))
            mapping[order]=np.roll(order,shift)
        if np.any(mapping==np.arange(len(pairs))):raise AssertionError('Random control is not a derangement')
        results.append(mapping)
    return results


def transformed_similarity(source: np.ndarray,target: np.ndarray,mean: torch.Tensor|None,
                           scale: torch.Tensor|None,device: torch.device) -> np.ndarray:
    a=torch.tensor(source,device=device);b=torch.tensor(target,device=device)
    if mean is not None:a=(a-mean)/scale;b=(b-mean)/scale
    return torch.bmm(F.normalize(a,dim=-1),F.normalize(b,dim=-1).transpose(1,2)).cpu().numpy()


def summarize(frame: pd.DataFrame,out: Path) -> pd.DataFrame:
    metrics=['diagonal_cosine','local_best_cosine','local_gain','global_best_cosine','global_gain',
        'same_token_top1_fraction','mutual_nn_fraction','top1_margin','top1_displacement_l1',
        'hungarian_cosine','hungarian_fixed_fraction','hungarian_displacement_l1',
        'translation_is_zero','translation_eval_gain','random_diagonal_cosine','random_global_best_cosine',
        'true_minus_random_diagonal']
    summary=frame.groupby(['space','split','pair_type'])[metrics].agg(['mean','std','median'])
    summary.to_csv(out/'alignment_summary.csv')
    counts=frame.groupby(['space','pair_type','translation_d0','translation_d1','translation_d2']).size().rename('count').reset_index()
    counts.to_csv(out/'translation_counts.csv',index=False)
    return summary


def figures(frame: pd.DataFrame,out: Path) -> None:
    plot=frame[frame.split.eq('test')]
    labels=[];diag=[];local=[];glob=[];random=[]
    for (space,pair_type),group in plot.groupby(['space','pair_type']):
        labels.append(f'{space}\n{pair_type}');diag.append(group.diagonal_cosine.mean())
        local.append(group.local_best_cosine.mean());glob.append(group.global_best_cosine.mean())
        random.append(group.random_diagonal_cosine.mean())
    x=np.arange(len(labels));width=.2
    fig,ax=plt.subplots(figsize=(10,5));ax.bar(x-1.5*width,random,width,label='random-patient diagonal')
    ax.bar(x-.5*width,diag,width,label='true diagonal');ax.bar(x+.5*width,local,width,label='true local best')
    ax.bar(x+1.5*width,glob,width,label='true global best');ax.set_xticks(x,labels);ax.set_ylabel('Cosine similarity')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'cosine_alignment.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for i,space in enumerate(['raw','standardized']):
        sub=plot[plot.space.eq(space)]
        data=[sub[sub.pair_type.eq(pair)].top1_displacement_l1 for pair in ['T0_T1','T1_T2']]
        axes[i].boxplot(data,tick_labels=['T0→T1','T1→T2'],showfliers=False)
        axes[i].set_title(space);axes[i].set_ylabel('Top-1 grid L1 displacement')
    fig.tight_layout();fig.savefig(out/'top1_displacement.png',dpi=180);plt.close(fig)


def write_report(frame: pd.DataFrame,out: Path) -> None:
    test=frame[frame.split.eq('test')]
    table=test.groupby(['space','pair_type'])[['diagonal_cosine','local_gain','global_gain',
        'same_token_top1_fraction','mutual_nn_fraction','top1_displacement_l1','hungarian_fixed_fraction',
        'translation_is_zero','translation_eval_gain','random_diagonal_cosine','true_minus_random_diagonal']].mean()
    text=('# COLIPRI longitudinal token-alignment audit\n\n'
          'No outcome labels or prognostic model are used. Metrics compare the native 4x4x4 token grid '
          'between observed scans. `raw` uses native cosine; `standardized` uses feature-wise statistics '
          'fit on training-patient scans only. Random controls pair each source with another patient in '
          'the same split and transition. Grid axes d0/d1/d2 are not assigned anatomical names.\n\n'
          'Local matching allows Chebyshev radius 1. Global matching allows all tokens. Hungarian matching '
          'is one-to-one. Patient-specific translation is selected on parity-even grid tokens and evaluated '
          'on held-out parity-odd tokens, so its gain is not measured on the selection tokens.\n\n'
          '## Test summary\n\n```text\n'+table.round(4).to_string()+'\n```\n\n'
          'See `alignment_per_pair.csv`, `alignment_summary.csv`, `translation_counts.csv` and figures. '
          'High nearest-neighbor gain alone does not prove anatomical registration: feature matching may '
          'pair semantically similar but anatomically different regions.\n')
    (out/'RESULTS.md').write_text(text)


def main() -> None:
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--config',type=Path,
        default=ROOT/'configs/model/colipri_token_alignment_audit.yaml');parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args();cfg=yaml.safe_load(args.config.read_text())
    if args.output_dir:cfg['output_dir']=str(args.output_dir)
    out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError('GPU required; no silent CPU fallback')
    device=torch.device('cuda');torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    paths=argparse.Namespace(**{key:Path(cfg[key]) for key in
        ['embedding_cache','prognostic_labels','diagnostic_manifest','patient_splits']})
    sources=[Path(__file__),args.config];inputs=[Path(cfg['token_cache']),Path(cfg['reference_pairs']),
        paths.embedding_cache/'embeddings.npy',paths.embedding_cache/'embedding_metadata.csv',paths.patient_splits]
    fingerprint={str(path):digest(path) for path in sources+inputs}
    for filename,value in [('config.json',cfg),('fingerprint.json',fingerprint)]:
        if (out/filename).exists():assert json.loads((out/filename).read_text())==value,f'Resume mismatch: {filename}'
        else:save_json(out/filename,value)
    if (out/'COMPLETE.json').exists():print('Already complete',flush=True);return
    pairs,embeddings=build_pairs(paths);reference=pd.read_csv(cfg['reference_pairs'],dtype={'patient_id':str})
    pd.testing.assert_frame_equal(pairs[['patient_id','pair_type','idx_t','idx_t1','split']],
                                  reference[['patient_id','pair_type','idx_t','idx_t1','split']])
    pairs.to_csv(out/'pairs_manifest.csv',index=False);tokens=np.load(cfg['token_cache'],mmap_mode='r')
    if tokens.shape!=(len(embeddings),64,864):raise ValueError(tokens.shape)
    metadata=pd.read_csv(paths.embedding_cache/'embedding_metadata.csv',dtype=str)
    split_table=pd.read_csv(paths.patient_splits,dtype=str);metadata=metadata.merge(split_table,on='patient_id',validate='many_to_one')
    metadata['embedding_index']=metadata.embedding_index.astype(int);metadata=metadata.sort_values('embedding_index')
    train_indices=metadata.loc[metadata.split.eq('train'),'embedding_index'].to_numpy()
    scaler=StandardScaler().fit(np.asarray(tokens[train_indices]).reshape(-1,864))
    np.savez(out/'feature_scaler.npz',mean=scaler.mean_,scale=scaler.scale_,train_scan_indices=train_indices)
    mean=torch.tensor(scaler.mean_,device=device,dtype=torch.float32);scale=torch.tensor(scaler.scale_,device=device,dtype=torch.float32)
    random_maps=random_derangements(pairs,cfg['random_controls'],cfg['random_seed'])
    random_diag={space:np.zeros(len(pairs)) for space in ['raw','standardized']}
    random_global={space:np.zeros(len(pairs)) for space in ['raw','standardized']}
    rows=[];batch=cfg['batch_size']
    status(out,stage='true_pairs',pairs=len(pairs))
    with torch.inference_mode():
        for start in range(0,len(pairs),batch):
            index=np.arange(start,min(start+batch,len(pairs)))
            source=np.asarray(tokens[pairs.idx_t.to_numpy()[index]],np.float32)
            target=np.asarray(tokens[pairs.idx_t1.to_numpy()[index]],np.float32)
            for space,mu,sigma in [('raw',None,None),('standardized',mean,scale)]:
                sims=transformed_similarity(source,target,mu,sigma,device)
                for local_i,pair_i in enumerate(index):
                    rows.append(dict(pair_index=int(pair_i),patient_id=str(pairs.iloc[pair_i].patient_id),
                        split=pairs.iloc[pair_i].split,pair_type=pairs.iloc[pair_i].pair_type,space=space,
                        **analyze_similarity(sims[local_i])))
            if (start//batch)%10==0:status(out,stage='true_pairs',completed=min(start+batch,len(pairs)),pairs=len(pairs))
        status(out,stage='random_controls',repeats=cfg['random_controls'])
        for mapping in random_maps:
            for start in range(0,len(pairs),batch):
                index=np.arange(start,min(start+batch,len(pairs)))
                source=np.asarray(tokens[pairs.idx_t.to_numpy()[index]],np.float32)
                target_pair=mapping[index]
                target=np.asarray(tokens[pairs.idx_t1.to_numpy()[target_pair]],np.float32)
                for space,mu,sigma in [('raw',None,None),('standardized',mean,scale)]:
                    sims=transformed_similarity(source,target,mu,sigma,device)
                    random_diag[space][index]+=np.diagonal(sims,axis1=1,axis2=2).mean(1)/len(random_maps)
                    random_global[space][index]+=sims.max(2).mean(1)/len(random_maps)
    frame=pd.DataFrame(rows)
    for space in ['raw','standardized']:
        mask=frame.space.eq(space);indices=frame.loc[mask,'pair_index'].to_numpy()
        frame.loc[mask,'random_diagonal_cosine']=random_diag[space][indices]
        frame.loc[mask,'random_global_best_cosine']=random_global[space][indices]
    frame['true_minus_random_diagonal']=frame.diagonal_cosine-frame.random_diagonal_cosine
    frame.to_csv(out/'alignment_per_pair.csv',index=False);summarize(frame,out);figures(frame,out);write_report(frame,out)
    save_json(out/'provenance.json',dict(pid=os.getpid(),gpu=torch.cuda.get_device_name(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),pairs=len(pairs),spaces=['raw','standardized'],
        random_controls=cfg['random_controls']))
    run=wandb.init(project=cfg['wandb_project'],group=cfg['wandb_group'],name='token-alignment-audit',
        dir=str(out),config=cfg)
    test_summary=frame[frame.split.eq('test')].groupby(['space','pair_type']).mean(numeric_only=True).reset_index()
    run.log({'test_summary':wandb.Table(dataframe=test_summary),
             'cosine_alignment':wandb.Image(str(out/'cosine_alignment.png')),
             'top1_displacement':wandb.Image(str(out/'top1_displacement.png'))})
    run.finish();save_json(out/'COMPLETE.json',dict(pairs=len(pairs),rows=len(frame),random_controls=cfg['random_controls']))
    status(out,stage='complete',pairs=len(pairs),rows=len(frame))


if __name__=='__main__':main()
