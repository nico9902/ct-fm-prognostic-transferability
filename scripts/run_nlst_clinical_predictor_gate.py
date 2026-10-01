#!/usr/bin/env python3
"""Clinical-loss gate on the first gate's EXACT cohort, targets, and splits."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import copy
import json
import logging
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping,ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger,WandbLogger
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset,DataLoader
from lung_jepa.models.clinical_future import OracleRiskHead,ClinicalFuturePredictor
from lung_jepa.models.disease_state import GateModule
from run_colipri_latent_world_model_ablation import DeltaPredictor
from run_nlst_gate import TASKS,METRICS,fit,weights,metrics,validate_splits,digest,save_json,markdown,seed_all


def inference(model,*arrays,logits=False):
    model.eval().requires_grad_(False).cuda()
    values=[]
    with torch.inference_mode():
        for start in range(0,len(arrays[0]),128):
            args=[torch.tensor(a[start:start+128],dtype=torch.float32,device='cuda') for a in arrays]
            values.append(model(*args).cpu().numpy())
    model.cpu()
    return np.concatenate(values)


def load_data(cfg,out):
    old=Path(cfg['previous_gate'])
    pairs=pd.read_csv(old/'pairs_manifest.csv',dtype={'patient_id':str})
    split=pd.read_csv(old/'patient_splits.csv',dtype={'patient_id':str})
    validate_splits(split)
    assert pairs.groupby('patient_id').split.nunique().eq(1).all()
    check=pairs[['patient_id','split']].drop_duplicates().merge(split[['patient_id','split']],on='patient_id',suffixes=('_pair','_patient'),validate='one_to_one')
    assert check.split_pair.eq(check.split_patient).all()
    em=pd.read_csv(Path(cfg['embedding_cache'])/'embedding_metadata.csv',dtype={'patient_id':str})
    prior_em=pd.read_csv(old/'feature_metadata.csv',dtype={'patient_id':str})
    pd.testing.assert_frame_equal(em[['patient_id','series_uid','screening_round','embedding_index']],
        prior_em[['patient_id','series_uid','screening_round','embedding_index']])
    source_cfg=json.loads((old/'configs/resolved.json').read_text())
    source_fp=json.loads((old/'fingerprints.json').read_text())
    cache=Path(cfg['embedding_cache'])/'embeddings.npy'
    assert digest(cache)==source_fp[str(cache)],'Feature cache changed since the first gate'
    assert list(cfg['seeds'])==source_cfg['seeds'] or cfg.get('smoke',False)
    z=np.load(cache).astype(np.float32)
    for task in TASKS:assert pairs[task].dropna().isin([0,1]).all()
    assert pairs.dt_days.gt(0).all()
    pairs.to_csv(out/'pairs_manifest.csv',index=False)
    split.to_csv(out/'patient_splits.csv',index=False)
    current=z[pairs.idx_t.to_numpy()];future=z[pairs.idx_t1.to_numpy()]
    y=pairs[TASKS].to_numpy(np.float32);s=pairs.split.to_numpy();train=s=='train'
    delta_scaler=StandardScaler().fit((future-current)[train])
    future_scaler=StandardScaler().fit(future[train])
    # Configured forecast horizon is a model query, NOT the realized future scan date.
    planned=np.full((len(pairs),1),cfg['forecast_interval_days'],np.float32)
    time_scaler=StandardScaler().fit(planned[train])
    t=time_scaler.transform(planned).astype(np.float32)
    np.savez(out/'configs/scalers.npz',delta_mean=delta_scaler.mean_,delta_scale=delta_scaler.scale_,
        future_mean=future_scaler.mean_,future_scale=future_scaler.scale_,
        time_mean=time_scaler.mean_,time_scale=time_scaler.scale_,pos_weights=weights(y[train]))
    save_json(out/'cohort_provenance.json',dict(patients=em.patient_id.nunique(),scans=len(z),pairs=len(pairs),
        split_counts=split.split.value_counts().to_dict(),prior_gate=str(old),
        cohort_unchanged=True,targets=TASKS,primary=cfg['primary_task'],
        feature_sha256=digest(cache),prior_split_sha256=digest(old/'patient_splits.csv'),
        interval_policy=cfg['interval_policy'],forecast_interval=cfg['forecast_interval_days'],
        test_usage='evaluation only, after oracle and loss-weight selections lock',
        target_definition='1/2/3y after the actual t+1 landmark; prior cancer excluded; short follow-up masked',
        input_policy='current embedding and configured 365-day query only; realized dt is legacy-only'))
    return pairs,current,future,y,s,t,delta_scaler,future_scaler


def train_oracle(cfg,out,future,y,split,future_scaler):
    train=split=='train';val=split=='val';scaled=future_scaler.transform(future).astype(np.float32)
    families={};rows=[]
    for kind in cfg['oracle_heads']:
        teacher=OracleRiskHead(future_scaler.mean_,future_scaler.scale_,kind,len(cfg['seeds']),cfg['hidden'])
        for i,seed in enumerate(cfg['seeds']):
            print('ORACLE',kind,seed,flush=True)
            seed_all(seed)
            model=GateModule(future.shape[1],3,weights(y[train]),kind=kind,hidden=cfg['hidden'],lr=cfg['lr'])
            model=fit(model,scaled,y,split,cfg,out/'checkpoints'/f'oracle_{kind}_seed{seed}',
                      f'oracle/{kind}/seed{seed}',seed,cfg['oracle_epochs'])
            teacher.heads[i].load_state_dict(model.head.state_dict())
            logits=inference(model,scaled)
            for j,task in enumerate(TASKS):
                mask=val & np.isfinite(y[:,j])
                rows.append(dict(kind=kind,seed=seed,task=task,**metrics(y[mask,j],expit(logits[mask,j]))))
        families[kind]=teacher
    choices=[]
    for kind,teacher in families.items():
        logits=inference(teacher,future[val])
        valid=np.isfinite(y[val,1])
        choices.append(dict(kind=kind,validation_2y_auroc=metrics(y[val,1][valid],expit(logits[valid,1]))['auroc']))
    chosen=max(choices,key=lambda r:(r['validation_2y_auroc'],r['kind']=='linear'))['kind']
    pd.DataFrame(rows).to_csv(out/'oracle_validation_per_seed.csv',index=False)
    pd.DataFrame(choices).to_csv(out/'oracle_family_selection.csv',index=False)
    teacher=families[chosen]
    save_json(out/'oracle_selection.json',dict(chosen_kind=chosen,selection='validation-only 2y ensemble AUROC',
        seeds=cfg['seeds'],frozen=True,checkpoint='checkpoints/oracle_risk_head.pt',
        ensemble='arithmetic mean of logits; no probability calibration',validation_scores=choices))
    torch.save(dict(kind=chosen,heads=len(cfg['seeds']),hidden=cfg['hidden'],
        mean=future_scaler.mean_.tolist(),scale=future_scaler.scale_.tolist(),state_dict=teacher.state_dict()),
        out/'checkpoints/oracle_risk_head.pt')
    return teacher,chosen,families


def old_predictions(cfg,out,pairs,current):
    base=Path(cfg['old_predictor_dir']);reference=pd.read_csv(base/'pairs_manifest.csv',dtype={'patient_id':str})
    pd.testing.assert_frame_equal(pairs,reference)
    planned=[];legacy=[];audit=[]
    for seed in cfg['old_predictor_seeds']:
        ck=base/'checkpoints'/f"{cfg['old_predictor_config']}__seed{seed}.pt"
        saved=torch.load(ck,map_location='cpu',weights_only=False)
        model=DeltaPredictor(saved['config']['time_mode'])
        model.load_state_dict(saved['model_state_dict'])
        for name,days in [('legacy',pairs.dt_days.to_numpy()[:,None]),
                          ('planned',np.full((len(pairs),1),cfg['forecast_interval_days']))]:
            time=((days-saved['time_mean'])/saved['time_scale']).astype(np.float32)
            delta=inference(model,current,time)*saved['delta_scale']+saved['delta_mean']
            pred=current+delta
            if name=='legacy':
                cached=np.load(base/'predicted_embeddings'/f"{cfg['old_predictor_config']}__seed{seed}.npy")
                error=float(np.max(np.abs(pred-cached)))
                assert error<1e-5,f'Old predictor reconstruction mismatch {error}'
                audit.append(dict(seed=seed,max_abs_cache_error=error))
                legacy.append(cached.astype(np.float32))
            else:planned.append(pred.astype(np.float32))
    pd.DataFrame(audit).to_csv(out/'old_checkpoint_reproduction.csv',index=False)
    for name,values in [('old',planned),('old_legacy_realized_interval',legacy)]:
        for seed,value in zip(cfg['old_predictor_seeds'],values):
            np.save(out/'predictions'/f'{name}_original_seed{seed}.npy',value)
    return dict(old=np.mean(planned,axis=0),old_legacy_realized_interval=np.mean(legacy,axis=0))


def predictor_fit(model,arrays,split,cfg,work,seed,name):
    work.mkdir(parents=True,exist_ok=True)
    if (work/'done.json').exists():
        return ClinicalFuturePredictor.load_from_checkpoint(work/'best.ckpt',map_location='cpu')
    seed_all(seed)
    logger=WandbLogger(project=cfg['wandb_project'],group=cfg['wandb_group'],name=name,
        save_dir=str(work),offline=True,log_model=False) if cfg['wandb_mode']!='disabled' else None
    loggers=[CSVLogger(str(work),name='history')]
    if logger:
        logger.experiment.config.update(dict(config=cfg,stage=name,seed=seed,
            predictor_parameters=sum(p.numel() for p in model.network.parameters()),
            oracle_frozen=all(not p.requires_grad for p in model.oracle.parameters()),
            checkpoint_sha256=cfg['checkpoint_sha256'],task=cfg['primary_task']))
        loggers.append(logger)
    checkpoint=ModelCheckpoint(dirpath=work,filename='best',monitor='val/primary_auroc',mode='max',
        save_top_k=1,enable_version_counter=False)
    trainer=pl.Trainer(accelerator='gpu',devices=1,max_epochs=cfg['predictor_epochs'],logger=loggers,
        callbacks=[checkpoint,EarlyStopping(monitor='val/primary_auroc',mode='max',patience=cfg['patience'],min_delta=1e-5)],
        deterministic=True,enable_progress_bar=False,enable_model_summary=False,num_sanity_val_steps=0,log_every_n_steps=5)
    datasets={s:TensorDataset(*(torch.from_numpy(a[split==s]) for a in arrays)) for s in ['train','val']}
    trainer.fit(model,DataLoader(datasets['train'],batch_size=cfg['batch_size'],shuffle=True,
        generator=torch.Generator().manual_seed(seed)),DataLoader(datasets['val'],batch_size=cfg['batch_size']))
    best=ClinicalFuturePredictor.load_from_checkpoint(checkpoint.best_model_path,map_location='cpu')
    save_json(work/'done.json',dict(epochs_run=trainer.current_epoch,best_val_2y_auroc=float(checkpoint.best_model_score),
        predictor_parameters=sum(p.numel() for p in model.network.parameters()),
        oracle_parameters_frozen=sum(p.numel() for p in model.oracle.parameters()),
        checkpoint=checkpoint.best_model_path))
    if logger:logger.experiment.finish()
    model.cpu();return best


def train_predictors(cfg,out,current,future,y,split,time,ds,fs,teacher,kind):
    train=split=='train';val=split=='val';candidates=[]
    for mode in cfg['modes']:
        for variant in cfg['variants']:
            grid=[0.] if variant=='C1' else cfg['clinical_weights']
            for weight in grid:candidates.append((mode,variant,weight))
    if cfg['geometry_only_retrain_control']:candidates.append(('residual','geometry_only',0.))
    if cfg['risk_only_control']:candidates.append(('residual','risk_only',1.))
    choices=[]
    for mode,variant,weight in candidates:
        name=f'{mode}_{variant}_w{weight:g}'
        save_json(out/'configs'/f'{name}.json',dict(mode=mode,variant=variant,clinical_weight=weight,
            lambda_state=0. if variant=='risk_only' else cfg['lambda_state'],
            lambda_delta=0. if variant in ['geometry_only','risk_only'] else cfg['lambda_delta'],
            input='current raw embedding + configured prospective interval only',
            oracle='frozen validation-selected head ensemble',seeds=cfg['seeds'],config=cfg))
        for seed in cfg['seeds']:
            print('PREDICTOR',name,seed,flush=True)
            seed_all(seed)
            model=ClinicalFuturePredictor(ds.mean_.tolist(),ds.scale_.tolist(),fs.mean_.tolist(),fs.scale_.tolist(),
                fs.mean_.tolist(),fs.scale_.tolist(),oracle_kind=kind,oracle_heads=len(cfg['seeds']),
                oracle_hidden=cfg['hidden'],mode=mode,variant=variant,clinical_weight=weight,
                lambda_state=cfg['lambda_state'],lambda_delta=0. if variant=='geometry_only' else cfg['lambda_delta'],
                pos_weights=weights(y[train]),hidden=cfg['predictor_hidden'],lr=cfg['lr'])
            model.oracle.load_state_dict(teacher.state_dict())
            initial_teacher={k:v.detach().clone() for k,v in model.oracle.state_dict().items()}
            work=out/'checkpoints'/name/f'seed{seed}'
            model=predictor_fit(model,[current,time,future,y],split,cfg,work,seed,f'{name}/seed{seed}')
            for k,v in model.oracle.state_dict().items():
                torch.testing.assert_close(v,initial_teacher[k],rtol=0,atol=0)
            pred=inference(model,current,time)
            np.save(out/'predictions'/f'{name}_seed{seed}.npy',pred)
            p=expit(inference(teacher,pred[val]))
            valid=np.isfinite(y[val,1])
            choices.append(dict(mode=mode,variant=variant,clinical_weight=weight,seed=seed,name=name,
                validation_2y_auroc=metrics(y[val,1][valid],p[valid,1])['auroc']))
            pd.DataFrame(choices).to_csv(out/'validation_selection_per_seed.csv',index=False)
    table=pd.DataFrame(choices)
    means=table.groupby(['mode','variant','clinical_weight','name']).validation_2y_auroc.mean().reset_index()
    means.to_csv(out/'validation_selection.csv',index=False)
    selected={}
    for (mode,variant),group in means.groupby(['mode','variant']):
        row=group.sort_values(['validation_2y_auroc','clinical_weight'],ascending=[False,True]).iloc[0]
        selected[f'{mode}_{variant}']=row.to_dict()
    clinical=means[means.variant.isin(['C2','C3','C4'])].sort_values(
        ['validation_2y_auroc','clinical_weight'],ascending=[False,True]).iloc[0].to_dict()
    save_json(out/'selection_lock.json',dict(selected=selected,primary_clinical=clinical,
        selection='mean validation-only 2y frozen-oracle AUROC across seeds; smaller loss weight breaks ties',
        test_not_used=True,seeds=cfg['seeds'],clinical_grid=cfg['clinical_weights']))
    predictions={}
    for model,row in selected.items():
        predictions[model]=[np.load(out/'predictions'/f"{row['name']}_seed{seed}.npy") for seed in cfg['seeds']]
    return predictions


def export_probabilities(out,channel,model,head,seed,pairs,prob):
    table=pairs[['patient_id','pair_type','split']+TASKS].copy()
    for j,task in enumerate(TASKS):table[task+'_probability']=prob[:,j]
    table.to_csv(out/'predictions'/f'{channel}_{model}_{head}_seed{seed}.csv',index=False)


def evaluation_rows(channel,model,head,seed,pairs,y,prob):
    rows=[]
    for split in ['train','val','test']:
        for j,task in enumerate(TASKS):
            mask=pairs.split.eq(split).to_numpy() & np.isfinite(y[:,j])
            rows.append(dict(channel=channel,model=model,head=head,seed=seed,split=split,task=task,
                n=int(mask.sum()),patients=pairs.loc[mask,'patient_id'].nunique(),positives=int(y[mask,j].sum()),
                **metrics(y[mask,j],prob[mask,j])))
    return rows


def evaluate(cfg,out,pairs,current,future,y,split,predictions,old,teacher,families):
    train=split=='train';rows=[];geometry=[];alignment=[];gaps=[]
    oracle_logit=inference(teacher,future);current_logit=inference(teacher,current)
    true_prob=expit(oracle_logit);current_prob=expit(current_logit)
    reps={'current':[current]*len(cfg['seeds']),'identity':[current]*len(cfg['seeds']),
          'real_future':[future]*len(cfg['seeds']),**{k:[v]*len(cfg['seeds']) for k,v in old.items()},**predictions}
    for name,values in reps.items():
        for seed,pred in zip(cfg['seeds'],values):
            pred_logit=inference(teacher,pred);prob=expit(pred_logit)
            export_probabilities(out,'risk',name,'frozen_oracle',seed,pairs,prob)
            rows.extend(evaluation_rows('risk',name,'frozen_oracle',seed,pairs,y,prob))
            for sp in ['train','val','test']:
                sm=split==sp
                geom=representation_metrics(current[sm],future[sm],pred[sm])
                geometry.append(dict(model=name,seed=seed,split=sp,**geom))
                for j,task in enumerate(TASKS):
                    mask=sm & np.isfinite(y[:,j]);p=prob[mask,j];target=true_prob[mask,j];base=current_prob[mask,j]
                    tp=np.clip(target,1e-7,1-1e-7);pp=np.clip(p,1e-7,1-1e-7)
                    kl=tp*np.log(tp/pp)+(1-tp)*np.log((1-tp)/(1-pp))
                    actual_delta=target-base;estimated=p-base
                    valid=np.abs(actual_delta)>1e-8
                    alignment.append(dict(model=name,seed=seed,split=sp,task=task,n=int(mask.sum()),
                        probability_mae=float(np.mean(np.abs(p-target))),probability_mse=float(np.mean((p-target)**2)),
                        logit_mae=float(np.mean(np.abs(pred_logit[mask,j]-oracle_logit[mask,j]))),
                        bernoulli_kl=float(np.mean(kl)),risk_delta_direction_agreement=float(np.mean(
                            np.sign(estimated[valid])==np.sign(actual_delta[valid]))) if valid.any() else np.nan))
                    auc=metrics(y[mask,j],p)['auroc'];oracle_auc=metrics(y[mask,j],target)['auroc'];cur_auc=metrics(y[mask,j],base)['auroc']
                    denom=oracle_auc-cur_auc
                    gaps.append(dict(model=name,seed=seed,split=sp,task=task,AUC_predicted=auc,
                        AUC_current=cur_auc,AUC_oracle=oracle_auc,OracleGap=oracle_auc-auc,
                        GapClosure=(auc-cur_auc)/denom if denom>0 else np.nan,
                        gap_denominator=denom,denominator_positive=denom>0))
    # Record oracle training-seed variability separately from its frozen ensemble.
    oracle_rows=[]
    fs=np.load(out/'configs/scalers.npz')
    for kind,head in families.items():
        scaled=(future-fs['future_mean'])/fs['future_scale']
        for component,seed in zip(head.heads,cfg['seeds']):
            prob=expit(inference(component,scaled.astype(np.float32)))
            oracle_rows.extend(evaluation_rows('oracle_component',kind,kind,seed,pairs,y,prob))
    pd.DataFrame(oracle_rows).to_csv(out/'oracle_per_seed_metrics.csv',index=False)
    # Reuse EXACT first-gate predictions and fitted heads for unchanged baselines.
    for exp,name in [('G0','G0'),('G1','G1'),('G2','G2'),('G3','G3_old_legacy')]:
        for head in ['linear','mlp']:
            for seed in cfg['seeds']:
                source=Path(cfg['previous_gate'])/'runs'/f'{exp}_{head}_seed{seed}'/'predictions.csv'
                prior=pd.read_csv(source,dtype={'patient_id':str})
                pd.testing.assert_frame_equal(prior[['patient_id','pair_type','split']],pairs[['patient_id','pair_type','split']])
                prob=prior[[t+'_probability' for t in TASKS]].to_numpy()
                export_probabilities(out,'direct',name,head,seed,pairs,prob)
                rows.extend(evaluation_rows('direct',name,head,seed,pairs,y,prob))
    for name,values in {'G3_old':reps['old'],**predictions}.items():
        for seed,pred in zip(cfg['seeds'],values):
            x=np.concatenate([current,pred],axis=1)
            scaler=StandardScaler().fit(x[train]);scaled=scaler.transform(x).astype(np.float32)
            np.savez(out/'configs'/f'probe_{name}_seed{seed}_scaler.npz',mean=scaler.mean_,scale=scaler.scale_)
            for head in ['linear','mlp']:
                print('DOWNSTREAM',name,head,seed,flush=True)
                seed_all(seed)
                model=GateModule(x.shape[1],3,weights(y[train]),kind=head,hidden=cfg['hidden'],lr=cfg['lr'])
                work=out/'checkpoints'/f'probe_{name}_{head}_seed{seed}'
                model=fit(model,scaled,y,split,cfg,work,f'probe/{name}/{head}/seed{seed}',seed,cfg['probe_epochs'])
                prob=expit(inference(model,scaled))
                export_probabilities(out,'direct',name,head,seed,pairs,prob)
                rows.extend(evaluation_rows('direct',name,head,seed,pairs,y,prob))
                pd.DataFrame(rows).to_csv(out/'per_seed_metrics.csv',index=False)
    for file,data in [('per_seed_metrics.csv',rows),('representation_metrics.csv',geometry),
                      ('clinical_alignment_metrics.csv',alignment),('oracle_gap_per_seed.csv',gaps)]:
        pd.DataFrame(data).to_csv(out/file,index=False)
    return reps


def representation_metrics(current,future,pred):
    delta_true=future-current;delta_pred=pred-current
    cosine=lambda a,b:np.sum(a*b,1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+1e-12)
    return dict(mse=float(np.mean((pred-future)**2)),cosine=float(cosine(pred,future).mean()),
        l2=float(np.linalg.norm(pred-future,axis=1).mean()),delta_mse=float(np.mean((delta_pred-delta_true)**2)),
        delta_cosine=float(cosine(delta_pred,delta_true).mean()),
        true_delta_norm=float(np.linalg.norm(delta_true,axis=1).mean()),
        predicted_delta_norm=float(np.linalg.norm(delta_pred,axis=1).mean()))


def cluster_weights(ids,draws=2000,seed=42):
    _,inverse=np.unique(ids,return_inverse=True);patients=inverse.max()+1
    rng=np.random.default_rng(seed)
    selected=rng.integers(patients,size=(draws,patients))
    count=np.zeros((draws,patients),dtype=np.int32)
    np.add.at(count,(np.broadcast_to(np.arange(draws)[:,None],selected.shape),selected),1)
    return count[:,inverse]


def weighted_scores(y,p,w):
    """Exact cluster-bootstrap AUROC/AP/MCC; fixed prediction ordering avoids refits."""
    y=np.asarray(y);p=np.asarray(p);w=np.asarray(w,dtype=np.float64)
    order=np.argsort(p,kind='stable');starts=np.r_[0,np.flatnonzero(np.diff(p[order]))+1]
    positive=np.add.reduceat(w[:,order]*y[order],starts,axis=1)
    total=np.add.reduceat(w[:,order],starts,axis=1);negative=total-positive
    pos=positive.sum(1);neg=negative.sum(1)
    before=np.cumsum(negative,axis=1)-negative
    with np.errstate(divide='ignore',invalid='ignore'):
        auc=np.sum(positive*(before+.5*negative),axis=1)/(pos*neg)
        cp=np.cumsum(positive[:,::-1],axis=1);ct=np.cumsum(total[:,::-1],axis=1)
        precision=np.divide(cp,ct,out=np.zeros_like(cp),where=ct>0)
        ap=np.sum(positive[:,::-1]*precision,axis=1)/pos
        tp=w@(y*(p>=.5));fp=w@((1-y)*(p>=.5));fn=pos-tp;tn=neg-fp
        den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        mcc=np.divide(tp*tn-fp*fn,den,out=np.zeros_like(den),where=den>0)
    scores=np.stack([auc,ap,mcc],axis=1)
    scores[(pos==0)|(neg==0)]=np.nan
    return scores


def bootstrap(cfg,out,pairs):
    print('BOOTSTRAP: patient clusters, all channels',flush=True)
    ensembles={}
    for path in sorted((out/'predictions').glob('*.csv')):
        table=pd.read_csv(path,dtype={'patient_id':str})
        pd.testing.assert_frame_equal(table[['patient_id','pair_type','split']],pairs[['patient_id','pair_type','split']])
        stem=path.stem.rsplit('_seed',1)[0]
        ensembles.setdefault(stem,[]).append(table[[t+'_probability' for t in TASKS]].to_numpy())
    ensembles={k:np.mean(v,axis=0) for k,v in ensembles.items()}
    selections=json.loads((out/'selection_lock.json').read_text())
    winner=selections['primary_clinical'];chosen=f"{winner['mode']}_{winner['variant']}"
    rows=[];intervals=[];gaprows=[]
    all_models=list(selections['selected'])
    for j,task in enumerate(TASKS):
        mask=pairs.split.eq('test').to_numpy() & pairs[task].notna().to_numpy()
        y=pairs.loc[mask,task].to_numpy(int);ids=pairs.loc[mask,'patient_id'].to_numpy()
        w=cluster_weights(ids,cfg['bootstrap_samples'],42+j)
        scores={k:weighted_scores(y,v[mask,j],w) for k,v in ensembles.items()}
        points={k:weighted_scores(y,v[mask,j],np.ones((1,len(y))))[0] for k,v in ensembles.items()}
        for key,values in scores.items():
            for m,metric in enumerate(['auroc','auprc','mcc']):
                valid=np.isfinite(values[:,m]);low,high=np.quantile(values[valid,m],[.025,.975])
                intervals.append(dict(arm=key,task=task,metric=metric,value=points[key][m],ci_low=low,ci_high=high,
                    valid_draws=int(valid.sum()),uncertainty='patient clusters, seed-ensemble probabilities, conditional on trained models'))
        def compare(channel,head,model,comparator):
            a=f'{channel}_{model}_{head}';b=f'{channel}_{comparator}_{head}'
            if a not in scores or b not in scores:return
            for m,metric in enumerate(['auroc','auprc','mcc']):
                diff=scores[a][:,m]-scores[b][:,m];valid=np.isfinite(diff)
                low,high=np.quantile(diff[valid],[.025,.975])
                rows.append(dict(channel=channel,head=head,model=model,comparator=comparator,task=task,
                    metric=metric,gain=points[a][m]-points[b][m],ci_low=low,ci_high=high,
                    valid_draws=int(valid.sum()),primary_candidate=model==chosen or comparator==chosen))
        for model in all_models:
            for comparator in ['old','identity','current']:
                compare('risk','frozen_oracle',model,comparator)
            compare('risk','frozen_oracle','real_future',model)
            if model!=chosen:compare('risk','frozen_oracle',chosen,model)
            for head in ['linear','mlp']:
                for comparator in ['G3_old','G0']:
                    compare('direct',head,model,comparator)
                compare('direct',head,'G1',model)
                if model!=chosen:compare('direct',head,chosen,model)
        for mode in ['residual','direct']:
            for variant in ['C2','C3','C4']:
                compare('risk','frozen_oracle',f'{mode}_{variant}',f'{mode}_C1')
        for variant in ['C1','C2','C3','C4']:
            compare('risk','frozen_oracle',f'residual_{variant}',f'direct_{variant}')
        oracle_key='risk_real_future_frozen_oracle';current_key='risk_current_frozen_oracle'
        denom=scores[oracle_key][:,0]-scores[current_key][:,0]
        for model in ['identity','old','old_legacy_realized_interval','real_future']+all_models:
            key=f'risk_{model}_frozen_oracle'
            gap=scores[oracle_key][:,0]-scores[key][:,0];valid=np.isfinite(gap)
            low,high=np.quantile(gap[valid],[.025,.975])
            ratio=np.divide(scores[key][:,0]-scores[current_key][:,0],denom,
                out=np.full_like(denom,np.nan),where=denom>0)
            ratios=ratio[np.isfinite(ratio)]
            ratio_low,ratio_high=np.quantile(ratios,[.025,.975]) if len(ratios) else (np.nan,np.nan)
            pointdenom=points[oracle_key][0]-points[current_key][0]
            gaprows.append(dict(model=model,task=task,ensemble_OracleGap=points[oracle_key][0]-points[key][0],
                OracleGap_ci_low=low,OracleGap_ci_high=high,
                ensemble_GapClosure=(points[key][0]-points[current_key][0])/pointdenom if pointdenom>0 else np.nan,
                GapClosure_ci_low=ratio_low,GapClosure_ci_high=ratio_high,
                positive_denominator_bootstrap_draws=len(ratios),requested_draws=cfg['bootstrap_samples']))
    pd.DataFrame(rows).drop_duplicates(['channel','head','model','comparator','task','metric']).to_csv(out/'paired_comparisons.csv',index=False)
    pd.DataFrame(intervals).to_csv(out/'bootstrap_intervals.csv',index=False)
    perseed=pd.read_csv(out/'oracle_gap_per_seed.csv')
    summary=perseed.groupby(['model','split','task'])[['AUC_predicted','AUC_current','AUC_oracle','OracleGap','GapClosure']].agg(['mean','std']).reset_index()
    summary.columns=['_'.join(c).strip('_') for c in summary.columns]
    summary=summary.merge(pd.DataFrame(gaprows),on=['model','task'],how='left')
    for col in pd.DataFrame(gaprows).columns:
        if col not in ['model','task']:summary.loc[summary.split.ne('test'),col]=np.nan
    summary.to_csv(out/'oracle_gap_summary.csv',index=False)
    return ensembles,chosen


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'results/nlst_clinical_predictor_gate/configs/gate.yaml')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--analysis-only',action='store_true')
    args=parser.parse_args();cfg=__import__('yaml').safe_load(args.config.read_text())
    if args.smoke:cfg.update(output_dir='results/nlst_clinical_predictor_gate_smoke',seeds=[42],oracle_epochs=2,
        predictor_epochs=2,probe_epochs=2,patience=2,clinical_weights=[1.],bootstrap_samples=20,wandb_mode='disabled',smoke=True)
    cfg['checkpoint_sha256']=json.loads(Path(cfg['extraction_metadata']).read_text())['checkpoint_sha256']
    out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    for name in ['configs','plots','checkpoints','runs','predictions','analysis']:(out/name).mkdir(exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError('This bounded gate requires an available CUDA device')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    logging.getLogger('lightning.pytorch').setLevel(logging.WARNING)
    files=[args.config,Path(__file__),ROOT/'src/lung_jepa/models/clinical_future.py',
        ROOT/'scripts/run_nlst_gate.py',ROOT/'scripts/run_colipri_latent_world_model_ablation.py',
        ROOT/'scripts/analyze_nlst_clinical_predictor_gate.py',ROOT/'tests/test_clinical_predictor_gate.py',
        ROOT/'src/lung_jepa/models/disease_state.py',ROOT/'src/lung_jepa/models/prognostic_adapter.py',
        Path(cfg['previous_gate'])/'pairs_manifest.csv',Path(cfg['previous_gate'])/'patient_splits.csv',
        Path(cfg['embedding_cache'])/'embeddings.npy',Path(cfg['extraction_metadata'])]
    files += [Path(cfg['old_predictor_dir'])/'checkpoints'/f"{cfg['old_predictor_config']}__seed{s}.pt" for s in cfg['old_predictor_seeds']]
    fingerprints={str(f):digest(f) for f in files}
    if not args.analysis_only:
        if (out/'fingerprints.json').exists():
            assert json.loads((out/'fingerprints.json').read_text())==fingerprints,'Resume fingerprint mismatch; use a new output directory'
        save_json(out/'fingerprints.json',fingerprints);save_json(out/'configs/resolved.json',cfg)
    save_json(out/'environment.json',dict(python=sys.version,torch=torch.__version__,lightning=pl.__version__,
        numpy=np.__version__,pandas=pd.__version__,device=torch.cuda.get_device_name(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),dataset_version='pilot_1000_balanced',
        seeds=cfg['seeds'],test_cohort_unchanged=True))
    pairs,current,future,y,split,time,ds,fs=load_data(cfg,out)
    if args.analysis_only:
        oracle=torch.load(out/'checkpoints/oracle_risk_head.pt',map_location='cpu',weights_only=True)
        teacher=OracleRiskHead(oracle['mean'],oracle['scale'],oracle['kind'],oracle['heads'],oracle['hidden'])
        teacher.load_state_dict(oracle['state_dict'])
        locked=json.loads((out/'selection_lock.json').read_text())
        predictions={name:[np.load(out/'predictions'/f"{row['name']}_seed{s}.npy") for s in cfg['seeds']]
                     for name,row in locked['selected'].items()}
        old={name:np.mean([np.load(out/'predictions'/f'{name}_original_seed{s}.npy') for s in cfg['old_predictor_seeds']],axis=0)
             for name in ['old','old_legacy_realized_interval']}
        reps={'current':[current]*len(cfg['seeds']),'identity':[current]*len(cfg['seeds']),
            'real_future':[future]*len(cfg['seeds']),**{k:[v]*len(cfg['seeds']) for k,v in old.items()},**predictions}
    else:
        teacher,kind,families=train_oracle(cfg,out,future,y,split,fs)
        old=old_predictions(cfg,out,pairs,current)
        predictions=train_predictors(cfg,out,current,future,y,split,time,ds,fs,teacher,kind)
        reps=evaluate(cfg,out,pairs,current,future,y,split,predictions,old,teacher,families)
    ensembles,chosen=bootstrap(cfg,out,pairs)
    # Analysis/report is separated to keep the training and selection code reviewable.
    from analyze_nlst_clinical_predictor_gate import analyze,report
    analyze(cfg,out,pairs,current,future,y,reps,ensembles,chosen)
    report(cfg,out,pairs,chosen)
    save_json(out/'COMPLETE.json',dict(smoke=cfg.get('smoke',False),seeds=cfg['seeds'],
        chosen=chosen,report=str(out/'FINAL_CLINICAL_PREDICTOR_GATE_REPORT.md')))
    print('COMPLETE',out,flush=True)


if __name__=='__main__':main()
