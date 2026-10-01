#!/usr/bin/env python3
"""Audited frozen-COLIPRI gate; no new dynamics/world model is trained."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                            f1_score, matthews_corrcoef, confusion_matrix)
from lung_jepa.models.disease_state import GateModule
from run_colipri_latent_world_model import build_pairs, DIAGNOSTIC_TASKS, seed_all

TASKS = ['primary_future_cancer_'+h for h in ['1y','2y','3y']]
METRICS = ['auroc','auprc','accuracy','balanced_accuracy','f1','mcc','sensitivity','specificity']
DESCRIPTIONS = dict(G0='current global', G1='current + true future global (oracle)',
 G2='current + true global delta (oracle)', G3='current + predicted global future',
 G_delta='true global delta only (oracle)', G_abs='current + absolute global delta (oracle)',
 G_identity='current + identity future', D0='current disease state',
 D1='current + true future disease state (oracle)', D2='current + disease delta (oracle)',
 D_delta='disease delta only (oracle)', D_abs='current + absolute disease delta (oracle)',
 D_identity='current + identity disease future')
COMPARISONS = [('G1','G0'),('G2','G0'),('G3','G0'),('D0','G0'),('D1','D0'),('D2','D0'),('G1','G3'),('G2','G1')]


def markdown(table):
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f'{value:.4f}' if np.isfinite(value) else 'NA'
        return str(value).replace('|', '/').replace('\n', ' ')
    lines = ['| ' + ' | '.join(map(str, table.columns)) + ' |',
             '| ' + ' | '.join(['---']*len(table.columns)) + ' |']
    lines += ['| ' + ' | '.join(cell(v) for v in row) + ' |' for row in table.itertuples(index=False, name=None)]
    return '\n'.join(lines)


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str))


def digest(path):
    h = hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_splits(splits):
    if splits.patient_id.duplicated().any():
        raise ValueError('Split must contain exactly one assignment per patient')
    if splits.split.isna().any() or not set(splits.split) <= {'train','val','test'}:
        raise ValueError('Missing or unknown split')
    sets = [set(splits.loc[splits.split.eq(s),'patient_id']) for s in ['train','val','test']]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Patient leakage')


def metrics(y, p):
    pred = p >= .5
    tn, fp, fn, tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
    sens = tp/(tp+fn) if tp+fn else np.nan
    spec = tn/(tn+fp) if tn+fp else np.nan
    two = len(np.unique(y)) == 2
    return dict(auroc=roc_auc_score(y,p) if two else np.nan,
                auprc=average_precision_score(y,p) if np.any(y) else np.nan,
                accuracy=accuracy_score(y,pred), balanced_accuracy=(sens+spec)/2,
                f1=f1_score(y,pred,zero_division=0),mcc=matthews_corrcoef(y,pred),
                sensitivity=sens,specificity=spec)


def weights(y):
    pos = np.nansum(y,axis=0)
    neg = np.sum(y==0,axis=0)
    if (pos == 0).any() or (neg == 0).any():
        raise ValueError('A training task lacks a class; do not invent supervision')
    return (neg/pos).tolist()


def load_and_audit(cfg, out):
    paths = argparse.Namespace(**{k:Path(cfg[k]) for k in
        ['embedding_cache','prognostic_labels','diagnostic_manifest','patient_splits']})
    splits = pd.read_csv(paths.patient_splits,dtype={'patient_id':str})
    validate_splits(splits)
    em = pd.read_csv(paths.embedding_cache/'embedding_metadata.csv',dtype={'patient_id':str})
    if em.duplicated(['patient_id','screening_round']).any():
        raise ValueError('Ambiguous patient/round cache')
    if not np.array_equal(em.embedding_index.to_numpy(),np.arange(len(em))):
        raise ValueError('Noncanonical feature index ordering')
    prog = pd.read_csv(paths.prognostic_labels,dtype={'patient_id':str})
    diag = pd.read_csv(paths.diagnostic_manifest,dtype={'patient_id':str})
    if 'screening_round' not in diag:
        diag = diag.rename(columns={'timepoint':'screening_round'})
    joined = em.merge(prog.drop(columns='screening_round'),on=['patient_id','series_uid'],
                      validate='one_to_one',how='left',indicator=True)
    if not joined._merge.eq('both').all():
        raise ValueError('Incomplete prognostic metadata linkage')
    joined = joined.drop(columns='_merge').merge(
        diag[['patient_id','series_uid']+DIAGNOSTIC_TASKS],on=['patient_id','series_uid'],
        validate='one_to_one',how='left').merge(splits[['patient_id','split']],on='patient_id',
        validate='many_to_one',how='left')
    if joined.split.isna().any():
        raise ValueError('Unassigned cached patients')
    source_splits = diag[['patient_id','split']].drop_duplicates()
    validate_splits(source_splits)
    check = source_splits.merge(splits[['patient_id','split']],on='patient_id',suffixes=('_scan','_patient'))
    if not check.split_scan.eq(check.split_patient).all():
        raise ValueError('Diagnostic and patient split disagree')
    pairs,z = build_pairs(paths)
    for task in TASKS:
        if not pairs[task].dropna().isin([0,1]).all():
            raise ValueError('Non-binary endpoint')
    prior = joined.prognostic_label_quality_flags.fillna('').str.contains('excluded_prior_cancer_diagnosis')
    if joined.loc[prior,['future_cancer_1y','future_cancer_2y','future_cancer_3y']].notna().any().any():
        raise ValueError('Prior cancer included in prognosis')
    tokens = np.load(cfg['token_cache'],mmap_mode='r')
    if tokens.shape != (len(em),64,z.shape[1]):
        raise ValueError(f'Token shape/order mismatch: {tokens.shape}')
    pooling_max = 0.
    for start in range(0,len(em),128):
        chunk = np.asarray(tokens[start:start+128])
        if not np.isfinite(chunk).all():
            raise ValueError('Nonfinite token cache')
        pooling_max = max(pooling_max,float(np.max(np.abs(chunk.mean(1)-z[start:start+128]))))
    if pooling_max > 1e-5:
        raise ValueError('Global/token pooling mismatch')
    selected = prog.merge(em[['patient_id','series_uid','embedding_index']],
                          on=['patient_id','series_uid'],how='left',validate='one_to_one')
    selected = selected.merge(splits[['patient_id','split']],on='patient_id',how='left',validate='many_to_one')
    selected['feature_available'] = selected.embedding_index.notna()
    selected['scan_id'] = selected.series_uid
    selected['timepoint'] = selected.screening_round
    selected = selected.sort_values(['patient_id','timepoint_index'])
    selected['time_from_previous_scan'] = selected.groupby('patient_id').scan_day_from_randomization.diff()
    selected['significant_change'] = np.nan
    selected['change_label_status'] = 'unavailable: no validated scan-level significant-change field'
    selected.to_csv(out/'longitudinal_table.csv',index=False)
    pairs.to_csv(out/'pairs_manifest.csv',index=False)
    joined.to_csv(out/'feature_metadata.csv',index=False)
    splits.to_csv(out/'patient_splits.csv',index=False)
    subjects = pd.read_csv(cfg['subject_manifest'],dtype={'patient_id':str})
    if set(subjects.patient_id) != set(selected.patient_id):
        raise ValueError('Cohort manifest and target metadata disagree')
    audit=[]
    def row(scope, name, value):
        audit.append(dict(scope=scope,metric=name,value=value))
    def distribution(scope,name,values):
        s = pd.Series(values).dropna()
        row(scope,name+'_n',len(s))
        for stat,value in s.describe(percentiles=[.25,.5,.75]).items():
            row(scope,name+'_'+stat,float(value))
    for scope,table in [('selected',selected),('cached',joined)]:
        counts = table.groupby('patient_id').size()
        row(scope,'patients',len(counts));row(scope,'scans',len(table))
        for n in [1,2,3]:row(scope,f'patients_with_{n}_scans',int(counts.eq(n).sum()))
        patterns = table.groupby('patient_id').screening_round.agg(lambda v:'+'.join(sorted(v)))
        for pattern,n in patterns.value_counts().items():row(scope,'pattern_'+pattern,int(n))
        patient = table.drop_duplicates('patient_id')
        cancer = patient.candx_days.notna()
        row(scope,'patients_developing_lung_cancer',int(cancer.sum()))
        row(scope,'cancer_prevalence',float(cancer.mean()))
        distribution(scope,'diagnosis_days_from_randomization',patient.loc[cancer,'candx_days'])
        eligible_event = table.cancer_event.eq(1)
        distribution(scope,'time_to_cancer_after_scan_days',table.loc[eligible_event,'cancer_time_days'])
    for kind,n in pairs.pair_type.value_counts().items():row('cached','usable_pairs_'+kind,int(n))
    distribution('cached','interscan_days',pairs.dt_days)
    for split in ['train','val','test']:
        row('cached',split+'_patients',int(splits.split.eq(split).sum()))
        for task in TASKS:
            s = pairs.loc[pairs.split.eq(split),task].dropna()
            row('cached',split+'_'+task+'_n',len(s));row('cached',split+'_'+task+'_positive',int(s.sum()))
    row('cached','significant_change_labels',0)
    row('selected','requested_approximate_patients',2000)
    pd.DataFrame(audit).to_csv(out/'cohort_audit.csv',index=False)
    # Preserve real comparison fields without declaring unknown codes stable/change.
    clinical = Path(cfg['clinical_dir'])
    schema={}
    for name in ['prsn','screen','ctab','ctabc','canc']:
        table = pd.read_csv(clinical/f'nlst_{name}.csv',dtype=str)
        schema[name] = dict(rows=len(table),columns=table.columns.tolist())
        if name=='ctabc':
            cols=['dicom_patient_id','study_yr','sct_ab_num','sct_ab_code',
                  'sct_ab_gwth','sct_ab_invg','sct_ab_preexist','sct_ab_attn','visible_days']
            table[[c for c in cols if c in table]].to_csv(out/'representation_analysis/raw_lesion_comparison_fields.csv',index=False)
            schema[name]['candidate_distributions'] = {c:table[c].value_counts(dropna=False).to_dict()
                for c in ['sct_ab_gwth','sct_ab_invg','sct_ab_preexist']}
    schema['significant_change_status']='No validated scan-level significant-change variable or local code dictionary. Lesion growth/involution codes exist; no unsupported code mapping or absent-record negatives.'
    save_json(out/'representation_analysis/clinical_schema_audit.json',schema)
    text = ('# Cohort audit\n\nUser approved the verified cached cohort after inspection: the active '
        'snapshot is pilot_1000_balanced, not a complete 2,000-patient COLIPRI cohort. '
        'The manifest contains 1,000 patients, 3,000 selected scans, and 500 cancer cases. '
        f'The matched cache contains {joined.patient_id.nunique()} patients, {len(joined)} scans, '
        f'and {len(pairs)} consecutive pairs. Selected scans and usable cached scans are counted separately.\n\n'
        'Relative time uses scr_days0/1/2 via scan_day_from_randomization. Manifest study dates are '
        'deidentified; clinical intervals do not use their apparent calendar differences. '
        'Patients with only T0 or T0+T1 are retained in the audit; only observed consecutive pairs '
        'enter prognostic comparisons. One selected patient has no cached feature and no gate split.\n\n'
        'Significant-change supervision is unavailable. sct_ab_gwth, sct_ab_invg, and '
        'sct_ab_preexist are lesion-record fields without a verified local scan-level coding rule. '
        'any_significant_abnormality and comparison_suspicious are current findings, not change labels.\n\n'
        'The cohort is cancer-enriched; its prevalence and AUPRC do not estimate NLST population performance.\n\n'
        '```text\n'+pd.DataFrame(audit).to_string(index=False)+'\n```\n')
    (out/'cohort_audit.md').write_text(text)
    return joined,pairs,z,tokens


def fit(model,x,y,split,cfg,work,name,seed,epochs):
    work.mkdir(parents=True,exist_ok=True)
    if (work/'done.json').exists():
        return GateModule.load_from_checkpoint(work/'best.ckpt',map_location='cpu')
    seed_all(seed)
    loggers=[CSVLogger(str(work),name='history')]
    wandb_logger=None
    if cfg['wandb_mode'] != 'disabled':
        wandb_logger=WandbLogger(project=cfg['wandb_project'],group=cfg['wandb_group'],name=name,
            save_dir=str(work),offline=cfg['wandb_mode']=='offline',log_model=False)
        wandb_logger.experiment.config.update(dict(config=cfg,seed=seed,stage=name,
            parameters=sum(p.numel() for p in model.parameters()),input_shape=list(x.shape),
            split='preserved patient-level 599/200/200',dataset_version='pilot_1000_balanced',
            checkpoint_sha256=cfg['checkpoint_sha256']))
        loggers.append(wandb_logger)
    ckpt=ModelCheckpoint(dirpath=work,filename='best',monitor='val/loss',mode='min',
                         save_top_k=1,enable_version_counter=False)
    trainer=pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu',devices=1,
        max_epochs=epochs,logger=loggers,callbacks=[ckpt,EarlyStopping(monitor='val/loss',
        mode='min',patience=cfg['patience'],min_delta=1e-6)],deterministic=True,
        enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=5,
        num_sanity_val_steps=0)
    datasets={s:TensorDataset(torch.from_numpy(x[split==s]),torch.from_numpy(y[split==s])) for s in ['train','val']}
    trainer.fit(model,DataLoader(datasets['train'],batch_size=cfg['batch_size'],shuffle=True,
                                generator=torch.Generator().manual_seed(seed)),
                      DataLoader(datasets['val'],batch_size=cfg['batch_size'],shuffle=False))
    save_json(work/'done.json',dict(epochs_run=trainer.current_epoch,best_validation=float(ckpt.best_model_score),
        best_checkpoint=ckpt.best_model_path,parameters=sum(p.numel() for p in model.parameters())))
    if wandb_logger is not None:wandb_logger.experiment.finish()
    model.cpu();del trainer
    return GateModule.load_from_checkpoint(work/'best.ckpt',map_location='cpu')


def apply(model,x,device,projector=False):
    model.eval().requires_grad_(False).to(device)
    chunks=[]
    with torch.inference_mode():
        for start in range(0,len(x),128):
            tensor=torch.from_numpy(np.asarray(x[start:start+128])).to(device)
            result=model.projector(tensor) if projector else torch.sigmoid(model(tensor))
            chunks.append(result.cpu().numpy())
    model.cpu()
    return np.concatenate(chunks)


def representations(current,future,predicted=None,prefix='G'):
    zero=np.zeros_like(current)
    reps={prefix+'0':np.concatenate([current,zero],1),
          prefix+'1':np.concatenate([current,future],1),
          prefix+'2':np.concatenate([current,future-current],1),
          prefix+'_delta':np.concatenate([zero,future-current],1),
          prefix+'_abs':np.concatenate([current,np.abs(future-current)],1),
          prefix+'_identity':np.concatenate([current,current],1)}
    if predicted is not None:reps['G3']=np.concatenate([current,predicted],1)
    return reps


def predictor(cfg,pairs,z,out):
    directory=Path(cfg['predictor_dir'])
    ref=pd.read_csv(directory/'pairs_manifest.csv',dtype={'patient_id':str})
    pd.testing.assert_frame_equal(pairs[['patient_id','pair_type','idx_t','idx_t1','split']],
        ref[['patient_id','pair_type','idx_t','idx_t1','split']])
    selection=json.loads((directory/'selection.json').read_text())
    if selection['best_objective_config'] != cfg['predictor_config']:
        raise ValueError('Predictor disagrees with existing validation selection')
    predictions=[np.load(directory/'predicted_embeddings'/f"{cfg['predictor_config']}__seed{s}.npy")
                 for s in cfg['predictor_seeds']]
    if any(p.shape != (len(pairs),z.shape[1]) or not np.isfinite(p).all() for p in predictions):
        raise ValueError('Invalid cached forecasts')
    true=z[pairs.idx_t1];current=z[pairs.idx_t]
    rows=[]
    for split in ['train','val','test']:
        mask=pairs.split.eq(split).to_numpy()
        for name,p in [('identity',current)]+[(f'learned_seed{s}',p) for s,p in zip(cfg['predictor_seeds'],predictions)]+[('learned_ensemble',np.mean(predictions,axis=0))]:
            a,b,c=p[mask],true[mask],current[mask]
            delta,real=a-c,b-c
            cos=lambda x,y:np.sum(x*y,1)/(np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1)+1e-12)
            rows.append(dict(space='global',split=split,predictor=name,mse=float(np.mean((a-b)**2)),
                cosine=float(cos(a,b).mean()),l2=float(np.linalg.norm(a-b,axis=1).mean()),
                delta_cosine=float(cos(delta,real).mean()),delta_mse=float(np.mean((delta-real)**2))))
    pd.DataFrame(rows).to_csv(out/'no_change_baseline.csv',index=False)
    save_json(out/'predictor_provenance.json',dict(selection=selection,config=cfg['predictor_config'],
        seeds=cfg['predictor_seeds'],G3='fixed seed ensemble of existing validation-selected standardized-delta residual MLP',
        uncertainty='probe seeds; predictor seed metrics also reported separately',
        inputs='current global state and interscan interval only; no clinical/future-scan inputs',
        manifest_verified=True))
    return np.mean(predictions,axis=0).astype(np.float32)


def train_all(cfg,out,scans,pairs,z,tokens):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    tr=scans.split.eq('train').to_numpy()
    sc=StandardScaler().fit(np.asarray(tokens[tr]).reshape(-1,tokens.shape[-1]))
    raw=np.asarray(tokens)
    scaled=((raw-sc.mean_)/sc.scale_).astype(np.float32)
    np.savez(out/'token_scaler.npz',mean=sc.mean_,scale=sc.scale_)
    own=['future_cancer_'+h for h in ['1y','2y','3y']]+DIAGNOSTIC_TASKS
    yscan=scans[own].to_numpy(np.float32)
    y=pairs[TASKS].to_numpy(np.float32)
    train=pairs.split.eq('train').to_numpy()
    predicted=predictor(cfg,pairs,z,out)
    g=representations(z[pairs.idx_t],z[pairs.idx_t1],predicted)
    records=[]
    for seed in cfg['seeds']:
        print(f'PROJECTOR seed={seed}',flush=True)
        seed_all(seed)
        model=GateModule(tokens.shape[-1],len(own),weights(yscan[tr]),kind='projector',
            d_state=cfg['d_state'],queries=cfg['queries'],tokens=tokens.shape[1],
            auxiliary_weight=cfg['auxiliary_weight'],lr=cfg['lr'])
        model=fit(model,scaled,yscan,scans.split.to_numpy(),cfg,out/'runs'/f'projector_seed{seed}',
                  f'projector/seed{seed}',seed,cfg['projector_epochs'])
        state=apply(model,scaled,device,projector=True)
        np.save(out/'representation_analysis'/f'disease_states_seed{seed}.npy',state)
        d=representations(state[pairs.idx_t].reshape(len(pairs),-1),
                          state[pairs.idx_t1].reshape(len(pairs),-1),prefix='D')
        for exp,x in {**g,**d}.items():
            if exp not in cfg['experiments']:continue
            scaler=StandardScaler().fit(x[train])
            xscaled=scaler.transform(x).astype(np.float32)
            np.savez(out/'configs'/f'{exp}_seed{seed}_scaler.npz',mean=scaler.mean_,scale=scaler.scale_)
            save_json(out/'configs'/f'{exp}.json',dict(experiment=exp,description=DESCRIPTIONS[exp],
                input_dim=x.shape[1],zero_padding='current-only/delta-only padded to matched concatenation width',
                tasks=TASKS,seeds=cfg['seeds'],heads=['linear','mlp'],oracle=exp not in ['G0','G3','D0','G_identity','D_identity'],
                projector='same frozen current-scan multi-task projector shared across D arms',config=cfg))
            for head in ['linear','mlp']:
                print(f'PROBE {exp} {head} seed={seed}',flush=True)
                seed_all(seed)
                probe=GateModule(x.shape[1],3,weights(y[train]),kind=head,hidden=cfg['hidden'],lr=cfg['lr'])
                work=out/'runs'/f'{exp}_{head}_seed{seed}'
                probe=fit(probe,xscaled,y,pairs.split.to_numpy(),cfg,work,
                          f'{exp}/{head}/seed{seed}',seed,cfg['probe_epochs'])
                probability=apply(probe,xscaled,device)
                export=pairs[['patient_id','pair_type','split']+TASKS].copy()
                for j,task in enumerate(TASKS):export[task+'_probability']=probability[:,j]
                export.to_csv(work/'predictions.csv',index=False)
                for split in ['train','val','test']:
                    for j,task in enumerate(TASKS):
                        mask=pairs.split.eq(split).to_numpy() & np.isfinite(y[:,j])
                        records.append(dict(experiment=exp,head=head,seed=seed,split=split,task=task,
                            n=int(mask.sum()),patients=pairs.loc[mask,'patient_id'].nunique(),
                            positives=int(y[mask,j].sum()),parameters=sum(p.numel() for p in probe.parameters()),
                            **metrics(y[mask,j],probability[mask,j])))
                pd.DataFrame(records).to_csv(out/'per_seed_metrics.csv',index=False)
    return pd.DataFrame(records)


def bootstrap(cfg,out,pairs):
    """Paired cluster bootstrap; both longitudinal rows follow their patient."""
    arm_probs={}
    for head in ['linear','mlp']:
        for exp in cfg['experiments']:
            files=[out/'runs'/f'{exp}_{head}_seed{s}'/'predictions.csv' for s in cfg['seeds']]
            if not all(f.exists() for f in files):continue
            probs=[]
            for f in files:
                table=pd.read_csv(f,dtype={'patient_id':str})
                pd.testing.assert_frame_equal(table[['patient_id','pair_type','split']],pairs[['patient_id','pair_type','split']])
                probs.append(table[[t+'_probability' for t in TASKS]].to_numpy())
            arm_probs[exp,head]=np.mean(probs,axis=0)
    interval_rows=[];comparison_rows=[]
    for j,task in enumerate(TASKS):
        print('BOOTSTRAP',task,flush=True)
        mask=pairs.split.eq('test').to_numpy() & pairs[task].notna().to_numpy()
        y=pairs.loc[mask,task].to_numpy(int)
        ids=pairs.loc[mask,'patient_id'].to_numpy()
        patients=np.unique(ids)
        lookup={pid:np.flatnonzero(ids==pid) for pid in patients}
        rng=np.random.default_rng(42+j)
        draws=[]
        for _ in range(cfg['bootstrap_samples']):
            idx=np.concatenate([lookup[p] for p in rng.choice(patients,len(patients),replace=True)])
            if len(np.unique(y[idx]))==2:draws.append(idx)
        scores={}
        points={}
        for key,p in arm_probs.items():
            p=p[mask,j]
            samples=np.array([[roc_auc_score(y[idx],p[idx]),average_precision_score(y[idx],p[idx]),
                               matthews_corrcoef(y[idx],p[idx]>=.5)] for idx in draws])
            points[key]=np.array([roc_auc_score(y,p),average_precision_score(y,p),matthews_corrcoef(y,p>=.5)])
            scores[key]=samples
            for k,metric in enumerate(['auroc','auprc','mcc']):
                low,high=np.quantile(samples[:,k],[.025,.975])
                interval_rows.append(dict(experiment=key[0],head=key[1],task=task,metric=metric,
                    ensemble_value=points[key][k],ci_low=low,ci_high=high,valid_draws=len(draws),
                    bootstrap='paired patient clusters; seed ensemble; conditional on fitted models'))
        def comparison(name,head,point,samples):
            for k,metric in enumerate(['auroc','auprc','mcc']):
                low,high=np.quantile(samples[:,k],[.025,.975])
                comparison_rows.append(dict(comparison=name,head=head,task=task,metric=metric,
                    gain=point[k],ci_low=low,ci_high=high,valid_draws=len(draws)))
        for head in ['linear','mlp']:
            for a,b in COMPARISONS:
                if (a,head) in scores and (b,head) in scores:
                    comparison(a+' - '+b,head,points[a,head]-points[b,head],scores[a,head]-scores[b,head])
            for future in ['1','2']:
                keys=[('D'+future,head),('D0',head),('G'+future,head),('G0',head)]
                if all(k in scores for k in keys):
                    comparison(f'(D{future}-D0) - (G{future}-G0)',head,
                        points[keys[0]]-points[keys[1]]-points[keys[2]]+points[keys[3]],
                        scores[keys[0]]-scores[keys[1]]-scores[keys[2]]+scores[keys[3]])
    pd.DataFrame(interval_rows).to_csv(out/'bootstrap_intervals.csv',index=False)
    pd.DataFrame(comparison_rows).to_csv(out/'paired_comparisons.csv',index=False)


def analysis(cfg,out,scans,pairs,z):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[]
    baseline=pd.read_csv(out/'no_change_baseline.csv')
    for seed in cfg['seeds']:
        state=np.load(out/'representation_analysis'/f'disease_states_seed{seed}.npy')
        for space,x in [('global',z),('disease',state.reshape(len(state),-1))]:
            current,future=x[pairs.idx_t],x[pairs.idx_t1]
            cos=np.sum(current*future,1)/(np.linalg.norm(current,axis=1)*np.linalg.norm(future,axis=1)+1e-12)
            distance=np.linalg.norm(future-current,axis=1)
            table=pairs[['patient_id','pair_type','split']+TASKS].copy()
            table['space']=space;table['seed']=seed;table['cosine']=cos;table['l2']=distance
            rows.append(table)
            if space=='disease':
                for split in ['train','val','test']:
                    m=pairs.split.eq(split).to_numpy()
                    baseline=pd.concat([baseline,pd.DataFrame([dict(space='disease',split=split,
                        predictor=f'identity_projector_seed{seed}',mse=float(np.mean((current[m]-future[m])**2)),
                        cosine=float(cos[m].mean()),l2=float(distance[m].mean()),delta_cosine=0.,
                        delta_mse=float(np.mean((current[m]-future[m])**2)))])],ignore_index=True)
            if seed != cfg['seeds'][0]:continue
            # PCA fits training patients only. Display held-out trajectories only.
            scaler=StandardScaler().fit(x[scans.split.eq('train')])
            xs=scaler.transform(x)
            pca=PCA(2,random_state=42).fit(xs[scans.split.eq('train')])
            coords=pca.transform(xs)
            frame=scans[['patient_id','screening_round','split','candx_days']].copy()
            frame[['pc1','pc2']]=coords
            frame.to_csv(out/'representation_analysis'/f'{space}_pca_coordinates.csv',index=False)
            np.savez(out/'representation_analysis'/f'{space}_pca_fit.npz',mean=scaler.mean_,scale=scaler.scale_,
                components=pca.components_,pca_mean=pca.mean_,explained_variance_ratio=pca.explained_variance_ratio_)
            fig,ax=plt.subplots(figsize=(7,5))
            for pid,group in frame[frame.split.eq('test')].groupby('patient_id'):
                group=group.sort_values('screening_round')
                color='crimson' if group.candx_days.notna().any() else 'steelblue'
                ax.plot(group.pc1,group.pc2,color=color,alpha=.18,linewidth=.5)
                ax.scatter(group.pc1,group.pc2,color=color,s=6,alpha=.4)
            ax.set(xlabel='PC1',ylabel='PC2',title=f'{space}: held-out patient trajectories (red = ever cancer)')
            fig.tight_layout();fig.savefig(out/'plots'/f'{space}_trajectories.png',dpi=160);plt.close(fig)
            delta=future-current
            ds=StandardScaler().fit(delta[pairs.split.eq('train')]);dx=ds.transform(delta)
            dpca=PCA(2,random_state=42).fit(dx[pairs.split.eq('train')]);dcoords=dpca.transform(dx)
            df=pairs[['patient_id','pair_type','split']+TASKS].copy();df[['pc1','pc2']]=dcoords
            df.to_csv(out/'representation_analysis'/f'{space}_delta_pca_coordinates.csv',index=False)
            fig,axs=plt.subplots(1,3,figsize=(14,4))
            test=pairs.split.eq('test').to_numpy();target=pairs[cfg['primary_endpoint']].to_numpy()
            for label,color in [(0,'steelblue'),(1,'crimson')]:
                m=test & (target==label)
                axs[0].hist(cos[m],bins=25,alpha=.5,density=True,color=color,label=str(label))
                axs[1].hist(distance[m],bins=25,alpha=.5,density=True,color=color,label=str(label))
                axs[2].scatter(dcoords[m,0],dcoords[m,1],s=10,alpha=.5,color=color,label=str(label))
            for ax,title in zip(axs,['consecutive cosine','consecutive L2','delta PCA']):
                ax.set_title(title);ax.legend(title='post-future cancer 2y')
            fig.suptitle(f'{space}: held-out pairs; qualitative geometry only')
            fig.tight_layout();fig.savefig(out/'plots'/f'{space}_changes.png',dpi=160);plt.close(fig)
    distances=pd.concat(rows,ignore_index=True)
    distances.to_csv(out/'representation_analysis/longitudinal_distances.csv',index=False)
    distances.groupby(['space','seed','split',cfg['primary_endpoint']])[['cosine','l2']].agg(['mean','std','count']).to_csv(out/'representation_analysis/distance_summary.csv')
    baseline.to_csv(out/'no_change_baseline.csv',index=False)


def report(cfg,out,records):
    summary=records.groupby(['experiment','head','split','task'])[METRICS].agg(['mean','std','count']).reset_index()
    summary.columns=['_'.join(c).strip('_') if isinstance(c,tuple) else c for c in summary.columns]
    ci=pd.read_csv(out/'bootstrap_intervals.csv')
    for metric in ['auroc','auprc','mcc']:
        table=ci[ci.metric.eq(metric)][['experiment','head','task','ci_low','ci_high']].rename(
            columns={'ci_low':metric+'_bootstrap_low','ci_high':metric+'_bootstrap_high'})
        summary=summary.merge(table,on=['experiment','head','task'],how='left')
        for col in [metric+'_bootstrap_low',metric+'_bootstrap_high']:
            summary.loc[~summary.split.eq('test'),col]=np.nan
    summary.to_csv(out/'experiment_summary.csv',index=False)
    formatted=[]
    test=summary[summary.split.eq('test')]
    for _,r in test.iterrows():
        row=dict(experiment=r.experiment,head=r['head'],task=r.task)
        for m in METRICS:row[m]=f'{r[m+"_mean"]:.3f} ± {r[m+"_std"]:.3f}'
        formatted.append(row)
    table=markdown(pd.DataFrame(formatted))
    (out/'experiment_summary.md').write_text('# Gate experiment summary\n\nTest metrics are mean ± sample SD across '
        f"{cfg['seeds']}. Disease arms include projector and probe seed variation; global arms include "
        'probe variation. G3 uses the fixed five-seed existing predictor ensemble.\n\n'+table+'\n\n'
        'AUROC/AUPRC/MCC bootstrap intervals are in experiment_summary.csv; they apply to seed-ensemble '
        'probabilities, conditional on trained models, not the distribution of training seeds. '
        'Both rows of a patient are resampled together. Threshold = 0.5, fixed before evaluation.\n')
    boot=pd.read_csv(out/'paired_comparisons.csv')
    primary=boot[boot.task.eq(cfg['primary_endpoint']) & boot.metric.eq('auroc')]
    margin=cfg['meaningful_auroc_gain']
    def status(name,head):
        r=primary[primary.comparison.eq(name) & primary['head'].eq(head)]
        if len(r)!=1:return 'unavailable'
        r=r.iloc[0]
        if r.ci_low>0 and r.gain>=margin:return 'improves'
        if r.ci_low>-margin and r.ci_high<margin:return 'equivalent'
        return 'inconclusive'
    decisions=[]
    for head in ['linear','mlp']:
        g1,g2,g3,c,d1,d2=[status(name,head) for name in ['G1 - G0','G2 - G0','G3 - G0',
                                                     'D0 - G0','D1 - D0','D2 - D0']]
        bigger=any(status(f'(D{n}-D0) - (G{n}-G0)',head)=='improves' for n in ['1','2'])
        if c=='improves' and (d1=='improves' or d2=='improves') and bigger:
            verdict='D candidate: strong disease state; independent validation still required'
        elif g1=='equivalent' and (d1=='improves' or d2=='improves'):
            verdict='C: change state space'
        elif (g1=='improves' or g2=='improves') and g3=='equivalent':
            verdict='B: improve predictor'
        elif g1==g2=='equivalent':
            verdict='A: STOP global latent forecasting'
        else:verdict='INCONCLUSIVE: do not build the full world model'
        decisions.append(dict(head=head,G1_vs_G0=g1,G2_vs_G0=g2,G3_vs_G0=g3,D0_vs_G0=c,
                              D1_vs_D0=d1,D2_vs_D0=d2,larger_disease_gain=bigger,verdict=verdict))
    save_json(out/'gate_decision.json',dict(primary_endpoint=cfg['primary_endpoint'],
        meaningful_gain=margin,rule='gain >= margin and paired 95% CI above zero; equivalence requires CI wholly within ±margin',
        decisions=decisions))
    same=decisions[0]['verdict']==decisions[1]['verdict']
    recommendation=decisions[0]['verdict'] if same else 'INCONCLUSIVE across heads: do not implement the world model'
    extraction=json.loads(Path(cfg['extraction_metadata']).read_text())
    baseline=pd.read_csv(out/'no_change_baseline.csv');baseline=baseline[baseline.split.eq('test')]
    report=('# NLST disease-state gate report\n\n'
        + ('**SMOKE RUN: pipeline verification only; no scientific gate conclusion.**\n\n' if len(cfg['seeds'])<3 else '')
        + f'**Recommendation: {recommendation}.** No RSSM or probabilistic world model was implemented.\n\n'
        '## 1. Dataset and cohort audit\n\n'
        'The user approved the available cached cohort: 1,000 selected cancer-balanced patients; '
        '999 embedded patients, 2,984 scans, 995 T0→T1 and 989 T1→T2 pairs. This is a development '
        'cohort previously used in multiple experiments, not a new independent test set. '
        'It does not satisfy a fresh 2,000-patient or population-representative validation claim. '
        'See cohort_audit.md/CSV and longitudinal_table.csv for exact counts and timing distributions.\n\n'
        '## 2. Exact targets\n\n'
        'All probes predict existing future_cancer_1y/2y/3y of the **t+1 landmark**, using 365/730/1095 days. '
        'Diagnosis on or before t+1 is excluded. Known later events or adequate cancer-free follow-up '
        'are negatives; inadequate follow-up is missing, never negative. Event timing uses candx_days '
        'minus scr_days{round}; censor timing uses canc_free_days minus scr_days{round}. '
        'The primary gate endpoint was predeclared as cancer within 2 years after t+1; 1y/3y are secondary. '
        'This oracle experiment conditions on an observed future scan and survival without diagnosed '
        'cancer to that landmark; it is not a deployable baseline-screening risk estimate. '
        'Mortality targets are unavailable. Existing binary endpoints are retained; no new survival '
        'task or unvalidated time-dependent metrics were introduced.\n\n'
        '## 3. Splits and leakage controls\n\n'
        'Existing 599/200/200 train/validation/test patient assignments are preserved. One assignment '
        'per patient, source scan split consistency, all cache identities, and predictor pair order '
        'are checked explicitly. Both transitions remain in the same split. Split seed 42 is recorded '
        'as requested; no new split was generated. All scalers and PCA fit training patients only. '
        'Validation masked BCE selects the restored best checkpoint; test is evaluated after fitting '
        f"and never selects models. Seeds: {cfg['seeds']}.\n\n"
        '## 4. Frozen COLIPRI representations\n\n'
        f'Checkpoint SHA256: `{extraction["checkpoint_sha256"]}`; revision `{extraction["revision"]}`. '
        'Reused official RAS/2mm/HU-clamp/192³ preprocessing and raw mean-pooled 864-dimensional '
        'embeddings, matching prior experiments. Full pre-pooling dense features are cached at '
        'outputs/colipri_decodability/cache/dense as [864,24,24,24]. The gate uses existing native '
        'spatial block means as [64,864] (4×4×4, no PCA) to keep attention small; this retains spatial '
        'structure but coarsens the dense grid and can dilute focal lesions. The full tensor is '
        'available for a later resolution ablation. It is not called the untouched dense token sequence. '
        'Mean-pooling equivalence is verified for every scan. No COLIPRI parameters are trained.\n\n'
        '## 5. DiseaseStateProjector\n\n'
        'Linear 864→128 token projection; learned 4×4×4 positional embeddings; K=4 learned queries; '
        '4-head cross-attention; residual LayerNorm; 128→256→128 MLP and residual LayerNorm. '
        'Output S=[4,128]; query identities are shared across scans; deltas are query-wise. '
        'States are flattened (512 dimensions) for probes. Parameters are logged for each training '
        'stage in runs/*/done.json and W&B. Supervision: masked equal-horizon prognosis from the '
        'individual scan plus 0.25 × masked diagnostic BCE for isolation_suspicious, nodule_ge_4mm, '
        'emphysema, fibrosis_or_scarring. All class weights use training scans only. This multi-task '
        'objective does not make stable-state similarity dominant. No state predictor is fitted. '
        'The same current-scan-trained projector is frozen across D0/D1/D2 for each seed; this removes '
        'condition-specific projector learning as a confound. Probe inputs for D0 contain no future scan.\n\n'
        '## 6–7. Experiments, metrics, uncertainty\n\n'+table+'\n\n'
        'Heads: single linear logistic layer or MLP with one 128-unit GELU hidden layer and dropout 0.1. '
        'Current-only and delta-only inputs are zero-padded to the concatenated width so within each '
        'representation space the architectures and parameter counts match. Global and disease inputs '
        'differ in width by design; head family and hidden width remain fixed. Train/val/test metrics, '
        'sample counts, and parameter counts are in per_seed_metrics.csv. AUROC, AUPRC and MCC are '
        'emphasized. Accuracy and threshold measures use 0.5 fixed in advance; weighted BCE '
        'probabilities are not calibrated absolute risks.\n\n'
        f"Mean ± SD describes {len(cfg['seeds'])} independent training seeds. Bootstrap uses {cfg['bootstrap_samples']} paired patient "
        'cluster draws of held-out seed-ensemble predictions and preserves repeated pairs. '
        'Its intervals are conditional on fitted models and do not replace training-seed variability. '
        'Paired differences and disease-minus-global improvement intervals are in paired_comparisons.csv. '
        'No multiplicity adjustment is applied: auxiliary endpoints/ablations are exploratory. '
        'Lack of statistical significance is not evidence of equivalence.\n\n'
        '## 8. Mandatory no-change control\n\n'
        +markdown(baseline)+'\n\n'
        'G3 reuses the existing validation-selected objective A standardized-delta residual MLP, '
        'fixed ensemble of prior predictor seeds 0–4. It takes current embedding and interscan interval; '
        'future scan, event metadata, and diagnostics are not predictor inputs. Pair indices are verified '
        'against its saved manifest. Disease identity metrics are reported separately for each projector '
        'seed; there is no learned disease dynamics predictor in this gate. High future-state cosine '
        'alone does not establish clinically meaningful forecasting. Learned global forecasts must '
        'beat identity reconstruction **and** show downstream prognostic benefit.\n\n'
        '## 9. Representation analysis\n\n'
        'representation_analysis contains per-pair cosine/L2, cancer-stratified summaries, train-fitted '
        'PCA coordinates, and states for all three seeds. plots contains held-out global/disease '
        'trajectories connecting scans of each patient and change distributions/delta PCA colored by '
        'post-future cancer. Scan trajectories use ever-cancer coloring, explicitly distinct from the '
        'landmark-safe pair target. PCA plots are qualitative; geometry does not prove domination '
        'by stable anatomy. Significant-change stratification is unavailable.\n\n'
        '## 10. Ablations and unavailable tasks\n\n'
        'Raw future versus signed delta, delta alone, absolute delta, identity future, linear versus '
        'MLP, global versus disease state, and three projector seeds are evaluated. Lesion comparison '
        'fields are saved without inventing labels; their codes lack a validated local scan-level '
        'significant-change mapping. Progression/change classification and change BCE are unavailable. '
        'Diagnostic supervision is not substituted for radiological progression supervision. '
        'No pseudo-labels are created. Dense-resolution, K=8, and prognosis-only projector ablations '
        'remain untested, so spatial attention versus extra supervision cannot be isolated causally.\n\n'
        '## 11–12. Interpretation and gate recommendation\n\n'
        +markdown(primary)+'\n\n'+markdown(pd.DataFrame(decisions))+'\n\n'
        'A meaningful AUROC advantage requires gain ≥0.02 and a paired 95% interval above zero; '
        'equivalence requires the entire interval inside ±0.02. These criteria were recorded before '
        'training. Discordant heads or intervals covering meaningful gains are inconclusive. '
        'Scenario D requires stronger evidence than D0 alone; because validated change labels and '
        'independent confirmation are absent, an apparent D candidate warrants replication before '
        'building a probabilistic progression model.\n\n'
        '**Information equivalence:** [z_t,z_t+1] and [z_t,z_t+1−z_t] are related by an invertible '
        'linear transformation. The same holds for disease queries. If G2 outperforms G1, it indicates '
        'regularization/conditioning/optimization or finite-sample effects, not stronger information '
        'in the change. High similarity is also compatible with sparse meaningful changes. '
        'Neither failed probes nor PCA by themselves establish that stable anatomy is the cause.\n\n'
        f'**Final recommendation: {recommendation}.** Interpret all evidence within this enriched, '
        'previously inspected cohort; COLIPRI vision pretraining itself used NLST images.\n\n'
        'Reproduce with `PYTHONPATH=src:scripts .venv/bin/python scripts/run_nlst_gate.py '
        '--config results/nlst_gate/configs/gate.yaml`. W&B is logged offline for this run; '
        'CSV histories, checkpoints, scalers, input hashes, environment versions, and predictions '
        'are saved locally. No external service is necessary to reproduce the report.\n')
    (out/'FINAL_GATE_REPORT.md').write_text(report)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'results/nlst_gate/configs/gate.yaml')
    parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    cfg=__import__('yaml').safe_load(args.config.read_text())
    if args.smoke:
        cfg.update(output_dir='results/nlst_gate_smoke',seeds=[42],projector_epochs=2,probe_epochs=2,
                   patience=2,bootstrap_samples=20,experiments=['G0','G1','G2','G3','D0','D1','D2'],wandb_mode='disabled')
    out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    for sub in ['configs','plots','representation_analysis','runs']:(out/sub).mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    logging.getLogger('lightning.pytorch').setLevel(logging.WARNING)
    scans,pairs,z,tokens=load_and_audit(cfg,out)
    if args.audit_only:return
    cfg['checkpoint_sha256']=json.loads(Path(cfg['extraction_metadata']).read_text())['checkpoint_sha256']
    if not args.report_only:
        inputs=[args.config,Path(__file__),ROOT/'src/lung_jepa/models/disease_state.py',
                ROOT/'src/lung_jepa/models/prognostic_adapter.py',ROOT/'scripts/run_colipri_latent_world_model.py',
                Path(cfg['embedding_cache'])/'embeddings.npy',Path(cfg['embedding_cache'])/'embedding_metadata.csv',
                Path(cfg['token_cache']),Path(cfg['extraction_metadata']),Path(cfg['checkpoint']),
                *[Path(cfg[k]) for k in ['subject_manifest','series_manifest','prognostic_labels','diagnostic_manifest','patient_splits']],
                Path(cfg['predictor_dir'])/'selection.json',Path(cfg['predictor_dir'])/'pairs_manifest.csv']
        inputs += [Path(cfg['predictor_dir'])/'predicted_embeddings'/f"{cfg['predictor_config']}__seed{s}.npy" for s in cfg['predictor_seeds']]
        fingerprints={str(p):digest(p) for p in inputs}
        if fingerprints[str(Path(cfg['checkpoint']))] != cfg['checkpoint_sha256']:
            raise ValueError('Cached feature checkpoint fingerprint mismatch')
        if (out/'fingerprints.json').exists() and json.loads((out/'fingerprints.json').read_text())!=fingerprints:
            raise ValueError('Resume refused: source/config/data fingerprint changed; use a new output directory')
        save_json(out/'fingerprints.json',fingerprints)
        save_json(out/'configs/resolved.json',cfg)
        save_json(out/'environment.json',dict(python=sys.version,torch=torch.__version__,lightning=pl.__version__,
            numpy=np.__version__,pandas=pd.__version__,device=torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
            dataset_version='pilot_1000_balanced',split='preserved existing patient assignments',
            spatial_tokens=list(tokens.shape),global_embeddings=list(z.shape)))
        records=train_all(cfg,out,scans,pairs,z,tokens)
    else:records=pd.read_csv(out/'per_seed_metrics.csv')
    bootstrap(cfg,out,pairs)
    analysis(cfg,out,scans,pairs,z)
    report(cfg,out,records)
    save_json(out/'COMPLETE.json',dict(seeds=cfg['seeds'],smoke=args.smoke,
              report=str(out/'FINAL_GATE_REPORT.md'),experiments=cfg['experiments']))
    print('COMPLETE',out,flush=True)


if __name__=='__main__':main()
