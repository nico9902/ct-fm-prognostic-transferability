#!/usr/bin/env python3
"""Resumable preservation -> frozen transition -> matched probe battery."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import lightning as L
import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from lung_jepa.models.preserving_adapter import PreservingAdapter, FrozenSpaceTransition
from run_colipri_latent_world_model import build_pairs, latent_metrics, probe_metrics, seed_all

SCOPES = ['pooled','T0_T1','T1_T2']
TASKS = [f'primary_future_cancer_{h}' for h in ['1y','2y','3y']] + [
    f'diag_t1_{t}' for t in ['isolation_suspicious','nodule_ge_4mm','emphysema','fibrosis_or_scarring']]


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def status(out: Path, **values: object) -> None:
    values['updated_at'] = time.strftime('%Y-%m-%d %H:%M:%S %z')
    save_json(out/'status.json',values)
    print(json.dumps(values),flush=True)


def fit_stage(model: L.LightningModule, train: TensorDataset, val: TensorDataset,
              cfg: dict, directory: Path, name: str, epochs: int, monitor: str) -> L.LightningModule:
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'done.json').exists():
        return type(model).load_from_checkpoint(str(directory/'best.ckpt'),map_location='cpu')
    logger = WandbLogger(project=cfg['wandb_project'],group=cfg['wandb_group'],name=name,
                         save_dir=str(directory),log_model=False)
    logger.experiment.config.update({'battery_config':cfg,'stage_name':name})
    csv = CSVLogger(str(directory),name='history')
    checkpoint = ModelCheckpoint(dirpath=directory,filename='best',monitor=monitor,mode='min',save_top_k=1,
                                 enable_version_counter=False)
    trainer = L.Trainer(accelerator='gpu',devices=1,max_epochs=epochs,
        logger=[logger,csv], callbacks=[checkpoint,EarlyStopping(monitor=monitor,patience=cfg['patience'],
        min_delta=1e-6,mode='min')],deterministic=True,enable_progress_bar=False,
        enable_model_summary=False,log_every_n_steps=1,default_root_dir=str(directory))
    trainer.fit(model,DataLoader(train,batch_size=cfg['batch_size'],shuffle=True),
                 DataLoader(val,batch_size=cfg['batch_size'],shuffle=False))
    save_json(directory/'done.json',{'checkpoint':checkpoint.best_model_path,
                                   'best_validation':float(checkpoint.best_model_score),
                                   'epochs':trainer.current_epoch,'wandb_url':logger.experiment.url})
    logger.experiment.finish()
    model.cpu()
    return type(model).load_from_checkpoint(str(directory/'best.ckpt'),map_location='cpu')


def encode_all(model: PreservingAdapter | None, x: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    if model is None:
        return x.copy(),x.copy()
    model.eval().requires_grad_(False).cuda()
    latent, reconstructed = [], []
    with torch.inference_mode():
        for chunk in np.array_split(x,max(1,int(np.ceil(len(x)/256)))):
            z,r = model(torch.tensor(chunk,device='cuda'))
            latent.append(z.cpu().numpy()); reconstructed.append(r.cpu().numpy())
    return np.concatenate(latent),np.concatenate(reconstructed)


def decode(model: PreservingAdapter | None, z: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    if model is None:
        return (z*scale+mean).astype(np.float32)
    with torch.inference_mode():
        parts = [model.decoder(torch.tensor(chunk,device='cuda')).cpu().numpy()
                 for chunk in np.array_split(z,max(1,int(np.ceil(len(z)/256))))]
    return (np.concatenate(parts)*scale+mean).astype(np.float32)


def forecast_metrics(pred: np.ndarray, target: np.ndarray, current: np.ndarray) -> dict:
    delta, truth = pred-current, target-current
    cosine = np.sum(delta*truth,axis=1)/(np.linalg.norm(delta,axis=1)*np.linalg.norm(truth,axis=1)+1e-12)
    return {**latent_metrics(pred,target),'r2':float(r2_score(target,pred,multioutput='variance_weighted')),
            'delta_mse':float(np.mean((delta-truth)**2)),'delta_cosine':float(cosine.mean())}


def export_variant(cfg: dict, v: dict, seed: int, pairs: pd.DataFrame, h: np.ndarray, x: np.ndarray,
                   scan_split: np.ndarray, mean: np.ndarray, scale: np.ndarray,
                   dt: np.ndarray, out: Path) -> Path:
    name = f'{v["name"]}__seed{seed}'
    work = out/'runs'/name
    work.mkdir(parents=True,exist_ok=True)
    export = work/'representations.npz'
    if export.exists() and (work/'export_complete.json').exists():
        return export
    seed_all(seed)
    adapter = None
    if not v.get('native',False):
        status(out,stage='adapter',run=name)
        adapter_hidden = max(cfg['hidden_dim'],v['dim']) if v['dim']<h.shape[1] else cfg['hidden_dim']
        adapter = PreservingAdapter(input_dim=h.shape[1],latent_dim=v['dim'],hidden_dim=adapter_hidden,
            decoder_hidden=cfg['decoder_hidden'],gamma=cfg['gamma'],anchor_weight=cfg['anchor_weight'],
            sigreg_weight=v['sigreg'],projections=cfg['sigreg_projections'],lr=cfg['lr'])
        adapter = fit_stage(adapter,TensorDataset(torch.from_numpy(x[scan_split=='train'])),
            TensorDataset(torch.from_numpy(x[scan_split=='val'])),cfg,work/'adapter',name+'/adapter',
            cfg['adapter_epochs'],'val/preservation')
    z,reconstructed = encode_all(adapter,x)
    np.savez(work/'scan_states.npz',z=z,reconstructed_h=reconstructed*scale+mean,
             embedding_index=np.arange(len(h)),split=scan_split.astype(str))
    current, target = z[pairs.idx_t],z[pairs.idx_t1]
    train, val = pairs.split.eq('train').to_numpy(),pairs.split.eq('val').to_numpy()
    delta = target-current
    ds = StandardScaler().fit(delta[train])
    np.savez(work/'delta_scaler.npz',mean=ds.mean_,scale=ds.scale_)
    # Predictor initialization and loader seeds are matched across variants.
    seed_all(seed)
    transition = FrozenSpaceTransition(v['dim'],cfg['hidden_dim'],cfg['lr'])
    transition.delta_mean.copy_(torch.tensor(ds.mean_,dtype=torch.float32))
    transition.delta_scale.copy_(torch.tensor(ds.scale_,dtype=torch.float32))
    status(out,stage='transition',run=name,adapter_frozen=True)
    loaders = {s:TensorDataset(*(torch.from_numpy(a[m]) for a in [current,dt,target]))
               for s,m in [('train',train),('val',val)]}
    transition = fit_stage(transition,loaders['train'],loaders['val'],cfg,work/'transition',
                           name+'/transition',cfg['transition_epochs'],'val/loss')
    transition.eval().requires_grad_(False).cuda()
    if adapter is not None:
        assert all(not p.requires_grad for p in adapter.parameters())
    with torch.inference_mode():
        predicted = np.concatenate([transition(torch.tensor(current[i:i+256],device='cuda'),
            torch.tensor(dt[i:i+256],device='cuda')).cpu().numpy() for i in range(0,len(pairs),256)])
    reps = {'current':current,'predicted_future':predicted,'actual_future':target,
        'decoded_current':decode(adapter,current,mean,scale),
        'decoded_predicted':decode(adapter,predicted,mean,scale),
        'decoded_actual':decode(adapter,target,mean,scale)}
    assert all(len(a)==len(pairs) and np.isfinite(a).all() for a in reps.values())
    ids = (pairs.patient_id.astype(str)+':'+pairs.pair_type).to_numpy(dtype=str)
    assert len(np.unique(ids))==len(ids)
    np.savez(export,**reps,pair_ids=ids,idx_t=pairs.idx_t.to_numpy(),idx_t1=pairs.idx_t1.to_numpy())
    rows=[]
    for split in ['train','val','test']:
        mask = pairs.split.eq(split).to_numpy()
        for scope in SCOPES:
            m=mask if scope=='pooled' else mask & pairs.pair_type.eq(scope).to_numpy()
            for space,pred,targ,cur in [('latent',predicted,target,current),
                    ('native_decoded',reps['decoded_predicted'],h[pairs.idx_t1],h[pairs.idx_t])]:
                for reference,value in [('prediction',pred),('identity',cur)]:
                    rows.append({'model':v['name'],'wm_seed':seed,'split':split,'scope':scope,
                        'space':space,'reference':reference,**forecast_metrics(value[m],targ[m],cur[m])})
        for key,value in [('current',current),('actual_future',target)]:
            std=value[mask].std(0)
            save_json(work/f'{split}_{key}_distribution.json',{'mean_feature_std':float(std.mean()),
                'min_feature_std':float(std.min()),'mean_norm':float(np.linalg.norm(value[mask],axis=1).mean()),
                'between_sample_energy_fraction':float(std.dot(std)/np.mean(np.sum(value[mask]**2,axis=1)))})
    pd.DataFrame(rows).to_csv(work/'latent_metrics.csv',index=False)
    save_json(work/'export_complete.json',{'rows':len(pairs),'ids_verified':True,
        'adapter_checkpoint_sha256':digest(work/'adapter/best.ckpt') if adapter is not None else None,
        'transition_checkpoint_sha256':digest(work/'transition/best.ckpt')})
    transition.cpu()
    if adapter is not None: adapter.cpu()
    torch.cuda.empty_cache()
    return export


def fit_probe(x: np.ndarray, pairs: pd.DataFrame, task: str, seed: int, cfg: dict,
              path: Path) -> dict:
    """Matched historical mini-batch linear probe, with train-only scaling."""
    if (path/'done.json').exists():
        return json.loads((path/'done.json').read_text())
    path.mkdir(parents=True,exist_ok=True)
    seed_all(seed)
    masks={s:pairs.split.eq(s).to_numpy() & pairs[task].notna().to_numpy() for s in ['train','val','test']}
    ys={s:pairs.loc[m,task].to_numpy(dtype=np.float32) for s,m in masks.items()}
    if any(len(np.unique(y))<2 for y in ys.values()):
        raise ValueError(f'{task}: both classes are required in each pooled split')
    scaler=StandardScaler().fit(x[masks['train']])
    xs={s:torch.tensor(scaler.transform(x[m]),dtype=torch.float32,device='cuda') for s,m in masks.items()}
    yy={s:torch.tensor(y,device='cuda') for s,y in ys.items()}
    clf=torch.nn.Linear(x.shape[1],1,device='cuda')
    opt=torch.optim.AdamW(clf.parameters(),lr=.001,weight_decay=.0001)
    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=((1-yy['train']).sum()/yy['train'].sum()).reshape(1))
    best,stale,state=float('inf'),0,None
    for epoch in range(cfg['probe_epochs']):
        order=torch.randperm(len(ys['train']),generator=torch.Generator().manual_seed(seed)).cuda()
        for indices in order.split(128):
            opt.zero_grad(set_to_none=True)
            criterion(clf(xs['train'][indices]).squeeze(1),yy['train'][indices]).backward()
            opt.step()
        with torch.no_grad(): vl=float(criterion(clf(xs['val']).squeeze(1),yy['val']))
        if vl<best-1e-5:
            best,stale=vl,0
            state={k:v.detach().cpu().clone() for k,v in clf.state_dict().items()}
        else: stale+=1
        if stale>=cfg['probe_patience']: break
    clf.load_state_dict(state)
    result={'validation_loss':best,'epochs':epoch+1,'rows':[]}
    probs={}
    for split in ['val','test']:
        with torch.no_grad(): prob=clf(xs[split]).squeeze(1).sigmoid().cpu().numpy()
        probs[split+'_probability']=prob
        probs[split+'_pair_ids']=(pairs.loc[masks[split],'patient_id'].astype(str)+':'+pairs.loc[masks[split],'pair_type']).to_numpy(dtype=str)
        probs[split+'_label']=ys[split]
        types=pairs.loc[masks[split],'pair_type'].to_numpy()
        for scope in SCOPES:
            m=np.ones(len(prob),bool) if scope=='pooled' else types==scope
            metrics=probe_metrics(ys[split][m],prob[m]) if len(np.unique(ys[split][m]))==2 else {
                k:float('nan') for k in ['auroc','auprc','mcc','f1']}
            result['rows'].append({'split':split,'scope':scope,'task':task,'probe_seed':seed,
                'n':int(m.sum()),'positives':int(ys[split][m].sum()),**metrics})
    torch.save({'state_dict':state,'feature_mean':scaler.mean_,'feature_scale':scaler.scale_,
                'feature_dim':x.shape[1],'seed':seed},path/'probe.pt')
    np.savez(path/'probabilities.npz',**probs)
    save_json(path/'done.json',result)
    return result


def summarize(out: Path, cfg: dict) -> None:
    frame=pd.read_csv(out/'downstream_metrics.csv')
    summary=frame.groupby(['model','representation','split','scope','task'])[['auroc','auprc','mcc','f1']].agg(['mean','std','count'])
    summary.to_csv(out/'downstream_summary.csv')
    latent=pd.concat([pd.read_csv(p) for p in (out/'runs').glob('*/latent_metrics.csv')],ignore_index=True)
    latent.to_csv(out/'latent_per_seed.csv',index=False)
    latent.groupby(['model','split','scope','space','reference'])[['mse','cosine_similarity','l2_distance','delta_mse','delta_cosine','r2']].agg(['mean','std']).to_csv(out/'latent_summary.csv')
    val=frame[(frame.split=='val')&(frame.scope=='pooled')&frame.task.str.startswith('primary_')]
    reference=val[(val.model=='native_controls')&(val.representation=='current')].groupby('task').auroc.mean()
    gate=[]
    for (model,rep,task),value in val.groupby(['model','representation','task']).auroc.mean().items():
        if rep!='current': continue
        gap=float(value-reference[task])
        gate.append({'model':model,'task':task,'val_auroc':value,'native_val_auroc':reference[task],
            'gap':gap,'preserved_within_margin':gap>=-cfg['validation_auroc_margin']})
    pd.DataFrame(gate).to_csv(out/'validation_preservation_gate.csv',index=False)
    # Paired seed-level gains; not confidence intervals over independent patients.
    keys=['model','wm_seed','task','probe_seed','scope','split']
    current=frame[frame.representation=='current'][keys+['auroc','auprc']]
    pred=frame[frame.representation=='predicted_future'][keys+['auroc','auprc']]
    gains=pred.merge(current,on=keys,suffixes=('_prediction','_identity'),validate='one_to_one')
    for metric in ['auroc','auprc']: gains[metric+'_gain']=gains[metric+'_prediction']-gains[metric+'_identity']
    gains.to_csv(out/'paired_gains_over_identity.csv',index=False)
    test=frame[(frame.split=='test')&(frame.scope=='pooled')&frame.task.str.startswith('primary_')]
    table=test.groupby(['model','representation','task']).auroc.mean().unstack('task')
    with (out/'RESULTS.md').open('w') as f:
        f.write('# Preservation adapter battery results\n\nCompleted; all figures below use matched probes and ordered exports.\n\n')
        f.write('```text\n'+table.round(4).to_string()+'\n```\n\n')
        f.write('See validation_preservation_gate.csv before interpreting future prediction. '
                'Transition-specific metrics use the pooled-fitted probe. Seed SD is not a patient confidence interval. '
                'Diagnostic targets are consistently t+1 labels.\n')


def main() -> None:
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/model/colipri_preserving_adapter.yaml')
    parser.add_argument('--output-dir',type=Path)
    a=parser.parse_args()
    cfg=yaml.safe_load(a.config.read_text())
    if a.output_dir: cfg['output_dir']=str(a.output_dir)
    out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available(): raise RuntimeError('GPU required, no CPU fallback')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    if (out/'config.json').exists():
        assert json.loads((out/'config.json').read_text())==cfg,'Refusing to mix configurations'
    else: save_json(out/'config.json',cfg)
    if (out/'COMPLETE.json').exists():
        print('Battery already complete',flush=True);return
    paths=argparse.Namespace(**{k:Path(cfg[k]) for k in ['embedding_cache','prognostic_labels','diagnostic_manifest','patient_splits']})
    pairs,h=build_pairs(paths)
    keys=['patient_id','pair_type','idx_t','idx_t1','split']
    reference=pd.read_csv(cfg['reference_pairs'],dtype={'patient_id':str})
    pd.testing.assert_frame_equal(pairs[keys],reference[keys])
    assert pairs.groupby('patient_id').split.nunique().max()==1
    em=pd.read_csv(paths.embedding_cache/'embedding_metadata.csv',dtype=str)
    splits=pd.read_csv(paths.patient_splits,dtype=str)
    em=em.merge(splits[['patient_id','split']],on='patient_id',validate='many_to_one')
    em['embedding_index']=em.embedding_index.astype(int)
    em=em.sort_values('embedding_index').reset_index(drop=True)
    np.testing.assert_array_equal(em.embedding_index,np.arange(len(h)))
    assert em.split.notna().all() and np.isfinite(h).all()
    scaler=StandardScaler().fit(h[em.split.eq('train')])
    x=scaler.transform(h).astype(np.float32)
    mean,scale=scaler.mean_.astype(np.float32),scaler.scale_.astype(np.float32)
    ts=StandardScaler().fit(pairs.loc[pairs.split.eq('train'),['dt_days']])
    dt=ts.transform(pairs[['dt_days']]).astype(np.float32)
    np.savez(out/'scalers.npz',feature_mean=mean,feature_scale=scale,time_mean=ts.mean_,time_scale=ts.scale_,
             train_scan_indices=em.loc[em.split.eq('train'),'embedding_index'].to_numpy())
    pairs.to_csv(out/'pairs_manifest.csv',index=False)
    em.to_csv(out/'scan_manifest.csv',index=False)
    sources=[Path(__file__),ROOT/'src/lung_jepa/models/preserving_adapter.py',a.config]
    save_json(out/'provenance.json',{'pid':os.getpid(),'gpu':torch.cuda.get_device_name(),
        'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'sources':{str(p):digest(p) for p in sources},'patient_splits_sha256':digest(paths.patient_splits),
        'pairs':len(pairs),'unique_patients':pairs.patient_id.nunique(),'scan_split_counts':em.split.value_counts().to_dict()})
    exports=[]
    # Train all predefined variants without downstream test-based selection.
    for v in cfg['variants']:
        for seed in cfg['seeds']:
            exports.append((v['name'],seed,export_variant(cfg,v,seed,pairs,h,x,em.split.to_numpy(),mean,scale,dt,out)))
    status(out,stage='downstream',completed_training_runs=len(exports))
    run=wandb.init(project=cfg['wandb_project'],group=cfg['wandb_group'],name='downstream-'+out.name,
                   dir=str(out),config=cfg)
    rows=[]
    native={'current':h[pairs.idx_t],'actual_future':h[pairs.idx_t1]}
    all_reps=[('native_controls',-1,native)]+exports
    expected_ids=(pairs.patient_id.astype(str)+':'+pairs.pair_type).to_numpy(dtype=str)
    done=0
    for model,seed,entry in all_reps:
        if isinstance(entry,Path):
            data=np.load(entry)
            np.testing.assert_array_equal(data['pair_ids'],expected_ids)
            np.testing.assert_array_equal(data['idx_t'],pairs.idx_t)
            reps={k:data[k] for k in ['current','predicted_future','actual_future','decoded_current','decoded_predicted','decoded_actual']}
        else: reps=entry
        for rep,features in reps.items():
            array_hash=hashlib.sha256(features.astype(np.float32).tobytes()).hexdigest()
            for task in cfg.get('tasks',TASKS):
                for probe_seed in cfg['probe_seeds']:
                    # Exact feature equality allows safe reuse of identical controls.
                    cache=out/'probes'/array_hash/task/f'seed{probe_seed}'
                    result=fit_probe(features,pairs,task,probe_seed,cfg,cache)
                    rows.extend([{'model':model,'wm_seed':seed,'representation':rep,'cache':str(cache),**r} for r in result['rows']])
                    done+=1
                    run.log({'completed_probe_evaluations':done,'validation_loss':result['validation_loss'],
                             'model':model,'representation':rep,'task':task})
                pd.DataFrame(rows).to_csv(out/'downstream_metrics.csv',index=False)
                status(out,stage='downstream',model=model,wm_seed=seed,representation=rep,task=task,completed_probes=done)
    summarize(out,cfg)
    summary=pd.read_csv(out/'downstream_metrics.csv')
    compact=summary.groupby(['model','representation','split','scope','task'])[['auroc','auprc','mcc','f1']].mean().reset_index()
    run.log({'summary':wandb.Table(dataframe=compact),
             'latent':wandb.Table(dataframe=pd.read_csv(out/'latent_per_seed.csv')),
             'validation_gate':wandb.Table(dataframe=pd.read_csv(out/'validation_preservation_gate.csv'))})
    run.finish()
    save_json(out/'COMPLETE.json',{'training_runs':len(exports),'probe_evaluations':done})
    status(out,stage='complete',completed_probes=done)


if __name__=='__main__':
    main()
