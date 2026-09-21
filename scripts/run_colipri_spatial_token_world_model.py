#!/usr/bin/env python3
"""Compact spatial/token-level frozen-COLIPRI world-model experiment."""
from __future__ import annotations

import argparse
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_colipri_latent_world_model import DIAGNOSTIC_TASKS, HORIZONS, build_pairs, latent_metrics, probe_metrics, seed_all

DIM_IN, DIM, TOKENS, SEEDS = 864, 256, 64, [0, 1, 2, 3, 4]


def args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dense-cache', type=Path, default=Path('outputs/colipri_decodability/cache/dense'))
    p.add_argument('--output-dir', type=Path, default=Path('outputs/colipri_spatial_token_world_model'))
    p.add_argument('--previous-output', type=Path, default=Path('outputs/colipri_latent_world_model'))
    p.add_argument('--device', default='auto'); p.add_argument('--epochs', type=int, default=150); p.add_argument('--patience', type=int, default=20); p.add_argument('--batch-size', type=int, default=16); p.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    p.add_argument('--smoke-only', action='store_true'); p.add_argument('--no-wandb', action='store_true'); p.add_argument('--wandb-project', default='nlst-colipri-latent-world-model'); p.add_argument('--wandb-group', default='spatial-token-world-model')
    return p.parse_args()


def pool_grid(path: Path) -> np.ndarray:
    x = np.load(path, mmap_mode='r')
    if x.shape != (864, 24, 24, 24): raise ValueError(f'{path}: expected (864,24,24,24), got {x.shape}')
    # [C, 4*6, 4*6, 4*6] -> block mean -> [4,4,4,C] -> [64,C]
    y = np.asarray(x, dtype=np.float32).reshape(864, 4, 6, 4, 6, 4, 6).mean(axis=(2, 4, 6)).transpose(1, 2, 3, 0)
    return y.reshape(TOKENS, DIM_IN).astype(np.float32)


def prepare_tokens(a, em, splits):
    out = a.output_dir / 'cached_spatial_tokens.npy'; meta = a.output_dir / 'spatial_token_metadata.json'
    if out.exists() and meta.exists(): return np.load(out, mmap_mode='r'), json.loads(meta.read_text())
    train_ids = set(splits.loc[splits.split.eq('train'), 'patient_id'])
    train_indices = em.loc[em.patient_id.isin(train_ids), 'embedding_index'].astype(int).tolist()
    ipca = IncrementalPCA(n_components=DIM, batch_size=2048)
    buf=[]
    for j, idx in enumerate(train_indices):
        buf.append(pool_grid(a.dense_cache / f'{idx:05d}.npy'))
        if sum(x.shape[0] for x in buf) >= 2048 or j == len(train_indices)-1:
            ipca.partial_fit(np.concatenate(buf, axis=0)); buf=[]
        if (j+1) % 200 == 0: print(f'PCA fit {j+1}/{len(train_indices)}', flush=True)
    tokens = np.lib.format.open_memmap(out, mode='w+', dtype='float32', shape=(len(em), TOKENS, DIM))
    for j, idx in enumerate(em.embedding_index.astype(int)):
        tokens[j] = ipca.transform(pool_grid(a.dense_cache / f'{idx:05d}.npy')).astype(np.float32)
        if (j+1) % 200 == 0: print(f'PCA transform {j+1}/{len(em)}', flush=True)
    tokens.flush(); np.savez(a.output_dir / 'token_pca.npz', components=ipca.components_, mean=ipca.mean_, explained_variance=ipca.explained_variance_, explained_variance_ratio=ipca.explained_variance_ratio_)
    info={'n_scans':len(em),'shape':[len(em),TOKENS,DIM],'source_shape':[864,24,24,24],'spatial_pool':'24x24x24 -> 4x4x4 block mean','pca_components':DIM,'pca_fit':'all tokens from training patients only','train_scan_count':len(train_indices)}; meta.write_text(json.dumps(info,indent=2)); return tokens, info


class SpatialTokenWorldModel(pl.LightningModule):
    def __init__(self, delta_mean, delta_scale, lr=1e-3):
        super().__init__(); self.save_hyperparameters(ignore=['delta_mean','delta_scale']); self.register_buffer('delta_mean', torch.tensor(delta_mean, dtype=torch.float32)); self.register_buffer('delta_scale', torch.tensor(delta_scale, dtype=torch.float32))
        self.input_norm=nn.LayerNorm(DIM); self.time_embed=nn.Sequential(nn.Linear(1,32),nn.GELU(),nn.Linear(32,DIM)); self.pos=nn.Parameter(torch.zeros(1,TOKENS,DIM)); nn.init.trunc_normal_(self.pos,std=.02)
        layer=nn.TransformerEncoderLayer(d_model=DIM,nhead=4,dim_feedforward=4*DIM,dropout=.1,activation='gelu',batch_first=True,norm_first=True); self.encoder=nn.TransformerEncoder(layer,num_layers=4); self.head=nn.Sequential(nn.LayerNorm(DIM),nn.Linear(DIM,DIM)); self.lr=lr
    def forward(self, h, dt):
        x=self.input_norm(h)+self.pos+self.time_embed(dt).unsqueeze(1); return self.head(self.encoder(x))
    def step(self,batch,stage):
        h,dt,y=batch; dnorm=self(h,dt); loss=nn.functional.mse_loss(dnorm,y); dhat=dnorm*self.delta_scale+self.delta_mean; pred=h+dhat; target=h+y*self.delta_scale+self.delta_mean; fmse=nn.functional.mse_loss(pred,target); cos=(1-nn.functional.cosine_similarity(pred.reshape(len(pred),-1),target.reshape(len(target),-1),dim=-1)).mean(); self.log(f'{stage}_loss',loss,on_epoch=True,prog_bar=False); self.log(f'{stage}_forecast_mse',fmse,on_epoch=True,prog_bar=False); self.log(f'{stage}_forecast_cosine_loss',cos,on_epoch=True,prog_bar=False); return loss
    def training_step(self,batch,batch_idx): return self.step(batch,'train')
    def validation_step(self,batch,batch_idx): self.step(batch,'val')
    def configure_optimizers(self): return torch.optim.AdamW(self.parameters(),lr=self.lr,weight_decay=1e-4)


def smoke(device):
    model=SpatialTokenWorldModel(np.zeros(DIM),np.ones(DIM)).to(device); h=torch.randn(2,TOKENS,DIM,device=device); dt=torch.randn(2,1,device=device); y=torch.randn(2,TOKENS,DIM,device=device); pred=model(h,dt); loss=nn.functional.mse_loss(pred,y); print(json.dumps({'H_t_shape':list(h.shape),'dt_shape':list(dt.shape),'H_hat_t1_shape':list(pred.shape),'H_t1_shape':list(y.shape),'loss':float(loss)},indent=2),flush=True)


def train_seed(a, h, pairs, seed, delta_mean, delta_scale, dt_scaler, device, logger):
    seed_all(seed); tr=pairs.split.eq('train').to_numpy(); va=pairs.split.eq('val').to_numpy(); h_t=h[pairs.idx_t.to_numpy()]; h_t1=h[pairs.idx_t1.to_numpy()]; delta=h_t1-h_t; y=((delta-delta_mean)/delta_scale).astype(np.float32); dt=dt_scaler.transform(pairs[['dt_days']]).astype(np.float32)
    train=TensorDataset(torch.from_numpy(h_t[tr]),torch.from_numpy(dt[tr]),torch.from_numpy(y[tr])); val=TensorDataset(torch.from_numpy(h_t[va]),torch.from_numpy(dt[va]),torch.from_numpy(y[va])); gen=torch.Generator().manual_seed(seed); dl=DataLoader(train,batch_size=a.batch_size,shuffle=True,generator=gen); vl=DataLoader(val,batch_size=a.batch_size)
    model=SpatialTokenWorldModel(delta_mean,delta_scale).to(device); ck=a.output_dir/'checkpoints'/f'seed{seed}'; ck.mkdir(parents=True,exist_ok=True); cb=ModelCheckpoint(dirpath=str(ck),monitor='val_forecast_mse',mode='min',save_top_k=1,filename='best'); es=EarlyStopping(monitor='val_forecast_mse',mode='min',patience=a.patience)
    trainer=pl.Trainer(accelerator='gpu' if device.type=='cuda' else 'cpu',devices=1,max_epochs=a.epochs,logger=logger,callbacks=[cb,es],enable_progress_bar=False,enable_model_summary=False,deterministic=True)
    trainer.fit(model,dl,vl); best=SpatialTokenWorldModel.load_from_checkpoint(cb.best_model_path,delta_mean=delta_mean,delta_scale=delta_scale).to(device); best.eval()
    with torch.inference_mode(): pred_norm=best(torch.from_numpy(h_t).to(device),torch.from_numpy(dt).to(device)).cpu().numpy()
    pred=h_t+pred_norm*delta_scale+delta_mean; np.save(a.output_dir/'predictions'/f'seed{seed}.npy',pred); np.save(a.output_dir/'predictions'/f'delta_seed{seed}.npy',pred-h_t); return pred,{'seed':seed,'best_checkpoint':cb.best_model_path,'best_val_forecast_mse':cb.best_model_score.item()}


def latent_metrics_table(a,pairs,h,preds,identity):
    true=h[pairs.idx_t1.to_numpy()]; true_delta=true-h[pairs.idx_t.to_numpy()]; rows=[]
    for scope,mask in [('pooled',np.ones(len(pairs),bool)),('T0_T1',pairs.pair_type.eq('T0_T1').to_numpy()),('T1_T2',pairs.pair_type.eq('T1_T2').to_numpy())]:
      mask &= pairs.split.eq('test').to_numpy()
      for name,p in {'identity':identity,**preds,'actual_H_t1':true}.items():
       if name=='actual_H_t1': row={'scope':scope,'representation':name,'mse':0.,'cosine':1.,'l2':0.,'delta_mse':0.,'delta_cosine':1.}
       else:
        d=p-h[pairs.idx_t.to_numpy()]; lm=latent_metrics(p[mask].mean(1),true[mask].mean(1)); dm=latent_metrics(d[mask].reshape(mask.sum(),-1),true_delta[mask].reshape(mask.sum(),-1)); row={'scope':scope,'representation':name,'mse':float(np.mean((p[mask]-true[mask])**2)),'cosine':float(np.mean(np.sum(p[mask]*true[mask],(1,2))/(np.linalg.norm(p[mask].reshape(mask.sum(),-1),axis=1)*np.linalg.norm(true[mask].reshape(mask.sum(),-1),axis=1)+1e-12))),'l2':float(np.mean(np.linalg.norm((p[mask]-true[mask]).reshape(mask.sum(),-1),axis=1))),'delta_mse':float(np.mean((d[mask]-true_delta[mask])**2)),'delta_cosine':float(np.mean(np.sum(d[mask]*true_delta[mask],(1,2))/(np.linalg.norm(d[mask].reshape(mask.sum(),-1),axis=1)*np.linalg.norm(true_delta[mask].reshape(mask.sum(),-1),axis=1)+1e-12)))}
       rows.append(row)
    pd.DataFrame(rows).to_csv(a.output_dir/'latent_per_seed_metrics.csv',index=False)


def probes(a,pairs,reps,device,out_name):
    rows=[]
    for scope,sm in [('pooled',np.ones(len(pairs),bool)),('T0_T1',pairs.pair_type.eq('T0_T1').to_numpy()),('T1_T2',pairs.pair_type.eq('T1_T2').to_numpy())]:
      for rep,x in reps.items():
       x=x.mean(1) if x.ndim==3 else x
       for label in [f'primary_future_cancer_{h}' for h in HORIZONS]+[f'diag_t1_{d}' for d in DIAGNOSTIC_TASKS]:
        valid=sm & pairs[label].notna().to_numpy(); sub=pairs.loc[valid].reset_index(drop=True); y=sub[label].to_numpy(float); split=sub.split.to_numpy(); tr,va,te=split=='train',split=='val',split=='test'
        if tr.sum()==0 or te.sum()==0 or len(np.unique(y[tr]))<2 or y[te].sum()==0: continue
        sc=StandardScaler().fit(x[valid][tr]); xs=sc.transform(x[valid]).astype(np.float32)
        for ps in a.seeds:
         seed_all(ps); clf=nn.Linear(xs.shape[1],1).to(device); pos=max(y[tr].sum(),1); neg=max(tr.sum()-y[tr].sum(),1); lf=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/pos],device=device)); opt=torch.optim.AdamW(clf.parameters(),lr=1e-3,weight_decay=1e-4); ds=TensorDataset(torch.from_numpy(xs[tr]),torch.from_numpy(y[tr].astype(np.float32))); best=(1e9,None); stale=0
         for _ in range(100):
          clf.train()
          for xb,yb in DataLoader(ds,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(ps)):
           opt.zero_grad(); loss=lf(clf(xb.to(device)).squeeze(1),yb.to(device)); loss.backward(); opt.step()
          clf.eval();
          with torch.inference_mode(): vl=float(lf(clf(torch.from_numpy(xs[va]).to(device)).squeeze(1),torch.from_numpy(y[va].astype(np.float32)).to(device)))
          if vl<best[0]-1e-5: best=(vl,{k:v.detach().cpu().clone() for k,v in clf.state_dict().items()}); stale=0
          else: stale+=1
          if stale>=15: break
         clf.load_state_dict(best[1]); clf.eval();
         with torch.inference_mode(): pr=torch.sigmoid(clf(torch.from_numpy(xs[te]).to(device)).squeeze(1)).cpu().numpy()
         rows.append({'scope':scope,'representation':rep,'label':label,'probe_seed':ps,'n_test':int(te.sum()),'n_test_positive':int(y[te].sum()),**probe_metrics(y[te],pr)})
    pd.DataFrame(rows).to_csv(a.output_dir/out_name,index=False)


def main():
    a=args(); a.output_dir.mkdir(parents=True,exist_ok=True); [ (a.output_dir/d).mkdir(exist_ok=True) for d in ['predictions','checkpoints','training_history'] ]; device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device)
    base=Path('outputs/colipri_decodability'); em=pd.read_csv(base/'mean/cache/embedding_metadata.csv',dtype=str); splits=pd.read_csv(base/'patient_splits.csv',dtype=str)
    em['embedding_index']=em.embedding_index.astype(int); pairs,z=build_pairs(argparse.Namespace(embedding_cache=base/'mean/cache',prognostic_labels=Path('data/nlst/metadata/nlst/prognostic_labels/pilot_1000_balanced/nlst_scan_prognostic_labels.csv'),diagnostic_manifest=Path('outputs/colipri_decodability/experiment_scan_manifest.csv'),patient_splits=base/'patient_splits.csv')); pairs.to_csv(a.output_dir/'pairs_manifest.csv',index=False)
    h,info=prepare_tokens(a,em,splits); delta=(h[pairs.idx_t1.to_numpy()]-h[pairs.idx_t.to_numpy()]); tr=pairs.split.eq('train').to_numpy(); ds=StandardScaler().fit(delta[tr].reshape(-1,DIM)); dt_scaler=StandardScaler().fit(pairs.loc[tr,['dt_days']]); np.savez(a.output_dir/'delta_scaler.npz',mean=ds.mean_,scale=ds.scale_); np.savez(a.output_dir/'time_scaler.npz',mean=dt_scaler.mean_,scale=dt_scaler.scale_); (a.output_dir/'run_metadata.json').write_text(json.dumps({'device':str(device),'token_info':info,'pairs':len(pairs),'architecture':'4-layer Transformer, 4 heads, dim 256, ff 1024, dropout .1'},indent=2))
    smoke(device)
    if a.smoke_only:return
    pred={}; rows=[]; identity=h[pairs.idx_t.to_numpy()]
    for seed in a.seeds:
      logger=None
      if not a.no_wandb:
       logger=WandbLogger(project=a.wandb_project,group=a.wandb_group,name=f'spatial_token_seed{seed}',save_dir=str(a.output_dir/'wandb'),log_model=False); logger.experiment.config.update({'seed':seed,'tokens':64,'input_dim':864,'pca_dim':256,'layers':4,'heads':4,'objective':'standardized_token_delta_mse'})
      p,r=train_seed(a,h,pairs,seed,ds.mean_,ds.scale_,dt_scaler,device,logger); pred[f'spatial_transformer__seed{seed}']=p; rows.append(r)
      if logger is not None: logger.experiment.summary.update(r); logger.experiment.finish()
    pd.DataFrame(rows).to_csv(a.output_dir/'training_per_seed_metrics.csv',index=False); latent_metrics_table(a,pairs,h,pred,identity)
    prev={f'previous_pooled_mlp__seed{s}':np.load(a.previous_output/'predicted_embeddings'/f'seed{s}.npy') for s in a.seeds}; reps={'current_H':h[pairs.idx_t.to_numpy()],'actual_H_t1':h[pairs.idx_t1.to_numpy()],**prev,**pred}; probes(a,pairs,reps,device,'downstream_per_seed_metrics.csv')
    pd.DataFrame({'representation':list(reps),'shape':[list(x.shape) for x in reps.values()]}).to_csv(a.output_dir/'representation_manifest.csv',index=False); print(json.dumps({'device':str(device),'pairs':len(pairs),'token_shape':list(h.shape),'representations':list(reps)},indent=2),flush=True)


if __name__=='__main__':main()
