#!/usr/bin/env python3
"""No-PCA native 864-d spatial-token COLIPRI world model."""
from __future__ import annotations
import argparse, json, os, random, sys
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np, pandas as pd, torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_colipri_latent_world_model import DIAGNOSTIC_TASKS, HORIZONS, build_pairs, probe_metrics, seed_all

TOKENS, DIM, SEEDS = 64, 864, [0,1,2,3,4]

def parse():
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dense-cache',type=Path,default=Path('outputs/colipri_decodability/cache/dense'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/colipri_native_spatial_token_world_model'))
    p.add_argument('--previous-pooled-output',type=Path,default=Path('outputs/colipri_latent_world_model'))
    p.add_argument('--previous-pca-output',type=Path,default=Path('outputs/colipri_spatial_token_world_model'))
    p.add_argument('--device',default='auto'); p.add_argument('--epochs',type=int,default=150); p.add_argument('--patience',type=int,default=20); p.add_argument('--batch-size',type=int,default=8); p.add_argument('--seeds',nargs='+',type=int,default=SEEDS); p.add_argument('--no-wandb',action='store_true'); p.add_argument('--wandb-project',default='nlst-colipri-latent-world-model'); p.add_argument('--wandb-group',default='native-864-spatial-token')
    return p.parse_args()

def pool_grid(path):
    x=np.load(path,mmap_mode='r')
    if x.shape!=(864,24,24,24): raise ValueError(f'{path}: {x.shape}')
    return np.asarray(x,dtype=np.float32).reshape(864,4,6,4,6,4,6).mean((2,4,6)).transpose(1,2,3,0).reshape(64,864).astype(np.float32)

def cache_tokens(a, em, z):
    path=a.output_dir/'native_spatial_tokens.npy'; report=a.output_dir/'pooling_equivalence.json'; path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and report.exists(): return np.load(path,mmap_mode='r')
    tokens=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(len(em),64,864)); ms=[]; cs=[]; mx=[]
    for i in range(len(em)):
        r=pool_grid(a.dense_cache/f'{i:05d}.npy'); tokens[i]=r; p=r.mean(0); q=z[i]; ms.append(float(np.mean((p-q)**2))); cs.append(float(np.dot(p,q)/(np.linalg.norm(p)*np.linalg.norm(q)+1e-12))); mx.append(float(np.max(np.abs(p-q))))
        if (i+1)%200==0: print(f'cached {i+1}/{len(em)}',flush=True)
    tokens.flush(); out={'n_scans':len(em),'token_shape':[len(em),64,864],'mean_pool_mse_mean':float(np.mean(ms)),'mean_pool_mse_max':float(np.max(ms)),'mean_pool_cosine_mean':float(np.mean(cs)),'mean_pool_cosine_min':float(np.min(cs)),'mean_pool_max_abs_diff_mean':float(np.mean(mx)),'mean_pool_max_abs_diff_max':float(np.max(mx)),'criterion':'mean pool of native R_t versus existing z_t'}; report.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2),flush=True); return tokens

class NativeTokenTransformer(pl.LightningModule):
    def __init__(self,mean,scale):
        super().__init__(); self.save_hyperparameters(ignore=['mean','scale']); self.register_buffer('mean',torch.tensor(mean,dtype=torch.float32)); self.register_buffer('scale',torch.tensor(scale,dtype=torch.float32)); self.norm=nn.LayerNorm(DIM); self.time=nn.Sequential(nn.Linear(1,32),nn.GELU(),nn.Linear(32,DIM)); self.pos=nn.Parameter(torch.zeros(1,TOKENS,DIM)); nn.init.trunc_normal_(self.pos,std=.02); layer=nn.TransformerEncoderLayer(d_model=DIM,nhead=8,dim_feedforward=2*DIM,dropout=.1,activation='gelu',batch_first=True,norm_first=True); self.enc=nn.TransformerEncoder(layer,num_layers=3); self.head=nn.Sequential(nn.LayerNorm(DIM),nn.Linear(DIM,DIM))
    def forward(self,r,dt): return self.head(self.enc(self.norm(r)+self.pos+self.time(dt).unsqueeze(1)))
    def step(self,b,stage):
        r,dt,y=b; yh=self(r,dt); loss=nn.functional.mse_loss(yh,y); dh=yh*self.scale+self.mean; pred=r+dh; target=r+y*self.scale+self.mean; fm=nn.functional.mse_loss(pred,target); co=(1-nn.functional.cosine_similarity(pred.reshape(len(pred),-1),target.reshape(len(target),-1),dim=-1)).mean(); self.log(f'{stage}_loss',loss,on_epoch=True); self.log(f'{stage}_forecast_mse',fm,on_epoch=True); self.log(f'{stage}_forecast_cosine_loss',co,on_epoch=True); return loss
    def training_step(self,b,i): return self.step(b,'train')
    def validation_step(self,b,i): self.step(b,'val')
    def configure_optimizers(self): return torch.optim.AdamW(self.parameters(),lr=1e-3,weight_decay=1e-4)

def train_seed(a,R,pairs,seed,mean,scale,dt_scaler,device,logger):
    seed_all(seed); tr=pairs.split.eq('train').to_numpy(); va=pairs.split.eq('val').to_numpy(); rt=R[pairs.idx_t.to_numpy()]; rt1=R[pairs.idx_t1.to_numpy()]; y=((rt1-rt-mean)/scale).astype('float32'); dt=dt_scaler.transform(pairs[['dt_days']]).astype('float32'); ds=TensorDataset(torch.from_numpy(rt[tr]),torch.from_numpy(dt[tr]),torch.from_numpy(y[tr])); vs=TensorDataset(torch.from_numpy(rt[va]),torch.from_numpy(dt[va]),torch.from_numpy(y[va])); dl=DataLoader(ds,batch_size=a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed)); vl=DataLoader(vs,batch_size=a.batch_size); model=NativeTokenTransformer(mean,scale); ck=a.output_dir/'checkpoints'/f'seed{seed}'; ck.mkdir(parents=True,exist_ok=True); cb=ModelCheckpoint(dirpath=str(ck),monitor='val_forecast_mse',mode='min',save_top_k=1,filename='best'); es=EarlyStopping(monitor='val_forecast_mse',mode='min',patience=a.patience); trainer=pl.Trainer(accelerator='gpu' if device.type=='cuda' else 'cpu',devices=1,max_epochs=a.epochs,logger=logger,callbacks=[cb,es],enable_progress_bar=False,enable_model_summary=False,deterministic=True); trainer.fit(model,dl,vl); best=NativeTokenTransformer.load_from_checkpoint(cb.best_model_path,mean=mean,scale=scale).to(device); best.eval();
    with torch.inference_mode(): yh=best(torch.from_numpy(rt).to(device),torch.from_numpy(dt).to(device)).cpu().numpy()
    pred=rt+yh*scale+mean; np.save(a.output_dir/'predictions'/f'seed{seed}.npy',pred); np.save(a.output_dir/'predictions'/f'delta_seed{seed}.npy',pred-rt); return pred,{'seed':seed,'best_checkpoint':cb.best_model_path,'best_val_forecast_mse':float(cb.best_model_score)}

def latent_table(a,pairs,R,preds,identity):
    true=R[pairs.idx_t1.to_numpy()]; td=true-R[pairs.idx_t.to_numpy()]; rows=[]
    for scope,mask in [('pooled',np.ones(len(pairs),bool)),('T0_T1',pairs.pair_type.eq('T0_T1').to_numpy()),('T1_T2',pairs.pair_type.eq('T1_T2').to_numpy())]:
        mask &= pairs.split.eq('test').to_numpy()
        for name,p in {'identity':identity,**preds,'actual_R_t1':true}.items():
            if name=='actual_R_t1': row={'scope':scope,'representation':name,'mse':0.,'cosine':1.,'l2':0.,'delta_mse':0.,'delta_cosine':1.}
            else:
                d=p-R[pairs.idx_t.to_numpy()]; flat=lambda x:x.reshape(x.shape[0],-1); row={'scope':scope,'representation':name,'mse':float(np.mean((p[mask]-true[mask])**2)),'cosine':float(np.mean(np.sum(p[mask]*true[mask],(1,2))/(np.linalg.norm(flat(p[mask]),axis=1)*np.linalg.norm(flat(true[mask]),axis=1)+1e-12))),'l2':float(np.mean(np.linalg.norm(flat(p[mask]-true[mask]),axis=1))),'delta_mse':float(np.mean((d[mask]-td[mask])**2)),'delta_cosine':float(np.mean(np.sum(d[mask]*td[mask],(1,2))/(np.linalg.norm(flat(d[mask]),axis=1)*np.linalg.norm(flat(td[mask]),axis=1)+1e-12)))}
            rows.append(row)
    pd.DataFrame(rows).to_csv(a.output_dir/'latent_per_seed_metrics.csv',index=False)

def run_probes(a,pairs,reps,device):
    rows=[]
    for scope,sm in [('pooled',np.ones(len(pairs),bool)),('T0_T1',pairs.pair_type.eq('T0_T1').to_numpy()),('T1_T2',pairs.pair_type.eq('T1_T2').to_numpy())]:
      for name,x in reps.items():
       x=x.mean(1) if x.ndim==3 else x
       for label in [f'primary_future_cancer_{h}' for h in HORIZONS]+[f'diag_t1_{d}' for d in DIAGNOSTIC_TASKS]:
        valid=sm & pairs[label].notna().to_numpy(); sub=pairs.loc[valid].reset_index(drop=True); y=sub[label].to_numpy(float); sp=sub.split.to_numpy(); tr,va,te=sp=='train',sp=='val',sp=='test'
        if len(np.unique(y[tr]))<2 or y[te].sum()==0: continue
        sc=StandardScaler().fit(x[valid][tr]); xs=sc.transform(x[valid]).astype('float32')
        for ps in a.seeds:
         seed_all(ps); clf=nn.Linear(xs.shape[1],1).to(device); pos=max(y[tr].sum(),1); neg=max(tr.sum()-y[tr].sum(),1); lf=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/pos],device=device)); opt=torch.optim.AdamW(clf.parameters(),lr=1e-3,weight_decay=1e-4); ds=TensorDataset(torch.from_numpy(xs[tr]),torch.from_numpy(y[tr].astype('float32'))); best=(1e9,None); stale=0
         for _ in range(100):
          clf.train()
          for xb,yb in DataLoader(ds,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(ps)):
           opt.zero_grad(); l=lf(clf(xb.to(device)).squeeze(1),yb.to(device)); l.backward(); opt.step()
          clf.eval();
          with torch.inference_mode(): vl=float(lf(clf(torch.from_numpy(xs[va]).to(device)).squeeze(1),torch.from_numpy(y[va].astype('float32')).to(device)))
          if vl<best[0]-1e-5: best=(vl,{k:v.detach().cpu().clone() for k,v in clf.state_dict().items()}); stale=0
          else: stale+=1
          if stale>=15: break
         clf.load_state_dict(best[1]); clf.eval();
         with torch.inference_mode(): pr=torch.sigmoid(clf(torch.from_numpy(xs[te]).to(device)).squeeze(1)).cpu().numpy()
         rows.append({'scope':scope,'representation':name,'label':label,'probe_seed':ps,'n_test':int(te.sum()),'n_test_positive':int(y[te].sum()),**probe_metrics(y[te],pr)})
    pd.DataFrame(rows).to_csv(a.output_dir/'downstream_per_seed_metrics.csv',index=False)

def main():
    a=parse(); [ (a.output_dir/d).mkdir(parents=True,exist_ok=True) for d in ['predictions','checkpoints','training_history'] ]; device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device); base=Path('outputs/colipri_decodability'); em=pd.read_csv(base/'mean/cache/embedding_metadata.csv',dtype=str); em['embedding_index']=em.embedding_index.astype(int); pairs,z=build_pairs(argparse.Namespace(embedding_cache=base/'mean/cache',prognostic_labels=Path('data/nlst/metadata/nlst/prognostic_labels/pilot_1000_balanced/nlst_scan_prognostic_labels.csv'),diagnostic_manifest=Path('outputs/colipri_decodability/experiment_scan_manifest.csv'),patient_splits=base/'patient_splits.csv')); pairs.to_csv(a.output_dir/'pairs_manifest.csv',index=False); R=cache_tokens(a,em,z); tr=pairs.split.eq('train').to_numpy(); delta=R[pairs.idx_t1.to_numpy()]-R[pairs.idx_t.to_numpy()]; ds=StandardScaler().fit(delta[tr].reshape(-1,DIM)); dt=StandardScaler().fit(pairs.loc[tr,['dt_days']]); np.savez(a.output_dir/'delta_scaler.npz',mean=ds.mean_,scale=ds.scale_); np.savez(a.output_dir/'time_scaler.npz',mean=dt.mean_,scale=dt.scale_); print(json.dumps({'device':str(device),'token_shape':list(R.shape),'delta_shape':list(delta.shape),'delta_mean_std':{'mean':float(ds.mean_.mean()),'std':float(ds.scale_.mean())}},indent=2),flush=True)
    preds={}; rows=[]
    for seed in a.seeds:
        logger=None
        if not a.no_wandb:
            logger=WandbLogger(project=a.wandb_project,group=a.wandb_group,name=f'native864_seed{seed}',save_dir=str(a.output_dir/'wandb'),log_model=False); logger.experiment.config.update({'seed':seed,'tokens':64,'dim':864,'blocks':3,'heads':8,'mlp_ratio':2,'objective':'standardized_native_token_delta_mse'})
        p,r=train_seed(a,R,pairs,seed,ds.mean_,ds.scale_,dt,device,logger); preds[f'native864_transformer__seed{seed}']=p; rows.append(r)
        if logger is not None: logger.experiment.summary.update(r); logger.experiment.finish()
    pd.DataFrame(rows).to_csv(a.output_dir/'training_per_seed_metrics.csv',index=False); latent_table(a,pairs,R,preds,R[pairs.idx_t.to_numpy()])
    prev={f'previous_pooled_mlp__seed{s}':np.load(a.previous_pooled_output/'predicted_embeddings'/f'seed{s}.npy') for s in a.seeds}; pca={f'pca256_spatial__seed{s}':np.load(a.previous_pca_output/'predictions'/f'seed{s}.npy').mean(1) for s in a.seeds}; reps={'current_z_t':z[pairs.idx_t.to_numpy()],'actual_z_t1':z[pairs.idx_t1.to_numpy()],**prev,**pca,**{k:v.mean(1) for k,v in preds.items()}}; run_probes(a,pairs,reps,device); pd.DataFrame({'representation':list(reps),'shape':[list(x.shape) for x in reps.values()]}).to_csv(a.output_dir/'representation_manifest.csv',index=False); print(json.dumps({'finished':True,'representations':list(reps)},indent=2),flush=True)

if __name__=='__main__': main()
