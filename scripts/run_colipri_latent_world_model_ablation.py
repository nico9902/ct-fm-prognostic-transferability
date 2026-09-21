#!/usr/bin/env python3
"""Controlled objective/time ablation for the frozen COLIPRI world model."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import wandb
except ImportError:
    wandb = None

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_colipri_latent_world_model import (DIAGNOSTIC_TASKS, DIM, HORIZONS,
    build_pairs, latent_metrics, probe_metrics, seed_all)

SEEDS = [0, 1, 2, 3, 4]
TIME_MODES = ["none", "scalar", "log", "fourier"]
BETA_GRID = [0.0, 1e-4, 1e-3, 1e-2]


class DeltaPredictor(nn.Module):
    def __init__(self, time_mode: str, dim: int = DIM, hidden: int = 1024):
        super().__init__(); self.time_mode = time_mode
        self.z_norm = nn.LayerNorm(dim)
        if time_mode == "none": self.time_embed = None; time_dim = 0
        elif time_mode in ("scalar", "log"): self.time_embed = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 32)); time_dim = 32
        elif time_mode == "fourier": self.time_embed = nn.Sequential(nn.Linear(8, 32), nn.GELU(), nn.Linear(32, 32)); time_dim = 32
        else: raise ValueError(time_mode)
        self.mlp = nn.Sequential(nn.Linear(dim + time_dim, hidden), nn.GELU(), nn.Dropout(.1), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(.1), nn.Linear(hidden, dim))

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.z_norm(z)
        if self.time_embed is not None: h = torch.cat([h, self.time_embed(t)], -1)
        return self.mlp(h)


def args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/colipri_latent_world_model_ablation"))
    p.add_argument("--previous-output", type=Path, default=Path("outputs/colipri_latent_world_model"))
    p.add_argument("--device", default="auto"); p.add_argument("--epochs", type=int, default=250); p.add_argument("--patience", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=128); p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--no-wandb", action="store_true"); p.add_argument("--wandb-project", default="nlst-colipri-latent-world-model"); p.add_argument("--wandb-group", default="standardized-delta-ablation")
    return p.parse_args()


def time_features(dt, mode, scaler):
    x = np.log1p(dt) if mode == "log" else dt
    x = scaler.transform(x.reshape(-1, 1)).astype(np.float32)
    if mode != "fourier": return x
    return np.concatenate([np.sin(x * f) for f in [1, 2, 4, 8]] + [np.cos(x * f) for f in [1, 2, 4, 8]], axis=1).astype(np.float32)


def objective_name(obj, beta):
    return f"{obj}_beta{beta:g}" if obj == "C" else obj


def train_one(a, pairs, z, delta_scaler, time_scaler, config, seed, device):
    objective, beta, time_mode = config["objective"], config["beta"], config["time_mode"]
    seed_all(seed); tr = pairs.split.eq("train").to_numpy(); va = pairs.split.eq("val").to_numpy()
    zt = z[pairs.idx_t.to_numpy()]; zt1 = z[pairs.idx_t1.to_numpy()]; delta = zt1 - zt
    target = delta_scaler.transform(delta).astype(np.float32); tf = time_features(pairs.dt_days.to_numpy(float), time_mode, time_scaler) if time_mode != "none" else np.zeros((len(pairs), 0), np.float32)
    model = DeltaPredictor(time_mode).to(device); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = TensorDataset(torch.from_numpy(zt[tr]), torch.from_numpy(tf[tr]), torch.from_numpy(target[tr])); vs = TensorDataset(torch.from_numpy(zt[va]), torch.from_numpy(tf[va]), torch.from_numpy(target[va]))
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed)); vloader = DataLoader(vs, batch_size=a.batch_size)
    best = (float("inf"), None, 0); stale = 0; history = []
    def calc(zb, tb, yb):
        dnorm = model(zb, tb); dhat = dnorm * torch.as_tensor(delta_scaler.scale_, device=device) + torch.as_tensor(delta_scaler.mean_, device=device); pred = zb + dhat; target_z = zb + yb * torch.as_tensor(delta_scaler.scale_, device=device) + torch.as_tensor(delta_scaler.mean_, device=device)
        dmse = ((dnorm - yb) ** 2).mean(); cos = (1 - nn.functional.cosine_similarity(pred, target_z, dim=-1)).mean(); shrink = (dhat ** 2).mean(); total = dmse + (cos if objective in ("B", "C") else 0) + beta * shrink
        return total, dmse, cos, shrink, pred, target_z
    for epoch in range(1, a.epochs + 1):
        model.train(); sums = np.zeros(4); n = 0
        for xb, tb, yb in loader:
            xb, tb, yb = xb.to(device), tb.to(device), yb.to(device); opt.zero_grad(set_to_none=True); total, dmse, cos, shrink, _, _ = calc(xb, tb, yb); total.backward(); opt.step(); sums += np.array([float(total), float(dmse), float(cos), float(shrink)]) * len(xb); n += len(xb)
        model.eval(); vsum = np.zeros(5); vn = 0; pred_v = []; truth_v = []
        with torch.inference_mode():
            for xb, tb, yb in vloader:
                vals = calc(xb.to(device), tb.to(device), yb.to(device)); vsum[:4] += np.array([float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])]) * len(xb); vn += len(xb); pred_v.append(vals[4].cpu().numpy()); truth_v.append(vals[5].cpu().numpy())
        pv, tv = np.concatenate(pred_v), np.concatenate(truth_v); lm = latent_metrics(pv, tv); row = {"config": config["name"], "seed": seed, "epoch": epoch, "train_loss": sums[0]/n, "train_delta_norm_mse": sums[1]/n, "val_loss": vsum[0]/vn, "val_delta_norm_mse": vsum[1]/vn, "val_cosine_loss": vsum[2]/vn, "val_shrinkage": vsum[3]/vn, "val_latent_mse": lm["mse"], "val_latent_cosine": lm["cosine_similarity"], "val_latent_l2": lm["l2_distance"]}; history.append(row)
        if wandb is not None and wandb.run is not None: wandb.log(row)
        if row["val_latent_mse"] < best[0] - 1e-6: best = (row["val_latent_mse"], {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}, epoch); stale = 0
        else: stale += 1
        if stale >= a.patience: break
    model.load_state_dict(best[1]); model.eval();
    with torch.inference_mode(): dnorm = model(torch.from_numpy(zt).to(device), torch.from_numpy(tf).to(device)).cpu().numpy()
    dhat = dnorm * delta_scaler.scale_ + delta_scaler.mean_; pred = zt + dhat
    name = config["name"]; (a.output_dir/"checkpoints").mkdir(exist_ok=True); (a.output_dir/"training_history").mkdir(exist_ok=True); (a.output_dir/"predicted_embeddings").mkdir(exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": config, "seed": seed, "delta_mean": delta_scaler.mean_, "delta_scale": delta_scaler.scale_, "time_mean": time_scaler.mean_, "time_scale": time_scaler.scale_}, a.output_dir/"checkpoints"/f"{name}__seed{seed}.pt")
    np.save(a.output_dir/"predicted_embeddings"/f"{name}__seed{seed}.npy", pred); np.save(a.output_dir/"predicted_embeddings"/f"{name}__delta__seed{seed}.npy", dhat); pd.DataFrame(history).to_csv(a.output_dir/"training_history"/f"{name}__seed{seed}.csv", index=False)
    return pred, {"config": name, "seed": seed, "best_epoch": best[2], "best_val_latent_mse": best[0], "n_epochs": len(history)}


def latent_table(pairs, z, pred_by_config, identity, out):
    rows=[]; true=z[pairs.idx_t1.to_numpy()]; true_delta=true-z[pairs.idx_t.to_numpy()]
    for scope, mask in [("pooled",np.ones(len(pairs),bool)),("T0_T1",pairs.pair_type.eq("T0_T1").to_numpy()),("T1_T2",pairs.pair_type.eq("T1_T2").to_numpy())]:
        mask &= pairs.split.eq("test").to_numpy()
        for name,pred in {"identity":identity,**pred_by_config,"actual_z_t1":true}.items():
            if name == "actual_z_t1": row={"scope":scope,"representation":name,"mse":0.,"cosine_similarity":1.,"l2_distance":0.,"delta_mse":0.,"delta_cosine_similarity":1.}
            else:
                pdlt=pred-z[pairs.idx_t.to_numpy()]; lm=latent_metrics(pred[mask],true[mask]); d=latent_metrics(pdlt[mask],true_delta[mask]); row={"scope":scope,"representation":name,**lm,"delta_mse":d["mse"],"delta_cosine_similarity":d["cosine_similarity"]}
            rows.append(row)
    pd.DataFrame(rows).to_csv(out,index=False); return pd.DataFrame(rows)


def probe_all(a, pairs, reps, labels_by_rep, out_name, device):
    rows=[]
    for scope, smask in [("pooled",np.ones(len(pairs),bool)),("T0_T1",pairs.pair_type.eq("T0_T1").to_numpy()),("T1_T2",pairs.pair_type.eq("T1_T2").to_numpy())]:
      for rep,X in reps.items():
       for label in labels_by_rep[rep]:
        valid=smask & pairs[label].notna().to_numpy(); sub=pairs.loc[valid].reset_index(drop=True); y=sub[label].to_numpy(float); split=sub.split.to_numpy(); tr,va,te=split=="train",split=="val",split=="test"
        if tr.sum()==0 or te.sum()==0 or len(np.unique(y[tr]))<2 or y[te].sum()==0: continue
        scaler=StandardScaler().fit(X[valid][tr]); Xs=scaler.transform(X[valid]).astype(np.float32)
        for pseed in a.seeds:
         seed_all(pseed); clf=nn.Linear(Xs.shape[1],1).to(device); pos=max(y[tr].sum(),1); neg=max(tr.sum()-y[tr].sum(),1); loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/pos],device=device)); opt=torch.optim.AdamW(clf.parameters(),lr=1e-3,weight_decay=1e-4); trds=TensorDataset(torch.from_numpy(Xs[tr]),torch.from_numpy(y[tr].astype(np.float32))); best=(float('inf'),None); stale=0
         for _ in range(100):
          clf.train()
          for xb,yb in DataLoader(trds,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(pseed)):
           opt.zero_grad(); l=loss_fn(clf(xb.to(device)).squeeze(1),yb.to(device)); l.backward(); opt.step()
          clf.eval();
          with torch.inference_mode(): vl=float(loss_fn(clf(torch.from_numpy(Xs[va]).to(device)).squeeze(1),torch.from_numpy(y[va].astype(np.float32)).to(device)))
          if vl<best[0]-1e-5: best=(vl,{k:v.detach().cpu().clone() for k,v in clf.state_dict().items()}); stale=0
          else: stale+=1
          if stale>=15: break
         clf.load_state_dict(best[1]); clf.eval()
         with torch.inference_mode(): prob=torch.sigmoid(clf(torch.from_numpy(Xs[te]).to(device)).squeeze(1)).cpu().numpy()
         rows.append({"scope":scope,"representation":rep,"label":label,"probe_seed":pseed,"n_test":int(te.sum()),"n_test_positive":int(y[te].sum()),**probe_metrics(y[te],prob)})
    d=pd.DataFrame(rows); d.to_csv(a.output_dir/out_name,index=False); d.groupby(["scope","representation","label"])[["auroc","auprc","mcc","f1"]].agg(["mean","std"]).to_csv(a.output_dir/out_name.replace("per_seed","summary")); return d


def main():
    a=args(); a.output_dir.mkdir(parents=True,exist_ok=True); device=torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a.device=="auto" else a.device)
    pairs,z=build_pairs(argparse.Namespace(embedding_cache=Path("outputs/colipri_decodability/mean/cache"),prognostic_labels=Path("data/nlst/metadata/nlst/prognostic_labels/pilot_1000_balanced/nlst_scan_prognostic_labels.csv"),diagnostic_manifest=Path("outputs/colipri_decodability/experiment_scan_manifest.csv"),patient_splits=Path("outputs/colipri_decodability/patient_splits.csv")))
    pairs.to_csv(a.output_dir/"pairs_manifest.csv",index=False); tr=pairs.split.eq("train").to_numpy(); delta=z[pairs.idx_t1.to_numpy()]-z[pairs.idx_t.to_numpy()]; delta_scaler=StandardScaler().fit(delta[tr]); time_scalers={m:StandardScaler().fit(((np.log1p(pairs.dt_days) if m=="log" else pairs.dt_days).to_numpy()[tr,None])) for m in ["scalar","log","fourier"]}; np.savez(a.output_dir/"delta_normalization.npz",mean=delta_scaler.mean_,scale=delta_scaler.scale_)
    (a.output_dir/"delta_normalization.json").write_text(json.dumps({"fit_patients":"train","mean":delta_scaler.mean_.tolist(),"scale":delta_scaler.scale_.tolist()}))
    identity=z[pairs.idx_t.to_numpy()]; configs=[]
    for obj in ["A","B"]: configs.append({"name":objective_name(obj,0),"objective":obj,"beta":0.,"time_mode":"scalar"})
    for b in BETA_GRID: configs.append({"name":objective_name("C",b),"objective":"C","beta":b,"time_mode":"scalar"})
    configs=list({c["name"]:c for c in configs}.values()); all_preds={}; train_rows=[]
    for c in configs:
      for seed in a.seeds:
       run=None
       if wandb is not None and not a.no_wandb: run=wandb.init(project=a.wandb_project,group=a.wandb_group,name=f"{c['name']}__seed{seed}",config={**c,"seed":seed,"device":str(device)},reinit="finish_previous",settings=wandb.Settings(root_dir=str(a.output_dir/"wandb"),x_disable_stats=True))
       pred,meta=train_one(a,pairs,z,delta_scaler,time_scalers["scalar"],c,seed,device); all_preds[f"{c['name']}__seed{seed}"]=pred; train_rows.append(meta)
       if run is not None: run.summary.update(meta); run.finish()
    pd.DataFrame(train_rows).to_csv(a.output_dir/"training_per_seed_metrics.csv",index=False)
    val_rows=pd.DataFrame(train_rows); best_name=val_rows.groupby("config").best_val_latent_mse.mean().idxmin(); best_cfg=next(c for c in configs if c["name"]==best_name)
    time_configs=[]
    for tm in TIME_MODES: time_configs.append({**best_cfg,"name":f"{best_name}__time_{tm}","time_mode":tm})
    for c in time_configs:
      for seed in a.seeds:
       run=None
       if wandb is not None and not a.no_wandb: run=wandb.init(project=a.wandb_project,group=a.wandb_group,name=f"{c['name']}__seed{seed}",config={**c,"seed":seed},reinit="finish_previous",settings=wandb.Settings(root_dir=str(a.output_dir/"wandb"),x_disable_stats=True))
       pred,meta=train_one(a,pairs,z,delta_scaler,time_scalers.get(c["time_mode"], time_scalers["scalar"]),c,seed,device); all_preds[f"{c['name']}__seed{seed}"]=pred; train_rows.append(meta)
       if run is not None: run.summary.update(meta); run.finish()
    pd.DataFrame(train_rows).to_csv(a.output_dir/"training_per_seed_metrics.csv",index=False); (a.output_dir/"selection.json").write_text(json.dumps({"best_objective_config":best_name,"selection":"mean validation latent MSE across world-model seeds","time_configs":[c["name"] for c in time_configs]},indent=2))
    latent_table(pairs,z,all_preds,identity,a.output_dir/"latent_per_seed_metrics.csv")
    prev={f"previous__seed{s}":np.load(a.previous_output/"predicted_embeddings"/f"seed{s}.npy") for s in a.seeds}
    # Downstream probing is reserved for the scientific comparison set: identity,
    # the previous raw-residual baseline, every time-conditioning variant of the
    # validation-selected objective, and the observed-future oracle. All objective
    # variants still receive complete latent metrics above.
    probe_reps={"z_t":identity,"actual_z_t1":z[pairs.idx_t1.to_numpy()]}; probe_reps.update(prev)
    probe_reps.update({k:v for k,v in all_preds.items() if any(k.startswith(c["name"]+"__seed") for c in time_configs)})
    prog={k:[f"primary_future_cancer_{h}" for h in HORIZONS] for k in probe_reps}; diag={k:[f"diag_t1_{x}" for x in DIAGNOSTIC_TASKS] for k in probe_reps}
    probe_all(a,pairs,probe_reps,prog,"prognostic_per_seed_metrics.csv",device); probe_all(a,pairs,probe_reps,diag,"diagnostic_t1_per_seed_metrics.csv",device)
    print(json.dumps({"device":str(device),"pairs":len(pairs),"best_objective":best_name,"n_probe_representations":len(probe_reps)}, indent=2), flush=True)


if __name__=="__main__": main()
