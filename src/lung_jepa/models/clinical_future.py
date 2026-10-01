"""Frozen future-risk semantics for a capacity-matched deterministic predictor."""
from __future__ import annotations
import lightning.pytorch as pl
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from .disease_state import GateModule
from .prognostic_adapter import masked_prognostic_loss
# Preserve the EXACT existing predictor architecture and its state-dict keys.
from run_colipri_latent_world_model_ablation import DeltaPredictor


class OracleRiskHead(nn.Module):
    """Validation-selected frozen ensemble, accepting RAW future embeddings."""
    def __init__(self, mean, scale, kind='linear', heads=3, hidden=128):
        super().__init__()
        self.register_buffer('mean',torch.tensor(mean,dtype=torch.float32))
        self.register_buffer('scale',torch.tensor(scale,dtype=torch.float32))
        self.heads=nn.ModuleList([GateModule(len(mean),3,[1.,1.,1.],kind=kind,hidden=hidden).head
                                  for _ in range(heads)])
        self.requires_grad_(False)
        self.eval()

    def train(self,mode=True):
        # Predictor.train() must NEVER enable oracle dropout or unfreeze the head.
        return super().train(False)

    def forward(self,z):
        x=(z-self.mean)/self.scale
        return torch.stack([head(x) for head in self.heads]).mean(0)


class ClinicalFuturePredictor(pl.LightningModule):
    """Forward sees current state and planned interval ONLY; supervision is separate."""
    def __init__(self, delta_mean,delta_scale,future_mean,future_scale,oracle_mean,oracle_scale,
                 oracle_kind='linear',oracle_heads=3,oracle_hidden=128,mode='residual',variant='C1',
                 clinical_weight=1.,lambda_state=1.,lambda_delta=.25,pos_weights=None,
                 hidden=1024,lr=.001):
        super().__init__()
        self.save_hyperparameters()
        if mode not in {'residual','direct'}:raise ValueError(mode)
        self.network=DeltaPredictor('scalar',dim=len(delta_mean),hidden=hidden)
        self.oracle=OracleRiskHead(oracle_mean,oracle_scale,oracle_kind,oracle_heads,oracle_hidden)
        for name,value in [('delta_mean',delta_mean),('delta_scale',delta_scale),
                           ('future_mean',future_mean),('future_scale',future_scale)]:
            self.register_buffer(name,torch.tensor(value,dtype=torch.float32))
        self.register_buffer('pos_weights',torch.tensor(pos_weights or [1.,1.,1.],dtype=torch.float32))
        self.validation_outputs=[]

    def forward(self,current,planned_interval):
        output=self.network(current,planned_interval)
        if self.hparams.mode=='residual':
            return current+output*self.delta_scale+self.delta_mean
        return output*self.future_scale+self.future_mean

    def losses(self,current,predicted,future,labels):
        # This is the old objective A EXACTLY (up to floating-point reconstruction):
        # standardized future-state error == standardized delta-prediction error.
        state=((predicted-future)/self.delta_scale).square().mean()
        delta_true=(future-current-self.delta_mean)/self.delta_scale
        delta_pred=(predicted-current-self.delta_mean)/self.delta_scale
        delta=F.smooth_l1_loss(delta_pred,delta_true)
        logits=self.oracle(predicted)  # Retain gradients wrt predicted state.
        with torch.no_grad():
            target=self.oracle(future).sigmoid()
            target=target.masked_fill(~torch.isfinite(labels),float('nan'))
        distill=masked_prognostic_loss(logits,target,torch.ones_like(self.pos_weights))
        label=masked_prognostic_loss(logits,labels,self.pos_weights)
        variant=self.hparams.variant
        semantic=(distill if variant in {'C2','C4'} else logits.sum()*0)
        semantic=semantic+(label if variant in {'C3','C4','risk_only'} else logits.sum()*0)
        geometric=self.hparams.lambda_state*state+self.hparams.lambda_delta*delta
        total=(logits.sum()*0 if variant=='risk_only' else geometric)+self.hparams.clinical_weight*semantic
        return dict(loss=total,state=state,delta=delta,distillation=distill,label=label)

    def training_step(self,batch,batch_idx):
        current,time,future,labels=batch
        predicted=self(current,time)
        losses=self.losses(current,predicted,future,labels)
        for name,value in losses.items():
            self.log('train/'+name,value,on_step=False,on_epoch=True,batch_size=len(current))
        return losses['loss']

    def validation_step(self,batch,batch_idx):
        current,time,future,labels=batch
        predicted=self(current,time)
        self.validation_outputs.append((current.detach(),predicted.detach(),future.detach(),labels.detach()))

    def on_validation_epoch_end(self):
        current,predicted,future,labels=[torch.cat([b[i] for b in self.validation_outputs]) for i in range(4)]
        for name,value in self.losses(current,predicted,future,labels).items():
            self.log('val/'+name,value)
        mask=torch.isfinite(labels[:,1])
        probability=self.oracle(predicted).sigmoid()
        auc=roc_auc_score(labels[mask,1].cpu().numpy(),probability[mask,1].detach().cpu().numpy())
        self.log('val/primary_auroc',auc)
        self.validation_outputs.clear()

    def configure_optimizers(self):
        # Oracle is frozen AND excluded from the optimizer explicitly.
        return torch.optim.AdamW(self.network.parameters(),lr=self.hparams.lr,weight_decay=.0001)
