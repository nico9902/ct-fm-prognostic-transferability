"""Held-out analysis and conservative report for the clinical-predictor gate."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from run_nlst_gate import TASKS,METRICS,metrics,markdown,save_json


def analyze(cfg,out,pairs,current,future,y,reps,ensembles,chosen):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    test=pairs.split.eq('test').to_numpy();train=pairs.split.eq('train').to_numpy()
    primary=test & np.isfinite(y[:,1])
    true_delta=future-current
    true_cos=np.sum(future*current,1)/(np.linalg.norm(future,axis=1)*np.linalg.norm(current,axis=1)+1e-12)
    groups={'all_test':test,'T0_T1':test & pairs.pair_type.eq('T0_T1').to_numpy(),
        'T1_T2':test & pairs.pair_type.eq('T1_T2').to_numpy(),
        'future_cancer_2y_positive':test & (y[:,1]==1),
        'future_cancer_2y_negative':test & (y[:,1]==0),
        'future_cancer_within_1y':test & pairs.primary_cancer_event.eq(1).to_numpy() & pairs.primary_cancer_time_days.le(365).to_numpy(),
        'future_cancer_after_1y':test & pairs.primary_cancer_event.eq(1).to_numpy() & pairs.primary_cancer_time_days.gt(365).to_numpy()}
    for diagnostic in ['isolation_suspicious','nodule_ge_4mm','emphysema','fibrosis_or_scarring']:
        for value in [0,1]:
            groups[f'current_{diagnostic}_{value}']=test & pairs[f'diag_t_{diagnostic}'].eq(value).to_numpy()
    subgroup=[];errors=[];correlations=[];distance_rows=[]
    true_risk=ensembles['risk_real_future_frozen_oracle']
    current_risk=ensembles['risk_current_frozen_oracle']
    for name,values in reps.items():
        pred=np.mean(values,axis=0)
        predicted_delta=pred-current
        probability=ensembles[f'risk_{name}_frozen_oracle']
        error=np.abs(probability[:,1]-true_risk[:,1])
        dcos=np.sum(predicted_delta*true_delta,1)/(np.linalg.norm(predicted_delta,axis=1)*np.linalg.norm(true_delta,axis=1)+1e-12)
        table=pairs[['patient_id','pair_type','split']+TASKS+['primary_cancer_event','primary_cancer_time_days']].copy()
        table['model']=name;table['clinical_probability_error_2y']=error
        table['clinical_logit_error_2y']=np.abs(np.log(np.clip(probability[:,1],1e-7,1-1e-7)/np.clip(1-probability[:,1],1e-7,1))-
            np.log(np.clip(true_risk[:,1],1e-7,1-1e-7)/np.clip(1-true_risk[:,1],1e-7,1)))
        table['true_delta_norm']=np.linalg.norm(true_delta,axis=1)
        table['predicted_delta_norm']=np.linalg.norm(predicted_delta,axis=1)
        table['true_current_future_cosine']=true_cos;table['delta_cosine']=dcos
        table['current_risk']=current_risk[:,1];table['predicted_risk']=probability[:,1];table['true_future_risk']=true_risk[:,1]
        errors.append(table.loc[primary])
        distance_rows.append(table.loc[test])
        for variable in ['primary_cancer_time_days','true_delta_norm','true_current_future_cosine']:
            mask=primary & pairs.primary_cancer_event.eq(1).to_numpy() if variable=='primary_cancer_time_days' else primary.copy()
            x=table[variable].to_numpy();mask &= np.isfinite(x)
            if mask.sum()>10 and np.std(error[mask])>0 and np.std(x[mask])>0:
                rho,pvalue=spearmanr(error[mask],x[mask])
                correlations.append(dict(model=name,variable=variable,n=int(mask.sum()),
                    spearman_rho=rho,exploratory_pair_level_pvalue=pvalue,
                    warning='descriptive only; repeated pairs and multiplicity not accounted for; no causal claim'))
        for group,sm in groups.items():
            n=int(sm.sum())
            if n==0:continue
            mse=float(np.mean((pred[sm]-future[sm])**2))
            for j,task in enumerate(TASKS):
                mask=sm & np.isfinite(y[:,j]);labels=y[mask,j]
                clinical=metrics(labels,probability[mask,j]) if mask.any() else {m:np.nan for m in METRICS}
                # Rare one-class subsets cannot support discrimination claims.
                if len(labels)<10 or len(np.unique(labels))<2:
                    for m in ['auroc','auprc','mcc']:clinical[m]=np.nan
                subgroup.append(dict(model=name,subgroup=group,task=task,n=len(labels),
                    patients=pairs.loc[mask,'patient_id'].nunique(),positives=int(labels.sum()),
                    class_count=len(np.unique(labels)),representation_n=n,mse=mse,
                    clinical_probability_error=float(np.mean(np.abs(probability[mask,j]-true_risk[mask,j]))) if mask.any() else np.nan,
                    delta_cosine=float(dcos[sm].mean()),**clinical))
    error_table=pd.concat(errors,ignore_index=True)
    error_table.to_csv(out/'analysis/test_pair_errors.csv',index=False)
    pd.concat(distance_rows,ignore_index=True).to_csv(out/'analysis/test_delta_metrics.csv',index=False)
    pd.DataFrame(correlations).to_csv(out/'analysis/error_associations.csv',index=False)
    pd.DataFrame(subgroup).to_csv(out/'subgroup_analysis.csv',index=False)
    patients=error_table.groupby(['model','patient_id']).agg(
        clinical_probability_error_2y=('clinical_probability_error_2y','mean'),
        n_pairs=('pair_type','size'),cancer_2y=(TASKS[1],'max'),
        time_to_cancer_days=('primary_cancer_time_days','min'),
        true_delta_norm=('true_delta_norm','mean'),true_current_future_cosine=('true_current_future_cosine','mean')).reset_index()
    patients.to_csv(out/'analysis/test_patient_errors.csv',index=False)
    for name in [chosen,'old','residual_C1']:
        sub=patients[patients.model.eq(name)].sort_values('clinical_probability_error_2y')
        ranked=pd.concat([sub.head(10).assign(rank_group='best'),sub.tail(10).assign(rank_group='worst')])
        ranked.to_csv(out/'analysis'/f'{name}_best_worst_patients.csv',index=False)
    error_table.groupby(['model','pair_type',TASKS[1]])[['clinical_probability_error_2y','true_delta_norm','true_current_future_cosine']].agg(['mean','std','count']).to_csv(out/'analysis/error_summary.csv')
    # One train-fitted delta PCA coordinate system for true AND predicted deltas.
    scaler=StandardScaler().fit(true_delta[train]);scaled=scaler.transform(true_delta)
    pca=PCA(2,random_state=42).fit(scaled[train]);true_coordinates=pca.transform(scaled)
    np.savez(out/'analysis/delta_pca_fit.npz',mean=scaler.mean_,scale=scaler.scale_,
        components=pca.components_,pca_mean=pca.mean_,explained_variance_ratio=pca.explained_variance_ratio_)
    visual=['identity','old','residual_C1',chosen]
    visual=list(dict.fromkeys(name for name in visual if name in reps))
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for name in visual:
        pred=np.mean(reps[name],axis=0);delta=pred-current
        cosine=np.sum(delta*true_delta,1)/(np.linalg.norm(delta,axis=1)*np.linalg.norm(true_delta,axis=1)+1e-12)
        axs[0].hist(cosine[test],bins=35,alpha=.4,density=True,label=name)
        axs[1].hist(np.linalg.norm(delta[test],axis=1),bins=35,alpha=.4,density=True,label=name)
    axs[1].hist(np.linalg.norm(true_delta[test],axis=1),bins=35,alpha=.35,density=True,label='true delta')
    axs[0].set_title('Predicted vs true delta cosine');axs[1].set_title('Delta norm')
    for ax in axs:ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'plots/delta_distributions.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,len(visual),figsize=(5*len(visual),4),squeeze=False)
    for ax,name in zip(axs[0],visual):
        pred=np.mean(reps[name],axis=0);coords=pca.transform(scaler.transform(pred-current))
        frame=pairs[['patient_id','pair_type','split']].copy();frame[['true_pc1','true_pc2']]=true_coordinates
        frame[['predicted_pc1','predicted_pc2']]=coords
        frame.to_csv(out/'analysis'/f'{name}_delta_pca_coordinates.csv',index=False)
        # Each connecting line represents a matched patient-transition prediction.
        for label,color in [(0,'steelblue'),(1,'crimson')]:
            mask=test & (y[:,1]==label)
            ax.scatter(true_coordinates[mask,0],true_coordinates[mask,1],s=10,color=color,alpha=.25,marker='o')
            ax.scatter(coords[mask,0],coords[mask,1],s=10,color=color,alpha=.4,marker='x')
            for i in np.flatnonzero(mask)[::3]:
                ax.plot([true_coordinates[i,0],coords[i,0]],[true_coordinates[i,1],coords[i,1]],color=color,alpha=.12,lw=.5)
        ax.set(title=name,xlabel='PC1',ylabel='PC2')
    fig.suptitle('Held-out matched deltas: true circles, predicted crosses; qualitative PCA')
    fig.tight_layout();fig.savefig(out/'plots/paired_delta_pca.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,len(visual),figsize=(5*len(visual),4),squeeze=False)
    for ax,name in zip(axs[0],visual):
        predicted=ensembles[f'risk_{name}_frozen_oracle'][:,1]
        ax.scatter(current_risk[primary,1],true_risk[primary,1]-current_risk[primary,1],s=12,alpha=.4,label='true risk change')
        ax.scatter(current_risk[primary,1],predicted[primary]-current_risk[primary,1],s=12,alpha=.4,label='predicted change')
        ax.axhline(0,color='black',lw=.5);ax.set(title=name,xlabel='R(current)',ylabel='R(future) − R(current)');ax.legend(fontsize=7)
    fig.suptitle('Frozen-head scores are class-weighted, uncalibrated probabilities')
    fig.tight_layout();fig.savefig(out/'plots/risk_change.png',dpi=160);plt.close(fig)
    chosen_error=error_table[error_table.model.eq(chosen)]
    plot_patients=chosen_error.groupby('patient_id').clinical_probability_error_2y.mean().sort_values()
    ids=list(plot_patients.head(8).index)+list(plot_patients.tail(8).index)
    fig,axs=plt.subplots(4,4,figsize=(14,11))
    for ax,pid in zip(axs.flatten(),ids):
        g=chosen_error[chosen_error.patient_id.eq(pid)].sort_values('pair_type')
        for _,r in g.iterrows():ax.plot([0,1,2],[r.current_risk,r.predicted_risk,r.true_future_risk],marker='o',label=r.pair_type)
        ax.set(title=pid,ylim=(0,1),xticks=[0,1,2],xticklabels=['current','predicted','real']);ax.legend(fontsize=6)
    fig.suptitle('Patient-level risk scores: 8 best and 8 worst predictions; descriptive extremes')
    fig.tight_layout();fig.savefig(out/'plots/patient_risk_pairs.png',dpi=160);plt.close(fig)
    gaps=pd.read_csv(out/'oracle_gap_summary.csv');gaps=gaps[gaps.split.eq('test') & gaps.task.eq(cfg['primary_task'])]
    subset=gaps[gaps.model.isin(['identity','old']+list(json.loads((out/'selection_lock.json').read_text())['selected']))]
    fig,ax=plt.subplots(figsize=(11,5));sub=subset.sort_values('ensemble_GapClosure')
    vals=sub.ensemble_GapClosure.to_numpy()*100
    ax.barh(sub.model,vals,color=['crimson' if n==chosen else 'steelblue' for n in sub.model])
    for i,(_,r) in enumerate(sub.iterrows()):
        ax.plot([100*r.GapClosure_ci_low,100*r.GapClosure_ci_high],[i,i],color='black',lw=1)
    ax.axvline(0,color='black',lw=.5);ax.axvline(100,color='gray',ls='--')
    ax.set(xlabel='Oracle gap closure (%) with paired patient bootstrap 95% CI',title='Primary two-year frozen-head oracle gap')
    fig.tight_layout();fig.savefig(out/'plots/oracle_gap_closure.png',dpi=160);plt.close(fig)


def report(cfg,out,pairs,chosen):
    perseed=pd.read_csv(out/'per_seed_metrics.csv')
    summary=perseed.groupby(['channel','model','head','split','task'])[METRICS].agg(['mean','std','count']).reset_index()
    summary.columns=['_'.join(c).strip('_') for c in summary.columns]
    interval=pd.read_csv(out/'bootstrap_intervals.csv')
    for metric in ['auroc','auprc','mcc']:
        rows=interval[interval.metric.eq(metric)].copy()
        rows=rows.rename(columns={'ci_low':metric+'_ensemble_ci_low','ci_high':metric+'_ensemble_ci_high',
            'value':metric+'_ensemble_value'})
        summary['arm']=summary.channel+'_'+summary.model+'_'+summary['head']
        summary=summary.merge(rows[['arm','task',metric+'_ensemble_ci_low',metric+'_ensemble_ci_high',metric+'_ensemble_value']],on=['arm','task'],how='left')
        for col in [metric+'_ensemble_ci_low',metric+'_ensemble_ci_high',metric+'_ensemble_value']:
            summary.loc[summary.split.ne('test'),col]=np.nan
    summary.drop(columns='arm').to_csv(out/'experiment_summary.csv',index=False)
    paired=pd.read_csv(out/'paired_comparisons.csv')
    primary=paired[paired.task.eq(cfg['primary_task']) & paired.metric.eq('auroc')]
    def row(channel,head,model,comparator):
        sub=primary[primary.channel.eq(channel)&primary['head'].eq(head)&primary.model.eq(model)&primary.comparator.eq(comparator)]
        return sub.iloc[0] if len(sub)==1 else None
    def improves(r):return r is not None and r.gain>=cfg['meaningful_auroc_gain'] and r.ci_low>0
    def equivalent(r):return r is not None and r.ci_low>-cfg['meaningful_auroc_gain'] and r.ci_high<cfg['meaningful_auroc_gain']
    risk_gain=row('risk','frozen_oracle',chosen,'old')
    direct=[row('direct',h,chosen,'G0') for h in ['linear','mlp']]
    improved_risk=improves(risk_gain);improved_direct=any(improves(r) for r in direct)
    align=pd.read_csv(out/'clinical_alignment_metrics.csv')
    align=align[align.split.eq('test')&align.task.eq(cfg['primary_task'])]
    geometry=pd.read_csv(out/'representation_metrics.csv');geometry=geometry[geometry.split.eq('test')]
    a=align.groupby('model')[['probability_mae','bernoulli_kl']].mean()
    g=geometry.groupby('model')[['delta_mse','delta_cosine','mse','cosine']].mean()
    geometric_gain=(g.loc[chosen,'delta_mse']<g.loc['old','delta_mse'] and g.loc[chosen,'delta_cosine']>g.loc['old','delta_cosine'])
    aligned=a.loc[chosen,'probability_mae']<a.loc['old','probability_mae']
    all_equivalent=all(equivalent(r) for r in direct)
    if cfg.get('smoke',False):
        outcome='SMOKE ONLY';recommendation='No scientific gate interpretation'
    elif improved_risk or improved_direct:
        outcome='A';recommendation='Clinical future information is partially predictable; a probabilistic follow-up is justified, subject to independent confirmation and control comparisons.'
    elif geometric_gain and aligned and all_equivalent:
        outcome='B';recommendation='Reconstruction/alignment improve without downstream prognostic gain; investigate conditional uncertainty rather than enlarging a deterministic predictor.'
    else:
        intervals=[risk_gain]+direct
        excludes_gain=all(r is not None and r.ci_high<cfg['meaningful_auroc_gain'] for r in intervals)
        if excludes_gain:
            outcome='C (tested deterministic family)';recommendation='No meaningful positive gain recovered by the validation-selected predictor; stop enlarging this deterministic forecasting family.'
        else:
            outcome='INCONCLUSIVE';recommendation='Do not advance to a full world model on this evidence; meaningful gains have not been established or excluded.'
    decision=dict(outcome=outcome,recommendation=recommendation,primary_candidate=chosen,
        meaningful_auroc_gain=cfg['meaningful_auroc_gain'],risk_oracle_gap_reduction_supported=improved_risk,
        downstream_improvement_supported=improved_direct,geometry_improves=bool(geometric_gain),alignment_mae_improves=bool(aligned),
        downstream_equivalence_supported=all_equivalent,
        rule='Primary candidate selected by validation. Gain >=0.02 and paired 95% interval above zero. Equivalence requires interval entirely inside ±0.02. Outcome C requires upper bounds to exclude +0.02; otherwise inconclusive.',
        scope='Evidence about tested architectures/losses, not a proof that no deterministic mapping can recover future information.',
        no_world_model_implemented=True)
    save_json(out/'gate_decision.json',decision)
    locked=json.loads((out/'selection_lock.json').read_text());oracle=json.loads((out/'oracle_selection.json').read_text())
    cohort=json.loads((out/'cohort_provenance.json').read_text())
    primary_summary=summary[summary.split.eq('test')&summary.task.eq(cfg['primary_task'])]
    printable=[]
    for _,r in primary_summary.iterrows():
        rec=dict(channel=r.channel,model=r.model,head=r['head'])
        for m in ['auroc','auprc','mcc']:rec[m]=f'{r[m+"_mean"]:.3f} ± {r[m+"_std"]:.3f}'
        printable.append(rec)
    gap=pd.read_csv(out/'oracle_gap_summary.csv');gap=gap[gap.split.eq('test')&gap.task.eq(cfg['primary_task'])]
    gapcols=['model','AUC_current_mean','AUC_oracle_mean','AUC_predicted_mean','OracleGap_mean','OracleGap_std',
        'GapClosure_mean','GapClosure_std','ensemble_OracleGap','OracleGap_ci_low','OracleGap_ci_high',
        'ensemble_GapClosure','GapClosure_ci_low','GapClosure_ci_high']
    comparisons=primary[(primary.model.eq(chosen)|primary.comparator.eq(chosen))]
    subgroup=pd.read_csv(out/'subgroup_analysis.csv')
    subgroup=subgroup[subgroup.model.isin([chosen,'old','identity']) & subgroup.task.eq(cfg['primary_task'])]
    output=('# Clinical future-predictor gate\n\n'
        f'**Gate outcome: {outcome}. {recommendation}**\n\n'
        '**Improved embedding reconstruction is not sufficient evidence of clinically meaningful future-state prediction.** '
        'All comparisons below distinguish geometric errors from future-risk discrimination. No RSSM, Dreamer, '
        'stochastic dynamics, diffusion, JEPA rollout, or full world model was implemented.\n\n'
        '## 1. Cohort, exact labels, and reproduced old baseline\n\n'
        f'The first gate cohort is unchanged: {cohort["patients"]} patients, {cohort["scans"]} cached CTs, '
        f'{cohort["pairs"]} pairs; 599/200/200 patient-level splits. Pair order, feature cache checksum, '
        'split assignments, and old checkpoint outputs are verified. The same 864-dimensional frozen '
        'mean-pooled COLIPRI representation and extraction provenance are used. Targets are the original '
        'future_cancer_1y/2y/3y at the t+1 landmark (365/730/1095 days after that scan), exposed as '
        'primary_future_cancer_* in the saved pair manifest. Diagnosis on/before t+1 is excluded and '
        'inadequate follow-up is missing, never negative. All current-state conditions share this '
        'observed-future-scan eligibility; these are not deployable population-risk estimates.\n\n'
        'The exact existing objective-A standardized-delta predictor is reloaded for seeds 0–4; '
        'original realized-interval predictions match its cached arrays within 1e-5. Its original '
        'five-seed ensemble is retained as old_legacy_realized_interval and first-gate G3_old_legacy. '
        'These are retrospective legacy references, not eligible primary current-only baselines. '
        'The primary old model uses the SAME checkpoints at the configured 365-day forecast query. '
        'The fixed query is available before scanning and removes realized future timing from the new '
        'predictor input. Historical old checkpoints were trained with realized intervals; this limitation '
        'is explicit. A newly trained geometry_only control uses the same prospective query as C1–C4.\n\n'
        '## 2. OracleRiskHead\n\n'
        f'Linear and 128-unit GELU MLP future-only heads were trained first for seeds {cfg["seeds"]}, '
        'with training-future-only StandardScaler and training-only class weights. Each head jointly '
        'predicts the existing three horizon targets using masked macro BCE, restoring the best '
        'validation-BCE checkpoint. The ensemble family was selected solely by validation two-year '
        f'AUROC: **{oracle["chosen_kind"]}**. Its three logits are averaged and frozen, including dropout '
        'and normalization. The oracle is excluded from predictor optimization; gradients pass through '
        'it only to the predicted embedding. The oracle is an evaluator/teacher, not a deployment model. '
        'Future-label weighting makes its sigmoid scores uncalibrated probabilities. Per-head seed '
        'variation is separately recorded in oracle_per_seed_metrics.csv; the main frozen ensemble '
        'is fixed for all predictors.\n\n'+markdown(pd.DataFrame(oracle['validation_scores']))+'\n\n'
        '## 3. Predictor variants and loss isolation\n\n'
        'The exact old DeltaPredictor class is reused: LayerNorm, time MLP 1→32→32, and '
        '(864+32)→1024→1024→864 MLP with GELU and dropout 0.1. No backbone capacity is added. '
        'Residual mode emits standardized delta and reconstructs z_current + delta; direct mode emits '
        'standardized future state. Delta and future scalers fit training pairs only. C1–C4 share '
        'architecture and checkpoint-selection rules. The state term retains objective A: '
        'mean(((predicted_future−true_future)/train_delta_std)²). It is algebraically equivalent to '
        'the original standardized-delta MSE. The auxiliary delta term is standardized-delta SmoothL1 '
        f'with weight {cfg["lambda_delta"]}; state weight is {cfg["lambda_state"]}.\n\n'
        '| Variant | Clinical term |\n| --- | --- |\n'
        '| C1 | None: state + delta only |\n'
        '| C2 | BCE distillation from stop-gradient oracle true-future probabilities |\n'
        '| C3 | Class-weighted masked BCE against true future labels |\n'
        '| C4 | Distillation + label BCE |\n\n'
        'Distillation is unweighted soft-target BCE; label BCE uses training-only class weights. '
        'Both mask invalid/missing horizon targets, including prior-cancer exclusions. They never '
        'become negatives. The clinical weight is identical for both terms in C4. State and delta '
        'errors are algebraically related; the robust auxiliary loss changes weighting, not the '
        'information content of the target. A geometry_only control uses the original loss alone; '
        'risk_only uses the identical residual backbone with label loss only, providing a capacity '
        'and outcome-supervision control rather than an anatomic forecasting claim.\n\n'
        f'Clinical weights were predeclared as {cfg["clinical_weights"]}. Each variant/mode selects '
        'one weight by mean validation-only 2y frozen-head AUROC across seeds. Predictor early stopping '
        'also uses validation 2y AUROC, so the geometry_only retrain control is essential for separating '
        'clinical loss from the historical geometric checkpoint criterion. Best checkpoints are '
        'restored. The primary clinical candidate is locked before new test evaluation: '
        f'**{chosen}**, weight **{locked["primary_clinical"]["clinical_weight"]}**.\n\n'
        +markdown(pd.DataFrame(locked['selected'].values()))+'\n\n'
        '## 4. State versus delta prediction\n\n'
        'Both output parameterizations use identical backbone parameter counts and the same clinical '
        'losses. Their output scaling and initialization-induced predictions differ; this is an explicit '
        'parameterization ablation. Paired residual-minus-direct results are in paired_comparisons.csv. '
        'No claim that a delta carries independent information beyond current+future is made.\n\n'
        '## 5–7. Identity, geometric quality, and clinical metrics\n\n'
        +markdown(g.reset_index())+'\n\n'+markdown(pd.DataFrame(printable))+'\n\n'
        'risk evaluates R(predicted_future) with the SAME frozen head; current/identity use R(current), '
        'and real_future is the oracle upper reference. direct evaluates a separately trained '
        '[current,predicted_future] linear or 128-unit MLP, with the unchanged first-gate G0/G1/G2 '
        'heads reused verbatim. G0 is padded current-only and G1 is current+real future. New probes '
        'use training-only feature scaling and validation-best masked BCE, never test-based selection. '
        'Train/validation/test AUROC, AUPRC, MCC and additional threshold metrics are in '
        'per_seed_metrics.csv and experiment_summary.csv. Threshold 0.5 is fixed.\n\n'
        'Metrics are mean ± sample SD across 42,123,2026; predictor variation is included for new '
        'models, while frozen-head and legacy ensembles are fixed. Baseline risk SD=0 reflects a '
        'fixed ensemble, not absence of uncertainty. Downstream new-model seeds couple predictor '
        'and probe variation. Legacy predictors retain their five original seeds, with individual '
        'outputs preserved.\n\n'
        '## 8. Oracle gap and relative closure\n\n'
        'OracleGap = AUROC(R(real future)) − AUROC(R(predicted future)). GapClosure = '
        '(AUROC(R(predicted future)) − AUROC(R(current))) / '
        '(AUROC(R(real future)) − AUROC(R(current))). The numerator/denominator use the same '
        'teacher and eligible samples; the R(current) baseline is distinct from G0, which trains '
        'a head on current features for this endpoint. Ratios are reported only for positive '
        'denominators. Confidence intervals for ratios can be unstable when the oracle-current '
        'gap is small; positive-denominator bootstrap draw counts are supplied. Values can be '
        'negative or exceed 100% and are not clipped.\n\n'+markdown(gap[gapcols])+'\n\n'
        '## 9. Paired uncertainty\n\n'
        +markdown(comparisons)+'\n\n'
        f'Bootstrap uses {cfg["bootstrap_samples"]} patient-cluster resamples of the held-out seed-ensemble '
        'predictions. All eligible transitions from a sampled patient travel together, and the same '
        'draws evaluate every arm. Exact weighted AUROC/AP/MCC is used for speed and tested against '
        'explicit sklearn resampling, including ties. Intervals are conditional on fitted models; '
        'training-seed SD remains separate. Paired gap reduction equals the candidate-minus-old '
        'AUROC difference because both share the oracle. Primary candidate selection used only '
        'validation; other model/endpoint comparisons are exploratory without multiplicity correction. '
        'Nonsignificance is not equivalence. Meaningful gain is predeclared as AUROC ≥0.02 with paired '
        '95% CI above zero; equivalence requires the whole interval inside ±0.02.\n\n'
        '## 10. Subgroups, error analysis, and representations\n\n'
        +markdown(subgroup[['model','subgroup','n','patients','positives','auroc','auprc','mcc','mse','clinical_probability_error']])+'\n\n'
        'Subgroups include transition type, future cancer/non-cancer, cancer within 1y/later, and '
        'validated current suspicion, nodule≥4mm, emphysema, fibrosis/scarring fields. No new clinical '
        'label is invented. Tiny or one-class subgroups have undefined discrimination metrics; '
        'they still receive geometric and risk-error summaries. These subgroup analyses are descriptive.\n\n'
        'analysis/test_pair_errors.csv and test_patient_errors.csv record probability/logit errors, '
        'event timing, transitions, true-delta norm and current-future cosine. Best/worst patient '
        'tables aggregate eligible transitions within a patient. Rank extremes are descriptive, '
        'not a new validation cohort. Error associations are exploratory and not causal; pair-level '
        'Spearman p-values do not account for repeated transitions and must not be used as clinical '
        'evidence. Plots compare delta norms/cosines, train-fitted true-delta PCA with predicted '
        'deltas in the SAME coordinate system, patient-transition pairing, current/predicted/real '
        'risk scores, and oracle-gap closure. No UMAP or geometric plot establishes prognosis.\n\n'
        '## 11. Limitations\n\n'
        'This enriched 999-patient cohort and its test split have already been inspected in earlier '
        'research. It is not independent confirmation or representative NLST risk calibration. '
        'COLIPRI pretraining included NLST images. The oracle is trained on the same training '
        'patients as the predictor; teacher in-sample overfitting is possible, although validation '
        'and test patients are never used in gradients. Cross-fitted teacher supervision is untested. '
        'Predicted training features are in-sample for subsequent probes; frozen-head evaluation '
        'therefore provides an important complementary view.\n\n'
        'Clinical-label supervision can learn a stronger current-state classifier rather than '
        'recover patient-specific future anatomy. Risk-only and geometry-only controls help assess '
        'this, but success through the optimized teacher alone does not prove general future-state '
        'quality. Independent probes and geometric errors should be read together; adversarial '
        'teacher directions or off-manifold states remain possible. An oracle advantage can reflect '
        'irreducible disease/scan uncertainty and does not establish that its entire gap is predictable.\n\n'
        'The prospective query is fixed at 365 days, whereas future scans occur at variable actual '
        'intervals; the target remains after the actual future landmark for comparability. Fixed query '
        'controls prevent future timing leakage but do not eliminate uncertainty from off-schedule '
        'scanning. There is no validated significant-change label, new survival model, or additional '
        'clinical conditioning. Optional 8³ tokens/K=8/supervision disease-projector ablations are '
        'deferred so this remains a global loss experiment. A negative result is about this finite '
        'deterministic family, not proof of information-theoretic impossibility.\n\n'
        '## 12. Gate decision\n\n'+markdown(pd.DataFrame([decision]))+'\n\n'
        f'**Final recommendation: {recommendation}** A positive outcome justifies planning an independent '
        'probabilistic follow-up; it does not cause a world model to be built in this gate. Where '
        'confidence intervals remain wide, the result stays inconclusive rather than being forced '
        'into Outcome B/C. Risk-only control comparisons are included in the paired table; if the '
        'candidate does not outperform that control, avoid attributing gains uniquely to future '
        'anatomic dynamics.\n\n'
        'Reproduce: `PYTHONPATH=src:scripts .venv/bin/python scripts/run_nlst_clinical_predictor_gate.py`. '
        '`--smoke` writes a separate verification run; `--analysis-only` regenerates completed '
        'analysis without training. Configs, source/input hashes, patient manifests, scalers, '
        'validation selections, checkpoints, CSV logs and offline W&B are retained.\n')
    (out/'FINAL_CLINICAL_PREDICTOR_GATE_REPORT.md').write_text(output)
    (out/'experiment_summary.md').write_text('# Clinical predictor gate summary\n\n'+markdown(pd.DataFrame(printable))+'\n')
