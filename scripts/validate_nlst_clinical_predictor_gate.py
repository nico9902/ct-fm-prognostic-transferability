#!/usr/bin/env python3
"""Verify completed second-gate artifacts and document loss/control attribution."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_nlst_gate import TASKS,markdown


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('results/nlst_clinical_predictor_gate'))
    parser.add_argument('--log-wandb',action='store_true')
    args=parser.parse_args();out=args.output_dir
    completed=json.loads((out/'COMPLETE.json').read_text())
    assert not completed['smoke']
    cfg=json.loads((out/'configs/resolved.json').read_text())
    assert cfg['seeds']==[42,123,2026]
    files=['experiment_summary.csv','oracle_gap_summary.csv','paired_comparisons.csv','per_seed_metrics.csv',
        'representation_metrics.csv','clinical_alignment_metrics.csv','subgroup_analysis.csv',
        'FINAL_CLINICAL_PREDICTOR_GATE_REPORT.md','oracle_selection.json','selection_lock.json',
        'old_checkpoint_reproduction.csv','environment.json']
    for name in files:assert (out/name).stat().st_size>0,name
    pairs=pd.read_csv(out/'pairs_manifest.csv',dtype={'patient_id':str})
    prior=pd.read_csv(Path(cfg['previous_gate'])/'pairs_manifest.csv',dtype={'patient_id':str})
    pd.testing.assert_frame_equal(pairs,prior)
    split=pd.read_csv(out/'patient_splits.csv',dtype={'patient_id':str})
    previous_split=pd.read_csv(Path(cfg['previous_gate'])/'patient_splits.csv',dtype={'patient_id':str})
    pd.testing.assert_frame_equal(split,previous_split)
    assert not split.patient_id.duplicated().any()
    assert pairs.groupby('patient_id').split.nunique().eq(1).all()
    metric=pd.read_csv(out/'per_seed_metrics.csv')
    key=['channel','model','head','seed','split','task']
    assert not metric.duplicated(key).any()
    assert set(metric.seed)==set(cfg['seeds'])
    assert np.isfinite(metric[['auroc','auprc','mcc']]).all().all()
    for (s,t),group in metric.groupby(['split','task']):
        mask=pairs.split.eq(s)&pairs[t].notna()
        assert group.n.eq(mask.sum()).all()
        assert group.patients.eq(pairs.loc[mask,'patient_id'].nunique()).all()
        assert group.positives.eq(pairs.loc[mask,t].sum()).all()
    lock=json.loads((out/'selection_lock.json').read_text());chosen=completed['chosen']
    assert lock['test_not_used']
    assert chosen==lock['primary_clinical']['mode']+'_'+lock['primary_clinical']['variant']
    choices=pd.read_csv(out/'validation_selection.csv')
    best=choices[choices.variant.isin(['C2','C3','C4'])].sort_values(
        ['validation_2y_auroc','clinical_weight'],ascending=[False,True]).iloc[0]
    assert best['name']==lock['primary_clinical']['name']
    parameters=[]
    checkpoints=list((out/'checkpoints').glob('*/seed*/done.json'))
    expected_configs=2*(1+3*len(cfg['clinical_weights']))+2
    assert len(checkpoints)==expected_configs*len(cfg['seeds'])
    for p in checkpoints:
        d=json.loads(p.read_text());parameters.append(d['predictor_parameters'])
        assert (p.parent/'best.ckpt').stat().st_size>0
        assert d['oracle_parameters_frozen']>0
    assert len(set(parameters))==1,'Changed architecture capacity across losses/modes'
    for p in (out/'predictions').glob('*.csv'):
        data=pd.read_csv(p,dtype={'patient_id':str})
        pd.testing.assert_frame_equal(data[['patient_id','pair_type','split']],pairs[['patient_id','pair_type','split']])
        probs=data.filter(like='_probability')
        assert probs.notna().all().all() and probs.ge(0).all().all() and probs.le(1).all().all()
    reproduction=pd.read_csv(out/'old_checkpoint_reproduction.csv')
    assert reproduction.max_abs_cache_error.lt(1e-5).all()
    paired=pd.read_csv(out/'paired_comparisons.csv')
    assert not paired.duplicated(['channel','head','model','comparator','task','metric']).any()
    assert paired.valid_draws.ge(1900).all()
    assert paired.ci_low.le(paired.ci_high).all()
    primary=paired[paired.task.eq(cfg['primary_task']) & paired.metric.eq('auroc')]
    controls=['old','current','identity',chosen.split('_')[0]+'_C1','residual_geometry_only','residual_risk_only']
    control_table=primary[primary.model.eq(chosen)&primary.comparator.isin(controls)]
    main_control=control_table[control_table.channel.eq('risk')]
    margin=cfg['meaningful_auroc_gain']
    def superiority(name):
        row=main_control[main_control.comparator.eq(name)]
        return len(row)==1 and row.iloc[0].gain>=margin and row.iloc[0].ci_low>0
    loss_specific=superiority(chosen.split('_')[0]+'_C1')
    risk_only=superiority('residual_risk_only')
    attribution=dict(primary_candidate=chosen,clinical_loss_advantage_over_matched_geometry=bool(loss_specific),
        advantage_over_risk_only_classifier_control=bool(risk_only),
        interpretation='A significant candidate-vs-old gap reduction is not automatically an advantage of clinical training loss. Read matched C1, geometry-only and risk-only comparisons. No claim of patient-specific future anatomy recovery follows from risk scores alone.')
    (out/'control_attribution.json').write_text(json.dumps(attribution,indent=2))
    # Identical current/identity inputs have identical bootstrap gaps; fill the redundant baseline row.
    gap_path=out/'oracle_gap_summary.csv'
    gap=pd.read_csv(gap_path)
    original_gap=gap.copy()
    supplemental=[c for c in gap.columns if c.startswith('ensemble_') or '_ci_' in c or c.endswith('draws')]
    for index,r in gap[gap.model.eq('current')].iterrows():
        same=gap[gap.model.eq('identity') & gap.split.eq(r['split']) & gap.task.eq(r.task)]
        if len(same)==1:
            for c in supplemental:gap.loc[index,c]=same.iloc[0][c]
    gap.to_csv(gap_path,index=False)
    decision_path=out/'gate_decision.json'
    decision=json.loads(decision_path.read_text())
    for field in ['risk_oracle_gap_reduction_supported','downstream_improvement_supported',
                  'geometry_improves','alignment_mae_improves','downstream_equivalence_supported']:
        if isinstance(decision.get(field),str):decision[field]=decision[field].lower()=='true'
    decision_path.write_text(json.dumps(decision,indent=2))
    # Supplement the locked gate decision; do not modify its criterion or choose a different winner.
    report=out/'FINAL_CLINICAL_PREDICTOR_GATE_REPORT.md'
    text=report.read_text().split('\n\n## Validation and control attribution')[0]
    gapcols=['model','AUC_current_mean','AUC_oracle_mean','AUC_predicted_mean','OracleGap_mean','OracleGap_std',
        'GapClosure_mean','GapClosure_std','ensemble_OracleGap','OracleGap_ci_low','OracleGap_ci_high',
        'ensemble_GapClosure','GapClosure_ci_low','GapClosure_ci_high']
    selected_rows=gap.split.eq('test')&gap.task.eq(cfg['primary_task'])
    old_rows=original_gap.split.eq('test')&original_gap.task.eq(cfg['primary_task'])
    text=text.replace(markdown(original_gap.loc[old_rows,gapcols]),markdown(gap.loc[selected_rows,gapcols]))
    text += ('\n\n## Validation and control attribution\n\n'+markdown(control_table)+'\n\n'
        f'Against the capacity-matched state+delta control, supported loss-specific superiority: **{loss_specific}**. '
        f'Against the same-backbone risk-only control, supported superiority: **{risk_only}**. '
        'Both use the predeclared 0.02 AUROC gain and paired CI above zero; absence of superiority '
        'does not demonstrate equivalence. The main gate outcome concerns clinically recoverable '
        'information, while these controls constrain attribution to clinical loss or future anatomy.\n\n'
        'Risk/error tables use seed-ensemble probabilities. Delta/PCA plots use mean predicted '
        'embeddings across seeds. For nonlinear teachers, averaging embeddings and averaging '
        'their probabilities are different operations; per-seed alignment metrics remain available.\n\n'
        'Every inferred state is a deterministic function of the current embedding and fixed forecast query. '
        'Thus [current,predicted future] contains no new information beyond those inputs. A downstream gain '
        'indicates improved clinical decodability/inductive bias, not acquisition of an unobserved future '
        'measurement. The learned oracle AUROC is a head-dependent reference, not an information-theoretic upper bound. '
        'The oracle-gap analysis tests preservation of future-risk semantics rather than '
        'proof of patient-specific future anatomy.\n\n'
        'The validated report includes unchanged pairs/splits, all prediction identities, exact old '
        'checkpoint reproduction, identical backbone counts, masked endpoint sample counts, and '
        '2,000 patient-cluster bootstrap draws.\n')
    report.write_text(text)
    assert len(list((out/'plots').glob('*.png')))>=5
    sources=json.loads((out/'fingerprints.json').read_text())
    for p,expected in sources.items():
        if p.endswith('.py'):
            assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==expected,p
    architecture=dict(predictor_parameters=parameters[0],same_backbone_all_variants=True,
        selected_oracle=json.loads((out/'oracle_selection.json').read_text()),
        colipri_frozen=True,world_model_implemented=False)
    (out/'configs/architecture_validation.json').write_text(json.dumps(architecture,indent=2))
    done=dict(status='PASS',metric_rows=len(metric),predictor_fits=len(checkpoints),
        oracle_fits=2*len(cfg['seeds']),downstream_probe_fits=(len(lock['selected'])+1)*2*len(cfg['seeds']),
        selected=chosen,test_patients=int(split.split.eq('test').sum()),all_pairs=len(pairs),
        trainable_predictor_parameters=parameters[0],loss_attribution=attribution)
    if args.log_wandb:
        import wandb
        with wandb.init(project=cfg['wandb_project'],group=cfg['wandb_group'],name='clinical-gate-final-tables',
            dir=str(out),mode='offline',config={**cfg,'dataset_version':'pilot_1000_balanced',
                'cohort':json.loads((out/'cohort_provenance.json').read_text())}) as run:
            for name in ['experiment_summary','oracle_gap_summary','paired_comparisons','per_seed_metrics',
                         'representation_metrics','clinical_alignment_metrics','subgroup_analysis']:
                run.log({name:wandb.Table(dataframe=pd.read_csv(out/(name+'.csv')))})
            run.summary.update(done)
            done['wandb_offline_run_id']=run.id
    (out/'validation.json').write_text(json.dumps(done,indent=2))
    (out/'validation_source_sha256.json').write_text(json.dumps(
        {str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
    print(json.dumps(done,indent=2))


if __name__=='__main__':main()
