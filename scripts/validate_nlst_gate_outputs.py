#!/usr/bin/env python3
"""Validate completed gate artifacts; optionally log final tables to offline W&B."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('results/nlst_gate'))
    parser.add_argument('--log-wandb',action='store_true')
    args=parser.parse_args();out=args.output_dir
    complete=json.loads((out/'COMPLETE.json').read_text())
    assert not complete['smoke'],'Cannot scientifically validate a smoke run'
    assert len(complete['seeds'])>=3
    required=['cohort_audit.csv','cohort_audit.md','experiment_summary.csv','experiment_summary.md',
              'FINAL_GATE_REPORT.md','paired_comparisons.csv','bootstrap_intervals.csv',
              'longitudinal_table.csv','patient_splits.csv','feature_metadata.csv','no_change_baseline.csv']
    for file in required:assert (out/file).stat().st_size>0,file
    splits=pd.read_csv(out/'patient_splits.csv',dtype={'patient_id':str})
    assert not splits.patient_id.duplicated().any()
    features=pd.read_csv(out/'feature_metadata.csv',dtype={'patient_id':str})
    assert features.groupby('patient_id').split.nunique().eq(1).all()
    pairs=pd.read_csv(out/'pairs_manifest.csv',dtype={'patient_id':str})
    assert pairs.groupby('patient_id').split.nunique().eq(1).all()
    comparison=pairs.merge(splits[['patient_id','split']],on='patient_id',suffixes=('_pair','_patient'))
    assert comparison.split_pair.eq(comparison.split_patient).all()
    expected=[t for t in pairs if t.startswith('primary_future_cancer_')]
    metrics=pd.read_csv(out/'per_seed_metrics.csv')
    assert set(metrics.experiment)==set(complete['experiments'])
    assert set(metrics.seed)==set(complete['seeds'])
    assert set(metrics['head'])=={'linear','mlp'}
    assert len(metrics)==len(complete['experiments'])*2*len(complete['seeds'])*3*len(expected)
    assert not metrics.duplicated(['experiment','head','seed','split','task']).any()
    assert np.isfinite(metrics[['auroc','auprc','mcc','f1']]).all().all()
    for (split,task),group in metrics.groupby(['split','task']):
        mask=pairs.split.eq(split)&pairs[task].notna()
        assert group.n.eq(mask.sum()).all()
        assert group.patients.eq(pairs.loc[mask,'patient_id'].nunique()).all()
        assert group.positives.eq(pairs.loc[mask,task].sum()).all()
    for exp in complete['experiments']:
        for head in ['linear','mlp']:
            for seed in complete['seeds']:
                work=out/'runs'/f'{exp}_{head}_seed{seed}'
                assert (work/'best.ckpt').stat().st_size>0
                p=pd.read_csv(work/'predictions.csv',dtype={'patient_id':str})
                pd.testing.assert_frame_equal(p[['patient_id','pair_type','split']],pairs[['patient_id','pair_type','split']])
                assert p.filter(like='_probability').ge(0).all().all()
                assert p.filter(like='_probability').le(1).all().all()
                assert p.filter(like='_probability').notna().all().all()
    baseline=pd.read_csv(out/'no_change_baseline.csv')
    assert {'global','disease'}==set(baseline.space)
    assert baseline[baseline.predictor.str.startswith('identity')].delta_cosine.eq(0).all()
    ci=pd.read_csv(out/'bootstrap_intervals.csv')
    assert ci.ci_low.le(ci.ci_high).all()
    assert ci.valid_draws.ge(1900).all()
    assert len(list((out/'plots').glob('*.png')))>=4
    metadata=json.loads((out/'environment.json').read_text())
    assert metadata['spatial_tokens']==[len(features),64,864]
    assert metadata['global_embeddings']==[len(features),864]
    # Add measured primary comparisons and theoretical limits, without changing the predeclared gate.
    paired=pd.read_csv(out/'paired_comparisons.csv')
    cfg=json.loads((out/'configs/resolved.json').read_text())
    primary=paired[paired.task.eq(cfg['primary_endpoint']) & paired.metric.eq('auroc')]
    architecture=json.loads((out/'configs/architecture.json').read_text())
    # Practical action is distinct from the original two-sided equivalence verdict.
    # All thresholds below use the already-declared meaningful-gain margin.
    margin=cfg['meaningful_auroc_gain']
    def rows_for(name):return primary[primary.comparison.eq(name)]
    oracle=rows_for('G1 - G0');predicted=rows_for('G3 - G0');oracle_gap=rows_for('G1 - G3')
    predictor_action=(len(oracle)==len(predicted)==len(oracle_gap)==2 and
        oracle.ci_low.gt(0).all() and oracle.gain.ge(margin).all() and
        predicted.ci_high.lt(margin).all() and oracle_gap.ci_low.gt(0).all())
    practical=('IMPROVE PREDICTOR; defer the full probabilistic world model' if predictor_action else
               'DEFER the world model; gate evidence is inconclusive')
    recommendation=dict(recommendation=practical,
        scenario='B-like predictor limitation' if predictor_action else 'inconclusive',
        evidence='Both heads support an oracle benefit and true-versus-predicted gap; both G3 upper confidence bounds exclude a +0.02 AUROC gain.' if predictor_action else 'No supported practical scenario.',
        distinction='The predeclared two-sided equivalence rule remains inconclusive: possible G3 deterioration beyond -0.02 is not ruled out. Lack of meaningful positive gain does not prove exact equivalence.',
        disease_state='No supported current-state superiority or greater longitudinal gain over global state.',
        no_full_world_model=True)
    (out/'practical_recommendation.json').write_text(json.dumps(recommendation,indent=2))
    lines=['\n\n## Measured primary comparisons and scientific limits',
        '', f"The projector alone has {architecture['projector_parameters']:,} trainable parameters; the frozen image encoder checkpoint has {architecture['checkpoint_tensor_elements_by_component']['image_encoder']:,} tensor elements.",
        '', f'**Practical recommendation: {practical}.** '+recommendation['distinction'],
        '', 'These paired differences use **seed-ensemble AUROC**,  distinct from the mean ± SD across individual seeds above.']
    names={'A: true global future':'G1 - G0','B: global change':'G2 - G0',
           'C: current disease versus global':'D0 - G0',
           'D: excess disease oracle gain':'(D1-D0) - (G1-G0)',
           'D: excess disease delta gain':'(D2-D0) - (G2-G0)',
           'Predicted future versus current':'G3 - G0',
           'True versus predicted future':'G1 - G3'}
    for title,name in names.items():
        rows=primary[primary.comparison.eq(name)]
        entries=[f"{r['head']}: {r.gain:+.3f} [95% CI {r.ci_low:+.3f}, {r.ci_high:+.3f}]" for _,r in rows.iterrows()]
        lines += ['', f"**{title}:** "+'; '.join(entries)+'.']
    btest=baseline[baseline.space.eq('global') & baseline.split.eq('test')]
    identity=btest[btest.predictor.eq('identity')].iloc[0]
    learned=btest[btest.predictor.eq('learned_ensemble')].iloc[0]
    lines += ['', f"The global predictor ensemble changes test MSE from {identity.mse:.6f} (identity) to {learned.mse:.6f}, a {100*(identity.mse-learned.mse)/identity.mse:.1f}% relative reduction. Its test cosine is {learned.cosine:.4f}, versus {identity.cosine:.4f} for identity. Reconstruction and clinical discrimination are separate requirements.",
        '', 'An oracle advantage establishes that actual future images contain decodable prognostic information. It does **not** establish that all of that information can be predicted from the current scan. Stochastic disease evolution, scan variation, and events between scans can leave an irreducible oracle gap. Improving a predictor is a research recommendation, not a guarantee that a larger dynamics model will close this gap.',
        '', 'A higher disease-state score, if present, can reflect spatial readout, additional supervision, nonlinear capacity, or regularization. This first gate does not isolate those causes. Greater disease longitudinal gain can also arise from a weaker current-state baseline; both absolute oracle performance and paired improvements should be examined. Without validated radiological-change labels, sensitivity to real radiological progression remains untested.',
        '', 'No result here proves that stable anatomy dominates the global state. The paired oracle test addresses whether clinically useful future information is present; high consecutive cosine alone cannot establish anatomical domination or forecasting success.']
    report=out/'FINAL_GATE_REPORT.md'
    base=report.read_text().split('\n\n## Measured primary comparisons and scientific limits')[0]
    base=base.replace('**Recommendation:', '**Predeclared equivalence-based gate verdict:').replace('**Final recommendation:', '**Predeclared gate verdict:')
    heading='# NLST disease-state gate report\n\n'
    base=base.replace(heading+'**Practical recommendation:',heading+'**Practical recommendation:',1)
    if not base.startswith(heading+'**Practical recommendation:'):
        base=base.replace(heading,heading+f'**Practical recommendation: {practical}.**\n\n',1)
    report.write_text(base+'\n'.join(lines)+'\n')
    result=dict(status='PASS',neural_probe_fits=len(complete['experiments'])*2*len(complete['seeds']),
        projector_fits=len(complete['seeds']),metric_rows=len(metrics),patients=features.patient_id.nunique(),
        scans=len(features),pairs=len(pairs),checks=required)
    if args.log_wandb:
        import wandb
        cfg=json.loads((out/'configs/resolved.json').read_text())
        with wandb.init(project=cfg['wandb_project'],group=cfg['wandb_group'],name='consolidated-gate-results',
                        dir=str(out),mode='offline',config=cfg) as run:
            run.log({'train_val_test_metrics':wandb.Table(dataframe=metrics),
                     'paired_patient_comparisons':wandb.Table(dataframe=pd.read_csv(out/'paired_comparisons.csv')),
                     'test_bootstrap_intervals':wandb.Table(dataframe=ci),
                     'cohort_audit':wandb.Table(dataframe=pd.read_csv(out/'cohort_audit.csv')),
                     'no_change_baseline':wandb.Table(dataframe=baseline)})
            run.summary.update(result)
            result['wandb_offline_run_id']=run.id
    (out/'validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
