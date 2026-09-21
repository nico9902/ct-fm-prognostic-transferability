#!/usr/bin/env python3
"""Write completed COLIPRI comparisons; refuse incomplete five-seed results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_tangerine_decodability import KEY, TASKS, METRICS


def table(d):
    return '\n'.join(['| '+' | '.join(d.columns)+' |', '| '+' | '.join(['---']*len(d.columns))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in d.itertuples(index=False,name=None)])


def main():
    root = Path('outputs/colipri_decodability')
    d = pd.read_csv(root/'experiment_scan_manifest.csv',dtype=str)
    meta = json.loads((root/'metadata.json').read_text())
    expected = {(t,p,s) for t in TASKS for p in ['mean','max','attention'] for s in range(5)}
    runs = pd.concat([pd.read_csv(root/p/'per_seed_results.csv').assign(pooling=p) for p in ['mean','max','attention']])
    linear = runs[runs.model.eq('linear')]
    assert len(linear)==60 and set(zip(linear.task,linear.pooling,linear.seed))==expected
    for p,dim in [('mean',864),('max',864),('attention',768)]:
        x=np.load(root/p/'cache/embeddings.npy')
        assert x.shape==(len(d),dim) and np.isfinite(x).all()
        em=pd.read_csv(root/p/'cache/embedding_metadata.csv',dtype=str).rename(columns={'screening_round':'timepoint'})
        pd.testing.assert_frame_equal(em[KEY],d[KEY])
    for r in runs.itertuples():
        pred=pd.read_csv(root/r.pooling/'runs'/f'{r.task}_{r.model}_{r.seed}'/'test_predictions.csv',dtype={k:str for k in KEY})
        pd.testing.assert_frame_equal(pred[KEY],d.loc[d.split.eq('test'),KEY].reset_index(drop=True))
        np.testing.assert_array_equal(pred.label.to_numpy(),d.loc[d.split.eq('test'),r.task].astype(float).astype(int).to_numpy())
    auroc=pd.read_csv(root/'comparison_auroc.csv').drop(columns='Rad-JEPA representation')
    report=['# Completed frozen COLIPRI NLST results',
        'These results measure clinical decodability of pretrained representations. COLIPRI used NLST images through vision-only MAE pretraining; this is not fully independent external generalization.',
        'Attention uses the frozen official pretrained pooling module; mean/max use raw dense features. All encoders remain frozen. Five-seed variability is sample SD, not a confidence interval.',
        table(auroc)]
    for m in METRICS[1:]:
        report += [f'## {m.upper()}',table(pd.read_csv(root/f'comparison_{m}.csv').drop(columns='Rad-JEPA representation'))]
    report += ['## Smoke test','```json',json.dumps(json.loads((root/'smoke_shapes.json').read_text()),indent=2),'```',
        'Rad-JEPA uses the historical best pure linear representation per target selected by test AUROC, a descriptive retrospective baseline. Interpret suspicion/nodule gains alongside AUPRC and MCC, not AUROC alone. This scan-level test does not demonstrate spatial localization. Preprocessing differs according to each model’s intended contract.']
    (root/'completed_report.md').write_text('\n\n'.join(report)+'\n')
    (root/'completion.json').write_text(json.dumps({'status':'complete','linear_runs':60,'cohort_sha256':meta['cohort_sha256'],'split_sha256':meta['split_sha256']},indent=2))
    print(table(auroc),flush=True)


if __name__=='__main__':
    main()
