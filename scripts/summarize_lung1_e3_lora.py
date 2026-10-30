#!/usr/bin/env python3
"""Summarize E3 without treating overlapping draws as independent patients."""
import argparse
import json
from pathlib import Path
import pandas as pd
from run_lung1_frozen_cox_e1 import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs/lung1_e3_lora')
    output = parser.parse_args().output_root
    rows = []
    for model in ('colipri', 'tangerine'):
        for budget in (10, 25, 50, 100):
            for draw in range(1 if budget == 100 else 10):
                folder = output / f'pct{budget}/draw{draw}/lora/{model}'
                for fold in range(5):
                    record = json.loads((folder / f'folds/fold_{fold}.json').read_text())
                    rows.append({**record['metrics'], 'budget_pct': budget, 'draw': draw})
    fits = pd.DataFrame(rows)
    fits.to_csv(output / 'fit_metrics.csv', index=False)
    # Average draws within each test fold first, then weight the five folds equally.
    fold = fits.groupby(['model', 'budget_pct', 'fold'], as_index=False).uno_c_36m.mean()
    fold.to_csv(output / 'fold_metrics.csv', index=False)
    summary = fold.groupby(['model', 'budget_pct'], as_index=False).uno_c_36m.mean()
    summary.to_csv(output / 'summary.csv', index=False)
    for components, directory in [(16, 'lung1_e3_frozen_linear_pca16'),
                                   (8, 'lung1_e3_frozen_linear')]:
        frozen = pd.read_csv(ROOT / f'outputs/{directory}/fold_metrics.csv')
        keys = ['model', 'budget_pct', 'draw', 'fold']
        paired = fits[keys + ['uno_c_36m']].merge(
            frozen[keys + ['uno_c_36m']], on=keys, how='left',
            validate='one_to_one', suffixes=('_lora', '_frozen'))
        if paired.uno_c_36m_frozen.isna().any():
            raise ValueError('Missing matched frozen metrics')
        paired['difference'] = paired.uno_c_36m_lora - paired.uno_c_36m_frozen
        paired.to_csv(output / f'paired_frozen_pca{components}.csv', index=False)
        by_fold = paired.groupby(['model', 'budget_pct', 'fold']).difference.mean()
        by_fold.groupby(['model', 'budget_pct']).mean().to_csv(
            output / f'difference_summary_pca{components}.csv')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
