#!/usr/bin/env python3
"""Export manuscript figures and aggregate frozen-model results for GitHub."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'docs/assets'
RESULTS = ROOT / 'docs/results'


def main():
    ASSETS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    sources = []
    for stem in ('study_overview', 'results_rq1', 'results_rq2', 'results_rq3', 'results_rq4'):
        path = ROOT / f'manuscript/figures/{stem}.pdf'
        subprocess.run(['pdftoppm', '-singlefile', '-png', '-scale-to', '2200', str(path), str(ASSETS / stem)], check=True)
        sources.append(path)
    source = ROOT / 'outputs/phase4_external/external_validation.json'
    primary = json.loads(source.read_text())['populations']['primary']
    fields = ['model', 'internal_uno_c_36m', 'external_uno_c_36m', 'external_ci95_lower', 'external_ci95_upper', 'calibration_slope', 'predicted_mortality_36m', 'observed_mortality_36m']
    with (RESULTS / 'frozen_model_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, m in primary['models'].items():
            writer.writerow(dict(zip(fields, [name, m['internal_lung1_uno_c_36m'], m['uno_c_36m'], *m['ci95'], m.get('calibration_slope'), m.get('expected_risk_36m'), m.get('observed_risk_36m')])))
    sources.append(source)
    (RESULTS / 'provenance.json').write_text(json.dumps({
        'metric': 'Uno C at 36 months', 'internal': 'LUNG1 mean-fold estimate',
        'external_n': primary['n'], 'external_events_by_36m': primary['events_by_36m'],
        'sources': [{'path': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources]
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
