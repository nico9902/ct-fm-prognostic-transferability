"""Generate manuscript tables/figure from aggregate outputs; never read patient data."""
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lung-manuscript-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'manuscript'
records = []
def read(rel):
    p = ROOT / rel
    raw = p.read_bytes()
    records.append({'path': str(rel), 'sha256': hashlib.sha256(raw).hexdigest()})
    return json.loads(raw)

settings = [('lora', ['1e-5','1e-4','1e-3']), ('partial', ['1e-6','1e-5','1e-4'])]
models = ['colipri', 'tangerine']
values = {}
rows = []
for method, rates in settings:
    for lr in rates:
        for ep in [5,20]:
            scores = []
            for model in models:
                d = read(f'outputs/lung1_phase3/lr{lr}_ep{ep}/{method}/{model}/summary.json')
                assert d['completed_folds'] == 5
                score = d['mean_fold']['uno_c_36m']
                assert 0 <= score <= 1
                values[method, model, lr, ep] = score
                scores.append(score)
            rows.append(f'{ {"lora": "LoRA", "partial": "Partial fine-tuning"}[method]} & {lr} & {ep} & {scores[0]:.4f} & {scores[1]:.4f} \\\\')
(OUT/'tables/phase3.tex').write_text(r'''\begin{longtable}{llrrr}
\caption{All 24 configurations of the preliminary fixed-epoch analysis (Appendix~\ref{app:fixed}). Each value is mean-fold Uno C at 36 months over five completed folds, evaluated at the final epoch.}\label{tab:phase3}\\
\toprule Method & Learning rate & Epochs & COLIPRI & TANGERINE \\ \midrule
\endfirsthead
\toprule Method & Learning rate & Epochs & COLIPRI & TANGERINE \\ \midrule
\endhead
'''+ '\n'.join(rows)+r'''
\bottomrule
\end{longtable}
''')
fig, axes = plt.subplots(2,2,figsize=(8,5.6),layout='constrained')
for i,(method,rates) in enumerate(settings):
    for j,model in enumerate(models):
        ax=axes[i,j]
        a=[[values[method,model,lr,ep] for ep in [5,20]] for lr in rates]
        im=ax.imshow(a,vmin=.48,vmax=.65,cmap='cividis',aspect='auto')
        ax.set(xticks=[0,1],xticklabels=['5','20'],yticks=[0,1,2],yticklabels=rates,
               xlabel='Fixed epochs',ylabel='Learning rate',title=f'{model.upper()} / {dict(lora="LoRA", partial="partial fine-tuning")[method]}')
        for y,row in enumerate(a):
            for x,v in enumerate(row):
                ax.text(x,y,f'{v:.3f}',ha='center',va='center',color='white' if v<.565 else 'black')
fig.colorbar(im,ax=axes,label='Mean-fold Uno C at 36 months',shrink=.85)
fig.savefig(OUT/'figures/phase3_sweep.pdf')
fig.savefig(OUT/'figures/phase3_sweep.png',dpi=180)
plt.close(fig)

# The external table is written by build_results_figures.py; record its source here for provenance.
external=read('outputs/phase4_external/external_validation.json')
assert external['populations']['primary']['n']==143 and external['populations']['primary']['events_by_36m']==33
(OUT/'artifact_sources.json').write_text(json.dumps({'metric':'Uno C at 36 months','sources':records,
    'phase3_configurations':24,'phase3_completed_folds':120,'new_statistical_tests':False},indent=2)+'\n')
print(f'Generated the sweep table, sweep PDF/PNG, and SHA256 provenance for {len(records)} source JSON files.')
