"""Generate manuscript figures and an ablation table from recorded trials."""
import gzip
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    out = Path('results_algorithm1')
    with gzip.open(out / 'experiments.json.gz', 'rt', encoding='utf-8') as f:
        data = json.load(f)
    cases = data['cases']
    assert len(cases) == 45 and all(len(r['trials']) == 300 for r in cases)
    for covariance in ['identity', 'equicorr']:
        paired = [r for r in cases if r['config']['group']=='ablation'
                  and r['config']['covariance']==covariance
                  and r['config']['variant'] in {'ecc','simple_diff'}]
        assert paired[0]['trials'] == paired[1]['trials']
    plt.rcParams.update({'font.size': 9})

    def save(fig, name):
        fig.savefig(out / (name+'.pdf'), bbox_inches='tight')
        fig.savefig(out / (name+'.png'), dpi=180, bbox_inches='tight')
        plt.close(fig)

    tr = data['trajectory']
    h = tr['history']
    fig, axes = plt.subplots(3, 1, figsize=(7, 5.6), sharex=True)
    scores = np.array([r['scores'] for r in h]).T
    actions = np.array([r['action'] for r in h]).T
    for ax, values, title in zip(axes[:2], [scores, actions],
                                  ['(a) Pseudo-scores (not posterior probabilities)', '(b) Sensing weights']):
        limit = np.abs(values).max()
        im = ax.imshow(values, origin='lower', aspect='auto', cmap='RdBu_r', vmin=-limit, vmax=limit,
                       extent=[.5,len(h)+.5,.5,15.5])
        ax.set_ylabel('Stream')
        ax.set_title(title, loc='left')
        fig.colorbar(im, ax=ax, fraction=.025, pad=.02)
    axes[2].plot([r['t'] for r in h], [r['likelihood_gap'] for r in h])
    axes[2].axhline(math.log((math.comb(15,3)-1)/.05), ls='--', color='k', label='Stopping boundary')
    axes[2].set_title('(c) Exact best-versus-second likelihood gap', loc='left')
    axes[2].set_xlabel('Observation t')
    axes[2].set_ylabel('Log likelihood gap')
    axes[2].legend(frameon=False)
    fig.tight_layout()
    save(fig,'fig_inside_algorithm1')

    fig, axes = plt.subplots(2,3,figsize=(9,5))
    specs = [('anomalies','n','Anomaly count (K=12)'),
             ('confidence','delta','Target error'),
             ('correlation','rho','Correlation'),
             ('budget','B','Amplitude budget'),
             ('signal','signal','Mean shift'),
             ('geometry','subset','Anomaly locations')]
    for ax, (group,key,label) in zip(axes.flat,specs):
        rows = [r for r in cases if r['config']['group']==group]
        families = ['equicorr','toeplitz'] if group=='correlation' else [None]
        for family in families:
            rs = [r for r in rows if family is None or r['config']['covariance']==family]
            rs.sort(key=lambda r:r['config'][key])
            xs = range(len(rs)) if key=='subset' else [r['config'][key] for r in rs]
            ax.errorbar(xs, [r['summary']['mean_tau_capped'] for r in rs],
                        yerr=[1.96*r['summary']['mean_tau_se'] for r in rs],
                        marker='o', ms=3, capsize=3, label=family)
            for x,r in zip(xs,rs):
                if r['summary']['timeouts']:
                    ax.annotate(f"{r['summary']['timeouts']} timeout", (x,r['summary']['mean_tau_capped']),
                                xytext=(6,-12), textcoords='offset points', fontsize=8)
            if key=='subset':
                ax.set_xticks(list(xs), ['{'+','.join(str(k+1) for k in r['config'][key])+'}' for r in rs])
        if group=='confidence':
            ax.set_xscale('log')
        if group=='correlation':
            ax.legend(frameon=False)
        ax.set_xlabel(label)
        ax.set_ylabel('Mean capped observations')
        ax.grid(alpha=.2)
    fig.tight_layout()
    save(fig,'fig_algorithm1_sweeps')

    names = {'ecc':'ECC-AHT','simple_diff':'Plain contrast', 'coordinate':'Uniform coordinates',
             'random_pair':'Random pair','no_exploration':'No exploration'}
    lines = [r'\begin{table*}[t]',r'\centering\footnotesize',
             r'\caption{Each sensing and exploration ablation uses 300 trials with $K=20$, $n=2$, $S^\star=\{1,2\}$, and $\delta=.05$. All rows use the same exact stopping rule. We count a timeout when a run reaches 5000 observations without a decision.}',
             r'\label{tab:algorithm1-ablation}',r'\begin{tabular}{llrrr}',r'\toprule',
             r'Covariance & Sensing policy & Mean $\tau$ (SE) & Errors & Timeouts\\',r'\midrule']
    for row in cases:
        c,s = row['config'],row['summary']
        if c['group']!='ablation':
            continue
        cov = '$I$' if c['covariance']=='identity' else ('Equicorr' if c['covariance']=='equicorr' else 'Toeplitz')+f"$({c['rho']})$"
        lines.append(f"{cov} & {names[c['variant']]} & {s['mean_tau_capped']:.2f} ({s['mean_tau_se']:.2f}) & {s['wrong_outputs']} & {s['timeouts']}"+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    (out/'algorithm1_results.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f"Generated 2 figures and ablation table; trajectory tau={tr['tau']}, correct={tr['correct']}")


if __name__=='__main__':
    main()
