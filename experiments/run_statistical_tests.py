"""Statistical comparison of BTC against RFC and the unsupervised baselines.

Two levels of evidence are reported:
  * Per dataset: paired Wilcoxon signed-rank test over the 30 seeds, Holm-adjusted
    across datasets. This reflects algorithmic randomness on that dataset only.
  * Across datasets: Wilcoxon signed-rank test on the 13 per-dataset mean ARIs.
    This is the test relevant to claims about performance in general.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
TABLE_PATH = os.path.join(BASE_DIR, '..', 'paper', 'tables', 'statistical_tests.tex')
BTC = 'BTC_Stochastic_Delta_sqrt_none'
ALPHA = 0.05
DISPLAY = {
    'arrhythmia': 'Arrhythmia', 'brca': 'BRCA', 'ctg': 'CTG', 'dermatology': 'Dermatology',
    'diabetes': 'Diabetes', 'gbmlgg': 'GBMLGG', 'har': 'HAR', 'leukemia': 'Leukemia', 'mice': 'Mice',
    'paul15': 'Paul15', 'segment': 'Segment', 'thyroid': 'Thyroid', 'vehicle': 'Vehicle',
}
BASELINES = {
    'Baseline_RFC': 'RFC', 'Baseline_KMeans': '$K$-Means',
    'Baseline_PCA_KMeans': 'PCA + $K$-Means', 'Baseline_UMAP_KMeans': 'UMAP + $K$-Means',
}


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def fmt_p(p):
    return '$<0.001$' if p < 0.001 else f'${p:.3f}$'


def main():
    canon = pd.read_csv(os.path.join(RESULTS_DIR, 'canonical_results_30seeds.csv'))
    km = pd.read_csv(os.path.join(RESULTS_DIR, 'kmeans_baselines.csv'))
    ari = pd.concat([canon, km]).pivot_table(index=['Dataset', 'seed'], columns='Method', values='ARI')

    rows = []
    for dataset, group in ari.groupby('Dataset'):
        diff = group[BTC] - group['Baseline_RFC']
        p = 1.0 if np.allclose(diff, 0) else wilcoxon(group[BTC], group['Baseline_RFC']).pvalue
        rows.append({'Dataset': dataset, 'BTC': group[BTC].mean(), 'RFC': group['Baseline_RFC'].mean(),
                     'Difference': diff.mean(), 'p': p})
    per_dataset = pd.DataFrame(rows)
    per_dataset['p_holm'] = holm(per_dataset['p'])
    per_dataset['Outcome'] = np.where(
        per_dataset['p_holm'] >= ALPHA, 'No clear difference',
        np.where(per_dataset['Difference'] > 0, 'BTC higher', 'RFC higher'))
    per_dataset.to_csv(os.path.join(RESULTS_DIR, 'statistical_tests_per_dataset.csv'), index=False)

    means = ari.groupby('Dataset').mean()
    overall = []
    for method, label in BASELINES.items():
        res = wilcoxon(means[BTC], means[method])
        overall.append({'Method': method, 'Label': label, 'Mean_ARI': means[method].mean(),
                        'BTC_mean_ARI': means[BTC].mean(), 'BTC_higher_on': int((means[BTC] > means[method]).sum()),
                        'p': res.pvalue})
    overall = pd.DataFrame(overall)
    overall['p_holm'] = holm(overall['p'])
    overall.to_csv(os.path.join(RESULTS_DIR, 'statistical_tests_overall.csv'), index=False)

    tex = [
        r'\begin{table}[htbp]', r'\centering', r'\small',
        r'\caption{Statistical comparison of BTC with the baselines. \textbf{Top:} per-dataset comparison with RFC using paired Wilcoxon signed-rank tests over 30 seeds, Holm-adjusted across the 13 datasets; these tests capture algorithmic randomness on each dataset, not generalization to new datasets. \textbf{Bottom:} Wilcoxon signed-rank tests on the 13 per-dataset mean ARIs, Holm-adjusted across the four baselines.}',
        r'\label{tab:app_stats}', r'\begin{tabular}{lcccc}', r'\toprule',
        r'\textbf{Dataset} & \textbf{BTC} & \textbf{RFC} & \textbf{Adjusted $p$} & \textbf{Outcome} \\', r'\midrule',
    ]
    for _, r in per_dataset.iterrows():
        tex.append(f"{DISPLAY[r['Dataset']]} & ${r['BTC']:.3f}$ & ${r['RFC']:.3f}$ & {fmt_p(r['p_holm'])} & {r['Outcome']} \\\\")
    tex += [r'\midrule', r'\textbf{BTC versus} & \textbf{Baseline mean} & \textbf{BTC mean} & \textbf{Adjusted $p$} & \textbf{BTC higher on} \\', r'\midrule']
    for _, r in overall.iterrows():
        tex.append(f"{r['Label']} & ${r['Mean_ARI']:.3f}$ & ${r['BTC_mean_ARI']:.3f}$ & {fmt_p(r['p_holm'])} & {r['BTC_higher_on']} of 13 \\\\")
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    os.makedirs(os.path.dirname(TABLE_PATH), exist_ok=True)
    with open(TABLE_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tex) + '\n')
    print(per_dataset.round(4).to_string(index=False))
    print(overall.drop(columns='Label').round(4).to_string(index=False))


if __name__ == '__main__':
    main()
