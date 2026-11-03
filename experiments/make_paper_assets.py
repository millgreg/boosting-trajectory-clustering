"""Generate every table, figure, and reported number of the manuscript from experiments/results.

Writes paper/tables/*.tex, paper/figures/*.pdf, and paper/numbers.tex. The manuscript refers to
numbers only through the macros in numbers.tex, so text and data cannot drift apart.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RES = os.path.join(ROOT, 'experiments', 'results')
CONF = os.path.join(ROOT, 'experiments', 'confirmation')
PAPER = os.path.join(ROOT, 'paper')
TABLES = os.path.join(PAPER, 'tables')
FIGURES = os.path.join(PAPER, 'figures')

BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, INK2, NEUTRAL, GRID = '#0b0b0b', '#52514e', '#9a9893', '#e6e5e1'
DEV_NAMES = {'arrhythmia': 'Arrhythmia', 'brca': 'BRCA', 'ctg': 'CTG', 'dermatology': 'Dermatology', 'diabetes': 'Diabetes',
             'gbmlgg': 'GBMLGG', 'har': 'HAR', 'leukemia': 'Leukemia', 'mice': 'Mice', 'paul15': 'Paul15',
             'segment': 'Segment', 'thyroid': 'Thyroid', 'vehicle': 'Vehicle'}
macros = {}


def bootstrap(x, n=20000, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return np.percentile(x[np.random.RandomState(seed).randint(0, len(x), (n, len(x)))].mean(axis=1), [2.5, 97.5])


def cluster_bootstrap(values, groups, n=20000, seed=0):
    frame = pd.DataFrame({'v': values, 'g': groups}).dropna()
    sums, counts = frame.groupby('g').v.sum().values, frame.groupby('g').v.count().values
    draws = np.random.RandomState(seed).randint(0, len(sums), (n, len(sums)))
    return np.percentile(sums[draws].sum(axis=1) / counts[draws].sum(axis=1), [2.5, 97.5])


def f3(x):
    if np.isnan(x):
        return '--'
    if round(x, 3) == 0:
        x = 0.0
    return f'{x:.3f}'.replace('-', '$-$')


def num(name, value, digits=3):
    text = f'{value:.{digits}f}' if isinstance(value, (float, np.floating)) else str(value)
    macros[name] = text.replace('-', '-')


def ci(name, lo, hi):
    num(name + 'Lo', lo)
    num(name + 'Hi', hi)


def write(name, lines):
    with open(os.path.join(TABLES, name), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def latex_name(s):
    return s.replace('&', r'\&').replace('%', r'\%').replace('_', r'\_').replace('#', r'\#')


def style(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(NEUTRAL)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(axis='x', color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- development benchmarks
def development():
    d = pd.read_csv(os.path.join(RES, 'history_study_development.csv'))
    t = d.groupby(['Dataset', 'Method']).ARI.mean().unstack()
    rows = [('Boosting', 'BTC tree updates', 'BTC (per-tree updates)'),
            ('Boosting', 'Stump leaves, unweighted', 'Stump leaves, unweighted'),
            ('Boosting', 'Per-feature sums, SHAP-equivalent', 'Per-feature sums'),
            ('Boosting', 'TreeSHAP, XGBoost depth 3', 'TreeSHAP (depth 3)'),
            ('History', 'Historic residuals', 'Pseudo-residual history'),
            ('History', 'Historic residuals, order shuffled', 'Pseudo-residuals, tree order shuffled'),
            ('History', 'Residual change', 'Round-to-round residual change'),
            ('History', 'Learning-speed summaries', 'Learning-speed summaries'),
            ('History', 'Final margin', 'Final margin only'),
            ('Control', 'BTC, permuted proxy labels', 'BTC, permuted proxy labels'),
            ('Forest', 'RF leaves with SVD', 'RF leaves with SVD'),
            ('Forest', 'RFC spectral', 'Spectral RFC'),
            ('Unsupervised', 'Raw K-Means', '$K$-Means, raw features'),
            ('Unsupervised', 'K-Means scaled', '$K$-Means, standardized'),
            ('Unsupervised', 'UMAP + K-Means', 'UMAP + $K$-Means, raw'),
            ('Unsupervised', 'UMAP scaled + K-Means', 'UMAP + $K$-Means, standardized')]
    lines = [r'\begin{table}[t]', r'\centering', r'\small',
             r'\caption{Development benchmarks (13 datasets, 30 seeds). Mean ARI across datasets and the difference from BTC with a 95\% bootstrap interval over datasets. These datasets informed all design choices and are not used for confirmatory claims.}',
             r'\label{tab:development}', r'\begin{tabular}{llcc}', r'\toprule',
             r'& \textbf{Representation} & \textbf{Mean ARI} & \textbf{Method minus BTC [95\% CI]} \\', r'\midrule']
    last = None
    for group, key, label in rows:
        if last is not None and group != last:
            lines.append(r'\addlinespace')
        diff = t[key] - t['BTC tree updates']
        cell = '--' if key == 'BTC tree updates' else f'{f3(diff.mean())} [{f3(bootstrap(diff)[0])}, {f3(bootstrap(diff)[1])}]'
        lines.append(f'{group if group != last else ""} & {label} & {f3(t[key].mean())} & {cell} \\\\')
        last = group
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    write('development.tex', lines)
    num('DevBTC', t['BTC tree updates'].mean())
    num('DevResidual', t['Historic residuals'].mean())
    num('DevMargin', t['Final margin'].mean())
    num('DevSpeed', t['Learning-speed summaries'].mean())
    num('DevPermuted', t['BTC, permuted proxy labels'].mean())
    num('DevShapThree', t['TreeSHAP, XGBoost depth 3'].mean())
    num('DevRFSVD', t['RF leaves with SVD'].mean())
    num('DevKMRaw', t['Raw K-Means'].mean())
    num('DevKMScaled', t['K-Means scaled'].mean())
    x = t['Historic residuals'] - t['Historic residuals, order shuffled']
    num('DevOrderDiff', x.mean()); ci('DevOrderDiff', *bootstrap(x))
    x = t['BTC tree updates'] - t['BTC tree updates, order shuffled']
    num('DevBTCOrderDiff', abs(x).max())
    btc_runs = d[d.Method == 'BTC tree updates'].groupby('Dataset')[['Mean_homogeneity', 'Trees_gt_95']].mean()
    num('HarTrees', btc_runs.loc['har', 'Trees_gt_95'], 1)
    num('HarHomogeneity', 100 * btc_runs.loc['har', 'Mean_homogeneity'], 1)
    num('HarBTC', t.loc['har', 'BTC tree updates']); num('HarUMAP', t.loc['har', 'UMAP + K-Means'])

    proxies = {'arrhythmia': 'Normal ECG vs.\\ arrhythmia', 'brca': 'ER-negative vs.\\ ER-positive', 'ctg': 'Normal vs.\\ abnormal tracing',
               'dermatology': 'Psoriasis vs.\\ other$^\\dagger$', 'diabetes': 'No readmission vs.\\ readmission', 'gbmlgg': 'Alive vs.\\ deceased',
               'har': 'Static vs.\\ dynamic activity', 'leukemia': 'AML vs.\\ ALL', 'mice': 'Control vs.\\ trisomic',
               'paul15': 'Non-erythroid vs.\\ erythroid', 'segment': 'Brickface vs.\\ other$^\\dagger$', 'thyroid': 'Normal vs.\\ hypothyroid',
               'vehicle': 'Bus vs.\\ other$^\\dagger$'}
    targets = {'arrhythmia': 'Arrhythmia classes', 'brca': 'PAM50 subtypes', 'ctg': 'Suspect, pathologic', 'dermatology': 'Skin diseases',
               'diabetes': 'ICD-9 diagnosis groups', 'gbmlgg': 'Histological grade', 'har': 'Walking activities', 'leukemia': 'B-cell, T-cell ALL',
               'mice': 'Behavior and treatment', 'paul15': 'Erythroid clusters', 'segment': 'Image segments', 'thyroid': 'Hypothyroid types',
               'vehicle': 'Vehicle types'}
    meta = d.groupby('Dataset')[['N_full', 'N_target', 'N_features', 'K']].first()
    lines = [r'\begin{table}[ht]', r'\centering', r'\small',
             r'\caption{Development benchmarks. $N$ is the number of samples used to train the proxy model, $N_{\text{target}}$ the cohort that is clustered, $P$ the number of features after preprocessing, and $K$ the number of subtypes. BRCA and GBMLGG cluster all samples (cross-cutting proxy); the others cluster the $y=1$ class. $^\dagger$Arbitrary proxy: the first class in sorted order.}',
             r'\label{tab:datasets}', r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{lrrrrll}', r'\toprule',
             r'\textbf{Dataset} & $N$ & $N_{\text{target}}$ & $P$ & $K$ & \textbf{Proxy ($y=0$ vs.\ $y=1$)} & \textbf{Subtypes} \\', r'\midrule']
    for ds, row in meta.iterrows():
        lines.append(f'{DEV_NAMES[ds]} & {int(row.N_full):,} & {int(row.N_target):,} & {int(row.N_features):,} & {int(row.K)} & {proxies[ds]} & {targets[ds]} \\\\'.replace(',', '{,}'))
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    write('datasets.tex', lines)

    lines = [r'\begin{table}[ht]', r'\centering', r'\scriptsize', r'\setlength{\tabcolsep}{3pt}',
             r'\caption{Development benchmarks by dataset (mean ARI over 30 seeds).}', r'\label{tab:development_by_dataset}',
             r'\begin{tabular}{lcccccccc}', r'\toprule',
             r'\textbf{Dataset} & \textbf{BTC} & \textbf{Per-feat.} & \textbf{TreeSHAP} & \textbf{Residuals} & \textbf{RF SVD} & \textbf{RFC} & \textbf{$K$-M raw} & \textbf{$K$-M std.} \\', r'\midrule']
    cols = ['BTC tree updates', 'Per-feature sums, SHAP-equivalent', 'TreeSHAP, XGBoost depth 3', 'Historic residuals',
            'RF leaves with SVD', 'RFC spectral', 'Raw K-Means', 'K-Means scaled']
    for ds in t.index:
        lines.append(DEV_NAMES[ds] + ' & ' + ' & '.join(f3(t.loc[ds, c]) for c in cols) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    write('development_by_dataset.tex', lines)


def brca_receptors():
    import sys
    sys.path.insert(0, ROOT)
    from experiments.dataset_loader import load_dataset
    _, er, _, subtype, _ = load_dataset('brca')
    names = np.array(['Basal', 'HER2', 'LumA', 'LumB'])[subtype]
    num('BrcaBasalERNeg', 100 * float(np.mean(er[names == 'Basal'] == 0)), 0)
    num('BrcaLumAERPos', 100 * float(np.mean(er[names == 'LumA'] == 1)), 0)


def target_direction():
    d = pd.read_csv(os.path.join(RES, 'target_direction.csv'))
    t = d.groupby(['Dataset', 'Method']).ARI.mean().unstack()
    num('TdResidual', t['Residual history'].mean())
    num('TdResidualPCRemoved', t['Residual history, first component removed'].mean())
    num('TdResidualMarginRemoved', t['Residual history, final margin regressed out'].mean())
    num('TdBTC', t['BTC tree updates'].mean())
    num('TdBTCMarginRemoved', t['BTC tree updates, final margin regressed out'].mean())


def sensitivity():
    d = pd.read_csv(os.path.join(RES, 'default_sensitivity.csv'))
    t = d.groupby(['Dataset', 'proxy', 'Method']).ARI.mean().groupby(['Dataset', 'Method']).mean().unstack()
    cc = t.index.str.startswith('cc18_')
    order = ['Default', 'Trees 50', 'Trees 300', 'Learning rate 0.05', 'Learning rate 0.3', 'Depth 2', 'Depth 3',
             'Features all', 'Features 10%', 'Row subsample 0.5', 'Rescaled per tree', 'Agglomerative backend',
             'RF leaves with SVD', 'Raw K-Means']
    lines = [r'\begin{table}[ht]', r'\centering', r'\small',
             f'\\caption{{Sensitivity of the BTC default, one setting changed at a time (13 benchmarks and {int(cc.sum())} OpenML-CC18 datasets with random class groupings, 5 seeds). Mean ARI, and the difference from the default over all {len(t)} datasets with a 95\\% bootstrap interval.}}',
             r'\label{tab:sensitivity}', r'\begin{tabular}{lcccc}', r'\toprule',
             r'\textbf{Setting} & \textbf{13 benchmarks} & \textbf{CC18} & \textbf{All} & \textbf{Minus default [95\% CI]} \\', r'\midrule']
    for m in order:
        diff = t[m] - t['Default']
        cell = '--' if m == 'Default' else f'{f3(diff.mean())} [{f3(bootstrap(diff)[0])}, {f3(bootstrap(diff)[1])}]'
        label = latex_name(m.replace('Features', 'Feature subsampling').replace('Rescaled per tree', 'Updates standardized per tree'))
        lines.append(f'{label} & {f3(t.loc[~cc, m].mean())} & {f3(t.loc[cc, m].mean())} & {f3(t[m].mean())} & {cell} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    write('sensitivity.tex', lines)
    num('SensNCC', int(cc.sum()))
    num('SensCCBTC', t.loc[cc, 'Default'].mean()); num('SensCCRF', t.loc[cc, 'RF leaves with SVD'].mean())
    num('SensCCKM', t.loc[cc, 'Raw K-Means'].mean())
    x = (t['Default'] - t['RF leaves with SVD'])[cc]
    num('SensCCDiff', x.mean()); ci('SensCCDiff', *bootstrap(x))


def shap_setups():
    d = pd.read_csv(os.path.join(RES, 'dev_shap_setups.csv'))
    t = d.groupby(['Dataset', 'Method']).ARI.mean().unstack()
    best = t.drop(columns='BTC').mean().idxmax()
    x = t['BTC'] - t[best]
    num('ShapNSetups', t.shape[1] - 1)
    num('ShapDevDiff', x.mean()); ci('ShapDevDiff', *bootstrap(x))
    num('ShapDevBest', t[best].mean())
    lines = [r'\begin{table}[ht]', r'\centering', r'\small',
             f'\\caption{{TreeSHAP clustering setups on the 13 benchmarks (10 seeds), ranked by mean ARI. The best setup was frozen as the TreeSHAP comparator for confirmation; BTC scored {f3(t["BTC"].mean())}.}}',
             r'\label{tab:shap}', r'\begin{tabular}{lccc}', r'\toprule',
             r'\textbf{Tree depth} & \textbf{Column subsampling} & \textbf{Clustering} & \textbf{Mean ARI} \\', r'\midrule']
    for m, v in t.drop(columns='BTC').mean().sort_values(ascending=False).items():
        depth, cols, how = m.replace('TreeSHAP depth ', '').split(', ')
        how = {'kmeans': '$K$-Means', 'kmeans_scaled': '$K$-Means, standardized', 'umap': 'UMAP + $K$-Means', 'ward': 'Ward'}[how]
        lines.append(f'{depth} & {cols.replace("columns ", "")} & {how} & {f3(v)} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    write('shap_setups.tex', lines)


def homogeneity():
    from scipy.stats import spearmanr
    h = pd.read_csv(os.path.join(RES, 'dev_homogeneity_fixed_cohort.csv'))
    g = h.groupby(['Dataset', 'background']).mean(numeric_only=True).reset_index()
    rho = [spearmanr(x.Mean_homogeneity, x.BTC).statistic for _, x in g.groupby('Dataset') if x.BTC.nunique() > 2]
    rho = np.array([r for r in rho if not np.isnan(r)])
    num('HomFixedRho', rho.mean()); ci('HomFixedRho', *bootstrap(rho)); num('HomFixedN', len(rho))


def proxy_contrast():
    c = pd.read_csv(os.path.join(RES, 'proxy_contrast.csv'))
    c['gain'] = c['BTC'] - c['K-Means scaled']
    u = c.groupby(['Dataset', 'Proxy', 'grouping']).gain.mean().groupby(['Dataset', 'Proxy']).mean().unstack()
    primary = ['arrhythmia', 'ctg', 'har', 'leukemia', 'mice', 'paul15', 'thyroid']
    x = (u.real - u.random).loc[primary]
    num('DevContrast', x.mean()); ci('DevContrast', *bootstrap(x))


# ---------------------------------------------------------------- confirmation
def confirmation():
    m = pd.read_csv(os.path.join(RES, 'confirmation_results.csv'))
    r = pd.read_csv(os.path.join(RES, 'confirmation_rfc_exploratory.csv'))
    m = m[m.Method != 'FAILED']
    d = pd.concat([m[['did', 'Dataset', 'grouping', 'seed', 'Method', 'ARI']], r[['did', 'Dataset', 'grouping', 'seed', 'Method', 'ARI']]])
    t = d.groupby(['did', 'Dataset', 'grouping', 'Method']).ARI.mean().groupby(['did', 'Dataset', 'Method']).mean().unstack()
    s = pd.read_csv(os.path.join(RES, 'confirmation_summary.csv'))
    audit = pd.read_csv(os.path.join(CONF, 'confirmation_audit.csv'))
    num('ConfN', t.shape[0]); num('ConfCandidates', len(audit)); num('ConfExcluded', int(audit.excluded.notna().sum()))
    for key, name in [('BTC', 'BTC'), ('TreeSHAP', 'Shap'), ('Per-feature sums', 'PerFeature'), ('RF leaves with SVD', 'RFSVD'),
                      ('RFC spectral', 'RFC'), ('K-Means raw', 'KMRaw'), ('K-Means scaled', 'KMScaled'), ('UMAP raw', 'UMAPRaw'),
                      ('UMAP scaled', 'UMAPScaled'), ('Historic residuals', 'Residual'), ('Final margin', 'Margin')]:
        num('Conf' + name, t[key].mean())

    labels = {'BTC minus TreeSHAP': ('C1', 'BTC $-$ TreeSHAP', 'Shap'),
              'BTC minus Per-feature sums': ('C1', 'BTC $-$ per-feature sums', 'PerFeature'),
              'Historic residuals minus Final margin': ('C2', 'Residual history $-$ final margin', 'ResMargin'),
              'Historic residuals minus Historic residuals, order shuffled': ('C2', 'Residual history $-$ shuffled order', 'ResOrder'),
              'BTC minus RF leaves with SVD': ('C3', 'BTC $-$ RF leaves with SVD', 'RFSVD'),
              'BTC minus K-Means raw': ('C3', 'BTC $-$ $K$-Means, raw', 'KMRaw'),
              'BTC minus K-Means scaled': ('C3', 'BTC $-$ $K$-Means, standardized', 'KMScaled'),
              'BTC minus UMAP raw': ('C3', 'BTC $-$ UMAP, raw', 'UMAPRaw'),
              'BTC minus UMAP scaled': ('C3', 'BTC $-$ UMAP, standardized', 'UMAPScaled')}
    lines = [r'\begin{table}[t]', r'\centering', r'\small',
             f'\\caption{{Confirmation on {t.shape[0]} held-out OpenML datasets with random class groupings, analysed exactly as pre-specified. Differences in mean ARI with 95\\% bootstrap intervals over datasets; a difference is claimed only when the interval excludes zero. Spectral RFC was added after the protocol was frozen and is exploratory.}}',
             r'\label{tab:confirmation}', r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{llccc}', r'\toprule',
             r'\textbf{Claim} & \textbf{Comparison} & \textbf{Difference [95\% CI]} & \textbf{A higher / lower} & \textbf{Verdict} \\', r'\midrule']
    for _, row in s.iterrows():
        claim, label, name = labels[row.Comparison]
        status = row.Claim.split()[1]
        verdict = 'difference' if row.Verdict == 'difference' else 'no claim'
        lines.append(f'{claim} ({status}) & {label} & {f3(row.Difference)} [{f3(row.Lower)}, {f3(row.Upper)}] & {row["A higher on"]} / {row["A lower on"]} & {verdict} \\\\')
        num('Conf' + name + 'Diff', row.Difference); ci('Conf' + name + 'Diff', row.Lower, row.Upper)
    x = t['BTC'] - t['RFC spectral']
    lo, hi = bootstrap(x)
    lines.append(r'\midrule')
    lines.append(f'Exploratory & BTC $-$ spectral RFC & {f3(x.mean())} [{f3(lo)}, {f3(hi)}] & {int((x > 5e-4).sum())} / {int((x < -5e-4).sum())} & difference \\\\')
    num('ConfRFCDiff', x.mean()); ci('ConfRFCDiff', lo, hi); num('ConfRFCLower', int((x < -5e-4).sum()))
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    write('confirmation.tex', lines)

    gap = (t['K-Means raw'] - t['K-Means scaled']).abs()
    num('ConfScaleGap', gap.mean()); num('ConfScaleGapN', int((gap > 0.05).sum()))
    from scipy.stats import spearmanr
    btc = m[m.Method == 'BTC'].groupby(['did', 'grouping'])[['ARI', 'Mean_homogeneity']].mean()
    num('ConfHomRho', spearmanr(btc.Mean_homogeneity, btc.ARI).statistic)

    # Figure: mean ARI with dataset-bootstrap intervals.
    methods = [('RFC spectral', 'Spectral RFC*'), ('TreeSHAP', 'TreeSHAP'), ('UMAP scaled', 'UMAP + $K$-Means, std.'), ('BTC', 'BTC'),
               ('RF leaves with SVD', 'RF leaves + SVD'), ('UMAP raw', 'UMAP + $K$-Means, raw'), ('Per-feature sums', 'Per-feature sums'),
               ('K-Means scaled', '$K$-Means, std.'), ('Historic residuals', 'Residual history'), ('Final margin', 'Final margin'),
               ('K-Means raw', '$K$-Means, raw')]
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    for i, (key, label) in enumerate(methods[::-1]):
        lo, hi = bootstrap(t[key])
        color = BLUE if key == 'BTC' else INK2
        ax.plot([lo, hi], [i, i], color=color, linewidth=1.5, solid_capstyle='round')
        ax.plot(t[key].mean(), i, 'o', color=color, markersize=5, markeredgecolor='white', markeredgewidth=0.8)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([lab for _, lab in methods[::-1]])
    for tick in ax.get_yticklabels():
        if tick.get_text() == 'BTC':
            tick.set_color(BLUE); tick.set_fontweight('bold')
    ax.set_xlabel('Mean ARI across 40 held-out datasets (95% bootstrap interval)', color=INK2, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'confirmation.pdf'))
    plt.close(fig)

    # Figure: sensitivity of K-Means to feature scaling.
    fig, ax = plt.subplots(figsize=(3.3, 3.2))
    ax.plot([0, 1], [0, 1], color=NEUTRAL, linewidth=1, linestyle='--')
    ax.plot(t['K-Means raw'], t['K-Means scaled'], 'o', color=BLUE, markersize=4.5, markeredgecolor='white', markeredgewidth=0.6)
    ax.set_xlabel('$K$-Means ARI, raw features', color=INK2, fontsize=8)
    ax.set_ylabel('$K$-Means ARI, standardized features', color=INK2, fontsize=8)
    lim = max(t['K-Means raw'].max(), t['K-Means scaled'].max()) + 0.05
    ax.set_xlim(-0.05, lim); ax.set_ylim(-0.05, lim)
    style(ax); ax.grid(axis='y', color=GRID, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'scaling.pdf'))
    plt.close(fig)

    header = r'\textbf{Dataset} & \textbf{$N$} & \textbf{$K$} & \textbf{BTC} & \textbf{TreeSHAP} & \textbf{RF SVD} & \textbf{RFC} & \textbf{UMAP std.} & \textbf{$K$-M raw} & \textbf{$K$-M std.} \\'
    lines = [r'\begin{longtable}{lrrccccccc}',
             f'\\caption{{Confirmation results by dataset (mean ARI over three groupings and five seeds).}} \\label{{tab:confirmation_by_dataset}} \\\\',
             r'\toprule', header, r'\midrule', r'\endfirsthead',
             r'\multicolumn{10}{l}{\textit{Continued from the previous page}} \\', r'\toprule', header, r'\midrule', r'\endhead']
    meta = m.groupby('did')[['N_full', 'K']].agg({'N_full': 'first', 'K': 'max'})
    for (did, name), row in t.iterrows():
        cells = ' & '.join(f3(row[c]) for c in ['BTC', 'TreeSHAP', 'RF leaves with SVD', 'RFC spectral', 'UMAP scaled', 'K-Means raw', 'K-Means scaled'])
        lines.append(f'{latex_name(name[:26])} &{int(meta.loc[did, "N_full"])} & {int(meta.loc[did, "K"])} & {cells} \\\\')
    lines += [r'\bottomrule', r'\end{longtable}']
    write('confirmation_by_dataset.tex', lines)

    reasons = audit.excluded.fillna('').str.split(';').explode().str.strip()
    reasons = reasons[reasons != ''].str.replace(r'^ordinal target:.*', 'Ordinal target', regex=True) \
        .str.replace(r'^review:.*', 'Review: duplicate, ordinal, or overlapping classes', regex=True) \
        .str.replace(r'^load failure:.*', 'Load failure', regex=True).str.replace('single feature determines a class', 'A single feature separates a class') \
        .str.replace('duplicated rows', 'At least 20\\% duplicated rows').str.replace('too few features', 'Fewer than 4 features') \
        .str.replace('mostly missing', 'At least 50\\% missing values')
    lines = [r'\begin{table}[ht]', r'\centering', r'\small',
             f'\\caption{{Exclusions while building the confirmation set ({len(audit)} candidates, {t.shape[0]} included). A dataset can have several reasons.}}',
             r'\label{tab:audit}', r'\begin{tabular}{lc}', r'\toprule', r'\textbf{Reason} & \textbf{Datasets} \\', r'\midrule']
    for reason, n in reasons.value_counts().items():
        lines.append(f'{reason} & {n} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    write('audit.tex', lines)


# ---------------------------------------------------------------- TCGA
def tcga():
    h = pd.read_csv(os.path.join(RES, 'tcga_hierarchy.csv'))
    t = h.groupby(['Level', 'Task', 'Proxy', 'grouping', 'Method']).ARI.mean().unstack()
    t['Best K-Means'] = t[['K-Means raw', 'K-Means scaled']].max(axis=1)
    methods = ['BTC', 'TreeSHAP', 'RF leaves with SVD', 'RFC spectral', 'K-Means raw', 'K-Means scaled']
    a = t.loc['A']
    real = a.xs('real', level='Proxy').droplevel('grouping')
    random = a.xs('random', level='Proxy').groupby('Task').mean()
    lines = [r'\begin{table}[t]', r'\centering', r'\small',
             r'\caption{TCGA, organ system to cancer type. Mean ARI over 5 seeds for each PanCanAtlas grouping (proxy: the group versus all other cancers; subtypes: its cancer types), with the mean over three random groupings of the same size in the last row.}',
             r'\label{tab:tcga_a}', r'\begin{tabular}{lcccccc}', r'\toprule',
             r'\textbf{Grouping} & \textbf{BTC} & \textbf{TreeSHAP} & \textbf{RF SVD} & \textbf{RFC} & \textbf{$K$-M raw} & \textbf{$K$-M std.} \\', r'\midrule']
    for task, row in real.iterrows():
        lines.append(f'{task} & ' + ' & '.join(f3(row[m]) for m in methods) + r' \\')
    lines.append(r'\midrule')
    lines.append(r'Mean, real groupings & ' + ' & '.join(f3(real[m].mean()) for m in methods) + r' \\')
    lines.append(r'Mean, random groupings & ' + ' & '.join(f3(random[m].mean()) for m in methods) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    write('tcga_organ.tex', lines)
    for m, name in [('BTC', 'BTC'), ('TreeSHAP', 'Shap'), ('RF leaves with SVD', 'RFSVD'), ('RFC spectral', 'RFC'), ('Best K-Means', 'KM')]:
        num('TcgaA' + name, real[m].mean())
        num('TcgaARandom' + name, random[m].mean())
    num('TcgaLungBTC', real.loc['Lung', 'BTC']); num('TcgaLungKM', real.loc['Lung', 'Best K-Means'])
    num('TcgaKidneyBTC', real.loc['Pan-Kidney', 'BTC']); num('TcgaKidneyKM', real.loc['Pan-Kidney', 'Best K-Means'])

    s = pd.read_csv(os.path.join(RES, 'tcga_hierarchy_summary.csv'))
    for _, row in s.iterrows():
        key = {'BTC: real minus random gain': 'ContrastBTC', 'TreeSHAP: real minus random gain': 'ContrastShap',
               'RF leaves with SVD: real minus random gain': 'ContrastRFSVD', 'RFC spectral: real minus random gain': 'ContrastRFC',
               'BTC gain over best K-Means': 'BGainBTC', 'BTC minus TreeSHAP': 'BBTCShap', 'BTC minus RF leaves with SVD': 'BBTCRFSVD',
               'BTC minus RFC spectral': 'BBTCRFC', 'RFC spectral gain over best K-Means': 'BGainRFC',
               'TreeSHAP gain over best K-Means': 'BGainShap'}[row.Comparison]
        num('Tcga' + key, row.Difference); ci('Tcga' + key, row.Lower, row.Upper)
        if key.startswith('BGain'):
            num('Tcga' + key + 'Pos', int(row['Positive in']))
    b = t.loc['B'].droplevel(['Proxy', 'grouping'])
    gains_b = pd.DataFrame({m: b[m] - b['Best K-Means'] for m in ['BTC', 'TreeSHAP', 'RF leaves with SVD', 'RFC spectral']})
    lo, hi = bootstrap(gains_b['RF leaves with SVD'])
    num('TcgaBGainRFSVD', gains_b['RF leaves with SVD'].mean()); ci('TcgaBGainRFSVD', lo, hi)
    num('TcgaBTasks', len(b))

    w = pd.read_csv(os.path.join(RES, 'tcga_within_cohort.csv'))
    wt = w.groupby(['Task', 'Proxy', 'Method']).ARI.mean().unstack()
    wbest = wt[['K-Means raw', 'K-Means scaled']].max(axis=1)
    wg = pd.DataFrame({m: wt[m] - wbest for m in ['BTC', 'TreeSHAP', 'RF leaves with SVD', 'RFC spectral']}).reset_index()
    ws = pd.read_csv(os.path.join(RES, 'tcga_within_cohort_summary.csv'))
    for _, row in ws.iterrows():
        q = 'Gain' if row.Question.startswith('Q1') else 'Vs'
        name = {'BTC': 'BTC', 'TreeSHAP': 'Shap', 'RF leaves with SVD': 'RFSVD', 'RFC spectral': 'RFC'}[row.Method]
        num(f'Within{q}{name}', row.Difference); ci(f'Within{q}{name}', row.Lower, row.Upper)
    num('WithinPairs', len(wg)); num('WithinTypes', wg.Task.nunique())
    num('WithinPosBTC', int((wg.BTC > 5e-4).sum()))
    er = wg[wg.Proxy.isin(['er', 'pr'])].set_index('Proxy').BTC
    num('WithinER', er.get('er', np.nan)); num('WithinPR', er.get('pr', np.nan))

    # Subtype tasks: one table with the cohort-versus-rest gain and the mean within-cohort gain.
    within_mean = wg.groupby('Task')[['BTC', 'RF leaves with SVD', 'RFC spectral']].mean()
    lines = [r'\begin{table}[t]', r'\centering', r'\small',
             r'\caption{TCGA, cancer type to molecular subtype. Gain in ARI over the better of raw and standardized $K$-Means on the same cohort, for the cohort-versus-rest proxy and averaged over the within-cohort clinical proxies.}',
             r'\label{tab:tcga_b}', r'\begin{tabular}{lrcccccc}', r'\toprule',
             r'& & \multicolumn{3}{c}{\textbf{Cohort versus rest}} & \multicolumn{3}{c}{\textbf{Within-cohort proxies}} \\',
             r'\cmidrule(lr){3-5} \cmidrule(lr){6-8}',
             r'\textbf{Cancer type} & \textbf{$K$} & \textbf{BTC} & \textbf{RF SVD} & \textbf{RFC} & \textbf{BTC} & \textbf{RF SVD} & \textbf{RFC} \\', r'\midrule']
    kk = h[(h.Level == 'B')].groupby('Task').K.first()
    for task in b.index:
        wm = within_mean.loc[task] if task in within_mean.index else pd.Series({'BTC': np.nan, 'RF leaves with SVD': np.nan, 'RFC spectral': np.nan})
        lines.append(f'{task.replace("_", "/")} & {int(kk.loc[task])} & {f3(gains_b.loc[task, "BTC"])} & {f3(gains_b.loc[task, "RF leaves with SVD"])} & {f3(gains_b.loc[task, "RFC spectral"])} & {f3(wm["BTC"])} & {f3(wm["RF leaves with SVD"])} & {f3(wm["RFC spectral"])} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    write('tcga_subtypes.tex', lines)

    by_proxy = wg.groupby('Proxy')[['BTC', 'TreeSHAP', 'RF leaves with SVD', 'RFC spectral']].mean()
    counts = wg.Proxy.value_counts()
    names = {'age': 'Age at or above median', 'er': 'Estrogen receptor (BRCA)', 'grade': 'Histological grade', 'histology': 'Most common histology',
             'pr': 'Progesterone receptor (BRCA)', 'progression': 'Progression event', 'sex': 'Sex', 'stage': 'Pathologic stage I--II vs.\\ III--IV',
             'survival': 'Death during follow-up'}
    lines = [r'\begin{table}[ht]', r'\centering', r'\small',
             r'\caption{TCGA within-cohort proxies: mean gain over the better $K$-Means baseline by proxy type.}', r'\label{tab:within_proxy}',
             r'\begin{tabular}{lccccc}', r'\toprule',
             r'\textbf{Proxy} & \textbf{Cancer types} & \textbf{BTC} & \textbf{TreeSHAP} & \textbf{RF SVD} & \textbf{RFC} \\', r'\midrule']
    for p, row in by_proxy.sort_values('BTC', ascending=False).iterrows():
        lines.append(f'{names[p]} & {counts[p]} & ' + ' & '.join(f3(row[c]) for c in ['BTC', 'TreeSHAP', 'RF leaves with SVD', 'RFC spectral']) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    write('within_proxy.tex', lines)

    # Figure: per-task gains for BTC and spectral RFC, both TCGA designs.
    order = gains_b.sort_values('BTC').index
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 4.1), sharey=True)
    for ax, (title, frame) in zip(axes, [('Cohort versus rest', gains_b), ('Within-cohort proxies (mean)', within_mean)]):
        ys = np.arange(len(order))
        ax.axvline(0, color=INK2, linewidth=0.8)
        for col, color, label in [('BTC', BLUE, 'BTC'), ('RFC spectral', ORANGE, 'Spectral RFC')]:
            vals = frame.reindex(order)[col].values
            ax.plot(vals, ys, 'o', color=color, markersize=4.5, markeredgecolor='white', markeredgewidth=0.6, label=label)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel('ARI gain over best $K$-Means', color=INK2, fontsize=8)
        style(ax)
    axes[0].set_yticks(np.arange(len(order)))
    axes[0].set_yticklabels([o.replace('_', '/') for o in order])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc='upper center', ncol=2, bbox_to_anchor=(0.55, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(FIGURES, 'tcga_gains.pdf'))
    plt.close(fig)


def main():
    os.makedirs(TABLES, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    development(); brca_receptors(); target_direction(); sensitivity(); shap_setups(); homogeneity(); proxy_contrast(); confirmation(); tcga()
    with open(os.path.join(PAPER, 'numbers.tex'), 'w', encoding='utf-8') as f:
        f.write('% Generated by experiments/make_paper_assets.py. Do not edit.\n')
        for k, v in sorted(macros.items()):
            assert k.isalpha(), k
            f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
    print(f'{len(macros)} numbers written')


if __name__ == '__main__':
    main()
