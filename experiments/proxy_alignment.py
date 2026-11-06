"""Exploratory, added after the alignment study: how aligned are the real TCGA within-cohort proxies?

For each task-proxy pair of run_tcga_within_cohort.py, the adjusted mutual information between the
binary clinical proxy and the molecular subtypes, on the same samples. This places the real proxies
on the alignment scale of run_alignment.py. Not part of any pre-specified plan.
"""
import os
import sys

import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from experiments import run_tcga_within_cohort as within
from experiments.run_tcga_hierarchy import load

OUT = os.path.join(ROOT, 'experiments', 'results', 'within_proxy_alignment.csv')


def main():
    rows = [{'Task': prefix, 'Proxy': proxy, 'AMI': adjusted_mutual_info_score(labels, y)}
            for prefix, proxy, idx, y, labels in within.tasks(load(), within.clinical_proxies())]
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    print(f'{len(d)} pairs, median AMI {d.AMI.median():.3f}')
    print(d.sort_values('AMI', ascending=False).head(5).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
