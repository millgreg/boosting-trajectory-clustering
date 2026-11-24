"""Download the TCGA PanCanAtlas files used by the TCGA hierarchy and within-cohort studies.

Files come from the public UCSC Xena PanCanAtlas hub. The expression matrix is about 331 MB.
The studies build a 2,000-gene cache from these files on first use.
"""
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from checksums import verify

BASE = 'https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/'
FILES = {
    'EB_AdjustPANCAN_geneExp.xena.gz': 'EB%2B%2BAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz',
    'TCGASubtype.20170308.tsv.gz': 'TCGASubtype.20170308.tsv.gz',
    'Survival_SupplementalTable_S1_20171025_xena_sp': 'Survival_SupplementalTable_S1_20171025_xena_sp',
}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    for local, remote in FILES.items():
        path = os.path.join(here, local)
        if os.path.exists(path):
            print(f'{local} already present')
            continue
        print(f'Downloading {local}...')
        urllib.request.urlretrieve(BASE + remote, path)
    verify(['tcga_pancan/' + local for local in FILES])


if __name__ == '__main__':
    main()
