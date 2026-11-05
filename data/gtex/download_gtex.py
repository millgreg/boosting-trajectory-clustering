"""Download the GTEx files used by the GTEx study (experiments/run_gtex.py).

Expression (RSEM gene TPM, about 546 MB) and tissue labels come from the public UCSC Xena Toil hub,
donor and sample attributes from the public GTEx v8 annotation files. The study builds a
2,000-gene cache from these files on first use.
"""
import gzip
import os
import urllib.request

XENA = 'https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/'
GTEX = 'https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/'
FILES = {
    'gtex_RSEM_gene_tpm.gz': XENA + 'gtex_RSEM_gene_tpm.gz',
    'GTEX_phenotype.gz': XENA + 'GTEX_phenotype.gz',
    'GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt': GTEX + 'GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt',
    'GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt': GTEX + 'GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt',
}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    for local, url in FILES.items():
        path = os.path.join(here, local)
        if os.path.exists(path):
            print(f'{local} already present')
            continue
        print(f'Downloading {local}...')
        urllib.request.urlretrieve(url, path)
    with gzip.open(os.path.join(here, 'gtex_RSEM_gene_tpm.gz')) as f:
        while f.read(1 << 24):
            pass
    print('Expression archive verified.')


if __name__ == '__main__':
    main()
