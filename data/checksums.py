"""Verify downloaded data files against the SHA-256 checksums of the files used for the paper.

The checksums in checksums.sha256 were computed from the files behind the reported results. A
mismatch means the download is incomplete or the server now provides a different version, in which
case results may differ from the paper. Delete the file and download it again; if the mismatch
persists, the source has changed.
"""
import hashlib
import os
import sys

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKSUMS = os.path.join(DATA_DIR, 'checksums.sha256')


def expected():
    with open(CHECKSUMS, encoding='utf-8') as f:
        return {path: digest for digest, path in (line.split() for line in f if line.strip())}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(paths):
    """Check files given relative to the data directory; exit with an explanation on a mismatch."""
    known = expected()
    for path in paths:
        actual = sha256(os.path.join(DATA_DIR, path))
        if actual != known[path]:
            sys.exit(f'Checksum mismatch for data/{path}: expected {known[path]}, got {actual}. '
                     'The download is incomplete or the source file has changed; delete the file and download it again.')
        print(f'data/{path}: checksum verified')
