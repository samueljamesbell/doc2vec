import argparse
import sys

import h5py
import pandas as pd


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path to doc embeddings HDF5 file')
    return parser.parse_args()


def _load_embeddings(path):
    with h5py.File(path, 'r') as f:
        return f.get('doc_embeddings').value


def main():
    args = _parse_args()

    embeddings = _load_embeddings(args.path)
    pd.DataFrame(embeddings).to_csv(sys.stdout, sep='\t', header=False, index=False)
