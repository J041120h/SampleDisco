#!/usr/bin/env python3
# mds_from_distance_simple.py
# Edit INPUT_CSV (and N_DIMS) below, then run: python mds_from_distance_simple.py

import numpy as np
import pandas as pd
from pathlib import Path

# ====== USER SETTINGS ======
INPUT_CSV = "/users/hjiang/r/GloScope/25_sample/knn_divergence.csv"
N_DIMS = 10                                      # e.g. 2, 3, 10, 50 ...
# ===========================

def read_distance_csv(p):
    p = Path(p)
    try:
        df = pd.read_csv(p, index_col=0)
        D = df.values
        labels = df.index.astype(str).tolist()
        if D.shape[0] != D.shape[1]:
            raise ValueError("Matrix not square; retrying unlabeled read.")
    except Exception:
        df = pd.read_csv(p, header=None)
        D = df.values
        labels = [f"item_{i}" for i in range(D.shape[0])]
    if D.shape[0] != D.shape[1]:
        raise ValueError("Distance CSV must be square (n x n).")
    return D.astype(float), labels

def classical_mds(D, k=2):
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    D[D < 0] = 0.0

    n = D.shape[0]
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J

    vals, vecs = np.linalg.eigh(B)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    pos = vals > 1e-12
    if not np.any(pos):
        raise RuntimeError("No positive eigenvalues; MDS failed.")
    vals = vals[pos][:k]
    vecs = vecs[:, pos][:, :k]

    X = vecs * np.sqrt(vals)
    return X

def save_embedding(X, labels, input_path, k):
    in_path = Path(input_path)
    out_csv = in_path.with_name(f"{in_path.stem}_mds_{k}d.csv")
    out_npy = in_path.with_name(f"{in_path.stem}_mds_{k}d.npy")

    df = pd.DataFrame(X, index=labels, columns=[f"dim_{i+1}" for i in range(X.shape[1])])
    df.to_csv(out_csv)
    np.save(out_npy, X)
    print(f"[Saved] {out_csv}")
    print(f"[Saved] {out_npy}")

def main():
    D, labels = read_distance_csv(INPUT_CSV)
    X = classical_mds(D, k=N_DIMS)
    save_embedding(X, labels, INPUT_CSV, N_DIMS)

if __name__ == "__main__":
    main()
