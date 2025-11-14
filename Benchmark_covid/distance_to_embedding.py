#!/usr/bin/env python3
# mds_from_distance_simple.py
# Loops over multiple sample sizes and runs classical MDS on each distance matrix.

import numpy as np
import pandas as pd
from pathlib import Path

# ====== USER SETTINGS ======
BASE_DIR = "/dcs07/hongkai/data/harry/result/QOT"
SAMPLE_SIZES = [25, 50, 100, 200, 400]          # sample sizes to loop through
N_DIMS = 10                                      # e.g. 2, 3, 10, 50 ...
# ===========================

def read_distance_csv(p):
    p = Path(p)
    try:
        df = pd.read_csv(p, index_col=0)
        D = df.values
        labels = df.index.astype(str).tolist()
        
        # If 'sample' column is not found, create it from default labels
        if 'sample' not in df.columns:
            print("[INFO] 'sample' column not found. Using default labels for samples.")
            df.insert(0, 'sample', labels)
            labels = df['sample'].tolist()  # Set the sample column as the labels
        
        if D.shape[0] != D.shape[1]:
            raise ValueError("Matrix not square; retrying unlabeled read.")
    except Exception:
        df = pd.read_csv(p, header=None)
        D = df.values
        labels = [f"item_{i}" for i in range(D.shape[0])]
        print("[INFO] No headers found; default labels used.")
    
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
    for n in SAMPLE_SIZES:
        input_csv = Path(BASE_DIR) / f"{n}_sample" / f"{n}_qot_distance_matrix.csv"
        print(f"\n[INFO] Processing {n} samples")
        print(f"[INFO] Input CSV: {input_csv}")

        D, labels = read_distance_csv(input_csv)
        X = classical_mds(D, k=N_DIMS)
        save_embedding(X, labels, input_csv, N_DIMS)

if __name__ == "__main__":
    main()
