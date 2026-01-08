#!/usr/bin/env python3
# mds_from_single_distance.py
# User edits INPUT_CSV directly in this file.

import numpy as np
import pandas as pd
from pathlib import Path


# ===== USER EDIT HERE =====
INPUT_CSV = "/dcs07/hongkai/data/harry/result/Benchmark_heart_rna/pilot/wasserstein_distance.csv"
N_DIMS = 10
# ===========================


def read_distance_csv(p: str | Path):
    """
    Read a distance matrix CSV and return (D, labels).
    1) Try labeled with index_col=0.
    2) If not square or fails, read as raw matrix with no labels.
    """
    p = Path(p)
    try:
        df = pd.read_csv(p, index_col=0)
        D = df.values
        labels = df.index.astype(str).tolist()
        if "sample" in df.columns:
            labels = df["sample"].astype(str).tolist()

        if D.shape[0] != D.shape[1]:
            raise ValueError("Matrix not square; trying fallback read.")
    except Exception:
        df = pd.read_csv(p, header=None)
        D = df.values
        labels = [f"item_{i}" for i in range(D.shape[0])]
        print("[INFO] No labels found; using default item labels.")

    if D.shape[0] != D.shape[1]:
        raise ValueError(f"CSV must be square; got {D.shape}.")

    return D.astype(float), labels


def classical_mds(D: np.ndarray, k: int = 2) -> np.ndarray:
    """Run classical MDS and return an n x k embedding."""
    D = 0.5*(D + D.T)
    np.fill_diagonal(D, 0.0)
    D[D < 0] = 0.0

    n = D.shape[0]
    J = np.eye(n) - np.ones((n,n))/n
    B = -0.5 * J @ (D**2) @ J

    vals, vecs = np.linalg.eigh(B)
    idx = vals.argsort()[::-1]
    vals, vecs = vals[idx], vecs[:,idx]

    pos = vals > 1e-12
    if not pos.any():
        raise RuntimeError("No positive eigenvalues; MDS failed.")
    vals = vals[pos][:k]
    vecs = vecs[:, pos][:, :k]
    return vecs * np.sqrt(vals)


def save_embedding(X, labels, input_path, k):
    """Save embedding next to input file."""
    in_path = Path(input_path)
    out_csv = in_path.with_name(f"{in_path.stem}_mds_{k}d.csv")
    out_npy = in_path.with_name(f"{in_path.stem}_mds_{k}d.npy")

    pd.DataFrame(
        X, index=labels, columns=[f"dim_{i+1}" for i in range(X.shape[1])]
    ).to_csv(out_csv)

    np.save(out_npy, X)
    print(f"[Saved] {out_csv}")
    print(f"[Saved] {out_npy}")


def main():
    print("="*60)
    print(f"[INFO] Input  : {INPUT_CSV}")
    print(f"[INFO] Dim    : {N_DIMS}")

    p = Path(INPUT_CSV)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")

    D, labels = read_distance_csv(p)
    print(f"[INFO] Distance shape: {D.shape}")

    X = classical_mds(D, k=N_DIMS)
    print(f"[INFO] Embedding shape: {X.shape}")

    save_embedding(X, labels, p, N_DIMS)


if __name__ == "__main__":
    main()
