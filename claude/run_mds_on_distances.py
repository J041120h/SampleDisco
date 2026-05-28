"""
Convert distance matrices (PILOT/QOT/GloScope) to 10-d sample embeddings
via classical MDS. Same approach as Benchmark_covid/distance_to_embedding.py.

Saves each embedding next to its source distance file:
    <stem>_mds_10d.csv  (labeled)
    <stem>_mds_10d.npy
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

N_DIMS = 10

# Targets — extend as more distance matrices arrive
INPUTS = [
    "/dcs07/hongkai/data/harry/result/health_aging_PBMC/other_methods/pilot/wasserstein_distance.csv",
    "/dcs07/hongkai/data/harry/result/health_aging_PBMC/other_methods/QOT/316_qot_distance_matrix.csv",
    # GloScope distance matrix path — add when run completes
    # "/dcs07/hongkai/data/harry/result/health_aging_PBMC/other_methods/GloScope/<file>.csv",
]


def read_distance_csv(p: Path):
    """Load square distance matrix with row/column labels."""
    df = pd.read_csv(p, index_col=0)
    D = df.values.astype(float)
    labels = df.index.astype(str).tolist()
    if D.shape[0] != D.shape[1]:
        raise ValueError(f"{p.name}: not square, got {D.shape}")
    return D, labels


def classical_mds(D: np.ndarray, k: int) -> np.ndarray:
    """Classical MDS via double centering + eigendecomposition."""
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    D[D < 0] = 0.0

    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J

    vals, vecs = np.linalg.eigh(B)
    idx = vals.argsort()[::-1]
    vals, vecs = vals[idx], vecs[:, idx]

    pos = vals > 1e-12
    if not pos.any():
        raise RuntimeError("No positive eigenvalues; MDS failed.")
    vals = vals[pos][:k]
    vecs = vecs[:, pos][:, :k]
    if len(vals) < k:
        print(f"[WARN] only {len(vals)} positive eigenvalues, padding with zeros")
        pad = np.zeros((vecs.shape[0], k - len(vals)))
        return np.hstack([vecs * np.sqrt(vals), pad])
    return vecs * np.sqrt(vals)


def process(path_str: str, k: int):
    p = Path(path_str)
    if not p.exists():
        print(f"[SKIP] not found: {p}")
        return
    print(f"\n=== {p.relative_to('/dcs07/hongkai/data/harry/result/health_aging_PBMC/other_methods')} ===")
    D, labels = read_distance_csv(p)
    print(f"  D shape: {D.shape}; symmetric err: {np.abs(D - D.T).max():.2e}")

    X = classical_mds(D, k=k)
    print(f"  Embedding shape: {X.shape}; var explained per dim:")
    var = X.var(axis=0)
    print("   ", np.round(var / var.sum() * 100, 2).tolist(), "(% of MDS variance)")

    out_csv = p.with_name(f"{p.stem}_mds_{k}d.csv")
    out_npy = p.with_name(f"{p.stem}_mds_{k}d.npy")
    pd.DataFrame(
        X, index=labels, columns=[f"dim_{i+1}" for i in range(X.shape[1])]
    ).to_csv(out_csv, index_label="sample")
    np.save(out_npy, X)
    print(f"  -> {out_csv.name}")
    print(f"  -> {out_npy.name}")


def main():
    for p in INPUTS:
        process(p, N_DIMS)
    print("\nDONE.")


if __name__ == "__main__":
    main()
