#!/usr/bin/env python3
# mds_from_single_distance.py
# Run classical MDS on a single distance matrix CSV and save the embedding.

import numpy as np
import pandas as pd
from pathlib import Path
import argparse


# ====== DEFAULT USER SETTINGS ======
# Default distance matrix to process (can override with --input)
DEFAULT_INPUT_CSV = (
    "/dcs07/hongkai/data/harry/result/Benchmark_multiomics/Gloscope/"
    "knn_divergence.csv"
)

DEFAULT_N_DIMS = 10
# ===================================


def read_distance_csv(p: str | Path):
    """
    Read a distance matrix CSV and return (D, labels).

    Strategy:
      1. Try reading with index as labels (index_col=0).
      2. If that fails or matrix is not square, fall back to header=None read.
    """
    p = Path(p)

    # First attempt: assume first column is an index of sample IDs
    try:
        df = pd.read_csv(p, index_col=0)
        D = df.values
        labels = df.index.astype(str).tolist()

        # If there is a 'sample' column, prefer that as labels
        if "sample" in df.columns:
            labels = df["sample"].astype(str).tolist()

        if D.shape[0] != D.shape[1]:
            raise ValueError("Matrix not square; retrying unlabeled read.")
    except Exception:
        # Fallback: no useful headers, treat as raw numeric matrix
        df = pd.read_csv(p, header=None)
        D = df.values
        labels = [f"item_{i}" for i in range(D.shape[0])]
        print("[INFO] No usable headers found; default labels used.")

    if D.shape[0] != D.shape[1]:
        raise ValueError(f"Distance CSV must be square (n x n), got {D.shape}.")

    return D.astype(float), labels


def classical_mds(D: np.ndarray, k: int = 2) -> np.ndarray:
    """
    Classical MDS on a distance matrix D, returning an n x k embedding.

    Steps:
      1. Symmetrize, zero diagonal, clip negatives.
      2. Double-center the squared distances.
      3. Eigen-decompose and take top k positive components.
    """
    # Force symmetry and clean up numerical junk
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

    return vecs * np.sqrt(vals)


def save_embedding(X: np.ndarray, labels, input_path: str | Path, k: int):
    """
    Save MDS embedding next to the input file.

    Example:
      knn_divergence.csv ->
      knn_divergence_mds_10d.csv
      knn_divergence_mds_10d.npy
    """
    in_path = Path(input_path)
    out_csv = in_path.with_name(f"{in_path.stem}_mds_{k}d.csv")
    out_npy = in_path.with_name(f"{in_path.stem}_mds_{k}d.npy")

    df = pd.DataFrame(
        X,
        index=labels,
        columns=[f"dim_{i + 1}" for i in range(X.shape[1])],
    )
    df.to_csv(out_csv)
    np.save(out_npy, X)

    print(f"[Saved] {out_csv}")
    print(f"[Saved] {out_npy}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run classical MDS on a single distance matrix CSV."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_CSV,
        help=f"Path to distance matrix CSV (default: {DEFAULT_INPUT_CSV})",
    )
    parser.add_argument(
        "--dims",
        type=int,
        default=DEFAULT_N_DIMS,
        help=f"Number of embedding dimensions (default: {DEFAULT_N_DIMS})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_csv = Path(args.input)
    k = args.dims

    print("=" * 60)
    print(f"[INFO] Input CSV : {input_csv}")
    print(f"[INFO] n_dims    : {k}")

    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    D, labels = read_distance_csv(input_csv)
    print(f"[INFO] Distance matrix shape: {D.shape}")

    X = classical_mds(D, k=k)
    print(f"[INFO] MDS embedding shape  : {X.shape}")

    save_embedding(X, labels, input_csv, k)


if __name__ == "__main__":
    main()
