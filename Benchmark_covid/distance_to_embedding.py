#!/usr/bin/env python3
# mds_from_distance_simple.py
# Loops over multiple sample sizes and runs classical MDS on each distance matrix.

import numpy as np
import pandas as pd
from pathlib import Path

# ================= USER SETTINGS =================
# Which method are we processing?
#   "QOT"      -> /QOT/{n}_sample/{n}_qot_distance_matrix.csv
#   "Gloscope" -> /Gloscope/{n}_sample/knn_divergence.csv
#   "pilot"    -> /pilot/{n}_sample/wasserstein_distance.csv
METHOD = "pilot"

SAMPLE_SIZES = [25, 50, 100, 200, 279, 400]
N_DIMS = 10

METHOD_CONFIG = {
    "QOT": {
        "BASE_DIR": "/dcs07/hongkai/data/harry/result/QOT",
        "FILE_PATTERN": "{n}_sample/{n}_qot_distance_matrix.csv",
    },
    "Gloscope": {
        "BASE_DIR": "/dcs07/hongkai/data/harry/result/Gloscope",
        "FILE_PATTERN": "{n}_sample/knn_divergence.csv",
    },
    "pilot": {
        "BASE_DIR": "/dcs07/hongkai/data/harry/result/pilot",
        "FILE_PATTERN": "{n}_sample/wasserstein_distance.csv",
    },
}
# ==================================================


def read_distance_csv(p: str | Path):
    """
    Read a distance matrix CSV and return (D, labels).

    First tries to read with index labels; if that fails, falls back
    to a raw numeric matrix with default labels.
    """
    p = Path(p)
    try:
        df = pd.read_csv(p, index_col=0)
        D = df.values
        labels = df.index.astype(str).tolist()

        if "sample" not in df.columns:
            print("[INFO] 'sample' column not found. Using index as sample labels.")
            df.insert(0, "sample", labels)
            labels = df["sample"].astype(str).tolist()

        if D.shape[0] != D.shape[1]:
            raise ValueError("Matrix not square; retrying unlabeled read.")
    except Exception:
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
    """
    D = 0.5 * (D + D.T)  # symmetrize
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

    e.g. wasserstein_distance.csv -> wasserstein_distance_mds_10d.csv/.npy
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


def main():
    if METHOD not in METHOD_CONFIG:
        raise ValueError(f"Unknown METHOD='{METHOD}'. Valid: {list(METHOD_CONFIG)}")

    base_dir = Path(METHOD_CONFIG[METHOD]["BASE_DIR"])
    file_pattern = METHOD_CONFIG[METHOD]["FILE_PATTERN"]

    print(f"[INFO] METHOD     : {METHOD}")
    print(f"[INFO] BASE_DIR   : {base_dir}")
    print(f"[INFO] PATTERN    : {file_pattern}")
    print(f"[INFO] SAMPLE_SIZES: {SAMPLE_SIZES}")

    for n in SAMPLE_SIZES:
        input_csv = base_dir / file_pattern.format(n=n)

        print("\n" + "=" * 60)
        print(f"[INFO] Processing {n} samples")
        print(f"[INFO] Input CSV: {input_csv}")

        if not input_csv.exists():
            print(f"[ERROR] File not found: {input_csv}")
            continue

        try:
            D, labels = read_distance_csv(input_csv)
            print(f"[INFO] Distance matrix shape: {D.shape}")

            X = classical_mds(D, k=N_DIMS)
            print(f"[INFO] MDS embedding shape: {X.shape}")

            save_embedding(X, labels, input_csv, N_DIMS)

        except Exception as e:
            print(f"[ERROR] Failed on n={n} with file {input_csv}")
            print(f"        {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
