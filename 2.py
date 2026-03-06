#!/usr/bin/env python3
"""
Complete inspection / QC summary for 10x Genomics HDF5 count matrix files.

Designed for files like:
- filtered_feature_bc_matrix.h5
- raw_feature_bc_matrix.h5

Checks:
1. HDF5 structure
2. Matrix dimensions and sparse structure sanity
3. Feature-type composition
4. Per-barcode count / feature statistics
5. Gene Expression only statistics
6. Optional mitochondrial statistics
7. Survival under common QC thresholds

This does NOT densify the matrix, so it is safe for large files.
"""

import os
import argparse
from collections import Counter

import h5py
import numpy as np
from scipy.sparse import csc_matrix


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def decode_array(arr):
    """Decode bytes array to Python strings."""
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return np.array(out, dtype=object)


def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def summarize_numeric(arr, name):
    arr = np.asarray(arr)
    if arr.size == 0:
        print(f"{name}: empty")
        return

    print(f"{name}:")
    print(f"  min    = {np.min(arr)}")
    print(f"  1%     = {np.percentile(arr, 1):.3f}")
    print(f"  5%     = {np.percentile(arr, 5):.3f}")
    print(f"  25%    = {np.percentile(arr, 25):.3f}")
    print(f"  median = {np.median(arr):.3f}")
    print(f"  75%    = {np.percentile(arr, 75):.3f}")
    print(f"  95%    = {np.percentile(arr, 95):.3f}")
    print(f"  99%    = {np.percentile(arr, 99):.3f}")
    print(f"  max    = {np.max(arr)}")
    print(f"  mean   = {np.mean(arr):.3f}")


def inspect_h5_structure(group, indent=0, max_depth=10):
    """Recursively inspect HDF5 structure."""
    if indent > max_depth:
        return

    for key in group.keys():
        item = group[key]
        prefix = "  " * indent

        if isinstance(item, h5py.Group):
            print(f"{prefix}[GROUP] {key}")
            inspect_h5_structure(item, indent + 1, max_depth=max_depth)
        elif isinstance(item, h5py.Dataset):
            print(f"{prefix}[DATASET] {key} | shape={item.shape} | dtype={item.dtype}")


def compute_barcode_stats(X):
    """
    X is sparse matrix of shape (features, barcodes).
    Returns:
        total_counts_per_barcode
        detected_features_per_barcode
    """
    total_counts = np.asarray(X.sum(axis=0)).ravel()
    detected_features = np.asarray((X > 0).sum(axis=0)).ravel()
    return total_counts, detected_features


def check_thresholds(counts, features, thresholds):
    """Report number of barcodes passing thresholds."""
    print_header("QC THRESHOLD SURVIVAL")
    n = len(counts)
    for min_counts, min_features in thresholds:
        keep = (counts >= min_counts) & (features >= min_features)
        n_keep = int(np.sum(keep))
        pct_keep = 100 * n_keep / n if n > 0 else 0
        print(
            f"min_counts >= {min_counts:>5}, "
            f"min_features >= {min_features:>5}  "
            f"-> keep {n_keep:>8}/{n:<8} ({pct_keep:6.2f}%)"
        )


# =============================================================================
# MAIN INSPECTION
# =============================================================================

def inspect_10x_h5(
    h5_path,
    mito_prefixes=("MT-", "mt-"),
    top_n_features=20,
):
    print_header("FILE")
    print(f"path: {h5_path}")
    print(f"exists: {os.path.exists(h5_path)}")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)

    with h5py.File(h5_path, "r") as f:
        # ---------------------------------------------------------------------
        # 1. HDF5 structure
        # ---------------------------------------------------------------------
        print_header("HDF5 STRUCTURE")
        inspect_h5_structure(f)

        # ---------------------------------------------------------------------
        # 2. Required 10x paths
        # ---------------------------------------------------------------------
        required_paths = [
            "matrix/barcodes",
            "matrix/data",
            "matrix/indices",
            "matrix/indptr",
            "matrix/shape",
            "matrix/features/id",
            "matrix/features/name",
            "matrix/features/feature_type",
        ]

        print_header("REQUIRED PATH CHECK")
        for p in required_paths:
            exists = p in f
            print(f"{p}: {'FOUND' if exists else 'MISSING'}")

        if not all(p in f for p in required_paths):
            raise ValueError("This file does not look like a standard 10x count H5.")

        # ---------------------------------------------------------------------
        # 3. Read core arrays
        # ---------------------------------------------------------------------
        barcodes = decode_array(f["matrix/barcodes"][:])

        data = f["matrix/data"][:]
        indices = f["matrix/indices"][:]
        indptr = f["matrix/indptr"][:]
        shape = tuple(f["matrix/shape"][:])   # usually (features, barcodes)

        feature_ids = decode_array(f["matrix/features/id"][:])
        feature_names = decode_array(f["matrix/features/name"][:])
        feature_types = decode_array(f["matrix/features/feature_type"][:])

        genome = decode_array(f["matrix/features/genome"][:]) if "matrix/features/genome" in f else None

        # ---------------------------------------------------------------------
        # 4. Sparse matrix sanity checks
        # ---------------------------------------------------------------------
        print_header("SPARSE MATRIX SANITY CHECK")

        print(f"matrix shape (features x barcodes): {shape}")
        print(f"n_features from shape: {shape[0]}")
        print(f"n_barcodes from shape: {shape[1]}")
        print(f"len(feature_ids): {len(feature_ids)}")
        print(f"len(feature_names): {len(feature_names)}")
        print(f"len(feature_types): {len(feature_types)}")
        print(f"len(barcodes): {len(barcodes)}")
        print(f"len(data): {len(data)}")
        print(f"len(indices): {len(indices)}")
        print(f"len(indptr): {len(indptr)}")

        expected_indptr_len = shape[1] + 1
        print(f"expected len(indptr) = n_barcodes + 1 = {expected_indptr_len}")
        print(f"indptr length OK: {len(indptr) == expected_indptr_len}")
        print(f"indices length == data length: {len(indices) == len(data)}")
        print(f"feature arrays length == n_features: {len(feature_ids) == shape[0] == len(feature_names) == len(feature_types)}")
        print(f"barcode length == n_barcodes: {len(barcodes) == shape[1]}")

        nnz = len(data)
        total_entries = int(shape[0]) * int(shape[1])
        sparsity = 1 - (nnz / total_entries if total_entries > 0 else 0)
        print(f"nnz (nonzero entries): {nnz}")
        print(f"total matrix entries: {total_entries}")
        print(f"sparsity: {sparsity:.8f}")

        if nnz > 0:
            summarize_numeric(data, "nonzero values (data array)")

        # ---------------------------------------------------------------------
        # 5. Feature type summary
        # ---------------------------------------------------------------------
        print_header("FEATURE TYPE SUMMARY")
        ft_counter = Counter(feature_types)
        for k, v in sorted(ft_counter.items(), key=lambda x: (-x[1], x[0])):
            print(f"{k}: {v}")

        # Optional: genome summary
        if genome is not None:
            print_header("GENOME SUMMARY")
            g_counter = Counter(genome)
            for k, v in sorted(g_counter.items(), key=lambda x: (-x[1], x[0])):
                print(f"{k}: {v}")

        # ---------------------------------------------------------------------
        # 6. Build sparse matrix
        # ---------------------------------------------------------------------
        print_header("BUILD SPARSE MATRIX")
        X = csc_matrix((data, indices, indptr), shape=shape)
        print(f"constructed sparse matrix: shape={X.shape}, nnz={X.nnz}")

        # ---------------------------------------------------------------------
        # 7. All-feature per-barcode stats
        # ---------------------------------------------------------------------
        print_header("ALL-FEATURE BARCODE STATS")
        total_counts_all, detected_features_all = compute_barcode_stats(X)

        summarize_numeric(total_counts_all, "total counts per barcode (all features)")
        summarize_numeric(detected_features_all, "detected features per barcode (all features)")

        zero_count_barcodes = int(np.sum(total_counts_all == 0))
        one_count_barcodes = int(np.sum(total_counts_all == 1))
        low_count_barcodes = int(np.sum(total_counts_all <= 10))

        print(f"barcodes with 0 total counts: {zero_count_barcodes} / {len(barcodes)}")
        print(f"barcodes with 1 total count: {one_count_barcodes} / {len(barcodes)}")
        print(f"barcodes with <=10 total counts: {low_count_barcodes} / {len(barcodes)}")

        # ---------------------------------------------------------------------
        # 8. Top features by total counts
        # ---------------------------------------------------------------------
        print_header(f"TOP {top_n_features} FEATURES BY TOTAL COUNTS (ALL FEATURE TYPES)")
        feature_total_counts = np.asarray(X.sum(axis=1)).ravel()
        top_idx = np.argsort(feature_total_counts)[::-1][:top_n_features]
        for rank, idx in enumerate(top_idx, start=1):
            print(
                f"{rank:>2}. "
                f"name={feature_names[idx]} | "
                f"id={feature_ids[idx]} | "
                f"type={feature_types[idx]} | "
                f"total_counts={feature_total_counts[idx]}"
            )

        # ---------------------------------------------------------------------
        # 9. Gene Expression only
        # ---------------------------------------------------------------------
        gene_mask = feature_types == "Gene Expression"

        print_header("GENE EXPRESSION FEATURE CHECK")
        print(f"n_gene_expression_features: {int(np.sum(gene_mask))}")

        if np.sum(gene_mask) == 0:
            print("No 'Gene Expression' features found.")
            return

        X_gene = X[gene_mask, :]
        gene_names = feature_names[gene_mask]
        gene_ids = feature_ids[gene_mask]

        total_counts_gene, detected_genes = compute_barcode_stats(X_gene)

        print_header("GENE-EXPRESSION-ONLY BARCODE STATS")
        summarize_numeric(total_counts_gene, "total gene-expression counts per barcode")
        summarize_numeric(detected_genes, "detected genes per barcode")

        zero_gene_barcodes = int(np.sum(total_counts_gene == 0))
        one_gene_barcodes = int(np.sum(total_counts_gene == 1))
        low_gene_barcodes = int(np.sum(total_counts_gene <= 10))

        print(f"barcodes with 0 gene-expression counts: {zero_gene_barcodes} / {len(barcodes)}")
        print(f"barcodes with 1 gene-expression count: {one_gene_barcodes} / {len(barcodes)}")
        print(f"barcodes with <=10 gene-expression counts: {low_gene_barcodes} / {len(barcodes)}")

        # ---------------------------------------------------------------------
        # 10. Mitochondrial fraction (if detectable)
        # ---------------------------------------------------------------------
        print_header("MITOCHONDRIAL CHECK")
        mito_mask = np.zeros(len(gene_names), dtype=bool)
        for prefix in mito_prefixes:
            mito_mask |= np.char.startswith(gene_names.astype(str), prefix)

        n_mito = int(np.sum(mito_mask))
        print(f"detected mitochondrial genes using prefixes {mito_prefixes}: {n_mito}")

        if n_mito > 0:
            mito_counts = np.asarray(X_gene[mito_mask, :].sum(axis=0)).ravel()
            pct_mito = np.divide(
                mito_counts,
                total_counts_gene,
                out=np.zeros_like(mito_counts, dtype=float),
                where=total_counts_gene > 0
            ) * 100

            summarize_numeric(mito_counts, "mitochondrial counts per barcode")
            summarize_numeric(pct_mito, "percent mitochondrial counts per barcode")
        else:
            print("No mitochondrial genes detected from name prefixes.")

        # ---------------------------------------------------------------------
        # 11. Common QC threshold survival
        # ---------------------------------------------------------------------
        thresholds = [
            (1, 1),
            (10, 10),
            (50, 50),
            (100, 50),
            (200, 100),
            (500, 200),
            (1000, 200),
        ]
        check_thresholds(total_counts_gene, detected_genes, thresholds)

        # ---------------------------------------------------------------------
        # 12. Top gene-expression features by counts
        # ---------------------------------------------------------------------
        print_header(f"TOP {top_n_features} GENE-EXPRESSION FEATURES BY TOTAL COUNTS")
        gene_total_counts = np.asarray(X_gene.sum(axis=1)).ravel()
        top_gene_idx = np.argsort(gene_total_counts)[::-1][:top_n_features]

        for rank, idx in enumerate(top_gene_idx, start=1):
            print(
                f"{rank:>2}. "
                f"gene={gene_names[idx]} | "
                f"id={gene_ids[idx]} | "
                f"total_counts={gene_total_counts[idx]}"
            )

        # ---------------------------------------------------------------------
        # 13. Final interpretation
        # ---------------------------------------------------------------------
        print_header("INTERPRETATION SUMMARY")

        med_counts = np.median(total_counts_gene)
        med_genes = np.median(detected_genes)
        pct_nonzero_gene = 100 * np.mean(total_counts_gene > 0)

        print(f"median gene-expression counts per barcode: {med_counts:.3f}")
        print(f"median detected genes per barcode: {med_genes:.3f}")
        print(f"percent barcodes with >0 gene-expression counts: {pct_nonzero_gene:.2f}%")

        if med_counts <= 1 and med_genes <= 1:
            print("Interpretation: sample is extremely sparse at the barcode level.")
            print("This is consistent with nearly all cells failing standard RNA QC.")
        elif med_counts < 50 and med_genes < 50:
            print("Interpretation: sample has very weak RNA complexity.")
            print("Aggressive QC thresholds will likely remove most or all barcodes.")
        else:
            print("Interpretation: sample has nontrivial RNA signal.")
            print("If cells are still being removed, inspect the exact QC thresholds and filtering code.")


# =============================================================================
# CLI
# =============================================================================

def main():

    inspect_10x_h5(
        h5_path='/dcs07/antar/data/cellranger/feature_matrices/21--228-M6_sample_filtered_feature_bc_matrix.h5',
        top_n_features=20,
    )


if __name__ == "__main__":
    main()