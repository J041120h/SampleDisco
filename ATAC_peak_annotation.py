#!/usr/bin/env python3
"""
Parallel ATAC-seq peak-to-gene annotation

*Only change from the previous version*:  
➜ Added the **output_dir** parameter (default: `"."`) so every result file
  is written inside that directory.  
  No other logic or behaviour was altered.
"""

# ────────────────────────────── imports ──────────────────────────────
import os                                # NEW
import json
import pickle
import warnings
import multiprocessing as mp
from functools import partial
from collections import defaultdict

import numpy as np
import pandas as pd
import anndata as ad
import pyensembl
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────── helper functions ─────────────────────────
def parse_peak(peak_str):
    """Parse a 'chr-start-end' style peak string into tuple(chrom, start, end)."""
    parts = peak_str.split("-")
    if len(parts) != 3:
        return None
    chrom = parts[0].replace("chr", "") if parts[0].startswith("chr") else parts[0]
    try:
        return chrom, int(parts[1]), int(parts[2])
    except Exception:
        return None


def process_chromosome_batch(args):
    """Worker that scores peak↔gene overlaps for one chromosome."""
    chrom, peaks_chr, genes_chr, params = args

    sigma = params["sigma"]
    promoter_weight_factor = params["promoter_weight_factor"]
    promoter_upstream = params["promoter_upstream"]
    promoter_downstream = params["promoter_downstream"]

    overlaps = []

    for _, peak in peaks_chr.iterrows():
        # genes whose windows overlap this peak
        overlapping_genes = genes_chr[
            (genes_chr["window_start"] <= peak["end"])
            & (genes_chr["window_end"] >= peak["start"])
        ]

        if len(overlapping_genes) == 0:
            continue

        for _, gene in overlapping_genes.iterrows():
            # distance metrics ------------------------------------------------
            dist_to_tss = abs(gene["tss"] - peak["center"])

            if gene["gene_start"] <= peak["center"] <= gene["gene_end"]:
                dist_to_gene = 0
                in_gene_body = True
            else:
                dist_to_gene = min(
                    abs(peak["center"] - gene["gene_start"]),
                    abs(peak["center"] - gene["gene_end"]),
                )
                in_gene_body = False

            in_promoter = gene["promoter_start"] <= peak["center"] <= gene["promoter_end"]

            # strand-aware relative position
            relative_pos = (
                peak["center"] - gene["tss"] if gene["strand"] == "+" else gene["tss"] - peak["center"]
            )

            # weighting -------------------------------------------------------
            tss_weight = np.exp(-0.5 * (dist_to_tss / sigma) ** 2)
            gene_body_weight = 0.5 if in_gene_body else 0
            promoter_weight = promoter_weight_factor if in_promoter else 0
            directional_weight = 1.0 if relative_pos < 0 else 0.8

            combined_weight = (tss_weight + gene_body_weight + promoter_weight) * directional_weight

            overlaps.append(
                {
                    "peak": peak["peak"],
                    "peak_idx": peak["peak_idx"],
                    "gene_name": gene["gene_name"],
                    "gene_id": gene["gene_id"],
                    "distance_to_tss": dist_to_tss,
                    "distance_to_gene": dist_to_gene,
                    "in_promoter": in_promoter,
                    "in_gene_body": in_gene_body,
                    "relative_position": relative_pos,
                    "tss_weight": tss_weight,
                    "gene_body_weight": gene_body_weight,
                    "promoter_weight": promoter_weight,
                    "directional_weight": directional_weight,
                    "combined_weight": combined_weight,
                    "peak_accessibility": peak["mean_accessibility"],
                }
            )

    return overlaps


# ─────────────────────── main annotation function ──────────────────────
def annotate_atac_peaks_parallel(
    atac_file_path,
    *,
    ensembl_release=110,
    extend_upstream=100_000,
    extend_downstream=100_000,
    use_gene_bounds=True,
    promoter_upstream=2_000,
    promoter_downstream=2_000,
    distance_weight_sigma=50_000,
    promoter_weight_factor=5.0,
    min_peak_accessibility=0.01,
    n_threads=None,
    output_prefix="atac_annotation",
    output_dir=".",          # NEW
):
    """
    Annotate ATAC peaks to nearby genes in parallel.

    The *output_dir* argument specifies where all result files are written.
    """

    # ── I/O set-up ──────────────────────────────────────────────────────
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if n_threads is None:
        n_threads = mp.cpu_count()

    print(f"Starting parallel ATAC peak annotation with {n_threads} threads…")
    print(f"• Output directory : {output_dir}")

    # ── load ATAC matrix ───────────────────────────────────────────────
    print("Loading ATAC data…")
    adata = ad.read_h5ad(atac_file_path)

    # mean & sum accessibility
    print("Calculating peak statistics…")
    if hasattr(adata.X, "toarray"):  # sparse
        peak_means = np.asarray(adata.X.mean(axis=0)).ravel()
        peak_sums = np.asarray(adata.X.sum(axis=0)).ravel()
    else:  # dense
        peak_means = np.asarray(adata.X.mean(axis=0)).ravel()
        peak_sums = np.asarray(adata.X.sum(axis=0)).ravel()

    valid_peaks = peak_means >= min_peak_accessibility
    print(f"Filtering peaks: {valid_peaks.sum()}/{len(valid_peaks)} pass accessibility threshold")

    # ── gene annotations ───────────────────────────────────────────────
    print("Loading gene annotations…")
    ensembl = pyensembl.EnsemblRelease(release=ensembl_release, species="homo_sapiens")
    try:
        genes = ensembl.genes()
    except Exception:
        print("Downloading Ensembl data (first time only)…")
        ensembl.download()
        ensembl.index()
        genes = ensembl.genes()

    protein_coding_genes = [g for g in genes if g.biotype == "protein_coding"]
    print(f"Processing {len(protein_coding_genes)} protein-coding genes…")

    gene_windows = []
    for gene in tqdm(protein_coding_genes, desc="Processing genes"):
        try:
            tss = gene.start if gene.strand == "+" else gene.end

            # define window ---------------------------------------------------
            if use_gene_bounds:
                # gene body ± extensions
                if gene.strand == "+":
                    window_start = max(0, gene.start - extend_upstream)
                    window_end = gene.end + extend_downstream
                else:
                    window_start = max(0, gene.start - extend_downstream)
                    window_end = gene.end + extend_upstream
            else:
                # TSS-centred window
                window_start = max(0, tss - extend_upstream)
                window_end = tss + extend_downstream

            # promoter
            if gene.strand == "+":
                promoter_start = max(0, tss - promoter_upstream)
                promoter_end = tss + promoter_downstream
            else:
                promoter_start = max(0, tss - promoter_downstream)
                promoter_end = tss + promoter_upstream

            gene_windows.append(
                {
                    "gene_name": gene.gene_name,
                    "gene_id": gene.gene_id,
                    "chromosome": gene.contig.replace("chr", ""),
                    "tss": tss,
                    "gene_start": gene.start,
                    "gene_end": gene.end,
                    "strand": gene.strand,
                    "window_start": window_start,
                    "window_end": window_end,
                    "promoter_start": promoter_start,
                    "promoter_end": promoter_end,
                }
            )
        except Exception:
            continue

    genes_df = pd.DataFrame(gene_windows)

    # ── parse peaks ────────────────────────────────────────────────────
    print("Parsing peak coordinates…")
    peaks_data = []
    for i, peak_name in enumerate(tqdm(adata.var_names, desc="Parsing peaks")):
        if not valid_peaks[i]:
            continue
        parsed = parse_peak(peak_name)
        if parsed is None:
            continue
        chrom, start, end = parsed
        peaks_data.append(
            {
                "peak": peak_name,
                "peak_idx": i,
                "chromosome": chrom,
                "start": start,
                "end": end,
                "center": (start + end) // 2,
                "width": end - start,
                "accessibility": peak_sums[i],
                "mean_accessibility": peak_means[i],
            }
        )

    peaks_df = pd.DataFrame(peaks_data)
    print(f"Parsed {len(peaks_df)} valid peaks")

    # ── batching by chromosome ─────────────────────────────────────────
    peaks_by_chr = peaks_df.groupby("chromosome")
    genes_by_chr = genes_df.groupby("chromosome")

    chromosome_batches = []
    for chrom in peaks_df["chromosome"].unique():
        peaks_chr = peaks_by_chr.get_group(chrom)

        genes_chr = None
        for variant in [chrom, chrom.replace("chr", ""), f"chr{chrom}"]:
            if variant in genes_df["chromosome"].values:
                genes_chr = genes_by_chr.get_group(variant)
                break
        if genes_chr is None:
            continue

        params = {
            "sigma": distance_weight_sigma,
            "promoter_weight_factor": promoter_weight_factor,
            "promoter_upstream": promoter_upstream,
            "promoter_downstream": promoter_downstream,
        }
        chromosome_batches.append((chrom, peaks_chr, genes_chr, params))

    # ── parallel processing ────────────────────────────────────────────
    print(f"Processing {len(chromosome_batches)} chromosomes in parallel…")
    with mp.Pool(n_threads) as pool:
        results = list(
            tqdm(
                pool.imap(process_chromosome_batch, chromosome_batches),
                total=len(chromosome_batches),
                desc="Processing chromosomes",
            )
        )

    # ── combine & normalise weights ────────────────────────────────────
    print("Combining results…")
    annotation_df = pd.DataFrame([x for batch in results for x in batch])
    print(f"Found {len(annotation_df)} peak-gene associations")

    print("Normalising weights per peak…")
    for peak in tqdm(annotation_df["peak"].unique(), desc="Normalising"):
        mask = annotation_df["peak"] == peak
        weights = annotation_df.loc[mask, "combined_weight"].values
        total = weights.sum()
        if total > 0:
            annotation_df.loc[mask, "combined_weight"] = weights / total

    # ── build peak-level summary dict ──────────────────────────────────
    print("Creating annotation dictionary…")
    peak_annotation = {}
    for peak_name, grp in tqdm(annotation_df.groupby("peak"), desc="Building annotations"):
        sorted_grp = grp.sort_values("combined_weight", ascending=False)
        peak_annotation[peak_name] = {
            "genes": sorted_grp["gene_name"].tolist(),
            "gene_ids": sorted_grp["gene_id"].tolist(),
            "distances": sorted_grp["distance_to_tss"].tolist(),
            "weights": sorted_grp["combined_weight"].tolist(),
            "tss_weights": sorted_grp["tss_weight"].tolist(),
            "in_promoter": sorted_grp["in_promoter"].tolist(),
            "in_gene_body": sorted_grp["in_gene_body"].tolist(),
            "best_gene": sorted_grp.iloc[0]["gene_name"],
            "best_weight": float(sorted_grp.iloc[0]["combined_weight"]),
        }

    # ── summary stats ──────────────────────────────────────────────────
    stats = {
        "total_peaks": len(peaks_df),
        "annotated_peaks": len(peak_annotation),
        "coverage_percent": 100 * len(peak_annotation) / len(peaks_df),
        "mean_genes_per_peak": annotation_df.groupby("peak").size().mean(),
        "mean_peaks_per_gene": annotation_df.groupby("gene_name").size().mean(),
        "total_associations": len(annotation_df),
        "n_unique_genes": annotation_df["gene_name"].nunique(),
    }

    parameters = {
        "ensembl_release": ensembl_release,
        "extend_upstream": extend_upstream,
        "extend_downstream": extend_downstream,
        "use_gene_bounds": use_gene_bounds,
        "promoter_upstream": promoter_upstream,
        "promoter_downstream": promoter_downstream,
        "distance_weight_sigma": distance_weight_sigma,
        "promoter_weight_factor": promoter_weight_factor,
        "min_peak_accessibility": min_peak_accessibility,
    }

    # ── write outputs ──────────────────────────────────────────────────
    print("\nSaving results…")
    output_files = {}

    def _path(fname):
        return os.path.join(output_dir, fname)

    # 1) full annotation table
    annotation_file = _path(f"{output_prefix}_full_annotations.parquet")
    annotation_df.to_parquet(annotation_file, index=False)
    output_files["annotation_df"] = annotation_file
    print(f"  • Full annotations : {annotation_file}")

    # 2) peak-to-gene mapping (pickle)
    peak2gene_file = _path(f"{output_prefix}_peak2gene.pkl")
    with open(peak2gene_file, "wb") as f:
        pickle.dump(peak_annotation, f)
    output_files["peak2gene"] = peak2gene_file
    print(f"  • Peak-to-gene map  : {peak2gene_file}")

    # 3) stats
    stats_file = _path(f"{output_prefix}_stats.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    output_files["stats"] = stats_file
    print(f"  • Statistics       : {stats_file}")

    # 4) parameters
    params_file = _path(f"{output_prefix}_parameters.json")
    with open(params_file, "w") as f:
        json.dump(parameters, f, indent=2)
    output_files["parameters"] = params_file
    print(f"  • Parameters       : {params_file}")

    # 5) per-gene summary CSV
    gene_summary = (
        annotation_df.groupby("gene_name").agg(
            n_peaks=("peak", "count"),
            n_promoter_peaks=("in_promoter", "sum"),
            n_gene_body_peaks=("in_gene_body", "sum"),
            mean_tss_distance=("distance_to_tss", "mean"),
            total_weight=("combined_weight", "sum"),
        )
    )
    gene_summary_file = _path(f"{output_prefix}_gene_summary.csv")
    gene_summary.to_csv(gene_summary_file)
    output_files["gene_summary"] = gene_summary_file
    print(f"  • Gene summary     : {gene_summary_file}")

    # ── console summary ────────────────────────────────────────────────
    print("\n=== Annotation Summary ===")
    print(f"Total peaks processed            : {stats['total_peaks']:,}")
    print(
        f"Peaks with ≥1 gene annotation    : {stats['annotated_peaks']:,} "
        f"({stats['coverage_percent']:.1f}%)"
    )
    print(f"Unique genes annotated           : {stats['n_unique_genes']:,}")
    print(f"Total peak-gene associations     : {stats['total_associations']:,}")
    print(f"Mean genes per peak              : {stats['mean_genes_per_peak']:.2f}")
    print(f"Mean peaks per gene              : {stats['mean_peaks_per_gene']:.2f}")

    return output_files


# ────────────────────── convenience reload helper ─────────────────────
def load_annotations(*, output_prefix="atac_annotation", output_dir="."):
    """
    Reload results produced by `annotate_atac_peaks_parallel`.

    Parameters
    ----------
    output_prefix : str
        File prefix used during saving.
    output_dir : str
        Directory where the files reside.
    """
    output_dir = os.path.abspath(output_dir)
    print(f"Loading annotation files from {output_dir}…")

    def _path(fname):
        return os.path.join(output_dir, fname)

    with open(_path(f"{output_prefix}_peak2gene.pkl"), "rb") as f:
        peak2gene = pickle.load(f)

    annotation_df = pd.read_parquet(_path(f"{output_prefix}_full_annotations.parquet"))

    with open(_path(f"{output_prefix}_stats.json"), "r") as f:
        stats = json.load(f)

    with open(_path(f"{output_prefix}_parameters.json"), "r") as f:
        parameters = json.load(f)

    gene_summary = pd.read_csv(_path(f"{output_prefix}_gene_summary.csv"), index_col=0)

    return {
        "peak2gene": peak2gene,
        "annotation_df": annotation_df,
        "stats": stats,
        "parameters": parameters,
        "gene_summary": gene_summary,
    }


# ────────────────────────────── CLI demo ──────────────────────────────
if __name__ == "__main__":
    # Example usage
    atac_path = "/Users/harry/Desktop/GenoDistance/Data/test_ATAC.h5ad"
    results = annotate_atac_peaks_parallel(
        atac_path,
        ensembl_release=110,
        extend_upstream=100_000,
        extend_downstream=100_000,
        promoter_weight_factor=5.0,
        distance_weight_sigma=50_000,
        min_peak_accessibility=0.01,
        n_threads=8,
        output_prefix="atac_annotation",
        output_dir="/Users/harry/Desktop/GenoDistance/result/peak_annotation",   # write everything here
    )

    print("\nOutput files:")
    for k, v in results.items():
        print(f"  {k:<15} : {v}")
