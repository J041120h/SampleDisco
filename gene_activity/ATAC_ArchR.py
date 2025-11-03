#!/usr/bin/env python3

from __future__ import annotations

import multiprocessing as mp
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from pyensembl import EnsemblRelease
from scipy.sparse import csr_matrix, diags, hstack, issparse, lil_matrix
from tqdm.auto import tqdm


def inspect_atac_data(adata: ad.AnnData, n_peaks_show: int = 10) -> None:
    print("=" * 60)
    print("ATAC-seq Data Inspection")
    print("=" * 60)
    
    print(f"Data shape: {adata.shape[0]:,} cells × {adata.shape[1]:,} peaks")
    print(f"Data type: {type(adata.X)}")
    print(f"Sparse: {issparse(adata.X)}")
    
    print(f"\n--- Peak Naming Analysis ---")
    peak_names = adata.var_names.tolist()
    print(f"First {n_peaks_show} peak names:")
    for i, peak in enumerate(peak_names[:n_peaks_show]):
        print(f"  {i+1:2d}. {peak}")
    
    formats = {
        "chr:start-end": 0,
        "chr-start-end": 0,
        "chr_start_end": 0,
        "chrN:start-end": 0,
        "N:start-end": 0,
        "other": 0
    }
    
    chromosomes = set()
    malformed = 0
    
    for peak in peak_names[:1000]:
        if ":" in peak and "-" in peak:
            try:
                chrom, coord = peak.split(":", 1)
                start_str, end_str = coord.split("-", 1)
                start, end = int(start_str), int(end_str)
                
                chromosomes.add(chrom)
                
                if chrom.startswith("chr") and chrom[3:].isdigit():
                    formats["chrN:start-end"] += 1
                elif chrom.startswith("chr"):
                    formats["chr:start-end"] += 1
                elif chrom.isdigit():
                    formats["N:start-end"] += 1
                else:
                    formats["other"] += 1
                    
            except (ValueError, IndexError):
                malformed += 1
        elif peak.count("-") == 2 and ":" not in peak:
            try:
                parts = peak.split("-")
                if len(parts) == 3:
                    chrom = parts[0]
                    start, end = int(parts[1]), int(parts[2])
                    chromosomes.add(chrom)
                    formats["chr-start-end"] += 1
                else:
                    malformed += 1
            except (ValueError, IndexError):
                malformed += 1
        elif "_" in peak:
            formats["chr_start_end"] += 1
        else:
            formats["other"] += 1
    
    print(f"\nPeak naming formats (from first 1,000 peaks):")
    for fmt, count in formats.items():
        if count > 0:
            print(f"  {fmt}: {count:,} peaks")
    
    if malformed > 0:
        print(f"  Malformed: {malformed:,} peaks")
    
    print(f"\nChromosomes found (from first 1,000 peaks): {sorted(list(chromosomes))}")
    
    print(f"\n--- Data Statistics ---")
    if issparse(adata.X):
        total_counts = adata.X.sum()
        nonzero_entries = adata.X.nnz
        sparsity = 1 - (nonzero_entries / (adata.X.shape[0] * adata.X.shape[1]))
    else:
        total_counts = adata.X.sum()
        nonzero_entries = np.count_nonzero(adata.X)
        sparsity = 1 - (nonzero_entries / adata.X.size)
    
    print(f"Total counts: {total_counts:,.0f}")
    print(f"Sparsity: {sparsity:.3f} ({100*sparsity:.1f}% zeros)")
    print(f"Mean counts per cell: {total_counts/adata.shape[0]:.1f}")
    print(f"Mean counts per peak: {total_counts/adata.shape[1]:.1f}")
    
    if 'n_genes_by_counts' not in adata.obs.columns:
        sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    
    print(f"\n--- Cell Statistics ---")
    print(f"Peaks per cell - Mean: {adata.obs['n_genes_by_counts'].mean():.0f}, "
          f"Median: {adata.obs['n_genes_by_counts'].median():.0f}")
    print(f"Peaks per cell - Min: {adata.obs['n_genes_by_counts'].min():.0f}, "
          f"Max: {adata.obs['n_genes_by_counts'].max():.0f}")
    print(f"Total counts per cell - Mean: {adata.obs['total_counts'].mean():.0f}, "
          f"Median: {adata.obs['total_counts'].median():.0f}")
    
    print(f"\n--- Peak Statistics ---")
    print(f"Cells per peak - Mean: {adata.var['n_cells_by_counts'].mean():.1f}, "
          f"Median: {adata.var['n_cells_by_counts'].median():.1f}")
    print(f"Cells per peak - Min: {adata.var['n_cells_by_counts'].min():.0f}, "
          f"Max: {adata.var['n_cells_by_counts'].max():.0f}")
    print(f"Total counts per peak - Mean: {adata.var['total_counts'].mean():.1f}, "
          f"Median: {adata.var['total_counts'].median():.1f}")
    
    print(f"\n--- Suggested Parameters ---")
    q10_genes = int(adata.obs['n_genes_by_counts'].quantile(0.1))
    q90_genes = int(adata.obs['n_genes_by_counts'].quantile(0.9))
    q10_counts = int(adata.obs['total_counts'].quantile(0.1))
    
    print(f"min_genes: {q10_genes:,} (10th percentile)")
    print(f"max_genes: {q90_genes:,} (90th percentile)")
    print(f"min_counts_per_cell: {q10_counts:,} (10th percentile)")
    
    mean_accessibility = np.asarray(adata.X.mean(axis=0)).ravel()
    suggested_min_access = np.percentile(mean_accessibility, 25)
    print(f"min_peak_accessibility: {suggested_min_access:.4f} (25th percentile)")
    
    print("=" * 60)


def debug_gene_peak_matching(
    peak_tiles: Dict[str, dict], 
    genes: Dict[str, dict], 
    gene_window: int,
    n_genes_check: int = 5,
    n_peaks_check: int = 20
) -> None:
    print(f"\n--- Gene-Peak Matching Debug ---")
    print(f"Total genes: {len(genes):,}")
    print(f"Total peak tiles: {len(peak_tiles):,}")
    
    gene_chroms = set(g["chrom"] for g in genes.values())
    peak_chroms = set(t["chrom"] for t in peak_tiles.values())
    
    print(f"\nChromosomes in genes: {sorted(gene_chroms)}")
    print(f"Chromosomes in peaks: {sorted(peak_chroms)}")
    print(f"Chromosome overlap: {sorted(gene_chroms & peak_chroms)}")
    print(f"Genes missing chromosomes: {sorted(gene_chroms - peak_chroms)}")
    print(f"Peaks missing chromosomes: {sorted(peak_chroms - gene_chroms)}")
    
    print(f"\n--- Detailed Matching for First {n_genes_check} Genes ---")
    for i, (gid, gene) in enumerate(list(genes.items())[:n_genes_check]):
        print(f"\nGene {i+1}: {gid} ({gene.get('gene_name', 'unnamed')})")
        print(f"  Location: {gene['chrom']}:{gene['start']}-{gene['end']} (TSS: {gene['tss']})")
        print(f"  Length: {gene['length']:,} bp")
        
        nearby_peaks = []
        for peak_name, tile in list(peak_tiles.items())[:n_peaks_check]:
            if tile["chrom"] == gene["chrom"]:
                center = tile["tile_center"]
                if gene["start"] - gene_window <= center <= gene["end"] + gene_window:
                    distance = min(
                        abs(center - gene["start"]),
                        abs(center - gene["end"]),
                        abs(center - gene["tss"])
                    )
                    nearby_peaks.append((peak_name, center, distance))
        
        nearby_peaks.sort(key=lambda x: x[2])
        print(f"  Nearby peaks: {len(nearby_peaks)}")
        for j, (peak_name, center, dist) in enumerate(nearby_peaks[:3]):
            print(f"    {j+1}. {peak_name} (center: {center}, distance: {dist:,} bp)")


def filter_atac(
    adata: ad.AnnData,
    *,
    min_cells: int = 1,
    min_genes: int = 1_000,
    max_genes: int = 50_000,
    min_counts_per_cell: int = 1_000,
    max_counts_per_cell: Optional[int] = None,
    min_counts_per_peak: int = 1,
    max_pct_mt: Optional[float] = None,
    verbose: bool = True,
) -> ad.AnnData:
    if verbose:
        print(f"Starting QC filtering: {adata.shape[0]:,} cells × {adata.shape[1]:,} peaks")
    
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    
    keep_vars = (
        (adata.var["n_cells_by_counts"] >= min_cells) &
        (adata.var["total_counts"] >= min_counts_per_peak)
    )
    
    keep_obs = (
        (adata.obs["n_genes_by_counts"] >= min_genes) &
        (adata.obs["n_genes_by_counts"] <= max_genes) &
        (adata.obs["total_counts"] >= min_counts_per_cell)
    )
    
    if max_counts_per_cell is not None:
        keep_obs &= (adata.obs["total_counts"] <= max_counts_per_cell)
    
    if max_pct_mt is not None and 'pct_counts_mt' in adata.obs.columns:
        keep_obs &= (adata.obs["pct_counts_mt"] <= max_pct_mt)
    
    if verbose:
        print(f"  → Keeping {keep_obs.sum():,}/{len(keep_obs):,} cells")
        print(f"  → Keeping {keep_vars.sum():,}/{len(keep_vars):,} peaks")
    
    filtered_adata = adata[keep_obs, :][:, keep_vars].copy()
    
    if verbose:
        print(f"Filtered data: {filtered_adata.shape[0]:,} cells × {filtered_adata.shape[1]:,} peaks")
    
    return filtered_adata


@dataclass
class _Params:
    tile_size: int
    gene_window: int
    scale_max: float
    decay_distance: int


def _process_gene_batch(
    args: Tuple[
        Sequence[str],
        Dict[str, dict],
        Dict[str, dict],
        Dict[str, int],
        csr_matrix,
        set[str],
        _Params,
    ]
) -> Tuple[csr_matrix, List[dict]]:
    gene_batch, all_genes, peak_tiles, peak_to_idx, X, valid_peaks, params = args
    
    n_cells = X.shape[0]
    batch_activity = lil_matrix((n_cells, len(gene_batch)))
    batch_stats: List[dict] = []

    min_weight = float(np.exp(-1))

    def calc_weight(distance: int, gene_len: int) -> float:
        w_dist = float(np.exp(-abs(distance) / params.decay_distance) + min_weight)
        
        if gene_len <= 500:
            size_w = params.scale_max
        elif gene_len >= 100_000:
            size_w = 1.0
        else:
            frac = (100_000 - gene_len) / 99_500
            size_w = 1.0 + frac * (params.scale_max - 1.0)
        
        return w_dist * size_w

    gene_window = params.gene_window

    for j, gid in enumerate(gene_batch):
        try:
            if gid not in all_genes:
                batch_stats.append({
                    "gene_id": gid,
                    "gene_name": gid,
                    "gene_length": 0,
                    "n_peaks": 0,
                })
                continue
                
            g = all_genes[gid]
            g_chrom = g["chrom"]
            g_start, g_end, g_len, g_tss = g["start"], g["end"], g["length"], g["tss"]

            gene_peaks: List[int] = []
            gene_weights: List[float] = []

            for peak, tile in peak_tiles.items():
                if tile["chrom"] != g_chrom:
                    continue

                center = tile["tile_center"]

                if center < g_start - gene_window or center > g_end + gene_window:
                    continue

                closer = False
                for other_gid, other in all_genes.items():
                    if other["chrom"] != g_chrom or other_gid == gid:
                        continue
                    if other["start"] - gene_window <= center <= other["end"] + gene_window:
                        dist_other = min(
                            abs(center - other["start"]),
                            abs(center - other["end"]),
                            abs(center - other["tss"]),
                        )
                        dist_self = min(
                            abs(center - g_start),
                            abs(center - g_end),
                            abs(center - g_tss),
                        )
                        if dist_other < dist_self:
                            closer = True
                            break
                
                if closer:
                    continue

                if g_start <= center <= g_end:
                    distance = 0
                else:
                    distance = min(
                        abs(center - g_start), 
                        abs(center - g_end), 
                        abs(center - g_tss)
                    )
                
                if peak in peak_to_idx and peak in valid_peaks:
                    gene_peaks.append(peak_to_idx[peak])
                    gene_weights.append(calc_weight(distance, g_len))

            if gene_peaks:
                peak_counts = X[:, gene_peaks]
                w = np.array(gene_weights)
                
                if issparse(peak_counts):
                    activity = peak_counts.multiply(w).sum(axis=1)
                else:
                    activity = (peak_counts * w).sum(axis=1)
                
                batch_activity[:, j] = np.ravel(activity)

            batch_stats.append({
                "gene_id": gid,
                "gene_name": g.get("gene_name", gid),
                "gene_length": g_len,
                "n_peaks": len(gene_peaks),
            })

        except Exception as e:
            warnings.warn(f"Error processing gene {gid}: {str(e)}")
            batch_stats.append({
                "gene_id": gid,
                "gene_name": gid,
                "gene_length": 0,
                "n_peaks": 0,
            })

    return batch_activity.tocsr(), batch_stats


class ArchRGeneActivity:

    def __init__(
        self,
        tile_size: int = 500,
        gene_window: int = 100_000,
        scale_max: float = 5.0,
        scale_to: int = 10_000,
        decay_distance: int = 5_000,
    ):
        self.tile_size = tile_size
        self.gene_window = gene_window
        self.scale_max = scale_max
        self.scale_to = scale_to
        self.decay_distance = decay_distance

    def load_genes_from_ensembl(
        self, 
        species: str = "homo_sapiens", 
        release: Optional[int] = None,
        chromosomes: Optional[List[str]] = None,
    ) -> Dict[str, dict]:
        release = release or 110
        print(f"→ Loading Ensembl {species} release {release}...")
        
        try:
            ens = EnsemblRelease(release, species=species)
            ens.download()
            ens.index()
        except Exception as e:
            raise RuntimeError(f"Failed to load Ensembl data: {str(e)}")

        genes: Dict[str, dict] = {}
        skipped_contigs = set()
        
        for g in ens.genes():
            if g.contig.startswith("MT") or g.contig.startswith("chrM"):
                continue
            
            chrom = g.contig if g.contig.startswith("chr") else f"chr{g.contig}"
            
            if chromosomes is not None and chrom not in chromosomes:
                skipped_contigs.add(chrom)
                continue
            
            tss = g.start if g.strand == "+" else g.end
            
            gid = g.gene_id.split(".")[0]
            genes[gid] = {
                "gene_id": gid,
                "gene_name": g.gene_name or gid,
                "chrom": chrom,
                "start": g.start,
                "end": g.end,
                "strand": g.strand,
                "length": g.end - g.start,
                "tss": tss,
                "biotype": g.biotype,
            }

        print(f"  → Loaded {len(genes):,} genes")
        if skipped_contigs:
            print(f"  → Skipped contigs: {sorted(skipped_contigs)}")
        
        return genes

    def create_peak_tiles(self, peaks: Sequence[str], verbose: bool = True) -> Dict[str, dict]:
        tiles: Dict[str, dict] = {}
        malformed = 0
        format_stats = {"standard": 0, "dash_only": 0, "underscore": 0, "other": 0}
        
        for peak in peaks:
            try:
                chrom, start, end = None, None, None
                
                if ":" in peak and "-" in peak:
                    chrom, coord = peak.split(":", 1)
                    start_str, end_str = coord.split("-", 1)
                    start, end = int(start_str), int(end_str)
                    format_stats["standard"] += 1
                
                elif peak.count("-") == 2 and ":" not in peak:
                    parts = peak.split("-")
                    if len(parts) == 3:
                        chrom = parts[0]
                        start, end = int(parts[1]), int(parts[2])
                        format_stats["dash_only"] += 1
                    else:
                        malformed += 1
                        continue
                
                elif peak.count("_") >= 2:
                    parts = peak.split("_")
                    if len(parts) >= 3:
                        chrom = parts[0]
                        start, end = int(parts[1]), int(parts[2])
                        format_stats["underscore"] += 1
                    else:
                        malformed += 1
                        continue
                
                else:
                    malformed += 1
                    format_stats["other"] += 1
                    continue
                
                if start is None or end is None or start >= end:
                    malformed += 1
                    continue
                
                if not chrom.startswith("chr"):
                    chrom = f"chr{chrom}"
                
                tile_start = (start // self.tile_size) * self.tile_size
                tile_end = ((end // self.tile_size) + 1) * self.tile_size
                
                tiles[peak] = {
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "tile_center": (tile_start + tile_end) // 2,
                }
                
            except (ValueError, IndexError):
                malformed += 1
                format_stats["other"] += 1
                continue
        
        if verbose:
            print(f"  → Peak format statistics:")
            for fmt, count in format_stats.items():
                if count > 0:
                    print(f"    {fmt}: {count:,} peaks")
            if malformed > 0:
                print(f"  → Warning: {malformed:,} malformed peak names skipped")
        
        return tiles

    def create_gene_activity(
        self,
        atac_adata: ad.AnnData,
        *,
        species: str = "homo_sapiens",
        release: Optional[int] = None,
        chromosomes: Optional[List[str]] = None,
        min_peak_accessibility: float = 0.01,
        min_cells: int = 3,
        min_genes: int = 1_000,
        max_genes: int = 50_000,
        min_counts_per_cell: int = 1_000,
        max_counts_per_cell: Optional[int] = None,
        min_counts_per_peak: int = 3,
        max_pct_mt: Optional[float] = 20.0,
        n_threads: Optional[int] = None,
        protein_coding_only: bool = False,
        verbose: bool = True,
    ) -> ad.AnnData:
        
        if verbose:
            inspect_atac_data(atac_adata)
        
        genes = self.load_genes_from_ensembl(species, release, chromosomes)
        if protein_coding_only:
            genes = {k: v for k, v in genes.items() if v["biotype"] == "protein_coding"}
            if verbose:
                print(f"  → Filtered to {len(genes):,} protein-coding genes")

        atac_adata = filter_atac(
            atac_adata,
            min_cells=min_cells,
            min_genes=min_genes,
            max_genes=max_genes,
            min_counts_per_cell=min_counts_per_cell,
            max_counts_per_cell=max_counts_per_cell,
            min_counts_per_peak=min_counts_per_peak,
            max_pct_mt=max_pct_mt,
            verbose=verbose,
        )

        X = atac_adata.X
        if not issparse(X):
            X = csr_matrix(X)
        X = X.tocsr()

        peak_means = np.asarray(X.mean(axis=0)).ravel()
        valid_mask = peak_means >= min_peak_accessibility
        valid_peaks = {
            p for p, m in zip(atac_adata.var_names, valid_mask) if m
        }
        
        if verbose:
            print(f"  → {len(valid_peaks):,}/{len(atac_adata.var_names):,} peaks "
                  f"pass accessibility ≥ {min_peak_accessibility}")

        peak_tiles = self.create_peak_tiles(valid_peaks, verbose=verbose)
        peak_to_idx = {p: i for i, p in enumerate(atac_adata.var_names)}
        
        if verbose:
            debug_gene_peak_matching(peak_tiles, genes, self.gene_window)

        gid_list = sorted(genes.keys())
        n_threads = n_threads or mp.cpu_count()
        n_threads = max(1, min(n_threads, mp.cpu_count()))
        
        if n_threads == 1:
            gene_batches = [gid_list]
        else:
            batch_size = max(1, len(gid_list) // (n_threads * 4))
            gene_batches = [
                gid_list[i:i + batch_size] 
                for i in range(0, len(gid_list), batch_size)
            ]

        params = _Params(
            tile_size=self.tile_size,
            gene_window=self.gene_window,
            scale_max=self.scale_max,
            decay_distance=self.decay_distance,
        )

        if verbose:
            print(f"→ Computing gene activity scores for {len(gid_list):,} genes using {n_threads} threads...")

        if n_threads == 1:
            results = []
            for i, batch in enumerate(tqdm(gene_batches, disable=not verbose, desc="Processing gene batches")):
                args = (batch, genes, peak_tiles, peak_to_idx, X, valid_peaks, params)
                result = _process_gene_batch(args)
                results.append(result)
                if verbose:
                    print(f"  → Processed batch {i+1}/{len(gene_batches)} ({len(batch)} genes)")
        else:
            args_list = [
                (batch, genes, peak_tiles, peak_to_idx, X, valid_peaks, params)
                for batch in gene_batches
            ]
            
            with mp.Pool(processes=n_threads) as pool:
                results = []
                for i, result in enumerate(tqdm(
                    pool.map(_process_gene_batch, args_list),
                    total=len(args_list),
                    disable=not verbose,
                    desc="Processing gene batches"
                )):
                    results.append(result)
                    if verbose and (i + 1) % 5 == 0:
                        print(f"  → Completed {i+1}/{len(args_list)} batches")

        mats = [r[0] for r in results]
        stats = [s for r in results for s in r[1]]
        
        gene_activity = hstack(mats).tocsr()
        stats_df = pd.DataFrame(stats)

        active_mask = stats_df["n_peaks"] > 0
        gene_activity = gene_activity[:, active_mask.values]
        stats_df = stats_df.loc[active_mask].set_index("gene_id")

        if verbose:
            print(f"  → {active_mask.sum():,}/{len(active_mask):,} genes have associated peaks")

        totals = np.asarray(gene_activity.sum(axis=1)).ravel()
        size_factors = self.scale_to / np.maximum(totals, 1e-10)
        gene_activity = diags(size_factors) @ gene_activity

        adata_gene = ad.AnnData(
            X=gene_activity,
            obs=atac_adata.obs.copy(),
            var=stats_df
        )

        if verbose:
            print(f"\n✓ Gene activity matrix: {adata_gene.shape[0]:,} cells × {adata_gene.shape[1]:,} genes")
        
        return adata_gene


def create_gene_activity_archR(
    atac_adata: Union[str, ad.AnnData],
    output_dir: str,
    *,
    species: str = "homo_sapiens",
    release: Optional[int] = None,
    chromosomes: Optional[List[str]] = None,
    min_peak_accessibility: float = 0.01,
    min_cells: int = 3,
    min_genes: int = 1_000,
    max_genes: int = 50_000,
    min_counts_per_cell: int = 1_000,
    max_counts_per_cell: Optional[int] = None,
    min_counts_per_peak: int = 3,
    max_pct_mt: Optional[float] = 20.0,
    n_threads: Optional[int] = None,
    protein_coding_only: bool = False,
    tile_size: int = 500,
    gene_window: int = 100_000,
    scale_max: float = 5.0,
    scale_to: int = 10_000,
    decay_distance: int = 5_000,
    verbose: bool = True,
) -> ad.AnnData:
    
    if isinstance(atac_adata, str):
        if verbose:
            print(f"Loading ATAC data from: {atac_adata}")
        atac_adata = ad.read_h5ad(atac_adata)
    
    if verbose:
        print(f"\n=== Quick Data Overview ===")
        print(f"Data shape: {atac_adata.shape[0]:,} cells × {atac_adata.shape[1]:,} peaks")
        if hasattr(atac_adata.X, 'nnz'):
            sparsity = 1 - (atac_adata.X.nnz / (atac_adata.X.shape[0] * atac_adata.X.shape[1]))
            print(f"Sparsity: {sparsity:.3f} ({100*sparsity:.1f}% zeros)")
        
        sample_peaks = atac_adata.var_names[:5].tolist()
        print(f"Sample peak names: {sample_peaks}")
    
    archr = ArchRGeneActivity(
        tile_size=tile_size,
        gene_window=gene_window,
        scale_max=scale_max,
        scale_to=scale_to,
        decay_distance=decay_distance,
    )
    
    gene_activity = archr.create_gene_activity(
        atac_adata,
        species=species,
        release=release,
        chromosomes=chromosomes,
        min_peak_accessibility=min_peak_accessibility,
        min_cells=min_cells,
        min_genes=min_genes,
        max_genes=max_genes,
        min_counts_per_cell=min_counts_per_cell,
        max_counts_per_cell=max_counts_per_cell,
        min_counts_per_peak=min_counts_per_peak,
        max_pct_mt=max_pct_mt,
        n_threads=n_threads,
        protein_coding_only=protein_coding_only,
        verbose=verbose,
    )
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "gene_activity_matrix.h5ad")
    gene_activity.write(output_path)
    
    if verbose:
        print(f"Gene activity matrix saved to: {output_path}")
    
    return gene_activity


if __name__ == "__main__":
    print("Starting ArchR Gene Activity Analysis...")
    
    gene_activity = create_gene_activity_archR(
        atac_adata="/dcl01/hongkai/data/data/hjiang/Data/paired/atac/placenta.h5ad",
        output_dir="/dcs07/hongkai/data/harry/result/gene_activity/ATAC_ArchR",
        species="homo_sapiens",
        release=98,
        protein_coding_only=True,
        min_peak_accessibility=0.003,
        min_cells=1,
        min_counts_per_peak=1,
        min_genes=1_500,
        max_genes=5_000,
        min_counts_per_cell=1_500,
        max_counts_per_cell=None,
        max_pct_mt=None,
        n_threads=8,
        verbose=True,
        gene_window=1_000,
        scale_max=5.0,
        decay_distance=5_000,
    )
    