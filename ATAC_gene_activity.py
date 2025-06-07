#!/usr/bin/env python3
"""
Enhanced Peak-to-Gene Activity Matrix Generator

*Modified to use gene IDs as primary identifiers*:
➜ Changed to use gene_ids instead of gene_names as primary keys
➜ Output matrix is structured as cells × gene_ids
➜ Gene names included for readability but gene_id is the unique identifier
➜ All grouping and indexing now uses gene_id for consistency
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path
from scipy.sparse import csr_matrix, lil_matrix, issparse
from collections import defaultdict
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

warnings.filterwarnings("ignore")


def process_gene_batch(args):
    """Worker function to process a batch of genes in parallel."""
    gene_batch, gene2peaks_weighted, peak_to_idx, X, aggregation_method, decay_params = args
    
    n_cells = X.shape[0]
    n_genes = len(gene_batch)
    
    # Initialize batch results
    batch_activity = lil_matrix((n_cells, n_genes))
    batch_stats = []
    
    for gene_idx, gene_id in enumerate(gene_batch):
        if gene_id not in gene2peaks_weighted:
            batch_stats.append({
                'gene_id': gene_id,
                'gene_name': gene2peaks_weighted.get(gene_id, {}).get('gene_name', 'Unknown'),
                'n_peaks': 0,
                'total_weight': 0,
                'mean_distance': np.nan,
                'n_promoter_peaks': 0,
                'n_gene_body_peaks': 0
            })
            continue
        
        peak_data = gene2peaks_weighted[gene_id]
        peak_indices = []
        weights = []
        distances = []
        n_promoter = 0
        n_gene_body = 0
        
        for peak_info in peak_data:
            peak = peak_info['peak']
            if peak in peak_to_idx:
                peak_indices.append(peak_to_idx[peak])
                
                # Use the combined weight from annotation
                weight = peak_info.get('combined_weight', 1.0)
                weights.append(weight)
                distances.append(peak_info.get('distance_to_tss', 0))
                
                if peak_info.get('in_promoter', False):
                    n_promoter += 1
                if peak_info.get('in_gene_body', False):
                    n_gene_body += 1
        
        if len(peak_indices) > 0:
            weights = np.array(weights)
            
            # Get peak counts
            peak_counts = X[:, peak_indices]
            
            if aggregation_method == 'weighted_sum':
                # ArchR-style weighted sum
                if issparse(peak_counts):
                    gene_activity_values = peak_counts.multiply(weights).sum(axis=1).A.flatten()
                else:
                    gene_activity_values = (peak_counts * weights).sum(axis=1)
                    
            elif aggregation_method == 'weighted_mean':
                # Weighted mean (normalized by total weight)
                if issparse(peak_counts):
                    weighted_counts = peak_counts.multiply(weights).sum(axis=1).A.flatten()
                else:
                    weighted_counts = (peak_counts * weights).sum(axis=1)
                gene_activity_values = weighted_counts / weights.sum()
                
            elif aggregation_method == 'max_weighted':
                # Maximum weighted peak (useful for sharp peaks)
                if issparse(peak_counts):
                    weighted_counts = peak_counts.multiply(weights)
                    gene_activity_values = weighted_counts.max(axis=1).A.flatten()
                else:
                    weighted_counts = peak_counts * weights
                    gene_activity_values = weighted_counts.max(axis=1)
                    
            else:  # 'sum' - simple sum without weights
                gene_activity_values = peak_counts.sum(axis=1).A.flatten() if issparse(peak_counts) else peak_counts.sum(axis=1)
            
            batch_activity[:, gene_idx] = gene_activity_values.reshape(-1, 1)
        
        # Get gene name for this gene_id (first occurrence in the data)
        gene_name = "Unknown"
        for peak_info in peak_data:
            if 'gene_name' in peak_info:
                gene_name = peak_info['gene_name']
                break
        
        batch_stats.append({
            'gene_id': gene_id,
            'gene_name': gene_name,
            'n_peaks': len(peak_indices),
            'total_weight': weights.sum() if len(weights) > 0 else 0,
            'mean_distance': np.mean(distances) if len(distances) > 0 else np.nan,
            'n_promoter_peaks': n_promoter,
            'n_gene_body_peaks': n_gene_body
        })
    
    return batch_activity.tocsr(), batch_stats


def peak_to_gene_activity_weighted(
    atac,
    annotation_results,
    output_dir,
    layer=None,
    aggregation_method='weighted_sum',
    distance_threshold=None,
    weight_threshold=0.01,
    n_threads=None,
    normalize_by='none',
    log_transform=False,
    scale_factors=None,
    verbose=True
):
    """
    Convert ATAC-seq peak counts to gene activity scores using weighted aggregation.
    Uses gene IDs as primary identifiers for robust mapping.
    
    Parameters:
    -----------
    atac : AnnData
        ATAC-seq data with peaks as features
    annotation_results : dict or str
        Either the annotation results dictionary from annotate_atac_peaks_parallel
        or path to the pickle file containing peak2gene mapping
    output_dir : str or Path
        Directory to save the gene activity AnnData
    layer : str or None
        Layer to use for counts (default: X)
    aggregation_method : str
        Method for aggregating peaks to genes:
        - 'weighted_sum': ArchR-style weighted sum (default)
        - 'weighted_mean': Weighted mean normalized by total weight
        - 'max_weighted': Maximum weighted peak
        - 'sum': Simple sum without weights
    distance_threshold : int or None
        Maximum TSS distance to include peaks (bp)
    weight_threshold : float
        Minimum weight to include a peak (default: 0.01)
    n_threads : int or None
        Number of threads for parallel processing
    normalize_by : str
        Normalization method:
        - 'none': No normalization
        - 'n_peaks': Normalize by number of peaks per gene
        - 'total_weight': Normalize by total weight per gene
        - 'archR': ArchR-style normalization (log2(counts + 1))
    log_transform : bool
        Apply log1p transformation after aggregation
    scale_factors : dict or None
        Optional scaling factors per cell type
    verbose : bool
        Print progress messages
    
    Returns:
    --------
    AnnData
        Gene activity matrix (cells × gene_ids) saved to output_dir/gene_activity_weighted.h5ad
    """
    
    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if n_threads is None:
        n_threads = mp.cpu_count()
    
    if verbose:
        print(f"Creating gene activity matrix using {n_threads} threads")
        print(f"Aggregation method: {aggregation_method}")
        print(f"Normalization: {normalize_by}")
        print(f"Output format: cells × gene_ids")
    
    # Load annotation results
    if isinstance(annotation_results, (str, Path)):
        with open(annotation_results, 'rb') as f:
            peak2gene = pickle.load(f)
        if verbose:
            print(f"Loaded peak annotations from {annotation_results}")
    elif isinstance(annotation_results, dict) and 'peak2gene' in annotation_results:
        peak2gene = annotation_results['peak2gene']
    else:
        peak2gene = annotation_results
    
    # Get count matrix
    if layer is not None:
        X = atac.layers[layer]
        if verbose:
            print(f"Using counts from layer '{layer}'")
    else:
        X = atac.X
    
    # Convert to sparse if needed
    if not issparse(X):
        X = csr_matrix(X)
    else:
        X = X.tocsr()
    
    # Build gene-to-peaks mapping with weights and filtering
    # Now using gene_id as the primary key
    gene2peaks_weighted = defaultdict(list)
    peak_stats = {
        'total_annotated': 0,
        'used_after_filtering': 0,
        'filtered_by_distance': 0,
        'filtered_by_weight': 0
    }
    
    for peak, annotation in peak2gene.items():
        peak_stats['total_annotated'] += 1
        
        if not isinstance(annotation, dict):
            continue
            
        # Use gene_ids as primary identifiers
        gene_ids = annotation.get('gene_ids', [])
        gene_names = annotation.get('gene_names', [])  # For readability
        weights = annotation.get('weights', [])
        distances = annotation.get('distances', [])
        in_promoter = annotation.get('in_promoter', [])
        in_gene_body = annotation.get('in_gene_body', [])
        tss_weights = annotation.get('tss_weights', [])
        
        # Process each gene annotation for this peak
        peak_used = False
        for i, gene_id in enumerate(gene_ids):
            # Get values with bounds checking
            weight = weights[i] if i < len(weights) else 0
            distance = distances[i] if i < len(distances) else float('inf')
            gene_name = gene_names[i] if i < len(gene_names) else "Unknown"
            
            # Apply filters
            if weight_threshold is not None and weight < weight_threshold:
                peak_stats['filtered_by_weight'] += 1
                continue
                
            if distance_threshold is not None and distance > distance_threshold:
                peak_stats['filtered_by_distance'] += 1
                continue
            
            # Store peak info for this gene_id
            peak_info = {
                'peak': peak,
                'gene_id': gene_id,
                'gene_name': gene_name,
                'combined_weight': weight,
                'distance_to_tss': distance,
                'in_promoter': in_promoter[i] if i < len(in_promoter) else False,
                'in_gene_body': in_gene_body[i] if i < len(in_gene_body) else False,
                'tss_weight': tss_weights[i] if i < len(tss_weights) else weight
            }
            
            gene2peaks_weighted[gene_id].append(peak_info)
            peak_used = True
        
        if peak_used:
            peak_stats['used_after_filtering'] += 1
    
    # Get valid gene IDs (primary identifiers)
    gene_ids = sorted(list(gene2peaks_weighted.keys()))
    gene_ids = [g for g in gene_ids if g and str(g).strip() and str(g).lower() != 'nan']
    n_genes = len(gene_ids)
    n_cells = atac.n_obs
    
    if verbose:
        print(f"\nPeak filtering statistics:")
        print(f"  Total annotated peaks: {peak_stats['total_annotated']:,}")
        print(f"  Used after filtering: {peak_stats['used_after_filtering']:,}")
        if weight_threshold:
            print(f"  Filtered by weight (<{weight_threshold}): {peak_stats['filtered_by_weight']:,}")
        if distance_threshold:
            print(f"  Filtered by distance (>{distance_threshold:,} bp): {peak_stats['filtered_by_distance']:,}")
        print(f"\nProcessing {n_genes:,} genes (by gene_id) from {atac.n_vars:,} peaks")
    
    # Create peak name to index mapping
    peak_to_idx = {peak: i for i, peak in enumerate(atac.var_names)}
    
    # Prepare gene batches for parallel processing
    batch_size = max(1, n_genes // (n_threads * 4))  # 4 batches per thread
    gene_batches = [gene_ids[i:i + batch_size] for i in range(0, n_genes, batch_size)]
    
    # Prepare arguments for parallel processing
    decay_params = {
        'sigma': 50000  # Default value
    }
    
    process_args = [
        (batch, gene2peaks_weighted, peak_to_idx, X, aggregation_method, decay_params)
        for batch in gene_batches
    ]
    
    # Process in parallel
    if verbose:
        print(f"Processing {len(gene_batches)} gene batches in parallel...")
    
    with mp.Pool(n_threads) as pool:
        results = list(tqdm(
            pool.imap(process_gene_batch, process_args),
            total=len(gene_batches),
            desc="Processing genes",
            disable=not verbose
        ))
    
    # Combine results
    if verbose:
        print("Combining results from parallel processing...")
    
    # Concatenate sparse matrices
    activity_matrices = [r[0] for r in results]
    gene_stats_lists = [r[1] for r in results]
    
    # Stack horizontally to get full gene activity matrix
    gene_activity = activity_matrices[0]
    for mat in activity_matrices[1:]:
        gene_activity = csr_matrix(np.hstack([gene_activity.toarray(), mat.toarray()]))
    
    # Combine gene statistics
    all_gene_stats = []
    for stats_list in gene_stats_lists:
        all_gene_stats.extend(stats_list)
    
    gene_stats_df = pd.DataFrame(all_gene_stats).set_index('gene_id')
    
    # Apply normalization
    if normalize_by == 'n_peaks':
        if verbose:
            print("Normalizing by number of peaks per gene...")
        for i, gene_id in enumerate(gene_ids):
            n_peaks = gene_stats_df.loc[gene_id, 'n_peaks']
            if n_peaks > 0:
                gene_activity[:, i] = gene_activity[:, i] / n_peaks
                
    elif normalize_by == 'total_weight':
        if verbose:
            print("Normalizing by total weight per gene...")
        for i, gene_id in enumerate(gene_ids):
            total_weight = gene_stats_df.loc[gene_id, 'total_weight']
            if total_weight > 0:
                gene_activity[:, i] = gene_activity[:, i] / total_weight
                
    elif normalize_by == 'archR':
        if verbose:
            print("Applying ArchR-style normalization (log2(counts + 1))...")
        gene_activity = csr_matrix(np.log2(gene_activity.toarray() + 1))
    
    # Apply log transformation if requested
    if log_transform and normalize_by != 'archR':
        if verbose:
            print("Applying log1p transformation...")
        gene_activity = csr_matrix(np.log1p(gene_activity.toarray()))
    
    # Apply cell-type specific scale factors if provided
    if scale_factors is not None:
        if verbose:
            print("Applying cell-type specific scale factors...")
        for cell_type, factor in scale_factors.items():
            mask = atac.obs['cell_type'] == cell_type if 'cell_type' in atac.obs else []
            gene_activity[mask, :] = gene_activity[mask, :] * factor
    
    # Create AnnData object with gene_ids as var_names
    adata_gene = ad.AnnData(
        X=gene_activity,
        obs=atac.obs.copy(),
        var=gene_stats_df.loc[gene_ids].copy()
    )
    
    # Set gene_ids as the index/var_names (primary identifiers)
    adata_gene.var_names = gene_ids
    adata_gene.var_names.name = 'gene_id'
    
    # Ensure gene_name is available for readability
    if 'gene_name' not in adata_gene.var.columns:
        adata_gene.var['gene_name'] = adata_gene.var.index  # Fallback
    
    # Copy relevant uns data
    for key in ['sample_name', 'genome', 'species']:
        if key in atac.uns:
            adata_gene.uns[key] = atac.uns[key]
    
    # Add processing metadata
    adata_gene.uns['gene_activity_params'] = {
        'method': 'weighted_aggregation_gene_id',
        'aggregation': aggregation_method,
        'normalization': normalize_by,
        'source_peaks': atac.n_vars,
        'target_genes': n_genes,
        'distance_threshold': distance_threshold,
        'weight_threshold': weight_threshold,
        'log_transformed': log_transform or normalize_by == 'archR',
        'n_threads': n_threads,
        'filtering_stats': peak_stats,
        'identifier_type': 'gene_id'
    }
    
    # Add ArchR-compatible metadata
    adata_gene.uns['GeneScoreMatrix'] = {
        'method': 'ArchR-compatible weighted aggregation (gene_id indexed)',
        'date': pd.Timestamp.now().isoformat(),
        'parameters': {
            'aggregation': aggregation_method,
            'normalization': normalize_by,
            'distance_threshold': distance_threshold,
            'weight_threshold': weight_threshold,
            'identifier': 'gene_id'
        }
    }
    
    # Save results
    output_path = output_dir / 'gene_activity_weighted.h5ad'
    adata_gene.write(output_path)
    
    if verbose:
        print(f"\nGene activity matrix created: {n_cells:,} cells × {n_genes:,} genes")
        print(f"Matrix structure: cells × gene_ids")
        print(f"Results saved to: {output_path}")
        
        # Summary statistics
        non_zero_genes = (adata_gene.X.sum(axis=0) > 0).A1.sum()
        sparsity = 1 - (adata_gene.X.nnz / (n_cells * n_genes))
        
        print(f"\nMatrix statistics:")
        print(f"  Non-zero genes: {non_zero_genes:,}/{n_genes:,} ({100*non_zero_genes/n_genes:.1f}%)")
        print(f"  Sparsity: {100*sparsity:.1f}%")
        print(f"  Total counts: {adata_gene.X.sum():,.0f}")
        
        # Gene statistics
        print(f"\nGene statistics:")
        print(f"  Peaks per gene: {adata_gene.var['n_peaks'].mean():.1f} ± {adata_gene.var['n_peaks'].std():.1f}")
        print(f"  Promoter peaks per gene: {adata_gene.var['n_promoter_peaks'].mean():.1f}")
        print(f"  Gene body peaks per gene: {adata_gene.var['n_gene_body_peaks'].mean():.1f}")
        
        valid_distances = adata_gene.var['mean_distance'].dropna()
        if len(valid_distances) > 0:
            print(f"  Mean TSS distance: {valid_distances.mean():.0f} bp")
        
        # Show example gene IDs vs names
        print(f"\nExample gene ID → name mapping:")
        for i in range(min(5, len(adata_gene.var))):
            gene_id = adata_gene.var_names[i]
            gene_name = adata_gene.var.iloc[i]['gene_name']
            print(f"  {gene_id} → {gene_name}")
    
    return adata_gene


def brief_gene_activity_overview(adata):
    """
    Enhanced overview of gene activity AnnData object with gene ID focus.
    """
    print(f"Gene Activity Matrix Overview")
    print(f"=" * 50)
    print(f"Shape: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
    print(f"Matrix type: {'Sparse' if issparse(adata.X) else 'Dense'}")
    print(f"Gene identifiers: {adata.var_names.name or 'gene_id'}")
    
    # Matrix statistics
    if issparse(adata.X):
        total_counts = adata.X.sum()
        non_zero_pct = (adata.X.nnz / adata.X.size) * 100
    else:
        total_counts = adata.X.sum()
        non_zero_pct = (np.count_nonzero(adata.X) / adata.X.size) * 100
    
    print(f"Total counts: {total_counts:,.0f}")
    print(f"Non-zero values: {non_zero_pct:.1f}%")
    
    # Processing information
    if 'gene_activity_params' in adata.uns:
        params = adata.uns['gene_activity_params']
        print(f"\nProcessing parameters:")
        print(f"  Aggregation: {params.get('aggregation', 'unknown')}")
        print(f"  Normalization: {params.get('normalization', 'unknown')}")
        print(f"  Distance threshold: {params.get('distance_threshold', 'None')}")
        print(f"  Weight threshold: {params.get('weight_threshold', 'None')}")
        print(f"  Identifier type: {params.get('identifier_type', 'unknown')}")
    
    # Gene statistics
    if 'n_peaks' in adata.var.columns:
        print(f"\nGene statistics:")
        print(f"  Peaks per gene: {adata.var['n_peaks'].mean():.1f} (range: {adata.var['n_peaks'].min()}-{adata.var['n_peaks'].max()})")
    
    if 'n_promoter_peaks' in adata.var.columns:
        print(f"  Promoter peaks: {adata.var['n_promoter_peaks'].sum():,} total")
    
    if 'n_gene_body_peaks' in adata.var.columns:
        print(f"  Gene body peaks: {adata.var['n_gene_body_peaks'].sum():,} total")
    
    # Gene ID and name mapping
    if 'gene_name' in adata.var.columns:
        print(f"\nGene ID → Name examples:")
        for i in range(min(3, len(adata.var))):
            gene_id = adata.var_names[i]
            gene_name = adata.var.iloc[i]['gene_name']
            print(f"  {gene_id} → {gene_name}")
    
    # Sample of the matrix
    print(f"\nCount matrix sample (first 3 genes × 3 cells):")
    if issparse(adata.X):
        sample_matrix = adata.X[:3, :3].toarray().T
    else:
        sample_matrix = adata.X[:3, :3].T
    
    sample_df = pd.DataFrame(
        sample_matrix,
        index=adata.var_names[:3],
        columns=adata.obs_names[:3]
    )
    print(sample_df.round(4))


# Example usage
if __name__ == "__main__":
    # Load ATAC data
    atac = ad.read_h5ad("/Users/harry/Desktop/GenoDistance/Data/test_ATAC.h5ad")
    
    # Load annotation results (from the new gene_id-based annotation function)
    with open("/Users/harry/Desktop/GenoDistance/result/peak_annotation/atac_annotation_peak2gene.pkl", "rb") as f:
        annotation_results = pickle.load(f)
    
    # Create gene activity matrix with gene IDs as primary identifiers
    adata_gene = peak_to_gene_activity_weighted(
        atac=atac,
        annotation_results=annotation_results,
        output_dir="/Users/harry/Desktop/GenoDistance/result/gene_activity/",
        aggregation_method='weighted_sum',  # ArchR-style
        distance_threshold=100_000,  # 100kb
        weight_threshold=0.01,  # Minimum weight
        normalize_by='archR',  # ArchR normalization
        verbose=True
    )
    
    # Show overview
    brief_gene_activity_overview(adata_gene)