#!/usr/bin/env python3
"""
GPU-accelerated peak-to-gene activity matrix generator.
Falls back to CPU multiprocessing when CuPy is unavailable.
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

# GPU imports
try:
    import cupy as cp
    import cupyx.scipy.sparse as cusparse
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not available. Install with: pip install cupy-cuda11x (adjust for your CUDA version)")

warnings.filterwarnings("ignore")


def process_gene_batch_gpu(args):
    """GPU-accelerated worker function to process a batch of genes."""
    gene_batch, gene2peaks_weighted, peak_to_idx, X_gpu, aggregation_method, decay_params, device_id = args
    
    cp.cuda.Device(device_id).use()
    
    n_cells = X_gpu.shape[0]
    n_genes = len(gene_batch)
    
    # Initialize batch results on GPU
    batch_activity = cp.zeros((n_cells, n_genes), dtype=cp.float32)
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
                weight = peak_info.get('combined_weight', 1.0)
                weights.append(weight)
                distances.append(peak_info.get('distance_to_tss', 0))
                
                if peak_info.get('in_promoter', False):
                    n_promoter += 1
                if peak_info.get('in_gene_body', False):
                    n_gene_body += 1
        
        if len(peak_indices) > 0:
            peak_indices_gpu = cp.array(peak_indices, dtype=cp.int32)
            weights_gpu = cp.array(weights, dtype=cp.float32)

            peak_counts = X_gpu[:, peak_indices_gpu]

            if aggregation_method == 'weighted_sum':
                if hasattr(peak_counts, 'multiply'):  # sparse
                    gene_activity_values = peak_counts.multiply(weights_gpu).sum(axis=1)
                    if hasattr(gene_activity_values, 'toarray'):
                        gene_activity_values = gene_activity_values.toarray().flatten()
                else:  # dense
                    gene_activity_values = (peak_counts * weights_gpu).sum(axis=1)

            elif aggregation_method == 'weighted_mean':
                if hasattr(peak_counts, 'multiply'):  # sparse
                    weighted_counts = peak_counts.multiply(weights_gpu).sum(axis=1)
                    if hasattr(weighted_counts, 'toarray'):
                        weighted_counts = weighted_counts.toarray().flatten()
                else:  # dense
                    weighted_counts = (peak_counts * weights_gpu).sum(axis=1)
                gene_activity_values = weighted_counts / weights_gpu.sum()

            elif aggregation_method == 'max_weighted':
                if hasattr(peak_counts, 'multiply'):  # sparse
                    weighted_counts = peak_counts.multiply(weights_gpu)
                    gene_activity_values = weighted_counts.max(axis=1)
                    if hasattr(gene_activity_values, 'toarray'):
                        gene_activity_values = gene_activity_values.toarray().flatten()
                else:  # dense
                    weighted_counts = peak_counts * weights_gpu
                    gene_activity_values = weighted_counts.max(axis=1)

            else:  # 'sum' - simple sum without weights
                gene_activity_values = peak_counts.sum(axis=1)
                if hasattr(gene_activity_values, 'toarray'):
                    gene_activity_values = gene_activity_values.toarray().flatten()
            
            batch_activity[:, gene_idx] = gene_activity_values.flatten()
        
        gene_name = "Unknown"
        for peak_info in peak_data:
            if 'gene_name' in peak_info:
                gene_name = peak_info['gene_name']
                break
        
        batch_stats.append({
            'gene_id': gene_id,
            'gene_name': gene_name,
            'n_peaks': len(peak_indices),
            'total_weight': float(cp.asnumpy(weights_gpu.sum())) if len(weights) > 0 else 0,
            'mean_distance': np.mean(distances) if len(distances) > 0 else np.nan,
            'n_promoter_peaks': n_promoter,
            'n_gene_body_peaks': n_gene_body
        })
    
    batch_activity_cpu = cp.asnumpy(batch_activity)
    return csr_matrix(batch_activity_cpu), batch_stats


def peak_to_gene_activity_weighted_gpu(
    atac,
    annotation_results,
    output_dir,
    layer=None,
    aggregation_method='weighted_sum',
    distance_threshold=None,
    weight_threshold=0.01,
    min_peak_accessibility=0.01,
    n_gpu_workers=None,
    gpu_batch_size=None,
    normalize_by='total_weight',
    log_transform=False,
    scale_factors=None,
    verbose=True,
    use_gpu=True
):
    """
    GPU-accelerated conversion of ATAC-seq peak counts to gene activity scores.
    
    Parameters:
    -----------
    atac : AnnData
        ATAC-seq data with peaks as features
    annotation_results : dict or str
        Peak2gene mapping from annotation
    output_dir : str or Path
        Directory to save the gene activity AnnData
    layer : str or None
        Layer to use for counts (default: X)
    aggregation_method : str
        Method for aggregating peaks to genes
    distance_threshold : int or None
        Maximum TSS distance to include peaks (bp)
    weight_threshold : float
        Minimum weight to include a peak
    min_peak_accessibility : float
        Minimum peak accessibility to include
    n_gpu_workers : int or None
        Number of GPU workers (default: 1)
    gpu_batch_size : int or None
        Batch size for GPU processing
    normalize_by : str
        Normalization method
    log_transform : bool
        Apply log1p transformation
    scale_factors : dict or None
        Optional scaling factors per cell type
    verbose : bool
        Print progress messages
    use_gpu : bool
        Whether to use GPU acceleration
    
    Returns:
    --------
    AnnData
        Gene activity matrix saved to output_dir/gene_activity_weighted_gpu.h5ad
    """
    
    if use_gpu and not GPU_AVAILABLE:
        print("GPU requested but CuPy not available. Falling back to CPU.")
        use_gpu = False
    
    if use_gpu:
        # Check available GPUs
        n_gpus = cp.cuda.runtime.getDeviceCount()
        if n_gpus == 0:
            print("No GPUs detected. Falling back to CPU.")
            use_gpu = False
        else:
            if verbose:
                print(f"GPU acceleration enabled. Detected {n_gpus} GPU(s)")
                for i in range(n_gpus):
                    props = cp.cuda.runtime.getDeviceProperties(i)
                    print(f"  GPU {i}: {props['name'].decode()} ({props['totalGlobalMem'] / 1e9:.1f} GB)")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if n_gpu_workers is None:
        n_gpu_workers = min(4, n_gpus) if use_gpu else mp.cpu_count()
    
    if verbose:
        mode = "GPU" if use_gpu else "CPU"
        print(f"Creating gene activity matrix using {mode} with {n_gpu_workers} workers")
        print(f"Aggregation method: {aggregation_method}")
        print(f"Normalization: {normalize_by}")
        print(f"Min peak accessibility: {min_peak_accessibility}")
    
    if isinstance(annotation_results, (str, Path)):
        with open(annotation_results, 'rb') as f:
            peak2gene = pickle.load(f)
    elif isinstance(annotation_results, dict) and 'peak2gene' in annotation_results:
        peak2gene = annotation_results['peak2gene']
    else:
        peak2gene = annotation_results
    
    if layer is not None:
        X = atac.layers[layer]
    else:
        X = atac.X

    if not issparse(X):
        X = csr_matrix(X)
    else:
        X = X.tocsr()
    
    if min_peak_accessibility is not None and min_peak_accessibility > 0:
        peak_means = np.asarray(X.mean(axis=0)).ravel()
        valid_peak_mask = peak_means >= min_peak_accessibility
        valid_peak_names = [peak for i, peak in enumerate(atac.var_names) 
                           if valid_peak_mask[i]]
        if verbose:
            print(f"Accessibility filter: {len(valid_peak_names)}/{len(atac.var_names)} peaks pass")
    else:
        valid_peak_names = list(atac.var_names)
    
    peak_to_idx = {peak: i for i, peak in enumerate(atac.var_names)
                   if peak in valid_peak_names}

    gene2peaks_weighted = defaultdict(list)
    peak_stats = {
        'total_annotated': 0,
        'peaks_in_atac_data': 0,
        'peaks_pass_accessibility': 0,
        'used_after_filtering': 0,
        'filtered_by_accessibility': 0,
        'filtered_by_distance': 0,
        'filtered_by_weight': 0,
        'not_in_atac_data': 0
    }
    
    for peak, annotation in peak2gene.items():
        peak_stats['total_annotated'] += 1
        
        if peak not in atac.var_names:
            peak_stats['not_in_atac_data'] += 1
            continue
        
        peak_stats['peaks_in_atac_data'] += 1
        
        if peak not in valid_peak_names:
            peak_stats['filtered_by_accessibility'] += 1
            continue
        
        peak_stats['peaks_pass_accessibility'] += 1
        
        if not isinstance(annotation, dict):
            continue
            
        gene_ids = annotation.get('gene_ids', [])
        gene_names = annotation.get('gene_names', [])
        weights = annotation.get('weights', [])
        distances = annotation.get('distances', [])
        in_promoter = annotation.get('in_promoter', [])
        in_gene_body = annotation.get('in_gene_body', [])
        tss_weights = annotation.get('tss_weights', [])
        
        peak_used = False
        for i, gene_id in enumerate(gene_ids):
            weight = weights[i] if i < len(weights) else 0
            distance = distances[i] if i < len(distances) else float('inf')
            gene_name = gene_names[i] if i < len(gene_names) else "Unknown"
            
            if weight_threshold is not None and weight < weight_threshold:
                peak_stats['filtered_by_weight'] += 1
                continue
                
            if distance_threshold is not None and distance > distance_threshold:
                peak_stats['filtered_by_distance'] += 1
                continue
            
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
    
    gene_ids = sorted(list(gene2peaks_weighted.keys()))
    gene_ids = [g for g in gene_ids if g and str(g).strip() and str(g).lower() != 'nan']
    n_genes = len(gene_ids)
    n_cells = atac.n_obs
    
    if verbose:
        print(f"\nProcessing {n_genes:,} genes from {atac.n_vars:,} peaks")
        print(f"Using {len(valid_peak_names):,} accessibility-filtered peaks")
    
    if use_gpu:
        if gpu_batch_size is None:
            # 25% of GPU memory, assuming ~100 peaks per gene
            gpu_mem = cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem']
            estimated_batch_size = int((gpu_mem * 0.25) / (n_cells * 8 * 100))
            gpu_batch_size = max(50, min(500, estimated_batch_size))
        
        if verbose:
            print(f"GPU batch size: {gpu_batch_size} genes per batch")
        
        if verbose:
            print("Transferring data to GPU...")

        X_gpu_list = []
        for gpu_id in range(min(n_gpu_workers, n_gpus)):
            with cp.cuda.Device(gpu_id):
                if issparse(X):
                    X_gpu = cusparse.csr_matrix(X)
                else:
                    X_gpu = cp.array(X)
                X_gpu_list.append(X_gpu)
        
        gene_batches = [gene_ids[i:i + gpu_batch_size] for i in range(0, n_genes, gpu_batch_size)]

        process_args = []
        for i, batch in enumerate(gene_batches):
            gpu_id = i % min(n_gpu_workers, n_gpus)
            args = (batch, gene2peaks_weighted, peak_to_idx, X_gpu_list[gpu_id], 
                   aggregation_method, {'sigma': 50000}, gpu_id)
            process_args.append(args)
        
        if verbose:
            print(f"Processing {len(gene_batches)} gene batches on GPU...")

        if n_gpu_workers > 1 and n_gpus > 1:
            with mp.Pool(n_gpu_workers) as pool:
                results = list(tqdm(
                    pool.imap(process_gene_batch_gpu, process_args),
                    total=len(gene_batches),
                    desc="GPU processing",
                    disable=not verbose
                ))
        else:
            results = []
            for args in tqdm(process_args, desc="GPU processing", disable=not verbose):
                results.append(process_gene_batch_gpu(args))
        
        for X_gpu in X_gpu_list:
            del X_gpu
        cp.get_default_memory_pool().free_all_blocks()
        
    else:
        from peak_to_gene_activity_weighted import process_gene_batch
        
        batch_size = max(1, n_genes // (n_gpu_workers * 4))
        gene_batches = [gene_ids[i:i + batch_size] for i in range(0, n_genes, batch_size)]
        
        process_args = [
            (batch, gene2peaks_weighted, peak_to_idx, X, aggregation_method, {'sigma': 50000})
            for batch in gene_batches
        ]
        
        with mp.Pool(n_gpu_workers) as pool:
            results = list(tqdm(
                pool.imap(process_gene_batch, process_args),
                total=len(gene_batches),
                desc="CPU processing",
                disable=not verbose
            ))
    
    if verbose:
        print("Combining results...")

    activity_matrices = [r[0] for r in results]
    gene_stats_lists = [r[1] for r in results]

    gene_activity = activity_matrices[0]
    for mat in activity_matrices[1:]:
        gene_activity = csr_matrix(np.hstack([gene_activity.toarray(), mat.toarray()]))

    all_gene_stats = []
    for stats_list in gene_stats_lists:
        all_gene_stats.extend(stats_list)
    
    gene_stats_df = pd.DataFrame(all_gene_stats).set_index('gene_id')
    
    if normalize_by != 'none' and use_gpu:
        if verbose:
            print(f"Applying {normalize_by} normalization on GPU...")
        
        with cp.cuda.Device(0):
            gene_activity_gpu = cp.array(gene_activity.toarray(), dtype=cp.float32)
            
            if normalize_by == 'n_peaks':
                for i, gene_id in enumerate(gene_ids):
                    n_peaks = gene_stats_df.loc[gene_id, 'n_peaks']
                    if n_peaks > 0:
                        gene_activity_gpu[:, i] /= n_peaks
                        
            elif normalize_by == 'total_weight':
                for i, gene_id in enumerate(gene_ids):
                    total_weight = gene_stats_df.loc[gene_id, 'total_weight']
                    if total_weight > 0:
                        gene_activity_gpu[:, i] /= total_weight
                        
            elif normalize_by == 'archR':
                gene_activity_gpu = cp.log2(gene_activity_gpu + 1)
            
            gene_activity = csr_matrix(cp.asnumpy(gene_activity_gpu))
            del gene_activity_gpu
            cp.get_default_memory_pool().free_all_blocks()
    
    elif normalize_by != 'none':
        if normalize_by == 'n_peaks':
            for i, gene_id in enumerate(gene_ids):
                n_peaks = gene_stats_df.loc[gene_id, 'n_peaks']
                if n_peaks > 0:
                    gene_activity[:, i] = gene_activity[:, i] / n_peaks
                    
        elif normalize_by == 'total_weight':
            for i, gene_id in enumerate(gene_ids):
                total_weight = gene_stats_df.loc[gene_id, 'total_weight']
                if total_weight > 0:
                    gene_activity[:, i] = gene_activity[:, i] / total_weight
                    
        elif normalize_by == 'archR':
            gene_activity = csr_matrix(np.log2(gene_activity.toarray() + 1))
    
    # Apply log transformation if requested
    if log_transform and normalize_by != 'archR':
        if verbose:
            print("Applying log1p transformation...")
        gene_activity = csr_matrix(np.log1p(gene_activity.toarray()))
    
    adata_gene = ad.AnnData(
        X=gene_activity,
        obs=atac.obs.copy(),
        var=gene_stats_df.loc[gene_ids].copy()
    )
    
    adata_gene.var_names = gene_ids
    adata_gene.var_names.name = 'gene_id'
    
    if 'gene_name' not in adata_gene.var.columns:
        adata_gene.var['gene_name'] = adata_gene.var.index
    
    for key in ['sample_name', 'genome', 'species']:
        if key in atac.uns:
            adata_gene.uns[key] = atac.uns[key]
    
    adata_gene.uns['gene_activity_params'] = {
        'method': 'gpu_accelerated_weighted_aggregation' if use_gpu else 'weighted_aggregation',
        'aggregation': aggregation_method,
        'normalization': normalize_by,
        'source_peaks': atac.n_vars,
        'target_genes': n_genes,
        'distance_threshold': distance_threshold,
        'weight_threshold': weight_threshold,
        'min_peak_accessibility': min_peak_accessibility,
        'log_transformed': log_transform or normalize_by == 'archR',
        'n_workers': n_gpu_workers,
        'gpu_used': use_gpu,
        'gpu_batch_size': gpu_batch_size if use_gpu else None,
        'filtering_stats': peak_stats,
        'identifier_type': 'gene_id'
    }
    
    suffix = '_gpu' if use_gpu else ''
    output_path = output_dir / f'gene_activity_weighted{suffix}.h5ad'
    adata_gene.write(output_path)
    
    if verbose:
        print(f"\nGene activity matrix created: {n_cells:,} cells × {n_genes:,} genes")
        print(f"Results saved to: {output_path}")
        
        non_zero_genes = (adata_gene.X.sum(axis=0) > 0).A1.sum()
        sparsity = 1 - (adata_gene.X.nnz / (n_cells * n_genes))
        
        print(f"\nMatrix statistics:")
        print(f"  Non-zero genes: {non_zero_genes:,}/{n_genes:,} ({100*non_zero_genes/n_genes:.1f}%)")
        print(f"  Sparsity: {100*sparsity:.1f}%")
        print(f"  Total counts: {adata_gene.X.sum():,.0f}")
        
        if use_gpu:
            print(f"\nGPU acceleration statistics:")
            print(f"  GPUs used: {min(n_gpu_workers, n_gpus)}")
            print(f"  Batch size: {gpu_batch_size} genes")
    
    return adata_gene
