import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix, lil_matrix, issparse
from collections import defaultdict
import anndata as ad
import os
from pathlib import Path

def brief_gene_activity_overview(adata):
    """
    Brief overview of gene activity AnnData object.
    
    Parameters:
    -----------
    adata : AnnData
        Gene activity AnnData object
    """
    
    print(f"Gene Activity Data Overview")
    print(f"=" * 30)
    print(f"Shape: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
    print(f"Matrix type: {'Sparse' if issparse(adata.X) else 'Dense'}")
    
    # Count stats
    if issparse(adata.X):
        total_counts = adata.X.sum()
        non_zero_pct = (adata.X.nnz / adata.X.size) * 100
    else:
        total_counts = adata.X.sum()
        non_zero_pct = (np.count_nonzero(adata.X) / adata.X.size) * 100
    
    print(f"Total counts: {total_counts:,.0f}")
    print(f"Non-zero values: {non_zero_pct:.1f}%")
    
    # Metadata
    print(f"Cell metadata columns: {adata.obs.shape[1]}")
    print(f"Gene metadata columns: {adata.var.shape[1]}")
    
    # Gene activity specific
    if 'n_peaks' in adata.var.columns:
        peak_counts = adata.var['n_peaks']
        print(f"Peaks per gene: {peak_counts.mean():.1f} (range: {peak_counts.min()}-{peak_counts.max()})")
    
    if 'peak_to_gene_conversion' in adata.uns:
        method = adata.uns['peak_to_gene_conversion'].get('method', 'unknown')
        normalization = adata.uns['peak_to_gene_conversion'].get('normalization', 'unknown')
        print(f"Conversion: {method}, normalized by {normalization}")
    
    # Count matrix example (3 genes × 3 cells)
    print(f"\nCount matrix (first 3 genes × 3 cells):")
    if issparse(adata.X):
        sample_matrix = adata.X[:3, :3].toarray().T  # Transpose to get genes × cells
    else:
        sample_matrix = adata.X[:3, :3].T
    
    sample_df = pd.DataFrame(
        sample_matrix,
        index=adata.var_names[:3],  # genes as rows
        columns=adata.obs_names[:3]  # cells as columns
    )
    print(sample_df.round(4))

def peak_to_gene_activity(atac, output_dir, peak2gene_key='peak2gene', layer=None, 
                         distance_threshold=None, use_closest_only=False, verbose=False):
    """
    Convert ATAC-seq peak counts to gene activity scores with mandatory normalization.
    Optimized for annotation format from annotate_atac_peaks function.
    
    Parameters:
    -----------
    atac : AnnData
        Annotated data matrix with peaks as features and cells as observations
    output_dir : str or Path
        Directory path where the gene activity AnnData object will be saved
    peak2gene_key : str
        Key path in atac.uns containing peak-to-gene mapping 
        (default: 'peak2gene' looks for atac.uns['atac']['peak2gene'])
    layer : str or None
        If specified, use counts from this layer. Otherwise use atac.X
    distance_threshold : int or None
        Maximum distance from TSS to include peaks (in bp). If None, use all annotated peaks
    use_closest_only : bool
        If True, only use the closest gene per peak. If False, use all genes within threshold
    verbose : bool
        Whether to print progress messages
    
    Returns:
    --------
    AnnData
        New AnnData object with genes as features and gene activity scores
        (also saved to output_dir/gene_activity.h5ad)
    
    Notes:
    ------
    Normalization by number of peaks per gene is applied by default as it's
    standard practice in ATAC-seq analysis to account for varying numbers
    of regulatory elements per gene.
    """
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get peak-to-gene mapping with flexible key handling
    peak2gene = None
    
    # Handle nested structure from annotate_atac_peaks
    if 'atac' in atac.uns and peak2gene_key in atac.uns['atac']:
        peak2gene = atac.uns['atac'][peak2gene_key]
        if verbose:
            print(f"Using peak-to-gene mapping from atac.uns['atac']['{peak2gene_key}']")
    elif peak2gene_key in atac.uns:
        peak2gene = atac.uns[peak2gene_key]
        if verbose:
            print(f"Using peak-to-gene mapping from atac.uns['{peak2gene_key}']")
    else:
        raise ValueError(f"Peak-to-gene mapping not found. Checked atac.uns['atac']['{peak2gene_key}'] and atac.uns['{peak2gene_key}']")
    
    # Get count matrix
    if layer is not None:
        X = atac.layers[layer]
        if verbose:
            print(f"Using counts from layer '{layer}'")
    else:
        X = atac.X
        if verbose:
            print("Using counts from atac.X")
    
    # Convert to sparse matrix if needed
    if not hasattr(X, 'tocsr'):
        X = csr_matrix(X)
    else:
        X = X.tocsr()
    
    # Create gene-to-peaks mapping with distance filtering
    gene2peaks = defaultdict(list)
    peak_stats = {'total_annotated': 0, 'used_after_filtering': 0, 'filtered_by_distance': 0}
    
    # Handle the annotation format from annotate_atac_peaks
    if isinstance(peak2gene, dict):
        for peak, annotation in peak2gene.items():
            peak_stats['total_annotated'] += 1
            
            if isinstance(annotation, dict):
                # Format from annotate_atac_peaks: {peak: {'genes': [...], 'distances': [...], ...}}
                genes = annotation.get('genes', [])
                distances = annotation.get('distances', [])
                
                if use_closest_only:
                    # Use only the closest gene
                    closest_gene = annotation.get('closest_gene')
                    if closest_gene:
                        closest_distance = annotation.get('closest_distance', 0)
                        if distance_threshold is None or closest_distance <= distance_threshold:
                            gene2peaks[closest_gene].append(peak)
                            peak_stats['used_after_filtering'] += 1
                        else:
                            peak_stats['filtered_by_distance'] += 1
                else:
                    # Use all genes within distance threshold
                    genes_used = False
                    for gene, distance in zip(genes, distances):
                        if distance_threshold is None or distance <= distance_threshold:
                            gene2peaks[gene].append(peak)
                            genes_used = True
                        else:
                            peak_stats['filtered_by_distance'] += 1
                    
                    if genes_used:
                        peak_stats['used_after_filtering'] += 1
                        
            elif isinstance(annotation, (list, tuple)):
                # Simple format: {peak: [genes]} or {peak: gene}
                genes = annotation if isinstance(annotation, (list, tuple)) else [annotation]
                for gene in genes:
                    gene2peaks[gene].append(peak)
                peak_stats['used_after_filtering'] += 1
            else:
                # Single gene format
                gene2peaks[annotation].append(peak)
                peak_stats['used_after_filtering'] += 1
    
    elif isinstance(peak2gene, pd.DataFrame):
        # Handle DataFrame format
        peak_stats['total_annotated'] = len(peak2gene)
        
        if distance_threshold is not None and 'distance' in peak2gene.columns:
            filtered_df = peak2gene[peak2gene['distance'] <= distance_threshold]
            peak_stats['filtered_by_distance'] = len(peak2gene) - len(filtered_df)
        else:
            filtered_df = peak2gene
        
        peak_stats['used_after_filtering'] = len(filtered_df)
        
        if 'peak' in filtered_df.columns and 'gene' in filtered_df.columns:
            for _, row in filtered_df.iterrows():
                gene2peaks[row['gene']].append(row['peak'])
        else:
            raise ValueError("DataFrame must have 'peak' and 'gene' columns")
    
    else:
        raise ValueError("peak2gene must be a dict or DataFrame")
    
    # Get all unique genes and filter out invalid ones
    genes = sorted(list(gene2peaks.keys()))
    original_count = len(genes)
    
    # Debug: check for problematic gene names
    if verbose:
        print(f"Original gene count: {original_count}")
        print(f"First 10 genes: {genes[:10]}")
        empty_genes = [i for i, g in enumerate(genes) if not g or not str(g).strip()]
        if empty_genes:
            print(f"Found empty genes at indices: {empty_genes}")
    
    # Remove empty, whitespace-only, or NaN gene names
    valid_genes = []
    for g in genes:
        if g and str(g).strip() and str(g).lower() != 'nan' and str(g) != 'None':
            valid_genes.append(g)
        elif verbose:
            print(f"Filtering out invalid gene: '{g}' (type: {type(g)})")
    
    genes = sorted(valid_genes)
    
    if len(genes) < original_count:
        filtered_count = original_count - len(genes)
        print(f"Filtered out {filtered_count} invalid gene names")
        
        # Update gene2peaks to only include valid genes
        gene2peaks = {g: peaks for g, peaks in gene2peaks.items() 
                     if g and str(g).strip() and str(g).lower() != 'nan' and str(g) != 'None'}
    
    n_genes = len(genes)
    n_cells = atac.n_obs
    
    if verbose:
        print(f"Peak annotation statistics:")
        print(f"  Total annotated peaks: {peak_stats['total_annotated']:,}")
        print(f"  Used after filtering: {peak_stats['used_after_filtering']:,}")
        if distance_threshold:
            print(f"  Filtered by distance (>{distance_threshold:,} bp): {peak_stats['filtered_by_distance']:,}")
        if use_closest_only:
            print(f"  Using closest gene only per peak")
        else:
            print(f"  Using all genes within threshold per peak")
        
        print(f"Found {len(gene2peaks)} unique genes associated with peaks")
        peak_counts = [len(peaks) for peaks in gene2peaks.values()]
        print(f"Peak counts per gene - Mean: {np.mean(peak_counts):.1f}, "
              f"Median: {np.median(peak_counts):.1f}, "
              f"Range: {min(peak_counts)}-{max(peak_counts)}")
    
    # Create mapping from peak names to indices
    peak_to_idx = {peak: i for i, peak in enumerate(atac.var_names)}
    
    # Initialize gene activity matrix
    gene_activity = lil_matrix((n_cells, n_genes))
    
    # Calculate gene activity scores
    if verbose:
        print(f"Converting {atac.n_vars} peaks to {n_genes} gene activity scores...")
        print("Applying normalization by number of peaks per gene (standard practice)")
    
    genes_with_no_peaks = 0
    total_missing_peaks = 0
    
    for gene_idx, gene in enumerate(genes):
        if verbose and gene_idx % 1000 == 0:
            print(f"Processing gene {gene_idx + 1}/{n_genes}...")
        
        # Get peaks associated with this gene
        peaks = gene2peaks[gene]
        
        # Get indices of these peaks
        peak_indices = []
        missing_peaks = 0
        for peak in peaks:
            if peak in peak_to_idx:
                peak_indices.append(peak_to_idx[peak])
            else:
                missing_peaks += 1
        
        total_missing_peaks += missing_peaks
        
        if len(peak_indices) > 0:
            # Sum counts across all peaks for this gene
            gene_counts = X[:, peak_indices].sum(axis=1).A.flatten()
            
            # Mandatory normalization by number of peaks
            # This is standard practice to account for genes with different
            # numbers of regulatory elements
            gene_counts = gene_counts / len(peak_indices)
            
            gene_activity[:, gene_idx] = gene_counts.reshape(-1, 1)
        else:
            genes_with_no_peaks += 1
    
    if verbose:
        if genes_with_no_peaks > 0:
            print(f"Warning: {genes_with_no_peaks} genes had no matching peaks in the dataset")
        if total_missing_peaks > 0:
            print(f"Warning: {total_missing_peaks} annotated peaks were not found in the data")
    
    # Convert to CSR format for efficiency
    gene_activity = gene_activity.tocsr()
    
    # Create new AnnData object
    adata_gene = ad.AnnData(
        X=gene_activity,
        obs=atac.obs.copy(),
        var=pd.DataFrame(index=genes)
    )
    
    # Add gene metadata if available
    if 'gene_info' in atac.uns:
        gene_info = atac.uns['gene_info']
        if isinstance(gene_info, pd.DataFrame) and 'gene' in gene_info.columns:
            # Merge gene info
            adata_gene.var = adata_gene.var.merge(
                gene_info.set_index('gene'), 
                left_index=True, 
                right_index=True, 
                how='left'
            )
            if verbose:
                print("Added gene metadata from atac.uns['gene_info']")
    
    # Add peak count per gene and normalization info
    peak_counts = pd.Series({gene: len(gene2peaks[gene]) for gene in genes})
    adata_gene.var['n_peaks'] = peak_counts
    adata_gene.var['normalized_by_n_peaks'] = True  # Flag indicating normalization was applied
    
    # Add distance statistics if available
    if distance_threshold is not None or use_closest_only:
        gene_distances = {}
        for gene in genes:
            distances = []
            for peak in gene2peaks[gene]:
                if peak in peak2gene and isinstance(peak2gene[peak], dict):
                    peak_annotation = peak2gene[peak]
                    if 'genes' in peak_annotation and 'distances' in peak_annotation:
                        gene_idx_in_peak = None
                        try:
                            gene_idx_in_peak = peak_annotation['genes'].index(gene)
                            distances.append(peak_annotation['distances'][gene_idx_in_peak])
                        except (ValueError, IndexError):
                            continue
            
            if distances:
                gene_distances[gene] = {
                    'mean_distance': np.mean(distances),
                    'min_distance': min(distances),
                    'max_distance': max(distances)
                }
        
        if gene_distances:
            distance_df = pd.DataFrame.from_dict(gene_distances, orient='index')
            adata_gene.var = adata_gene.var.merge(distance_df, left_index=True, right_index=True, how='left')
    
    # Copy relevant uns data
    for key in ['sample_name', 'genome', 'species']:
        if key in atac.uns:
            adata_gene.uns[key] = atac.uns[key]
    
    # Copy ATAC annotation stats if available
    if 'atac' in atac.uns and 'annotation_stats' in atac.uns['atac']:
        adata_gene.uns['atac_annotation_stats'] = atac.uns['atac']['annotation_stats']
    
    # Add processing metadata
    adata_gene.uns['peak_to_gene_conversion'] = {
        'method': 'sum_and_normalize',
        'normalization': 'by_n_peaks',
        'source_peaks': atac.n_vars,
        'target_genes': n_genes,
        'peak2gene_key': peak2gene_key,
        'layer_used': layer if layer is not None else 'X',
        'distance_threshold': distance_threshold,
        'use_closest_only': use_closest_only,
        'annotation_stats': peak_stats,
        'filtered_invalid_genes': original_count - len(genes)
    }
    
    # Save to output directory
    output_path = output_dir / 'gene_activity.h5ad'
    adata_gene.write(output_path)
    
    if verbose:
        print(f"Gene activity matrix created: {n_cells} cells × {n_genes} genes")
        print(f"Results saved to: {output_path}")
        
        # Print summary statistics
        non_zero_genes = (adata_gene.X.sum(axis=0) > 0).A1.sum()
        print(f"Summary: {non_zero_genes}/{n_genes} genes have non-zero activity")
        
        # Print distance statistics if available
        if 'mean_distance' in adata_gene.var.columns:
            mean_distances = adata_gene.var['mean_distance'].dropna()
            if len(mean_distances) > 0:
                print(f"Distance stats - Mean: {mean_distances.mean():.0f} bp, "
                      f"Median: {mean_distances.median():.0f} bp")
    
    return adata_gene

if __name__ == "__main__":
    atac = ad.read_h5ad("/Users/harry/Desktop/GenoDistance/Data/test_ATAC.h5ad")
    output_dir = "/Users/harry/Desktop/GenoDistance/result/gene_activity"
    adata = peak_to_gene_activity(atac, output_dir, peak2gene_key='peak2gene', layer=None, verbose=True)
    # adata = ad.read_h5ad("/Users/harry/Desktop/GenoDistance/result/gene_activity/gene_activity.h5ad")
    # adata = ad.read_h5ad("/Users/harry/Desktop/GenoDistance/Data/count_data.h5ad")
    brief_gene_activity_overview(adata)