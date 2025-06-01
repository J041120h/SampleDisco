import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix, lil_matrix
from collections import defaultdict
import anndata as ad

def peak_to_gene_activity(adata, peak2gene_key='peak2gene', layer=None, normalize=True):
    """
    Convert ATAC-seq peak counts to gene activity scores.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix with peaks as features and cells as observations
    peak2gene_key : str
        Key in adata.uns containing peak-to-gene mapping
    layer : str or None
        If specified, use counts from this layer. Otherwise use adata.X
    normalize : bool
        Whether to normalize by the number of peaks per gene
    
    Returns:
    --------
    AnnData
        New AnnData object with genes as features and gene activity scores
    """
    
    # Get peak-to-gene mapping
    if peak2gene_key not in adata.uns:
        raise ValueError(f"Peak-to-gene mapping not found in adata.uns['{peak2gene_key}']")
    
    peak2gene = adata.uns[peak2gene_key]
    
    # Get count matrix
    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X
    
    # Convert to sparse matrix if needed
    if not hasattr(X, 'tocsr'):
        X = csr_matrix(X)
    else:
        X = X.tocsr()
    
    # Create gene-to-peaks mapping
    gene2peaks = defaultdict(list)
    
    # Handle different formats of peak2gene mapping
    if isinstance(peak2gene, dict):
        # Format: {peak: gene} or {peak: [genes]}
        for peak, genes in peak2gene.items():
            if isinstance(genes, (list, tuple)):
                for gene in genes:
                    gene2peaks[gene].append(peak)
            else:
                gene2peaks[genes].append(peak)
    
    elif isinstance(peak2gene, pd.DataFrame):
        # Format: DataFrame with 'peak' and 'gene' columns
        if 'peak' in peak2gene.columns and 'gene' in peak2gene.columns:
            for _, row in peak2gene.iterrows():
                gene2peaks[row['gene']].append(row['peak'])
        else:
            raise ValueError("DataFrame must have 'peak' and 'gene' columns")
    
    else:
        raise ValueError("peak2gene must be a dict or DataFrame")
    
    # Get all unique genes
    genes = sorted(list(gene2peaks.keys()))
    n_genes = len(genes)
    n_cells = adata.n_obs
    
    # Create mapping from peak names to indices
    peak_to_idx = {peak: i for i, peak in enumerate(adata.var_names)}
    
    # Initialize gene activity matrix
    gene_activity = lil_matrix((n_cells, n_genes))
    
    # Calculate gene activity scores
    print(f"Converting {adata.n_vars} peaks to {n_genes} gene activity scores...")
    
    for gene_idx, gene in enumerate(genes):
        if gene_idx % 1000 == 0:
            print(f"Processing gene {gene_idx}/{n_genes}...")
        
        # Get peaks associated with this gene
        peaks = gene2peaks[gene]
        
        # Get indices of these peaks
        peak_indices = []
        for peak in peaks:
            if peak in peak_to_idx:
                peak_indices.append(peak_to_idx[peak])
        
        if len(peak_indices) > 0:
            # Sum counts across all peaks for this gene
            gene_counts = X[:, peak_indices].sum(axis=1).A.flatten()
            
            # Normalize by number of peaks if requested
            if normalize and len(peak_indices) > 1:
                gene_counts = gene_counts / len(peak_indices)
            
            gene_activity[:, gene_idx] = gene_counts.reshape(-1, 1)
    
    # Convert to CSR format for efficiency
    gene_activity = gene_activity.tocsr()
    
    # Create new AnnData object
    adata_gene = ad.AnnData(
        X=gene_activity,
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=genes)
    )
    
    # Add gene metadata if available
    if 'gene_info' in adata.uns:
        gene_info = adata.uns['gene_info']
        if isinstance(gene_info, pd.DataFrame) and 'gene' in gene_info.columns:
            # Merge gene info
            adata_gene.var = adata_gene.var.merge(
                gene_info.set_index('gene'), 
                left_index=True, 
                right_index=True, 
                how='left'
            )
    
    # Add peak count per gene
    peak_counts = pd.Series({gene: len(gene2peaks[gene]) for gene in genes})
    adata_gene.var['n_peaks'] = peak_counts
    
    # Copy relevant uns data
    for key in ['sample_name', 'genome', 'species']:
        if key in adata.uns:
            adata_gene.uns[key] = adata.uns[key]
    
    print(f"Gene activity matrix created: {n_cells} cells × {n_genes} genes")
    
    return adata_gene
