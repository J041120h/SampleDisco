#!/usr/bin/env python3
"""
Sample embedding calculation wrapper.
"""

import os
import sys

# Add parent directory to path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Dict, List, Optional, Tuple, Union
import scanpy as sc
from sample_embedding.DR import dimension_reduction
from utils.random_seed import set_global_seed


def calculate_sample_embedding(
    adata: sc.AnnData,
    sample_col: str = 'sample',
    celltype_col: str = 'cell_type',
    batch_col: Optional[Union[str, List[str]]] = None,
    output_dir: str = './',
    sample_hvg_number: int = 2000,
    combat_timeout: Optional[float] = None,
    n_expression_components: int = 10,
    n_proportion_components: int = 10,
    harmony_for_proportion: bool = True,
    preserve_cols_in_sample_embedding: Optional[Union[str, List[str]]] = None,
    use_gpu: bool = False,
    atac: bool = False,
    save: bool = True,
    verbose: bool = True,
) -> Tuple[Dict, sc.AnnData]:
    """
    Calculate sample-level embeddings from single-cell data.
    
    Parameters
    ----------
    adata : sc.AnnData
        Input single-cell AnnData object
    sample_col : str
        Column name for sample identifiers
    celltype_col : str
        Column name for cell type labels
    batch_col : str or list, optional
        Column(s) for batch correction
    output_dir : str
        Output directory for results
    sample_hvg_number : int
        Number of highly variable genes per cell type
    combat_timeout : float, optional
        Timeout for ComBat in seconds (default: 20s for GPU, 1800s for CPU)
    n_expression_components : int
        Number of components for expression dimension reduction
    n_proportion_components : int
        Number of components for proportion dimension reduction
    harmony_for_proportion : bool
        Whether to apply Harmony to proportion embeddings
    preserve_cols : str or list, optional
        Columns to preserve during batch correction
    use_gpu : bool
        Whether to use GPU acceleration
    atac : bool
        Whether data is ATAC-seq (uses TF-IDF/LSI instead of normalization/PCA)
    save : bool
        Whether to save results to disk
    verbose : bool
        Whether to print progress messages
    
    Returns
    -------
    pseudobulk_result_dict : dict
        Dictionary containing 'cell_expression_corrected' and 'cell_proportion'
    sample_embedding_adata : sc.AnnData
        AnnData with sample embeddings in .uns and .obsm
    """
    set_global_seed(seed=42)
    
    output_dir = os.path.abspath(output_dir)
    os.makedirs(os.path.join(output_dir, "pseudobulk"), exist_ok=True)

    if use_gpu:
        from sample_embedding.pseudo_adata_linux import compute_pseudobulk_adata_linux
        
        pseudobulk_result_dict, pseudobulk_adata = compute_pseudobulk_adata_linux(
            adata=adata,
            batch_col=batch_col,
            sample_col=sample_col,
            celltype_col=celltype_col,
            output_dir=output_dir,
            save=save,
            sample_hvg_number=sample_hvg_number,
            atac=atac,
            verbose=verbose,
            preserve_cols_in_sample_embedding=preserve_cols_in_sample_embedding,
        )
    else:
        from sample_embedding.pseudo_adata import compute_pseudobulk_adata
        
        pseudobulk_result_dict, pseudobulk_adata = compute_pseudobulk_adata(
            adata=adata,
            batch_col=batch_col,
            sample_col=sample_col,
            celltype_col=celltype_col,
            output_dir=output_dir,
            save=save,
            n_features=sample_hvg_number,
            atac=atac,
            verbose=verbose,
            combat_timeout=combat_timeout if combat_timeout is not None else 1800.0,
            preserve_cols_in_sample_embedding=preserve_cols_in_sample_embedding,
        )

    return pseudobulk_result_dict, dimension_reduction(
        adata=adata,
        pseudobulk=pseudobulk_result_dict,
        pseudobulk_anndata=pseudobulk_adata,
        sample_col=sample_col,
        n_expression_components=n_expression_components,
        n_proportion_components=n_proportion_components,
        batch_col=batch_col,
        harmony_for_proportion=harmony_for_proportion,
        preserve_cols_in_sample_embedding=preserve_cols_in_sample_embedding,
        output_dir=output_dir,
        not_save=not save,
        atac=atac,
        verbose=verbose,
    )