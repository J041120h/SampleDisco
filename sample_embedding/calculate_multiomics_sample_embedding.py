#!/usr/bin/env python3
"""
Multi-omics sample embedding calculation wrapper.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_embedding.DR import dimension_reduction
from sample_embedding.multi_omics_pseudobulk_cpu import compute_pseudobulk_adata
from utils.random_seed import set_global_seed


def calculate_multiomics_sample_embedding(
    adata: sc.AnnData,
    sample_col: str = 'sample',
    celltype_col: str = 'cell_type',
    batch_col: Optional[Union[str, List[str]]] = None,
    output_dir: str = './',
    sample_hvg_number: int = 2000,
    n_expression_components: int = 10,
    n_proportion_components: int = 10,
    harmony_for_proportion: bool = True,
    preserve_cols_in_sample_embedding: Optional[Union[str, List[str]]] = None,
    use_gpu: bool = False,
    atac: bool = False,
    save: bool = True,
    verbose: bool = True,
    hvg_modality: str = 'RNA',
    modality_col: str = 'modality',
) -> Tuple[Dict, sc.AnnData]:
    set_global_seed(seed=42)

    output_dir = os.path.abspath(output_dir)
    os.makedirs(os.path.join(output_dir, "pseudobulk"), exist_ok=True)

    if use_gpu:
        from sample_embedding.multi_omics_pseudobulk_gpu import compute_pseudobulk_adata_linux
        compute_pseudobulk_func = compute_pseudobulk_adata_linux
    else:
        compute_pseudobulk_func = compute_pseudobulk_adata

    pseudobulk_result_dict, pseudobulk_adata = compute_pseudobulk_func(
        adata=adata,
        batch_col=batch_col,
        sample_col=sample_col,
        celltype_col=celltype_col,
        output_dir=output_dir,
        save=save,
        n_features=sample_hvg_number,
        atac=atac,
        verbose=verbose,
        preserve_cols=preserve_cols_in_sample_embedding,
        hvg_modality=hvg_modality,
        modality_col=modality_col,
    )

    sample_embedding_adata = dimension_reduction(
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

    return pseudobulk_result_dict, sample_embedding_adata

