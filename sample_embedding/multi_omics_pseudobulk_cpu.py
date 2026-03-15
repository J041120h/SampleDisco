#!/usr/bin/env python3
"""
CPU wrapper for multi-omics pseudobulk computation.
"""

from typing import Dict, List, Optional, Tuple, Union

import scanpy as sc

from sample_embedding.pseudo_adata import compute_pseudobulk_adata


def compute_pseudobulk_adata_cpu(
    adata: sc.AnnData,
    batch_col: Optional[Union[str, List[str]]] = None,
    sample_col: str = 'sample',
    celltype_col: str = 'cell_type',
    output_dir: str = './',
    save: bool = True,
    n_features: int = 2000,
    sample_hvg_number: Optional[int] = None,
    atac: bool = False,
    verbose: bool = False,
    combat_timeout: float = 1800.0,
    preserve_cols: Optional[Union[str, List[str]]] = None,
    hvg_modality: str = 'RNA',
    modality_col: str = 'modality',
) -> Tuple[Dict, sc.AnnData]:
    # hvg_modality/modality_col are accepted for API parity with GPU path.
    return compute_pseudobulk_adata(
        adata=adata,
        batch_col=batch_col,
        sample_col=sample_col,
        celltype_col=celltype_col,
        output_dir=output_dir,
        save=save,
        n_features=sample_hvg_number if sample_hvg_number is not None else n_features,
        atac=atac,
        verbose=verbose,
        combat_timeout=combat_timeout,
        preserve_cols_in_sample_embedding=preserve_cols,
    )

