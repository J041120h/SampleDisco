"""Sample embedding module.

Public entry: `compute_sample_embedding(adata, output_dir, *, use_gpu=False, ...)`

The CPU implementation lives in `sample_embedding.sample_embedding` and the
GPU implementation in `sample_embedding.sample_embedding_gpu`. The function
exported here dispatches based on `use_gpu`.
"""

from __future__ import annotations

from typing import List, Optional, Union

from anndata import AnnData


def compute_sample_embedding(
    adata: AnnData,
    output_dir: str,
    *,
    use_gpu: bool = False,
    sample_col: str = "sample",
    celltype_col: str = "cell_type",
    cluster_emb_key: str = "Z_clust",
    cmd_emb_key: Optional[str] = None,
    modality_col: Optional[str] = None,
    batch_col: Optional[Union[str, List[str]]] = None,
    medium_K: int = 120,
    fine_K: int = 300,
    cmd_dim_per_cluster: int = 8,
    use_clr: bool = False,
    use_cmd: bool = True,
    block_weights: Optional[List[float]] = None,
    cmd_weight: float = 0.60,
    pca_components: int = 10,
    batch_method: str = "harmony",
    save: bool = True,
    verbose: bool = True,
    seed: int = 42,
) -> AnnData:
    """Dispatch to CPU or GPU implementation."""
    if use_gpu:
        from sample_embedding.sample_embedding_gpu import (
            compute_sample_embedding as _impl,
        )
    else:
        from sample_embedding.sample_embedding import (
            compute_sample_embedding as _impl,
        )
    return _impl(
        adata, output_dir,
        sample_col=sample_col,
        celltype_col=celltype_col,
        cluster_emb_key=cluster_emb_key,
        cmd_emb_key=cmd_emb_key,
        modality_col=modality_col,
        batch_col=batch_col,
        medium_K=medium_K,
        fine_K=fine_K,
        cmd_dim_per_cluster=cmd_dim_per_cluster,
        use_clr=use_clr,
        use_cmd=use_cmd,
        block_weights=block_weights,
        cmd_weight=cmd_weight,
        pca_components=pca_components,
        batch_method=batch_method,
        save=save,
        verbose=verbose,
        seed=seed,
    )


__all__ = ["compute_sample_embedding"]
