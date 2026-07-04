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
    rmd_emb_key: Optional[str] = None,
    modality_col: Optional[str] = None,
    batch_col: Optional[Union[str, List[str]]] = None,
    medium_K: int = 120,
    fine_K: int = 300,
    rmd_dim_per_cluster: int = 8,
    use_clr: bool = False,
    use_rmd: bool = True,
    block_weights: Optional[List[float]] = None,
    rmd_weight: float = 0.60,
    pca_components: int = 10,
    batch_method: str = "harmony",
    save: bool = True,
    verbose: bool = True,
    seed: int = 42,
) -> AnnData:
    """Dispatch to CPU or GPU implementation.

    ``use_gpu=True`` on a machine without the RAPIDS stack (e.g. macOS, or any
    CPU-only box) falls back cleanly to the CPU implementation instead of
    crashing on the import.
    """
    if use_gpu:
        try:
            from sampledisco.sample_embedding.sample_embedding_gpu import (
                compute_sample_embedding as _impl,
            )
        except ImportError as e:
            print(
                f"[sampledisco] GPU sample embedding unavailable ({e}); "
                "falling back to the CPU implementation."
            )
            from sampledisco.sample_embedding.sample_embedding import (
                compute_sample_embedding as _impl,
            )
    else:
        from sampledisco.sample_embedding.sample_embedding import (
            compute_sample_embedding as _impl,
        )
    return _impl(
        adata, output_dir,
        sample_col=sample_col,
        celltype_col=celltype_col,
        cluster_emb_key=cluster_emb_key,
        rmd_emb_key=rmd_emb_key,
        modality_col=modality_col,
        batch_col=batch_col,
        medium_K=medium_K,
        fine_K=fine_K,
        rmd_dim_per_cluster=rmd_dim_per_cluster,
        use_clr=use_clr,
        use_rmd=use_rmd,
        block_weights=block_weights,
        rmd_weight=rmd_weight,
        pca_components=pca_components,
        batch_method=batch_method,
        save=save,
        verbose=verbose,
        seed=seed,
    )


__all__ = ["compute_sample_embedding"]
