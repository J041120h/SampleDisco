"""Strip an AnnData down to its embedding-only payload.

The SampleDisco pipeline only needs ``obs`` (cell metadata) and ``obsm``
(cell-level embeddings) downstream of GLUE — sample-embedding, cell typing,
clustering, distance, and trajectory all read those exclusively. The heavy
expression / gene-activity ``X`` matrix dominates h5ad file size (typically
10× to 100× larger than obs+obsm) and is only required when running
differential analysis. ``slim_adata_drop_expression`` replaces ``X`` with
an all-zero sparse matrix of the same shape, drops ``layers`` / ``varm`` /
``varp``, and leaves ``obs`` / ``obsm`` / ``obsp`` / ``var`` untouched so
downstream code that introspects sample metadata or var names keeps working.
"""
from __future__ import annotations

import numpy as np
from anndata import AnnData
from scipy import sparse


def slim_adata_drop_expression(adata: AnnData) -> AnnData:
    """In-place: replace ``adata.X`` with a same-shape all-zero CSR and drop
    expression-side mappings (``layers``, ``varm``, ``varp``). Returns the
    same object for chaining."""
    n_obs, n_var = adata.shape
    adata.X = sparse.csr_matrix((n_obs, n_var), dtype=np.float32)
    for attr in ("layers", "varm", "varp"):
        store = getattr(adata, attr)
        for key in list(store.keys()):
            del store[key]
    return adata
