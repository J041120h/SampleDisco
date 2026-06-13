"""Drop the expression payload from an integrated AnnData.

After GLUE, downstream modules only read ``obs`` and ``obsm``; ``X`` and
associated layers dominate file size (10–100×). Keeping ``var`` intact
lets code that inspects gene/peak names keep working.
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
