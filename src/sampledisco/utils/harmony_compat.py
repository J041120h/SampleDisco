"""Harmony backend shim so PyTorch stays an optional dependency.

Prefers ``harmony-pytorch`` (fast, GPU-capable) when it — and therefore torch —
is importable; otherwise falls back to the pure-NumPy ``harmonypy``. Behaviour
with torch installed is identical to calling ``harmony.harmonize`` directly, so
existing (torch) environments are unaffected. A core ``pip install sampledisco``
runs Harmony on CPU via harmonypy; ``pip install sampledisco[multiomics]`` pulls
torch and the faster harmony-pytorch is used automatically.
"""
from typing import Optional, Sequence, Union
import numpy as np


def harmonize_embedding(
    embedding: np.ndarray,
    obs,
    batch_key: Optional[Union[str, Sequence[str]]] = None,
    max_iter_harmony: int = 30,
    use_gpu: bool = False,
    seed: int = 42,
) -> np.ndarray:
    """Batch-correct an (n_cells x n_dims) embedding with Harmony.

    Returns a float32 array of the same shape. If ``batch_key`` is empty/None the
    embedding is returned unchanged (no correction).
    """
    X = np.asarray(embedding)
    keys = [batch_key] if isinstance(batch_key, str) else list(batch_key or [])
    if not keys:
        return X.astype(np.float32, copy=False)

    try:
        from harmony import harmonize  # harmony-pytorch (needs torch)
    except ImportError:
        import harmonypy as hm
        ho = hm.run_harmony(
            X, obs, keys, max_iter_harmony=max_iter_harmony, random_state=seed,
        )
        Z = ho.Z_corr
        if Z.shape[0] != X.shape[0]:  # harmonypy returns (n_dims x n_cells)
            Z = Z.T
        return np.asarray(Z, dtype=np.float32)

    Z = harmonize(
        X, obs, batch_key=keys, max_iter_harmony=max_iter_harmony, use_gpu=use_gpu,
        random_state=seed,
    )
    return np.asarray(Z, dtype=np.float32)
