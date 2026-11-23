"""Harmony backend shim for the cell-level Harmony passes.

Uses ``harmony-pytorch`` (a core dependency since 0.6.1, so pip and conda
installs run the same implementation). With ``n_jobs`` fixed to the usable CPU
count it is reproducible run to run. If it cannot be imported, falls back to
``harmonypy`` run single-threaded (``utils.cpus.single_threaded``), since
harmonypy's PyTorch backend is not bit-reproducible with several threads.
"""
from typing import Optional, Sequence, Union
import numpy as np


def harmonypy_cpu_kwargs(hm) -> dict:
    """``{'device': 'cpu'}`` when this harmonypy's ``run_harmony`` takes ``device``
    (>= 0.2, PyTorch backend, which otherwise picks CUDA/MPS whenever torch sees a
    GPU); ``{}`` for older harmonypy, which always runs on the CPU."""
    import inspect
    try:
        params = inspect.signature(hm.run_harmony).parameters
    except (TypeError, ValueError):
        return {}
    return {"device": "cpu"} if "device" in params else {}


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
        from sampledisco.utils.cpus import single_threaded
        with single_threaded():   # reproducible; restores the thread settings after
            ho = hm.run_harmony(
                X, obs, keys, max_iter_harmony=max_iter_harmony, random_state=seed,
                **({} if use_gpu else harmonypy_cpu_kwargs(hm)),
            )
        Z = ho.Z_corr
        if Z.shape[0] != X.shape[0]:  # harmonypy returns (n_dims x n_cells)
            Z = Z.T
        return np.asarray(Z, dtype=np.float32)

    # harmony-pytorch's default n_jobs=-1 sets torch to every physical core of the
    # node, ignoring the SLURM allocation; give it the usable CPU count instead.
    import inspect
    from sampledisco.utils.cpus import usable_cpu_count
    try:
        takes_n_jobs = "n_jobs" in inspect.signature(harmonize).parameters
    except (TypeError, ValueError):
        takes_n_jobs = False
    Z = harmonize(
        X, obs, batch_key=keys, max_iter_harmony=max_iter_harmony, use_gpu=use_gpu,
        random_state=seed, **({"n_jobs": usable_cpu_count()} if takes_n_jobs else {}),
    )
    return np.asarray(Z, dtype=np.float32)
