"""SampleDisco — cross-omics, cross-condition sample embedding for single-cell data.

Public API (imported lazily so ``import sampledisco`` stays light and does not
pull scanpy / torch / scGLUE until you actually call into the pipeline):

    import sampledisco
    sampledisco.wrapper(...)                 # full pipeline (RNA / ATAC / multi-omics)
    sampledisco.compute_sample_embedding(...)  # the core method only

The CLI entry point is ``sampledisco --config <yaml>`` (see ``sampledisco.cli``).
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["wrapper", "compute_sample_embedding"]


def __getattr__(name: str):
    # PEP 562 lazy attribute access — defer heavy imports to first use.
    if name == "wrapper":
        from sampledisco.wrapper.wrapper import wrapper
        return wrapper
    if name == "compute_sample_embedding":
        from sampledisco.sample_embedding import compute_sample_embedding
        return compute_sample_embedding
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
