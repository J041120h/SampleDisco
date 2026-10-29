"""SampleDisco — cross-omics, cross-condition sample embedding for single-cell data.

Public API (imported lazily so ``import sampledisco`` stays light and does not
pull scanpy / torch / scGLUE until you actually call into the pipeline):

    import sampledisco
    sampledisco.wrapper(...)                 # full pipeline (RNA / ATAC / multi-omics)
    sampledisco.compute_sample_embedding(...)  # the core method only
    sampledisco.predict_phenotype(...)       # cross-validated phenotype prediction from the PCs
    sampledisco.transfer_labels(...)         # kNN label transfer in the sample embedding

The CLI entry point is ``sampledisco --config <yaml>`` (see ``sampledisco.cli``).
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sampledisco")
except PackageNotFoundError:  # running from a source tree that isn't installed
    __version__ = "0.0.0+unknown"

__all__ = ["wrapper", "compute_sample_embedding", "predict_phenotype", "transfer_labels"]


def __getattr__(name: str):
    # PEP 562 lazy attribute access — defer heavy imports to first use.
    if name == "wrapper":
        from sampledisco.wrapper.wrapper import wrapper
        return wrapper
    if name == "compute_sample_embedding":
        from sampledisco.sample_embedding import compute_sample_embedding
        return compute_sample_embedding
    if name in ("predict_phenotype", "transfer_labels"):
        from sampledisco.sample_prediction import prediction
        return getattr(prediction, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
