"""Usable CPU count for worker pools and thread settings.

The CPU-affinity mask respects cgroups/containers, but on SLURM it can still be
wider than the allocation (e.g. both hyperthreads of every allocated core: 16 on
an 8-CPU job), so it is capped at SLURM_CPUS_PER_TASK when that is set.
"""
import contextlib
import os


def usable_cpu_count() -> int:
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:  # sched_getaffinity is Linux-only
        n = os.cpu_count() or 1
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
    if slurm_cpus.isdigit() and int(slurm_cpus) > 0:
        n = min(n, int(slurm_cpus))
    return n


def resolve_n_jobs(n_jobs):
    """``usable_cpu_count()`` for n_jobs None/-1; any other value is returned unchanged."""
    return usable_cpu_count() if n_jobs in (None, -1) else n_jobs


@contextlib.contextmanager
def single_threaded():
    """Run the block with torch's intra-op pool and the BLAS/OpenMP pools
    (threadpoolctl) at one thread, then restore the previous settings.

    Both are needed: torch.set_num_threads() (called by harmony-pytorch) sets MKL's
    *thread-local* thread count, which overrides the global count threadpoolctl sets,
    so with MKL threadpool_limits alone stops working once torch has set threads.
    Used where multi-threaded reductions broke run-to-run reproducibility: harmonypy
    >= 0.2 (PyTorch backend), the cell-level PCA / LSI, and the serial GAM fits."""
    import sys
    from threadpoolctl import threadpool_limits
    torch = sys.modules.get("torch")   # only loaded torch can have set MKL's thread-local count
    prev = torch.get_num_threads() if torch is not None else None
    try:
        if torch is not None:
            torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            yield
    finally:
        if torch is not None:
            torch.set_num_threads(prev)
