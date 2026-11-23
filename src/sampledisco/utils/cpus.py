"""Usable CPU count for worker pools and thread settings.

The CPU-affinity mask respects cgroups/containers, but on SLURM it can still be
wider than the allocation (e.g. both hyperthreads of every allocated core: 16 on
an 8-CPU job), so it is capped at SLURM_CPUS_PER_TASK when that is set.
"""
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
