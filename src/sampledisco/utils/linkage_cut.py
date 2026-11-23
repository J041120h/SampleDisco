"""Cut a hierarchical-clustering linkage into exactly k groups.

``scipy.cluster.hierarchy.fcluster(Z, t=k, criterion="maxclust")`` can return fewer
than k clusters even when the merge heights are distinct (seen on the ATAC demo:
9 Leiden clusters, t=8 -> 7 groups). The cell-type merges need exactly k.
"""
import numpy as np
import scipy.cluster.hierarchy as sch


def cut_linkage(Z, k: int) -> np.ndarray:
    """Labels 1..k (int array, one per leaf) for exactly ``k`` groups, ``1 <= k <= n``.

    The groups are the first ``n - k`` merges of the agglomeration (the rows of ``Z``
    are in merge order), i.e. the agglomeration stopped when k groups remain. For
    monotonic linkages (single, complete, average, weighted, ward) that is
    ``scipy.cluster.hierarchy.cut_tree(Z, n_clusters=k)``, which is used. cut_tree
    does not support centroid / median linkages (inversions); there the first n - k
    merges are taken directly.

    Whenever ``fcluster(Z, k, "maxclust")`` returns k groups its result is returned
    unchanged (same partition and labels as before). Only when it returns fewer are
    the first n - k merges used, numbered by fcluster itself (criterion "monocrit").
    """
    Z = np.asarray(Z, dtype=float)
    n = Z.shape[0] + 1
    if not 1 <= k <= n:
        raise ValueError(f"k must be between 1 and the number of leaves ({n}), got {k}")
    if n == 1:
        return np.ones(1, dtype=int)
    fc = np.asarray(sch.fcluster(Z, t=k, criterion="maxclust"), dtype=int)
    if len(np.unique(fc)) == k:
        return fc
    # monocrit 0 for the first n - k merges, 1 for the rest. It is monotonic for any
    # linkage (a merge's children always come from earlier rows), so fcluster(t=0.5)
    # returns exactly those merges as k groups, numbered the way fcluster numbers them.
    mc = np.zeros(n - 1, dtype=float)
    mc[n - k:] = 1.0
    labels = np.asarray(sch.fcluster(Z, t=0.5, criterion="monocrit", monocrit=mc), dtype=int)
    if sch.is_monotonic(Z):
        part = sch.cut_tree(Z, n_clusters=k).ravel()
        if len(set(zip(labels.tolist(), part.tolist()))) != k:   # not expected: same merges
            _, labels = np.unique(part, return_inverse=True)
            labels = labels + 1
    return labels
