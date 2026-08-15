#!/usr/bin/env python
"""How much does each block actually contribute to the final embedding?

Two measures per dataset, both read straight from the cached blocks:

  share_F   fraction of the stacked matrix's squared Frobenius norm held by
            each block. frobenius_stack rescales block b to ||B||_F = sqrt(N)*w_b,
            so this is exactly w_b^2 / sum(w^2) — fixed by the weights alone,
            independent of the data.

  share_PC  fraction of the top-10 PC energy that comes from each block's
            columns, i.e. sum over PCs of (eigenvalue * squared loading mass on
            that block's columns), normalised. This is what the "Combined"
            panel actually shows.
"""
import glob
import json
import os
import sys

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, "/users/hjiang/GenoDistance/code/src")
from sampledisco.sample_embedding.blocks import frobenius_stack  # noqa: E402

ROOT = "/users/hjiang/GenoDistance/figure/figure2"
BLOCKS = ["A1", "A2", "A3", "RMD"]


def analyse(cache):
    with open(f"{cache}/block_meta.json") as fh:
        meta = json.load(fh)
    w = [meta["weights"][b] for b in BLOCKS]
    blocks = [np.load(f"{cache}/{b}.npy") for b in BLOCKS]
    dims = [b.shape[1] for b in blocks]

    share_F = np.array(w) ** 2
    share_F = share_F / share_F.sum()

    F = frobenius_stack(blocks, w)
    n_pc = min(10, F.shape[0] - 1, F.shape[1])
    pca = PCA(n_components=n_pc, random_state=42).fit(F)
    # energy of each block's columns across the retained PCs
    edges = np.cumsum([0] + dims)
    comp = pca.components_                      # (n_pc, n_feat)
    ev = pca.explained_variance_                # (n_pc,)
    energy = np.array([
        float((ev[:, None] * comp[:, edges[i]:edges[i + 1]] ** 2).sum())
        for i in range(len(BLOCKS))
    ])
    share_PC = energy / energy.sum()
    return meta, w, dims, share_F, share_PC


def main():
    caches = sorted(glob.glob(f"{ROOT}/*/_cache/block_meta.json") +
                    glob.glob(f"{ROOT}/*/*/_cache/block_meta.json") +
                    glob.glob(f"{ROOT}/*/*/*/_cache/block_meta.json"))
    print(f"{'dataset':<26}{'K_c':>4}{'w_A1':>7}{'w_RMD':>7}"
          f"{'  --- share of Frobenius --- ':^32}{'  --- share of top-10 PC --- ':^32}")
    print(f"{'':<26}{'':>4}{'':>7}{'':>7}"
          f"{'A1':>8}{'A2':>8}{'A3':>8}{'RMD':>8}"
          f"{'A1':>8}{'A2':>8}{'A3':>8}{'RMD':>8}")
    print("-" * 122)
    rows = []
    for c in caches:
        cache = os.path.dirname(c)
        name = os.path.relpath(os.path.dirname(cache), ROOT)
        try:
            meta, w, dims, sF, sP = analyse(cache)
        except Exception as e:
            print(f"{name:<26} FAILED {type(e).__name__}: {e}")
            continue
        print(f"{name:<26}{meta['K_c']:>4}{w[0]:>7.2f}{w[3]:>7.2f}"
              + "".join(f"{v * 100:>7.1f}%" for v in sF)
              + "".join(f"{v * 100:>7.1f}%" for v in sP))
        rows.append((name, meta["K_c"], sF[3], sP[3]))

    if rows:
        rmdF = np.array([r[2] for r in rows]) * 100
        rmdP = np.array([r[3] for r in rows]) * 100
        print("-" * 122)
        print(f"RMD share of Frobenius : min {rmdF.min():.2f}%  "
              f"median {np.median(rmdF):.2f}%  max {rmdF.max():.2f}%")
        print(f"RMD share of top-10 PC : min {rmdP.min():.2f}%  "
              f"median {np.median(rmdP):.2f}%  max {rmdP.max():.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
