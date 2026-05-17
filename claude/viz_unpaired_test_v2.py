"""Generate figure3-style embedding plots for the V2 unpaired_test SE run
(cell typing on X_glue_harmony at default res=0.8).

3 projections × N built-in plot types from figure3/embedding/1.py:
  first2pc, best2pc, umap

Output: /users/hjiang/GenoDistance/figure/figure3/embedding/sampledisco_unpaired_test_v2/
"""
from __future__ import annotations
import os, sys, time, shutil, importlib.util
sys.path.insert(0, "/users/hjiang/GenoDistance/code")

import numpy as np, pandas as pd, scanpy as sc, anndata as ad

EMB_CSV  = '/dcs07/hongkai/data/harry/result/multi_omics_unpaired_test/multiomics/sampledisco_tuned_v2_celltype-on-harmony/sample_embedding/sample_embedding.csv'
H5       = '/dcs07/hongkai/data/harry/result/multi_omics_unpaired_test/multiomics/preprocess/atac_rna_integrated.h5ad'
OUT_BASE = '/users/hjiang/GenoDistance/figure/figure3/embedding/sampledisco_unpaired_test_v2'


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


log(f"reading embedding CSV: {EMB_CSV}")
emb = pd.read_csv(EMB_CSV, index_col=0)
log(f"  embedding shape: {emb.shape}")

log(f"reading cell-level meta from {H5} (backed, no X)")
a = ad.read_h5ad(H5, backed='r')
cell_obs = pd.DataFrame({
    'sample':    a.obs['sample'].astype(str).values,
    'modality':  a.obs['modality'].astype(str).values,
    'batch':     a.obs['batch'].astype(str).values,
    'sev.level': a.obs['sev.level'].astype(str).values,
})
a.file.close()

def majority(s):
    s = s.dropna()
    return s.mode().iloc[0] if not s.empty else 'UNK'

unit_meta = (cell_obs
             .groupby(['sample', 'modality'])
             .agg({'batch': majority, 'sev.level': majority})
             .reset_index())
unit_meta['uid'] = unit_meta['sample'] + '_' + unit_meta['modality']
unit_meta = unit_meta.set_index('uid')

common = emb.index.intersection(unit_meta.index)
emb = emb.loc[common]
meta = unit_meta.loc[common]
log(f"  common units: {len(common)}")

sample_ad = sc.AnnData(X=np.zeros((len(common), 1), dtype=np.float32),
                       obs=meta[['sample', 'modality', 'batch', 'sev.level']].copy())
sample_ad.obsm['X_DR_sampledisco'] = emb.values.astype(np.float32)

spec = importlib.util.spec_from_file_location(
    "fig3_emb", "/users/hjiang/GenoDistance/figure/figure3/embedding/1.py")
fig3_emb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fig3_emb)

if os.path.exists(OUT_BASE):
    shutil.rmtree(OUT_BASE)

for proj in ("first2pc", "best2pc", "umap"):
    out_dir = os.path.join(OUT_BASE, proj)
    log(f"rendering {proj} → {out_dir}")
    fig3_emb.run_embedding_analysis(
        adata=sample_ad,
        embedding_key='X_DR_sampledisco',
        output_dir=out_dir,
        title_suffix=" (SampleDisco — unpaired_test V2, celltype-on-harmony)",
        projection=proj,
    )
log("DONE")
