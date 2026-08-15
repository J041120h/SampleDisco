import anndata as ad, pandas as pd, numpy as np
p="/dcs07/hongkai/data/harry/result/1M-scBloodNL/V3/rna/preprocess/adata_cell.h5ad"
a=ad.read_h5ad(p, backed="r")
o=a.obs[["id","seq_batch","timepoint","stimulation_conditions"]].astype(str)
u=o.drop_duplicates("id").set_index("id")
print("units:", len(u))
ct=pd.crosstab(u["seq_batch"], u["timepoint"])
print("\nseq_batch x timepoint (first 12 batches):"); print(ct.head(12))
# Cramer's V
chi2=((ct-np.outer(ct.sum(1),ct.sum(0))/ct.values.sum())**2/(np.outer(ct.sum(1),ct.sum(0))/ct.values.sum())).values.sum()
n=ct.values.sum(); v=np.sqrt(chi2/(n*(min(ct.shape)-1)))
print(f"\nCramer's V(seq_batch, timepoint) = {v:.3f}   (1.0 = fully confounded)")
print("units per seq_batch: min=%d median=%.0f max=%d" % (ct.sum(1).min(), ct.sum(1).median(), ct.sum(1).max()))
print("\nbatches whose units are ALL one timepoint: %d / %d" % ((ct>0).sum(1).eq(1).sum(), len(ct)))
a.file.close()
