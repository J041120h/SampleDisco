import anndata as ad, pandas as pd
R="/dcs07/hongkai/data/harry/result/1M-scBloodNL"
for tag, p in [("V1","rna"), ("V3","V3/rna"), ("V3_remove","V3/rna_remove"),
               ("V2_reg","V2/rna_regular"), ("V2_rmdonor","V2/rna_remove_donor")]:
    path=f"{R}/{p}/preprocess/adata_cell.h5ad"
    try: a=ad.read_h5ad(path, backed="r")
    except Exception as e: print(f"\n### {tag}: OPEN FAILED {e}"); continue
    print(f"\n### {tag}  {path}\n n_obs={a.n_obs}  obsm={list(a.obsm.keys())}")
    for c in ["assignment","id","timepoint","stimulation_conditions","chem",
              "seq_batch","seq_lane","seq_date","cell_type","cell_type_lowerres"]:
        if c not in a.obs.columns: continue
        s=a.obs[c].astype(str)
        print(f"  {c:<24} nunique={s.nunique():<6} e.g. {sorted(s.unique())[:6]}")
    # candidate composite sample units
    o=a.obs
    for combo in [("assignment","timepoint"),("assignment","stimulation_conditions"),
                  ("id","timepoint"),("assignment","timepoint","stimulation_conditions")]:
        if all(c in o.columns for c in combo):
            n=o[list(combo)].astype(str).agg("_".join,axis=1).nunique()
            print(f"  COMBO {'+'.join(combo):<50} -> {n} units")
    a.file.close()
