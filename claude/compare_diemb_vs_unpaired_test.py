"""Compare unpaired_diemb (Mode B: 2-run scGLUE, Z_clust + Z_cmd) against
unpaired_test (Mode A: X_glue + Harmony post-pass) on the same set of
sample embedding diagnostics.

Per sample-embedding matrix we compute:
  - autotune cmd_weight        (from autotune_record.txt)
  - autotune proxy score       (from autotune_record.txt)
  - mean PC R²(batch)          (per-PC linear R² ~ batch one-hot, averaged)
  - ASW(batch)                 (silhouette on sample embedding by batch)
  - ASW(modality)              (silhouette on sample embedding by RNA / ATAC suffix in sample id)
  - CCA(emb, sev.level)        (first canonical correlation with sev.level numeric)
"""
from __future__ import annotations
import os, re, sys
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

META = "/dcl01/hongkai/data/data/hjiang/Data/merged_rna_atac_metadata.csv"

RUNS = [
    ("diemb_alltune (Mode B: Z_clust + Z_cmd, autotune on RNA+ATAC)",
     "/dcs07/hongkai/data/harry/result/multi_omics_unpaired_diemb/multiomics/sample_embedding"),
    ("diemb_RNAtune (Mode B: Z_clust + Z_cmd, autotune on RNA only)",
     "/dcs07/hongkai/data/harry/result/multi_omics_unpaired_diemb/multiomics/sample_embedding_tune-on-RNA"),
    ("test_RETUNE (Mode A: X_glue + dual-Harmony, autotune on RNA+ATAC)",
     "/dcs07/hongkai/data/harry/result/multi_omics_unpaired_test/multiomics/sampledisco_tuned_v2_RETUNE/sample_embedding"),
    ("test_celltype-on-harmony (Mode A: X_glue + dual-Harmony, autotune on RNA+ATAC)",
     "/dcs07/hongkai/data/harry/result/multi_omics_unpaired_test/multiomics/sampledisco_tuned_v2_celltype-on-harmony/sample_embedding"),
]

DROP_FIRST_PC_FOR_R2 = False   # if True, skip PC1 (often dominates total var)


def parse_autotune(path: str) -> dict:
    out = {"cmd_weight": np.nan, "score": np.nan, "K_c": np.nan}
    if not os.path.exists(path): return out
    txt = open(path).read()
    m = re.search(r"best cmd_weight\s*:\s*([\d.]+)", txt);  out["cmd_weight"] = float(m.group(1)) if m else np.nan
    m = re.search(r"best score\s*:\s*([\d.]+)", txt);       out["score"]      = float(m.group(1)) if m else np.nan
    m = re.search(r"K_c\s*\(cell types\)\s*:\s*(\d+)", txt); out["K_c"]       = int(m.group(1))   if m else np.nan
    return out


def load_meta() -> pd.DataFrame:
    df = pd.read_csv(META)
    if "sample" not in df.columns:
        raise KeyError("metadata missing 'sample' column")
    return df.set_index("sample")


def strip_modality(idx: pd.Index) -> tuple[pd.Index, pd.Series]:
    """Sample IDs in SE are like `CoV-1.1-Wilk_RNA` / `..._ATAC`. Return
    cleaned base id + a modality series."""
    mod = idx.to_series().str.extract(r"_(RNA|ATAC)$")[0].fillna("RNA").values
    base = pd.Index([re.sub(r"_(RNA|ATAC)$", "", s) for s in idx], name="sample")
    return base, pd.Series(mod, index=idx, name="modality")


def pc_R2_batch(emb: pd.DataFrame, batch: pd.Series) -> float:
    """Mean per-PC R² of regressing each PC on batch one-hot."""
    common = emb.index.intersection(batch.dropna().index)
    if len(common) < 5 or batch.loc[common].nunique() < 2:
        return np.nan
    X = pd.get_dummies(batch.loc[common], drop_first=True).values
    if X.shape[1] == 0:
        return np.nan
    Y = emb.loc[common].values
    if DROP_FIRST_PC_FOR_R2 and Y.shape[1] > 1:
        Y = Y[:, 1:]
    r2s = []
    for j in range(Y.shape[1]):
        lr = LinearRegression().fit(X, Y[:, j])
        yhat = lr.predict(X)
        ss_res = float(((Y[:, j] - yhat) ** 2).sum())
        ss_tot = float(((Y[:, j] - Y[:, j].mean()) ** 2).sum())
        r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else np.nan)
    return float(np.nanmean(r2s))


def asw_label(emb: pd.DataFrame, label: pd.Series) -> float:
    common = emb.index.intersection(label.dropna().index)
    lab = label.loc[common]
    if lab.nunique() < 2 or len(common) < 3:
        return np.nan
    return float(silhouette_score(emb.loc[common].values, lab.values))


def cca_grouping(emb: pd.DataFrame, group: pd.Series) -> float:
    common = emb.index.intersection(group.dropna().index)
    if len(common) < 5: return np.nan
    Y = group.loc[common].astype(float).values.reshape(-1, 1)
    X = emb.loc[common].values
    n_comp = 1
    cca = CCA(n_components=n_comp, max_iter=2000)
    try:
        cca.fit(X, Y)
        x_c, y_c = cca.transform(X, Y)
        return float(np.corrcoef(x_c[:, 0], y_c[:, 0])[0, 1])
    except Exception:
        return np.nan


def evaluate_one(emb_path: str, autotune_path: str, meta: pd.DataFrame) -> dict:
    emb = pd.read_csv(emb_path, index_col=0)
    base, modality = strip_modality(emb.index)
    md_aligned = meta.reindex(base.values)
    md_aligned.index = emb.index   # align meta back onto suffixed ids

    batch = md_aligned["batch"]    if "batch"    in md_aligned else pd.Series(index=emb.index, dtype=object)
    sev   = md_aligned["sev.level"] if "sev.level" in md_aligned else pd.Series(index=emb.index, dtype=float)
    sev_num = pd.to_numeric(sev, errors="coerce")

    rna_idx  = emb.index[modality.values == "RNA"]
    atac_idx = emb.index[modality.values == "ATAC"]

    tune = parse_autotune(autotune_path)
    return {
        "n_units":            int(emb.shape[0]),
        "n_RNA":              int(len(rna_idx)),
        "n_ATAC":             int(len(atac_idx)),
        "K_c":                tune["K_c"],
        "cmd_weight":         tune["cmd_weight"],
        "proxy_score":        tune["score"],
        "mean_PC_R2_batch":   pc_R2_batch(emb, batch),
        "ASW_batch":          asw_label(emb, batch),
        "ASW_modality":       asw_label(emb, modality),
        "CCA_sev.level":      cca_grouping(emb, sev_num),
        "CCA_sev.level_RNA":  cca_grouping(emb.loc[rna_idx],  sev_num.loc[rna_idx]),
        "CCA_sev.level_ATAC": cca_grouping(emb.loc[atac_idx], sev_num.loc[atac_idx]),
        "ASW_batch_RNA":      asw_label(emb.loc[rna_idx],  batch.loc[rna_idx]),
        "ASW_batch_ATAC":     asw_label(emb.loc[atac_idx], batch.loc[atac_idx]),
    }


def main():
    meta = load_meta()
    print(f"[meta] {len(meta):,} samples; cols head: {list(meta.columns)[:8]}")
    rows = {}
    for name, base in RUNS:
        emb_path = os.path.join(base, "sample_embedding.csv")
        at_path  = os.path.join(base, "autotune_record.txt")
        if not os.path.exists(emb_path):
            print(f"[skip] {name}: no sample_embedding.csv at {emb_path}"); continue
        rows[name] = evaluate_one(emb_path, at_path, meta)
        print(f"[done] {name}")
    df = pd.DataFrame(rows).T
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print()
    print(df.to_string())
    out = "/dcs07/hongkai/data/harry/result/multi_omics_unpaired_diemb/multiomics/comparison_vs_unpaired_test.csv"
    df.to_csv(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
