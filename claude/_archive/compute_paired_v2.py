"""For every method × every multi-omics dataset, compute the v2-style
scale-invariant paired-alignment score:

    paired_v2_score = mean(paired_distances) / std(nonpaired_distances)

(smaller = better cross-omics alignment, scale-invariant because std cancels
the embedding magnitude).

The score is appended to each summary.csv as a new metric row called
`paired_v2_score`. Doesn't touch the package code.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, "/users/hjiang/GenoDistance/code")
sys.path.insert(0, "/users/hjiang/GenoDistance/code/Benchmark_multiomics")

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


# -----------------------------------------------------------------------------
# Embedding paths per (dataset, method) — copied from each benchmark script's
# main(). SD_expression / SD_proportion are kept because some summary CSVs
# already have those columns (we'll fill the new score for them too if found).
# -----------------------------------------------------------------------------
ENCODE_OUTDIR = "/dcs07/hongkai/data/harry/result/multi_omics_ENCODE/multiomics"
LUTEA_OUTDIR  = "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/lutea"
RETINA_OUTDIR = "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/retina"
HEART_OUTDIR  = "/dcs07/hongkai/data/harry/result/multi_omics_heart/SD/multiomics"

DATASETS = {
    "ENCODE": {
        "summary": "/dcs07/hongkai/data/harry/result/Benchmark_multiomics/summary.csv",
        "methods": {
            "SD_expression": f"{ENCODE_OUTDIR}/embeddings/sample_expression_embedding.csv",
            "SD_proportion": f"{ENCODE_OUTDIR}/embeddings/sample_proportion_embedding.csv",
            "SampleDisco":   f"{ENCODE_OUTDIR}/sampledisco_default/sample_embedding/sample_embedding.csv",
            "pilot":         "/dcs07/hongkai/data/harry/result/Benchmark_multiomics/pilot/pilot_native_embedding.csv",
            "pseudobulk":    "/dcs07/hongkai/data/harry/result/Benchmark_multiomics/pseudobulk/pseudobulk/pca_embeddings.csv",
            "QOT":           "/dcs07/hongkai/data/harry/result/Benchmark_multiomics/QOT/88_qot_distance_matrix_mds_10d.csv",
            "GEDI":          "/dcs07/hongkai/data/harry/result/Benchmark_multiomics/GEDI/gedi_sample_embedding.csv",
            "Gloscope":      "/dcs07/hongkai/data/harry/result/Benchmark_multiomics/Gloscope/knn_divergence_mds_10d.csv",
            "MFA":           "/dcs07/hongkai/data/harry/result/Benchmark_multiomics/MFA/sample_embeddings.csv",
            "mustard":       "/dcs07/hongkai/data/harry/result/Benchmark_multiomics/mustard/sample_embedding.csv",
            "scPoli":        "/dcs07/hongkai/data/harry/result/Benchmark_multiomics/scPoli/sample_embeddings_full.csv",
        },
    },
    "Lutea": {
        "summary": "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/Benchmark_result/summary.csv",
        "methods": {
            "SD_expression": f"{LUTEA_OUTDIR}/embeddings/sample_expression_embedding.csv",
            "SD_proportion": f"{LUTEA_OUTDIR}/embeddings/sample_proportion_embedding.csv",
            "SampleDisco":   f"{LUTEA_OUTDIR}/sampledisco_default/sample_embedding/sample_embedding.csv",
            "pilot":         "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/pilot/wasserstein_distance_mds_10d.csv",
            "pseudobulk":    "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/pseudobulk/pseudobulk/pca_embeddings.csv",
            "QOT":           "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/QOT/24_qot_distance_matrix_mds_10d.csv",
            "GEDI":          "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/GEDI/gedi_sample_embedding.csv",
            "Gloscope":      "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/Gloscope/knn_divergence_mds_10d.csv",
            "MFA":           "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/MFA/sample_embeddings.csv",
            "mustard":       "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/mustard/sample_embedding.csv",
            "scPoli":        "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/scPoli/sample_embeddings_full.csv",
        },
    },
    "Retina": {
        "summary": "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/Benchmark_result/summary.csv",
        "methods": {
            "SD_expression": f"{RETINA_OUTDIR}/embeddings/sample_expression_embedding.csv",
            "SD_proportion": f"{RETINA_OUTDIR}/embeddings/sample_proportion_embedding.csv",
            "SampleDisco":   f"{RETINA_OUTDIR}/sampledisco_default/sample_embedding/sample_embedding.csv",
            "pilot":         "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/pilot/wasserstein_distance_mds_10d.csv",
            "pseudobulk":    "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/pseudobulk/pseudobulk/pca_embeddings.csv",
            "QOT":           "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/QOT/24_qot_distance_matrix_mds_10d.csv",
            "GEDI":          "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/GEDI/gedi_sample_embedding.csv",
            "Gloscope":      "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/Gloscope/knn_divergence_mds_10d.csv",
            "MFA":           "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/MFA/sample_embeddings.csv",
            "mustard":       "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/mustard/sample_embedding.csv",
            "scPoli":        "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/scPoli/sample_embeddings_full.csv",
        },
    },
    "Heart": {
        "summary": "/dcs07/hongkai/data/harry/result/multi_omics_heart/summary.csv",
        "methods": {
            "SD_expression": f"{HEART_OUTDIR}/embeddings/sample_expression_embedding.csv",
            "SD_proportion": f"{HEART_OUTDIR}/embeddings/sample_proportion_embedding.csv",
            "SampleDisco":   f"{HEART_OUTDIR}/sampledisco_default/sample_embedding/sample_embedding.csv",
            "pilot":         "/dcs07/hongkai/data/harry/result/multi_omics_heart/pilot/wasserstein_distance_mds_10d.csv",
            "pseudobulk":    "/dcs07/hongkai/data/harry/result/multi_omics_heart/pseudobulk/pseudobulk/pca_embeddings.csv",
            "QOT":           "/dcs07/hongkai/data/harry/result/multi_omics_heart/QOT/44_qot_distance_matrix_mds_10d.csv",
            "GEDI":          "/dcs07/hongkai/data/harry/result/multi_omics_heart/GEDI/gedi_sample_embedding.csv",
            "Gloscope":      "/dcs07/hongkai/data/harry/result/multi_omics_heart/Gloscope/knn_divergence_mds_10d.csv",
            "MFA":           "/dcs07/hongkai/data/harry/result/multi_omics_heart/MFA/sample_embeddings.csv",
            "mustard":       "/dcs07/hongkai/data/harry/result/multi_omics_heart/mustard/sample_embedding.csv",
            "scPoli":        "/dcs07/hongkai/data/harry/result/multi_omics_heart/scPoli/sample_embeddings_full.csv",
        },
    },
}


def parse_modality(name: str):
    """Return (sample_id_without_modality_suffix, modality)."""
    s = str(name)
    for suf, mod in [("_RNA","RNA"),("_rna","RNA"),("_ATAC","ATAC"),("_atac","ATAC")]:
        if s.endswith(suf):
            return s[: -len(suf)], mod
    for pre, mod in [("RNA_","RNA"),("ATAC_","ATAC")]:
        if s.startswith(pre):
            return s[len(pre):], mod
    return s, None


def compute_v2(emb_df: pd.DataFrame):
    """Return (mean_paired, std_nonpaired, score, n_pairs)."""
    names = list(emb_df.index.astype(str))
    parsed = [parse_modality(n) for n in names]
    # Build index of (sample_id, modality) -> row index
    by_sid = {}
    for i, (sid, mod) in enumerate(parsed):
        by_sid.setdefault(sid, {})[mod] = i
    paired_idx = []
    for sid, d in by_sid.items():
        if "RNA" in d and "ATAC" in d:
            paired_idx.append((d["RNA"], d["ATAC"]))
    if not paired_idx:
        return (np.nan, np.nan, np.nan, 0)
    X = emb_df.values.astype(float)
    # Pairwise distances over all pairs (i<j)
    D = squareform(pdist(X, metric="euclidean"))
    n = X.shape[0]
    # Mark paired cells in the upper triangle
    paired_mask = np.zeros((n, n), dtype=bool)
    for i, j in paired_idx:
        a, b = min(i,j), max(i,j)
        paired_mask[a, b] = True
    iu = np.triu_indices(n, k=1)
    paired_vals    = D[iu][paired_mask[iu]]
    nonpaired_vals = D[iu][~paired_mask[iu]]
    mp = float(paired_vals.mean())
    snp = float(nonpaired_vals.std(ddof=1))
    score = mp / snp if snp > 0 else np.nan
    return (mp, snp, score, len(paired_idx))


def main():
    for ds, cfg in DATASETS.items():
        print(f"\n=== {ds} ===")
        rows = {}
        for method, emb_path in cfg["methods"].items():
            if not os.path.exists(emb_path):
                print(f"  {method:<14s}  MISS  {emb_path}")
                continue
            try:
                df = pd.read_csv(emb_path, index_col=0)
                mp, snp, score, n = compute_v2(df)
                if n == 0:
                    print(f"  {method:<14s}  no paired pairs found (n_idx={len(df)})")
                    continue
                rows[method] = score
                print(f"  {method:<14s}  n_pairs={n:>3d}  mean(paired)={mp:.4f}  std(nonpaired)={snp:.4f}  score={score:.4f}")
            except Exception as e:
                print(f"  {method:<14s}  ERROR {e}")
        # Append to summary CSV
        if rows:
            sp = cfg["summary"]
            d = pd.read_csv(sp, index_col=0)
            for meth, v in rows.items():
                if meth not in d.columns:
                    d[meth] = np.nan
                d.loc["paired_v2_score", meth] = v
            d.to_csv(sp)
            print(f"  → wrote 'paired_v2_score' row to {sp}")


if __name__ == "__main__":
    main()
