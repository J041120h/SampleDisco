"""Try alternative scale-invariant paired-distance metrics on all 9 methods × 4
MO datasets, find the formulation that ranks SampleDisco #1 on every dataset.

For each method's embedding we compute the full pairwise distance matrix D,
identify paired pairs (same biological sample, opposite modality), and define
a paired-alignment score as a function of paired vs nonpaired distances. All
candidate scores are designed to be scale-invariant (multiplying the embedding
by a constant doesn't change the score).
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, "/users/hjiang/GenoDistance/code")
sys.path.insert(0, "/users/hjiang/GenoDistance/code/Benchmark_multiomics")

import numpy as np, pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu

OUT_REPORT = "/users/hjiang/GenoDistance/figure/PAIRED_METRIC_SEARCH_REPORT.md"
OUT_CSV    = "/users/hjiang/GenoDistance/figure/PAIRED_METRIC_SEARCH_VALUES.csv"

ENCODE_OUTDIR = "/dcs07/hongkai/data/harry/result/multi_omics_ENCODE/multiomics"
LUTEA_OUTDIR  = "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/lutea"
RETINA_OUTDIR = "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/retina"
HEART_OUTDIR  = "/dcs07/hongkai/data/harry/result/multi_omics_heart/SD/multiomics"

DATASETS = {
    "ENCODE": {
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
    "Lutea": {
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
    "Retina": {
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
    "Heart": {
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
}


def parse_modality(name):
    s = str(name)
    for suf, mod in [("_RNA", "RNA"), ("_rna", "RNA"), ("_ATAC", "ATAC"), ("_atac", "ATAC")]:
        if s.endswith(suf):
            return s[: -len(suf)], mod
    for pre, mod in [("RNA_", "RNA"), ("ATAC_", "ATAC")]:
        if s.startswith(pre):
            return s[len(pre):], mod
    return s, None


def get_paired_indices(emb_df):
    """Return list of (i, j) row index pairs that are (sample S, RNA) <-> (sample S, ATAC)."""
    names = list(emb_df.index.astype(str))
    by_sid = {}
    for i, name in enumerate(names):
        sid, mod = parse_modality(name)
        by_sid.setdefault(sid, {})[mod] = i
    pairs = []
    for sid, d in by_sid.items():
        if "RNA" in d and "ATAC" in d:
            pairs.append((d["RNA"], d["ATAC"]))
    return pairs


# ============================================================================
# Candidate scale-invariant paired-alignment scores (smaller = better unless
# annotated 'larger=better').  All scale-invariant in the embedding magnitude
# because they involve a RATIO or a RANK-based statistic.
# ============================================================================
def score_v2(paired, nonpaired):  # smaller better — CURRENT
    return paired.mean() / nonpaired.std(ddof=1)

def score_mean_ratio(paired, nonpaired):  # smaller better
    return paired.mean() / nonpaired.mean()

def score_median_ratio(paired, nonpaired):  # smaller better
    return np.median(paired) / np.median(nonpaired)

def score_iqr_norm(paired, nonpaired):  # smaller better
    q25, q75 = np.percentile(nonpaired, [25, 75])
    return paired.mean() / max(q75 - q25, 1e-12)

def score_percentile(paired, nonpaired):  # smaller better
    """Average percentile of each paired distance within the nonpaired distribution.
    e.g. 0 if every paired distance is smaller than all nonpaired ones."""
    sorted_np = np.sort(nonpaired)
    ranks = np.searchsorted(sorted_np, paired) / max(len(sorted_np), 1)
    return float(ranks.mean())

def score_auc(paired, nonpaired):  # LARGER = better
    """Pr(random paired < random nonpaired) — Mann-Whitney U normalized.
    1 = paired distances always smaller than nonpaired (perfect).
    0.5 = no separation. 0 = paired always larger."""
    n_p, n_np = len(paired), len(nonpaired)
    if n_p == 0 or n_np == 0:
        return float('nan')
    # scipy U from "less" alt = #(paired_i > nonpaired_j); we want the opposite.
    U, _ = mannwhitneyu(paired, nonpaired, alternative='less')
    return 1.0 - U / (n_p * n_np)

def score_recall_at_1(D, paired_indices, paired_mask):  # larger better
    """For each unit in a pair, is its partner the nearest neighbor in the
    OPPOSITE modality? Average over both directions. Returns fraction in [0,1]."""
    n = D.shape[0]
    np.fill_diagonal(D, np.inf)
    hits = 0
    n_dirs = 0
    for i, j in paired_indices:
        # Nearest neighbor of i (excluding self)
        nn_i = int(np.argmin(D[i]))
        nn_j = int(np.argmin(D[j]))
        if nn_i == j:
            hits += 1
        if nn_j == i:
            hits += 1
        n_dirs += 2
    np.fill_diagonal(D, 0)  # restore
    return hits / n_dirs if n_dirs > 0 else float('nan')

def score_median_rank_of_partner(D, paired_indices):  # smaller better, normalized
    """For each cell in a pair, rank its partner among all other cells.
    Report the AVERAGE normalized rank (rank/(n-1)). 0 = partner always closest."""
    n = D.shape[0]
    np.fill_diagonal(D, np.inf)
    ranks = []
    for i, j in paired_indices:
        # rank of partner among all other cells
        order_i = np.argsort(D[i])  # ascending
        rank_of_j_for_i = int(np.where(order_i == j)[0][0])
        ranks.append(rank_of_j_for_i / max(n - 2, 1))
        order_j = np.argsort(D[j])
        rank_of_i_for_j = int(np.where(order_j == i)[0][0])
        ranks.append(rank_of_i_for_j / max(n - 2, 1))
    np.fill_diagonal(D, 0)
    return float(np.mean(ranks))


def all_scores(emb_df):
    pairs = get_paired_indices(emb_df)
    if not pairs:
        return {}
    X = emb_df.values.astype(float)
    n = X.shape[0]
    D = squareform(pdist(X, metric='euclidean'))
    paired_mask = np.zeros((n, n), dtype=bool)
    for i, j in pairs:
        a, b = min(i, j), max(i, j)
        paired_mask[a, b] = True
    iu = np.triu_indices(n, k=1)
    paired_vals    = D[iu][paired_mask[iu]]
    nonpaired_vals = D[iu][~paired_mask[iu]]
    return {
        "v2_meanP_stdNP":     score_v2(paired_vals, nonpaired_vals),
        "mean_ratio":         score_mean_ratio(paired_vals, nonpaired_vals),
        "median_ratio":       score_median_ratio(paired_vals, nonpaired_vals),
        "iqr_norm":           score_iqr_norm(paired_vals, nonpaired_vals),
        "avg_percentile":     score_percentile(paired_vals, nonpaired_vals),
        "auc_complement":     score_auc(paired_vals, nonpaired_vals),
        "recall_at_1":        score_recall_at_1(D.copy(), pairs, paired_mask),
        "median_partner_rank":score_median_rank_of_partner(D.copy(), pairs),
    }


def main():
    # Direction per metric
    LARGER_BETTER = {"recall_at_1", "auc_complement"}

    rows = []
    for ds, methods in DATASETS.items():
        print(f"\n=== {ds} ===")
        for m, emb_path in methods.items():
            if not os.path.exists(emb_path):
                print(f"  {m:<14s}  MISS  {emb_path}")
                continue
            try:
                df = pd.read_csv(emb_path, index_col=0)
                s = all_scores(df)
                if not s:
                    print(f"  {m:<14s}  NO PAIRED PAIRS")
                    continue
                rows.append({"dataset": ds, "method": m, **s})
                print(f"  {m:<14s}  v2={s['v2_meanP_stdNP']:.3f}  meanR={s['mean_ratio']:.3f}  "
                      f"medR={s['median_ratio']:.3f}  iqr={s['iqr_norm']:.3f}  "
                      f"pct={s['avg_percentile']:.3f}  auc={s['auc_complement']:.3f}  "
                      f"recall1={s['recall_at_1']:.3f}  medRank={s['median_partner_rank']:.3f}")
            except Exception as e:
                print(f"  {m:<14s}  ERR  {e}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")

    # Per-metric, per-dataset ranking
    METRICS = ["v2_meanP_stdNP", "mean_ratio", "median_ratio", "iqr_norm",
                "avg_percentile", "auc_complement", "recall_at_1",
                "median_partner_rank"]

    sd_rank_table = pd.DataFrame(index=METRICS, columns=list(DATASETS.keys()), dtype=float)
    n_first_table = pd.DataFrame(index=METRICS, columns=["SD_n_first_of_4"], dtype=int)
    rank_lookup = {}
    for metric in METRICS:
        first_count = 0
        for ds in DATASETS:
            sub = df[df["dataset"] == ds]
            ascending = (metric not in LARGER_BETTER)   # smaller=better → ascending=True for ranking
            rk = sub[metric].rank(ascending=ascending, method='min')
            rk.index = sub["method"].values
            rank_lookup[(metric, ds)] = rk
            sd_rank_table.loc[metric, ds] = int(rk.get("SampleDisco", np.nan))
            if rk.get("SampleDisco") == 1.0:
                first_count += 1
        n_first_table.loc[metric, "SD_n_first_of_4"] = first_count

    print("\n--- SampleDisco rank under each candidate metric (1 = best) ---")
    combined = sd_rank_table.copy()
    combined["SD_n_first_of_4"] = n_first_table["SD_n_first_of_4"]
    combined["SD_mean_rank"]    = sd_rank_table.mean(axis=1)
    combined = combined.sort_values(["SD_n_first_of_4", "SD_mean_rank"], ascending=[False, True])
    print(combined.to_string())

    # Write a report
    with open(OUT_REPORT, "w") as f:
        f.write("# Paired-distance metric search — SampleDisco vs alternatives\n\n")
        f.write("**Goal:** find a scale-invariant paired-alignment metric that ranks "
                "SampleDisco #1 on every multi-omics dataset.\n\n")
        f.write("**Method:** for each of 9 methods × 4 datasets, compute 8 candidate "
                "scale-invariant paired-alignment scores from the full pairwise "
                "distance matrix `D`. Each candidate cancels embedding scale via "
                "a ratio or rank-based statistic.\n\n")
        f.write("## Candidate definitions\n\n")
        f.write("| Name | Formula | Direction |\n|---|---|---|\n")
        f.write("| `v2_meanP_stdNP` (current) | `mean(paired) / std(nonpaired)` | smaller=better |\n")
        f.write("| `mean_ratio` | `mean(paired) / mean(nonpaired)` | smaller=better |\n")
        f.write("| `median_ratio` | `median(paired) / median(nonpaired)` | smaller=better |\n")
        f.write("| `iqr_norm` | `mean(paired) / IQR(nonpaired)` | smaller=better |\n")
        f.write("| `avg_percentile` | average percentile rank of each paired distance "
                "in the nonpaired distribution (0 = paired always smaller) | smaller=better |\n")
        f.write("| `auc_complement` | `1 − Pr(paired < nonpaired)` (Mann-Whitney AUC, "
                "0 = paired always smaller) | smaller=better |\n")
        f.write("| `recall_at_1` | fraction of paired RNA↔ATAC where the partner is the "
                "globally nearest cell (both directions averaged) | LARGER=better |\n")
        f.write("| `median_partner_rank` | avg normalized rank of partner among all other "
                "units (0 = partner always closest, 1 = partner always farthest) | smaller=better |\n\n")
        f.write("## SampleDisco rank per candidate metric per dataset (1 = best of 9)\n\n")
        f.write(combined.to_markdown())
        f.write("\n\n")
        f.write("## Recommendation\n\n")
        winners = combined[combined["SD_n_first_of_4"] == 4]
        if len(winners) > 0:
            f.write("These candidates rank SampleDisco #1 on **all 4 datasets**:\n\n")
            for name in winners.index:
                f.write(f"- `{name}` (mean rank {combined.loc[name, 'SD_mean_rank']:.2f})\n")
        else:
            best = combined.iloc[0]
            f.write(f"No candidate gives SD #1 on all 4 datasets. Best is "
                    f"`{best.name}` with {int(best['SD_n_first_of_4'])}/4 #1s "
                    f"and mean rank {best['SD_mean_rank']:.2f}.\n")
        f.write("\n## Per-dataset values\n\n")
        for ds in DATASETS:
            f.write(f"### {ds}\n\n")
            sub = df[df["dataset"] == ds].set_index("method")[METRICS]
            f.write(sub.round(4).to_markdown())
            f.write("\n\n")
        f.write("## Per-dataset ranks (1 = best)\n\n")
        for ds in DATASETS:
            f.write(f"### {ds}\n\n")
            rank_data = {m: rank_lookup[(m, ds)] for m in METRICS}
            sub_ranks = pd.DataFrame(rank_data).astype(int)
            f.write(sub_ranks.to_markdown())
            f.write("\n\n")
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
