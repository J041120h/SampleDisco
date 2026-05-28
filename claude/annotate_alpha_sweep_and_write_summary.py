"""Annotate the existing alpha_sweep plot with the chosen α from both tunes
and emit a human-readable summary.txt.

Reads:
  alpha_sweep/alpha_sweep.csv                       (already-computed sweep)
  sample_embedding/autotune_record.txt              (all-modality tune)
  sample_embedding_tune-on-RNA/autotune_record.txt  (RNA-only tune)
  comparison_vs_unpaired_test.csv                   (per-modality eval table)

Writes:
  alpha_sweep/alpha_sweep_annotated.png             (sweep + α markers)
  summary.txt                                       (one-page narrative)
"""
from __future__ import annotations
import os, re
import numpy as np, pandas as pd

ROOT = "/dcs07/hongkai/data/harry/result/multi_omics_unpaired_diemb/multiomics"
SWEEP = f"{ROOT}/alpha_sweep/alpha_sweep.csv"
ALLTUNE = f"{ROOT}/sample_embedding/autotune_record.txt"
RNATUNE = f"{ROOT}/sample_embedding_tune-on-RNA/autotune_record.txt"
CMP_CSV = f"{ROOT}/comparison_vs_unpaired_test.csv"
OUT_PNG = f"{ROOT}/alpha_sweep/alpha_sweep_annotated.png"
OUT_TXT = f"{ROOT}/summary.txt"


def best_alpha(path):
    txt = open(path).read()
    m = re.search(r"best cmd_weight\s*:\s*([\d.]+)", txt)
    s = re.search(r"best score\s*:\s*([\d.]+)", txt)
    return float(m.group(1)) if m else None, float(s.group(1)) if s else None


def interp_at(df, alpha, col):
    """Linear interp of df[col] vs df['alpha'] at the requested alpha."""
    x = df["alpha"].values; y = df[col].values
    order = np.argsort(x); x = x[order]; y = y[order]
    return float(np.interp(alpha, x, y))


# ── load ──────────────────────────────────────────────────────────────────
df  = pd.read_csv(SWEEP)
a_alltune, s_alltune = best_alpha(ALLTUNE)
a_rnatune, s_rnatune = best_alpha(RNATUNE)
cmp_df = pd.read_csv(CMP_CSV, index_col=0)

print(f"all-modality tune  α = {a_alltune:.4f}  proxy = {s_alltune:.4f}")
print(f"RNA-only     tune  α = {a_rnatune:.4f}  proxy = {s_rnatune:.4f}")


# ── annotated sweep PNG ───────────────────────────────────────────────────
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for axi, (cols, label) in enumerate((
    (("r_rna_full", "r_atac_full"), "Full 10-PC CCA vs sev.level"),
    (("r_rna_jointBP", "r_atac_jointBP"), "Joint best-2-PC CCA vs sev.level"),
)):
    a_, b_ = cols
    ax[axi].plot(df["alpha"], df[a_], 'o-', color='tab:blue',   label='RNA',  ms=4)
    ax[axi].plot(df["alpha"], df[b_], 's-', color='tab:orange', label='ATAC', ms=4)
    pr, _ = pearsonr(df[a_], df[b_])

    # vertical lines + annotations at the two chosen alphas
    for alpha, color, tag in [(a_alltune, 'tab:green',  'all-modality tune'),
                              (a_rnatune, 'tab:red',    'RNA-only tune')]:
        ax[axi].axvline(alpha, color=color, ls='--', lw=1.2, alpha=0.8)
        y_rna  = interp_at(df, alpha, a_)
        y_atac = interp_at(df, alpha, b_)
        ax[axi].annotate(f"{tag}\nα={alpha:.2f}\nRNA={y_rna:.3f}\nATAC={y_atac:.3f}",
                          xy=(alpha, max(y_rna, y_atac)),
                          xytext=(alpha * 1.1, max(y_rna, y_atac) + 0.02),
                          fontsize=8, color=color,
                          arrowprops=dict(arrowstyle='->', color=color, lw=0.8))

    ax[axi].set_xscale("log")
    ax[axi].set_xlabel("cmd_weight α (log)")
    ax[axi].set_ylabel("CCA r vs sev.level")
    ax[axi].set_title(f"{label}\nPearson(RNA, ATAC across α) = {pr:+.3f}", fontsize=10)
    ax[axi].legend(loc='lower right')
    ax[axi].grid(alpha=0.3)
fig.suptitle("Mode B (diemb, Z_clust + Z_cmd) — per-modality CCA across α, "
              "with chosen α from each autotune objective", fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=130, bbox_inches='tight')
plt.close(fig)
print(f"wrote {OUT_PNG}")


# ── summary.txt ───────────────────────────────────────────────────────────
def fmt(v): return f"{v:.4f}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v)

# pull the per-modality numbers we want to highlight
def row(name):
    matches = [r for r in cmp_df.index if r.startswith(name)]
    return cmp_df.loc[matches[0]] if matches else None

r_alltune = row("diemb_alltune")
r_rnatune = row("diemb_RNAtune")
r_modeA   = row("test_RETUNE")

# per-α CCA at each chosen α (from the sweep)
sweep_at_alltune = {c: interp_at(df, a_alltune, c) for c in
                     ["r_rna_full", "r_atac_full", "r_rna_jointBP", "r_atac_jointBP"]}
sweep_at_rnatune = {c: interp_at(df, a_rnatune, c) for c in
                     ["r_rna_full", "r_atac_full", "r_rna_jointBP", "r_atac_jointBP"]}

lines = []
lines.append("=" * 78)
lines.append("Unpaired multi-omics SampleDisco — autotune-objective comparison")
lines.append("=" * 78)
lines.append("")
lines.append("Dataset")
lines.append("-" * 78)
lines.append("  source        : COVID unpaired RNA + ATAC (multi_omics_unpaired_diemb)")
lines.append("  cell count    : 987,395  (898,435 RNA + 88,960 ATAC)")
lines.append("  sample units  : 431  (405 RNA + 26 ATAC)")
lines.append("  batches       : 13")
lines.append("  grouping col  : sev.level  (numeric severity 1..4)")
lines.append("")
lines.append("Pipeline (Mode B = 2-run scGLUE)")
lines.append("-" * 78)
lines.append("  scGLUE run 1  : use_batch=batch     → obsm['Z_cmd']    (sample-preserved)")
lines.append("  scGLUE run 2  : use_batch=sample    → obsm['Z_clust']  (sample-removed)")
lines.append("  cell typing   : Leiden on Z_clust, K_c = 17, RNA→ATAC label transfer via")
lines.append("                  Jaccard-SNN (torch-GPU KNN fallback because cuml/cupy is")
lines.append("                  binary-incompatible in the conda env on these nodes)")
lines.append("  SE            : composition (A1+A2+A3) + CMD; α (cmd_weight) auto-tuned")
lines.append("                  via Bayesian search over [0.1, 10]; 15 evals each run")
lines.append("")
lines.append("Two autotune objectives compared")
lines.append("-" * 78)
lines.append(f"  ALL-MODALITY  (sample_embedding/):                 score on all 431 units")
lines.append(f"                 best α = {a_alltune:.4f}   proxy = {s_alltune:.4f}")
lines.append(f"  RNA-ONLY      (sample_embedding_tune-on-RNA/):     score on 405 RNA units")
lines.append(f"                 best α = {a_rnatune:.4f}   proxy = {s_rnatune:.4f}")
lines.append("  (Final embedding always built on all 431 units; only the autotune")
lines.append("   objective's unit subset differs.)")
lines.append("")
lines.append("Headline result — per-modality eval on the final embeddings")
lines.append("-" * 78)
lines.append(f"  {'':52s} {'all-mod tune':>14s} {'RNA-only tune':>14s} {'Δ':>9s}")
def cmp_row(label, key, fmtspec=".4f"):
    a = r_alltune[key] if r_alltune is not None else float('nan')
    b = r_rnatune[key] if r_rnatune is not None else float('nan')
    d = b - a
    lines.append(f"  {label:52s} {a:>14{fmtspec}} {b:>14{fmtspec}} {d:>+9{fmtspec}}")
cmp_row("cmd_weight α",                           "cmd_weight")
cmp_row("proxy_score",                            "proxy_score")
cmp_row("CCA(emb, sev.level) — full set",        "CCA_sev.level")
cmp_row("CCA(emb_RNA,  sev.level)",              "CCA_sev.level_RNA")
cmp_row("CCA(emb_ATAC, sev.level)",              "CCA_sev.level_ATAC")
cmp_row("mean PC R²(batch)",                     "mean_PC_R2_batch")
cmp_row("ASW(batch)",                            "ASW_batch")
cmp_row("ASW(modality)",                         "ASW_modality")
lines.append("")
lines.append("Per-α sweep CCA at the two chosen α (from alpha_sweep.csv)")
lines.append("-" * 78)
lines.append(f"  {'':30s} {'all-mod α=' + f'{a_alltune:.2f}':>20s} {'RNA-only α=' + f'{a_rnatune:.2f}':>20s}")
for col, label in [("r_rna_full",   "RNA full-10PC CCA"),
                    ("r_atac_full",  "ATAC full-10PC CCA"),
                    ("r_rna_jointBP", "RNA joint best-2-PC CCA"),
                    ("r_atac_jointBP","ATAC joint best-2-PC CCA")]:
    lines.append(f"  {label:30s} {sweep_at_alltune[col]:>20.4f} {sweep_at_rnatune[col]:>20.4f}")
lines.append("")
lines.append("Comparison to Mode A (X_glue + dual-Harmony)")
lines.append("-" * 78)
lines.append("  For reference, the same metrics on the previous Mode-A run")
lines.append("  (sampledisco_tuned_v2_RETUNE on multi_omics_unpaired_test):")
if r_modeA is not None:
    lines.append(f"    cmd_weight α                       : {r_modeA['cmd_weight']:.4f}")
    lines.append(f"    proxy_score                        : {r_modeA['proxy_score']:.4f}")
    lines.append(f"    CCA(emb_RNA,  sev.level)           : {r_modeA['CCA_sev.level_RNA']:.4f}")
    lines.append(f"    CCA(emb_ATAC, sev.level)           : {r_modeA['CCA_sev.level_ATAC']:.4f}")
    lines.append(f"    mean PC R²(batch)                  : {r_modeA['mean_PC_R2_batch']:.4f}")
    lines.append(f"    ASW(modality)                      : {r_modeA['ASW_modality']:.4f}")
lines.append("")
lines.append("Interpretation")
lines.append("-" * 78)
lines.append("  1. RNA-only tuning selects a LOWER α (1.63 vs 2.76) — composition")
lines.append("     blocks get more relative weight; CMD less. The autotune is willing")
lines.append("     to give up some CMD weight because RNA's bio signal is already")
lines.append("     well-encoded in composition (Z_clust is sample-removed, K_c=17).")
lines.append("")
lines.append("  2. RNA CCA is essentially unchanged (0.698 → 0.684, Δ = -0.014) — within")
lines.append("     noise of the 15-eval Bayesian search.")
lines.append("")
lines.append("  3. ATAC CCA actually IMPROVES (0.799 → 0.852, Δ = +0.053). Tuning the")
lines.append("     objective on RNA labels does not penalise ATAC; the dual-embedding")
lines.append("     (Z_clust + Z_cmd) carries sev.level signal coherently across both")
lines.append("     modalities, so a well-chosen α for RNA happens to also be well-chosen")
lines.append("     for ATAC.")
lines.append("")
lines.append("  4. This is consistent with the α-sweep correlation analysis: in Mode B")
lines.append("     joint-best-2-PC, Pearson(r_RNA, r_ATAC across α) ≈ -0.07 (n.s.) — the")
lines.append("     two modalities' CCA scores are essentially decoupled across α, so")
lines.append("     tuning on one does not force a trade-off against the other.")
lines.append("     (For contrast: Mode A's Pearson ≈ -0.61.)")
lines.append("")
lines.append("  5. Batch correction is slightly tighter under RNA-only tuning")
lines.append("     (ASW_batch -0.241 → -0.271; more negative = better batch mixing).")
lines.append("     mean_PC_R²(batch) is essentially unchanged (0.056 → 0.059).")
lines.append("")
lines.append("Artifacts")
lines.append("-" * 78)
lines.append("  sample_embedding/sample_embedding.csv               — all-modality tune SE")
lines.append("  sample_embedding/autotune_record.txt                — all-modality tune log")
lines.append("  sample_embedding_tune-on-RNA/sample_embedding.csv   — RNA-only tune SE")
lines.append("  sample_embedding_tune-on-RNA/autotune_record.txt    — RNA-only tune log")
lines.append("  comparison_vs_unpaired_test.csv                     — full eval table")
lines.append("  alpha_sweep/alpha_sweep.csv                         — 38-α grid sweep")
lines.append("  alpha_sweep/alpha_sweep_annotated.png               — sweep w/ chosen α marked")
lines.append("  summary.txt                                         — this file")
lines.append("")
lines.append("Reproduction")
lines.append("-" * 78)
lines.append("  Config         : config/config_unpaired.yaml          (all-modality tune)")
lines.append("                 : config/config_unpaired_RNA_tune.yaml (RNA-only tune)")
lines.append("  Launcher       : claude/run_diemb_test.py             (all-modality)")
lines.append("                 : claude/run_diemb_RNA_tune.py         (RNA-only)")
lines.append("  Wrapper change : parameter_selection/autotune.py adds tune_on_modality;")
lines.append("                   wrapper/multiomics_wrapper.py + wrapper/wrapper.py thread")
lines.append("                   it through as multiomics_autotune_tune_on_modality.")
lines.append("")

with open(OUT_TXT, "w") as f:
    f.write("\n".join(lines))
print(f"wrote {OUT_TXT}  ({len(lines)} lines)")
