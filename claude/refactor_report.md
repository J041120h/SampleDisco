# Sample-embedding refactor — implementation report

Status: code-complete. **No smoke test has been run** — wait for explicit authorization before invoking `SampleDisc.py`.

## What changed (per file)

### New files

- **`sample_embedding/blocks.py`** — shared math primitives:
  `soft_assign` (Gaussian-RBF, NumPy/Cupy-agnostic), `composition_per_unit`,
  `clr_transform` (opt-in only), `loo_cmd` (per-(group, cluster) leave-one-out
  displacement; group = modality for MO, batch for single-omics),
  `derive_weights` (inverse-variance: `w_A1 = √(K_fine/K_c)`, `w_A2 = √(K_fine/K_med)`,
  `w_A3 = 1.0`, `w_CMD = cmd_weight`), `frobenius_stack`,
  `regress_out_batch_linear`, `composite_batch_labels`, `build_emb_from_blocks`,
  `assemble_units`.

- **`sample_embedding/sample_embedding.py`** — CPU `compute_sample_embedding()`.
  Algorithm (singleCMD recipe, **no CLR by default**):
  A1 = one-hot cell-type composition · A2/A3 = soft-kmeans compositions at
  `medium_K` / `fine_K` on `cluster_emb_key` · CMD = LOO displacement on
  `cmd_emb_key` per (group, coarse cluster) · Frobenius stack with
  inverse-variance weights · PCA to `pca_components` (default **10**) ·
  composite-batch Harmony at the sample level. Output written to
  `<output_dir>/sample_embedding/{adata_sample.h5ad, sample_embedding.csv}`
  with `.uns['X_DR_sample']` (DataFrame) and `.obsm['X_DR_sample']`.

- **`sample_embedding/sample_embedding_gpu.py`** — GPU twin with the same API.
  Uses `cuml.cluster.MiniBatchKMeans`, `cupy` for RBF soft-assign,
  `cuml.decomposition.PCA`, and `harmony.harmonize(use_gpu=True)`
  (matching the library mix already used by `rna_preprocess_gpu.py`).

- **`sample_embedding/__init__.py`** — public dispatcher
  `compute_sample_embedding(..., use_gpu=False, ...)` that routes to CPU or
  GPU implementation.

- **`parameter_selection/autotune.py`** — `run_autotune()` with adaptive
  proxy gating. Default: `search='bayesian'`, `scoring='auto'`,
  `scope='alpha_only'`, bounds `(0.1, 10.0)`, `n_init=5`, `n_iter=10`.
  `make_scorer` drops `ilisi_batch` / `neg_asw_batch` when the data has no
  usable batch column and drops `cca` / `sps` / `cv_knn_severity` /
  `pseudotime_spearman` when there is no grouping/trajectory label.
  If neither is available, the autotune short-circuits and uses fixed
  defaults `[3.0, 1.55, 1.0, 0.60]`.

### Modified files

- **`preparation/rna_preprocess_cpu.py` / `rna_preprocess_gpu.py`** — dual
  Harmony: pass 1 batches `cell_level_batch_key + [sample_col]` →
  `obsm['X_pca_harmony']` (sample-removed); pass 2 batches
  `cell_level_batch_key` only → `obsm['X_pca_harmony_nosamp']`
  (sample-preserved). HVG flag set with `subset=False`; raw counts moved to
  `.layers['counts']` before normalize/log1p. Single
  `adata_preprocessed.h5ad` is written. The previous `anndata_sample`
  function was deleted.

- **`preparation/atac_preprocess_cpu.py` / `atac_preprocess_gpu.py`** — same
  pattern with `muon.atac.tfidf` + `ac.tl.lsi`. LSI runs on the HVG subset
  and the embedding is written back to the full-feature adata. Two Harmony
  passes produce `obsm['X_lsi_harmony']` (sample-removed) and
  `obsm['X_lsi_harmony_nosamp']` (sample-preserved). Single
  `adata_preprocessed.h5ad` is written.

- **`preparation/cell_type_cpu.py` / `cell_type_gpu.py`** — dropped the
  `anndata_sample` argument (and `defined_sample_output_path`). Now save
  `adata_preprocessed.h5ad` instead of `adata_cell.h5ad`. Returns a single
  AnnData.

- **`preparation/multi_omics_preprocess.py`** — writes
  `preprocess/adata_preprocessed.h5ad` (renamed from the misleadingly-named
  `adata_sample.h5ad`). Stale `from sample_embedding.pseudo_adata import *` /
  `from sample_embedding.DR import *` imports were removed.

- **`preparation/multi_omics_cell_type_cpu.py` / `multi_omics_cell_type_gpu.py`** —
  default save path renamed to `adata_preprocessed.h5ad`.

- **`sample_trajectory/trajectory_diff_gene.py`** — `_build_sample_pseudobulk()`
  helper added (mirrors the legacy `compute_pseudobulk_adata` recipe: per
  `(sample × celltype)` aggregation, optional Limma batch correction within
  each cell type, optional first-round HVG per cell type, concatenate into
  `samples × (celltype-gene)`). **No double normalization, no second HVG
  round.** `run_trajectory_gam_differential_gene_analysis` now takes
  `adata` (cell-level `adata_preprocessed`) as its first arg plus
  `celltype_col`, `batch_col`, `n_features_per_celltype`,
  `columns_to_preserve` kwargs.

- **`sample_association/association.py`** — `_available_embeddings()`
  returns `['X_DR_sample']` (with legacy `X_DR_expression` /
  `X_DR_proportion` fallback for old artifacts).

- **`sample_distance/sample_distance.py`** — collapsed the legacy
  `get_best_expression_dr_key` / `get_best_proportion_dr_key` into a single
  `get_best_sample_dr_key()`. `sample_distance_vector` now writes a single
  `sample_DR_distance/` directory (no expression/proportion split). The
  legacy aliases are kept as thin wrappers for any external callers.

- **`sample_trajectory/CCA.py`** — picks `X_DR_sample` first; falls back to
  legacy keys. Returns the legacy 4-tuple shape; when only `X_DR_sample`
  exists, the legacy slots resolve to the same data so existing call sites
  still work.

- **`sample_trajectory/multi_omics_CCA_test.py`** — picks up `X_DR_sample`
  in addition to the legacy keys when shape-aligning `.uns` arrays.

- **`sample_prediction/predict_sample_phenotype.py`** —
  `_get_feature_matrix` now resolves the sample DR to `X_DR_sample`
  (with legacy fallback), and `cluster_sample_kmeans` /
  `pseudotime_sample` likewise fall back to legacy column names.

- **`cluster.py`** — single-key K-means on `X_DR_sample` (legacy keys still
  accepted). Returns `(label_map, label_map)` for back-compat with callers
  that expected an `(expr, prop)` pair.

- **`wrapper/rna_wrapper.py`, `wrapper/atac_wrapper.py`,
  `wrapper/multiomics_wrapper.py`** — collapsed the dual-adata output into
  a single `adata` plus the new `sample_adata`. Sample embedding now routes
  through `sample_embedding.compute_sample_embedding(..., use_gpu=...)`,
  optionally followed by `parameter_selection.autotune.run_autotune` when
  `autotune_enable=True`. Drops the legacy `cca_based_cell_resolution_selection`
  step entirely.

- **`wrapper/wrapper.py`** — `downstream_analysis()` keeps the
  `pseudo_adata=` / `adata_cell=` / `adata_sample=` parameter names but the
  orchestrator now passes `sample_adata` to `pseudo_adata` and `adata` to
  both `adata_cell` and `adata_sample`. Trajectory analysis defaults
  `column="X_DR_sample"`; trajectory DGE now calls
  `run_trajectory_gam_differential_gene_analysis(adata=..., celltype_col=..., ...)`
  with the pseudobulk-on-the-fly helper. Multiomics embedding viz uses the
  new `multiomics_sample_embedding_key`. Deprecated config keys are still
  accepted in the orchestrator signature (and silently ignored) so legacy
  YAMLs keep parsing.

### Deleted files

- `sample_embedding/DR.py`
- `sample_embedding/pseudo_adata.py`
- `sample_embedding/pseudo_adata_linux.py`
- `sample_embedding/multi_omics_pseudobulk_cpu.py`
- `sample_embedding/multi_omics_pseudobulk_gpu.py`
- `sample_embedding/calculate_sample_embedding.py`
- `sample_embedding/calculate_multiomics_sample_embedding.py`
- `sample_embedding/embedding_selection.py`
- `parameter_selection/cpu_optimal_resolution.py`
- `parameter_selection/gpu_optimal_resolution.py`
- `parameter_selection/multi_omics_optimal_resolution_cpu.py`
- `parameter_selection/multi_omics_optimal_resolution_gpu.py`
- `parameter_selection/multi_omics_unify_optimal.py`
- The `anndata_sample()` function inside `rna_preprocess_*.py` /
  `atac_preprocess_*.py` (and the `adata_sample.h5ad` write path).

## New API surface

```python
from sample_embedding import compute_sample_embedding

sample_adata = compute_sample_embedding(
    adata,                                 # cell-level AnnData (adata_preprocessed)
    output_dir,
    use_gpu=False,                          # True ⇒ cuml/cupy/harmony.harmonize(use_gpu=True)
    sample_col='sample',
    celltype_col='cell_type',
    cluster_emb_key='X_pca_harmony',       # 'X_lsi_harmony' (ATAC) / 'X_glue' (MO)
    cmd_emb_key=None,                       # defaults to cluster_emb_key + '_nosamp'
    modality_col=None,                      # 'modality' for MO
    batch_col=None,
    medium_K=120, fine_K=300,
    cmd_dim_per_cluster=8,
    use_clr=False, use_cmd=True,
    block_weights=None,                     # None ⇒ inverse-variance auto-derive
    cmd_weight=0.60,
    pca_components=10,                      # default 10 (was 20 in legacy)
    batch_method='harmony',                 # 'harmony' or 'linear'
    save=True, verbose=True, seed=42,
)
```

```python
from parameter_selection.autotune import run_autotune

result = run_autotune(
    adata, output_dir,
    sample_col='sample', celltype_col='cell_type',
    cluster_emb_key='X_pca_harmony',
    cmd_emb_key='X_pca_harmony_nosamp',
    modality_col=None,
    batch_col=None,
    grouping_col=None,                      # e.g. 'sev.level' for supervised proxies
    scoring='auto',                         # gated multi_metric_proxy
    search='bayesian',
    scope='alpha_only',
    alpha_bounds=(0.1, 10.0),
)
# result['sample_adata'], result['best_params'], result['best_score'], result['trace']
```

```python
from sample_trajectory.trajectory_diff_gene import run_trajectory_gam_differential_gene_analysis

run_trajectory_gam_differential_gene_analysis(
    adata,                                  # cell-level adata_preprocessed
    pseudotime_source,
    sample_col='sample',
    celltype_col='cell_type',
    batch_col=None,
    n_features_per_celltype=2000,           # first-round HVG per cell type; None disables
    columns_to_preserve=None,
    ...
)
```

## Config migration table

| Removed (per `{rna|atac|multiomics}_` prefix) | Replaced by |
| --- | --- |
| `sample_hvg_number`, `sample_embedding_dimension`, `harmony_for_proportion`, `preserve_cols_in_sample_embedding` | `sample_embedding_medium_K`, `sample_embedding_fine_K`, `sample_embedding_cmd_dim`, `sample_embedding_use_clr`, `sample_embedding_use_cmd`, `sample_embedding_block_weights`, `sample_embedding_cmd_weight`, `sample_embedding_pca_components`, `sample_embedding_batch_method` |
| `n_expression_components`, `n_proportion_components` (MO) | `sample_embedding_pca_components` (single PC count) |
| `cca_compute_corrected_pvalues`, `cca_coarse_start/end/step`, `cca_fine_range/step` | `autotune_enable`, `autotune_search`, `autotune_scoring`, `autotune_scope`, `autotune_alpha_bounds`, `autotune_n_init`, `autotune_n_iter`, `autotune_grouping_col` |
| `cca_based_cell_resolution_selection` | `autotune_enable` |
| `multiomics_expression_key`, `multiomics_proportion_key` | `multiomics_sample_embedding_key` (default `X_DR_sample`) |
| `multiomics_dimensionality_reduction`, `multiomics_find_optimal_resolution`, `multiomics_optimization_target`, `multiomics_sev_col`, `multiomics_resolution_use_rep`, `multiomics_num_pcs`, `multiomics_visualize_cell_types`, `multiomics_compute_corrected_pvalues`, `multiomics_analyze_modality_alignment` | retained as no-op in the orchestrator signature; the new flow uses `multiomics_derive_sample_embedding` + `multiomics_autotune_*` |

Deprecated keys are still **accepted** by the orchestrator's `wrapper()`
signature so existing YAMLs continue to parse. They are not forwarded to
the modality wrappers.

## Output schema

`<output_dir>/preprocess/adata_preprocessed.h5ad`:
- `.X` — normalized + log1p expression (RNA) or TF-IDF + log1p (ATAC), all genes/features
- `.var['highly_variable']` (RNA) / `.var['HVF']` + `.var['highly_variable']` (ATAC) — HVG flag, no subsetting
- `.layers['counts']` — original raw counts (preserved for DGE)
- `.obsm['X_pca']` / `.obsm['X_lsi']` — embedding on HVG subset
- `.obsm['X_pca_harmony']` (RNA) / `.obsm['X_lsi_harmony']` (ATAC) — Harmony pass 1 (sample-removed)
- `.obsm['X_pca_harmony_nosamp']` (RNA) / `.obsm['X_lsi_harmony_nosamp']` (ATAC) — Harmony pass 2 (sample-preserved; used by CMD)
- Multi-omics: `.obsm['X_glue']` from GLUE (sample-preserved by design)
- `.obs[...]` — full cell-level metadata + `cell_type` after clustering

`<output_dir>/sample_embedding/adata_sample.h5ad`:
- `.X` — `samples × pca_components` ndarray (default 10 PCs)
- `.obs[...]` — per-unit metadata aggregated from cell-level obs
- `.uns['X_DR_sample']` — DataFrame with `PC1..PC{pca_components}` columns
- `.obsm['X_DR_sample']` — same ndarray
- `.uns['sample_embedding_params']` — recipe parameters (K_c, K_med, K_fine,
  weights, etc.) for reproducibility

`<output_dir>/sample_embedding/sample_embedding.csv` — same matrix as CSV.

When autotune runs: also writes `<output_dir>/sample_embedding/autotune_record.json`
with `best_params`, `best_score`, `search`, `scoring`, `scope`, `weights`, `trace`.

## Downstream contract

All downstream modules now read the single key `X_DR_sample` (with legacy
`X_DR_expression` / `X_DR_proportion` as fallback). Function signatures
preserved where possible; the wrapper orchestrator now passes:
- `pseudo_adata = sample_adata` (sample-level, has `X_DR_sample`)
- `adata_cell = adata = adata_preprocessed` (cell-level, all genes, dual-Harmony embeddings)

Trajectory DGE explicitly takes the cell-level `adata`; it builds pseudobulk
internally per cell type with optional Limma batch correction and an
optional first-round HVG per cell type (no double normalization, no second
HVG round).

## Auto-weight schedule

When `block_weights=None` and any of `medium_K` / `fine_K` / the data's
actual `K_c` (cell-type count) change, weights are auto-rescaled via the
inverse-variance schedule:

    w_A1 = √(K_fine / K_c)
    w_A2 = √(K_fine / K_med)
    w_A3 = 1.0
    w_CMD = cmd_weight   (literal; not scaled by K)

Reference values for the published defaults (K_c≈15, K_med=120, K_fine=300):
`[w_A1≈4.47, w_A2≈1.58, w_A3=1.00, w_CMD=0.60]`. If the user passes an
explicit `block_weights` list, it is used as-is.

## Trajectory DGE adaptation

`run_trajectory_gam_differential_gene_analysis(adata, ...)` first calls
`_build_sample_pseudobulk` to materialize a samples × (celltype-gene) AnnData:

1. Aggregate cells per `(sample × celltype)` by mean (no double normalization —
   `adata.X` is already normalized + log1p from preprocessing).
2. Per cell type:
   - Drop NaN-only genes.
   - If `batch_col` is given: Limma correction inside the cell type's pseudobulk
     (reuses `utils.limma.limma`, identical primitive the legacy code used).
   - If `n_features_per_celltype` is set: select top HVGs **once** per cell
     type (no second HVG round across cell types).
3. Concatenate per-celltype HVGs → `samples × Σ_c HVGs(c)` matrix with
   features named `f"{celltype} - {gene}"`.

The existing GAM stack (`prepare_gam_input_data_improved` → `fit_gam_models_for_genes`
→ `calculate_effect_size_and_direction`) is unchanged and operates on this
features-as-`celltype - gene` matrix.

## Known limitations / follow-up

- Autotune `scope` is restricted to `'alpha_only'` in this generalized port.
  The legacy `'k_med_alpha'` scope can be added later if needed.
- `sample_trajectory/multi_omics_CCA_test.py` still includes explicit
  `column="X_DR_proportion"` / `column="X_DR_expression"` call sites for
  its specific RNA/ATAC split-by-modality test; these are not in the
  wrapper flow but will need updating if the function is reactivated.
- Benchmark scripts under `Benchmark_covid/` and `Benchmark_multiomics/`
  still reference `sample_expression_embedding.csv` /
  `sample_proportion_embedding.csv` and `pseudotime_expression.csv` /
  `pseudotime_proportion.csv`. These are external benchmarking tools that
  read saved CSVs; they were intentionally not modified in this refactor.
  The plan includes a follow-up commit for benchmark fairness fixes
  (uniform `StandardScaler`, PC1+PC2 only for CCA, paired-distance scale
  normalization, etc.) — see `claude/sample_embedding_refactor_plan.md`
  benchmark improvements section.
- The `multi_omics_glue.py` gene-activity kNN step was left untouched per
  the user's instruction; it remains controllable via
  `multiomics_run_glue_gene_activity`.

## Validation status

- ✅ All modified files parse cleanly (`python -m py_compile` equivalent).
- ✅ `blocks.derive_weights(15, 120, 300, 0.6) == [4.47, 1.58, 1.0, 0.6]`
  (correct inverse-variance derivation).
- ⏳ End-to-end smoke test on
  `/users/hjiang/GenoDistance/code/config/config_covid_rna.yaml` —
  **not run yet**. Awaiting user authorization.

Canonical smoke-test command:

```bash
python -u SampleDisc.py -m complex --config "/users/hjiang/GenoDistance/code/config/config_covid_rna.yaml" > test.out 2>test.err
```

Expected artifacts after the smoke test:
1. `/dcs07/hongkai/data/harry/result/test/rna/preprocess/adata_preprocessed.h5ad` (single file; no separate `adata_sample.h5ad` / `adata_cell.h5ad`).
2. `/dcs07/hongkai/data/harry/result/test/rna/sample_embedding/adata_sample.h5ad` with `.uns['X_DR_sample']` (DataFrame, samples × 10) and `.obsm['X_DR_sample']`.
3. `/dcs07/hongkai/data/harry/result/test/rna/sample_embedding/sample_embedding.csv`.
4. `variance_explained_X_DR_sample.csv` from dimension association analysis.
5. No `KeyError` for `X_DR_expression` or `X_DR_proportion` anywhere in `test.err`.
