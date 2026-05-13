# Plan: Replace SD's sample embedding with the new composition-and-displacement method (generalized)

## Context

The current sample embedding pipeline in [sample_embedding/](/users/hjiang/GenoDistance/code/sample_embedding/) builds per-(sample × cell type) pseudobulk, runs ComBat/Limma batch correction, applies PCA/LSI, and stores **two** DataFrames — `X_DR_expression` and `X_DR_proportion`. The new method ([/dcs07/hongkai/data/harry/result/cluade_generative/](/dcs07/hongkai/data/harry/result/cluade_generative/)) is composition-based + cross-modality counterfactual displacement (CMD). It must replace the current method and be **generalized** (no dataset hardcoding), with **CPU and GPU versions**, while collapsing the `adata_cell`/`adata_sample` outputs into a single adata.

## Which exact variant we're porting (clarified)

From [WIRE_VARIANTS_REPORT.md](/dcs07/hongkai/data/harry/result/cluade_generative_review/WIRE_VARIANTS_REPORT.md) and [wire_canonical_runs.py](/dcs07/hongkai/data/harry/result/cluade_generative/code/wire_framework/wire_canonical_runs.py):

| Domain | Winning method | Source file | CLR? | Composition weights | CMD weight |
|---|---|---|---|---|---|
| Multi-omics (ENCODE/heart/retina/lutea) | **`wire_singleCMD`** | [wire_multires_cmd.py](/dcs07/hongkai/data/harry/result/cluade_generative/code/wire_framework/wire_multires_cmd.py) with `cmd_resolutions=("coarse",)` | **No** | `[3.0, 1.55, 1.0]` | `0.60` |
| COVID single-omics | **`wire_singleCMD_dualembed`** | [wire_dualembed_covid.py](/dcs07/hongkai/data/harry/result/cluade_generative/code/wire_framework/wire_dualembed_covid.py) | **No** (no `use_clr` arg exists) | `[3.0, 1.55, 1.0]` | `cmd_weight=0.60` |

Both winners use **raw `composition_per_unit(...)` outputs — no CLR transform**. `wire_unified.py` does have a CLR option (default True) but is **not the winning variant**. I will:
- Port the `singleCMD` recipe (composition A1+A2+A3 + CMD-coarse, no CLR by default).
- Keep `use_clr` as an opt-in flag (default `False`) only for completeness, not as the recommended path.
- For single-omics: use the dual-Harmony pattern from `wire_dualembed_covid.py` (Z_clust = sample-removed, Z_cmd = sample-preserved); CMD groups by **batch** (not modality).
- For multi-omics: use the single-embedding pattern from `wire_multires_cmd.py` (Z_clust = Z_cmd = X_glue); CMD groups by **modality**.

## User decisions (locked in)

| Question | Decision |
| --- | --- |
| Embedding interface | **Single unified key** `X_DR_sample`; refactor downstream readers |
| Preprocessing | **Dual-Harmony for RNA and ATAC**; preserve ATAC's distinct flow (TF-IDF / LSI / drop_first / scale_factor). Multi-omics uses `X_glue` directly |
| Single-omics scope | One generic sample-embedding function for RNA, ATAC, and multi-omics (cell-level + cmd embedding keys are parameters); single-omics CMD groups by batch, multi-omics by modality |
| Parameter selection | Port autotune v2 (bayesian × multi_metric_proxy × alpha_only); gate scoring proxies on user config (drop ASW/iLISI when no batch column; drop CCA/SPS/CV-kNN when no grouping label; fall back to fixed defaults if neither) |
| CPU + GPU | Split into [sample_embedding/sample_embedding.py](/users/hjiang/GenoDistance/code/sample_embedding/sample_embedding.py) and [sample_embedding/sample_embedding_gpu.py](/users/hjiang/GenoDistance/code/sample_embedding/sample_embedding_gpu.py) with the same API; **mirror the same library mix the existing preprocessing uses** — `rapids_singlecell` (rsc) + `harmony.harmonize(use_gpu=True)` + `cupy` for GPU; `scanpy` + `harmonypy` + `numpy/sklearn` for CPU |
| Naming | No "WIRE" prefix — `sample_embedding.py` / `sample_embedding_gpu.py` and `compute_sample_embedding()` |
| Remove embedding_selection | Drop [sample_embedding/embedding_selection.py](/users/hjiang/GenoDistance/code/sample_embedding/embedding_selection.py); single embedding ⇒ no selection step needed |
| Single saved adata | Collapse `adata_cell.h5ad` + `adata_sample.h5ad` into one file per pipeline; new filename is **`adata_preprocessed.h5ad`** (clearer than `adata_cell.h5ad` since it carries embeddings + raw counts + HVG flag) |
| Trajectory DGE pseudobulk | Use the existing [compute_pseudobulk_adata](/users/hjiang/GenoDistance/code/sample_embedding/pseudo_adata.py) recipe — per (sample × celltype) mean, optional Limma batch correction per cell type, optional **first-round HVG per cell type** for noise reduction, concatenate across cell types into `samples × (celltype-gene)`. **No double normalization**; **no second HVG round** |
| Benchmark — paired distance | Normalize embeddings to unit Frobenius norm (or per-PC z-score) before computing the MO paired-distance metric — apply uniformly across methods |
| Auto-weight scaling | When `block_weights=None` (default) and user has changed `medium_K` / `fine_K` / `coarse_K`, weights are **auto-rescaled via the inverse-variance schedule** so the relative balance among A1/A2/A3 doesn't drift |
| PC count | Default `pca_components = 10` (was 20). User can override |

## Target architecture

### New module layout

```
code/
├── sample_embedding/
│   ├── sample_embedding.py          ← NEW: CPU compute_sample_embedding() (singleCMD recipe; CLR off by default)
│   ├── sample_embedding_gpu.py      ← NEW: GPU twin (rsc + cupy + GPU harmonize)
│   └── blocks.py                    ← NEW: shared math primitives (soft_assign, composition_per_unit, loo_cmd, build_emb_from_blocks, derive_weights)
├── preparation/
│   ├── rna_preprocess_cpu.py        ← MODIFY: dual-Harmony; single adata_preprocessed.h5ad with HVG flag + raw counts layer
│   ├── rna_preprocess_gpu.py        ← MODIFY: same; keep rsc.pp pipeline (rsc.get.anndata_to_GPU → rsc.pp.normalize_total → rsc.pp.log1p → rsc.pp.pca → harmony.harmonize(use_gpu=True) pass 1 + pass 2)
│   ├── atac_preprocess_cpu.py       ← MODIFY: dual LSI-Harmony; single adata_preprocessed.h5ad (keeps muon.atac.tfidf + ac.tl.lsi + drop_first + tfidf_scale_factor)
│   ├── atac_preprocess_gpu.py       ← MODIFY: same
│   ├── cell_type_cpu.py / _gpu.py   ← MODIFY: single-adata signature (drop anndata_sample arg)
│   ├── multi_omics_preprocess.py    ← MODIFY: output `adata_preprocessed.h5ad`
│   └── multi_omics_cell_type_*.py   ← MODIFY: single-adata signature
├── parameter_selection/
│   └── autotune.py                  ← NEW: bayesian × multi_metric_proxy × alpha_only with adaptive proxy gating
├── wrapper/
│   ├── rna_wrapper.py               ← MODIFY: route to compute_sample_embedding; drop adata_sample
│   ├── atac_wrapper.py              ← MODIFY: same
│   ├── multiomics_wrapper.py        ← MODIFY: route to compute_sample_embedding
│   └── wrapper.py                   ← MODIFY: downstream_analysis() reads adata_preprocessed + sample_adata (X_DR_sample)
├── sample_distance/                 ← MODIFY: single-key DR consumer
├── sample_association/              ← MODIFY: single-key DR consumer
├── sample_trajectory/
│   ├── CCA.py / CCA_test.py / multi_omics_CCA_test.py  ← MODIFY: read X_DR_sample
│   └── trajectory_diff_gene.py      ← MODIFY: build per-celltype pseudobulk from adata_preprocessed (existing recipe; no double normalization)
├── sample_clustering/               ← unchanged (consumes distance matrix or cell-level adata)
└── config/                          ← MODIFY: replace DR params with new method params + autotune params
```

### Single saved adata after preprocessing

One file per pipeline: **`adata_preprocessed.h5ad`** containing:
- `.X` = normalized + log1p expression (RNA) or TF-IDF + log1p (ATAC), **all genes/features**
- `.var['highly_variable']` (RNA) / `.var['HVF']` (ATAC) flag (Seurat v3, per sample)
- `.layers['counts']` = raw counts (preserved for DGE pseudobulk-on-the-fly)
- `.obsm['X_pca_harmony']` (RNA) / `.obsm['X_lsi_harmony']` (ATAC) / `.obsm['X_glue']` (MO) — sample-removed
- `.obsm['X_pca_harmony_nosamp']` / `.obsm['X_lsi_harmony_nosamp']` (sample-preserved); MO reuses `X_glue`
- `.obs` with sample, celltype, batch metadata

PCA / Harmony / clustering operate on the HVG subset via `use_highly_variable=True` or by gating the HVG-mask before PCA. `anndata_sample()` is **deleted**.

### `compute_sample_embedding()` — the new single entry point

CPU at [sample_embedding/sample_embedding.py](/users/hjiang/GenoDistance/code/sample_embedding/sample_embedding.py):

```python
def compute_sample_embedding(
    adata,                          # cell-level AnnData with cluster_emb and cmd_emb in obsm
    output_dir: str,
    sample_col: str,
    celltype_col: str,
    cluster_emb_key: str,           # 'X_pca_harmony' (RNA) / 'X_lsi_harmony' (ATAC) / 'X_glue' (MO)
    cmd_emb_key: Optional[str] = None,   # defaults to cluster_emb_key + '_nosamp' if present, else cluster_emb_key
    modality_col: Optional[str] = None,  # MO ⇒ 'modality'; None ⇒ single-omics, CMD groups by batch
    batch_col: Optional[Union[str, List[str]]] = None,
    medium_K: int = 120, fine_K: int = 300,
    cmd_dim_per_cluster: int = 8,
    use_clr: bool = False,           # OFF by default; matches the winning variant
    use_cmd: bool = True,
    block_weights: Optional[List[float]] = None,
                                     # None ⇒ auto-rescale via inverse-variance schedule:
                                     #   w_A1 = √(K_fine/K_c), w_A2 = √(K_fine/K_med), w_A3 = 1.0,
                                     #   w_CMD = cmd_weight (literal). When user changes any K,
                                     #   composition weights auto-adjust to keep relative balance.
    cmd_weight: float = 0.60,        # default matches wire_singleCMD / wire_singleCMD_dualembed
    pca_components: int = 10,        # default 10
    batch_method: str = 'harmony',   # 'harmony' or 'linear'
    save: bool = True, verbose: bool = True,
    seed: int = 42,
) -> AnnData:                        # sample-level AnnData; .uns['X_DR_sample'] (DataFrame), .obsm['X_DR_sample'] (ndarray)
```

Algorithm (matches `wire_singleCMD` recipe; **no CLR**):
1. **A1** — for each unit, one-hot from `celltype_col`, mean over the unit's cells → raw composition.
2. **A2** — MiniBatchKMeans(K_med) on `cluster_emb_key` cells; Gaussian-RBF soft-assign; mean over unit's cells → raw composition.
3. **A3** — same with K_fine.
4. **CMD-coarse** — per-(group, cluster) leave-one-out displacement using `cmd_emb_key`. `group = modality_col` for MO; `group = batch_col` for single-omics. Per-cluster PCA reduces to `cmd_dim_per_cluster` PCs per cluster; relevance-weighted by `√N_{u,k}`.
5. **Auto weight derivation** (in `blocks.derive_weights`): if `block_weights is None`, compute `[w_A1, w_A2, w_A3, w_CMD]` from the *actual* `K_c` (number of cell-type labels in the data), `K_med`, `K_fine` via the inverse-variance schedule so user-changed cluster counts auto-rescale.
6. **Frobenius stack**: each block centered, scaled to `‖B‖_F = √N · w_b`, concatenated.
7. **PCA** to `pca_components` (default 10).
8. **Composite-batch Harmony** at sample level: composite label = `f"{group}__{within_group_batch}"` (degenerates to group-only if 1:1 with units).

GPU at [sample_embedding/sample_embedding_gpu.py](/users/hjiang/GenoDistance/code/sample_embedding/sample_embedding_gpu.py) — mirrors the same library mix used by [rna_preprocess_gpu.py](/users/hjiang/GenoDistance/code/preparation/rna_preprocess_gpu.py):
- `rapids_singlecell` (rsc) for cluster-on-GPU ergonomics where applicable; `rsc.get.anndata_to_GPU()` / `rsc.get.anndata_to_CPU()` for transfers
- `cuml.cluster.MiniBatchKMeans` for k-means
- `cupy` for pairwise-distance + RBF soft-assign (dominant cost: ~400k cells × 300 anchors)
- `cuml.decomposition.PCA` for the Frobenius-stack PCA
- `harmony.harmonize(..., use_gpu=True)` (the same `harmony` package the preprocessor uses) for sample-level Harmony; fallback to CPU `harmonypy` if not available

Shared primitives live in [sample_embedding/blocks.py](/users/hjiang/GenoDistance/code/sample_embedding/blocks.py); they detect `numpy` vs `cupy` arrays and route to the right backend. Dispatcher in `sample_embedding/__init__.py` exports `compute_sample_embedding` from CPU or GPU module based on `use_gpu`.

Output sample AnnData:
- One row per `sample` for single-omics; one row per `(sample, modality)` for multi-omics
- `.uns['X_DR_sample']` = DataFrame (samples × `pca_components`)
- `.obsm['X_DR_sample']` = ndarray
- Written to `<output_dir>/sample_embedding/adata_sample.h5ad` and `<output_dir>/sample_embedding/sample_embedding.csv`

### Dual-Harmony preprocessor

Both passes use the existing library mix to stay aligned with the rest of the package:

**RNA — GPU** ([rna_preprocess_gpu.py](/users/hjiang/GenoDistance/code/preparation/rna_preprocess_gpu.py)): keep the existing GPU flow (`rsc.get.anndata_to_GPU` → `rsc.pp.normalize_total` → `rsc.pp.log1p` → `rsc.pp.pca`) and run Harmony **twice** via `harmony.harmonize(..., use_gpu=True)`:
- Pass 1: batch keys = `cell_level_batch_key + [sample_col]` → `obsm['X_pca_harmony']`
- Pass 2: batch keys = `cell_level_batch_key` only → `obsm['X_pca_harmony_nosamp']`
- If no extra batch covariate, pass 2 stores raw `X_pca`.
- HVG flag set with `subset=False` so all genes stay in `.X`; raw counts in `.layers['counts']` before normalize/log1p.
- Delete `anndata_sample()` and the `adata_sample.h5ad` write path; return only `adata_preprocessed`.

**RNA — CPU** ([rna_preprocess_cpu.py](/users/hjiang/GenoDistance/code/preparation/rna_preprocess_cpu.py)): same skeleton but with `sc.pp.normalize_total`, `sc.pp.log1p`, `sc.tl.pca`, and `harmonypy.run_harmony` for both passes.

**ATAC — GPU** ([atac_preprocess_gpu.py](/users/hjiang/GenoDistance/code/preparation/atac_preprocess_gpu.py)): preserve `muon.atac.tfidf`, `ac.tl.lsi`, `drop_first_lsi`, `tfidf_scale_factor`. Add pass 2 LSI-Harmony excluding sample → `obsm['X_lsi_harmony_nosamp']`. Use `harmony.harmonize(use_gpu=True)`.

**ATAC — CPU** ([atac_preprocess_cpu.py](/users/hjiang/GenoDistance/code/preparation/atac_preprocess_cpu.py)): same ATAC-specific flow, CPU harmonypy for the two passes.

**Multi-omics** ([multi_omics_preprocess.py](/users/hjiang/GenoDistance/code/preparation/multi_omics_preprocess.py)): write a single `adata_preprocessed.h5ad`. GLUE's `X_glue` serves as both `cluster_emb_key` and `cmd_emb_key`.

### Trajectory DGE — pseudobulk on-the-fly (matches the original recipe)

New helper inside [trajectory_diff_gene.py](/users/hjiang/GenoDistance/code/sample_trajectory/trajectory_diff_gene.py), porting [compute_pseudobulk_adata](/users/hjiang/GenoDistance/code/sample_embedding/pseudo_adata.py)'s recipe:

```python
def _build_sample_pseudobulk(
    adata,                                  # adata_preprocessed: cell-level, .X already normalized + log1p
    sample_col: str,
    celltype_col: str,                      # required: defines the per-celltype layers
    batch_col: Optional[Union[str, List[str]]] = None,   # for Limma correction inside each cell type
    n_features_per_celltype: Optional[int] = 2000,       # first-round HVG per cell type for noise reduction
                                                          # set None to disable HVG
    columns_to_preserve: Optional[List[str]] = None,     # passed through to limma covariate_formula
    verbose: bool = False,
) -> sc.AnnData:                            # samples × (celltype-gene) AnnData
```

Behavior (mirrors [aggregate_pseudobulk](/users/hjiang/GenoDistance/code/sample_embedding/pseudo_adata.py#L207) + [process_celltype_layer](/users/hjiang/GenoDistance/code/sample_embedding/pseudo_adata.py#L277)):
1. **Aggregate cells → (sample × celltype)**: per cell type, mean of cells of that type within each sample → one matrix per celltype, stored as `pseudobulk_adata.layers[celltype]`.
2. **Per cell type processing**:
   - **No re-normalization** — `.X` is already normalized + log1p from preprocessing.
   - Remove NaN genes.
   - If `batch_col` is given: Limma correction within this celltype's pseudobulk (uses [utils/limma.py](/users/hjiang/GenoDistance/code/utils/limma.py)). Keeps any covariates listed in `columns_to_preserve` in the design matrix.
   - **First-round HVG per cell type** (`n_features_per_celltype` features) for noise reduction. **Skip** if `n_features_per_celltype is None` or larger than n_genes.
3. **Concatenate across cell types** → `samples × Σ_c HVGs(c)` matrix with features named `f"{celltype} - {gene}"`. **No second HVG round.**
4. Return `sc.AnnData` (samples × concatenated features) with sample metadata in `.obs`.

The existing GAM stack (`prepare_gam_input_data_improved` → `fit_gam_models_for_genes` → `calculate_effect_size_and_direction`) operates on this samples × features matrix unchanged — it already fits one GAM per feature.

Function signature change in [run_trajectory_gam_differential_gene_analysis](/users/hjiang/GenoDistance/code/sample_trajectory/trajectory_diff_gene.py#L758): first arg becomes `adata` (was `pseudobulk_adata`); add kwargs `celltype_col`, `batch_col`, `n_features_per_celltype`, `columns_to_preserve`. Callers in [wrapper/wrapper.py:282-348](/users/hjiang/GenoDistance/code/wrapper/wrapper.py#L282-L348) updated accordingly.

### Parameter selection — `parameter_selection/autotune.py`

Port [wire_autotune_dualembed_v2.py](/dcs07/hongkai/data/harry/result/cluade_generative/code/parameter_selection/wire_autotune_dualembed_v2.py) with path/dataset hardcoding stripped out:
- `build_blocks(adata, sample_col, celltype_col, cluster_emb_key, cmd_emb_key, modality_col, batch_col, grouping_col, medium_K, fine_K, ...)` replaces the COVID-specific block builder.
- `build_emb_from_blocks(blocks, weights, ...)` is module-shared with `sample_embedding.py` (lives in `blocks.py`).
- `make_scorer(name, blocks, has_batch, has_grouping)` adapts the ensemble:
  - `has_batch=False` ⇒ drop `ilisi_batch`, `neg_asw_batch` from `multi_metric_proxy`.
  - `has_grouping=False` ⇒ drop `cca`, `sps`, `cv_knn_severity`, `pseudotime_spearman`.
  - both absent ⇒ short-circuit autotune (warn + use fixed defaults `[3.0, 1.55, 1.0, 0.60]`).
- Default search: `bayesian` (Matern kernel GP + EI), scope: `alpha_only`, bounds: `(0.1, 10.0)`, n_init=5, n_iter=10.
- Public entry: `run_autotune(adata, output_dir, scoring='auto', search='bayesian', scope='alpha_only', **kwargs) -> dict` returns best params + final sample AnnData.

### Downstream consumer updates (single-key)

| Module | Change | File |
| --- | --- | --- |
| sample_association | `_available_embeddings()` returns `['X_DR_sample']`; outer loop runs once | [association.py:101-110](/users/hjiang/GenoDistance/code/sample_association/association.py#L101-L110) |
| sample_distance | Collapse `get_best_expression_dr_key` / `get_best_proportion_dr_key` into `get_best_sample_dr_key()`; `sample_distance_vector` returns `{'sample_DR': ...}` | [sample_distance.py:498-622](/users/hjiang/GenoDistance/code/sample_distance/sample_distance.py#L498-L622) |
| sample_trajectory CCA | Default `column='X_DR_sample'` | [CCA.py](/users/hjiang/GenoDistance/code/sample_trajectory/CCA.py), [CCA_test.py](/users/hjiang/GenoDistance/code/sample_trajectory/CCA_test.py), [multi_omics_CCA_test.py](/users/hjiang/GenoDistance/code/sample_trajectory/multi_omics_CCA_test.py) |
| sample_trajectory DGE | Accept `adata_preprocessed` + celltype/batch cols; pseudobulk-on-the-fly with the existing recipe | [trajectory_diff_gene.py:758-965](/users/hjiang/GenoDistance/code/sample_trajectory/trajectory_diff_gene.py#L758-L965) |
| wrapper.downstream_analysis | Pass `adata` (cell-level, all genes) where the old code passed `adata_sample`; rename `pseudo_adata=` → `sample_adata=`; drop `adata_sample=` | [wrapper.py:63-529](/users/hjiang/GenoDistance/code/wrapper/wrapper.py#L63-L529) |
| sample_clustering proportion_test / cluster_DGE | Already cell-level; route `adata_preprocessed` (no change to its own code) | [proportion_test.py](/users/hjiang/GenoDistance/code/sample_clustering/proportion_test.py), [wrapper.py:355-411](/users/hjiang/GenoDistance/code/wrapper/wrapper.py#L355-L411) |

### Wrapper signature changes

```python
# rna_wrapper.py (mirror for atac_wrapper, multiomics_wrapper)
def rna_wrapper(...):
    # 1. preprocess → single adata_preprocessed (raw counts in .layers; dual Harmony in .obsm)
    adata = preprocess_func(...)        # was: (adata_cell, adata_sample)

    # 2. cell typing on adata_preprocessed
    adata = cell_types_func(adata, ...)

    # 3. sample embedding (singleCMD recipe; CLR off by default; weights auto-scale on K)
    from sample_embedding import compute_sample_embedding   # dispatches CPU/GPU on use_gpu
    sample_adata = compute_sample_embedding(
        adata, output_dir,
        sample_col=..., celltype_col=..., batch_col=...,
        cluster_emb_key='X_pca_harmony',           # ATAC: 'X_lsi_harmony'; MO: 'X_glue'
        cmd_emb_key='X_pca_harmony_nosamp',        # ATAC: 'X_lsi_harmony_nosamp'; MO: 'X_glue'
        modality_col=None,                         # MO: 'modality'
        medium_K=cfg.sample_embedding_medium_K, fine_K=cfg.sample_embedding_fine_K,
        block_weights=cfg.sample_embedding_block_weights,
        cmd_weight=cfg.sample_embedding_cmd_weight,
        use_clr=cfg.sample_embedding_use_clr,      # default False (matches winning variant)
        use_cmd=cfg.sample_embedding_use_cmd,
        pca_components=cfg.sample_embedding_pca_components,   # default 10
        save=True, verbose=verbose, seed=42,
    )

    # 4. optional autotune
    if cfg.autotune_enable:
        result = run_autotune(adata, output_dir,
                              scoring=cfg.autotune_scoring, search=cfg.autotune_search,
                              scope=cfg.autotune_scope, ...)
        sample_adata = result['sample_adata']

    return {'adata': adata, 'sample_adata': sample_adata, 'status_flags': ...}
```

Drops `adata_sample`; renames `pseudo_adata` → `sample_adata`. The orchestrator at [wrapper.py:1139-1543](/users/hjiang/GenoDistance/code/wrapper/wrapper.py#L1139-L1543) updates keyword names everywhere they're forwarded into `downstream_analysis()`.

### Config rewrite

For each pipeline prefix `{rna|atac|multiomics}_`, **remove**:
- `sample_hvg_number`, `sample_embedding_dimension`, `harmony_for_proportion`, `preserve_cols_in_sample_embedding`
- `n_expression_components`, `n_proportion_components` (MO)
- `cca_coarse_start/end/step`, `cca_fine_range/step`, `cca_compute_corrected_pvalues`
- `expression_key`, `proportion_key` (MO viz; collapse into `sample_embedding_key`)

**Add**:
- `sample_embedding_medium_K` (120), `sample_embedding_fine_K` (300), `sample_embedding_cmd_dim` (8)
- `sample_embedding_use_clr` (**false** — winning default), `sample_embedding_use_cmd` (true)
- `sample_embedding_block_weights` (null → auto-rescale via inverse-variance schedule using actual K values)
- `sample_embedding_cmd_weight` (0.60)
- `sample_embedding_pca_components` (**10**), `sample_embedding_batch_method` ("harmony")
- `sample_embedding_key` (default "X_DR_sample")
- `dge_pseudobulk_celltype_col` (default = `celltype_col`), `dge_pseudobulk_batch_col` (default = `batch_col`), `dge_pseudobulk_n_features_per_celltype` (2000, null disables), `dge_pseudobulk_columns_to_preserve` (null)
- `autotune_enable` (false), `autotune_search` ("bayesian"), `autotune_scoring` ("auto"), `autotune_scope` ("alpha_only")
- `autotune_alpha_bounds` (`[0.1, 10.0]`), `autotune_n_init` (5), `autotune_n_iter` (10)

The wrapper raises a clear deprecation error for any removed key.

### Files to delete (after refactor is validated)

- [sample_embedding/DR.py](/users/hjiang/GenoDistance/code/sample_embedding/DR.py)
- [sample_embedding/pseudo_adata.py](/users/hjiang/GenoDistance/code/sample_embedding/pseudo_adata.py)
- [sample_embedding/pseudo_adata_linux.py](/users/hjiang/GenoDistance/code/sample_embedding/pseudo_adata_linux.py)
- [sample_embedding/multi_omics_pseudobulk_cpu.py](/users/hjiang/GenoDistance/code/sample_embedding/multi_omics_pseudobulk_cpu.py)
- [sample_embedding/calculate_sample_embedding.py](/users/hjiang/GenoDistance/code/sample_embedding/calculate_sample_embedding.py)
- [sample_embedding/calculate_multiomics_sample_embedding.py](/users/hjiang/GenoDistance/code/sample_embedding/calculate_multiomics_sample_embedding.py)
- [sample_embedding/embedding_selection.py](/users/hjiang/GenoDistance/code/sample_embedding/embedding_selection.py)
- [parameter_selection/cpu_optimal_resolution.py](/users/hjiang/GenoDistance/code/parameter_selection/cpu_optimal_resolution.py)
- [parameter_selection/gpu_optimal_resolution.py](/users/hjiang/GenoDistance/code/parameter_selection/gpu_optimal_resolution.py)
- [parameter_selection/multi_omics_optimal_resolution_cpu.py](/users/hjiang/GenoDistance/code/parameter_selection/multi_omics_optimal_resolution_cpu.py)
- [parameter_selection/multi_omics_optimal_resolution_gpu.py](/users/hjiang/GenoDistance/code/parameter_selection/multi_omics_optimal_resolution_gpu.py)
- [parameter_selection/multi_omics_unify_optimal.py](/users/hjiang/GenoDistance/code/parameter_selection/multi_omics_unify_optimal.py)
- The `anndata_sample()` function inside `rna_preprocess_*.py` / `atac_preprocess_*.py`

## Implementation order

1. **Block primitives + CPU `sample_embedding.py`**: port `soft_assign`, `composition_per_unit`, `loo_cmd` (MO + single-omics-by-batch variants), `build_emb_from_blocks`, `derive_weights` into `sample_embedding/blocks.py`; build CPU entry on top. Match `wire_singleCMD` defaults (no CLR, weights `[3.0, 1.55, 1.0, 0.60]`, K_med=120, K_fine=300, pca_components=10).
2. **GPU `sample_embedding_gpu.py`**: same library mix as `rna_preprocess_gpu.py` — rsc + cupy + `harmony.harmonize(use_gpu=True)`.
3. **Preprocessing refactor**: collapse to `adata_preprocessed.h5ad`; dual-Harmony / dual-LSI-Harmony; preserve `.layers['counts']` + `highly_variable` flag.
4. **Autotune port** at `parameter_selection/autotune.py` with adaptive scorer gating.
5. **Downstream refactor**: single-key migration; pseudobulk-on-the-fly with celltype + Limma + per-celltype-HVG in trajectory_diff_gene (no double normalization, no second HVG round).
6. **Wrapper + config rewrite**: route through the new entries; deprecate old keys with clear errors.
7. **Delete old files** only after all callers are migrated.
8. **Benchmark improvements** (separate commit) — see below.
9. **Generate update report** — see below.

## Critical files

| File | Action |
| --- | --- |
| [sample_embedding/sample_embedding.py](/users/hjiang/GenoDistance/code/sample_embedding/sample_embedding.py) | **Create** — CPU `compute_sample_embedding()` (singleCMD recipe) |
| [sample_embedding/sample_embedding_gpu.py](/users/hjiang/GenoDistance/code/sample_embedding/sample_embedding_gpu.py) | **Create** — GPU twin |
| [sample_embedding/blocks.py](/users/hjiang/GenoDistance/code/sample_embedding/blocks.py) | **Create** — soft_assign, composition_per_unit, loo_cmd, build_emb_from_blocks, derive_weights |
| [preparation/rna_preprocess_cpu.py](/users/hjiang/GenoDistance/code/preparation/rna_preprocess_cpu.py) / [rna_preprocess_gpu.py](/users/hjiang/GenoDistance/code/preparation/rna_preprocess_gpu.py) | **Modify** — single adata_preprocessed.h5ad; dual-Harmony |
| [preparation/atac_preprocess_cpu.py](/users/hjiang/GenoDistance/code/preparation/atac_preprocess_cpu.py) / [atac_preprocess_gpu.py](/users/hjiang/GenoDistance/code/preparation/atac_preprocess_gpu.py) | **Modify** — single adata_preprocessed.h5ad; dual LSI-Harmony |
| [preparation/cell_type_cpu.py](/users/hjiang/GenoDistance/code/preparation/cell_type_cpu.py) / [cell_type_gpu.py](/users/hjiang/GenoDistance/code/preparation/cell_type_gpu.py) / multi_omics_cell_type_*.py | **Modify** — single-adata signature |
| [preparation/multi_omics_preprocess.py](/users/hjiang/GenoDistance/code/preparation/multi_omics_preprocess.py) | **Modify** — output `adata_preprocessed.h5ad` |
| [parameter_selection/autotune.py](/users/hjiang/GenoDistance/code/parameter_selection/autotune.py) | **Create** — bayesian × multi_metric_proxy × alpha_only |
| [wrapper/rna_wrapper.py](/users/hjiang/GenoDistance/code/wrapper/rna_wrapper.py) / [atac_wrapper.py](/users/hjiang/GenoDistance/code/wrapper/atac_wrapper.py) / [multiomics_wrapper.py](/users/hjiang/GenoDistance/code/wrapper/multiomics_wrapper.py) / [wrapper.py](/users/hjiang/GenoDistance/code/wrapper/wrapper.py) | **Modify** — single adata; new entry; rename `pseudo_adata` → `sample_adata`; drop `adata_sample` |
| [sample_association/association.py](/users/hjiang/GenoDistance/code/sample_association/association.py) | **Modify** — single-key lookup |
| [sample_distance/sample_distance.py](/users/hjiang/GenoDistance/code/sample_distance/sample_distance.py) | **Modify** — single-key DR; collapse `get_best_*_dr_key` |
| [sample_trajectory/CCA.py](/users/hjiang/GenoDistance/code/sample_trajectory/CCA.py), [CCA_test.py](/users/hjiang/GenoDistance/code/sample_trajectory/CCA_test.py), [multi_omics_CCA_test.py](/users/hjiang/GenoDistance/code/sample_trajectory/multi_omics_CCA_test.py) | **Modify** — default column to `X_DR_sample` |
| [sample_trajectory/trajectory_diff_gene.py](/users/hjiang/GenoDistance/code/sample_trajectory/trajectory_diff_gene.py) | **Modify** — pseudobulk-on-the-fly with celltype + Limma + per-celltype HVG (no double normalization, no second HVG round) |
| [config/config.yaml](/users/hjiang/GenoDistance/code/config/config.yaml) + dataset YAMLs in [config/](/users/hjiang/GenoDistance/code/config/) | **Modify** — new keys |

Reused existing utilities:
- [utils/limma.py](/users/hjiang/GenoDistance/code/utils/limma.py) — for trajectory DGE pseudobulk batch correction
- [utils/safe_save.py](/users/hjiang/GenoDistance/code/utils/safe_save.py), [utils/random_seed.py](/users/hjiang/GenoDistance/code/utils/random_seed.py), [utils/merge_sample_meta.py](/users/hjiang/GenoDistance/code/utils/merge_sample_meta.py)
- Original pseudobulk recipe at [pseudo_adata.py](/users/hjiang/GenoDistance/code/sample_embedding/pseudo_adata.py) — port `aggregate_pseudobulk` + `process_celltype_layer` logic into `_build_sample_pseudobulk`, omitting the double normalization and second HVG round
- Existing GPU library mix from [rna_preprocess_gpu.py](/users/hjiang/GenoDistance/code/preparation/rna_preprocess_gpu.py) (rsc + harmony.harmonize) and ATAC's [atac_preprocess_gpu.py](/users/hjiang/GenoDistance/code/preparation/atac_preprocess_gpu.py) (muon.atac.tfidf + ac.tl.lsi)

Source code being ported from:
- Composition primitives + `loo_cmd` + `build_emb_from_blocks` from [wire_multires_cmd.py](/dcs07/hongkai/data/harry/result/cluade_generative/code/wire_framework/wire_multires_cmd.py) (singleCMD MO) and [wire_dualembed_covid.py](/dcs07/hongkai/data/harry/result/cluade_generative/code/wire_framework/wire_dualembed_covid.py) (singleCMD_dualembed COVID); see [wire_unified.py:64-169](/dcs07/hongkai/data/harry/result/cluade_generative/code/wire_framework/wire_unified.py#L64-L169) for the shared primitives
- Inverse-variance weight schedule from [wire_autotune_dualembed_v2.py:68-94 (derive_weights)](/dcs07/hongkai/data/harry/result/cluade_generative/code/parameter_selection/wire_autotune_dualembed_v2.py#L68-L94)
- Composite-batch label builder from [common.py:325-361](/dcs07/hongkai/data/harry/result/cluade_generative/code/wire_framework/common.py#L325-L361)
- Dual-Harmony pattern from [rna_preprocess_dualharmony_gpu.py:82-115](/dcs07/hongkai/data/harry/result/cluade_generative/code/wire_framework/preparation/rna_preprocess_dualharmony_gpu.py#L82-L115)
- Autotune scorers + search from [wire_autotune_dualembed_v2.py:198-430](/dcs07/hongkai/data/harry/result/cluade_generative/code/parameter_selection/wire_autotune_dualembed_v2.py#L198-L430)

## Benchmark improvements (separate commit)

| # | Issue | Current location | Fix |
| --- | --- | --- | --- |
| 1 | `StandardScaler` applied per-method, post-hoc — penalizes/rewards methods unevenly based on natural PC scales | [Benchmark_covid/customized_benchmark.py:229-230](/users/hjiang/GenoDistance/code/Benchmark_covid/customized_benchmark.py#L229-L230) | Wrap scaling in a single `_normalize_embedding(emb, scheme)` helper applied identically to every method; record the scheme in the summary CSV |
| 2 | Fresh PCA reducing >2D embeddings to 2D before CCA; ≤2D used as-is — destroys information unevenly | [Benchmark_multiomics/benchmark_heart.py:826-827](/users/hjiang/GenoDistance/code/Benchmark_multiomics/benchmark_heart.py#L826-L827), [benchmark_eye.py:575](/users/hjiang/GenoDistance/code/Benchmark_multiomics/benchmark_eye.py#L575) | Match [common.py:158-166](/dcs07/hongkai/data/harry/result/cluade_generative/code/wire_framework/common.py#L158-L166): use PC1+PC2 of the existing embedding for CCA; mark as N/A if <2 PCs available |
| 3 | Anchor-only ANOVA vs cross-batch k-NN gap operate on different populations | [Benchmark_covid/customized_benchmark.py:101-250](/users/hjiang/GenoDistance/code/Benchmark_covid/customized_benchmark.py#L101-L250) | One consistent population per metric: restrict k-NN to anchor batch OR extend ANOVA to cross-batch pool. Apply uniformly |
| 4 | **MO paired distance is scale-sensitive** — methods with tiny PC norms get artificially small distances, huge norms get huge distances | [Benchmark_multiomics/benchmark_ENCODE.py](/users/hjiang/GenoDistance/code/Benchmark_multiomics/benchmark_ENCODE.py) + other `benchmark_*` paired-distance call sites | Normalize embeddings to unit Frobenius norm (or per-PC z-score across samples) **before** computing paired distance, identically across methods. Record the normalization in the summary |

Minor cleanup:
- Hardcoded `"...sample_metadata_fixed.csv"` ([Benchmark_covid/benchmark_ENCODE.py:653](/users/hjiang/GenoDistance/code/Benchmark_covid/benchmark_ENCODE.py#L653)) → CLI/config arg
- Case-insensitive sample matching ([1M_blood/Benchmark_1M-scBloodNL/common_io.py:69-71](/users/hjiang/GenoDistance/1M_blood/Benchmark_1M-scBloodNL/common_io.py#L69-L71)) → keep, but emit a stderr count of dropped samples

## Post-implementation deliverable: update report

After all code modifications are complete, write a detailed report at `/users/hjiang/GenoDistance/code/claude/refactor_report.md` covering:
- **What changed**: per-file diff summary (one paragraph per file), with rationale
- **New API surface**: exported function signatures (`compute_sample_embedding`, `run_autotune`, `_build_sample_pseudobulk`)
- **Config migration table**: every removed key with its new replacement(s)
- **Output schema**: what files preprocessing/sample-embedding write now (`adata_preprocessed.h5ad`, `sample_embedding/adata_sample.h5ad`), and their internal structure (`.X`, `.obs`, `.obsm`, `.uns`, `.layers`, `.var` keys)
- **Downstream contract**: which DR key each downstream module reads now; function signature changes (`pseudo_adata` → `sample_adata`; `pseudobulk_adata` → `adata` in trajectory DGE)
- **Auto-weight schedule**: explanation of how block_weights auto-rescale on K
- **Trajectory DGE adaptation**: pseudobulk-on-the-fly recipe with celltype + Limma + per-celltype HVG (no double normalization, no second HVG round)
- **Benchmark fixes** (if shipped in the same pass): per-issue before/after summary
- **Known limitations**: anything left for follow-up
- **Validation status**: which smoke tests have run and what they confirmed

## Verification

Smoke test (only run **after the user authorizes**):

```bash
python -u SampleDisc.py -m complex --config "/users/hjiang/GenoDistance/code/config/config_covid_rna.yaml" > test.out 2>test.err
```

- Test config: [/users/hjiang/GenoDistance/code/config/config_covid_rna.yaml](/users/hjiang/GenoDistance/code/config/config_covid_rna.yaml) (RNA pipeline, COVID dataset)
- Output dir: `/dcs07/hongkai/data/harry/result/test`
- Input data: `/dcl01/hongkai/data/data/hjiang/Data/test_RNA.h5ad` (per the config); user mentions `/dcl01/hongkai/data/data/hjiang/Data/tutorial_test_data` as the canonical location for tutorial test files
- **The config will need its DR-param keys replaced** with the new `sample_embedding_*` and `autotune_*` keys per the rewrite section above; do that before running.
- Inspect `test.out` for the sample-embedding completion log and `test.err` for any deprecation errors from old config keys.

What the smoke test should produce:
1. Single `adata_preprocessed.h5ad` under `/dcs07/hongkai/data/harry/result/test/rna/preprocess/` (no `adata_sample.h5ad`, no `adata_cell.h5ad`).
2. `sample_embedding/adata_sample.h5ad` with `.uns['X_DR_sample']` (DataFrame, samples × 10) and `.obsm['X_DR_sample']`.
3. `sample_embedding/sample_embedding.csv`.
4. Trajectory analysis succeeds against `X_DR_sample`; dimension association writes `variance_explained_X_DR_sample.csv`; phenotype prediction succeeds.
5. No KeyError on `X_DR_expression` / `X_DR_proportion` anywhere.

**Important:** Do not run the smoke test until explicitly authorized.
