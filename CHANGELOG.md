# Changelog

## 0.9.0

**Trajectory DGE: batch as a covariate in the GAM instead of limma pre-correction**

- With `batch_col` set, the pseudobulk values passed to the GAM are no longer limma-corrected. The
  batch enters the model as an unpenalised categorical covariate (expression ~ s(pseudotime) +
  batch; one-hot dummies, `lam=0`), and the p-value tests the pseudotime term only.
- The variance-ratio gene filter keeps its definition (residual after a batch regression on the
  retained groups, same genes pass). A batch left with a single retained sample is kept instead of
  set to NaN, and its warning is removed.
- Effect size and direction are computed from the pseudotime term's partial dependence, so batch
  differences no longer inflate the range or flip the direction. Without covariates the numbers are
  unchanged. Fitted-curve plots draw the batch-adjusted curve.
- Fixed a crash with covariates: the one-hot design was boolean (object dtype), which broke the
  effect-size calculation; the design is now float.

## 0.8.0

**Trajectory DGE: variance-ratio gene filter, minimum cells per group, zeros fitted**

- The "top 2,000 HVG per cell type" first round is replaced by a variance-ratio filter. Within each
  cell type a gene is tested when its between-sample variance (residual variance after the limma
  batch correction) is at least `min_variance_ratio = 2.0` times the variance expected from
  sampling the cells of each group alone (per-group within-cell variance / n cells, with a
  leverage correction for the batch regression). The ratio is unit-free and does not use
  pseudotime. A cell type with no residual degrees of freedom is skipped with a warning.
- A sample x cell-type group with fewer than `min_cells_per_group = 10` cells (at least 2) is
  missing (NaN), not zero, and is dropped before limma, which now runs on the retained groups
  only. A batch left with a single retained sample is set to NaN (limma would fit it exactly) and
  warned about per cell type.
- The per-gene "value != 0" mask is removed everywhere (GAM fit, effect size, plots). Each gene is
  fitted on all samples with a finite value, true zeros included; NaN is no longer turned into 0.
  Previously exact zeros survived mostly in the limma reference batch, so which samples a gene was
  fitted on depended on the batch layout.
- `n_features_per_celltype` / `dge_pseudobulk_n_features_per_celltype` are replaced by
  `min_cells_per_group` / `min_variance_ratio` (`dge_pseudobulk_min_cells_per_group`,
  `dge_pseudobulk_min_variance_ratio` in `downstream_analysis`). They are not `wrapper()` / YAML keys.
  Trajectory gene sets, and the TF motif analysis built on them, change.

## 0.7.0

**Batch columns at two levels**

- New default batch keys `rna_batch_col`, `atac_batch_col` (and the existing `multiomics_batch_col`),
  used at both levels unless a level is set. Cell level (`*_cell_level_batch_key`, new
  `multiomics_cell_level_batch_key`) corrects `Z_comp` and `Z_rmd`. Sample level
  (`*_sample_level_batch_col`, new `multiomics_sample_level_batch_col`) defines the RMD reference
  group, the sample-level Harmony covariates, the autotune batch term and the downstream DE batch.
  A level left `null` inherits the default; `[]` turns it off. The four new keys are optional, so
  existing configs validate and run as before.
- **The RMD reference group now uses every sample-level batch column.** Single-omics used only
  the first column (`[batch, sex]` and `[sex, batch]` gave different embeddings). It is now the
  joint label of all columns. Multi-omics used modality only; it is now modality crossed with the
  sample-level batch columns.
- **Autotune matches the final embedding.** Its batch term uses the same joint label (crossed
  with modality for multi-omics, where it used batch alone), and its sample-level Harmony now
  uses every batch column (it used only the first).
- Multi-omics sample-level Harmony with two or more batch columns now includes modality.
- A warning names reference groups that hold a single unit; such a unit has no leave-one-out
  reference and borrows the nearest other group.
- Results are bit-identical to 0.6.1 for no batch column or one column in single-omics, and
  for no batch column in multi-omics. They change for several sample-level columns, and for
  multi-omics with a batch column.

**Phenotype prediction: MLP by default, KNN for small cohorts**

- `predict_phenotype` takes `model="mlp"` (default) or `"knn"`.
  - `"mlp"`: the paper's classifier. Standardized PCs into a multilayer perceptron (one
    hidden layer of 100 ReLU units, Adam, L2 weight penalty 10, seeded by `random_state`);
    nothing is tuned. With the paper's settings it reproduces the Sound Life out-of-fold
    predictions and donor-level AUCs exactly.
  - `"knn"`: vote (or mean) of the `n_neighbors` nearest training samples (default 5,
    capped at the training-split size) on the raw PCs, as in `transfer_labels`. It fits no
    parameters, for cohorts of tens of samples.
- The logistic / ridge model with the inner-CV penalty (0.5.0-0.6.1) is removed, and with it
  the `C` / `alpha` entries of `metrics`. Folds, grouping, seeding and outputs are unchanged;
  `metrics` gains `model`, and `n_neighbors` for knn.
- Tests: `tests/test_sample_prediction.py`.

## 0.6.1

Reproducible runs, faster trajectory DGE, and fixes for the remaining known issues. Config
files are unchanged and validate as before.

**Reproducibility**

- Rerunning the same input on the same kind of machine (CPU type and CPU count) now gives
  bit-identical results, whichever way the package was installed. Results still differ
  slightly between CPU generations (for example AVX2 vs AVX-512), because the
  floating-point arithmetic differs.
  - **harmony-pytorch 0.1.8 is now a core dependency**, so every install, pip or conda,
    runs cell-level Harmony with the same implementation. harmonypy remains the fallback.
  - The sample-level Harmony (harmonypy), the RNA PCA and the ATAC LSI run single-threaded.
    Multithreaded BLAS made them differ between identical runs. The extra time on the demo
    data is negligible.
- **Trajectory DGE p-values no longer depend on the number of workers.** One pyGAM term
  object was shared across genes and kept the spline knots of the first fit, so the
  serial run and `n_jobs=8` disagreed on about 300 of 10,000 genes (p-value differences
  up to 0.26). The terms are now built for each gene. On the demo, no gene changes
  significance.

**Performance**

- Each trajectory-DGE (GAM) worker uses one BLAS thread. With MKL (the conda environments)
  every worker used to start as many threads as the job had CPUs; on AMD CPUs this made
  the stage up to 20× slower (about 220 s → 12 s on the demo).
- Worker pools (GAM, k-means threads) are capped at the Slurm allocation.
- The cell-typing step saves its h5ad gzip-compressed (RNA demo 718 MB → 208 MB, values
  identical).
- RAISIN no longer keeps dense copies of the expression matrix after ComBat.

**Fixes**

- **Merging cell types now always gives the requested number of types.** With scipy 1.14
  (the version in the conda environments), `fcluster(..., criterion="maxclust")` can return
  fewer clusters than asked for. On the ATAC demo, 9 Leiden clusters merged to 7 types
  instead of 8. Results are unchanged wherever the old merge already gave the requested
  number, as in the RNA demo.
- **`build_sample_adata` keeps only sample-level columns** (constant within each sample,
  ignoring NaN), and logs which per-cell columns it drops. It used to copy the first
  cell's value of every column, so dimension association tested per-cell columns such as
  `n_genes` or `celltype` as if they were sample covariates.
- **RAISIN with `.raw`:** the ComBat correction was discarded (same genes) or crashed
  (more genes in `.raw`). The corrected values are now used.
- **`cell_types_atac`** no longer fails with `NameError` when merging clusters.
- **Gene activity (CPU path)** no longer fails on a missing import.
- **Multi-omics:**
  - `multiomics_target_modality` / `multiomics_color_col` now produce the
    modality-highlight plots.
  - `rna_file` / `atac_file` are no longer required when starting from a pre-integrated
    file.
- **GPU preprocessing** defaults `sample_level_batch_key` to `None`, as on CPU, so data
  without a `batch` column no longer fails.
- **`batch_method`** must be `"harmony"`, `"linear"` or `"none"`. Other values raise
  instead of silently running Harmony.
- **Warnings and messages:**
  - A warning when existing cell-type labels will be merged to fewer
    `*_n_target_cell_clusters`.
  - Importing optional modules no longer switches off Python warnings.
  - No "CuPy not available" message on import.
- `uns['sample_embedding_params']['pca_components']` records the number of PCs used (at
  most units − 1); the requested value is in `pca_components_requested`.
- CCA resolution-selection p-values use `(1 + k) / (N + 1)`, like the CCA p-value.
- The requirement on `python-igraph` is now `igraph`, the name conda-forge registers, so
  `pip check` passes in the conda environments.
- Tests: `tests/test_061_fixes.py`.

## 0.6.0

One environment for the whole package, and fixes for every bug the rewritten tutorials hit.
Config files are unchanged: every existing config validates as before. Results on the
default configs are unchanged except where a fix below says otherwise.

**Installation**

- **One environment runs everything.** `environment-cpu.yml` now installs the core pipeline,
  scGLUE 0.3.2 (with the `bedtools` binary), harmony-pytorch, MaxFuse 0.0.2 and SampleDisco
  itself. `environment-gpu.yml` is the same plus RAPIDS 24.12: create it, then run
  `pip install --no-deps rapids-singlecell==0.13.1 sampledisco==0.6.0`. See `INSTALL.md`.
- **New pip extras:** `[glue]` (alias `[multiomics]`), `[maxfuse]` and `[all]`.
  `pip install "sampledisco[all]"` plus `conda install -c bioconda bedtools` gives the same
  feature set without an env file. The extras pin what the upstream packages leave out:
  scGLUE 0.3.2 needs `anndata<0.11` and `torch<2.7`, and MaxFuse 0.0.2 needs `anndata<0.13`.
- **`pynndescent<0.7`** is now a dependency. pynndescent 0.7 (and umap-learn 0.6) break
  `scanpy.pp.neighbors` on more than about 8,000 cells
  (`'csr_array' object has no attribute 'getnnz'`), so a fresh `pip install sampledisco`
  0.5.0 failed on real data.
- The unused tree-clustering dependencies (biopython, scikit-bio, dendropy) are gone from
  the env files. The project URLs now point to the SampleDisco repository.

**Fixes**

- **`from sampledisco import wrapper` works.** It used to return the `sampledisco.wrapper`
  subpackage, so `wrapper(**config)` raised `TypeError: 'module' object is not callable`.
  The subpackage is now callable. Existing import paths work unchanged, including
  `from sampledisco.wrapper.wrapper import wrapper`.
- **Multi-omics no longer overwrites its input file.** With `multiomics_integration: false`,
  the wrapper used to write cell types and the sample embedding back into
  `multiomics_integrated_h5ad_path`, the user's input. It now writes to
  `<output_dir>/multiomics/preprocess/adata_sample.h5ad` (the same file the integration
  route writes), and reads that file first on later runs.
- **Embedding keys:** passing `comp_emb_key="Z_comp"` or `rmd_emb_key="Z_rmd"` explicitly
  now falls back to the legacy alias (`Z_clust`, `Z_cmd`, …) with a `FutureWarning`, as
  automatic resolution already did. This fixes the demo `test_multiomics_integrated.h5ad`.
- **Cell-type labels after cell typing:** once the wrapper's cell-typing stage has run, the
  sample embedding and downstream stages use the labels it wrote (`obs['cell_type']`).
  Before, a different `*_celltype_col` together with `*_existing_cell_types: true` made the
  embedding silently use the original, unmerged labels.
- **Custom sample column plus a sample-metadata CSV:** merging the metadata no longer
  renames the sample column away, which caused `KeyError: '<sample_col>'`. The column is
  kept and copied to `sample`.
- **`cell_types_multiomics(save=True)`** writes `<output_dir>/preprocess/adata_preprocessed.h5ad`.
  It used to write to `preprocess/preprocess/`, so `compute_sample_embedding` never saved the
  embedding back.
- **Sample distance:**
  - `chi_square` and `jensen_shannon` no longer raise `NameError`
    (`plot_cell_type_abundances`), and they now receive `grouping_columns`.
  - The group check works when it is given cell-level AnnData: it matches on
    `obs['sample']`.
  - With no grouping column, the check is skipped with a clear message.
- **CCA p-value:** the permutation p-value is now `(1 + k) / (N + 1)` (k = permutations at
  least as extreme as the observed score, N = permutations), so it is never reported as 0.
  It was `k / N`. A degenerate permutation null no longer crashes the histogram. That null
  (all |r| ≈ 1) happens when the number of PCs is at least the number of samples minus 1;
  use fewer CCA PCs than that, or the p-value is meaningless. The statistic itself is
  unchanged.
- **Supervised CCA pseudotime is oriented** so it increases with `trajectory_col` (it is
  flipped, with a log line, when its Spearman correlation with the trajectory variable is
  negative). Its sign was arbitrary, so trajectory-DGE "up/down" could mean the reverse of
  the trajectory variable. The CCA score and p-value are unchanged.
- **RAISIN:**
  - Comparisons that involve a group with a single sample are skipped, with a warning, and
    marked `FAILED`. Such a group has zero residual degrees of freedom and its variance was
    set to 0, which produced thousands of false "significant" genes. Comparisons between
    groups of two or more samples give identical results.
  - Automatic worker counts are capped at the Slurm allocation (`SLURM_CPUS_PER_TASK`).
- **Harmony:**
  - harmony-pytorch is now given the usable CPU count (the CPU affinity mask, capped at
    `SLURM_CPUS_PER_TASK`). It used to set torch to every physical core of the machine,
    which oversubscribed shared nodes (about 4× slower RNA preprocessing on an 8-CPU job)
    and made cell-level Harmony non-deterministic between runs.
  - harmonypy (the sample-level Harmony, and the cell-level fallback) is pinned to the CPU
    unless `use_gpu` is set. harmonypy 0.2.0 otherwise moves itself to any visible GPU, so
    the same CPU run gave different results on GPU nodes. The GPU sample-embedding path now
    also runs this step on the CPU, so GPU and CPU sample embeddings agree.
- **Multi-omics cell typing** with a second modality other than `"ATAC"` (for example
  `"FLOW"`) no longer drops every cell in the modality-balance filter.
- **Multi-omics from a pre-integrated file** (`multiomics_integration: false`) with
  `multiomics_trajectory_dge` or `multiomics_cluster_dge` switched on used to crash
  (`No features survived pseudobulk aggregation`). There is no per-modality expression file
  in that case, so both stages are now skipped with one warning and the other downstream
  stages run.
- **Trajectory DGE** warns once about batches that contain a single sample, because the
  batch regression removes their information.
- **Trajectory DGE:** a clear warning when the spline settings leave no gene fittable (for
  example `spline_order=3` on 8 samples). Importing the DGE plotting module no longer turns
  off Python warnings for the rest of the session.
- **`*_summary_sample_csv_path`:** a user-supplied file that already existed is no longer
  deleted at the end of the run.
- **Config template:** the demo-data DOI is now 10.5281/zenodo.22549610. The
  `*_tscan_pseudotime_mode` comments list the valid values, `"rank"` and `"distance"`, instead
  of `"projection"`. This is a comment-only change.
- Tests: `tests/test_tutorial_fixes.py`.

## 0.5.0

Two new supervised downstream functions on the sample embedding. Python API only: the
config file and `wrapper()` are unchanged (existing configs validate and run as before).

- **New `predict_phenotype`** (`sampledisco.predict_phenotype`, module
  `sampledisco.sample_prediction`): out-of-fold prediction of a sample phenotype from the
  first `n_pcs` PCs of `uns['X_DR_sample']`. The PCs are standardized and fed to an L2
  logistic regression (binary or multiclass) or, with `task="regression"`, a ridge
  regression. The penalty (C in {10, 1, 0.1, 0.01}; alpha in {0.1, 1, 10, 100}) is chosen
  by an inner 3-fold grouped cross-validation inside each training split, and a tie goes to
  the weakest penalty. The outer split is `n_splits`-fold (default 5), stratified by class
  and seeded by `random_state`. With `group_col` (for example a donor column) every sample
  of a group stays in one fold, so repeated draws of a donor are never in training and test
  at once. Returns the per-sample out-of-fold predictions (with class probabilities) and
  AUC (binary), accuracy and balanced accuracy, or R². `score_by="group"` averages the
  out-of-fold probabilities (or predictions) per group before scoring. This is the recipe
  of the paper's Sound Life trait prediction: with `score_by="group"` it reproduces the
  paper's within-assay donor-level AUCs exactly.
- **New `transfer_labels`** (`sampledisco.transfer_labels`): assigns labels to query samples
  (unlabelled samples, or a boolean `query` mask) by a vote of their `n_neighbors` nearest
  labelled samples in the embedding (Euclidean on the first `n_pcs` PCs; majority vote, or
  1/distance-weighted with `weighted=True`; ties go to the nearest tied neighbour). Returns
  each label with its confidence (vote share), and a leave-one-out accuracy and balanced
  accuracy on the labelled samples; with `group_col` the other samples of the same group
  are left out too (use `group_col="sample"` for multi-omics units).
- Tests: `tests/test_sample_prediction.py` (synthetic data with known answers).

## 0.4.2

- **New autotune scoring option `scoring="ilisi_label"`**: a two-term objective, the mean of one
  biology term (grouping tracking: canonical correlation for a numeric label, PC-R² for a
  categorical one) and one batch term (iLISI on the batch key), each min-max scaled as in
  `"auto"`. It drops the second biology term (SPS) and the second batch term (−ASW) of the
  four-term `"auto"` ensemble. Terms are gated by the available metadata as in `"auto"`.
  Available through `run_autotune(scoring=...)` and `*_autotune_scoring` in the config;
  the default stays `"auto"`.
- **Fix:** the autotune report is written as UTF-8, so `run_autotune(save=True)` no longer
  fails on nodes with an ASCII locale.

## 0.4.1

- **Fix a deadlock in 0.4.0** in the concurrent k-means fits (`kmeans_centers_pair`, CPU and GPU
  paths): a threadpoolctl library scan in one worker thread could deadlock with a first import of
  scikit-learn's compiled extensions in the other. Observed in 2 of 3 warm GPU runs on the
  25-sample COVID object. `sklearn.cluster` is now imported in the calling thread, so no library is
  loaded inside the worker threads; the k-means centres are unchanged (bit-identical).
  `threadpoolctl>=3.1` is now a direct requirement.

## 0.4.0

New default RMD weight (`rmd_weight="equal"`), a faster sample-embedding step with
unchanged results, and the embedding is now stored in the cell-level h5ad in place.

- **harmonypy pinned to 0.2.0** (was `>=0.0.9,<0.0.10`), the version used for the published
  results; sample-level Harmony output depends on the harmonypy version. harmonypy 0.2.0
  requires torch, so `pip install sampledisco` now also installs PyTorch.

### ⚠️ Behaviour change: default RMD weight is now `"equal"`

- **`rmd_weight="equal"`** (new default of `compute_sample_embedding`, the wrappers and
  `*_sample_embedding_rmd_weight` in the bundled config) gives the RMD displacement block
  the same energy as the three composition blocks together:
  α² = w_A1² + w_A2² + w_A3², with w_A1 = √(K_fine/K_c), w_A2 = √(K_fine/K_med),
  w_A3 = 1, rounded to 2 decimals. α depends only on the number of cell types and the
  two k-means resolutions (with the defaults 120/300 and at least 30,000 cells, where
  neither resolution is capped by the cell count, it is √(300/K_c + 3.5)); the
  composition weights are unchanged. A number keeps the previous behaviour
  (`0.60` reproduces earlier releases exactly). The value used is recorded in
  `.uns['sample_embedding_params']['rmd_weight']`. Only `"equal"` or a finite positive
  number is accepted; anything else raises `ValueError` up front (also in the CLI
  config check), before any GPU attempt.
- `run_autotune` is unchanged when it searches; with no batch and no grouping column
  (no search) it now uses the `"equal"` α instead of 0.60.

### Embedding stored in the cell-level h5ad in place

- With `save_cell_adata=True` (default), `.uns['X_DR_sample']` and
  `.uns['sample_embedding_params']` are written into the existing
  `preprocess/adata_preprocessed.h5ad` (or the multi-omics union h5ad) in place:
  only those two entries are replaced, X, layers, obs and obsm are not re-written.
  At 405 COVID samples (898k cells, 3.3 GB) this takes about 5 s, including the check
  below, instead of the full gzip re-write (273 s of the previous 295 s step). The in-place path is
  taken only when the file holds the same object as memory: identical obs and var
  (names, columns and values), the same obsm/layers/obsp/varm keys, the same shape,
  dtype and a fixed sample of values for X and every matrix, and the same uns key
  names (values of other uns entries are not compared, so a changed value under an
  existing uns key is not detected). Otherwise (for
  example cell types relabelled in memory, or an old-format file) the whole file is
  re-written as before. Each entry is written under a temporary key and then moved
  into place, so an interrupted update never leaves the file without an embedding. The
  stored values are identical to a full re-write
  (`tests/test_inplace_uns_and_equal_weight.py`).

### Faster sample embedding, identical output

- **New config keys `rna_/atac_/multiomics_save_cell_adata_after_embedding`**
  (default `true`; `compute_sample_embedding(save_cell_adata=)` and
  `run_autotune(save_cell_adata=)`). `false` skips storing `.uns['X_DR_sample']` in
  the cell-level h5ad; the CSV and block files are still written. If that h5ad (or the multi-omics union h5ad) already
  holds an older `.uns['X_DR_sample']`, a `UserWarning` is raised: a later run with
  `derive_sample_embedding: false` would reuse that stale embedding.
- **Backward-compatible config:** the bundled config now has 304 keys; existing
  301-key configs (0.2.0 and earlier) still validate and run. `validate_config`
  treats only these three keys as optional (absent = `true`); every other key is
  still required (`tests/test_config_validation.py`).
- **Speed-ups with identical output** (`compute_sample_embedding`, CPU and GPU):
  units and cell types indexed by integer codes instead of per-cell id dicts;
  RMD sums by `np.bincount` (same float64 accumulation order); MiniBatchKMeans
  with `compute_labels=False` (the skipped final labelling pass never fed the
  embedding); on CPU the softmax and per-unit mean run unit by unit on the
  n_cells x K distance matrix, so the separate soft-assignment matrix and its
  temporaries are no longer built; the two independent k-means fits run
  concurrently. Compared with commit 1c4168a at the same α, A1, A2, A3, RMD and the final
  embedding are bit-identical (max abs difference 0) on COVID 25 and 405 samples,
  heart, lifespan and Sound Life (`tests/test_sample_embedding_equivalence.py`
  checks the helpers on synthetic data).

## 0.3.0

Cell-level embedding keys are renamed to match the manuscript's notation, behind a
full backward-compatibility layer. **Every legacy key still reads**; nothing on
disk needs to be rewritten. Autotuning of the RMD weight α is also fixed: it now
searches on a log scale, and its ceiling no longer truncates the optimum.

### ⚠️ Behaviour-changing fix: α autotuning searched the wrong scale

α is a scale parameter, but all three search strategies explored it linearly, and
the ceiling was too low. Both faults pushed the tuned α away from the optimum.

- **All searches now operate on log10(α).** `search_bayesian` seeded with
  `np.linspace(lo, hi, 5)`, so over a wide interval such as [0.01, 1000] its first
  five probes were 0.01/250/500/750/1000 and the sub-unit region — where
  composition-dominated cohorts have their optimum — was never examined.
  `search_golden` bisected linearly for the same reason. Both now work in log10
  and report the trace back in α.
- **`search_grid` honours `alpha_bounds`.** It previously ignored them entirely
  and swept a hardcoded `[0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0]`, so the `grid`
  strategy silently disregarded the user's configured range. It now builds a
  log-spaced grid from the bounds (`n=15` by default). Passing an explicit
  `alpha_grid=` list still works unchanged.
- **`DEFAULT_ALPHA_BOUNDS` ceiling raised, `(0.1, 10.0)` → `(0.1, 100.0)`,** and
  the matching defaults in `wrapper()`, `rna_wrapper`, `atac_wrapper`,
  `multiomics_wrapper`, and `config_demo.yaml` are raised with it. The wrapper
  signatures previously pinned `(0.1, 10.0)` and so overrode the module constant.
  The floor is deliberately unchanged: no dataset has been observed to optimise
  against it, whereas on a composition-dominated cohort the objective is flat
  below α≈1.5, so a lower floor only lets α drift until the RMD block is
  numerically absent.
- **The α autotune record now reports the plateau, not just the winner.**
  `autotune_record.txt` gains an `α within 1% of best` line, and flags the case
  where the selected α sits exactly on a search bound. On a flat objective the
  single winning α is arbitrary within the plateau, and the previous record gave
  no way to see that.
- **Impact.** On a stimulation time-course (1M-scBloodNL, 790 sample–condition
  units) a 40-point dense sweep of the same objective peaks at α≈16; the old
  search returned α pinned at the 10.0 ceiling, the new one returns α≈17.8.
  Saved runs for COVID-ATAC, COVID-279, and the unpaired-paper cohort had likewise
  returned α pinned at 10.0 and should be re-tuned.
  On a composition-dominated cohort the selected α also moves, but only inside a
  region the objective cannot distinguish: on ENCODE the old search returned
  α≈0.66 and the new one α≈0.15, while a dense sweep puts the peak at α≈0.62 and
  everything in [0.01, 1.5] within 1% of it. Such cohorts put ≲1% of the embedding
  in the RMD block at any α in that range, so the fitted sample embedding is
  effectively unchanged — but the reported α is only weakly identified and should
  not be quoted as a precise value. Use the new plateau line to see the range.
  **Any α obtained from a previous release should be re-tuned; embeddings built
  from a pinned α are not at the objective's optimum.**

### ⚠️ Results-changing fix (numbers move on GEN-2 objects)

- **The RMD key resolver now recognises the legacy `Z_cmd`.** Through 0.2.0,
  `_resolve_rmd_emb_key` checked only `Z_rmd` and otherwise returned the
  *composition* key. On an h5ad that carried `Z_clust` + `Z_cmd` (the pre-`Z_rmd`
  generation), the RMD displacement block was therefore computed on the
  **sample-REMOVED** embedding — silently, with no error and no warning — making
  that block near-degenerate. Any sample embedding regenerated from such an object
  under 0.2.0 is wrong and must be recomputed.
  **Verification item for maintainers:** check whether any current metric in
  `R/multi_omics_unpaired_diemb/` or `R/test_REFERENCE/` was regenerated under
  0.2.0 against `Z_cmd`-carrying objects; both are of that generation and both are
  actively referenced. This was not verified as part of this release.
- When no sample-preserved embedding exists at all, the fallback to the
  composition key is still taken but now emits a `RuntimeWarning` stating that the
  result is **not** the method described in the paper (previously silent).

### Renamed

- `obsm['Z_clust']` → **`obsm['Z_comp']`**, matching $z_i^{\mathrm{comp}}$ in the
  paper. `Z_rmd` is unchanged.
- Reads of `Z_clust`, `Z_cmd`, `X_pca_harmony`, `X_pca_harmony_nosamp`,
  `X_lsi_harmony`, `X_lsi_harmony_nosamp` and `X_glue_harmony` all still work,
  each with a `FutureWarning`. Removal in 1.0.
- 0.3.0 also **writes** `Z_clust` as a duplicate of `Z_comp`, so h5ads produced by
  0.3.0 remain readable by an already-installed 0.2.0. That duplicate goes away in
  1.0.

### Deprecated

- `cluster_emb_key=` → `comp_emb_key=` on `compute_sample_embedding` and
  `run_autotune`; `z_clust_key=` → `z_comp_key=` on the 2-run GLUE merge helper.
  Old names accepted with a `FutureWarning`, removed in 1.0.
- `uns['sample_embedding_params']` gains `comp_emb_key` and keeps writing
  `cluster_emb_key` (same value) for existing readers; the old field goes away in
  1.0. Both now record the **resolved** key rather than the caller's argument.
- `sample_embedding.sample_embedding._resolve_rmd_emb_key` is a thin shim over
  `utils.embedding_keys.resolve_rmd_key`.

### Added

- `sampledisco.utils.embedding_keys` — the single owner of cell-level embedding
  key names: `COMP_KEY` / `RMD_KEY`, the alias tables, and
  `resolve_comp_key` / `resolve_rmd_key` / `resolve_embedding_keys`, which every
  read site in the package now routes through (four separate ad-hoc resolvers
  previously disagreed with each other).
- `migrate_obsm_keys(adata)` — copies legacy obsm embeddings onto their canonical
  names **in memory only**, non-destructively. It writes no files.

### Fixed

- `cell_types_multiomics` / `cell_types_multiomics_gpu` defaulted to
  `use_rep="X_glue"`, which is sample-PRESERVED and contradicted their own
  docstrings; called directly (not via the wrapper) they leaked per-sample
  variance into cell typing. The default is now `None` = auto-resolve to the
  sample-removed view.
- Stale docstrings that named `X_DR_expression` / `X_DR_proportion` defaults where
  the code reads `X_DR_sample` (`multi_omics_visualization`, `association`).

### No config change

All 301 `wrapper()` / YAML keys are unchanged; existing configs run unmodified.
The three affected `config_demo.yaml` lines are comments only.

### Retroactive note

The `Z_cmd` → `Z_rmd` rename (and the `loo_cmd` → `loo_rmd` / `cmd_*` → `rmd_*`
identifier rename) shipped unannounced in 0.2.0. It is recorded here for the
first time.

## 0.2.0

A correctness- and robustness-focused release. Some fixes change saved outputs —
if you have benchmark/figure numbers derived from CCA scores/p-values, RAISIN, or
the sample embedding, **re-generate the saved metrics for affected datasets**
(figure scripts only re-rank saved metrics; do not re-run competing methods).

### ⚠️ Results-changing fixes (numbers move)

- **CCA trajectory p-value is now valid.** The permutation null is built on the
  same `n_cca_pcs` dimensions as the observed statistic (was hard-wired to 2 PCs
  → anti-conservative); NaN trajectory samples are dropped (not mean-imputed) and
  masked identically on both sides; the permutation RNG is seeded. P-values become
  more conservative.
- **RAISIN validated against the reference R implementation (`zji90/raisin`) and
  aligned to it.** `mean`, `omega2` (cell-level variance) and fold changes are
  bit-identical to R; the sigma2 EB-estimation formula is bit-identical given the
  same inputs. Two real R-mismatches were fixed: (1) the non-finite EB fallback is
  now `1.0` (matching R's `est[is.na] <- 1`); (2) variance components are estimated
  in R's `unique(group)` first-appearance order (the sequential done-group
  correction is order-dependent). A residual sigma2 difference remains — it is the
  random-orthonormal-basis Monte-Carlo component inherent to the estimator (present
  in R too, ±~10% across seeds).
- **proportion test:** a degenerate (1-vs-1) group no longer NaN-propagates through
  the pooled BH-FDR and blanks every comparison; degenerate pairs are skipped and
  flagged.
- **Deterministic tie-breaking** in per-unit majority-vote group/batch labels, so a
  seeded config reproduces the same embedding (was hash-order dependent on ties).
- **Multi-omics Harmony** now honors the seed and falls back to CPU `harmonypy`.

### Robustness / correctness

- Sample-level batch correction now degrades **harmonypy → linear regression → raw
  PCA** (was harmonypy → raw PCA), and the GPU path honors `batch_method="none"`.
- GPU dispatch and GPU cell-typing fall back to CPU on CUDA runtime errors (not just
  import errors).
- GLUE training records its batch-design fields and retrains if they change (no more
  silent reuse of a stale model on a changed design).
- `dimension_association` no longer marks success after a swallowed failure.
- RAISIN pair-tests record failed/skipped comparisons; cluster plot no longer
  IndexErrors on an empty k-means cluster; `cell_proportions` orientation is checked
  rather than guessed; sample-metadata aggregation uses exact `(sample, modality)`
  keys.
- RAISIN parallel uses a threading backend (bounds memory on wide matrices; numerically
  identical) and gained an opt-in `max_features` cap (off by default).
- Multi-omics wrapper warns up front when `integration=False` skips downstream DGE.

### Packaging & docs

- Version 0.2.0; PEP 639 license metadata (`license = "MIT"`, `license-files`).
- CLI `--init-config` labels the emitted file as the demo config and warns its
  thresholds are demo-tuned.
- Documentation site: corrected install recipes (GLUE-from-scratch deps, macOS
  `curl`), output filenames, API signatures/defaults, and a landing-page quickstart;
  `mkdocs --strict` enforced.
