# Changelog

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
