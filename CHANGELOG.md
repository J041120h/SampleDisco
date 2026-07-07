# Changelog

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
