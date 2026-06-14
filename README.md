# SampleDisco

A cross-omics, cross-condition **sample embedding** tool for single-cell data.

SampleDisco takes a cell-level embedding (from any standard scRNA / scATAC / multi-omics integration method) and lifts it to a **sample-level embedding** that captures both cell-type composition and the per-cell-type state of each sample. Every downstream analysis — sample-to-sample distance, clustering, trajectory inference, phenotype association — then runs on that single shared sample embedding, regardless of modality.

Paper draft: [`/users/hjiang/GenoDistance/SampleDisco_Draft-11.pdf`](../SampleDisco_Draft-11.pdf)

---

## What the method does

For each modality (RNA, ATAC, or integrated multi-omics) the pipeline produces two cell-level views:

| Key | Role | Source |
|---|---|---|
| **`Z_clust`** | sample-removed embedding — used for clustering and composition blocks | Harmony (single-omics) / Harmony post-pass on scGLUE (multi-omics) |
| **`Z_cmd`**  | sample-preserved embedding — used for the counterfactual displacement (CMD) block | second Harmony pass (single-omics) / scGLUE primary output (multi-omics) |

It then assembles **four blocks** per sample (or per sample × modality for multi-omics):

1. **A1** — one-hot cell-type composition
2. **A2** — soft k-means composition at K_med (≈120)
3. **A3** — soft k-means composition at K_fine (≈300)
4. **CMD** — leave-one-out cell-type-resolved displacement on `Z_cmd`

The four blocks are inverse-variance weighted, Frobenius-stacked, PCA-reduced to 10 dimensions, and Harmony-corrected at sample level. The result is stored as `adata.uns['X_DR_sample']` and feeds every downstream module.

---

## Repository layout

```
code/
├── SampleDisc.py              # CLI entry point (simple or complex mode)
├── config/                    # 9 YAML configs covering covid / blood / eye / heart / ENCODE / tabula / long_covid / unpaired / default
├── wrapper/                   # Orchestration
│   ├── wrapper.py             # Master wrapper; gates RNA + ATAC + multiomics + shared downstream
│   ├── rna_wrapper.py
│   ├── atac_wrapper.py
│   └── multiomics_wrapper.py
├── preparation/               # Preprocessing
│   ├── rna_preprocess_{cpu,gpu}.py   # QC → HVG → PCA → dual Harmony → Z_clust + Z_cmd
│   ├── atac_preprocess_{cpu,gpu}.py  # QC → TF-IDF → HVF → LSI → dual Harmony → Z_clust + Z_cmd
│   ├── cell_type_{cpu,gpu}.py        # Leiden clustering on Z_clust (RNA or ATAC)
│   ├── ATAC_cell_type{,_gpu}.py      # ATAC-specific cell typing variants
│   ├── multi_omics_glue.py           # scGLUE integration (cross-modality VAE + guidance graph)
│   ├── multi_omics_batch_correction.py # Harmony post-pass on X_glue → Z_clust
│   ├── multi_omics_merge.py          # post-GLUE merge + per-modality preprocess/slimming
│   └── multi_omics_cell_type_{cpu,gpu}.py  # RNA-Leiden + k-NN label transfer to ATAC
├── sample_embedding/          # Core method
│   ├── blocks.py              # composition, CMD, weighting, Frobenius stack, final PCA + Harmony
│   ├── sample_embedding.py    # CPU pipeline
│   └── sample_embedding_gpu.py # GPU pipeline (cuML + cupy)
├── parameter_selection/
│   └── autotune.py            # Bayesian GP sweep over CMD α; adaptive proxy ensemble
├── sample_distance/           # Pairwise sample distances (DR / EMD / chi-square / JS)
├── sample_clustering/         # Hierarchical (HRA / HRC / NN / UPGMA / consensus), K-means, proportion test, RAISIN
├── sample_trajectory/         # CCA (supervised) and TSCAN (unsupervised) + GAM-based trajectory DGE
├── sample_association/        # Per-PC variance explained vs sample-level covariates (permutation FDR)
├── visualization/             # Embedding plots, dendrograms, DGE volcanos, modality-aware multi-omics scatters
├── utils/                     # Shared helpers: seed, safe h5ad I/O, limma, TF-IDF, batch regress, Grouping
├── gene_activity/             # ATAC peak → gene activity inference + RNA-ATAC validation
└── claude/                    # Active one-off run scripts (rerun launchers, monitored SE, parameter sweeps)
```

---

## Usage

### Complex mode (recommended) — YAML-driven

```bash
python SampleDisc.py -m complex --config config/config.yaml
```

The YAML drives every flag and parameter for all three pipelines:

- **Pipeline gates** (top-level): `run_rna_pipeline`, `run_atac_pipeline`, `run_multiomics_pipeline`
- **Per-modality phase gates** (Phase 1): `*_preprocessing`, `*_cell_type_cluster`, `*_derive_sample_embedding`
- **Per-modality downstream gates** (Phase 2): `*_sample_distance_calculation`, `*_trajectory_analysis`, `*_trajectory_dge`, `*_sample_cluster`, `*_proportion_test`, `*_cluster_dge`, `*_visualize_data`, `*_dimension_association_analysis`
- **Multi-omics-specific**: `multiomics_run_glue_*`, `multiomics_treat_sample_as_batch`, `multiomics_run_glue_twice_for_sample_removal`

The 9 ready-to-use configs in `config/` are point-in-time snapshots for the datasets used in the paper; copy one and adjust paths / column names for your own data.

### Simple mode — one positional file, defaults everywhere

```bash
python SampleDisc.py -m simple -c <count_data.h5ad> -o <output_dir>
```

---

## Inputs

A standard scanpy AnnData file with at minimum:
- `.X` — count matrix (genes for RNA, peaks for ATAC)
- `.obs['sample']` — sample column (required)
- Optional: `.obs['batch']`, `.obs['cell_type']`, sample-level metadata file (CSV) to merge

For multi-omics, the pipeline takes two separate h5ads (RNA + ATAC) and integrates them via scGLUE; samples may be **paired** (1:1 cell correspondence) or **unpaired**.

---

## Outputs (under `output_dir`)

```
<output_dir>/
├── rna/
│   ├── preprocess/adata_preprocessed.h5ad
│   ├── sample_embedding/sample_embedding.csv
│   ├── Sample_distance/{cosine,correlation}/*
│   ├── CCA/  or  TSCAN/                       # whichever trajectory mode
│   ├── trajectoryDEG/
│   ├── sample_cluster/{kmeans_*,proportion_test/}
│   ├── sample_association/variance_explained_sample.csv + figures/
│   └── visualization/*.png
├── atac/   (parallel structure)
├── multiomics/
│   ├── integration/glue/{rna-pp,atac-pp,guidance.graphml.gz}
│   ├── preprocess/adata_sample.h5ad           # post-GLUE merged adata with Z_clust + Z_cmd
│   ├── sample_embedding/sample_embedding.csv
│   └── (same downstream subdirs as single-omics)
└── sys_log/main_process_status.json           # which stages completed
```

---

## Installation

SampleDisco is **one package**. The CPU install is pip-only; **GPU acceleration is
activated simply by installing the GPU libraries separately** — the same package
detects and uses them at runtime. There is no separate "GPU build" of SampleDisco.

### 1. Core install (CPU)

```bash
pip install sampledisco          # once published — or `pip install -e .` from a clone
```

### 2. System prerequisite — bedtools

scGLUE (the multi-omics integrator) calls the `bedtools` binary, which pip cannot
provide:

```bash
conda install -c bioconda bedtools
```

### 3. GPU acceleration (optional, install yourself)

The GPU functions (RAPIDS-accelerated normalization, Harmony, k-means / PCA, Leiden,
scGLUE training) turn on **only when the RAPIDS stack is present** in your
environment. RAPIDS is CUDA-driver-specific and conda-only, so you install it
separately, matching your driver (the pins below target a CUDA-12.5 driver such as
the cluster's GPU nodes):

```bash
conda install -c rapidsai -c conda-forge -c nvidia \
    cuml=24.12 cudf=24.12 cugraph=24.12 rmm=24.12 cuvs=24.12 cupy=13 cuda-version=12.5
pip install rapids-singlecell==0.13.1 --no-deps
```

Then set `use_gpu: true` in your config. **You do not reinstall SampleDisco** — once
those packages are importable the GPU paths activate automatically; if they are
missing or the driver is too old, SampleDisco falls back to CPU equivalents
(`harmonypy` / linear regression, scikit-learn k-means, PyTorch CPU).

### One-command environments (recommended)

For a fully reproducible environment (including bedtools), use the provided conda
files instead of the manual steps above — see `INSTALL.md` for the driver/version
notes:

```bash
conda env create -f environment-cpu.yml      # CPU
conda env create -f environment-gpu.yml      # GPU (RAPIDS 24.12)
```
