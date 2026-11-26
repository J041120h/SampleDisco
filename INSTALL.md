# Installing SampleDisco

SampleDisco is **one package** (`sampledisco`), and every part of it — the core
pipeline, scGLUE (RNA + ATAC), harmony-pytorch and MaxFuse (RNA + protein / flow) —
runs in **one environment**. Pick the
CPU or the GPU file; the GPU one is the same environment plus NVIDIA RAPIDS. The
code dispatches CPU vs GPU at runtime via the `use_gpu` flag, so the same
scripts/configs work in either.

---

## CPU environment

```bash
conda env create -f environment-cpu.yml
conda activate sampledisco-cpu
```

That is all: the env file installs SampleDisco itself (its last pip line).

Validated stack (fresh build, 2026-11-22, JHPCE): python 3.10, numpy 2.0.2,
scanpy 1.11.4, anndata 0.10.9, scikit-learn 1.6.1, numba 0.60, pynndescent 0.6,
umap-learn 0.5, torch 2.5.1 (CPU, conda-forge), scGLUE 0.3.2, harmony-pytorch 0.1.8,
MaxFuse 0.0.2, bedtools 2.31.

## GPU environment (NVIDIA, CUDA-12 driver)

```bash
conda env create -f environment-gpu.yml      # no GPU on this machine? prefix CONDA_OVERRIDE_CUDA=12.5
conda activate sampledisco-gpu
pip install --no-deps rapids-singlecell==0.13.1 sampledisco==0.8.0
```

Validated stack: the CPU stack **+ RAPIDS 24.12** (cuml/cudf/cugraph/rmm/cuvs 24.12,
cupy 13) + torch 2.5.1+cu121, all coexisting with numpy 2.0.2. The last step uses
`--no-deps` on purpose: the env file already installs every dependency, and a
dependency-resolving pip would replace the pyarrow that RAPIDS needs (< 18) because
the pyensembl → gtfparse chain declares pyarrow ≥ 18 in its pip metadata.

> **Driver note.** RAPIDS is pinned to **24.12** on purpose. The cluster's GPU
> nodes run driver **555.42.06 (CUDA 12.5)**; RAPIDS 25.04+'s compiled libraries
> need a newer driver (≥ CUDA 12.6) and fail at import with
> `cudaErrorInsufficientDriver` on these nodes. RAPIDS 24.12 is built for CUDA
> 12.0–12.5 and runs on the 12.5 driver. On nodes with driver ≥ CUDA 12.6 you may
> bump the `cuml/cudf/cugraph/rmm/cuvs` pins to 25.x.

## pip only (no env file)

```bash
conda create -n sampledisco -c conda-forge -c bioconda python=3.10 pip bedtools   # bedtools: binary that scGLUE/pybedtools call
conda activate sampledisco
pip install "sampledisco[all]"
```

Create the env with Python and bedtools in one step. On Miniconda (default channel
`defaults`), installing bedtools from conda-forge into an existing env can swap its
Python for an incompatible build (observed: GraalPy), after which nothing imports.

`pip install sampledisco` alone gives the core pipeline. The extras add the
optional parts: `[glue]` (scGLUE 0.3.2 + harmony-pytorch), `[maxfuse]`, or `[all]`
for everything. This route takes
PyPI's default torch wheel and does not pin numpy/scanpy to the validated versions;
use an env file when you want the validated stack.

---

## Version constraints (already handled by the files above)

- **scGLUE** must come from PyPI, **never** `conda install -c bioconda scglue` —
  the bioconda recipe caps `numpy<1.22` and breaks the stack. scGLUE 0.3.2 also
  needs **`anndata<0.11`** (it imports `anndata._core.sparse_dataset.SparseDataset`,
  which anndata 0.11 removed) and **`torch<2.7`** (its training loop passes
  `ReduceLROnPlateau(verbose=...)`, which torch 2.7 removed). Its own metadata
  declares neither cap.
- **MaxFuse 0.0.2** calls `AnnData(..., dtype=...)`, which anndata 0.13 removed:
  keep **`anndata<0.13`**.
- **pynndescent < 0.7** (and therefore umap-learn < 0.6): pynndescent 0.7 /
  umap-learn 0.6 break `scanpy.pp.neighbors` (scanpy 1.11) on more than ~8k cells.
- **CPU requirement:** the env files pull in `polars` (a gtfparse dependency), whose
  conda build needs an AVX2 CPU. On older nodes (e.g. AMD Opteron 6234) importing
  scGLUE or pyensembl dies with `Illegal instruction`; run on a newer node.
- **System CUDA libraries on `LD_LIBRARY_PATH`:** if a cluster module puts its own
  CUDA (e.g. `/usr/local/cuda/lib64`) on `LD_LIBRARY_PATH`, importing torch (and so
  anndata/scanpy) can fail with `libtorch_cuda.so: undefined symbol: ncclCommResume`.
  Run with that path removed (e.g. `unset LD_LIBRARY_PATH`) — the envs ship their own
  CUDA libraries.

---

## Run

```bash
sampledisco --config config/config.yaml      # or: python -m sampledisco.cli ...
```

GPU acceleration is requested with `use_gpu: true` in the config; if RAPIDS is
unavailable (CPU env, or an incompatible driver) the pipeline falls back to CPU.
