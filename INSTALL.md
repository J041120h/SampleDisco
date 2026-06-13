# Installing SampleDisco

SampleDisco ships as **one package** (`sampledisco`) with **two environments** — a
CPU environment and a GPU (NVIDIA RAPIDS) environment. They install the *same*
package; the only difference is the environment (the GPU one adds the RAPIDS
stack). The code dispatches CPU vs GPU at runtime via the `use_gpu` flag, so the
same scripts/configs work in either.

Both environments were built and smoke-tested end-to-end on 2026-06-13 (JHPCE):
all imports + scGLUE coexistence + (GPU) `torch`/`cupy`/`cuml`/`rapids_singlecell`
all using an A100 in one process.

---

## CPU environment

```bash
conda env create -f environment-cpu.yml
conda activate sampledisco-cpu
pip install -e . --no-deps            # add the sampledisco package
```

Validated stack: python 3.10, numpy 2.0.2, scanpy 1.11.4, anndata 0.10.9,
scikit-learn 1.6.1, numba 0.60, torch 2.5.1 (CPU), **scGLUE 0.3.2**, bedtools 2.31.

## GPU environment (NVIDIA, CUDA-12 driver)

```bash
conda env create -f environment-gpu.yml
conda activate sampledisco-gpu
pip install rapids-singlecell==0.13.1 --no-deps   # pip-only; rapids deps come from conda
pip install -e . --no-deps
```

Validated stack: the CPU stack **+ RAPIDS 24.12** (cuml/cudf/rmm/cuvs 24.12,
cupy 13.6) + torch 2.5.1+cu121, all coexisting with numpy 2.0.2.

> **Driver note.** RAPIDS is pinned to **24.12** on purpose. The cluster's GPU
> nodes run driver **555.42.06 (CUDA 12.5)**; RAPIDS 25.04+'s compiled libraries
> need a newer driver (≥ CUDA 12.6) and fail at import with
> `cudaErrorInsufficientDriver` on these nodes. RAPIDS 24.12 is built for CUDA
> 12.0–12.5 and runs on the 12.5 driver. On nodes with driver ≥ CUDA 12.6 you may
> bump the `cuml/cudf/rmm/cuvs` pins to 25.x.

---

## scGLUE (multi-omics)

scGLUE is included in **both** environments — it does **not** need a separate
environment. The one rule: it **must** be installed from PyPI (`pip install
scglue==0.3.2`, which the env files do), **never** `conda install -c bioconda
scglue` — the bioconda recipe caps `numpy<1.22` and breaks the rest of the stack.
scGLUE needs the `bedtools` system binary, which the env files provide via bioconda.

## Optional: phylogenetic-tree clustering

The NN/UPGMA/consensus tree-clustering methods are off the default pipeline path,
so their heavy deps are optional:

```bash
pip install -e '.[trees]'      # biopython, scikit-bio, dendropy
```

---

## Run

```bash
sampledisco --config config/config.yaml      # or: python -m sampledisco.cli ...
```

GPU acceleration is requested with `use_gpu: true` in the config; if RAPIDS is
unavailable (CPU env, or an incompatible driver) the pipeline falls back to CPU.
