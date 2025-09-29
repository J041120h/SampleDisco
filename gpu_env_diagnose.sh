#!/usr/bin/env bash
# gpu_env_diagnose.sh — NO MODIFICATIONS, just prints diagnostics
set -u

have() { command -v "$1" >/dev/null 2>&1; }

echo "=== 0) Basics ==============================================="
date
echo "Hostname: $(hostname)"
echo "User: $(whoami)"
echo

echo "=== 1) Active Python/Conda =================================="
if have conda; then
  conda info --envs | sed -n '1p;/\*/p'
else
  echo "conda: not found"
fi
echo "PYTHON: $(which python || echo 'python: not found')"
python -V 2>&1 || true
echo

echo "=== 2) Environment variables (CUDA-related) =================="
echo "CONDA_PREFIX=${CONDA_PREFIX-<unset>}"
echo "CUDA_HOME=${CUDA_HOME-<unset>}"
echo "CUDA_PATH=${CUDA_PATH-<unset>}"
echo "PATH=$PATH"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH-<unset>}"
echo

echo "=== 3) System CUDA toolchain (if any) ========================"
if have nvidia-smi; then
  echo "[nvidia-smi]"
  nvidia-smi || true
else
  echo "nvidia-smi: not found"
fi
if have nvcc; then
  echo "[nvcc --version]"
  nvcc --version || true
else
  echo "nvcc: not found"
fi
# NVRTC shared library locations (common names)
echo "[Locate libnvrtc*]"
ldconfig -p 2>/dev/null | grep -E 'libnvrtc\.so' || echo "ldconfig not available or libnvrtc not listed"
echo

echo "=== 4) Conda packages of interest ============================"
if have conda; then
  conda list | grep -E '^(cupy|cudf|cuml|rapids|rmm|libraft|raft|cugraph|cuda|nvrtc)\b' || echo "(none matched)"
else
  echo "conda: not found"
fi
echo

echo "=== 5) Pip packages (in this env) ============================"
if have pip; then
  pip show cupy cupy-cuda12x rmm cudf cuml libcugraph-cu12 libraft-cu12 2>/dev/null || true
else
  echo "pip: not found"
fi
echo

echo "=== 6) Check for cuda_fp16.h in common include paths ========="
check_path() {
  local p="$1"
  if [ -n "$p" ] && [ -f "$p/include/cuda_fp16.h" ]; then
    echo "FOUND: $p/include/cuda_fp16.h"
  else
    echo "Missing: $p/include/cuda_fp16.h"
  fi
}
check_path "${CUDA_HOME-}"
check_path "${CUDA_PATH-}"
check_path "${CONDA_PREFIX-}"
# Common system locations
for base in /usr/local/cuda /usr/local/cuda-12.4 /usr/local/cuda-12.3 /usr/local/cuda-12.2 /usr/local/cuda-12.1 /usr/local/cuda-12.0; do
  if [ -d "$base" ]; then
    if [ -f "$base/include/cuda_fp16.h" ]; then
      echo "FOUND: $base/include/cuda_fp16.h"
    else
      echo "Missing: $base/include/cuda_fp16.h"
    fi
  fi
done
# Last-resort quick search (bounded to avoid long scans)
if [ -d /usr/local ]; then
  echo "[Quick scan under /usr/local (depth<=3)]"
  find /usr/local -maxdepth 3 -name cuda_fp16.h 2>/dev/null | sed 's/^/FOUND: /' || true
fi
echo

echo "=== 7) Python-side probes (no JIT to avoid your error) ======="
python - <<'PY' 2>/dev/null || true
import sys, os
def safe_import(name):
    try:
        m=__import__(name)
        print(f"import {name}: OK ({getattr(m,'__version__','no __version__')})")
        return m
    except Exception as e:
        print(f"import {name}: FAIL ({e})")
        return None

print("Python:", sys.version.replace('\n',' '))
print("sys.executable:", sys.executable)

cp = safe_import("cupy")
cuml = safe_import("cuml")
rmm = safe_import("rmm")
rsc = safe_import("rapids_singlecell")

if cp:
    try:
        from cupy.cuda import runtime
        v = runtime.runtimeGetVersion()
        print("CUDA runtime version (from CuPy):", v)
    except Exception as e:
        print("CuPy runtime probe failed:", e)
    try:
        # Do NOT trigger elementwise JIT. Just a device count.
        from cupy.cuda import Device
        import cupy
        ndev = cupy.cuda.runtime.getDeviceCount()
        print("Visible CUDA devices:", ndev)
    except Exception as e:
        print("Device count failed:", e)
PY
echo

echo "=== 8) RAPIDS coherence quick check =========================="
# Compare versions that often need to match
# (Best-effort parse from conda list)
if have conda; then
  echo "[libraft/libcugraph/cuml versions]"
  conda list | grep -E '^(libraft|libraft-cu12|libcugraph|libcugraph-cu12|cuml)\b' || echo "(none matched)"
fi
echo

echo "=== 9) Module system (if available) =========================="
if have module; then
  module list 2>&1 | sed 's/^/module: /'
else
  echo "Environment Modules: not present"
fi

echo
echo "=== Done. Interpretations ===================================="
cat <<'NOTE'
- If ALL 'Missing: .../cuda_fp16.h' → your environment lacks CUDA headers; NVRTC cannot include <cuda_fp16.h>.
- If you see both 'cupy' and 'cupy-cuda12x' in pip/conda, that's a duplicate installation risk.
- If RAPIDS packages show mixed minor versions (e.g., libcugraph 25.4, libraft 25.8), expect ABI/runtime errors later.
- If CUDA runtime version is shown but JIT still fails in your pipeline, it's almost always missing headers (this script avoided JIT).
NOTE
