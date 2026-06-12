Quarantined dead code (NOT deleted) — 2026-06-11

EMD.py: a superseded/duplicate EMD implementation. The LIVE EMD path is
sample_distance/sample_distance.py::compute_emd_distances (called via emd_distance()).
EMD.py was imported nowhere in the package. Moved here instead of deleting; fully
reversible — to restore: move it back to sample_distance/.

--- 2026-06-12 P2 hygiene ---
consumer.py: intentional OOM "allocate-until-crash" script that lived in utils/ (a shared package). Imported nowhere. Quarantined, not deleted.
Kmeans_cluster.py: legacy folder-based k-means clustering, superseded by sample_clustering/cluster.py (the live wrapper path). Imported nowhere. Quarantined, not deleted.
requirement.txt.pipfreeze: original non-installable pip-freeze dump (~90 `@ file://` conda artifacts). Replaced by requirements.txt (core) + requirements-gpu.txt. Quarantined, not deleted.

--- 2026-06-12 P0-1 cleanup ---
ATAC_cell_type.py, ATAC_cell_type_gpu.py: legacy ATAC-specific cell-typing modules. The live ATAC path routes through cell_type_{cpu,gpu}.py (atac_wrapper dispatches cell_types/cell_types_gpu). These contained residual rsc.tl.leiden (cuGraph) — a CPU/GPU-parity footgun. Imported nowhere. Quarantined, not deleted.
