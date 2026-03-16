#!/usr/bin/env python3
import os
import scanpy as sc
import celltypist
from celltypist import models
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple


def _ensure_dense_X(adata) -> None:
    """CellTypist expects dense matrix in many setups."""
    if not isinstance(adata.X, np.ndarray):
        adata.X = adata.X.toarray()


def _load_celltypist_model(model_name: Optional[str], custom_model_path: Optional[str]):
    if (model_name is None and custom_model_path is None) or (model_name and custom_model_path):
        raise ValueError("You must provide exactly one of `model_name` or `custom_model_path`.")

    if model_name:
        models.download_models(force_update=True, model=[model_name])
        return models.Model.load(model_name)

    if not os.path.exists(custom_model_path):
        raise FileNotFoundError(f"Custom model not found: {custom_model_path}")
    return models.Model.load(custom_model_path)


def annotate_cell_types_with_celltypist(
    adata,
    output_dir,
    model_name=None,
    custom_model_path=None,
    majority_voting=True,
    save=True
):
    """
    Annotate a single AnnData with CellTypist.
    NOTE: Assumes adata.X is already normalized + log-transformed (as user stated).
    """
    # === Validation of model input ===
    from utils.random_seed import set_global_seed
    set_global_seed(seed=42)

    # === Prepare output directory ===
    plot_dir = os.path.join(output_dir, "cell_annotated")
    os.makedirs(plot_dir, exist_ok=True)

    # === Make a copy and preprocess for CellTypist (do NOT modify original .X) ===
    adata_ct = adata.copy()
    _ensure_dense_X(adata_ct)

    # === Load model ===
    model = _load_celltypist_model(model_name=model_name, custom_model_path=custom_model_path)

    # Ensure enough PCs for CellTypist majority voting / over-clustering
    if "X_pca" not in adata_ct.obsm or adata_ct.obsm["X_pca"].shape[1] < 50:
        sc.pp.pca(adata_ct, n_comps=50, svd_solver="arpack")

    # === Run CellTypist annotation ===
    predictions = celltypist.annotate(
        adata_ct,
        model=model,
        majority_voting=majority_voting
    )
    pred_adata = predictions.to_adata()

    # === Assign predicted labels to original AnnData (no .X change) ===
    # Align by obs index
    if not adata.obs.index.equals(pred_adata.obs.index):
        # keep only overlapping, then reindex to adata order
        overlap = adata.obs.index.intersection(pred_adata.obs.index)
        if len(overlap) == 0:
            raise ValueError(
                "No overlapping cell barcodes between input AnnData and CellTypist output. "
                "Cannot transfer labels."
            )
        pred_adata = pred_adata[overlap].copy()

    # majority_voting column exists only if majority_voting=True; otherwise CellTypist uses 'predicted_labels'
    label_col = "majority_voting" if majority_voting else "predicted_labels"
    if label_col not in pred_adata.obs.columns:
        # fallback safety
        for cand in ["majority_voting", "predicted_labels"]:
            if cand in pred_adata.obs.columns:
                label_col = cand
                break

    adata.obs["cell_type"] = pred_adata.obs.reindex(adata.obs.index)[label_col].values
    if "conf_score" in pred_adata.obs.columns:
        adata.obs["celltypist_conf_score"] = pred_adata.obs.reindex(adata.obs.index)["conf_score"].values

    # === Cell type bar plot ===
    plt.figure(figsize=(8, 6))
    adata.obs["cell_type"].value_counts().plot(kind="bar")
    plt.title("Predicted Cell Types")
    plt.ylabel("Number of Cells")
    plt.xlabel("Cell Type")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "celltypist_cell_label_distribution.png"))
    plt.close()

    # === UMAP if not yet computed (do not change .X) ===
    if "X_umap" not in adata.obsm:
        sc.pp.pca(adata, svd_solver="arpack")
        sc.pp.neighbors(adata, n_pcs=20)
        sc.tl.umap(adata)

    # === UMAP plot ===
    fig = sc.pl.umap(
        adata,
        color="cell_type",
        title="CellTypist Annotated Cell Types",
        show=False,
        return_fig=True
    )
    fig.savefig(os.path.join(plot_dir, "celltypist_umap.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # === Save AnnData if needed ===
    if save:
        output_file = os.path.join(plot_dir, "adata_sample.h5ad")
        adata.write(output_file)
        print(f"[INFO] Annotated AnnData saved to: {output_file}")

    return adata


def _transfer_celltypist_labels(
    adata_src,
    adata_tgt,
    cell_type_key: str = "cell_type",
    conf_key: str = "celltypist_conf_score",
) -> Tuple[int, int]:
    """
    Transfer celltypist columns from adata_src.obs -> adata_tgt.obs by matching obs_names.

    Returns:
      (n_matched, n_total_tgt)
    """
    if cell_type_key not in adata_src.obs.columns:
        raise ValueError(f"Source AnnData missing '{cell_type_key}' in .obs; run CellTypist first.")

    overlap = adata_tgt.obs.index.intersection(adata_src.obs.index)
    if len(overlap) == 0:
        raise ValueError(
            "No overlapping cell barcodes between annotation AnnData and raw-count AnnData. "
            "Cannot transfer labels."
        )

    # Write labels into target (only for overlapping cells; others become NaN)
    adata_tgt.obs[cell_type_key] = adata_src.obs.reindex(adata_tgt.obs.index)[cell_type_key].values

    if conf_key in adata_src.obs.columns:
        adata_tgt.obs[conf_key] = adata_src.obs.reindex(adata_tgt.obs.index)[conf_key].values

    return len(overlap), adata_tgt.n_obs


def run_celltypist_transfer_two_h5ad(
    h5ad_path_for_annotation: str,
    h5ad_path_raw_counts: str,
    output_dir: str,
    celltypist_model_name: Optional[str] = None,
    custom_model_path: Optional[str] = None,
    majority_voting: bool = True,
) -> None:
    """
    NEW functionality:
      - Load 2 h5ad files
      - Run CellTypist on the FIRST (already normalized+log) AnnData
      - Transfer inferred labels to the SECOND (raw counts) AnnData by matching obs_names
      - Overwrite BOTH original h5ad files with new .obs columns (do NOT change .X for either)
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Load annotation AnnData (normalized/log) ---
    print(f"[INFO] Loading annotation AnnData (normalized/log) from: {h5ad_path_for_annotation}")
    adata_norm = sc.read_h5ad(h5ad_path_for_annotation)
    print(f"[INFO] Annotation shape: {adata_norm.n_obs} cells × {adata_norm.n_vars} genes")

    # --- Run CellTypist annotation on normalized/log adata ---
    adata_norm = annotate_cell_types_with_celltypist(
        adata=adata_norm,
        output_dir=output_dir,
        model_name=celltypist_model_name,
        custom_model_path=custom_model_path,
        majority_voting=majority_voting,
        save=True,  # keep your existing behavior (also writes a copy in output_dir/cell_annotated)
    )

    # --- Load raw-count AnnData ---
    print(f"[INFO] Loading raw-count AnnData from: {h5ad_path_raw_counts}")
    adata_raw = sc.read_h5ad(h5ad_path_raw_counts)
    print(f"[INFO] Raw-count shape: {adata_raw.n_obs} cells × {adata_raw.n_vars} genes")

    # --- Transfer labels ---
    n_matched, n_total = _transfer_celltypist_labels(adata_src=adata_norm, adata_tgt=adata_raw)
    print(f"[INFO] Transferred labels for {n_matched}/{n_total} cells (matched by obs_names).")

    # --- Overwrite BOTH original .h5ad files (no .X change) ---
    print(f"[INFO] Writing annotated AnnData back to (overwrite): {h5ad_path_for_annotation}")
    adata_norm.write_h5ad(h5ad_path_for_annotation, compression="gzip")

    print(f"[INFO] Writing raw-count AnnData with transferred labels back to (overwrite): {h5ad_path_raw_counts}")
    adata_raw.write_h5ad(h5ad_path_raw_counts, compression="gzip")

    print("[INFO] Two-h5ad CellTypist annotation + transfer complete.")


def run_celltypist_only(
    h5ad_path: str,
    output_dir: str,
    celltypist_model_name: Optional[str] = None,
    custom_model_path: Optional[str] = None,
    majority_voting: bool = True,
) -> None:
    """
    Original behavior preserved:
      - Load AnnData
      - Run CellTypist annotation ONLY
      - Overwrite the SAME .h5ad file
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Loading AnnData from: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"[INFO] Shape: {adata.n_obs} cells × {adata.n_vars} genes")

    adata = annotate_cell_types_with_celltypist(
        adata=adata,
        output_dir=output_dir,
        model_name=celltypist_model_name,
        custom_model_path=custom_model_path,
        majority_voting=majority_voting,
    )

    print(f"[INFO] Writing annotated AnnData back to: {h5ad_path}")
    adata.write_h5ad(h5ad_path, compression="gzip")
    print("[INFO] CellTypist-only annotation complete.")


if __name__ == "__main__":
    # ====== EDIT THESE PATHS MANUALLY ======
    # 1) normalized + log-transformed h5ad for CellTypist
    H5AD_NORM_PATH = "/dcs07/hongkai/data/harry/result/long_covid/rna/preprocess/adata_cell.h5ad"
    # 2) raw-count h5ad to receive transferred labels
    H5AD_RAW_PATH  = "/dcs07/hongkai/data/harry/result/long_covid/rna/preprocess/adata_sample.h5ad"
    OUTPUT_DIR     = "/dcs07/hongkai/data/harry/result/long_covid/rna/preprocess"

    # --- NEW: run two-h5ad mode (annotation + transfer + overwrite both) ---
    run_celltypist_transfer_two_h5ad(
        h5ad_path_for_annotation=H5AD_NORM_PATH,
        h5ad_path_raw_counts=H5AD_RAW_PATH,
        output_dir=OUTPUT_DIR,
        celltypist_model_name=None,  # keep None when using a custom .pkl
        custom_model_path="/users/hjiang/GenoDistance/long_covid/PaediatricAdult_COVID19_PBMC.pkl",
        majority_voting=True,
    )