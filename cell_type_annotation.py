import os
import scanpy as sc
import celltypist
from celltypist import models
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

def annotate_cell_types_with_celltypist(
    adata,
    output_dir,
    model_name=None,
    custom_model_path=None,
    majority_voting=True,
    save=True
):
    # === Validation of model input ===
    from utils.random_seed import set_global_seed
    set_global_seed(seed = 42)
    if (model_name is None and custom_model_path is None) or (model_name and custom_model_path):
        raise ValueError("You must provide exactly one of `model_name` or `custom_model_path`.")

    # === Prepare output directory ===
    plot_dir = os.path.join(output_dir, "preprocess")
    os.makedirs(plot_dir, exist_ok=True)

    # === Make a copy and preprocess for CellTypist ===
    adata_ct = adata.copy()
    sc.pp.normalize_total(adata_ct, target_sum=1e4)
    sc.pp.log1p(adata_ct)
    if not isinstance(adata_ct.X, np.ndarray):
        adata_ct.X = adata_ct.X.toarray()

    # === Load model ===
    if model_name:
        models.download_models(force_update=True, model=[model_name])
        model = models.Model.load(model_name)
    else:
        if not os.path.exists(custom_model_path):
            raise FileNotFoundError(f"Custom model not found: {custom_model_path}")
        model = models.Model.load(custom_model_path)

    # === Run CellTypist annotation ===
    predictions = celltypist.annotate(
        adata_ct,
        model=model,
        majority_voting=majority_voting
    )
    pred_adata = predictions.to_adata()

    # === Assign predicted labels to original AnnData ===
    if not adata.obs.index.equals(pred_adata.obs.index):
        pred_adata = pred_adata[pred_adata.obs.index.isin(adata.obs.index)]
    adata.obs["cell_type"] = pred_adata.obs.loc[adata.obs.index, "majority_voting"]
    adata.obs["celltypist_conf_score"] = pred_adata.obs.loc[adata.obs.index, "conf_score"]

    # === Cell type bar plot ===
    plt.figure(figsize=(8, 6))
    adata.obs["cell_type"].value_counts().plot(kind='bar')
    plt.title("Predicted Cell Types")
    plt.ylabel("Number of Cells")
    plt.xlabel("Cell Type")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "celltypist_cell_label_distribution.png"))
    plt.close()

    # === UMAP if not yet computed ===
    if 'X_umap' not in adata.obsm:
        sc.pp.pca(adata, svd_solver='arpack')
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

def run_celltypist_only(
    h5ad_path: str,
    output_dir: str,
    celltypist_model_name: Optional[str] = None,
    custom_model_path: Optional[str] = None,
    majority_voting: bool = True,
) -> None:
    """
    Load AnnData, run CellTypist annotation ONLY, and overwrite the SAME .h5ad file.

    This is a stripped-down wrapper around `annotate_cell_types_with_celltypist`:
      - no Leiden, no marker analysis, no extra steps
      - same model loading behavior as in the full pipeline
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Loading AnnData from: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"[INFO] Shape: {adata.n_obs} cells × {adata.n_vars} genes")

    # --- Run CellTypist annotation only ---
    adata = annotate_cell_types_with_celltypist(
        adata=adata,
        output_dir=output_dir,
        model_name=celltypist_model_name,
        custom_model_path=custom_model_path,
        majority_voting=majority_voting,
    )

    # --- Overwrite the same .h5ad file ---
    print(f"[INFO] Writing annotated AnnData back to: {h5ad_path}")
    adata.write_h5ad(h5ad_path, compression="gzip")
    print("[INFO] CellTypist-only annotation complete.")


if __name__ == "__main__":
    # ====== EDIT THESE PATHS MANUALLY ======
    H5AD_PATH = "/dcs07/hongkai/data/harry/result/long_covid/rna/preprocess/adata_cell.h5ad"
    OUTPUT_DIR = "/dcs07/hongkai/data/harry/result/long_covid/rna/preprocess"

    run_celltypist_only(
        h5ad_path=H5AD_PATH,
        output_dir=OUTPUT_DIR,
        celltypist_model_name=None,  # keep None when using a custom .pkl
        # Same model location/usage pattern as your example:
        custom_model_path="/users/hjiang/GenoDistance/long_covid/PaediatricAdult_COVID19_PBMC.pkl",
        majority_voting=True,
    )
