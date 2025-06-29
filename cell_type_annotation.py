import os
import scanpy as sc
import celltypist
from celltypist import models
import matplotlib.pyplot as plt
import numpy as np

def annotate_cell_types_with_celltypist(
    adata,
    output_dir,
    model_name=None,
    custom_model_path=None,
    majority_voting=True,
    save=True
):
    # === Validation of model input ===
    if (model_name is None and custom_model_path is None) or (model_name and custom_model_path):
        raise ValueError("You must provide exactly one of `model_name` or `custom_model_path`.")

    # === Prepare output directory ===
    plot_dir = os.path.join(output_dir, "harmony")
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

if __name__ == "__main__":
    input_file = "/dcl01/hongkai/data/data/hjiang/Test/result/harmony/adata_sample.h5ad"
    output_dir = "/dcl01/hongkai/data/data/hjiang/Test/result"
    
    adata = sc.read_h5ad(input_file)

    # Use either `model_name=...` or `custom_model_path=...`
    annotate_cell_types_with_celltypist(
        adata=adata,
        output_dir=output_dir,
        model_name="Healthy_COVID19_PBMC.pkl",  # or custom_model_path="/path/to/model.pkl"
        majority_voting=True,
        save=True
    )
