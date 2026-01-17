import os
from typing import Optional

import scanpy as sc
import celltypist
from celltypist import models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pyensembl import EnsemblRelease


def set_global_seed(seed=42):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)


def auto_align_gene_names_with_pyensembl(
    adata_rna,
    ensembl_release: int = 98,
    species: str = "human",
    min_ensembl_fraction: float = 0.2,
):
    """
    Detect if var_names look like Ensembl IDs and, if so, map them to gene symbols
    using pyensembl (given an Ensembl release).

    - Strips version suffixes like '.1', '.2', ...
    - Uses EnsemblRelease(ensembl_release, species)
    - If mapping succeeds for a reasonable fraction of genes, replaces var_names with symbols.
    """
    var_names = pd.Index(adata_rna.var_names.astype(str))

    # Strip version suffix (safe even if they are already symbols)
    stripped = var_names.str.replace(r"\.\d+$", "", regex=True)

    # Heuristic: fraction that look like Ensembl IDs
    ensembl_like_mask = stripped.str.match(r"^ENS[A-Z0-9]+")
    frac_ensembl_like = ensembl_like_mask.mean()
    print(f"[INFO] pyensembl: fraction of Ensembl-like IDs in var_names: {frac_ensembl_like:.3f}")

    if frac_ensembl_like < min_ensembl_fraction:
        print(
            "[INFO] pyensembl: var_names do not look predominantly like Ensembl IDs; "
            "keeping existing gene names."
        )
        return adata_rna  # no change

    print(
        f"[INFO] pyensembl: detected Ensembl-style gene IDs; mapping to symbols "
        f"(Ensembl release {ensembl_release}, species={species})"
    )

    # Initialize / download the Ensembl DB (only first run downloads)
    ensembl = EnsemblRelease(ensembl_release, species=species)
    print("[INFO] pyensembl: building / loading index (this may take a bit the first time)...")
    ensembl.index()

    new_names = []
    n_mapped = 0

    for g in stripped:
        symbol = None

        # Try mapping as a gene ID
        try:
            symbol = ensembl.gene_name_of_gene_id(g)
        except Exception:
            # If that fails, try mapping as a transcript ID → gene ID → gene name
            try:
                gene_id = ensembl.gene_id_of_transcript_id(g)
                symbol = ensembl.gene_name_of_gene_id(gene_id)
            except Exception:
                # Still nothing → leave as is
                symbol = None

        if symbol is not None:
            new_names.append(symbol)
            n_mapped += 1
        else:
            # Fallback: keep the original name if mapping fails
            new_names.append(g)

    mapping_frac = n_mapped / len(stripped)
    print(
        f"[INFO] pyensembl: successfully mapped {n_mapped}/{len(stripped)} genes "
        f"({mapping_frac:.3%}) to gene symbols."
    )

    if n_mapped == 0:
        print("[WARN] pyensembl: no genes were mapped; leaving var_names unchanged.")
        return adata_rna

    # Apply new names and make them unique
    adata_rna.var_names = new_names
    adata_rna.var_names_make_unique()
    print(f"[DEBUG] pyensembl: example mapped gene names: {list(adata_rna.var_names[:5])}")

    return adata_rna


def annotate_and_transfer_cell_types(
    adata,
    output_dir: str,
    model_name: Optional[str] = None,
    custom_model_path: Optional[str] = None,
    majority_voting: bool = True,
    modality_col: str = "modality",
    original_celltype_col: str = "cell_type",   # input column name (overwritten in-place)
    rna_modality_value: str = "RNA",
    atac_modality_value: str = "ATAC",
    ensembl_release: int = 98,
    ensembl_species: str = "human",
):
    """
    Annotate RNA cells using CellTypist and transfer labels to ATAC cells
    based on shared original cell type labels.

    Key constraints (per user requirement):
      - Overwrites the original cell type column in-place (default: 'cell_type')
      - DOES NOT add any new columns to adata.obs
        (no original_barcode, label_transfer_confidence, cell_type_original,
         celltypist_conf_score, etc.)

    Returns
    -------
    (adata, label_mapping)
    """
    set_global_seed(42)

    if (model_name is None and custom_model_path is None) or (model_name and custom_model_path):
        raise ValueError("You must provide exactly one of `model_name` or `custom_model_path`.")

    os.makedirs(output_dir, exist_ok=True)

    # Separate RNA and ATAC cells
    rna_mask = adata.obs[modality_col] == rna_modality_value
    atac_mask = adata.obs[modality_col] == atac_modality_value

    print(f"[INFO] Total cells: {adata.n_obs}")
    print(f"[INFO] RNA cells: {rna_mask.sum()}")
    print(f"[INFO] ATAC cells: {atac_mask.sum()}")

    # Extract RNA cells for CellTypist annotation
    adata_rna = adata[rna_mask].copy()
    print(f"[INFO] RNA subset shape: {adata_rna.n_obs} cells × {adata_rna.n_vars} genes")

    # ======== use pyensembl to align gene names (Ensembl → symbols) =========
    adata_rna = auto_align_gene_names_with_pyensembl(
        adata_rna,
        ensembl_release=ensembl_release,
        species=ensembl_species,
    )
    # ========================================================================

    # Preprocess RNA data for CellTypist (does not touch original adata.X)
    adata_ct = adata_rna.copy()
    sc.pp.normalize_total(adata_ct, target_sum=1e4)
    sc.pp.log1p(adata_ct)
    if not isinstance(adata_ct.X, np.ndarray):
        adata_ct.X = adata_ct.X.toarray()

    # Load CellTypist model
    if model_name:
        print(f"[INFO] Downloading CellTypist model: {model_name}")
        models.download_models(force_update=True, model=[model_name])
        model = models.Model.load(model_name)
    else:
        if not os.path.exists(custom_model_path):
            raise FileNotFoundError(f"Custom model not found: {custom_model_path}")
        print(f"[INFO] Loading custom model from: {custom_model_path}")
        model = models.Model.load(custom_model_path)

    # Overlap diagnostic to prevent cryptic "no features overlap" error
    try:
        model_genes = set(model.genes) if hasattr(model, "genes") else set(model.features)
    except Exception:
        model_genes = set(getattr(model, "features", []))

    adata_genes = set(adata_ct.var_names)
    overlap = model_genes & adata_genes

    print(f"[INFO] #genes in data    : {len(adata_genes)}")
    print(f"[INFO] #genes in model   : {len(model_genes)}")
    print(f"[INFO] #overlapping genes: {len(overlap)}")

    if len(overlap) == 0:
        raise ValueError(
            "No overlapping genes between input AnnData (after pyensembl mapping) "
            "and CellTypist model.\n"
            "Check that:\n"
            "  1) pyensembl species and Ensembl release are correct, and\n"
            "  2) The dataset and model are from the same species.\n"
        )
    else:
        print(f"[DEBUG] Example overlapping genes: {list(overlap)[:10]}")

    # Run CellTypist annotation on RNA cells
    print("[INFO] Running CellTypist annotation on RNA cells...")
    predictions = celltypist.annotate(
        adata_ct,
        model=model,
        majority_voting=majority_voting,
    )
    pred_adata = predictions.to_adata()

    # Get the annotation column name
    annot_col = "majority_voting" if majority_voting else "predicted_labels"

    # Create mapping from original cell type to CellTypist annotation
    rna_original_labels = adata_rna.obs[original_celltype_col].values
    rna_celltypist_labels = pred_adata.obs.loc[adata_rna.obs.index, annot_col].values

    # Build mapping dictionary (original label -> most common CellTypist label)
    label_mapping = {}
    mapping_df = pd.DataFrame(
        {
            "original": rna_original_labels,
            "celltypist": rna_celltypist_labels,
        }
    )

    for orig_label in mapping_df["original"].unique():
        subset = mapping_df[mapping_df["original"] == orig_label]
        most_common = subset["celltypist"].value_counts().idxmax()
        label_mapping[orig_label] = most_common

    print("\n[INFO] Label mapping (original -> CellTypist):")
    for orig, new in sorted(label_mapping.items(), key=lambda x: str(x[0])):
        print(f"  {orig} -> {new}")

    # Overwrite original_celltype_col in-place (NO new obs columns)
    orig_series = adata.obs[original_celltype_col].copy()
    new_series = orig_series.map(label_mapping)

    # Preserve any labels that are not in the mapping dictionary
    unmapped_mask = new_series.isna()
    if unmapped_mask.any():
        print(
            f"[WARN] {unmapped_mask.sum()} cells have cell type labels not in mapping; "
            "keeping their original labels."
        )
        new_series[unmapped_mask] = orig_series[unmapped_mask]

    adata.obs[original_celltype_col] = new_series

    # Save mapping to file (disk only, no obs additions)
    mapping_df_out = pd.DataFrame(
        list(label_mapping.items()), columns=["original_label", "celltypist_label"]
    )
    mapping_path = os.path.join(output_dir, "celltype_mapping.csv")
    mapping_df_out.to_csv(mapping_path, index=False)
    print(f"[INFO] Label mapping saved to: {mapping_path}")

    return adata, label_mapping


def visualize_glue_embedding(
    adata,
    output_dir: str,
    embedding_key: str = "X_glue",
    color_by: str = "cell_type",
    filename: str = "glue_umap_celltype.png",
    n_neighbors: int = 15,
    min_dist: float = 0.5,
):
    """
    Compute UMAP from GLUE embedding and visualize colored by cell type.

    Note: This function does NOT modify adata.obs; it only writes to
    adata.uns, adata.obsm, and adata.obsp.
    """
    os.makedirs(output_dir, exist_ok=True)

    if embedding_key not in adata.obsm:
        raise KeyError(
            f"Embedding key '{embedding_key}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )

    print(f"[INFO] Computing UMAP from {embedding_key}...")

    sc.pp.neighbors(adata, use_rep=embedding_key, n_neighbors=n_neighbors)
    sc.tl.umap(adata, min_dist=min_dist)

    # Plot UMAP colored by cell type
    fig, ax = plt.subplots(figsize=(10, 8))
    sc.pl.umap(
        adata,
        color=color_by,
        ax=ax,
        show=False,
        title=f"GLUE Embedding UMAP - {color_by}",
        legend_loc="on data",
        legend_fontsize=8,
        frameon=False,
    )
    plt.tight_layout()

    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] UMAP plot saved to: {output_path}")

    # Also save a version with legend outside
    fig, ax = plt.subplots(figsize=(12, 8))
    sc.pl.umap(
        adata,
        color=color_by,
        ax=ax,
        show=False,
        title=f"GLUE Embedding UMAP - {color_by}",
        legend_loc="right margin",
        frameon=False,
    )
    plt.tight_layout()

    output_path_legend = os.path.join(
        output_dir, filename.replace(".png", "_legend_outside.png")
    )
    fig.savefig(output_path_legend, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] UMAP plot (legend outside) saved to: {output_path_legend}")

    return adata


def plot_modality_celltype_heatmap(
    adata,
    output_dir: str,
    modality_col: str = "modality",
    celltype_col: str = "cell_type",
    filename: str = "modality_celltype_heatmap.png",
):
    """
    Create a heatmap showing the distribution of RNA/ATAC cells in each cell type.

    Note: This function does NOT modify adata.obs; it only reads from it.
    """
    os.makedirs(output_dir, exist_ok=True)

    crosstab = pd.crosstab(
        adata.obs[celltype_col],
        adata.obs[modality_col],
    )

    crosstab = crosstab.loc[crosstab.sum(axis=1).sort_values(ascending=False).index]

    print("\n[INFO] Cell count distribution (Modality x Cell Type):")
    print(crosstab)

    crosstab.to_csv(os.path.join(output_dir, "modality_celltype_counts.csv"))

    crosstab_norm = crosstab.div(crosstab.sum(axis=1), axis=0)

    # Plot 1: Raw counts heatmap
    fig, ax = plt.subplots(figsize=(8, max(6, len(crosstab) * 0.4)))
    sns.heatmap(
        crosstab,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        cbar_kws={"label": "Cell Count"},
    )
    ax.set_title("Cell Count Distribution: Modality × Cell Type")
    ax.set_xlabel("Modality")
    ax.set_ylabel("Cell Type")
    plt.tight_layout()

    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Count heatmap saved to: {output_path}")

    # Plot 2: Normalized heatmap (proportions)
    fig, ax = plt.subplots(figsize=(8, max(6, len(crosstab) * 0.4)))
    sns.heatmap(
        crosstab_norm,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        ax=ax,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Proportion"},
    )
    ax.set_title("Modality Proportion within Each Cell Type")
    ax.set_xlabel("Modality")
    ax.set_ylabel("Cell Type")
    plt.tight_layout()

    output_path_norm = os.path.join(
        output_dir, filename.replace(".png", "_normalized.png")
    )
    fig.savefig(output_path_norm, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Normalized heatmap saved to: {output_path_norm}")

    # Plot 3: Stacked bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    crosstab.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
    ax.set_title("Cell Count Distribution by Cell Type and Modality")
    ax.set_xlabel("Cell Type")
    ax.set_ylabel("Cell Count")
    ax.legend(title="Modality")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path_bar = os.path.join(output_dir, "modality_celltype_barplot.png")
    fig.savefig(output_path_bar, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Stacked bar plot saved to: {output_path_bar}")

    return crosstab


def run_full_pipeline(
    h5ad_path: str,
    output_dir: str,
    celltypist_model_name: Optional[str] = None,
    custom_model_path: Optional[str] = None,
    majority_voting: bool = True,
    modality_col: str = "modality",
    original_celltype_col: str = "cell_type",
    rna_modality_value: str = "RNA",
    atac_modality_value: str = "ATAC",
    embedding_key: str = "X_glue",
    overwrite_input_h5ad: bool = True,   # overwrite input by default
    ensembl_release: int = 98,
    ensembl_species: str = "human",
):
    """
    Run the full pipeline:
    1. Annotate RNA cells with CellTypist and transfer labels to ATAC cells
       (overwrite cell_type in-place, no new obs columns)
    2. Visualize GLUE embedding with UMAP colored by cell type
    3. Create heatmap of modality distribution across cell types
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Loading AnnData from: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"[INFO] Shape: {adata.n_obs} cells × {adata.n_vars} genes/peaks")
    print(f"[INFO] Modalities: {adata.obs[modality_col].value_counts().to_dict()}")

    # Step 1: Annotate cell types
    print("\n" + "=" * 60)
    print("STEP 1: Cell Type Annotation and Transfer")
    print("=" * 60)
    adata, label_mapping = annotate_and_transfer_cell_types(
        adata=adata,
        output_dir=output_dir,
        model_name=celltypist_model_name,
        custom_model_path=custom_model_path,
        majority_voting=majority_voting,
        modality_col=modality_col,
        original_celltype_col=original_celltype_col,  # this column is overwritten in-place
        rna_modality_value=rna_modality_value,
        atac_modality_value=atac_modality_value,
        ensembl_release=ensembl_release,
        ensembl_species=ensembl_species,
    )

    # Step 2: Visualize GLUE embedding
    print("\n" + "=" * 60)
    print("STEP 2: GLUE Embedding Visualization")
    print("=" * 60)
    adata = visualize_glue_embedding(
        adata=adata,
        output_dir=output_dir,
        embedding_key=embedding_key,
        color_by=original_celltype_col,  # now it's 'cell_type' (overwritten)
    )

    # Also visualize by modality
    fig, ax = plt.subplots(figsize=(10, 8))
    sc.pl.umap(
        adata,
        color=modality_col,
        ax=ax,
        show=False,
        title="GLUE Embedding UMAP - Modality",
        frameon=False,
    )
    plt.tight_layout()
    modality_umap_path = os.path.join(output_dir, "glue_umap_modality.png")
    fig.savefig(modality_umap_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Modality UMAP saved to: {modality_umap_path}")

    # Step 3: Modality-Cell Type heatmap
    print("\n" + "=" * 60)
    print("STEP 3: Modality-Cell Type Distribution Heatmap")
    print("=" * 60)
    _ = plot_modality_celltype_heatmap(
        adata=adata,
        output_dir=output_dir,
        modality_col=modality_col,
        celltype_col=original_celltype_col,  # now 'cell_type'
    )

    # Overwrite or save new h5ad
    if overwrite_input_h5ad:
        output_h5ad = h5ad_path
        print(f"\n[INFO] Overwriting input AnnData file: {output_h5ad}")
    else:
        output_h5ad = os.path.join(output_dir, "adata_annotated.h5ad")
        print(f"\n[INFO] Saving annotated AnnData to: {output_h5ad}")

    adata.write_h5ad(output_h5ad, compression="gzip")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"[INFO] All outputs saved to: {output_dir}")
    print(f"[INFO] H5AD written to: {output_h5ad}")

    return adata


if __name__ == "__main__":
    # ====== EDIT THESE PATHS ======
    H5AD_PATH = (
        "/dcs07/hongkai/data/harry/result/multi_omics_unpaired/"
        "multiomics/preprocess/adata_sample.h5ad"
    )
    OUTPUT_DIR = (
        "/dcs07/hongkai/data/harry/result/multi_omics_unpaired/"
        "multiomics/preprocess/annotation_celltypist"
    )

    adata = run_full_pipeline(
        h5ad_path=H5AD_PATH,
        output_dir=OUTPUT_DIR,
        celltypist_model_name=None,  # Set to None when using custom model
        custom_model_path="/dcl01/hongkai/data/data/hjiang/Data/Adult_COVID19_PBMC.pkl",
        majority_voting=True,
        modality_col="modality",
        original_celltype_col="cell_type",  # will be overwritten in-place
        rna_modality_value="RNA",
        atac_modality_value="ATAC",
        embedding_key="X_glue",
        overwrite_input_h5ad=True,  # overwrites H5AD_PATH
        ensembl_release=98,
        ensembl_species="human",
    )
