import numpy as np
import pandas as pd
import os
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
import scanpy as sc

from visualization.visualization_helper import generate_umap_visualizations


def cell_types(
    anndata_cell,
    anndata_sample=None,
    cell_type_column="cell_type",
    existing_cell_types=False,
    n_target_clusters=None,
    umap=True,
    save=False,
    output_dir=None,
    defined_output_path=None,
    defined_sample_output_path=None,
    leiden_cluster_resolution=0.8,
    cell_embedding_column="X_pca_harmony",
    cell_embedding_num_PCs=20,
    verbose=True,
    umap_plots=True,
    _recursion_depth=0
):
    MAX_RESOLUTION = 5.0
    RESOLUTION_STEP = 0.5
    MAX_RECURSION_DEPTH = 10

    if _recursion_depth > MAX_RECURSION_DEPTH:
        raise RuntimeError(
            f"Maximum recursion depth exceeded. "
            f"Could not achieve {n_target_clusters} clusters."
        )

    adata = anndata_cell
    indent = "  " * _recursion_depth

    if _recursion_depth == 0:
        from utils.random_seed import set_global_seed
        set_global_seed(seed=42)

    if cell_type_column in adata.obs.columns and existing_cell_types:
        if verbose and _recursion_depth == 0:
            print("[cell_types] Found existing cell type annotation.")

        adata.obs["cell_type"] = adata.obs[cell_type_column].astype(str)
        current_n_types = adata.obs["cell_type"].nunique()

        if verbose:
            print(f"{indent}[cell_types] Current number of cell types: {current_n_types}")

        if n_target_clusters is not None and current_n_types > n_target_clusters:
            if verbose:
                print(
                    f"{indent}[cell_types] Aggregating {current_n_types} → "
                    f"{n_target_clusters} using dendrogram"
                )

            adata = cell_type_dendrogram(
                adata=adata,
                n_clusters=n_target_clusters,
                groupby="cell_type",
                cell_embedding_column=cell_embedding_column,
                cell_embedding_num_PCs=cell_embedding_num_PCs,
            )

        if _recursion_depth == 0:
            if verbose:
                print("[cell_types] Building neighborhood graph...")
            sc.pp.neighbors(
                adata, use_rep=cell_embedding_column, n_pcs=cell_embedding_num_PCs, random_state=42
            )

    else:
        if verbose and _recursion_depth == 0:
            print("[cell_types] No cell type annotation found. Performing clustering.")

        if _recursion_depth == 0:
            if verbose:
                print("[cell_types] Building neighborhood graph...")
            sc.pp.neighbors(
                adata, use_rep=cell_embedding_column, n_pcs=cell_embedding_num_PCs, random_state=42
            )

        if n_target_clusters is not None:
            if verbose:
                print(
                    f"{indent}[cell_types] Target={n_target_clusters}, "
                    f"resolution={leiden_cluster_resolution:.2f}"
                )

            sc.tl.leiden(
                adata,
                resolution=leiden_cluster_resolution,
                flavor="igraph",
                directed=False,
                key_added="cell_type",
                random_state=42,
            )

            adata.obs["cell_type"] = (
                adata.obs["cell_type"].astype(int) + 1
            ).astype(str).astype("category")

            num_clusters_found = adata.obs["cell_type"].nunique()
            if verbose:
                print(f"{indent}[cell_types] Found {num_clusters_found} clusters")

            if num_clusters_found >= n_target_clusters:
                if num_clusters_found > n_target_clusters and verbose:
                    print(
                        f"{indent}[cell_types] Over-shot target; "
                        f"recursing with dendrogram aggregation"
                    )

                return cell_types(
                    anndata_cell=adata,
                    anndata_sample=anndata_sample,
                    cell_type_column="cell_type",
                    existing_cell_types=True,
                    n_target_clusters=n_target_clusters,
                    umap=False,
                    save=False,
                    cell_embedding_column=cell_embedding_column,
                    cell_embedding_num_PCs=cell_embedding_num_PCs,
                    verbose=verbose,
                    umap_plots=False,
                    _recursion_depth=_recursion_depth + 1,
                )

            new_resolution = leiden_cluster_resolution + RESOLUTION_STEP
            if new_resolution <= MAX_RESOLUTION:
                return cell_types(
                    anndata_cell=adata,
                    anndata_sample=anndata_sample,
                    cell_type_column=cell_type_column,
                    existing_cell_types=False,
                    n_target_clusters=n_target_clusters,
                    umap=False,
                    save=False,
                    leiden_cluster_resolution=new_resolution,
                    cell_embedding_column=cell_embedding_column,
                    cell_embedding_num_PCs=cell_embedding_num_PCs,
                    verbose=verbose,
                    umap_plots=False,
                    _recursion_depth=_recursion_depth + 1,
                )

        else:
            if verbose:
                print(
                    f"{indent}[cell_types] Standard Leiden "
                    f"(resolution={leiden_cluster_resolution})"
                )

            sc.tl.leiden(
                adata,
                resolution=leiden_cluster_resolution,
                flavor="igraph",
                directed=False,
                key_added="cell_type",
                random_state=42,
            )

            adata.obs["cell_type"] = (
                adata.obs["cell_type"].astype(int) + 1
            ).astype(str).astype("category")

    if _recursion_depth == 0:
        if verbose:
            print("[cell_types] Finished assigning cell types.")

        if umap:
            if verbose:
                print("[cell_types] Computing UMAP...")
            sc.tl.umap(adata, min_dist=0.5)

        if umap_plots and umap and output_dir:
            if verbose:
                print("[cell_types] Generating UMAP plots...")
            generate_umap_visualizations(
                adata=adata,
                output_dir=output_dir,
                groupby="cell_type",
                figsize=(12, 8),
                point_size=20,
                dpi=300,
                palette="tab20",
                verbose=verbose,
            )

        if anndata_sample is not None:
            if verbose:
                print("[cell_types] Assigning cell_type to anndata_sample")

            if anndata_sample.obs_names.equals(adata.obs_names):
                anndata_sample.obs["cell_type"] = adata.obs["cell_type"].values
            else:
                cell_type_series = pd.Series(
                    adata.obs["cell_type"].values,
                    index=adata.obs_names,
                    name="cell_type",
                )
                anndata_sample.obs["cell_type"] = cell_type_series.reindex(
                    anndata_sample.obs_names
                ).values

        if output_dir:
            preprocess_output_dir = os.path.join(output_dir, "preprocess")
            os.makedirs(preprocess_output_dir, exist_ok=True)

            celltype_df = pd.DataFrame({
                "cell_id": adata.obs.index,
                "cell_type": adata.obs["cell_type"].astype(str),
            })
            csv_path = os.path.join(preprocess_output_dir, "cell_type.csv")
            celltype_df.to_csv(csv_path, index=False)

            if verbose:
                print(f"[cell_types] Saved cell type CSV to {csv_path}")

        if save and output_dir:
            preprocess_output_dir = os.path.join(output_dir, "preprocess")
            os.makedirs(preprocess_output_dir, exist_ok=True)

            cell_save_path = (
                defined_output_path
                or os.path.join(preprocess_output_dir, "adata_cell.h5ad")
            )
            adata.write(cell_save_path)

            if anndata_sample is not None:
                sample_save_path = (
                    defined_sample_output_path
                    or os.path.join(preprocess_output_dir, "adata_sample.h5ad")
                )
                anndata_sample.write(sample_save_path)

            if verbose:
                print("[cell_types] Saved output anndatas to preprocess/")

    if anndata_sample is None:
        return adata
    return adata, anndata_sample


def cell_type_dendrogram(
    adata,
    n_clusters,
    groupby="cell_type",
    cell_embedding_column="X_pca_harmony",
    cell_embedding_num_PCs=20,
    verbose=True,
):
    LINKAGE_METHOD = "average"
    DISTANCE_METRIC = "euclidean"

    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1")
    if groupby not in adata.obs:
        raise ValueError(f"{groupby} not found in adata.obs")
    if cell_embedding_column not in adata.obsm:
        raise ValueError(f"{cell_embedding_column} not found in adata.obsm")

    if cell_embedding_num_PCs is not None and cell_embedding_column.startswith("X_pca"):
        embedding_data = adata.obsm[cell_embedding_column][:, :cell_embedding_num_PCs]
    else:
        embedding_data = adata.obsm[cell_embedding_column]

    embedding_df = pd.DataFrame(embedding_data, index=adata.obs_names)
    embedding_df[groupby] = adata.obs[groupby].values

    cell_type_centroids = embedding_df.groupby(groupby).mean()
    centroid_distance_matrix = pdist(cell_type_centroids.values, metric=DISTANCE_METRIC)
    linkage_matrix = sch.linkage(centroid_distance_matrix, method=LINKAGE_METHOD)

    n_clusters_capped = min(n_clusters, cell_type_centroids.shape[0])
    hierarchical_cluster_labels = fcluster(linkage_matrix, t=n_clusters_capped, criterion="maxclust")
    celltype_to_cluster_mapping = dict(zip(cell_type_centroids.index, map(str, hierarchical_cluster_labels)))

    adata.obs[f"{groupby}_original"] = adata.obs[groupby].copy()
    adata.obs[groupby] = adata.obs[groupby].map(celltype_to_cluster_mapping).astype("category")

    cluster_composition_mapping = {}
    for original_type, new_cluster in celltype_to_cluster_mapping.items():
        cluster_composition_mapping.setdefault(new_cluster, []).append(original_type)

    adata.uns["cluster_mapping"] = cluster_composition_mapping

    if verbose:
        print(f"[cell_type_dendrogram] Aggregated {len(celltype_to_cluster_mapping)} cell types "
              f"into {len(set(hierarchical_cluster_labels))} clusters")

    return adata