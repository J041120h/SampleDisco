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
    Save=False,
    output_dir=None,
    defined_output_path=None,
    defined_sample_output_path=None,
    cluster_resolution=0.8,
    use_rep="X_pca_harmony",
    markers=None,
    num_PCs=20,
    verbose=True,
    generate_plots=True,
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
        set_global_seed(seed=42, verbose=verbose)

    if cell_type_column in adata.obs.columns and existing_cell_types:
        if verbose and _recursion_depth == 0:
            print("[cell_types] Found existing cell type annotation.")

        adata.obs["cell_type"] = adata.obs[cell_type_column].astype(str)
        current_n = adata.obs["cell_type"].nunique()

        if verbose:
            print(f"{indent}[cell_types] Current number of cell types: {current_n}")

        if n_target_clusters is not None and current_n > n_target_clusters:
            if verbose:
                print(
                    f"{indent}[cell_types] Aggregating {current_n} → "
                    f"{n_target_clusters} using dendrogram"
                )

            adata = cell_type_dendrogram(
                adata=adata,
                n_clusters=n_target_clusters,
                groupby="cell_type",
                use_rep=use_rep,
                num_PCs=num_PCs,
            )

        if _recursion_depth == 0:
            if verbose:
                print("[cell_types] Building neighborhood graph...")
            sc.pp.neighbors(
                adata, use_rep=use_rep, n_pcs=num_PCs, random_state=42
            )

    else:
        if verbose and _recursion_depth == 0:
            print("[cell_types] No cell type annotation found. Performing clustering.")

        if _recursion_depth == 0:
            if verbose:
                print("[cell_types] Building neighborhood graph...")
            sc.pp.neighbors(
                adata, use_rep=use_rep, n_pcs=num_PCs, random_state=42
            )

        if n_target_clusters is not None:
            if verbose:
                print(
                    f"{indent}[cell_types] Target={n_target_clusters}, "
                    f"resolution={cluster_resolution:.2f}"
                )

            sc.tl.leiden(
                adata,
                resolution=cluster_resolution,
                flavor="igraph",
                directed=False,
                key_added="cell_type",
                random_state=42,
            )

            adata.obs["cell_type"] = (
                adata.obs["cell_type"].astype(int) + 1
            ).astype(str).astype("category")

            n_found = adata.obs["cell_type"].nunique()
            if verbose:
                print(f"{indent}[cell_types] Found {n_found} clusters")

            if n_found >= n_target_clusters:
                if n_found > n_target_clusters and verbose:
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
                    Save=False,
                    use_rep=use_rep,
                    num_PCs=num_PCs,
                    verbose=verbose,
                    generate_plots=False,
                    _recursion_depth=_recursion_depth + 1,
                )

            new_res = cluster_resolution + RESOLUTION_STEP
            if new_res <= MAX_RESOLUTION:
                return cell_types(
                    anndata_cell=adata,
                    anndata_sample=anndata_sample,
                    cell_type_column=cell_type_column,
                    existing_cell_types=False,
                    n_target_clusters=n_target_clusters,
                    umap=False,
                    Save=False,
                    cluster_resolution=new_res,
                    use_rep=use_rep,
                    markers=markers,
                    num_PCs=num_PCs,
                    verbose=verbose,
                    generate_plots=False,
                    _recursion_depth=_recursion_depth + 1,
                )

        else:
            if verbose:
                print(
                    f"{indent}[cell_types] Standard Leiden "
                    f"(resolution={cluster_resolution})"
                )

            sc.tl.leiden(
                adata,
                resolution=cluster_resolution,
                flavor="igraph",
                directed=False,
                key_added="cell_type",
                random_state=42,
            )

            adata.obs["cell_type"] = (
                adata.obs["cell_type"].astype(int) + 1
            ).astype(str).astype("category")

    if _recursion_depth == 0:
        n_final = adata.obs["cell_type"].nunique()
        if markers is not None and len(markers) == n_final:
            if verbose:
                print("[cell_types] Applying marker names")
            mapping = {str(i): markers[i - 1] for i in range(1, n_final + 1)}
            adata.obs["cell_type"] = adata.obs["cell_type"].map(mapping)

        if umap:
            if verbose:
                print("[cell_types] Computing UMAP...")
            sc.tl.umap(adata, min_dist=0.5)

        if generate_plots and umap and output_dir:
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
                series = pd.Series(
                    adata.obs["cell_type"].values,
                    index=adata.obs_names,
                    name="cell_type",
                )
                anndata_sample.obs["cell_type"] = series.reindex(
                    anndata_sample.obs_names
                ).values

        if Save and output_dir:
            preprocess = os.path.join(output_dir, "preprocess")
            os.makedirs(preprocess, exist_ok=True)

            cell_path = (
                defined_output_path
                or os.path.join(preprocess, "adata_cell.h5ad")
            )
            adata.write(cell_path)

            if anndata_sample is not None:
                sample_path = (
                    defined_sample_output_path
                    or os.path.join(preprocess, "adata_sample.h5ad")
                )
                anndata_sample.write(sample_path)

            if verbose:
                print("[cell_types] Saved outputs to preprocess/")

    if anndata_sample is None:
        return adata
    return adata, anndata_sample


def cell_type_dendrogram(
    adata,
    n_clusters,
    groupby="cell_type",
    use_rep="X_pca_harmony",
    num_PCs=20,
):
    METHOD = "average"
    METRIC = "euclidean"

    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1")
    if groupby not in adata.obs:
        raise ValueError(f"{groupby} not found in adata.obs")
    if use_rep not in adata.obsm:
        raise ValueError(f"{use_rep} not found in adata.obsm")

    if num_PCs is not None and use_rep.startswith("X_pca"):
        X = adata.obsm[use_rep][:, :num_PCs]
    else:
        X = adata.obsm[use_rep]

    df = pd.DataFrame(X, index=adata.obs_names)
    df[groupby] = adata.obs[groupby].values

    centroids = df.groupby(groupby).mean()
    dist = pdist(centroids.values, metric=METRIC)
    Z = sch.linkage(dist, method=METHOD)

    labels = fcluster(Z, t=min(n_clusters, centroids.shape[0]), criterion="maxclust")
    mapping = dict(zip(centroids.index, map(str, labels)))

    adata.obs[f"{groupby}_original"] = adata.obs[groupby].copy()
    adata.obs[groupby] = adata.obs[groupby].map(mapping).astype("category")
    adata.uns["cluster_mapping"] = {
        k: list(v)
        for k, v in pd.Series(mapping).groupby(mapping.values()).groups.items()
    }

    return adata