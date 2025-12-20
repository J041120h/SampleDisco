import numpy as np
import pandas as pd
import os
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
import rapids_singlecell as rsc
import scanpy as sc
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from visualization.visualization_helper import generate_umap_visualizations
from utils.safe_save import safe_h5ad_write, ensure_cpu_arrays


def cell_types_linux(
    anndata_cell,
    anndata_sample=None,
    cell_type_column="cell_type",
    existing_cell_types=False,
    n_target_clusters=None,
    umap=True,
    save=False,
    output_dir=None,
    defined_output_path=None,
    cluster_resolution=0.8,
    use_rep="X_pca_harmony",
    markers=None,
    num_PCs=20,
    verbose=True,
    generate_plots=True,
    _recursion_depth=0,
):
    adata = anndata_cell

    MAX_RESOLUTION = 5.0
    RESOLUTION_STEP = 0.5
    MAX_RECURSION_DEPTH = 10

    if _recursion_depth > MAX_RECURSION_DEPTH:
        raise RuntimeError(
            f"Maximum recursion depth exceeded. Could not achieve {n_target_clusters} clusters."
        )

    if _recursion_depth == 0:
        from utils.random_seed import set_global_seed

        set_global_seed(seed=42, verbose=verbose)
        rsc.get.anndata_to_GPU(adata)

    indent = "  " * _recursion_depth

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
                    f"{indent}[cell_types] Aggregating {current_n_types} cell types "
                    f"into {n_target_clusters} clusters using dendrogram."
                )
                print(
                    f"{indent}[cell_types] Using dimension reduction ({use_rep}) "
                    f"for dendrogram construction..."
                )

            adata = cell_type_dendrogram_linux(
                adata=adata,
                n_clusters=n_target_clusters,
                groupby="cell_type",
                use_rep=use_rep,
                num_PCs=num_PCs,
                verbose=verbose,
            )

            if verbose:
                final_n_types = adata.obs["cell_type"].nunique()
                print(
                    f"{indent}[cell_types] Successfully aggregated to {final_n_types} cell types."
                )

        elif n_target_clusters is not None and verbose:
            print(
                f"{indent}[cell_types] Current cell types ({current_n_types}) <= "
                f"target clusters ({n_target_clusters}). Using as-is."
            )

        if _recursion_depth == 0:
            if verbose:
                print("[cell_types] Building neighborhood graph...")
            rsc.pp.neighbors(adata, use_rep=use_rep, n_pcs=num_PCs, random_state=42)

    else:
        if verbose and _recursion_depth == 0:
            print("[cell_types] No cell type annotation found. Performing clustering.")

        if _recursion_depth == 0:
            if verbose:
                print("[cell_types] Building neighborhood graph...")
            rsc.pp.neighbors(adata, use_rep=use_rep, n_pcs=num_PCs, random_state=42)

        if n_target_clusters is not None:
            if verbose:
                print(
                    f"{indent}[cell_types] Target: {n_target_clusters} clusters. "
                    f"Trying resolution: {cluster_resolution:.1f}"
                )

            rsc.tl.leiden(
                adata,
                resolution=cluster_resolution,
                key_added="cell_type",
                random_state=42,
            )
            adata.obs["cell_type"] = (
                (adata.obs["cell_type"].astype(int) + 1).astype(str).astype("category")
            )
            num_clusters = adata.obs["cell_type"].nunique()

            if verbose:
                print(
                    f"{indent}[cell_types] Leiden clustering produced {num_clusters} clusters."
                )

            if num_clusters >= n_target_clusters:
                if num_clusters == n_target_clusters:
                    if verbose:
                        print(
                            f"{indent}[cell_types] Perfect! Got exactly {n_target_clusters} clusters."
                        )
                else:
                    if verbose:
                        print(
                            f"{indent}[cell_types] Got {num_clusters} clusters (>= target). "
                            f"Recursing with existing_cell_types=True..."
                        )

                    return cell_types_linux(
                        anndata_cell=adata,
                        anndata_sample=anndata_sample,
                        cell_type_column="cell_type",
                        existing_cell_types=True,
                        n_target_clusters=n_target_clusters,
                        umap=False,
                        save=False,
                        use_rep=use_rep,
                        num_PCs=num_PCs,
                        _recursion_depth=_recursion_depth + 1,
                        verbose=verbose,
                        generate_plots=False,
                    )
            else:
                new_resolution = cluster_resolution + RESOLUTION_STEP

                if new_resolution > MAX_RESOLUTION:
                    if verbose:
                        print(
                            f"{indent}[cell_types] Warning: Reached max resolution ({MAX_RESOLUTION}). "
                            f"Got {num_clusters} clusters instead of {n_target_clusters}."
                        )
                else:
                    if verbose:
                        print(
                            f"{indent}[cell_types] Need more clusters. "
                            f"Increasing resolution to {new_resolution:.1f}..."
                        )

                    return cell_types_linux(
                        anndata_cell=adata,
                        anndata_sample=anndata_sample,
                        cell_type_column=cell_type_column,
                        existing_cell_types=False,
                        n_target_clusters=n_target_clusters,
                        umap=False,
                        save=False,
                        cluster_resolution=new_resolution,
                        use_rep=use_rep,
                        markers=markers,
                        num_PCs=num_PCs,
                        _recursion_depth=_recursion_depth + 1,
                        verbose=verbose,
                        generate_plots=False,
                    )

        else:
            if verbose:
                print(
                    f"{indent}[cell_types] No target clusters specified. "
                    f"Using standard Leiden clustering (resolution={cluster_resolution})..."
                )

            rsc.tl.leiden(
                adata,
                resolution=cluster_resolution,
                key_added="cell_type",
                random_state=42,
            )
            adata.obs["cell_type"] = (
                (adata.obs["cell_type"].astype(int) + 1).astype(str).astype("category")
            )

            if verbose:
                num_clusters = adata.obs["cell_type"].nunique()
                print(f"[cell_types] Found {num_clusters} clusters after Leiden clustering.")

    if _recursion_depth == 0:
        final_cluster_count = adata.obs["cell_type"].nunique()
        if markers is not None:
            if len(markers) == final_cluster_count:
                if verbose:
                    print(
                        f"[cell_types] Applying custom marker names to {final_cluster_count} clusters..."
                    )
                marker_dict = {str(i): markers[i - 1] for i in range(1, len(markers) + 1)}
                adata.obs["cell_type"] = adata.obs["cell_type"].map(marker_dict)
            elif verbose:
                print(
                    f"[cell_types] Warning: Marker list length ({len(markers)}) "
                    f"doesn't match cluster count ({final_cluster_count}). Skipping marker mapping."
                )

        if verbose:
            print("[cell_types] Finished assigning cell types.")

        if umap:
            if verbose:
                print("[cell_types] Computing UMAP...")
            rsc.tl.umap(adata, min_dist=0.5)

        if verbose:
            print("[cell_types] Converting GPU arrays to CPU...")
        rsc.get.anndata_to_CPU(adata)
        adata = ensure_cpu_arrays(adata)

        if generate_plots and umap and output_dir:
            if verbose:
                print("[cell_types] Generating UMAP visualizations...")
            adata = generate_umap_visualizations(
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
                print("[cell_types] Assigning cell_type labels to anndata_sample...")

            if (
                anndata_sample.n_obs == adata.n_obs
                and anndata_sample.obs_names.equals(adata.obs_names)
            ):
                anndata_sample.obs["cell_type"] = adata.obs["cell_type"].astype(str).values
            else:
                common = anndata_sample.obs_names.intersection(adata.obs_names)
                if len(common) == 0:
                    raise ValueError(
                        "[cell_types] anndata_sample and anndata_cell share 0 common obs_names; "
                        "cannot safely assign cell_type."
                    )

                anndata_sample.obs["cell_type"] = pd.Series(
                    pd.NA, index=anndata_sample.obs_names, dtype="object"
                )
                anndata_sample.obs.loc[common, "cell_type"] = (
                    adata.obs.loc[common, "cell_type"].astype(str).values
                )

                if verbose and len(common) != anndata_sample.n_obs:
                    missing = anndata_sample.n_obs - len(common)
                    print(
                        f"[cell_types] Warning: Only assigned cell_type for {len(common)}/{anndata_sample.n_obs} "
                        f"cells in anndata_sample (missing {missing})."
                    )

            anndata_sample = ensure_cpu_arrays(anndata_sample)

        if save and output_dir and not defined_output_path:
            out_pre = os.path.join(output_dir, "preprocess")
            os.makedirs(out_pre, exist_ok=True)
            save_path = os.path.join(out_pre, "adata_cell.h5ad")
            safe_h5ad_write(adata, save_path, verbose=verbose)

            if anndata_sample is not None:
                sample_path = os.path.join(out_pre, "adata_sample.h5ad")
                safe_h5ad_write(anndata_sample, sample_path, verbose=verbose)

        if defined_output_path:
            safe_h5ad_write(adata, defined_output_path, verbose=verbose)

    if anndata_sample is None:
        return adata
    return adata, anndata_sample


def cell_type_dendrogram_linux(
    adata,
    n_clusters,
    groupby="cell_type",
    use_rep="X_pca_harmony",
    num_PCs=20,
    verbose=True,
):
    METHOD = "average"
    METRIC = "euclidean"
    DISTANCE_MODE = "centroid"

    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1")

    if groupby not in adata.obs.columns:
        raise ValueError(f"The groupby key '{groupby}' is not present in adata.obs.")

    if use_rep not in adata.obsm:
        raise ValueError(f"The representation '{use_rep}' is not present in adata.obsm.")

    if verbose:
        print(f"=== Preparing data for dendrogram (using {use_rep}) ===")

    temp_adata = adata.copy()
    obsm_data = (
        temp_adata.obsm[use_rep].get()
        if hasattr(temp_adata.obsm[use_rep], "get")
        else temp_adata.obsm[use_rep]
    )

    if num_PCs is not None and use_rep.startswith("X_pca"):
        dim_data = obsm_data[:, :num_PCs]
        if verbose:
            print(f"Using first {num_PCs} components from {use_rep}")
    else:
        dim_data = obsm_data
        if verbose:
            print(f"Using all {dim_data.shape[1]} components from {use_rep}")

    df_dims = pd.DataFrame(
        dim_data,
        index=adata.obs_names,
        columns=[f"PC{i+1}" for i in range(dim_data.shape[1])],
    )
    df_dims[groupby] = adata.obs[groupby].values

    if verbose:
        print(f"=== Computing centroids of cell types in {use_rep} space ===")

    centroids = df_dims.groupby(groupby).mean()
    original_n_types = centroids.shape[0]

    if verbose:
        print(f"Calculated centroids for {original_n_types} cell types.")
        print(f"Centroid shape: {centroids.shape}")
        print(f"=== Computing distance matrix between centroids using {METRIC} distance ===")

    dist_matrix = pdist(centroids.values, metric=METRIC)

    if verbose:
        print(f"=== Performing hierarchical clustering on {use_rep} centroids ===")
        print(f"Linkage method: {METHOD}, Distance metric: {METRIC}")

    Z = sch.linkage(dist_matrix, method=METHOD)
    adata.uns["cell_type_linkage"] = Z

    n_clusters = min(n_clusters, original_n_types)
    if verbose and n_clusters < original_n_types:
        print(f"=== Aggregating cell types into {n_clusters} clusters ===")
    elif verbose:
        print(
            f"Warning: Requested {n_clusters} clusters matches original {original_n_types} types. "
            f"No aggregation needed."
        )

    cluster_labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    actual_n_clusters = len(np.unique(cluster_labels))

    if verbose:
        print(f"Successfully created {actual_n_clusters} clusters")

    celltype_to_cluster = dict(
        zip(centroids.index, [str(label) for label in cluster_labels])
    )

    adata.obs[f"{groupby}_original"] = adata.obs[groupby].copy()
    adata.obs[groupby] = adata.obs[groupby].map(celltype_to_cluster).astype("category")

    cluster_mapping = {}
    for original_type, new_cluster in celltype_to_cluster.items():
        cluster_mapping.setdefault(new_cluster, []).append(original_type)

    adata.uns["cluster_mapping"] = cluster_mapping

    if verbose:
        print("\n=== Cluster Composition ===")
        for cluster_id in sorted(cluster_mapping.keys()):
            original_types = cluster_mapping[cluster_id]
            print(f'Cluster {cluster_id}: {", ".join(map(str, sorted(original_types)))}')

        print("\n=== Cluster Quality Metrics ===")
        for cluster_id in sorted(cluster_mapping.keys()):
            cluster_types = cluster_mapping[cluster_id]
            if len(cluster_types) > 1:
                cluster_centroids = centroids.loc[cluster_types]
                if cluster_centroids.shape[0] > 1:
                    within_cluster_dist = pdist(cluster_centroids.values, metric=METRIC)
                    avg_dist = np.mean(within_cluster_dist)
                    print(
                        f"Cluster {cluster_id}: Average within-cluster distance = {avg_dist:.4f}"
                    )

    return adata