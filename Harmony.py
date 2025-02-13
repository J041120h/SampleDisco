import os
import numpy as np
import pandas as pd
import scanpy as sc
import harmonypy as hm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Local imports from your project
from HierarchicalConstruction import cell_type_dendrogram
from HVG import find_hvgs
from Grouping import find_sample_grouping

def treecor_harmony(h5ad_path,
                    sample_meta_path,
                    output_dir,
                    cell_meta_path=None,
                    markers=None,
                    cluster_resolution=1,
                    num_PCs=30,
                    num_harmony=10,
                    num_features=2000,
                    min_cells=5,
                    min_features=5,
                    pct_mito_cutoff=20,
                    exclude_genes=None,
                    method='average',
                    metric='euclidean',
                    distance_mode='centroid',
                    vars_to_regress=[],
                    verbose=True):
    """
    Harmony Integration with proportional HVG selection by cell type,
    now reading an existing H5AD file that only contains raw counts (no meta).
    """

    # 0. Create output directories if not present
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print("Automatically generating output directory")

    # Append 'harmony' subdirectory
    output_dir = os.path.join(output_dir, 'harmony')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print("Automatically generating harmony subdirectory")

    # 1. Read the raw count data from an existing H5AD
    if verbose:
        print('=== Read input dataset ===')
    adata = sc.read_h5ad(h5ad_path)
    if verbose:
        print(f'Dimension of raw data (cells x genes): {adata.shape[0]} x {adata.shape[1]}')

    # 2. Filter out any genes that have zero expression across all cells
    #    If your AnnData already uses the shape (cells, genes), do:
    sc.pp.filter_genes(adata, min_cells=1)

    # Optionally, also filter out any cells that have zero genes expressed
    sc.pp.filter_cells(adata, min_genes=1)

    # Additional user-defined filtering by min_cells and min_features
    #   (Note: These can be redundant if you've already filtered above;
    #    but we keep them for completeness.)
    if min_cells > 0:
        sc.pp.filter_genes(adata, min_cells=min_cells)
    if min_features > 0:
        sc.pp.filter_cells(adata, min_genes=min_features)

    # 3. Calculate mitochondrial gene percentage & filter
    #    We'll assume your gene symbols for mitochondrial genes start with "MT-"
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    # Filter out cells with > pct_mito_cutoff% mitochondrial counts
    adata = adata[adata.obs['pct_counts_mt'] < pct_mito_cutoff].copy()

    # Exclude (remove) mitochondrial genes and any other user-specified genes
    mt_genes = adata.var_names[adata.var_names.str.startswith('MT-')]
    if exclude_genes is not None:
        genes_to_exclude = set(exclude_genes) | set(mt_genes)
    else:
        genes_to_exclude = set(mt_genes)
    adata = adata[:, ~adata.var_names.isin(genes_to_exclude)].copy()

    if verbose:
        print(f'Dimension of processed data (cells x genes): {adata.shape[0]} x {adata.shape[1]}')

    # 4. Attach cell metadata if provided
    if cell_meta_path is None:
        # If no cell metadata, try to parse sample IDs from cell names
        # e.g. if cell name is "SAMPLE1:ATCGG"
        adata.obs['sample'] = adata.obs_names.str.split(':').str[0]
    else:
        cell_meta = pd.read_csv(cell_meta_path)
        # Assuming 'barcode' is the column with cell IDs
        cell_meta.set_index('barcode', inplace=True)
        # Join with adata.obs
        adata.obs = adata.obs.join(cell_meta, how='left')

    # 5. Attach sample metadata
    sample_meta = pd.read_csv(sample_meta_path)
    # Assuming 'sample' is the column to merge with
    adata.obs = adata.obs.merge(sample_meta, on='sample', how='left')

    # ----------------------------------------------------------------------
    # We now split the data for two analyses:
    #    (a) Clustering with Harmony (adata_cluster)
    #    (b) Sample differences (adata_sample_diff)
    # ----------------------------------------------------------------------
    adata_cluster = adata.copy()      # with batch effect correction
    adata_sample_diff = adata.copy()  # without batch effect correction

    # ==================== (a) Clustering with Harmony ====================
    if verbose:
        print('=== Processing data for clustering (mediating batch effects) ===')

    sc.pp.normalize_total(adata_cluster, target_sum=1e4)
    # Normalize and log transform
    sc.pp.log1p(adata_cluster)
    adata_cluster.raw = adata_cluster.copy()

    # Find highly variable genes (HVGs)
    sc.pp.highly_variable_genes(
        adata_cluster,
        n_top_genes=num_features,
        flavor='seurat_v3',
        batch_key='sample'
    )

    adata_cluster = adata_cluster[:, adata_cluster.var['highly_variable']].copy()

    # Scale data
    sc.pp.scale(adata_cluster, max_value=10)
    # PCA
    sc.pp.regress_out(adata_cluster, ['batch'])
    sc.tl.pca(adata_cluster, n_comps=num_PCs, svd_solver='arpack')
    # Harmony batch correction
    if verbose:
        print('=== Running Harmony integration for clustering ===')
        print('Variables to be regressed out: ', ','.join(vars_to_regress))
        print(f'Clustering cluster_resolution: {cluster_resolution}')

    vars_to_regress_for_harmony = vars_to_regress.copy()
    if "sample" not in vars_to_regress_for_harmony:
        vars_to_regress_for_harmony.append("sample")

    sc.external.pp.harmony_integrate(adata_cluster, vars_to_regress_for_harmony)

    # Clustering
    # If "celltype" is not in the metadata, we perform Leiden clustering.
    if 'celltype' in adata_cluster.obs.columns:
        adata_cluster.obs['cell_type'] = adata_cluster.obs['celltype'].astype('category')
        if markers is not None:
            marker_dict = {i: markers[i - 1] for i in range(1, len(markers) + 1)}
            adata_cluster.obs['cell_type'] = adata_cluster.obs['cell_type'].map(marker_dict)
    else:
        sc.tl.leiden(
            adata_cluster,
            resolution=cluster_resolution,
            flavor='igraph',
            n_iterations=1,
            directed=False,
            key_added='cell_type'
        )
        # Convert cluster IDs to "1, 2, 3..."
        adata_cluster.obs['cell_type'] = (adata_cluster.obs['cell_type'].astype(int) + 1).astype(str)

    # Marker genes for dendrogram
    sc.tl.rank_genes_groups(adata_cluster, groupby='cell_type', method='wilcoxon', n_genes=100)
    rank_results = adata_cluster.uns['rank_genes_groups']
    groups = rank_results['names'].dtype.names
    all_marker_genes = []
    for group in groups:
        all_marker_genes.extend(rank_results['names'][group])
    all_marker_genes = list(set(all_marker_genes))

    # Neighbors and UMAP
    sc.pp.neighbors(adata_cluster, use_rep='X_pca_harmony', n_pcs=num_harmony)
    sc.tl.umap(adata_cluster, min_dist=0.5)

    # Save results
    adata_cluster.write(os.path.join(output_dir, 'adata_cell.h5ad'))

    # ============== (b) Sample Differences (No batch correction) ==========
    if verbose:
        print('=== Processing data for sample differences (without batch effect correction) ===')

    # Ensure 'cell_type' is carried over
    if 'cell_type' not in adata_cluster.obs.columns or adata_cluster.obs['cell_type'].nunique() == 0:
        adata_cluster.obs['cell_type'] = '1'
    adata_sample_diff.obs['cell_type'] = adata_cluster.obs['cell_type']

    sc.pp.normalize_total(adata_sample_diff, target_sum=1e4)
    sc.pp.log1p(adata_sample_diff)
    adata_sample_diff.raw = adata_sample_diff.copy()

    if verbose:
        print('=== Preprocessing ===')
    # Find HVGs on raw counts
    find_hvgs(
        adata=adata_sample_diff,
        sample_column='sample',
        num_features=num_features,
        batch_key='cell_type',
        check_values=True,
        inplace=True
    )
    adata_sample_diff = adata_sample_diff[:, adata_sample_diff.var['highly_variable']].copy()

    if verbose:
        print('=== HVG ===')
    sc.pp.scale(adata_sample_diff, max_value=10)

    # PCA + Harmony (optional for sample difference dimension reduction)
    sc.pp.regress_out(adata_sample_diff, vars_to_regress_for_harmony)
    sc.tl.pca(adata_sample_diff, n_comps=num_PCs, svd_solver='arpack', zero_center=True)

    if verbose:
        print('=== Begin Harmony ===')
    sc.external.pp.harmony_integrate(adata_sample_diff, vars_to_regress_for_harmony)

    sc.pp.neighbors(adata_sample_diff, use_rep='X_pca_harmony', n_pcs=num_harmony, n_neighbors=15, metric='cosine')
    sc.tl.umap(adata_sample_diff, min_dist=0.3, spread=1.0)

    # Save final integrated data
    adata_sample_diff.write(os.path.join(output_dir, 'adata_sample.h5ad'))

    # === Visualizing sample relationship in 2D (PCA on average HVG expression) ===
    if verbose:
        print('=== Computing average HVG expression per sample ===')

    # HVG genes
    hvg_genes = adata_sample_diff.var_names[adata_sample_diff.var['highly_variable']]
    hvg_data = adata_sample_diff[:, hvg_genes].to_df()
    hvg_data['sample'] = adata_sample_diff.obs['sample'].values
    sample_means = hvg_data.groupby('sample').mean()

    if verbose:
        print(f'Computed average expression for {sample_means.shape[0]} samples and {sample_means.shape[1]} HVGs.')

    if verbose:
        print('=== End of processing ===')

    return adata_cluster, adata_sample_diff

import os
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# For interactive 3D plot
import plotly.express as px
import plotly.io as pio

def visualization_harmony(
    adata_cluster,
    adata_sample_diff,
    output_dir,
    grouping_columns=['sev.level'],
    age_bin_size=None,
    verbose=True
):
    # -----------------------------
    # 1. Ensure output directory
    # -----------------------------
    output_dir = os.path.join(output_dir, 'harmony')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # -----------------------------
    # 2. Build Group Assignments
    # -----------------------------
    def find_sample_grouping(adata, samples, grouping_columns, age_bin_size=None):
        group_dict = {}
        for sample in samples:
            row = adata.obs.loc[adata.obs['sample'] == sample].iloc[0]
            group_key = []
            for col in grouping_columns:
                group_key.append(str(row[col]))
            if age_bin_size and 'age' in adata.obs.columns:
                age_val = row['age']
                age_bin = int(age_val // age_bin_size * age_bin_size)
                group_key.append(f"AgeBin_{age_bin}")
            group_name = "_".join(group_key)
            group_dict[sample] = group_name
        return group_dict

    # Group assignments for both AnnData objects
    cluster_samples = adata_cluster.obs['sample'].unique().tolist()
    cluster_groups = find_sample_grouping(
        adata_cluster, cluster_samples, grouping_columns, age_bin_size
    )
    adata_cluster.obs['plot_group'] = adata_cluster.obs['sample'].map(cluster_groups)

    diff_samples = adata_sample_diff.obs['sample'].unique().tolist()
    diff_groups = find_sample_grouping(
        adata_sample_diff, diff_samples, grouping_columns, age_bin_size
    )
    adata_sample_diff.obs['plot_group'] = adata_sample_diff.obs['sample'].map(diff_groups)

    if verbose:
        print("[visualization_harmony] 'plot_group' assigned via find_sample_grouping.")

    # --------------------------------
    # 3. Dendrogram (by cell_type)
    # --------------------------------
    sc.pl.dendrogram(adata_cluster, groupby='cell_type', show=False)
    plt.savefig(os.path.join(output_dir, 'phylo_tree.pdf'))
    plt.close()

    # --------------------------------
    # 4. UMAP colored by plot_group (Clusters)
    # --------------------------------
    plt.figure(figsize=(12, 10))
    sc.pl.umap(
        adata_cluster,
        color='plot_group',
        legend_loc=None,
        frameon=False,
        size=3,
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_umap_by_plot_group.pdf'), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 10))
    sc.pl.umap(
        adata_cluster,
        color='cell_type',
        legend_loc=None,
        frameon=False,
        size=3,
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_umap_cell_type.pdf'), bbox_inches='tight')
    plt.close()

    # --------------------------------
    # 5. UMAP colored by plot_group (Sample Differences)
    # --------------------------------
    plt.figure(figsize=(12, 10))
    sc.pl.umap(
        adata_sample_diff,
        color='plot_group',
        legend_loc=None,
        frameon=False,
        size=3,
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_umap_by_plot_group.pdf'), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 10))
    sc.pl.umap(
        adata_sample_diff,
        color='cell_type',
        legend_loc=None,
        frameon=False,
        size=3,
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_umap_by_cell_type.pdf'), bbox_inches='tight')
    plt.close()

    # --------------------------------
    # 6. PCA of Average HVG Expression
    # --------------------------------
    if verbose:
        print("[visualization_harmony] Computing sample-level PCA from average HVG expression.")

    df = pd.DataFrame(
        adata_sample_diff.X,
        index=adata_sample_diff.obs_names,
        columns=adata_sample_diff.var_names
    )
    df['sample'] = adata_sample_diff.obs['sample']
    sample_means = df.groupby('sample').mean()
    sample_to_group = adata_sample_diff.obs[['sample', 'plot_group']].drop_duplicates().set_index('sample')

    pca_2d = PCA(n_components=2)
    pca_coords_2d = pca_2d.fit_transform(sample_means)
    pca_2d_df = pd.DataFrame(pca_coords_2d, index=sample_means.index, columns=['PC1', 'PC2'])
    pca_2d_df = pca_2d_df.join(sample_to_group, how='left')

    plt.figure(figsize=(8, 6))
    unique_groups = pca_2d_df['plot_group'].unique()
    colors = plt.cm.get_cmap('tab10', len(unique_groups))
    for i, grp in enumerate(unique_groups):
        mask = (pca_2d_df['plot_group'] == grp)
        plt.scatter(pca_2d_df.loc[mask, 'PC1'], pca_2d_df.loc[mask, 'PC2'], color=colors(i), s=80, alpha=0.8)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('2D PCA of Avg HVG Expression ')
    plt.grid(True)
    plt.tight_layout()
    plt.legend().remove()
    plt.savefig(os.path.join(output_dir, 'sample_relationship_pca_2D_no_legend.pdf'))
    plt.close()

    # 3D Interactive PCA
    pca_3d = PCA(n_components=3)
    pca_coords_3d = pca_3d.fit_transform(sample_means)
    pca_3d_df = pd.DataFrame(pca_coords_3d, index=sample_means.index, columns=['PC1', 'PC2', 'PC3'])
    pca_3d_df = pca_3d_df.join(sample_to_group, how='left')

    fig_3d = px.scatter_3d(
        pca_3d_df,
        x='PC1', y='PC2', z='PC3',
        color='plot_group',
        hover_data={'plot_group': False}
    )
    fig_3d.update_layout(showlegend=False)
    fig_3d.update_traces(marker=dict(size=5), hovertemplate='<extra></extra>')
    output_html_path = os.path.join(output_dir, 'sample_relationship_pca_3D.html')
    pio.write_html(fig_3d, file=output_html_path, auto_open=False)

    # --------------------------------
    # 7. 3D Visualization of Cell-level Harmony PCA
    # --------------------------------
    if verbose:
        print("[visualization_harmony] Generating 3D cell-level Harmony PCA visualization.")

    # If using obsm (standard):
    harmony_coords = adata_sample_diff.obsm['X_pca_harmony'][:, :3]
    pca_cell_df = pd.DataFrame(
        harmony_coords,
        columns=['PC1', 'PC2', 'PC3'],
        index=adata_sample_diff.obs.index
    )
    pca_cell_df['plot_group'] = adata_sample_diff.obs['plot_group']

    # Create interactive plot
    fig_cell_3d = px.scatter_3d(
        pca_cell_df,
        x='PC1',
        y='PC2',
        z='PC3',
        color='plot_group',
        hover_data={'plot_group': False}
    )
    fig_cell_3d.update_layout(showlegend=False)
    fig_cell_3d.update_traces(marker=dict(size=2), hovertemplate='<extra></extra>')

    # Save plot
    cell_3d_path = os.path.join(output_dir, 'cell_pca_sample.html')
    pio.write_html(fig_cell_3d, file=cell_3d_path, auto_open=False)

    if verbose:
        print(f"[visualization_harmony] 3D cell-level PCA saved to {cell_3d_path}")

    # --------------------------------
    # 8. 3D Visualization of Cell-level Harmony PCA from cluster
    # --------------------------------
    if verbose:
        print("[visualization_harmony] Generating 3D cell-level Harmony PCA visualization.")

    # If using obsm (standard):
    harmony_coords = adata_cluster.obsm['X_pca_harmony'][:, :3]
    pca_cell_df = pd.DataFrame(
        harmony_coords,
        columns=['PC1', 'PC2', 'PC3'],
        index=adata_cluster.obs.index
    )
    pca_cell_df['plot_group'] = adata_cluster.obs['plot_group']

    # Create interactive plot
    fig_cell_3d = px.scatter_3d(
        pca_cell_df,
        x='PC1',
        y='PC2',
        z='PC3',
        color='plot_group',
        hover_data={'plot_group': False}
    )
    fig_cell_3d.update_layout(showlegend=False)
    fig_cell_3d.update_traces(marker=dict(size=2), hovertemplate='<extra></extra>')

    # Save plot
    cell_3d_path = os.path.join(output_dir, 'cell_pca_cluster.html')
    pio.write_html(fig_cell_3d, file=cell_3d_path, auto_open=False)

    if verbose:
        print(f"[visualization_harmony] 3D cell-level PCA saved to {cell_3d_path}")

    # --------------------------------
    # Done
    # --------------------------------
    if verbose:
        print("[visualization_harmony] All visualizations saved.")