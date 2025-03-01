import os
import numpy as np
import pandas as pd
import scanpy as sc
import harmonypy as hm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from Visualization import visualization_harmony
from combat.pycombat import pycombat

# Local imports from your project
from HierarchicalConstruction import cell_type_dendrogram
from HVG import find_hvgs
from Grouping import find_sample_grouping

def treecor_harmony(h5ad_path,
                    sample_meta_path,
                    output_dir,
                    cell_meta_path=None,
                    markers=None,
                    cluster_resolution=0.8,
                    num_PCs=20,
                    num_harmony=30,
                    num_features=2000,
                    min_cells=500,
                    min_features=500,
                    pct_mito_cutoff=20,
                    exclude_genes=None,
                    doublet = True,
                    combat = True,
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

    vars_to_regress_for_harmony = vars_to_regress.copy()
    if "sample" not in vars_to_regress_for_harmony:
        vars_to_regress_for_harmony.append("sample")

    if cell_meta_path is None:
        adata.obs['sample'] = adata.obs_names.str.split(':').str[0]
    else:
        cell_meta = pd.read_csv(cell_meta_path)
        cell_meta.set_index('barcode', inplace=True)
        adata.obs = adata.obs.join(cell_meta, how='left')

    sample_meta = pd.read_csv(sample_meta_path)
    adata.obs = adata.obs.merge(sample_meta, on='sample', how='left')

    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_cells(adata, min_genes=min_features)

    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs['pct_counts_mt'] < pct_mito_cutoff].copy()

    mt_genes = adata.var_names[adata.var_names.str.startswith('MT-')]
    if exclude_genes is not None:
        genes_to_exclude = set(exclude_genes) | set(mt_genes)
    else:
        genes_to_exclude = set(mt_genes)
    adata = adata[:, ~adata.var_names.isin(genes_to_exclude)].copy()

    cell_counts_per_patient = adata.obs.groupby('sample').size()
    patients_to_keep = cell_counts_per_patient[cell_counts_per_patient >= min_cells].index
    adata = adata[adata.obs['sample'].isin(patients_to_keep)].copy()

    min_cells_for_gene = int(0.01 * adata.n_obs)
    sc.pp.filter_genes(adata, min_cells=min_cells_for_gene)

    if verbose:
        print(f'Dimension of processed data (cells x genes): {adata.shape[0]} x {adata.shape[1]}')

    # Doublet detection is now optional
    if doublet:
        sc.pp.scrublet(adata, batch_key="sample")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if verbose:
        print("Preprocessing complete!")
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

    adata_cluster.raw = adata_cluster.copy()

    # Find highly variable genes (HVGs)
    sc.pp.highly_variable_genes(
        adata_cluster,
        n_top_genes=num_features,
        flavor='seurat_v3',
        batch_key='sample'
    )

    adata_cluster = adata_cluster[:, adata_cluster.var['highly_variable']].copy()  
    sc.tl.pca(adata_cluster, n_comps=num_PCs, svd_solver='arpack')
    # Harmony batch correction
    if verbose:
        print('=== Running Harmony integration for clustering ===')
        print('Variables to be regressed out: ', ','.join(vars_to_regress))
        print(f'Clustering cluster_resolution: {cluster_resolution}')

    ho = hm.run_harmony(adata_cluster.obsm['X_pca'], adata_cluster.obs, vars_to_regress_for_harmony, max_iter_harmony=num_harmony, max_iter_kmeans=50)
    adata_cluster.obsm['X_pca_harmony'] = ho.Z_corr.T

    if verbose:
        print("End of harmony for adata_cluster.")

    if 'celltype' in adata_cluster.obs.columns:
        adata_cluster.obs['cell_type'] = adata_cluster.obs['celltype'].astype(str)
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
        adata_cluster.obs['cell_type'] = (adata_cluster.obs['leiden'].astype(int) + 1).astype('category')
    if verbose:
        print("Finish find cell type")

    sc.tl.rank_genes_groups(adata_cluster, groupby='cell_type', method='logreg', n_genes=100)
    rank_results = adata_cluster.uns['rank_genes_groups']
    groups = rank_results['names'].dtype.names
    all_marker_genes = []
    for group in groups:
        all_marker_genes.extend(rank_results['names'][group])
    all_marker_genes = list(set(all_marker_genes))

    adata_cluster = cell_type_dendrogram(
        adata=adata_cluster,
        resolution=cluster_resolution,
        groupby='cell_type',
        method='average',
        metric='euclidean',
        distance_mode='centroid',
        marker_genes=all_marker_genes,
        verbose=True
    )

    # Neighbors and UMAP
    sc.pp.neighbors(adata_cluster, use_rep='X_pca_harmony', n_pcs=num_PCs)
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

    if verbose:
        print('=== Preprocessing ===')
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

    sc.tl.pca(adata_sample_diff, n_comps=num_PCs, svd_solver='arpack', zero_center=True)

    if verbose:
        print('=== Begin Harmony ===')
    ho = hm.run_harmony(adata_sample_diff.obsm['X_pca'], adata_sample_diff.obs, ['batch'], max_iter_harmony=num_harmony, max_iter_kmeans=50)
    adata_sample_diff.obsm['X_pca_harmony'] = ho.Z_corr.T
    
    sc.pp.neighbors(adata_sample_diff, use_rep='X_pca_harmony', n_pcs=num_PCs, n_neighbors=15, metric='cosine')
    sc.tl.umap(adata_sample_diff, min_dist=0.3, spread=1.0)

    # At this point, we have create our comparison group as well as find the correct grouping of cell. 
    # Now all our operation should be based on a new round of operation of our new sample_diff adata.

    compute_pseudobulk_dataframes(adata_sample_diff, 'batch', 'sample', 'cell_type', output_dir)
    sc.write(os.path.join(output_dir, 'adata_sample.h5ad'), adata_sample_diff)
    return adata_cluster, adata_sample_diff

def compute_pseudobulk_dataframes(
    adata: sc.AnnData,
    batch_col: str = 'batch',
    sample_col: str = 'sample',
    celltype_col: str = 'cell_type',
    output_dir: str = './'
):
    """
    Creates two DataFrames:

    1) `cell_expression_df` with rows = cell types, columns = samples.
       Each cell is a vector of average gene expressions for that cell type in that sample.
    2) `cell_proportion_df` with rows = cell types, columns = samples.
       Each cell is a single float for proportion of that cell type in that sample.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (cells x genes).
        Must have `sample_col` and `celltype_col` in .obs.
    sample_col : str
        Column in `adata.obs` indicating sample ID.
    celltype_col : str
        Column in `adata.obs` indicating cell type.
    output_dir : str
        Directory where the output might be saved. 
        (Optional in this snippet; you can omit if not saving.)

    Returns
    -------
    cell_expression_df : DataFrame
        Rows = cell types, columns = samples, each element is a 1D numpy array of shape (n_genes,).
    cell_proportion_df : DataFrame
        Rows = cell types, columns = samples, each element is a float (the proportion).
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract relevant columns
    samples = adata.obs[sample_col].unique()
    cell_types = adata.obs[celltype_col].unique()
    gene_names = adata.var_names

    # Convert sparse matrix to dense if needed
    X_data = adata.X
    if not isinstance(X_data, np.ndarray):
        X_data = X_data.toarray()

    # Create empty DataFrames
    # Each cell in cell_expression_df is initially set to None (or np.nan).
    # We'll store arrays in them, so we use dtype=object for the expression DF.
    cell_expression_df = pd.DataFrame(
        index=cell_types,
        columns=samples,
        dtype=object
    )
    cell_proportion_df = pd.DataFrame(
        index=cell_types,
        columns=samples,
        dtype=float
    )

    for sample in samples:
        # Mask: all cells from this sample
        sample_mask = (adata.obs[sample_col] == sample)
        total_cells = np.sum(sample_mask)

        for ctype in cell_types:
            # Further subset to this cell type
            ctype_mask = sample_mask & (adata.obs[celltype_col] == ctype)
            num_cells = np.sum(ctype_mask)

            if num_cells > 0:
                # Average expression across genes for the subset
                expr_values = X_data[ctype_mask, :].mean(axis=0)
                proportion = num_cells / total_cells
            else:
                # No cells of this (sample, cell_type) combination
                expr_values = np.zeros(len(gene_names))
                proportion = 0.0

            # Store results in the DataFrames
            cell_expression_df.loc[ctype, sample] = expr_values
            cell_proportion_df.loc[ctype, sample] = proportion
    
    print("Successfuly computed pseudobulk dataframes.")

    # Save DataFrames as CSV files without modifying their structure
    cell_expression_df.to_csv(os.path.join(output_dir, "cell_expression.csv"), index=True)
    cell_proportion_df.to_csv(os.path.join(output_dir, "cell_proportion.csv"), index=True)

    # Compute corrected expression and save it
    cell_expression_corrected_df = combat_correct_cell_expressions(adata, cell_expression_df)
    cell_expression_corrected_df.to_csv(os.path.join(output_dir, "cell_expression_corrected.csv"), index=True)
    return cell_expression_df, cell_proportion_df

def combat_correct_cell_expressions(
    adata: sc.AnnData,
    cell_expression_df: pd.DataFrame,
    batch_col: str = 'batch',
    sample_col: str = 'sample'
) -> pd.DataFrame:
    """
    Applies ComBat batch correction to each cell type across samples for a
    DataFrame where:
      - Rows = cell types
      - Columns = sample IDs
      - Each cell is a 1D array of shape (n_genes,).

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object that includes `batch_col` and `sample_col` in `adata.obs`
        for each cell. We'll look up the batch for each sample from here.
    cell_expression_df : DataFrame
        Rows = cell types, columns = samples.
        Each cell in the table is a 1D numpy array of length n_genes.
    batch_col : str
        The name of the column in `adata.obs` that indicates batch.
    sample_col : str
        The name of the column in `adata.obs` that indicates sample ID.

    Returns
    -------
    corrected_df : DataFrame
        Same shape as `cell_expression_df` (rows = cell types, columns = samples),
        but each cell's array is now ComBat-corrected.
    """

    sample_batch_map = (adata.obs[[sample_col, batch_col]]
                        .drop_duplicates()
                        .set_index(sample_col)[batch_col]
                        .to_dict())

    # We'll also gather the list of genes from the first row's first cell
    # (assuming all arrays have the same length, i.e., same set/order of genes).
    # If your code has a known gene ordering, you can pass it directly.
    # Here we assume at least 1 row x 1 column is present.
    example_row = cell_expression_df.iloc[0].dropna()
    example_array = next((arr for arr in example_row if arr is not None and len(arr) > 0), None)
    if example_array is None:
        raise ValueError("Unable to find a non-empty array in cell_expression_df.")
    n_genes = len(example_array)

    # Make a copy to store corrected values
    corrected_df = cell_expression_df.copy(deep=True)

    # ---------------------------
    # 2. Loop over cell types
    # ---------------------------
    for ctype in corrected_df.index:  # each row
        # Extract the row as a Series: index=sample IDs, values=arrays of shape (n_genes,)
        row_data = corrected_df.loc[ctype]

        # Build an (n_samples x n_genes) matrix by stacking the arrays in row order
        # Also collect the batch labels in the same order
        arrays_for_this_ctype = []
        batch_labels = []
        samples_in_row = row_data.index

        for sample_id in samples_in_row:
            expr_array = row_data[sample_id]
            # It might be None or an empty array if data is missing
            if expr_array is None or len(expr_array) == 0:
                expr_array = np.zeros(n_genes, dtype=float)
                print(f"\nWarning: Missing data for cell type '{ctype}' in sample '{sample_id}'.\n")
            
            arrays_for_this_ctype.append(expr_array)

            # Lookup the batch for this sample. If not found, use a placeholder or skip.
            batch = sample_batch_map.get(sample_id, "missing_batch")
            batch_labels.append(batch)

        # Convert to shape (n_samples, n_genes)
        expr_matrix = np.vstack(arrays_for_this_ctype)  # shape: (n_samples, n_genes)

        # If there's only one sample or only one unique batch, ComBat doesn't do anything meaningful.
        # We can skip or at least check for that here:
        unique_batches = pd.unique(batch_labels)
        if len(batch_labels) < 2 or len(unique_batches) < 2:
            # Skip ComBat, leave row as is
            print(f"Skipping ComBat correction for cell type '{ctype}' due to insufficient samples or batches.")
            continue

        # ---------------------------
        # 3. Apply ComBat
        # ---------------------------
        # pycombat expects shape (n_genes x n_samples), so transpose
        expr_matrix_t = expr_matrix.T  # shape: (n_genes, n_samples)
        expr_df_t = pd.DataFrame(
        expr_matrix_t,
        columns=samples_in_row,      # samples_in_row is your list of sample IDs
        index=range(n_genes)         # or gene names if you have them
        )

        # Convert the batch labels into a Series, indexed by sample:
        batch_series = pd.Series(batch_labels, index=samples_in_row, name='batch')

        # Now pass DataFrame + Series to pycombat
        corrected_df_t = pycombat(expr_df_t, batch=batch_series)

        # Convert the DataFrame back to a NumPy array if needed
        corrected_matrix_t = corrected_df_t.values

        # ---------------------------
        # 4. Write corrected arrays back to corrected_df
        # ---------------------------
        for i, sample_id in enumerate(samples_in_row):
            corrected_df.loc[ctype, sample_id] = corrected_matrix_t[i]
        
        print(f"ComBat correction complete for cell type '{ctype}'.")

    return corrected_df