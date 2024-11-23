def compute_ground_distance_matrix_cell_types(
    avg_expr: Dict[str, np.ndarray],
    cell_types: list,
    hvg: list
) -> np.ndarray:
    """
    Compute the ground distance matrix between different cell types based on Earth Mover's Distance (EMD).

    Each highly variable gene (HVG) is treated as a bin, and their average expressions across cell types are considered as the frequencies.
    The ground distance between HVGs is defined based on the Euclidean distance between their average expression profiles across cell types.

    Parameters:
    ----------
    avg_expr : dict
        Dictionary where keys are cell types and values are average expression arrays for HVGs.
    cell_types : list
        List of unique cell types.
    hvg : list
        List of highly variable genes.

    Returns:
    -------
    ground_distance_matrix : np.ndarray
        A symmetric matrix of ground distances between cell types.
    """
    # Create a DataFrame from the avg_expr dictionary
    avg_expr_df = pd.DataFrame(avg_expr, index=hvg).T  # Shape: (cell_types, hvg)

    # Normalize the distributions so that each cell type sums to 1
    avg_expr_normalized = avg_expr_df.div(avg_expr_df.sum(axis=1), axis=0).fillna(0).values  # Shape: (cell_types, hvg)

    # Define ground distance between HVGs based on Euclidean distance between their expression profiles across cell types
    hvg_profiles = avg_expr_df.values  # Shape: (cell_types, hvg)
    # Transpose to get (hvg, cell_types) for distance computation
    hvg_profiles_transposed = hvg_profiles.T  # Shape: (hvg, cell_types)
    ground_distance_hvg = squareform(pdist(hvg_profiles_transposed, metric='euclidean')).astype(np.float64)

    # Ensure ground_distance_hvg is C-contiguous
    ground_distance_hvg = ground_distance_hvg.copy(order='C')

    # Compute the ground distance matrix between cell types using EMD
    num_cell_types = len(cell_types)
    ground_distance_matrix = np.zeros((num_cell_types, num_cell_types), dtype=np.float64)

    for i in range(num_cell_types):
        for j in range(i + 1, num_cell_types):
            # Ensure the histograms are C-contiguous
            hist_i = avg_expr_normalized[i].astype(np.float64, order='C').copy(order='C')
            hist_j = avg_expr_normalized[j].astype(np.float64, order='C').copy(order='C')
            distance = emd(hist_i, hist_j, ground_distance_hvg)
            ground_distance_matrix[i, j] = distance
            ground_distance_matrix[j, i] = distance  # Symmetric matrix

    # Ensure the ground_distance_matrix is C-contiguous
    return ground_distance_matrix.copy(order='C')

Harmony:
# for count_path, sample_meta_path in zip(counts_path, sample_meta_paths):
    #     # Process count data
    #     temp_count = pd.read_csv(count_path, index_col=0)
    #     temp_count = temp_count.sort_index()
        
    #     # Extract sample name from the count file path
    #     sample_name = extract_sample_name_from_path(count_path)
        
    #     # Prefix cell barcodes with sample name
    #     temp_count.columns = [f"{sample_name}:{cell_barcode}" for cell_barcode in temp_count.columns]
        
    #     # Initialize or concatenate counts
    #     if count is None:
    #         count = temp_count
    #     else:
    #         count = count.sort_index()
    #         if not temp_count.index.equals(count.index):
    #             raise ValueError(f"Gene names do not match between files: {count_path}")
    #         else:
    #             count = pd.concat([count, temp_count], axis=1)
        
    #     # Process sample metadata
    #     temp_meta = pd.read_csv(sample_meta_path)
    #     temp_meta['sample'] = sample_name
    #     sample_meta_list.append(temp_meta)

    # # Combine all sample metadata into a single DataFrame
    # sample_meta = pd.concat(sample_meta_list, ignore_index=True)

    # # Optionally save the combined count matrix and sample metadata
    # count.to_csv('combined_counts.csv')
    # sample_meta.to_csv('combined_sample_meta.csv', index=False)

    # # Find marker genes for each cluster
    # if verbose:
    #     print('=== Find gene markers for each cell cluster ===')
    
    # if issparse(adata.X):
    #     adata.X.data += 1e-6
    # else:
    #     adata.X += 1e-6

    # if issparse(adata.X):
    #     has_nan = np.isnan(adata.X.data).any()
    #     has_zero = np.any(adata.X.data == 0)
    # else:
    #     has_nan = np.isnan(adata.X).any()
    #     has_zero = np.any(adata.X == 0)

    # print(f"Contains NaNs: {has_nan}, Contains Zeros: {has_zero}")

    # sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
    # markers = sc.get.rank_genes_groups_df(adata, group=None)
    # markers.to_csv(os.path.join(output_dir, 'markers.csv'), index=False)
    
    # # Get top 10 markers per cluster
    # top10 = markers.groupby('group', observed=True).head(10)
    # top10.to_csv(os.path.join(output_dir, 'markers_top10.csv'), index=False)



sampleSimilarityExpression:
# # 2. Compute ground distance matrix between cell types based on average expression profiles
    # # Compute global average expression profiles for each cell type across all samples
    # global_avg_expression = {}
    # for cell_type in cell_types:
    #     cell_type_data = hvg[hvg.obs[cell_type_column] == cell_type]
    #     if cell_type_data.shape[0] > 0:
    #         if issparse(cell_type_data.X):
    #             avg_expr = cell_type_data.X.mean(axis=0).A1.astype(np.float64)
    #         else:
    #             avg_expr = cell_type_data.X.mean(axis=0).astype(np.float64)
    #         global_avg_expression[cell_type] = avg_expr
    #     else:
    #         global_avg_expression[cell_type] = np.zeros(hvg.shape[1], dtype=np.float64)

    # Create a list of cell types to maintain order
   
    # # Initialize the ground distance matrix
    # ground_distance = np.zeros((num_cell_types, num_cell_types), dtype=np.float64)

    # # Populate the ground distance matrix with Euclidean distances between cell type centroids
    # for i in range(num_cell_types):
    #     for j in range(num_cell_types):
    #         expr_i = global_avg_expression[cell_type_list[i]]
    #         expr_j = global_avg_expression[cell_type_list[j]]
    #         distance = np.linalg.norm(expr_i - expr_j)
    #         ground_distance[i, j] = distance

    # # 3. Normalize the ground distance matrix (optional but recommended)
    # # This ensures that the distances are scaled appropriately for EMD
    # max_distance = ground_distance.max()
    # if max_distance > 0:
    #     ground_distance /= max_distance

    # # Ensure ground_distance is float64
    # ground_distance = ground_distance.astype(np.float64)

    # 2. Compute ground distance matrix between cell types
    # We'll use the centroids of cell types in PCA space


    def calculate_sample_distances_average_expression(
    adata: AnnData,
    output_dir: str,
    method: str,
    summary_csv_path: str,
    cell_type_column: str = 'leiden',
    sample_column: str = 'sample'
) -> pd.DataFrame:
    """
    Calculate distance matrix based on average gene expression per cell type for each sample.

    Parameters:
    - adata: AnnData object containing single-cell data.
    - output_dir: Directory to save the distance matrix and related plots.
    - cell_type_column: Column in adata.obs indicating cell types.
    - sample_column: Column in adata.obs indicating sample identifiers.

    Returns:
    - distance_df: DataFrame containing the pairwise distances between samples.
    """
    output_dir = os.path.join(output_dir, 'avarage_expression')
    os.makedirs(output_dir, exist_ok=True)

    avg_expression = adata.to_df().groupby([adata.obs[sample_column], adata.obs[cell_type_column]]).mean()
    # Reshape to have samples as rows and (cell_type, gene) as columns
    avg_expression = avg_expression.unstack(level=1)
    # Flatten the MultiIndex columns
    avg_expression.columns = ['{}_{}'.format(cell_type, gene) for cell_type, gene in avg_expression.columns]
    avg_expression.fillna(0, inplace=True)  # Handle any missing values

    # Save average expression to a CSV file
    avg_expression.to_csv(os.path.join(output_dir, 'average_expression_per_cell_type.csv'))
    print("Average expression per cell type saved to 'average_expression_per_cell_type.csv'.")

    # Calculate distance matrix
    distance_matrix = pdist(avg_expression.values, metric = method)
    distance_df = pd.DataFrame(
        squareform(distance_matrix),
        index=avg_expression.index,
        columns=avg_expression.index
    )
    distance_df = np.log1p(np.maximum(distance_df, 0))
    distance_df = distance_df / distance_df.max().max()

    # Save the distance matrix
    distance_matrix_path = os.path.join(output_dir, 'distance_matrix_average_expression.csv')
    distance_df.to_csv(distance_matrix_path)
    distanceCheck(distance_matrix_path, 'average_expression', method, summary_csv_path)
    print(f"Sample distance avarage expresission matrix saved to {distance_matrix_path}")

    # generate a heatmap for sample distance
    heatmap_path = os.path.join(output_dir, 'sample_distance_average_expression_heatmap.pdf')
    visualizeDistanceMatrix(distance_df, heatmap_path)
    visualizeGroupRelationship(distance_df, outputDir=output_dir, heatmap_path=os.path.join(output_dir, 'sample_avarage_expression_relationship.pdf'))
    print("Distance matrix based on average expression per cell type saved to 'distance_matrix_average_expression.csv'.")
    return distance_df
    #possibly used for vectordistance