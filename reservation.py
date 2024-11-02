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