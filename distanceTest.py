import os
import pandas as pd
import numpy as np
from Grouping import find_sample_grouping

def distanceCheck(df_path, 
                  row, 
                  method, 
                  summary_csv_path, 
                  adata=None,
                  grouping_columns=['sample','batch'],
                  age_bin_size=10):
    """
    Calculate in-group vs. between-group distances based on a grouping of samples,
    then update a summary CSV file with the resulting score.

    Parameters
    ----------
    df_path : str
        Path to the CSV file containing the distance matrix.
    row : str
        Row name in the summary CSV to update (e.g., 'cell_proportion' or 'gene_expression').
    method : str
        Column name in the summary CSV to update.
    summary_csv_path : str
        Path to the summary CSV file.
    adata : anndata.AnnData or None
        An AnnData object where per-sample metadata is stored in `adata.obs`.
        If None or grouping_columns is None, grouping will be by the first two letters of each sample.
    grouping_columns : list of str or None
        Column names in `adata.obs` to use for grouping the samples.
    age_bin_size : int or None
        If 'age' is in grouping_columns, this controls the bin width for age groups.
    """

    # --------------------------------------------------------------------------
    # 1) Read the distance matrix
    # --------------------------------------------------------------------------
    df = pd.read_csv(df_path, index_col=0)

    # Get the directory of the CSV file (for writing outputs)
    directory = os.path.dirname(os.path.abspath(df_path))

    # The sample names from the distance matrix (index)
    samples = df.index.tolist()

    # --------------------------------------------------------------------------
    # 2) Find sample grouping
    #    (Using the new function that handles multi-column grouping, age bins, etc.)
    # --------------------------------------------------------------------------
    # Import here or at the top (if the function is in the same file, just call it)

    groups = find_sample_grouping(
        adata=adata,
        samples=samples,
        grouping_columns=grouping_columns,
        age_bin_size=age_bin_size
    )
    # 'groups' is now a dict like {sample_name: group_label}

    # --------------------------------------------------------------------------
    # 3) Compute in-group vs. between-group distances
    # --------------------------------------------------------------------------
    in_group_distances = []
    between_group_distances = []

    for i, sample_i in enumerate(samples):
        for j, sample_j in enumerate(samples):
            if i >= j:
                continue  # Avoid redundant pairs & self-distances
            distance = df.iloc[i, j]
            if groups[sample_i] == groups[sample_j]:
                in_group_distances.append(distance)
            else:
                between_group_distances.append(distance)

    average_in_group_distance = np.mean(in_group_distances) if in_group_distances else np.nan
    average_between_group_distance = np.mean(between_group_distances) if between_group_distances else np.nan

    # --------------------------------------------------------------------------
    # 4) Calculate the final score
    # --------------------------------------------------------------------------
    if np.isnan(average_in_group_distance) or average_in_group_distance == 0:
        score = np.nan
    else:
        score = average_between_group_distance / average_in_group_distance

    # Prepare a string to save / print
    result_str = (
        f"Average in-group distance: {average_in_group_distance}\n"
        f"Average between-group distance: {average_between_group_distance}\n"
        f"Score: {score}"
    )

    # --------------------------------------------------------------------------
    # 5) Write the results to a text file
    # --------------------------------------------------------------------------
    output_file = os.path.join(directory, 'distance_results.txt')
    with open(output_file, 'w') as f:
        f.write(result_str)

    print(f"Results saved to {output_file}")

    # --------------------------------------------------------------------------
    # 6) Update the summary CSV
    # --------------------------------------------------------------------------
    if os.path.isfile(summary_csv_path):
        summary_df = pd.read_csv(summary_csv_path, index_col=0)
    else:
        # Create a new DataFrame with some default rows
        rows = ['cell_proportion', 'gene_expression', "gene_pseudobulk"]
        summary_df = pd.DataFrame(index=rows)

    if method not in summary_df.columns:
        summary_df[method] = np.nan

    summary_df.loc[row, method] = score
    summary_df.to_csv(summary_csv_path)
    print(f"Summary updated in {summary_csv_path}")