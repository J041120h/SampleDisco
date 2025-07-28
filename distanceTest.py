import os
import pandas as pd
import numpy as np
from anndata import AnnData
from typing import List, Optional, Dict
from Grouping import find_sample_grouping

def distanceCheck(
    distance_df: pd.DataFrame,
    row: str,
    method: str,
    output_dir: str,
    adata: AnnData = None,
    grouping_columns: List[str] = ['sev.level'],
    age_bin_size: int = 10,
    summary_csv_path: Optional[str] = None
) -> float:
    """
    Calculate in-group vs. between-group distances based on a grouping of samples,
    using data directly from AnnData object and distance DataFrame.

    Parameters
    ----------
    distance_df : pd.DataFrame
        Distance matrix with samples as both index and columns
    row : str
        Row name in the summary CSV to update (e.g., 'expression_DR' or 'proportion_DR').
    method : str
        Distance method used (e.g., 'cosine', 'euclidean').
    output_dir : str
        Directory to save results
    adata : anndata.AnnData or None
        An AnnData object where per-sample metadata is stored in `adata.obs`.
        Sample names should match the distance_df index.
    grouping_columns : list of str
        Column names in `adata.obs` to use for grouping the samples.
    age_bin_size : int
        If 'age' is in grouping_columns, this controls the bin width for age groups.
    summary_csv_path : str or None
        Path to the summary CSV file. If None, no summary file is updated.

    Returns
    -------
    float
        The distance score (between-group distance / in-group distance)
    """

    # Get the sample names from the distance matrix
    samples = distance_df.index.tolist()
    
    print(f"Debug: Distance matrix samples: {samples}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # 1) Find sample grouping using available metadata
    # --------------------------------------------------------------------------
    if adata is not None and hasattr(adata, 'obs') and not adata.obs.empty:
        # Check which samples from distance matrix exist in adata.obs
        available_samples = []
        sample_metadata = {}
        
        for sample in samples:
            if sample in adata.obs.index:
                available_samples.append(sample)
                sample_metadata[sample] = adata.obs.loc[sample].to_dict()
            else:
                print(f"Warning: Sample '{sample}' not found in adata.obs")
        
        if not available_samples:
            print("Warning: No samples from distance matrix found in adata.obs, using fallback grouping")
            # Fallback to simple grouping by first few characters
            groups = {sample: sample[:2] for sample in samples}
        else:
            # Create a temporary AnnData-like structure for grouping function
            temp_obs = pd.DataFrame.from_dict(sample_metadata, orient='index')
            
            # Create a minimal AnnData object for the grouping function
            temp_adata = type('TempAnnData', (), {})()
            temp_adata.obs = temp_obs
            
            try:
                groups = find_sample_grouping(
                    adata=temp_adata,
                    samples=available_samples,
                    grouping_columns=grouping_columns,
                    age_bin_size=age_bin_size
                )
                
                # Add any missing samples with a default group
                for sample in samples:
                    if sample not in groups:
                        groups[sample] = 'Unknown'
                        
            except Exception as e:
                print(f"Warning: Grouping function failed: {e}")
                print("Using fallback grouping by sample prefix")
                groups = {sample: sample[:2] for sample in samples}
    else:
        print("Warning: No adata provided or adata.obs is empty, using fallback grouping")
        # Fallback grouping by first two characters
        groups = {sample: sample[:2] for sample in samples}

    print(f"Debug: Sample groups: {groups}")
    
    # --------------------------------------------------------------------------
    # 2) Compute in-group vs. between-group distances
    # --------------------------------------------------------------------------
    in_group_distances = []
    between_group_distances = []

    for i, sample_i in enumerate(samples):
        for j, sample_j in enumerate(samples):
            if i >= j:
                continue  # Avoid redundant pairs & self-distances
            
            distance = distance_df.iloc[i, j]
            
            if groups[sample_i] == groups[sample_j]:
                in_group_distances.append(distance)
            else:
                between_group_distances.append(distance)

    average_in_group_distance = np.mean(in_group_distances) if in_group_distances else np.nan
    average_between_group_distance = np.mean(between_group_distances) if between_group_distances else np.nan

    # --------------------------------------------------------------------------
    # 3) Calculate the final score
    # --------------------------------------------------------------------------
    if np.isnan(average_in_group_distance) or average_in_group_distance == 0:
        score = np.nan if np.isnan(average_between_group_distance) else np.inf
    else:
        score = average_between_group_distance / average_in_group_distance

    # --------------------------------------------------------------------------
    # 4) Prepare results summary
    # --------------------------------------------------------------------------
    result_str = (
        f"Distance Check Results for {row} using {method}\n"
        f"{'='*50}\n"
        f"Number of samples: {len(samples)}\n"
        f"Number of groups: {len(set(groups.values()))}\n"
        f"Group distribution: {dict(pd.Series(list(groups.values())).value_counts())}\n"
        f"Number of in-group pairs: {len(in_group_distances)}\n"
        f"Number of between-group pairs: {len(between_group_distances)}\n"
        f"Average in-group distance: {average_in_group_distance:.6f}\n"
        f"Average between-group distance: {average_between_group_distance:.6f}\n"
        f"Score (between/in-group): {score:.6f}\n"
        f"\nInterpretation:\n"
        f"- Higher scores indicate better separation between groups\n"
        f"- Score > 1: Groups are more distant from each other than within groups\n"
        f"- Score < 1: Groups are closer to each other than within groups\n"
    )

    # --------------------------------------------------------------------------
    # 5) Write the results to a text file
    # --------------------------------------------------------------------------
    output_file = os.path.join(output_dir, f'distance_check_results_{row}_{method}.txt')
    with open(output_file, 'w') as f:
        f.write(result_str)

    print(f"Distance check results saved to {output_file}")
    print(f"Score for {row} ({method}): {score:.6f}")

    # --------------------------------------------------------------------------
    # 6) Update the summary CSV if path provided
    # --------------------------------------------------------------------------
    if summary_csv_path is not None:
        try:
            if os.path.isfile(summary_csv_path):
                summary_df = pd.read_csv(summary_csv_path, index_col=0)
            else:
                # Create a new DataFrame
                summary_df = pd.DataFrame()

            if method not in summary_df.columns:
                summary_df[method] = np.nan

            summary_df.loc[row, method] = score
            summary_df.to_csv(summary_csv_path)
            print(f"Summary updated in {summary_csv_path}")
        except Exception as e:
            print(f"Warning: Failed to update summary CSV: {e}")

    return score