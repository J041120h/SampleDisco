import os
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.spatial.distance import pdist, squareform
from Visualization import visualizeGroupRelationship, visualizeDistanceMatrix
from typing import Optional, List
from distance_test import distanceCheck
from .EMD import EMD_distances
from .ChiSquare import chi_square_distance
from .jensenshannon import jensen_shannon_distance

def calculate_sample_distances_DR(
    adata: AnnData,
    DR_key: str,
    output_dir: str,
    method: str = 'euclidean',
    grouping_columns: Optional[List[str]] = None,
    dr_name: str = 'DR',
    summary_csv_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute a sample distance matrix using dimensionality reduction data stored in adata.uns.
    
    Parameters:
    - adata: AnnData object containing DR results in .uns[DR_key] and sample metadata in .obs
    - DR_key: Key in adata.uns where the DR DataFrame is stored
    - output_dir: Directory for output files
    - method: Distance metric for pdist (default: 'euclidean')
    - grouping_columns: Optional list of columns for grouping analysis
    - dr_name: Name of the dimensionality reduction method for file naming
    - summary_csv_path: Optional path to summary CSV for logging results
    
    Returns:
    - distance_df: Sample distance matrix (samples x samples)
    """
    
    # Validate inputs
    if DR_key not in adata.uns:
        raise KeyError(f"DR key '{DR_key}' not found in adata.uns. Available keys: {list(adata.uns.keys())}")
    
    DR = adata.uns[DR_key]
    
    if DR is None or DR.empty:
        raise ValueError(f"DR DataFrame for key '{DR_key}' is empty or None.")
    
    # Create output directory
    output_dir = os.path.join(output_dir, f'{dr_name}_distance')
    os.makedirs(output_dir, exist_ok=True)
    
    # Use DR data directly (it should already be averaged per sample)
    # Fill NaN values with 0
    dr_data = DR.fillna(0)
    
    # Handle potential case mismatch between DR index and adata.obs index
    # Try to find matching samples with case-insensitive comparison
    dr_samples = set(dr_data.index)
    adata_samples = set(adata.obs.index)
    
    # Check for exact matches first
    exact_matches = dr_samples.intersection(adata_samples)
    
    if len(exact_matches) == 0:
        # Try case-insensitive matching
        # Create mapping from lowercase to original names
        dr_lower_to_orig = {name.lower(): name for name in dr_samples}
        adata_lower_to_orig = {name.lower(): name for name in adata_samples}
        
        # Find matches in lowercase
        lowercase_matches = set(dr_lower_to_orig.keys()).intersection(set(adata_lower_to_orig.keys()))
        
        if len(lowercase_matches) > 0:
            # Create mapping from DR sample names to AnnData sample names
            sample_mapping = {}
            for lower_name in lowercase_matches:
                dr_name_orig = dr_lower_to_orig[lower_name]
                adata_name_orig = adata_lower_to_orig[lower_name]
                sample_mapping[dr_name_orig] = adata_name_orig
            
            # Filter DR data to only include matching samples
            matching_dr_samples = list(sample_mapping.keys())
            dr_data_filtered = dr_data.loc[matching_dr_samples].copy()
            
            # Rename the index to match AnnData sample names for consistency
            dr_data_filtered.index = [sample_mapping[name] for name in dr_data_filtered.index]
            
            # Use filtered data for distance calculation
            dr_data = dr_data_filtered
        else:
            raise ValueError(f"No matching samples found between DR results and AnnData. "
                           f"DR samples: {sorted(list(dr_samples))[:5]}, "
                           f"AnnData samples: {sorted(list(adata_samples))[:5]}")
    else:
        # Use exact matches
        matching_samples = list(exact_matches)
        dr_data = dr_data.loc[matching_samples].copy()
    
    # Save DR data for reference
    dr_data.to_csv(os.path.join(output_dir, f'{dr_name}_coordinates.csv'))
    
    # Compute distances
    distance_matrix = pdist(dr_data.values, metric=method)
    distance_df = pd.DataFrame(
        squareform(distance_matrix), 
        index=dr_data.index, 
        columns=dr_data.index
    )
    
    # Apply log transformation and normalization
    distance_df = np.log1p(np.maximum(distance_df, 0))
    if distance_df.max().max() > 0:  # Avoid division by zero
        distance_df = distance_df / distance_df.max().max()
    
    # Save distance matrix
    distance_matrix_path = os.path.join(output_dir, f'distance_matrix_{dr_name}.csv')
    distance_df.to_csv(distance_matrix_path)
    
    # Perform distance check using the improved distanceCheck function
    try:
        score = distanceCheck(
            distance_df=distance_df,
            row=dr_name,
            method=method,
            output_dir=output_dir,
            adata=adata,
            grouping_columns=grouping_columns,
            summary_csv_path=summary_csv_path  # Pass through the summary CSV path
        )
        print(f"Distance check completed for {dr_name}: score = {score:.6f}")
    except Exception as e:
        print(f"Warning: Distance check failed for {dr_name}: {e}")
    
    # Generate visualizations
    try:
        visualizeDistanceMatrix(
            distance_df, 
            os.path.join(output_dir, f'sample_distance_{dr_name}_heatmap.pdf')
        )
    except Exception as e:
        print(f"Warning: Failed to create distance heatmap for {dr_name}: {e}")
    
    try:
        visualizeGroupRelationship(
            distance_df, 
            outputDir=output_dir, 
            adata=adata, 
            grouping_columns=grouping_columns,
            heatmap_path=os.path.join(output_dir, f'sample_{dr_name}_relationship.pdf')
        )
    except Exception as e:
        print(f"Warning: Failed to create group relationship visualization for {dr_name}: {e}")
    
    print(f"{dr_name}-based distance matrix saved to: {output_dir}")
    return distance_df

def sample_distance_vector(
    adata: AnnData,
    output_dir: str,
    method: str,
    grouping_columns: Optional[List[str]] = None,
    summary_csv_path: Optional[str] = None
) -> None:
    """
    Compute and save sample distance matrices using dimension reduction vector results.
    
    This function looks for DR results stored in adata.uns under the unified keys:
    - 'X_DR_expression': Expression-based dimension reduction
    - 'X_DR_proportion': Proportion-based dimension reduction
    
    Sample metadata is automatically extracted from adata.obs.
    
    Parameters:
    - adata: AnnData object with DR results in .uns and sample metadata in .obs
    - output_dir: Directory for outputs
    - method: Distance metric
    - grouping_columns: List of columns for grouping analysis
    - summary_csv_path: Optional path to global summary CSV for logging results across all methods
    
    Returns:
    - None (results are saved to disk)
    """
    output_dir = os.path.join(output_dir, method)
    os.makedirs(output_dir, exist_ok=True)
    
    # Print available sample metadata columns for reference
    if hasattr(adata, 'obs') and not adata.obs.empty:
        print(f"Available sample metadata columns: {list(adata.obs.columns)}")
        print(f"Number of samples: {adata.n_obs}")
        
        # If grouping_columns specified, validate they exist
        if grouping_columns:
            missing_cols = [col for col in grouping_columns if col not in adata.obs.columns]
            if missing_cols:
                print(f"Warning: Grouping columns not found in adata.obs: {missing_cols}")
                grouping_columns = [col for col in grouping_columns if col in adata.obs.columns]
                print(f"Using available grouping columns: {grouping_columns}")
    else:
        print("Warning: No sample metadata found in adata.obs")
    
    # Track which distance matrices were computed and their scores
    computed_distances = []
    distance_results = {}
    
    # Compute distance matrix for expression DR if available
    if 'X_DR_expression' in adata.uns:
        try:
            print("Computing sample distances using expression dimension reduction...")
            distance_df = calculate_sample_distances_DR(
                adata=adata,
                DR_key='X_DR_expression',
                output_dir=output_dir,
                method=method,
                grouping_columns=grouping_columns,
                dr_name='expression_DR',
                summary_csv_path=summary_csv_path
            )
            computed_distances.append('expression_DR')
            distance_results['expression_DR'] = distance_df
        except Exception as e:
            print(f"Failed to compute expression DR distances: {e}")
    else:
        print("Warning: 'X_DR_expression' not found in adata.uns")
    
    # Compute distance matrix for proportion DR if available
    if 'X_DR_proportion' in adata.uns:
        try:
            print("Computing sample distances using proportion dimension reduction...")
            distance_df = calculate_sample_distances_DR(
                adata=adata,
                DR_key='X_DR_proportion',
                output_dir=output_dir,
                method=method,
                grouping_columns=grouping_columns,
                dr_name='proportion_DR',
                summary_csv_path=summary_csv_path
            )
            computed_distances.append('proportion_DR')
            distance_results['proportion_DR'] = distance_df
        except Exception as e:
            print(f"Failed to compute proportion DR distances: {e}")
    else:
        print("Warning: 'X_DR_proportion' not found in adata.uns")
    
    # Also check for any method-specific DR results for additional analysis
    method_specific_keys = [
        ('X_pca_expression_method', 'PCA_expression'),
        ('X_lsi_expression_method', 'LSI_expression'),
        ('X_spectral_expression_method', 'spectral_expression')
    ]
    
    for dr_key, dr_name in method_specific_keys:
        if dr_key in adata.uns:
            try:
                print(f"Computing sample distances using {dr_name}...")
                distance_df = calculate_sample_distances_DR(
                    adata=adata,
                    DR_key=dr_key,
                    output_dir=output_dir,
                    method=method,
                    grouping_columns=grouping_columns,
                    dr_name=dr_name,
                    summary_csv_path=summary_csv_path
                )
                computed_distances.append(dr_name)
                distance_results[dr_name] = distance_df
            except Exception as e:
                print(f"Failed to compute {dr_name} distances: {e}")
    
    if not computed_distances:
        raise ValueError("No dimension reduction results found in adata.uns. "
                        "Please run dimension reduction first.")
    
    print(f"Sample distance computations completed for: {', '.join(computed_distances)}")
    
    # Optional: Create additional summary statistics
    if distance_results:
        try:
            # Create a summary of basic distance statistics
            summary_stats = {}
            for dr_name, dist_df in distance_results.items():
                # Get upper triangle values (excluding diagonal)
                dist_values = dist_df.values[np.triu_indices_from(dist_df.values, k=1)]
                summary_stats[f"{dr_name}_mean"] = np.mean(dist_values)
                summary_stats[f"{dr_name}_std"] = np.std(dist_values)
                summary_stats[f"{dr_name}_min"] = np.min(dist_values)
                summary_stats[f"{dr_name}_max"] = np.max(dist_values)
                summary_stats[f"{dr_name}_median"] = np.median(dist_values)
            
            # Save basic statistics summary
            stats_file = os.path.join(output_dir, f'distance_statistics_summary_{method}.csv')
            stats_df = pd.DataFrame([summary_stats])
            stats_df.to_csv(stats_file)
            print(f"Distance statistics summary saved to: {stats_file}")
            
        except Exception as e:
            print(f"Warning: Failed to create distance statistics summary: {e}")

def sample_distance(
    adata: AnnData,
    output_dir: str,
    method: str,
    grouping_columns: Optional[List[str]] = None,
    summary_csv_path: Optional[str] = None,
    cell_adata: Optional[AnnData] = None,
    cell_type_column: str = 'cell_type',
    sample_column: str = 'sample',
    pseudobulk_adata: Optional[AnnData] = None
) -> None:
    """
    Unified function to compute sample distance matrices using various methods.
    
    This function handles both standard distance metrics (using dimension reduction results)
    and specialized distance methods (EMD, chi-square, Jensen-Shannon).
    
    Parameters:
    - adata: AnnData object with DR results in .uns and sample metadata in .obs
    - output_dir: Directory for outputs
    - method: Distance metric ('euclidean', 'cosine', 'EMD', 'chi_square', 'jensen_shannon', etc.)
    - grouping_columns: List of columns for grouping analysis
    - summary_csv_path: Optional path to summary CSV for logging results across methods
    - cell_adata: AnnData object with single-cell data (required for EMD, chi_square, jensen_shannon)
    - cell_type_column: Column name for cell types in cell_adata (default: 'cell_type')
    - sample_column: Column name for samples in cell_adata (default: 'sample')
    - pseudobulk_adata: Pseudobulk AnnData object (required for EMD, chi_square, jensen_shannon)
    """
    
    # Standard distance metrics that work with dimension reduction results
    valid_pdist_metrics = {
        "euclidean",
        "sqeuclidean",
        "minkowski",
        "cityblock",
        "chebyshev",
        "cosine",
        "correlation",
        "hamming",
        "jaccard",
        "canberra",
        "braycurtis",
        "matching",
    }
    
    # Specialized distance methods
    specialized_methods = {"EMD", "chi_square", "jensen_shannon"}
    
    if method in valid_pdist_metrics:
        # Handle standard distance metrics
        print(f"Computing {method} distance using dimension reduction results...")
        sample_distance_vector(
            adata=adata,
            output_dir=output_dir,
            method=method,
            grouping_columns=grouping_columns,
            summary_csv_path=summary_csv_path
        )
        
    elif method in specialized_methods:
        # Handle specialized distance methods
        if cell_adata is None:
            raise ValueError(f"cell_adata is required for {method} distance calculation")
        
        if method == "EMD":
            print("Computing Earth Mover's Distance (EMD)...")
            emd_output_dir = os.path.join(output_dir, 'sample_level_EMD')
            EMD_distances(
                adata=cell_adata,
                output_dir=emd_output_dir,
                summary_csv_path=summary_csv_path,
                cell_type_column=cell_type_column,
                sample_column=sample_column,
                pseudobulk_adata=pseudobulk_adata
            )
            
        elif method == "chi_square":
            print("Computing Chi-square distance...")
            chi_output_dir = os.path.join(output_dir, 'Chi_square_sample')
            chi_square_distance(
                adata=cell_adata,
                output_dir=chi_output_dir,
                summary_csv_path=summary_csv_path,
                cell_type_column=cell_type_column,
                sample_column=sample_column,
                pseudobulk_adata=pseudobulk_adata
            )
            
        elif method == "jensen_shannon":
            print("Computing Jensen-Shannon distance...")
            js_output_dir = os.path.join(output_dir, 'jensen_shannon_sample')
            jensen_shannon_distance(
                adata=cell_adata,
                output_dir=js_output_dir,
                summary_csv_path=summary_csv_path,
                cell_type_column=cell_type_column,
                sample_column=sample_column,
                pseudobulk_adata=pseudobulk_adata
            )
    else:
        print(f"Warning: Unknown distance method '{method}'. Skipping...")
        return