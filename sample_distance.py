import os
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.spatial.distance import pdist, squareform
from Visualization import visualizeGroupRelationship, visualizeDistanceMatrix
from typing import Optional, List

def calculate_sample_distances_DR(
    adata: AnnData,
    DR_key: str,
    output_dir: str,
    method: str = 'euclidean',
    grouping_columns: Optional[List[str]] = None,
    dr_name: str = 'DR'
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
    
    Returns:
    - distance_df: Sample distance matrix (samples x samples)
    """
    
    # Validate inputs
    if DR_key not in adata.uns:
        raise KeyError(f"DR key '{DR_key}' not found in adata.uns. Available keys: {list(adata.uns.keys())}")
    
    DR = adata.uns[DR_key]
    
    if DR is None or DR.empty:
        raise ValueError(f"DR DataFrame for key '{DR_key}' is empty or None.")
    
    print(f"Debug: DR data shape: {DR.shape}")
    print(f"Debug: DR index (first 5): {DR.index[:5].tolist()}")
    print(f"Debug: AnnData obs index (first 5): {adata.obs.index[:5].tolist()}")
    
    # Create output directory
    output_dir = os.path.join(output_dir, f'{dr_name}_distance')
    os.makedirs(output_dir, exist_ok=True)
    
    # Use DR data directly (it should already be averaged per sample)
    # Fill NaN values with 0
    dr_data = DR.fillna(0)
    
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
    
    # Create a temporary CSV with sample metadata for distanceCheck compatibility
    # (in case distanceCheck still expects a CSV path)
    temp_summary_path = os.path.join(output_dir, f'temp_sample_metadata_{dr_name}.csv')
    
    # Handle potential case mismatch between DR index and adata.obs index
    # Try to find matching samples with case-insensitive comparison
    dr_samples = set(dr_data.index)
    adata_samples = set(adata.obs.index)
    
    print(f"Debug: DR samples: {sorted(list(dr_samples))[:10]}")  # Show first 10
    print(f"Debug: AnnData samples: {sorted(list(adata_samples))[:10]}")  # Show first 10
    
    # Check for exact matches first
    exact_matches = dr_samples.intersection(adata_samples)
    print(f"Debug: Exact matches found: {len(exact_matches)}")
    
    if len(exact_matches) == 0:
        # Try case-insensitive matching
        print("Debug: No exact matches, trying case-insensitive matching...")
        
        # Create mapping from lowercase to original names
        dr_lower_to_orig = {name.lower(): name for name in dr_samples}
        adata_lower_to_orig = {name.lower(): name for name in adata_samples}
        
        # Find matches in lowercase
        lowercase_matches = set(dr_lower_to_orig.keys()).intersection(set(adata_lower_to_orig.keys()))
        print(f"Debug: Case-insensitive matches found: {len(lowercase_matches)}")
        
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
            
            # Rename the index to match AnnData sample names
            dr_data_filtered.index = [sample_mapping[name] for name in dr_data_filtered.index]
            
            # Extract corresponding sample metadata from adata.obs
            matching_adata_samples = list(sample_mapping.values())
            sample_metadata = adata.obs.loc[matching_adata_samples].copy()
            
            print(f"Debug: Using {len(matching_dr_samples)} matched samples")
        else:
            raise ValueError(f"No matching samples found between DR results and AnnData. "
                           f"DR samples: {sorted(list(dr_samples))[:5]}, "
                           f"AnnData samples: {sorted(list(adata_samples))[:5]}")
    else:
        # Use exact matches
        matching_samples = list(exact_matches)
        dr_data_filtered = dr_data.loc[matching_samples].copy()
        sample_metadata = adata.obs.loc[matching_samples].copy()
        print(f"Debug: Using {len(matching_samples)} exactly matched samples")
    
    # Use the filtered DR data for distance calculation
    dr_data = dr_data_filtered
    sample_metadata.to_csv(temp_summary_path)
    
    # Perform integrated distance check
    try:
        score = _distanceCheck_integrated(
            distance_df=distance_df,
            dr_name=dr_name,
            method=method,
            output_dir=output_dir,
            adata=adata,
            grouping_columns=grouping_columns
        )
    except Exception as e:
        print(f"Warning: Distance check failed for {dr_name}: {e}")
        score = np.nan
    
    visualizeDistanceMatrix(
        distance_df, 
        os.path.join(output_dir, f'sample_distance_{dr_name}_heatmap.pdf')
    )
    
    visualizeGroupRelationship(
        distance_df, 
        outputDir=output_dir, 
        adata=adata, 
        grouping_columns=grouping_columns,
        heatmap_path=os.path.join(output_dir, f'sample_{dr_name}_relationship.pdf')
    )
    
    print(f"{dr_name}-based distance matrix saved to: {output_dir}")
    return distance_df

def sample_distance_dr_only(
    adata: AnnData,
    output_dir: str,
    method: str,
    grouping_columns: Optional[List[str]] = None
) -> None:
    """
    Compute and save sample distance matrices using dimension reduction results only.
    
    This function looks for DR results stored in adata.uns under the unified keys:
    - 'X_DR_expression': Expression-based dimension reduction
    - 'X_DR_proportion': Proportion-based dimension reduction
    
    Sample metadata is automatically extracted from adata.obs.
    
    Parameters:
    - adata: AnnData object with DR results in .uns and sample metadata in .obs
    - output_dir: Directory for outputs
    - method: Distance metric
    - grouping_columns: List of columns for grouping analysis
    
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
    
    # Debug: Print available DR keys
    print(f"Debug: Available keys in adata.uns: {list(adata.uns.keys())}")
    
    # Track which distance matrices were computed
    computed_distances = []
    
    # Compute distance matrix for expression DR if available
    if 'X_DR_expression' in adata.uns:
        try:
            print("Computing sample distances using expression dimension reduction...")
            calculate_sample_distances_DR(
                adata=adata,
                DR_key='X_DR_expression',
                output_dir=output_dir,
                method=method,
                grouping_columns=grouping_columns,
                dr_name='expression_DR'
            )
            computed_distances.append('expression_DR')
        except Exception as e:
            print(f"Failed to compute expression DR distances: {e}")
    else:
        print("Warning: 'X_DR_expression' not found in adata.uns")
    
    # Compute distance matrix for proportion DR if available
    if 'X_DR_proportion' in adata.uns:
        try:
            print("Computing sample distances using proportion dimension reduction...")
            calculate_sample_distances_DR(
                adata=adata,
                DR_key='X_DR_proportion',
                output_dir=output_dir,
                method=method,
                grouping_columns=grouping_columns,
                dr_name='proportion_DR'
            )
            computed_distances.append('proportion_DR')
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
                calculate_sample_distances_DR(
                    adata=adata,
                    DR_key=dr_key,
                    output_dir=output_dir,
                    method=method,
                    grouping_columns=grouping_columns,
                    dr_name=dr_name
                )
                computed_distances.append(dr_name)
            except Exception as e:
                print(f"Failed to compute {dr_name} distances: {e}")
    
    if not computed_distances:
        raise ValueError("No dimension reduction results found in adata.uns. "
                        "Please run dimension reduction first.")
    
    print(f"Sample distance computations completed for: {', '.join(computed_distances)}")

# Backward compatibility: alias to the original function name
def sample_distance(
    adata: AnnData,
    output_dir: str,
    method: str,
    grouping_columns: Optional[List[str]] = None
) -> None:
    """
    Backward compatibility wrapper for the original sample_distance function.
    
    This function now only uses dimension reduction results stored in adata.uns
    and sample metadata stored in adata.obs.
    
    Deprecated parameters (ignored):
    - summary_csv_path: Sample metadata is now extracted from adata.obs
    - pseudobulk: Dimension reduction results are used instead
    - sample_column: Sample information is implicit in adata structure
    """
    
    sample_distance_dr_only(
        adata=adata,
        output_dir=output_dir,
        method=method,
        grouping_columns=grouping_columns
    )