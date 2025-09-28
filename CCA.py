import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from anndata import AnnData
from sklearn.cross_decomposition import CCA
import time
from itertools import combinations


def find_best_2pc_combination(
    pca_coords: np.ndarray,
    sev_levels: np.ndarray,
    verbose: bool = False
):
    """
    Find the best 2-PC combination that maximizes CCA correlation.
    
    Parameters:
    -----------
    pca_coords : np.ndarray
        PCA coordinates with shape (n_samples, n_components)
    sev_levels : np.ndarray
        Severity levels for each sample
    verbose : bool
        Whether to print detailed information
        
    Returns:
    --------
    tuple: (best_pc_indices, best_score, best_cca_model, best_pca_coords_2d)
    """
    n_components = pca_coords.shape[1]
    
    if n_components < 2:
        raise ValueError("Need at least 2 PC components")
    
    if n_components == 2:
        # Only one combination possible
        sev_levels_2d = sev_levels.reshape(-1, 1)
        cca = CCA(n_components=1)
        cca.fit(pca_coords, sev_levels_2d)
        U, V = cca.transform(pca_coords, sev_levels_2d)
        score = np.corrcoef(U[:, 0], V[:, 0])[0, 1]
        return (0, 1), score, cca, pca_coords
    
    # Test all possible 2-PC combinations
    best_score = -1
    best_combination = None
    best_cca = None
    best_coords_2d = None
    
    sev_levels_2d = sev_levels.reshape(-1, 1)
    
    if verbose:
        print(f"Testing {n_components * (n_components - 1) // 2} PC combinations...")
    
    for pc1, pc2 in combinations(range(n_components), 2):
        pca_subset = pca_coords[:, [pc1, pc2]]
        
        try:
            cca = CCA(n_components=1)
            cca.fit(pca_subset, sev_levels_2d)
            U, V = cca.transform(pca_subset, sev_levels_2d)
            score = abs(np.corrcoef(U[:, 0], V[:, 0])[0, 1])  # Use absolute value
            
            if verbose:
                print(f"PC{pc1+1} + PC{pc2+1}: CCA score = {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_combination = (pc1, pc2)
                best_cca = cca
                best_coords_2d = pca_subset
                
        except Exception as e:
            if verbose:
                print(f"Error with PC{pc1+1} + PC{pc2+1}: {str(e)}")
            continue
    
    if best_combination is None:
        raise ValueError("Could not find valid PC combination for CCA")
    
    if verbose:
        print(f"Best combination: PC{best_combination[0]+1} + PC{best_combination[1]+1} "
              f"with score {best_score:.4f}")
    
    return best_combination, best_score, best_cca, best_coords_2d


def run_cca_on_pca_from_adata(
    adata: AnnData,
    column: str,
    sev_col: str = "sev.level",
    n_components: int = 2,
    verbose: bool = False
):
    """
    Run CCA analysis on PCA coordinates from AnnData object.
    Now returns full PCA coordinates and lets visualization function handle PC selection.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    column : str
        Key in adata.uns containing PCA coordinates
    sev_col : str
        Column name in adata.obs containing severity levels
    n_components : int
        Number of PC components to use (default: 2)
    verbose : bool
        Whether to print detailed information
        
    Returns:
    --------
    tuple: (pca_coords_full, sev_levels, samples, n_components_used)
    """
    if column not in adata.uns:
        raise KeyError(f"'{column}' not found in adata.uns. Available keys: {list(adata.uns.keys())}")
    
    if sev_col not in adata.obs.columns:
        raise KeyError(f"'{sev_col}' column is missing in adata.obs. Available columns: {list(adata.obs.columns)}")

    pca_coords = adata.uns[column]
    
    # Convert to numpy array if it's a DataFrame
    if hasattr(pca_coords, 'iloc'):
        pca_coords_array = pca_coords.values
    else:
        pca_coords_array = pca_coords
    
    # Check if we have enough components
    available_components = pca_coords_array.shape[1]
    if available_components < n_components:
        if verbose:
            print(f"Warning: Only {available_components} components available, using all of them.")
        n_components = available_components
    
    if n_components < 2:
        raise ValueError("Need at least 2 PC components for CCA analysis.")
    
    # Extract the requested number of components
    pca_coords_subset = pca_coords_array[:, :n_components]
    
    # Process severity levels
    sev_levels = pd.to_numeric(adata.obs[sev_col], errors='coerce').values
    missing = np.isnan(sev_levels).sum()
    if missing > 0:
        if verbose:
            print(f"Warning: {missing} sample(s) missing severity level. Imputing with mean.")
        sev_levels[np.isnan(sev_levels)] = np.nanmean(sev_levels)
    
    if len(sev_levels) != pca_coords_subset.shape[0]:
        raise ValueError(f"Mismatch between PCA rows ({pca_coords_subset.shape[0]}) and severity levels ({len(sev_levels)}).")

    samples = adata.obs.index.values
    
    return pca_coords_subset, sev_levels, samples, n_components


def plot_cca_on_2d_pca(
    pca_coords_full: np.ndarray,
    sev_levels: np.ndarray,
    auto_select_best_2pc: bool = True,
    pc_indices: tuple = None,
    output_path: str = None,
    sample_labels=None,
    title_suffix: str = "",
    verbose: bool = False
):
    """
    Plot 2D PCA with CCA direction overlay, with PC selection logic integrated.
    
    Parameters:
    -----------
    pca_coords_full : np.ndarray
        Full PCA coordinates (can be >2D)
    sev_levels : np.ndarray
        Severity levels for coloring
    auto_select_best_2pc : bool
        If True, automatically select best 2-PC combination
    pc_indices : tuple
        Specific PC indices to use (if auto_select_best_2pc is False)
    output_path : str
        Path to save the plot
    sample_labels : array-like
        Labels for each sample
    title_suffix : str
        Additional text for the title
    verbose : bool
        Whether to print information
        
    Returns:
    --------
    tuple: (cca_score, pc_indices_used, cca_model)
    """
    n_components = pca_coords_full.shape[1]
    
    # PC selection logic
    if auto_select_best_2pc and n_components > 2:
        if verbose:
            print(f"Auto-selecting best 2-PC combination from {n_components} components...")
        
        pc_indices_used, cca_score, cca_model, pca_coords_2d = find_best_2pc_combination(
            pca_coords_full, sev_levels, verbose
        )
        
    elif pc_indices is not None:
        # Use specified PC indices
        if len(pc_indices) != 2:
            raise ValueError("pc_indices must contain exactly 2 indices")
        if max(pc_indices) >= n_components:
            raise ValueError(f"PC index {max(pc_indices)} exceeds available components ({n_components})")
        
        pc_indices_used = pc_indices
        pca_coords_2d = pca_coords_full[:, list(pc_indices)]
        
        # Run CCA on specified PCs
        sev_levels_2d = sev_levels.reshape(-1, 1)
        cca_model = CCA(n_components=1)
        cca_model.fit(pca_coords_2d, sev_levels_2d)
        U, V = cca_model.transform(pca_coords_2d, sev_levels_2d)
        cca_score = np.corrcoef(U[:, 0], V[:, 0])[0, 1]
        
        if verbose:
            print(f"Using specified PC{pc_indices[0]+1} + PC{pc_indices[1]+1}: CCA score = {cca_score:.4f}")
            
    else:
        # Use first 2 components by default
        pc_indices_used = (0, 1)
        pca_coords_2d = pca_coords_full[:, :2]
        
        # Run CCA on first 2 PCs
        sev_levels_2d = sev_levels.reshape(-1, 1)
        cca_model = CCA(n_components=1)
        cca_model.fit(pca_coords_2d, sev_levels_2d)
        U, V = cca_model.transform(pca_coords_2d, sev_levels_2d)
        cca_score = np.corrcoef(U[:, 0], V[:, 0])[0, 1]
        
        if verbose:
            print(f"Using default PC1 + PC2: CCA score = {cca_score:.4f}")

    # Create the visualization
    plt.figure(figsize=(10, 8))

    # Normalize severity levels for coloring
    norm_sev = (sev_levels - np.min(sev_levels)) / (np.max(sev_levels) - np.min(sev_levels) + 1e-16)

    # Create scatter plot
    sc = plt.scatter(
        pca_coords_2d[:, 0],
        pca_coords_2d[:, 1],
        c=norm_sev,
        cmap='viridis_r',
        edgecolors='k',
        alpha=0.8,
        s=60
    )
    cbar = plt.colorbar(sc, label='Normalized Severity Level')

    # Draw CCA direction vector
    dx, dy = cca_model.x_weights_[:, 0]
    scale = 0.5 * max(np.ptp(pca_coords_2d[:, 0]), np.ptp(pca_coords_2d[:, 1]))
    x_start, x_end = -scale * dx, scale * dx
    y_start, y_end = -scale * dy, scale * dy

    plt.plot([x_start, x_end], [y_start, y_end],
             linestyle="--", color="red", linewidth=3, label="CCA Direction", alpha=0.9)

    # Add sample labels if requested
    if sample_labels is not None:
        for i, label in enumerate(sample_labels):
            plt.text(pca_coords_2d[i, 0], pca_coords_2d[i, 1], 
                    str(label), fontsize=8, alpha=0.7)

    # Set labels and title
    plt.xlabel(f"PC{pc_indices_used[0]+1}")
    plt.ylabel(f"PC{pc_indices_used[1]+1}")
    title = f"PCA (PC{pc_indices_used[0]+1} vs PC{pc_indices_used[1]+1}) with CCA Direction"
    if title_suffix:
        title += f" - {title_suffix}"
    if auto_select_best_2pc and n_components > 2:
        title += f" (Auto-selected, Score: {cca_score:.3f})"
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save or show plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"Saved CCA plot to: {output_path}")
    else:
        plt.show()
    
    return cca_score, pc_indices_used, cca_model


def assign_pseudotime_from_cca(
    pca_coords_2d: np.ndarray, 
    cca: CCA, 
    sample_labels: np.ndarray,
    scale_to_unit: bool = True
) -> dict:
    """
    Assign pseudotime values based on CCA projection.
    
    Parameters:
    -----------
    pca_coords_2d : np.ndarray
        2D PCA coordinates
    cca : CCA
        Fitted CCA model
    sample_labels : np.ndarray
        Sample identifiers
    scale_to_unit : bool
        Whether to scale pseudotime to [0, 1] range
        
    Returns:
    --------
    dict: Mapping from sample labels to pseudotime values
    """
    direction = cca.x_weights_[:, 0]
    raw_projection = pca_coords_2d @ direction

    if scale_to_unit:
        min_proj, max_proj = np.min(raw_projection), np.max(raw_projection)
        denom = max_proj - min_proj
        if denom < 1e-16:
            denom = 1e-16
        pseudotimes = (raw_projection - min_proj) / denom
    else:
        pseudotimes = raw_projection

    return {str(sample_labels[i]): pseudotimes[i] for i in range(len(sample_labels))}


def CCA_Call(
    adata: AnnData,
    output_dir: str = None,
    sev_col: str = "sev.level",
    n_components: int = 2,
    auto_select_best_2pc: bool = True,
    ptime: bool = False,
    verbose: bool = False,
    show_sample_labels: bool = False
):
    """
    Main function to run CCA analysis with PC selection integrated into visualization.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    output_dir : str
        Directory to save output files
    sev_col : str
        Column name for severity levels
    n_components : int
        Number of PC components to use (default: 2)
    auto_select_best_2pc : bool
        If True, automatically select best 2-PC combination for visualization
    ptime : bool
        Whether to compute pseudotime (kept for compatibility)
    verbose : bool
        Whether to print detailed information
    show_sample_labels : bool
        Whether to show sample labels on plots
        
    Returns:
    --------
    tuple: (proportion_score, expression_score, proportion_pseudotime, expression_pseudotime)
    """
    start_time = time.time() if verbose else None
    
    if output_dir:
        output_dir = os.path.join(output_dir, 'CCA')
        os.makedirs(output_dir, exist_ok=True)

    paths = {
        "X_DR_proportion": os.path.join(output_dir, f"pca_{n_components}d_cca_proportion.pdf") if output_dir else None,
        "X_DR_expression": os.path.join(output_dir, f"pca_{n_components}d_cca_expression.pdf") if output_dir else None
    }

    results = {}
    sample_dicts = {}
    pc_info = {}

    for key in ["X_DR_proportion", "X_DR_expression"]:
        if verbose:
            print(f"\nProcessing {key}...")
            
        try:
            # Get full PCA coordinates and metadata
            pca_coords_full, sev_levels, samples, n_components_used = run_cca_on_pca_from_adata(
                adata=adata,
                column=key,
                sev_col=sev_col,
                n_components=n_components,
                verbose=verbose
            )

            # Run visualization with integrated PC selection
            cca_score, pc_indices_used, cca_model = plot_cca_on_2d_pca(
                pca_coords_full=pca_coords_full,
                sev_levels=sev_levels,
                auto_select_best_2pc=auto_select_best_2pc,
                pc_indices=None,  # Let auto-selection handle it
                output_path=paths[key],
                sample_labels=samples if show_sample_labels else None,
                title_suffix=key.replace("X_DR_", "").title(),
                verbose=verbose
            )
            
            results[key] = cca_score
            pc_info[key] = pc_indices_used

            # Get the 2D coordinates used for pseudotime calculation
            pca_coords_2d = pca_coords_full[:, list(pc_indices_used)]
            
            # Assign pseudotime
            sample_dicts[key] = assign_pseudotime_from_cca(
                pca_coords_2d=pca_coords_2d, 
                cca=cca_model, 
                sample_labels=samples
            )
            
        except Exception as e:
            print(f"Error processing {key}: {str(e)}")
            results[key] = np.nan
            sample_dicts[key] = {}
            pc_info[key] = None

    # Save pseudotime data to CSV files
    if output_dir:
        for key in ["X_DR_proportion", "X_DR_expression"]:
            if sample_dicts[key]:  # Only save if we have data
                # Convert dictionary to DataFrame
                pseudotime_df = pd.DataFrame([
                    {'sample': sample_id, 'pseudotime': pseudotime_value}
                    for sample_id, pseudotime_value in sample_dicts[key].items()
                ])
                
                # Create CSV filename
                data_type = key.replace("X_DR_", "")
                csv_filename = f"pseudotime_{data_type}.csv"
                csv_path = os.path.join(output_dir, csv_filename)
                
                # Save to CSV
                pseudotime_df.to_csv(csv_path, index=False)
                
                if verbose:
                    print(f"Saved {data_type} pseudotime data to: {csv_path}")

    if verbose:
        print("\n" + "="*50)
        print("CCA ANALYSIS SUMMARY")
        print("="*50)
        for key in ["X_DR_proportion", "X_DR_expression"]:
            score = results.get(key, np.nan)
            pc_indices = pc_info.get(key, None)
            if pc_indices:
                print(f"{key}: CCA score = {score:.4f} (using PC{pc_indices[0]+1} + PC{pc_indices[1]+1})")
            else:
                print(f"{key}: Failed")
        
        if start_time:
            print(f"\nTotal runtime: {time.time() - start_time:.2f} seconds")
        print("="*50)

    return (results.get("X_DR_proportion", np.nan), 
            results.get("X_DR_expression", np.nan), 
            sample_dicts.get("X_DR_proportion", {}), 
            sample_dicts.get("X_DR_expression", {}))