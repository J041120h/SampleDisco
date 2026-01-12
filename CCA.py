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
    from utils.random_seed import set_global_seed
    set_global_seed(seed = 42, verbose = verbose)
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


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr
def plot_cca_on_2d_pca(
    pca_coords_full: np.ndarray,
    sev_levels: np.ndarray,
    auto_select_best_2pc: bool = True,
    pc_indices: tuple = None,
    output_path: str = None,
    sample_labels=None,
    title_suffix: str = "",
    verbose: bool = False,
    create_contribution_plot: bool = True
):
    """
    Plot 2D PCA with CCA direction overlay, with PC selection logic integrated.
    Optionally creates a companion plot showing PC contributions to CCA.
    """
    from utils.random_seed import set_global_seed
    set_global_seed(seed=42, verbose=verbose)

    # ------------------------------------------------------------------
    # 1) Make severity levels numeric & 1D float
    # ------------------------------------------------------------------
    sev_levels = np.asarray(sev_levels).reshape(-1)

    if not np.issubdtype(sev_levels.dtype, np.number):
        # Try direct numeric cast; if that fails, fall back to category codes
        try:
            sev_levels = sev_levels.astype(float)
        except (ValueError, TypeError):
            # Map unique categories to 0,1,2,...
            _, sev_codes = np.unique(sev_levels, return_inverse=True)
            sev_levels = sev_codes.astype(float)

    # At this point sev_levels is numeric float array
    n_components = pca_coords_full.shape[1]

    # ------------------------------------------------------------------
    # 2) PC selection + CCA
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 3) MAIN PLOT
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))

    # Normalize severity levels for coloring (handle constant case)
    sev_min = np.min(sev_levels)
    sev_max = np.max(sev_levels)
    sev_range = sev_max - sev_min

    if sev_range < 1e-16:
        norm_sev = np.zeros_like(sev_levels, dtype=float)
    else:
        norm_sev = (sev_levels - sev_min) / (sev_range + 1e-16)

    # Create scatter plot
    sc = ax.scatter(
        pca_coords_2d[:, 0],
        pca_coords_2d[:, 1],
        c=norm_sev,
        cmap='viridis_r',
        edgecolors='k',
        alpha=0.8,
        s=60,
    )
    cbar = plt.colorbar(sc, ax=ax, label='Normalized Severity Level')

    # Draw CCA direction vector
    dx, dy = cca_model.x_weights_[:, 0]
    scale = 0.5 * max(np.ptp(pca_coords_2d[:, 0]), np.ptp(pca_coords_2d[:, 1]))
    x_start, x_end = -scale * dx, scale * dx
    y_start, y_end = -scale * dy, scale * dy

    ax.plot(
        [x_start, x_end],
        [y_start, y_end],
        linestyle="--",
        color="red",
        linewidth=3,
        label="CCA Direction",
        alpha=0.9,
    )

    # Add sample labels if requested
    if sample_labels is not None:
        for i, label in enumerate(sample_labels):
            ax.text(
                pca_coords_2d[i, 0],
                pca_coords_2d[i, 1],
                str(label),
                fontsize=8,
                alpha=0.7,
            )

    # Set labels and title
    ax.set_xlabel(f"PC{pc_indices_used[0]+1}", fontsize=12)
    ax.set_ylabel(f"PC{pc_indices_used[1]+1}", fontsize=12)
    title = f"PCA (PC{pc_indices_used[0]+1} vs PC{pc_indices_used[1]+1}) with CCA Direction"
    if title_suffix:
        title += f" - {title_suffix}"
    if auto_select_best_2pc and n_components > 2:
        title += f" (Auto-selected, Score: {cca_score:.3f})"
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    # Save or show main plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"Saved CCA plot to: {output_path}")
    else:
        plt.show()

    plt.close()

    # ------------------------------------------------------------------
    # 4) CONTRIBUTION PLOT
    # ------------------------------------------------------------------
    if create_contribution_plot:
        # Handle constant or near-constant severity gracefully
        sev_std = np.std(sev_levels)
        if sev_std < 1e-16:
            pc1_corr = 0.0
            pc2_corr = 0.0
        else:
            pc1_corr, _ = pearsonr(pca_coords_2d[:, 0], sev_levels)
            pc2_corr, _ = pearsonr(pca_coords_2d[:, 1], sev_levels)

        fig, ax = plt.subplots(figsize=(10, 6))

        x_pos = np.arange(3)
        colors = ['#3498db', '#e74c3c', '#2ecc71']

        # Show: individual correlations and CCA score
        values = [pc1_corr, pc2_corr, cca_score]
        labels = [
            f'PC{pc_indices_used[0]+1}\n(r={pc1_corr:.3f})',
            f'PC{pc_indices_used[1]+1}\n(r={pc2_corr:.3f})',
            f'CCA Combined\n(r={cca_score:.3f})',
        ]

        bars = ax.bar(
            x_pos,
            values,
            color=colors,
            alpha=0.7,
            edgecolor='black',
            linewidth=2,
            width=0.6,
        )

        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel('Correlation with Severity', fontsize=12, fontweight='bold')
        ax.set_ylim([min(values) - 0.1, max(values) + 0.15])
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f'{val:.3f}',
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold',
            )

        # Add CCA weights as text annotation
        weight_text = (
            f"CCA Weights: PC{pc_indices_used[0]+1}={dx:.3f}, "
            f"PC{pc_indices_used[1]+1}={dy:.3f}\n"
            f"(Direction: {dx:.2f}×PC{pc_indices_used[0]+1} + {dy:.2f}×PC{pc_indices_used[1]+1})"
        )
        ax.text(
            0.5,
            0.98,
            weight_text,
            transform=ax.transAxes,
            fontsize=10,
            ha='center',
            va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        )

        title = "PC Contributions to CCA"
        if title_suffix:
            title += f" - {title_suffix}"
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        # Save or show contribution plot
        if output_path:
            import os
            base, ext = os.path.splitext(output_path)
            contribution_path = f"{base}_contributions{ext}"
            plt.savefig(contribution_path, dpi=300, bbox_inches='tight')
            if verbose:
                print(f"Saved CCA contributions plot to: {contribution_path}")
        else:
            plt.show()

        plt.close()

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
    from utils.random_seed import set_global_seed
    set_global_seed(seed = 42, verbose = verbose)
    
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