import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from anndata import AnnData
from sklearn.cross_decomposition import CCA
from itertools import combinations


def find_best_2pc_combination(
    pca_coords: np.ndarray,
    sev_levels: np.ndarray
):
    """
    Pick the 2-PC pair best aligned with severity FOR VISUALIZATION ONLY,
    and report an unbiased CCA score computed on the FULL PC set.

    Selecting the C(n,2) PC pair with maximum |corr| cherry-picks the score
    if returned as the statistical metric. Instead we compute the CCA score
    on the full PC matrix (an unbiased estimate of the linear severity-
    correlation) and use the 2-PC pair only for plotting.

    Returns:
        (best_pc_indices, cca_score, cca_model_2d, best_pca_coords_2d)
        where cca_score is the full-PC CCA correlation (unbiased) and
        cca_model_2d is the 2-PC model used to generate plot coordinates.
    """
    n_components = pca_coords.shape[1]

    if n_components < 2:
        raise ValueError("Need at least 2 PC components")

    sev_levels_2d = sev_levels.reshape(-1, 1)

    full_cca = CCA(n_components=1)
    full_cca.fit(pca_coords, sev_levels_2d)
    U_full, V_full = full_cca.transform(pca_coords, sev_levels_2d)
    full_score = abs(np.corrcoef(U_full[:, 0], V_full[:, 0])[0, 1])

    if n_components == 2:
        return (0, 1), full_score, full_cca, pca_coords

    best_2d_corr = -1.0
    best_combination = None
    best_cca_2d = None
    best_coords_2d = None

    for pc1, pc2 in combinations(range(n_components), 2):
        pca_subset = pca_coords[:, [pc1, pc2]]
        try:
            cca = CCA(n_components=1)
            cca.fit(pca_subset, sev_levels_2d)
            U, V = cca.transform(pca_subset, sev_levels_2d)
            score = abs(np.corrcoef(U[:, 0], V[:, 0])[0, 1])
            if score > best_2d_corr:
                best_2d_corr = score
                best_combination = (pc1, pc2)
                best_cca_2d = cca
                best_coords_2d = pca_subset
        except Exception:
            continue

    if best_combination is None:
        raise ValueError("Could not find valid PC combination for CCA")

    return best_combination, full_score, best_cca_2d, best_coords_2d


def run_cca_on_pca_from_adata(
    adata: AnnData,
    column: str,
    trajectory_col: str = "sev.level",
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
    trajectory_col : str
        Column name in adata.obs containing trajectory levels
    n_components : int
        Number of PC components to use (default: 2)
    verbose : bool
        Whether to print detailed information
        
    Returns:
    --------
    tuple: (pca_coords_full, sev_levels, samples, n_components_used)
    """
    # The sample embedding is normally in .uns (DataFrame); fall back to .obsm
    # (ndarray) so callers that only populate .obsm still work.
    if column in adata.uns:
        pca_coords = adata.uns[column]
    elif column in adata.obsm:
        pca_coords = adata.obsm[column]
    else:
        raise KeyError(f"'{column}' not found in adata.uns or adata.obsm. "
                       f"uns keys: {list(adata.uns.keys())}; obsm keys: {list(adata.obsm.keys())}")

    if trajectory_col not in adata.obs.columns:
        raise KeyError(f"'{trajectory_col}' column is missing in adata.obs. Available columns: {list(adata.obs.columns)}")
    
    if hasattr(pca_coords, 'iloc'):
        pca_coords_array = pca_coords.values
    else:
        pca_coords_array = pca_coords
    
    available_components = pca_coords_array.shape[1]
    if available_components < n_components:
        if verbose:
            print(f"Warning: Only {available_components} components available, using all of them.")
        n_components = available_components
    
    if n_components < 2:
        raise ValueError("Need at least 2 PC components for CCA analysis.")
    
    pca_coords_subset = pca_coords_array[:, :n_components]
    
    sev_levels = pd.to_numeric(adata.obs[trajectory_col], errors='coerce').values
    missing = np.isnan(sev_levels).sum()
    if missing > 0:
        if verbose:
            print(f"Warning: {missing} sample(s) missing trajectory level. Imputing with mean.")
        sev_levels[np.isnan(sev_levels)] = np.nanmean(sev_levels)
    
    if len(sev_levels) != pca_coords_subset.shape[0]:
        raise ValueError(f"Mismatch between PCA rows ({pca_coords_subset.shape[0]}) and severity levels ({len(sev_levels)}).")

    samples = adata.obs.index.values
    
    return pca_coords_subset, sev_levels, samples, n_components


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
    sev_levels = np.asarray(sev_levels).reshape(-1)

    if not np.issubdtype(sev_levels.dtype, np.number):
        try:
            sev_levels = sev_levels.astype(float)
        except (ValueError, TypeError):
            _, sev_codes = np.unique(sev_levels, return_inverse=True)
            sev_levels = sev_codes.astype(float)

    n_components = pca_coords_full.shape[1]

    if auto_select_best_2pc and n_components > 2:
        pc_indices_used, cca_score, cca_model, pca_coords_2d = find_best_2pc_combination(
            pca_coords_full, sev_levels
        )

    elif pc_indices is not None:
        if len(pc_indices) != 2:
            raise ValueError("pc_indices must contain exactly 2 indices")
        if max(pc_indices) >= n_components:
            raise ValueError(f"PC index {max(pc_indices)} exceeds available components ({n_components})")

        pc_indices_used = pc_indices
        pca_coords_2d = pca_coords_full[:, list(pc_indices)]

        sev_levels_2d = sev_levels.reshape(-1, 1)
        cca_model = CCA(n_components=1)
        cca_model.fit(pca_coords_2d, sev_levels_2d)
        U, V = cca_model.transform(pca_coords_2d, sev_levels_2d)
        cca_score = abs(np.corrcoef(U[:, 0], V[:, 0])[0, 1])

    else:
        pc_indices_used = (0, 1)
        pca_coords_2d = pca_coords_full[:, :2]

        sev_levels_2d = sev_levels.reshape(-1, 1)
        cca_model = CCA(n_components=1)
        cca_model.fit(pca_coords_2d, sev_levels_2d)
        U, V = cca_model.transform(pca_coords_2d, sev_levels_2d)
        cca_score = abs(np.corrcoef(U[:, 0], V[:, 0])[0, 1])

    fig, ax = plt.subplots(figsize=(10, 8))

    sev_min = np.min(sev_levels)
    sev_max = np.max(sev_levels)
    sev_range = sev_max - sev_min

    if sev_range < 1e-16:
        norm_sev = np.zeros_like(sev_levels, dtype=float)
    else:
        norm_sev = (sev_levels - sev_min) / (sev_range + 1e-16)

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

    if sample_labels is not None:
        for i, label in enumerate(sample_labels):
            ax.text(
                pca_coords_2d[i, 0],
                pca_coords_2d[i, 1],
                str(label),
                fontsize=8,
                alpha=0.7,
            )

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

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()

    if create_contribution_plot:
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

        if output_path:
            base, ext = os.path.splitext(output_path)
            contribution_path = f"{base}_contributions{ext}"
            plt.savefig(contribution_path, dpi=300, bbox_inches='tight')
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
    trajectory_col: str = "sev.level",
    n_components: int = 2,
    auto_select_best_2pc: bool = True,
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
    trajectory_col : str
        Column name for trajectory levels
    n_components : int
        Number of PC components to use (default: 2)
    auto_select_best_2pc : bool
        If True, automatically select best 2-PC combination for visualization
    verbose : bool
        Whether to print detailed information
    show_sample_labels : bool
        Whether to show sample labels on plots
        
    Returns:
    --------
    tuple: (proportion_score, expression_score, proportion_pseudotime, expression_pseudotime)
    """
    if output_dir:
        output_dir = os.path.join(output_dir, 'CCA')
        os.makedirs(output_dir, exist_ok=True)

    if "X_DR_sample" not in adata.uns and "X_DR_sample" not in adata.obsm:
        raise KeyError(
            "CCA_Call: 'X_DR_sample' not found in adata.uns or adata.obsm. "
            "Run compute_sample_embedding first (it populates both)."
        )

    dr_keys = ["X_DR_sample"]
    paths = {
        k: (os.path.join(output_dir, f"pca_{n_components}d_cca_{k.replace('X_DR_', '')}.pdf")
            if output_dir else None)
        for k in dr_keys
    }

    results = {}
    sample_dicts = {}
    pc_info = {}

    for key in dr_keys:
        try:
            pca_coords_full, sev_levels, samples, n_components_used = run_cca_on_pca_from_adata(
                adata=adata,
                column=key,
                trajectory_col=trajectory_col,
                n_components=n_components,
                verbose=verbose,
            )

            cca_score, pc_indices_used, cca_model = plot_cca_on_2d_pca(
                pca_coords_full=pca_coords_full,
                sev_levels=sev_levels,
                auto_select_best_2pc=auto_select_best_2pc,
                pc_indices=None,
                output_path=paths[key],
                sample_labels=samples if show_sample_labels else None,
                title_suffix=key.replace("X_DR_", "").title(),
                verbose=verbose,
            )

            results[key] = cca_score
            pc_info[key] = pc_indices_used

            pca_coords_2d = pca_coords_full[:, list(pc_indices_used)]
            sample_dicts[key] = assign_pseudotime_from_cca(
                pca_coords_2d=pca_coords_2d,
                cca=cca_model,
                sample_labels=samples,
            )

        except Exception as e:
            if verbose:
                print(f"Error processing {key}: {str(e)}")
            results[key] = np.nan
            sample_dicts[key] = {}
            pc_info[key] = None

    if output_dir:
        for key in dr_keys:
            if sample_dicts[key]:
                pseudotime_df = pd.DataFrame([
                    {'sample': sample_id, 'pseudotime': pseudotime_value}
                    for sample_id, pseudotime_value in sample_dicts[key].items()
                ])
                data_type = key.replace("X_DR_", "")
                csv_filename = f"pseudotime_{data_type}.csv"
                pseudotime_df.to_csv(os.path.join(output_dir, csv_filename), index=False)

    if verbose:
        print("\nCCA Analysis Summary:")
        for key in dr_keys:
            score = results.get(key, np.nan)
            pc_indices = pc_info.get(key, None)
            data_type = key.replace("X_DR_", "").title()
            if pc_indices:
                print(f"  {data_type}: score={score:.4f} (PC{pc_indices[0]+1} + PC{pc_indices[1]+1})")
            else:
                print(f"  {data_type}: Failed")

    # CCA_Call still returns a 4-tuple shape for back-compat with the wrapper;
    # in the single-key pipeline the two slots collapse to the same sample DR.
    score = results.get("X_DR_sample", np.nan)
    ptime = sample_dicts.get("X_DR_sample", {})
    return (score, score, ptime, ptime)