import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from anndata import AnnData
from sklearn.cross_decomposition import CCA
import time


def run_cca_on_2d_pca_from_adata(
    adata: AnnData,
    column: str,
    sev_col: str = "sev.level"
):
    pca_coords = adata.uns[column]
    if pca_coords.shape[1] < 2:
        raise ValueError("PCA must have at least 2 components for CCA.")

    pca_coords_2d = pca_coords.iloc[:, :2].values if hasattr(pca_coords, 'iloc') else pca_coords[:, :2]

    sev_levels = pd.to_numeric(adata.obs[sev_col], errors='coerce').values
    missing = np.isnan(sev_levels).sum()
    if missing > 0:
        print(f"Warning: {missing} sample(s) missing severity level. Imputing with mean.")
        sev_levels[np.isnan(sev_levels)] = np.nanmean(sev_levels)
    
    if len(sev_levels) != pca_coords_2d.shape[0]:
        raise ValueError(f"Mismatch between PCA rows ({pca_coords_2d.shape[0]}) and severity levels ({len(sev_levels)}).")

    sev_levels_2d = sev_levels.reshape(-1, 1)
    samples = adata.obs.index.values

    cca = CCA(n_components=1)
    cca.fit(pca_coords_2d, sev_levels_2d)
    U, V = cca.transform(pca_coords_2d, sev_levels_2d)
    first_component_score = np.corrcoef(U[:, 0], V[:, 0])[0, 1]

    print(f"\nThe CCA score for {column} is {first_component_score:.4f}\n")
    return pca_coords_2d, sev_levels, cca, first_component_score, samples


def plot_cca_on_2d_pca(
    pca_coords_2d: np.ndarray,
    sev_levels: np.ndarray,
    cca: CCA,
    output_path: str = None,
    sample_labels=None,
    title_suffix: str = ""
):
    plt.figure(figsize=(8, 6))

    norm_sev = (sev_levels - np.min(sev_levels)) / (np.max(sev_levels) - np.min(sev_levels) + 1e-16)

    sc = plt.scatter(
        pca_coords_2d[:, 0],
        pca_coords_2d[:, 1],
        c=norm_sev,
        cmap='viridis_r',
        edgecolors='k',
        alpha=0.8,
        s=60
    )
    cbar = plt.colorbar(sc, label='Severity Level')

    dx, dy = cca.x_weights_[:, 0]
    scale = 0.5 * max(np.ptp(pca_coords_2d[:, 0]), np.ptp(pca_coords_2d[:, 1]))
    x_start, x_end = -scale * dx, scale * dx
    y_start, y_end = -scale * dy, scale * dy

    plt.plot([x_start, x_end], [y_start, y_end],
             linestyle="--", color="red", linewidth=2, label="CCA Direction", alpha=0.8)

    if sample_labels is not None:
        for i, label in enumerate(sample_labels):
            plt.text(pca_coords_2d[i, 0], pca_coords_2d[i, 1], 
                    str(label), fontsize=8, alpha=0.7)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    title = f"2D PCA with CCA Direction{' - ' + title_suffix if title_suffix else ''}"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved CCA plot to: {output_path}")
    else:
        plt.show()


def assign_pseudotime_from_cca(
    pca_coords_2d: np.ndarray, 
    cca: CCA, 
    sample_labels: np.ndarray,
    scale_to_unit: bool = True
) -> dict:
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
    ptime: bool = False,
    verbose: bool = False,
    show_sample_labels: bool = False
):
    start_time = time.time() if verbose else None
    
    if output_dir:
        output_dir = os.path.join(output_dir, 'CCA')
        os.makedirs(output_dir, exist_ok=True)

    paths = {
        "X_pca_proportion": os.path.join(output_dir, "pca_2d_cca_proportion.pdf") if output_dir else None,
        "X_pca_expression": os.path.join(output_dir, "pca_2d_cca_expression.pdf") if output_dir else None
    }

    results = {}
    sample_dicts = {}

    for key in ["X_pca_proportion", "X_pca_expression"]:
        if verbose:
            print(f"Processing {key}...")
            
        try:
            (pca_coords_2d, 
             sev_levels, 
             cca_model, 
             score, 
             samples) = run_cca_on_2d_pca_from_adata(
                adata=adata,
                column=key,
                sev_col=sev_col
            )

            plot_cca_on_2d_pca(
                pca_coords_2d=pca_coords_2d,
                sev_levels=sev_levels,
                cca=cca_model,
                output_path=paths[key],
                sample_labels=samples if show_sample_labels else None,
                title_suffix=key.replace("X_pca_", "").title()
            )
            
            results[key] = score

            sample_dicts[key] = assign_pseudotime_from_cca(
                pca_coords_2d=pca_coords_2d, 
                cca=cca_model, 
                sample_labels=samples
            )
            
        except Exception as e:
            print(f"Error processing {key}: {str(e)}")
            results[key] = np.nan
            sample_dicts[key] = {}

    if verbose:
        print("CCA analysis completed.")
        if start_time:
            print(f"\n[CCA] Total runtime: {time.time() - start_time:.2f} seconds\n")

    return (results.get("X_pca_proportion", np.nan), 
            results.get("X_pca_expression", np.nan), 
            sample_dicts.get("X_pca_proportion", {}), 
            sample_dicts.get("X_pca_expression", {}))