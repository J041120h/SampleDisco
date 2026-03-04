#!/usr/bin/env python3
"""
Optimal cell resolution finder using CCA correlation optimization - Linux/GPU version.
"""

import os
import sys
import time
import shutil
from typing import Optional, Union, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from anndata import AnnData
from sklearn.cross_decomposition import CCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preparation.Cell_type_linux import cell_types_linux
from sample_embedding.calculate_sample_embedding import calculate_sample_embedding
from sample_trajectory.CCA import run_cca_on_pca_from_adata, plot_cca_on_2d_pca
from sample_trajectory.CCA_test import (
    generate_null_distribution,
    generate_corrected_null_distribution,
    compute_corrected_pvalues_rna,
    create_comprehensive_summary_rna,
)


def find_optimal_cell_resolution_linux(
    adata_cell: AnnData,
    adata_sample: AnnData,
    output_dir: str,
    column: str,
    trajectory_col: str = "sev.level",
    batch_col: Optional[Union[str, List[str]]] = "batch",
    sample_col: str = "sample",
    cell_embedding_column: str = "X_pca",
    cell_embedding_num_pcs: int = 20,
    n_hvg_features: int = 2000,
    sample_embedding_dimension: int = 30,
    n_cca_pcs: int = 10,
    compute_corrected_pvalues: bool = True,
    preserve_cols_in_sample_embedding: Optional[Union[str, List[str]]] = None,
    verbose: bool = True,
) -> Tuple[float, pd.DataFrame]:
    """
    Find optimal clustering resolution by maximizing CCA correlation between
    dimension reduction and trajectory levels - Linux/GPU version.

    Parameters
    ----------
    adata_cell : AnnData
        Cell-level AnnData object
    adata_sample : AnnData
        Sample-level AnnData object
    output_dir : str
        Output directory for results
    column : str
        Column name in adata.uns for dimension reduction results
    trajectory_col : str
        Column name for trajectory/severity levels
    batch_col : str or list, optional
        Column name(s) for batch information
    sample_col : str
        Column name for sample identifiers
    cell_embedding_column : str
        Representation to use for clustering
    cell_embedding_num_pcs : int
        Number of PCs for clustering
    n_hvg_features : int
        Number of highly variable genes for pseudobulk
    sample_embedding_dimension : int
        Number of dimension reduction components for sample embedding
    n_cca_pcs : int
        Number of PCs for CCA analysis and null distribution
    compute_corrected_pvalues : bool
        Whether to compute corrected p-values
    preserve_cols_in_sample_embedding : str or list, optional
        Columns to preserve during batch correction in sample embedding
    verbose : bool
        Whether to print verbose output

    Returns
    -------
    Tuple[float, pd.DataFrame]
        Optimal resolution and results dataframe
    """
    if verbose:
        print("Using GPU/Linux version of optimal resolution finder")

    # Setup directories
    dr_type = column.replace("X_DR_", "") if column.startswith("X_DR_") else column
    main_output_dir = os.path.join(output_dir, f"RNA_resolution_optimization_{dr_type}")
    resolutions_dir = os.path.join(main_output_dir, "resolutions")
    summary_dir = os.path.join(main_output_dir, "summary")

    for dir_path in [main_output_dir, resolutions_dir, summary_dir]:
        os.makedirs(dir_path, exist_ok=True)

    if verbose:
        print(f"Starting RNA-seq resolution optimization for {column}...")
        print(f"Using representation: {cell_embedding_column} with {cell_embedding_num_pcs} components")
        print(f"Using {n_cca_pcs} PCs for CCA analysis")
        if compute_corrected_pvalues:
            print("Will compute corrected p-values with default simulations")

    # Storage for results
    all_results: List[Dict[str, Any]] = []
    all_null_results: List[Dict[str, Any]] = []

    def process_resolution(resolution: float, search_pass: str) -> Tuple[Dict, Dict]:
        """Process a single resolution and return results."""
        if verbose:
            print(f"\n{'='*50}\nTesting resolution: {resolution:.3f}\n{'='*50}")

        resolution_dir = os.path.join(resolutions_dir, f"resolution_{resolution:.3f}")
        os.makedirs(resolution_dir, exist_ok=True)

        result = {
            "resolution": resolution,
            "cca_score": np.nan,
            "p_value": np.nan,
            "corrected_pvalue": np.nan,
            "pass": search_pass,
            "n_clusters": 0,
            "n_samples": 0,
            "n_pcs_used": n_cca_pcs,
            "pc_indices_used": None,
        }
        null_result = {"resolution": resolution, "null_scores": None}

        try:
            # Clean up previous cell type assignments
            for adata in [adata_cell, adata_sample]:
                if "cell_type" in adata.obs.columns:
                    adata.obs.drop(columns=["cell_type"], inplace=True, errors="ignore")

            # Perform clustering using Linux/GPU version
            updated_cell, updated_sample = cell_types_linux(
                anndata_cell=adata_cell,
                anndata_sample=adata_sample,
                cell_type_column="cell_type",
                existing_cell_types=False,
                umap=False,
                save=False,
                output_dir=resolution_dir,
                leiden_cluster_resolution=resolution,
                cell_embedding_column=cell_embedding_column,
                cell_embedding_num_PCs=cell_embedding_num_pcs,
                verbose=False,
                umap_plots=False,
            )

            # Update references
            adata_cell.obs = updated_cell.obs
            adata_sample.obs = updated_sample.obs

            n_clusters = adata_sample.obs["cell_type"].nunique()
            result["n_clusters"] = n_clusters
            if verbose:
                print(f"Number of clusters: {n_clusters}")

            # Calculate sample embedding using the wrapper with GPU
            pseudobulk_dict, pseudobulk_adata = calculate_sample_embedding(
                adata=adata_sample,
                sample_col=sample_col,
                celltype_col="cell_type",
                batch_col=batch_col,
                output_dir=resolution_dir,
                sample_hvg_number=n_hvg_features,
                n_expression_components=sample_embedding_dimension,
                n_proportion_components=sample_embedding_dimension,
                harmony_for_proportion=True,
                preserve_cols_in_sample_embedding=preserve_cols_in_sample_embedding,
                use_gpu=True,
                atac=False,
                save=False,
                verbose=False,
            )

            result["n_samples"] = len(pseudobulk_adata)
            pseudobulk_adata.write_h5ad(os.path.join(resolution_dir, "pseudobulk_sample.h5ad"))

            if column not in pseudobulk_adata.uns:
                print(f"Warning: {column} not found in pseudobulk_adata.uns. Skipping.")
                return result, null_result

            # Run CCA analysis
            pca_coords, sev_levels, _, _ = run_cca_on_pca_from_adata(
                adata=pseudobulk_adata,
                column=column,
                trajectory_col=trajectory_col,
                n_components=n_cca_pcs,
                verbose=False,
            )

            # Calculate CCA score
            n_pcs_actual = min(n_cca_pcs, pca_coords.shape[1])
            pca_subset = pca_coords[:, :n_pcs_actual]

            cca = CCA(n_components=1)
            cca.fit(pca_subset, sev_levels.reshape(-1, 1))
            U, V = cca.transform(pca_subset, sev_levels.reshape(-1, 1))
            cca_score = np.corrcoef(U[:, 0], V[:, 0])[0, 1]

            result.update({"cca_score": cca_score, "n_pcs_used": n_pcs_actual})
            if verbose:
                print(f"CCA Score = {cca_score:.4f} (using {n_pcs_actual} PCs)")

            # Create CCA visualization
            try:
                plot_path = os.path.join(resolution_dir, f"cca_plot_res_{resolution:.3f}.png")
                _, pc_indices, _ = plot_cca_on_2d_pca(
                    pca_coords_full=pca_coords,
                    sev_levels=sev_levels,
                    auto_select_best_2pc=True,
                    output_path=plot_path,
                    title_suffix=f"Resolution {resolution:.3f}",
                    verbose=verbose,
                )
                result["pc_indices_used"] = pc_indices
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to create CCA visualization: {e}")

            # Generate null distribution
            if compute_corrected_pvalues:
                try:
                    null_dist = generate_null_distribution(
                        pseudobulk_adata=pseudobulk_adata,
                        column=column,
                        trajectory_col=trajectory_col,
                        n_pcs=n_cca_pcs,
                        save_path=os.path.join(resolution_dir, f"null_dist_{resolution:.3f}.npy"),
                        verbose=False,
                    )
                    null_result["null_scores"] = null_dist
                    result["p_value"] = np.mean(null_dist >= cca_score)
                    if verbose:
                        print(f"Standard p-value = {result['p_value']:.4f}")
                except Exception as e:
                    print(f"Warning: Failed to generate null distribution: {e}")

        except Exception as e:
            print(f"Error at resolution {resolution:.3f}: {e}")

        return result, null_result

    # === COARSE SEARCH ===
    if verbose:
        print("\n" + "=" * 60)
        print("FIRST PASS: Coarse Search (0.1 to 1.0)")
        print("=" * 60)

    for resolution in np.arange(0.1, 1.01, 0.1):
        result, null_result = process_resolution(resolution, "coarse")
        all_results.append(result)
        all_null_results.append(null_result)

    # Find best coarse resolution
    valid_coarse = [r for r in all_results if not np.isnan(r["cca_score"])]
    if not valid_coarse:
        raise ValueError("No valid CCA scores in coarse search. Check data and parameters.")

    best_coarse = max(valid_coarse, key=lambda x: x["cca_score"])
    if verbose:
        print(f"\nBest coarse resolution: {best_coarse['resolution']:.2f} (CCA: {best_coarse['cca_score']:.4f})")

    # === FINE SEARCH ===
    if verbose:
        print("\n" + "=" * 60)
        print("SECOND PASS: Fine-tuned Search")
        print("=" * 60)

    fine_start = max(0.01, best_coarse["resolution"] - 0.02)
    fine_end = min(1.00, best_coarse["resolution"] + 0.02)
    tested_resolutions = {r["resolution"] for r in all_results}

    for resolution in np.arange(fine_start, fine_end + 0.001, 0.01):
        resolution = round(resolution, 3)
        if any(abs(resolution - tested) < 0.001 for tested in tested_resolutions):
            continue

        result, null_result = process_resolution(resolution, "fine")
        all_results.append(result)
        all_null_results.append(null_result)

    # Create results dataframe
    df_results = pd.DataFrame(all_results).sort_values("resolution")

    # === CORRECTED P-VALUES ===
    if compute_corrected_pvalues:
        valid_nulls = [r for r in all_null_results if r["null_scores"] is not None]

        if valid_nulls:
            if verbose:
                print("\n" + "=" * 60)
                print("GENERATING CORRECTED NULL DISTRIBUTION")
                print("=" * 60)

            corrected_null = generate_corrected_null_distribution(
                all_resolution_results=valid_nulls,
            )

            # Save and visualize corrected null distribution
            corrected_null_dir = os.path.join(main_output_dir, "corrected_null")
            os.makedirs(corrected_null_dir, exist_ok=True)
            np.save(os.path.join(corrected_null_dir, f"corrected_null_{dr_type}.npy"), corrected_null)

            df_results = compute_corrected_pvalues_rna(
                df_results=df_results,
                corrected_null_distribution=corrected_null,
                output_dir=main_output_dir,
                column=column,
            )

            # Plot corrected null distribution
            _plot_corrected_null_distribution(corrected_null, column, corrected_null_dir)
        else:
            print("Warning: No valid null distributions, skipping corrected p-values")
            compute_corrected_pvalues = False

    # === FINAL RESULTS ===
    valid_results = df_results[df_results["cca_score"].notna()]
    if valid_results.empty:
        raise ValueError("No valid results obtained.")

    best_idx = valid_results["cca_score"].idxmax()
    best_row = valid_results.loc[best_idx]

    optimal_resolution = best_row["resolution"]
    optimal_score = best_row["cca_score"]

    if verbose:
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Optimal resolution: {optimal_resolution:.3f}")
        print(f"Best CCA score: {optimal_score:.4f}")
        print(f"Number of clusters: {best_row['n_clusters']}")
        if best_row.get("pc_indices_used") is not None:
            pcs = best_row["pc_indices_used"]
            print(f"Best visualization PCs: PC{pcs[0]+1} + PC{pcs[1]+1}")
        if not np.isnan(best_row.get("p_value", np.nan)):
            print(f"Standard p-value: {best_row['p_value']:.4f}")
        if compute_corrected_pvalues and not np.isnan(best_row.get("corrected_pvalue", np.nan)):
            print(f"Corrected p-value: {best_row['corrected_pvalue']:.4f}")

    # Save outputs
    create_comprehensive_summary_rna(
        df_results=df_results,
        best_resolution=optimal_resolution,
        column=column,
        output_dir=summary_dir,
        has_corrected_pvalues=compute_corrected_pvalues,
    )

    df_results.to_csv(os.path.join(summary_dir, f"all_resolution_results_{dr_type}.csv"), index=False)

    # Copy optimal pseudobulk to summary
    optimal_pb_src = os.path.join(resolutions_dir, f"resolution_{optimal_resolution:.3f}", "pseudobulk_sample.h5ad")
    if os.path.exists(optimal_pb_src):
        shutil.copy2(optimal_pb_src, os.path.join(summary_dir, "optimal.h5ad"))

    # Write final summary
    _write_final_summary_linux(
        main_output_dir, summary_dir, column, cell_embedding_column, cell_embedding_num_pcs,
        sample_embedding_dimension, n_cca_pcs, n_hvg_features,
        best_row, valid_results, compute_corrected_pvalues
    )

    return optimal_resolution, df_results


def _plot_corrected_null_distribution(corrected_null: np.ndarray, column: str, output_dir: str) -> None:
    """Plot and save corrected null distribution."""
    plt.figure(figsize=(10, 6))
    plt.hist(corrected_null, bins=50, alpha=0.7, color="lightblue", density=True, edgecolor="black")
    plt.xlabel("Maximum CCA Score (across resolutions)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title(f"Corrected Null Distribution\n{column} - Accounts for Resolution Selection", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)

    stats_text = (
        f"Mean: {np.mean(corrected_null):.4f}\n"
        f"Std: {np.std(corrected_null):.4f}\n"
        f"95th percentile: {np.percentile(corrected_null, 95):.4f}"
    )
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.savefig(os.path.join(output_dir, "corrected_null_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()


def _write_final_summary_linux(
    main_output_dir: str, summary_dir: str, column: str,
    cell_embedding_column: str, cell_embedding_num_pcs: int,
    sample_embedding_dimension: int, n_cca_pcs: int, n_hvg_features: int,
    best_row: pd.Series, valid_results: pd.DataFrame, compute_corrected_pvalues: bool
) -> None:
    """Write final summary report to file (Linux/GPU version)."""
    summary_path = os.path.join(main_output_dir, "FINAL_SUMMARY.txt")

    with open(summary_path, "w") as f:
        f.write("RNA-SEQ RESOLUTION OPTIMIZATION FINAL SUMMARY (GPU/Linux Version)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Analysis completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("OPTIMIZATION PARAMETERS:\n")
        f.write(f"  - Column analyzed: {column}\n")
        f.write(f"  - Cell embedding: {cell_embedding_column}\n")
        f.write(f"  - Cell embedding PCs: {cell_embedding_num_pcs}\n")
        f.write(f"  - Sample embedding dimension: {sample_embedding_dimension}\n")
        f.write(f"  - CCA PCs: {n_cca_pcs}\n")
        f.write(f"  - HVG features: {n_hvg_features}\n\n")

        f.write("RESULTS:\n")
        f.write(f"  - Optimal resolution: {best_row['resolution']:.3f}\n")
        f.write(f"  - Best CCA score: {best_row['cca_score']:.4f}\n")
        f.write(f"  - Number of clusters: {best_row['n_clusters']}\n")

        if best_row.get("pc_indices_used") is not None:
            pcs = best_row["pc_indices_used"]
            f.write(f"  - Visualization PCs: PC{pcs[0]+1} + PC{pcs[1]+1}\n")
        if not np.isnan(best_row.get("p_value", np.nan)):
            f.write(f"  - Standard p-value: {best_row['p_value']:.4f}\n")
        if compute_corrected_pvalues and not np.isnan(best_row.get("corrected_pvalue", np.nan)):
            f.write(f"  - Corrected p-value: {best_row['corrected_pvalue']:.4f}\n")

        n_coarse = len(valid_results[valid_results["pass"] == "coarse"])
        n_fine = len(valid_results[valid_results["pass"] == "fine"])
        f.write(f"\nResolutions tested: {len(valid_results)} (coarse: {n_coarse}, fine: {n_fine})\n")

        f.write("\nOUTPUT LOCATIONS:\n")
        f.write(f"  - Main: {main_output_dir}\n")
        f.write(f"  - Summary: {summary_dir}\n")

    print(f"Final summary saved to: {summary_path}")