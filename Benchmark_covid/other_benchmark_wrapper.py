import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from ANOVA import run_trajectory_anova_analysis
from batch_removal_test import evaluate_batch_removal
from embedding_effective import evaluate_ari_clustering
from spearman_test import run_trajectory_analysis
from customized_benchmark import benchmark_pseudotime_embeddings_custom  # <-- ADDED

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BenchmarkWrapper:
    """
    A comprehensive wrapper for running various benchmark analyses with EXPLICIT paths.

    Parameters
    ----------
    meta_csv_path : str
        Path to metadata CSV file (required for all benchmarks)
    pseudotime_csv_path : str, optional
        Path to pseudotime CSV file (required for trajectory_* benchmarks)
    embedding_csv_path : str, optional
        Path to embedding/coordinates CSV file (required for ARI and batch-removal benchmarks)
    mode : str, optional
        Label used ONLY for naming the output directory scope (default: 'expression')
    output_base_dir : str, optional
        Base directory for all outputs. If None, defaults to parent of the meta CSV file.
    """

    def __init__(
        self,
        meta_csv_path: str,
        pseudotime_csv_path: Optional[str] = None,
        embedding_csv_path: Optional[str] = None,
        mode: str = "expression",
        output_base_dir: Optional[str] = None,
    ):
        # Store and validate core inputs
        self.meta_csv_path = Path(meta_csv_path).resolve()
        self.pseudotime_csv_path = Path(pseudotime_csv_path).resolve() if pseudotime_csv_path else None
        self.embedding_csv_path = Path(embedding_csv_path).resolve() if embedding_csv_path else None
        self.mode = mode.lower()

        if not self.meta_csv_path.exists() or not self.meta_csv_path.is_file():
            raise FileNotFoundError(f"Metadata CSV does not exist or is not a file: {self.meta_csv_path}")

        # Output base directory strategy
        if output_base_dir is None:
            # Default to the parent of the meta CSV so there's always a stable place to write
            self.output_base_dir = self.meta_csv_path.parent
        else:
            self.output_base_dir = Path(output_base_dir).resolve()

        # Mode-scoped output directory
        self.mode_output_dir = self.output_base_dir / f"benchmark_results_{self.mode}"
        self.mode_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initialized BenchmarkWrapper (explicit paths) with:")
        logger.info(f"  Meta CSV:          {self.meta_csv_path}")
        logger.info(f"  Pseudotime CSV:    {self.pseudotime_csv_path if self.pseudotime_csv_path else '(not provided)'}")
        logger.info(f"  Embedding CSV:     {self.embedding_csv_path if self.embedding_csv_path else '(not provided)'}")
        logger.info(f"  Mode label:        {self.mode}")
        logger.info(f"  Output base dir:   {self.output_base_dir}")
        logger.info(f"  Mode output dir:   {self.mode_output_dir}")

    # ------------------------- helpers -------------------------

    def _create_output_dir(self, benchmark_name: str) -> Path:
        out = self.mode_output_dir / benchmark_name
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _check_file_exists(self, file_path: Optional[Path], file_description: str) -> bool:
        """
        Check if a file exists and log helpful diagnostics if not.
        """
        if file_path is None:
            logger.error(f"ERROR: {file_description} was not provided.")
            return False

        if not file_path.exists():
            logger.error(f"ERROR: {file_description} not found!")
            logger.error(f"  Expected path: {file_path}")
            parent = file_path.parent
            logger.error(f"  Parent directory exists: {parent.exists()}")
            if parent.exists():
                logger.error("  Contents of parent directory:")
                try:
                    for item in parent.iterdir():
                        logger.error(f"    - {item.name}")
                except Exception as e:
                    logger.error(f"    Could not list directory contents: {e}")
            else:
                logger.error(f"  Parent directory does not exist: {parent}")
            return False

        if not file_path.is_file():
            logger.error(f"ERROR: {file_description} path is not a file: {file_path}")
            return False

        return True

    def _save_summary_csv(self, results: Dict[str, Dict[str, Any]], summary_csv_path: Optional[Path] = None) -> None:
        """Save a summary of all benchmark results to a CSV file."""
        if not summary_csv_path:
            summary_csv_path = self.mode_output_dir / "benchmark_summary.csv"

        # Initialize the CSV if it doesn't exist
        if not summary_csv_path.exists():
            header = [
                "Benchmark", "ARI", "iLISI_norm", "ASW-batch", "Mean Per-Su NN",
                "Spearman Correlation", "Batch Effect Size", "Severity Effect Size", "Interaction Effect Size"
            ]
            summary_df = pd.DataFrame(columns=header)
            summary_df.to_csv(summary_csv_path, index=False)
            logger.info(f"Created new summary CSV at: {summary_csv_path}")
        else:
            summary_df = pd.read_csv(summary_csv_path)

        # Add the results as a new row for each benchmark
        for benchmark_name, result in results.items():
            row = {
                "Benchmark": benchmark_name,
                "ARI": result.get("ari", None),
                "iLISI_norm": result.get("ilisi_norm", None),
                "ASW-batch": result.get("asw_batch", None),
                "Mean Per-Su NN": result.get("mean_per_su_nn", None),
                "Spearman Correlation": result.get("spearman_corr", None),
                "Batch Effect Size": result.get("batch_effect_size", None),
                "Severity Effect Size": result.get("severity_effect_size", None),
                "Interaction Effect Size": result.get("interaction_effect_size", None)
            }
            summary_df = summary_df.append(row, ignore_index=True)

        # Save the updated CSV file
        summary_df.to_csv(summary_csv_path, index=False)
        logger.info(f"Updated summary CSV at: {summary_csv_path}")


    def run_embedding_visualization(
        self,
        n_components: int = 2,
        figsize: tuple = (12, 5),
        dpi: int = 300,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Visualize embeddings colored by sev.level (continuous) and batch (categorical).

        Requires:
            - meta_csv_path (must include columns: 'sev.level', 'batch', and typically 'sample')
            - embedding_csv_path (rows indexed by sample IDs matching meta['sample'] or meta index)
        """
        logger.info("Running Embedding Visualization...")
        output_dir = self._create_output_dir("embedding_visualization")

        if not self._check_file_exists(self.embedding_csv_path, "Embedding/coordinates CSV file"):
            return {"status": "error", "message": "Missing or invalid embedding CSV path."}

        try:
            # -------------------- LOAD --------------------
            logger.info(f"Loading metadata from: {self.meta_csv_path}")
            meta_df = pd.read_csv(self.meta_csv_path)
            print(f"[DEBUG] Metadata loaded: shape={meta_df.shape}, columns={meta_df.columns.tolist()}")

            logger.info(f"Loading embeddings from: {self.embedding_csv_path}")
            embedding_df = pd.read_csv(self.embedding_csv_path, index_col=0)
            print(f"[DEBUG] Embedding matrix shape: {embedding_df.shape}")
            print(f"[DEBUG] First 3 embedding indices: {embedding_df.index[:3].tolist()}")

            # -------------------- REQUIREMENTS --------------------
            required_cols = ['sev.level', 'batch']
            missing_cols = [c for c in required_cols if c not in meta_df.columns]
            if missing_cols:
                logger.warning(f"Missing columns in metadata: {missing_cols}")
                print(f"[DEBUG] ERROR: Missing required columns in metadata: {missing_cols}")
                return {"status": "error", "message": f"Missing required columns in metadata: {missing_cols}"}

            # -------------------- ALIGN BY SAMPLE ID --------------------
            # Prefer a 'sample' column if present; otherwise assume meta_df is already indexed by IDs
            if 'sample' in meta_df.columns:
                print("[DEBUG] Setting meta_df index to 'sample'")
                meta_df = meta_df.set_index('sample')
            else:
                print("[DEBUG][WARN] 'sample' column not found; using meta_df.index for alignment")

            # Report overlaps for auditing
            common_ids = meta_df.index.intersection(embedding_df.index)
            only_meta = meta_df.index.difference(embedding_df.index)
            only_emb = embedding_df.index.difference(meta_df.index)
            print(f"[DEBUG] Overlap report: common={len(common_ids)}, meta_only={len(only_meta)}, embed_only={len(only_emb)}")
            if len(only_meta) > 0:
                print(f"[DEBUG] Example meta_only IDs: {list(only_meta[:5])}")
            if len(only_emb) > 0:
                print(f"[DEBUG] Example embed_only IDs: {list(only_emb[:5])}")

            if len(common_ids) == 0:
                err = ("No overlapping sample IDs between metadata and embedding! "
                    "Ensure meta_df['sample'] (or meta index) matches embedding_df.index.")
                print(f"[DEBUG] ERROR: {err}")
                raise ValueError(err)

            # Subset BOTH to common IDs and order meta to match embedding
            meta_before, emb_before = meta_df.shape[0], embedding_df.shape[0]
            embedding_df = embedding_df.loc[common_ids]
            meta_df = meta_df.loc[embedding_df.index]
            print(f"[DEBUG] After alignment: meta_df={meta_df.shape}, embedding_df={embedding_df.shape}")
            print(f"[DEBUG] Dropped rows -> meta: {meta_before - meta_df.shape[0]}, embed: {emb_before - embedding_df.shape[0]}")

            # -------------------- PCA --------------------
            logger.info(f"Performing PCA to {n_components} components...")
            print("[DEBUG] Running PCA on aligned embedding_df...")
            pca = PCA(n_components=n_components)
            embedding_2d = pca.fit_transform(embedding_df)
            variance_explained = pca.explained_variance_ratio_
            print(f"[DEBUG] PCA done. Explained variance ratio: {variance_explained}")

            logger.info(f"Variance explained by PC1: {variance_explained[0]:.2%}")
            if n_components >= 2:
                logger.info(f"Variance explained by PC2: {variance_explained[1]:.2%}")

            # -------------------- VISUALIZATION --------------------
            print("[DEBUG] Creating figure...")
            fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

            # Left panel: sev.level (continuous)
            print("[DEBUG] Plotting sev.level panel...")
            sev_levels = pd.to_numeric(meta_df['sev.level'], errors='coerce')
            sev_min, sev_max = sev_levels.min(), sev_levels.max()
            sev_range = (sev_max - sev_min) if pd.notnull(sev_max) and pd.notnull(sev_min) else 0.0
            if sev_range == 0.0:
                print("[DEBUG][WARN] sev.level has zero/invalid range; coloring will be constant.")
            sev_norm = (sev_levels - (sev_min if pd.notnull(sev_min) else 0.0)) / (sev_range + 1e-16)
            print(f"[DEBUG] sev.level raw range: min={sev_min}, max={sev_max}")
            print(f"[DEBUG] sev.level normalized range: {sev_norm.min():.3f}–{sev_norm.max():.3f}")

            ax1 = axes[0]
            scatter1 = ax1.scatter(
                embedding_2d[:, 0],
                embedding_2d[:, 1],
                c=sev_norm,
                cmap='viridis',
                edgecolors='black',
                alpha=0.8,
                s=100,
                linewidths=0.5
            )
            ax1.set_xlabel(f'PC1 ({variance_explained[0]:.1%})', fontsize=12, fontweight='bold')
            ax1.set_ylabel(f'PC2 ({variance_explained[1]:.1%})', fontsize=12, fontweight='bold' if n_components >= 2 else None)
            ax1.set_title('Embeddings colored by sev.level', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, linestyle='--')

            cbar1 = plt.colorbar(scatter1, ax=ax1)
            cbar1.set_label('Normalized sev.level', fontsize=10)
            cbar1.ax.tick_params(labelsize=10)

            # Right panel: batch (categorical)
            print("[DEBUG] Plotting batch panel...")
            unique_batches = sorted(meta_df['batch'].astype(str).unique().tolist())
            print(f"[DEBUG] Unique batches ({len(unique_batches)}): {unique_batches[:10]}{'...' if len(unique_batches)>10 else ''}")
            batch_to_num = {b: i for i, b in enumerate(unique_batches)}
            batch_colors = [batch_to_num[str(b)] for b in meta_df['batch']]

            ax2 = axes[1]
            scatter2 = ax2.scatter(
                embedding_2d[:, 0],
                embedding_2d[:, 1],
                c=batch_colors,
                cmap='tab10',
                edgecolors='black',
                alpha=0.8,
                s=100,
                linewidths=0.5
            )
            ax2.set_xlabel(f'PC1 ({variance_explained[0]:.1%})', fontsize=12, fontweight='bold')
            ax2.set_ylabel(f'PC2 ({variance_explained[1]:.1%})', fontsize=12, fontweight='bold' if n_components >= 2 else None)
            ax2.set_title('Embeddings colored by batch', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')

            # Colorbar with batch labels
            cbar2 = plt.colorbar(scatter2, ax=ax2, ticks=range(len(unique_batches)))
            cbar2.set_label('batch', fontsize=10)
            cbar2.ax.set_yticklabels(unique_batches, fontsize=9)
            cbar2.ax.tick_params(labelsize=9)

            plt.tight_layout()

            # -------------------- SAVE OUTPUTS --------------------
            output_path = output_dir / 'embedding_overview.png'
            print(f"[DEBUG] Saving figure to: {output_path}")
            plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
            plt.close()

            pca_results = pd.DataFrame(
                embedding_2d,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=embedding_df.index
            )
            pca_path = output_dir / 'pca_coordinates.csv'
            print(f"[DEBUG] Saving PCA coordinates to: {pca_path}")
            pca_results.to_csv(pca_path)

            result = {
                "variance_explained": variance_explained.tolist(),
                "n_samples": int(embedding_df.shape[0]),
                "n_features": int(embedding_df.shape[1]),
                "sev_level_range": [float(sev_min) if pd.notnull(sev_min) else None,
                                    float(sev_max) if pd.notnull(sev_max) else None],
                "unique_batches": len(unique_batches),
                "batch_labels": unique_batches,
                "output_plot": str(output_path),
                "output_pca": str(pca_path)
            }

            print("[DEBUG] Visualization completed successfully.")
            logger.info(f"Embedding visualization completed. Results saved to: {output_dir}")
            return {"status": "success", "output_dir": str(output_dir), "result": result}

        except Exception as e:
            logger.error(f"Error in embedding visualization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            print("[DEBUG] ERROR:", e)
            return {"status": "error", "message": str(e)}

    def run_trajectory_anova(self, **kwargs) -> Dict[str, Any]:
        """
        Run trajectory ANOVA analysis.

        Requires:
            - meta_csv_path
            - pseudotime_csv_path
        """
        logger.info("Running Trajectory ANOVA Analysis...")
        output_dir = self._create_output_dir("trajectory_anova")

        if not self._check_file_exists(self.pseudotime_csv_path, "Pseudotime CSV file"):
            return {"status": "error", "message": "Missing or invalid pseudotime CSV path."}

        try:
            result = run_trajectory_anova_analysis(
                meta_csv_path=str(self.meta_csv_path),
                pseudotime_csv_path=str(self.pseudotime_csv_path),
                output_dir_path=str(output_dir),
                **kwargs,
            )
            logger.info(f"Trajectory ANOVA completed. Results saved to: {output_dir}")
            return {"status": "success", "output_dir": str(output_dir), "result": result}
        except Exception as e:
            logger.error(f"Error in trajectory ANOVA: {e}")
            return {"status": "error", "message": str(e)}

    def run_batch_removal_evaluation(
        self, k: int = 15, include_self: bool = False, **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate batch removal performance.

        Requires:
            - meta_csv_path
            - embedding_csv_path
        """
        logger.info("Running Batch Removal Evaluation...")
        output_dir = self._create_output_dir("batch_removal")

        if not self._check_file_exists(self.embedding_csv_path, "Embedding/coordinates CSV file"):
            return {"status": "error", "message": "Missing or invalid embedding CSV path."}

        try:
            result = evaluate_batch_removal(
                meta_csv=str(self.meta_csv_path),
                data_csv=str(self.embedding_csv_path),
                mode="embedding",
                outdir=str(output_dir),
                k=k,
                include_self=include_self,
                **kwargs,
            )
            logger.info(f"Batch removal evaluation completed. Results saved to: {output_dir}")
            return {"status": "success", "output_dir": str(output_dir), "result": result}
        except Exception as e:
            logger.error(f"Error in batch removal evaluation: {e}")
            return {"status": "error", "message": str(e)}

    def run_ari_clustering(
        self,
        k_neighbors: int = 15,
        n_clusters: Optional[int] = None,
        create_plots: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run ARI clustering evaluation.

        Requires:
            - meta_csv_path
            - embedding_csv_path
        """
        logger.info("Running ARI Clustering Evaluation...")
        output_dir = self._create_output_dir("ari_clustering")

        if not self._check_file_exists(self.embedding_csv_path, "Embedding/coordinates CSV file"):
            return {"status": "error", "message": "Missing or invalid embedding CSV path."}

        try:
            result = evaluate_ari_clustering(
                meta_csv=str(self.meta_csv_path),
                data_csv=str(self.embedding_csv_path),
                mode="embedding",
                outdir=str(output_dir),
                k_neighbors=k_neighbors,
                n_clusters=n_clusters,
                create_plots=create_plots,
                **kwargs,
            )
            logger.info(f"ARI clustering evaluation completed. Results saved to: {output_dir}")
            return {"status": "success", "output_dir": str(output_dir), "result": result}
        except Exception as e:
            logger.error(f"Error in ARI clustering evaluation: {e}")
            return {"status": "error", "message": str(e)}

    def run_trajectory_analysis(self, **kwargs) -> Dict[str, Any]:
        """
        Run trajectory analysis.

        Requires:
            - meta_csv_path
            - pseudotime_csv_path
        """
        logger.info("Running Trajectory Analysis...")
        output_dir = self._create_output_dir("trajectory_analysis")

        if not self._check_file_exists(self.pseudotime_csv_path, "Pseudotime CSV file"):
            return {"status": "error", "message": "Missing or invalid pseudotime CSV path."}

        try:
            result = run_trajectory_analysis(
                meta_csv_path=str(self.meta_csv_path),
                pseudotime_csv_path=str(self.pseudotime_csv_path),
                output_dir_path=str(output_dir),
                **kwargs,
            )
            logger.info(f"Trajectory analysis completed. Results saved to: {output_dir}")
            return {"status": "success", "output_dir": str(output_dir), "result": result}
        except Exception as e:
            logger.error(f"Error in trajectory analysis: {e}")
            return {"status": "error", "message": str(e)}

    def run_pseudotime_embeddings_custom(
        self,
        anchor_batch: str = "Su",
        batch_col: str = "batch",
        sev_col: str = "sev.level",
        severity_transform: str = "raw",
        neighbor_batches_include: Optional[List[str]] = None,
        neighbor_batches_exclude: Optional[List[str]] = None,
        k_neighbors: int = 1,
        metric: str = "euclidean",
        nn_agg: str = "mean",
        standardize_embedding: bool = True,
        make_plots: bool = True,
        random_state: int = 0,
        embedding_label: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Simple custom benchmark expecting pseudotime CSV with columns:
            sample,pseudotime
        Uses 'pseudotime' directly, aligned by sample ID.
        """
        logger.info("Running Custom Pseudotime Embeddings Benchmark (simple schema)...")
        output_dir = self._create_output_dir("pseudotime_embeddings_custom")

        # Required files
        if not self._check_file_exists(self.embedding_csv_path, "Embedding/coordinates CSV file"):
            return {"status": "error", "message": "Missing or invalid embedding CSV path."}
        if not self._check_file_exists(self.pseudotime_csv_path, "Pseudotime CSV file"):
            return {"status": "error", "message": "Missing or invalid pseudotime CSV path."}

        try:
            # --- Load metadata and index by 'sample'
            meta_df = pd.read_csv(self.meta_csv_path)
            for c in (batch_col, sev_col):
                if c not in meta_df.columns:
                    return {"status": "error", "message": f"Missing column in metadata: '{c}'"}
            if 'sample' in meta_df.columns:
                meta_df['sample'] = meta_df['sample'].astype(str).str.strip()
                meta_df = meta_df.set_index('sample')
            else:
                return {"status": "error", "message": "Metadata must contain a 'sample' column."}

            # --- Load embedding (indexed by sample IDs)
            embedding_df = pd.read_csv(self.embedding_csv_path, index_col=0)
            embedding_df.index = embedding_df.index.astype(str).str.strip()

            # --- Align meta & embedding
            common_ids = meta_df.index.intersection(embedding_df.index)
            if common_ids.empty:
                return {"status": "error", "message": "No overlapping sample IDs between metadata and embedding."}
            meta_df = meta_df.loc[common_ids]
            embedding_df = embedding_df.loc[common_ids]

            # --- Load pseudotime with fixed schema: sample,pseudotime
            pt = pd.read_csv(self.pseudotime_csv_path, usecols=['sample', 'pseudotime'])
            pt['sample'] = pt['sample'].astype(str).str.strip()
            pt = pt.drop_duplicates(subset='sample')
            pt = pt.set_index('sample')

            # --- Join pseudotime and drop rows without it
            meta_df = meta_df.join(pt[['pseudotime']], how='left')
            keep = meta_df['pseudotime'].notna()
            if not keep.any():
                return {"status": "error", "message": "All pseudotime values are missing after alignment."}
            if (~keep).any():
                dropped = int((~keep).sum())
                logger.warning(f"Dropping {dropped} samples with missing pseudotime.")
            meta_df = meta_df.loc[keep]
            embedding_df = embedding_df.loc[keep]

            # --- Run benchmark
            method_name = embedding_label if embedding_label else self.mode
            result = benchmark_pseudotime_embeddings_custom(
                df=meta_df,
                embedding=embedding_df.values,
                method_name=method_name,
                anchor_batch=anchor_batch,
                batch_col=batch_col,
                sev_col=sev_col,
                pseudotime_col='pseudotime',  # fixed, simple
                severity_transform=severity_transform,
                neighbor_batches_include=neighbor_batches_include,
                neighbor_batches_exclude=neighbor_batches_exclude,
                k_neighbors=k_neighbors,
                metric=metric,
                nn_agg=nn_agg,
                standardize_embedding=standardize_embedding,
                make_plots=make_plots,
                save_dir=str(output_dir),
                random_state=random_state,
                **kwargs
            )

            return {"status": "success", "output_dir": str(output_dir), "result": result}

        except Exception as e:
            logger.error(f"Error in custom pseudotime benchmark: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}

    # ------------------------- orchestration -------------------------

    def run_all_benchmarks(
        self,
        skip_benchmarks: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run all benchmark analyses.

        Parameters
        ----------
        skip_benchmarks : list of str, optional
            List of benchmark names to skip
        **kwargs : dict
            Parameters to pass to individual benchmarks (use per-benchmark keys)
            e.g., kwargs = {
                "embedding_visualization": {"dpi": 300, "figsize": (12, 5)},
                "trajectory_anova": {...},
                "batch_removal": {"k": 20},
                "ari_clustering": {"k_neighbors": 30, "n_clusters": 8},
                "trajectory_analysis": {...},
                "pseudotime_embeddings_custom": {"k_neighbors": 5, "anchor_batch": "Su"}
            }

        Returns
        -------
        dict
            Dictionary with results from all benchmarks
        """
        skip_benchmarks = skip_benchmarks or []
        results: Dict[str, Dict[str, Any]] = {}

        benchmark_methods = {
            "embedding_visualization": self.run_embedding_visualization,
            "trajectory_anova": self.run_trajectory_anova,
            "batch_removal": self.run_batch_removal_evaluation,
            "ari_clustering": self.run_ari_clustering,
            "trajectory_analysis": self.run_trajectory_analysis,
            "pseudotime_embeddings_custom": self.run_pseudotime_embeddings_custom,  # <-- ADDED
        }

        for name, method in benchmark_methods.items():
            if name in skip_benchmarks:
                logger.info(f"Skipping {name}...")
                continue

            logger.info(f"\n{'=' * 50}")
            logger.info(f"Running {name}...")
            logger.info(f"{'=' * 50}")

            method_kwargs = kwargs.get(name, {})
            results[name] = method(**method_kwargs)

        self._save_summary(results)
        return results

def run_benchmarks(
    meta_csv_path: str,
    pseudotime_csv_path: Optional[str] = None,
    embedding_csv_path: Optional[str] = None,
    mode: str = "expression",
    benchmarks_to_run: Optional[List[str]] = None,
    output_base_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to run benchmarks with explicit paths.

    Parameters
    ----------
    meta_csv_path : str
        Path to metadata CSV file
    pseudotime_csv_path : str, optional
        Path to pseudotime CSV file (required for trajectory_* benchmarks)
    embedding_csv_path : str, optional
        Path to embedding/coordinates CSV file (required for ARI/batch-removal)
    mode : str
        Label used ONLY for output folder naming
    benchmarks_to_run : list of str, optional
        Specific benchmarks to run. If None, runs all.
        Options: ['embedding_visualization', 'trajectory_anova', 'batch_removal', 'ari_clustering', 'trajectory_analysis', 'pseudotime_embeddings_custom']
    output_base_dir : str, optional
        Base directory for outputs
    **kwargs : dict
        Per-benchmark kwargs, e.g.:
        {
          "embedding_visualization": {"dpi": 300, "figsize": (12, 5)},
          "ari_clustering": {"k_neighbors": 30, "n_clusters": 8},
          "batch_removal": {"k": 20, "include_self": True},
          "pseudotime_embeddings_custom": {"k_neighbors": 5, "anchor_batch": "Su"}
        }

    Returns
    -------
    dict
        Results from all benchmarks
    """
    try:
        wrapper = BenchmarkWrapper(
            meta_csv_path=meta_csv_path,
            pseudotime_csv_path=pseudotime_csv_path,
            embedding_csv_path=embedding_csv_path,
            mode=mode,
            output_base_dir=output_base_dir,
        )

        if benchmarks_to_run:
            all_benchmarks = ["embedding_visualization", "trajectory_anova", "batch_removal", "ari_clustering", "trajectory_analysis", "pseudotime_embeddings_custom"]  # <-- ADDED
            skip_benchmarks = [b for b in all_benchmarks if b not in benchmarks_to_run]
            return wrapper.run_all_benchmarks(skip_benchmarks=skip_benchmarks, **kwargs)
        else:
            return wrapper.run_all_benchmarks(**kwargs)

    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        logger.error(f"Failed to initialize BenchmarkWrapper: {e}")
        return {"initialization_error": {"status": "error", "message": str(e)}}
 
# ------------------------- examples -------------------------
if __name__ == "__main__":
    # Example: run all with explicit paths
    results = run_benchmarks(
        meta_csv_path="/dcl01/hongkai/data/data/hjiang/Data/covid_data/sample_data.csv",
        pseudotime_csv_path='/users/hjiang/r/gedi_out_25/trajectory/pseudotime_results.csv',
        embedding_csv_path="/users/hjiang/r/gedi_out_25/gedi_sample_embedding.csv",
        summary_csv_path="/dcs07/hongkai/data/harry/result/benchmark_summary_all_methods.csv",
        mode="expression",
        output_base_dir = '/users/hjiang/r/gedi_out_25',
        # per-benchmark overrides (optional)
        embedding_visualization={"dpi": 300, "figsize": (12, 5)},
        ari_clustering={"k_neighbors": 20, "n_clusters": None, "create_plots": True},
        batch_removal={"k": 15, "include_self": False},
    )
