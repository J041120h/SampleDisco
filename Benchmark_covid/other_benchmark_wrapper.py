import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

from ANOVA import run_trajectory_anova_analysis
from batch_removal_test import evaluate_batch_removal
from embedding_effective import evaluate_ari_clustering
from spearman_test import run_trajectory_analysis
from customized_benchmark import benchmark_pseudotime_embeddings_custom

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global plotting style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "figure.dpi": 100,
})


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
    method_name : str
        Name of the method being benchmarked (e.g., 'GEDI', 'scVI'). Used for summary CSV columns.
    output_base_dir : str, optional
        Base directory for all outputs. If None, defaults to parent of the meta CSV file.
    summary_csv_path : str, optional
        Path to the summary CSV file for aggregating results across runs.
    """

    def __init__(
        self,
        meta_csv_path: str,
        pseudotime_csv_path: Optional[str] = None,
        embedding_csv_path: Optional[str] = None,
        method_name: str = "method",
        output_base_dir: Optional[str] = None,
        summary_csv_path: Optional[str] = None,
    ):
        # Store and validate core inputs
        self.meta_csv_path = Path(meta_csv_path).resolve()
        self.pseudotime_csv_path = Path(pseudotime_csv_path).resolve() if pseudotime_csv_path else None
        self.embedding_csv_path = Path(embedding_csv_path).resolve() if embedding_csv_path else None
        self.method_name = method_name

        if not self.meta_csv_path.exists() or not self.meta_csv_path.is_file():
            raise FileNotFoundError(f"Metadata CSV does not exist or is not a file: {self.meta_csv_path}")

        # Output base directory strategy
        if output_base_dir is None:
            self.output_base_dir = self.meta_csv_path.parent
        else:
            self.output_base_dir = Path(output_base_dir).resolve()

        # Summary CSV path
        if summary_csv_path is not None:
            self.summary_csv_path = Path(summary_csv_path).resolve()
        else:
            self.summary_csv_path = self.output_base_dir / "benchmark_summary.csv"

        # Output directory for this run
        self.run_output_dir = self.output_base_dir / f"benchmark_results_{self.method_name}"
        self.run_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initialized BenchmarkWrapper (explicit paths) with:")
        logger.info(f"  Meta CSV:          {self.meta_csv_path}")
        logger.info(f"  Pseudotime CSV:    {self.pseudotime_csv_path if self.pseudotime_csv_path else '(not provided)'}")
        logger.info(f"  Embedding CSV:     {self.embedding_csv_path if self.embedding_csv_path else '(not provided)'}")
        logger.info(f"  Method name:       {self.method_name}")
        logger.info(f"  Output base dir:   {self.output_base_dir}")
        logger.info(f"  Run output dir:    {self.run_output_dir}")
        logger.info(f"  Summary CSV:       {self.summary_csv_path}")

    # ------------------------- helpers -------------------------

    def _create_output_dir(self, benchmark_name: str) -> Path:
        out = self.run_output_dir / benchmark_name
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _check_file_exists(self, file_path: Optional[Path], file_description: str) -> bool:
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

    def _save_summary_csv(self, results: Dict[str, Dict[str, Any]]) -> None:
        summary_csv_path = self.summary_csv_path
        summary_csv_path.parent.mkdir(parents=True, exist_ok=True)

        all_metrics = {}
        sample_size = None
        
        for benchmark_name, bench_result in results.items():
            if bench_result.get("status") != "success":
                logger.warning(f"Skipping {benchmark_name} in summary - status was not 'success'")
                continue
            
            result = bench_result.get("result", {})
            if result is None:
                result = {}
            
            logger.info(f"[DEBUG] {benchmark_name} result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
            
            if sample_size is None:
                sample_size = result.get("n_samples")
            
            if benchmark_name == "embedding_visualization":
                if "n_samples" in result:
                    all_metrics["n_samples"] = result["n_samples"]
                    
            elif benchmark_name == "trajectory_anova":
                anova_table = result.get("anova_table")
                if anova_table is not None and hasattr(anova_table, 'loc'):
                    try:
                        if 'C(batch)' in anova_table.index and 'partial_eta_sq' in anova_table.columns:
                            all_metrics["batch_partial_eta_sq"] = float(anova_table.loc['C(batch)', 'partial_eta_sq'])
                        if 'C(severity_level)' in anova_table.index and 'partial_eta_sq' in anova_table.columns:
                            all_metrics["severity_partial_eta_sq"] = float(anova_table.loc['C(severity_level)', 'partial_eta_sq'])
                        if 'C(batch):C(severity_level)' in anova_table.index and 'partial_eta_sq' in anova_table.columns:
                            all_metrics["interaction_partial_eta_sq"] = float(anova_table.loc['C(batch):C(severity_level)', 'partial_eta_sq'])
                    except Exception as e:
                        logger.warning(f"Could not extract ANOVA metrics: {e}")
                    
            elif benchmark_name == "batch_removal":
                if "iLISI_norm_mean" in result:
                    all_metrics["iLISI_norm"] = result["iLISI_norm_mean"]
                if "ASW_batch_overall" in result:
                    all_metrics["ASW_batch"] = result["ASW_batch_overall"]
                    
            elif benchmark_name == "ari_clustering":
                metrics_dict = result.get("metrics", {})
                if isinstance(metrics_dict, dict):
                    if "ari" in metrics_dict:
                        all_metrics["ARI"] = metrics_dict["ari"]
                    if "nmi" in metrics_dict:
                        all_metrics["NMI"] = metrics_dict["nmi"]
                    if "avg_purity" in metrics_dict:
                        all_metrics["Avg_Purity"] = metrics_dict["avg_purity"]
                    
            elif benchmark_name == "trajectory_analysis":
                if "spearman_corr" in result:
                    all_metrics["Spearman_Correlation"] = result["spearman_corr"]
                if "spearman_p" in result:
                    all_metrics["Spearman_pval"] = result["spearman_p"]
                    
            elif benchmark_name == "pseudotime_embeddings_custom":
                nn_gap_summary = result.get("nn_gap_summary")
                if nn_gap_summary is not None and hasattr(nn_gap_summary, 'iloc'):
                    try:
                        row = nn_gap_summary.iloc[0]
                        if "mean_|Δsev|" in row:
                            all_metrics["Mean_NN_Severity_Gap"] = float(row["mean_|Δsev|"])
                        if "n_anchor" in row:
                            sample_size = int(row["n_anchor"]) if sample_size is None else sample_size
                    except Exception as e:
                        logger.warning(f"Could not extract nn_gap_summary metrics: {e}")
                
                anova_scipy = result.get("anova_anchor_scipy")
                if isinstance(anova_scipy, dict):
                    if "eta_sq" in anova_scipy:
                        all_metrics["Custom_ANOVA_eta_sq"] = anova_scipy["eta_sq"]
                    if "omega_sq" in anova_scipy:
                        all_metrics["Custom_ANOVA_omega_sq"] = anova_scipy["omega_sq"]
            
            elif benchmark_name == "batch_mixing":
                if "mean_same_batch_proportion" in result:
                    all_metrics["Mean_Same_Batch_Proportion"] = result["mean_same_batch_proportion"]
                if "expected_same_batch_proportion" in result:
                    all_metrics["Expected_Same_Batch_Proportion"] = result["expected_same_batch_proportion"]
        
        logger.info(f"[DEBUG] Collected metrics: {all_metrics}")
        
        if not all_metrics:
            logger.warning("No metrics collected from benchmarks - nothing to save to summary CSV")
            return
        
        col_name = f"{self.method_name}-{sample_size}" if sample_size else self.method_name
        
        if summary_csv_path.exists():
            summary_df = pd.read_csv(summary_csv_path, index_col=0)
        else:
            summary_df = pd.DataFrame()
        
        for metric, value in all_metrics.items():
            summary_df.loc[metric, col_name] = value
        
        summary_df.to_csv(summary_csv_path, index_label="Metric")
        logger.info(f"Updated summary CSV at: {summary_csv_path} with column '{col_name}'")

    @staticmethod
    def _parse_severity_numeric(sev_series: pd.Series) -> pd.Series:
        """
        Attempt to convert severity levels to numeric values.

        Strategy:
        1. Try direct numeric conversion (handles int/float strings).
        2. If that fails for >50% of values, try ordinal mapping from
        known COVID severity labels (e.g., healthy < mild < moderate < severe < critical).
        3. Fall back to alphabetical rank order if nothing else works.

        Returns a numeric Series aligned with the input index.
        """
        # Step 1: try direct numeric
        numeric = pd.to_numeric(sev_series, errors='coerce')
        frac_valid = numeric.notna().mean()
        if frac_valid > 0.5:
            # Most values are already numeric; fill any remaining NaNs with median
            if numeric.isna().any():
                numeric = numeric.fillna(numeric.median())
            return numeric

        # Step 2: known ordinal mapping (case-insensitive)
        known_order = {
            'healthy': 0, 'control': 0, 'normal': 0, 'none': 0,
            'mild': 1, 'moderate': 2,
            'severe': 3, 'critical': 4, 'deceased': 5, 'death': 5,
            'convalescent': 2.5, 'conv': 2.5,
        }
        sev_lower = sev_series.astype(str).str.strip().str.lower()
        mapped = sev_lower.map(known_order)
        frac_mapped = mapped.notna().mean()
        if frac_mapped > 0.5:
            if mapped.isna().any():
                mapped = mapped.fillna(mapped.median())
            return mapped

        # Step 3: rank unique values alphabetically
        unique_vals = sorted(sev_series.astype(str).unique())
        rank_map = {v: i for i, v in enumerate(unique_vals)}
        return sev_series.astype(str).map(rank_map).astype(float)

    def _plot_sample_distance_heatmap_by_severity(
        self,
        embedding_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        output_dir: Path,
        distance_metric: str = "euclidean",
        dpi: int = 300,
    ) -> Dict[str, Any]:
        """
        Create a sample-by-sample distance heatmap from embeddings, ordered by
        severity level (low → high) with clean styling.
        """
        logger.info("Creating sample distance heatmap ordered by severity level...")

        # --- Parse severity to numeric and sort ---
        sev_numeric = self._parse_severity_numeric(meta_df['sev.level'])
        sev_numeric.name = 'severity_numeric'
        print(f"[DEBUG] Severity numeric range: {sev_numeric.min():.3f} – {sev_numeric.max():.3f}")
        print(f"[DEBUG] Severity numeric unique values: {sorted(sev_numeric.unique())}")

        # Sort samples by severity (low → high), break ties by batch then sample ID
        sort_df = pd.DataFrame({
            'severity_numeric': sev_numeric,
            'batch': meta_df['batch'].astype(str),
        }, index=meta_df.index)
        sort_df = sort_df.sort_values(['severity_numeric', 'batch'])
        sorted_ids = sort_df.index

        # Reorder embedding and meta
        embedding_sorted = embedding_df.loc[sorted_ids]
        sev_sorted = sev_numeric.loc[sorted_ids]
        batch_sorted = meta_df.loc[sorted_ids, 'batch'].astype(str)

        # --- Compute pairwise distance matrix ---
        dist_condensed = pdist(embedding_sorted.values, metric=distance_metric)
        dist_matrix = squareform(dist_condensed)
        print(f"[DEBUG] Distance matrix shape: {dist_matrix.shape}, metric={distance_metric}")

        # --- Figure with clean styling ---
        fig = plt.figure(figsize=(10, 10), dpi=dpi)
        
        # GridSpec: top for annotation bars, middle for heatmap, right for colorbar
        gs = fig.add_gridspec(
            nrows=3, ncols=2,
            height_ratios=[0.02, 0.02, 1],
            width_ratios=[1, 0.02],
            hspace=0.02, wspace=0.02,
        )

        ax_sev_bar = fig.add_subplot(gs[0, 0])
        ax_batch_bar = fig.add_subplot(gs[1, 0])
        ax_heatmap = fig.add_subplot(gs[2, 0])
        ax_cbar = fig.add_subplot(gs[2, 1])

        n = len(sorted_ids)

        # --- Severity annotation bar ---
        sev_vals = sev_sorted.values
        sev_norm_vals = (sev_vals - sev_vals.min()) / (sev_vals.max() - sev_vals.min() + 1e-16)
        sev_colors = plt.cm.viridis(sev_norm_vals).reshape(1, n, 4)
        ax_sev_bar.imshow(sev_colors, aspect='auto', interpolation='nearest')
        ax_sev_bar.set_xticks([])
        ax_sev_bar.set_yticks([])
        ax_sev_bar.set_xlim(-0.5, n - 0.5)
        # Remove spines
        for spine in ax_sev_bar.spines.values():
            spine.set_visible(False)

        # --- Batch annotation bar ---
        unique_batches = sorted(batch_sorted.unique())
        batch_to_num = {b: i for i, b in enumerate(unique_batches)}
        batch_num = np.array([batch_to_num[b] for b in batch_sorted])
        batch_cmap = plt.cm.tab10
        batch_colors = batch_cmap(batch_num / max(len(unique_batches) - 1, 1)).reshape(1, n, 4)
        ax_batch_bar.imshow(batch_colors, aspect='auto', interpolation='nearest')
        ax_batch_bar.set_xticks([])
        ax_batch_bar.set_yticks([])
        ax_batch_bar.set_xlim(-0.5, n - 0.5)
        # Remove spines
        for spine in ax_batch_bar.spines.values():
            spine.set_visible(False)

        # --- Heatmap ---
        im = ax_heatmap.imshow(
            dist_matrix,
            cmap='magma_r',
            aspect='auto',
            interpolation='nearest',
        )
        ax_heatmap.set_xlabel('Samples (ordered by severity)', fontsize=11)
        ax_heatmap.set_ylabel('Samples (ordered by severity)', fontsize=11)
        ax_heatmap.set_title(
            f'Sample Distance Heatmap ({distance_metric})',
            fontsize=12, pad=12,
        )

        # Minimal ticks
        ax_heatmap.set_xticks([])
        ax_heatmap.set_yticks([])
        
        # Thin spines
        for spine in ax_heatmap.spines.values():
            spine.set_linewidth(0.8)

        # Distance colorbar
        cbar = plt.colorbar(im, cax=ax_cbar)
        cbar.set_label(f'{distance_metric} distance', fontsize=10)
        cbar.outline.set_linewidth(0.8)
        cbar.ax.tick_params(labelsize=9)

        # --- Save PNG only ---
        heatmap_path = output_dir / 'sample_distance_heatmap_severity.png'
        fig.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[DEBUG] Severity-ordered heatmap saved to: {heatmap_path}")

        # Save distance matrix CSV
        dist_df = pd.DataFrame(dist_matrix, index=sorted_ids, columns=sorted_ids)
        dist_csv_path = output_dir / 'sample_distance_matrix_severity.csv'
        dist_df.to_csv(dist_csv_path)

        # Save the severity ordering CSV
        order_df = pd.DataFrame({
            'sample': sorted_ids,
            'severity_raw': meta_df.loc[sorted_ids, 'sev.level'].values,
            'severity_numeric': sev_sorted.values,
            'batch': batch_sorted.values,
        })
        order_csv_path = output_dir / 'sample_severity_order.csv'
        order_df.to_csv(order_csv_path, index=False)

        return {
            "heatmap_path": str(heatmap_path),
            "distance_matrix_csv": str(dist_csv_path),
            "severity_order_csv": str(order_csv_path),
            "distance_metric": distance_metric,
        }

    def _plot_sample_distance_heatmap_by_batch(
        self,
        embedding_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        output_dir: Path,
        distance_metric: str = "euclidean",
        dpi: int = 300,
    ) -> Dict[str, Any]:
        """
        Create a sample-by-sample distance heatmap from embeddings, ordered by
        batch with clean styling.
        """
        logger.info("Creating sample distance heatmap ordered by batch...")

        # Sort samples by batch, then sample ID
        sort_df = pd.DataFrame({
            'batch': meta_df['batch'].astype(str),
        }, index=meta_df.index)
        sort_df = sort_df.sort_values(['batch'])
        sorted_ids = sort_df.index

        # Reorder
        embedding_sorted = embedding_df.loc[sorted_ids]
        batch_sorted = meta_df.loc[sorted_ids, 'batch'].astype(str)

        # --- Compute pairwise distance matrix ---
        dist_condensed = pdist(embedding_sorted.values, metric=distance_metric)
        dist_matrix = squareform(dist_condensed)
        print(f"[DEBUG] Distance matrix shape: {dist_matrix.shape}, metric={distance_metric}")

        # --- Figure with clean styling ---
        fig = plt.figure(figsize=(10, 10), dpi=dpi)
        
        gs = fig.add_gridspec(
            nrows=2, ncols=2,
            height_ratios=[0.02, 1],
            width_ratios=[1, 0.02],
            hspace=0.02, wspace=0.02,
        )

        ax_batch_bar = fig.add_subplot(gs[0, 0])
        ax_heatmap = fig.add_subplot(gs[1, 0])
        ax_cbar = fig.add_subplot(gs[1, 1])

        n = len(sorted_ids)

        # --- Batch annotation bar ---
        unique_batches = sorted(batch_sorted.unique())
        batch_to_num = {b: i for i, b in enumerate(unique_batches)}
        batch_num = np.array([batch_to_num[b] for b in batch_sorted])
        batch_cmap = plt.cm.tab10
        batch_colors = batch_cmap(batch_num / max(len(unique_batches) - 1, 1)).reshape(1, n, 4)
        ax_batch_bar.imshow(batch_colors, aspect='auto', interpolation='nearest')
        ax_batch_bar.set_xticks([])
        ax_batch_bar.set_yticks([])
        ax_batch_bar.set_xlim(-0.5, n - 0.5)
        for spine in ax_batch_bar.spines.values():
            spine.set_visible(False)

        # --- Heatmap ---
        im = ax_heatmap.imshow(
            dist_matrix,
            cmap='magma_r',
            aspect='auto',
            interpolation='nearest',
        )
        ax_heatmap.set_xlabel('Samples (ordered by batch)', fontsize=11)
        ax_heatmap.set_ylabel('Samples (ordered by batch)', fontsize=11)
        ax_heatmap.set_title(
            f'Sample Distance Heatmap ({distance_metric})',
            fontsize=12, pad=12,
        )

        ax_heatmap.set_xticks([])
        ax_heatmap.set_yticks([])
        
        for spine in ax_heatmap.spines.values():
            spine.set_linewidth(0.8)

        # Distance colorbar
        cbar = plt.colorbar(im, cax=ax_cbar)
        cbar.set_label(f'{distance_metric} distance', fontsize=10)
        cbar.outline.set_linewidth(0.8)
        cbar.ax.tick_params(labelsize=9)

        # --- Save PNG only ---
        heatmap_path = output_dir / 'sample_distance_heatmap_batch.png'
        fig.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[DEBUG] Batch-ordered heatmap saved to: {heatmap_path}")

        # Save distance matrix CSV
        dist_df = pd.DataFrame(dist_matrix, index=sorted_ids, columns=sorted_ids)
        dist_csv_path = output_dir / 'sample_distance_matrix_batch.csv'
        dist_df.to_csv(dist_csv_path)

        # Save the batch ordering CSV
        order_df = pd.DataFrame({
            'sample': sorted_ids,
            'batch': batch_sorted.values,
        })
        order_csv_path = output_dir / 'sample_batch_order.csv'
        order_df.to_csv(order_csv_path, index=False)

        return {
            "heatmap_path": str(heatmap_path),
            "distance_matrix_csv": str(dist_csv_path),
            "batch_order_csv": str(order_csv_path),
            "distance_metric": distance_metric,
        }

    # -------------------- NEW: Batch Mixing Visualizations --------------------

    def _compute_same_batch_proportions(
        self,
        embedding_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        k: int = 20
    ) -> tuple:
        """
        Compute same-batch neighbor proportions.
        Returns: same_batch_pct array, mean_val, expected_val
        """
        batch = meta_df['batch'].values
        
        # Compute kNN
        X = embedding_df.values
        k_actual = min(k, len(X) - 1)
        knn = NearestNeighbors(n_neighbors=k_actual + 1).fit(X)
        distances, indices = knn.kneighbors(X)
        
        # For each sample: % of neighbors from same batch
        same_batch_pct = []
        for i, neighbors in enumerate(indices):
            sample_batch = batch[i]
            neighbor_batches = batch[neighbors[1:]]
            same_batch_count = (neighbor_batches == sample_batch).sum()
            same_batch_pct.append(same_batch_count / k_actual)
        
        # Global expected proportion
        batch_counts = pd.Series(batch).value_counts()
        expected_per_sample = [(batch_counts[batch[i]] - 1) / (len(batch) - 1) for i in range(len(batch))]
        global_expected = np.mean(expected_per_sample)
        
        mean_val = np.mean(same_batch_pct)
        
        return np.array(same_batch_pct), mean_val, global_expected

    def _plot_knn_same_batch_distribution(
        self,
        embedding_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        output_dir: Path,
        k: int = 20,
        ylim: Optional[tuple] = None
    ) -> Dict[str, Any]:
        """
        Plot clean histogram of same-batch neighbor proportions.
        Square format, journal style, no annotations.
        """
        same_batch_pct, mean_val, global_expected = self._compute_same_batch_proportions(
            embedding_df, meta_df, k
        )
        
        # Create square figure with equal aspect
        fig = plt.figure(figsize=(7, 7))
        
        # Use gridspec to ensure main plot is perfectly square
        gs = fig.add_gridspec(1, 1, left=0.15, right=0.95, top=0.92, bottom=0.12)
        ax = fig.add_subplot(gs[0, 0])
        
        # Better binning strategy
        n_bins = min(30, max(15, len(same_batch_pct) // 10))
        
        # Histogram with better styling
        counts, bins, patches = ax.hist(
            same_batch_pct, 
            bins=n_bins, 
            color='#5A9BD4',
            edgecolor='white',
            linewidth=0.5,
            alpha=0.85
        )
        
        # Labels and title
        ax.set_xlabel('Proportion of same-batch neighbors', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Same-batch neighbor distribution', fontsize=14, pad=15)
        
        # Set x-axis limits to [0, 1] for consistency
        ax.set_xlim(0, 1)
        
        # Set y-axis limits if provided (for consistency across methods)
        if ylim is not None:
            ax.set_ylim(ylim)
        
        # Clean styling
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(0.8)
        
        # Better tick formatting
        ax.tick_params(axis='both', which='major', labelsize=10, width=0.8)
        
        out_path = output_dir / 'same_batch_distribution_histogram.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {out_path}")
        
        # Print interpretation
        median_val = np.median(same_batch_pct)
        interpretation = f"""
        Expected same-batch proportion: {global_expected:.2f}
        Observed median: {median_val:.2f}
        Observed mean: {mean_val:.2f}
        
        Interpretation:
        - Batch effect removed: distribution centered around {global_expected:.2f}
        - Batch effect present: distribution shifted toward 1.0
        - Current shift: {median_val - global_expected:.2f}
        """
        logger.info(interpretation)
        
        return {
            "histogram_path": str(out_path),
            "mean_same_batch_proportion": float(mean_val),
            "median_same_batch_proportion": float(median_val),
            "expected_same_batch_proportion": float(global_expected),
        }

    def _plot_knn_same_batch_violin(
        self,
        embedding_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        output_dir: Path,
        k: int = 20
    ) -> Dict[str, Any]:
        """
        Plot violin plot of same-batch neighbor proportions.
        Square format, only show mean line (no label).
        """
        same_batch_pct, mean_val, _ = self._compute_same_batch_proportions(
            embedding_df, meta_df, k
        )
        
        # Create square figure
        fig = plt.figure(figsize=(7, 7))
        gs = fig.add_gridspec(1, 1, left=0.15, right=0.95, top=0.92, bottom=0.12)
        ax = fig.add_subplot(gs[0, 0])
        
        # Violin plot
        parts = ax.violinplot(
            [same_batch_pct], 
            positions=[0], 
            widths=0.7,
            showmeans=False,
            showmedians=False,
            showextrema=False
        )
        
        # Style the violin
        for pc in parts['bodies']:
            pc.set_facecolor('#5A9BD4')
            pc.set_edgecolor('black')
            pc.set_alpha(0.85)
            pc.set_linewidth(0.8)
        
        # Add horizontal line for mean (NO LABEL)
        ax.axhline(
            mean_val,
            color='#E15759',
            linestyle='--',
            linewidth=2.5,
            alpha=0.8,
            zorder=10
        )
        
        # Labels and title
        ax.set_ylabel('Proportion of same-batch neighbors', fontsize=12)
        ax.set_title('Same-batch neighbor distribution', fontsize=14, pad=15)
        
        # Set y-axis limits to [0, 1] for consistency
        ax.set_ylim(0, 1)
        
        # Remove x-axis ticks and labels
        ax.set_xticks([])
        ax.set_xlim(-0.5, 0.5)
        
        # Clean styling
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        
        # Better tick formatting
        ax.tick_params(axis='y', which='major', labelsize=10, width=0.8)
        
        out_path = output_dir / 'same_batch_distribution_violin.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {out_path}")
        
        return {
            "violin_path": str(out_path),
        }

    def _plot_batch_mixing_matrix_normalized(
        self,
        embedding_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        output_dir: Path,
        k: int = 20
    ) -> Dict[str, Any]:
        """
        Plot normalized mixing matrix in square format, journal style.
        Creates TWO versions: one with annotations, one without.
        """
        batch = meta_df['batch'].values
        
        # Compute kNN
        X = embedding_df.values
        k_actual = min(k, len(X) - 1)
        knn = NearestNeighbors(n_neighbors=k_actual + 1).fit(X)
        distances, indices = knn.kneighbors(X)
        
        # Count cross-batch connections (observed)
        batch_labels = sorted(set(batch))
        mixing_observed = pd.DataFrame(0, index=batch_labels, columns=batch_labels, dtype=float)
        
        for i, neighbors in enumerate(indices):
            source = batch[i]
            for n in neighbors[1:]:
                target = batch[n]
                mixing_observed.loc[source, target] += 1
        
        # Normalize by row (observed proportions)
        mixing_obs_norm = mixing_observed.div(mixing_observed.sum(axis=1), axis=0)
        
        # Expected proportions (batch sizes)
        batch_counts = pd.Series(batch).value_counts()
        batch_props = batch_counts / batch_counts.sum()
        expected = pd.DataFrame(
            np.tile(batch_props.values, (len(batch_labels), 1)),
            index=batch_labels,
            columns=batch_props.index
        )
        
        # Normalized mixing score: observed / expected
        mixing_norm = mixing_obs_norm / expected
        
        # VERSION 1: With annotations
        fig, ax = plt.subplots(figsize=(7, 6.5))
        
        sns.heatmap(mixing_norm, annot=True, fmt='.2f', cmap='RdBu_r', 
                    center=1.0, vmin=0, vmax=2, square=True, ax=ax,
                    cbar_kws={'label': 'Obs/Exp ratio'})
        ax.set_title(f'Normalized Batch Mixing (n={len(X)} samples, k={k_actual})')
        ax.set_xlabel('Neighbor batch')
        ax.set_ylabel('Sample batch')
        
        # Batch sizes as subtitle
        batch_size_str = ", ".join([f"{b}: n={batch_counts[b]}" for b in batch_labels])
        fig.text(0.5, 0.02, f'Batch sizes: {batch_size_str}', 
                 ha='center', fontsize=9, style='italic')
        
        plt.tight_layout()
        
        out_path_annot = output_dir / 'batch_mixing_matrix_normalized_annotated.png'
        plt.savefig(out_path_annot, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {out_path_annot}")
        
        # VERSION 2: Without annotations
        fig, ax = plt.subplots(figsize=(7, 6.5))
        
        sns.heatmap(mixing_norm, annot=False, cmap='RdBu_r', 
                    center=1.0, vmin=0, vmax=2, square=True, ax=ax,
                    cbar_kws={'label': 'Obs/Exp ratio'})
        ax.set_title(f'Normalized Batch Mixing (n={len(X)} samples, k={k_actual})')
        ax.set_xlabel('Neighbor batch')
        ax.set_ylabel('Sample batch')
        
        # Batch sizes as subtitle
        fig.text(0.5, 0.02, f'Batch sizes: {batch_size_str}', 
                 ha='center', fontsize=9, style='italic')
        
        plt.tight_layout()
        
        out_path_clean = output_dir / 'batch_mixing_matrix_normalized_clean.png'
        plt.savefig(out_path_clean, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {out_path_clean}")
        
        return {
            "mixing_matrix_annotated_path": str(out_path_annot),
            "mixing_matrix_clean_path": str(out_path_clean),
        }

    # -------------------- Embedding Visualization --------------------

    def run_embedding_visualization(
        self,
        n_components: int = 2,
        figsize: tuple = (12, 5),
        dpi: int = 300,
        heatmap_distance_metric: str = "euclidean",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Visualize embeddings colored by sev.level (continuous) and batch (categorical),
        PLUS sample-by-sample distance heatmaps ordered by severity and batch separately.

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
            if 'sample' in meta_df.columns:
                print("[DEBUG] Setting meta_df index to 'sample'")
                meta_df = meta_df.set_index('sample')
            else:
                print("[DEBUG][WARN] 'sample' column not found; using meta_df.index for alignment")

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

            # -------------------- VISUALIZATION (PCA scatter plots) --------------------
            print("[DEBUG] Creating figure...")
            
            # Create two square panels side by side
            fig = plt.figure(figsize=(13.0, 6.0), dpi=dpi)
            ax1 = fig.add_axes([0.06, 0.12, 0.35, 0.62])
            ax2 = fig.add_axes([0.56, 0.12, 0.35, 0.62])

            # Helper function for axis styling
            def style_embedding_axes(ax, xlabel="PC1", ylabel="PC2", title=None):
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                if title is not None:
                    ax.set_title(title, pad=12)
                ax.grid(False)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(0.8)
                ax.set_aspect(1.0, adjustable="box")

            # Left panel: sev.level (continuous)
            print("[DEBUG] Plotting sev.level panel...")
            sev_levels = pd.to_numeric(meta_df['sev.level'], errors='coerce')
            sev_min, sev_max = sev_levels.min(), sev_levels.max()
            sev_range = (sev_max - sev_min) if pd.notnull(sev_max) and pd.notnull(sev_min) else 0.0
            if sev_range == 0.0:
                print("[DEBUG][WARN] sev.level has zero/invalid range; coloring will be constant.")
                sev_norm = pd.Series(0.5, index=sev_levels.index)
            else:
                sev_norm = (sev_levels - sev_min) / sev_range
            print(f"[DEBUG] sev.level raw range: min={sev_min}, max={sev_max}")
            print(f"[DEBUG] sev.level normalized range: {sev_norm.min():.3f}–{sev_norm.max():.3f}")

            scatter1 = ax1.scatter(
                embedding_2d[:, 0],
                embedding_2d[:, 1],
                c=sev_norm,
                cmap='viridis',
                s=50,
                alpha=0.7,
                edgecolors='none'
            )

            style_embedding_axes(ax1, xlabel="PC1", ylabel="PC2", title="Embeddings colored by severity")
            
            # Equal aspect ratio with padding
            x_min, x_max = embedding_2d[:, 0].min(), embedding_2d[:, 0].max()
            y_min, y_max = embedding_2d[:, 1].min(), embedding_2d[:, 1].max()
            cx = 0.5 * (x_min + x_max)
            cy = 0.5 * (y_min + y_max)
            dx = x_max - x_min
            dy = y_max - y_min
            half_range = 0.5 * max(dx, dy)
            pad = 0.10
            half_range *= (1.0 + pad)
            if half_range == 0:
                half_range = 1.0
            ax1.set_xlim(cx - half_range, cx + half_range)
            ax1.set_ylim(cy - half_range, cy + half_range)

            cbar1 = plt.colorbar(scatter1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.set_label('Severity (normalized)')
            cbar1.outline.set_linewidth(0.8)

            # Right panel: batch (categorical) with clean legend
            print("[DEBUG] Plotting batch panel...")
            unique_batches = sorted(meta_df['batch'].astype(str).unique().tolist())
            print(f"[DEBUG] Unique batches ({len(unique_batches)}): {unique_batches[:10]}{'...' if len(unique_batches)>10 else ''}")
            batch_to_num = {b: i for i, b in enumerate(unique_batches)}
            batch_colors_numeric = [batch_to_num[str(b)] for b in meta_df['batch']]
            
            # Get actual colors from tab10
            batch_cmap = plt.cm.tab10
            n_batches = len(unique_batches)

            # Plot each batch separately for legend
            for batch in unique_batches:
                mask = meta_df['batch'].astype(str) == batch
                color_idx = batch_to_num[batch]
                color = batch_cmap(color_idx / max(n_batches - 1, 1))
                ax2.scatter(
                    embedding_2d[mask, 0],
                    embedding_2d[mask, 1],
                    c=[color],
                    s=50,
                    alpha=0.7,
                    edgecolors='none',
                    label=batch
                )

            style_embedding_axes(ax2, xlabel="PC1", ylabel="PC2", title="Embeddings colored by batch")
            
            # Same equal aspect ratio
            ax2.set_xlim(cx - half_range, cx + half_range)
            ax2.set_ylim(cy - half_range, cy + half_range)

            # Clean legend outside plot
            leg = ax2.legend(
                frameon=True,
                bbox_to_anchor=(1.25, 1.0),
                loc="upper left",
                borderpad=0.5,
                framealpha=1.0,
                edgecolor="black",
                fontsize=9,
            )
            leg.get_frame().set_linewidth(0.8)

            # -------------------- SAVE PCA SCATTER (PNG only) --------------------
            output_path = output_dir / 'embedding_overview.png'
            print(f"[DEBUG] Saving figure to: {output_path}")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            pca_results = pd.DataFrame(
                embedding_2d,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=embedding_df.index
            )
            pca_path = output_dir / 'pca_coordinates.csv'
            print(f"[DEBUG] Saving PCA coordinates to: {pca_path}")
            pca_results.to_csv(pca_path)

            # -------------------- SAMPLE DISTANCE HEATMAPS --------------------
            # Heatmap 1: Ordered by severity
            heatmap_severity = self._plot_sample_distance_heatmap_by_severity(
                embedding_df=embedding_df,
                meta_df=meta_df,
                output_dir=output_dir,
                distance_metric=heatmap_distance_metric,
                dpi=dpi,
            )

            # Heatmap 2: Ordered by batch
            heatmap_batch = self._plot_sample_distance_heatmap_by_batch(
                embedding_df=embedding_df,
                meta_df=meta_df,
                output_dir=output_dir,
                distance_metric=heatmap_distance_metric,
                dpi=dpi,
            )

            result = {
                "variance_explained": variance_explained.tolist(),
                "n_samples": int(embedding_df.shape[0]),
                "n_features": int(embedding_df.shape[1]),
                "sev_level_range": [float(sev_min) if pd.notnull(sev_min) else None,
                                    float(sev_max) if pd.notnull(sev_max) else None],
                "unique_batches": len(unique_batches),
                "batch_labels": unique_batches,
                "output_plot": str(output_path),
                "output_pca": str(pca_path),
                "heatmap_severity": heatmap_severity,
                "heatmap_batch": heatmap_batch,
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

    # -------------------- NEW: Batch Mixing Analysis --------------------

    def run_batch_mixing_analysis(
        self,
        k: int = 20,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run batch mixing analysis with histogram, violin, and mixing matrix plots.

        Requires:
            - meta_csv_path (must include 'batch' column)
            - embedding_csv_path
        """
        logger.info("Running Batch Mixing Analysis...")
        output_dir = self._create_output_dir("batch_mixing")

        if not self._check_file_exists(self.embedding_csv_path, "Embedding/coordinates CSV file"):
            return {"status": "error", "message": "Missing or invalid embedding CSV path."}

        try:
            # Load data
            logger.info(f"Loading metadata from: {self.meta_csv_path}")
            meta_df = pd.read_csv(self.meta_csv_path)
            
            if 'batch' not in meta_df.columns:
                return {"status": "error", "message": "Missing 'batch' column in metadata."}

            logger.info(f"Loading embeddings from: {self.embedding_csv_path}")
            embedding_df = pd.read_csv(self.embedding_csv_path, index_col=0)

            # Align
            if 'sample' in meta_df.columns:
                meta_df = meta_df.set_index('sample')

            common_ids = meta_df.index.intersection(embedding_df.index)
            if len(common_ids) == 0:
                raise ValueError("No overlapping sample IDs between metadata and embedding!")

            meta_df = meta_df.loc[common_ids]
            embedding_df = embedding_df.loc[common_ids]

            # Generate all plots
            histogram_result = self._plot_knn_same_batch_distribution(
                embedding_df, meta_df, output_dir, k=k
            )
            
            violin_result = self._plot_knn_same_batch_violin(
                embedding_df, meta_df, output_dir, k=k
            )
            
            mixing_matrix_result = self._plot_batch_mixing_matrix_normalized(
                embedding_df, meta_df, output_dir, k=k
            )

            result = {
                **histogram_result,
                **violin_result,
                **mixing_matrix_result,
                "k_neighbors": k,
                "n_samples": len(common_ids),
            }

            logger.info(f"Batch mixing analysis completed. Results saved to: {output_dir}")
            return {"status": "success", "output_dir": str(output_dir), "result": result}

        except Exception as e:
            logger.error(f"Error in batch mixing analysis: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}

    # -------------------- Existing Benchmark Methods --------------------

    def run_trajectory_anova(self, **kwargs) -> Dict[str, Any]:
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
        label_col: str = "sev.level",  # <-- ADD THIS PARAMETER
        **kwargs,
    ) -> Dict[str, Any]:
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
                label_col=label_col,  # <-- ADD THIS LINE
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
        logger.info("Running Custom Pseudotime Embeddings Benchmark (simple schema)...")
        output_dir = self._create_output_dir("pseudotime_embeddings_custom")

        if not self._check_file_exists(self.embedding_csv_path, "Embedding/coordinates CSV file"):
            return {"status": "error", "message": "Missing or invalid embedding CSV path."}
        if not self._check_file_exists(self.pseudotime_csv_path, "Pseudotime CSV file"):
            return {"status": "error", "message": "Missing or invalid pseudotime CSV path."}

        try:
            meta_df = pd.read_csv(self.meta_csv_path)
            for c in (batch_col, sev_col):
                if c not in meta_df.columns:
                    return {"status": "error", "message": f"Missing column in metadata: '{c}'"}
            if 'sample' in meta_df.columns:
                meta_df['sample'] = meta_df['sample'].astype(str).str.strip()
                meta_df = meta_df.set_index('sample')
            else:
                return {"status": "error", "message": "Metadata must contain a 'sample' column."}

            embedding_df = pd.read_csv(self.embedding_csv_path, index_col=0)
            embedding_df.index = embedding_df.index.astype(str).str.strip()

            common_ids = meta_df.index.intersection(embedding_df.index)
            if common_ids.empty:
                return {"status": "error", "message": "No overlapping sample IDs between metadata and embedding."}
            meta_df = meta_df.loc[common_ids]
            embedding_df = embedding_df.loc[common_ids]

            pt = pd.read_csv(self.pseudotime_csv_path, usecols=['sample', 'pseudotime'])
            pt['sample'] = pt['sample'].astype(str).str.strip()
            pt = pt.drop_duplicates(subset='sample')
            pt = pt.set_index('sample')

            meta_df = meta_df.join(pt[['pseudotime']], how='left')
            keep = meta_df['pseudotime'].notna()
            if not keep.any():
                return {"status": "error", "message": "All pseudotime values are missing after alignment."}
            if (~keep).any():
                dropped = int((~keep).sum())
                logger.warning(f"Dropping {dropped} samples with missing pseudotime.")
            meta_df = meta_df.loc[keep]
            embedding_df = embedding_df.loc[keep]

            method_label = embedding_label if embedding_label else self.method_name
            result = benchmark_pseudotime_embeddings_custom(
                df=meta_df,
                embedding=embedding_df.values,
                method_name=method_label,
                anchor_batch=anchor_batch,
                batch_col=batch_col,
                sev_col=sev_col,
                pseudotime_col='pseudotime',
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
        skip_benchmarks = skip_benchmarks or []
        results: Dict[str, Dict[str, Any]] = {}

        benchmark_methods = {
            "embedding_visualization": self.run_embedding_visualization,
            "trajectory_anova": self.run_trajectory_anova,
            "batch_removal": self.run_batch_removal_evaluation,
            "ari_clustering": self.run_ari_clustering,
            "trajectory_analysis": self.run_trajectory_analysis,
            "pseudotime_embeddings_custom": self.run_pseudotime_embeddings_custom,
            "batch_mixing": self.run_batch_mixing_analysis,  # NEW
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

        self._save_summary_csv(results)
        return results


def run_benchmarks(
    meta_csv_path: str,
    pseudotime_csv_path: Optional[str] = None,
    embedding_csv_path: Optional[str] = None,
    method_name: str = "method",
    benchmarks_to_run: Optional[List[str]] = None,
    output_base_dir: Optional[str] = None,
    summary_csv_path: Optional[str] = None,
    **kwargs,
) -> Dict[str, Dict[str, Any]]:
    try:
        wrapper = BenchmarkWrapper(
            meta_csv_path=meta_csv_path,
            pseudotime_csv_path=pseudotime_csv_path,
            embedding_csv_path=embedding_csv_path,
            method_name=method_name,
            output_base_dir=output_base_dir,
            summary_csv_path=summary_csv_path,
        )

        if benchmarks_to_run:
            all_benchmarks = [
                "embedding_visualization", "trajectory_anova", "batch_removal", 
                "ari_clustering", "trajectory_analysis", "pseudotime_embeddings_custom",
                "batch_mixing"  # NEW
            ]
            skip_benchmarks = [b for b in all_benchmarks if b not in benchmarks_to_run]
            return wrapper.run_all_benchmarks(skip_benchmarks=skip_benchmarks, **kwargs)
        else:
            return wrapper.run_all_benchmarks(**kwargs)

    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        logger.error(f"Failed to initialize BenchmarkWrapper: {e}")
        return {"initialization_error": {"status": "error", "message": str(e)}}


# ------------------------- simplified main -------------------------
if __name__ == "__main__":
    sample_sizes = [25, 50, 100, 200, 279, 400]

    META_CSV = "/dcl01/hongkai/data/data/hjiang/Data/covid_data/sample_data.csv"
    SUMMARY_CSV = "/dcs07/hongkai/data/harry/result/benchmark_summary_all_methods.csv"

    # NEW: all outputs go here
    COMMON_OUT_ROOT = Path("/dcs07/hongkai/data/harry/result/ALL_BENCHMARK_OUTPUTS")
    COMMON_OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # common per-benchmark overrides (same for all methods)
    COMMON_KWARGS = dict(
        embedding_visualization={"dpi": 300, "figsize": (12, 5)},
        ari_clustering={
            "k_neighbors": 20, 
            "n_clusters": None, 
            "create_plots": True,
            "label_col": "sev.level"  # <-- ADD THIS LINE
        },
        batch_removal={"k": 15, "include_self": False},
        batch_mixing={"k": 20},
    )

    # Each entry defines how to build embedding/pseudotime paths for a given method + size
    METHODS = [
        {
            "name": "SD_expression",
            "paths": lambda size: (
                f"/dcs07/hongkai/data/harry/result/Benchmark_covid/covid_{size}_sample/"
                f"rna/Sample_distance/correlation/expression_DR_distance/expression_DR_coordinates.csv",
                f"/dcs07/hongkai/data/harry/result/Benchmark_covid/covid_{size}_sample/"
                f"rna/CCA/pseudotime_expression.csv",
            ),
        },
        {
            "name": "SD_proportion",
            "paths": lambda size: (
                f"/dcs07/hongkai/data/harry/result/Benchmark_covid/covid_{size}_sample/"
                f"rna/Sample_distance/correlation/proportion_DR_distance/proportion_DR_coordinates.csv",
                f"/dcs07/hongkai/data/harry/result/Benchmark_covid/covid_{size}_sample/"
                f"rna/CCA/pseudotime_proportion.csv",
            ),
        },
        {
            "name": "GEDI",
            "paths": lambda size: (
                f"/dcs07/hongkai/data/harry/result/GEDI/{size}_sample/gedi_sample_embedding.csv",
                f"/dcs07/hongkai/data/harry/result/GEDI/{size}_sample/trajectory/pseudotime_results.csv",
            ),
        },
        {
            "name": "Gloscope",
            "paths": lambda size: (
                f"/dcs07/hongkai/data/harry/result/Gloscope/{size}_sample/knn_divergence_mds_10d.csv",
                f"/dcs07/hongkai/data/harry/result/Gloscope/{size}_sample/trajectory/pseudotime_results.csv",
            ),
        },
        {
            "name": "MFA",
            "paths": lambda size: (
                f"/dcs07/hongkai/data/harry/result/MFA/{size}_sample/sample_embeddings.csv",
                f"/dcs07/hongkai/data/harry/result/MFA/{size}_sample/trajectory/pseudotime_results.csv",
            ),
        },
        {
            "name": "pseudobulk",
            "paths": lambda size: (
                f"/dcs07/hongkai/data/harry/result/naive_pseudobulk/covid_{size}_sample/"
                f"pseudobulk/pca_embeddings.csv",
                f"/dcs07/hongkai/data/harry/result/naive_pseudobulk/covid_{size}_sample/"
                f"pseudobulk/trajectory/pseudotime_results.csv",
            ),
        },
        {
            "name": "pilot",
            "paths": lambda size: (
                f"/dcs07/hongkai/data/harry/result/pilot/{size}_sample/wasserstein_distance_mds_10d.csv",
                f"/dcs07/hongkai/data/harry/result/pilot/{size}_sample/pilot_native_pseudotime.csv",
            ),
        },
        {
            "name": "QOT",
            "paths": lambda size: (
                f"/dcs07/hongkai/data/harry/result/QOT/{size}_sample/{size}_qot_distance_matrix_mds_10d.csv",
                f"/dcs07/hongkai/data/harry/result/QOT/{size}_sample/trajectory/pseudotime_results.csv",
            ),
        },
        {
            "name": "scPoli",
            "paths": lambda size: (
                f"/dcs07/hongkai/data/harry/result/scPoli/{size}_sample/sample_embeddings_full.csv",
                f"/dcs07/hongkai/data/harry/result/scPoli/{size}_sample/trajectory/pseudotime_results.csv",
            ),
        },
        {
            "name": "MUSTARD",
            "paths": lambda size: (
                f"/dcs07/hongkai/data/harry/result/MUSTARD/{size}_sample/sample_embedding.csv",
                f"/dcs07/hongkai/data/harry/result/MUSTARD/{size}_sample/trajectory/pseudotime_results.csv",
            ),
        },
    ]

    # Run everything: output folder name = "method_name-Sample_size"
    for method in METHODS:
        method_name = method["name"]
        for size in sample_sizes:
            embedding_csv_path, pseudotime_csv_path = method["paths"](size)

            # NEW: put outputs in one common place
            output_base_dir = COMMON_OUT_ROOT / f"{method_name}-{size}"
            output_base_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n=== Running {method_name} @ n={size} ===")
            print(f"  output_base_dir:   {output_base_dir}")
            print(f"  embedding_csv:     {embedding_csv_path}")
            print(f"  pseudotime_csv:    {pseudotime_csv_path}")

            results = run_benchmarks(
                meta_csv_path=META_CSV,
                pseudotime_csv_path=pseudotime_csv_path,
                embedding_csv_path=embedding_csv_path,
                summary_csv_path=SUMMARY_CSV,
                method_name=method_name,
                output_base_dir=str(output_base_dir),
                **COMMON_KWARGS,
            )