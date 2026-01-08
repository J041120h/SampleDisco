import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from embedding_effective import evaluate_ari_clustering

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BenchmarkWrapper:
    """
    A streamlined wrapper for running embedding-based benchmark analyses (visualization + clustering only).

    Parameters
    ----------
    meta_csv_path : str
        Path to metadata CSV file (required for all benchmarks)
        Must contain: 'sample', label_col (e.g., 'tissue'), and optionally 'batch'
    embedding_csv_path : str
        Path to embedding/coordinates CSV file (required for all benchmarks)
    method_name : str
        Name of the method being benchmarked (e.g., 'GEDI', 'scVI'). Used for summary CSV columns.
    label_col : str, default='tissue'
        Name of the label column in metadata for clustering evaluation (e.g., 'tissue', 'sev.level', 'cell_type')
    output_base_dir : str, optional
        Base directory for all outputs. If None, defaults to parent of the meta CSV file.
    summary_csv_path : str, optional
        Path to the summary CSV file for aggregating results across runs.
    """

    def __init__(
        self,
        meta_csv_path: str,
        embedding_csv_path: str,
        method_name: str = "method",
        label_col: str = "disease_state",
        output_base_dir: Optional[str] = None,
        summary_csv_path: Optional[str] = None,
    ):
        # Store and validate core inputs
        self.meta_csv_path = Path(meta_csv_path).resolve()
        self.embedding_csv_path = Path(embedding_csv_path).resolve()
        self.method_name = method_name
        self.label_col = label_col

        if not self.meta_csv_path.exists() or not self.meta_csv_path.is_file():
            raise FileNotFoundError(f"Metadata CSV does not exist or is not a file: {self.meta_csv_path}")
        
        if not self.embedding_csv_path.exists() or not self.embedding_csv_path.is_file():
            raise FileNotFoundError(f"Embedding CSV does not exist or is not a file: {self.embedding_csv_path}")

        # Output base directory strategy
        if output_base_dir is None:
            # Default to the parent of the meta CSV so there's always a stable place to write
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

        logger.info("Initialized BenchmarkWrapper (visualization + clustering only) with:")
        logger.info(f"  Meta CSV:          {self.meta_csv_path}")
        logger.info(f"  Embedding CSV:     {self.embedding_csv_path}")
        logger.info(f"  Method name:       {self.method_name}")
        logger.info(f"  Label column:      {self.label_col}")
        logger.info(f"  Output base dir:   {self.output_base_dir}")
        logger.info(f"  Run output dir:    {self.run_output_dir}")
        logger.info(f"  Summary CSV:       {self.summary_csv_path}")

    # ------------------------- helpers -------------------------

    def _create_output_dir(self, benchmark_name: str) -> Path:
        out = self.run_output_dir / benchmark_name
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

    @staticmethod
    def _normalize_sample_ids(series_or_index):
        """
        Normalize sample IDs for robust matching.
        
        Steps:
        1. Convert to string
        2. Strip whitespace
        3. Convert to lowercase
        4. Remove common prefixes/suffixes if needed
        
        Returns a pandas Index or Series with normalized values.
        """
        if isinstance(series_or_index, pd.Index):
            normalized = series_or_index.astype(str).str.strip().str.lower()
        else:  # Series
            normalized = series_or_index.astype(str).str.strip().str.lower()
        
        return normalized

    @staticmethod
    def _load_csv_robust(filepath: Path, index_col: Optional[int] = None) -> pd.DataFrame:
        """
        Load CSV with robust handling of encoding issues (BOM, etc.)
        """
        # Try reading with UTF-8-sig first (handles BOM)
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig', index_col=index_col)
            return df
        except Exception as e:
            logger.warning(f"Failed to read with utf-8-sig encoding: {e}")
        
        # Fallback to default encoding
        try:
            df = pd.read_csv(filepath, index_col=index_col)
            return df
        except Exception as e:
            logger.error(f"Failed to read CSV file {filepath}: {e}")
            raise

    def _align_data(self, meta_df: pd.DataFrame, embedding_df: pd.DataFrame) -> tuple:
        """
        Align metadata and embedding DataFrames by sample IDs with robust matching.
        
        Returns:
            tuple: (aligned_meta_df, aligned_embedding_df, alignment_stats)
        """
        print("\n" + "="*80)
        print("SAMPLE ID ALIGNMENT DIAGNOSTICS")
        print("="*80)
        
        # Step 1: Identify sample ID column in metadata
        if 'sample' in meta_df.columns:
            meta_df = meta_df.copy()
            meta_sample_ids = self._normalize_sample_ids(meta_df['sample'])
            meta_df['_normalized_sample'] = meta_sample_ids
            meta_df = meta_df.set_index('_normalized_sample')
            print(f"✓ Using 'sample' column from metadata")
        else:
            # Use existing index
            meta_sample_ids = self._normalize_sample_ids(meta_df.index)
            meta_df = meta_df.copy()
            meta_df.index = meta_sample_ids
            print(f"⚠ No 'sample' column found; using metadata index")
        
        # Step 2: Normalize embedding index
        embedding_sample_ids = self._normalize_sample_ids(embedding_df.index)
        embedding_df = embedding_df.copy()
        embedding_df.index = embedding_sample_ids
        
        # Step 3: Report on sample ID formats
        print(f"\nMetadata samples (n={len(meta_df)}):")
        print(f"  First 5: {list(meta_df.index[:5])}")
        print(f"  Last 5:  {list(meta_df.index[-5:])}")
        
        print(f"\nEmbedding samples (n={len(embedding_df)}):")
        print(f"  First 5: {list(embedding_df.index[:5])}")
        print(f"  Last 5:  {list(embedding_df.index[-5:])}")
        
        # Step 4: Find overlap
        common_ids = meta_df.index.intersection(embedding_df.index)
        only_meta = meta_df.index.difference(embedding_df.index)
        only_emb = embedding_df.index.difference(meta_df.index)
        
        # Step 5: Report alignment statistics
        print(f"\n{'─'*80}")
        print("ALIGNMENT STATISTICS")
        print(f"{'─'*80}")
        print(f"  Common samples:        {len(common_ids):>6} ({len(common_ids)/len(meta_df)*100:>5.1f}% of metadata)")
        print(f"  Metadata-only:         {len(only_meta):>6} samples")
        print(f"  Embedding-only:        {len(only_emb):>6} samples")
        
        if len(only_meta) > 0:
            print(f"\n  Examples of metadata-only samples (first 10):")
            for i, sid in enumerate(list(only_meta[:10]), 1):
                print(f"    {i:>2}. {sid}")
        
        if len(only_emb) > 0:
            print(f"\n  Examples of embedding-only samples (first 10):")
            for i, sid in enumerate(list(only_emb[:10]), 1):
                print(f"    {i:>2}. {sid}")
        
        # Step 6: Check for potential partial matches (debugging)
        if len(common_ids) == 0 and len(only_meta) > 0 and len(only_emb) > 0:
            print(f"\n⚠ WARNING: No exact matches found. Checking for partial matches...")
            meta_samples_set = set(only_meta)
            emb_samples_set = set(only_emb)
            
            # Check if any meta sample is a substring of any emb sample or vice versa
            partial_matches = []
            for meta_s in list(meta_samples_set)[:20]:  # Check first 20
                for emb_s in list(emb_samples_set)[:20]:
                    if meta_s in emb_s or emb_s in meta_s:
                        partial_matches.append((meta_s, emb_s))
            
            if partial_matches:
                print(f"  Found {len(partial_matches)} potential partial matches:")
                for meta_s, emb_s in partial_matches[:5]:
                    print(f"    Meta: '{meta_s}' <-> Emb: '{emb_s}'")
        
        print("="*80 + "\n")
        
        # Step 7: Validate and align
        if len(common_ids) == 0:
            raise ValueError(
                "No overlapping sample IDs between metadata and embedding! "
                "Check the alignment diagnostics above for details."
            )
        
        # Align both dataframes to common IDs (in the same order)
        meta_aligned = meta_df.loc[common_ids]
        embedding_aligned = embedding_df.loc[common_ids]
        
        # Step 8: Return alignment stats
        alignment_stats = {
            'n_common': len(common_ids),
            'n_meta_only': len(only_meta),
            'n_emb_only': len(only_emb),
            'pct_aligned': len(common_ids) / len(meta_df) * 100,
            'common_ids': list(common_ids)
        }
        
        return meta_aligned, embedding_aligned, alignment_stats

    def _save_summary_csv(self, results: Dict[str, Dict[str, Any]]) -> None:
        """
        Save a summary of benchmark results to a CSV file.
        
        Structure:
        - Rows: benchmark metric categories (ARI, NMI, Avg_Purity, n_samples)
        - Columns: method_name (e.g., GEDI, scVI, pilot)
        """
        summary_csv_path = self.summary_csv_path
        
        # Ensure parent directory exists
        summary_csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Collect metrics from all successful benchmarks
        all_metrics = {}
        sample_size = None
        
        for benchmark_name, bench_result in results.items():
            if bench_result.get("status") != "success":
                logger.warning(f"Skipping {benchmark_name} in summary - status was not 'success'")
                continue
            
            result = bench_result.get("result", {})
            if result is None:
                result = {}
            
            # Debug: log all keys in result
            logger.info(f"[DEBUG] {benchmark_name} result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
            
            # Map benchmark-specific metrics to standard row names
            if benchmark_name == "embedding_visualization":
                if "n_samples" in result:
                    all_metrics["n_samples"] = result["n_samples"]
                    sample_size = result["n_samples"]
                    
            elif benchmark_name == "ari_clustering":
                # embedding_effective.py returns: metrics dict nested with 'ari', 'nmi', 'avg_purity'
                metrics_dict = result.get("metrics", {})
                if isinstance(metrics_dict, dict):
                    if "ari" in metrics_dict:
                        all_metrics["ARI"] = metrics_dict["ari"]
                    if "nmi" in metrics_dict:
                        all_metrics["NMI"] = metrics_dict["nmi"]
                    if "avg_purity" in metrics_dict:
                        all_metrics["Avg_Purity"] = metrics_dict["avg_purity"]
        
        logger.info(f"[DEBUG] Collected metrics: {all_metrics}")
        
        if not all_metrics:
            logger.warning("No metrics collected from benchmarks - nothing to save to summary CSV")
            return
        
        # Build column name: just method_name (no sample size since it's single-size per method now)
        col_name = self.method_name
        
        # Load existing summary or create new one
        if summary_csv_path.exists():
            summary_df = pd.read_csv(summary_csv_path, index_col=0)
        else:
            summary_df = pd.DataFrame()
        
        # Add/update the column for this run
        for metric, value in all_metrics.items():
            summary_df.loc[metric, col_name] = value
        
        # Save
        summary_df.to_csv(summary_csv_path, index_label="Metric")
        logger.info(f"Updated summary CSV at: {summary_csv_path} with column '{col_name}'")


    def run_embedding_visualization(
        self,
        n_components: int = 2,
        figsize: tuple = (12, 5),
        dpi: int = 300,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Visualize embeddings colored by label_col (continuous) and optionally by batch (categorical).

        Requires:
            - meta_csv_path (must include columns: label_col, 'sample', and optionally 'batch')
            - embedding_csv_path (rows indexed by sample IDs matching meta['sample'] or meta index)
        
        Note: If 'batch' column is not present, creates single-panel visualization with only label_col.
        """
        logger.info("Running Embedding Visualization...")
        output_dir = self._create_output_dir("embedding_visualization")

        if not self._check_file_exists(self.embedding_csv_path, "Embedding/coordinates CSV file"):
            return {"status": "error", "message": "Missing or invalid embedding CSV path."}

        try:
            # -------------------- LOAD --------------------
            logger.info(f"Loading metadata from: {self.meta_csv_path}")
            meta_df = self._load_csv_robust(self.meta_csv_path)
            print(f"[DEBUG] Metadata loaded: shape={meta_df.shape}, columns={meta_df.columns.tolist()}")

            logger.info(f"Loading embeddings from: {self.embedding_csv_path}")
            embedding_df = self._load_csv_robust(self.embedding_csv_path, index_col=0)
            print(f"[DEBUG] Embedding matrix shape: {embedding_df.shape}")
            print(f"[DEBUG] First 3 embedding indices: {embedding_df.index[:3].tolist()}")

            # -------------------- REQUIREMENTS --------------------
            required_cols = [self.label_col]
            missing_cols = [c for c in required_cols if c not in meta_df.columns]
            if missing_cols:
                logger.warning(f"Missing columns in metadata: {missing_cols}")
                print(f"[DEBUG] ERROR: Missing required columns in metadata: {missing_cols}")
                return {"status": "error", "message": f"Missing required columns in metadata: {missing_cols}"}
            
            # Check if batch column is available
            has_batch = 'batch' in meta_df.columns
            if not has_batch:
                logger.info("'batch' column not found in metadata - will create single-panel visualization")
                print("[DEBUG] 'batch' column not found - creating single-panel visualization")

            # -------------------- ALIGN DATA --------------------
            meta_df, embedding_df, align_stats = self._align_data(meta_df, embedding_df)
            
            logger.info(f"Aligned {align_stats['n_common']} samples ({align_stats['pct_aligned']:.1f}%)")
            print(f"[DEBUG] Final aligned shapes: meta={meta_df.shape}, embedding={embedding_df.shape}")

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
            n_panels = 2 if has_batch else 1
            fig, axes = plt.subplots(1, n_panels, figsize=(figsize[0] if has_batch else figsize[0]/2, figsize[1]), dpi=dpi)
            if n_panels == 1:
                axes = [axes]  # Make it iterable for consistent indexing

            # Left panel (or only panel): label_col
            print(f"[DEBUG] Plotting {self.label_col} panel...")
            
            # Detect if label_col is numerical or categorical
            label_values_raw = meta_df[self.label_col]
            label_numeric = pd.to_numeric(label_values_raw, errors='coerce')
            is_numerical = label_numeric.notna().sum() / len(label_numeric) > 0.5  # >50% convertible to numeric
            
            if is_numerical:
                # Numerical: use continuous colormap (viridis)
                print(f"[DEBUG] {self.label_col} detected as NUMERICAL - using continuous colormap")
                label_min, label_max = label_numeric.min(), label_numeric.max()
                label_range = (label_max - label_min) if pd.notnull(label_max) and pd.notnull(label_min) else 0.0
                if label_range == 0.0:
                    print(f"[DEBUG][WARN] {self.label_col} has zero/invalid range; coloring will be constant.")
                label_norm = (label_numeric - (label_min if pd.notnull(label_min) else 0.0)) / (label_range + 1e-16)
                print(f"[DEBUG] {self.label_col} raw range: min={label_min}, max={label_max}")
                print(f"[DEBUG] {self.label_col} normalized range: {label_norm.min():.3f}–{label_norm.max():.3f}")
                
                ax1 = axes[0]
                scatter1 = ax1.scatter(
                    embedding_2d[:, 0],
                    embedding_2d[:, 1],
                    c=label_norm,
                    cmap='viridis',
                    edgecolors='black',
                    alpha=0.8,
                    s=100,
                    linewidths=0.5
                )
                cbar1 = plt.colorbar(scatter1, ax=ax1)
                cbar1.set_label(f'{self.label_col}', fontsize=10)
                cbar1.ax.tick_params(labelsize=10)
            else:
                # Categorical: use discrete colormap (tab10/tab20)
                print(f"[DEBUG] {self.label_col} detected as CATEGORICAL - using discrete colormap")
                unique_labels = sorted(label_values_raw.astype(str).unique().tolist())
                n_unique = len(unique_labels)
                print(f"[DEBUG] Unique {self.label_col} values ({n_unique}): {unique_labels[:10]}{'...' if n_unique>10 else ''}")
                
                label_to_num = {lbl: i for i, lbl in enumerate(unique_labels)}
                label_colors = [label_to_num[str(lbl)] for lbl in label_values_raw]
                
                # Choose colormap based on number of categories
                cmap = 'tab20' if n_unique > 10 else 'tab10'
                
                ax1 = axes[0]
                scatter1 = ax1.scatter(
                    embedding_2d[:, 0],
                    embedding_2d[:, 1],
                    c=label_colors,
                    cmap=cmap,
                    edgecolors='black',
                    alpha=0.8,
                    s=100,
                    linewidths=0.5
                )
                cbar1 = plt.colorbar(scatter1, ax=ax1, ticks=range(n_unique))
                cbar1.set_label(f'{self.label_col}', fontsize=10)
                # Set categorical labels on colorbar
                if n_unique <= 20:  # Only show labels if not too many
                    cbar1.ax.set_yticklabels(unique_labels, fontsize=8)
                cbar1.ax.tick_params(labelsize=8)

            ax1.set_xlabel(f'PC1 ({variance_explained[0]:.1%})', fontsize=12, fontweight='bold')
            ax1.set_ylabel(f'PC2 ({variance_explained[1]:.1%})', fontsize=12, fontweight='bold' if n_components >= 2 else None)
            ax1.set_title(f'Embeddings colored by {self.label_col}', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, linestyle='--')

            # Right panel: batch (categorical) - only if batch column exists
            unique_batches = []
            if has_batch:
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

            # Get label min/max safely
            if is_numerical:
                label_min_val = float(label_min) if pd.notnull(label_min) else None
                label_max_val = float(label_max) if pd.notnull(label_max) else None
            else:
                label_min_val = None
                label_max_val = None

            result = {
                "variance_explained": variance_explained.tolist(),
                "n_samples": int(embedding_df.shape[0]),
                "n_features": int(embedding_df.shape[1]),
                f"{self.label_col}_range": [label_min_val, label_max_val],
                "unique_batches": len(unique_batches),
                "batch_labels": unique_batches,
                "output_plot": str(output_path),
                "output_pca": str(pca_path),
                "alignment_stats": align_stats
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
                label_col=self.label_col,
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
                "ari_clustering": {"k_neighbors": 30, "n_clusters": 8}
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
            "ari_clustering": self.run_ari_clustering,
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
    embedding_csv_path: str,
    method_name: str = "method",
    label_col: str = "disease_state",
    benchmarks_to_run: Optional[List[str]] = None,
    output_base_dir: Optional[str] = None,
    summary_csv_path: Optional[str] = None,
    **kwargs,
) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to run benchmarks with explicit paths.

    Parameters
    ----------
    meta_csv_path : str
        Path to metadata CSV file
    embedding_csv_path : str
        Path to embedding/coordinates CSV file
    method_name : str
        Name of the method being benchmarked (e.g., 'GEDI', 'scVI', 'pilot')
    label_col : str, default='disease_state'
        Name of the label column in metadata for clustering evaluation (e.g., 'disease_state', 'sev.level', 'cell_type')
    benchmarks_to_run : list of str, optional
        Specific benchmarks to run. If None, runs all.
        Options: ['embedding_visualization', 'ari_clustering']
    output_base_dir : str, optional
        Base directory for outputs
    summary_csv_path : str, optional
        Path to the summary CSV file for aggregating results across runs.
    **kwargs : dict
        Per-benchmark kwargs, e.g.:
        {
          "embedding_visualization": {"dpi": 300, "figsize": (12, 5)},
          "ari_clustering": {"k_neighbors": 30, "n_clusters": 8}
        }

    Returns
    -------
    dict
        Results from all benchmarks
    """
    try:
        wrapper = BenchmarkWrapper(
            meta_csv_path=meta_csv_path,
            embedding_csv_path=embedding_csv_path,
            method_name=method_name,
            label_col=label_col,
            output_base_dir=output_base_dir,
            summary_csv_path=summary_csv_path,
        )

        if benchmarks_to_run:
            all_benchmarks = ["embedding_visualization", "ari_clustering"]
            skip_benchmarks = [b for b in all_benchmarks if b not in benchmarks_to_run]
            return wrapper.run_all_benchmarks(skip_benchmarks=skip_benchmarks, **kwargs)
        else:
            return wrapper.run_all_benchmarks(**kwargs)

    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        logger.error(f"Failed to initialize BenchmarkWrapper: {e}")
        return {"initialization_error": {"status": "error", "message": str(e)}}

 
if __name__ == "__main__":

    # Base paths - update to the new folder path
    base_dir = '/dcs07/hongkai/data/harry/result/Benchmark_heart_rna'
    meta_csv_path = "/dcs07/hongkai/data/harry/result/multi_omics_heart/data/multi_omics_heart_sample_meta.csv"
    summary_csv_path = f'{base_dir}/benchmark_summary_heart_rna.csv'

    # ========== SD_expression (sample distance - expression) ==========
    results = run_benchmarks(
        meta_csv_path=meta_csv_path,
        embedding_csv_path=f'{base_dir}/rna/embeddings/sample_expression_embedding.csv',
        summary_csv_path=summary_csv_path,
        method_name="SD_expression",
        label_col="disease_state",
        output_base_dir=f'{base_dir}/rna',
        embedding_visualization={"dpi": 300, "figsize": (12, 5)},
        ari_clustering={"k_neighbors": 20, "n_clusters": None, "create_plots": True},
    )

    # ========== SD_proportion (sample distance - proportion) ==========
    results = run_benchmarks(
        meta_csv_path=meta_csv_path,
        embedding_csv_path=f'{base_dir}/rna/embeddings/sample_proportion_embedding.csv',
        summary_csv_path=summary_csv_path,
        method_name="SD_proportion",
        label_col="disease_state",
        output_base_dir=f'{base_dir}/rna',
        embedding_visualization={"dpi": 300, "figsize": (12, 5)},
        ari_clustering={"k_neighbors": 20, "n_clusters": None, "create_plots": True},
    )

    # ========== GEDI ==========
    results = run_benchmarks(
        meta_csv_path=meta_csv_path,
        embedding_csv_path=f'{base_dir}/GEDI/gedi_sample_embedding.csv',
        summary_csv_path=summary_csv_path,
        method_name="GEDI",
        label_col="disease_state",
        output_base_dir=f'{base_dir}/GEDI',
        embedding_visualization={"dpi": 300, "figsize": (12, 5)},
        ari_clustering={"k_neighbors": 20, "n_clusters": None, "create_plots": True},
    )

    # ========== Gloscope ==========
    results = run_benchmarks(
        meta_csv_path=meta_csv_path,
        embedding_csv_path=f'{base_dir}/Gloscope/knn_divergence_mds_10d.csv',
        summary_csv_path=summary_csv_path,
        method_name="Gloscope",
        label_col="disease_state",
        output_base_dir=f'{base_dir}/Gloscope',
        embedding_visualization={"dpi": 300, "figsize": (12, 5)},
        ari_clustering={"k_neighbors": 20, "n_clusters": None, "create_plots": True},
    )

    # ========== MFA ==========
    results = run_benchmarks(
        meta_csv_path=meta_csv_path,
        embedding_csv_path=f'{base_dir}/MFA/sample_embeddings.csv',
        summary_csv_path=summary_csv_path,
        method_name="MFA",
        label_col="disease_state",
        output_base_dir=f'{base_dir}/MFA',
        embedding_visualization={"dpi": 300, "figsize": (12, 5)},
        ari_clustering={"k_neighbors": 20, "n_clusters": None, "create_plots": True},
    )

    # ========== pseudobulk (naive) ==========
    results = run_benchmarks(
        meta_csv_path=meta_csv_path,
        embedding_csv_path=f'{base_dir}/pseudobulk/pseudobulk/pca_embeddings.csv',
        summary_csv_path=summary_csv_path,
        method_name="pseudobulk",
        label_col="disease_state",
        output_base_dir=f'{base_dir}/pseudobulk',
        embedding_visualization={"dpi": 300, "figsize": (12, 5)},
        ari_clustering={"k_neighbors": 20, "n_clusters": None, "create_plots": True},
    )

    # ========== pilot ==========
    results = run_benchmarks(
        meta_csv_path=meta_csv_path,
        embedding_csv_path=f'{base_dir}/pilot/wasserstein_distance_mds_10d.csv',
        summary_csv_path=summary_csv_path,
        method_name="pilot",
        label_col="disease_state",
        output_base_dir=f'{base_dir}/pilot',
        embedding_visualization={"dpi": 300, "figsize": (12, 5)},
        ari_clustering={"k_neighbors": 20, "n_clusters": None, "create_plots": True},
    )

    # ========== QOT ==========
    results = run_benchmarks(
        meta_csv_path=meta_csv_path,
        embedding_csv_path=f'{base_dir}/QOT/44_qot_distance_matrix_mds_10d.csv',
        summary_csv_path=summary_csv_path,
        method_name="QOT",
        label_col="disease_state",
        output_base_dir=f'{base_dir}/QOT',
        embedding_visualization={"dpi": 300, "figsize": (12, 5)},
        ari_clustering={"k_neighbors": 20, "n_clusters": None, "create_plots": True},
    )

    # ========== scPoli ==========
    results = run_benchmarks(
        meta_csv_path=meta_csv_path,
        embedding_csv_path=f'{base_dir}/scPoli/sample_embeddings_full.csv',
        summary_csv_path=summary_csv_path,
        method_name="scPoli",
        label_col="disease_state",
        output_base_dir=f'{base_dir}/scPoli',
        embedding_visualization={"dpi": 300, "figsize": (12, 5)},
        ari_clustering={"k_neighbors": 20, "n_clusters": None, "create_plots": True},
    )

    # ========== MUSTARD ==========
    results = run_benchmarks(
        meta_csv_path=meta_csv_path,
        embedding_csv_path=f'{base_dir}/mustard/sample_embedding.csv',
        summary_csv_path=summary_csv_path,
        method_name="MUSTARD",
        label_col="disease_state",
        output_base_dir=f'{base_dir}/mustard',
        embedding_visualization={"dpi": 300, "figsize": (12, 5)},
        ari_clustering={"k_neighbors": 20, "n_clusters": None, "create_plots": True},
    )