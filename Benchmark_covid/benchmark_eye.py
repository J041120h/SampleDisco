"""
Benchmark Wrapper for Eye Dataset (Age-based Trajectory Analysis Only)

This wrapper provides trajectory-focused benchmarking for single-cell multiomics embeddings
with age as a continuous trajectory variable. It includes:
- Embedding visualization (age coloring only, no batch)
- Trajectory ANOVA analysis (age effects on pseudotime)
- Spearman correlation analysis (age vs pseudotime; **absolute value** used as metric)

Features:
- Case-insensitive sample ID matching
- Automatic numerical label detection
- Robust error handling and detailed logging
- Trajectory-focused metrics only (no clustering, no batch)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from spearman_test import run_trajectory_analysis

# NEW: imports for one-way ANOVA
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BenchmarkWrapper:
    """
    Benchmark wrapper for age-based trajectory analysis (eye dataset).

    This wrapper orchestrates trajectory-focused benchmark analyses for evaluating
    single-cell multiomics integration methods using age as a continuous variable.

    Parameters
    ----------
    meta_csv_path : str
        Path to metadata CSV file (required for all benchmarks)
        Must contain: 'sample', label_col (e.g., 'age')
    embedding_csv_path : str
        Path to embedding/coordinates CSV file (required for visualization)
        Rows should be indexed by sample IDs
    method_name : str
        Name of the method being benchmarked (e.g., 'GEDI', 'scVI', 'pilot')
        Used as column name in summary CSV
    label_col : str, default='age'
        Name of the label column in metadata for analysis
        Typically a continuous variable (e.g., 'age')
    pseudotime_csv_path : str, optional
        Path to pseudotime CSV file (required for trajectory_* benchmarks)
        Should contain columns: 'sample', 'pseudotime'
        Pseudotime is inferred from the embedding (predicted trajectory).
    output_base_dir : str, optional
        Base directory for all outputs. If None, defaults to parent of meta CSV file
    summary_csv_path : str, optional
        Path to the summary CSV file for aggregating results across methods

    Attributes
    ----------
    run_output_dir : Path
        Directory where results for this method will be saved
    """

    def __init__(
        self,
        meta_csv_path: str,
        embedding_csv_path: str,
        method_name: str = "method",
        label_col: str = "age",
        pseudotime_csv_path: Optional[str] = None,
        output_base_dir: Optional[str] = None,
        summary_csv_path: Optional[str] = None,
    ):
        # Store and validate core inputs
        self.meta_csv_path = Path(meta_csv_path).resolve()
        self.embedding_csv_path = Path(embedding_csv_path).resolve()
        self.pseudotime_csv_path = Path(pseudotime_csv_path).resolve() if pseudotime_csv_path else None
        self.method_name = method_name
        self.label_col = label_col

        if not self.meta_csv_path.exists() or not self.meta_csv_path.is_file():
            raise FileNotFoundError(f"Metadata CSV does not exist or is not a file: {self.meta_csv_path}")
        
        if not self.embedding_csv_path.exists() or not self.embedding_csv_path.is_file():
            raise FileNotFoundError(f"Embedding CSV does not exist or is not a file: {self.embedding_csv_path}")

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

        logger.info("=" * 70)
        logger.info("Initialized BenchmarkWrapper (trajectory-only analysis)")
        logger.info("=" * 70)
        logger.info(f"  Meta CSV:          {self.meta_csv_path}")
        logger.info(f"  Embedding CSV:     {self.embedding_csv_path}")
        logger.info(f"  Pseudotime CSV:    {self.pseudotime_csv_path if self.pseudotime_csv_path else '(not provided)'}")
        logger.info(f"  Method name:       {self.method_name}")
        logger.info(f"  Label column:      {self.label_col}")
        logger.info(f"  Output base dir:   {self.output_base_dir}")
        logger.info(f"  Run output dir:    {self.run_output_dir}")
        logger.info(f"  Summary CSV:       {self.summary_csv_path}")
        logger.info("=" * 70)

    # ------------------------- Helper Methods -------------------------

    def _create_output_dir(self, benchmark_name: str) -> Path:
        """Create and return output directory for a specific benchmark."""
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
            if parent.exists():
                logger.error("  Contents of parent directory:")
                try:
                    for item in sorted(parent.iterdir())[:10]:
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

    def _normalize_sample_ids(self, series_or_index) -> pd.Index:
        """Normalize sample IDs for case-insensitive matching."""
        if isinstance(series_or_index, pd.Series):
            return pd.Index(series_or_index.astype(str).str.lower().str.strip())
        else:
            return pd.Index(series_or_index.astype(str).str.lower().str.strip())

    def _save_summary_csv(self, results: Dict[str, Dict[str, Any]]) -> None:
        """
        Save a summary of benchmark results to a CSV file.
        
        Creates or updates a summary CSV where:
        - Rows = benchmark metrics (Spearman_Correlation, Trajectory_ANOVA_eta_sq, etc.)
        - Columns = method names (GEDI, scVI, pilot, etc.)
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
            
            # Get sample size (take from first benchmark that has it)
            if sample_size is None:
                sample_size = result.get("n_samples")
            
            # Map benchmark-specific metrics to standard row names
            if benchmark_name == "embedding_visualization":
                if "n_samples" in result:
                    all_metrics["n_samples"] = result["n_samples"]

            elif benchmark_name == "trajectory_anova":
                # Our custom one-way ANOVA returns: anova_table with partial_eta_sq column
                anova_table = result.get("anova_table")
                if anova_table is not None and hasattr(anova_table, 'loc'):
                    try:
                        # In run_trajectory_anova, the row is named directly as self.label_col
                        target_row = self.label_col
                        
                        if target_row in anova_table.index and 'partial_eta_sq' in anova_table.columns:
                            # Extract the value
                            val = anova_table.loc[target_row, 'partial_eta_sq']
                            # Save as metric
                            all_metrics[f"One_way_ANOVA_eta_sq"] = float(val)
                            logger.info(f"[DEBUG] Extracted One-way ANOVA eta_sq: {val}")
                        else:
                            logger.warning(f"Could not find row '{target_row}' or col 'partial_eta_sq' in ANOVA table")
                    except Exception as e:
                        logger.warning(f"Could not extract ANOVA metrics: {e}")
                    
            elif benchmark_name == "trajectory_analysis":
                # spearman_test.run_trajectory_analysis returns: spearman_corr, spearman_p
                if "spearman_corr" in result:
                    # Store ABSOLUTE value as the metric
                    all_metrics["Spearman_Correlation"] = abs(result["spearman_corr"])
                if "spearman_p" in result:
                    all_metrics["Spearman_pval"] = result["spearman_p"]
        
        logger.info(f"[DEBUG] Collected metrics for summary: {all_metrics}")
        
        if not all_metrics:
            logger.warning("No metrics collected from benchmarks - nothing to save to summary CSV")
            return
        
        # Build column name: just method_name
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
        logger.info(f"✓ Updated summary CSV at: {summary_csv_path} with column '{col_name}'")

    # ------------------------- Benchmark Methods -------------------------

    def run_embedding_visualization(
        self,
        n_components: int = 2,
        figsize: tuple = (8, 6),
        dpi: int = 300,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Visualize embeddings colored by label_col (age).
        """
        logger.info("Running Embedding Visualization...")
        output_dir = self._create_output_dir("embedding_visualization")

        if not self._check_file_exists(self.embedding_csv_path, "Embedding/coordinates CSV file"):
            return {"status": "error", "message": "Missing or invalid embedding CSV path."}

        try:
            # -------------------- LOAD --------------------
            logger.info(f"Loading metadata from: {self.meta_csv_path}")
            meta_df = pd.read_csv(self.meta_csv_path)
            
            # Normalize sample IDs if 'sample' column exists
            if 'sample' in meta_df.columns:
                meta_df['sample'] = self._normalize_sample_ids(meta_df['sample'])
            
            logger.info(f"Loading embeddings from: {self.embedding_csv_path}")
            embedding_df = pd.read_csv(self.embedding_csv_path, index_col=0)
            embedding_df.index = self._normalize_sample_ids(embedding_df.index)

            # -------------------- REQUIREMENTS --------------------
            required_cols = [self.label_col]
            missing_cols = [c for c in required_cols if c not in meta_df.columns]
            if missing_cols:
                return {"status": "error", "message": f"Missing required columns in metadata: {missing_cols}"}

            # -------------------- ALIGN BY SAMPLE ID --------------------
            if 'sample' in meta_df.columns:
                meta_df = meta_df.set_index('sample')

            meta_df.index = self._normalize_sample_ids(meta_df.index)
            embedding_df.index = self._normalize_sample_ids(embedding_df.index)
            
            common_ids = meta_df.index.intersection(embedding_df.index)
            if len(common_ids) == 0:
                raise ValueError("No overlapping sample IDs between metadata and embedding!")

            embedding_df = embedding_df.loc[common_ids]
            meta_df = meta_df.loc[embedding_df.index]

            # -------------------- PCA --------------------
            pca = PCA(n_components=n_components)
            embedding_2d = pca.fit_transform(embedding_df)
            variance_explained = pca.explained_variance_ratio_

            # -------------------- VISUALIZATION --------------------
            fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
            
            label_values_raw = meta_df[self.label_col]
            label_numeric = pd.to_numeric(label_values_raw, errors='coerce')
            is_numerical = label_numeric.notna().sum() / len(label_numeric) > 0.5
            
            if is_numerical:
                # Numerical: use continuous colormap (viridis)
                scatter = ax.scatter(
                    embedding_2d[:, 0],
                    embedding_2d[:, 1],
                    c=label_numeric,
                    cmap='viridis',
                    edgecolors='black',
                    alpha=0.8,
                    s=100,
                    linewidths=0.5
                )
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(f'{self.label_col}', fontsize=10)
            else:
                unique_labels = sorted(label_values_raw.astype(str).unique().tolist())
                n_unique = len(unique_labels)
                label_to_num = {lbl: i for i, lbl in enumerate(unique_labels)}
                label_colors = [label_to_num[str(lbl)] for lbl in label_values_raw]
                
                scatter = ax.scatter(
                    embedding_2d[:, 0],
                    embedding_2d[:, 1],
                    c=label_colors,
                    cmap='tab20',
                    edgecolors='black',
                    alpha=0.8,
                    s=100,
                    linewidths=0.5
                )
                cbar = plt.colorbar(scatter, ax=ax, ticks=range(n_unique))
                cbar.set_label(f'{self.label_col}', fontsize=10)

            ax.set_xlabel(f'PC1 ({variance_explained[0]:.1%})', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'PC2 ({variance_explained[1]:.1%})', fontsize=12, fontweight='bold')
            ax.set_title(f'Embeddings colored by {self.label_col}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')

            plt.tight_layout()
            output_path = output_dir / 'embedding_overview.png'
            plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
            plt.close()

            pca_results = pd.DataFrame(
                embedding_2d,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=embedding_df.index
            )
            pca_path = output_dir / 'pca_coordinates.csv'
            pca_results.to_csv(pca_path)

            result = {
                "variance_explained": variance_explained.tolist(),
                "n_samples": int(embedding_df.shape[0]),
                "output_plot": str(output_path),
                "output_pca": str(pca_path)
            }

            logger.info(f"✓ Embedding visualization completed. Results saved to: {output_dir}")
            return {"status": "success", "output_dir": str(output_dir), "result": result}

        except Exception as e:
            logger.error(f"✗ Error in embedding visualization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}

    def run_trajectory_anova(self, pseudotime_col: str = "pseudotime", **kwargs) -> Dict[str, Any]:
        """
        Run one-way regression ANOVA for pseudotime ~ age.

        Model:
        - Dependent variable: pseudotime
        - Predictor: label_col (e.g., 'age'), treated as a continuous covariate.

        It computes:
        - Standard ANOVA table (statsmodels, Type I)
        - Partial eta-squared for the age effect:
          SS_age / (SS_age + SS_residual)
        """
        logger.info("Running Trajectory ANOVA Analysis (pseudotime ~ age)...")
        output_dir = self._create_output_dir("trajectory_anova")

        if not self._check_file_exists(self.pseudotime_csv_path, "Pseudotime CSV file"):
            return {"status": "error", "message": "Missing or invalid pseudotime CSV path."}

        try:
            # ---------- Load data ----------
            meta_df = pd.read_csv(self.meta_csv_path)
            pseudotime_df = pd.read_csv(self.pseudotime_csv_path)

            # ---------- Basic checks ----------
            if "sample" not in meta_df.columns or "sample" not in pseudotime_df.columns:
                raise ValueError("'sample' column missing in metadata or pseudotime CSV")

            if self.label_col not in meta_df.columns:
                raise ValueError(f"Label column '{self.label_col}' not found in metadata.")

            if pseudotime_col not in pseudotime_df.columns:
                raise ValueError(f"Pseudotime column '{pseudotime_col}' not found in pseudotime CSV.")

            # ---------- Normalize and Merge ----------
            meta_df["sample"] = self._normalize_sample_ids(meta_df["sample"])
            pseudotime_df["sample"] = self._normalize_sample_ids(pseudotime_df["sample"])
            
            merged_df = pd.merge(meta_df, pseudotime_df, on="sample", how="inner")
            clean_df = merged_df[["sample", self.label_col, pseudotime_col]].copy()
            clean_df[self.label_col] = pd.to_numeric(clean_df[self.label_col], errors="coerce")
            clean_df = clean_df.dropna(subset=[self.label_col, pseudotime_col])
            
            n_samples = clean_df.shape[0]
            if n_samples < 3:
                raise ValueError(f"Not enough samples for ANOVA (n={n_samples}).")

            # ---------- Regression ANOVA ----------
            formula = f"{pseudotime_col} ~ {self.label_col}"
            model = ols(formula, data=clean_df).fit()
            anova_table = anova_lm(model, typ=1)

            # effect_row should simply be the label_col name (e.g., 'age')
            effect_row = self.label_col
            if effect_row not in anova_table.index:
                # Fallback logging if something is weird with statsmodels version
                logger.warning(f"Expected row '{effect_row}' not found. Indices: {anova_table.index.tolist()}")
                # Try to guess row
                possible_rows = [i for i in anova_table.index if self.label_col in i]
                if possible_rows:
                    effect_row = possible_rows[0]

            # ---------- Partial eta-squared ----------
            ss_effect = float(anova_table.loc[effect_row, "sum_sq"])
            ss_resid = float(anova_table.loc["Residual", "sum_sq"])
            partial_eta_sq = ss_effect / (ss_effect + ss_resid) if (ss_effect + ss_resid) > 0 else np.nan

            anova_table["partial_eta_sq"] = np.nan
            anova_table.loc[effect_row, "partial_eta_sq"] = partial_eta_sq

            # ---------- Save outputs ----------
            anova_csv_path = output_dir / "trajectory_anova_table.csv"
            anova_table.to_csv(anova_csv_path)

            summary_lines = [
                "TRAJECTORY REGRESSION ANOVA",
                f"Formula: {formula}",
                f"Partial eta-squared for {self.label_col}: {partial_eta_sq:.4f}",
            ]
            with open(output_dir / "trajectory_anova_summary.txt", "w") as f:
                f.write("\n".join(summary_lines))

            result = {
                "anova_table": anova_table,
                "n_samples": int(n_samples),
                "label_col": self.label_col
            }

            logger.info(f"✓ Trajectory ANOVA completed. Eta-sq: {partial_eta_sq:.4f}")
            return {"status": "success", "output_dir": str(output_dir), "result": result}

        except Exception as e:
            logger.error(f"✗ Error in trajectory ANOVA: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}

    def run_trajectory_analysis(self, pseudotime_col: str = "pseudotime", **kwargs) -> Dict[str, Any]:
        """Run trajectory analysis (Spearman correlation)."""
        logger.info("Running Trajectory Analysis (Spearman correlation)...")
        output_dir = self._create_output_dir("trajectory_analysis")

        if not self._check_file_exists(self.pseudotime_csv_path, "Pseudotime CSV file"):
            return {"status": "error", "message": "Missing or invalid pseudotime CSV path."}

        try:
            raw_result = run_trajectory_analysis(
                meta_csv_path=str(self.meta_csv_path),
                pseudotime_csv_path=str(self.pseudotime_csv_path),
                output_dir_path=str(output_dir),
                severity_col=self.label_col,
                pseudotime_col=pseudotime_col,
                **kwargs,
            )

            # Ensure absolute value for metric consistency
            if isinstance(raw_result, dict) and "spearman_corr" in raw_result:
                raw_corr = raw_result["spearman_corr"]
                abs_corr = abs(raw_corr) if raw_corr is not None else None
                raw_result["spearman_corr_raw"] = raw_corr
                raw_result["spearman_corr"] = abs_corr

            logger.info(f"✓ Trajectory analysis completed.")
            return {"status": "success", "output_dir": str(output_dir), "result": raw_result}
        except Exception as e:
            logger.error(f"✗ Error in trajectory analysis: {e}")
            return {"status": "error", "message": str(e)}

    # ------------------------- Orchestration -------------------------

    def run_all_benchmarks(
        self,
        skip_benchmarks: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Dict[str, Any]]:
        """Run all benchmark analyses."""
        skip_benchmarks = skip_benchmarks or []
        results: Dict[str, Dict[str, Any]] = {}

        benchmark_methods = {
            "embedding_visualization": self.run_embedding_visualization,
            "trajectory_anova": self.run_trajectory_anova,
            "trajectory_analysis": self.run_trajectory_analysis,
        }

        for name, method in benchmark_methods.items():
            if name in skip_benchmarks:
                continue

            logger.info(f"Running: {name}")
            method_kwargs = kwargs.get(name, {})
            results[name] = method(**method_kwargs)
            
        self._save_summary_csv(results)
        return results


# ------------------------- Convenience Function -------------------------

def run_benchmarks(
    meta_csv_path: str,
    embedding_csv_path: str,
    method_name: str = "method",
    label_col: str = "age",
    pseudotime_csv_path: Optional[str] = None,
    benchmarks_to_run: Optional[List[str]] = None,
    output_base_dir: Optional[str] = None,
    summary_csv_path: Optional[str] = None,
    **kwargs,
) -> Dict[str, Dict[str, Any]]:
    """Convenience function to run benchmarks."""
    try:
        wrapper = BenchmarkWrapper(
            meta_csv_path=meta_csv_path,
            embedding_csv_path=embedding_csv_path,
            method_name=method_name,
            label_col=label_col,
            pseudotime_csv_path=pseudotime_csv_path,
            output_base_dir=output_base_dir,
            summary_csv_path=summary_csv_path,
        )

        if benchmarks_to_run:
            all_benchmarks = ["embedding_visualization", "trajectory_anova", "trajectory_analysis"]
            skip_benchmarks = [b for b in all_benchmarks if b not in benchmarks_to_run]
            return wrapper.run_all_benchmarks(skip_benchmarks=skip_benchmarks, **kwargs)
        else:
            return wrapper.run_all_benchmarks(**kwargs)

    except Exception as e:
        logger.error(f"✗ Failed to initialize BenchmarkWrapper: {e}")
        return {"initialization_error": {"status": "error", "message": str(e)}}

 
# ------------------------- Usage Examples -------------------------

if __name__ == "__main__":
    
    # Base paths
    base_dir = '/dcs07/hongkai/data/harry/result/Benchmark_eye_rna/retina'
    meta_csv_path = '/dcs07/hongkai/data/harry/result/multi_omics_eye/data/scMultiomics_database.csv'
    summary_csv_path = f'{base_dir}/benchmark_summary_eye_retina.csv'
    
    print("\n" + "=" * 80)
    print("BENCHMARK SUITE - Eye Dataset (Age-based Trajectory Only)")
    print("=" * 80)
    
    # Common parameters for all methods
    common_params = {
        "meta_csv_path": meta_csv_path,
        "summary_csv_path": summary_csv_path,
        "label_col": "age",
        "embedding_visualization": {"dpi": 300, "figsize": (8, 6)},
    }
    
    # List of method configurations
    methods = [
        ("SD_expression", f'{base_dir}/rna/embeddings/sample_expression_embedding.csv', f'{base_dir}/rna/CCA/pseudotime_expression.csv', f'{base_dir}/rna'),
        ("SD_proportion", f'{base_dir}/rna/embeddings/sample_proportion_embedding.csv', f'{base_dir}/rna/CCA/pseudotime_proportion.csv', f'{base_dir}/rna'),
        ("GEDI", f'{base_dir}/GEDI/gedi_sample_embedding.csv', f'{base_dir}/GEDI/trajectory/pseudotime_results.csv', f'{base_dir}/GEDI'),
        ("Gloscope", f'{base_dir}/Gloscope/knn_divergence_mds_10d.csv', f'{base_dir}/Gloscope/trajectory/pseudotime_results.csv', f'{base_dir}/Gloscope'),
        ("MFA", f'{base_dir}/MFA/sample_embeddings.csv', f'{base_dir}/MFA/trajectory/pseudotime_results.csv', f'{base_dir}/MFA'),
        ("pseudobulk", f'{base_dir}/pseudobulk/pseudobulk/pca_embeddings.csv', f'{base_dir}/pseudobulk/pseudobulk/trajectory/pseudotime_results.csv', f'{base_dir}/pseudobulk'),
        ("pilot", f'{base_dir}/pilot/wasserstein_distance_mds_10d.csv', f'{base_dir}/pilot/trajectory/pseudotime_results.csv', f'{base_dir}/pilot'),
        ("QOT", f'{base_dir}/QOT/12_qot_distance_matrix_mds_10d.csv', f'{base_dir}/QOT/trajectory/pseudotime_results.csv', f'{base_dir}/QOT'),
        ("scPoli", f'{base_dir}/scPoli/sample_embeddings_full.csv', f'{base_dir}/scPoli/trajectory/pseudotime_results.csv', f'{base_dir}/scPoli'),
        ("MUSTARD", f'{base_dir}/mustard/sample_embedding.csv', f'{base_dir}/mustard/trajectory/pseudotime_results.csv', f'{base_dir}/mustard')
    ]

    for i, (m_name, m_emb, m_pseudo, m_out) in enumerate(methods):
        print(f"\n[{i+1}/{len(methods)}] Running {m_name}...")
        run_benchmarks(
            embedding_csv_path=m_emb,
            pseudotime_csv_path=m_pseudo,
            method_name=m_name,
            output_base_dir=m_out,
            **common_params
        )
    
    print("\n" + "=" * 80)
    print("ALL BENCHMARKS COMPLETED!")
    print(f"Summary CSV saved to: {summary_csv_path}")
    print("=" * 80 + "\n")