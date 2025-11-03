import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging

from ANOVA import run_trajectory_anova_analysis
from batch_removal_test import evaluate_batch_removal
from embedding_effective import evaluate_ari_clustering
from spearman_test import run_trajectory_analysis

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

    # ------------------------- benchmarks -------------------------

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
                "trajectory_anova": {...},
                "batch_removal": {"k": 20},
                "ari_clustering": {"k_neighbors": 30, "n_clusters": 8},
                "trajectory_analysis": {...}
            }

        Returns
        -------
        dict
            Dictionary with results from all benchmarks
        """
        skip_benchmarks = skip_benchmarks or []
        results: Dict[str, Dict[str, Any]] = {}

        benchmark_methods = {
            "trajectory_anova": self.run_trajectory_anova,
            "batch_removal": self.run_batch_removal_evaluation,
            "ari_clustering": self.run_ari_clustering,
            "trajectory_analysis": self.run_trajectory_analysis,
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

    def _save_summary(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Save a summary of all benchmark results."""
        summary_path = self.mode_output_dir / "benchmark_summary.txt"
        with open(summary_path, "w") as f:
            f.write("Benchmark Summary Report\n")
            f.write("========================\n\n")
            f.write(f"Mode label:        {self.mode}\n")
            f.write(f"Meta CSV:          {self.meta_csv_path}\n")
            f.write(f"Pseudotime CSV:    {self.pseudotime_csv_path if self.pseudotime_csv_path else '(not provided)'}\n")
            f.write(f"Embedding CSV:     {self.embedding_csv_path if self.embedding_csv_path else '(not provided)'}\n")
            f.write(f"Output Directory:  {self.mode_output_dir}\n\n")

            for benchmark_name, result in results.items():
                f.write(f"\n{benchmark_name.upper()}\n")
                f.write(f"{'-' * len(benchmark_name)}\n")
                f.write(f"Status: {result.get('status', 'unknown')}\n")
                if result.get("status") == "success":
                    f.write(f"Output: {result.get('output_dir', 'N/A')}\n")
                else:
                    f.write(f"Error: {result.get('message', 'Unknown error')}\n")

        logger.info(f"Summary saved to: {summary_path}")


# ------------------------- convenience function -------------------------

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
        Options: ['trajectory_anova', 'batch_removal', 'ari_clustering', 'trajectory_analysis']
    output_base_dir : str, optional
        Base directory for outputs
    **kwargs : dict
        Per-benchmark kwargs, e.g.:
        {
          "ari_clustering": {"k_neighbors": 30, "n_clusters": 8},
          "batch_removal": {"k": 20, "include_self": True}
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
            all_benchmarks = ["trajectory_anova", "batch_removal", "ari_clustering", "trajectory_analysis"]
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
        pseudotime_csv_path='/users/hjiang/r/GloScope/25_sample/trajectory/pseudotime_results.csv',
        embedding_csv_path="/users/hjiang/r/GloScope/25_sample/knn_divergence_mds_10d.csv",
        mode="expression",
        output_base_dir = '/users/hjiang/r/GloScope/25_sample',
        # per-benchmark overrides (optional)
        ari_clustering={"k_neighbors": 20, "n_clusters": None, "create_plots": True},
        batch_removal={"k": 15, "include_self": False},
    )