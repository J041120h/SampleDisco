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
    A comprehensive wrapper for running various benchmark analyses.
    
    Parameters
    ----------
    base_input_path : str
        Base path for input data (e.g., '/dcs07/hongkai/data/harry/result/Benchmark/covid_25_sample/rna/')
    meta_csv_path : str
        Path to metadata CSV file
    mode : str
        Analysis mode - either 'expression' or 'proportion'
    output_base_dir : str, optional
        Base directory for all outputs. If None, uses parent of base_input_path
    """
    
    def __init__(
        self,
        base_input_path: str,
        meta_csv_path: str,
        mode: str = 'expression',
        output_base_dir: Optional[str] = None
    ):
        self.base_input_path = Path(base_input_path).resolve()
        self.meta_csv_path = Path(meta_csv_path).resolve()
        self.mode = mode.lower()
        
        if self.mode not in ['expression', 'proportion']:
            raise ValueError("Mode must be either 'expression' or 'proportion'")
        
        # Validate input paths exist
        if not self.base_input_path.exists():
            error_msg = f"ERROR: Base input path does not exist: {self.base_input_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        if not self.base_input_path.is_dir():
            error_msg = f"ERROR: Base input path is not a directory: {self.base_input_path}"
            logger.error(error_msg)
            raise NotADirectoryError(error_msg)
        
        if not self.meta_csv_path.exists():
            error_msg = f"ERROR: Metadata CSV file does not exist: {self.meta_csv_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        if not self.meta_csv_path.is_file():
            error_msg = f"ERROR: Metadata CSV path is not a file: {self.meta_csv_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Set output base directory
        if output_base_dir is None:
            # Use the parent directory of base_input_path for outputs
            self.output_base_dir = self.base_input_path.parent
        else:
            self.output_base_dir = Path(output_base_dir).resolve()
        
        # Create mode-specific output directory
        self.mode_output_dir = self.output_base_dir / f"benchmark_results_{self.mode}"
        self.mode_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized BenchmarkWrapper with:")
        logger.info(f"  Base input path: {self.base_input_path}")
        logger.info(f"  Meta CSV: {self.meta_csv_path}")
        logger.info(f"  Mode: {self.mode}")
        logger.info(f"  Output directory: {self.mode_output_dir}")
    
    def _get_pseudotime_path(self) -> Path:
        """Get path to pseudotime CSV file based on mode."""
        return self.base_input_path / "CCA" / f"pseudotime_{self.mode}.csv"
    
    def _get_embedding_path(self) -> Path:
        """Get path to embedding/coordinates CSV file based on mode."""
        return (self.base_input_path / "Sample_distance" / "cosine" / 
                f"{self.mode}_DR_distance" / f"{self.mode}_DR_coordinates.csv")
    
    def _create_output_dir(self, benchmark_name: str) -> Path:
        """Create and return output directory for specific benchmark."""
        output_dir = self.mode_output_dir / benchmark_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def _check_file_exists(self, file_path: Path, file_description: str) -> bool:
        """
        Check if a file exists and print detailed error if not.
        
        Parameters
        ----------
        file_path : Path
            Path to check
        file_description : str
            Description of the file for error message
        
        Returns
        -------
        bool
            True if file exists, False otherwise
        """
        if not file_path.exists():
            error_msg = f"ERROR: {file_description} not found!"
            logger.error(error_msg)
            logger.error(f"  Expected path: {file_path}")
            logger.error(f"  Parent directory exists: {file_path.parent.exists()}")
            
            if file_path.parent.exists():
                logger.error(f"  Contents of parent directory:")
                try:
                    for item in file_path.parent.iterdir():
                        logger.error(f"    - {item.name}")
                except Exception as e:
                    logger.error(f"    Could not list directory contents: {e}")
            else:
                logger.error(f"  Parent directory does not exist: {file_path.parent}")
                logger.error(f"  Please check the directory structure.")
            
            return False
        
        return True
    
    def run_trajectory_anova(self, **kwargs) -> Dict[str, Any]:
        """
        Run trajectory ANOVA analysis.
        
        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to run_trajectory_anova_analysis
        
        Returns
        -------
        dict
            Results dictionary with status and output path
        """
        logger.info("Running Trajectory ANOVA Analysis...")
        
        pseudotime_path = self._get_pseudotime_path()
        output_dir = self._create_output_dir("trajectory_anova")
        
        if not self._check_file_exists(pseudotime_path, "Pseudotime CSV file"):
            return {
                "status": "error", 
                "message": f"Pseudotime file not found: {pseudotime_path}"
            }
        
        try:
            result = run_trajectory_anova_analysis(
                meta_csv_path=str(self.meta_csv_path),
                pseudotime_csv_path=str(pseudotime_path),
                output_dir_path=str(output_dir),
                **kwargs
            )
            
            logger.info(f"Trajectory ANOVA completed. Results saved to: {output_dir}")
            return {
                "status": "success",
                "output_dir": str(output_dir),
                "result": result
            }
        except Exception as e:
            logger.error(f"Error in trajectory ANOVA: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def run_batch_removal_evaluation(self, k: int = 15, include_self: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Evaluate batch removal performance.
        
        Parameters
        ----------
        k : int
            Number of neighbors for KNN
        include_self : bool
            Whether to include self in KNN
        **kwargs : dict
            Additional parameters to pass to evaluate_batch_removal
        
        Returns
        -------
        dict
            Results dictionary with status and output path
        """
        logger.info("Running Batch Removal Evaluation...")
        
        embedding_path = self._get_embedding_path()
        output_dir = self._create_output_dir("batch_removal")
        
        if not self._check_file_exists(embedding_path, "Embedding/coordinates CSV file"):
            return {
                "status": "error", 
                "message": f"Embedding file not found: {embedding_path}"
            }
        
        try:
            result = evaluate_batch_removal(
                meta_csv=str(self.meta_csv_path),
                data_csv=str(embedding_path),
                mode="embedding",
                outdir=str(output_dir),
                k=k,
                include_self=include_self,
                **kwargs
            )
            
            logger.info(f"Batch removal evaluation completed. Results saved to: {output_dir}")
            return {
                "status": "success",
                "output_dir": str(output_dir),
                "result": result
            }
        except Exception as e:
            logger.error(f"Error in batch removal evaluation: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def run_ari_clustering(self, k_neighbors: int = 15, n_clusters: Optional[int] = None, 
                           create_plots: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Run ARI clustering evaluation.
        
        Parameters
        ----------
        k_neighbors : int
            Number of neighbors for KNN
        n_clusters : int or None
            Number of clusters (None for auto-detection)
        create_plots : bool
            Whether to create visualization plots
        **kwargs : dict
            Additional parameters to pass to evaluate_ari_clustering
        
        Returns
        -------
        dict
            Results dictionary with status and output path
        """
        logger.info("Running ARI Clustering Evaluation...")
        
        embedding_path = self._get_embedding_path()
        output_dir = self._create_output_dir("ari_clustering")
        
        if not self._check_file_exists(embedding_path, "Embedding/coordinates CSV file"):
            return {
                "status": "error", 
                "message": f"Embedding file not found: {embedding_path}"
            }
        
        try:
            result = evaluate_ari_clustering(
                meta_csv=str(self.meta_csv_path),
                data_csv=str(embedding_path),
                mode="embedding",
                outdir=str(output_dir),
                k_neighbors=k_neighbors,
                n_clusters=n_clusters,
                create_plots=create_plots,
                **kwargs
            )
            
            logger.info(f"ARI clustering evaluation completed. Results saved to: {output_dir}")
            return {
                "status": "success",
                "output_dir": str(output_dir),
                "result": result
            }
        except Exception as e:
            logger.error(f"Error in ARI clustering evaluation: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def run_trajectory_analysis(self, **kwargs) -> Dict[str, Any]:
        """
        Run trajectory analysis.
        
        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to run_trajectory_analysis
        
        Returns
        -------
        dict
            Results dictionary with status and output path
        """
        logger.info("Running Trajectory Analysis...")
        
        pseudotime_path = self._get_pseudotime_path()
        output_dir = self._create_output_dir("trajectory_analysis")
        
        if not self._check_file_exists(pseudotime_path, "Pseudotime CSV file"):
            return {
                "status": "error", 
                "message": f"Pseudotime file not found: {pseudotime_path}"
            }
        
        try:
            result = run_trajectory_analysis(
                meta_csv_path=str(self.meta_csv_path),
                pseudotime_csv_path=str(pseudotime_path),
                output_dir_path=str(output_dir),
                **kwargs
            )
            
            logger.info(f"Trajectory analysis completed. Results saved to: {output_dir}")
            return {
                "status": "success",
                "output_dir": str(output_dir),
                "result": result
            }
        except Exception as e:
            logger.error(f"Error in trajectory analysis: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def run_all_benchmarks(self, 
                          skip_benchmarks: Optional[List[str]] = None,
                          **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Run all benchmark analyses.
        
        Parameters
        ----------
        skip_benchmarks : list of str, optional
            List of benchmark names to skip
        **kwargs : dict
            Parameters to pass to individual benchmarks
        
        Returns
        -------
        dict
            Dictionary with results from all benchmarks
        """
        skip_benchmarks = skip_benchmarks or []
        results = {}
        
        benchmark_methods = {
            "trajectory_anova": self.run_trajectory_anova,
            "batch_removal": self.run_batch_removal_evaluation,
            "ari_clustering": self.run_ari_clustering,
            "trajectory_analysis": self.run_trajectory_analysis
        }
        
        for name, method in benchmark_methods.items():
            if name in skip_benchmarks:
                logger.info(f"Skipping {name}...")
                continue
            
            logger.info(f"\n{'='*50}")
            logger.info(f"Running {name}...")
            logger.info(f"{'='*50}")
            
            # Extract method-specific kwargs
            method_kwargs = kwargs.get(name, {})
            results[name] = method(**method_kwargs)
        
        # Save summary
        self._save_summary(results)
        
        return results
    
    def _save_summary(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Save a summary of all benchmark results."""
        summary_path = self.mode_output_dir / "benchmark_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write(f"Benchmark Summary Report\n")
            f.write(f"========================\n\n")
            f.write(f"Mode: {self.mode}\n")
            f.write(f"Base Input Path: {self.base_input_path}\n")
            f.write(f"Meta CSV: {self.meta_csv_path}\n")
            f.write(f"Output Directory: {self.mode_output_dir}\n\n")
            
            for benchmark_name, result in results.items():
                f.write(f"\n{benchmark_name.upper()}\n")
                f.write(f"{'-'*len(benchmark_name)}\n")
                f.write(f"Status: {result.get('status', 'unknown')}\n")
                
                if result.get('status') == 'success':
                    f.write(f"Output: {result.get('output_dir', 'N/A')}\n")
                else:
                    f.write(f"Error: {result.get('message', 'Unknown error')}\n")
        
        logger.info(f"Summary saved to: {summary_path}")


# Convenience function for quick benchmark runs
def run_benchmarks(
    base_input_path: str,
    meta_csv_path: str,
    mode: str = 'expression',
    benchmarks_to_run: Optional[List[str]] = None,
    output_base_dir: Optional[str] = None,
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to run benchmarks.
    
    Parameters
    ----------
    base_input_path : str
        Base path for input data
    meta_csv_path : str
        Path to metadata CSV file
    mode : str
        'expression' or 'proportion'
    benchmarks_to_run : list of str, optional
        Specific benchmarks to run. If None, runs all.
        Options: ['trajectory_anova', 'batch_removal', 'ari_clustering', 'trajectory_analysis']
    output_base_dir : str, optional
        Base directory for outputs
    **kwargs : dict
        Additional parameters for specific benchmarks
    
    Returns
    -------
    dict
        Results from all benchmarks
    
    Examples
    --------
    # Run all benchmarks for expression mode
    results = run_benchmarks(
        base_input_path="/dcs07/hongkai/data/harry/result/Benchmark/covid_25_sample/rna/",
        meta_csv_path="/dcl01/hongkai/data/data/hjiang/Data/covid_data/sample_data.csv",
        mode="expression"
    )
    
    # Run specific benchmarks with custom parameters
    results = run_benchmarks(
        base_input_path="/path/to/data/",
        meta_csv_path="/path/to/meta.csv",
        mode="proportion",
        benchmarks_to_run=["trajectory_anova", "ari_clustering"],
        trajectory_anova={"param1": value1},
        ari_clustering={"k_neighbors": 20, "create_plots": True}
    )
    """
    try:
        wrapper = BenchmarkWrapper(
            base_input_path=base_input_path,
            meta_csv_path=meta_csv_path,
            mode=mode,
            output_base_dir=output_base_dir
        )
        
        if benchmarks_to_run:
            # Run only specified benchmarks
            all_benchmarks = ["trajectory_anova", "batch_removal", "ari_clustering", "trajectory_analysis"]
            skip_benchmarks = [b for b in all_benchmarks if b not in benchmarks_to_run]
            return wrapper.run_all_benchmarks(skip_benchmarks=skip_benchmarks, **kwargs)
        else:
            # Run all benchmarks
            return wrapper.run_all_benchmarks(**kwargs)
    
    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        logger.error(f"Failed to initialize BenchmarkWrapper: {e}")
        return {"initialization_error": {"status": "error", "message": str(e)}}


# Example usage
if __name__ == "__main__":
    # Example 1: Run all benchmarks for expression mode
    results = run_benchmarks(
        base_input_path="/dcs07/hongkai/data/harry/result/Benchmark_covid/covid_400_sample/rna",
        meta_csv_path="/dcl01/hongkai/data/data/hjiang/Data/covid_data/sample_data.csv",
        mode="expression"
    )
    
    # Example 2: Run all benchmarks for proportion mode
    # results_prop = run_benchmarks(
    #     base_input_path="/dcs07/hongkai/data/harry/result/Benchmark/covid_25_sample/rna/",
    #     meta_csv_path="/dcl01/hongkai/data/data/hjiang/Data/covid_data/sample_data.csv",
    #     mode="proportion"
    # )