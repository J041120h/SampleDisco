# -*- coding: utf-8 -*-
"""
Method Ranking Analysis - Paper-Style Aggregation

VERSION: 2.1.0
DATE: 2025-02-09

UPDATED to match the BenchmarkVisualizer ranking methodology:
- Transforms metrics where "smaller is better" by inversion (1/value)
- Ranks methods within each sample-size variant first
- Averages ranks across variants to get single rank per metric
- Supports multiple dataset groups with equal weighting
- Computes group-level mean ranks, then averages across groups

This ensures consistency with the paper-style aggregation used in the
benchmark visualization script.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# File paths - can be extended to multiple datasets
COVID_PATH = "/dcs07/hongkai/data/harry/result/Benchmark_covid/ALL_BENCHMARK_OUTPUTS/benchmark_summary_all_methods.csv"
LONG_COVID_PATH = "/dcs07/hongkai/data/harry/result/Benchmark_long_covid/benchmark_summary_long_covid.csv"  # Optional

OUTPUT_DIR = "/dcs07/hongkai/data/harry/result/Benchmark_covid/ALL_BENCHMARK_OUTPUTS"

# Set to True if you have Long COVID data, False for COVID only
USE_LONG_COVID = False

# Verbose output
VERBOSE = True

# =============================================================================
# METRIC CONFIGURATION
# =============================================================================

# Metrics where LARGER is BETTER (no inversion needed)
ASCENDING_METRICS = {
    "iLISI_norm",
    "severity_partial_eta_sq",
    "One_way_ANOVA_eta_sq",
    "Spearman_Correlation",
    "Custom_ANOVA_eta_sq",
    "Month_Preservation_Score",
    "ARI",
    "NMI",
    "Avg_Purity",
}

# Metrics where SMALLER is BETTER (will be inverted: 1/value)
DESCENDING_METRICS = {
    "batch_partial_eta_sq",
    "ASW_batch",
    "Mean_NN_Severity_Gap",
}

# Metrics to include per dataset group
COVID_METRICS = [
    "batch_partial_eta_sq",
    "iLISI_norm",
    "ASW_batch",
    "severity_partial_eta_sq",
    "Spearman_Correlation",
    "Custom_ANOVA_eta_sq",
    "ARI",
    "NMI",
    "Avg_Purity",
    "Mean_NN_Severity_Gap",
]

LONG_COVID_METRICS = [
    "One_way_ANOVA_eta_sq",
    "Spearman_Correlation",
    "Month_Preservation_Score",
]

# Metrics to exclude from analysis
EXCLUDED_METRICS = {
    "n_samples",
    "Spearman_pval",
    "Custom_ANOVA_omega_sq",
    "interaction_partial_eta_sq",
}

# Methods to exclude entirely
EXCLUDED_METHODS = {
    "fusion_mfa",
    "fusion_concat",
    "cell_embedding_pseudobulk",
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def log(message: str, verbose: bool = True):
    """Print log message with timestamp if verbose mode is on."""
    if verbose:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")


def parse_column_name(col_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract method name and sample size variant from column name.
    
    Examples:
        'SD_expression-25' -> ('SD_expression', '25')
        'GEDI-25' -> ('GEDI', '25')
        'Metric' -> (None, None)
    
    Returns:
        Tuple of (method_name, variant) or (None, None) if invalid
    """
    if col_name == "Metric":
        return None, None
    
    # Split from the right to handle method names with hyphens
    if "-" in col_name:
        parts = col_name.rsplit("-", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0], parts[1]
    
    return col_name, None


def transform_value(value: float, metric: str) -> float:
    """
    Transform metric value so that LARGER is always BETTER.
    
    For metrics where smaller is better, we invert (1/value).
    This matches the BenchmarkVisualizer approach.
    """
    if pd.isna(value):
        return np.nan
    
    if metric in DESCENDING_METRICS:
        # Smaller is better -> invert so larger is better
        if value != 0:
            return 1.0 / value
        else:
            return np.nan
    else:
        # Larger is already better
        return value


# =============================================================================
# MAIN RANKING CLASS
# =============================================================================

class PaperStyleRankingAnalyzer:
    """
    Ranking analyzer that matches the BenchmarkVisualizer methodology.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.raw_df: Optional[pd.DataFrame] = None
        self.methods: List[str] = []
        self.variants: List[str] = []
        self.metrics: List[str] = []
        self.dataset_name: str = ""
        
        # Results
        self.ranks_df: Optional[pd.DataFrame] = None
        self.values_df: Optional[pd.DataFrame] = None
        self.avg_ranks: Optional[pd.Series] = None
        self.metric_ranks: Dict[str, pd.Series] = {}
    
    def load_csv(self, file_path: str, dataset_name: str, metrics_to_use: List[str]) -> None:
        """
        Load data from CSV file.
        
        CSV Format expected:
        - First column: 'Metric' containing metric names
        - Other columns: 'MethodName-SampleSize' (e.g., 'SD_expression-25')
        """
        log("\n" + "="*80, self.verbose)
        log("Loading Data", self.verbose)
        log("="*80, self.verbose)
        
        log(f"\nLoading from {file_path}...", self.verbose)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.raw_df = pd.read_csv(file_path)
        self.dataset_name = dataset_name
        
        log(f"  Shape: {self.raw_df.shape}", self.verbose)
        log(f"  Columns: {list(self.raw_df.columns)[:5]}... (showing first 5)", self.verbose)
        
        # Parse all columns to extract methods and variants
        method_variant_pairs = []
        for col in self.raw_df.columns:
            if col == "Metric":
                continue
            method, variant = parse_column_name(col)
            if method is not None and method not in EXCLUDED_METHODS:
                method_variant_pairs.append((method, variant, col))
        
        # Get unique methods and variants
        self.methods = sorted(set(m for m, v, c in method_variant_pairs if m is not None))
        self.variants = sorted(set(v for m, v, c in method_variant_pairs if v is not None))
        
        # Filter metrics
        available_metrics = self.raw_df["Metric"].tolist()
        self.metrics = [m for m in metrics_to_use 
                       if m in available_metrics and m not in EXCLUDED_METRICS]
        
        log(f"\n✓ Data loading complete:", self.verbose)
        log(f"  Methods found: {self.methods}", self.verbose)
        log(f"  Variants (sample sizes): {self.variants}", self.verbose)
        log(f"  Metrics to analyze: {self.metrics}", self.verbose)
        
        # Show which requested metrics were not found
        missing_metrics = [m for m in metrics_to_use 
                         if m not in available_metrics and m not in EXCLUDED_METRICS]
        if missing_metrics:
            log(f"  ⚠ Metrics not found in CSV: {missing_metrics}", self.verbose)
    
    def compute_ranks(self) -> None:
        """
        Compute ranks using paper-style aggregation.
        
        Process:
        1. For each metric, extract values for all method-variant combinations
        2. Rank methods within each variant (sample size)
        3. Average ranks across variants to get single rank per method per metric
        4. Compute overall average rank across all metrics
        """
        log("\n" + "="*80, self.verbose)
        log("Computing Ranks (Paper-Style Aggregation)", self.verbose)
        log("="*80, self.verbose)
        
        # Store ranks for each metric
        all_metric_ranks = {}
        all_metric_values = {}
        
        for metric in self.metrics:
            log(f"\n  Processing metric: {metric}", self.verbose)
            
            # Get the row for this metric
            metric_row = self.raw_df[self.raw_df["Metric"] == metric]
            if metric_row.empty:
                log(f"    ⚠ Metric '{metric}' not found - skipping", self.verbose)
                continue
            metric_row = metric_row.iloc[0]
            
            # Build a matrix: rows = variants, columns = methods
            # Each cell contains the (transformed) value for that method at that sample size
            values_matrix = pd.DataFrame(index=self.variants, columns=self.methods, dtype=float)
            
            for variant in self.variants:
                for method in self.methods:
                    col_name = f"{method}-{variant}"
                    if col_name in self.raw_df.columns:
                        raw_value = pd.to_numeric(metric_row[col_name], errors="coerce")
                        transformed_value = transform_value(raw_value, metric)
                        values_matrix.loc[variant, method] = transformed_value
                    else:
                        values_matrix.loc[variant, method] = np.nan
            
            # Rank within each variant (row)
            # Higher transformed value = better = rank 1
            ranks_matrix = values_matrix.apply(
                lambda row: row.rank(ascending=False, method="min", na_option="bottom"),
                axis=1
            )
            
            # Average ranks across variants for each method
            mean_ranks = ranks_matrix.mean(axis=0)
            mean_values = values_matrix.mean(axis=0, skipna=True)
            
            all_metric_ranks[metric] = mean_ranks
            all_metric_values[metric] = mean_values
            
            # Log best method for this metric
            best_method = mean_ranks.idxmin()
            log(f"    Best method: {best_method} (avg rank = {mean_ranks[best_method]:.2f})", self.verbose)
        
        # Create DataFrames: rows = metrics, columns = methods
        self.ranks_df = pd.DataFrame(all_metric_ranks).T
        self.values_df = pd.DataFrame(all_metric_values).T
        
        # Store per-metric ranks
        self.metric_ranks = all_metric_ranks
        
        # Compute overall average rank across all metrics
        self.avg_ranks = self.ranks_df.mean(axis=0)
        
        log(f"\n✓ Ranking complete", self.verbose)
        log(f"  Ranks DataFrame shape: {self.ranks_df.shape}", self.verbose)
        
        # Show top 3 overall
        sorted_avg = self.avg_ranks.sort_values()
        log(f"\n  Top 3 methods overall:", self.verbose)
        for i, (method, rank) in enumerate(sorted_avg.head(3).items(), 1):
            log(f"    {i}. {method}: avg_rank = {rank:.3f}", self.verbose)
    
    def get_rankings_by_variant(self, metric: str) -> pd.DataFrame:
        """
        Get rankings for each variant separately (before averaging).
        
        Returns:
            DataFrame with methods as rows and variants as columns
        """
        if metric not in self.metrics:
            raise ValueError(f"Metric '{metric}' not found")
        
        # Get the row for this metric
        metric_row = self.raw_df[self.raw_df["Metric"] == metric].iloc[0]
        
        # Build values matrix
        values_matrix = pd.DataFrame(index=self.variants, columns=self.methods, dtype=float)
        
        for variant in self.variants:
            for method in self.methods:
                col_name = f"{method}-{variant}"
                if col_name in self.raw_df.columns:
                    raw_value = pd.to_numeric(metric_row[col_name], errors="coerce")
                    transformed_value = transform_value(raw_value, metric)
                    values_matrix.loc[variant, method] = transformed_value
        
        # Rank within each variant
        ranks_matrix = values_matrix.apply(
            lambda row: row.rank(ascending=False, method="min", na_option="bottom"),
            axis=1
        )
        
        # Transpose so methods are rows, variants are columns
        result = ranks_matrix.T.copy()
        result["Average_Rank"] = result.mean(axis=1)
        result = result.sort_values("Average_Rank")
        
        return result
    
    def print_summary(self) -> None:
        """Print summary of rankings."""
        if self.avg_ranks is None:
            raise ValueError("Must call compute_ranks() first")
        
        print("\n" + "="*80)
        print("RANKING SUMMARY")
        print("="*80)
        
        # Overall rankings
        print(f"\n📊 OVERALL RANKINGS ({self.dataset_name})")
        print("-"*50)
        sorted_ranks = self.avg_ranks.sort_values()
        for i, (method, rank) in enumerate(sorted_ranks.items(), 1):
            print(f"  {i:2d}. {method:<25} avg_rank = {rank:.3f}")
        
        # Best method per metric
        print("\n📊 BEST METHOD PER METRIC")
        print("-"*50)
        for metric in self.metrics:
            if metric in self.metric_ranks:
                ranks = self.metric_ranks[metric]
                best_method = ranks.idxmin()
                best_rank = ranks[best_method]
                print(f"  {metric:<30} -> {best_method} (rank={best_rank:.2f})")
    
    def save_results(self, output_dir: str, prefix: str = "ranking") -> None:
        """
        Save all ranking results to CSV files.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        log("\n" + "="*80, self.verbose)
        log("Saving Results", self.verbose)
        log("="*80, self.verbose)
        
        # 1. Overall rankings
        overall_df = pd.DataFrame({
            "Method": self.avg_ranks.sort_values().index,
            "Rank_Position": range(1, len(self.avg_ranks) + 1),
            "Average_Rank": self.avg_ranks.sort_values().values
        })
        overall_path = os.path.join(output_dir, f"{prefix}_overall.csv")
        overall_df.to_csv(overall_path, index=False)
        log(f"  ✓ Overall rankings: {overall_path}", self.verbose)
        
        # 2. Rankings by metric (methods as rows, metrics as columns)
        by_metric_df = self.ranks_df.T.copy()
        by_metric_df["Overall_Avg_Rank"] = self.avg_ranks
        by_metric_df = by_metric_df.sort_values("Overall_Avg_Rank")
        by_metric_path = os.path.join(output_dir, f"{prefix}_by_metric.csv")
        by_metric_df.to_csv(by_metric_path, index_label="Method")
        log(f"  ✓ Rankings by metric: {by_metric_path}", self.verbose)
        
        # 3. Detailed rankings with values
        detailed_rows = []
        for metric in self.metrics:
            if metric not in self.metric_ranks:
                continue
            
            # Determine direction
            if metric in DESCENDING_METRICS:
                direction = "smaller_is_better (inverted)"
            else:
                direction = "larger_is_better"
            
            for method in self.methods:
                detailed_rows.append({
                    "Dataset": self.dataset_name,
                    "Metric": metric,
                    "Method": method,
                    "Rank": self.ranks_df.loc[metric, method] if metric in self.ranks_df.index else np.nan,
                    "Transformed_Value": self.values_df.loc[metric, method] if metric in self.values_df.index else np.nan,
                    "Direction": direction,
                })
        
        detailed_df = pd.DataFrame(detailed_rows)
        detailed_df = detailed_df.sort_values(["Dataset", "Metric", "Rank"])
        detailed_path = os.path.join(output_dir, f"{prefix}_detailed.csv")
        detailed_df.to_csv(detailed_path, index=False)
        log(f"  ✓ Detailed rankings: {detailed_path}", self.verbose)
        
        # 4. Full ranks matrix (metrics as rows, methods as columns)
        ranks_path = os.path.join(output_dir, f"{prefix}_ranks_matrix.csv")
        self.ranks_df.to_csv(ranks_path, index_label="Metric")
        log(f"  ✓ Ranks matrix: {ranks_path}", self.verbose)
        
        log(f"\n✓ All results saved to: {output_dir}", self.verbose)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("\n" + "="*80)
    print("Method Ranking Analysis - Paper-Style Aggregation")
    print("="*80 + "\n")
    
    # Initialize analyzer
    analyzer = PaperStyleRankingAnalyzer(verbose=VERBOSE)
    
    # Load COVID data
    analyzer.load_csv(COVID_PATH, "COVID", COVID_METRICS)
    
    # Compute ranks
    analyzer.compute_ranks()
    
    # Print summary
    analyzer.print_summary()
    
    # Save results
    analyzer.save_results(OUTPUT_DIR, prefix="paper_style_ranking")
    
    # Additional detailed output
    print("\n" + "="*80)
    print("DETAILED RANKINGS BY METRIC")
    print("="*80)
    
    # Show ranks for each metric
    print("\nRanks by Metric (rows=methods, columns=metrics):")
    print(analyzer.ranks_df.T.round(2).to_string())
    
    # Show variant-level detail for one metric as example
    print("\n" + "="*80)
    print("EXAMPLE: Rankings by Variant (iLISI_norm)")
    print("="*80)
    
    if "iLISI_norm" in analyzer.metrics:
        variant_rankings = analyzer.get_rankings_by_variant("iLISI_norm")
        print(variant_rankings.round(3).to_string())
    
    print("\n" + "="*80)
    print("✓ Analysis Complete")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()