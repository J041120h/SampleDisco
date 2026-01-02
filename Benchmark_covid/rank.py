# -*- coding: utf-8 -*-
"""Method Ranking Analysis - Average Rankings Across Datasets"""

import os
import numpy as np
import pandas as pd

# Configuration
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# File paths
DATA_PATH = "/dcs07/hongkai/data/harry/result/Benchmark_covid/benchmark_summary_all_methods.csv"
OUTPUT_DIR = "/dcs07/hongkai/data/harry/result/Benchmark_covid"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
print("Data loaded. Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Metrics where LARGER is BETTER (descending sort for ranking = rank 1 is best)
DESCENDING_METRICS = {
    "severity_partial_eta_sq",
    "iLISI_norm",
    "ARI",
    "NMI",
    "Avg_Purity",
    "Spearman_Correlation",
    "Custom_ANOVA_eta_sq",
}

# Metrics that use absolute value
ABS_METRICS = {"Spearman_Correlation", "severity_partial_eta_sq"}

# Metrics to exclude from analysis
EXCLUDED_METRICS = {
    "n_samples",
    "Spearman_pval",
    "Custom_ANOVA_omega_sq",
    "interaction_partial_eta_sq",
}


def parse_column_name(col_name):
    """Extract method name and sample size from column name like 'SD_expression-25'"""
    if col_name == "Metric":
        return None, None
    parts = col_name.rsplit("-", 1)  # Split from the right, only once
    if len(parts) == 2:
        method = parts[0]
        try:
            sample_size = int(parts[1])
            return method, sample_size
        except ValueError:
            return None, None
    return None, None


def get_sample_sizes_and_methods(df):
    """Extract all unique sample sizes and methods from column names"""
    sample_sizes = set()
    methods = set()
    
    for col in df.columns:
        if col == "Metric":
            continue
        method, sample_size = parse_column_name(col)
        if method and sample_size:
            methods.add(method)
            sample_sizes.add(sample_size)
    
    return sorted(sample_sizes), sorted(methods)


def calculate_rankings_for_metric(df, metric_name, sample_sizes):
    """
    Calculate rankings for each method at each sample size for a given metric.
    Returns a DataFrame with methods as rows and sample sizes as columns.
    """
    
    # Configuration for this metric
    use_abs = metric_name in ABS_METRICS
    sort_descending = metric_name in DESCENDING_METRICS
    
    # Get the metric row
    metric_row = df[df["Metric"] == metric_name]
    if metric_row.empty:
        print(f"[WARNING] Metric '{metric_name}' not found")
        return None
    
    metric_row = metric_row.iloc[0]
    
    # Store rankings: {method: {sample_size: rank}}
    rankings_dict = {}
    
    for sample_size in sample_sizes:
        # Find all columns for this sample size
        cols_for_size = []
        methods_for_size = []
        values_for_size = []
        
        for col in df.columns:
            if col == "Metric":
                continue
            method, size = parse_column_name(col)
            if method and size == sample_size:
                value = pd.to_numeric(metric_row[col], errors="coerce")
                
                # Skip NaN values
                if pd.isna(value):
                    continue
                
                # Apply absolute value if needed
                if use_abs:
                    value = abs(value)
                
                cols_for_size.append(col)
                methods_for_size.append(method)
                values_for_size.append(value)
        
        if not methods_for_size:
            continue
        
        # Sort and assign ranks
        # For descending: best (highest) gets rank 1
        # For ascending: best (lowest) gets rank 1
        sorted_pairs = sorted(
            zip(methods_for_size, values_for_size),
            key=lambda x: x[1],
            reverse=sort_descending
        )
        
        # Assign rankings (1-based)
        for rank, (method, value) in enumerate(sorted_pairs, start=1):
            if method not in rankings_dict:
                rankings_dict[method] = {}
            rankings_dict[method][sample_size] = rank
    
    # Convert to DataFrame
    rankings_df = pd.DataFrame(rankings_dict).T
    rankings_df.index.name = "Method"
    
    # Sort columns by sample size
    rankings_df = rankings_df[sorted(rankings_df.columns)]
    
    # Add average ranking column
    rankings_df["Average_Rank"] = rankings_df.mean(axis=1)
    
    # Sort by average rank
    rankings_df = rankings_df.sort_values("Average_Rank")
    
    return rankings_df


# Get all sample sizes and methods
sample_sizes, methods = get_sample_sizes_and_methods(df)
print(f"\nFound {len(sample_sizes)} sample sizes: {sample_sizes}")
print(f"Found {len(methods)} methods: {methods}")

# Get all metrics to analyze
all_metrics = df["Metric"].unique()
metrics_to_analyze = [m for m in all_metrics if m not in EXCLUDED_METRICS]

print(f"\nAnalyzing {len(metrics_to_analyze)} metrics:")
for m in metrics_to_analyze:
    direction = "LARGER is better" if m in DESCENDING_METRICS else "SMALLER is better"
    abs_note = " (using absolute value)" if m in ABS_METRICS else ""
    print(f"  - {m}: {direction}{abs_note}")

# Calculate rankings for each metric
all_rankings = {}

print("\n" + "="*80)
print("CALCULATING RANKINGS FOR EACH METRIC")
print("="*80)

for metric in metrics_to_analyze:
    print(f"\n{'='*80}")
    print(f"Metric: {metric}")
    print(f"{'='*80}")
    
    rankings_df = calculate_rankings_for_metric(df, metric, sample_sizes)
    
    if rankings_df is not None:
        all_rankings[metric] = rankings_df
        print(rankings_df)
        
        # Save individual metric ranking to CSV
        safe_name = metric.replace("/", "_").replace(" ", "_")
        output_path = os.path.join(OUTPUT_DIR, f"rankings_{safe_name}.csv")
        rankings_df.to_csv(output_path)
        print(f"\nSaved to: {output_path}")

# Create a summary with average ranks across all methods for each metric
print("\n" + "="*80)
print("SUMMARY: AVERAGE RANK FOR EACH METHOD (BY METRIC)")
print("="*80)

summary_data = {}
for metric, rankings_df in all_rankings.items():
    summary_data[metric] = rankings_df["Average_Rank"]

summary_df = pd.DataFrame(summary_data)
summary_df.index.name = "Method"

# Add overall average rank across all metrics
summary_df["Overall_Avg_Rank"] = summary_df.mean(axis=1)
summary_df = summary_df.sort_values("Overall_Avg_Rank")

print(summary_df)

# Save summary
summary_path = os.path.join(OUTPUT_DIR, "summary_average_rankings.csv")
summary_df.to_csv(summary_path)
print(f"\nSaved summary to: {summary_path}")

# Also create a "best method per metric" summary
print("\n" + "="*80)
print("BEST METHOD FOR EACH METRIC (Rank 1)")
print("="*80)

best_methods = {}
for metric, rankings_df in all_rankings.items():
    best_method = rankings_df.index[0]  # First row (already sorted by Average_Rank)
    best_rank = rankings_df.loc[best_method, "Average_Rank"]
    best_methods[metric] = f"{best_method} (avg rank: {best_rank:.2f})"
    print(f"{metric}: {best_methods[metric]}")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)