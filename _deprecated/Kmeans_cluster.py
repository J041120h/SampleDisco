import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

def cluster_samples_from_folder(folder_path: str, n_clusters: int):
    """
    Cluster samples using KMeans separately on expression and proportion data,
    and return two sample-to-clade mappings for valid clades (with ≥ 2 samples).
    
    Args:
        folder_path (str): Path to folder with 'expression.csv' and 'proportion.csv'.
        n_clusters (int): Number of clusters.
        
    Returns:
        tuple: (expression_sample_to_clade, proportion_sample_to_clade) - two dictionaries
               with sample-to-clade mappings (only for samples in valid clades).
    """
    # Load CSVs
    expr_path = os.path.join(folder_path, "expression.csv")
    prop_path = os.path.join(folder_path, "proportion.csv")
    
    expression_df = pd.read_csv(expr_path, index_col=0)
    proportion_df = pd.read_csv(prop_path, index_col=0)
    
    # Handle different orientation of data
    # In expression data, samples are rows
    # In proportion data, samples are columns, so we need to transpose
    proportion_df = proportion_df.T
    
    # Strip whitespace from sample names
    expression_df.index = expression_df.index.str.strip()
    proportion_df.index = proportion_df.index.str.strip()
    
    # Check if n_clusters is valid (at least 2 and not more than number of samples/2)
    expr_sample_count = len(expression_df)
    prop_sample_count = len(proportion_df)
    
    min_sample_count = min(expr_sample_count, prop_sample_count)
    max_possible_clusters = min_sample_count // 2
    
    if n_clusters < 2:
        raise ValueError(f"Number of clusters must be at least 2, but got {n_clusters}")
    
    if n_clusters > max_possible_clusters:
        raise ValueError(f"Number of clusters must not exceed half the number of samples. "
                         f"With {min_sample_count} samples, maximum allowed clusters is {max_possible_clusters}, "
                         f"but got {n_clusters}")
    
    # Convert all column names to strings to avoid mixed types error
    expression_df.columns = expression_df.columns.astype(str)
    proportion_df.columns = proportion_df.columns.astype(str)
    
    # Run KMeans on expression data
    expr_kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    expr_labels = expr_kmeans.fit_predict(expression_df)
    
    # Count how many in each expression cluster
    expr_label_counts = Counter(expr_labels)
    
    # Keep only valid clusters (>=2 members) for expression
    expr_sample_to_clade = {
        sample: int(label)
        for sample, label in zip(expression_df.index, expr_labels)
        if expr_label_counts[label] >= 2
    }
    
    # Check if we have enough valid groups after filtering
    valid_expr_clusters = len(set(expr_sample_to_clade.values()))
    if valid_expr_clusters < n_clusters:
        print(f"Warning: After filtering out singleton clusters from expression data, "
              f"only {valid_expr_clusters} valid clusters remain instead of the requested {n_clusters}")
    
    if valid_expr_clusters < 2:
        raise ValueError(f"After filtering out singleton clusters from expression data, "
                         f"less than 2 valid clusters remain ({valid_expr_clusters}). Cannot proceed.")
    
    # Run KMeans on proportion data
    prop_kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    prop_labels = prop_kmeans.fit_predict(proportion_df)
    
    # Count how many in each proportion cluster
    prop_label_counts = Counter(prop_labels)
    
    # Keep only valid clusters (>=2 members) for proportion
    prop_sample_to_clade = {
        sample: int(label)
        for sample, label in zip(proportion_df.index, prop_labels)
        if prop_label_counts[label] >= 2
    }
    
    # Check if we have enough valid groups after filtering
    valid_prop_clusters = len(set(prop_sample_to_clade.values()))
    if valid_prop_clusters < n_clusters:
        print(f"Warning: After filtering out singleton clusters from proportion data, "
              f"only {valid_prop_clusters} valid clusters remain instead of the requested {n_clusters}")
    
    if valid_prop_clusters < 2:
        raise ValueError(f"After filtering out singleton clusters from proportion data, "
                         f"less than 2 valid clusters remain ({valid_prop_clusters}). Cannot proceed.")
    
    return expr_sample_to_clade, prop_sample_to_clade
