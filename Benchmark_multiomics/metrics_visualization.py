import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_benchmark_metrics(csv_path, output_dir):
    """
    Generate bar plots comparing performance across different methods.
    
    Parameters:
    -----------
    csv_path : str
        Path to the summary CSV file
    output_dir : str
        Directory to save the output figure
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV file
    df = pd.read_csv(csv_path, index_col=0)
    
    # --- Robust numeric handling ---
    # 1. Strip percent signs if present (e.g., "85%")
    df = df.replace('%', '', regex=True)
    # 2. Try to convert everything to numeric; non-convertible become NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Get the methods (columns) and metrics (rows)
    methods = df.columns.tolist()
    metrics = df.index.tolist()
    
    # Skip n_samples and n_pairs as they're not performance metrics
    metrics_to_plot = [m for m in metrics if m not in ['n_samples', 'n_pairs']]
    
    # Further filter out metrics that are entirely NaN (no numeric data)
    valid_metrics = []
    for m in metrics_to_plot:
        row = df.loc[m]
        if np.isfinite(row.astype(float)).any():
            valid_metrics.append(m)
        else:
            print(f"[WARN] Skipping metric '{m}' because it has no valid numeric values.")
    
    if not valid_metrics:
        print("No valid metrics to plot after cleaning. Exiting without creating a figure.")
        return
    
    # Set up colors for each method
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    
    # Create a figure with subplots
    n_metrics = len(valid_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    
    # If there's only one row/col, axes may not be an array
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = np.array([axes])
    
    for idx, metric in enumerate(valid_metrics):
        ax = axes[idx]
        
        # Get numeric values for this metric
        row = df.loc[metric].astype(float)
        values = row.values
        
        # Replace NaNs with 0.0 for plotting (but annotate as 'NA')
        is_nan = np.isnan(values)
        plot_values = np.where(is_nan, 0.0, values)
        
        bars = ax.bar(methods, plot_values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title(metric, fontsize=11, fontweight='bold')
        ax.set_ylabel('Value', fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        
        # Add value labels on bars
        for bar, val, nan_flag in zip(bars, values, is_nan):
            height = bar.get_height()
            if nan_flag:
                label = 'NA'
            else:
                label = f'{val:.3f}'
            ax.annotate(label,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7, rotation=0)
        
        # Set y-limits robustly
        finite_vals = values[np.isfinite(values)]
        if finite_vals.size > 0:
            max_val = finite_vals.max()
            min_val = finite_vals.min()
            # Ensure some margin; handle constant / zero values gracefully
            if max_val == min_val:
                ax.set_ylim(min_val - 0.1 * (abs(min_val) + 1), max_val + 0.1 * (abs(max_val) + 1))
            else:
                padding = 0.15 * (max_val - min_val)
                ax.set_ylim(min(0, min_val - padding), max_val + padding)
        else:
            # Fallback if everything is NaN/0
            ax.set_ylim(0, 1)
    
    # Hide empty subplots if any
    for idx in range(len(valid_metrics), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Performance Comparison Across Methods', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    csv_path = '/dcs07/hongkai/data/harry/result/Benchmark_long_covid/benchmark_summary_long_covid.csv'
    output_dir = '/dcs07/hongkai/data/harry/result/Benchmark_long_covid'
    
    plot_benchmark_metrics(csv_path, output_dir)
