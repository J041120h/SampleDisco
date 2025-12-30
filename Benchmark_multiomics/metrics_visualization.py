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
    
    # Get the methods (columns) and metrics (rows)
    methods = df.columns.tolist()
    metrics = df.index.tolist()
    
    # Skip n_samples and n_pairs as they're not performance metrics
    metrics_to_plot = [m for m in metrics if m not in ['n_samples', 'n_pairs']]
    
    # Set up colors for each method
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    
    # Create a figure with subplots
    n_metrics = len(metrics_to_plot)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        values = df.loc[metric].values
        
        bars = ax.bar(methods, values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title(metric, fontsize=11, fontweight='bold')
        ax.set_ylabel('Value', fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7, rotation=0)
        
        ax.set_ylim(0, max(values) * 1.15)
    
    # Hide empty subplots if any
    for idx in range(len(metrics_to_plot), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Performance Comparison Across Methods', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    csv_path = '/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/summary.csv'
    output_dir = '/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina'
    
    plot_benchmark_metrics(csv_path, output_dir)