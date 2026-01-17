#!/usr/bin/env python3
"""
Benchmark Visualization Script - Multi-Component Version
Generates 4 separate PNG files for assembly in Illustrator:
1. Inverse average ranks bar plot
2. Method names column
3. Main metrics dot heatmap
4. Legends

Usage:
    1. Modify the file paths in the "USER CONFIGURATION" section below
    2. Run: python benchmark_visualization.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
from typing import Dict, List, Tuple
import os
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# USER CONFIGURATION - Modify these paths before running
# =============================================================================

# Input file paths
ENCODE_PATH = "/dcs07/hongkai/data/harry/result/Benchmark_multiomics/summary.csv"
RETINA_PATH = "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/Benchmark_result/summary.csv"
LUTEA_PATH = "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/Benchmark_result/summary.csv"
HEART_PATH = "/dcs07/hongkai/data/harry/result/multi_omics_heart/summary.csv"

# Output settings
OUTPUT_DIR = "/users/hjiang/GenoDistance/figure"  # Output directory (will be created if doesn't exist)
OUTPUT_PREFIX = "benchmark"  # Prefix for output files

# Figure settings
DPI = 300  # Resolution for saved figures
COLORMAP = "viridis"  # Colormap for dots: 'viridis', 'plasma', 'cividis', etc.

# =============================================================================
# END OF USER CONFIGURATION
# =============================================================================


class BenchmarkVisualizer:
    """
    A class to create benchmark visualization plots for method comparison.
    Outputs 4 separate PNG files for Illustrator assembly.
    """
    
    def __init__(self, 
                 encode_path: str, 
                 retina_path: str, 
                 lutea_path: str, 
                 heart_path: str):
        """Initialize the visualizer with data paths."""
        self.data, self.methods = self._load_data(
            encode_path, retina_path, lutea_path, heart_path
        )
        
        # Dataset header colors
        self.dataset_colors = {
            'ENCODE': '#27AE60',
            'Retina': '#E74C3C',
            'Lutea': '#8E44AD',
            'Heart': '#3498DB',
        }
        
        # Method categories for bar coloring
        self.method_categories = {
            'SD': 'SD-based',            # <-- SD_proportion renamed to SD; SD_expression removed
            'pilot': 'Integration',
            'pseudobulk': 'Pseudobulk',
            'QOT': 'Optimal Transport',
            'GEDI': 'Gene Expression',
            'Gloscope': 'Global Structure',
            'MFA': 'Factor Analysis',
            'mustard': 'Multi-task',
            'scPoli': 'Deep Learning'
        }
        
        # Colors for each method category
        self.category_colors = {
            'SD-based': '#1ABC9C',
            'Integration': '#3498DB',
            'Pseudobulk': '#9B59B6',
            'Optimal Transport': '#F1C40F',
            'Gene Expression': '#E67E22',
            'Global Structure': '#95A5A6',
            'Factor Analysis': '#E91E63',
            'Multi-task': '#00BCD4',
            'Deep Learning': '#FF5722',
            'Other': '#BDC3C7',
        }
        
        # Metric display names
        self.metric_display_names = {
            '1/mean_paired_distance': '1/paired_dist',
            'ASW_modality': 'ASW',
            'tissue_preservation_score': 'tissue_pres',
            'cca_score': 'CCA',
            'disease_state_score': 'disease'
        }
        
        # Compute ranks once
        self.ranks_df, self.avg_ranks, self.values_df = self.compute_ranks()
        self.sorted_methods = self.avg_ranks.sort_values().index.tolist()
        self.n_methods = len(self.sorted_methods)
        self.y_positions = np.arange(self.n_methods)[::-1]
    
    def _load_data(self, encode_path: str, retina_path: str, 
                   lutea_path: str, heart_path: str) -> Tuple[Dict, List]:
        """Load and process all CSV files."""
        encode_df = pd.read_csv(encode_path, index_col=0)
        retina_df = pd.read_csv(retina_path, index_col=0)
        lutea_df = pd.read_csv(lutea_path, index_col=0)
        heart_df = pd.read_csv(heart_path, index_col=0)
        
        # === ONLY KEEP SD_proportion, RENAME TO SD, DROP SD_expression ===
        for df in [encode_df, retina_df, lutea_df, heart_df]:
            if 'SD_expression' in df.columns:
                df.drop(columns=['SD_expression'], inplace=True)
            if 'SD_proportion' in df.columns:
                df.rename(columns={'SD_proportion': 'SD'}, inplace=True)
        # Methods list after cleanup (now includes SD instead of SD_proportion, no SD_expression)
        methods = encode_df.columns.tolist()
        # ================================================================
        
        data = {
            'ENCODE': {
                '1/mean_paired_distance': 1.0 / encode_df.loc['mean_paired_distance'].astype(float),
                'ASW_modality': encode_df.loc['ASW_modality'].astype(float),
                'tissue_preservation_score': encode_df.loc['tissue_preservation_score'].astype(float),
            },
            'Retina': {
                '1/mean_paired_distance': 1.0 / retina_df.loc['mean_paired_distance'].astype(float),
                'ASW_modality': retina_df.loc['ASW_modality'].astype(float),
                'cca_score': retina_df.loc['cca_score'].astype(float),
            },
            'Lutea': {
                '1/mean_paired_distance': 1.0 / lutea_df.loc['mean_paired_distance'].astype(float),
                'ASW_modality': lutea_df.loc['ASW_modality'].astype(float),
                'cca_score': lutea_df.loc['cca_score'].astype(float),
            },
            'Heart': {
                '1/mean_paired_distance': 1.0 / heart_df.loc['mean_paired_distance'].astype(float),
                'ASW_modality': heart_df.loc['ASW_modality'].astype(float),
                'disease_state_score': heart_df.loc['disease_state_score'].astype(float),
            }
        }
        
        return data, methods
    
    def compute_ranks(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Compute ranks for each method across all metrics."""
        all_ranks = []
        all_values = []
        index_tuples = []
        
        for dataset_name, metrics in self.data.items():
            for metric_name, values in metrics.items():
                ranks = pd.Series(values).rank(ascending=False, method='min').values
                all_ranks.append(ranks)
                all_values.append(values.values)
                index_tuples.append((dataset_name, metric_name))
        
        multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['Dataset', 'Metric'])
        
        ranks_df = pd.DataFrame(all_ranks, columns=self.methods, index=multi_index)
        values_df = pd.DataFrame(all_values, columns=self.methods, index=multi_index)
        avg_ranks = ranks_df.mean(axis=0)
        
        return ranks_df, avg_ranks, values_df
    
    def plot_bars(self, output_path: str, dpi: int = 300) -> None:
        """
        Plot 1: Horizontal bar chart showing inverse of average ranks.
        """
        fig, ax = plt.subplots(figsize=(4, 8), facecolor='white')
        
        inverse_ranks = 1.0 / self.avg_ranks[self.sorted_methods].values
        
        colors = [self.category_colors.get(
            self.method_categories.get(m, 'Other'), '#BDC3C7'
        ) for m in self.sorted_methods]
        
        ax.barh(self.y_positions, inverse_ranks, color=colors, 
                edgecolor='white', linewidth=0.5, height=0.75)
        
        # Styling
        ax.set_yticks([])
        ax.invert_xaxis()
        ax.set_xlim(max(inverse_ranks) * 1.15, 0)
        ax.set_ylim(-1.5, self.n_methods + 0.5)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(left=False, bottom=True, labelsize=10)
        
        # Header box
        header = FancyBboxPatch(
            (0.0, 1.02), 1.0, 0.05,
            transform=ax.transAxes,
            facecolor='#EAECEE',
            edgecolor='#2C3E50',
            linewidth=1.5,
            boxstyle='round,pad=0.01',
            clip_on=False
        )
        ax.add_patch(header)
        ax.text(0.5, 1.045, 'Inverse of Average Ranks', 
                transform=ax.transAxes, ha='center', va='center',
                fontsize=11, fontweight='bold', color='#2C3E50')
        
        # "Better" arrow
        arrow_x = max(inverse_ranks) * 0.9
        ax.annotate('', xy=(arrow_x, -1.0), xytext=(max(inverse_ranks) * 0.2, -1.0),
                   arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2),
                   annotation_clip=False)
        ax.text(arrow_x * 0.6, -1.35, 'Better', fontsize=10, 
               fontweight='bold', ha='center', color='#2C3E50')
        
        plt.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', transparent=False)
        plt.close(fig)
        print(f"Bar plot saved to: {output_path}")
    
    def plot_method_names(self, output_path: str, dpi: int = 300) -> None:
        """
        Plot 2: Method names column with header.
        """
        fig, ax = plt.subplots(figsize=(2, 8), facecolor='white')
        
        # Method names
        for i, method in enumerate(self.sorted_methods):
            y = self.y_positions[i]
            ax.text(0.5, y, method, ha='center', va='center',
                   fontsize=11, fontweight='medium', color='#2C3E50')
        
        # "Method" header box
        header_y = self.n_methods - 0.3
        header_height = 0.9
        
        method_rect = FancyBboxPatch(
            (-0.3, header_y), 1.6, header_height,
            facecolor='#2C3E50', edgecolor='white', linewidth=2,
            boxstyle='round,pad=0.02', zorder=5
        )
        ax.add_patch(method_rect)
        ax.text(0.5, header_y + header_height / 2, 'Method',
               ha='center', va='center', fontsize=12, 
               fontweight='bold', color='white', zorder=6)
        
        # Styling
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-1.5, self.n_methods + 1.2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', transparent=False)
        plt.close(fig)
        print(f"Method names saved to: {output_path}")
    
    def plot_dots(self, output_path: str, dpi: int = 300, colormap: str = 'viridis') -> None:
        """
        Plot 3: Main balloon/dot heatmap with dataset headers and metric labels.
        """
        datasets = list(self.data.keys())
        
        # Build column info
        columns_info = []
        x_pos = 0
        dataset_ranges = {}
        
        for dataset in datasets:
            start_x = x_pos
            for metric in self.data[dataset].keys():
                columns_info.append((dataset, metric, x_pos))
                x_pos += 1
            dataset_ranges[dataset] = (start_x, x_pos - 1)
        
        n_cols = len(columns_info)
        
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
        cmap = plt.cm.get_cmap(colormap)
        
        # Plot dots
        for dataset, metric, x in columns_info:
            metric_vals = self.values_df.loc[(dataset, metric), self.sorted_methods]
            val_min, val_max = metric_vals.min(), metric_vals.max()
            
            for method_idx, method in enumerate(self.sorted_methods):
                y = self.y_positions[method_idx]
                rank = self.ranks_df.loc[(dataset, metric), method]
                value = self.values_df.loc[(dataset, metric), method]
                
                max_size, min_size = 400, 50
                size = max_size - (rank - 1) * (max_size - min_size) / max(self.n_methods - 1, 1)
                
                norm_val = (value - val_min) / (val_max - val_min) if val_max > val_min else 0.5
                color = cmap(norm_val)
                
                ax.scatter(x, y, s=size, c=[color], 
                          edgecolors='white', linewidths=0.8, zorder=3)
        
        # Column labels (metric names) - LARGER FONT
        for dataset, metric, x in columns_info:
            display_name = self.metric_display_names.get(metric, metric)
            ax.text(x, -1.2, display_name, rotation=45, ha='right', va='top',
                   fontsize=11, color='#2C3E50', fontweight='medium')
        
        # Dataset header boxes
        header_y = self.n_methods - 0.3
        header_height = 0.9
        
        for dataset, (x_start, x_end) in dataset_ranges.items():
            width = x_end - x_start + 0.9
            rect = FancyBboxPatch(
                (x_start - 0.45, header_y), width, header_height,
                facecolor=self.dataset_colors[dataset],
                edgecolor='white', linewidth=2,
                boxstyle='round,pad=0.02', alpha=0.9, zorder=5
            )
            ax.add_patch(rect)
            
            text = ax.text((x_start + x_end) / 2, header_y + header_height / 2,
                          dataset, ha='center', va='center',
                          fontsize=12, fontweight='bold', color='white', zorder=6)
            text.set_path_effects([
                path_effects.withStroke(linewidth=2, foreground='black', alpha=0.3)
            ])
        
        # Grid lines
        for y in self.y_positions:
            ax.axhline(y=y, color='#EAECEE', linestyle='-', linewidth=0.5, zorder=1)
        for x in range(n_cols):
            ax.axvline(x=x, color='#EAECEE', linestyle='-', linewidth=0.5, zorder=1)
        
        # Styling
        ax.set_xlim(-0.6, n_cols - 0.4)
        ax.set_ylim(-2.5, self.n_methods + 1.2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', transparent=False)
        plt.close(fig)
        print(f"Dot plot saved to: {output_path}")
    
    def plot_legends(self, output_path: str, dpi: int = 300) -> None:
        """
        Plot 4: Combined legends (category legend + size legend).
        """
        fig, (ax_cat, ax_size) = plt.subplots(1, 2, figsize=(12, 2), facecolor='white')
        
        # === Category Legend ===
        ax_cat.axis('off')
        
        used_categories = set(self.method_categories.get(m, 'Other') 
                             for m in self.sorted_methods)
        
        legend_elements = []
        for cat in sorted(used_categories):
            color = self.category_colors.get(cat, '#BDC3C7')
            elem = Line2D([0], [0], marker='s', color='w',
                         markerfacecolor=color, markersize=14,
                         markeredgecolor='#2C3E50', markeredgewidth=1,
                         label=cat)
            legend_elements.append(elem)
        
        ncol = min(5, len(legend_elements))
        leg = ax_cat.legend(handles=legend_elements, loc='center', ncol=ncol,
                           frameon=True, fontsize=11, 
                           columnspacing=1.5, handletextpad=0.8,
                           facecolor='white', edgecolor='#EAECEE')
        leg.get_frame().set_linewidth(1.5)
        
        ax_cat.set_title('Method Categories', fontsize=12, fontweight='bold', 
                        color='#2C3E50', pad=10)
        
        # === Size Legend ===
        ax_size.axis('off')
        
        example_ranks = [1, 3, 5, 7, self.n_methods]
        example_ranks = [r for r in example_ranks if r <= self.n_methods]
        
        max_size, min_size = 400, 50
        x_positions = np.linspace(0.1, 0.9, len(example_ranks))
        
        for x, rank in zip(x_positions, example_ranks):
            size = max_size - (rank - 1) * (max_size - min_size) / max(self.n_methods - 1, 1)
            ax_size.scatter(x, 0.4, s=size * 0.8, c='#3498DB', 
                           edgecolors='#2C3E50', linewidths=1,
                           transform=ax_size.transAxes)
            ax_size.text(x, 0.05, f'{int(rank)}', ha='center', fontsize=11,
                        transform=ax_size.transAxes, color='#2C3E50', fontweight='medium')
        
        ax_size.set_title('Rank (1 = Best)', fontsize=12, fontweight='bold',
                         color='#2C3E50', pad=10)
        
        plt.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', transparent=False)
        plt.close(fig)
        print(f"Legends saved to: {output_path}")
    
    def plot_all(self, output_dir: str, prefix: str = "benchmark", 
                 dpi: int = 300, colormap: str = 'viridis') -> None:
        """
        Generate all 4 component plots.
        """
        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        # Generate all components
        self.plot_bars(
            os.path.join(output_dir, f"{prefix}_1_bars.png"), dpi=dpi
        )
        self.plot_method_names(
            os.path.join(output_dir, f"{prefix}_2_methods.png"), dpi=dpi
        )
        self.plot_dots(
            os.path.join(output_dir, f"{prefix}_3_dots.png"), dpi=dpi, colormap=colormap
        )
        self.plot_legends(
            os.path.join(output_dir, f"{prefix}_4_legends.png"), dpi=dpi
        )
        
        print(f"\nAll 4 components saved to: {output_dir}/")
        print("Assembly order in Illustrator (left to right):")
        print("  1. bars -> 2. methods -> 3. dots")
        print("  4. legends (below)")


def main():
    """Main function to generate the benchmark visualization components."""
    print("Loading data...")
    viz = BenchmarkVisualizer(
        encode_path=ENCODE_PATH,
        retina_path=RETINA_PATH,
        lutea_path=LUTEA_PATH,
        heart_path=HEART_PATH
    )
    
    print(f"Found {len(viz.methods)} methods: {viz.methods}")
    print(f"Sorted by performance: {viz.sorted_methods}")
    print("\nGenerating figure components...")
    
    viz.plot_all(
        output_dir=OUTPUT_DIR,
        prefix=OUTPUT_PREFIX,
        dpi=DPI,
        colormap=COLORMAP
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
