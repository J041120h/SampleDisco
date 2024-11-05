import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage

def plot_cell_type_abundances(proportions: pd.DataFrame, output_dir: str):
    """
    Generate a stacked bar plot to visualize the cell type proportions across samples.

    Parameters:
    ----------
    proportions : pd.DataFrame
        DataFrame containing cell type proportions for each sample.
        Rows represent samples, and columns represent cell types.
    output_dir : str
        Directory to save the output plot.

    Returns:
    -------
    None
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Automatically generating output directory")

    # Sort the samples for consistent plotting
    proportions = proportions.sort_index()

    # Define the order of cell types (optional: you can sort or specify a custom order)
    cell_types = proportions.columns.tolist()

    # Define a color palette with enough colors for all cell types
    num_cell_types = len(cell_types)
    colors = sns.color_palette('tab20', n_colors=num_cell_types)

    # Create a figure and axis
    plt.figure(figsize=(12, 8))

    # Plot stacked bar chart
    bottom = np.zeros(len(proportions))
    sample_indices = np.arange(len(proportions))

    for idx, cell_type in enumerate(cell_types):
        values = proportions[cell_type].values
        plt.bar(
            sample_indices,
            values,
            bottom=bottom,
            color=colors[idx],
            edgecolor='white',
            width=0.8,
            label=cell_type
        )
        bottom += values

    # Customize the plot
    plt.ylabel('Proportion', fontsize=14)
    plt.title('Cell Type Proportions Across Samples', fontsize=16)
    plt.xticks(sample_indices, proportions.index, rotation=90, fontsize=10)
    plt.yticks(fontsize=12)
    plt.legend(title='Cell Types', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save the figure
    plot_path = os.path.join(output_dir, 'cell_type_abundances.pdf')
    plt.savefig(plot_path)
    plt.close()
    print(f"Cell type abundance plot saved to {plot_path}")



def plot_cell_type_expression_heatmap(
    avg_expression: dict,
    output_dir: str,
    cell_type_order: list = None,
    sample_order: list = None,
    figsize: tuple = (10, 8),
    cmap: str = 'viridis',
    annot: bool = False
):
    """
    Generate a heatmap showing the average expression of each cell type across samples.
    
    Parameters:
    ----------
    avg_expression : dict
        Nested dictionary where avg_expression[sample][cell_type] = average_expression_array
    output_dir : str
        Directory to save the heatmap.
    cell_type_order : list, optional
        Order of cell types in the heatmap. If None, uses the order in the dictionary.
    sample_order : list, optional
        Order of samples in the heatmap. If None, uses the order in the dictionary.
    figsize : tuple, optional
        Size of the heatmap figure.
    cmap : str, optional
        Colormap for the heatmap.
    annot : bool, optional
        Whether to annotate the heatmap cells with their values.
    
    Returns:
    -------
    None
    """
    
    # Extract unique cell types and samples
    samples = list(avg_expression.keys())
    cell_types = list(next(iter(avg_expression.values())).keys()) if samples else []
    
    # Initialize a DataFrame with cell types as rows and samples as columns
    expression_matrix = pd.DataFrame(index=cell_types, columns=samples, dtype=np.float64)
    
    for sample in samples:
        for cell_type in cell_types:
            # Sum the average expression array to get a single scalar value
            # If the cell type is not present, it should already be 0
            # expression_value = avg_expression[sample].get(cell_type, np.zeros(1))[0] if avg_expression[sample].get(cell_type, np.zeros(1)).size > 0 else 0
            # Alternatively, sum across genes if avg_expression[sample][cell_type] is a vector
            expression_value = avg_expression[sample].get(cell_type, np.zeros(avg_expression[sample][list(avg_expression[sample].keys())[0]].shape)[0].astype(np.float64)).mean()
            expression_matrix.loc[cell_type, sample] = expression_value
    # Replace NaN with 0 (in case some cell types are missing in certain samples)
    expression_matrix.fillna(0, inplace=True)
    
    if cell_type_order:
        expression_matrix = expression_matrix.reindex(cell_type_order)
    if sample_order:
        expression_matrix = expression_matrix[sample_order]
    
    # Create the heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        expression_matrix,
        cmap=cmap,
        linewidths=0.5,
        linecolor='grey',
        annot=annot,
        fmt=".2f"
    )
    plt.title('Average Expression of Cell Types Across Samples')
    plt.xlabel('Samples')
    plt.ylabel('Cell Types')
    
    # Save the heatmap
    heatmap_path = os.path.join(output_dir, 'cell_type_expression_heatmap.pdf')
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Cell type expression heatmap saved to {heatmap_path}")

def visualizeGroupRelationship(sample_distance_matrix, outputDir, heatmap_path=None):
    # Ensure the output directory exists
    os.makedirs(outputDir, exist_ok=True)

    # Use the true sample names from the distance matrix index
    samples = sample_distance_matrix.index.tolist()

    # Making the distance matrix symmetric and setting diagonal to 0
    sample_distance_matrix = (sample_distance_matrix + sample_distance_matrix.T) / 2
    np.fill_diagonal(sample_distance_matrix.values, 0)

    # Perform MDS to get a 2D embedding based on the distance matrix
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    points_2d = mds.fit_transform(sample_distance_matrix)

    # Plot the 2D embedding
    plt.figure(figsize=(8, 6))
    plt.scatter(points_2d[:, 0], points_2d[:, 1], color='blue', s=100)

    # Annotate points with the actual sample names
    for i, (x, y) in enumerate(points_2d):
        plt.text(x, y, samples[i], fontsize=12, ha='right')

    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.title("2D MDS Visualization of Sample Distance Matrix")
    plt.grid(True)

    # Determine output path
    if heatmap_path is None:
        heatmap_path = os.path.join(outputDir, "sample_distance_matrix_MDS.png")

    # Save the plot to the output path
    plt.savefig(heatmap_path)
    plt.close()  # Close the plot to avoid displaying it in non-interactive environments
    print(f"Plot saved to {heatmap_path}")

def visualizeDistanceMatrix(sample_distance_matrix, heatmap_path):
    
    # Convert the square distance matrix to condensed form for linkage
    condensed_distances = squareform(sample_distance_matrix.values)

    # Compute the linkage matrix using the condensed distance matrix
    linkage_matrix = linkage(condensed_distances, method='average')

    # Generate the clustermap
    sns.clustermap(
        sample_distance_matrix,
        cmap='viridis',
        linewidths=0.5,
        annot=True,
        row_linkage=linkage_matrix,
        col_linkage=linkage_matrix
    )
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Sample distance heatmap saved to {heatmap_path}")