import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import os

def visualizeGroupRelationship(sampleNumber, sample_distance_matrix, outputDir, heatmap_path=None):
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
