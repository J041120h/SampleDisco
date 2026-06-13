import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio import Phylo
from Bio.Phylo.Newick import Tree, Clade
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree
import scipy.spatial.distance as ssd
from sampledisco.sample_clustering.cluster_helper import *

def loadDistanceMatrix(filePath):
    """Load a distance matrix CSV; mirrors upper triangle to lower to enforce symmetry."""
    df = pd.read_csv(filePath, index_col=0)
    df.replace("-", np.nan, inplace=True)
    matrix = df.to_numpy(dtype=float)
    iUpper = np.triu_indices_from(matrix, 1)
    matrix[(iUpper[1], iUpper[0])] = matrix[iUpper]
    return matrix, df.columns.tolist()

def linkageToPhylo(linkageMatrix, labels):
    """Convert a linkage matrix to a Bio.Phylo tree."""
    treeRoot, _ = to_tree(linkageMatrix, rd=True)
    
    def buildClade(node, parent=None):
        if node.is_leaf():
            return Clade(name=labels[node.id])
        clade = Clade(branch_length=node.dist)
        clade.clades.append(buildClade(node.left, clade))
        clade.clades.append(buildClade(node.right, clade))
        return clade
        
    return Tree(root=buildClade(treeRoot))

def upgmaTree(distanceMatrix, labels, title, ax):
    """Plot dendrogram using UPGMA (average linkage) and return Phylo tree."""
    condensedDist = ssd.squareform(distanceMatrix, checks=False)
    linkageMatrix = linkage(condensedDist, method="average")
    
    dendrogram(
        linkageMatrix,
        labels=labels,
        orientation="left",
        ax=ax,
    )
    
    ax.set_title(title)
    ax.set_xlabel("Distance")
    ax.set_ylabel("Samples")
    
    return linkageToPhylo(linkageMatrix, labels)

def UPGMA(inputFilePath, generalOutputDir, custom_tree_name=None):
    """Build a UPGMA tree from a distance-matrix CSV and write .nex + .png."""
    if not os.path.exists(inputFilePath):
        print(f"Input file '{inputFilePath}' not found.")
        return

    os.makedirs(generalOutputDir, exist_ok=True)
    baseName = custom_tree_name if custom_tree_name else os.path.splitext(os.path.basename(inputFilePath))[0]
    outputTreePath = os.path.join(generalOutputDir, f"{baseName}.nex")
    outputImagePath = os.path.join(generalOutputDir, f"{baseName}.png")

    print(f"\nProcessing file: {inputFilePath} with tree name '{baseName}'")

    matrix, labels = loadDistanceMatrix(inputFilePath)

    fig, ax = plt.subplots(figsize=(8, len(labels) * 0.3 + 2))
    tree = upgmaTree(matrix, labels, f"UPGMA Tree: {baseName}", ax)

    fig.tight_layout()
    fig.savefig(outputImagePath)
    plt.close(fig)
    print(f" - Saved tree visualization to '{outputImagePath}'.")

    Phylo.write([tree], outputTreePath, "nexus")
    print(f" - Saved UPGMA tree to '{outputTreePath}' in Nexus format.")

    # Recompute patristic distances from the fitted tree and re-render with visualizeTree
    # for a consistent dendrogram style shared with NN and HRA/HRC.
    distanceMatrix, reorderedLabels = calculate_distance_matrix_from_tree(tree)
    condensedDist = ssd.squareform(distanceMatrix)
    linkageMatrix = linkage(condensedDist, method="complete")
    visualizeTree(linkageMatrix, outputImagePath, "UPGMA", reorderedLabels)
    print(f" - Final linkage-based visualization completed for '{baseName}'.")