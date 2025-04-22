import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import numpy as np


def visualizeTree(linkageMatrix, outputImagePath, treeLabel, labels):
    plt.figure(figsize=(7, 5))
    plt.title(f"Phylogenetic Tree: {treeLabel}")
    plt.xlabel("Distance")
    plt.ylabel("Taxa")
    plt.gca().yaxis.set_label_position("right")
    dendrogram(linkageMatrix, orientation="left", labels=labels)
    plt.tight_layout()
    plt.savefig(outputImagePath, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Phylogenetic tree visualization saved to '{outputImagePath}'.")


def add_parent_references(tree):
    """
    Add parent references to all clades in the tree.
    This is necessary because Bio.Phylo trees don't have parent pointers by default.
    """
    for clade in tree.find_clades(order='preorder'):
        for subclade in clade:
            subclade.parent = clade
    return tree


def calculate_distance_matrix_from_tree(tree):
    """Calculate distance matrix from a phylogenetic tree"""
    # First add parent references to all nodes
    add_parent_references(tree)
    
    terminals = tree.get_terminals()
    n = len(terminals)
    distance_matrix = np.zeros((n, n))

    def get_distance_to_mrca(terminal, mrca):
        """Calculate the distance from a terminal to a specific ancestor"""
        distance = 0
        current = terminal
        while current != mrca and current is not None:
            if current.branch_length is not None:
                distance += current.branch_length
            current = current.parent
        return distance

    # For each pair of terminals, calculate the patristic distance
    for i, term_i in enumerate(terminals):
        for j, term_j in enumerate(terminals[i + 1 :], i + 1):
            # Find most recent common ancestor
            mrca = tree.common_ancestor(term_i, term_j)
            if mrca is not None:
                # Calculate distances from each terminal to MRCA
                dist1 = get_distance_to_mrca(term_i, mrca)
                dist2 = get_distance_to_mrca(term_j, mrca)
                distance = dist1 + dist2
            else:
                # If no MRCA found, use maximum possible distance
                distance = len(list(tree.find_clades()))

            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # symmetric matrix

    return distance_matrix, [term.name for term in terminals]