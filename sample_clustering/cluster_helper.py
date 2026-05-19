import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
from scipy.spatial.distance import pdist
from dendropy import Tree as DendroPyTree, TaxonNamespace


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


def _linkage_to_newick(linkageMatrix, labels):
    """Convert scipy linkage matrix to Newick string.

    Branch lengths use the dendrogram drop from parent to child
    (parent.dist - child.dist) — the mathematically correct definition.
    """
    tree = to_tree(linkageMatrix, rd=False)

    def _build(node):
        if node.is_leaf():
            return labels[node.id]
        left = _build(node.left)
        right = _build(node.right)
        leftLength = node.dist - node.left.dist
        rightLength = node.dist - node.right.dist
        return f"({left}:{leftLength:.2f},{right}:{rightLength:.2f})"

    return _build(tree) + ";"


def _save_trees_nexus(newickTrees, outputTreePath):
    """Write `[(newick_string, label)]` to a NEXUS trees block."""
    with open(outputTreePath, "w") as nexusFile:
        nexusFile.write("#NEXUS\nBEGIN TREES;\n")
        for newickStr, label in newickTrees:
            nexusFile.write(f"    TREE {label} = {newickStr}\n")
        nexusFile.write("END;\n")
    print(f"All trees saved to '{outputTreePath}' in NEXUS format.")


def expression_tree_to_nexus(
    inputFilePath: str,
    generalOutputDir: str,
    linkage_method: str,
    tree_label_prefix: str,
    custom_tree_name=None,
):
    """Shared driver for expression-based hierarchical tree builders.

    Reads samples × features expression CSV, runs scipy hierarchical
    clustering with `linkage_method` ('average', 'complete', ...),
    saves a PNG dendrogram and a NEXUS tree file.

    Used by HRA_VEC (method='average') and HRC_VEC (method='complete').
    """
    if not os.path.exists(inputFilePath):
        print(f"Input file '{inputFilePath}' not found.")
        return

    os.makedirs(generalOutputDir, exist_ok=True)

    baseName = os.path.basename(inputFilePath)
    treeLabel = custom_tree_name if custom_tree_name else os.path.splitext(baseName)[0]
    outputImagePath = os.path.join(generalOutputDir, f"{treeLabel}.png")
    outputTreePath = os.path.join(generalOutputDir, f"{treeLabel}.nex")

    print(f"\nProcessing '{inputFilePath}' with label '{treeLabel}'...")
    expressionDf = pd.read_csv(inputFilePath, index_col=0).transpose()
    condensed = pdist(expressionDf.values, metric="euclidean")
    linkageMatrix = sch.linkage(condensed, method=linkage_method)
    labels = expressionDf.index.tolist()

    visualizeTree(linkageMatrix, outputImagePath, tree_label_prefix, labels)
    newickStr = _linkage_to_newick(linkageMatrix, labels)

    dendroTree = DendroPyTree.get(
        data=newickStr, schema="newick", taxon_namespace=TaxonNamespace()
    )
    newickOut = dendroTree.as_string(schema="newick").strip()
    _save_trees_nexus([(newickOut, treeLabel)], outputTreePath)
    print(f"Tree saved as '{treeLabel}.nex'")