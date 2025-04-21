import os
import io
import numpy as np
import pandas as pd
from skbio import DistanceMatrix
from skbio.tree import nj
from dendropy import Tree as DendroPyTree
from dendropy import TaxonNamespace
from Bio import Phylo
from scipy.cluster.hierarchy import linkage
import scipy.spatial.distance as ssd
import utils


def read_distance_csv(filePath):
    """Reads a CSV file containing the distance matrix."""
    distanceDf = pd.read_csv(filePath, index_col=0, na_values=["-"])
    distanceDf = distanceDf.astype(float)
    return distanceDf


def construct_nj_tree(distanceDf):
    """Constructs a Neighbor-Joining tree from a distance matrix."""
    ids = distanceDf.index.tolist()
    dm = DistanceMatrix(distanceDf.values, ids)
    return nj(dm)


def skbio_to_dendropy_tree(skbioTree):
    """Converts a skbio tree to a dendropy tree."""
    with io.StringIO() as newickIo:
        skbioTree.write(newickIo, format="newick")
        newickStr = newickIo.getvalue()
    return DendroPyTree.get(data=newickStr, schema="newick", taxon_namespace=TaxonNamespace())


def get_linkage_matrix(distanceMatrix):
    """Converts square distance matrix to linkage matrix format."""
    condensed_dist = ssd.squareform(distanceMatrix, checks=False)
    return linkage(condensed_dist, method="average")


def save_trees_nexus(dendropyTrees, outputTreePath):
    """Saves all trees in Nexus format."""
    with open(outputTreePath, "w") as nexusFile:
        nexusFile.write("#NEXUS\nBEGIN TREES;\n")
        for idx, (tree, label) in enumerate(dendropyTrees, 1):
            treeStr = tree.as_string(schema='newick').strip()
            nexusFile.write(f"    TREE {label} = {treeStr}\n")
        nexusFile.write("END;\n")
    print(f"All trees saved to '{outputTreePath}' in NEXUS format.")


def process_single_csv(filePath, outputDir):
    """Processes a single CSV file containing a distance matrix."""
    baseName = os.path.basename(filePath)
    treeLabel = os.path.splitext(baseName)[0]
    print(f"\nProcessing file: '{filePath}' with label '{treeLabel}'.")

    # Output file paths
    outputImagePath = os.path.join(outputDir, f"{treeLabel}.png")
    outputTreePath = os.path.join(outputDir, f"{treeLabel}.nex")

    # Load distance matrix
    distanceDf = read_distance_csv(filePath)
    print(" - Loaded distance matrix.")
    labels = distanceDf.index.tolist()
    distanceMatrix = distanceDf.values

    # Create linkage matrix and visualize tree
    linkageMatrix = get_linkage_matrix(distanceMatrix)
    print(" - Converted to linkage matrix format.")
    utils.visualizeTree(linkageMatrix, outputImagePath, "NN", labels)
    print(f" - Saved tree visualization to '{outputImagePath}'.")

    # Build Neighbor-Joining tree and convert
    njTree = construct_nj_tree(distanceDf)
    dendropyTree = skbio_to_dendropy_tree(njTree)

    return dendropyTree, treeLabel, outputTreePath


def NN(inputFilePath, generalOutputDir):
    """Main function to process a single CSV and generate NJ tree."""
    if not os.path.exists(inputFilePath):
        print(f"Input file '{inputFilePath}' not found.")
        return

    os.makedirs(generalOutputDir, exist_ok=True)

    try:
        dendropyTree, treeLabel, outputTreePath = process_single_csv(inputFilePath, generalOutputDir)
        save_trees_nexus([(dendropyTree, treeLabel)], outputTreePath)
        print("\nNeighbor-Joining tree generation and saving completed.")
    except Exception as e:
        print(f"Error during processing: {e}")