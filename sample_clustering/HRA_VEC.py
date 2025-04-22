import os
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import to_tree
from dendropy import Tree as DendroPyTree
from dendropy import TaxonNamespace
from sample_clustering.cluster_helper import *

def readExpressionCsv(filePath):
    """Read expression data where rows are samples and columns are features."""
    expressionDf = pd.read_csv(filePath, index_col=0).transpose()
    return expressionDf

def hraVectorClustering(expressionMatrix):
    """
    Perform hierarchical clustering using average linkage method.
    
    Parameters:
        expressionMatrix (ndarray): Samples x Features matrix
    
    Returns:
        linkageMatrix (ndarray): Result of hierarchical clustering
    """
    distances = np.zeros((expressionMatrix.shape[0], expressionMatrix.shape[0]))
    for i in range(expressionMatrix.shape[0]):
        for j in range(expressionMatrix.shape[0]):
            distances[i, j] = np.linalg.norm(expressionMatrix[i] - expressionMatrix[j])
    condensedDistMatrix = sch.distance.squareform(distances)
    linkageMatrix = sch.linkage(condensedDistMatrix, method="average")
    return linkageMatrix

def linkageToNewick(linkageMatrix, labels):
    """Convert scipy linkage matrix to Newick format."""
    tree = to_tree(linkageMatrix, rd=False)

    def buildNewick(node):
        if node.is_leaf():
            return labels[node.id]
        else:
            left = buildNewick(node.left)
            right = buildNewick(node.right)
            leftLength = node.dist - node.left.dist
            rightLength = node.dist - node.right.dist
            return f"({left}:{leftLength:.2f},{right}:{rightLength:.2f})"

    return buildNewick(tree) + ";"

def saveTreesNexus(newickTrees, outputTreePath):
    """Save trees in NEXUS format."""
    with open(outputTreePath, "w") as nexusFile:
        nexusFile.write("#NEXUS\nBEGIN TREES;\n")
        for idx, (newickStr, label) in enumerate(newickTrees, 1):
            nexusFile.write(f"    TREE {label} = {newickStr}\n")
        nexusFile.write("END;\n")
    print(f"All trees saved to '{outputTreePath}' in NEXUS format.")

def processExpressionData(filePath, outputDir, custom_name=None):
    """
    Process one expression data file and return DendroPy tree and label.

    Parameters:
        filePath (str): Path to the expression CSV file
        outputDir (str): Base output directory
        custom_name (str, optional): Custom name for output files

    Returns:
        (DendroPyTree, str): The tree and its label
    """
    baseName = os.path.basename(filePath)
    treeLabel = custom_name if custom_name else os.path.splitext(baseName)[0]
    outputImagePath = os.path.join(outputDir, f"{treeLabel}.png")

    print(f"\nProcessing '{filePath}' with label '{treeLabel}'...")
    expressionDf = readExpressionCsv(filePath)
    linkageMatrix = hraVectorClustering(expressionDf.values)
    labels = expressionDf.index.tolist()

    visualizeTree(linkageMatrix, outputImagePath, "HRA", labels)
    newickStr = linkageToNewick(linkageMatrix, labels)
    dendroTree = DendroPyTree.get(data=newickStr, schema="newick", taxon_namespace=TaxonNamespace())

    return dendroTree, treeLabel

def HRA_VEC(inputFilePath, generalOutputDir, custom_tree_name=None):
    """
    Run hierarchical clustering on input data and save results.

    Parameters:
        inputFilePath (str): Path to CSV input file
        generalOutputDir (str): Output folder to store .nex and .png files
        custom_tree_name (str, optional): Custom name for the output tree file
                                         (without extension, e.g., "expression")
    """
    if not os.path.exists(inputFilePath):
        print(f"Input file '{inputFilePath}' not found.")
        return

    os.makedirs(generalOutputDir, exist_ok=True)
    newickTrees = []

    try:
        # Process the data to get the tree structure
        dendroTree, treeLabel = processExpressionData(
            filePath=inputFilePath, 
            outputDir=generalOutputDir,
            custom_name=custom_tree_name
        )
            
        newickStr = dendroTree.as_string(schema="newick").strip()
        newickTrees.append((newickStr, treeLabel))
        print(f"Completed processing for '{treeLabel}'.")
    except Exception as e:
        print(f"Error during processing: {e}")

    if newickTrees:
        outputTreePath = os.path.join(generalOutputDir, f"{treeLabel}.nex")
        saveTreesNexus(newickTrees, outputTreePath)
        print(f"Tree saved as '{treeLabel}.nex'")
    else:
        print("No trees were successfully processed.")
