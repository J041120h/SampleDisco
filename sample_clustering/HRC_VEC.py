import os
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import to_tree
from dendropy import Tree as DendroPyTree
from dendropy import TaxonNamespace
import utils

def readExpressionCsv(filePath):
    """Read expression data where rows are samples and columns are features"""
    return pd.read_csv(filePath, index_col=0).transpose()

def vectorBasedClustering(expressionMatrix):
    """Perform hierarchical clustering using complete linkage on Euclidean distances"""
    return sch.linkage(expressionMatrix, method="complete", metric="euclidean")

def linkageToNewick(linkageMatrix, labels):
    """Convert scipy linkage matrix to Newick format"""
    tree = to_tree(linkageMatrix, rd=False)

    def buildNewick(node):
        if node.is_leaf():
            return labels[node.id]
        else:
            left = buildNewick(node.left)
            right = buildNewick(node.right)
            leftLength = max(0.01, node.left.dist)
            rightLength = max(0.01, node.right.dist)
            return f"({left}:{leftLength:.2f},{right}:{rightLength:.2f})"

    return buildNewick(tree) + ";"

def saveTreesNexus(newickTrees, outputTreePath):
    """Save trees in NEXUS format"""
    with open(outputTreePath, "w") as nexusFile:
        nexusFile.write("#NEXUS\nBEGIN TREES;\n")
        for idx, (newickStr, label) in enumerate(newickTrees, 1):
            nexusFile.write(f"    TREE {label} = {newickStr}\n")
        nexusFile.write("END;\n")
    print(f"All trees saved to '{outputTreePath}' in NEXUS format.")

def processExpressionData(filePath, outputDir):
    """Process a single expression data file and generate tree/image paths automatically"""
    baseName = os.path.basename(filePath)
    treeLabel = os.path.splitext(baseName)[0]

    outputImagePath = os.path.join(outputDir, f"{treeLabel}.png")
    outputTreePath = os.path.join(outputDir, f"{treeLabel}.nex")

    print(f"\nProcessing '{filePath}' with label '{treeLabel}'...")

    expressionDf = readExpressionCsv(filePath)
    print(" - Loaded expression matrix.")

    linkageMatrix = vectorBasedClustering(expressionDf.values)
    print(" - Performed hierarchical clustering.")

    labels = expressionDf.index.tolist()
    utils.visualizeTree(linkageMatrix, outputImagePath, "HRC", labels)
    print(f" - Saved tree visualization to '{outputImagePath}'.")

    newickStr = linkageToNewick(linkageMatrix, labels)
    dendroTree = DendroPyTree.get(data=newickStr, schema="newick", taxon_namespace=TaxonNamespace())

    return dendroTree, treeLabel, outputTreePath

def HRC_VEC(inputFilePath, generalOutputDir):
    """Main function to run hierarchical clustering and save results"""
    if not os.path.exists(inputFilePath):
        print(f"Input file '{inputFilePath}' not found.")
        return

    os.makedirs(generalOutputDir, exist_ok=True)

    try:
        dendroTree, treeLabel, outputTreePath = processExpressionData(inputFilePath, generalOutputDir)
        newickStr = dendroTree.as_string(schema="newick").strip()
        saveTreesNexus([(newickStr, treeLabel)], outputTreePath)
        print("Tree generation and export complete.\n")
    except Exception as e:
        print(f"Error during processing: {e}")