import os
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import to_tree
from dendropy import Tree as DendroPyTree
from dendropy import TaxonNamespace
from sample_clustering.cluster_helper import *

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

def processExpressionData(filePath, outputDir, custom_name=None):
    """
    Process a single expression data file and generate tree/image paths with optional custom naming
    
    Parameters:
        filePath (str): Path to the expression CSV file
        outputDir (str): Base output directory
        custom_name (str, optional): Custom name for output files (without extension)
        
    Returns:
        tuple: (dendropy_tree, tree_label, output_tree_path)
    """
    baseName = os.path.basename(filePath)
    treeLabel = custom_name if custom_name else os.path.splitext(baseName)[0]
    outputImagePath = os.path.join(outputDir, f"{treeLabel}.png")
    outputTreePath = os.path.join(outputDir, f"{treeLabel}.nex")
    
    print(f"\nProcessing '{filePath}' with label '{treeLabel}'...")
    expressionDf = readExpressionCsv(filePath)
    print(" - Loaded expression matrix.")
    
    linkageMatrix = vectorBasedClustering(expressionDf.values)
    print(" - Performed hierarchical clustering.")
    
    labels = expressionDf.index.tolist()
    visualizeTree(linkageMatrix, outputImagePath, "HRC", labels)
    print(f" - Saved tree visualization to '{outputImagePath}'.")
    
    newickStr = linkageToNewick(linkageMatrix, labels)
    dendroTree = DendroPyTree.get(data=newickStr, schema="newick", taxon_namespace=TaxonNamespace())
    
    return dendroTree, treeLabel, outputTreePath

def HRC_VEC(inputFilePath, generalOutputDir, custom_tree_name=None):
    """
    Main function to run hierarchical clustering and save results
    
    Parameters:
        inputFilePath (str): Path to CSV input file
        generalOutputDir (str): Output folder to store .nex and .png files
        custom_tree_name (str, optional): Custom name for the output tree file (without extension)
    """
    if not os.path.exists(inputFilePath):
        print(f"Input file '{inputFilePath}' not found.")
        return
        
    os.makedirs(generalOutputDir, exist_ok=True)
    
    try:
        dendroTree, treeLabel, outputTreePath = processExpressionData(
            filePath=inputFilePath, 
            outputDir=generalOutputDir,
            custom_name=custom_tree_name
        )
        
        newickStr = dendroTree.as_string(schema="newick").strip()
        saveTreesNexus([(newickStr, treeLabel)], outputTreePath)
        print(f"Tree generation and export complete. Saved as '{os.path.basename(outputTreePath)}'.\n")
    except Exception as e:
        print(f"Error during processing: {e}")