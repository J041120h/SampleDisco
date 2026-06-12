import os
from Bio import Phylo
from Bio.Phylo.Newick import Tree, Clade
from collections import Counter
import glob
import re
import logging
import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import linkage
from sample_clustering.cluster_helper import *

from sample_clustering.NN import NN
from sample_clustering.UPGMA import UPGMA
from sample_clustering.HRA_VEC import HRA_VEC
from sample_clustering.HRC_VEC import HRC_VEC

def convert_to_unweighted_newick(weighted_newick):
    """Convert a weighted Newick tree string to an unweighted version."""
    unweighted_newick = re.sub(r":\d+(\.\d+)?", "", weighted_newick)
    return unweighted_newick

def setupLogging(logFilePath):
    """Configure logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(logFilePath), logging.StreamHandler()],
    )

def getSplits(tree):
    """Extract splits (bipartitions) from a phylogenetic tree."""
    splits = []
    clades = list(tree.find_clades(order="level"))
    for clade in clades:
        if clade.is_terminal():
            continue
        taxaNames = frozenset(taxon.name for taxon in clade.get_terminals())
        splits.append(taxaNames)
    return splits

def buildConsensusTree(allTaxa, majoritySplits):
    """Build a consensus tree from a set of majority splits."""
    consensusTree = Tree()
    consensusTree.root.clades = [
        Clade(name=taxon, branch_length=1.0) for taxon in sorted(allTaxa)
    ]

    consensusTree.root.branch_length = 0.0

    height = 1.0
    changed = True
    while changed:
        changed = False
        for split in list(majoritySplits):
            for clade in consensusTree.get_nonterminals(order="postorder"):
                cladeTaxa = {taxon.name for taxon in clade.get_terminals()}
                if split < cladeTaxa and split != cladeTaxa:
                    taxaInSplit = []
                    taxaNotInSplit = []
                    for subclade in clade.clades:
                        subcladeTaxa = {
                            taxon.name for taxon in subclade.get_terminals()
                        }
                        if subcladeTaxa & split:
                            taxaInSplit.append(subclade)
                        else:
                            taxaNotInSplit.append(subclade)
                    if taxaInSplit and taxaNotInSplit:
                        newClade = Clade()
                        newClade.clades = taxaInSplit
                        newClade.branch_length = height

                        for subclade in taxaInSplit:
                            if subclade.branch_length is None:
                                subclade.branch_length = height
                        for subclade in taxaNotInSplit:
                            if subclade.branch_length is None:
                                subclade.branch_length = height

                        clade.clades = [newClade] + taxaNotInSplit
                        changed = True
                        majoritySplits.remove(split)
                        height += 1.0
                        break

    for clade in consensusTree.find_clades():
        if clade.branch_length is None:
            clade.branch_length = 1.0

    for clade in consensusTree.find_clades():
        for subclade in clade.clades:
            subclade.parent = clade

    return consensusTree

def buildConsensus(sample_distance_paths, generalFolder, methods=None, run_methods=True, custom_tree_names=None):
    """
    Build a majority-rule consensus tree from multiple phylogenetic methods.

    Parameters:
        sample_distance_paths (dict or str): Distance matrix path(s) keyed by data type,
                                             or a single path string (backward compat).
        generalFolder (str): Base output folder; trees go in <generalFolder>/Tree/<method>/.
        methods (list): Tree-building methods to use (default: all four: NN, UPGMA, HRA_VEC, HRC_VEC).
        run_methods (bool): Run the tree-building methods before collecting trees.
        custom_tree_names (list): Output tree names per input data type.
    """
    resultFolder = generalFolder
    consensusFolder = os.path.join(resultFolder, "Tree", "consensus")
    os.makedirs(consensusFolder, exist_ok=True)
    logFilePath = os.path.join(generalFolder, "consensus_building.log")
    setupLogging(logFilePath)
    method_functions = {
        'NN': NN,
        'UPGMA': UPGMA,
        'HRA_VEC': HRA_VEC,
        'HRC_VEC': HRC_VEC,
        'HRA': HRA_VEC,
        'HRC': HRC_VEC,
    }

    if methods is None:
        methods = ['NN', 'UPGMA', 'HRA_VEC', 'HRC_VEC']
    
    logging.info(f"Building consensus tree using methods: {methods}")

    if isinstance(sample_distance_paths, str):
        sample_distance_paths = {"default": sample_distance_paths}
    
    if custom_tree_names is None:
        custom_tree_names = list(sample_distance_paths.keys())
    
    if run_methods:
        logging.info("Running tree building methods with custom tree names")
        for method in methods:
            method_key = method
            if method in ['HRA', 'HRC']:
                method_key = f"{method}_VEC"
                
            if method_key in method_functions:
                method_function = method_functions[method_key]
                tree_output_dir = os.path.join(generalFolder, "Tree", method)
                
                for data_type, path in sample_distance_paths.items():
                    tree_name = data_type
                    logging.info(f"Running {method} on {data_type} data, saving as {tree_name}")
                    method_function(
                        inputFilePath=path, 
                        generalOutputDir=tree_output_dir, 
                        custom_tree_name=custom_tree_names
                    )
            else:
                logging.warning(f"Method {method} not recognized, skipping...")
    
    all_trees = []

    for method in methods:
        methodFolder = os.path.join(resultFolder, "Tree", method)
        if not os.path.isdir(methodFolder):
            logging.warning(f"Method folder not found: {methodFolder}")
            continue
        
        treeFiles = glob.glob(os.path.join(methodFolder, "**", "*.nex"), recursive=True)
        logging.info(f"Found {len(treeFiles)} tree files in {methodFolder}")

        for file in treeFiles:
            logging.info(f"  - Found tree file: {file}")

        if not treeFiles:
            logging.warning(f"No .nex tree files found in {methodFolder}. Please check the directory structure.")
        
        for filePath in treeFiles:
            try:
                trees = list(Phylo.parse(filePath, "nexus"))
                all_trees.extend(trees)
                logging.info(f"Added {len(trees)} trees from {filePath}")
            except Exception as e:
                logging.warning(f"Could not parse {filePath}: {e}")
    
    if not all_trees:
        logging.error("No trees collected from any method, cannot build consensus")
        return
    
    logging.info(f"Total trees collected for consensus: {len(all_trees)}")
    
    splitCounts = Counter()
    allTaxa = set()
    
    for tree in all_trees:
        allTaxa.update(taxon.name for taxon in tree.get_terminals())
        splits = getSplits(tree)
        splitCounts.update(splits)
    
    majorityThreshold = len(all_trees) / 2
    majoritySplits = {split for split, count in splitCounts.items() if count > majorityThreshold}
    logging.info(f"{len(majoritySplits)} majority splits identified")
    
    if not majoritySplits:
        logging.warning("No majority splits found, cannot build consensus")
        return
    
    consensusTree = buildConsensusTree(allTaxa, majoritySplits)
    outputFileNex = os.path.join(consensusFolder, f"{custom_tree_names}.nex")
    outputFileNwk = os.path.join(consensusFolder, f"{custom_tree_names}.nwk")
    outputImagePath = os.path.join(consensusFolder, f"{custom_tree_names}.png")

    Phylo.write(consensusTree, outputFileNex, "nexus")
    Phylo.write(consensusTree, outputFileNwk, "newick")
    logging.info(f"Consensus tree written to {outputFileNex} and {outputFileNwk}")
    
    distance_matrix, labels = calculate_distance_matrix_from_tree(consensusTree)
    condensed_dist = ssd.squareform(distance_matrix)
    linkage_matrix = linkage(condensed_dist, method="complete")
    visualizeTree(linkage_matrix, outputImagePath, "Consensus Tree", labels)
    logging.info(f"Consensus tree plot saved at: {outputImagePath}")
    
    with open(outputFileNwk, "r+") as f:
        weightedNewick = f.read()
        f.seek(0)
        f.truncate()
        f.write(convert_to_unweighted_newick(weightedNewick))
        logging.info(f"Unweighted Newick tree written to {outputFileNwk}")