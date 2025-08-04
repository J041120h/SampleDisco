from sample_clustering.consensus import *
from sample_clustering.HRA_VEC import *
from sample_clustering.HRC_VEC import *
from sample_clustering.Kmeans_cluster import *
from sample_clustering.NN import *
from sample_clustering.tree_cut import *
from sample_clustering.UPGMA import *
from sample_clustering.proportion_test import *
import os

def cluster(
    generalFolder: str,
    Kmeans: bool = False,
    methods: list = ['HRA_VEC', 'HRC_VEC', 'NN', 'UPGMA'],
    prportion_test: bool = False,
    distance_method: str = "cosine",
    number_of_clusters: int = 5,
    sample_to_clade_user: dict = None,
):
    """
    Cluster samples using various methods.
    
    Parameters:
    -----------
    generalFolder : str
        General folder path (required)
    Kmeans : bool
        Whether to use K-means clustering
    methods : list
        List of clustering methods to use
    prportion_test : bool
        Whether to perform proportion test
    distance_method : str
        Distance method (used for path construction)
    number_of_clusters : int
        Number of clusters to create
    sample_to_clade_user : dict
        User-provided clustering (optional)
    """
    print(f"[INFO] Entering cluster function with parameters:\n  generalFolder={generalFolder},\n  Kmeans={Kmeans},\n  methods={methods},\n  proportion_test={prportion_test},\n  distance_method={distance_method},\n  number_of_clusters={number_of_clusters}")
    
    # Create sample_cluster directory for all clustering results
    sample_cluster_dir = os.path.join(generalFolder, "sample_cluster")
    os.makedirs(sample_cluster_dir, exist_ok=True)
    print(f"[INFO] Created sample_cluster directory: {sample_cluster_dir}")
    
    pseudobulk_folder_path = os.path.join(generalFolder, "pseudobulk")

    if Kmeans == False and methods is None:
        raise ValueError("Please provide at least one clustering method (Kmeans or methods).")
    
    if sample_to_clade_user is not None:
        # CASE 1: Use provided clustering directly
        print("[INFO] User-provided sample_to_clade detected. Using user input.")
        print("[INFO] Completed user-provided clustering path. Exiting function.")
        return sample_to_clade_user, sample_to_clade_user

    # CASE 2: No user clustering provided, proceed with clustering
    print("[INFO] No user-provided clustering found. Proceeding with default clustering methods.")
    
    # Check if distance method is proportion-only
    proportion_only_methods = ["EMD", "chi_square", "jensen_shannon"]
    use_proportion_only = distance_method in proportion_only_methods
    
    if use_proportion_only:
        print(f"[INFO] Distance method '{distance_method}' detected. Running clustering on proportion data only.")
    
    # Construct distance matrix paths
    sample_distance_path_proportion = os.path.join(generalFolder, "Sample_distance", distance_method, "proportion_DR_distance", "distance_matrix_proportion_DR.csv")
    sample_distance_path_expression = os.path.join(generalFolder, "Sample_distance", distance_method, "expression_DR_distance", "distance_matrix_expression_DR.csv")
    
    # Validate that the required CSV files exist
    if not os.path.exists(sample_distance_path_proportion):
        raise FileNotFoundError(f"Proportion distance matrix file not found: {sample_distance_path_proportion}")
    
    if not use_proportion_only and not os.path.exists(sample_distance_path_expression):
        raise FileNotFoundError(f"Expression distance matrix file not found: {sample_distance_path_expression}")
    
    print(f"[INFO] Using proportion distance matrix: {sample_distance_path_proportion}")
    if not use_proportion_only:
        print(f"[INFO] Using expression distance matrix: {sample_distance_path_expression}")
    
    expr_results_Kmeans = None
    prop_results_Kmeans = None

    if Kmeans:
        print(f"[INFO] Starting Kmeans clustering with {number_of_clusters} clusters.")
        try:
            expr_results_Kmeans, prop_results_Kmeans = cluster_samples_from_folder(folder_path=pseudobulk_folder_path, n_clusters=number_of_clusters)
            print("[INFO] Kmeans clustering completed.")
        except Exception as e:
            print(f"[ERROR] Error in Kmeans clustering: {e}")

    expr_results = None
    prop_results = None
    
    if methods is not None and len(methods) > 0:
        print(f"[INFO] Starting tree-based clustering methods: {methods}")
        try:
            if len(methods) == 1:
                print(f"[INFO] Running single method '{methods[0]}' for tree construction...")
                
                # Build trees for proportion data (always)
                if methods[0] == "HRA_VEC":
                    HRA_VEC(inputFilePath=sample_distance_path_proportion, generalOutputDir=os.path.join(sample_cluster_dir, "Tree", methods[0]), custom_tree_name="proportion")
                    if not use_proportion_only:
                        HRA_VEC(inputFilePath=sample_distance_path_expression, generalOutputDir=os.path.join(sample_cluster_dir, "Tree", methods[0]), custom_tree_name="expression")
                elif methods[0] == "HRC_VEC":
                    HRC_VEC(inputFilePath=sample_distance_path_proportion, generalOutputDir=os.path.join(sample_cluster_dir, "Tree", methods[0]), custom_tree_name="proportion")
                    if not use_proportion_only:
                        HRC_VEC(inputFilePath=sample_distance_path_expression, generalOutputDir=os.path.join(sample_cluster_dir, "Tree", methods[0]), custom_tree_name="expression")
                elif methods[0] == "NN":
                    NN(inputFilePath=sample_distance_path_proportion, generalOutputDir=os.path.join(sample_cluster_dir, "Tree", methods[0]), custom_tree_name="proportion")
                    if not use_proportion_only:
                        NN(inputFilePath=sample_distance_path_expression, generalOutputDir=os.path.join(sample_cluster_dir, "Tree", methods[0]), custom_tree_name="expression")
                elif methods[0] == "UPGMA":
                    UPGMA(inputFilePath=sample_distance_path_proportion, generalOutputDir=os.path.join(sample_cluster_dir, "Tree", methods[0]), custom_tree_name="proportion")
                    if not use_proportion_only:
                        UPGMA(inputFilePath=sample_distance_path_expression, generalOutputDir=os.path.join(sample_cluster_dir, "Tree", methods[0]), custom_tree_name="expression")
                
                # Set tree paths
                proportion_tree_path = os.path.join(sample_cluster_dir, "Tree", methods[0], "proportion.nex")
                if not use_proportion_only:
                    expression_tree_path = os.path.join(sample_cluster_dir, "Tree", methods[0], "expression.nex")
                
            elif len(methods) > 1:
                print(f"[INFO] Running consensus building for methods: {methods}")
                # Build consensus for proportion data (always)
                buildConsensus(sample_distance_paths=sample_distance_path_proportion, generalFolder=sample_cluster_dir, methods=methods, custom_tree_names="proportion")
                if not use_proportion_only:
                    buildConsensus(sample_distance_paths=sample_distance_path_expression, generalFolder=sample_cluster_dir, methods=methods, custom_tree_names="expression")
                
                # Set consensus tree paths
                proportion_tree_path = os.path.join(sample_cluster_dir, "Tree", "consensus", "proportion.nex")
                if not use_proportion_only:
                    expression_tree_path = os.path.join(sample_cluster_dir, "Tree", "consensus", "expression.nex")
            
            # Cut trees to get clustering results
            prop_results = cut_tree_by_group_count(proportion_tree_path, desired_groups=number_of_clusters, format='nexus', verbose=True, tol=0)
            if not use_proportion_only:
                expr_results = cut_tree_by_group_count(expression_tree_path, desired_groups=number_of_clusters, format='nexus', verbose=True, tol=0)
            else:
                # For proportion-only methods, set expression results to None or copy proportion results
                expr_results = None
                print("[INFO] Expression clustering skipped for proportion-only distance method.")
            
            print("[INFO] Tree construction completed.")
            print("[INFO] Tree-based clustering completed.")
        except Exception as e:
            print(f"[ERROR] Error in tree-based clustering: {e}")
            
    if Kmeans:
        if use_proportion_only:
            # For proportion-only methods with Kmeans, only use proportion results
            expr_results = None
            prop_results = prop_results_Kmeans
        else:
            expr_results = expr_results_Kmeans
            prop_results = prop_results_Kmeans

    if prportion_test:
        print("[INFO] Starting proportion tests...")
        try:    
            if expr_results is not None:
                unique_expr_clades = len(set(expr_results.values()))
                if unique_expr_clades <= 1:
                    print("[INFO] Only one clade found in expression results. Skipping proportion test.")
                else:
                    proportion_DGE_test(sample_cluster_dir, expr_results, sub_folder="expression", verbose=False)
            elif use_proportion_only:
                print("[INFO] Expression proportion test skipped for proportion-only distance method.")
                
            if prop_results is not None:
                unique_prop_clades = len(set(prop_results.values()))
                if unique_prop_clades <= 1:
                    print("[INFO] Only one clade found in proportion results. Skipping proportion test.")
                else:
                    proportion_DGE_test(sample_cluster_dir, prop_results, sub_folder="proportion", verbose=False)
            print("[INFO] Proportion tests completed.")
        except Exception as e:
            print(f"[ERROR] Error in proportion test: {e}")
    
    print("[INFO] cluster function completed.")
    return expr_results, prop_results