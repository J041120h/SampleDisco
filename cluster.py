from sample_clustering.cluster_DEG_visualization import *
from sample_clustering.consensus import *
from sample_clustering.HRA_VEC import *
from sample_clustering.HRC_VEC import *
from sample_clustering.Kmeans_cluster import *
from sample_clustering.NN import *
from sample_clustering.tree_cut import *
from sample_clustering.UPGMA import *
from sample_clustering.proportion_test import *
from sample_clustering.RAISIN import *

def cluster(
    Kmeans: bool = False,
    methods: list = ['HRA_VEC', 'HRC_VEC', 'NN', 'UPGMA'],
    prportion_test: bool = False,
    RAISIN_analysis: bool = False,
    generalFolder: str = None,
    distance_method: str = "cosine",
    number_of_clusters: int = 5,
    sample_to_clade_user: dict = None,  # <-- NEW PARAMETER
):
    if generalFolder is None:
        raise ValueError("Please provide a generalFolder path.")

    pseudobulk_folder_path = os.path.join(generalFolder, "pseudobulk")

    if sample_to_clade_user is not None:
        # CASE 1: Use provided clustering directly
        print("User-provided sample_to_clade detected. Skipping clustering, using user input.")
        output_dir = os.path.join(generalFolder, "Cluster_DEG", "User_cluster")
        os.makedirs(output_dir, exist_ok=True)

        cluster_dge_visualization(sample_to_clade=sample_to_clade_user, folder_path=pseudobulk_folder_path, output_dir=output_dir)

        unique_user_clades = len(set(sample_to_clade_user.values()))
        if unique_user_clades <= 1:
            print("Only one clade found in user-provided clustering. Skipping multi-clade DGE analysis.")
        elif unique_user_clades > 2:
            print("Conducting multi-clade DGE analysis for user-provided clustering.")
            multi_clade_dge_analysis(sample_to_clade=sample_to_clade_user, folder_path=pseudobulk_folder_path, output_dir=output_dir)

        return  # Important: Exit after handling user clustering

    # CASE 2: No user clustering provided, proceed with clustering
    sample_distance_path_proportion = os.path.join(generalFolder, "Sample", distance_method, "cell_proportion", "distance_matrix_proportion.csv")
    sample_distance_path_expression = os.path.join(generalFolder, "Sample", distance_method, "cell_expression", "distance_matrix_expression.csv")
    
    expr_results_Kmeans = None
    prop_results_Kmeans = None

    if Kmeans:
        try:
            expr_results_Kmeans, prop_results_Kmeans = cluster_samples_from_folder(folder_path=pseudobulk_folder_path, n_clusters=number_of_clusters)
            expr_output_dir = os.path.join(generalFolder, "Cluster_DEG", "Kmeans_expression")
            prop_output_dir = os.path.join(generalFolder, "Cluster_DEG", "Kmeans_proportion")

            os.makedirs(expr_output_dir, exist_ok=True)
            os.makedirs(prop_output_dir, exist_ok=True)

            cluster_dge_visualization(sample_to_clade=expr_results_Kmeans, folder_path=pseudobulk_folder_path, output_dir=expr_output_dir)
            cluster_dge_visualization(sample_to_clade=prop_results_Kmeans, folder_path=pseudobulk_folder_path, output_dir=prop_output_dir)
            
            unique_expr_clades = len(set(expr_results_Kmeans.values()))
            if unique_expr_clades <= 1:
                print("Only one clade found in expression-based Kmeans clustering. Skipping multi-clade DGE analysis.")
            elif unique_expr_clades > 2:
                print("Conducting multi-clade DGE analysis for expression-based Kmeans clustering.")
                multi_clade_dge_analysis(sample_to_clade=expr_results_Kmeans, folder_path=pseudobulk_folder_path, output_dir=expr_output_dir)

            unique_prop_clades = len(set(prop_results_Kmeans.values()))
            if unique_prop_clades <= 1:
                print("Only one clade found in proportion-based Kmeans clustering. Skipping multi-clade DGE analysis.")
            elif unique_prop_clades > 2:
                print("Conducting multi-clade DGE analysis for proportion-based Kmeans clustering.")
                multi_clade_dge_analysis(sample_to_clade=prop_results_Kmeans, folder_path=pseudobulk_folder_path, output_dir=prop_output_dir)
        except Exception as e:
            print(f"Error in Kmeans clustering: {e}")

    expr_results = None
    prop_results = None
    
    if methods is not None and len(methods) > 0:
        try:
            if len(methods) == 1:
                print(f"Running {methods[0]} for tree construction...")
                if methods[0] == "HRA_VEC":
                    HRA_VEC(inputFilePath=sample_distance_path_expression, generalOutputDir=os.path.join(generalFolder, "Tree", methods[0]), custom_tree_name="expression")
                    HRA_VEC(inputFilePath=sample_distance_path_proportion, generalOutputDir=os.path.join(generalFolder, "Tree", methods[0]), custom_tree_name="proportion")
                elif methods[0] == "HRC_VEC":
                    HRC_VEC(inputFilePath=sample_distance_path_expression, generalOutputDir=os.path.join(generalFolder, "Tree", methods[0]), custom_tree_name="expression")
                    HRC_VEC(inputFilePath=sample_distance_path_proportion, generalOutputDir=os.path.join(generalFolder, "Tree", methods[0]), custom_tree_name="proportion")
                elif methods[0] == "NN":
                    NN(inputFilePath=sample_distance_path_expression, generalOutputDir=os.path.join(generalFolder, "Tree", methods[0]), custom_tree_name="expression")
                    NN(inputFilePath=sample_distance_path_proportion, generalOutputDir=os.path.join(generalFolder, "Tree", methods[0]), custom_tree_name="proportion")
                elif methods[0] == "UPGMA":
                    UPGMA(inputFilePath=sample_distance_path_expression, generalOutputDir=os.path.join(generalFolder, "Tree", methods[0]), custom_tree_name="expression")
                    UPGMA(inputFilePath=sample_distance_path_proportion, generalOutputDir=os.path.join(generalFolder, "Tree", methods[0]), custom_tree_name="proportion")
                expression_tree_path = os.path.join(generalFolder, "Tree", methods[0], "expression.nex")
                proportion_tree_path = os.path.join(generalFolder, "Tree", methods[0], "proportion.nex")
            elif len(methods) > 1:
                buildConsensus(sample_distance_paths=sample_distance_path_expression, generalFolder=generalFolder, methods=methods, custom_tree_names="expression")
                buildConsensus(sample_distance_paths=sample_distance_path_proportion, generalFolder=generalFolder, methods=methods, custom_tree_names="proportion")
                expression_tree_path = os.path.join(generalFolder, "Tree", "consensus", "expression.nex")
                proportion_tree_path = os.path.join(generalFolder, "Tree", "consensus", "proportion.nex")
            
            expr_results = cut_tree_by_group_count(expression_tree_path, desired_groups=number_of_clusters, format='nexus', verbose=True, tol=0)
            prop_results = cut_tree_by_group_count(proportion_tree_path, desired_groups=number_of_clusters, format='nexus', verbose=True, tol=0)
            
            expr_output_dir = os.path.join(generalFolder, "Cluster_DEG", "Tree_expression")
            prop_output_dir = os.path.join(generalFolder, "Cluster_DEG", "Tree_proportion")
            
            os.makedirs(expr_output_dir, exist_ok=True)
            os.makedirs(prop_output_dir, exist_ok=True)
            
            cluster_dge_visualization(sample_to_clade=expr_results, folder_path=pseudobulk_folder_path, output_dir=expr_output_dir)
            cluster_dge_visualization(sample_to_clade=prop_results, folder_path=pseudobulk_folder_path, output_dir=prop_output_dir)
            
            unique_expr_clades = len(set(expr_results.values()))
            if unique_expr_clades <= 1:
                print("Only one clade found in expression-based tree clustering. Skipping multi-clade DGE analysis.")
            elif unique_expr_clades > 2:
                print("Conducting multi-clade DGE analysis for expression-based tree clustering.")
                multi_clade_dge_analysis(sample_to_clade=expr_results, folder_path=pseudobulk_folder_path, output_dir=expr_output_dir)

            unique_prop_clades = len(set(prop_results.values()))
            if unique_prop_clades <= 1:
                print("Only one clade found in proportion-based tree clustering. Skipping multi-clade DGE analysis.")
            elif unique_prop_clades > 2:
                print("Conducting multi-clade DGE analysis for proportion-based tree clustering.")
                multi_clade_dge_analysis(sample_to_clade=prop_results, folder_path=pseudobulk_folder_path, output_dir=prop_output_dir)
        except Exception as e:
            print(f"Error in tree-based clustering: {e}")

    if prportion_test:
        try:
            if Kmeans:
                expr_results = expr_results_Kmeans
                prop_results = prop_results_Kmeans
                
            if expr_results is not None:
                unique_expr_clades = len(set(expr_results.values()))
                if unique_expr_clades <= 1:
                    print("Only one clade found in expression results. Skipping proportion DGE test.")
                else:
                    proportion_DGE_test(generalFolder, expr_results, sub_folder="expression", verbose=False)
            
            if prop_results is not None:
                unique_prop_clades = len(set(prop_results.values()))
                if unique_prop_clades <= 1:
                    print("Only one clade found in proportion results. Skipping proportion DGE test.")
                else:
                    proportion_DGE_test(generalFolder, prop_results, sub_folder="proportion", verbose=False)
        except Exception as e:
            print(f"Error in proportion test: {e}")
    
    if RAISIN_analysis:
        try:
            if expr_results is not None:
                unique_expr_clades = len(set(expr_results.values()))
                if unique_expr_clades <= 1:
                    print("Only one clade found in expression results. Skipping RAISIN analysis.")
                else:
                    fit_results, test_results = RAISIN(generalFolder, expr_results)
                    print(expr_results)
            else:
                print("No expression results available. Skipping RAISIN analysis.")
            if prop_results is not None:
                unique_prop_clades = len(set(prop_results.values()))
                if unique_prop_clades <= 1:
                    print("Only one clade found in proportion results. Skipping RAISIN analysis.")
                else:
                    fit_results, test_results = RAISIN(generalFolder, prop_results)
                    print(prop_results)
            else:
                print("No proportion results available. Skipping RAISIN analysis.")
        except Exception as e:
            print(f"Error in RAISIN analysis: {e}")