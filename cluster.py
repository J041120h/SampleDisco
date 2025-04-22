from sample_clustering.cluster_DEG_visualization import *
from sample_clustering.consensus import *
from sample_clustering.HRA_VEC import *
from sample_clustering.HRC_VEC import *
from sample_clustering.Kmeans_cluster import *
from sample_clustering.NN import *
from sample_clustering.tree_cut import *
from sample_clustering.UPGMA import *

def cluster(
    Kmeans: bool = False,
    methods: list = ['HRA_VEC', 'HRC_VEC', 'NN', 'tree', 'UPGMA'],
    generalFolder: str = None,
    distance_method: str = "cosine",
    number_of_clusters: int = 5
):
    if Kmeans == False and methods is None:
        raise ValueError("Please provide at least one clustering method or set Kmeans to True.")
    
    pseudobulk_folder_path = os.path.join(generalFolder, "pseudobulk")
    sample_distance_path_proportion = os.path.join(generalFolder, "Sample", distance_method, "cell_expression", "distance_matrix_expression.csv")
    sample_distance_path_expression = os.path.join(generalFolder, "Sample", distance_method, "cell_proportion", "distance_matrix_proportion.csv")
    if Kmeans:
        expr_results, prop_results = cluster_samples_from_folder(folder_path = pseudobulk_folder_path , n_clusters = number_of_clusters)
        # Define output directories
        expr_output_dir = os.path.join(generalFolder, "Cluster_DEG", "Kmeans_expression")
        prop_output_dir = os.path.join(generalFolder, "Cluster_DEG", "Kmeans_proporton")

        # Make sure directories exist
        os.makedirs(expr_output_dir, exist_ok=True)
        os.makedirs(prop_output_dir, exist_ok=True)

        # Call the visualization function
        cluster_dge_visualization(sample_to_clade=expr_results, folder_path=pseudobulk_folder_path, output_dir=expr_output_dir)
        cluster_dge_visualization(sample_to_clade=prop_results, folder_path=pseudobulk_folder_path, output_dir=prop_output_dir)

    if len(methods) == 1:
        print(f"Running {methods[0]} for tree construction...")
        if methods[0] == "HRA_VEC":
            HRA_VEC(inputFilePath=sample_distance_path_proportion, generalOutputDir=os.path.join(generalFolder, "Tree", methods[0]), custom_tree_name = "expression")
            HRA_VEC(inputFilePath=sample_distance_path_expression, generalOutputDir=os.path.join(generalFolder, "Tree", methods[0]), custom_tree_name = "proportion")
        elif methods[0] == "HRC_VEC":
            HRC_VEC(inputFilePath=sample_distance_path_proportion, generalOutputDir=os.path.join(generalFolder, "Tree", methods[0]), custom_tree_name = "expression")
            HRC_VEC(inputFilePath=sample_distance_path_expression, generalOutputDir=os.path.join(generalFolder, "Tree", methods[0]), custom_tree_name = "proportion")
        elif methods[0] == "NN":
            NN(inputFilePath=sample_distance_path_proportion, generalOutputDir=os.path.join(generalFolder, "Tree", methods[0]), custom_tree_name = "expression")
            NN(inputFilePath=sample_distance_path_expression, generalOutputDir=os.path.join(generalFolder, "Tree", methods[0]), custom_tree_name = "proportion")
        elif methods[0] == "UPGMA":
            UPGMA(inputFilePath=sample_distance_path_proportion, generalOutputDir=os.path.join(generalFolder, "Tree", methods[0]), custom_tree_name = "expression")
            UPGMA(inputFilePath=sample_distance_path_expression, generalOutputDir=os.path.join(generalFolder, "Tree", methods[0]), custom_tree_name = "proportion")
    elif len(methods) > 1:
        buildConsensus(sample_distance_paths = sample_distance_path_expression, generalFolder = generalFolder, methods=methods, custom_tree_names="expression")
        buildConsensus(sample_distance_paths = sample_distance_path_proportion, generalFolder = generalFolder, methods=methods, custom_tree_names="proportion")
            
if __name__ == "__main__":
    cluster(
        Kmeans=False,
        methods=['UPGMA', 'NN'],
        generalFolder="/Users/harry/Desktop/GenoDistance/result/",
        number_of_clusters=4
    )