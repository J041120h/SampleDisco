from sampledisco.sample_clustering.cluster_helper import expression_tree_to_nexus


def HRA_VEC(inputFilePath, generalOutputDir, custom_tree_name=None):
    """Hierarchical clustering with AVERAGE linkage on Euclidean distances.

    Thin wrapper around `expression_tree_to_nexus(linkage_method='average')`.
    """
    expression_tree_to_nexus(
        inputFilePath=inputFilePath,
        generalOutputDir=generalOutputDir,
        linkage_method="average",
        tree_label_prefix="HRA",
        custom_tree_name=custom_tree_name,
    )
