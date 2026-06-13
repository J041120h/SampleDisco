from sampledisco.sample_clustering.cluster_helper import expression_tree_to_nexus


def HRC_VEC(inputFilePath, generalOutputDir, custom_tree_name=None):
    """Hierarchical clustering with COMPLETE linkage on Euclidean distances.

    Thin wrapper around `expression_tree_to_nexus(linkage_method='complete')`.
    """
    expression_tree_to_nexus(
        inputFilePath=inputFilePath,
        generalOutputDir=generalOutputDir,
        linkage_method="complete",
        tree_label_prefix="HRC",
        custom_tree_name=custom_tree_name,
    )
