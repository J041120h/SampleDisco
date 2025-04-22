from Bio import Phylo
import numpy as np
from collections import Counter

def cut_tree_by_group_count(tree_path, desired_groups, format='nexus', verbose=False, tol=0):
    """
    Cut a phylogenetic tree into approximately `desired_groups` clades with ≥2 samples each.
    Singleton clades (with only 1 sample) are ignored.

    Parameters:
        tree_path (str): Path to the tree file.
        desired_groups (int): Number of valid clades (groups with ≥2 samples).
        format (str): Tree file format. Default is 'nexus'.
        verbose (bool): Print intermediate results if True.
        max_iter (int): Max number of iterations for binary search on resolution.
        tol (int): Allowed deviation from desired number of groups.

    Returns:
        dict: sample_to_clade mapping (only for samples in valid clades).
    """
    tree = Phylo.read(tree_path, format)

    def get_max_depth(clade, current=0):
        depths = [get_max_depth(c, current + c.branch_length) for c in clade.clades]
        return max(depths) if depths else current

    def assign_clades_at_resolution(resolution):
        max_height = get_max_depth(tree.root)
        threshold = resolution * max_height

        clade_id = 1
        sample_to_clade = {}

        def collect_leaves(clade, cid):
            for t in clade.get_terminals():
                sample_to_clade[t.name] = cid

        def traverse(node, depth=0):
            nonlocal clade_id
            for child in node.clades:
                child_depth = depth + child.branch_length
                if depth < threshold <= child_depth:
                    collect_leaves(child, clade_id)
                    clade_id += 1
                else:
                    traverse(child, child_depth)

        traverse(tree.root)

        # Assign unassigned leaves to singleton clades
        for leaf in tree.get_terminals():
            if leaf.name not in sample_to_clade:
                sample_to_clade[leaf.name] = clade_id
                clade_id += 1

        # Filter to clades with ≥2 samples
        counts = Counter(sample_to_clade.values())
        valid_clades = {cid for cid, cnt in counts.items() if cnt >= 2}
        valid_mapping = {
            sample: cid for sample, cid in sample_to_clade.items()
            if cid in valid_clades
        }
        return valid_mapping

    # Validate desired group count
    all_samples = [leaf.name for leaf in Phylo.read(tree_path, format).get_terminals()]
    max_groups = len(all_samples) // 2
    if desired_groups < 2 or desired_groups > max_groups:
        raise ValueError(f"desired_groups must be between 2 and {max_groups} (got {desired_groups})")

    # Binary search on resolution
    low, high = 0.0, 1.0
    best_result = {}
    for _ in range(100):
        mid = (low + high) / 2
        sample_to_clade = assign_clades_at_resolution(mid)
        num_groups = len(set(sample_to_clade.values()))

        if verbose:
            print(f"Resolution={mid:.3f} → Valid Groups={num_groups}")

        if abs(num_groups - desired_groups) <= tol:
            return sample_to_clade

        if num_groups < desired_groups:
            high = mid
        else:
            low = mid

        best_result = sample_to_clade

    if verbose:
        print("Returning closest match found.")
    return best_result

# Example usage (not executed when imported):
if __name__ == "__main__":
    tree_path = "/Users/harry/Desktop/GenoDistance/result/Tree/Combined/Consensus.nex"
    desired_groups = 4
    cut_tree_by_group_count(tree_path, desired_groups, format='nexus', verbose=True, tol=0)
