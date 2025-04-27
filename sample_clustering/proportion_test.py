import os
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.special import logit
from statsmodels.stats.multitest import multipletests

def proportion_DGE_test(folder_path, sample_to_clade, sub_folder, verbose=False):
    """
    Perform proportion test on logit-transformed proportions.

    If there are multiple clades, perform all pairwise comparisons.

    Args:
        folder_path (str): General folder path containing the proportion CSV.
        sample_to_clade (dict): Mapping from sample name to clade (group).
        verbose (bool): If True, print progress.
    """
    csv_path = os.path.join(folder_path, "pseudobulk", "proportion.csv")
    output_dir = os.path.join(folder_path, "Cluster_DEG", "proportion_test", sub_folder)
    os.makedirs(output_dir, exist_ok=True)

    # Step 2: Read CSV
    if verbose:
        print(f"Reading proportion data from {csv_path}...")
    prop = pd.read_csv(csv_path, index_col=0)

    # Step 3: Filter columns (samples) based on sample_to_clade keys
    valid_samples = list(sample_to_clade.keys())
    prop = prop[valid_samples]

    if verbose:
        print(f"Filtered to {len(valid_samples)} samples.")

    # Step 4: Fix boundary proportions (0 and 1)
    if verbose:
        print("Fixing boundary proportions (0 and 1)...")
    min_nonzero = prop[prop > 0].min().min()
    prop = prop.clip(lower=min_nonzero, upper=1 - min_nonzero)

    # Step 5: Logit transform
    if verbose:
        print("Applying logit transform...")
    prop_logit = logit(prop)

    # Step 6: Find unique clades
    unique_clades = sorted(set(sample_to_clade.values()))
    if verbose:
        print(f"Unique clades found: {unique_clades}")

    # Step 7: For each pair of clades, do the proportion test
    all_results = {}

    for clade1, clade2 in itertools.combinations(unique_clades, 2):
        if verbose:
            print(f"Comparing clade {clade1} vs clade {clade2}...")

        # Find samples belonging to clade1 and clade2
        selected_samples = [sample for sample, clade in sample_to_clade.items() if clade in [clade1, clade2]]
        selected_prop_logit = prop_logit[selected_samples]

        # Build design matrix
        selected_group_labels = [1 if sample_to_clade[sample] == clade1 else 0 for sample in selected_samples]
        X = np.column_stack((np.ones(len(selected_group_labels)), selected_group_labels))

        # Fit model for each cell type
        results = []
        for cell_type in selected_prop_logit.index:
            y = selected_prop_logit.loc[cell_type].values
            model = sm.OLS(y, X).fit()
            coef = model.params[1]        # Group coefficient (clade1 vs clade2)
            pval = model.pvalues[1]        # P-value
            results.append((cell_type, coef, pval))

        # Multiple testing correction
        df_result = pd.DataFrame(results, columns=['celltype', 'logFC', 'p_value'])
        fdr = multipletests(df_result['p_value'], method='fdr_bh')[1]
        df_result['FDR'] = fdr

        # Save results
        comparison_name = f"{clade1}_vs_{clade2}"
        output_path = os.path.join(output_dir, f"proportion_test_{comparison_name}.csv")
        df_result.to_csv(output_path, index=False)

        if verbose:
            print(f"Saved result for {comparison_name} to {output_path}")

        all_results[comparison_name] = df_result

    if verbose:
        print(f"All comparisons finished. Results saved to {output_dir}")

    return all_results
