import os
import pandas as pd
import numpy as np
from anndata import AnnData
from typing import List, Optional


def distanceCheck(
    distance_df: pd.DataFrame,
    row: str,
    method: str,
    output_dir: str,
    adata: AnnData = None,
    grouping_columns: List[str] = ['sev.level'],
    summary_csv_path: Optional[str] = None,
    allow_prefix_fallback: bool = False
) -> float:
    """
    Calculate in-group vs. between-group distances based on sample grouping.

    Parameters
    ----------
    distance_df : pd.DataFrame
        Distance matrix with samples as both index and columns
    row : str
        Row name in the summary CSV to update
    method : str
        Distance method used (e.g., 'cosine', 'euclidean')
    output_dir : str
        Directory to save results
    adata : AnnData or None
        Pseudobulked AnnData with sample metadata in `adata.obs`
    grouping_columns : list of str
        Column names in `adata.obs` for grouping samples
    summary_csv_path : str or None
        Path to summary CSV file
    allow_prefix_fallback : bool
        If True, fall back to grouping samples by their first two characters
        when no valid grouping column is found. Off by default; when off,
        a missing/invalid grouping raises a ValueError.

    Returns
    -------
    float
        Distance score (between-group / in-group distance)
    """
    samples = distance_df.index.tolist()
    os.makedirs(output_dir, exist_ok=True)

    groups = _get_sample_groups(samples, adata, grouping_columns, allow_prefix_fallback)
    in_group, between_group = _compute_distances(samples, distance_df, groups)

    avg_in = np.mean(in_group) if in_group else np.nan
    avg_between = np.mean(between_group) if between_group else np.nan
    score = _calculate_score(avg_in, avg_between)

    _save_results(output_dir, row, method, samples, groups, in_group, between_group, avg_in, avg_between, score)
    _update_summary(summary_csv_path, row, method, score)

    return score


def _get_sample_groups(samples: List[str], adata: AnnData, grouping_columns: List[str],
                        allow_prefix_fallback: bool = False) -> dict:
    """Determine sample groupings from AnnData. Raises if no valid grouping column is
    found, unless `allow_prefix_fallback` explicitly opts into prefix-based grouping."""
    if adata is None or not hasattr(adata, 'obs') or adata.obs.empty:
        if allow_prefix_fallback:
            print("Warning: No adata provided or adata.obs is empty, using fallback grouping")
            return {sample: sample[:2] for sample in samples}
        raise ValueError("No adata provided or adata.obs is empty; cannot determine sample groups")

    available_samples = [s for s in samples if s in adata.obs.index]
    if not available_samples:
        if allow_prefix_fallback:
            print("Warning: No samples from distance matrix found in adata.obs, using fallback grouping")
            return {sample: sample[:2] for sample in samples}
        raise ValueError("No samples from distance matrix found in adata.obs; cannot determine sample groups")

    grouping_column = next((col for col in grouping_columns if col in adata.obs.columns), None)
    if grouping_column is None:
        if allow_prefix_fallback:
            print(f"Warning: None of the grouping columns {grouping_columns} found in adata.obs, using fallback grouping")
            return {sample: sample[:2] for sample in samples}
        raise ValueError(f"None of the grouping columns {grouping_columns} found in adata.obs")

    groups = {s: str(adata.obs.loc[s, grouping_column]) for s in available_samples}
    for sample in samples:
        groups.setdefault(sample, 'Unknown')
    return groups


def _compute_distances(samples: List[str], distance_df: pd.DataFrame, groups: dict) -> tuple:
    """Compute in-group and between-group distances."""
    in_group, between_group = [], []
    
    for i, sample_i in enumerate(samples):
        for j in range(i + 1, len(samples)):
            sample_j = samples[j]
            distance = distance_df.iloc[i, j]
            
            if groups[sample_i] == groups[sample_j]:
                in_group.append(distance)
            else:
                between_group.append(distance)
    
    return in_group, between_group


def _calculate_score(avg_in: float, avg_between: float) -> float:
    """Calculate the distance score."""
    if np.isnan(avg_in) or avg_in == 0:
        return np.nan
    return avg_between / avg_in


def _save_results(output_dir: str, row: str, method: str, samples: list, groups: dict,
                  in_group: list, between_group: list, avg_in: float, avg_between: float, score: float):
    """Save results to text file."""
    group_counts = dict(pd.Series(list(groups.values())).value_counts())
    
    result_str = f"""Distance Check Results for {row} using {method}
{'='*50}
Number of samples: {len(samples)}
Number of groups: {len(set(groups.values()))}
Group distribution: {group_counts}
Number of in-group pairs: {len(in_group)}
Number of between-group pairs: {len(between_group)}
Average in-group distance: {avg_in:.6f}
Average between-group distance: {avg_between:.6f}
Score (between/in-group): {score:.6f}

Interpretation:
- Higher scores indicate better separation between groups
- Score > 1: Groups are more distant from each other than within groups
- Score < 1: Groups are closer to each other than within groups
"""
    
    output_file = os.path.join(output_dir, f'distance_check_results_{row}_{method}.txt')
    with open(output_file, 'w') as f:
        f.write(result_str)
    
    print(f"Distance check results saved to {output_file}")
    print(f"Score for {row} ({method}): {score:.6f}")


def _update_summary(summary_csv_path: Optional[str], row: str, method: str, score: float):
    """Update summary CSV file if path provided."""
    if summary_csv_path is None:
        return
    
    try:
        summary_df = pd.read_csv(summary_csv_path, index_col=0) if os.path.isfile(summary_csv_path) else pd.DataFrame()
        if method not in summary_df.columns:
            summary_df[method] = np.nan
        summary_df.loc[row, method] = score
        summary_df.to_csv(summary_csv_path)
        print(f"Summary updated in {summary_csv_path}")
    except Exception as e:
        print(f"Warning: Failed to update summary CSV: {e}")