import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_data(h5ad_path: str, meta_csv_path: str) -> Tuple[ad.AnnData, pd.DataFrame]:
    """Load h5ad file and metadata CSV."""
    print(f"Loading h5ad file from: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    
    print(f"Loading metadata from: {meta_csv_path}")
    meta_df = pd.read_csv(meta_csv_path)
    
    return adata, meta_df

def analyze_original_distribution(adata: ad.AnnData, meta_df: pd.DataFrame, summary_lines: List[str]) -> Dict:
    """Analyze the original distribution of samples by severity level and batch."""
    # Get unique samples from adata
    samples_in_adata = adata.obs['sample'].unique()
    
    # Filter metadata to only include samples in adata
    meta_filtered = meta_df[meta_df['sample'].isin(samples_in_adata)].copy()
    
    # Count distribution
    sev_dist = meta_filtered['sev.level'].value_counts(normalize=True).to_dict()
    
    # Check if batch column exists
    batch_dist = None
    if 'batch' in meta_filtered.columns:
        batch_dist = meta_filtered['batch'].value_counts(normalize=True).to_dict()
    
    summary_lines.append("\n" + "="*60)
    summary_lines.append("ORIGINAL DATASET STATISTICS")
    summary_lines.append("="*60)
    summary_lines.append(f"Total samples: {len(samples_in_adata)}")
    summary_lines.append(f"Total cells: {adata.n_obs}")
    summary_lines.append(f"Average cells per sample: {adata.n_obs / len(samples_in_adata):.1f}")
    
    summary_lines.append("\nSeverity Level Distribution:")
    for level, prop in sorted(sev_dist.items()):
        count = meta_filtered['sev.level'].value_counts()[level]
        summary_lines.append(f"  {level}: {count} samples ({prop*100:.1f}%)")
    
    if batch_dist:
        summary_lines.append("\nBatch Distribution:")
        for batch, prop in sorted(batch_dist.items()):
            count = meta_filtered['batch'].value_counts()[batch]
            summary_lines.append(f"  {batch}: {count} samples ({prop*100:.1f}%)")
    
    print(f"Original dataset: {len(samples_in_adata)} samples, {adata.n_obs} cells")
    
    return {
        'sev_dist': sev_dist,
        'batch_dist': batch_dist,
        'meta_filtered': meta_filtered
    }

def stratified_sample(meta_df: pd.DataFrame, n_samples: int, 
                      target_dist: Dict[str, float],
                      min_batch_samples: int = 3) -> List[str]:
    """
    Perform stratified sampling to maintain severity level distribution.
    Ensures that if a batch is included, it has at least min_batch_samples samples.
    
    Parameters:
    -----------
    meta_df : pd.DataFrame
        Metadata dataframe with 'sample', 'sev.level', and optionally 'batch' columns
    n_samples : int
        Target number of samples to select
    target_dist : Dict[str, float]
        Target distribution of severity levels
    min_batch_samples : int
        Minimum number of samples required per batch (default: 3)
    
    Returns:
    --------
    List[str]
        List of selected sample IDs
    """
    has_batch = 'batch' in meta_df.columns
    
    # Initial stratified sampling by severity level
    sev_groups = meta_df.groupby('sev.level')
    selected_samples = []
    
    # Calculate how many samples to take from each severity level
    for sev_level, proportion in target_dist.items():
        if sev_level not in sev_groups.groups:
            continue
            
        n_target = int(np.round(proportion * n_samples))
        available_samples = sev_groups.get_group(sev_level)['sample'].tolist()
        n_to_sample = min(n_target, len(available_samples))
        
        if n_to_sample > 0:
            sampled = np.random.choice(available_samples, 
                                      size=n_to_sample, 
                                      replace=False)
            selected_samples.extend(sampled)
    
    # If no batch column, just adjust to target size and return
    if not has_batch:
        return _adjust_sample_size(meta_df, selected_samples, n_samples)
    
    # Enforce batch constraint: each included batch must have >= min_batch_samples
    selected_samples = _enforce_batch_constraint(
        meta_df, selected_samples, n_samples, min_batch_samples
    )
    
    return selected_samples

def _enforce_batch_constraint(meta_df: pd.DataFrame,
                              selected_samples: List[str],
                              n_samples: int,
                              min_batch_samples: int) -> List[str]:
    """
    Enforce the minimum batch samples constraint.
    If a batch has fewer than min_batch_samples, either complete it or remove it entirely.
    """
    selected_set = set(selected_samples)
    
    max_iterations = 50
    for iteration in range(max_iterations):
        # Get current batch counts
        meta_selected = meta_df[meta_df['sample'].isin(selected_set)]
        batch_counts = meta_selected['batch'].value_counts()
        
        # Find batches violating constraint (1 to min_batch_samples-1 samples)
        violating_batches = batch_counts[
            (batch_counts > 0) & (batch_counts < min_batch_samples)
        ].index.tolist()
        
        if not violating_batches:
            break
        
        for batch in violating_batches:
            current_count = batch_counts[batch]
            needed = min_batch_samples - current_count
            
            # Find available samples from this batch not yet selected
            available_from_batch = meta_df[
                (meta_df['batch'] == batch) & 
                (~meta_df['sample'].isin(selected_set))
            ]['sample'].tolist()
            
            if len(available_from_batch) >= needed:
                # Can complete this batch - add more samples from it
                to_add = np.random.choice(available_from_batch, size=needed, replace=False)
                selected_set.update(to_add)
            else:
                # Cannot complete this batch - remove all its samples
                samples_to_remove = meta_df[
                    (meta_df['batch'] == batch) & 
                    (meta_df['sample'].isin(selected_set))
                ]['sample'].tolist()
                selected_set -= set(samples_to_remove)
    
    selected_samples = list(selected_set)
    
    # Adjust to target n_samples while respecting batch constraint
    if len(selected_samples) > n_samples:
        selected_samples = _trim_with_batch_constraint(
            meta_df, selected_samples, n_samples, min_batch_samples
        )
    elif len(selected_samples) < n_samples:
        selected_samples = _fill_with_batch_constraint(
            meta_df, selected_samples, n_samples, min_batch_samples
        )
    
    return selected_samples

def _trim_with_batch_constraint(meta_df: pd.DataFrame,
                                selected_samples: List[str],
                                n_samples: int,
                                min_batch_samples: int) -> List[str]:
    """
    Remove samples to reach target size while maintaining batch constraint.
    Only removes samples from batches that would still have >= min_batch_samples after removal.
    """
    selected_set = set(selected_samples)
    
    while len(selected_set) > n_samples:
        meta_selected = meta_df[meta_df['sample'].isin(selected_set)]
        batch_counts = meta_selected['batch'].value_counts()
        
        # Find batches with more than min_batch_samples (safe to remove from)
        removable_batches = batch_counts[batch_counts > min_batch_samples].index.tolist()
        
        if not removable_batches:
            # Cannot remove more without violating constraint
            break
        
        # Get samples from removable batches
        removable_samples = meta_df[
            (meta_df['batch'].isin(removable_batches)) &
            (meta_df['sample'].isin(selected_set))
        ]['sample'].tolist()
        
        if not removable_samples:
            break
        
        # Remove one random sample
        to_remove = np.random.choice(removable_samples, size=1)[0]
        selected_set.remove(to_remove)
    
    return list(selected_set)

def _fill_with_batch_constraint(meta_df: pd.DataFrame,
                                selected_samples: List[str],
                                n_samples: int,
                                min_batch_samples: int) -> List[str]:
    """
    Add samples to reach target size while maintaining batch constraint.
    Prioritizes adding to existing batches, then adds new batches with >= min_batch_samples.
    """
    selected_set = set(selected_samples)
    
    while len(selected_set) < n_samples:
        meta_selected = meta_df[meta_df['sample'].isin(selected_set)]
        current_batches = meta_selected['batch'].unique() if len(meta_selected) > 0 else []
        
        # First priority: add samples from already-included batches
        available_from_current = meta_df[
            (meta_df['batch'].isin(current_batches)) &
            (~meta_df['sample'].isin(selected_set))
        ]['sample'].tolist()
        
        if available_from_current:
            to_add = np.random.choice(available_from_current, size=1)[0]
            selected_set.add(to_add)
            continue
        
        # Second priority: add a new batch (must add at least min_batch_samples)
        remaining_needed = n_samples - len(selected_set)
        
        # Find batches not yet included that have enough samples
        all_batches = meta_df['batch'].unique()
        new_batches = [b for b in all_batches if b not in current_batches]
        
        # Check which new batches have enough available samples
        eligible_new_batches = []
        for batch in new_batches:
            available = meta_df[
                (meta_df['batch'] == batch) &
                (~meta_df['sample'].isin(selected_set))
            ]['sample'].tolist()
            if len(available) >= min_batch_samples:
                eligible_new_batches.append((batch, available))
        
        if eligible_new_batches and remaining_needed >= min_batch_samples:
            # Add min_batch_samples from a random eligible batch
            batch, available = eligible_new_batches[np.random.randint(len(eligible_new_batches))]
            to_add = np.random.choice(available, size=min_batch_samples, replace=False)
            selected_set.update(to_add)
        else:
            # Cannot add more samples while respecting constraint
            break
    
    return list(selected_set)

def _adjust_sample_size(meta_df: pd.DataFrame, 
                        selected_samples: List[str], 
                        n_samples: int) -> List[str]:
    """Adjust sample list to target size (no batch constraint)."""
    selected_samples = list(selected_samples)
    
    if len(selected_samples) < n_samples:
        remaining = meta_df[~meta_df['sample'].isin(selected_samples)]['sample'].tolist()
        n_additional = min(n_samples - len(selected_samples), len(remaining))
        if n_additional > 0:
            additional = np.random.choice(remaining, size=n_additional, replace=False)
            selected_samples.extend(additional)
    
    if len(selected_samples) > n_samples:
        selected_samples = list(np.random.choice(selected_samples, size=n_samples, replace=False))
    
    return selected_samples

def subsample_adata(adata: ad.AnnData, selected_samples: List[str]) -> ad.AnnData:
    """Subsample adata to only include cells from selected samples."""
    mask = adata.obs['sample'].isin(selected_samples)
    adata_sub = adata[mask].copy()
    return adata_sub

def record_subsample_stats(adata_sub: ad.AnnData, meta_sub: pd.DataFrame, 
                          n_samples: int, output_path: str, summary_lines: List[str],
                          min_batch_samples: int = 3):
    """Record statistics for the subsampled dataset to summary."""
    summary_lines.append(f"\n{'='*60}")
    summary_lines.append(f"SUBSAMPLE: {n_samples} samples (requested)")
    summary_lines.append(f"{'='*60}")
    summary_lines.append(f"Output file: {output_path}")
    summary_lines.append(f"Total samples: {len(meta_sub)}")
    summary_lines.append(f"Total cells: {adata_sub.n_obs}")
    summary_lines.append(f"Average cells per sample: {adata_sub.n_obs / len(meta_sub):.1f}")
    
    # Severity level distribution
    summary_lines.append("\nSeverity Level Distribution:")
    sev_counts = meta_sub['sev.level'].value_counts()
    for level in sorted(sev_counts.index):
        count = sev_counts[level]
        prop = count / len(meta_sub)
        summary_lines.append(f"  {level}: {count} samples ({prop*100:.1f}%)")
    
    # Batch distribution if exists
    if 'batch' in meta_sub.columns:
        summary_lines.append("\nBatch Distribution:")
        batch_counts = meta_sub['batch'].value_counts()
        
        # Check constraint
        violating = batch_counts[(batch_counts > 0) & (batch_counts < min_batch_samples)]
        if len(violating) > 0:
            summary_lines.append(f"  WARNING: Batch constraint violated for: {violating.to_dict()}")
        else:
            summary_lines.append(f"  (All batches have >= {min_batch_samples} samples)")
        
        for batch in sorted(batch_counts.index):
            count = batch_counts[batch]
            prop = count / len(meta_sub)
            summary_lines.append(f"  {batch}: {count} samples ({prop*100:.1f}%)")

def main(h5ad_path: str, meta_csv_path: str, output_dir: str = None, 
         sample_sizes: List[int] = [25, 50, 100, 200], seed: int = 42,
         min_batch_samples: int = 3):
    """
    Main function to perform subsampling.
    
    Parameters:
    -----------
    h5ad_path : str
        Path to the input h5ad file
    meta_csv_path : str
        Path to the metadata CSV file
    output_dir : str
        Directory to save output files (default: same as input h5ad)
    sample_sizes : List[int]
        List of sample sizes to create
    seed : int
        Random seed for reproducibility
    min_batch_samples : int
        Minimum samples per batch if batch is included (default: 3)
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Initialize summary lines
    summary_lines = []
    
    # Add header information
    summary_lines.append("="*60)
    summary_lines.append("H5AD SUBSAMPLING SUMMARY REPORT")
    summary_lines.append("="*60)
    summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"Input h5ad: {h5ad_path}")
    summary_lines.append(f"Input metadata: {meta_csv_path}")
    summary_lines.append(f"Random seed: {seed}")
    summary_lines.append(f"Requested sample sizes: {sample_sizes}")
    summary_lines.append(f"Minimum samples per batch: {min_batch_samples}")
    
    print("Starting subsampling process...")
    print(f"Batch constraint: minimum {min_batch_samples} samples per batch")
    
    # Load data
    adata, meta_df = load_data(h5ad_path, meta_csv_path)
    
    # Ensure required columns exist
    if 'sample' not in meta_df.columns:
        raise ValueError("Metadata CSV must contain a 'sample' column")
    
    if 'sev.level' not in meta_df.columns:
        raise ValueError("Metadata CSV must contain a 'sev.level' column")
    
    # Analyze original distribution
    dist_info = analyze_original_distribution(adata, meta_df, summary_lines)
    original_sev_dist = dist_info['sev_dist']
    meta_filtered = dist_info['meta_filtered']
    
    # Set output directory
    if output_dir is None:
        output_dir = Path(h5ad_path).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_lines.append(f"\nOutput directory: {output_dir}")
    
    # Create subsamples
    for i, n_samples in enumerate(sample_sizes, 1):
        print(f"Creating subsample {i}/{len(sample_sizes)}: {n_samples} samples...")
        
        summary_lines.append(f"\n{'*'*60}")
        summary_lines.append(f"Creating subsample with {n_samples} samples...")
        summary_lines.append(f"{'*'*60}")
        
        # Check if we have enough samples
        if n_samples > len(meta_filtered):
            warning_msg = f"WARNING: Requested {n_samples} samples but only {len(meta_filtered)} available."
            summary_lines.append(warning_msg)
            print(f"  {warning_msg}")
            n_samples_actual = len(meta_filtered)
        else:
            n_samples_actual = n_samples
        
        # Perform stratified sampling with batch constraint
        selected_samples = stratified_sample(
            meta_filtered, n_samples_actual, original_sev_dist,
            min_batch_samples=min_batch_samples
        )
        
        # Note if final count differs from requested
        if len(selected_samples) != n_samples:
            note_msg = f"NOTE: Final sample count is {len(selected_samples)} (requested {n_samples}) due to batch constraint"
            summary_lines.append(note_msg)
            print(f"  {note_msg}")
        
        # Subsample adata
        adata_sub = subsample_adata(adata, selected_samples)
        
        # Get metadata for selected samples
        meta_sub = meta_filtered[meta_filtered['sample'].isin(selected_samples)]
        
        # Generate output filename (use actual count in filename)
        input_stem = Path(h5ad_path).stem
        output_filename = f"{input_stem}_subsample_{len(selected_samples)}samples.h5ad"
        output_path = output_dir / output_filename
        
        # Save subsampled data
        summary_lines.append(f"Saving to: {output_path}")
        print(f"  Saving {len(selected_samples)} samples ({adata_sub.n_obs} cells) to: {output_filename}")
        adata_sub.write_h5ad(output_path)
        
        # Record statistics
        record_subsample_stats(adata_sub, meta_sub, n_samples, output_path, summary_lines, 
                              min_batch_samples=min_batch_samples)
    
    summary_lines.append(f"\n{'='*60}")
    summary_lines.append("SUBSAMPLING COMPLETE!")
    summary_lines.append(f"{'='*60}")
    summary_lines.append(f"Created {len(sample_sizes)} subsampled files in: {output_dir}")
    
    # Save summary to file
    summary_path = output_dir / f"subsampling_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"\nSubsampling complete! Created {len(sample_sizes)} files.")
    print(f"Summary report saved to: {summary_path}")

# Example usage
if __name__ == "__main__":
    # Configure paths
    h5ad_path = "/dcl01/hongkai/data/data/hjiang/Data/covid_data/count_data.h5ad"
    meta_csv_path = "/dcl01/hongkai/data/data/hjiang/Data/covid_data/sample_data.csv"
    
    # Run subsampling
    main(
        h5ad_path=h5ad_path,
        meta_csv_path=meta_csv_path,
        output_dir='/dcl01/hongkai/data/data/hjiang/Data/covid_data/Benchmark',
        sample_sizes=[25, 50, 100],  # Create 4 subsampled files
        seed=42,  # For reproducibility
        min_batch_samples=3  # Minimum samples per batch
    )