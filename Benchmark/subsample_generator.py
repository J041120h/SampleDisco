import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path
from typing import List, Dict, Tuple
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
                      target_dist: Dict[str, float]) -> List[str]:
    """
    Perform stratified sampling to maintain severity level distribution.
    """
    # Group samples by severity level
    sev_groups = meta_df.groupby('sev.level')
    
    selected_samples = []
    
    # Calculate how many samples to take from each severity level
    for sev_level, proportion in target_dist.items():
        if sev_level not in sev_groups.groups:
            continue
            
        # Calculate target number for this severity level
        n_target = int(np.round(proportion * n_samples))
        
        # Get available samples for this severity level
        available_samples = sev_groups.get_group(sev_level)['sample'].tolist()
        
        # Sample without replacement (or take all if not enough)
        n_to_sample = min(n_target, len(available_samples))
        if n_to_sample > 0:
            sampled = np.random.choice(available_samples, 
                                      size=n_to_sample, 
                                      replace=False)
            selected_samples.extend(sampled)
    
    # If we have fewer samples than requested, fill up randomly from remaining
    if len(selected_samples) < n_samples:
        remaining_samples = meta_df[~meta_df['sample'].isin(selected_samples)]['sample'].tolist()
        n_additional = min(n_samples - len(selected_samples), len(remaining_samples))
        if n_additional > 0:
            additional = np.random.choice(remaining_samples, 
                                        size=n_additional, 
                                        replace=False)
            selected_samples.extend(additional)
    
    # If we have more samples than requested, randomly remove some
    if len(selected_samples) > n_samples:
        selected_samples = np.random.choice(selected_samples, 
                                          size=n_samples, 
                                          replace=False)
    
    return list(selected_samples)

def subsample_adata(adata: ad.AnnData, selected_samples: List[str]) -> ad.AnnData:
    """Subsample adata to only include cells from selected samples."""
    # Create mask for cells belonging to selected samples
    mask = adata.obs['sample'].isin(selected_samples)
    
    # Subsample
    adata_sub = adata[mask].copy()
    
    return adata_sub

def record_subsample_stats(adata_sub: ad.AnnData, meta_sub: pd.DataFrame, 
                          n_samples: int, output_path: str, summary_lines: List[str]):
    """Record statistics for the subsampled dataset to summary."""
    summary_lines.append(f"\n{'='*60}")
    summary_lines.append(f"SUBSAMPLE: {n_samples} samples")
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
        for batch in sorted(batch_counts.index):
            count = batch_counts[batch]
            prop = count / len(meta_sub)
            summary_lines.append(f"  {batch}: {count} samples ({prop*100:.1f}%)")

def main(h5ad_path: str, meta_csv_path: str, output_dir: str = None, 
         sample_sizes: List[int] = [25, 50, 100, 200], seed: int = 42):
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
    
    print("Starting subsampling process...")
    
    # Load data
    adata, meta_df = load_data(h5ad_path, meta_csv_path)
    
    # Ensure 'sample' column exists in metadata
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
            n_samples = len(meta_filtered)
        
        # Perform stratified sampling
        selected_samples = stratified_sample(meta_filtered, n_samples, original_sev_dist)
        
        # Subsample adata
        adata_sub = subsample_adata(adata, selected_samples)
        
        # Get metadata for selected samples
        meta_sub = meta_filtered[meta_filtered['sample'].isin(selected_samples)]
        
        # Generate output filename
        input_stem = Path(h5ad_path).stem
        output_filename = f"{input_stem}_subsample_{n_samples}samples.h5ad"
        output_path = output_dir / output_filename
        
        # Save subsampled data
        summary_lines.append(f"Saving to: {output_path}")
        print(f"  Saving {n_samples} samples ({adata_sub.n_obs} cells) to: {output_filename}")
        adata_sub.write_h5ad(output_path)
        
        # Record statistics
        record_subsample_stats(adata_sub, meta_sub, n_samples, output_path, summary_lines)
    
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
        sample_sizes=[25, 50, 100, 200],  # Create 4 subsampled files
        seed=42  # For reproducibility
    )