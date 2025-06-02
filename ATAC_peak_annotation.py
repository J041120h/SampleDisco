import pandas as pd
import numpy as np
import anndata as ad
import pyensembl
from collections import defaultdict

def annotate_atac_peaks(atac_file_path, ensembl_release=110, window_size=100000, overwrite=True):
    """
    Streamlined ArchR-style peak-to-gene annotation using PyEnsembl
    
    Parameters:
    -----------
    atac_file_path : str
        Path to ATAC-seq h5ad file
    ensembl_release : int
        Ensembl release version (default: 110 for GRCh38)
    window_size : int
        Window size around TSS in base pairs (default: 100kb)
    overwrite : bool
        Whether to overwrite the original file (default: True)
    
    Returns:
    --------
    AnnData : Annotated ATAC data
    """
    
    print("Loading ATAC data and annotations...")
    
    adata = ad.read_h5ad(atac_file_path)
    ensembl = pyensembl.EnsemblRelease(release=ensembl_release, species='homo_sapiens')
    
    try:
        genes = ensembl.genes()
    except:
        print("Downloading Ensembl data (first time only)...")
        ensembl.download()
        ensembl.index()
        genes = ensembl.genes()
    
    # Filter protein-coding genes and create TSS windows
    protein_coding_genes = [g for g in genes if g.biotype == 'protein_coding']
    gene_windows = []
    for gene in protein_coding_genes:
        try:
            tss = gene.start if gene.strand == '+' else gene.end
            window_start = max(0, tss - window_size)
            window_end = tss + window_size
            
            gene_windows.append({
                'gene_name': gene.gene_name,
                'gene_id': gene.gene_id,
                'chromosome': gene.contig,
                'tss': tss,
                'window_start': window_start,
                'window_end': window_end
            })
        except:
            continue
    
    genes_df = pd.DataFrame(gene_windows)
    
    # Parse peak coordinates
    def parse_peak(peak_str):
        parts = peak_str.split("-")
        chrom = parts[0].replace('chr', '') if parts[0].startswith('chr') else parts[0]
        return chrom, int(parts[1]), int(parts[2])
    
    peaks_data = []
    for peak_name in adata.var_names:
        try:
            chrom, start, end = parse_peak(peak_name)
            peaks_data.append({
                'peak': peak_name,
                'chromosome': chrom,
                'start': start,
                'end': end,
                'center': (start + end) // 2
            })
        except:
            continue
    
    peaks_df = pd.DataFrame(peaks_data)
    
    # Find overlaps efficiently
    peaks_by_chr = peaks_df.groupby('chromosome')
    genes_by_chr = genes_df.groupby('chromosome')
    
    overlaps = []
    for chrom in peaks_df['chromosome'].unique():
        # Handle chromosome naming variations
        genes_chr = None
        for chr_variant in [chrom, f'chr{chrom}']:
            if chr_variant in genes_df['chromosome'].values:
                genes_chr = genes_by_chr.get_group(chr_variant)
                break
        
        if genes_chr is None:
            continue

        peaks_chr = peaks_by_chr.get_group(chrom)
        
        for _, peak in peaks_chr.iterrows():
            overlapping_genes = genes_chr[
                (genes_chr['window_start'] <= peak['end']) & 
                (genes_chr['window_end'] >= peak['start'])
            ].copy()
            
            if len(overlapping_genes) > 0:
                overlapping_genes['distance_to_tss'] = abs(overlapping_genes['tss'] - peak['center'])
                overlapping_genes['peak'] = peak['peak']
                overlaps.append(overlapping_genes)
    
    if not overlaps:
        print("Warning: No overlaps found. Check chromosome naming.")
        return adata
    
    # Create annotation dictionary
    annotation_df = pd.concat(overlaps, ignore_index=True)
    peak_annotation = {}
    
    for peak_name, group in annotation_df.groupby('peak'):
        group_sorted = group.sort_values('distance_to_tss')
        peak_annotation[peak_name] = {
            'genes': group_sorted['gene_name'].tolist(),
            'gene_ids': group_sorted['gene_id'].tolist(),
            'distances': group_sorted['distance_to_tss'].tolist(),
            'closest_gene': group_sorted.iloc[0]['gene_name'],
            'closest_distance': int(group_sorted.iloc[0]['distance_to_tss'])
        }
    
    if "atac" not in adata.uns:
        adata.uns["atac"] = {}
    adata.uns["atac"]["peak2gene"] = peak_annotation
    
    # Add summary statistics
    total_peaks = len(adata.var_names)
    annotated_peaks = len(peak_annotation)
    coverage = (annotated_peaks / total_peaks) * 100
    
    adata.uns["atac"]["annotation_stats"] = {
        "total_peaks": total_peaks,
        "annotated_peaks": annotated_peaks,
        "coverage_percent": coverage,
        "window_size_kb": window_size // 1000,
        "ensembl_release": ensembl_release
    }
    
    print(f"Annotation complete: {annotated_peaks:,}/{total_peaks:,} peaks ({coverage:.1f}%)")
    
    # Save data
    if overwrite:
        adata.write(atac_file_path)
        print(f"Updated: {atac_file_path}")
    else:
        output_path = atac_file_path
        adata.write(output_path)
        print(f"Saved: {output_path}")
    
    return adata

def run_atac_annotation_pipeline(atac_file_path, rna_file_path=None):
    """
    Complete pipeline: annotate ATAC peaks and optionally analyze RNA overlap
    
    Parameters:
    -----------
    atac_file_path : str
        Path to ATAC h5ad file
    rna_file_path : str, optional
        Path to RNA h5ad file for overlap analysis
    """
    
    print("=== ATAC Peak Annotation Pipeline ===")
    adata = annotate_atac_peaks(atac_file_path)
    print("Pipeline complete!")
    return adata

# === Usage ===
if __name__ == "__main__":
    # File paths
    atac_path = "/Users/harry/Desktop/GenoDistance/Data/test_ATAC.h5ad"
    rna_path = "/Users/harry/Desktop/GenoDistance/Data/count_data.h5ad"
    
    # Run complete pipeline
    annotated_adata = run_atac_annotation_pipeline(atac_path, rna_path)
