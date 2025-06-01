import pandas as pd
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def explore_atac_annotation_data(atac_adata):
    """
    Explore the structure and content of ATAC peak annotation data
    
    Parameters:
    -----------
    atac_adata : AnnData
        Annotated ATAC-seq data with peak2gene annotations
    
    Returns:
    --------
    dict : Summary statistics and example data
    """
    
    print("=== EXPLORING ATAC ANNOTATION DATA ===")
    
    # Check if annotation exists
    if "atac" not in atac_adata.uns or "peak2gene" not in atac_adata.uns["atac"]:
        print("❌ No peak2gene annotation found in adata.uns['atac']['peak2gene']")
        return None
    
    peak_annotation = atac_adata.uns["atac"]["peak2gene"]
    
    # Basic statistics
    total_peaks = len(atac_adata.var_names)
    annotated_peaks = len(peak_annotation)
    coverage = (annotated_peaks / total_peaks) * 100
    
    print(f"📊 BASIC STATISTICS:")
    print(f"  Total peaks: {total_peaks:,}")
    print(f"  Annotated peaks: {annotated_peaks:,}")
    print(f"  Coverage: {coverage:.1f}%")
    
    # Extract all genes from annotations
    all_genes = []
    all_distances = []
    genes_per_peak = []
    
    for peak, annotation in peak_annotation.items():
        genes = annotation['genes']
        distances = annotation['distances']
        
        all_genes.extend(genes)
        all_distances.extend(distances)
        genes_per_peak.append(len(genes))
    
    unique_genes = set(all_genes)
    
    print(f"\n🧬 GENE STATISTICS:")
    print(f"  Total peak-gene links: {len(all_genes):,}")
    print(f"  Unique genes linked: {len(unique_genes):,}")
    print(f"  Average genes per peak: {np.mean(genes_per_peak):.1f}")
    print(f"  Median genes per peak: {np.median(genes_per_peak):.0f}")
    
    # Distance statistics
    print(f"\n📏 DISTANCE STATISTICS:")
    print(f"  Mean distance to TSS: {np.mean(all_distances):,.0f} bp")
    print(f"  Median distance to TSS: {np.median(all_distances):,.0f} bp")
    print(f"  Max distance to TSS: {np.max(all_distances):,.0f} bp")
    
    # Distance distribution
    distance_bins = [0, 5000, 10000, 25000, 50000, 100000]
    print(f"\n  Distance distribution:")
    for i in range(len(distance_bins)-1):
        count = sum(1 for d in all_distances if distance_bins[i] <= d < distance_bins[i+1])
        pct = (count / len(all_distances)) * 100
        print(f"    {distance_bins[i]/1000:.0f}-{distance_bins[i+1]/1000:.0f}kb: {count:,} links ({pct:.1f}%)")
    
    # Most common genes
    gene_counts = Counter(all_genes)
    most_common_genes = gene_counts.most_common(10)
    
    print(f"\n🔝 TOP 10 MOST LINKED GENES:")
    for gene, count in most_common_genes:
        print(f"  {gene}: {count} peaks")
    
    # Example annotations
    print(f"\n📋 EXAMPLE PEAK ANNOTATIONS:")
    example_peaks = list(peak_annotation.keys())[:5]
    
    for i, peak in enumerate(example_peaks, 1):
        annotation = peak_annotation[peak]
        print(f"\n  {i}. Peak: {peak}")
        print(f"     Closest gene: {annotation['closest_gene']} ({annotation['closest_distance']:,} bp)")
        
        # Show top 3 genes if multiple
        if len(annotation['genes']) > 1:
            top_genes = []
            for j in range(min(3, len(annotation['genes']))):
                gene = annotation['genes'][j]
                dist = annotation['distances'][j]
                top_genes.append(f"{gene} ({dist:,} bp)")
            print(f"     Top genes: {', '.join(top_genes)}")
            if len(annotation['genes']) > 3:
                print(f"     ... and {len(annotation['genes']) - 3} more genes")
    
    # Return summary for further analysis
    summary = {
        'total_peaks': total_peaks,
        'annotated_peaks': annotated_peaks,
        'coverage_percent': coverage,
        'unique_genes': list(unique_genes),
        'gene_counts': dict(gene_counts),
        'distance_stats': {
            'mean': np.mean(all_distances),
            'median': np.median(all_distances),
            'std': np.std(all_distances)
        }
    }
    
    return summary

def analyze_atac_rna_overlap(atac_adata, rna_adata):
    """
    Analyze overlap between ATAC-linked genes and RNA-seq genes
    
    Parameters:
    -----------
    atac_adata : AnnData
        Annotated ATAC-seq data with peak2gene annotations
    rna_adata : AnnData
        RNA-seq data with gene names in var_names
    
    Returns:
    --------
    dict : Overlap analysis results
    """
    
    print("\n" + "="*50)
    print("=== ATAC-RNA GENE OVERLAP ANALYSIS ===")
    
    # Extract ATAC-linked genes
    if "atac" not in atac_adata.uns or "peak2gene" not in atac_adata.uns["atac"]:
        print("❌ No ATAC peak annotation found!")
        return None
    
    peak_annotation = atac_adata.uns["atac"]["peak2gene"]
    
    # Get all genes linked to ATAC peaks
    atac_genes = set()
    for peak, annotation in peak_annotation.items():
        atac_genes.update(annotation['genes'])
    
    # Get RNA-seq genes
    rna_genes = set(rna_adata.var_names)
    
    # Remove any version numbers from gene names (e.g., ENSG00000000003.15 -> ENSG00000000003)
    def clean_gene_name(gene_name):
        return gene_name.split('.')[0] if '.' in gene_name else gene_name
    
    atac_genes_clean = {clean_gene_name(g) for g in atac_genes}
    rna_genes_clean = {clean_gene_name(g) for g in rna_genes}
    
    # Find overlaps
    overlap_genes = atac_genes_clean.intersection(rna_genes_clean)
    atac_only = atac_genes_clean - rna_genes_clean
    rna_only = rna_genes_clean - atac_genes_clean
    
    # Calculate statistics
    atac_count = len(atac_genes_clean)
    rna_count = len(rna_genes_clean)
    overlap_count = len(overlap_genes)
    
    atac_overlap_pct = (overlap_count / atac_count) * 100 if atac_count > 0 else 0
    rna_overlap_pct = (overlap_count / rna_count) * 100 if rna_count > 0 else 0
    
    print(f"📊 GENE COUNT SUMMARY:")
    print(f"  ATAC-linked genes: {atac_count:,}")
    print(f"  RNA-seq genes: {rna_count:,}")
    print(f"  Overlapping genes: {overlap_count:,}")
    print(f"  ATAC-only genes: {len(atac_only):,}")
    print(f"  RNA-only genes: {len(rna_only):,}")
    
    print(f"\n🎯 OVERLAP PERCENTAGES:")
    print(f"  % of ATAC genes found in RNA: {atac_overlap_pct:.1f}%")
    print(f"  % of RNA genes found in ATAC: {rna_overlap_pct:.1f}%")
    
    # Analyze overlap by distance to TSS
    print(f"\n📏 OVERLAP BY DISTANCE TO TSS:")
    
    # Get distance info for overlapping genes
    overlap_distances = []
    for peak, annotation in peak_annotation.items():
        for i, gene in enumerate(annotation['genes']):
            clean_gene = clean_gene_name(gene)
            if clean_gene in overlap_genes:
                overlap_distances.append(annotation['distances'][i])
    
    if overlap_distances:
        distance_bins = [0, 5000, 10000, 25000, 50000, 100000]
        for i in range(len(distance_bins)-1):
            count = sum(1 for d in overlap_distances if distance_bins[i] <= d < distance_bins[i+1])
            pct = (count / len(overlap_distances)) * 100 if overlap_distances else 0
            print(f"  {distance_bins[i]/1000:.0f}-{distance_bins[i+1]/1000:.0f}kb: {count:,} links ({pct:.1f}%)")
    
    # Show examples of overlapping and non-overlapping genes
    print(f"\n🔍 EXAMPLE OVERLAPPING GENES:")
    overlap_list = list(overlap_genes)[:10]
    for gene in overlap_list:
        print(f"  {gene}")
    if len(overlap_genes) > 10:
        print(f"  ... and {len(overlap_genes) - 10:,} more")
    
    print(f"\n❌ EXAMPLE ATAC-ONLY GENES (no RNA detection):")
    atac_only_list = list(atac_only)[:10]
    for gene in atac_only_list:
        print(f"  {gene}")
    if len(atac_only) > 10:
        print(f"  ... and {len(atac_only) - 10:,} more")
    
    # Quality assessment
    print(f"\n✅ QUALITY ASSESSMENT:")
    if atac_overlap_pct > 70:
        print(f"  🟢 Excellent overlap ({atac_overlap_pct:.1f}%) - Most ATAC peaks link to expressed genes")
    elif atac_overlap_pct > 50:
        print(f"  🟡 Good overlap ({atac_overlap_pct:.1f}%) - Majority of ATAC peaks link to expressed genes")
    elif atac_overlap_pct > 30:
        print(f"  🟠 Moderate overlap ({atac_overlap_pct:.1f}%) - Some ATAC peaks may be regulatory elements")
    else:
        print(f"  🔴 Low overlap ({atac_overlap_pct:.1f}%) - Check gene naming or annotation quality")
    
    # Return results for further analysis
    results = {
        'atac_genes': atac_genes_clean,
        'rna_genes': rna_genes_clean,
        'overlap_genes': overlap_genes,
        'atac_only': atac_only,
        'rna_only': rna_only,
        'overlap_stats': {
            'atac_count': atac_count,
            'rna_count': rna_count,
            'overlap_count': overlap_count,
            'atac_overlap_pct': atac_overlap_pct,
            'rna_overlap_pct': rna_overlap_pct
        },
        'overlap_distances': overlap_distances
    }
    
    return results

def create_overlap_visualization(overlap_results):
    """
    Create visualizations for ATAC-RNA overlap analysis
    """
    if overlap_results is None:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Venn diagram-style bar plot
    stats = overlap_results['overlap_stats']
    categories = ['ATAC-linked\nGenes', 'RNA-seq\nGenes', 'Overlapping\nGenes']
    counts = [stats['atac_count'], stats['rna_count'], stats['overlap_count']]
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    
    axes[0].bar(categories, counts, color=colors)
    axes[0].set_ylabel('Number of Genes')
    axes[0].set_title('Gene Count Comparison')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add count labels on bars
    for i, count in enumerate(counts):
        axes[0].text(i, count + max(counts)*0.01, f'{count:,}', 
                    ha='center', va='bottom', fontweight='bold')
    
    # 2. Distance distribution for overlapping genes
    if overlap_results['overlap_distances']:
        distances = np.array(overlap_results['overlap_distances']) / 1000  # Convert to kb
        axes[1].hist(distances, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[1].set_xlabel('Distance to TSS (kb)')
        axes[1].set_ylabel('Number of Gene-Peak Links')
        axes[1].set_title('Distance Distribution of Overlapping Genes')
        axes[1].axvline(np.median(distances), color='red', linestyle='--', 
                       label=f'Median: {np.median(distances):.1f} kb')
        axes[1].legend()
    
    plt.tight_layout()
    plt.show()

# === Usage Example ===
if __name__ == "__main__":
    print("Loading ATAC and RNA data...")
    
    # Load annotated ATAC data
    atac_adata = ad.read_h5ad("/Users/harry/Desktop/GenoDistance/Data/test_ATAC_annotated.h5ad")
    
    # Load RNA data
    rna_adata = ad.read_h5ad("/Users/harry/Desktop/GenoDistance/Data/count_data.h5ad")
    
    # Explore ATAC annotation structure
    atac_summary = explore_atac_annotation_data(atac_adata)
    
    # Analyze ATAC-RNA overlap
    overlap_results = analyze_atac_rna_overlap(atac_adata, rna_adata)
    
    # Create visualizations
    if overlap_results:
        create_overlap_visualization(overlap_results)
        
        # Save results
        print(f"\n💾 Saving overlap results...")
        
        # Add overlap info to ATAC data
        if "rna_overlap" not in atac_adata.uns:
            atac_adata.uns["rna_overlap"] = {}
        atac_adata.uns["rna_overlap"]["overlap_analysis"] = overlap_results['overlap_stats']
        atac_adata.uns["rna_overlap"]["overlapping_genes"] = list(overlap_results['overlap_genes'])
        
        # Save updated data
        atac_adata.write("/Users/harry/Desktop/GenoDistance/Data/test_ATAC_with_rna_overlap.h5ad")
        print(f"Saved results to: test_ATAC_with_rna_overlap.h5ad")