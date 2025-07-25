
import os
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd


def check_atac_annotations(*, output_prefix="atac_annotation", output_dir="."):
    """
    Simple quality check for ATAC peak annotations.
    
    Parameters
    ----------
    output_prefix : str
        File prefix used during annotation saving.
    output_dir : str
        Directory where annotation files reside.
        
    Returns
    -------
    dict
        Dictionary containing key quality metrics.
    """
    
    output_dir = Path(output_dir).resolve()
    
    # Check required files
    required_files = {
        'peak2gene': f"{output_prefix}_peak2gene.pkl",
        'annotation_df': f"{output_prefix}_full_annotations.parquet", 
        'stats': f"{output_prefix}_stats.json",
        'parameters': f"{output_prefix}_parameters.json"
    }
    
    missing_files = []
    for file_type, filename in required_files.items():
        if not (output_dir / filename).exists():
            missing_files.append(filename)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return None
    
    # Load data
    try:
        with open(output_dir / required_files['peak2gene'], 'rb') as f:
            peak2gene = pickle.load(f)
        
        annotation_df = pd.read_parquet(output_dir / required_files['annotation_df'])
        
        with open(output_dir / required_files['stats'], 'r') as f:
            saved_stats = json.load(f)
        
        with open(output_dir / required_files['parameters'], 'r') as f:
            parameters = json.load(f)
            
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return None
    
    # Key metrics
    total_peaks = len(peak2gene)
    total_genes = annotation_df['gene_id'].nunique()
    total_associations = len(annotation_df)
    
    # Weight normalization check
    weight_sums = annotation_df.groupby('peak')['combined_weight'].sum()
    weights_normalized = np.allclose(weight_sums, 1.0, rtol=1e-6)
    
    # Distance quality
    within_10kb = (annotation_df['distance_to_tss'] <= 10000).mean() * 100
    within_50kb = (annotation_df['distance_to_tss'] <= 50000).mean() * 100
    median_distance = annotation_df['distance_to_tss'].median()
    
    # Functional distribution
    promoter_pct = annotation_df['in_promoter'].mean() * 100
    gene_body_pct = annotation_df['in_gene_body'].mean() * 100
    
    # High confidence associations
    high_conf_pct = (annotation_df['combined_weight'] >= 0.5).mean() * 100
    
    # Coverage
    expected_peaks = saved_stats.get('total_peaks', total_peaks)
    coverage_pct = (total_peaks / expected_peaks) * 100
    
    # Check for issues
    issues = []
    if not weights_normalized:
        issues.append("Weights not properly normalized")
    if within_10kb < 30:
        issues.append("Low percentage of close associations")
    if promoter_pct < 10:
        issues.append("Very few promoter associations")
    if high_conf_pct < 20:
        issues.append("Low confidence associations")
    if coverage_pct < 80:
        issues.append("Low peak coverage")
    
    # Overall quality score
    quality_score = (
        min(100, coverage_pct) * 0.25 +           # Coverage
        within_50kb * 0.3 +                       # Distance quality  
        min(100, promoter_pct + gene_body_pct) * 0.25 +  # Functional annotation
        high_conf_pct * 0.2                      # Confidence
    )
    
    if weights_normalized:
        quality_score = min(100, quality_score + 10)  # Bonus for proper normalization
    
    quality_level = ('Excellent' if quality_score >= 85 else
                    'Good' if quality_score >= 70 else  
                    'Fair' if quality_score >= 55 else 'Poor')
    
    # Print summary
    print("🔍 ATAC Annotation Quality Check")
    print("=" * 40)
    
    # Print annotation parameters
    print("⚙️  Annotation Parameters:")
    print(f"  • Ensembl release: {parameters.get('ensembl_release', 'Unknown')}")
    print(f"  • Upstream extension: {parameters.get('extend_upstream', 'Unknown'):,} bp")
    print(f"  • Downstream extension: {parameters.get('extend_downstream', 'Unknown'):,} bp")
    print(f"  • Gene bounds mode: {parameters.get('use_gene_bounds', 'Unknown')}")
    print(f"  • Promoter upstream: {parameters.get('promoter_upstream', 'Unknown'):,} bp")
    print(f"  • Promoter downstream: {parameters.get('promoter_downstream', 'Unknown'):,} bp")
    print(f"  • Distance weight sigma: {parameters.get('distance_weight_sigma', 'Unknown'):,} bp")
    print(f"  • Promoter weight factor: {parameters.get('promoter_weight_factor', 'Unknown')}")
    print(f"  • Min peak accessibility: {parameters.get('min_peak_accessibility', 'Unknown')}")
    print()
    
    print(f"Overall Score: {quality_score:.1f}/100 ({quality_level})")
    print()
    print("📊 Key Metrics:")
    print(f"  • Annotated peaks: {total_peaks:,}")
    print(f"  • Unique genes: {total_genes:,}")
    print(f"  • Total associations: {total_associations:,}")
    print(f"  • Coverage: {coverage_pct:.1f}%")
    print()
    print("📏 Distance Quality:")
    print(f"  • Within 10kb of TSS: {within_10kb:.1f}%")
    print(f"  • Within 50kb of TSS: {within_50kb:.1f}%")
    print(f"  • Median distance: {median_distance:,.0f} bp")
    print()
    print("🎯 Association Quality:")
    print(f"  • Promoter associations: {promoter_pct:.1f}%")
    print(f"  • Gene body associations: {gene_body_pct:.1f}%")
    print(f"  • High confidence (≥0.5): {high_conf_pct:.1f}%")
    print(f"  • Weights normalized: {'✅' if weights_normalized else '❌'}")
    
    if issues:
        print()
        print("⚠️  Issues Found:")
        for issue in issues:
            print(f"  • {issue}")
    else:
        print()
        print("✅ No major issues detected")
    
    return {
        'quality_score': quality_score,
        'quality_level': quality_level,
        'total_peaks': total_peaks,
        'total_genes': total_genes,
        'coverage_pct': coverage_pct,
        'within_10kb_pct': within_10kb,
        'promoter_pct': promoter_pct,
        'high_confidence_pct': high_conf_pct,
        'weights_normalized': weights_normalized,
        'issues': issues,
        'parameters': parameters
    }


# Quick one-liner check
def quick_check(*, output_prefix="atac_annotation", output_dir="."):
    """One-line summary of annotation quality."""
    result = check_atac_annotations(output_prefix=output_prefix, output_dir=output_dir)
    if result:
        score = result['quality_score']
        level = result['quality_level'] 
        peaks = result['total_peaks']
        genes = result['total_genes']
        within_10kb = result['within_10kb_pct']
        print(f"Quality: {score:.1f}/100 ({level}) | {peaks:,} peaks → {genes:,} genes | {within_10kb:.1f}% within 10kb")
    return result

if __name__ == "__main__":
    results = check_atac_annotations(
    output_prefix="atac_annotation",
    output_dir="/dcl01/hongkai/data/data/hjiang/result/peak_annotation"
    )