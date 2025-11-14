#!/usr/bin/env python3
"""
RNA AnnData Gene Name to Gene ID Converter

This function reads an RNA AnnData object and converts gene names to gene IDs
if necessary, ensuring consistent gene_id indexing across datasets.

Features:
- Automatically detects if genes are labeled by name or ID
- Converts gene names to gene IDs using Ensembl
- Handles duplicates and missing mappings gracefully
- Preserves original gene names for reference
- Comprehensive logging and statistics
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import pyensembl
from collections import Counter, defaultdict
from pathlib import Path
import json

warnings.filterwarnings("ignore")


def detect_gene_identifier_type(gene_list, sample_size=1000):
    """
    Detect whether genes are labeled by gene names or gene IDs.
    
    Parameters:
    -----------
    gene_list : list or pandas.Index
        List of gene identifiers
    sample_size : int
        Number of genes to sample for detection
        
    Returns:
    --------
    str : 'gene_id', 'gene_name', or 'mixed'
    """
    
    # Sample genes for analysis
    genes_to_check = list(gene_list)[:sample_size] if len(gene_list) > sample_size else list(gene_list)
    
    ensembl_pattern = re.compile(r'^ENS[A-Z]*G\d{11}(\.\d+)?$')  # ENSG pattern with optional version
    refseq_pattern = re.compile(r'^(NM_|NR_|XM_|XR_)\d+(\.\d+)?$')  # RefSeq patterns
    
    gene_id_count = 0
    gene_name_count = 0
    
    for gene in genes_to_check:
        gene_str = str(gene).strip()
        
        # Check for Ensembl gene ID pattern
        if ensembl_pattern.match(gene_str):
            gene_id_count += 1
        # Check for RefSeq patterns
        elif refseq_pattern.match(gene_str):
            gene_id_count += 1
        # Check for typical gene name patterns (all caps, mixed case with numbers)
        elif (gene_str.isupper() and len(gene_str) <= 20 and gene_str.isalnum()) or \
             (re.match(r'^[A-Za-z][A-Za-z0-9-]*[0-9]*$', gene_str) and len(gene_str) <= 20):
            gene_name_count += 1
    
    total_checked = len(genes_to_check)
    gene_id_ratio = gene_id_count / total_checked
    gene_name_ratio = gene_name_count / total_checked
    
    # Decision thresholds
    if gene_id_ratio > 0.8:
        return 'gene_id'
    elif gene_name_ratio > 0.6:
        return 'gene_name'
    else:
        return 'mixed'


def create_gene_mapping(ensembl_release, species="homo_sapiens", verbose=True):
    """
    Create comprehensive gene name to gene ID mapping from Ensembl.
    
    Parameters:
    -----------
    ensembl_release : int
        Ensembl release version
    species : str
        Species name (default: "homo_sapiens")
    verbose : bool
        Print progress messages
        
    Returns:
    --------
    dict : Mapping from gene names to gene IDs
    dict : Mapping from gene IDs to gene names (reverse mapping)
    dict : Statistics about the mapping
    """
    
    if verbose:
        print(f"Creating gene mappings from Ensembl release {ensembl_release}...")
    
    # Initialize Ensembl
    ensembl = pyensembl.EnsemblRelease(release=ensembl_release, species=species)
    
    try:
        genes = ensembl.genes()
    except Exception:
        if verbose:
            print("Downloading Ensembl data (first time only)...")
        ensembl.download()
        ensembl.index()
        genes = ensembl.genes()
    
    # Build mappings
    name_to_id = {}
    id_to_name = {}
    duplicate_names = defaultdict(list)
    
    for gene in genes:
        gene_id = gene.gene_id
        gene_name = gene.gene_name
        
        if gene_name and gene_id:
            # Track duplicates
            if gene_name in name_to_id:
                duplicate_names[gene_name].append(gene_id)
            else:
                name_to_id[gene_name] = gene_id
                duplicate_names[gene_name] = [gene_id]
            
            id_to_name[gene_id] = gene_name
    
    # Handle duplicates by keeping the first occurrence
    for name, ids in duplicate_names.items():
        if len(ids) > 1:
            name_to_id[name] = ids[0]  # Keep first occurrence
    
    stats = {
        'total_genes': len(genes),
        'mapped_names': len(name_to_id),
        'mapped_ids': len(id_to_name),
        'duplicate_names': len([name for name, ids in duplicate_names.items() if len(ids) > 1]),
        'ensembl_release': ensembl_release,
        'species': species
    }
    
    if verbose:
        print(f"Created mappings for {stats['mapped_names']:,} gene names → {stats['mapped_ids']:,} gene IDs")
        if stats['duplicate_names'] > 0:
            print(f"Warning: {stats['duplicate_names']:,} gene names map to multiple IDs (kept first occurrence)")
    
    return name_to_id, id_to_name, stats


def convert_rna_to_gene_ids(
    adata_path,
    ensembl_release,
    output_path=None,
    species="homo_sapiens",
    force_conversion=False,
    handle_duplicates='first',
    min_mapping_rate=0.7,
    save_mapping_stats=True,
    verbose=True
):
    """
    Read RNA AnnData and convert gene names to gene IDs if necessary.
    
    Parameters:
    -----------
    adata_path : str or Path
        Path to the RNA AnnData file
    ensembl_release : int
        Ensembl release version to use for mapping
    output_path : str or Path or None
        Output path for converted AnnData (if None, saves to same location with suffix)
    species : str
        Species name for Ensembl (default: "homo_sapiens")
    force_conversion : bool
        Force conversion even if genes appear to be IDs already
    handle_duplicates : str
        How to handle duplicate gene names: 'first', 'drop', 'suffix'
    min_mapping_rate : float
        Minimum fraction of genes that must map successfully (0.0-1.0)
    save_mapping_stats : bool
        Save detailed mapping statistics to JSON
    verbose : bool
        Print detailed progress information
        
    Returns:
    --------
    AnnData
        RNA data with gene IDs as var_names and gene names preserved in var['gene_name']
    """
    
    # Load data
    adata_path = Path(adata_path)
    if verbose:
        print(f"Loading RNA data from: {adata_path}")
    
    adata = ad.read_h5ad(adata_path)
    original_shape = adata.shape
    
    if verbose:
        print(f"Original data shape: {original_shape[0]:,} cells × {original_shape[1]:,} genes")
    
    # Detect gene identifier type
    gene_type = detect_gene_identifier_type(adata.var_names)
    
    if verbose:
        print(f"Detected gene identifier type: {gene_type}")
        print(f"Sample genes: {list(adata.var_names[:5])}")
    
    # Decide if conversion is needed
    if gene_type == 'gene_id' and not force_conversion:
        if verbose:
            print("Genes appear to be already in gene ID format. No conversion needed.")
        
        # Ensure gene_name column exists for consistency
        if 'gene_name' not in adata.var.columns:
            # Try to get gene names from Ensembl if possible
            try:
                _, id_to_name, _ = create_gene_mapping(ensembl_release, species, verbose=False)
                gene_names = [id_to_name.get(gene_id, gene_id) for gene_id in adata.var_names]
                adata.var['gene_name'] = gene_names
                if verbose:
                    print("Added gene names from Ensembl mapping.")
            except Exception:
                adata.var['gene_name'] = adata.var_names.copy()
                if verbose:
                    print("Could not map gene IDs to names, using IDs as names.")
        
        return adata
    
    # Proceed with conversion
    if verbose:
        print(f"Converting gene names to gene IDs using Ensembl release {ensembl_release}...")
    
    # Create gene mappings
    name_to_id, id_to_name, mapping_stats = create_gene_mapping(
        ensembl_release, species, verbose
    )
    
    # Prepare conversion
    original_genes = list(adata.var_names)
    conversion_results = {
        'mapped': [],
        'unmapped': [],
        'duplicates': [],
        'original_names': []
    }
    
    new_gene_ids = []
    new_gene_names = []
    keep_indices = []
    
    gene_count = Counter(original_genes)
    
    for i, gene in enumerate(original_genes):
        gene_str = str(gene).strip()
        conversion_results['original_names'].append(gene_str)
        
        # Handle duplicates in original data
        if gene_count[gene] > 1:
            conversion_results['duplicates'].append(gene_str)
            
            if handle_duplicates == 'drop':
                continue
            elif handle_duplicates == 'suffix':
                # Add suffix to make unique
                occurrence = conversion_results['original_names'][:i].count(gene_str)
                gene_str = f"{gene_str}_{occurrence + 1}"
        
        # Try to map gene name to gene ID
        if gene_str in name_to_id:
            gene_id = name_to_id[gene_str]
            new_gene_ids.append(gene_id)
            new_gene_names.append(gene_str)
            keep_indices.append(i)
            conversion_results['mapped'].append(gene_str)
        else:
            conversion_results['unmapped'].append(gene_str)
            
            if handle_duplicates != 'drop':
                # Keep unmapped genes as-is or try to detect if they're already IDs
                if re.match(r'^ENS[A-Z]*G\d{11}', gene_str):
                    # Looks like a gene ID already
                    new_gene_ids.append(gene_str)
                    new_gene_names.append(id_to_name.get(gene_str, gene_str))
                    keep_indices.append(i)
                else:
                    # Skip this gene
                    continue
    
    # Calculate mapping rate
    mapping_rate = len(conversion_results['mapped']) / len(original_genes)
    
    if verbose:
        print(f"\nConversion results:")
        print(f"  Successfully mapped: {len(conversion_results['mapped']):,} ({mapping_rate:.2%})")
        print(f"  Unmapped genes: {len(conversion_results['unmapped']):,}")
        print(f"  Duplicate gene names: {len(set(conversion_results['duplicates'])):,}")
        print(f"  Genes retained: {len(keep_indices):,}")
    
    # Check if mapping rate is acceptable
    if mapping_rate < min_mapping_rate:
        raise ValueError(
            f"Mapping rate ({mapping_rate:.2%}) is below minimum threshold ({min_mapping_rate:.2%}). "
            f"Consider using a different Ensembl release or lowering the threshold."
        )
    
    # Filter AnnData to keep only successfully mapped genes
    if len(keep_indices) < len(original_genes):
        adata_filtered = adata[:, keep_indices].copy()
        if verbose:
            print(f"Filtered data shape: {adata_filtered.shape[0]:,} cells × {adata_filtered.shape[1]:,} genes")
    else:
        adata_filtered = adata.copy()
    
    # Update gene identifiers
    adata_filtered.var_names = new_gene_ids
    adata_filtered.var_names.name = 'gene_id'
    adata_filtered.var['gene_name'] = new_gene_names
    
    # Add conversion metadata
    adata_filtered.uns['gene_conversion'] = {
        'method': 'ensembl_name_to_id',
        'ensembl_release': ensembl_release,
        'species': species,
        'original_gene_count': len(original_genes),
        'mapped_gene_count': len(conversion_results['mapped']),
        'mapping_rate': mapping_rate,
        'duplicate_handling': handle_duplicates,
        'conversion_date': pd.Timestamp.now().isoformat(),
        'unmapped_genes': conversion_results['unmapped'][:50]  # Store first 50 for reference
    }
    
    # Save mapping statistics if requested
    if save_mapping_stats:
        stats_file = adata_path.parent / f"{adata_path.stem}_gene_conversion_stats.json"
        detailed_stats = {
            'mapping_stats': mapping_stats,
            'conversion_results': {
                'mapped_count': len(conversion_results['mapped']),
                'unmapped_count': len(conversion_results['unmapped']),
                'duplicate_count': len(set(conversion_results['duplicates'])),
                'mapping_rate': mapping_rate,
                'unmapped_genes': conversion_results['unmapped']
            },
            'parameters': {
                'ensembl_release': ensembl_release,
                'species': species,
                'handle_duplicates': handle_duplicates,
                'min_mapping_rate': min_mapping_rate
            }
        }
        
        with open(stats_file, 'w') as f:
            json.dump(detailed_stats, f, indent=2)
        
        if verbose:
            print(f"Detailed mapping statistics saved to: {stats_file}")
    
    # Save converted data
    if output_path is None:
        output_path = adata_path.parent / f"{adata_path.stem}.h5ad"
    else:
        output_path = Path(output_path)
    
    adata_filtered.write(output_path)
    
    if verbose:
        print(f"Converted RNA data saved to: {output_path}")
        print(f"\nFinal data structure:")
        print(f"  Shape: {adata_filtered.shape[0]:,} cells × {adata_filtered.shape[1]:,} genes")
        print(f"  Gene identifiers: gene_ids (var_names)")
        print(f"  Gene names: available in var['gene_name']")
        
        # Show example mapping
        print(f"\nExample gene ID → name mapping:")
        for i in range(min(5, len(adata_filtered.var))):
            gene_id = adata_filtered.var_names[i]
            gene_name = adata_filtered.var.iloc[i]['gene_name']
            print(f"  {gene_id} → {gene_name}")
    
    return adata_filtered


def quick_gene_overview(adata, title="Gene Data Overview"):
    """
    Quick overview of gene identifiers in AnnData object.
    """
    print(f"{title}")
    print("=" * len(title))
    print(f"Shape: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
    
    # Detect identifier type
    gene_type = detect_gene_identifier_type(adata.var_names)
    print(f"Gene identifier type: {gene_type}")
    print(f"var_names name: {adata.var_names.name or 'None'}")
    
    # Show sample genes
    print(f"Sample gene identifiers:")
    for i in range(min(5, len(adata.var_names))):
        gene_id = adata.var_names[i]
        if 'gene_name' in adata.var.columns:
            gene_name = adata.var.iloc[i]['gene_name']
            print(f"  {gene_id} → {gene_name}")
        else:
            print(f"  {gene_id}")
    
    # Conversion info if available
    if 'gene_conversion' in adata.uns:
        conv_info = adata.uns['gene_conversion']
        print(f"\nConversion info:")
        print(f"  Method: {conv_info.get('method', 'unknown')}")
        print(f"  Ensembl release: {conv_info.get('ensembl_release', 'unknown')}")
        print(f"  Mapping rate: {conv_info.get('mapping_rate', 0):.2%}")


# Example usage
if __name__ == "__main__":
    # Example: Convert RNA data to use gene IDs
    rna_path ="/dcl01/hongkai/data/data/hjiang/Data/paired/rna/heart.h5ad"
    
    # Convert with Ensembl release 98
    adata_converted = convert_rna_to_gene_ids(
        adata_path=rna_path,
        ensembl_release=98,
        species="homo_sapiens",
        handle_duplicates='first',
        min_mapping_rate=0.7,
        output_path = "/dcs07/hongkai/data/harry/result/gene_activity/signac_outputs/heart/rna_corrected.h5ad",
        verbose=True
    )
    
    # Show overview
    quick_gene_overview(adata_converted, "Converted RNA Data")