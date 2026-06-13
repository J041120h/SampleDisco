#!/usr/bin/env python3
"""Convert RNA AnnData var_names from gene symbols to Ensembl IDs via pyensembl."""

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
    """Return 'gene_id', 'gene_name', or 'mixed' for the dominant identifier style."""
    genes_to_check = list(gene_list)[:sample_size] if len(gene_list) > sample_size else list(gene_list)

    ensembl_pattern = re.compile(r'^ENS[A-Z]*G\d{11}(\.\d+)?$')
    refseq_pattern = re.compile(r'^(NM_|NR_|XM_|XR_)\d+(\.\d+)?$')
    
    gene_id_count = 0
    gene_name_count = 0
    
    for gene in genes_to_check:
        gene_str = str(gene).strip()
        if ensembl_pattern.match(gene_str):
            gene_id_count += 1
        elif refseq_pattern.match(gene_str):
            gene_id_count += 1
        elif (gene_str.isupper() and len(gene_str) <= 20 and gene_str.isalnum()) or \
             (re.match(r'^[A-Za-z][A-Za-z0-9-]*[0-9]*$', gene_str) and len(gene_str) <= 20):
            gene_name_count += 1

    total_checked = len(genes_to_check)
    gene_id_ratio = gene_id_count / total_checked
    gene_name_ratio = gene_name_count / total_checked

    if gene_id_ratio > 0.8:
        return 'gene_id'
    elif gene_name_ratio > 0.6:
        return 'gene_name'
    else:
        return 'mixed'


def create_gene_mapping(ensembl_release, species="homo_sapiens", verbose=True):
    """Return (name_to_id, id_to_name, stats) from a pyensembl release."""
    if verbose:
        print(f"Creating gene mappings from Ensembl release {ensembl_release}...")

    ensembl = pyensembl.EnsemblRelease(release=ensembl_release, species=species)
    
    try:
        genes = ensembl.genes()
    except Exception:
        if verbose:
            print("Downloading Ensembl data (first time only)...")
        ensembl.download()
        ensembl.index()
        genes = ensembl.genes()
    
    name_to_id = {}
    id_to_name = {}
    duplicate_names = defaultdict(list)

    for gene in genes:
        gene_id = gene.gene_id
        gene_name = gene.gene_name
        if gene_name and gene_id:
            if gene_name in name_to_id:
                duplicate_names[gene_name].append(gene_id)
            else:
                name_to_id[gene_name] = gene_id
                duplicate_names[gene_name] = [gene_id]
            id_to_name[gene_id] = gene_name

    # Keep first occurrence for symbols that map to multiple Ensembl IDs
    for name, ids in duplicate_names.items():
        if len(ids) > 1:
            name_to_id[name] = ids[0]
    
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
    Convert RNA AnnData var_names from gene symbols to Ensembl IDs.

    If var_names already look like IDs (and force_conversion=False), skips conversion
    but still backfills var['gene_name'] via Ensembl. Raises if mapping_rate < min_mapping_rate.
    Returns AnnData with gene IDs as var_names and original symbols in var['gene_name'].
    """
    adata_path = Path(adata_path)
    if verbose:
        print(f"Loading RNA data from: {adata_path}")

    adata = ad.read_h5ad(adata_path)
    original_shape = adata.shape

    if verbose:
        print(f"Original data shape: {original_shape[0]:,} cells × {original_shape[1]:,} genes")

    gene_type = detect_gene_identifier_type(adata.var_names)

    if verbose:
        print(f"Detected gene identifier type: {gene_type}")
        print(f"Sample genes: {list(adata.var_names[:5])}")

    if gene_type == 'gene_id' and not force_conversion:
        if verbose:
            print("Genes appear to be already in gene ID format. No conversion needed.")

        if 'gene_name' not in adata.var.columns:
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

    if verbose:
        print(f"Converting gene names to gene IDs using Ensembl release {ensembl_release}...")

    name_to_id, id_to_name, mapping_stats = create_gene_mapping(
        ensembl_release, species, verbose
    )

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

        if gene_count[gene] > 1:
            conversion_results['duplicates'].append(gene_str)
            if handle_duplicates == 'drop':
                continue
            elif handle_duplicates == 'suffix':
                occurrence = conversion_results['original_names'][:i].count(gene_str)
                gene_str = f"{gene_str}_{occurrence + 1}"

        if gene_str in name_to_id:
            gene_id = name_to_id[gene_str]
            new_gene_ids.append(gene_id)
            new_gene_names.append(gene_str)
            keep_indices.append(i)
            conversion_results['mapped'].append(gene_str)
        else:
            conversion_results['unmapped'].append(gene_str)
            if handle_duplicates != 'drop':
                if re.match(r'^ENS[A-Z]*G\d{11}', gene_str):
                    # Already an Ensembl ID; pass through
                    new_gene_ids.append(gene_str)
                    new_gene_names.append(id_to_name.get(gene_str, gene_str))
                    keep_indices.append(i)
                else:
                    continue

    mapping_rate = len(conversion_results['mapped']) / len(original_genes)
    
    if verbose:
        print(f"\nConversion results:")
        print(f"  Successfully mapped: {len(conversion_results['mapped']):,} ({mapping_rate:.2%})")
        print(f"  Unmapped genes: {len(conversion_results['unmapped']):,}")
        print(f"  Duplicate gene names: {len(set(conversion_results['duplicates'])):,}")
        print(f"  Genes retained: {len(keep_indices):,}")

    if mapping_rate < min_mapping_rate:
        raise ValueError(
            f"Mapping rate ({mapping_rate:.2%}) is below minimum threshold ({min_mapping_rate:.2%}). "
            f"Consider using a different Ensembl release or lowering the threshold."
        )
    
    if len(keep_indices) < len(original_genes):
        adata_filtered = adata[:, keep_indices].copy()
        if verbose:
            print(f"Filtered data shape: {adata_filtered.shape[0]:,} cells × {adata_filtered.shape[1]:,} genes")
    else:
        adata_filtered = adata.copy()

    adata_filtered.var_names = new_gene_ids
    adata_filtered.var_names.name = 'gene_id'
    adata_filtered.var['gene_name'] = new_gene_names

    adata_filtered.uns['gene_conversion'] = {
        'method': 'ensembl_name_to_id',
        'ensembl_release': ensembl_release,
        'species': species,
        'original_gene_count': len(original_genes),
        'mapped_gene_count': len(conversion_results['mapped']),
        'mapping_rate': mapping_rate,
        'duplicate_handling': handle_duplicates,
        'conversion_date': pd.Timestamp.now().isoformat(),
        'unmapped_genes': conversion_results['unmapped'][:50]
    }

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
        print(f"\nExample gene ID → name mapping:")
        for i in range(min(5, len(adata_filtered.var))):
            gene_id = adata_filtered.var_names[i]
            gene_name = adata_filtered.var.iloc[i]['gene_name']
            print(f"  {gene_id} → {gene_name}")

    return adata_filtered


def quick_gene_overview(adata, title="Gene Data Overview"):
    """Print a brief summary of gene identifier type and conversion provenance."""
    print(f"{title}")
    print("=" * len(title))
    print(f"Shape: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

    gene_type = detect_gene_identifier_type(adata.var_names)
    print(f"Gene identifier type: {gene_type}")
    print(f"var_names name: {adata.var_names.name or 'None'}")

    print(f"Sample gene identifiers:")
    for i in range(min(5, len(adata.var_names))):
        gene_id = adata.var_names[i]
        if 'gene_name' in adata.var.columns:
            gene_name = adata.var.iloc[i]['gene_name']
            print(f"  {gene_id} → {gene_name}")
        else:
            print(f"  {gene_id}")

    if 'gene_conversion' in adata.uns:
        conv_info = adata.uns['gene_conversion']
        print(f"\nConversion info:")
        print(f"  Method: {conv_info.get('method', 'unknown')}")
        print(f"  Ensembl release: {conv_info.get('ensembl_release', 'unknown')}")
        print(f"  Mapping rate: {conv_info.get('mapping_rate', 0):.2%}")


if __name__ == "__main__":
    rna_path = "/dcl01/hongkai/data/data/hjiang/Data/paired/rna/heart.h5ad"

    adata_converted = convert_rna_to_gene_ids(
        adata_path=rna_path,
        ensembl_release=98,
        species="homo_sapiens",
        handle_duplicates='first',
        min_mapping_rate=0.7,
        output_path="/dcs07/hongkai/data/harry/result/gene_activity/signac_outputs/heart/rna_corrected.h5ad",
        verbose=True
    )

    quick_gene_overview(adata_converted, "Converted RNA Data")