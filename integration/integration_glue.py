# Standard library
import os
import gc
import time
from pathlib import Path
from typing import Optional, Tuple, List
from itertools import chain
from concurrent.futures import ThreadPoolExecutor

# Third-party
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import networkx as nx
import scglue
import pyensembl
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
import psutil
import cupy as cp
from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
from cupyx.scipy import sparse as cusparse

# Local/project
from Cell_type import *
from linux.CellType_linux import cell_types_linux


def clean_anndata_for_saving(adata, verbose=True):
    """
    Clean AnnData object to ensure it can be saved to HDF5 format.
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object to clean
    verbose : bool
        Whether to print cleaning statistics
    
    Returns:
    --------
    adata : AnnData
        Cleaned AnnData object
    """
    if verbose:
        print("🧹 Cleaning AnnData object for HDF5 compatibility...")
    
    # Clean obs dataframe
    for col in adata.obs.columns:
        if verbose:
            print(f"   Processing column: {col}")
        
        # Convert object columns to string, handling NaN values
        if adata.obs[col].dtype == 'object':
            # Fill NaN values with 'Unknown' or appropriate default
            adata.obs[col] = adata.obs[col].fillna('Unknown')
            # Convert to string
            adata.obs[col] = adata.obs[col].astype(str)
            # Convert to category for memory efficiency
            adata.obs[col] = adata.obs[col].astype('category')
        
        # Handle numeric columns with NaN
        elif adata.obs[col].dtype in ['float64', 'float32']:
            # Fill NaN values with appropriate defaults
            if adata.obs[col].isna().any():
                adata.obs[col] = adata.obs[col].fillna(0.0)
        
        # Handle integer columns
        elif adata.obs[col].dtype in ['int64', 'int32']:
            # Ensure no NaN values in integer columns
            if adata.obs[col].isna().any():
                adata.obs[col] = adata.obs[col].fillna(0).astype('int64')
    
    # Clean var dataframe
    for col in adata.var.columns:
        if adata.var[col].dtype == 'object':
            # Fill NaN values and convert to string
            adata.var[col] = adata.var[col].fillna('Unknown').astype(str)
            # Convert to category for memory efficiency
            adata.var[col] = adata.var[col].astype('category')
        elif adata.var[col].dtype in ['float64', 'float32']:
            if adata.var[col].isna().any():
                adata.var[col] = adata.var[col].fillna(0.0)
        elif adata.var[col].dtype in ['int64', 'int32']:
            if adata.var[col].isna().any():
                adata.var[col] = adata.var[col].fillna(0).astype('int64')
    
    if verbose:
        print("✅ AnnData cleaning complete")
    
    return adata


def merge_sample_metadata(
    adata, 
    metadata_path, 
    sample_column="sample", 
    sep=",", 
    verbose=True
):
    """
    Merge sample-level metadata with AnnData object and standardize sample column to 'sample'.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    metadata_path : str
        Path to metadata CSV file
    sample_column : str
        Column name to use as index for merging
    sep : str
        Separator for CSV file
    verbose : bool
        Whether to print merge statistics
    
    Returns:
    --------
    adata : AnnData
        AnnData object with merged metadata and standardized 'sample' column
    """
    meta = pd.read_csv(metadata_path, sep=sep).set_index(sample_column)
    
    # Store original column count for comparison
    original_cols = adata.obs.shape[1]
    
    # Clean metadata before merging
    for col in meta.columns:
        if meta[col].dtype == 'object':
            meta[col] = meta[col].fillna('Unknown').astype(str)
    
    # Perform the merge
    adata.obs = adata.obs.join(meta, on=sample_column, how='left')
    
    # Standardize sample column name to 'sample'
    if sample_column != 'sample':
        if sample_column in adata.obs.columns:
            adata.obs['sample'] = adata.obs[sample_column]
            adata.obs = adata.obs.drop(columns=[sample_column])
            if verbose:
                print(f"   Standardized sample column '{sample_column}' to 'sample'")
        elif 'sample' not in adata.obs.columns:
            # If the original sample column doesn't exist, check if we can infer it
            if verbose:
                print(f"   Warning: Sample column '{sample_column}' not found")
    
    # Calculate merge statistics
    new_cols = adata.obs.shape[1] - original_cols
    matched_samples = adata.obs[meta.columns].notna().any(axis=1).sum()
    total_samples = adata.obs.shape[0]
    
    if verbose:
        print(f"   Merged {new_cols} sample-level columns")
        print(f"   Matched metadata for {matched_samples}/{total_samples} samples")
        if matched_samples < total_samples:
            print(f"   ⚠️ Warning: {total_samples - matched_samples} samples have no metadata")
    
    return adata


def glue_preprocess_pipeline(
    rna_file: str,
    atac_file: str,
    rna_sample_meta_file: Optional[str] = None,
    atac_sample_meta_file: Optional[str] = None,
    additional_hvg_file: Optional[str] = None,
    ensembl_release: int = 98,
    species: str = "homo_sapiens",
    output_dir: str = "./",
    use_highly_variable: bool = True,
    n_top_genes: int = 2000,
    n_pca_comps: int = 100,
    n_lsi_comps: int = 100,
    lsi_n_iter: int = 15,
    gtf_by: str = "gene_name",
    flavor: str = "seurat_v3",
    generate_umap: bool = False,
    compression: str = "gzip",
    random_state: int = 42,
    metadata_sep: str = ",",
    rna_sample_column: str = "sample",
    atac_sample_column: str = "sample"
) -> Tuple[ad.AnnData, ad.AnnData, nx.MultiDiGraph]:
    """
    Complete GLUE preprocessing pipeline for scRNA-seq and scATAC-seq data integration.
    
    Parameters:
    -----------
    rna_file : str
        Path to RNA h5ad file
    atac_file : str
        Path to ATAC h5ad file
    rna_sample_meta_file : str, optional
        Path to RNA sample metadata CSV
    atac_sample_meta_file : str, optional
        Path to ATAC sample metadata CSV
    additional_hvg_file : str, optional
        Path to txt file containing additional genes to be considered during multiomics integration.
        Each line should contain one gene name. These genes will be marked as HVG in addition
        to the ones selected by scanpy.
    ensembl_release : int
        Ensembl database release version
    species : str
        Species name for Ensembl
    output_dir : str
        Output directory path
    use_highly_variable : bool
        Whether to use highly variable features (True) or all features (False)
    n_top_genes : int
        Number of highly variable genes to select (only used if use_highly_variable=True)
    n_pca_comps : int
        Number of PCA components
    n_lsi_comps : int
        Number of LSI components for ATAC
    lsi_n_iter : int
        Number of LSI iterations
    gtf_by : str
        Gene annotation method
    flavor : str
        Method for highly variable gene selection
    generate_umap : bool
        Whether to generate UMAP embeddings
    compression : str
        Compression method for output files
    random_state : int
        Random seed for reproducibility
    metadata_sep : str
        Separator for metadata CSV files
    rna_sample_column : str
        Column name for RNA sample IDs in metadata
    atac_sample_column : str
        Column name for ATAC sample IDs in metadata
    
    Returns:
    --------
    Tuple[ad.AnnData, ad.AnnData, nx.MultiDiGraph]
        Preprocessed RNA data, ATAC data, and guidance graph
    """
    print("\n🚀 Starting GLUE preprocessing pipeline...\n")
    print(f"   Feature selection mode: {'Highly Variable' if use_highly_variable else 'All Features'}\n")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"📊 Loading data files...")
    print(f"   RNA: {rna_file}")
    print(f"   ATAC: {atac_file}")
    
    rna = ad.read_h5ad(rna_file)
    atac = ad.read_h5ad(atac_file)
    
    print(f"✅ Data loaded successfully")
    print(f"   RNA shape: {rna.shape}")
    print(f"   ATAC shape: {atac.shape}\n")
    
    # Load and integrate sample metadata using improved function
    if rna_sample_meta_file or atac_sample_meta_file:
        print("📋 Loading and merging sample metadata...")
        
        if rna_sample_meta_file:
            print(f"   Processing RNA metadata: {rna_sample_meta_file}")
            rna = merge_sample_metadata(
                rna, 
                rna_sample_meta_file, 
                sample_column=rna_sample_column,
                sep=metadata_sep,
                verbose=True
            )
        
        if atac_sample_meta_file:
            print(f"   Processing ATAC metadata: {atac_sample_meta_file}")
            atac = merge_sample_metadata(
                atac, 
                atac_sample_meta_file, 
                sample_column=atac_sample_column,
                sep=metadata_sep,
                verbose=True
            )
        
        print(f"\n✅ Metadata integration and standardization complete")
        print(f"   RNA obs columns: {list(rna.obs.columns)}")
        print(f"   ATAC obs columns: {list(atac.obs.columns)}\n")
    else:
        # Even if no metadata files are provided, ensure 'sample' column exists
        # This handles cases where the sample info might already be in the obs dataframe
        print("📋 Standardizing existing sample columns...")
        
        if rna_sample_column != 'sample' and rna_sample_column in rna.obs.columns:
            print(f"   Standardizing RNA sample column '{rna_sample_column}' to 'sample'")
            rna.obs['sample'] = rna.obs[rna_sample_column]
            rna.obs = rna.obs.drop(columns=[rna_sample_column])
        
        if atac_sample_column != 'sample' and atac_sample_column in atac.obs.columns:
            print(f"   Standardizing ATAC sample column '{atac_sample_column}' to 'sample'")
            atac.obs['sample'] = atac.obs[atac_sample_column]
            atac.obs = atac.obs.drop(columns=[atac_sample_column])
        
        print("✅ Sample column standardization complete\n")
    
    # Download and setup Ensembl annotation
    print(f"🧬 Setting up Ensembl annotation...")
    print(f"   Release: {ensembl_release}")
    print(f"   Species: {species}")
    
    ensembl = pyensembl.EnsemblRelease(release=ensembl_release, species=species)
    ensembl.download()
    ensembl.index()
    print("✅ Ensembl annotation ready\n")
    
    # Set random seed
    sc.settings.seed = random_state
    
    # Preprocess scRNA-seq data
    print(f"🧬 Preprocessing scRNA-seq data...")
    
    # Store counts
    rna.layers["counts"] = rna.X.copy()
    
    # Process based on feature selection strategy
    if use_highly_variable:
        print(f"   Selecting {n_top_genes} highly variable genes")
        sc.pp.highly_variable_genes(rna, n_top_genes=n_top_genes, flavor=flavor)
        
        # Load and process additional HVG genes if provided
        if additional_hvg_file:
            print(f"\n📝 Processing additional HVG genes from: {additional_hvg_file}")
            
            try:
                # Read the additional genes from file
                with open(additional_hvg_file, 'r') as f:
                    additional_genes = [line.strip() for line in f if line.strip()]
                
                print(f"   Found {len(additional_genes)} genes in additional HVG file")
                
                # Find which genes are present in the RNA data
                genes_in_data = [gene for gene in additional_genes if gene in rna.var_names]
                genes_not_found = [gene for gene in additional_genes if gene not in rna.var_names]
                
                # Simply mark all found genes as HVG (whether already HVG or not)
                for gene in genes_in_data:
                    gene_idx = rna.var_names.get_loc(gene)
                    rna.var.iloc[gene_idx, rna.var.columns.get_loc('highly_variable')] = True
                
                print(f"   Statistics for additional HVG genes:")
                print(f"     - Genes found and marked as HVG: {len(genes_in_data)}/{len(additional_genes)}")
                
                if genes_not_found:
                    print(f"     - Genes not found in RNA data: {len(genes_not_found)}")

            except Exception as e:
                print(f"   ⚠️ Error reading additional HVG file: {e}")
                print(f"   Continuing with scanpy-selected HVG only")
        
    else:
        print(f"   Using all {rna.n_vars} genes")
        # Mark all genes as highly variable for compatibility
        rna.var['highly_variable'] = True
        
        # Note: When use_highly_variable=False, additional_hvg_file is ignored
        if additional_hvg_file:
            print(f"   Note: additional_hvg_file is ignored when use_highly_variable=False")
    
    print(f"   Computing {n_pca_comps} PCA components")
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    sc.pp.scale(rna)
    sc.tl.pca(rna, n_comps=n_pca_comps, svd_solver="auto")
    
    if generate_umap:
        print("   Computing UMAP embedding...")
        sc.pp.neighbors(rna, metric="cosine")
        sc.tl.umap(rna)
    
    print("✅ RNA preprocessing complete\n")
    
    # Preprocess scATAC-seq data
    print(f"🏔️ Preprocessing scATAC-seq data...")
    
    # Process based on feature selection strategy
    if use_highly_variable:
        print(f"   Computing feature statistics for peak selection")
        # Calculate peak statistics for highly variable selection
        peak_counts = np.array(atac.X.sum(axis=0)).flatten()
        peak_cells = np.array((atac.X > 0).sum(axis=0)).flatten()
        
        # Select top peaks based on coverage
        n_top_peaks = min(50000, atac.n_vars)  # Default to top 50k peaks or all if less
        top_peak_indices = np.argsort(peak_counts)[-n_top_peaks:]
        
        # Mark highly variable peaks
        atac.var['highly_variable'] = False
        atac.var.iloc[top_peak_indices, atac.var.columns.get_loc('highly_variable')] = True
        atac.var['n_cells'] = peak_cells
        atac.var['n_counts'] = peak_counts
        
        print(f"   Selected {n_top_peaks} highly accessible peaks")
    else:
        print(f"   Using all {atac.n_vars} peaks")
        # Mark all peaks as highly variable for compatibility
        atac.var['highly_variable'] = True
        peak_counts = np.array(atac.X.sum(axis=0)).flatten()
        peak_cells = np.array((atac.X > 0).sum(axis=0)).flatten()
        atac.var['n_cells'] = peak_cells
        atac.var['n_counts'] = peak_counts
    
    print(f"   Computing {n_lsi_comps} LSI components ({lsi_n_iter} iterations)")
    scglue.data.lsi(atac, n_components=n_lsi_comps, n_iter=lsi_n_iter)
    
    if generate_umap:
        print("   Computing UMAP embedding...")
        sc.pp.neighbors(atac, use_rep="X_lsi", metric="cosine")
        sc.tl.umap(atac)
    
    print("✅ ATAC preprocessing complete\n")
    
    # Get gene coordinates from Ensembl
    def get_gene_coordinates(gene_names, ensembl_db):
        """Extract gene coordinates from pyensembl database with improved error handling"""
        coords = []
        failed_genes = []
        
        for gene_name in gene_names:
            try:
                genes = ensembl_db.genes_by_name(gene_name)
                if genes:
                    gene = genes[0]  # Take first match
                    # Convert strand to +/- format
                    strand = '+' if gene.strand == '+' else '-'
                    coords.append({
                        'chrom': f"chr{gene.contig}",
                        'chromStart': gene.start,
                        'chromEnd': gene.end,
                        'strand': strand
                    })
                else:
                    coords.append({'chrom': None, 'chromStart': None, 'chromEnd': None, 'strand': None})
                    failed_genes.append(gene_name)
            except Exception as e:
                coords.append({'chrom': None, 'chromStart': None, 'chromEnd': None, 'strand': None})
                failed_genes.append(gene_name)
        
        if failed_genes and len(failed_genes) <= 10:
            print(f"   ⚠️ Could not find coordinates for genes: {', '.join(failed_genes[:10])}")
        elif failed_genes:
            print(f"   ⚠️ Could not find coordinates for {len(failed_genes)} genes")
        
        return coords
    
    # Add gene coordinates to RNA data
    print(f"🗺️ Processing gene coordinates...")
    print(f"   Processing {len(rna.var_names)} genes...")
    
    gene_coords = get_gene_coordinates(rna.var_names, ensembl)
    rna.var['chrom'] = [c['chrom'] for c in gene_coords]
    rna.var['chromStart'] = [c['chromStart'] for c in gene_coords]
    rna.var['chromEnd'] = [c['chromEnd'] for c in gene_coords]
    rna.var['strand'] = [c['strand'] for c in gene_coords]
    
    # Remove genes without coordinates
    valid_genes = rna.var['chrom'].notna()
    n_valid = valid_genes.sum()
    n_invalid = (~valid_genes).sum()
    
    if n_invalid > 0:
        rna = rna[:, valid_genes].copy()
        print(f"   Filtered out {n_invalid} genes without coordinates")
    
    print(f"✅ Gene coordinate processing complete")
    print(f"   {n_valid} genes retained")
    print(f"   Final RNA shape: {rna.shape}\n")
    
    # Extract ATAC peak coordinates
    print(f"🏔️ Processing ATAC peak coordinates...")
    print(f"   Processing {len(atac.var_names)} peaks...")
    
    try:
        split = atac.var_names.str.split(r"[:-]")
        atac.var["chrom"] = split.map(lambda x: x[0])
        atac.var["chromStart"] = split.map(lambda x: x[1]).astype(int)
        atac.var["chromEnd"] = split.map(lambda x: x[2]).astype(int)
        print("✅ ATAC peak coordinates extracted successfully\n")
    except Exception as e:
        print(f"❌ Error processing ATAC peak coordinates: {e}")
        raise
    
    # Construct guidance graph
    print(f"🕸️ Constructing guidance graph...")
    print(f"   Using {'highly variable' if use_highly_variable else 'all'} features for graph construction")
    
    try:
        guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac)
        n_nodes = guidance.number_of_nodes()
        n_edges = guidance.number_of_edges()
        
        # Validate guidance graph
        scglue.graph.check_graph(guidance, [rna, atac])
        
        print(f"✅ Guidance graph constructed successfully")
        print(f"   Nodes: {n_nodes:,}")
        print(f"   Edges: {n_edges:,}\n")
        
    except Exception as e:
        print(f"❌ Error constructing guidance graph: {e}")
        raise
    
    # Clean data before saving
    print(f"🧹 Preparing data for saving...")
    rna = clean_anndata_for_saving(rna, verbose=True)
    atac = clean_anndata_for_saving(atac, verbose=True)
    
    # Save preprocessed data
    print(f"💾 Saving preprocessed data...")
    print(f"   Output directory: {output_dir}")
    
    rna_path = str(output_path / "rna-pp.h5ad")
    atac_path = str(output_path / "atac-pp.h5ad")
    guidance_path = str(output_path / "guidance.graphml.gz")
    
    try:
        print("   Saving RNA data...")
        rna.write(rna_path, compression=compression)
        print("   Saving ATAC data...")
        atac.write(atac_path, compression=compression)
        print("   Saving guidance graph...")
        nx.write_graphml(guidance, guidance_path)
        
    except Exception as e:
        print(f"❌ Error saving files: {e}")
        print("Debug info:")
        print(f"   RNA obs dtypes: {rna.obs.dtypes}")
        print(f"   RNA var dtypes: {rna.var.dtypes}")
        print(f"   ATAC obs dtypes: {atac.obs.dtypes}")
        print(f"   ATAC var dtypes: {atac.var.dtypes}")
        raise
    
    print("\n🎉 GLUE preprocessing pipeline completed successfully!\n")
    return rna, atac, guidance

def glue_train(preprocess_output_dir, output_dir="glue_output", 
               save_prefix="glue", consistency_threshold=0.05,
               treat_sample_as_batch=True,
               use_highly_variable=True):
    """
    Train SCGLUE model for single-cell multi-omics integration.
    
    Parameters:
    -----------
    preprocess_output_dir : str
        Directory containing preprocessed data
    output_dir : str
        Output directory for saving all training results
    save_prefix : str
        Prefix for saved files
    consistency_threshold : float
        Threshold for integration consistency
    use_highly_variable : bool
        Whether to use only highly variable features or all features
    """
    print("\n\n\n🚀 Starting GLUE training pipeline...\n\n\n")
    print(f"   Feature mode: {'Highly Variable Only' if use_highly_variable else 'All Features'}")
    print(f"   Output directory: {output_dir}\n")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load preprocessed data from preprocessing output directory
    rna_path = os.path.join(preprocess_output_dir, "rna-pp.h5ad")
    atac_path = os.path.join(preprocess_output_dir, "atac-pp.h5ad")
    guidance_path = os.path.join(preprocess_output_dir, "guidance.graphml.gz")
    
    # Check if files exist
    for file_path, file_type in [(rna_path, "RNA"), (atac_path, "ATAC"), (guidance_path, "Guidance")]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_type} file not found: {file_path}")
    
    print(f"\n\n\n📊 Loading preprocessed data from {preprocess_output_dir}...\n\n\n")
    rna = ad.read_h5ad(rna_path)
    atac = ad.read_h5ad(atac_path)
    guidance = nx.read_graphml(guidance_path)
    print(f"\n\n\nData loaded - RNA: {rna.shape}, ATAC: {atac.shape}, Graph: {guidance.number_of_nodes()} nodes\n\n\n")
    
    # 2. Configure datasets with negative binomial distribution
    print("\n\n\n⚙️ Configuring datasets...\n\n\n")

    if treat_sample_as_batch:
        scglue.models.configure_dataset(
            rna, "NB", use_highly_variable=use_highly_variable, 
            use_layer="counts", use_rep="X_pca", use_batch='sample'
        )
        scglue.models.configure_dataset(
            atac, "NB", use_highly_variable=use_highly_variable, 
            use_rep="X_lsi", use_batch='sample'
        )
    else:
        scglue.models.configure_dataset(
            rna, "NB", use_highly_variable=use_highly_variable, 
            use_layer="counts", use_rep="X_pca"
        )
        scglue.models.configure_dataset(
            atac, "NB", use_highly_variable=use_highly_variable, 
            use_rep="X_lsi"
        )
    
    # 3. Extract subgraph based on feature selection strategy
    if use_highly_variable:
        print("\n\n\n🔍 Extracting highly variable feature subgraph...\n\n\n")
        rna_hvf = rna.var.query("highly_variable").index
        atac_hvf = atac.var.query("highly_variable").index
        guidance_hvf = guidance.subgraph(chain(rna_hvf, atac_hvf)).copy()
        print(f"HVF subgraph extracted - RNA HVF: {len(rna_hvf)}, ATAC HVF: {len(atac_hvf)}")
        print(f"HVF graph: {guidance_hvf.number_of_nodes()} nodes, {guidance_hvf.number_of_edges()} edges\n\n\n")
    else:
        print("\n\n\n🔍 Using full feature graph...\n\n\n")
        guidance_hvf = guidance
        print(f"Full graph: {guidance_hvf.number_of_nodes()} nodes, {guidance_hvf.number_of_edges()} edges\n\n\n")
    
    # 4. Train GLUE model (create training subdirectory)
    train_dir = os.path.join(output_dir, "training")
    os.makedirs(train_dir, exist_ok=True)
    
    print(f"\n\n\n🤖 Training GLUE model...\n\n\n")
    glue = scglue.models.fit_SCGLUE(
    {"rna": rna, "atac": atac},
    guidance_hvf,
    fit_kws={
        "directory": train_dir,
        "max_epochs": 500
        }
    )
    
    # 6. Generate embeddings
    print(f"\n\n\n🎨 Generating embeddings...\n\n\n")
    rna.obsm["X_glue"] = glue.encode_data("rna", rna)
    atac.obsm["X_glue"] = glue.encode_data("atac", atac)
    
    feature_embeddings = glue.encode_graph(guidance_hvf)
    feature_embeddings = pd.DataFrame(feature_embeddings, index=glue.vertices)
    rna.varm["X_glue"] = feature_embeddings.reindex(rna.var_names).to_numpy()
    atac.varm["X_glue"] = feature_embeddings.reindex(atac.var_names).to_numpy()
    
    # 7. Save results to output directory
    print(f"\n\n\n💾 Saving results to {output_dir}...\n\n\n")
    model_path = os.path.join(output_dir, f"{save_prefix}.dill")
    rna_emb_path = os.path.join(output_dir, f"{save_prefix}-rna-emb.h5ad")
    atac_emb_path = os.path.join(output_dir, f"{save_prefix}-atac-emb.h5ad")
    guidance_hvf_path = os.path.join(output_dir, f"{save_prefix}-guidance-hvf.graphml.gz")
    
    glue.save(model_path)
    rna.write(rna_emb_path, compression="gzip")
    atac.write(atac_emb_path, compression="gzip")
    nx.write_graphml(guidance_hvf, guidance_hvf_path)
    os.remove(rna_path)
    os.remove(atac_path)
    
    # 5. Check integration consistency
    print(f"\n\n\n📊 Checking integration consistency...\n\n\n")
    consistency_scores = scglue.models.integration_consistency(
        glue, {"rna": rna, "atac": atac}, guidance_hvf
    )
    min_consistency = consistency_scores['consistency'].min()
    mean_consistency = consistency_scores['consistency'].mean()
    print(f"\n\n\nConsistency scores - Min: {min_consistency:.4f}, Mean: {mean_consistency:.4f}\n\n\n")
    # Check if integration is reliable
    is_reliable = min_consistency > consistency_threshold
    status = "✅ RELIABLE" if is_reliable else "❌ UNRELIABLE"
    print(f"\n\n\n📈 Integration Assessment:")
    print(f"Feature mode: {'Highly Variable' if use_highly_variable else 'All Features'}")
    print(f"Consistency threshold: {consistency_threshold}")
    print(f"Minimum consistency: {min_consistency:.4f}")
    print(f"Status: {status}\n\n\n")
    
    if not is_reliable:
        print("\n\n\n⚠️ Low consistency detected. Consider adjusting parameters or checking data quality.\n\n\n")
    
    print(f"\n\n\n🎉 GLUE training pipeline completed successfully!\nResults saved to: {output_dir}\n\n\n")

def compute_gene_activity_from_knn(
    glue_dir: str,
    output_path: str,
    raw_rna_path: str,
    k_neighbors: int = 50,
    use_rep: str = "X_glue",
    metric: str = "cosine",
    use_gpu: bool = True,
    verbose: bool = True,
) -> ad.AnnData:
    """
    Compute gene activity from k-NN with shift-scale weight normalization.
    
    Uses shift-scale approach that preserves all relative differences while
    ensuring weights are in [0,1] and sum to 1.
    """    
    def fix_sparse_matrix_dtype(X, verbose=False):
        """Fix sparse matrix by converting to int64 indices"""
        if not sparse.issparse(X):
            return X
            
        if verbose:
            print(f"   Converting sparse matrix indices to int64...")
            print(f"   Current dtypes - indices: {X.indices.dtype}, indptr: {X.indptr.dtype}")
        
        # Convert to COO then back to CSR with int64
        coo = X.tocoo()
        
        # Create new CSR matrix with int64 indices
        X_fixed = sparse.csr_matrix(
            (coo.data.astype(np.float64), 
             (coo.row.astype(np.int64), coo.col.astype(np.int64))),
            shape=X.shape,
            dtype=np.float64
        )
        
        # Clean up
        X_fixed.eliminate_zeros()
        X_fixed.sort_indices()
        
        if verbose:
            print(f"   Fixed dtypes - indices: {X_fixed.indices.dtype}, indptr: {X_fixed.indptr.dtype}")
        
        return X_fixed
    
    # Memory pool settings for GPU
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    
    # Get available memory for optimal batch sizing
    gpu_mem = cp.cuda.Device().mem_info[0] / 1e9  # Available GPU memory in GB
    cpu_mem = psutil.virtual_memory().available / 1e9  # Available CPU memory in GB
    
    # Construct file paths
    rna_processed_path = os.path.join(glue_dir, "glue-rna-emb.h5ad")
    atac_path = os.path.join(glue_dir, "glue-atac-emb.h5ad")
    
    # Check if files exist
    if not os.path.exists(rna_processed_path):
        raise FileNotFoundError(f"Processed RNA embedding file not found: {rna_processed_path}")
    if not os.path.exists(atac_path):
        raise FileNotFoundError(f"ATAC embedding file not found: {atac_path}")
    if not os.path.exists(raw_rna_path):
        raise FileNotFoundError(f"Raw RNA count file not found: {raw_rna_path}")
    
    if verbose:
        print(f"\n🧬 Computing gene activity using raw RNA counts (highly optimized)...")
        print(f"   k_neighbors: {k_neighbors}")
        print(f"   metric: {metric}")
        print(f"   GPU acceleration: enabled")
        print(f"   Available GPU memory: {gpu_mem:.2f} GB")
        print(f"   Available CPU memory: {cpu_mem:.2f} GB")
    
    # Load processed RNA - ONLY for embeddings
    if verbose:
        print("📂 Loading processed RNA embeddings...")
    
    rna_processed = ad.read_h5ad(rna_processed_path)
    rna_embedding = rna_processed.obsm[use_rep].copy()
    processed_rna_cells = rna_processed.obs.index.copy()  # Keep cell IDs for alignment
    
    # Store obsm from processed RNA (these are embeddings we want to keep)
    rna_obsm_dict = {k: v.copy() for k, v in rna_processed.obsm.items()}
    
    # Clean up memory
    del rna_processed
    gc.collect()
    
    # Load ATAC
    if verbose:
        print("📂 Loading ATAC embeddings...")
    
    atac = ad.read_h5ad(atac_path)
    atac_embedding = atac.obsm[use_rep].copy()
    atac_obs = atac.obs.copy()
    
    # Store all obsm from ATAC
    atac_obsm_dict = {k: v.copy() for k, v in atac.obsm.items()}
    
    n_atac_cells = atac.n_obs
    
    # Clean up memory
    del atac
    gc.collect()
    
    # Load raw RNA counts
    if verbose:
        print("📂 Loading raw RNA counts...")
    
    rna_raw = ad.read_h5ad(raw_rna_path, backed='r')
    
    # Get raw RNA metadata
    raw_rna_obs = rna_raw.obs.copy()
    raw_rna_var = rna_raw.var.copy()
    raw_rna_varm_dict = {k: v.copy() for k, v in rna_raw.varm.items()} if hasattr(rna_raw, 'varm') else {}
    
    # Align cells between processed RNA embeddings and raw RNA counts
    common_cells = processed_rna_cells.intersection(raw_rna_obs.index)
    if len(common_cells) != len(processed_rna_cells):
        if verbose:
            print(f"   Aligning to {len(common_cells)} common cells...")
        # Align embeddings
        embedding_mask = np.isin(processed_rna_cells, common_cells)
        rna_embedding = rna_embedding[embedding_mask]
        
        # Update obsm arrays with aligned cells
        for key in rna_obsm_dict:
            rna_obsm_dict[key] = rna_obsm_dict[key][embedding_mask]
    
    # Use raw RNA obs for the aligned cells
    rna_obs = raw_rna_obs.loc[common_cells].copy()
    
    # Get dimensions
    n_rna_cells = len(common_cells)
    n_genes = rna_raw.n_vars
    
    if verbose:
        print(f"   RNA cells: {n_rna_cells}, ATAC cells: {n_atac_cells}, Genes: {n_genes}")
    
    # Find k-nearest neighbors using GPU
    if verbose:
        print("\n🔍 Finding k-nearest RNA neighbors (GPU-accelerated)...")
        start_time = time.time()
    
    # Use float32 for k-NN computation (faster on GPU)
    rna_embedding_gpu = cp.asarray(rna_embedding, dtype=cp.float32)
    atac_embedding_gpu = cp.asarray(atac_embedding, dtype=cp.float32)
    
    # Clear CPU arrays immediately
    del rna_embedding, atac_embedding
    gc.collect()
    
    # Optimized k-NN search with algorithm selection
    nn = cuNearestNeighbors(
        n_neighbors=k_neighbors, 
        metric=metric,
        algorithm='brute' if n_rna_cells < 50000 else 'auto'
    )
    nn.fit(rna_embedding_gpu)
    distances_gpu, indices_gpu = nn.kneighbors(atac_embedding_gpu)
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"   k-NN search completed in {elapsed:.2f} seconds")
    
    # Compute weights using shift-scale normalization
    if verbose:
        print("\n📐 Computing similarity weights (shift-scale normalization)...")
    
    # Convert distances to similarities based on metric
    if metric == 'cosine':
        # For cosine metric, cuML returns cosine distance in range [0, 2]
        # Convert to cosine similarity in range [-1, 1]
        similarities = 1 - (distances_gpu / 2)
    else:
        # For other metrics, use inverse distance
        similarities = 1 / (1 + distances_gpu)
    
    # Shift and scale to [0, 1] while preserving relative differences
    min_sim = cp.min(similarities, axis=1, keepdims=True)
    max_sim = cp.max(similarities, axis=1, keepdims=True)
    
    # Handle case where all similarities are the same
    sim_range = max_sim - min_sim
    sim_range = cp.where(sim_range > 0, sim_range, 1.0)
    
    # Shift minimum to 0, maximum to 1
    similarities = (similarities - min_sim) / sim_range
    
    # Normalize to sum to 1 (create weights)
    weights_gpu = similarities / cp.sum(similarities, axis=1, keepdims=True)
    
    # Verify weights are valid
    if verbose:
        min_weight = cp.min(weights_gpu)
        max_weight = cp.max(weights_gpu)
        mean_weight = cp.mean(weights_gpu)
        std_weight = cp.std(weights_gpu)
        print(f"   Weight statistics: min={min_weight:.6f}, max={max_weight:.6f}, "
              f"mean={mean_weight:.6f}, std={std_weight:.6f}")
        
        # Check that weights sum to 1 for each cell
        weight_sums = cp.sum(weights_gpu, axis=1)
        print(f"   Weight sums: min={cp.min(weight_sums):.6f}, max={cp.max(weight_sums):.6f}")
        
        # Check effective number of neighbors (inverse Simpson index)
        squared_weights = weights_gpu ** 2
        eff_neighbors = 1.0 / cp.mean(cp.sum(squared_weights, axis=1))
        print(f"   Effective neighbors: {eff_neighbors:.1f} out of {k_neighbors}")
    
    del similarities, distances_gpu
    
    # Determine optimal batch size based on memory
    estimated_memory_per_cell = (k_neighbors * n_genes * 8) / 1e9  # GB per cell
    optimal_batch_size = int(min(
        gpu_mem * 0.5 / estimated_memory_per_cell,  # Use 50% of available GPU memory
        10000,  # Max batch size
        n_atac_cells
    ))
    optimal_batch_size = max(optimal_batch_size, 100)  # Minimum batch size
    
    if verbose:
        print(f"\n🧮 Computing weighted gene activity...")
        print(f"   Optimal batch size: {optimal_batch_size}")
        start_time = time.time()
    
    n_batches = (n_atac_cells + optimal_batch_size - 1) // optimal_batch_size
    
    # Pre-allocate output matrix as float64 for final precision
    gene_activity_matrix = np.zeros((n_atac_cells, n_genes), dtype=np.float64)
    
    # Optimize RNA data loading - check if sparse
    is_sparse_rna = sparse.issparse(rna_raw.X)
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * optimal_batch_size
        end_idx = min((batch_idx + 1) * optimal_batch_size, n_atac_cells)
        batch_size_actual = end_idx - start_idx
        
        # Get batch indices and weights (already on GPU)
        batch_indices_gpu = indices_gpu[start_idx:end_idx]
        batch_weights_gpu = weights_gpu[start_idx:end_idx]
        
        # Get unique RNA indices for this batch
        batch_indices_cpu = cp.asnumpy(batch_indices_gpu).flatten()
        unique_rna_indices = np.unique(batch_indices_cpu)
        
        # Load RNA expression for batch neighbors
        if is_sparse_rna:
            rna_expr_subset = rna_raw[common_cells[unique_rna_indices], :].X
            if sparse.issparse(rna_expr_subset):
                if rna_expr_subset.nnz / rna_expr_subset.size > 0.1:  # If >10% dense
                    rna_expr_subset = rna_expr_subset.toarray()
                else:
                    # Keep sparse and use sparse GPU operations
                    rna_expr_subset = cusparse.csr_matrix(rna_expr_subset, dtype=cp.float32)
        else:
            rna_expr_subset = rna_raw[common_cells[unique_rna_indices], :].X[:]
        
        # Transfer to GPU if not already
        if not isinstance(rna_expr_subset, (cp.ndarray, cusparse.csr_matrix)):
            rna_expr_gpu = cp.asarray(rna_expr_subset, dtype=cp.float32)
        else:
            rna_expr_gpu = rna_expr_subset
        
        # Create mapping for indices
        idx_map = cp.zeros(n_rna_cells, dtype=cp.int32) - 1
        idx_map[unique_rna_indices] = cp.arange(len(unique_rna_indices), dtype=cp.int32)
        mapped_indices_gpu = idx_map[batch_indices_gpu]
        
        # Vectorized computation using advanced indexing
        batch_gene_activity_gpu = cp.zeros((batch_size_actual, n_genes), dtype=cp.float32)
        
        # Use Einstein summation for efficient weighted sum
        for i in range(batch_size_actual):
            cell_indices = mapped_indices_gpu[i]
            if isinstance(rna_expr_gpu, cusparse.csr_matrix):
                neighbor_expr = rna_expr_gpu[cell_indices].toarray()
            else:
                neighbor_expr = rna_expr_gpu[cell_indices]
            
            # Vectorized weighted sum
            batch_gene_activity_gpu[i] = cp.einsum('n,ng->g', 
                                                   batch_weights_gpu[i], 
                                                   neighbor_expr)
        
        # Transfer to CPU and convert to float64
        gene_activity_matrix[start_idx:end_idx] = cp.asnumpy(batch_gene_activity_gpu).astype(np.float64)
        
        # Clear GPU memory for this batch
        del batch_gene_activity_gpu, rna_expr_gpu, mapped_indices_gpu
        
        if verbose and ((batch_idx + 1) % max(1, n_batches // 10) == 0 or batch_idx == n_batches - 1):
            progress = (batch_idx + 1) / n_batches * 100
            print(f"   Progress: {progress:.1f}% ({batch_idx + 1}/{n_batches} batches)")
    
    # Clear GPU memory
    del weights_gpu, indices_gpu, rna_embedding_gpu, atac_embedding_gpu
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"   Gene activity computation completed in {elapsed:.2f} seconds")
    
    # Validate gene activity values
    if verbose:
        print("\n🔍 Validating gene activity counts...")
    
    # Vectorized cleaning
    gene_activity_matrix = np.nan_to_num(gene_activity_matrix, 0)
    np.clip(gene_activity_matrix, 0, None, out=gene_activity_matrix)
    
    # Convert to sparse CSR matrix with int64 indices
    if verbose:
        print("📦 Converting gene activity to sparse CSR matrix with int64 indices...")
        nnz_before = np.count_nonzero(gene_activity_matrix)
        sparsity = 1 - (nnz_before / gene_activity_matrix.size)
        print(f"   Sparsity: {sparsity:.1%} zeros")
    
    gene_activity_sparse = sparse.csr_matrix(gene_activity_matrix, dtype=np.float64)
    gene_activity_sparse = fix_sparse_matrix_dtype(gene_activity_sparse, verbose=verbose)
    
    # Clear dense matrix
    del gene_activity_matrix
    gc.collect()
    
    # Create gene activity AnnData with raw RNA var
    gene_activity_adata = ad.AnnData(
        X=gene_activity_sparse,
        obs=atac_obs.copy(),
        var=raw_rna_var.copy()  # Use raw RNA var
    )
    
    gene_activity_adata.obs['modality'] = 'ATAC'
    gene_activity_adata.layers['gene_activity'] = gene_activity_sparse.copy()
    
    # Add all obsm from ATAC
    for key, value in atac_obsm_dict.items():
        gene_activity_adata.obsm[key] = value
    
    # Add varm from raw RNA only
    for key, value in raw_rna_varm_dict.items():
        gene_activity_adata.varm[key] = value
    
    # Prepare RNA data for merging
    if verbose:
        print("\n🔗 Preparing RNA data for merging...")
    
    # Load RNA data efficiently
    if is_sparse_rna:
        rna_X = rna_raw[common_cells, :].X
        if sparse.issparse(rna_X):
            rna_X = rna_X.tocsr().astype(np.float64)
            # Fix sparse matrix dtype to int64
            rna_X = fix_sparse_matrix_dtype(rna_X, verbose=verbose)
            if verbose:
                rna_sparsity = 1 - (rna_X.nnz / (rna_X.shape[0] * rna_X.shape[1]))
                print(f"   RNA sparsity: {rna_sparsity:.1%} zeros")
        else:
            rna_X = rna_X[:].astype(np.float64)
            nnz = np.count_nonzero(rna_X)
            sparsity = 1 - (nnz / rna_X.size)
            if sparsity > 0.5:
                if verbose:
                    print(f"   Converting RNA to sparse (sparsity: {sparsity:.1%})")
                rna_X = sparse.csr_matrix(rna_X, dtype=np.float64)
                rna_X = fix_sparse_matrix_dtype(rna_X, verbose=verbose)
    else:
        rna_X = rna_raw[common_cells, :].X[:].astype(np.float64)
        nnz = np.count_nonzero(rna_X)
        sparsity = 1 - (nnz / rna_X.size)
        if sparsity > 0.5:
            if verbose:
                print(f"   Converting RNA to sparse (sparsity: {sparsity:.1%})")
            rna_X = sparse.csr_matrix(rna_X, dtype=np.float64)
            rna_X = fix_sparse_matrix_dtype(rna_X, verbose=verbose)
    
    # Close the backed file
    rna_raw.file.close()
    del rna_raw
    gc.collect()
    
    # Create RNA AnnData with raw RNA obs/var
    rna_for_merge = ad.AnnData(
        X=rna_X,
        obs=rna_obs.copy(),  # Using raw RNA obs
        var=raw_rna_var.copy()  # Using raw RNA var
    )
    
    rna_for_merge.obs['modality'] = 'RNA'
    
    # Add obsm from processed RNA (embeddings)
    for key, value in rna_obsm_dict.items():
        rna_for_merge.obsm[key] = value
    
    # Add varm from raw RNA only
    for key, value in raw_rna_varm_dict.items():
        rna_for_merge.varm[key] = value
    
    # Handle index overlaps
    if verbose:
        print("\n🔗 Merging gene activity with RNA data...")
        print("   Checking for index overlaps...")
    
    # Check for overlapping indices
    rna_indices = set(rna_for_merge.obs.index)
    atac_indices = set(gene_activity_adata.obs.index)
    overlap = rna_indices.intersection(atac_indices)
    
    if overlap:
        if verbose:
            print(f"   ⚠️ Found {len(overlap)} overlapping indices")
            print(f"   Adding modality suffix to ensure unique indices...")
    else:
        if verbose:
            print(f"   No overlapping indices found, but adding suffix for safety...")
    
    # ALWAYS add modality suffix for safety
    # Store original barcodes
    rna_for_merge.obs['original_barcode'] = rna_for_merge.obs.index
    gene_activity_adata.obs['original_barcode'] = gene_activity_adata.obs.index
    
    # Create unique indices with modality suffix
    rna_for_merge.obs.index = pd.Index([f"{idx}_RNA" for idx in rna_for_merge.obs.index])
    gene_activity_adata.obs.index = pd.Index([f"{idx}_ATAC" for idx in gene_activity_adata.obs.index])
    
    if verbose:
        print(f"   RNA cells: {rna_for_merge.n_obs}")
        print(f"   ATAC cells: {gene_activity_adata.n_obs}")
    
    # Merge datasets with fixed indices
    merged_adata = ad.concat(
        [rna_for_merge, gene_activity_adata], 
        axis=0, 
        join='inner',
        merge='same',
        label=None,  # Don't add batch labels
        keys=None,   # Don't add keys  
        index_unique=None  # Don't modify indices further
    )
    
    del rna_for_merge, gene_activity_adata
    gc.collect()
    
    # Verify indices are unique
    if not merged_adata.obs.index.is_unique:
        if verbose:
            print("   ⚠️ Warning: Non-unique indices detected, fixing...")
        merged_adata.obs_names_make_unique()
    
    # Ensure proper sparse matrix format with int64
    if sparse.issparse(merged_adata.X):
        if verbose:
            print("📦 Ensuring merged matrix has int64 indices...")
        
        # Convert to CSR format (better for row operations)
        if not isinstance(merged_adata.X, sparse.csr_matrix):
            merged_adata.X = merged_adata.X.tocsr()
        
        # Fix dtype to int64
        merged_adata.X = fix_sparse_matrix_dtype(merged_adata.X, verbose=verbose)
        
        # Final cleanup
        merged_adata.X.sort_indices()
        merged_adata.X.eliminate_zeros()
    
    # Clean for saving
    merged_adata = clean_anndata_for_saving(merged_adata, verbose=False)
    
    # Save with optimized compression
    output_dir = os.path.join(output_path, 'preprocess')
    os.makedirs(output_dir, exist_ok=True)
    output_path_anndata = os.path.join(output_dir, 'atac_rna_integrated.h5ad')
    
    merged_adata.write(output_path_anndata, 
                      compression='gzip',
                      compression_opts=4)
    
    if verbose:
        print(f"\n✅ Gene activity computation and merging complete!")
        print(f"\n📊 Summary:")
        print(f"   Output path: {output_path_anndata}")
        print(f"   Merged dataset shape: {merged_adata.shape}")
        print(f"   RNA cells: {(merged_adata.obs['modality'] == 'RNA').sum()}")
        print(f"   ATAC cells: {(merged_adata.obs['modality'] == 'ATAC').sum()}")
        print(f"   Unique cell indices: {merged_adata.obs.index.is_unique}")
        
        # Report matrix format
        if sparse.issparse(merged_adata.X):
            matrix_type = type(merged_adata.X).__name__
            sparsity = 1 - (merged_adata.X.nnz / (merged_adata.X.shape[0] * merged_adata.X.shape[1]))
            print(f"   Matrix format: {matrix_type} (sparse)")
            print(f"   Matrix index dtype: {merged_adata.X.indices.dtype}")
            print(f"   Matrix indptr dtype: {merged_adata.X.indptr.dtype}")
            print(f"   Data type: {merged_adata.X.dtype}")
            print(f"   Sparsity: {sparsity:.1%} zeros")
            print(f"   Memory usage: {merged_adata.X.data.nbytes / 1e6:.2f} MB (sparse)")
        else:
            print(f"   Matrix format: dense numpy array")
            print(f"   Data type: {merged_adata.X.dtype}")
            print(f"   Memory usage: {merged_adata.X.nbytes / 1e6:.2f} MB")
        
        print(f"   Optimization: GPU vectorization + dynamic batching + sparse matrices")
        print(f"   Batch processing: {n_batches} batches of size {optimal_batch_size}")
        print(f"   Weight normalization: shift-scale (preserves all relative differences)")

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    gc.collect()
    
    return merged_adata

matplotlib.use('Agg')  # Use non-interactive backend to prevent figure warnings

def glue_visualize(integrated_path, output_dir=None, plot_columns=None):
    """
    Load integrated RNA-ATAC data and create joint UMAP visualization
    
    Parameters:
    -----------
    integrated_path : str
        Path to integrated h5ad file (e.g., atac_rna_integrated.h5ad)
    output_dir : str, optional
        Output directory. If None, uses directory of integrated file
    plot_columns : list, optional
        List of column names to plot. If None, defaults to ['modality', 'cell_type']
    """
    
    # Load the integrated data
    print("Loading integrated RNA-ATAC data...")
    combined = ad.read_h5ad(integrated_path)
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(integrated_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if scGLUE embeddings exist
    if "X_glue" not in combined.obsm:
        print("Error: X_glue embeddings not found in integrated data. Run scGLUE integration first.")
        return
    
    print("Computing UMAP from scGLUE embeddings...")
    # Compute neighbors and UMAP using the scGLUE embeddings
    sc.pp.neighbors(combined, use_rep="X_glue", metric="cosine")
    sc.tl.umap(combined)
    
    # Set up plotting parameters
    sc.settings.set_figure_params(dpi=80, facecolor='white', figsize=(8, 6))
    plt.rcParams['figure.max_open_warning'] = 50  # Increase the warning threshold
    
    # Set default columns if none specified
    if plot_columns is None:
        plot_columns = ['modality', 'cell_type']
    
    print(f"Generating visualizations for columns: {plot_columns}")
    
    # Generate plots for specified columns
    for col in plot_columns:
        if col not in combined.obs.columns:
            print(f"Warning: Column '{col}' not found in data. Skipping...")
            continue
        
        # Skip columns that also exist in var_names to avoid ambiguity
        if col in combined.var_names:
            print(f"Warning: Column '{col}' exists in both obs.columns and var_names. Skipping...")
            continue
        
        try:
            plt.figure(figsize=(12, 8))
            sc.pl.umap(combined, color=col, 
                       title=f"scGLUE Integration: {col}",
                       save=False, show=False, wspace=0.65)
            plt.tight_layout()
            col_plot_path = os.path.join(output_dir, f"scglue_umap_{col}.png")
            plt.savefig(col_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {col} plot: {col_plot_path}")
        except Exception as e:
            print(f"Error plotting {col}: {str(e)}")
            plt.close()  # Close the figure even if there's an error
    
    # Create modality-cell type distribution heatmap if both columns exist and are in plot_columns
    if ("modality" in plot_columns and "modality" in combined.obs.columns and 
        "cell_type" in plot_columns and "cell_type" in combined.obs.columns):
        print("\nCreating modality-cell type distribution heatmap...")
        
        # Create a cross-tabulation of modality and cell type
        modality_celltype_counts = pd.crosstab(
            combined.obs['cell_type'], 
            combined.obs['modality']
        )
        
        # Calculate proportions (percentage of each modality within each cell type)
        modality_celltype_props = modality_celltype_counts.div(
            modality_celltype_counts.sum(axis=1), 
            axis=0
        ) * 100
        
        # Create two heatmaps: one for counts and one for proportions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Heatmap 1: Raw counts
        sns.heatmap(
            modality_celltype_counts,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar_kws={'label': 'Number of cells'},
            ax=ax1
        )
        ax1.set_title('Cell Count Distribution: Modality vs Cell Type', fontsize=14, pad=20)
        ax1.set_xlabel('Modality', fontsize=12)
        ax1.set_ylabel('Cell Type', fontsize=12)
        
        # Heatmap 2: Proportions
        sns.heatmap(
            modality_celltype_props,
            annot=True,
            fmt='.1f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Percentage (%)'},
            ax=ax2
        )
        ax2.set_title('Modality Distribution within Each Cell Type (%)', fontsize=14, pad=20)
        ax2.set_xlabel('Modality', fontsize=12)
        ax2.set_ylabel('Cell Type', fontsize=12)
        
        plt.tight_layout()
        heatmap_plot_path = os.path.join(output_dir, "scglue_modality_celltype_heatmap.png")
        plt.savefig(heatmap_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved modality-cell type heatmap: {heatmap_plot_path}")
        
        # Also create a stacked bar plot for better visualization
        plt.figure(figsize=(14, 8))
        
        # Calculate percentages for stacked bar
        modality_celltype_props_T = modality_celltype_props.T
        
        # Create stacked bar plot
        modality_celltype_props_T.plot(
            kind='bar',
            stacked=True,
            colormap='tab20',
            width=0.8
        )
        
        plt.title('Modality Composition by Cell Type', fontsize=16, pad=20)
        plt.xlabel('Modality', fontsize=12)
        plt.ylabel('Percentage of Cells (%)', fontsize=12)
        plt.legend(title='Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        stacked_plot_path = os.path.join(output_dir, "scglue_modality_celltype_stacked.png")
        plt.savefig(stacked_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved modality-cell type stacked plot: {stacked_plot_path}")
        
        # Print summary statistics
        print("\n=== Modality-Cell Type Distribution Summary ===")
        print(f"Total cell types: {len(modality_celltype_counts.index)}")
        print(f"Total modalities: {len(modality_celltype_counts.columns)}")
        
        # Print cells per modality per cell type
        print("\nCell counts by modality and cell type:")
        print(modality_celltype_counts.to_string())
        
        # Save the distribution table as CSV
        csv_path = os.path.join(output_dir, "scglue_modality_celltype_distribution.csv")
        modality_celltype_counts.to_csv(csv_path)
        print(f"\nSaved distribution table to: {csv_path}")
        
    # Generate summary statistics
    print("\n=== Integration Summary ===")
    print(f"Total cells: {combined.n_obs}")
    print(f"Total features: {combined.n_vars}")
    print(f"Available metadata columns: {list(combined.obs.columns)}")
    
    # Show breakdown by modality if available
    if "modality" in combined.obs.columns:
        modality_counts = combined.obs['modality'].value_counts()
        print(f"\nModality breakdown:")
        for modality, count in modality_counts.items():
            print(f"  {modality}: {count} cells")
    
    # Check feature usage
    hvg_used = combined.var['highly_variable'].sum() if 'highly_variable' in combined.var else combined.n_vars
    print(f"\nFeatures used: {hvg_used}/{combined.n_vars}")
    
    print("\nVisualization complete!")
    
def glue(
    # Data files
    rna_file: str,
    atac_file: str,
    rna_sample_meta_file: Optional[str] = None,
    atac_sample_meta_file: Optional[str] = None,
    additional_hvg_file: Optional[str] = None,
    
    # Process control flags
    run_preprocessing: bool = True,
    run_training: bool = True,
    run_gene_activity: bool = True,
    run_cell_types: bool = True,
    run_visualization: bool = True,
    
    # Preprocessing parameters
    ensembl_release: int = 98,
    species: str = "homo_sapiens",
    use_highly_variable: bool = True,
    n_top_genes: int = 2000,
    n_pca_comps: int = 50,
    n_lsi_comps: int = 50,
    lsi_n_iter: int = 15,
    gtf_by: str = "gene_name",
    flavor: str = "seurat_v3",
    generate_umap: bool = False,
    compression: str = "gzip",
    random_state: int = 42,
    metadata_sep: str = ",",
    rna_sample_column: str = "sample",
    atac_sample_column: str = "sample",
    
    # Training parameters
    consistency_threshold: float = 0.05,
    treat_sample_as_batch: bool = True,
    save_prefix: str = "glue",
    
    # Gene activity computation parameters
    k_neighbors: int = 10,
    use_rep: str = "X_glue",
    metric: str = "cosine",
    use_gpu: bool = True,
    verbose: bool = True,
    
    # Cell type assignment parameters
    existing_cell_types: bool = False,
    n_target_clusters: int = 10,
    cluster_resolution: float = 0.8,
    use_rep_celltype: str = "X_glue",
    markers: Optional[List] = None,
    method: str = 'average',
    metric_celltype: str = 'euclidean',
    distance_mode: str = 'centroid',
    generate_umap_celltype: bool = True,
    
    # Visualization parameters
    plot_columns: Optional[List[str]] = None,
    
    # Output directory
    output_dir: str = "./glue_results",
):
    """Complete GLUE pipeline that runs preprocessing, training, gene activity computation, cell type assignment, and visualization.
    
    Use process flags to control which steps to run:
    - run_preprocessing: Run data preprocessing
    - run_training: Run model training
    - run_gene_activity: Compute gene activity
    - run_cell_types: Assign cell types
    - run_visualization: Generate visualizations
    """
    
    os.makedirs(output_dir, exist_ok=True)
    glue_output_dir = os.path.join(output_dir, "integration", "glue")
    start_time = time.time()
    
    # Step 1: Preprocessing
    if run_preprocessing:
        print("Running preprocessing...")
        rna, atac, guidance = glue_preprocess_pipeline(
            rna_file=rna_file,
            atac_file=atac_file,
            rna_sample_meta_file=rna_sample_meta_file,
            atac_sample_meta_file=atac_sample_meta_file,
            additional_hvg_file = additional_hvg_file,
            ensembl_release=ensembl_release,
            species=species,
            output_dir=glue_output_dir,
            use_highly_variable=use_highly_variable,
            n_top_genes=n_top_genes,
            n_pca_comps=n_pca_comps,
            n_lsi_comps=n_lsi_comps,
            lsi_n_iter=lsi_n_iter,
            gtf_by=gtf_by,
            flavor=flavor,
            generate_umap=generate_umap,
            compression=compression,
            random_state=random_state,
            metadata_sep=metadata_sep,
            rna_sample_column=rna_sample_column,
            atac_sample_column=atac_sample_column
        )
        print("Preprocessing completed.")
    
    # Step 2: Training
    if run_training:
        print("Running training...")
        glue_train(
            preprocess_output_dir=glue_output_dir,
            save_prefix=save_prefix,
            consistency_threshold=consistency_threshold,
            treat_sample_as_batch = treat_sample_as_batch,
            use_highly_variable=use_highly_variable,
            output_dir=glue_output_dir
        )
        print("Training completed.")
    
    # Step 3: Memory management and gene activity computation
    if run_gene_activity:
        print("Computing gene activity...")        
        merged_adata = compute_gene_activity_from_knn(
            glue_dir=glue_output_dir,
            output_path=output_dir,
            raw_rna_path=rna_file,
            k_neighbors=k_neighbors,
            use_rep=use_rep,
            metric=metric,
            use_gpu=use_gpu,
            verbose=verbose
        )
        print("Gene activity computation completed.")
    else:
        # If gene activity step is skipped, load the existing merged data
        integrated_file = os.path.join(output_dir, "preprocess", "atac_rna_integrated.h5ad")
        if os.path.exists(integrated_file):
            merged_adata = ad.read_h5ad(integrated_file)
        else:
            raise FileNotFoundError(f"Integrated file not found: {integrated_file}. Run gene activity computation first.")
    
    # Step 4: Cell type assignment
    if run_cell_types:
        print("Assigning cell types...")
    
        # Apply cell type assignment
        if use_gpu:
                from linux.CellType_linux import cell_types_linux

                print(" Using Linux Cell Type Assignment for cell type assignment...")

                merged_adata = cell_types_linux(
                    merged_adata,
                    cell_type_column='cell_type',
                    existing_cell_types=existing_cell_types,
                    n_target_clusters=n_target_clusters,
                    umap=False,  # We'll handle UMAP separately
                    Save=False,  # We'll save the final result
                    output_dir=output_dir,
                    defined_output_path = os.path.join(output_dir, "preprocess", "atac_rna_integrated.h5ad"),
                    cluster_resolution=cluster_resolution,
                    use_rep=use_rep_celltype,
                    markers=markers,
                    method=method,
                    metric=metric_celltype,
                    distance_mode=distance_mode,
                    num_PCs=n_lsi_comps,
                    verbose=verbose
                )

    # Step 5: Visualization
    if run_visualization:
        print("Running visualization...")
        integrated_file_path = os.path.join(output_dir, "preprocess", "atac_rna_integrated.h5ad")
        glue_visualize(
            integrated_path=integrated_file_path,
            output_dir=os.path.join(output_dir, "visualization"),
            plot_columns=plot_columns
        )
        print("Visualization completed.")
    
    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    print(f"\nTotal runtime: {elapsed_minutes:.2f} minutes")

    # Return the merged data if it was computed in this run
    if run_gene_activity or run_cell_types:
        return merged_adata
    else:
        return None