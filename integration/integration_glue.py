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
from utils.safe_save import safe_h5ad_write
from utils.merge_sample_meta import merge_sample_metadata
from integration.integration_visualization import glue_visualize

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
    gtf_by: str = "gene_name",
    flavor: str = "seurat_v3",
    generate_umap: bool = False,
    rna_sample_column: str = "sample",
    atac_sample_column: str = "sample"
) -> Tuple[ad.AnnData, ad.AnnData, nx.MultiDiGraph]:
    """
    Complete GLUE preprocessing pipeline for scRNA-seq and scATAC-seq data integration.
    
    Now supports both gene symbols and Ensembl gene IDs for RNA data.
    
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
    gtf_by : str
        Gene annotation method
    flavor : str
        Method for highly variable gene selection
    generate_umap : bool
        Whether to generate UMAP embeddings
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
                verbose=True
            )
        
        if atac_sample_meta_file:
            print(f"   Processing ATAC metadata: {atac_sample_meta_file}")
            atac = merge_sample_metadata(
                atac, 
                atac_sample_meta_file, 
                sample_column=atac_sample_column,
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

    scglue.data.lsi(atac, n_components=n_lsi_comps)
    
    if generate_umap:
        print("   Computing UMAP embedding...")
        sc.pp.neighbors(atac, use_rep="X_lsi", metric="cosine")
        sc.tl.umap(atac)
    
    print("✅ ATAC preprocessing complete\n")
    
    # Detect gene ID format
    print(f"🔍 Detecting gene ID format...")
    sample_genes = rna.var_names[:100]
    ensembl_pattern = sample_genes.str.match(r'^ENS[A-Z]*G\d+')
    use_ensembl_ids = ensembl_pattern.sum() > 50  # If >50% are Ensembl IDs
    
    if use_ensembl_ids:
        print(f"   Detected Ensembl gene IDs format")
    else:
        print(f"   Detected gene symbol format")
    print()
    
    # Get gene coordinates from Ensembl
    def get_gene_coordinates(gene_ids, ensembl_db, use_ensembl_id=False):
        """Extract gene coordinates from pyensembl database with support for both gene symbols and Ensembl IDs"""
        coords = []
        failed_genes = []
        
        for gene_id in gene_ids:
            try:
                if use_ensembl_id:
                    # Handle Ensembl IDs - remove version suffix if present
                    clean_id = gene_id.split('.')[0] if '.' in gene_id else gene_id
                    try:
                        gene = ensembl_db.gene_by_id(clean_id)
                        strand = '+' if gene.strand == '+' else '-'
                        coords.append({
                            'chrom': f"chr{gene.contig}",
                            'chromStart': gene.start,
                            'chromEnd': gene.end,
                            'strand': strand
                        })
                    except ValueError:
                        # Gene ID not found
                        coords.append({'chrom': None, 'chromStart': None, 'chromEnd': None, 'strand': None})
                        failed_genes.append(gene_id)
                else:
                    # Handle gene symbols
                    genes = ensembl_db.genes_by_name(gene_id)
                    if genes:
                        gene = genes[0]  # Take first match
                        strand = '+' if gene.strand == '+' else '-'
                        coords.append({
                            'chrom': f"chr{gene.contig}",
                            'chromStart': gene.start,
                            'chromEnd': gene.end,
                            'strand': strand
                        })
                    else:
                        coords.append({'chrom': None, 'chromStart': None, 'chromEnd': None, 'strand': None})
                        failed_genes.append(gene_id)
            except Exception as e:
                coords.append({'chrom': None, 'chromStart': None, 'chromEnd': None, 'strand': None})
                failed_genes.append(gene_id)
        
        if failed_genes and len(failed_genes) <= 10:
            print(f"   ⚠️ Could not find coordinates for genes: {', '.join(failed_genes[:10])}")
        elif failed_genes:
            print(f"   ⚠️ Could not find coordinates for {len(failed_genes)} genes")
        
        return coords
    
    # Add gene coordinates to RNA data
    print(f"🗺️ Processing gene coordinates...")
    print(f"   Processing {len(rna.var_names)} genes...")
    
    gene_coords = get_gene_coordinates(rna.var_names, ensembl, use_ensembl_id=use_ensembl_ids)
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
        
        # Add strand information for ATAC peaks (default to '+' as peaks are strand-agnostic)
        if 'strand' not in atac.var.columns:
            atac.var['strand'] = '+'
        
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
    
    # Save preprocessed data
    print(f"💾 Saving preprocessed data...")
    print(f"   Output directory: {output_dir}")
    
    rna_path = str(output_path / "rna-pp.h5ad")
    atac_path = str(output_path / "atac-pp.h5ad")
    guidance_path = str(output_path / "guidance.graphml.gz")
    
    try:
        print("   Saving RNA data...")
        safe_h5ad_write(rna, rna_path)
        print("   Saving ATAC data...")
        safe_h5ad_write(atac, atac_path)
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
    k_neighbors: int = 1,
    use_rep: str = "X_glue",
    metric: str = "cosine",
    use_gpu: bool = True,
    verbose: bool = True,
) -> ad.AnnData:
    
    def fix_sparse_matrix_dtype(X, verbose=False):
        if not sparse.issparse(X):
            return X
            
        if verbose:
            print(f"   Converting sparse matrix indices to int64...")
        
        coo = X.tocoo()
        X_fixed = sparse.csr_matrix(
            (coo.data.astype(np.float64), 
             (coo.row.astype(np.int64), coo.col.astype(np.int64))),
            shape=X.shape,
            dtype=np.float64
        )
        X_fixed.eliminate_zeros()
        X_fixed.sort_indices()
        
        return X_fixed
    
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    
    gpu_mem = cp.cuda.Device().mem_info[0] / 1e9
    cpu_mem = psutil.virtual_memory().available / 1e9
    
    rna_processed_path = os.path.join(glue_dir, "glue-rna-emb.h5ad")
    atac_path = os.path.join(glue_dir, "glue-atac-emb.h5ad")
    
    if not os.path.exists(rna_processed_path):
        raise FileNotFoundError(f"Processed RNA embedding file not found: {rna_processed_path}")
    if not os.path.exists(atac_path):
        raise FileNotFoundError(f"ATAC embedding file not found: {atac_path}")
    if not os.path.exists(raw_rna_path):
        raise FileNotFoundError(f"Raw RNA count file not found: {raw_rna_path}")
    
    if verbose:
        print(f"\n🧬 Computing gene activity using raw RNA counts...")
        print(f"   k_neighbors: {k_neighbors}")
        print(f"   metric: {metric}")
        print(f"   GPU acceleration: {'enabled' if use_gpu else 'disabled'}")
        print(f"   Available GPU memory: {gpu_mem:.2f} GB")
        print(f"   Available CPU memory: {cpu_mem:.2f} GB")
    
    if verbose:
        print("\n📂 Loading processed RNA embeddings and metadata...")
    
    rna_processed = ad.read_h5ad(rna_processed_path)
    rna_embedding = rna_processed.obsm[use_rep].copy()
    processed_rna_cells = rna_processed.obs.index.copy()
    rna_obsm_dict = {k: v.copy() for k, v in rna_processed.obsm.items()}
    processed_rna_obs = rna_processed.obs.copy()
    
    if verbose:
        print(f"   RNA cells: {len(processed_rna_cells)}")
        print(f"   Obs columns: {list(processed_rna_obs.columns)}")
    
    del rna_processed
    gc.collect()

    if verbose:
        print("\n📂 Loading ATAC embeddings...")
    
    atac = ad.read_h5ad(atac_path)
    atac_embedding = atac.obsm[use_rep].copy()
    atac_obs = atac.obs.copy()
    atac_obsm_dict = {k: v.copy() for k, v in atac.obsm.items()}
    n_atac_cells = atac.n_obs
    
    if verbose:
        print(f"   ATAC cells: {n_atac_cells}")
    
    del atac
    gc.collect()
    
    if verbose:
        print("\n📂 Loading raw RNA counts...")
    
    rna_raw = ad.read_h5ad(raw_rna_path)
    raw_rna_var = rna_raw.var.copy()
    raw_rna_varm_dict = {k: v.copy() for k, v in rna_raw.varm.items()} if hasattr(rna_raw, 'varm') else {}
    raw_rna_obs_index = rna_raw.obs.index.copy()
    
    if sparse.issparse(rna_raw.X):
        rna_X_full = rna_raw.X.tocsr()
    else:
        rna_X_full = rna_raw.X
    
    if verbose:
        print(f"   RNA matrix shape: {rna_X_full.shape}")
    
    del rna_raw
    gc.collect()

    if verbose:
        print("\n🔗 Aligning cells...")
    
    common_cells = processed_rna_cells.intersection(raw_rna_obs_index)
    
    if len(common_cells) == 0:
        raise ValueError("No common cells between processed and raw RNA!")
    
    if len(common_cells) != len(processed_rna_cells):
        if verbose:
            print(f"   Aligning to {len(common_cells)} common cells...")
        embedding_mask = np.isin(processed_rna_cells, common_cells)
        rna_embedding = rna_embedding[embedding_mask]
        
        for key in rna_obsm_dict:
            rna_obsm_dict[key] = rna_obsm_dict[key][embedding_mask]
    
    rna_obs = processed_rna_obs.loc[common_cells].copy()
    
    raw_rna_cell_to_idx = {cell: idx for idx, cell in enumerate(raw_rna_obs_index)}
    common_cells_list = list(common_cells)
    common_cells_raw_indices = np.array([raw_rna_cell_to_idx[cell] for cell in common_cells_list], dtype=np.int64)
    
    n_rna_cells = len(common_cells_list)
    n_genes = rna_X_full.shape[1]
    
    if verbose:
        print(f"   RNA cells: {n_rna_cells}, ATAC cells: {n_atac_cells}, Genes: {n_genes}")

    if verbose:
        print("\n🔍 Finding k-nearest RNA neighbors...")
    
    rna_embedding_gpu = cp.asarray(rna_embedding, dtype=cp.float32)
    atac_embedding_gpu = cp.asarray(atac_embedding, dtype=cp.float32)
    
    del rna_embedding, atac_embedding
    gc.collect()
    
    nn = cuNearestNeighbors(
        n_neighbors=k_neighbors, 
        metric=metric,
        algorithm='brute' if n_rna_cells < 50000 else 'auto'
    )
    nn.fit(rna_embedding_gpu)
    distances_gpu, indices_gpu = nn.kneighbors(atac_embedding_gpu)
    
    if verbose:
        print("\n📐 Computing similarity weights...")
    
    if metric == 'cosine':
        similarities = 1 - (distances_gpu / 2)
    else:
        similarities = 1 / (1 + distances_gpu)
    
    min_sim = cp.min(similarities, axis=1, keepdims=True)
    max_sim = cp.max(similarities, axis=1, keepdims=True)
    sim_range = max_sim - min_sim
    all_equal = sim_range == 0
    
    if cp.any(all_equal):
        weights_gpu = cp.ones_like(similarities, dtype=cp.float32) / k_neighbors
        if not cp.all(all_equal):
            similarities = cp.where(all_equal, 0, (similarities - min_sim) / sim_range)
            similarities = similarities / cp.sum(similarities, axis=1, keepdims=True)
            weights_gpu = cp.where(all_equal, weights_gpu, similarities)
    else:
        similarities = (similarities - min_sim) / sim_range
        weights_gpu = similarities / cp.sum(similarities, axis=1, keepdims=True)
    
    if verbose:
        print(f"   Weight stats: min={float(cp.min(weights_gpu)):.6f}, max={float(cp.max(weights_gpu)):.6f}")
    
    del similarities, distances_gpu
    
    estimated_memory_per_cell = (k_neighbors * n_genes * 8) / 1e9
    optimal_batch_size = int(min(
        gpu_mem * 0.5 / estimated_memory_per_cell,
        10000,
        n_atac_cells
    ))
    optimal_batch_size = max(optimal_batch_size, 100)
    
    if verbose:
        print(f"\n🧮 Computing weighted gene activity...")
        print(f"   Batch size: {optimal_batch_size}")
    
    n_batches = (n_atac_cells + optimal_batch_size - 1) // optimal_batch_size
    gene_activity_matrix = np.zeros((n_atac_cells, n_genes), dtype=np.float64)
    is_sparse_rna = sparse.issparse(rna_X_full)
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * optimal_batch_size
        end_idx = min((batch_idx + 1) * optimal_batch_size, n_atac_cells)
        batch_size_actual = end_idx - start_idx
        
        batch_indices_gpu = indices_gpu[start_idx:end_idx]
        batch_weights_gpu = weights_gpu[start_idx:end_idx]
        
        batch_indices_cpu = cp.asnumpy(batch_indices_gpu).flatten()
        unique_common_indices = np.unique(batch_indices_cpu)
        unique_raw_indices = common_cells_raw_indices[unique_common_indices]
        
        if is_sparse_rna:
            rna_expr_subset = rna_X_full[unique_raw_indices, :]
            if sparse.issparse(rna_expr_subset):
                if rna_expr_subset.nnz / rna_expr_subset.size > 0.1:
                    rna_expr_subset = rna_expr_subset.toarray()
                else:
                    rna_expr_subset = cusparse.csr_matrix(rna_expr_subset, dtype=cp.float32)
            else:
                rna_expr_subset = np.asarray(rna_expr_subset)
        else:
            rna_expr_subset = rna_X_full[unique_raw_indices, :]
        
        raw_idx_to_subset_idx = {raw_idx: subset_idx for subset_idx, raw_idx in enumerate(unique_raw_indices)}
        common_to_subset = np.array([raw_idx_to_subset_idx[common_cells_raw_indices[ci]] for ci in unique_common_indices], dtype=np.int32)
        
        if not isinstance(rna_expr_subset, (cp.ndarray, cusparse.csr_matrix)):
            rna_expr_gpu = cp.asarray(rna_expr_subset, dtype=cp.float32)
        else:
            rna_expr_gpu = rna_expr_subset
        
        idx_map = cp.zeros(n_rna_cells, dtype=cp.int32) - 1
        idx_map[unique_common_indices] = cp.asarray(common_to_subset, dtype=cp.int32)
        mapped_indices_gpu = idx_map[batch_indices_gpu]
        
        batch_gene_activity_gpu = cp.zeros((batch_size_actual, n_genes), dtype=cp.float32)
        
        for i in range(batch_size_actual):
            cell_indices = mapped_indices_gpu[i]
            if isinstance(rna_expr_gpu, cusparse.csr_matrix):
                neighbor_expr = rna_expr_gpu[cell_indices].toarray()
            else:
                neighbor_expr = rna_expr_gpu[cell_indices]
            
            batch_gene_activity_gpu[i] = cp.einsum('n,ng->g', 
                                                   batch_weights_gpu[i], 
                                                   neighbor_expr)
        
        gene_activity_matrix[start_idx:end_idx] = cp.asnumpy(batch_gene_activity_gpu).astype(np.float64)
        
        del batch_gene_activity_gpu, rna_expr_gpu, mapped_indices_gpu
        
        if verbose and ((batch_idx + 1) % max(1, n_batches // 10) == 0 or batch_idx == n_batches - 1):
            progress = (batch_idx + 1) / n_batches * 100
            print(f"   Progress: {progress:.1f}% ({batch_idx + 1}/{n_batches} batches)")
    
    del weights_gpu, indices_gpu, rna_embedding_gpu, atac_embedding_gpu
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    
    if verbose:
        print("\n📦 Creating gene activity AnnData...")
    
    gene_activity_matrix = np.nan_to_num(gene_activity_matrix, 0)
    np.clip(gene_activity_matrix, 0, None, out=gene_activity_matrix)
    
    gene_activity_sparse = sparse.csr_matrix(gene_activity_matrix, dtype=np.float64)
    gene_activity_sparse = fix_sparse_matrix_dtype(gene_activity_sparse, verbose=verbose)
    
    del gene_activity_matrix
    gc.collect()
    
    gene_activity_adata = ad.AnnData(
        X=gene_activity_sparse,
        obs=atac_obs.copy(),
        var=raw_rna_var.copy()
    )
    
    gene_activity_adata.obs['modality'] = 'ATAC'
    gene_activity_adata.layers['gene_activity'] = gene_activity_sparse.copy()
    
    for key, value in atac_obsm_dict.items():
        gene_activity_adata.obsm[key] = value
    
    for key, value in raw_rna_varm_dict.items():
        gene_activity_adata.varm[key] = value
    
    if verbose:
        print("\n📦 Creating RNA AnnData for merging...")
    
    if is_sparse_rna:
        rna_X = rna_X_full[common_cells_raw_indices, :]
        if sparse.issparse(rna_X):
            rna_X = rna_X.tocsr().astype(np.float64)
            rna_X = fix_sparse_matrix_dtype(rna_X, verbose=verbose)
        else:
            rna_X = np.asarray(rna_X).astype(np.float64)
            nnz = np.count_nonzero(rna_X)
            sparsity = 1 - (nnz / rna_X.size)
            if sparsity > 0.5:
                rna_X = sparse.csr_matrix(rna_X, dtype=np.float64)
                rna_X = fix_sparse_matrix_dtype(rna_X, verbose=verbose)
    else:
        rna_X = rna_X_full[common_cells_raw_indices, :].astype(np.float64)
        nnz = np.count_nonzero(rna_X)
        sparsity = 1 - (nnz / rna_X.size)
        if sparsity > 0.5:
            rna_X = sparse.csr_matrix(rna_X, dtype=np.float64)
            rna_X = fix_sparse_matrix_dtype(rna_X, verbose=verbose)
    
    del rna_X_full
    gc.collect()
    
    rna_for_merge = ad.AnnData(
        X=rna_X,
        obs=rna_obs.copy(),
        var=raw_rna_var.copy()
    )
    
    rna_for_merge.obs['modality'] = 'RNA'
    
    for key, value in rna_obsm_dict.items():
        rna_for_merge.obsm[key] = value
    
    for key, value in raw_rna_varm_dict.items():
        rna_for_merge.varm[key] = value
    
    if verbose:
        print("\n🔗 Merging RNA and ATAC datasets...")
    
    rna_indices = set(rna_for_merge.obs.index)
    atac_indices = set(gene_activity_adata.obs.index)
    overlap = rna_indices.intersection(atac_indices)
    
    if verbose and overlap:
        print(f"   Found {len(overlap)} overlapping indices, adding modality suffix...")
    
    rna_for_merge.obs['original_barcode'] = rna_for_merge.obs.index
    gene_activity_adata.obs['original_barcode'] = gene_activity_adata.obs.index
    
    rna_for_merge.obs.index = pd.Index([f"{idx}_RNA" for idx in rna_for_merge.obs.index])
    gene_activity_adata.obs.index = pd.Index([f"{idx}_ATAC" for idx in gene_activity_adata.obs.index])
    
    if verbose:
        print(f"   RNA cells: {rna_for_merge.n_obs}")
        print(f"   ATAC cells: {gene_activity_adata.n_obs}")
    
    merged_adata = ad.concat(
        [rna_for_merge, gene_activity_adata], 
        axis=0, 
        join='inner',
        merge='same',
        label=None,
        keys=None,
        index_unique=None
    )
    
    del rna_for_merge, gene_activity_adata
    gc.collect()
    
    if not merged_adata.obs.index.is_unique:
        if verbose:
            print("   ⚠️ Fixing non-unique indices...")
        merged_adata.obs_names_make_unique()
    
    if verbose:
        print("\n💾 Saving merged dataset...")
    
    if sparse.issparse(merged_adata.X):
        if not isinstance(merged_adata.X, sparse.csr_matrix):
            merged_adata.X = merged_adata.X.tocsr()
        merged_adata.X = fix_sparse_matrix_dtype(merged_adata.X, verbose=verbose)
        merged_adata.X.sort_indices()
        merged_adata.X.eliminate_zeros()
    
    output_dir_path = os.path.join(output_path, 'preprocess')
    os.makedirs(output_dir_path, exist_ok=True)
    output_path_anndata = os.path.join(output_dir_path, 'atac_rna_integrated.h5ad')
    safe_h5ad_write(merged_adata, output_path_anndata)
    
    if verbose:
        print(f"\n✅ Gene activity computation complete!")
        print(f"   Output: {output_path_anndata}")
        print(f"   Shape: {merged_adata.shape}")
        print(f"   RNA cells: {(merged_adata.obs['modality'] == 'RNA').sum()}")
        print(f"   ATAC cells: {(merged_adata.obs['modality'] == 'ATAC').sum()}")
        print(f"   Obs columns: {list(merged_adata.obs.columns)}")

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    gc.collect()
    
    return merged_adata

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
    run_visualization: bool = True,
    
    # Preprocessing parameters
    ensembl_release: int = 98,
    species: str = "homo_sapiens",
    use_highly_variable: bool = True,
    n_top_genes: int = 2000,
    n_pca_comps: int = 50,
    n_lsi_comps: int = 50,
    gtf_by: str = "gene_name",
    flavor: str = "seurat_v3",
    generate_umap: bool = False,
    compression: str = "gzip",
    random_state: int = 42,
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
    
    # Visualization parameters
    plot_columns: Optional[List[str]] = None,
    
    # Output directory
    output_dir: str = "./glue_results",
):
    """Complete GLUE pipeline that runs preprocessing, training, gene activity computation, and visualization.
    
    Use process flags to control which steps to run:
    - run_preprocessing: Run data preprocessing
    - run_training: Run model training
    - run_gene_activity: Compute gene activity
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
            additional_hvg_file=additional_hvg_file,
            ensembl_release=ensembl_release,
            species=species,
            output_dir=glue_output_dir,
            use_highly_variable=use_highly_variable,
            n_top_genes=n_top_genes,
            n_pca_comps=n_pca_comps,
            n_lsi_comps=n_lsi_comps,
            flavor=flavor,
            generate_umap=generate_umap,
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
            treat_sample_as_batch=treat_sample_as_batch,
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
    
    # Step 4: Visualization
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
    if run_gene_activity:
        return merged_adata
    else:
        return None