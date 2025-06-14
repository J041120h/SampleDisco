import anndata as ad
import networkx as nx
import scanpy as sc
import scglue
import numpy as np
from typing import Optional, Tuple
from pathlib import Path
import pyensembl
from linux.CellType_linux import *
from CellType import *
import time

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
    import pandas as pd
    import numpy as np
    
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
    import pandas as pd
    
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
    ensembl_release: int = 98,
    species: str = "homo_sapiens",
    output_dir: str = "./",
    use_highly_variable: bool = True,  # NEW PARAMETER
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
    import anndata as ad
    import scanpy as sc
    import scglue
    import pyensembl
    import networkx as nx
    from pathlib import Path
    from typing import Optional, Tuple
    
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
    else:
        print(f"   Using all {rna.n_vars} genes")
        # Mark all genes as highly variable for compatibility
        rna.var['highly_variable'] = True
        rna.var['highly_variable_rank'] = range(1, rna.n_vars + 1)
        rna.var['means'] = np.array(rna.X.mean(axis=0)).flatten()
        rna.var['dispersions'] = np.array(rna.X.var(axis=0)).flatten()
        rna.var['dispersions_norm'] = rna.var['dispersions'] / rna.var['means']
    
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
    import anndata as ad
    import networkx as nx
    import scglue
    import pandas as pd
    import scanpy as sc
    import os
    from itertools import chain
    
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
    scglue.models.configure_dataset(
        rna, "NB", use_highly_variable=use_highly_variable, 
        use_layer="counts", use_rep="X_pca", use_batch='sample'
    )
    scglue.models.configure_dataset(
        atac, "NB", use_highly_variable=use_highly_variable, 
        use_rep="X_lsi", use_batch='sample'
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
        {"rna": rna, "atac": atac}, guidance_hvf,
        fit_kws={"directory": train_dir}
    )
    
    # 5. Check integration consistency
    print(f"\n\n\n📊 Checking integration consistency...\n\n\n")
    consistency_scores = scglue.models.integration_consistency(
        glue, {"rna": rna, "atac": atac}, guidance_hvf
    )
    min_consistency = consistency_scores['consistency'].min()
    mean_consistency = consistency_scores['consistency'].mean()
    print(f"\n\n\nConsistency scores - Min: {min_consistency:.4f}, Mean: {mean_consistency:.4f}\n\n\n")
    
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

def compute_gene_activity_from_knn_with_celltype(
    glue_dir: str,
    output_path: str,
    raw_rna_path: str,  # New parameter for raw RNA counts
    k_neighbors: int = 50,
    use_rep: str = "X_glue",
    metric: str = "cosine",
    use_gpu: bool = True,
    verbose: bool = True,
    # Cell type assignment parameters
    existing_cell_types: bool = False,
    n_target_clusters: int = 3,
    cluster_resolution: float = 0.8,
    use_rep_celltype: str = "X_glue",
    markers: list = None,
    method: str = 'average',
    metric_celltype: str = 'euclidean',
    distance_mode: str = 'centroid',
    num_PCs: int = 20,
    generate_umap: bool = True,
) -> ad.AnnData:
    """
    Compute gene activity for ATAC cells using weighted k-nearest neighbors from RNA cells,
    validate and correct the computed counts, merge with RNA data, assign cell types, 
    and generate UMAP visualizations.
    
    This enhanced function:
    1. Uses raw RNA counts for gene activity computation and final merging
    2. Preserves embeddings and metadata from processed RNA file
    3. Computes gene activity for ATAC cells using k-NN from RNA cells with cosine similarity weights
    4. Validates and corrects the computed gene activity counts (NaN, Inf, negatives)
    5. Merges the gene activity data with original raw RNA data
    6. Assigns cell types using the provided cell_types_linux function
    7. Generates UMAP visualizations
    8. Saves the final merged dataset
    
    Parameters:
    -----------
    raw_rna_path : str
        Path to the raw RNA count matrix (not log-transformed or normalized)
    
    Returns:
    --------
    merged_adata : ad.AnnData
        Merged AnnData object with validated gene activity, raw RNA data, cell types, and visualizations
    """
    import anndata as ad
    import numpy as np
    import pandas as pd
    import time
    from pathlib import Path
    import os
    import scanpy as sc
    import matplotlib.pyplot as plt
    from sklearn.metrics.pairwise import cosine_similarity
    
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
    
    gpu_available = False
    if use_gpu:
        try:
            import cupy as cp
            from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
            gpu_available = True
            if verbose:
                print("🚀 GPU acceleration enabled (cuML/CuPy detected)")
        except ImportError:
            if verbose:
                print("⚠️  GPU libraries not found, falling back to CPU")
            from sklearn.neighbors import NearestNeighbors
            from sklearn.preprocessing import normalize
    else:
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import normalize

    if verbose:
        print(f"\n🧬 Computing gene activity using raw RNA counts and merging data...")
        print(f"   k_neighbors: {k_neighbors}")
        print(f"   metric: {metric}")
        print(f"   weight method: cosine similarity")
    
    # Load processed RNA (for embeddings and metadata)
    rna_processed = ad.read_h5ad(rna_processed_path)
    # Load raw RNA counts
    rna_raw = ad.read_h5ad(raw_rna_path)
    # Load ATAC
    atac = ad.read_h5ad(atac_path)
    
    if verbose:
        print(f"   Processed RNA shape: {rna_processed.shape}")
        print(f"   Raw RNA shape: {rna_raw.shape}")
        print(f"   ATAC shape: {atac.shape}")
    
    # Verify cell matching between processed and raw RNA
    if not rna_processed.obs.index.equals(rna_raw.obs.index):
        if verbose:
            print("   ⚠️  Cell indices don't match between processed and raw RNA, aligning...")
        # Align the datasets by cell indices
        common_cells = rna_processed.obs.index.intersection(rna_raw.obs.index)
        if len(common_cells) == 0:
            raise ValueError("No common cells found between processed and raw RNA data")
        rna_processed = rna_processed[common_cells].copy()
        rna_raw = rna_raw[common_cells].copy()
        if verbose:
            print(f"   Aligned to {len(common_cells)} common cells")
    
    # Verify gene matching
    if not rna_processed.var.index.equals(rna_raw.var.index):
        if verbose:
            print("   ⚠️  Gene indices don't match between processed and raw RNA, aligning...")
        # Align the datasets by gene indices
        common_genes = rna_processed.var.index.intersection(rna_raw.var.index)
        if len(common_genes) == 0:
            raise ValueError("No common genes found between processed and raw RNA data")
        rna_processed = rna_processed[:, common_genes].copy()
        rna_raw = rna_raw[:, common_genes].copy()
        if verbose:
            print(f"   Aligned to {len(common_genes)} common genes")
    
    # Check if embeddings exist
    if use_rep not in rna_processed.obsm:
        raise ValueError(f"Embedding '{use_rep}' not found in processed RNA data. Available: {list(rna_processed.obsm.keys())}")
    if use_rep not in atac.obsm:
        raise ValueError(f"Embedding '{use_rep}' not found in ATAC data. Available: {list(atac.obsm.keys())}")
    
    # Get embeddings from processed data
    rna_embedding = rna_processed.obsm[use_rep]
    atac_embedding = atac.obsm[use_rep]
    
    # Get raw expression data for gene activity computation
    rna_raw_expression = rna_raw.X
    
    # Ensure we're working with dense arrays
    if hasattr(rna_raw_expression, 'toarray'):
        rna_raw_expression = rna_raw_expression.toarray()
    
    # Check for negative values in original RNA expression
    rna_neg_count = np.sum(rna_raw_expression < 0)
    if verbose and rna_neg_count > 0:
        print(f"   ⚠️  Found {rna_neg_count:,} negative values in raw RNA expression data")
        print(f"   This may contribute to negative gene activity values")
    
    # Find k-nearest neighbors
    if verbose:
        print("🔍 Finding k-nearest RNA neighbors for each ATAC cell...")
        start_time = time.time()
    
    if gpu_available:
        # GPU implementation
        rna_embedding_gpu = cp.asarray(rna_embedding)
        atac_embedding_gpu = cp.asarray(atac_embedding)
        
        nn = cuNearestNeighbors(n_neighbors=k_neighbors, metric=metric)
        nn.fit(rna_embedding_gpu)
        distances_gpu, indices_gpu = nn.kneighbors(atac_embedding_gpu)
        
        indices = cp.asnumpy(indices_gpu)
        distances = cp.asnumpy(distances_gpu)
        
    else:
        # CPU implementation
        nn = NearestNeighbors(n_neighbors=k_neighbors, metric=metric)
        nn.fit(rna_embedding)
        distances, indices = nn.kneighbors(atac_embedding)
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"   k-NN search completed in {elapsed:.2f} seconds\n")
    
    # Compute cosine similarity weights
    if verbose:
        print("📐 Computing cosine similarity weights...")
        start_time = time.time()
    
    # Convert cosine distances to similarities and then to weights
    if metric == 'cosine':
        # For cosine distance: similarity = 1 - distance
        similarities = 1 - distances
    else:
        # If not using cosine metric for k-NN, compute cosine similarity manually
        similarities = np.zeros_like(distances)
        for i in range(atac_embedding.shape[0]):
            neighbor_indices = indices[i]
            atac_vec = atac_embedding[i:i+1]  # Keep 2D shape
            rna_neighbors = rna_embedding[neighbor_indices]
            # Compute cosine similarity
            sim_scores = cosine_similarity(atac_vec, rna_neighbors)[0]
            similarities[i] = sim_scores
    
    # Ensure similarities are non-negative (cosine similarity ranges from -1 to 1)
    similarities = np.clip(similarities, 0, 1)
    
    # Normalize similarities to create weights (sum to 1 for each ATAC cell)
    weights = similarities / (similarities.sum(axis=1, keepdims=True) + 1e-8)  # Add small epsilon to avoid division by zero
    
    if verbose:
        elapsed = time.time() - start_time
        avg_similarity = np.mean(similarities)
        print(f"   Cosine similarity computation completed in {elapsed:.2f} seconds")
        print(f"   Average cosine similarity: {avg_similarity:.4f}\n")
    
    # Compute weighted gene activity
    if verbose:
        print("🧮 Computing weighted gene activity using raw RNA counts...")
        print(f"   Computing activity for {rna_raw.n_vars} genes across {atac.n_obs} ATAC cells")
        start_time = time.time()
    
    if gpu_available:
        # GPU-accelerated computation
        rna_expression_gpu = cp.asarray(rna_raw_expression, dtype=cp.float32)
        weights_gpu = cp.asarray(weights, dtype=cp.float32)
        
        gene_activity_gpu = cp.zeros((atac.n_obs, rna_raw.n_vars), dtype=cp.float32)
        
        batch_size = 5000
        n_batches = (atac.n_obs + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, atac.n_obs)
            
            batch_indices = indices[start_idx:end_idx]
            batch_weights = weights_gpu[start_idx:end_idx]
            
            for i in range(end_idx - start_idx):
                cell_indices = batch_indices[i]
                neighbor_expression = rna_expression_gpu[cell_indices]
                gene_activity_gpu[start_idx + i] = cp.sum(
                    neighbor_expression * batch_weights[i][:, cp.newaxis], axis=0
                )
            
            if verbose and (batch_idx + 1) % 5 == 0:
                progress = (batch_idx + 1) / n_batches * 100
                print(f"   Progress: {progress:.1f}% ({batch_idx + 1}/{n_batches} batches)")
        
        gene_activity_matrix = cp.asnumpy(gene_activity_gpu)
        
    else:
        # CPU implementation
        gene_activity_matrix = np.zeros((atac.n_obs, rna_raw.n_vars), dtype=np.float32)
        
        batch_size = 1000
        n_batches = (atac.n_obs + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, atac.n_obs)
            
            batch_indices = indices[start_idx:end_idx]
            batch_weights = weights[start_idx:end_idx]
            
            for i, (cell_indices, cell_weights) in enumerate(zip(batch_indices, batch_weights)):
                neighbor_expression = rna_raw_expression[cell_indices]
                gene_activity_matrix[start_idx + i] = np.sum(
                    neighbor_expression * cell_weights[:, np.newaxis], axis=0
                )
            
            if verbose and (batch_idx + 1) % 10 == 0:
                progress = (batch_idx + 1) / n_batches * 100
                print(f"   Progress: {progress:.1f}% ({batch_idx + 1}/{n_batches} batches)")
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"   Gene activity computation completed in {elapsed:.2f} seconds\n")
    
    # Data validation and correction
    if verbose:
        print("🔍 Validating gene activity counts...")
        
    # Check for problematic values
    n_nan = np.sum(np.isnan(gene_activity_matrix))
    n_inf = np.sum(np.isinf(gene_activity_matrix))
    n_neg = np.sum(gene_activity_matrix < 0)
    
    if verbose and (n_nan > 0 or n_inf > 0 or n_neg > 0):
        print(f"   Found: {n_nan:,} NaN, {n_inf:,} Inf, {n_neg:,} negative values")
    
    # Fix NaN values (set to 0)
    if n_nan > 0:
        gene_activity_matrix[np.isnan(gene_activity_matrix)] = 0
        if verbose:
            print(f"   ✓ Fixed {n_nan:,} NaN values → 0")
    
    # Fix Inf values (set to maximum finite value or 0)
    if n_inf > 0:
        finite_mask = np.isfinite(gene_activity_matrix)
        if np.any(finite_mask):
            max_finite = np.max(gene_activity_matrix[finite_mask])
            gene_activity_matrix[np.isinf(gene_activity_matrix)] = max_finite
        else:
            gene_activity_matrix[np.isinf(gene_activity_matrix)] = 0
        if verbose:
            print(f"   ✓ Fixed {n_inf:,} Inf values")
    
    # Fix negative values (set to 0)
    if n_neg > 0:
        gene_activity_matrix[gene_activity_matrix < 0] = 0
        if verbose:
            print(f"   ✓ Fixed {n_neg:,} negative values → 0")
    
    if verbose:
        final_range = f"[{np.min(gene_activity_matrix):.3f}, {np.max(gene_activity_matrix):.3f}]"
        print(f"   Final count range: {final_range}\n")
    
    # Create gene activity AnnData object using metadata from processed RNA
    if verbose:
        print("📦 Creating gene activity AnnData object...")
    
    gene_activity = ad.AnnData(
        X=gene_activity_matrix,
        obs=atac.obs.copy(),
        var=rna_raw.var.copy()  # Use raw RNA var info
    )
    
    # Copy embeddings from ATAC data
    if use_rep in atac.obsm:
        gene_activity.obsm[use_rep] = atac.obsm[use_rep].copy()
    if 'X_umap' in atac.obsm:
        gene_activity.obsm['X_umap'] = atac.obsm['X_umap'].copy()
    
    gene_activity.layers['gene_activity'] = gene_activity.X.copy()
    
    # Add modality labels
    gene_activity.obs['modality'] = 'ATAC'
    rna_raw.obs['modality'] = 'RNA'
    
    # Copy embeddings and other metadata from processed RNA to raw RNA
    if use_rep in rna_processed.obsm:
        rna_raw.obsm[use_rep] = rna_processed.obsm[use_rep].copy()
    if 'X_umap' in rna_processed.obsm:
        rna_raw.obsm['X_umap'] = rna_processed.obsm['X_umap'].copy()
    
    # Copy any additional metadata from processed RNA
    for key in rna_processed.obs.columns:
        if key not in rna_raw.obs.columns:
            rna_raw.obs[key] = rna_processed.obs[key]
    
    # Merge with raw RNA data
    if verbose:
        print("🔗 Merging gene activity data with raw RNA data...")
    
    # Concatenate the datasets
    merged_adata = ad.concat([rna_raw, gene_activity], axis=0, join='outer', merge='same')
    
    # Copy embeddings to merged object
    merged_embeddings = np.vstack([rna_raw.obsm[use_rep], gene_activity.obsm[use_rep]])
    merged_adata.obsm[use_rep] = merged_embeddings
    
    if verbose:
        print(f"   Merged shape: {merged_adata.shape}")
        print(f"   RNA cells: {(merged_adata.obs['modality'] == 'RNA').sum()}")
        print(f"   ATAC cells: {(merged_adata.obs['modality'] == 'ATAC').sum()}\n")
    
    # Assign cell types using the cell_types_linux function
    if verbose:
        print("🏷️  Assigning cell types...")
    
    # Create output directory for intermediate files
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Apply cell type assignment
    if gpu_available:
        merged_adata = cell_types_linux(
            merged_adata,
            cell_column='cell_type',
            existing_cell_types=existing_cell_types,
            n_target_clusters=n_target_clusters,
            umap=False,  # We'll handle UMAP separately
            Save=False,  # We'll save the final result
            output_dir=str(output_dir),
            cluster_resolution=cluster_resolution,
            use_rep=use_rep_celltype,
            markers=markers,
            method=method,
            metric=metric_celltype,
            distance_mode=distance_mode,
            num_PCs=num_PCs,
            verbose=verbose
        )
    else:
        merged_adata = cell_types(
            merged_adata,
            cell_column='cell_type',
            existing_cell_types=existing_cell_types,
            n_target_clusters=n_target_clusters,
            umap=False,  # We'll handle UMAP separately
            Save=False,  # We'll save the final result
            output_dir=str(output_dir),
            cluster_resolution=cluster_resolution,
            use_rep=use_rep_celltype,
            markers=markers,
            method=method,
            metric=metric_celltype,
            distance_mode=distance_mode,
            num_PCs=num_PCs,
            verbose=verbose
        )
    
    # Clean the object for savingd
    merged_adata = clean_anndata_for_saving(merged_adata, verbose=False)
    merged_adata.write(os.path.join(output_path, "atac_rna_integrated.h5ad"), compression='gzip')
    
    if verbose:
        print(f"✅ Gene activity computation and merging complete!")
        print(f"\n📊 Final Summary:")
        print(f"   Merged dataset shape: {merged_adata.shape}")
        print(f"   Total cells: {merged_adata.n_obs}")
        print(f"   RNA cells: {(merged_adata.obs['modality'] == 'RNA').sum()}")
        print(f"   ATAC cells: {(merged_adata.obs['modality'] == 'ATAC').sum()}")
        print(f"   Genes: {merged_adata.n_vars}")
        print(f"   Cell types identified: {merged_adata.obs['cell_type'].nunique()}")
        print(f"   GPU acceleration: {'Yes' if gpu_available else 'No'}")
        print(f"   Data corrections: {n_nan + n_inf + n_neg} total fixes applied")
        print(f"   Weight method: Cosine similarity")
        print(f"   Expression data: Raw RNA counts")
        print(f"   UMAP visualizations: {'Generated' if generate_umap else 'Skipped'}")
    
    return merged_adata

import os
import sys
import argparse
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
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
    
import time
import os
from typing import Optional, List

def glue(
    # Data files
    rna_file: str,
    atac_file: str,
    rna_sample_meta_file: Optional[str] = None,
    atac_sample_meta_file: Optional[str] = None,
    
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
    save_prefix: str = "glue",
    
    # Gene activity computation parameters
    k_neighbors: int = 50,
    use_rep: str = "X_glue",
    metric: str = "cosine",
    use_gpu: bool = True,
    verbose: bool = True,
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
    """Complete GLUE pipeline that runs preprocessing, training, gene activity computation, and visualization."""
    
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, "integration", "glue")
    start_time = time.time()
    
    # # Step 1: Preprocessing
    # rna, atac, guidance = glue_preprocess_pipeline(
    #     rna_file=rna_file,
    #     atac_file=atac_file,
    #     rna_sample_meta_file=rna_sample_meta_file,
    #     atac_sample_meta_file=atac_sample_meta_file,
    #     ensembl_release=ensembl_release,
    #     species=species,
    #     output_dir=output_dir,
    #     use_highly_variable=use_highly_variable,
    #     n_top_genes=n_top_genes,
    #     n_pca_comps=n_pca_comps,
    #     n_lsi_comps=n_lsi_comps,
    #     lsi_n_iter=lsi_n_iter,
    #     gtf_by=gtf_by,
    #     flavor=flavor,
    #     generate_umap=generate_umap,
    #     compression=compression,
    #     random_state=random_state,
    #     metadata_sep=metadata_sep,
    #     rna_sample_column=rna_sample_column,
    #     atac_sample_column=atac_sample_column
    # )
    
    # # Step 2: Training
    # glue_train(
    #     preprocess_output_dir=output_dir,
    #     save_prefix=save_prefix,
    #     consistency_threshold=consistency_threshold,
    #     use_highly_variable=use_highly_variable,
    #     output_dir=output_dir
    # )
    
    # Step 3: Memory management and gene activity computation
    if use_gpu:
        try:
            import rmm
            from rmm.allocators.cupy import rmm_cupy_allocator
            import cupy as cp
            
            rmm.reinitialize(
                managed_memory=True,
                pool_allocator=False,
            )
            cp.cuda.set_allocator(rmm_cupy_allocator)
        except:
            pass
    
    compute_gene_activity_from_knn_with_celltype(
        glue_dir=output_dir,
        output_path=output_dir,
        raw_rna_path = rna_file,
        k_neighbors=k_neighbors,
        use_rep=use_rep,
        metric=metric,
        use_gpu=use_gpu,
        verbose=verbose,
        existing_cell_types=existing_cell_types,
        n_target_clusters=n_target_clusters,
        cluster_resolution=cluster_resolution,
        use_rep_celltype=use_rep_celltype,
        markers=markers,
        method=method,
        metric_celltype=metric_celltype,
        distance_mode=distance_mode,
        num_PCs=n_lsi_comps,
        generate_umap=generate_umap_celltype
    )
    
    # Step 4: Visualization
    integrated_file = os.path.join(output_dir, "atac_rna_integrated.h5ad")
    glue_visualize(
        integrated_path=integrated_file,
        output_dir=output_dir,
        plot_columns=plot_columns
    )
    
    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    print(f"\nTotal runtime: {elapsed_minutes:.2f} minutes")

if __name__ == "__main__":
    glue(
    # Data files
    rna_file = "/dcl01/hongkai/data/data/hjiang/Data/count_data.h5ad",
    atac_file = "/dcl01/hongkai/data/data/hjiang/Data/ATAC.h5ad",
    rna_sample_meta_file = "/dcl01/hongkai/data/data/hjiang/Data/sample_data.csv",
    atac_sample_meta_file= "/dcl01/hongkai/data/data/hjiang/Data/ATAC_Metadata.csv",
    output_dir = "/dcl01/hongkai/data/data/hjiang/result",
)
