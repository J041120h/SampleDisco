# Core data manipulation and scientific computing
import pandas as pd
import numpy as np
import anndata as ad

# System and file operations
import os
import time
from pathlib import Path
import contextlib
import io

# Single-cell analysis
import scanpy as sc

# Machine learning and similarity metrics
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# Visualization
import matplotlib.pyplot as plt

# GPU acceleration (optional - will be imported conditionally in the code)
try:
    import cupy as cp
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("GPU libraries (cupy/cuml) not available, will use CPU implementations")

# Custom modules (these appear to be user-defined modules)
# Note: These imports will only work if these modules exist in your environment
try:
    from pseudo_adata import *
    from DR import *
    from Cell_type import *
except ImportError as e:
    print(f"Custom module import failed: {e}")
    print("Make sure pseudo_adata.py, DR.py, and Cell_type.py are in your Python path")

# Additional packages that may be needed for full functionality
# (These are used implicitly through scanpy or other functions)
try:
    import scrublet  # For doublet detection (used via scanpy)
except ImportError:
    print("scrublet not available - doublet detection may not work")

try:
    import harmonypy  # For Harmony integration (if using Harmony)
except ImportError:
    print("harmonypy not available - Harmony integration may not work")


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
    Compute gene activity for ATAC cells using weighted k-nearest neighbors from RNA cells,
    validate and correct the computed counts, and merge with RNA data.
    
    This function:
    1. Uses raw RNA counts for gene activity computation and final merging
    2. Preserves embeddings and metadata from processed RNA file
    3. Computes gene activity for ATAC cells using k-NN from RNA cells with cosine similarity weights
    4. Validates and corrects the computed gene activity counts (NaN, Inf, negatives)
    5. Merges the gene activity data with original raw RNA data
    6. Saves the merged dataset (without cell type assignment)
    
    Parameters:
    -----------
    glue_dir : str
        Directory containing GLUE results
    output_path : str
        Path to save the merged dataset
    raw_rna_path : str
        Path to the raw RNA count matrix (not log-transformed or normalized)
    k_neighbors : int
        Number of nearest neighbors to use
    use_rep : str
        Representation to use for neighbor finding
    metric : str
        Distance metric for neighbor finding
    use_gpu : bool
        Whether to use GPU acceleration
    verbose : bool
        Whether to print progress information
    
    Returns:
    --------
    merged_adata : ad.AnnData
        Merged AnnData object with validated gene activity and raw RNA data
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
    
    # Clean the object for saving
    merged_adata = clean_anndata_for_saving(merged_adata, verbose=False)

    output_dir = os.path.join(output_path, 'preprocess')
    os.makedirs(output_dir, exist_ok=True)
    output_path_anndata = os.path.join(output_dir, 'atac_rna_integrated.h5ad')

    merged_adata.write(output_path_anndata, compression='gzip')

    if verbose:
        print(f"✅ Gene activity computation and merging complete!")
        print(f"\n📊 Summary:")
        print(f"   Output path: {output_path_anndata}")
        print(f"   Merged dataset shape: {merged_adata.shape}")
        print(f"   Total cells: {merged_adata.n_obs}")
        print(f"   RNA cells: {(merged_adata.obs['modality'] == 'RNA').sum()}")
        print(f"   ATAC cells: {(merged_adata.obs['modality'] == 'ATAC').sum()}")
        print(f"   Genes: {merged_adata.n_vars}")
        print(f"   GPU acceleration: {'Yes' if gpu_available else 'No'}")
        print(f"   Data corrections: {n_nan + n_inf + n_neg} total fixes applied")
        print(f"   Weight method: Cosine similarity")
        print(f"   Expression data: Raw RNA counts")
    
    return merged_adata

import os
import scanpy as sc
import time
import contextlib
import io

def integrate_preprocess(
    output_dir,
    h5ad_path = None,
    sample_column = 'sample',
    modality_col = 'modality',
    min_cells_sample=1,
    min_cell_gene=10,
    min_features=500,
    pct_mito_cutoff=20,
    exclude_genes=None,
    doublet=True,
    verbose=True
):
    """
    Harmony Integration with proportional HVG selection by cell type,
    now reading an existing H5AD file that only contains raw counts (no meta).

    This function:
      1. Reads and preprocesses the data (filter genes/cells, remove MT genes, etc.).
      2. Splits into two branches for:
         (a) adata_cluster used for clustering with Harmony
         (b) adata_sample_diff used for sample-level analysis (minimal batch correction).
      3. Returns both AnnData objects.
    """
    # Start timing
    start_time = time.time()

    if h5ad_path == None:
        h5ad_path = os.path.join(output_dir, 'glue/atac_rna_integrated.h5ad')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print("Automatically generating output_dir")
    output_dir = os.path.join(output_dir, 'preprocess')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print("Automatically generating preprocess subdirectory")

    if doublet and min_cells_sample < 30:
        min_cells_sample = 30
        print("Minimum dimension requested by scrublet is 30, raise sample standard accordingly")
    
    # 1. Read the raw count data from an existing H5AD
    if verbose:
        print('=== Read input dataset ===')
    adata = sc.read_h5ad(h5ad_path)
    if verbose:
        print(f'Dimension of raw data (cells x genes): {adata.shape[0]} x {adata.shape[1]}')

    # Modify sample IDs by adding modality information
    if modality_col is not None and modality_col in adata.obs.columns:
        adata.obs[sample_column] = adata.obs[sample_column].astype(str) + '_' + adata.obs[modality_col].astype(str)
        if verbose:
            print(f"Modified sample IDs by adding modality information from '{modality_col}' column")
    adata.var_names_make_unique()
    adata.var = adata.var.dropna(axis=1, how="all")
    sc.pp.calculate_qc_metrics(
        adata, 
        qc_vars=["MT"],  # mitochondrial genes if annotated
        log1p=False, 
        inplace=True
    )
    sc.pp.filter_cells(adata, min_genes=min_features)
    if verbose:
        print(f"After cell filtering -- Cells remaining: {adata.n_obs}, Genes remaining: {adata.n_vars}")
    sc.pp.filter_genes(adata, min_cells=min_cell_gene)
    if verbose:
        print(f"After gene filtering -- Cells remaining: {adata.n_obs}, Genes remaining: {adata.n_vars}")

    # Mito QC
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs['pct_counts_mt'] < pct_mito_cutoff].copy()

    # Exclude genes if needed
    mt_genes = adata.var_names[adata.var_names.str.startswith('MT-')]
    if exclude_genes is not None:
        genes_to_exclude = set(exclude_genes) | set(mt_genes)
    else:
        genes_to_exclude = set(mt_genes)
    adata = adata[:, ~adata.var_names.isin(genes_to_exclude)].copy()
    if verbose:
        print(f"After remove MT_gene and user input cell -- Cells remaining: {adata.n_obs}, Genes remaining: {adata.n_vars}")

    cell_counts_per_patient = adata.obs.groupby(sample_column).size()
    if verbose:
        print("Sample counts BEFORE filtering:")
        print(cell_counts_per_patient.sort_values(ascending=False))
    patients_to_keep = cell_counts_per_patient[cell_counts_per_patient >= min_cells_sample].index
    if verbose:
        print(f"\nSamples retained (>= {min_cells_sample} cells): {list(patients_to_keep)}")
    adata = adata[adata.obs[sample_column].isin(patients_to_keep)].copy()
    cell_counts_after = adata.obs[sample_column].value_counts()
    if verbose:
        print("\nSample counts AFTER filtering:")
        print(cell_counts_after.sort_values(ascending=False))

    # Drop genes that are too rare in these final cells
    min_cells_for_gene = int(0.01 * adata.n_obs)
    sc.pp.filter_genes(adata, min_cells=min_cells_for_gene)
    if verbose:
        print(f"Final filtering -- Cells remaining: {adata.n_obs}, Genes remaining: {adata.n_vars}")

    # Optional doublet detection
    if doublet:
        if verbose:
            print(f"Running doublet detection with scrublet on {adata.n_obs} cells...")
        
        try:
            # Store original cell count for comparison
            original_n_cells = adata.n_obs
            
            # Create a copy for scrublet to avoid modifying original
            adata_scrub = adata.copy()
            
            # Run scrublet with suppressed output
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                sc.pp.scrublet(adata_scrub, batch_key=sample_column)
            
            # Check if scrublet results match our data
            if 'predicted_doublet' not in adata_scrub.obs.columns:
                if verbose:
                    print("Warning: Scrublet did not add 'predicted_doublet' column. Skipping doublet removal.")
            elif adata_scrub.n_obs != original_n_cells:
                if verbose:
                    print(f"Warning: Scrublet changed cell count from {original_n_cells} to {adata_scrub.n_obs}. Using original data without doublet removal.")
            else:
                # Successfully ran scrublet, now filter doublets
                n_doublets = adata_scrub.obs['predicted_doublet'].sum()
                if verbose:
                    print(f"Detected {n_doublets} doublets out of {original_n_cells} cells")
                
                # Copy the scrublet results back to original adata
                adata.obs['predicted_doublet'] = adata_scrub.obs['predicted_doublet']
                adata.obs['doublet_score'] = adata_scrub.obs.get('doublet_score', 0)
                
                # Filter out doublets
                adata = adata[~adata.obs['predicted_doublet']].copy()
                
                if verbose:
                    print(f"After doublet removal: {adata.n_obs} cells remaining")
        
        except Exception as e:
            if verbose:
                print(f"Warning: Scrublet failed with error: {str(e)}")
                print("Continuing without doublet detection...")
            # Continue without doublet detection

    fill_obs_nan_with_unknown(adata)
    adata.raw = adata.copy()
    if verbose:
        print("Preprocessing complete!")

    # Save to new file instead of overwriting original
    output_h5ad_path = os.path.join(output_dir, 'adata_sample.h5ad')
    sc.write(output_h5ad_path, adata)
    if verbose:
        print(f"Preprocessed data saved to: {output_h5ad_path}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print execution time
    if verbose:
        print(f"Function execution time: {elapsed_time:.2f} seconds")

    return adata

import pandas as pd
import scanpy as sc

def fill_obs_nan_with_unknown(
    adata: sc.AnnData,
    fill_value: str = "unKnown",
    verbose: bool = False,
) -> None:
    """
    Replace NaN values in all .obs columns with `fill_value`.
    Works transparently for Categorical, string/object, numeric, or mixed types.
    Operates in-place on `adata`.
    """
    for col in adata.obs.columns:
        ser = adata.obs[col]

        # Skip if the column has no missing values
        if not ser.isnull().any():
            continue

        # --- Handle categoricals ------------------------------------------------
        if pd.api.types.is_categorical_dtype(ser):
            if fill_value not in ser.cat.categories:
                # add the new category then continue using categorical dtype
                ser = ser.cat.add_categories([fill_value])
            ser = ser.fillna(fill_value)

        # --- Handle everything else (string, numeric, mixed) --------------------
        else:
            # Cast to object first if it's numeric; keeps mixed dtypes safe
            if pd.api.types.is_numeric_dtype(ser):
                ser = ser.astype("object")
            ser = ser.fillna(fill_value)

        # Write back to AnnData
        adata.obs[col] = ser

        if verbose:
            print(f"✓ Filled NaNs in .obs['{col}'] with '{fill_value}'")


if __name__ == "__main__":
    compute_gene_activity_from_knn(
        glue_dir = '/dcs07/hongkai/data/harry/result/Benchmark/multiomics/integration/glue',
        output_path = '/dcs07/hongkai/data/harry/result/Benchmark/multiomics',
        raw_rna_path = '/dcl01/hongkai/data/data/hjiang/Data/paired/rna/all.h5ad'
    )

    integrate_preprocess(
        output_dir = '/dcs07/hongkai/data/harry/result/Benchmark/multiomics',
        h5ad_path = '/dcs07/hongkai/data/harry/result/Benchmark/multiomics/preprocess/atac_rna_integrated.h5ad'
    )