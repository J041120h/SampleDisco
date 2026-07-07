# Standard library
import os
import sys
import gc
import json
import time
import shutil as _shutil
from pathlib import Path
from typing import Optional, Tuple, List
from itertools import chain
from concurrent.futures import ThreadPoolExecutor


# ---------------------------------------------------------------------------
# Auto-resolve bedtools BEFORE importing scglue (which imports pybedtools at
# module load and caches each legacy-binary's availability — once cached, no
# late PATH change will recover sortBed/intersectBed/etc.).
#
# When the user invokes Python directly without ``conda activate``, the env's
# bin/ is absent from PATH; scglue would then raise a misleading "exited with
# code 127" or "sortBed does not appear to be installed".
# ---------------------------------------------------------------------------
def _resolve_bedtools_dir() -> str:
    """Directory containing the bedtools binary, or '' if not found."""
    found = _shutil.which("bedtools")
    if found:
        return os.path.dirname(found)
    candidate = os.path.join(os.path.dirname(sys.executable), "bedtools")
    return os.path.dirname(candidate) if os.path.exists(candidate) else ""


_bin_dir = _resolve_bedtools_dir()
if _bin_dir:
    if _bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = _bin_dir + os.pathsep + os.environ.get("PATH", "")
        print(f"[multi_omics_glue] prepended {_bin_dir} to PATH for bedtools")
else:
    print("[multi_omics_glue] WARNING: bedtools not found on PATH or alongside "
          "python; rna_anchored_guidance_graph will fail.")


# Third-party
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import networkx as nx
import scglue  # pulls in pybedtools — PATH must already be correct above
import pyensembl
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
import psutil

# Local/project
from sampledisco.utils.safe_save import safe_h5ad_write
from sampledisco.utils.merge_sample_meta import merge_sample_metadata
from sampledisco.visualization.multi_omics_visualization import glue_visualize

def _peak_hvf_with_retries(adata_sub: ad.AnnData, n_top: int) -> pd.Index:
    """seurat_v3 HVG on raw-count ATAC subset with LOESS span retries.

    Mirrors rna_preprocess_cpu._hvg_with_retries — tries spans [0.3, 0.5,
    0.8, 1.0] and catches the reciprocal-condition-number LOESS failure.
    Returns the index of selected peak names (n_top or fewer).
    """
    spans = [0.3, 0.5, 0.8, 1.0]
    last_err = None
    for span in spans:
        try:
            sc.pp.highly_variable_genes(
                adata_sub, n_top_genes=n_top, flavor="seurat_v3", span=span,
            )
            return adata_sub.var_names[adata_sub.var["highly_variable"]]
        except ValueError as exc:
            arg = exc.args[0] if exc.args else ""
            msg = arg.decode("utf-8", "ignore") if isinstance(arg, bytes) else str(arg)
            if "reciprocal condition number" not in msg:
                raise
            last_err = exc
    raise last_err if last_err else RuntimeError("Peak HVF selection failed")


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
    n_top_peaks: int = 50000,
    atac_min_cells_floor: int = 10,
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

    # Release the dense scaled matrix: PCA result is in obsm['X_pca'] and
    # GLUE reads raw counts from layers['counts'] (set at line 176). Keeping
    # rna.X dense would carry ~50 GB unnecessarily into guidance-graph
    # construction below (the documented OOM hotspot on large RNA datasets).
    rna.X = rna.layers["counts"]
    gc.collect()
    print(f"   Freed dense rna.X (restored to sparse counts from layers['counts'])")

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
        peak_counts = np.array(atac.X.sum(axis=0)).flatten()
        peak_cells = np.array((atac.X > 0).sum(axis=0)).flatten()
        atac.var['n_cells'] = peak_cells
        atac.var['n_counts'] = peak_counts

        # This highly_variable flag drives BOTH the LSI basis (scglue.data.lsi at line ~300
        # auto-uses it) AND the scGLUE training subgraph (~line 630). Switching
        # coverage->variability improves both consistently.
        effective_floor = max(atac_min_cells_floor, int(np.ceil(0.001 * atac.n_obs)))
        passes_floor = peak_cells >= effective_floor
        n_eligible = int(passes_floor.sum())

        atac.var['highly_variable'] = False

        if n_eligible <= n_lsi_comps:
            # Degenerate: too few peaks pass floor — use all floor-passing peaks.
            selected_names = atac.var_names[passes_floor] if n_eligible > 0 else atac.var_names
        else:
            n_top = max(1, min(n_top_peaks, n_eligible))
            # seurat_v3 needs raw counts — atac.X is raw at this point (loaded at
            # ~line 141; lsi at ~line 300 does its own internal TF-IDF without
            # mutating .X).
            adata_sub = atac[:, passes_floor].copy()
            selected_names = _peak_hvf_with_retries(adata_sub, n_top)

        atac.var.loc[selected_names, 'highly_variable'] = True
        print(f"   Selected {int(atac.var['highly_variable'].sum())} VARIABLE peaks "
              f"(floor: open in >= {effective_floor} cells; "
              f"{n_eligible} eligible of {atac.n_vars})")
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

def _resolve_glue_batch_design(rna, atac, *, treat_sample_as_batch, batch_key, sample_key):
    """Decide the single ``use_batch`` column to pass to scGLUE.

    Each sample belongs to exactly one batch (single-cell biology
    invariant), so ``sample`` is a strict refinement of ``batch`` —
    removing per-sample variance with scGLUE already removes batch
    variance implicitly. Therefore ``treat_sample_as_batch=True`` just
    uses ``sample_key`` as use_batch; no synthetic combined column is
    needed (and adding one would inflate scGLUE's per-batch internal
    layers without giving the model any new information).
    """
    def _has(adata, col): return col and col in adata.obs.columns

    if treat_sample_as_batch:
        if not _has(rna, sample_key) or not _has(atac, sample_key):
            raise KeyError(
                f"sample_key={sample_key!r} missing from rna/atac obs — "
                "required for treat_sample_as_batch=True.")
        return sample_key

    if _has(rna, batch_key) and _has(atac, batch_key):
        return batch_key
    if batch_key:
        print(f"   [glue_train] batch_key={batch_key!r} not in obs of rna/atac — "
              "scGLUE will run without batch correction.")
    return None


def glue_train(preprocess_output_dir, output_dir="glue_output",
               save_prefix="glue", consistency_threshold=0.05,
               treat_sample_as_batch=False,
               batch_key=None,
               sample_key="sample",
               use_highly_variable=True,
               data_batch_size: int = 1024,
               max_epochs: Optional[int] = None,
               dataloader_num_workers: int = 0,
               dataloader_fetches_per_worker: int = 4,
               array_shuffle_num_workers: int = 0,
               graph_shuffle_num_workers: int = 0):
    """
    Train scGLUE model for single-cell multi-omics integration.

    scGLUE's ``configure_dataset(use_batch=...)`` accepts a single column.
    Batch design:

      treat_sample_as_batch  batch_key  →  use_batch
      ─────────────────────  ─────────     ──────────
      False (V2 default)     set       →  batch_key   (remove batch, keep sample)
      False                  None      →  None        (no correction in GLUE)
      True                   any       →  sample_key  (removes sample;
                                                       since each sample is in
                                                       exactly one batch, batch
                                                       is implicitly removed too)

    Skip-if-output-exists: if the final ``<save_prefix>.dill`` and both
    ``<save_prefix>-{rna,atac}-emb.h5ad`` already exist in ``output_dir``
    *and* the saved ``<save_prefix>.design.json`` (batch design + key
    training knobs) matches the current call's arguments, this function
    returns early without retraining. This lets the V2 dual scGLUE pass
    (primary + sample-removal) resume across kill/restart boundaries
    without redoing the run whose artifacts are already saved. A design
    mismatch (e.g. ``treat_sample_as_batch`` or ``batch_key`` changed)
    forces a retrain so stale artifacts are never reused under the wrong
    design.

    The combined-column path lets the optional second scGLUE run yield a
    truly batch- *and* sample-removed cluster embedding when both keys are
    provided.

    Parameters
    ----------
    preprocess_output_dir : str
        Directory containing the preprocessed rna-pp.h5ad / atac-pp.h5ad /
        guidance graph.
    output_dir : str
        Output directory for trained model + embedding h5ads.
    save_prefix : str
        Filename prefix for ``<prefix>-rna-emb.h5ad`` / ``<prefix>-atac-emb.h5ad``.
    consistency_threshold : float
        Min integration consistency to flag the run as reliable.
    treat_sample_as_batch : bool
        See table above.
    batch_key, sample_key : str | None
        obs columns describing technical batch and biological sample.
    use_highly_variable : bool
        Use HVGs only when training (else all features).
    data_batch_size : int
        Cells per minibatch. scGLUE's library default is 128; we lift it to
        1024 to better saturate modern GPUs (V100 / A100 easily fit it).
    max_epochs : int | None
        Hard upper bound on training epochs. ``None`` (default) passes
        scGLUE's ``"AUTO"`` sentinel through to ``fit`` so the library
        picks adaptively (typically ~100–200); early stopping via patience
        usually fires well before the cap regardless.
    dataloader_num_workers : int
        scGLUE's torch DataLoader worker count
        (``scglue.config.DATALOADER_NUM_WORKERS``). Library default 0
        (single-threaded fetch → GPU often idle between steps); 4 overlaps
        I/O with GPU compute (typical 1.5–2× speedup, no convergence impact).
    dataloader_fetches_per_worker : int
        Prefetch depth — number of batches each worker pre-fetches ahead of
        the GPU (``scglue.config.DATALOADER_FETCHES_PER_WORKER``). Library
        default 4. Larger = deeper pipeline = better GPU saturation, at the
        cost of host RAM.
    array_shuffle_num_workers : int
        Background workers for the cell-data shuffle
        (``scglue.config.ARRAY_SHUFFLE_NUM_WORKERS``). Default 0.
    graph_shuffle_num_workers : int
        Background workers for the guidance-graph shuffle
        (``scglue.config.GRAPH_SHUFFLE_NUM_WORKERS``). Default 0.
    """
    # Skip-if-output-exists: a previous run already produced this
    # save_prefix's final artifacts. Don't redo (lets the V2 dual scGLUE
    # pass resume across kill/restart without re-running the first pass).
    model_path    = os.path.join(output_dir, f"{save_prefix}.dill")
    rna_emb_path  = os.path.join(output_dir, f"{save_prefix}-rna-emb.h5ad")
    atac_emb_path = os.path.join(output_dir, f"{save_prefix}-atac-emb.h5ad")
    design_path   = os.path.join(output_dir, f"{save_prefix}.design.json")
    design = {
        "treat_sample_as_batch": treat_sample_as_batch,
        "batch_key": batch_key,
        "sample_key": sample_key,
        "use_highly_variable": use_highly_variable,
        "max_epochs": max_epochs,
        "data_batch_size": data_batch_size,
    }
    if all(os.path.exists(p) for p in (model_path, rna_emb_path, atac_emb_path)):
        prev_design = None
        if os.path.exists(design_path):
            with open(design_path) as f:
                prev_design = json.load(f)
        if prev_design == design:
            print(f"\n\n\n⏭️  Skipping glue_train(save_prefix={save_prefix!r}): "
                  f"all final artifacts already exist in {output_dir} with matching design.")
            print(f"      {os.path.basename(model_path)}")
            print(f"      {os.path.basename(rna_emb_path)}")
            print(f"      {os.path.basename(atac_emb_path)}\n\n\n")
            return
        print(f"\n\n\n⚠️  glue_train(save_prefix={save_prefix!r}): final artifacts exist in "
              f"{output_dir} but batch design does not match — retraining.\n"
              f"      previous: {prev_design}\n"
              f"      current:  {design}\n\n\n")

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

    use_batch = _resolve_glue_batch_design(
        rna, atac,
        treat_sample_as_batch=treat_sample_as_batch,
        batch_key=batch_key,
        sample_key=sample_key,
    )
    print(f"   scGLUE use_batch = {use_batch!r}")

    rna_kwargs  = dict(use_highly_variable=use_highly_variable,
                       use_layer="counts", use_rep="X_pca")
    atac_kwargs = dict(use_highly_variable=use_highly_variable, use_rep="X_lsi")
    if use_batch is not None:
        rna_kwargs["use_batch"]  = use_batch
        atac_kwargs["use_batch"] = use_batch
    scglue.models.configure_dataset(rna,  "NB", **rna_kwargs)
    scglue.models.configure_dataset(atac, "NB", **atac_kwargs)
    
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
    
    # 4. Train GLUE model. The training subdir is per-run (keyed by
    # save_prefix) so multiple glue_train calls under the same output_dir
    # — e.g. the V2 dual scGLUE pass (primary + sample-removal) — do NOT
    # overwrite each other's pretrain.dill / checkpoint_*.pt / tensorboard
    # logs inside training/pretrain/ and training/fine-tune/.
    train_dir = os.path.join(output_dir, "training", save_prefix)
    os.makedirs(train_dir, exist_ok=True)
    
    fit_kws = {
        "directory": train_dir,
        "data_batch_size": data_batch_size,
    }
    if max_epochs is not None:
        fit_kws["max_epochs"] = max_epochs
    # scglue.config is a module-level singleton used by the dataloaders
    # inside fit_SCGLUE. Setting these knobs here, just before training,
    # scopes the changes cleanly to this call.
    scglue.config.DATALOADER_NUM_WORKERS         = dataloader_num_workers
    scglue.config.DATALOADER_FETCHES_PER_WORKER   = dataloader_fetches_per_worker
    scglue.config.ARRAY_SHUFFLE_NUM_WORKERS      = array_shuffle_num_workers
    scglue.config.GRAPH_SHUFFLE_NUM_WORKERS      = graph_shuffle_num_workers
    print(f"\n\n\n🤖 Training GLUE model (fit_kws={fit_kws}, "
          f"workers={dataloader_num_workers}, prefetch={dataloader_fetches_per_worker}, "
          f"array_shuf={array_shuffle_num_workers}, graph_shuf={graph_shuffle_num_workers})...\n\n\n")
    glue = scglue.models.fit_SCGLUE(
        {"rna": rna, "atac": atac},
        guidance_hvf,
        fit_kws=fit_kws,
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
    with open(design_path, "w") as f:
        json.dump(design, f, indent=2)
    # NOTE: do NOT delete rna-pp.h5ad / atac-pp.h5ad — the optional second
    # scGLUE pass (sample-removal run, V2 architecture) re-reads them.
    # They are small relative to the X_glue embeddings and only matter
    # at training time.
    
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


def _merge_second_glue_embedding_into_primary_h5ads(
    glue_dir: str,
    primary_prefix: str,
    secondary_prefix: str,
    z_clust_key: str = "Z_clust",
    z_rmd_key: str = "Z_rmd",
) -> None:
    """Align the primary RNA/ATAC h5ads with paper-named cell-level views.

    Writes into each of ``<primary_prefix>-{rna,atac}-emb.h5ad``:

      * ``obsm[z_rmd_key]``   = primary's ``obsm['X_glue']`` aliased
                                (sample-PRESERVED — RMD displacement role)
      * ``obsm[z_clust_key]`` = secondary's ``obsm['X_glue']`` merged in
                                (sample-REMOVED — cluster role)

    ``obsm['X_glue']`` on the primary is left untouched as the raw scGLUE
    output. Downstream code reads ``Z_rmd`` / ``Z_clust``.
    """
    import os as _os
    for mod in ("rna", "atac"):
        primary  = _os.path.join(glue_dir, f"{primary_prefix}-{mod}-emb.h5ad")
        secondary = _os.path.join(glue_dir, f"{secondary_prefix}-{mod}-emb.h5ad")
        if not _os.path.exists(primary):
            raise FileNotFoundError(f"primary GLUE h5ad missing: {primary}")
        if not _os.path.exists(secondary):
            raise FileNotFoundError(f"secondary GLUE h5ad missing: {secondary}")
        a_primary = ad.read_h5ad(primary)
        a_secondary = ad.read_h5ad(secondary)
        if "X_glue" not in a_primary.obsm:
            raise KeyError(f"primary h5ad {primary} missing obsm['X_glue']")
        if "X_glue" not in a_secondary.obsm:
            raise KeyError(f"secondary h5ad {secondary} missing obsm['X_glue']")
        if a_primary.n_obs != a_secondary.n_obs:
            raise ValueError(
                f"primary/secondary cell count mismatch for {mod}: "
                f"{a_primary.n_obs} vs {a_secondary.n_obs}")
        a_primary.obsm[z_rmd_key]   = a_primary.obsm["X_glue"]
        a_primary.obsm[z_clust_key] = a_secondary.obsm["X_glue"]
        a_primary.write(primary, compression="gzip")
        print(f"  ✓ {mod}: obsm[{z_rmd_key!r}] (from primary X_glue) "
              f"+ obsm[{z_clust_key!r}] (from {secondary_prefix}) → {primary}")


def multiomics_preparation(
    # Data files
    rna_file: str,
    atac_file: str,
    rna_sample_meta_file: Optional[str] = None,
    atac_sample_meta_file: Optional[str] = None,
    additional_hvg_file: Optional[str] = None,
    
    # Process control flags
    run_preprocessing: bool = True,
    run_training: bool = True,
    run_merge: bool = True,
    run_preprocess_per_modality: bool = True,
    run_visualization: bool = True,
    
    # Preprocessing parameters
    ensembl_release: int = 98,
    species: str = "homo_sapiens",
    use_highly_variable: bool = True,
    n_top_genes: int = 2000,
    n_top_peaks: int = 50000,
    atac_min_cells_floor: int = 10,
    n_pca_comps: int = 50,
    n_lsi_comps: int = 50,
    gtf_by: str = "gene_name",
    flavor: str = "seurat_v3",
    generate_umap: bool = False,
    rna_sample_column: str = "sample",
    atac_sample_column: str = "sample",
    
    # Training parameters
    consistency_threshold: float = 0.05,
    treat_sample_as_batch: bool = False,
    save_prefix: str = "glue",
    # Batch design for scGLUE.configure_dataset(use_batch=...). With
    # batch_key set + treat_sample_as_batch=False (V2 default), scGLUE
    # removes the named batch column while preserving per-sample variance,
    # so the primary X_glue is suitable as the RMD embedding.
    batch_key: Optional[str] = None,
    sample_key: str = "sample",
    # scGLUE training throughput knobs (see glue_train docstring)
    data_batch_size: int = 1024,
    max_epochs: Optional[int] = None,
    dataloader_num_workers: int = 0,
    dataloader_fetches_per_worker: int = 4,
    array_shuffle_num_workers: int = 0,
    graph_shuffle_num_workers: int = 0,
    # Optional second scGLUE training run for the sample-REMOVED cluster
    # embedding (paper's ``Z_clust``). When True, scGLUE is invoked a
    # SECOND time with ``treat_sample_as_batch=True`` (use_batch=sample,
    # which implicitly removes batch too since each sample is in exactly
    # one batch). After training, the merge helper writes both
    # ``obsm['Z_rmd']`` (primary's X_glue, aliased) and
    # ``obsm['Z_clust']`` (secondary's X_glue) into the primary RNA + ATAC
    # h5ads so downstream code reads paper-aligned keys uniformly.
    run_second_glue_for_sample_removal: bool = False,
    second_run_save_prefix: str = "glue_no_sample",
    
    # Per-modality preprocess QC params (mirrored from rna/atac_preprocess_cpu).
    rna_min_cells: int = 500,
    rna_min_genes: int = 500,
    rna_pct_mito_cutoff: float = 20.0,
    rna_exclude_genes: Optional[List[str]] = None,
    atac_min_cells: int = 1,
    atac_min_features: int = 2000,
    atac_max_features: int = 15000,
    atac_min_cells_per_sample: int = 1,
    atac_exclude_features: Optional[List[str]] = None,
    atac_doublet_detection: bool = True,
    atac_tfidf_scale_factor: float = 1e4,
    atac_log_transform: bool = True,
    verbose: bool = True,

    # Visualization parameters
    plot_columns: Optional[List[str]] = None,
    
    # Output directory
    output_dir: str = "./glue_results",
):
    """Complete GLUE pipeline: scGLUE training + cell-union merge + per-modality preprocess.

    Flow:
      - run_preprocessing            scGLUE preprocess (HVG, lsi, guidance)
      - run_training                 scGLUE training (single primary; optional
                                     second run for sample-removal — Mode B)
      - run_merge                    Build embedding-only union AnnData
                                     (preprocess/adata_sample.h5ad). No
                                     expression X — see multi_omics_merge.py.
      - run_preprocess_per_modality  Per-modality QC + normalize, writes
                                     preprocess/adata_{rna,atac}_preprocessed.
                                     Used by downstream DGE / RAISIN.
      - run_visualization            UMAP/scatter on the union obsm
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
            n_top_peaks=n_top_peaks,
            atac_min_cells_floor=atac_min_cells_floor,
            n_pca_comps=n_pca_comps,
            n_lsi_comps=n_lsi_comps,
            gtf_by=gtf_by,
            flavor=flavor,
            generate_umap=generate_umap,
            rna_sample_column=rna_sample_column,
            atac_sample_column=atac_sample_column
        )
        print("Preprocessing completed.")
    
    # Step 2: Training (single primary run — produces X_glue =
    # batch-removed (when batch_key is set), sample-preserved).
    if run_training:
        print("Running training...")
        glue_train(
            preprocess_output_dir=glue_output_dir,
            save_prefix=save_prefix,
            consistency_threshold=consistency_threshold,
            treat_sample_as_batch=treat_sample_as_batch,
            batch_key=batch_key,
            sample_key=sample_key,
            use_highly_variable=use_highly_variable,
            data_batch_size=data_batch_size,
            max_epochs=max_epochs,
            dataloader_num_workers=dataloader_num_workers,
            dataloader_fetches_per_worker=dataloader_fetches_per_worker,
            array_shuffle_num_workers=array_shuffle_num_workers,
            graph_shuffle_num_workers=graph_shuffle_num_workers,
            output_dir=glue_output_dir,
        )
        print("Training completed.")

        # Step 2b (optional): SECOND scGLUE run that ALSO removes per-sample
        # variance → produces the paper's Z_clust. With sample as use_batch
        # (each sample lives in exactly one batch, so removing sample also
        # removes batch), this is the end-to-end alternative to running a
        # Harmony post-pass on Z_rmd.
        if run_second_glue_for_sample_removal:
            print(f"Running second scGLUE pass for sample removal "
                  f"(treat_sample_as_batch=True) → obsm['Z_clust']")
            glue_train(
                preprocess_output_dir=glue_output_dir,
                save_prefix=second_run_save_prefix,
                consistency_threshold=consistency_threshold,
                treat_sample_as_batch=True,
                batch_key=batch_key,
                sample_key=sample_key,
                use_highly_variable=use_highly_variable,
                data_batch_size=data_batch_size,
                max_epochs=max_epochs,
                dataloader_num_workers=dataloader_num_workers,
                dataloader_fetches_per_worker=dataloader_fetches_per_worker,
                array_shuffle_num_workers=array_shuffle_num_workers,
                graph_shuffle_num_workers=graph_shuffle_num_workers,
                output_dir=glue_output_dir,
            )
            _merge_second_glue_embedding_into_primary_h5ads(
                glue_dir=glue_output_dir,
                primary_prefix=save_prefix,
                secondary_prefix=second_run_save_prefix,
            )
            print("Second GLUE pass merged.")
    
    # Step 3: build the embedding-only union AnnData from the per-modality
    # scGLUE outputs (no KNN, no synthetic gene-activity — see archive/
    # multi_omics_gene_activity_knn.py for the removed approach).
    rna_emb_path  = os.path.join(glue_output_dir, f"{save_prefix}-rna-emb.h5ad")
    atac_emb_path = os.path.join(glue_output_dir, f"{save_prefix}-atac-emb.h5ad")
    union_path    = os.path.join(output_dir, "preprocess", "adata_sample.h5ad")
    rna_pre_path  = os.path.join(output_dir, "preprocess", "adata_rna_preprocessed.h5ad")
    atac_pre_path = os.path.join(output_dir, "preprocess", "adata_atac_preprocessed.h5ad")

    merged_adata = None
    if run_merge:
        print("Building embedding-only union AnnData...")
        from sampledisco.preparation.multi_omics_merge import build_embedding_union
        merged_adata = build_embedding_union(
            rna_emb_path=rna_emb_path,
            atac_emb_path=atac_emb_path,
            output_path=union_path,
            rna_sample_meta_path=rna_sample_meta_file,
            atac_sample_meta_path=atac_sample_meta_file,
            sample_column=sample_key,
            modality_col="modality",
            verbose=verbose,
        )
    elif os.path.exists(union_path):
        merged_adata = ad.read_h5ad(union_path)

    if run_preprocess_per_modality:
        from sampledisco.preparation.multi_omics_merge import (
            preprocess_rna_for_downstream,
            preprocess_atac_for_downstream,
        )
        print("Per-modality preprocess (RNA): QC + normalize for DGE/RAISIN...")
        preprocess_rna_for_downstream(
            rna_emb_path=rna_emb_path, output_path=rna_pre_path,
            sample_column=rna_sample_column,
            sample_meta_path=rna_sample_meta_file,
            min_cells=rna_min_cells, min_genes=rna_min_genes,
            pct_mito_cutoff=rna_pct_mito_cutoff,
            exclude_genes=rna_exclude_genes, verbose=verbose,
        )
        print("Per-modality preprocess (ATAC): QC + TF-IDF...")
        preprocess_atac_for_downstream(
            atac_emb_path=atac_emb_path, output_path=atac_pre_path,
            sample_column=atac_sample_column,
            sample_meta_path=atac_sample_meta_file,
            min_cells=atac_min_cells,
            min_features=atac_min_features, max_features=atac_max_features,
            min_cells_per_sample=atac_min_cells_per_sample,
            exclude_features=atac_exclude_features,
            doublet_detection=atac_doublet_detection,
            tfidf_scale_factor=atac_tfidf_scale_factor,
            log_transform=atac_log_transform, verbose=verbose,
        )

    if run_visualization:
        print("Running visualization...")
        glue_visualize(
            integrated_path=union_path,
            output_dir=os.path.join(output_dir, "visualization"),
            plot_columns=plot_columns,
        )
        print("Visualization completed.")

    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    print(f"\nTotal runtime: {elapsed_minutes:.2f} minutes")

    return merged_adata