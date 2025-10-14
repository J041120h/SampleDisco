# =========================
# GLUE pipeline with RNA decoder cross-reconstruction (improved version)
# =========================

import anndata as ad
import networkx as nx
import scanpy as sc
import scglue
import numpy as np
from typing import Optional, Tuple, List
from pathlib import Path
import pyensembl
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

# -------------------------
# Utilities (unchanged)
# -------------------------

def clean_anndata_for_saving(adata, verbose=True):
    import pandas as pd
    import numpy as np
    if verbose:
        print("🧹 Cleaning AnnData object for HDF5 compatibility...")
    for col in adata.obs.columns:
        if adata.obs[col].dtype == 'object':
            adata.obs[col] = adata.obs[col].fillna('Unknown').astype(str).astype('category')
        elif adata.obs[col].dtype in ['float64', 'float32']:
            if adata.obs[col].isna().any():
                adata.obs[col] = adata.obs[col].fillna(0.0)
        elif adata.obs[col].dtype in ['int64', 'int32']:
            if adata.obs[col].isna().any():
                adata.obs[col] = adata.obs[col].fillna(0).astype('int64')
    for col in adata.var.columns:
        if adata.var[col].dtype == 'object':
            adata.var[col] = adata.var[col].fillna('Unknown').astype(str).astype('category')
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
    import pandas as pd
    meta = pd.read_csv(metadata_path, sep=sep).set_index(sample_column)
    original_cols = adata.obs.shape[1]
    for col in meta.columns:
        if meta[col].dtype == 'object':
            meta[col] = meta[col].fillna('Unknown').astype(str)
    adata.obs = adata.obs.join(meta, on=sample_column, how='left')
    if sample_column != 'sample':
        if sample_column in adata.obs.columns:
            adata.obs['sample'] = adata.obs[sample_column]
            adata.obs = adata.obs.drop(columns=[sample_column])
            if verbose:
                print(f"   Standardized sample column '{sample_column}' to 'sample'")
        elif 'sample' not in adata.obs.columns and verbose:
            print(f"   Warning: Sample column '{sample_column}' not found")
    new_cols = adata.obs.shape[1] - original_cols
    matched_samples = adata.obs[meta.columns].notna().any(axis=1).sum()
    total_samples = adata.obs.shape[0]
    if verbose:
        print(f"   Merged {new_cols} sample-level columns")
        print(f"   Matched metadata for {matched_samples}/{total_samples} samples")
        if matched_samples < total_samples:
            print(f"   ⚠️ Warning: {total_samples - matched_samples} samples have no metadata")
    return adata


# -------------------------
# Preprocess (unchanged)
# -------------------------

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

    print("\n🚀 Starting GLUE preprocessing pipeline...\n")
    print(f"   Feature selection mode: {'Highly Variable' if use_highly_variable else 'All Features'}\n")
    output_path = Path(output_dir); output_path.mkdir(parents=True, exist_ok=True)

    print(f"📊 Loading data files...")
    print(f"   RNA: {rna_file}")
    print(f"   ATAC: {atac_file}")
    rna = ad.read_h5ad(rna_file)
    atac = ad.read_h5ad(atac_file)
    print(f"✅ Data loaded successfully")
    print(f"   RNA shape: {rna.shape}")
    print(f"   ATAC shape: {atac.shape}\n")

    if rna_sample_meta_file or atac_sample_meta_file:
        print("📋 Loading and merging sample metadata...")
        if rna_sample_meta_file:
            print(f"   Processing RNA metadata: {rna_sample_meta_file}")
            rna = merge_sample_metadata(rna, rna_sample_meta_file, sample_column=rna_sample_column, sep=metadata_sep, verbose=True)
        if atac_sample_meta_file:
            print(f"   Processing ATAC metadata: {atac_sample_meta_file}")
            atac = merge_sample_metadata(atac, atac_sample_meta_file, sample_column=atac_sample_column, sep=metadata_sep, verbose=True)
        print(f"\n✅ Metadata integration and standardization complete")
        print(f"   RNA obs columns: {list(rna.obs.columns)}")
        print(f"   ATAC obs columns: {list(atac.obs.columns)}\n")
    else:
        print("📋 Standardizing existing sample columns...")
        if rna_sample_column != 'sample' and rna_sample_column in rna.obs.columns:
            print(f"   Standardizing RNA sample column '{rna_sample_column}' to 'sample'")
            rna.obs['sample'] = rna.obs[rna_sample_column]; rna.obs = rna.obs.drop(columns=[rna_sample_column])
        if atac_sample_column != 'sample' and atac_sample_column in atac.obs.columns:
            print(f"   Standardizing ATAC sample column '{atac_sample_column}' to 'sample'")
            atac.obs['sample'] = atac.obs[atac_sample_column]; atac.obs = atac.obs.drop(columns=[atac_sample_column])
        print("✅ Sample column standardization complete\n")

    print(f"🧬 Setting up Ensembl annotation...")
    print(f"   Release: {ensembl_release}  Species: {species}")
    ensembl = pyensembl.EnsemblRelease(release=ensembl_release, species=species)
    ensembl.download(); ensembl.index()
    print("✅ Ensembl annotation ready\n")

    sc.settings.seed = random_state

    print(f"🧬 Preprocessing scRNA-seq data...")
    rna.layers["counts"] = rna.X.copy()
    if use_highly_variable:
        print(f"   Selecting {n_top_genes} highly variable genes")
        sc.pp.highly_variable_genes(rna, n_top_genes=n_top_genes, flavor=flavor)
        if additional_hvg_file:
            print(f"\n📝 Processing additional HVG genes from: {additional_hvg_file}")
            try:
                with open(additional_hvg_file, 'r') as f:
                    additional_genes = [line.strip() for line in f if line.strip()]
                genes_in_data = [g for g in additional_genes if g in rna.var_names]
                rna.var.loc[genes_in_data, 'highly_variable'] = True
                not_found = [g for g in additional_genes if g not in rna.var_names]
                print(f"   Marked {len(genes_in_data)}/{len(additional_genes)} extra genes as HVG")
                if not_found: print(f"   {len(not_found)} genes not found in data")
            except Exception as e:
                print(f"   ⚠️ Error reading additional HVG file: {e}\n   Continuing with scanpy-selected HVG only")
    else:
        print(f"   Using all {rna.n_vars} genes")
        rna.var['highly_variable'] = True
        if additional_hvg_file:
            print(f"   Note: additional_hvg_file is ignored when use_highly_variable=False")

    print(f"   Computing {n_pca_comps} PCA components")
    sc.pp.normalize_total(rna); sc.pp.log1p(rna); sc.pp.scale(rna); sc.tl.pca(rna, n_comps=n_pca_comps, svd_solver="auto")
    if generate_umap:
        print("   Computing UMAP embedding..."); sc.pp.neighbors(rna, metric="cosine"); sc.tl.umap(rna)
    print("✅ RNA preprocessing complete\n")

    print(f"🏔️ Preprocessing scATAC-seq data...")
    if use_highly_variable:
        print(f"   Computing feature statistics for peak selection")
        peak_counts = np.array(atac.X.sum(axis=0)).flatten()
        n_top_peaks = min(50000, atac.n_vars)
        top_peak_indices = np.argsort(peak_counts)[-n_top_peaks:]
        atac.var['highly_variable'] = False
        atac.var.iloc[top_peak_indices, atac.var.columns.get_loc('highly_variable')] = True
        atac.var['n_counts'] = peak_counts
        atac.var['n_cells'] = np.array((atac.X > 0).sum(axis=0)).flatten()
        print(f"   Selected {n_top_peaks} highly accessible peaks")
    else:
        print(f"   Using all {atac.n_vars} peaks")
        atac.var['highly_variable'] = True
        atac.var['n_counts'] = np.array(atac.X.sum(axis=0)).flatten()
        atac.var['n_cells'] = np.array((atac.X > 0).sum(axis=0)).flatten()

    print(f"   Computing {n_lsi_comps} LSI components ({lsi_n_iter} iterations)")
    scglue.data.lsi(atac, n_components=n_lsi_comps, n_iter=lsi_n_iter)
    if generate_umap:
        print("   Computing UMAP embedding..."); sc.pp.neighbors(atac, use_rep="X_lsi", metric="cosine"); sc.tl.umap(atac)
    print("✅ ATAC preprocessing complete\n")

    def get_gene_coordinates(gene_names, ensembl_db):
        coords = []; failed = []
        for gene_name in gene_names:
            try:
                genes = ensembl_db.genes_by_name(gene_name)
                if genes:
                    gene = genes[0]
                    strand = '+' if gene.strand == '+' else '-'
                    coords.append({'chrom': f"chr{gene.contig}", 'chromStart': gene.start, 'chromEnd': gene.end, 'strand': strand})
                else:
                    coords.append({'chrom': None, 'chromStart': None, 'chromEnd': None, 'strand': None}); failed.append(gene_name)
            except Exception:
                coords.append({'chrom': None, 'chromStart': None, 'chromEnd': None, 'strand': None}); failed.append(gene_name)
        if failed:
            print(f"   ⚠️ Could not find coordinates for {len(failed)} genes")
        return coords

    print(f"🗺️ Processing gene coordinates...")
    gene_coords = get_gene_coordinates(rna.var_names, ensembl)
    rna.var['chrom'] = [c['chrom'] for c in gene_coords]
    rna.var['chromStart'] = [c['chromStart'] for c in gene_coords]
    rna.var['chromEnd'] = [c['chromEnd'] for c in gene_coords]
    rna.var['strand'] = [c['strand'] for c in gene_coords]
    valid = rna.var['chrom'].notna()
    n_invalid = (~valid).sum()
    if n_invalid > 0:
        rna = rna[:, valid].copy()
        print(f"   Filtered out {n_invalid} genes without coordinates")
    print(f"✅ Gene coordinate processing complete")
    print(f"   Final RNA shape: {rna.shape}\n")

    print(f"🏔️ Processing ATAC peak coordinates...")
    split = atac.var_names.str.split(r"[:-]")
    atac.var["chrom"] = split.map(lambda x: x[0])
    atac.var["chromStart"] = split.map(lambda x: int(x[1]))
    atac.var["chromEnd"] = split.map(lambda x: int(x[2]))
    print("✅ ATAC peak coordinates extracted successfully\n")

    print(f"🕸️ Constructing guidance graph...")
    guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac)
    scglue.graph.check_graph(guidance, [rna, atac])
    print(f"✅ Guidance graph constructed successfully")
    print(f"   Nodes: {guidance.number_of_nodes():,}  Edges: {guidance.number_of_edges():,}\n")

    print(f"🧹 Preparing data for saving...")
    rna = clean_anndata_for_saving(rna, verbose=True)
    atac = clean_anndata_for_saving(atac, verbose=True)

    print(f"💾 Saving preprocessed data to: {output_dir}")
    rna_path = str(Path(output_dir) / "rna-pp.h5ad")
    atac_path = str(Path(output_dir) / "atac-pp.h5ad")
    guidance_path = str(Path(output_dir) / "guidance.graphml.gz")
    rna.write(rna_path, compression=compression)
    atac.write(atac_path, compression=compression)
    nx.write_graphml(guidance, guidance_path)
    print("\n🎉 GLUE preprocessing pipeline completed successfully!\n")
    return rna, atac, guidance


# -------------------------
# IMPROVED Custom RNA encoder + decoder following tutorial best practices
# -------------------------

import torch
import torch.nn.functional as F
from scglue.models.sc import DataEncoder, DataDecoder
from scglue.models.scglue import register_prob_model
from scglue.models import prob as D  # SCGLUE distributions

# Define EPS constant for numerical stability
EPS = 1e-8

class ImprovedNBEncoder(DataEncoder):
    """
    Improved Negative Binomial encoder following SCGLUE tutorial.
    - Computes library size as raw count sum per cell
    - Normalizes to TOTAL_COUNT then applies log1p transformation
    """
    TOTAL_COUNT = 1e4
    
    def compute_l(self, x: torch.Tensor) -> torch.Tensor:
        """Compute library size from raw counts"""
        return x.sum(dim=1, keepdim=True)
    
    def normalize(self, x: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        """Normalize and log-transform the data"""
        # Add small epsilon to avoid division by zero
        normalized = x * (self.TOTAL_COUNT / (l + EPS))
        return normalized.log1p()


import torch
import torch.nn.functional as F
import torch.distributions as TD
from scglue.models.sc import DataDecoder

EPS = 1e-8  # small constant for numerical stability


class ImprovedNBDecoder(DataDecoder):
    """
    Improved Negative Binomial decoder following SCGLUE tutorial.
    Includes batch-specific parameters for handling batch effects.
    """

    def __init__(self, out_features: int, n_batches: int = 1) -> None:
        """
        Initialize decoder with batch-specific parameters.

        Args:
            out_features: Number of output features (genes)
            n_batches: Number of batches for batch effect correction
        """
        super().__init__(out_features, n_batches=n_batches)

        # Batch-specific parameters
        self.scale_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.bias = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.log_theta = torch.nn.Parameter(torch.zeros(n_batches, out_features))

        # Small random init for stability
        torch.nn.init.normal_(self.scale_lin, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.bias, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.log_theta, mean=2.0, std=0.5)  # reasonable dispersion

    def forward(
        self,
        u: torch.Tensor,  # (n_cells, latent_dim)
        v: torch.Tensor,  # (n_genes, latent_dim)
        b: torch.Tensor,  # (n_cells,)
        l: torch.Tensor   # (n_cells, 1) or (n_cells,)
    ) -> TD.NegativeBinomial:
        """
        Return a NegativeBinomial distribution for RNA counts with:
          total_count = theta (inverse-dispersion)
          logits      = log(mu / theta)

        Enforces compositional rates via softmax so that sum(mu_i) per cell ~= library size.
        """
        # Ensure types/shapes
        if b.dtype != torch.long:
            b = b.long()
        if l.ndim == 1:
            l = l.unsqueeze(1)  # (n_cells, 1)

        # Broadcasted per-batch parameters: (n_cells, n_genes)
        scale = F.softplus(self.scale_lin[b])                  # positive
        bias = self.bias[b]
        theta = self.log_theta[b].exp().clamp_min(1e-6)        # avoid 0; (n_cells, n_genes)

        # Gene proportions per cell (stabilized softmax)
        # (n_cells, d) @ (d, n_genes) -> (n_cells, n_genes)
        logits_raw = scale * (u @ v.t()) + bias
        logits_ctr = logits_raw - logits_raw.max(dim=1, keepdim=True).values  # row-wise centering
        p = torch.softmax(logits_ctr, dim=1).clamp_min(EPS)  # (n_cells, n_genes)

        # Expected counts per cell/gene
        mu = p * l  # broadcasts to (n_cells, n_genes)

        # NB parameterization via logits: log(mu / theta)
        nb_logits = (mu + EPS).log() - (theta + EPS).log()
        nb_logits = torch.where(torch.isfinite(nb_logits), nb_logits, torch.zeros_like(nb_logits))

        return TD.NegativeBinomial(
            total_count=theta,  # (n_cells, n_genes)
            logits=nb_logits    # (n_cells, n_genes)
        )

    
import torch
import torch.nn.functional as F
import torch.distributions as TD

EPS = 1e-8  # small constant for numerical stability

def forward(
    self,
    u: torch.Tensor,  # (n_cells, latent_dim)
    v: torch.Tensor,  # (n_genes, latent_dim)
    b: torch.Tensor,  # (n_cells,)
    l: torch.Tensor   # (n_cells, 1) or (n_cells,)
) -> TD.NegativeBinomial:
    """
    Return a NegativeBinomial distribution for RNA counts with:
      total_count = theta (inverse-dispersion)
      logits      = log(mu / theta)

    Enforces compositional rates via softmax so that sum(mu_i) per cell ~= library size.
    """
    # Ensure types/shapes
    if b.dtype != torch.long:
        b = b.long()
    if l.ndim == 1:
        l = l.unsqueeze(1)  # (n_cells, 1)

    # Broadcasted per-batch parameters: (n_cells, n_genes)
    scale = F.softplus(self.scale_lin[b])                      # positive
    bias  = self.bias[b]
    theta = self.log_theta[b].exp().clamp_min(1e-6)            # avoid 0; (n_cells, n_genes)

    # Gene proportions per cell (stabilized softmax)
    # inner product: (n_cells, latent_dim) @ (latent_dim, n_genes) -> (n_cells, n_genes)
    logits_raw = scale * (u @ v.t()) + bias
    logits_ctr = logits_raw - logits_raw.max(dim=1, keepdim=True).values  # subtract row max for stability
    p = torch.softmax(logits_ctr, dim=1).clamp_min(EPS)  # (n_cells, n_genes)

    # Expected counts per cell/gene
    mu = p * l  # broadcasts (n_cells, n_genes)

    # NB parameterization via logits: log(mu / theta)
    nb_logits = (mu + EPS).log() - (theta + EPS).log()

    # Replace non-finite logits (can happen if inputs are pathological) with zeros (neutral)
    nb_logits = torch.where(torch.isfinite(nb_logits), nb_logits, torch.zeros_like(nb_logits))

    return TD.NegativeBinomial(
        total_count=theta,  # (n_cells, n_genes)
        logits=nb_logits    # (n_cells, n_genes)
    )



# Register the improved model
register_prob_model("ImprovedNB", ImprovedNBEncoder, ImprovedNBDecoder)


# -------------------------
# Training (using improved model)
# -------------------------

def glue_train(preprocess_output_dir, output_dir="glue_output", 
               save_prefix="glue", consistency_threshold=0.05,
               treat_sample_as_batch=True,
               use_highly_variable=True):
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
    os.makedirs(output_dir, exist_ok=True)

    rna_path = os.path.join(preprocess_output_dir, "rna-pp.h5ad")
    atac_path = os.path.join(preprocess_output_dir, "atac-pp.h5ad")
    guidance_path = os.path.join(preprocess_output_dir, "guidance.graphml.gz")
    for file_path, file_type in [(rna_path, "RNA"), (atac_path, "ATAC"), (guidance_path, "Guidance")]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_type} file not found: {file_path}")

    print(f"\n\n\n📊 Loading preprocessed data from {preprocess_output_dir}...\n\n\n")
    rna = ad.read_h5ad(rna_path)
    atac = ad.read_h5ad(atac_path)
    guidance = nx.read_graphml(guidance_path)
    print(f"\n\n\nData loaded - RNA: {rna.shape}, ATAC: {atac.shape}, Graph: {guidance.number_of_nodes()} nodes\n\n\n")

    print("\n\n\n⚙️ Configuring datasets...\n\n\n")
    if treat_sample_as_batch:
        # RNA uses our improved custom decoder
        scglue.models.configure_dataset(
            rna, prob_model="ImprovedNB",
            use_highly_variable=use_highly_variable,
            use_layer="counts", use_rep="X_pca", use_batch='sample'
        )
        # ATAC uses default NB decoder
        scglue.models.configure_dataset(
            atac, "NB",
            use_highly_variable=use_highly_variable,
            use_rep="X_lsi", use_batch='sample'
        )
    else:
        scglue.models.configure_dataset(
            rna, prob_model="ImprovedNB",
            use_highly_variable=use_highly_variable,
            use_layer="counts", use_rep="X_pca"
        )
        scglue.models.configure_dataset(
            atac, "NB",
            use_highly_variable=use_highly_variable,
            use_rep="X_lsi"
        )

    if use_highly_variable:
        print("\n\n\n🔍 Extracting highly variable feature subgraph...\n\n\n")
        rna_hvf = rna.var.query("highly_variable").index
        atac_hvf = atac.var.query("highly_variable").index
        guidance_hvf = guidance.subgraph(list(rna_hvf) + list(atac_hvf)).copy()
        print(f"HVF subgraph extracted - RNA HVF: {len(rna_hvf)}, ATAC HVF: {len(atac_hvf)}")
        print(f"HVF graph: {guidance_hvf.number_of_nodes()} nodes, {guidance_hvf.number_of_edges()} edges\n\n\n")
    else:
        print("\n\n\n🔍 Using full feature graph...\n\n\n")
        guidance_hvf = guidance
        print(f"Full graph: {guidance_hvf.number_of_nodes()} nodes, {guidance_hvf.number_of_edges()} edges\n\n\n")

    train_dir = os.path.join(output_dir, "training"); os.makedirs(train_dir, exist_ok=True)

    print(f"\n\n\n🤖 Training GLUE model with improved decoder...\n\n\n")
    glue = scglue.models.fit_SCGLUE(
        {"rna": rna, "atac": atac},
        guidance_hvf,
        fit_kws={
            "directory": train_dir,
            "max_epochs": 500
        }
    )

    print(f"\n\n\n🎨 Generating embeddings...\n\n\n")
    rna.obsm["X_glue"] = glue.encode_data("rna", rna)
    atac.obsm["X_glue"] = glue.encode_data("atac", atac)

    feature_embeddings = glue.encode_graph(guidance_hvf)
    feature_embeddings = pd.DataFrame(feature_embeddings, index=glue.vertices)
    rna.varm["X_glue"] = feature_embeddings.reindex(rna.var_names).to_numpy()
    atac.varm["X_glue"] = feature_embeddings.reindex(atac.var_names).to_numpy()

    print(f"\n\n\n💾 Saving results to {output_dir}...\n\n\n")
    model_path = os.path.join(output_dir, f"{save_prefix}.dill")
    rna_emb_path = os.path.join(output_dir, f"{save_prefix}-rna-emb.h5ad")
    atac_emb_path = os.path.join(output_dir, f"{save_prefix}-atac-emb.h5ad")
    guidance_hvf_path = os.path.join(output_dir, f"{save_prefix}-guidance-hvf.graphml.gz")

    glue.save(model_path)
    rna.write(rna_emb_path, compression="gzip")
    atac.write(atac_emb_path, compression="gzip")
    nx.write_graphml(guidance_hvf, guidance_hvf_path)

    # Remove heavy preprocess files to save space
    if os.path.exists(rna_path): os.remove(rna_path)
    if os.path.exists(atac_path): os.remove(atac_path)

    print(f"\n\n\n📊 Checking integration consistency...\n\n\n")
    consistency_scores = scglue.models.integration_consistency(glue, {"rna": rna, "atac": atac}, guidance_hvf)
    min_consistency = consistency_scores['consistency'].min()
    mean_consistency = consistency_scores['consistency'].mean()
    print(f"\n\n\nConsistency scores - Min: {min_consistency:.4f}, Mean: {mean_consistency:.4f}\n\n\n")
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

def decode_rna_from_atac_with_decoder(
    glue_dir: str,
    output_path: str,
    verbose: bool = True
) -> ad.AnnData:
    """
    Use the trained RNA decoder to generate expected RNA expression for ATAC cells
    and produce a merged AnnData where:
      - RNA cells have their original RNA counts
      - ATAC cells carry decoder-predicted RNA expression

    Assumes files in {glue_dir}:
      - glue.dill
      - glue-rna-emb.h5ad
      - glue-atac-emb.h5ad
    """
    import torch
    import numpy as np
    import pandas as pd
    from scipy import sparse

    model_path = os.path.join(glue_dir, "glue.dill")
    rna_emb_path = os.path.join(glue_dir, "glue-rna-emb.h5ad")
    atac_emb_path = os.path.join(glue_dir, "glue-atac-emb.h5ad")

    if not os.path.exists(model_path):  
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(rna_emb_path): 
        raise FileNotFoundError(f"RNA embeddings not found: {rna_emb_path}")
    if not os.path.exists(atac_emb_path): 
        raise FileNotFoundError(f"ATAC embeddings not found: {atac_emb_path}")

    if verbose:
        print("\n🧠 Loading trained SCGLUE model and embedded datasets...")
    glue = scglue.models.load_model(model_path)
    
    rna = ad.read_h5ad(rna_emb_path)
    atac = ad.read_h5ad(atac_emb_path)

    if "X_glue" not in atac.obsm or "X_glue" not in rna.obsm:
        raise RuntimeError("X_glue embeddings missing from rna/atac AnnData!")

    # 1) Get cell embeddings u for ATAC and feature embeddings v for RNA genes
    if verbose:
        print("🔌 Encoding ATAC cells and fetching RNA feature embeddings...")
    u_atac = atac.obsm["X_glue"]  # (n_atac, d)
    if "X_glue" not in rna.varm:
        raise RuntimeError("RNA feature embeddings (varm['X_glue']) not found!")
    v_rna = rna.varm["X_glue"]    # (n_genes, d)

    # 2) Map ATAC 'sample' to RNA batch indices (if present)
    if "sample" in rna.obs:
        rna_batches = pd.Categorical(rna.obs["sample"]).categories
        rna_batch_index = {s: i for i, s in enumerate(rna_batches)}
        if "sample" in atac.obs:
            atac_b = atac.obs["sample"].map(lambda s: rna_batch_index.get(s, 0)).astype(int).to_numpy()
        else:
            atac_b = np.zeros(atac.n_obs, dtype=int)
    else:
        atac_b = np.zeros(atac.n_obs, dtype=int)

    # 3) Choose ATAC->RNA decoding library size: median RNA library
    if "counts" in (rna.layers or {}):
        rna_lib = np.array(rna.layers["counts"].sum(axis=1)).flatten()
    else:
        X = rna.X.A if sparse.issparse(rna.X) else rna.X
        rna_lib = np.expm1(X).sum(axis=1)
    lib_med = float(np.median(rna_lib))
    l_atac = np.full((atac.n_obs, 1), lib_med, dtype=np.float32)

    # 4) Torch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")
    
    u_t = torch.as_tensor(u_atac, dtype=torch.float32, device=device)
    v_t = torch.as_tensor(v_rna, dtype=torch.float32, device=device)
    b_t = torch.as_tensor(atac_b, dtype=torch.long, device=device)
    l_t = torch.as_tensor(l_atac, dtype=torch.float32, device=device)

    # 5) Locate the correct RNA decoder key robustly (minimal change: only decoder selection logic)
    if not hasattr(glue, "u2x") or not isinstance(glue.u2x, dict) or len(glue.u2x) == 0:
        have = [] if not hasattr(glue, "u2x") else list(getattr(glue, "u2x", {}).keys())
        raise RuntimeError(
            "This SCGLUE checkpoint doesn't expose any u2x decoders. "
            f"Available u2x keys found: {have}. "
            "Re-train/compile with a count likelihood for RNA (e.g., ImprovedNB/NB/Poisson)."
        )

    u2x_keys = list(glue.u2x.keys())

    # Prefer decoders whose output dimensionality matches the number of RNA genes
    size_matched = [k for k, dec in glue.u2x.items()
                    if getattr(dec, "out_features", None) == rna.n_vars]

    # Common aliases to try if multiple keys exist
    preferred_aliases = ["rna", "gex", "RNA", "rna_counts", "GeneExpression"]

    rna_key = None
    if "rna" in size_matched:
        rna_key = "rna"
    elif len(size_matched) == 1:
        rna_key = size_matched[0]
    else:
        # Fallback to aliases
        rna_key = next((k for k in preferred_aliases if k in glue.u2x), None)
        # If still unresolved and there is only one key overall, use it
        if rna_key is None and len(u2x_keys) == 1:
            rna_key = u2x_keys[0]

    if rna_key is None:
        raise RuntimeError(
            "Could not locate the RNA decoder in glue.u2x. "
            f"Available keys: {u2x_keys}. "
            f"Keys with out_features={rna.n_vars}: {size_matched}. "
            "If one of these corresponds to RNA (e.g., 'gex'), use that as the decoder key."
        )

    rna_decoder = glue.u2x[rna_key].to(device).eval()

    if verbose:
        print(f"🧪 Decoding expected RNA expression for ATAC cells via RNA decoder key: '{rna_key}'")
    with torch.no_grad():
        dist = rna_decoder(u_t, v_t, b_t, l_t)  # torch.distributions.NegativeBinomial (or compatible)
        mu_atac_rna = dist.mean  # (n_atac, n_genes)

    mu_np = mu_atac_rna.detach().cpu().numpy().astype(np.float64)

    # 6) Build merged AnnData
    #    RNA rows = original (prefer counts); ATAC rows = predicted mu
    if "counts" in (rna.layers or {}):
        X_rna = rna.layers["counts"]
        X_rna = X_rna.tocsr().astype(np.float64) if sparse.issparse(X_rna) \
            else np.asarray(X_rna).astype(np.float64)
    else:
        X_rna = rna.X.tocsr().astype(np.float64) if sparse.issparse(rna.X) \
            else np.asarray(rna.X).astype(np.float64)

    genes = rna.var_names.copy()
    X_atac_pred = mu_np

    def to_csr_if_sparse(x):
        if sparse.issparse(x): 
            return x.tocsr()
        nnz = np.count_nonzero(x)
        sparsity = 1 - nnz / (x.shape[0] * x.shape[1])
        return sparse.csr_matrix(x) if sparsity > 0.5 else x

    X_rna = to_csr_if_sparse(X_rna)
    X_atac_pred = to_csr_if_sparse(X_atac_pred)

    rna_block = ad.AnnData(
        X=X_rna,
        obs=rna.obs.copy(),
        var=rna.var.loc[genes].copy()
    )
    rna_block.obs['modality'] = 'RNA'

    atac_block = ad.AnnData(
        X=X_atac_pred,
        obs=atac.obs.copy(),
        var=rna.var.loc[genes].copy()  # same genes as RNA
    )
    atac_block.obs['modality'] = 'ATAC'

    # carry over embeddings
    for k, v in rna.obsm.items(): 
        rna_block.obsm[k] = v
    for k, v in atac.obsm.items(): 
        atac_block.obsm[k] = v

    # unique indices across modalities
    rna_block.obs['original_barcode'] = rna_block.obs.index
    atac_block.obs['original_barcode'] = atac_block.obs.index
    rna_block.obs.index = pd.Index([f"{x}_RNA" for x in rna_block.obs.index])
    atac_block.obs.index = pd.Index([f"{x}_ATAC" for x in atac_block.obs.index])

    merged = ad.concat([rna_block, atac_block], axis=0, join="inner", merge="same")

    # Save
    out_dir = os.path.join(output_path, 'preprocess')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'atac_rna_integrated.h5ad')
    merged = clean_anndata_for_saving(merged, verbose=False)
    merged.write(out_path, compression='gzip', compression_opts=4)

    if verbose:
        print(f"\n✅ RNA decoder transfer complete!")
        print(f"   Output path: {out_path}")
        print(f"   Merged dataset shape: {merged.shape}")
        print(f"   RNA cells: {(merged.obs['modality'] == 'RNA').sum()}")
        print(f"   ATAC cells: {(merged.obs['modality'] == 'ATAC').sum()}")

    return merged

# -------------------------
# Visualization (unchanged)
# -------------------------

def glue_visualize(integrated_path, output_dir=None, plot_columns=None):
    print("Loading integrated RNA-ATAC data...")
    combined = ad.read_h5ad(integrated_path)
    if output_dir is None:
        output_dir = os.path.dirname(integrated_path)
    os.makedirs(output_dir, exist_ok=True)

    if "X_glue" not in combined.obsm:
        print("Error: X_glue embeddings not found in integrated data. Run scGLUE integration first.")
        return

    print("Computing UMAP from scGLUE embeddings...")
    sc.pp.neighbors(combined, use_rep="X_glue", metric="cosine")
    sc.tl.umap(combined)
    sc.settings.set_figure_params(dpi=80, facecolor='white', figsize=(8, 6))
    plt.rcParams['figure.max_open_warning'] = 50
    if plot_columns is None:
        plot_columns = ['modality', 'cell_type']
    print(f"Generating visualizations for columns: {plot_columns}")
    for col in plot_columns:
        if col not in combined.obs.columns:
            print(f"Warning: Column '{col}' not found in data. Skipping...")
            continue
        if col in combined.var_names:
            print(f"Warning: Column '{col}' exists in both obs.columns and var_names. Skipping...")
            continue
        try:
            plt.figure(figsize=(12, 8))
            sc.pl.umap(combined, color=col, title=f"scGLUE Integration: {col}", save=False, show=False, wspace=0.65)
            plt.tight_layout()
            col_plot_path = os.path.join(output_dir, f"scglue_umap_{col}.png")
            plt.savefig(col_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {col} plot: {col_plot_path}")
        except Exception as e:
            print(f"Error plotting {col}: {str(e)}"); plt.close()

    if ("modality" in plot_columns and "modality" in combined.obs.columns and 
        "cell_type" in plot_columns and "cell_type" in combined.obs.columns):
        print("\nCreating modality-cell type distribution heatmap...")
        modality_celltype_counts = pd.crosstab(combined.obs['cell_type'], combined.obs['modality'])
        modality_celltype_props = modality_celltype_counts.div(modality_celltype_counts.sum(axis=1), axis=0) * 100
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        sns.heatmap(modality_celltype_counts, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Number of cells'}, ax=ax1)
        ax1.set_title('Cell Count Distribution: Modality vs Cell Type', fontsize=14, pad=20)
        ax1.set_xlabel('Modality'); ax1.set_ylabel('Cell Type')
        sns.heatmap(modality_celltype_props, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': 'Percentage (%)'}, ax=ax2)
        ax2.set_title('Modality Distribution within Each Cell Type (%)', fontsize=14, pad=20)
        ax2.set_xlabel('Modality'); ax2.set_ylabel('Cell Type')
        plt.tight_layout()
        heatmap_plot_path = os.path.join(output_dir, "scglue_modality_celltype_heatmap.png")
        plt.savefig(heatmap_plot_path, dpi=300, bbox_inches='tight'); plt.close()
        print(f"Saved modality-cell type heatmap: {heatmap_plot_path}")
        plt.figure(figsize=(14, 8))
        modality_celltype_props_T = modality_celltype_props.T
        modality_celltype_props_T.plot(kind='bar', stacked=True, colormap='tab20', width=0.8)
        plt.title('Modality Composition by Cell Type', fontsize=16, pad=20)
        plt.xlabel('Modality'); plt.ylabel('Percentage of Cells (%)')
        plt.legend(title='Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right'); plt.tight_layout()
        stacked_plot_path = os.path.join(output_dir, "scglue_modality_celltype_stacked.png")
        plt.savefig(stacked_plot_path, dpi=300, bbox_inches='tight'); plt.close()
        csv_path = os.path.join(output_dir, "scglue_modality_celltype_distribution.csv")
        modality_celltype_counts.to_csv(csv_path); print(f"Saved distribution table to: {csv_path}")

    print("\n=== Integration Summary ===")
    print(f"Total cells: {combined.n_obs}")
    print(f"Total features: {combined.n_vars}")
    print(f"Available metadata columns: {list(combined.obs.columns)}")
    if "modality" in combined.obs.columns:
        modality_counts = combined.obs['modality'].value_counts()
        print(f"\nModality breakdown:")
        for modality, count in modality_counts.items():
            print(f"  {modality}: {count} cells")
    hvg_used = combined.var['highly_variable'].sum() if 'highly_variable' in combined.var else combined.n_vars
    print(f"\nFeatures used: {hvg_used}/{combined.n_vars}")
    print("\nVisualization complete!")


# -------------------------
# Main wrapper function (unchanged except for model name)
# -------------------------

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
    run_gene_activity: bool = True,   # kept for compatibility; now runs decoder-based transfer
    run_cell_types: bool = False,     # set default False; you can rewire your Linux cell-types as before
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

    # Visualization parameters
    plot_columns: Optional[List[str]] = None,

    # Output directory
    output_dir: str = "./glue_results",
):
    """
    Complete GLUE pipeline with improved decoder:
    - preprocess
    - train (RNA uses improved custom decoder)
    - "gene activity" step now = decoder-based RNA transfer for ATAC
    - (optional) cell types
    - visualization
    """
    os.makedirs(output_dir, exist_ok=True)
    glue_output_dir = os.path.join(output_dir, "integration", "glue")
    start_time = __import__("time").time()

    if run_preprocessing:
        print("Running preprocessing...")
        glue_preprocess_pipeline(
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

    if run_training:
        print("Running training with improved decoder...")
        glue_train(
            preprocess_output_dir=glue_output_dir,
            save_prefix=save_prefix,
            consistency_threshold=consistency_threshold,
            treat_sample_as_batch=treat_sample_as_batch,
            use_highly_variable=use_highly_variable,
            output_dir=glue_output_dir
        )
        print("Training completed.")

    # Decoder-based RNA transfer step
    if run_gene_activity:
        print("Decoding RNA for ATAC cells using the improved RNA decoder...")
        merged_adata = decode_rna_from_atac_with_decoder(
            glue_dir=glue_output_dir,
            output_path=output_dir,
            verbose=True
        )
        print("Decoder transfer completed.")
    else:
        integrated_file = os.path.join(output_dir, "preprocess", "atac_rna_integrated.h5ad")
        if os.path.exists(integrated_file):
            merged_adata = ad.read_h5ad(integrated_file)
        else:
            raise FileNotFoundError(f"Integrated file not found: {integrated_file}. Run the decoder step first.")

    # (Optional) your existing cell-type routine can be reattached here as before

    if run_visualization:
        print("Running visualization...")
        integrated_file_path = os.path.join(output_dir, "preprocess", "atac_rna_integrated.h5ad")
        glue_visualize(
            integrated_path=integrated_file_path,
            output_dir=os.path.join(output_dir, "visualization"),
            plot_columns=plot_columns
        )
        print("Visualization completed.")

    elapsed_minutes = (__import__("time").time() - start_time) / 60
    print(f"\nTotal runtime: {elapsed_minutes:.2f} minutes")

    return merged_adata if run_gene_activity else None


if __name__ == "__main__":
    import os

    # ========================================
    # MAIN PIPELINE PARAMETERS
    # ========================================
    output_dir = "/dcs07/hongkai/data/harry/result/Benchmark_omics/decoder"

    # ========================================
    # PIPELINE CONTROL FLAGS
    # ========================================
    run_rna_pipeline = False
    run_atac_pipeline = False
    run_multiomics_pipeline = True

    use_gpu = True
    initialization = False
    verbose = True
    save_intermediate = True

    # ========================================
    # MULTIOMICS PIPELINE PARAMETERS
    # ========================================
    multiomics_rna_file = "/dcl01/hongkai/data/data/hjiang/Data/paired/rna/placenta.h5ad"
    multiomics_atac_file = "/dcl01/hongkai/data/data/hjiang/Data/paired/atac/placenta.h5ad"
    multiomics_output_dir = None  # will default below if None

    # Main control flags
    multiomics_run_glue = True
    multiomics_run_integrate_preprocess = True
    multiomics_run_dimensionality_reduction = True
    multiomics_run_visualize_embedding = True
    multiomics_run_find_optimal_resolution = False

    # GLUE sub-steps
    multiomics_run_glue_preprocessing = False
    multiomics_run_glue_training = False
    multiomics_run_glue_gene_activity = True      # decoder-based transfer step
    multiomics_run_glue_cell_types = True
    multiomics_run_glue_visualization = False

    # Basics
    multiomics_rna_sample_meta_file = None
    multiomics_atac_sample_meta_file = None
    multiomics_additional_hvg_file = None
    multiomics_rna_sample_column = "sample"
    multiomics_atac_sample_column = "sample"
    multiomics_sample_column = "sample"
    multiomics_sample_col = "sample"
    multiomics_batch_col = "modality"
    multiomics_celltype_col = "cell_type"
    multiomics_verbose = True
    multiomics_use_gpu = True
    large_data_need_extra_memory = True
    multiomics_random_state = 42

    # GLUE integration params
    multiomics_ensembl_release = 98
    multiomics_species = "homo_sapiens"
    multiomics_use_highly_variable = True
    multiomics_n_top_genes = 1500
    multiomics_n_pca_comps = 50
    multiomics_n_lsi_comps = 50
    multiomics_lsi_n_iter = 15
    multiomics_gtf_by = "gene_name"
    multiomics_flavor = "seurat_v3"
    multiomics_generate_umap = False
    multiomics_compression = "gzip"
    multiomics_metadata_sep = ","
    multiomics_consistency_threshold = 0.05
    treat_sample_as_batch = False
    multiomics_save_prefix = "glue"
    # (kNN-related params are intentionally ignored; decoder is used instead)
    multiomics_existing_cell_types = False
    multiomics_n_target_clusters = None
    multiomics_cluster_resolution = 0.8
    multiomics_use_rep_celltype = "X_glue"
    multiomics_markers = None
    multiomics_method = "average"
    multiomics_metric_celltype = "euclidean"
    multiomics_distance_mode = "centroid"
    multiomics_generate_umap_celltype = True
    multiomics_plot_columns = None

    # Output directory for this run
    final_output_dir = multiomics_output_dir or os.path.join(output_dir, "multiomics")

    # =========================
    # Execute multiomics GLUE (improved decoder)
    # =========================
    if run_multiomics_pipeline and multiomics_run_integrate_preprocess:
        merged_adata = glue(
            # Required files
            rna_file=multiomics_rna_file,
            atac_file=multiomics_atac_file,

            # Optional metadata / HVG list
            rna_sample_meta_file=multiomics_rna_sample_meta_file,
            atac_sample_meta_file=multiomics_atac_sample_meta_file,
            additional_hvg_file=multiomics_additional_hvg_file,

            # GLUE sub-step controls
            run_preprocessing=multiomics_run_glue_preprocessing,
            run_training=multiomics_run_glue_training,
            run_gene_activity=multiomics_run_glue_gene_activity,   # uses improved RNA decoder
            run_cell_types=multiomics_run_glue_cell_types,         # keep if you later reattach your cell-type module
            run_visualization=multiomics_run_glue_visualization,

            # Preprocessing / training params
            ensembl_release=multiomics_ensembl_release,
            species=multiomics_species,
            use_highly_variable=multiomics_use_highly_variable,
            n_top_genes=multiomics_n_top_genes,
            n_pca_comps=multiomics_n_pca_comps,
            n_lsi_comps=multiomics_n_lsi_comps,
            lsi_n_iter=multiomics_lsi_n_iter,
            gtf_by=multiomics_gtf_by,
            flavor=multiomics_flavor,
            generate_umap=multiomics_generate_umap,
            compression=multiomics_compression,
            random_state=multiomics_random_state,
            metadata_sep=multiomics_metadata_sep,
            rna_sample_column=multiomics_rna_sample_column,
            atac_sample_column=multiomics_atac_sample_column,

            # Training thresholds/flags
            consistency_threshold=multiomics_consistency_threshold,
            treat_sample_as_batch=treat_sample_as_batch,
            save_prefix=multiomics_save_prefix,

            # Visualization
            plot_columns=multiomics_plot_columns,

            # Where to write outputs
            output_dir=final_output_dir,
        )

        print("\n✅ Multiomics GLUE with improved decoder run finished.")
        print(f"   Output directory: {final_output_dir}")
    else:
        print("Multiomics pipeline disabled or integrate_preprocess flag is False; nothing to run.")