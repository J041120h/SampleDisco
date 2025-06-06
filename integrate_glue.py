import anndata as ad
import networkx as nx
import scanpy as sc
import scglue
import numpy as np
from typing import Optional, Tuple
from pathlib import Path
import pyensembl

def glue_preprocess_pipeline(
    rna_file: str,
    atac_file: str,
    ensembl_release: int = 110,
    species: str = "homo_sapiens",
    output_dir: str = "./",
    n_top_genes: int = 2000,
    n_pca_comps: int = 100,
    n_lsi_comps: int = 100,
    lsi_n_iter: int = 15,
    gtf_by: str = "gene_name",
    flavor: str = "seurat_v3",
    generate_umap: bool = False,
    compression: str = "gzip",
    random_state: int = 42
) -> Tuple[ad.AnnData, ad.AnnData, nx.MultiDiGraph]:
    """
    Complete GLUE preprocessing pipeline for scRNA-seq and scATAC-seq data integration.
    
    Parameters
    ----------
    rna_file : str
        Path to the scRNA-seq data file (h5ad format)
    atac_file : str
        Path to the scATAC-seq data file (h5ad format)
    ensembl_release : int, default 110
        Ensembl release version to download GTF annotation
    species : str, default "homo_sapiens"
        Species name (homo_sapiens, mus_musculus, etc.)
    output_dir : str, default "./"
        Directory to save preprocessed files
    n_top_genes : int, default 2000
        Number of highly variable genes to select
    n_pca_comps : int, default 100
        Number of PCA components for RNA data
    n_lsi_comps : int, default 100
        Number of LSI components for ATAC data
    lsi_n_iter : int, default 15
        Number of iterations for LSI randomized SVD
    gtf_by : str, default "gene_name"
        GTF attribute to match gene names
    flavor : str, default "seurat_v3"
        Method for highly variable gene selection
    generate_umap : bool, default False
        Whether to generate UMAP embeddings
    compression : str, default "gzip"
        Compression method for output files
    random_state : int, default 42
        Random state for reproducibility
        
    Returns
    -------
    Tuple[ad.AnnData, ad.AnnData, nx.MultiDiGraph]
        Preprocessed RNA data, ATAC data, and guidance graph
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    rna = ad.read_h5ad(rna_file)
    atac = ad.read_h5ad(atac_file)
    
    # Download and setup Ensembl annotation
    ensembl = pyensembl.EnsemblRelease(release=ensembl_release, species=species)
    ensembl.download()
    ensembl.index()
    
    # Set random seed
    sc.settings.seed = random_state
    
    # Preprocess scRNA-seq data
    rna.layers["counts"] = rna.X.copy()
    sc.pp.highly_variable_genes(rna, n_top_genes=n_top_genes, flavor=flavor)
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    sc.pp.scale(rna)
    sc.tl.pca(rna, n_comps=n_pca_comps, svd_solver="auto")
    
    if generate_umap:
        sc.pp.neighbors(rna, metric="cosine")
        sc.tl.umap(rna)
    
    # Preprocess scATAC-seq data
    scglue.data.lsi(atac, n_components=n_lsi_comps, n_iter=lsi_n_iter)
    
    if generate_umap:
        sc.pp.neighbors(atac, use_rep="X_lsi", metric="cosine")
        sc.tl.umap(atac)
    
    # Get gene coordinates from Ensembl
    def get_gene_coordinates(gene_names, ensembl_db):
        """Extract gene coordinates from pyensembl database"""
        coords = []
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
            except:
                coords.append({'chrom': None, 'chromStart': None, 'chromEnd': None, 'strand': None})
        return coords
    
    # Add gene coordinates to RNA data
    gene_coords = get_gene_coordinates(rna.var_names, ensembl)
    rna.var['chrom'] = [c['chrom'] for c in gene_coords]
    rna.var['chromStart'] = [c['chromStart'] for c in gene_coords]
    rna.var['chromEnd'] = [c['chromEnd'] for c in gene_coords]
    rna.var['strand'] = [c['strand'] for c in gene_coords]
    
    # Remove genes without coordinates
    valid_genes = rna.var['chrom'].notna()
    rna = rna[:, valid_genes].copy()
    
    # Extract ATAC peak coordinates
    split = atac.var_names.str.split(r"[:-]")
    atac.var["chrom"] = split.map(lambda x: x[0])
    atac.var["chromStart"] = split.map(lambda x: x[1]).astype(int)
    atac.var["chromEnd"] = split.map(lambda x: x[2]).astype(int)
    
    # Construct guidance graph
    guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac)
    
    # Validate guidance graph
    scglue.graph.check_graph(guidance, [rna, atac])
    
    # Save preprocessed data
    rna.write(str(output_path / "rna-pp.h5ad"), compression=compression)
    atac.write(str(output_path / "atac-pp.h5ad"), compression=compression)
    nx.write_graphml(guidance, str(output_path / "guidance.graphml.gz"))
    
    return rna, atac, guidance

def glue_train(preprocess_output_dir, train_output_dir="glue", 
               save_prefix="glue", consistency_threshold=0.05):
    """
    Train SCGLUE model for single-cell multi-omics integration.
    
    Args:
        preprocess_output_dir: Directory containing preprocessed files from stage 1
                              (expects rna-pp.h5ad, atac-pp.h5ad, guidance.graphml.gz)
        train_output_dir: Directory for model snapshots and logs
        save_prefix: Prefix for saved files
        consistency_threshold: Minimum consistency score for reliable integration
        
    Returns:
        dict: Contains trained model, consistency scores, and file paths
    """
    import anndata as ad
    import networkx as nx
    import scglue
    import pandas as pd
    import scanpy as sc
    import os
    from itertools import chain
    
    # 1. Load preprocessed data from preprocessing output directory
    rna_path = os.path.join(preprocess_output_dir, "rna-pp.h5ad")
    atac_path = os.path.join(preprocess_output_dir, "atac-pp.h5ad")
    guidance_path = os.path.join(preprocess_output_dir, "guidance.graphml.gz")
    
    # Check if files exist
    for file_path, file_type in [(rna_path, "RNA"), (atac_path, "ATAC"), (guidance_path, "Guidance")]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_type} file not found: {file_path}")
    
    rna = ad.read_h5ad(rna_path)
    atac = ad.read_h5ad(atac_path)
    guidance = nx.read_graphml(guidance_path)
    
    # 2. Configure datasets with negative binomial distribution
    scglue.models.configure_dataset(
        rna, "NB", use_highly_variable=True, 
        use_layer="counts", use_rep="X_pca"
    )
    scglue.models.configure_dataset(
        atac, "NB", use_highly_variable=True, 
        use_rep="X_lsi"
    )
    
    # 3. Extract subgraph for highly variable features
    guidance_hvf = guidance.subgraph(chain(
        rna.var.query("highly_variable").index,
        atac.var.query("highly_variable").index
    )).copy()
    
    # 4. Train GLUE model
    glue = scglue.models.fit_SCGLUE(
        {"rna": rna, "atac": atac}, guidance_hvf,
        fit_kws={"directory": train_output_dir}
    )
    
    # 5. Check integration consistency
    consistency_scores = scglue.models.integration_consistency(
        glue, {"rna": rna, "atac": atac}, guidance_hvf
    )
    
    # 6. Generate embeddings
    rna.obsm["X_glue"] = glue.encode_data("rna", rna)
    atac.obsm["X_glue"] = glue.encode_data("atac", atac)
    
    # Feature embeddings
    feature_embeddings = glue.encode_graph(guidance_hvf)
    feature_embeddings = pd.DataFrame(feature_embeddings, index=glue.vertices)
    rna.varm["X_glue"] = feature_embeddings.reindex(rna.var_names).to_numpy()
    atac.varm["X_glue"] = feature_embeddings.reindex(atac.var_names).to_numpy()
    
    # 7. Save results
    model_path = f"{save_prefix}.dill"
    rna_emb_path = f"{save_prefix}-rna-emb.h5ad"
    atac_emb_path = f"{save_prefix}-atac-emb.h5ad"
    guidance_hvf_path = f"{save_prefix}-guidance-hvf.graphml.gz"
    
    glue.save(model_path)
    rna.write(rna_emb_path, compression="gzip")
    atac.write(atac_emb_path, compression="gzip")
    nx.write_graphml(guidance_hvf, guidance_hvf_path)
    
    # Check if integration is reliable
    min_consistency = consistency_scores['consistency'].min()
    is_reliable = min_consistency > consistency_threshold
    
    return {
        'model': glue,
        'consistency_scores': consistency_scores,
        'is_reliable': is_reliable,
        'min_consistency': min_consistency,
        'rna_embedded': rna,
        'atac_embedded': atac,
        'guidance_hvf': guidance_hvf,
        'files': {
            'model': model_path,
            'rna_embeddings': rna_emb_path,
            'atac_embeddings': atac_emb_path,
            'guidance_graph': guidance_hvf_path
        }
    }

# Example usage
if __name__ == "__main__":
    # rna, atac, guidance = glue_preprocess_pipeline(
    #     rna_file="/Users/harry/Desktop/GenoDistance/Data/count_data.h5ad",
    #     atac_file="/Users/harry/Desktop/GenoDistance/Data/test_ATAC.h5ad", 
    #     ensembl_release=98,  # Latest human Ensembl release
    #     species="homo_sapiens",
    #     output_dir="/Users/harry/Desktop/GenoDistance/result/glue"
    # )
    glue_train( preprocess_output_dir = "/Users/harry/Desktop/GenoDistance/result/glue", 
               save_prefix="glue", consistency_threshold=0.05)