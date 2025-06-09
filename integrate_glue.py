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
    rna_sample_meta_file: Optional[str] = None,
    atac_sample_meta_file: Optional[str] = None,
    ensembl_release: int = 98,
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
    """
    
    print("\n\n\n🚀 Starting GLUE preprocessing pipeline...\n\n\n")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\n\n\n📊 Loading data files...\nRNA: {rna_file}\nATAC: {atac_file}\n\n\n")
    rna = ad.read_h5ad(rna_file)
    atac = ad.read_h5ad(atac_file)
    print(f"\n\n\nData loaded - RNA shape: {rna.shape}, ATAC shape: {atac.shape}\n\n\n")
    
    # Load and integrate sample metadata
    if rna_sample_meta_file or atac_sample_meta_file:
        print(f"\n\n\n📋 Loading sample metadata...\n\n\n")
        
        if rna_sample_meta_file:
            print(f"Loading RNA metadata: {rna_sample_meta_file}")
            import pandas as pd
            rna_meta = pd.read_csv(rna_sample_meta_file, index_col=0)
            
            # Match metadata to RNA obs
            common_samples = rna.obs.index.intersection(rna_meta.index)
            if len(common_samples) == 0:
                print("⚠️ Warning: No matching sample IDs between RNA data and metadata")
            else:
                print(f"Matched {len(common_samples)} RNA samples with metadata")
                # Add metadata columns to RNA obs
                for col in rna_meta.columns:
                    rna.obs[col] = rna_meta.loc[rna.obs.index, col] if col not in rna.obs.columns else rna.obs[col]
        
        if atac_sample_meta_file:
            print(f"Loading ATAC metadata: {atac_sample_meta_file}")
            atac_meta = pd.read_csv(atac_sample_meta_file, index_col=0)
            
            # Match metadata to ATAC obs
            common_samples = atac.obs.index.intersection(atac_meta.index)
            if len(common_samples) == 0:
                print("⚠️ Warning: No matching sample IDs between ATAC data and metadata")
            else:
                print(f"Matched {len(common_samples)} ATAC samples with metadata")
                # Add metadata columns to ATAC obs
                for col in atac_meta.columns:
                    atac.obs[col] = atac_meta.loc[atac.obs.index, col] if col not in atac.obs.columns else atac.obs[col]
        
        print(f"\n\n\nMetadata integration complete\nRNA obs columns: {list(rna.obs.columns)}\nATAC obs columns: {list(atac.obs.columns)}\n\n\n")
    
    # Download and setup Ensembl annotation
    print(f"\n\n\n🧬 Setting up Ensembl annotation (release: {ensembl_release}, species: {species})\n\n\n")
    ensembl = pyensembl.EnsemblRelease(release=ensembl_release, species=species)
    ensembl.download()
    ensembl.index()
    
    # Set random seed
    sc.settings.seed = random_state
    
    # Preprocess scRNA-seq data
    print(f"\n\n\n🧬 Preprocessing scRNA-seq data (n_top_genes={n_top_genes})...\n\n\n")
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
    print(f"\n\n\n🧬 Preprocessing scATAC-seq data (LSI components={n_lsi_comps})...\n\n\n")
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
    print(f"\n\n\n🗺️ Processing gene coordinates ({len(rna.var_names)} genes)...\n\n\n")
    gene_coords = get_gene_coordinates(rna.var_names, ensembl)
    rna.var['chrom'] = [c['chrom'] for c in gene_coords]
    rna.var['chromStart'] = [c['chromStart'] for c in gene_coords]
    rna.var['chromEnd'] = [c['chromEnd'] for c in gene_coords]
    rna.var['strand'] = [c['strand'] for c in gene_coords]
    
    # Remove genes without coordinates
    valid_genes = rna.var['chrom'].notna()
    n_valid = valid_genes.sum()
    n_invalid = (~valid_genes).sum()
    rna = rna[:, valid_genes].copy()
    print(f"\n\n\nGene filtering complete - {n_valid} genes kept, {n_invalid} removed\nFinal RNA shape: {rna.shape}\n\n\n")
    
    # Extract ATAC peak coordinates
    print(f"\n\n\n🏔️ Processing ATAC peak coordinates ({len(atac.var_names)} peaks)...\n\n\n")
    split = atac.var_names.str.split(r"[:-]")
    atac.var["chrom"] = split.map(lambda x: x[0])
    atac.var["chromStart"] = split.map(lambda x: x[1]).astype(int)
    atac.var["chromEnd"] = split.map(lambda x: x[2]).astype(int)
    
    # Construct guidance graph
    print(f"\n\n\n🕸️ Constructing guidance graph...\n\n\n")
    guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac)
    n_nodes = guidance.number_of_nodes()
    n_edges = guidance.number_of_edges()
    
    # Validate guidance graph
    scglue.graph.check_graph(guidance, [rna, atac])
    print(f"\n\n\nGuidance graph created - {n_nodes} nodes, {n_edges} edges\n\n\n")
    
    # Save preprocessed data
    print(f"\n\n\n💾 Saving preprocessed data to {output_dir}...\n\n\n")
    rna_path = str(output_path / "rna-pp.h5ad")
    atac_path = str(output_path / "atac-pp.h5ad")
    guidance_path = str(output_path / "guidance.graphml.gz")
    
    rna.write(rna_path, compression=compression)
    atac.write(atac_path, compression=compression)
    nx.write_graphml(guidance, guidance_path)
    
    print("\n\n\n🎉 Preprocessing pipeline completed successfully!\n\n\n")
    return rna, atac, guidance

def glue_train(preprocess_output_dir, train_output_dir="glue", 
               save_prefix="glue", consistency_threshold=0.05):
    """
    Train SCGLUE model for single-cell multi-omics integration.
    """
    import anndata as ad
    import networkx as nx
    import scglue
    import pandas as pd
    import scanpy as sc
    import os
    from itertools import chain
    
    print("\n\n\n🚀 Starting GLUE training pipeline...\n\n\n")
    
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
        rna, "NB", use_highly_variable=True, 
        use_layer="counts", use_rep="X_pca"
    )
    scglue.models.configure_dataset(
        atac, "NB", use_highly_variable=True, 
        use_rep="X_lsi"
    )
    
    # 3. Extract subgraph for highly variable features
    rna_hvf = rna.var.query("highly_variable").index
    atac_hvf = atac.var.query("highly_variable").index
    guidance_hvf = guidance.subgraph(chain(rna_hvf, atac_hvf)).copy()
    print(f"\n\n\nHVF subgraph extracted - RNA HVF: {len(rna_hvf)}, ATAC HVF: {len(atac_hvf)}\nHVF graph: {guidance_hvf.number_of_nodes()} nodes, {guidance_hvf.number_of_edges()} edges\n\n\n")
    
    # 4. Train GLUE model
    print(f"\n\n\n🤖 Training GLUE model...\n\n\n")
    glue = scglue.models.fit_SCGLUE(
        {"rna": rna, "atac": atac}, guidance_hvf,
        fit_kws={"directory": train_output_dir}
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
    
    # 7. Save results
    print(f"\n\n\n💾 Saving results...\n\n\n")
    model_path = f"{save_prefix}.dill"
    rna_emb_path = f"{save_prefix}-rna-emb.h5ad"
    atac_emb_path = f"{save_prefix}-atac-emb.h5ad"
    guidance_hvf_path = f"{save_prefix}-guidance-hvf.graphml.gz"
    
    glue.save(model_path)
    rna.write(rna_emb_path, compression="gzip")
    atac.write(atac_emb_path, compression="gzip")
    nx.write_graphml(guidance_hvf, guidance_hvf_path)
    
    # Check if integration is reliable
    is_reliable = min_consistency > consistency_threshold
    status = "✅ RELIABLE" if is_reliable else "❌ UNRELIABLE"
    print(f"\n\n\n📈 Integration Assessment:\nConsistency threshold: {consistency_threshold}\nMinimum consistency: {min_consistency:.4f}\nStatus: {status}\n\n\n")
    
    if not is_reliable:
        print("\n\n\n⚠️ Low consistency detected. Consider adjusting parameters or checking data quality.\n\n\n")
    
    print("\n\n\n🎉 GLUE training pipeline completed successfully!\n\n\n")
    
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

import os
import sys
import argparse
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns

def glue_visualize(rna_path, atac_path, output_dir=None):
    """
    Load processed RNA and ATAC data, create joint UMAP visualization
    
    Parameters:
    -----------
    rna_path : str
        Path to processed RNA h5ad file
    atac_path : str
        Path to processed ATAC h5ad file
    output_dir : str, optional
        Output directory. If None, uses directory of RNA file
    """
    
    # Load the processed data
    print("Loading RNA data...")
    rna = ad.read_h5ad(rna_path)
    
    print("Loading ATAC data...")
    atac = ad.read_h5ad(atac_path)
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(rna_path)
    
    # Check if scGLUE embeddings exist
    if "X_glue" not in rna.obsm:
        print("Error: X_glue embeddings not found in RNA data. Run scGLUE integration first.")
        return
    
    if "X_glue" not in atac.obsm:
        print("Error: X_glue embeddings not found in ATAC data. Run scGLUE integration first.")
        return
    
    print("Creating combined dataset...")
    # Combine the datasets for joint visualization
    combined = ad.concat([rna, atac])
    
    # Add modality information
    combined.obs['modality'] = ['RNA'] * rna.n_obs + ['ATAC'] * atac.n_obs
    
    print("Computing UMAP from scGLUE embeddings...")
    # Compute neighbors and UMAP using the scGLUE embeddings
    sc.pp.neighbors(combined, use_rep="X_glue", metric="cosine")
    sc.tl.umap(combined)
    
    # Set up plotting parameters
    sc.settings.set_figure_params(dpi=80, facecolor='white', figsize=(8, 6))
    
    print("Generating visualizations...")
    
    # Create visualization by modality
    plt.figure(figsize=(10, 8))
    sc.pl.umap(combined, color="modality", 
               title="scGLUE Integration: RNA vs ATAC",
               save=False, show=False)
    plt.tight_layout()
    modality_plot_path = os.path.join(output_dir, "scglue_umap_modality.png")
    plt.savefig(modality_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved modality plot: {modality_plot_path}")
    
    # Create visualization by cell type (if available)
    if "cell_type" in combined.obs.columns:
        plt.figure(figsize=(12, 8))
        sc.pl.umap(combined, color="cell_type", 
                   title="scGLUE Integration: Cell Types",
                   save=False, show=False, wspace=0.65)
        plt.tight_layout()
        celltype_plot_path = os.path.join(output_dir, "scglue_umap_celltype.png")
        plt.savefig(celltype_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved cell type plot: {celltype_plot_path}")
    
    # Create visualization by domain (if available)
    if "domain" in combined.obs.columns:
        plt.figure(figsize=(12, 8))
        sc.pl.umap(combined, color="domain", 
                   title="scGLUE Integration: Domains",
                   save=False, show=False, wspace=0.65)
        plt.tight_layout()
        domain_plot_path = os.path.join(output_dir, "scglue_umap_domain.png")
        plt.savefig(domain_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved domain plot: {domain_plot_path}")
    
    # Create visualization by sample/batch (if available)
    if "sample" in combined.obs.columns:
        plt.figure(figsize=(12, 8))
        sc.pl.umap(combined, color="sample", 
                   title="scGLUE Integration: Samples/Batches",
                   save=False, show=False, wspace=0.65)
        plt.tight_layout()
        sample_plot_path = os.path.join(output_dir, "scglue_umap_sample.png")
        plt.savefig(sample_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved sample plot: {sample_plot_path}")
    
    # Save the combined dataset with UMAP coordinates
    combined_output_path = os.path.join(output_dir, "scglue_combined_with_umap.h5ad")
    combined.write(combined_output_path)
    print(f"Saved combined dataset: {combined_output_path}")
    
    # Generate summary statistics
    print("\n=== Integration Summary ===")
    print(f"RNA cells: {rna.n_obs}")
    print(f"ATAC cells: {atac.n_obs}")
    print(f"Total cells: {combined.n_obs}")
    print(f"Available metadata columns: {list(combined.obs.columns)}")
    
    # Save summary to file
    summary_path = os.path.join(output_dir, "scglue_integration_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("scGLUE Integration Summary\n")
        f.write("=" * 30 + "\n")
        f.write(f"RNA cells: {rna.n_obs}\n")
        f.write(f"ATAC cells: {atac.n_obs}\n")
        f.write(f"Total cells: {combined.n_obs}\n")
        f.write(f"Available metadata: {', '.join(combined.obs.columns)}\n")
        f.write(f"Files generated:\n")
        f.write(f"  - {os.path.basename(modality_plot_path)}\n")
        if "cell_type" in combined.obs.columns:
            f.write(f"  - {os.path.basename(celltype_plot_path)}\n")
        if "domain" in combined.obs.columns:
            f.write(f"  - {os.path.basename(domain_plot_path)}\n")
        if "sample" in combined.obs.columns:
            f.write(f"  - {os.path.basename(sample_plot_path)}\n")
        f.write(f"  - {os.path.basename(combined_output_path)}\n")
    
    print(f"Saved summary: {summary_path}")
    print("\nVisualization complete!")

if __name__ == "__main__":
    # rna, atac, guidance = glue_preprocess_pipeline(
    #     rna_file="/Users/harry/Desktop/GenoDistance/Data/count_data.h5ad",
    #     atac_file="/Users/harry/Desktop/GenoDistance/Data/test_ATAC.h5ad", 
    #     ensembl_release=98,  # Latest human Ensembl release
    #     species="homo_sapiens",
    #     output_dir="/Users/harry/Desktop/GenoDistance/result/glue"
    # )
    # rna, atac, guidance = glue_preprocess_pipeline(
    #     rna_file="/users/hjiang/GenoDistance/Data/test_rna.h5ad",
    #     atac_file="/users/hjiang/GenoDistance/Data/test_ATAC.h5ad",
    #     rna_sample_meta_file="/users/hjiang/GenoDistance/Data/sample_data.csv",  # Optional
    #     atac_sample_meta_file="/users/hjiang/GenoDistance/Data/ATAC_Metadata.csv",  # Optional
    #     ensembl_release=98,
    #     species="homo_sapiens",
    #     output_dir="/users/hjiang/GenoDistance/result/glue"
    # )
    glue_train( preprocess_output_dir = "/users/hjiang/GenoDistance/result/glue", 
               save_prefix="glue", consistency_threshold=0.05)
    glue_visualize("/users/hjiang/GenoDistance/result/glue/glue-rna-emb.h5ad", "/users/hjiang/GenoDistance/result/glue/glue-atac-emb.h5ad", "/users/hjiang/GenoDistance/result/glue")