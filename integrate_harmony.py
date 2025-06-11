import os
import scanpy as sc
from pseudo_adata import *
from PCA import *
import time
import contextlib
import io

def integrate_preprocess(
    output_dir,
    h5ad_path = None,
    sample_column = 'sample',
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

    # Append 'harmony' subdirectory
    output_dir = os.path.join(output_dir, 'preprocess')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print("Automatically generating harmony subdirectory")

    if doublet and min_cells_sample < 30:
        min_cells_sample = 30
        print("Minimum dimension requested by scrublet is 30, raise sample standard accordingly")
    # 1. Read the raw count data from an existing H5AD
    if verbose:
        print('=== Read input dataset ===')
    adata = sc.read_h5ad(h5ad_path)
    if verbose:
        print(f'Dimension of raw data (cells x genes): {adata.shape[0]} x {adata.shape[1]}')

    sc.pp.filter_cells(adata, min_genes=min_features)
    sc.pp.filter_genes(adata, min_cells=min_cell_gene)
    if verbose:
        print(f"After basic filtering -- Cells remaining: {adata.n_obs}, Genes remaining: {adata.n_vars}")

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
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            sc.pp.scrublet(adata, batch_key=sample_column)
            adata = adata[~adata.obs['predicted_doublet']].copy()
    
    adata.raw = adata.copy()
    if verbose:
        print("Preprocessing complete!")

    sc.write(os.path.join(h5ad_path), adata)
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print execution time
    if verbose:
        print(f"Function execution time: {elapsed_time:.2f} seconds")

    return adata

if __name__ == "__main__":
    # adata = integrate_preprocess(
    #     output_dir = "/users/hjiang/GenoDistance/result",
    #     sample_column = 'sample',
    #     min_cells_sample=1,
    #     min_cell_gene=10,
    #     min_features=500,
    #     pct_mito_cutoff=20,
    #     exclude_genes=None,
    #     doublet=True, 
    #     verbose=True
    # )
    adata = sc.read_h5ad("/users/hjiang/GenoDistance/result/integration/glue/atac_rna_integrated.h5ad")
    atac_pseudobulk_df, pseudobulk_adata = compute_pseudobulk_adata(
                adata=adata,
                batch_col='batch',
                sample_col='sample',
                output_dir="/users/hjiang/GenoDistance/result/"
            )

    process_anndata_with_pca(
        adata=adata,
        pseudobulk=atac_pseudobulk_df,
        pseudobulk_anndata = pseudobulk_adata,
        sample_col = "sample",
        output_dir= "/users/hjiang/GenoDistance/result/integration",
        integrated_data = True
    )