import scanpy as sc

def print_dataset_info(rna_file: str, atac_file: str):
    # Load RNA and ATAC data
    try:
        rna = sc.read(rna_file)
        atac = sc.read(atac_file)
        print(f"✅ Successfully loaded RNA and ATAC files")
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return

    # Print basic information about RNA and ATAC data
    print("\n🔍 RNA Data Information")
    print(f"   RNA Shape: {rna.shape}")
    print(f"   RNA var_names (first 10 features):\n{rna.var_names[:10]}")
    print(f"   RNA var columns: {list(rna.var.columns)}")
    
    # Check for strand information in RNA
    if 'strand' in rna.var.columns:
        print(f"   RNA strand information:\n{rna.var['strand'].value_counts()}")
    else:
        print("   RNA does not contain 'strand' column")
    
    print("\n🔍 ATAC Data Information")
    print(f"   ATAC Shape: {atac.shape}")
    print(f"   ATAC var_names (first 10 features):\n{atac.var_names[:10]}")
    print(f"   ATAC var columns: {list(atac.var.columns)}")
    
    # Check for strand information in ATAC
    if 'strand' in atac.var.columns:
        print(f"   ATAC strand information:\n{atac.var['strand'].value_counts()}")
    else:
        print("   ATAC does not contain 'strand' column")

# Run the debug function on your files
rna_file = '/dcs07/hongkai/data/harry/result/multi_omics_heart/data/rna_raw.h5ad'
atac_file = '/dcs07/hongkai/data/harry/result/multi_omics_heart/data/atac_raw.h5ad'

print_dataset_info(rna_file, atac_file)
