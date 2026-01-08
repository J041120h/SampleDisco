import pandas as pd
import matplotlib.pyplot as plt
import os
def debug_embedding_check(meta_csv_path: str, embedding_csv_path: str, label_col: str = "disease_state"):
    """
    Debug version of the benchmark that inspects the metadata and embedding files
    to check for alignment issues and label problems.

    Parameters:
    ----------
    meta_csv_path : str
        Path to metadata CSV file
    embedding_csv_path : str
        Path to embedding/coordinates CSV file
    label_col : str
        Name of the column used for color coding in visualizations
    """
    
    print(f"[DEBUG] Loading metadata from: {meta_csv_path}")
    try:
        meta_df = pd.read_csv(meta_csv_path)
        print(f"[DEBUG] Metadata loaded successfully. Shape: {meta_df.shape}, Columns: {meta_df.columns.tolist()}")
    except Exception as e:
        print(f"[ERROR] Failed to load metadata CSV: {e}")
        return
    
    print(f"[DEBUG] Checking if '{label_col}' exists in metadata...")
    if label_col not in meta_df.columns:
        print(f"[ERROR] '{label_col}' column not found in metadata.")
        return
    
    print(f"[DEBUG] Inspecting unique values in '{label_col}' column...")
    print(meta_df[label_col].value_counts())
    
    print(f"[DEBUG] Loading embeddings from: {embedding_csv_path}")
    try:
        embedding_df = pd.read_csv(embedding_csv_path, index_col=0)
        print(f"[DEBUG] Embedding loaded successfully. Shape: {embedding_df.shape}")
        print(f"[DEBUG] First 3 embedding indices: {embedding_df.index[:3]}")
    except Exception as e:
        print(f"[ERROR] Failed to load embedding CSV: {e}")
        return
    
    # Normalize sample IDs in both metadata and embedding files
    print(f"[DEBUG] Normalizing sample IDs to lowercase for case-insensitive matching...")
    meta_df['sample'] = meta_df['sample'].astype(str).str.lower().str.strip()
    embedding_df.index = embedding_df.index.astype(str).str.lower().str.strip()

    # Check overlap between metadata and embedding sample IDs
    common_ids = meta_df['sample'].index.intersection(embedding_df.index)
    only_meta = meta_df['sample'].index.difference(embedding_df.index)
    only_emb = embedding_df.index.difference(meta_df['sample'].index)
    
    print(f"[DEBUG] Overlap check:")
    print(f"  Common IDs: {len(common_ids)}")
    print(f"  Only in metadata: {len(only_meta)}")
    print(f"  Only in embedding: {len(only_emb)}")
    print(f"[DEBUG] Example meta-only IDs: {only_meta[:5]}")
    print(f"[DEBUG] Example embed-only IDs: {only_emb[:5]}")
    
    if len(common_ids) == 0:
        print("[ERROR] No common sample IDs between metadata and embedding!")
        return

    # Subset both metadata and embeddings to common IDs
    print(f"[DEBUG] Subsetting both metadata and embedding to common IDs...")
    meta_df = meta_df[meta_df['sample'].isin(common_ids)]
    embedding_df = embedding_df.loc[common_ids]

    print(f"[DEBUG] After alignment - Meta shape: {meta_df.shape}, Embedding shape: {embedding_df.shape}")
    
    # Check the label column and its values
    label_values_raw = meta_df[label_col]
    print(f"[DEBUG] Checking label column values:")
    print(f"  Unique values in '{label_col}': {label_values_raw.unique()}")
    
    # Check if label column is numeric or categorical
    label_numeric = pd.to_numeric(label_values_raw, errors='coerce')
    is_numerical = label_numeric.notna().sum() / len(label_numeric) > 0.5
    
    if is_numerical:
        print(f"[DEBUG] {label_col} is numerical. Min: {label_numeric.min()}, Max: {label_numeric.max()}")
    else:
        print(f"[DEBUG] {label_col} is categorical.")
    
    # Perform a PCA on the embeddings to see if they are well-conditioned
    print(f"[DEBUG] Performing PCA to inspect embedding...")
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    embedding_2d = pca.fit_transform(embedding_df)
    variance_explained = pca.explained_variance_ratio_
    
    print(f"[DEBUG] PCA completed. Explained variance: {variance_explained}")
    
    # Visualize the first two components with the label column
    print(f"[DEBUG] Creating PCA visualization...")
    plt.figure(figsize=(8, 6))
    
    # Plot if numerical or categorical
    if is_numerical:
        plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=label_numeric, cmap='viridis', edgecolors='black', alpha=0.8)
        plt.colorbar(label=label_col)
    else:
        label_to_num = {lbl: i for i, lbl in enumerate(label_values_raw.unique())}
        label_colors = [label_to_num[lbl] for lbl in label_values_raw]
        plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=label_colors, cmap='tab10', edgecolors='black', alpha=0.8)
        plt.colorbar(label=label_col)
    
    plt.xlabel(f"PC1 ({variance_explained[0]:.2%})")
    plt.ylabel(f"PC2 ({variance_explained[1]:.2%})")
    plt.title(f"Embedding PCA colored by {label_col}")
    plt.show()


# Debug using one of the files in the logs
meta_csv_path = "/dcs07/hongkai/data/harry/result/multi_omics_heart/data/multi_omics_heart_sample_meta.csv"
embedding_csv_path = "/dcs07/hongkai/data/harry/result/Benchmark_heart_rna/rna/embeddings/sample_expression_embedding.csv"

debug_embedding_check(meta_csv_path, embedding_csv_path, label_col="disease_state")
