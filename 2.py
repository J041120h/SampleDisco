import scanpy as sc
import pandas as pd

h5ad_path = "/dcs07/hongkai/data/harry/result/long_covid/analysis/preprocess/adata_sample.h5ad"
annotation_csv_path = "/dcs07/hongkai/data/harry/result/long_covid/analysis/clustering/AI_celltype_annotations.csv"

adata = sc.read_h5ad(h5ad_path)
anno = pd.read_csv(annotation_csv_path)

if "cell_type" not in adata.obs.columns:
    raise ValueError("Column 'cell_type' not found in adata.obs")

if "leiden_1" not in adata.obs.columns:
    raise ValueError("Column 'leiden_1' not found in adata.obs")

required_cols = ["Resolution", "Cluster", "Identified Cell Type"]
missing_cols = [col for col in required_cols if col not in anno.columns]
if missing_cols:
    raise ValueError(f"Missing required columns in annotation CSV: {missing_cols}")

adata.obs.rename(columns={"cell_type": "cell_type_0.25"}, inplace=True)

anno_res1 = anno[pd.to_numeric(anno["Resolution"], errors="coerce") == 1.0].copy()
if anno_res1.empty:
    raise ValueError("No rows found in annotation CSV with Resolution == 1")

anno_res1["Cluster"] = anno_res1["Cluster"].astype(str).str.strip()
anno_res1["Identified Cell Type"] = (
    anno_res1["Identified Cell Type"]
    .astype(str)
    .str.strip()
    .str.rstrip(",")
)

cluster_to_celltype = dict(zip(anno_res1["Cluster"], anno_res1["Identified Cell Type"]))

adata.obs["leiden_1"] = adata.obs["leiden_1"].astype(str).str.strip()
adata.obs["cell_type_1"] = adata.obs["leiden_1"].map(cluster_to_celltype)

missing_cluster_labels = sorted(
    set(adata.obs["leiden_1"].unique()) - set(cluster_to_celltype.keys())
)
if missing_cluster_labels:
    raise ValueError(
        f"The following leiden_1 clusters have no matching annotation: {missing_cluster_labels}"
    )

adata.obs["cell_type_1"] = pd.Categorical(adata.obs["cell_type_1"])

adata.write_h5ad(h5ad_path)

print("Finished updating AnnData.")
print("Renamed 'cell_type' to 'cell_type_0.25'")
print("Created 'cell_type_1' from 'leiden_1' using 'Identified Cell Type'")
print(f"Overwrote original file: {h5ad_path}")