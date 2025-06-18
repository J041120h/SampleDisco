import anndata as ad
import pandas as pd
from sklearn.cross_decomposition import CCA
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
import time
from pseudobulk import compute_pseudobulk_dataframes
from DR import process_anndata_with_pca
from CellType import cell_types, cell_type_assign
from linux.CellType_linux import cell_types_linux, cell_type_assign_linux

def find_optimal_cell_resolution_linux(
    AnnData_cell,
    AnnData_sample,
    output_dir,
    summary_sample_csv_path,
    AnnData_sample_path,
    column,
    sev_col: str = "sev.level",
    sample_col: str = "sample"
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.cross_decomposition import CCA
    import os
    import time

    start_time = time.time()
    score_counter = {}

    for resolution in np.arange(0.01, 1.01, 0.1):
        print(f"\n\nTesting resolution: {resolution}\n\n")

        cell_types_linux(
            AnnData_cell,
            cell_column='cell_type',
            Save=False,
            output_dir=output_dir,
            cluster_resolution=resolution,
            markers=None,
            method='average',
            metric='euclidean',
            distance_mode='centroid',
            num_PCs=20,
            verbose=False
        )

        cell_type_assign_linux(AnnData_cell, AnnData_sample, Save=False, output_dir=output_dir, verbose=False)

        pseudobulk = compute_pseudobulk_dataframes(AnnData_sample, 'batch', sample_col, 'cell_type', output_dir)

        process_anndata_with_pca(
            adata=AnnData_sample,
            pseudobulk=pseudobulk,
            output_dir=output_dir,
            sample_col=sample_col,
            not_save=True,
            verbose=False
        )

        pca_coords = AnnData_sample.uns[column]
        pca_coords_2d = pca_coords.iloc[:, :2].values

        samples = AnnData_sample.obs[sample_col].unique()
        sev_levels_2d = load_severity_levels(summary_sample_csv_path, samples, sample_col=sample_col, sev_col=sev_col)

        cca = CCA(n_components=1)
        cca.fit(pca_coords_2d, sev_levels_2d)
        U, V = cca.transform(pca_coords_2d, sev_levels_2d)
        score = np.corrcoef(U[:, 0], V[:, 0])[0, 1]

        print(score)
        score_counter[resolution] = score

        AnnData_cell.obs.drop(columns=['cell_type'], errors='ignore', inplace=True)
        AnnData_sample.obs.drop(columns=['cell_type'], errors='ignore', inplace=True)

    best_resolution = max(score_counter, key=score_counter.get)
    print(f"Best resolution from first pass: {best_resolution}")
    fine_score_counter = {}

    for resolution in np.arange(max(0.1, best_resolution - 0.02), min(1.0, best_resolution + 0.02) + 0.01, 0.01):
        cell_types_linux(
            AnnData_cell,
            cell_column='cell_type',
            Save=False,
            output_dir=output_dir,
            cluster_resolution=resolution,
            markers=None,
            method='average',
            metric='euclidean',
            distance_mode='centroid',
            num_PCs=20,
            verbose=False
        )

        cell_type_assign_linux(AnnData_cell, AnnData_sample, Save=False, output_dir=output_dir, verbose=False)

        pseudobulk = compute_pseudobulk_dataframes(AnnData_sample, 'batch', sample_col, 'cell_type', output_dir)

        process_anndata_with_pca(
            adata=AnnData_sample,
            pseudobulk=pseudobulk,
            output_dir=output_dir,
            sample_col=sample_col,
            not_save=True,
            verbose=False
        )

        pca_coords = AnnData_sample.uns[column]
        pca_coords_2d = pca_coords.iloc[:, :2].values

        samples = AnnData_sample.obs[sample_col].unique()
        sev_levels_2d = load_severity_levels(summary_sample_csv_path, samples, sample_col=sample_col, sev_col=sev_col)

        cca = CCA(n_components=1)
        cca.fit(pca_coords_2d, sev_levels_2d)
        U, V = cca.transform(pca_coords_2d, sev_levels_2d)
        score = np.corrcoef(U[:, 0], V[:, 0])[0, 1]

        print(f"Fine-tuned Resolution {resolution:.2f}: Score {score}")
        fine_score_counter[resolution] = score

        AnnData_cell.obs.drop(columns=['cell_type'], errors='ignore', inplace=True)
        AnnData_sample.obs.drop(columns=['cell_type'], errors='ignore', inplace=True)

    final_best_resolution = max(fine_score_counter, key=fine_score_counter.get)
    print(f"Final best resolution: {final_best_resolution}")

    df_coarse = pd.DataFrame(score_counter.items(), columns=["resolution", "score"])
    df_fine = pd.DataFrame(fine_score_counter.items(), columns=["resolution", "score"])
    df_results = pd.concat([df_coarse, df_fine], ignore_index=True)

    output_dir = os.path.join(output_dir, "CCA_test")
    os.makedirs(output_dir, exist_ok=True)

    to_csv_path = os.path.join(output_dir, f"resolution_scores_{column}.csv")
    df_results.to_csv(to_csv_path, index=False)

    plt.figure(figsize=(8, 6))
    plt.plot(df_results["resolution"], df_results["score"], marker='o', linestyle='None', color='b', label="CCA Score")
    plt.xlabel("Resolution")
    plt.ylabel("CCA Score")
    plt.title("Resolution vs. CCA Score")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(output_dir, f"resolution_vs_cca_score_{column}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Resolution vs. CCA Score plot saved as '{plot_path}'.")
    print(f"Resolution scores saved to '{to_csv_path}'.")
    print("All data saved locally.")

    print(f"\n[Find Optimal Resolution] Total runtime: {time.time() - start_time:.2f} seconds\n")

    return final_best_resolution