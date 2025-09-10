import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc

def harmonize_proportion_and_plot_from_path(
    pseudobulk_path: str,
    modality_col: str = "modality",
    output_basename: str = "proportion_embedding_harmony_by_modality",
    overwrite_h5ad: bool = False,
    verbose: bool = True,
):
    """
    Use harmony-pytorch to remove the 'modality' batch effect from the existing
    proportion embedding (X_DR_proportion). Keeps ALL components.

    Saves:
      - <output_basename>.png (HPC1 vs HPC2), next to the input .h5ad
      - updated .h5ad with:
          .uns["X_DR_proportion_harmony"]  (DataFrame, index=obs_names)
          .obsm["X_DR_proportion_harmony"] (ndarray in obs order)
    """
    if not os.path.exists(pseudobulk_path):
        raise FileNotFoundError(f"pseudobulk_path does not exist: {pseudobulk_path}")

    out_dir = os.path.dirname(os.path.abspath(pseudobulk_path))
    os.makedirs(out_dir, exist_ok=True)
    if verbose:
        print(f"[Harmony] Reading AnnData from: {pseudobulk_path}")

    ad = sc.read_h5ad(pseudobulk_path)

    if modality_col not in ad.obs.columns:
        raise KeyError(f"Missing '{modality_col}' in .obs.")

    # ---- Load existing proportion embedding ----
    if "X_DR_proportion" in ad.uns:
        prop_df = ad.uns["X_DR_proportion"]
        if not isinstance(prop_df, pd.DataFrame):
            prop_df = pd.DataFrame(prop_df, index=ad.obs_names)
    elif "X_DR_proportion" in ad.obsm:
        prop_df = pd.DataFrame(ad.obsm["X_DR_proportion"], index=ad.obs_names)
    else:
        raise KeyError("Could not find 'X_DR_proportion' in .uns or .obsm.")

    # ---- Case-insensitive alignment to ad.obs_names ----
    prop_norm_to_orig = pd.Series(
        prop_df.index.values,
        index=pd.Index(prop_df.index.astype(str).str.strip().str.lower()),
    ).groupby(level=0).first()

    obs_norm = pd.Index(pd.Series(ad.obs_names).astype(str).str.strip().str.lower())
    try:
        ordered_prop_names = [prop_norm_to_orig[nm] for nm in obs_norm]
    except KeyError as e:
        missing = [nm for nm in obs_norm if nm not in prop_norm_to_orig]
        raise KeyError(f"Some obs_names not found in X_DR_proportion (after normalization): {missing[:10]} ...") from e

    prop_df_aligned = prop_df.loc[ordered_prop_names]
    assert prop_df_aligned.shape[0] == ad.n_obs

    X = prop_df_aligned.values  # (n_samples, n_dims)
    # Ensure float dtype for torch
    X = np.asarray(X, dtype=np.float32)

    # ---- Harmony (pytorch) ----
    try:
        from harmony import harmonize  # harmony-pytorch
    except Exception as e:
        raise ImportError(
            "harmony-pytorch is required. Install via `pip install harmony-pytorch` "
            "or `conda install -c bioconda harmony-pytorch`."
        ) from e

    if verbose:
        print(f"[Harmony] Running harmony-pytorch on {X.shape} with batch_key='{modality_col}' ...")

    # harmonize expects N x d embedding and a DataFrame of metadata with matching row order
    Z = harmonize(X, ad.obs, batch_key=modality_col)  # returns N x d (numpy)

    # ---- Store back into AnnData ----
    harm_cols = [f"HPC{i+1}" for i in range(Z.shape[1])]
    harm_df = pd.DataFrame(Z, index=ad.obs_names, columns=harm_cols)
    ad.uns["X_DR_proportion"] = harm_df
    ad.obsm["X_DR_proportion"] = Z  # ndarray in obs order

    # ---- Plot HPC1 vs HPC2 ----
    if Z.shape[1] >= 2:
        if verbose:
            print("[Harmony] Plotting HPC1 vs HPC2 ...")
        groups = ad.obs[modality_col].astype(str)
        cats = groups.unique().tolist()

        fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=150)
        for cat in cats:
            idx = (groups == cat).values
            ax.scatter(Z[idx, 0], Z[idx, 1], s=28, alpha=0.9, label=str(cat), edgecolors="none")
        ax.set_xlabel("HPC1")
        ax.set_ylabel("HPC2")
        ax.set_title("Proportion Embedding (Harmony-corrected by modality)")
        ax.legend(title=modality_col, frameon=False, fontsize=8, title_fontsize=9)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        png_path = os.path.join(out_dir, f"{output_basename}.png")
        fig.savefig(png_path, bbox_inches="tight")
        plt.close(fig)
        if verbose:
            print(f"[Harmony] Saved plot → {png_path}")
    else:
        png_path = None
        if verbose:
            print("[Harmony] <2 components after correction; skipping plot.")

    # ---- Save updated AnnData ----
    if overwrite_h5ad:
        h5ad_path = os.path.join(out_dir, os.path.basename(pseudobulk_path))
    else:
        base, ext = os.path.splitext(os.path.basename(pseudobulk_path))
        h5ad_path = os.path.join(out_dir, f"{base}.harmony{ext or '.h5ad'}")

    try:
        sc.write(h5ad_path, ad)
        if verbose:
            print(f"[Harmony] Saved updated AnnData → {h5ad_path}")
    except Exception as e:
        if verbose:
            print(f"[Harmony] WARNING: Failed to save AnnData: {e}")

    return {"png": png_path, "h5ad": h5ad_path}


import os
import pandas as pd
import scanpy as sc

def debug_print_proportion_vs_obs(
    pseudobulk_path: str,
    modality_col: str = "modality",
    verbose: bool = True,
):
    """
    Debugging helper: load pseudobulk .h5ad, print all names from
    X_DR_proportion vs ad.obs_names (original + normalized) so we can
    see mismatches.
    """
    if not os.path.exists(pseudobulk_path):
        raise FileNotFoundError(f"pseudobulk_path does not exist: {pseudobulk_path}")

    ad = sc.read_h5ad(pseudobulk_path)

    if "X_DR_proportion" in ad.uns:
        prop_df = ad.uns["X_DR_proportion"]
        if not isinstance(prop_df, pd.DataFrame):
            prop_df = pd.DataFrame(prop_df, index=ad.obs_names)
    elif "X_DR_proportion" in ad.obsm:
        prop_df = pd.DataFrame(ad.obsm["X_DR_proportion"], index=ad.obs_names)
    else:
        raise KeyError("Could not find 'X_DR_proportion' in .uns or .obsm.")

    # Normalize indices
    prop_idx_norm = pd.Index(prop_df.index.astype(str).str.strip().str.lower())
    obs_idx_norm  = pd.Index(pd.Series(ad.obs_names).astype(str).str.strip().str.lower())

    # Debug printing
    print("=== DEBUG: X_DR_proportion index (original) ===")
    print(prop_df.index.tolist()[:10], "...")
    print("... total:", len(prop_df.index))
    print()

    print("=== DEBUG: ad.obs_names (original) ===")
    print(ad.obs_names[:10].tolist(), "...")
    print("... total:", len(ad.obs_names))
    print()

    print("=== DEBUG: X_DR_proportion index (normalized) ===")
    print(prop_idx_norm[:10].tolist(), "...")
    print()

    print("=== DEBUG: ad.obs_names (normalized) ===")
    print(obs_idx_norm[:10].tolist(), "...")
    print()

    common = prop_idx_norm.intersection(obs_idx_norm)
    print(f"=== DEBUG: Intersection size = {len(common)} ===")
    if len(common) > 0:
        print("Example common names:", list(common)[:10])

    return {
        "prop_index": prop_df.index.tolist(),
        "obs_names": ad.obs_names.tolist(),
        "prop_index_norm": prop_idx_norm.tolist(),
        "obs_names_norm": obs_idx_norm.tolist(),
        "common": list(common)
    }


if __name__ == "__main__":
    # debug_info = debug_print_proportion_vs_obs(
    #     "/dcs07/hongkai/data/harry/result/heart/multiomics/pseudobulk/pseudobulk_sample.h5ad"
    # )
    
    paths = harmonize_proportion_and_plot_from_path(
        pseudobulk_path="/dcs07/hongkai/data/harry/result/heart/multiomics/pseudobulk/pseudobulk_sample.h5ad",
        modality_col="modality",
        output_basename="proportion_embedding_harmony_by_modality",
        overwrite_h5ad=False,
        verbose=True,
    )
    print(paths)
