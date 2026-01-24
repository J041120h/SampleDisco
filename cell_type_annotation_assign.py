#!/usr/bin/env python3
"""
Update pseudobulk var_names to use annotated cell types instead of Leiden clusters.

We assume pseudobulk .var_names are of the form:
    "<original_leiden_label> - <gene_symbol>"

We use the mapping stored in:
    annotated_adata.uns["celltypist_label_mapping"]
which is a dict: {original_leiden_label -> celltypist_label}

Example:
    "1 - IGHD"  +  {"1": "CD14 Mono"}  -->  "CD14 Mono - IGHD"
"""

import os
import scanpy as sc
import pandas as pd


ANNOTATED_H5AD = "/dcs07/hongkai/data/harry/result/multi_omics_unpaired_test/multiomics/preprocess/adata_sample.h5ad"
PSEUDOBULK_H5AD = "/dcs07/hongkai/data/harry/result/multi_omics_unpaired_test/multiomics/pseudobulk/pseudobulk_sample.h5ad"


def update_pseudobulk_celltype_in_varnames(
    annotated_h5ad_path: str,
    pseudobulk_h5ad_path: str,
    output_path: str | None = None,
    separator: str = " - ",
):
    """
    Replace the Leiden-derived cell-type prefix in pseudobulk var_names
    with the annotated CellTypist cell type using a mapping stored in
    annotated_adata.uns["celltypist_label_mapping"].

    Parameters
    ----------
    annotated_h5ad_path : str
        Path to annotated single-cell AnnData (with celltypist_label_mapping in .uns).
    pseudobulk_h5ad_path : str
        Path to pseudobulk AnnData to update.
    output_path : str or None
        Where to write the updated pseudobulk h5ad.
        If None, overwrite `pseudobulk_h5ad_path` in place.
    separator : str
        String separating the prefix and gene symbol in var_names
        (default: " - ").
    """
    print(f"[INFO] Loading annotated AnnData from: {annotated_h5ad_path}")
    adata_annot = sc.read_h5ad(annotated_h5ad_path)

    if "celltypist_label_mapping" not in adata_annot.uns:
        raise KeyError(
            "celltypist_label_mapping not found in adata.uns. "
            "Make sure you ran the annotation pipeline that stores "
            "this mapping in `adata.uns['celltypist_label_mapping']`."
        )

    # Mapping: original_leiden_label -> annotated_cell_type
    mapping_raw = adata_annot.uns["celltypist_label_mapping"]
    # Make sure keys/values are strings
    label_mapping = {str(k): str(v) for k, v in mapping_raw.items()}
    print(f"[INFO] Loaded label mapping with {len(label_mapping)} entries from adata.uns['celltypist_label_mapping'].")

    print(f"[INFO] Loading pseudobulk AnnData from: {pseudobulk_h5ad_path}")
    adata_pb = sc.read_h5ad(pseudobulk_h5ad_path)

    old_var = pd.Index(adata_pb.var_names.astype(str))
    n_var = len(old_var)
    print(f"[INFO] Pseudobulk has {n_var} features (var). Example var_names:")
    print("       ", list(old_var[:10]))

    new_var = []
    replaced_count = 0
    no_mapping_prefixes: set[str] = set()
    malformed_names: list[str] = []

    for name in old_var:
        # Expect pattern "<prefix> - <gene>"
        if separator not in name:
            # Keep unchanged, but log that it's not in the expected pattern
            malformed_names.append(name)
            new_var.append(name)
            continue

        prefix, gene = name.split(separator, 1)
        prefix_stripped = prefix.strip()

        if prefix_stripped in label_mapping:
            new_label = label_mapping[prefix_stripped]
            new_name = f"{new_label}{separator}{gene}"
            replaced_count += 1
        else:
            # No mapping for this prefix – keep as-is, but remember which
            new_name = name
            no_mapping_prefixes.add(prefix_stripped)

        new_var.append(new_name)

    adata_pb.var_names = new_var
    adata_pb.var_names_make_unique()

    if output_path is None:
        output_path = pseudobulk_h5ad_path

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(
        f"[INFO] Updated {replaced_count} / {n_var} var_names "
        f"({replaced_count / max(1, n_var) * 100:.1f}%)."
    )
    if no_mapping_prefixes:
        print("[WARN] Some prefixes had no mapping and were left unchanged. Examples:")
        for p in sorted(list(no_mapping_prefixes))[:20]:
            print(f"       - '{p}'")
        if len(no_mapping_prefixes) > 20:
            print(f"       ... and {len(no_mapping_prefixes) - 20} more")

    if malformed_names:
        print("[WARN] Some var_names did not contain the expected separator "
              f"'{separator}' and were left unchanged. Examples:")
        for n in malformed_names[:20]:
            print(f"       - '{n}'")
        if len(malformed_names) > 20:
            print(f"       ... and {len(malformed_names) - 20} more")

    print(f"[INFO] Writing updated pseudobulk AnnData to: {output_path}")
    adata_pb.write_h5ad(output_path, compression="gzip")
    print("[INFO] Done.")


if __name__ == "__main__":
    update_pseudobulk_celltype_in_varnames(
        annotated_h5ad_path=ANNOTATED_H5AD,
        pseudobulk_h5ad_path=PSEUDOBULK_H5AD,
        output_path=None,  # overwrite in place
    )

