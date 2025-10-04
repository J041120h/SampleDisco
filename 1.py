def qc_var_checker(
    adata,
    gene_symbol_cols=("gene_symbol","gene_symbols","symbol","Gene","name"),
    expected_qc_cols=("mt","ribo","hb","MT"),  # include 'MT' for legacy code that expects uppercase
    peek_n=10
):
    """
    Check that adata.var contains the expected QC boolean columns and that
    gene names are in a format where mito/ribo/hb flags can be derived.
    Does NOT modify adata.

    Returns:
        report (dict): {
            'present_qc_cols': {col: bool, ...},
            'missing_qc_cols': [ ... ],
            'gene_symbol_source': 'var[col]' or 'var_names',
            'symbols_look_like': 'human_symbols' | 'mouse_symbols' | 'ensembl' | 'unknown',
            'counts': {
                'total_genes': int,
                'starts_MT-': int,
                'starts_mt-': int,
                'starts_RPS/RPL': int,
                'starts_HB': int
            },
            'suggested_qc_vars_for_scanpy': ['mt','ribo','hb'] or subset,
            'notes': [ ... ]
        }
    """
    import pandas as pd

    report = {}
    var = adata.var
    total_genes = var.shape[0]
    report['counts'] = {'total_genes': int(total_genes)}

    # 1) Which QC cols are present?
    present = {col: (col in var.columns and var[col].dtype == bool) for col in expected_qc_cols}
    report['present_qc_cols'] = present
    report['missing_qc_cols'] = [c for c, ok in present.items() if not ok]

    # 2) Choose a gene symbol source for *inspection only*
    symbol_src = None
    for c in gene_symbol_cols:
        if c in var.columns and pd.api.types.is_string_dtype(var[c]):
            symbol_src = f"var[{c}]"
            genes = var[c].astype(str)
            break
    if symbol_src is None:
        symbol_src = "var_names"
        genes = pd.Index(adata.var_names.astype(str))

    report['gene_symbol_source'] = symbol_src

    # 3) Quick heuristics on gene naming
    gl = genes.str.lower()
    starts_MTdash = genes.str.startswith("MT-").sum()
    starts_mtdash = gl.str.startswith("mt-").sum()
    starts_RPSRPL = (genes.str.startswith("RPS") | genes.str.startswith("RPL") |
                     gl.str.startswith("rps")  | gl.str.startswith("rpl")).sum()
    starts_HB = (genes.str.startswith("HB") | gl.str.startswith("hb")).sum()

    report['counts'].update({
        'starts_MT-': int(starts_MTdash),
        'starts_mt-': int(starts_mtdash),
        'starts_RPS/RPL': int(starts_RPSRPL),
        'starts_HB': int(starts_HB),
    })

    # Ensembl heuristic
    looks_ensembl = gl.str.startswith(("ensg","ensmusg","fbgn","spac","spbc")).mean() > 0.2
    if looks_ensembl:
        style = "ensembl"
    elif starts_MTdash > 5:
        style = "human_symbols"
    elif starts_mtdash > 5:
        style = "mouse_symbols"
    else:
        style = "unknown"

    report['symbols_look_like'] = style

    # 4) Suggest qc_vars to pass to scanpy (based on what’s present)
    # Prefer lower-case names that Scanpy examples use.
    suggested = []
    if present.get('mt', False) or present.get('MT', False):
        suggested.append('mt' if present.get('mt', False) else 'MT')
    if present.get('ribo', False):
        suggested.append('ribo')
    if present.get('hb', False):
        suggested.append('hb')
    report['suggested_qc_vars_for_scanpy'] = suggested

    # 5) Notes
    notes = []
    if any(not ok for ok in present.values()):
        notes.append("Some expected QC boolean columns are missing; Scanpy will raise KeyError if referenced.")
    if style == "ensembl":
        notes.append("Gene IDs look like Ensembl. Prefix-based mito/ribo/hb detection won’t work unless you also have a gene-symbol column.")
    if style in ("human_symbols","mouse_symbols"):
        notes.append("Gene symbols look like standard symbols; if QC columns are missing, you can derive them from symbol prefixes.")
    if not suggested:
        notes.append("No valid qc_vars detected to pass to sc.pp.calculate_qc_metrics.")
    # Peek a few names
    head = genes[:peek_n].tolist()
    notes.append(f"First {min(peek_n, len(head))} gene names from {symbol_src}: {head}")
    report['notes'] = notes

    # 6) Pretty print (optional)
    print("=== QC Var Checker ===")
    print(f"Total genes: {total_genes}")
    print("Present QC columns (bool dtype expected):")
    for c in expected_qc_cols:
        print(f"  - {c}: {'OK' if present[c] else 'MISSING or non-bool'}")
    print(f"Gene symbol source inspected: {symbol_src}")
    print(f"Symbols look like: {style}")
    print("Heuristic counts:")
    for k, v in report['counts'].items():
        if k != 'total_genes':
            print(f"  - {k}: {v}")
    print(f"Suggested qc_vars for Scanpy: {suggested if suggested else 'None'}")
    if notes:
        print("Notes:")
        for n in notes:
            print(f"  - {n}")
    print("======================")

    return report

if __name__ == "__main__":
    import anndata
    # adata = anndata.read_h5ad("/dcs07/hongkai/data/harry/result/Benchmark/multiomics/preprocess/atac_rna_integrated.h5ad")
    # adata = anndata.read_h5ad('/dcl01/hongkai/data/data/hjiang/Data/paired/rna/all.h5ad')
    # report = qc_var_checker(adata)
    import time
    time(5)