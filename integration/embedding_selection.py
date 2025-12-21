#!/usr/bin/env python3
"""
Cross-Modal Embedding Evaluation for Multiomics Pseudobulk

Evaluates embeddings by checking: For each ATAC sample, does its closest RNA sample
in embedding space have correlated gene expression/activity profiles?

Key idea: Good embedding -> ATAC samples are close to RNA samples with similar biology.
"""

import os
import re
import signal
import io
import contextlib
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse, csr_matrix
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple, Optional, Union

warnings.filterwarnings('ignore')


# =============================================================================
# USER CONFIGURATION - Set your paths and parameters here
# =============================================================================
CELL_ADATA_PATH = "/dcs07/hongkai/data/harry/result/Benchmark_multiomics/adata_cell.h5ad"
PSEUDOBULK_PATH = "/dcs07/hongkai/data/harry/result/multi_omics_SD/multiomics/pseudobulk/pseudobulk_sample.h5ad"
OUTPUT_DIR = "/dcs07/hongkai/data/harry/result/multi_omics_SD/embedding_selection"

SAMPLE_COL = "sample"
CELLTYPE_COL = "cell_type"
BATCH_COL = None

SELECTION_METRIC = "avg_nearest_rna_corr"  # Options: avg_nearest_rna_corr, avg_knn_corr, paired_corr
K_NEIGHBORS = 3  # Number of nearest RNA neighbors to consider
COMBAT_TIMEOUT = 20.0  # Timeout for ComBat in seconds
VERBOSE = True


# =============================================================================
# Sample Name Utilities
# =============================================================================

def normalize_sample_name(name: str) -> str:
    """Remove modality suffixes (_RNA, _ATAC) from sample name."""
    return re.sub(r'[_-]?(RNA|ATAC|rna|atac)$', '', str(name).strip())


def get_modality(name: str) -> str:
    """Extract modality (RNA or ATAC) from sample name."""
    if '_RNA' in name or '-RNA' in name or name.endswith('RNA'):
        return 'RNA'
    elif '_ATAC' in name or '-ATAC' in name or name.endswith('ATAC'):
        return 'ATAC'
    return 'unknown'


def get_sample_modality_info(sample_names: List[str]) -> Tuple[List[int], List[int], Dict[str, str]]:
    """
    Separate ATAC and RNA sample indices and build base name mapping.
    
    Returns:
        atac_indices: List of indices for ATAC samples
        rna_indices: List of indices for RNA samples
        base_to_samples: Dict mapping base name -> {modality: index}
    """
    atac_indices = []
    rna_indices = []
    base_to_samples = {}
    
    for i, name in enumerate(sample_names):
        modality = get_modality(name)
        base_name = normalize_sample_name(name)
        
        if modality == 'ATAC':
            atac_indices.append(i)
        elif modality == 'RNA':
            rna_indices.append(i)
        
        if base_name not in base_to_samples:
            base_to_samples[base_name] = {}
        base_to_samples[base_name][modality] = i
    
    return atac_indices, rna_indices, base_to_samples


# =============================================================================
# Gene Feature Parsing
# =============================================================================

def parse_gene_features(var_names: pd.Index) -> Dict[str, List[str]]:
    """Parse gene names like '1 - ABCB11' into {cell_type: [gene_names]}."""
    ct_genes = {}
    for feat in var_names:
        feat_str = str(feat)
        if ' - ' in feat_str:
            parts = feat_str.split(' - ', 1)
            ct, gene = parts[0].strip(), parts[1].strip()
            ct_genes.setdefault(ct, []).append(gene)
        else:
            ct_genes.setdefault('unknown', []).append(feat_str)
    return ct_genes


# =============================================================================
# Batch Correction (matching original pseudobulk code)
# =============================================================================

def _try_combat(adata: sc.AnnData, batch_col: str, timeout: float, verbose: bool) -> bool:
    """Attempt ComBat correction with timeout."""
    def timeout_handler(signum, frame):
        raise TimeoutError("ComBat timed out")

    try:
        old_handler = signal.signal(signal.SIGALRM, timeout_handler) if timeout else None
        if timeout:
            signal.alarm(int(timeout))

        try:
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                sc.pp.combat(adata, key=batch_col, inplace=True)
            if verbose:
                print(f"    Applied ComBat")
            return True
        finally:
            if timeout:
                signal.alarm(0)
                if old_handler:
                    signal.signal(signal.SIGALRM, old_handler)

    except (TimeoutError, Exception) as e:
        if verbose:
            print(f"    ComBat failed: {type(e).__name__}: {e}")
        return False


def _try_limma(adata: sc.AnnData, batch_col: str, verbose: bool) -> bool:
    """Attempt limma-style regression correction."""
    try:
        from utils.limma import limma
        X = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)
        remove_formula = f'~ Q("{batch_col}")'
        adata.X = limma(
            pheno=adata.obs, exprs=X,
            covariate_formula="1", design_formula=remove_formula,
            rcond=1e-8, verbose=False
        )
        if verbose:
            print(f"    Applied limma")
        return True
    except Exception as e:
        if verbose:
            print(f"    Limma failed: {type(e).__name__}: {e}")
        return False


def apply_batch_correction(adata: sc.AnnData, batch_col: str, combat_timeout: float, verbose: bool = True) -> sc.AnnData:
    """Apply batch correction using ComBat with limma fallback."""
    if batch_col not in adata.obs.columns:
        if verbose:
            print(f"[BatchCorrect] Column '{batch_col}' not found, skipping")
        return adata
    
    adata.obs[batch_col] = adata.obs[batch_col].fillna('Unknown')
    batch_counts = adata.obs[batch_col].value_counts()
    n_batches = len(batch_counts)
    
    if verbose:
        print(f"[BatchCorrect] {n_batches} batches, sizes: {batch_counts.min()}-{batch_counts.max()}")
    
    if n_batches <= 1:
        if verbose:
            print("[BatchCorrect] Only 1 batch, skipping")
        return adata
    
    adata = adata.copy()
    
    if batch_counts.min() >= 2:
        if _try_combat(adata, batch_col, combat_timeout, verbose):
            return adata
    else:
        if verbose:
            print(f"    Skipping ComBat: batches <2 samples")
    
    if _try_limma(adata, batch_col, verbose):
        return adata
    
    # Final fallback: mean centering
    if verbose:
        print("[BatchCorrect] Using mean centering fallback")
    X = adata.X.toarray() if issparse(adata.X) else np.array(adata.X)
    global_mean = np.nanmean(X, axis=0)
    for batch in adata.obs[batch_col].unique():
        mask = adata.obs[batch_col] == batch
        if mask.sum() > 0:
            batch_mean = np.nanmean(X[mask], axis=0)
            X[mask] = X[mask] - batch_mean + global_mean
    adata.X = np.nan_to_num(X, 0)
    return adata


# =============================================================================
# Pseudobulk Aggregation by Modality
# =============================================================================

def aggregate_pseudobulk_by_modality(
    adata: sc.AnnData,
    target_genes: Dict[str, List[str]],
    sample_col: str,
    celltype_col: str,
    batch_col: Optional[str],
    modality: str,
    verbose: bool
) -> sc.AnnData:
    """Aggregate cells of a specific modality into pseudobulk."""
    
    # Filter to specified modality
    if 'modality' in adata.obs.columns:
        mod_mask = adata.obs['modality'] == modality
        adata_mod = adata[mod_mask].copy()
        if verbose:
            print(f"[Aggregate {modality}] Filtered to {modality}: {adata_mod.n_obs} cells")
    else:
        raise ValueError("No 'modality' column found in adata.obs")
    
    samples = sorted(adata_mod.obs[sample_col].unique())
    cell_types_in_data = set(adata_mod.obs[celltype_col].astype(str).unique())
    
    if verbose:
        print(f"[Aggregate {modality}] Samples: {len(samples)}, Cell types: {len(cell_types_in_data)}")
    
    # Build feature list
    gene_to_idx = {str(g): i for i, g in enumerate(adata_mod.var_names)}
    
    features = []
    feature_info = []
    
    for ct in sorted(target_genes.keys()):
        ct_str = str(ct)
        if ct_str not in cell_types_in_data:
            continue
        for gene in target_genes[ct]:
            gene_str = str(gene)
            if gene_str in gene_to_idx:
                feat_name = f"{ct} - {gene}"
                features.append(feat_name)
                feature_info.append((ct_str, gene_str, gene_to_idx[gene_str]))
    
    if verbose:
        print(f"[Aggregate {modality}] Target features: {len(features)}")
    
    if len(features) == 0:
        raise ValueError(f"No matching features found for {modality}")
    
    # Aggregate
    sample_idx = {s: i for i, s in enumerate(samples)}
    cell_sample = adata_mod.obs[sample_col].map(sample_idx).values
    cell_ct = adata_mod.obs[celltype_col].astype(str).values
    X = adata_mod.X.tocsr() if issparse(adata_mod.X) else adata_mod.X
    
    n_samples = len(samples)
    n_features = len(features)
    pb_matrix = np.zeros((n_samples, n_features), dtype=np.float32)
    
    for feat_idx, (ct, gene, var_idx) in enumerate(feature_info):
        ct_mask = cell_ct == ct
        for sample in samples:
            s_idx = sample_idx[sample]
            sample_mask = cell_sample == s_idx
            combined_mask = ct_mask & sample_mask
            if combined_mask.sum() > 0:
                cell_indices = np.where(combined_mask)[0]
                if issparse(X):
                    vals = np.array(X[cell_indices, var_idx].todense()).flatten()
                else:
                    vals = X[cell_indices, var_idx]
                pb_matrix[s_idx, feat_idx] = np.mean(vals)
    
    # Build AnnData
    obs = pd.DataFrame(index=samples)
    obs.index.name = 'sample'
    obs['modality'] = modality
    
    if batch_col and batch_col in adata_mod.obs.columns:
        mapping = adata_mod.obs.groupby(sample_col)[batch_col].first().to_dict()
        obs[batch_col] = [mapping.get(s) for s in samples]
    
    var = pd.DataFrame(index=features)
    pb_adata = sc.AnnData(X=pb_matrix, obs=obs, var=var)
    
    if verbose:
        print(f"[Aggregate {modality}] Output shape: {pb_adata.shape}")
    
    return pb_adata


# =============================================================================
# Cross-Modal Embedding Evaluation
# =============================================================================

def evaluate_embedding_cross_modal(
    embedding: np.ndarray,
    atac_expression: np.ndarray,
    rna_expression: np.ndarray,
    atac_pb_indices: np.ndarray,
    rna_pb_indices: np.ndarray,
    base_to_samples: Dict[str, Dict[str, int]],
    pb_sample_names: List[str],
    k_neighbors: int = 5,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Evaluate embedding by checking cross-modal ATAC-RNA correlations.
    
    For each ATAC sample:
    1. Find its k nearest RNA samples in embedding space
    2. Compute correlation between ATAC gene activity and RNA expression
    
    Also computes paired correlation (ATAC sample vs its matched RNA sample).
    """
    n_atac = len(atac_pb_indices)
    n_rna = len(rna_pb_indices)
    
    if n_atac == 0 or n_rna == 0:
        return {
            'avg_nearest_rna_corr': np.nan,
            'avg_knn_corr': np.nan,
            'paired_corr': np.nan,
            'n_atac_samples': n_atac,
            'n_rna_samples': n_rna
        }
    
    # Get embeddings for each modality
    emb_atac = embedding[atac_pb_indices]
    emb_rna = embedding[rna_pb_indices]
    
    # Compute cross-modal distances: ATAC (rows) vs RNA (cols)
    cross_dists = cdist(emb_atac, emb_rna, metric='euclidean')
    
    # For each ATAC sample, find nearest RNA and compute correlation
    nearest_rna_corrs = []
    knn_corrs = []
    paired_corrs = []
    
    k = min(k_neighbors, n_rna)
    
    for i, atac_idx in enumerate(atac_pb_indices):
        atac_sample_name = pb_sample_names[atac_idx]
        base_name = normalize_sample_name(atac_sample_name)
        atac_expr = atac_expression[i]
        
        # Get distances to all RNA samples
        dists_to_rna = cross_dists[i]
        
        # Nearest RNA sample
        nearest_rna_local_idx = np.argmin(dists_to_rna)
        nearest_rna_expr = rna_expression[nearest_rna_local_idx]
        corr, _ = spearmanr(atac_expr, nearest_rna_expr)
        if not np.isnan(corr):
            nearest_rna_corrs.append(corr)
        
        # K nearest RNA samples
        knn_local_indices = np.argsort(dists_to_rna)[:k]
        knn_expr_corrs = []
        for j in knn_local_indices:
            rna_expr = rna_expression[j]
            c, _ = spearmanr(atac_expr, rna_expr)
            if not np.isnan(c):
                knn_expr_corrs.append(c)
        if knn_expr_corrs:
            knn_corrs.append(np.mean(knn_expr_corrs))
        
        # Paired correlation (same biological sample, different modality)
        if base_name in base_to_samples and 'RNA' in base_to_samples[base_name]:
            paired_rna_pb_idx = base_to_samples[base_name]['RNA']
            # Find which local RNA index this corresponds to
            try:
                paired_rna_local_idx = list(rna_pb_indices).index(paired_rna_pb_idx)
                paired_rna_expr = rna_expression[paired_rna_local_idx]
                paired_c, _ = spearmanr(atac_expr, paired_rna_expr)
                if not np.isnan(paired_c):
                    paired_corrs.append(paired_c)
            except ValueError:
                pass  # RNA sample not in our indices
    
    return {
        'avg_nearest_rna_corr': np.mean(nearest_rna_corrs) if nearest_rna_corrs else np.nan,
        'avg_knn_corr': np.mean(knn_corrs) if knn_corrs else np.nan,
        'paired_corr': np.mean(paired_corrs) if paired_corrs else np.nan,
        'n_atac_samples': n_atac,
        'n_rna_samples': n_rna,
        'n_paired': len(paired_corrs)
    }


def select_best_embedding(
    embeddings: Dict[str, np.ndarray],
    atac_expression: np.ndarray,
    rna_expression: np.ndarray,
    atac_pb_indices: np.ndarray,
    rna_pb_indices: np.ndarray,
    base_to_samples: Dict[str, Dict[str, int]],
    pb_sample_names: List[str],
    metric: str,
    k_neighbors: int,
    verbose: bool
) -> Tuple[str, np.ndarray, pd.DataFrame]:
    """Evaluate embeddings and select the best one."""
    results = []
    
    if verbose:
        print("\n" + "=" * 70)
        print("CROSS-MODAL EMBEDDING EVALUATION")
        print("(For each ATAC sample, check correlation with nearest RNA sample)")
        print("=" * 70)
    
    for name, emb in embeddings.items():
        metrics = evaluate_embedding_cross_modal(
            emb, atac_expression, rna_expression,
            atac_pb_indices, rna_pb_indices,
            base_to_samples, pb_sample_names,
            k_neighbors, verbose
        )
        metrics['embedding'] = name
        results.append(metrics)
        
        if verbose:
            print(f"\n  {name}:")
            print(f"    Avg correlation (nearest RNA):     {metrics['avg_nearest_rna_corr']:.4f}")
            print(f"    Avg correlation (k={k_neighbors} nearest RNA): {metrics['avg_knn_corr']:.4f}")
            print(f"    Avg correlation (paired samples):  {metrics['paired_corr']:.4f}")
            print(f"    Samples: {metrics['n_atac_samples']} ATAC, {metrics['n_rna_samples']} RNA, {metrics['n_paired']} paired")
    
    results_df = pd.DataFrame(results).set_index('embedding')
    
    valid_results = results_df[~results_df[metric].isna()]
    if len(valid_results) == 0:
        if verbose:
            print("\n[Warning] No valid embeddings found!")
        return list(embeddings.keys())[0], list(embeddings.values())[0], results_df
    
    best_name = valid_results[metric].idxmax()
    best_emb = embeddings[best_name]
    
    if verbose:
        print("\n" + "-" * 70)
        print(f"BEST EMBEDDING: {best_name}")
        print(f"  {metric} = {results_df.loc[best_name, metric]:.4f}")
        print("=" * 70)
    
    return best_name, best_emb, results_df


# =============================================================================
# Main Pipeline
# =============================================================================

def evaluate_pseudobulk_embeddings(
    cell_adata_path: str,
    pseudobulk_path: str,
    output_dir: str,
    sample_col: str,
    celltype_col: str,
    batch_col: Optional[str],
    selection_metric: str,
    k_neighbors: int,
    combat_timeout: float,
    verbose: bool
) -> Dict:
    """
    Main pipeline to evaluate embeddings using cross-modal ATAC-RNA correlation.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    if verbose:
        print("[Pipeline] Loading data...")
    
    adata = sc.read_h5ad(cell_adata_path)
    pb_adata = sc.read_h5ad(pseudobulk_path)
    
    if verbose:
        print(f"  Cell data: {adata.shape}")
        print(f"  Pseudobulk: {pb_adata.shape}")
        print(f"  Pseudobulk samples: {list(pb_adata.obs_names[:5])}...")
    
    # Extract embeddings
    if verbose:
        print("\n[Pipeline] Extracting embeddings from pseudobulk...")
    
    embeddings = {}
    for key in ['X_DR_expression', 'X_DR_proportion']:
        if key in pb_adata.obsm:
            embeddings[key] = pb_adata.obsm[key]
            if verbose:
                print(f"  Found {key}: shape {embeddings[key].shape}")
    
    if not embeddings:
        raise ValueError("No embeddings found in pseudobulk AnnData")
    
    # Parse sample modalities from pseudobulk
    pb_sample_names = list(pb_adata.obs_names)
    atac_pb_indices, rna_pb_indices, base_to_samples = get_sample_modality_info(pb_sample_names)
    
    if verbose:
        print(f"\n[Pipeline] Pseudobulk modalities:")
        print(f"  ATAC samples: {len(atac_pb_indices)}")
        print(f"  RNA samples: {len(rna_pb_indices)}")
        print(f"  Paired samples: {len([b for b in base_to_samples if len(base_to_samples[b]) == 2])}")
    
    # Parse target genes
    target_genes = parse_gene_features(pb_adata.var_names)
    if verbose:
        n_ct = len(target_genes)
        n_genes = sum(len(v) for v in target_genes.values())
        print(f"\n[Pipeline] Target features: {n_ct} cell types, {n_genes} total genes")
    
    # Aggregate ATAC pseudobulk
    if verbose:
        print("\n[Pipeline] Creating ATAC pseudobulk...")
    
    atac_pb = aggregate_pseudobulk_by_modality(
        adata, target_genes, sample_col, celltype_col, batch_col, 'ATAC', verbose
    )
    
    # Aggregate RNA pseudobulk
    if verbose:
        print("\n[Pipeline] Creating RNA pseudobulk...")
    
    rna_pb = aggregate_pseudobulk_by_modality(
        adata, target_genes, sample_col, celltype_col, batch_col, 'RNA', verbose
    )
    
    # Apply batch correction
    if batch_col:
        if verbose:
            print(f"\n[Pipeline] Applying batch correction to ATAC...")
        atac_pb = apply_batch_correction(atac_pb, batch_col, combat_timeout, verbose)
        
        if verbose:
            print(f"\n[Pipeline] Applying batch correction to RNA...")
        rna_pb = apply_batch_correction(rna_pb, batch_col, combat_timeout, verbose)
    
    # Normalize both
    sc.pp.normalize_total(atac_pb, target_sum=1e4)
    sc.pp.log1p(atac_pb)
    sc.pp.normalize_total(rna_pb, target_sum=1e4)
    sc.pp.log1p(rna_pb)
    
    # Align features between ATAC and RNA pseudobulk
    common_features = list(set(atac_pb.var_names) & set(rna_pb.var_names))
    if verbose:
        print(f"\n[Pipeline] Common features between ATAC and RNA: {len(common_features)}")
    
    atac_pb = atac_pb[:, common_features].copy()
    rna_pb = rna_pb[:, common_features].copy()
    
    # Build mapping from cell-level sample names to pseudobulk indices
    # ATAC cell samples -> pseudobulk ATAC indices
    atac_cell_samples = list(atac_pb.obs_names)
    rna_cell_samples = list(rna_pb.obs_names)
    
    # Map: for each ATAC pseudobulk sample (e.g., "ENCSR033MDU"), find the corresponding 
    # index in pb_adata (e.g., "ENCSR033MDU_ATAC")
    atac_pb_to_multiomics = {}
    rna_pb_to_multiomics = {}
    
    for base_name, modality_dict in base_to_samples.items():
        if 'ATAC' in modality_dict:
            # Find this base_name in atac_pb
            matches = [i for i, s in enumerate(atac_cell_samples) if normalize_sample_name(s) == base_name]
            if matches:
                atac_pb_to_multiomics[matches[0]] = modality_dict['ATAC']
        
        if 'RNA' in modality_dict:
            matches = [i for i, s in enumerate(rna_cell_samples) if normalize_sample_name(s) == base_name]
            if matches:
                rna_pb_to_multiomics[matches[0]] = modality_dict['RNA']
    
    if verbose:
        print(f"\n[Pipeline] Matched samples:")
        print(f"  ATAC (cell-level -> multiomics pb): {len(atac_pb_to_multiomics)}")
        print(f"  RNA (cell-level -> multiomics pb): {len(rna_pb_to_multiomics)}")
    
    # Get expression matrices (using matched samples only)
    atac_local_indices = sorted(atac_pb_to_multiomics.keys())
    rna_local_indices = sorted(rna_pb_to_multiomics.keys())
    
    atac_multiomics_indices = np.array([atac_pb_to_multiomics[i] for i in atac_local_indices])
    rna_multiomics_indices = np.array([rna_pb_to_multiomics[i] for i in rna_local_indices])
    
    X_atac = atac_pb.X[atac_local_indices]
    X_rna = rna_pb.X[rna_local_indices]
    
    if issparse(X_atac):
        X_atac = X_atac.toarray()
    if issparse(X_rna):
        X_rna = X_rna.toarray()
    
    X_atac = np.nan_to_num(X_atac, 0)
    X_rna = np.nan_to_num(X_rna, 0)
    
    if verbose:
        print(f"  ATAC expression matrix: {X_atac.shape}")
        print(f"  RNA expression matrix: {X_rna.shape}")
    
    # Evaluate embeddings
    best_name, best_emb, eval_df = select_best_embedding(
        embeddings, X_atac, X_rna,
        atac_multiomics_indices, rna_multiomics_indices,
        base_to_samples, pb_sample_names,
        selection_metric, k_neighbors, verbose
    )
    
    # Save results
    if verbose:
        print(f"\n[Pipeline] Saving results to {output_dir}...")
    
    eval_path = os.path.join(output_dir, 'embedding_evaluation.csv')
    eval_df.to_csv(eval_path)
    
    info = {
        'best_embedding': best_name,
        'metric_used': selection_metric,
        'n_atac_samples': len(atac_multiomics_indices),
        'n_rna_samples': len(rna_multiomics_indices),
        'n_common_features': len(common_features)
    }
    info_df = pd.DataFrame([info])
    info_df.to_csv(os.path.join(output_dir, 'best_embedding_info.csv'), index=False)
    
    # Save pseudobulks
    sc.write(os.path.join(output_dir, 'atac_pseudobulk.h5ad'), atac_pb)
    sc.write(os.path.join(output_dir, 'rna_pseudobulk.h5ad'), rna_pb)
    
    if verbose:
        print(f"\n[Pipeline] Complete!")
        print(f"  Evaluation results: {eval_path}")
        print(f"  Best embedding: {best_name}")
    
    return {
        'best_embedding_name': best_name,
        'best_embedding': embeddings[best_name],
        'evaluation': eval_df,
        'embeddings': embeddings,
        'atac_pseudobulk': atac_pb,
        'rna_pseudobulk': rna_pb
    }


# =============================================================================
# Run Pipeline
# =============================================================================

if __name__ == '__main__':
    results = evaluate_pseudobulk_embeddings(
        cell_adata_path=CELL_ADATA_PATH,
        pseudobulk_path=PSEUDOBULK_PATH,
        output_dir=OUTPUT_DIR,
        sample_col=SAMPLE_COL,
        celltype_col=CELLTYPE_COL,
        batch_col=BATCH_COL,
        selection_metric=SELECTION_METRIC,
        k_neighbors=K_NEIGHBORS,
        combat_timeout=COMBAT_TIMEOUT,
        verbose=VERBOSE
    )
    
    print(f"\nBest embedding: {results['best_embedding_name']}")
    print("\nEvaluation summary:")
    print(results['evaluation'])