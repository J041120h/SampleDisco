"""Re-run the multi-omics benchmarks for the legacy SD_proportion embedding,
using the UPDATED benchmark code (paired_v2_score now computed inside the
evaluator; summary CSV slimmed to 3 metrics).

Reads the existing SD_proportion sample_proportion_embedding.csv files (no
re-derivation needed — we just want fresh metric values from the new code)."""
from __future__ import annotations
import os, sys, traceback
sys.path.insert(0, "/users/hjiang/GenoDistance/code")
sys.path.insert(0, "/users/hjiang/GenoDistance/code/Benchmark_multiomics")


CASES = [
    ("ENCODE",
     "/dcs07/hongkai/data/harry/result/multi_omics_ENCODE/multiomics/embeddings/sample_proportion_embedding.csv",
     "/dcl01/hongkai/data/data/hjiang/Data/multiomics_benchmark_data/sample_metadata.csv",
     "/dcs07/hongkai/data/harry/result/Benchmark_multiomics",
     "encode", 5),
    ("Lutea",
     "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea/lutea/embeddings/sample_proportion_embedding.csv",
     "/dcs07/hongkai/data/harry/result/multi_omics_eye/data/scMultiomics_database.csv",
     "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_lutea",
     "eye", 3),
    ("Retina",
     "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina/retina/embeddings/sample_proportion_embedding.csv",
     "/dcs07/hongkai/data/harry/result/multi_omics_eye/data/scMultiomics_database.csv",
     "/dcs07/hongkai/data/harry/result/multi_omics_eye/benchmark_retina",
     "eye", 3),
    ("Heart",
     "/dcs07/hongkai/data/harry/result/multi_omics_heart/SD/multiomics/embeddings/sample_proportion_embedding.csv",
     "/dcs07/hongkai/data/harry/result/multi_omics_heart/data/multi_omics_heart_sample_meta.csv",
     "/dcs07/hongkai/data/harry/result/multi_omics_heart",
     "heart", 15),
]


def main():
    for name, emb, meta, outroot, kind, k in CASES:
        if not os.path.exists(emb):
            print(f"\n[{name}] SKIP — no embedding at {emb}")
            continue
        print(f"\n==== {name} ====")
        try:
            if kind == "encode":
                from benchmark_metrics_ENCODE import (
                    evaluate_multimodal_integration as ev, save_to_summary_csv as sv)
                summary = f"{outroot}/summary.csv"
            elif kind == "eye":
                from benchmark_eye import (
                    evaluate_multimodal_integration as ev, save_to_summary_csv as sv)
                summary = f"{outroot}/Benchmark_result/summary.csv"
            else:
                from benchmark_heart import (
                    evaluate_multimodal_integration as ev, save_to_summary_csv as sv)
                summary = f"{outroot}/summary.csv"
            r = ev(meta_csv=meta, embedding_csv=emb,
                     method_name="SD_proportion",
                     general_outdir=outroot,
                     k_neighbors=k, n_permutations=1000,
                     create_visualizations=False)
            sv(r, summary)
            print(f"[{name}] paired_v2={r['paired_v2_score']:.4f}  "
                  f"ASW={r['ASW_modality_overall']:.4f}")
        except Exception:
            print(f"[{name}] FAIL")
            traceback.print_exc()


if __name__ == "__main__":
    main()
