# import json
# import pandas as pd
# from pathlib import Path
# from typing import List

# def summarize_json_to_csv(sample_sizes: List[int], base_path: str, output_csv: str, method_name: str) -> pd.DataFrame:
#     """
#     Summarize multiple JSON files into a single CSV file.
    
#     Parameters:
#     -----------
#     sample_sizes : List[int]
#         List of sample sizes to process
#     base_path : str
#         Path template with {} placeholder for sample size
#     output_csv : str
#         Output CSV file path
#     method_name : str
#         Method name to use in labels
        
#     Returns:
#     --------
#     pd.DataFrame
#         Summary DataFrame
#     """
#     # Collect all dataframes in a list for efficient concatenation
#     dfs = []
    
#     # Loop over all sample sizes
#     for sample_size in sample_sizes:
#         # Construct the JSON file path (handles multiple placeholders)
#         json_path = Path(base_path.format(sample_size, sample_size))
        
#         # Check if the JSON file exists
#         if not json_path.exists():
#             print(f"Warning: File does not exist: {json_path}")
#             continue
        
#         try:
#             # Read the JSON file
#             with open(json_path, 'r') as f:
#                 data = json.load(f)
            
#             # Assign label based on method_name and sample size
#             data['label'] = f"{method_name}-{sample_size}"
            
#             # Create a DataFrame from the JSON data
#             dfs.append(pd.DataFrame([data]))
            
#         except json.JSONDecodeError as e:
#             print(f"Error reading JSON from {json_path}: {e}")
#         except Exception as e:
#             print(f"Unexpected error processing {json_path}: {e}")
    
#     # Concatenate all dataframes at once (more efficient)
#     if dfs:
#         summary_df = pd.concat(dfs, axis=0, ignore_index=True)
        
#         # Check if the output CSV file already exists
#         output_path = Path(output_csv)
#         if output_path.exists():
#             # Read existing data and append new data
#             existing_df = pd.read_csv(output_csv)
#             combined_df = pd.concat([existing_df, summary_df], axis=0, ignore_index=True)
#             combined_df.to_csv(output_csv, index=False)
#             print(f"Appended {len(summary_df)} rows to existing {output_csv}")
#             print(f"Total rows in file: {len(combined_df)}")
#         else:
#             # Create new file
#             summary_df.to_csv(output_csv, index=False)
#             print(f"Created new summary file: {output_csv}")
#             print(f"Total rows: {len(summary_df)}")
        
#         return summary_df
#     else:
#         print("No valid data found. No CSV file modified.")
#         return pd.DataFrame()



# # Example usage
# if __name__ == "__main__":
#     sample_sizes = [25, 50, 100, 200, 279, 400]  # List of sample sizes
#     base_path = "/dcs07/hongkai/data/harry/result/GEDI/{}_sample/GEDI_summary.json"  # Path template
#     output_csv = "/dcs07/hongkai/data/harry/result/benchmark_summary_run_time.csv"  # Output CSV file path
#     method_name = "GEDI"  # User-defined method name
#     summarize_json_to_csv(sample_sizes, base_path, output_csv, method_name)
    
#     base_path = "/dcs07/hongkai/data/harry/result/Gloscope/{}_sample/gloscope_summary.json"
#     output_csv = "/dcs07/hongkai/data/harry/result/benchmark_summary_run_time.csv"
#     method_name = "Gloscope"
#     df = summarize_json_to_csv(sample_sizes, base_path, output_csv, method_name)
    
#     base_path = "/dcs07/hongkai/data/harry/result/MFA/{}_sample/MFA_summary.json"
#     output_csv = "/dcs07/hongkai/data/harry/result/benchmark_summary_run_time.csv"
#     method_name = "MFA"
#     df = summarize_json_to_csv(sample_sizes, base_path, output_csv, method_name)
    
#     base_path = "/dcs07/hongkai/data/harry/result/MUSTARD/{}_sample/MUSTARD_summary.json"
#     output_csv = "/dcs07/hongkai/data/harry/result/benchmark_summary_run_time.csv"
#     method_name = "MUSTARRD"
#     df = summarize_json_to_csv(sample_sizes, base_path, output_csv, method_name)
    
#     base_path = "/dcs07/hongkai/data/harry/result/naive_pseudobulk/covid_{}_sample/naive_pseudobulk_summary.json"
#     output_csv = "/dcs07/hongkai/data/harry/result/benchmark_summary_run_time.csv"
#     method_name = "navie_pseudobulk"
#     df = summarize_json_to_csv(sample_sizes, base_path, output_csv, method_name)
    
#     base_path = "/dcs07/hongkai/data/harry/result/pilot/{}_sample/pilot_summary.json"
#     output_csv = "/dcs07/hongkai/data/harry/result/benchmark_summary_run_time.csv"
#     method_name = "pilot"
#     df = summarize_json_to_csv(sample_sizes, base_path, output_csv, method_name)
    
#     base_path = "/dcs07/hongkai/data/harry/result/QOT/{}_sample/QOT_summary.json"
#     output_csv = "/dcs07/hongkai/data/harry/result/benchmark_summary_run_time.csv"
#     method_name = "QOT"
#     df = summarize_json_to_csv(sample_sizes, base_path, output_csv, method_name)
    
    
#     base_path = "/dcs07/hongkai/data/harry/result/scPoli/{}_sample/scPoli_summary.json"
#     output_csv = "/dcs07/hongkai/data/harry/result/benchmark_summary_run_time.csv"
#     method_name = "scPoli"
#     df = summarize_json_to_csv(sample_sizes, base_path, output_csv, method_name)
    
#     base_path = "/dcs07/hongkai/data/harry/result/Benchmark_covid/covid_{}_sample/sampledisco_summary.json"
#     output_csv = "/dcs07/hongkai/data/harry/result/benchmark_summary_run_time.csv"
#     method_name = "SD"
#     df = summarize_json_to_csv(sample_sizes, base_path, output_csv, method_name)

"""
Generate plots for runtime and RAM metrics:

1) Across SAMPLE SIZE (x-axis = sample size) for each METHOD (line plots)
2) Across METHOD (x-axis = method) for each SAMPLE SIZE (bar charts / "histograms"),
   with methods sorted in descending order by the metric and the value marked on each bar.

User modifies CSV_PATH and OUTDIR directly.
"""

import os
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for HPC
import matplotlib.pyplot as plt


# ============================================================
# USER INPUT (MODIFY THESE TWO PATHS ONLY)
# ============================================================
CSV_PATH = "/dcs07/hongkai/data/harry/result/run_time_summary/benchmark_summary_run_time.csv"
OUTDIR   = "/dcs07/hongkai/data/harry/result/run_time_summary"


# ============================================================
# INTERNAL FUNCTIONS (DO NOT MODIFY)
# ============================================================
def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def add_method_and_sample_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "label" not in df.columns:
        raise KeyError("CSV must contain a 'label' column.")

    split = df["label"].astype(str).str.rsplit("-", n=1, expand=True)
    df["method"] = split[0]
    df["sample_size"] = pd.to_numeric(split[1], errors="coerce")

    return df


# ============================================================
# MAIN LOGIC
# ============================================================
def main():

    print(f"[INFO] Reading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    df = add_method_and_sample_columns(df)

    metrics = [
        "duration_s_from_csv",
        "avg_ram_mb_time_weighted",
    ]

    for m in metrics:
        if m not in df.columns:
            raise KeyError(f"Missing required metric: {m}")

    os.makedirs(OUTDIR, exist_ok=True)

    # ========================================================
    # 1. ACROSS SAMPLE SIZE (x-axis = sample size)
    #    One line plot per METHOD
    # ========================================================
    by_method_dir = os.path.join(OUTDIR, "across_sample_size")
    os.makedirs(by_method_dir, exist_ok=True)

    print("[INFO] Plotting across SAMPLE SIZE for each METHOD...")

    for method, sub in df.groupby("method"):
        sub = sub.sort_values("sample_size")
        x = sub["sample_size"]

        for metric in metrics:
            y = sub[metric]

            if y.dropna().empty:
                print(f"[WARN] No data for method={method}, metric={metric}, skipping.")
                continue

            plt.figure()
            plt.plot(x, y, marker="o")
            plt.xlabel("Sample Size")
            plt.ylabel(metric)
            plt.title(f"{method} - {metric} vs Sample Size")
            plt.tight_layout()

            safe_method = sanitize_name(method)
            safe_metric = sanitize_name(metric)

            out_path = os.path.join(
                by_method_dir,
                f"{safe_method}_by_sample_size_{safe_metric}.png",
            )

            plt.savefig(out_path, dpi=150)
            plt.close()

            print(f"[INFO] Saved: {out_path}")

    # ========================================================
    # 2. ACROSS METHOD (x-axis = method, bar charts)
    #    One bar chart per SAMPLE SIZE, sorted descending
    #    Each bar labeled with its numeric value
    # ========================================================
    by_sample_dir = os.path.join(OUTDIR, "across_method")
    os.makedirs(by_sample_dir, exist_ok=True)

    print("[INFO] Plotting across METHOD (bar charts) for each SAMPLE SIZE...")

    for sample_size, sub in df.groupby("sample_size"):

        for metric in metrics:
            # Drop NaNs and sort by metric descending
            sub_metric = sub[["method", metric]].dropna()
            if sub_metric.empty:
                print(
                    f"[WARN] No data for sample_size={sample_size}, metric={metric}, skipping."
                )
                continue

            sub_metric = sub_metric.sort_values(metric, ascending=False)

            methods = sub_metric["method"].tolist()
            values = sub_metric[metric].tolist()

            plt.figure()
            x_pos = range(len(methods))
            plt.bar(x_pos, values)
            plt.xticks(x_pos, methods, rotation=45, ha="right")
            plt.xlabel("Method")
            plt.ylabel(metric)
            plt.title(f"Sample Size {sample_size} - {metric} vs Method")
            plt.tight_layout()

            # Label each bar with its value
            for i, v in enumerate(values):
                plt.text(
                    i,
                    v,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )

            safe_sample = sanitize_name(sample_size)
            safe_metric = sanitize_name(metric)

            out_path = os.path.join(
                by_sample_dir,
                f"sample_{safe_sample}_by_method_{safe_metric}_bar.png",
            )

            plt.savefig(out_path, dpi=150)
            plt.close()

            print(f"[INFO] Saved: {out_path}")

    print("[INFO] All plots generated successfully.")


if __name__ == "__main__":
    main()

