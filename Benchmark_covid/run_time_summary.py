import json
import pandas as pd
from pathlib import Path
from typing import List

def summarize_json_to_csv(sample_sizes: List[int], base_path: str, output_csv: str, method_name: str) -> pd.DataFrame:
    """
    Summarize multiple JSON files into a single CSV file.
    
    Parameters:
    -----------
    sample_sizes : List[int]
        List of sample sizes to process
    base_path : str
        Path template with {} placeholder for sample size
    output_csv : str
        Output CSV file path
    method_name : str
        Method name to use in labels
        
    Returns:
    --------
    pd.DataFrame
        Summary DataFrame
    """
    # Collect all dataframes in a list for efficient concatenation
    dfs = []
    
    # Loop over all sample sizes
    for sample_size in sample_sizes:
        # Construct the JSON file path (handles multiple placeholders)
        json_path = Path(base_path.format(sample_size, sample_size))
        
        # Check if the JSON file exists
        if not json_path.exists():
            print(f"Warning: File does not exist: {json_path}")
            continue
        
        try:
            # Read the JSON file
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Assign label based on method_name and sample size
            data['label'] = f"{method_name}-{sample_size}"
            
            # Create a DataFrame from the JSON data
            dfs.append(pd.DataFrame([data]))
            
        except json.JSONDecodeError as e:
            print(f"Error reading JSON from {json_path}: {e}")
        except Exception as e:
            print(f"Unexpected error processing {json_path}: {e}")
    
    # Concatenate all dataframes at once (more efficient)
    if dfs:
        summary_df = pd.concat(dfs, axis=0, ignore_index=True)
        
        # Check if the output CSV file already exists
        output_path = Path(output_csv)
        if output_path.exists():
            # Read existing data and append new data
            existing_df = pd.read_csv(output_csv)
            combined_df = pd.concat([existing_df, summary_df], axis=0, ignore_index=True)
            combined_df.to_csv(output_csv, index=False)
            print(f"Appended {len(summary_df)} rows to existing {output_csv}")
            print(f"Total rows in file: {len(combined_df)}")
        else:
            # Create new file
            summary_df.to_csv(output_csv, index=False)
            print(f"Created new summary file: {output_csv}")
            print(f"Total rows: {len(summary_df)}")
        
        return summary_df
    else:
        print("No valid data found. No CSV file modified.")
        return pd.DataFrame()


# Example usage
if __name__ == "__main__":
    
    sample_sizes = [25, 50, 100, 200, 279, 400]  # List of sample sizes
    base_path = "/dcs07/hongkai/data/harry/result/GEDI/{}_sample/GEDI_summary.json"  # Path template
    output_csv = "/dcs07/hongkai/data/harry/result/benchmark_summary_run_time.csv"  # Output CSV file path
    method_name = "GEDI"  # User-defined method name
    summarize_json_to_csv(sample_sizes, base_path, output_csv, method_name)
    
    sample_sizes = [25, 50, 100, 200, 400]
    base_path = "/dcs07/hongkai/data/harry/result/Gloscope/{}_sample/gloscope_summary.json"
    output_csv = "/dcs07/hongkai/data/harry/result/benchmark_summary_run_time.csv"
    method_name = "Gloscope"
    df = summarize_json_to_csv(sample_sizes, base_path, output_csv, method_name)
    
    sample_sizes = [25, 50, 100, 200, 400]
    base_path = "/dcs07/hongkai/data/harry/result/MFA/{}_sample/MFA_summary.json"
    output_csv = "/dcs07/hongkai/data/harry/result/benchmark_summary_run_time.csv"
    method_name = "MFA"
    df = summarize_json_to_csv(sample_sizes, base_path, output_csv, method_name)
    
    sample_sizes = [25, 50, 100, 200, 400]
    base_path = "/dcs07/hongkai/data/harry/result/MUSTARD/{}_sample/MUSTARD_summary.json"
    output_csv = "/dcs07/hongkai/data/harry/result/benchmark_summary_run_time.csv"
    method_name = "MUSTARRD"
    df = summarize_json_to_csv(sample_sizes, base_path, output_csv, method_name)
    
    sample_sizes = [25, 50, 100, 200, 400]
    base_path = "/dcs07/hongkai/data/harry/result/naive_pseudobulk/covid_{}_sample/naive_pseudobulk_summary.json"
    output_csv = "/dcs07/hongkai/data/harry/result/benchmark_summary_run_time.csv"
    method_name = "navie_pseudobulk"
    df = summarize_json_to_csv(sample_sizes, base_path, output_csv, method_name)
    
    sample_sizes = [25, 50, 100, 200, 400]
    base_path = "/dcs07/hongkai/data/harry/result/pilot/{}_sample/pilot_summary.json"
    output_csv = "/dcs07/hongkai/data/harry/result/benchmark_summary_run_time.csv"
    method_name = "pilot"
    df = summarize_json_to_csv(sample_sizes, base_path, output_csv, method_name)
    
    sample_sizes = [25, 50, 100, 200, 400]
    base_path = "/dcs07/hongkai/data/harry/result/QOT/{}_sample/QOT_summary.json"
    output_csv = "/dcs07/hongkai/data/harry/result/benchmark_summary_run_time.csv"
    method_name = "QOT"
    df = summarize_json_to_csv(sample_sizes, base_path, output_csv, method_name)
    
    sample_sizes = [25, 50, 100, 200, 400]
    base_path = "/dcs07/hongkai/data/harry/result/scPoli/{}_sample/{}_scPoli_summary.json"
    output_csv = "/dcs07/hongkai/data/harry/result/benchmark_summary_run_time.csv"
    method_name = "scPoli"
    df = summarize_json_to_csv(sample_sizes, base_path, output_csv, method_name)