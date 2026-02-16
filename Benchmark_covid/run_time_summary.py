import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any


# ============================================================
# KEEP ONLY THESE METRICS (AS REQUESTED)
# ============================================================
KEEP_KEYS = [
    "duration_s_from_csv",
    "avg_ram_mb_time_weighted",
    "peak_ram_mb_from_csv",
]


def flatten_json(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flatten nested dictionaries (safe even if you later add nested metrics).
    Currently we only keep top-level metrics in KEEP_KEYS, but flattening is harmless.
    """
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(flatten_json(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def read_metrics_from_json(json_path: Path) -> Dict[str, Any]:
    """
    Read a single JSON and return only requested metrics.
    Missing keys are filled with None (so CSV columns are consistent).
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    data = flatten_json(data)

    out = {k: data.get(k, None) for k in KEEP_KEYS}
    return out


def write_one_csv_per_sample_size(
    sample_sizes: List[int],
    method_to_path_template: Dict[str, str],
    out_dir: str,
) -> None:
    """
    For each sample size N:
      - read each method's JSON file
      - create a CSV named: runtime_summary_{N}_sample.csv
      - rows are methods (method name only, no sample size suffix)
      - columns are ONLY:
          duration_s_from_csv, avg_ram_mb_time_weighted, peak_ram_mb_from_csv

    ------------------------------------------------------------
    ALL PATHS EXPLICITLY LISTED BELOW (method_to_path_template):
      GEDI:
        /dcs07/hongkai/data/harry/result/GEDI/{N}_sample/GEDI_summary.json
      GloScope:
        /dcs07/hongkai/data/harry/result/Gloscope/{N}_sample/gloscope_summary.json
      MFA:
        /dcs07/hongkai/data/harry/result/MFA/{N}_sample/MFA_summary.json
      MUSTARD:
        /dcs07/hongkai/data/harry/result/MUSTARD/{N}_sample/MUSTARD_summary.json
      naive_pseudobulk:
        /dcs07/hongkai/data/harry/result/naive_pseudobulk/covid_{N}_sample/naive_pseudobulk_summary.json
      pilot:
        /dcs07/hongkai/data/harry/result/pilot/{N}_sample/pilot_summary.json
      QOT:
        /dcs07/hongkai/data/harry/result/QOT/{N}_sample/QOT_summary.json
      scPoli:
        /dcs07/hongkai/data/harry/result/scPoli/{N}_sample/scPoli_summary.json
      SampleDisc (NEW STYLE):
        /dcs07/hongkai/data/harry/result/Benchmark_covid/covid_{N}_sample/run_time/sampledisco_summary.json
    ------------------------------------------------------------
    """
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    for n in sample_sizes:
        rows = []

        for method, template in method_to_path_template.items():
            json_path = Path(template.format(n))

            if not json_path.exists():
                print(f"[WARN] Missing JSON for sample_size={n}, method={method}: {json_path}")
                continue

            try:
                metrics = read_metrics_from_json(json_path)
                metrics["method"] = method  # method name only (NO sample size)
                metrics["sample_size"] = n  # optional but useful
                rows.append(metrics)
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON decode failed for {json_path}: {e}")
            except Exception as e:
                print(f"[ERROR] Failed reading {json_path}: {e}")

        if not rows:
            print(f"[INFO] No data found for sample_size={n}; no CSV written.")
            continue

        df = pd.DataFrame(rows)

        # Enforce column order: method, sample_size, metrics...
        df = df[["method", "sample_size"] + KEEP_KEYS]

        out_csv = out_dir_path / f"runtime_summary_{n}_sample.csv"
        df.to_csv(out_csv, index=False)
        print(f"[INFO] Wrote {len(df)} rows -> {out_csv}")


# ============================================================
# EXAMPLE USAGE (EDIT ONLY IF YOU WANT DIFFERENT SIZES/OUTDIR)
# ============================================================
if __name__ == "__main__":

    sample_sizes = [25, 50, 100, 200, 279, 400]

    OUTDIR = "/dcs07/hongkai/data/harry/result/run_time_summary"

    method_to_path_template = {
        # GEDI
        "GEDI": "/dcs07/hongkai/data/harry/result/GEDI/{}_sample/GEDI_summary.json",

        # GloScope
        "Gloscope": "/dcs07/hongkai/data/harry/result/Gloscope/{}_sample/gloscope_summary.json",

        # MFA
        "MFA": "/dcs07/hongkai/data/harry/result/MFA/{}_sample/MFA_summary.json",

        # MUSTARD
        "MUSTARD": "/dcs07/hongkai/data/harry/result/MUSTARD/{}_sample/MUSTARD_summary.json",

        # naive_pseudobulk
        "naive_pseudobulk": (
            "/dcs07/hongkai/data/harry/result/"
            "naive_pseudobulk/covid_{}_sample/naive_pseudobulk_summary.json"
        ),

        # PILOT
        "pilot": "/dcs07/hongkai/data/harry/result/pilot/{}_sample/pilot_summary.json",

        # QOT
        "QOT": "/dcs07/hongkai/data/harry/result/QOT/{}_sample/QOT_summary.json",

        # scPoli
        "scPoli": "/dcs07/hongkai/data/harry/result/scPoli/{}_sample/scPoli_summary.json",

        # SampleDisc (NEW DIRECTORY STYLE)
        "SD": (
            "/dcs07/hongkai/data/harry/result/"
            "Benchmark_covid/covid_{}_sample/run_time/sampledisco_summary.json"
        ),
    }

    write_one_csv_per_sample_size(
        sample_sizes=sample_sizes,
        method_to_path_template=method_to_path_template,
        out_dir=OUTDIR,
    )
