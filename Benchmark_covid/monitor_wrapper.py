#!/usr/bin/env python3
"""
run_metrics_v2.py

Exact totals (CPU time, peak RAM) + time-weighted averages (RAM, GPU util/mem)
Optional: CPU cycles (perf), per-kernel GPU time (Nsight Systems).

Outputs into --outdir:
  - <label>.out / <label>.err                   : your program's stdout/stderr
  - <label>_timeseries.csv                      : RAM/GPU time series (streaming)
  - <label>_summary.json                        : final JSON report (exact+averages)
  - <label>_nsys.nsys-rep or _nsys.qdrep        : Nsight Systems trace (if --nsys)
  - <label>_gpukernels.csv (if --nsys stats succeed)
  - perf stats mixed into stderr parsed for cycles (if --perf)
"""

import argparse
import csv
import json
import os
import re
import shlex
import sys
import time
import threading
import subprocess
from shutil import which as _which

import psutil

# Optional NVML
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False

TIME_BIN = "/usr/bin/time"  # posix time(1)
PERF_BIN = "perf"
NSYS_BIN = "nsys"


def which(x):
    return _which(x)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return os.path.abspath(p)


def detect_gpus():
    if not _NVML_OK:
        return 0, []
    try:
        n = pynvml.nvmlDeviceGetCount()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
        return n, handles
    except Exception:
        return 0, []


def build_timeseries_header(num_gpus):
    cols = ["ts", "wall_s", "rss_mb", "cpu_percent_sum"]
    for gi in range(num_gpus):
        cols += [f"gpu{gi}_util", f"gpu{gi}_mem_used_mb", f"gpu{gi}_mem_total_mb"]
    return cols


def sample_loop(pid, csv_path, interval, include_children=True, flush_every=5):
    """
    Writes a wide CSV timeseries (1 row per timestamp) with CPU RAM and per-GPU util/mem.
    Designed for multi-day runs: streaming writes + periodic flush.
    """
    proc = psutil.Process(pid)
    num_gpus, gpu_handles = detect_gpus()
    header = build_timeseries_header(num_gpus)
    t0 = time.time()
    wrote_header = False
    last_flush = time.time()

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        while True:
            if not proc.is_running():
                break
            now = time.time()
            wall = now - t0

            try:
                procs = [proc] + (proc.children(recursive=True) if include_children else [])
                rss = sum((p.memory_info().rss for p in procs if p.is_running()), 0) / 1e6
                for p in procs:
                    try:
                        p.cpu_percent(interval=None)
                    except Exception:
                        pass
                cpu_sum = sum((p.cpu_percent(interval=None) for p in procs if p.is_running()), 0.0)
            except Exception:
                rss, cpu_sum = 0.0, 0.0

            row = [f"{now:.6f}", f"{wall:.3f}", f"{rss:.2f}", f"{cpu_sum:.1f}"]

            if num_gpus > 0:
                try:
                    for h in gpu_handles:
                        util = pynvml.nvmlDeviceGetUtilizationRates(h)
                        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                        row += [f"{util.gpu}", f"{mem.used/1e6:.2f}", f"{mem.total/1e6:.2f}"]
                except Exception:
                    for _ in range(num_gpus):
                        row += ["", "", ""]

            if not wrote_header:
                w.writerow(header)
                wrote_header = True
            w.writerow(row)

            if now - last_flush >= flush_every:
                f.flush()
                os.fsync(f.fileno())
                last_flush = now

            time.sleep(interval)


def parse_time_v(stderr_text):
    # Extract exact counters from /usr/bin/time -v
    get = lambda k: re.search(rf"^{k}:\s+(.*)$", stderr_text, re.MULTILINE)

    def to_sec(s):
        try:
            return float(s)
        except Exception:
            if ":" in s:
                parts = [float(x) for x in s.split(":")]
                acc = 0.0
                for p in parts:
                    acc = acc * 60 + p
                return acc
            return None

    out = {}
    m = get(r"User time \(seconds\)")
    if m:
        out["cpu_user_s"] = to_sec(m.group(1))
    m = get(r"System time \(seconds\)")
    if m:
        out["cpu_sys_s"] = to_sec(m.group(1))
    m = get(r"Elapsed \(wall clock\) time")
    if m:
        out["wall_s"] = to_sec(m.group(1))
    m = get(r"Maximum resident set size \(kbytes\)")
    if m:
        out["peak_ram_mb_exact"] = float(m.group(1)) / 1024.0
    return out


def parse_perf_cycles(stderr_text):
    # perf stat -x, yields lines like: 123456789,cycles
    m = re.search(r"^\s*([\d,]+),cycles\b", stderr_text, re.MULTILINE)
    if not m:
        return None
    return int(m.group(1).replace(",", ""))


def summarize_timeseries(csv_path):
    """
    Time-weighted averages + peaks, single pass over the CSV.
    """
    if not os.path.exists(csv_path):
        return {}
    with open(csv_path, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        if not header:
            return {}

        idx = {name: i for i, name in enumerate(header)}
        gpu_indices = []
        gi = 0
        while f"gpu{gi}_util" in idx:
            gpu_indices.append(gi)
            gi += 1

        prev_ts = None
        accum = {"ram_time_int": 0.0, "cpu_time_int": 0.0, "ram_peak": 0.0}
        gpu_accum = {
            gi: {"util_time_int": 0.0, "mem_time_int": 0.0, "mem_peak": 0.0, "mem_total": 0.0}
            for gi in gpu_indices
        }
        total_time = 0.0

        for row in r:
            try:
                ts = float(row[idx["ts"]])
                rss = float(row[idx["rss_mb"]]) if row[idx["rss_mb"]] else 0.0
                cpu = float(row[idx["cpu_percent_sum"]]) if row[idx["cpu_percent_sum"]] else 0.0
            except Exception:
                continue

            if prev_ts is not None and ts >= prev_ts:
                dt = ts - prev_ts
                total_time += dt
                accum["ram_time_int"] += rss * dt
                accum["cpu_time_int"] += cpu * dt
                if rss > accum["ram_peak"]:
                    accum["ram_peak"] = rss

                for gi in gpu_indices:
                    ucol = f"gpu{gi}_util"
                    mcol = f"gpu{gi}_mem_used_mb"
                    tcol = f"gpu{gi}_mem_total_mb"
                    try:
                        util = float(row[idx[ucol]]) if row[idx[ucol]] != "" else 0.0
                        gmem = float(row[idx[mcol]]) if row[idx[mcol]] != "" else 0.0
                        gtot = float(row[idx[tcol]]) if row[idx[tcol]] != "" else 0.0
                    except Exception:
                        util, gmem, gtot = 0.0, 0.0, 0.0

                    gpu_accum[gi]["util_time_int"] += util * dt
                    gpu_accum[gi]["mem_time_int"] += gmem * dt
                    gpu_accum[gi]["mem_total"] = max(gpu_accum[gi]["mem_total"], gtot)
                    gpu_accum[gi]["mem_peak"] = max(gpu_accum[gi]["mem_peak"], gmem)
            else:
                # first datapoint
                accum["ram_peak"] = max(accum["ram_peak"], rss)
                for gi in gpu_indices:
                    mcol = f"gpu{gi}_mem_used_mb"
                    tcol = f"gpu{gi}_mem_total_mb"
                    try:
                        gmem = float(row[idx[mcol]]) if row[idx[mcol]] != "" else 0.0
                        gtot = float(row[idx[tcol]]) if row[idx[tcol]] != "" else 0.0
                    except Exception:
                        gmem, gtot = 0.0, 0.0
                    gpu_accum[gi]["mem_peak"] = max(gpu_accum[gi]["mem_peak"], gmem)
                    gpu_accum[gi]["mem_total"] = max(gpu_accum[gi]["mem_total"], gtot)

            prev_ts = ts

        out = {
            "avg_ram_mb_time_weighted": (accum["ram_time_int"] / total_time) if total_time > 0 else None,
            "peak_ram_mb_from_csv": accum["ram_peak"],
            "duration_s_from_csv": total_time,
        }
        out["gpus"] = {}
        for gi in gpu_indices:
            d = gpu_accum[gi]
            out["gpus"][gi] = {
                "avg_gpu_util_percent_time_weighted": (d["util_time_int"] / total_time) if total_time > 0 else None,
                "avg_gpu_mem_used_mb_time_weighted": (d["mem_time_int"] / total_time) if total_time > 0 else None,
                "peak_gpu_mem_used_mb": d["mem_peak"],
                "gpu_mem_total_mb_seen": d["mem_total"],
            }
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="Directory to store ALL outputs.")
    ap.add_argument("--label", default="run")
    ap.add_argument("--cmd", required=True, help="Command to execute (quote it).")
    ap.add_argument("--interval", type=float, default=1.0, help="Sampling interval (s) for CSV.")
    ap.add_argument("--nsys", action="store_true", help="Capture Nsight Systems per-kernel times.")
    ap.add_argument("--perf", action="store_true", help="Capture CPU cycles via perf stat.")
    ap.add_argument("--workdir", default=None, help="cd into this directory before running.")
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    if args.workdir:
        os.chdir(args.workdir)

    inner = shlex.split(args.cmd)

    # Optional Nsight wrapper (NO --stats here)
    nsys_path_base = None
    if args.nsys and which(NSYS_BIN):
        nsys_path_base = os.path.join(outdir, f"{args.label}_nsys")
        inner = [
            NSYS_BIN, "profile",
            "-o", nsys_path_base,
            "--force-overwrite", "true",
            "--trace", "cuda,nvtx,osrt",
            "--sample", "none",
        ] + inner
    elif args.nsys:
        print("[WARN] nsys not found; skipping Nsight.", file=sys.stderr)

    # Optional perf wrapper (cycles)
    if args.perf and which(PERF_BIN):
        inner = [PERF_BIN, "stat", "-x,", "-e", "cycles"] + inner
    elif args.perf:
        print("[WARN] perf not found; skipping cycles.", file=sys.stderr)

    if not which(TIME_BIN):
        print("[ERROR] /usr/bin/time not found at /usr/bin/time", file=sys.stderr)
        sys.exit(2)

    timed = [TIME_BIN, "-v"] + inner

    stdout_path = os.path.join(outdir, f"{args.label}.out")
    stderr_path = os.path.join(outdir, f"{args.label}.err")
    csv_path = os.path.join(outdir, f"{args.label}_timeseries.csv")
    summary_json = os.path.join(outdir, f"{args.label}_summary.json")

    out_f = open(stdout_path, "wb")
    err_f = open(stderr_path, "wb")
    proc = subprocess.Popen(timed, stdout=out_f, stderr=err_f, preexec_fn=os.setsid)

    sampler = threading.Thread(
        target=sample_loop,
        args=(proc.pid, csv_path, args.interval),
        daemon=True,
    )
    sampler.start()

    proc.wait()
    returncode = proc.returncode
    # Make sure this appears in scPoli.err (stderr of the wrapper)
    print(f"[monitor_wrapper] child returncode={returncode}", file=sys.stderr)

    sampler.join()
    out_f.close()
    err_f.close()

    # Parse /usr/bin/time -v output
    with open(stderr_path, "r", encoding="utf-8", errors="ignore") as f:
        time_text = f.read()
    exact = parse_time_v(time_text)
    cycles = parse_perf_cycles(time_text) if args.perf else None

    # Figure out which Nsight report file exists (.nsys-rep vs .qdrep)
    nsys_report_path = None
    nsys_kernel_csv = None
    if args.nsys and which(NSYS_BIN) and nsys_path_base:
        cand_nsysrep = nsys_path_base + ".nsys-rep"
        cand_qdrep = nsys_path_base + ".qdrep"
        if os.path.exists(cand_nsysrep):
            nsys_report_path = cand_nsysrep
        elif os.path.exists(cand_qdrep):
            nsys_report_path = cand_qdrep

        if nsys_report_path is not None:
            nsys_kernel_csv = os.path.join(outdir, f"{args.label}_gpukernels.csv")
            # Run nsys stats, but don't crash if it fails
            try:
                subprocess.run(
                    [
                        NSYS_BIN, "stats",
                        "--report", "gpukernsum",
                        "-f", "csv",
                        "-o", nsys_kernel_csv,
                        nsys_report_path,
                    ],
                    check=False,
                )
            except Exception as e:
                print(f"[monitor_wrapper] WARNING: nsys stats failed: {e}", file=sys.stderr)
                nsys_kernel_csv = None
        else:
            print(
                f"[monitor_wrapper] WARNING: No Nsight report file (.nsys-rep/.qdrep) found at base {nsys_path_base}",
                file=sys.stderr,
            )

    # Summarize time-weighted averages from CSV
    tw = summarize_timeseries(csv_path)

    report = {
        "label": args.label,
        "started_at": None,
        "outdir": outdir,
        "wall_s": exact.get("wall_s") if exact else None,
        "cpu_user_s": exact.get("cpu_user_s") if exact else None,
        "cpu_sys_s": exact.get("cpu_sys_s") if exact else None,
        "cpu_total_s": (
            (exact.get("cpu_user_s") or 0) + (exact.get("cpu_sys_s") or 0)
        )
        if exact
        else None,
        "peak_ram_mb_exact": exact.get("peak_ram_mb_exact") if exact else None,
        "avg_ram_mb_time_weighted": tw.get("avg_ram_mb_time_weighted"),
        "peak_ram_mb_from_csv": tw.get("peak_ram_mb_from_csv"),
        "duration_s_from_csv": tw.get("duration_s_from_csv"),
        "gpus": tw.get("gpus", {}),
        "cycles_C": cycles,
        "stdout": stdout_path,
        "stderr": stderr_path,
        "timeseries_csv": csv_path,
        "nsys_report": nsys_report_path,
        "nsys_kernel_csv": nsys_kernel_csv,
        "returncode": returncode,
    }

    with open(summary_json, "w") as jf:
        json.dump(report, jf, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()