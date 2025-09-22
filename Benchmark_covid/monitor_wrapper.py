#!/usr/bin/env python3
import argparse, subprocess, time, psutil, threading, csv, os
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_OK = True
except Exception:
    NVML_OK = False

def monitor(pid, logfile, interval=1.0):
    proc = psutil.Process(pid)
    ghandles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())] if NVML_OK else []
    with open(logfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ts","rss_mb","cpu_percent","gpu_index","gpu_util","gpu_mem_used_mb"])
        while proc.is_running():
            ts = time.time()
            rss = proc.memory_info().rss/1e6
            cpu = proc.cpu_percent(interval=None)
            if ghandles:
                for i,h in enumerate(ghandles):
                    u = pynvml.nvmlDeviceGetUtilizationRates(h)
                    m = pynvml.nvmlDeviceGetMemoryInfo(h)
                    writer.writerow([ts,rss,cpu,i,u.gpu,m.used/1e6])
            else:
                writer.writerow([ts,rss,cpu,"","",""])
            f.flush()
            time.sleep(interval)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logfile", default="resource_log.csv")
    ap.add_argument("--cmd", required=True, help="python script and args as a string")
    args = ap.parse_args()

    child = subprocess.Popen(args.cmd.split(), preexec_fn=os.setsid)
    t = threading.Thread(target=monitor, args=(child.pid, args.logfile))
    t.start()
    child.wait()
    t.join()

if __name__ == "__main__":
    main()
