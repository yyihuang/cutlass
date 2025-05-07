
import matplotlib.pyplot as plt
import pandas as pd
import re

# Constants
BYTES_PER_HALF = 2  # half precision (float16)

# Load and parse the log file
def parse_log(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    records = []
    for line in lines:
        match = re.search(
            r"CUTE_GEMM (\d+) (\d+) (\d+) [NT] [NT]:\s+\[\s*([\d.]+)]GFlop/s\s+\(([\d.]+)\)ms",
            line
        )
        if match:
            m, n, k = map(int, match.group(1, 2, 3))
            gflops = float(match.group(4))
            latency_ms = float(match.group(5))
            time_s = latency_ms / 1000.0
            # GEMM: A (m×k), B (k×n), C (m×n)
            size_bytes = (m * k + k * n + m * n) * BYTES_PER_HALF
            bandwidth = size_bytes / time_s  # bytes per second
            records.append({"M": m, "N": n, "K": k, "Bandwidth": bandwidth / (1024 * 1024 * 1024)})  # GB/s
    return pd.DataFrame(records)

df = parse_log("raw.log")

# 1. For all groups of (M, N), plot Bandwidth vs K
plt.figure(figsize=(10, 6))
for (m, n), group in df.groupby(["M", "N"]):
    sorted_group = group.sort_values("K")
    plt.plot(sorted_group["K"], sorted_group["Bandwidth"], marker='o', label=f"M={m}, N={n}")
plt.title("Bandwidth vs K for fixed (M, N)")
plt.xlabel("K")
plt.ylabel("Bandwidth (GB/s)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("bandwidth_vs_k.png")

# 2. For all groups of (M, K), plot Bandwidth vs N
plt.figure(figsize=(10, 6))
for (m, k), group in df.groupby(["M", "K"]):
    sorted_group = group.sort_values("N")
    plt.plot(sorted_group["N"], sorted_group["Bandwidth"], marker='o', label=f"M={m}, K={k}")
plt.title("Bandwidth vs N for fixed (M, K)")
plt.xlabel("N")
plt.ylabel("Bandwidth (GB/s)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("bandwidth_vs_n.png")

# 3. For all groups of (N, K), plot Bandwidth vs M
plt.figure(figsize=(10, 6))
for (n, k), group in df.groupby(["N", "K"]):
    sorted_group = group.sort_values("M")
    plt.plot(sorted_group["M"], sorted_group["Bandwidth"], marker='o', label=f"N={n}, K={k}")
plt.title("Bandwidth vs M for fixed (N, K)")
plt.xlabel("M")
plt.ylabel("Bandwidth (GB/s)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("bandwidth_vs_m.png")

print("Graphs saved as bandwidth_vs_k.png, bandwidth_vs_n.png, bandwidth_vs_m.png")
