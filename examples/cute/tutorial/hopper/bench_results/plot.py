
import matplotlib.pyplot as plt
import pandas as pd
import re

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
            records.append({"M": m, "N": n, "K": k, "GFLOPS": gflops})
    return pd.DataFrame(records)

df = parse_log("/workspaces/cutlass/examples/cute/tutorial/hopper/bench_results/raw.log")

# 1. For all groups of (M, N), plot GFLOPS vs K
plt.figure(figsize=(10, 6))
for (m, n), group in df.groupby(["M", "N"]):
    sorted_group = group.sort_values("K")
    plt.plot(sorted_group["K"], sorted_group["GFLOPS"], marker='o', label=f"M={m}, N={n}")
plt.title("GFLOPS vs K for fixed (M, N)")
plt.xlabel("K")
plt.ylabel("GFLOPS")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("gflops_vs_k.png")

# 2. For all groups of (M, K), plot GFLOPS vs N
plt.figure(figsize=(10, 6))
for (m, k), group in df.groupby(["M", "K"]):
    sorted_group = group.sort_values("N")
    plt.plot(sorted_group["N"], sorted_group["GFLOPS"], marker='o', label=f"M={m}, K={k}")
plt.title("GFLOPS vs N for fixed (M, K)")
plt.xlabel("N")
plt.ylabel("GFLOPS")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("gflops_vs_n.png")

# 3. For all groups of (N, K), plot GFLOPS vs M
plt.figure(figsize=(10, 6))
for (n, k), group in df.groupby(["N", "K"]):
    sorted_group = group.sort_values("M")
    plt.plot(sorted_group["M"], sorted_group["GFLOPS"], marker='o', label=f"N={n}, K={k}")
plt.title("GFLOPS vs M for fixed (N, K)")
plt.xlabel("M")
plt.ylabel("GFLOPS")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("gflops_vs_m.png")

print("Graphs saved as gflops_vs_k.png, gflops_vs_n.png, gflops_vs_m.png")
