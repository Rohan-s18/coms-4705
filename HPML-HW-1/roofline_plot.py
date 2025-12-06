import matplotlib.pyplot as plt
import numpy as np

# Peak performance and bandwidth
peak_flops = 200  # GFLOP/s
peak_bw = 30      # GB/s
AI = 0.25         # arithmetic intensity for dot product

# Roofline curve
ai_vals = np.logspace(-2, 2, 100)
roofline = np.minimum(peak_bw * ai_vals, peak_flops)

# Results (AI = 0.25 for all dot products)
results = {
    "dp1 (1e6)": 1.950,
    "dp1 (3e8)": 0.770,
    "dp2 (1e6)": 7.864,
    "dp2 (3e8)": 1.157,
    "dp4 (1e6)": 0.007,
    "dp4 (3e8)": 0.007,
    "dp5 (1e6)": 12.386,
    "dp5 (3e8)": 0.668,
}

# Plot
plt.figure(figsize=(8,6))
plt.loglog(ai_vals, roofline, label="Roofline", linewidth=2, color="black")

# Horizontal line for peak performance
plt.axhline(y=peak_flops, linestyle="--", color="red", label="Peak FLOP/s")

# Plot points
for label, flops in results.items():
    plt.scatter(AI, flops, label=label)

plt.xlabel("Arithmetic Intensity (FLOPs/Byte)")
plt.ylabel("Performance (GFLOP/s)")
plt.title("Roofline Model for Dot Product Benchmarks")
plt.legend()
plt.grid(True, which="both", linestyle="--")
plt.savefig("roofline.png", dpi=300, bbox_inches="tight")
plt.show()
