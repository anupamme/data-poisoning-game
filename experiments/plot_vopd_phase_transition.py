"""
Generate VoPD phase transition figure (two-panel).
Left: % mixed-NE vs payoff noise σ.
Right: Median VoPD vs payoff noise σ, with Q25/Q75 error bars.

Data source: results/cifar10_10seeds/noise_sensitivity.json
Output: paper/figures/vopd_phase_transition.pdf
"""
import json, os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, "results", "cifar10_10seeds", "noise_sensitivity.json")
output_path = os.path.join(base_dir, "paper", "figures", "vopd_phase_transition.pdf")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(data_path) as f:
    d = json.load(f)

# Build data arrays
sigma_labels = ["0\n(bootstrap)", "0.01", "0.02", "0.05"]
sigma_vals = [0.0, 0.01, 0.02, 0.05]

pct_mixed = [
    d["bootstrap"]["pct_mixed_ne"],
    d["noise"]["0.01"]["pct_mixed_ne"],
    d["noise"]["0.02"]["pct_mixed_ne"],
    d["noise"]["0.05"]["pct_mixed_ne"],
]

median_vopd = [
    d["bootstrap"]["median_vopd"],
    d["noise"]["0.01"]["median_vopd"],
    d["noise"]["0.02"]["median_vopd"],
    d["noise"]["0.05"]["median_vopd"],
]

q25_vopd = [
    d["bootstrap"]["q25_vopd"],
    d["noise"]["0.01"]["q25_vopd"],
    d["noise"]["0.02"]["q25_vopd"],
    d["noise"]["0.05"]["q25_vopd"],
]

q75_vopd = [
    d["bootstrap"]["q75_vopd"],
    d["noise"]["0.01"]["q75_vopd"],
    d["noise"]["0.02"]["q75_vopd"],
    d["noise"]["0.05"]["q75_vopd"],
]

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.2))
x = np.arange(len(sigma_vals))
color_main = "#1565C0"
color_fill = "#90CAF9"

# ---- Left panel: % mixed NE ----
ax1.plot(x, pct_mixed, "o-", color=color_main, linewidth=2, markersize=7, zorder=3)
ax1.fill_between(x, pct_mixed, alpha=0.15, color=color_fill)
ax1.axhline(100/3, linestyle="--", color="gray", linewidth=1.2, label="Random baseline (33%)")
ax1.set_xticks(x)
ax1.set_xticklabels(sigma_labels)
ax1.set_xlabel(r"Payoff noise $\sigma$")
ax1.set_ylabel("% Bootstrap resamples\nyielding mixed NE")
ax1.set_ylim(0, 105)
ax1.set_yticks([0, 33, 50, 65, 80, 97.8])
ax1.set_yticklabels(["0", "33", "50", "65", "80", "97.8"])
ax1.legend(loc="lower left", framealpha=0.9)
ax1.set_title("(a) Mixed-NE frequency vs.\ noise")
# Annotate the drop
ax1.annotate("97.8%", xy=(0, 97.8), xytext=(0.15, 96), fontsize=8, color=color_main)
ax1.annotate("65.0%", xy=(1, 65.0), xytext=(1.1, 68), fontsize=8, color=color_main)
ax1.annotate("55.8%", xy=(3, 55.8), xytext=(2.6, 50), fontsize=8, color=color_main)

# ---- Right panel: Median VoPD with IQR ----
err_lo = [median_vopd[i] - q25_vopd[i] for i in range(len(x))]
err_hi = [q75_vopd[i] - median_vopd[i] for i in range(len(x))]
ax2.errorbar(x, median_vopd, yerr=[err_lo, err_hi],
             fmt="o-", color=color_main, linewidth=2, markersize=7,
             capsize=4, capthick=1.5, zorder=3, label="Median VoPD ± IQR")
ax2.fill_between(x, q25_vopd, q75_vopd, alpha=0.15, color=color_fill, label="IQR")
ax2.axhline(0, linestyle="--", color="gray", linewidth=1.0, alpha=0.6)
ax2.set_xticks(x)
ax2.set_xticklabels(sigma_labels)
ax2.set_xlabel(r"Payoff noise $\sigma$")
ax2.set_ylabel("VoPD")
ax2.set_ylim(-0.01, 0.14)
ax2.legend(loc="upper right", framealpha=0.9)
ax2.set_title("(b) Median VoPD vs.\ noise")
ax2.annotate("0.085", xy=(0, 0.085), xytext=(0.1, 0.088), fontsize=8, color=color_main)
ax2.annotate("0.013", xy=(3, 0.013), xytext=(2.6, 0.028), fontsize=8, color=color_main)

fig.suptitle("VoPD phase transition under payoff estimation noise\n"
             "(CIFAR-10, 10 seeds, $n=500$ bootstrap resamples per noise level)",
             fontsize=10, y=1.02)

plt.tight_layout()
plt.savefig(output_path)
plt.close()
print(f"Saved: {output_path}")
