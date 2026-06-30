"""
Generate per-seed VoPD bar chart for CIFAR-10 (seeds 42-51).
Output: paper/figures/perseed_vopd.pdf
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
summary_path = os.path.join(base_dir, "results", "cifar10_10seeds", "summary.json")
out_path = os.path.join(base_dir, "paper", "figures", "perseed_vopd.pdf")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(summary_path) as f:
    summary = json.load(f)

seeds = summary["seeds"]
vopds = summary["vopds"]
mean_vopd = np.mean(vopds)
median_vopd = np.median(vopds)

colors = ["#E87722" if v > 1e-4 else "#AAAAAA" for v in vopds]

fig, ax = plt.subplots(figsize=(7, 3.5))
bars = ax.bar([str(s) for s in seeds], vopds, color=colors, edgecolor="black", linewidth=0.7)

ax.axhline(mean_vopd, color="#E87722", linestyle="--", linewidth=1.2, label=f"Mean = {mean_vopd:.3f}")
ax.axhline(median_vopd, color="#555555", linestyle=":", linewidth=1.2, label=f"Median = {median_vopd:.3f}")

ax.set_xlabel("Seed", fontsize=11)
ax.set_ylabel("VoPD", fontsize=11)
ax.set_ylim(bottom=0)

mixed_patch = mpatches.Patch(color="#E87722", label="Mixed NE (VoPD > 0)")
pure_patch = mpatches.Patch(color="#AAAAAA", label="Pure NE (VoPD = 0)")
mean_line = plt.Line2D([0], [0], color="#E87722", linestyle="--", linewidth=1.2, label=f"Mean = {mean_vopd:.3f}")
median_line = plt.Line2D([0], [0], color="#555555", linestyle=":", linewidth=1.2, label=f"Median = {median_vopd:.3f}")
ax.legend(handles=[mixed_patch, pure_patch, mean_line, median_line], fontsize=9, loc="upper left")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(out_path, bbox_inches="tight")
print(f"Saved to {out_path}")
