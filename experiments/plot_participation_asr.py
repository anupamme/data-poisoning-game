"""
Per-seed ASR(model_scaling, FedAvg) vs. per-round adversarial participation probability.

No new training required: uses existing per_seed_results.json files.
Output: paper/figures/participation_asr_scatter.pdf
"""
import json
import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import comb

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Scale configurations: (N, K, f, result_dir, seeds)
CONFIGS = [
    (10,  5,  0.2, "cifar10_10seeds",    list(range(42, 52))),
    (20,  4,  0.2, "cifar10_20clients",  list(range(42, 47))),
    (50,  10, 0.2, "cifar10_50clients",  list(range(42, 47))),
    (100, 20, 0.2, "cifar10_100clients", list(range(42, 47))),
]


def participation_prob(N, K, f):
    """P(at least 1 adversarial client in a round of K from N total, fN adversarial)."""
    n_adv = int(N * f)
    n_ben = N - n_adv
    # P(0 adversarial) = C(n_ben, K) / C(N, K)
    if K > n_ben:
        return 1.0
    p_none = comb(n_ben, K, exact=True) / comb(N, K, exact=True)
    return 1.0 - p_none


def load_asr(result_dir, seed):
    path = os.path.join(base_dir, "results", result_dir, f"seed_{seed}", "per_seed_results.json")
    with open(path) as f:
        d = json.load(f)
    key = "model_scaling_fedavg"
    if key not in d:
        return None
    entry = d[key]
    if isinstance(entry, list):
        for e in entry:
            if e.get("seed") == seed:
                return e["attack_success_rate"]
        return entry[0]["attack_success_rate"]
    return entry["attack_success_rate"]


# Collect data
scale_data = []
for N, K, f, result_dir, seeds in CONFIGS:
    p_part = participation_prob(N, K, f)
    asr_vals = []
    for seed in seeds:
        try:
            asr = load_asr(result_dir, seed)
            if asr is not None:
                asr_vals.append(asr)
        except FileNotFoundError:
            pass
    scale_data.append({
        "N": N, "K": K, "f": f,
        "p_part": p_part,
        "asr_vals": asr_vals,
        "asr_mean": float(np.mean(asr_vals)) if asr_vals else None,
        "asr_std": float(np.std(asr_vals)) if asr_vals else None,
    })
    print(f"N={N}, K={K}, P(>=1 adv)={p_part:.3f}, ASR vals: {[f'{v:.3f}' for v in asr_vals]}")


# Plot
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

fig, ax = plt.subplots(figsize=(5.5, 3.8))

colors = ["#1565C0", "#d32f2f", "#388E3C", "#7B1FA2"]
n_labels = [10, 20, 50, 100]

for i, sd in enumerate(scale_data):
    p = sd["p_part"]
    asr_vals = sd["asr_vals"]
    N = sd["N"]
    color = colors[i]
    # Jitter x slightly so overlapping points are visible
    jitter = np.random.default_rng(N).uniform(-0.005, 0.005, len(asr_vals))
    xs = [p + j for j in jitter]
    ax.scatter(xs, asr_vals, color=color, s=45, zorder=4, alpha=0.85,
               label=f"$N={N}$ ($K={sd['K']}$)")
    if sd["asr_mean"] is not None:
        ax.errorbar([p], [sd["asr_mean"]], yerr=[sd["asr_std"]],
                    fmt="D", color=color, ms=7, capsize=4, capthick=1.5, zorder=5)

# Threshold annotation
ax.axhline(0.5, color="gray", lw=0.8, linestyle=":", alpha=0.7, label="ASR threshold (0.5)")

ax.set_xlabel("Per-round adversarial participation probability $P(\\geq\\!1\\,\\mathrm{adv/round})$")
ax.set_ylabel("ASR(model scaling, FedAvg)")
ax.set_xlim(0.55, 1.05)
ax.set_ylim(-0.05, 1.10)
ax.set_title("Per-seed ASR vs. adversarial participation probability\n"
             "(CIFAR-10, $f{=}0.2$, $4{\\times}5$ game; mean~$\\pm$~std shown as diamonds)",
             fontsize=8.5)
ax.legend(loc="lower right", fontsize=8)

# Annotate P values
for sd in scale_data:
    p = sd["p_part"]
    ax.annotate(f"$P={p:.3f}$\n$N={sd['N']}$",
                xy=(p, -0.04), ha="center", va="top", fontsize=7.5, color="black")

plt.tight_layout()
out_path = os.path.join(base_dir, "paper", "figures", "participation_asr_scatter.pdf")
plt.savefig(out_path)
plt.close()
print(f"Saved: {out_path}")
