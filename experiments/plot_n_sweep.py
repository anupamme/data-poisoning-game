"""
N-sweep figure: scale-dependent phase boundary.

Panel (a): Mixed-NE rate (fraction of seeds with VoPD > 0) vs N
Panel (b): ASR(model_scaling, FedAvg) mean ± std vs N

Data sources (pre-computed):
  N=10:  results/cifar10_10seeds/seed_{42-51}/per_seed_results.json
  N=20:  results/20clients/game_analysis.json  (1 seed only)
  N=50:  results/cifar10_50clients/summary.json
  N=100: results/cifar10_100clients_rich/    (per_seed_results.json, 5 seeds)

Output: paper/figures/n_sweep.pdf
"""
import sys, os, json
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from game_theory.game_solver import GameSolver
from game_theory.payoff_matrix import PayoffMatrix

fig_path = os.path.join(base_dir, "paper", "figures", "n_sweep.pdf")
os.makedirs(os.path.dirname(fig_path), exist_ok=True)

# ── N=10 data (4×5 game, from pre-computed summary) ───────────────────────────
base_10    = os.path.join(base_dir, "results", "cifar10_10seeds")
n10_path   = os.path.join(base_10, "n10_4x5_summary.json")
with open(n10_path) as f: s10 = json.load(f)
seeds_10      = s10["seeds"]
vopds_10      = s10["vopds"]
mixed_10      = s10["mixed_count"]
mean_vopd_10  = s10["mean_vopd"]
asr_info_10   = s10.get("asr_binding_cell", {})
asrs_10       = asr_info_10.get("per_seed", [])
asr_mean_10   = asr_info_10.get("mean") or (float(np.mean(asrs_10)) if asrs_10 else 0.0)
asr_std_10    = asr_info_10.get("std")  or (float(np.std(asrs_10))  if asrs_10 else 0.0)
print(f"N=10: mixed={mixed_10}/{len(seeds_10)}, VoPD={mean_vopd_10:.4f}±{float(np.std(vopds_10)):.4f}, ASR={asr_mean_10:.3f}±{asr_std_10:.3f}")

# ── N=20 data (prefer new 5-seed summary, fall back to 1-seed analysis) ────────
summary_20_path = os.path.join(base_dir, "results", "cifar10_20clients", "summary.json")
if os.path.exists(summary_20_path):
    with open(summary_20_path) as f: s20 = json.load(f)
    mixed_rate_20 = s20["mixed_count"] / s20["n_seeds"]
    asr_info_20   = s20.get("asr_binding_cell", {})
    asrs_20       = asr_info_20.get("per_seed", [])
    asr_mean_20   = asr_info_20.get("mean") or (float(np.mean(asrs_20)) if asrs_20 else None)
    asr_std_20    = asr_info_20.get("std")  or (float(np.std(asrs_20))  if asrs_20 else None)
    mean_vopd_20  = s20["mean_vopd"]
    std_vopd_20   = s20["std_vopd"]
    n_seeds_20    = s20["n_seeds"]
    print(f"N=20: mixed={s20['mixed_count']}/{n_seeds_20}, VoPD={mean_vopd_20:.4f}±{std_vopd_20:.4f}, ASR={asr_mean_20:.3f}±{asr_std_20:.3f}")
else:
    # Fall back to single-seed legacy data
    path_20 = os.path.join(base_dir, "results", "20clients", "game_analysis.json")
    with open(path_20) as f: d20 = json.load(f)
    pm20 = d20["payoff_matrix"]
    attacks_20  = pm20["attacks"]
    defenses_20 = pm20["defenses"]
    adv_pay_20  = pm20["adversary_payoffs"]
    ai20 = attacks_20.index("model_scaling")
    di20 = defenses_20.index("fedavg")
    asr_20 = adv_pay_20[ai20][di20]
    nes_20    = d20["nash_equilibria"]
    vopds_20  = [ne["value_of_information"] for ne in nes_20]
    best_v20  = max(vopds_20, default=0.0)
    mixed_rate_20 = 1.0 if best_v20 > 1e-4 else 0.0
    asr_mean_20   = float(asr_20)
    asr_std_20    = None
    mean_vopd_20  = float(best_v20)
    std_vopd_20   = None
    n_seeds_20    = 1
    print(f"N=20: mixed=1/1, VoPD={mean_vopd_20:.4f}, ASR={asr_mean_20:.3f} (1 seed, legacy)")

# ── N=50 data ──────────────────────────────────────────────────────────────────
summary_50_path = os.path.join(base_dir, "results", "cifar10_50clients", "summary.json")
if os.path.exists(summary_50_path):
    with open(summary_50_path) as f: s50 = json.load(f)
    mixed_rate_50 = s50["mixed_count"] / s50["n_seeds"]
    asr_info_50   = s50.get("asr_binding_cell", {})
    asrs_50       = asr_info_50.get("per_seed", [])
    asr_mean_50   = asr_info_50.get("mean") or (float(np.mean(asrs_50)) if asrs_50 else None)
    asr_std_50    = asr_info_50.get("std")  or (float(np.std(asrs_50))  if asrs_50 else None)
    mean_vopd_50  = s50["mean_vopd"]
    n_seeds_50    = s50["n_seeds"]
    print(f"N=50: mixed={s50['mixed_count']}/{n_seeds_50}, VoPD={mean_vopd_50:.4f}, ASR={asr_mean_50:.3f}±{asr_std_50:.3f}")
else:
    print("N=50: results not yet available — using placeholder")
    mixed_rate_50 = None
    asr_mean_50   = None
    asr_std_50    = None
    mean_vopd_50  = None
    n_seeds_50    = 0

# ── N=100 data ─────────────────────────────────────────────────────────────────
base_100 = os.path.join(base_dir, "results", "cifar10_100clients_rich")
seeds_100 = [42, 43, 44, 45, 46]
asrs_100, vopds_100, mixed_100 = [], [], 0
for seed in seeds_100:
    path = os.path.join(base_100, f"seed_{seed}", "per_seed_results.json")
    if os.path.exists(path):
        with open(path) as f: psr = json.load(f)
        entries = psr.get("model_scaling_fedavg", [])
        for e in entries:
            if e["seed"] == seed:
                asrs_100.append(e["attack_success_rate"])
                break

with open(os.path.join(base_100, "summary.json")) as f: s100 = json.load(f)
mixed_rate_100 = s100["mixed_count"] / s100["n_seeds"]
asr_mean_100   = float(np.mean(asrs_100))
asr_std_100    = float(np.std(asrs_100))
mean_vopd_100  = s100["mean_vopd"]
print(f"N=100: mixed={s100['mixed_count']}/{s100['n_seeds']}, VoPD={mean_vopd_100:.4f}, ASR={asr_mean_100:.3f}±{asr_std_100:.3f}")

# ── Plot ───────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

color_main  = "#1565C0"
color_asr   = "#d32f2f"
color_fill  = "#90CAF9"
color_fill2 = "#FFCDD2"

Ns = [10, 20, 50, 100]

mean_vopds = [mean_vopd_10, mean_vopd_20, mean_vopd_50, mean_vopd_100]
std_vopds  = [float(np.std(vopds_10)),
              std_vopd_20,
              float(np.std(s50["vopds"])) if mean_vopd_50 is not None else None,
              float(np.std(s100["vopds"]))]
asr_means  = [asr_mean_10, asr_mean_20, asr_mean_50, asr_mean_100]
asr_stds   = [asr_std_10,
              asr_std_20 if asr_std_20 is not None else 0.0,
              asr_std_50,
              asr_std_100]
n_seeds    = [len(seeds_10), n_seeds_20, n_seeds_50, s100["n_seeds"]]

valid_idx = [i for i, v in enumerate(mean_vopds) if v is not None]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.0))

# ── Panel (a): mean VoPD ± std vs N ──────────────────────────────────────────
Ns_valid    = [Ns[i]       for i in valid_idx]
vopd_valid  = [mean_vopds[i] for i in valid_idx]
vstd_valid  = [std_vopds[i] if std_vopds[i] is not None else 0.0 for i in valid_idx]
ns_valid    = [n_seeds[i]  for i in valid_idx]

ax1.errorbar(Ns_valid, vopd_valid, yerr=vstd_valid,
             fmt="o-", color=color_main, lw=2, ms=7, capsize=4, capthick=1.5, zorder=3)
ax1.fill_between(Ns_valid,
                 [v - s for v, s in zip(vopd_valid, vstd_valid)],
                 [v + s for v, s in zip(vopd_valid, vstd_valid)],
                 alpha=0.15, color=color_fill)
ax1.set_xscale("log")
ax1.set_xticks(Ns_valid)
ax1.set_xticklabels([str(n) for n in Ns_valid])
ax1.set_xlabel("Number of clients $N$")
ax1.set_ylabel("Mean VoPD")
ax1.set_ylim(0, 0.12)
ax1.set_title(r"(a) Mean VoPD vs.\ $N$", fontsize=9.5)
for x, v, s in zip(Ns_valid, vopd_valid, vstd_valid):
    ax1.text(x, v + s + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color=color_main)
# Annotate 1-seed N=20 if using legacy data
if n_seeds_20 == 1:
    ax1.annotate("(1 seed)", xy=(20, mean_vopd_20), xytext=(20, mean_vopd_20 + 0.020),
                 ha="center", fontsize=7, color="gray",
                 arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))

# ── Panel (b): ASR(model_scaling, FedAvg) mean±std vs N ───────────────────────
asr_valid   = [asr_means[i] for i in valid_idx]
astd_valid  = [asr_stds[i] if asr_stds[i] is not None else 0.0 for i in valid_idx]

ax2.errorbar(Ns_valid, asr_valid, yerr=astd_valid,
             fmt="o-", color=color_asr, lw=2, ms=7, capsize=4, capthick=1.5, zorder=3)
ax2.fill_between(Ns_valid,
                 [a - s for a, s in zip(asr_valid, astd_valid)],
                 [a + s for a, s in zip(asr_valid, astd_valid)],
                 alpha=0.12, color=color_fill2)
ax2.set_xscale("log")
ax2.set_xticks(Ns_valid)
ax2.set_xticklabels([str(n) for n in Ns_valid])
ax2.set_xlabel("Number of clients $N$")
ax2.set_ylabel(r"ASR(model\_scaling, FedAvg)")
ax2.set_ylim(-0.05, 1.10)
ax2.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
ax2.set_title(r"(b) Binding cell ASR vs.\ $N$", fontsize=9.5)
for x, a in zip(Ns_valid, asr_valid):
    ax2.text(x, a + 0.05, f"{a:.2f}", ha="center", va="bottom", fontsize=7.5, color=color_asr)
if n_seeds_20 == 1:
    ax2.annotate("(1 seed)", xy=(20, asr_mean_20), xytext=(20, asr_mean_20 - 0.18),
                 ha="center", fontsize=7, color="gray",
                 arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))

fig.suptitle(
    "Scale-dependent phase boundary (CIFAR-10, $f=0.2$)\n"
    "Complementarity erodes as adversarial sampling transitions from threshold-variable to reliable",
    fontsize=8.5, y=1.03
)

plt.tight_layout()
plt.savefig(fig_path)
plt.close()
print(f"\nSaved: {fig_path}")
