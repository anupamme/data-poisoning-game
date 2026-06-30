"""
Phase portrait: three independent axes of the complementarity phase boundary.

Panel (a): noise sensitivity  — x: payoff noise σ, y: % bootstrap with mixed NE
Panel (b): defense-menu       — x: menu name, y: % seeds with mixed NE (10 seeds)
Panel (c): seed-count k       — x: k seeds averaged (bootstrap), y: % mixed NE
                                 overlay: Theorem 2 bound on Pr[correct]

Data sources (all pre-computed, no new FL training):
  results/cifar10_10seeds/noise_sensitivity.json
  results/strategy_space/robustness.json
  results/cifar10_10seeds/seed_{42-51}/per_seed_results.json  (via same pipeline as noise script)

Output: paper/figures/phase_portrait.pdf
"""
import sys, os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from game_theory.game_solver import GameSolver
from game_theory.payoff_matrix import PayoffMatrix

noise_path = os.path.join(base_dir, "results", "cifar10_10seeds", "noise_sensitivity.json")
menu_path  = os.path.join(base_dir, "results", "strategy_space", "robustness.json")
seed_base  = os.path.join(base_dir, "results", "cifar10_10seeds")
fig_path   = os.path.join(base_dir, "paper", "figures", "phase_portrait.pdf")
os.makedirs(os.path.dirname(fig_path), exist_ok=True)

with open(noise_path) as f:  noise_data = json.load(f)
with open(menu_path)  as f:  menu_data  = json.load(f)

SEEDS = list(range(42, 52))

# ── load per-seed result dicts (same as payoff_noise_sensitivity.py) ────────
def load_seed_results(seeds):
    mats = {}
    attacks = defenses = None
    for seed in seeds:
        seed_path   = os.path.join(seed_base, f"seed_{seed}", "per_seed_results.json")
        payoff_path = os.path.join(seed_base, f"seed_{seed}", "payoff_results.json")
        with open(seed_path) as f:   per_seed = json.load(f)
        if attacks is None:
            attack_set, defense_set = set(), set()
            with open(payoff_path) as f:
                payoff = json.load(f)
            for v in payoff.values():
                attack_set.add(v["attack"])
                defense_set.add(v["defense"])
            attacks  = sorted(attack_set)
            defenses = sorted(defense_set)
        results = {}
        for a in attacks:
            for d in defenses:
                key = f"{a}_{d}"
                if key in per_seed:
                    for e in per_seed[key]:
                        if e["seed"] == seed:
                            results[(a, d)] = e
                            break
        mats[seed] = results
    return mats, attacks, defenses


def compute_vopd_mixed(results, attacks, defenses):
    """Returns (vopd, is_mixed) using the project's GameSolver pipeline."""
    try:
        pm = PayoffMatrix.from_experiment_results(results, attacks, defenses)
        solver = GameSolver(pm)
        equilibria = solver.solve_nash()
        if not equilibria:
            return 0.0, False
        best_v, best_mixed = 0.0, False
        for ne in equilibria:
            v = ne.value_of_information(pm)
            mixed = ((ne.adversary_strategy > 0.01).sum() > 1 or
                     (ne.server_strategy > 0.01).sum() > 1)
            if v > best_v:
                best_v = v
                best_mixed = mixed
        return best_v, bool(best_mixed)
    except Exception:
        return 0.0, False


def bootstrap_avg_results(sampled_seeds, mats, attacks, defenses):
    """Average results across a list of seeds (may repeat)."""
    avg = {}
    for a in attacks:
        for d in defenses:
            key = (a, d)
            accs, asrs, worsts = [], [], []
            for seed in sampled_seeds:
                e = mats[seed].get(key)
                if e:
                    accs.append(e["accuracy"])
                    asrs.append(e["attack_success_rate"])
                    worsts.append(e["worst_class_accuracy"])
            if accs:
                avg[key] = {
                    "accuracy": float(np.mean(accs)),
                    "attack_success_rate": float(np.mean(asrs)),
                    "worst_class_accuracy": float(np.mean(worsts)),
                }
    return avg


# ── Panel (c): bootstrap mixed-NE rate vs k seeds averaged ──────────────────
print("Loading per-seed matrices...")
mats, attacks, defenses = load_seed_results(SEEDS)

k_values = [2, 3, 4, 5, 7, 10]
N_BOOT   = 200
rng      = np.random.default_rng(0)
panel_c_pct = []

print(f"Computing Panel (c): bootstrap (n={N_BOOT}) mixed-NE rate vs k...")
for k in k_values:
    mixed_count = 0
    for _ in range(N_BOOT):
        chosen = list(rng.choice(SEEDS, size=k, replace=True))
        avg_r  = bootstrap_avg_results(chosen, mats, attacks, defenses)
        _, mixed = compute_vopd_mixed(avg_r, attacks, defenses)
        if mixed:
            mixed_count += 1
    pct = 100 * mixed_count / N_BOOT
    panel_c_pct.append(pct)
    print(f"  k={k:2d}: {pct:.1f}%")

# Theorem 2 Corollary bound is vacuous (≥1) for k≤96 at CIFAR-10 parameters;
# the empirical panel is shown without the overlay curve.
k_smooth = np.linspace(1.5, 10.5, 200)  # retained for axis computation

# ── plot ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "xtick.labelsize": 8,
    "ytick.labelsize": 8.5, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

fig = plt.figure(figsize=(9.5, 2.8))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.40)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

color_main = "#1565C0"
color_fill = "#90CAF9"
color_pred = "#d32f2f"

# ── (a) noise sensitivity ─────────────────────────────────────────────────────
pct_noise = [
    noise_data["bootstrap"]["pct_mixed_ne"],
    noise_data["noise"]["0.01"]["pct_mixed_ne"],
    noise_data["noise"]["0.02"]["pct_mixed_ne"],
    noise_data["noise"]["0.05"]["pct_mixed_ne"],
]
x_n = np.arange(4)
ax1.plot(x_n, pct_noise, "o-", color=color_main, lw=2, ms=6, zorder=3)
ax1.fill_between(x_n, pct_noise, alpha=0.13, color=color_fill)
ax1.axhline(50, ls="--", color="gray", lw=1.0, alpha=0.5)
ax1.set_xticks(x_n)
ax1.set_xticklabels(["0\n(boot)", "0.01", "0.02", "0.05"])
ax1.set_xlabel(r"Payoff noise $\sigma$")
ax1.set_ylabel("% bootstrap with\nmixed NE")
ax1.set_ylim(0, 105)
ax1.set_yticks([0, 25, 50, 75, 100])
ax1.set_title("(a) Noise axis", fontsize=9.5)
for x, p in zip(x_n, pct_noise):
    ax1.text(x, p + 2.5, f"{p:.0f}%", ha="center", va="bottom", fontsize=7.5, color=color_main)

# ── (b) defense-menu ─────────────────────────────────────────────────────────
menu_results = menu_data["defense_menu_robustness"]
menu_names   = [r["name"] for r in menu_results]
pct_menu     = [r["pct_mixed"] for r in menu_results]
x_m = np.arange(len(menu_names))
bar_colors = [color_main if p > 0 else "#BBDEFB" for p in pct_menu]
ax2.bar(x_m, pct_menu, color=bar_colors, edgecolor="white", lw=0.5, zorder=3)
ax2.axhline(50, ls="--", color="gray", lw=1.0, alpha=0.5)
ax2.set_xticks(x_m)
ax2.set_xticklabels(menu_names, rotation=22, ha="right", fontsize=7.5)
ax2.set_ylabel("% seeds (of 10)\nwith mixed NE")
ax2.set_ylim(0, 80)
ax2.set_yticks([0, 20, 40, 60, 80])
ax2.set_title("(b) Defense-menu axis", fontsize=9.5)
for x, p in zip(x_m, pct_menu):
    ax2.text(x, p + 1.5, f"{p:.0f}%", ha="center", va="bottom", fontsize=7.5,
             color=color_main if p > 0 else "gray")

# ── (c) seed-count axis ───────────────────────────────────────────────────────
ax3.plot(k_values, panel_c_pct, "o-", color=color_main, lw=2, ms=6, zorder=3)
ax3.axhline(50, ls=":", color="gray", lw=1.0, alpha=0.5)
ax3.set_xlabel("Seeds averaged ($k$)")
ax3.set_ylabel("% bootstrap with\nmixed NE")
ax3.set_xlim(1.5, 10.5)
ax3.set_xticks(k_values)
ax3.set_ylim(0, 105)
ax3.set_yticks([0, 25, 50, 75, 100])
ax3.set_title("(c) Seed-count axis", fontsize=9.5)
for k, p in zip(k_values, panel_c_pct):
    ax3.text(k, p + 2.5, f"{p:.0f}%", ha="center", va="bottom", fontsize=7.5, color=color_main)

fig.suptitle(
    "Phase boundary of defense complementarity: three independent axes\n"
    "(CIFAR-10, 10 seeds)",
    fontsize=9, y=1.02
)

plt.savefig(fig_path)
plt.close()
print(f"\nSaved: {fig_path}")
