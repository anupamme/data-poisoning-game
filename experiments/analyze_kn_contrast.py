"""
Analyze K/N-fixed contrast: N=20 K=10 (K/N=0.5) vs N=20 K=4 (K/N=0.2).

Reads:
  results/cifar10_20clients_k10/seed_{42-46}/  (new K/N=0.5 run)
  results/cifar10_20clients/seed_{42-46}/      (existing K/N=0.2 run)

Outputs:
  results/cifar10_20clients_k10/kn_contrast_table.json
  results/cifar10_20clients_k10/kn_contrast_latex.tex
"""
import json
import os
import sys
import numpy as np
from scipy.special import comb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GameConfig
from game_theory.game_solver import GameSolver
from game_theory.payoff_matrix import PayoffMatrix

game_config = GameConfig(
    attacks=["no_attack", "backdoor_pixel", "model_scaling", "dba"],
    defenses=["fedavg", "norm_clip", "rfa", "trimmed_mean", "coord_median"],
)
SEEDS = [42, 43, 44, 45, 46]
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def participation_prob(N, K, f=0.2):
    n_adv = int(N * f)
    n_ben = N - n_adv
    if K > n_ben:
        return 1.0
    p_none = comb(n_ben, K, exact=True) / comb(N, K, exact=True)
    return 1.0 - p_none


def analyze_config(label, result_dir, N, K, seeds=SEEDS):
    p_part = participation_prob(N, K)
    vopds = []
    mixed_count = 0
    diag_correct_count = 0
    asr_vals = []
    per_seed = []

    for seed in seeds:
        seed_dir = os.path.join(base_dir, "results", result_dir, f"seed_{seed}")
        psr_path = os.path.join(seed_dir, "per_seed_results.json")
        if not os.path.exists(psr_path):
            print(f"  WARNING: {label} seed {seed}: missing {psr_path}")
            continue

        with open(psr_path) as f:
            psr = json.load(f)

        results = {}
        for a in game_config.attacks:
            for d in game_config.defenses:
                key = f"{a}_{d}"
                if key in psr:
                    for e in psr[key]:
                        if e.get("seed") == seed:
                            results[(a, d)] = e
                            break

        pm = PayoffMatrix.from_experiment_results(results, game_config.attacks, game_config.defenses)
        solver = GameSolver(pm)
        equilibria = solver.solve_nash()

        best_vopd = max((ne.value_of_information(pm) for ne in equilibria), default=0.0)
        is_mixed = any(
            (ne.adversary_strategy > 0.01).sum() > 1 or (ne.server_strategy > 0.01).sum() > 1
            for ne in equilibria
        )
        if is_mixed:
            mixed_count += 1
        vopds.append(best_vopd)

        # Complementarity check
        all_correct = True
        for ne in equilibria:
            srv = [game_config.defenses[j] for j, p in enumerate(ne.server_strategy) if p > 0.01]
            vopd_val = float(ne.value_of_information(pm))
            srv_indices = [game_config.defenses.index(d) for d in srv]
            best_attacks_per_def = [
                set(np.where(pm.adversary_payoffs[:, j] == pm.adversary_payoffs[:, j].max())[0])
                for j in srv_indices
            ]
            intersection = best_attacks_per_def[0]
            for s in best_attacks_per_def[1:]:
                intersection = intersection & s
            has_common_best_response = len(intersection) > 0
            theorem_predicts_null = has_common_best_response
            actual_null = vopd_val < 1e-4
            if theorem_predicts_null != actual_null:
                all_correct = False
        if all_correct:
            diag_correct_count += 1

        # ASR for binding cell
        asr_scaling_fa = None
        for e in psr.get("model_scaling_fedavg", []):
            if e.get("seed") == seed:
                asr_scaling_fa = e["attack_success_rate"]
                break
        if asr_scaling_fa is not None:
            asr_vals.append(asr_scaling_fa)

        per_seed.append({
            "seed": seed,
            "vopd": float(best_vopd),
            "mixed": is_mixed,
            "diag_correct": all_correct,
            "asr_scaling_fa": asr_scaling_fa,
        })
        fa_str = f"{asr_scaling_fa:.3f}" if asr_scaling_fa is not None else "N/A"
        print(f"  {label} seed {seed}: VoPD={best_vopd:.4f}, mixed={is_mixed}, ASR(sca,FA)={fa_str}")

    return {
        "label": label,
        "N": N,
        "K": K,
        "kn_ratio": K / N,
        "participation_prob": float(p_part),
        "n_seeds": len(vopds),
        "mean_vopd": float(np.mean(vopds)) if vopds else None,
        "std_vopd": float(np.std(vopds)) if vopds else None,
        "mixed_count": mixed_count,
        "diag_correct_count": diag_correct_count,
        "mean_asr_scaling_fa": float(np.mean(asr_vals)) if asr_vals else None,
        "std_asr_scaling_fa": float(np.std(asr_vals)) if asr_vals else None,
        "per_seed": per_seed,
    }


print("=== K/N-fixed contrast analysis ===\n")

configs = [
    ("N=20, K=4 (K/N=0.20, existing)",  "cifar10_20clients",    20, 4),
    ("N=20, K=10 (K/N=0.50, new)",       "cifar10_20clients_k10", 20, 10),
]

results = {}
for label, result_dir, N, K in configs:
    print(f"--- {label} ---")
    r = analyze_config(label, result_dir, N, K)
    results[label] = r

# Print summary
print("\n" + "="*80)
print("K/N CONTRAST SUMMARY")
print(f"{'Config':>30} | {'K/N':>5} | {'P(adv)':>6} | {'MixNE':>5} | {'MeanVoPD':>8} | {'DiagOK':>6}")
print("-"*80)
for label, r in results.items():
    n = r["n_seeds"]
    vopd = f"{r['mean_vopd']:.4f}±{r['std_vopd']:.4f}" if r["mean_vopd"] is not None else "N/A"
    print(f"{label[:30]:>30} | {r['kn_ratio']:.2f} | {r['participation_prob']:.3f} | "
          f"{r['mixed_count']}/{n} | {vopd} | {r['diag_correct_count']}/{n}")

# Save
out_dir = os.path.join(base_dir, "results", "cifar10_20clients_k10")
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(out_dir, "kn_contrast_table.json"), "w") as f:
    json.dump({k: v for k, v in results.items()}, f, indent=2)
print(f"\nSaved: {out_dir}/kn_contrast_table.json")

# LaTeX table row for the N-sweep table
latex_lines = []
for label, r in results.items():
    n = r["n_seeds"]
    if n == 0:
        continue
    vopd_str = f"${r['mean_vopd']:.3f}\\pm{r['std_vopd']:.3f}$" if r["mean_vopd"] is not None else "---"
    row = (f"$N={r['N']}$, $K={r['K']}$ & {r['kn_ratio']:.2f} & {r['participation_prob']:.3f} & "
           f"{r['mixed_count']}/{n} & {vopd_str} & {r['diag_correct_count']}/{n} \\\\")
    latex_lines.append(f"% {label}\n" + row)

latex_fragment = "\n".join(latex_lines)
with open(os.path.join(out_dir, "kn_contrast_latex.tex"), "w") as f:
    f.write("% K/N contrast rows for N-sweep table\n% N, K/N, P(>=1 adv), MixedNE, MeanVoPD, DiagOK\n")
    f.write(latex_fragment + "\n")
print(f"Saved: {out_dir}/kn_contrast_latex.tex")
print("\nLaTeX rows:")
print(latex_fragment)
