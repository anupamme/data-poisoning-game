"""
Analyze τ sweep results and generate paper-ready table.

Reads:
  results/cifar10_tau_sweep/tau_{1,10}/seed_{42-46}/per_seed_results.json  (new runs)
  results/cifar10_10seeds/seed_{42-46}/per_seed_results.json               (τ=5 reused)

Outputs:
  results/cifar10_tau_sweep/tau_sweep_table.json  — machine-readable
  results/cifar10_tau_sweep/tau_sweep_latex.tex   — copy-paste LaTeX table
"""
import json
import os
import sys
import numpy as np

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

TAU_DIRS = {
    1.0:  os.path.join(base_dir, "results", "cifar10_tau_sweep", "tau_1"),
    5.0:  os.path.join(base_dir, "results", "cifar10_10seeds"),      # reused
    10.0: os.path.join(base_dir, "results", "cifar10_tau_sweep", "tau_10"),
}


def analyze_tau(tau_val, tau_dir, seeds=SEEDS):
    vopds = []
    gaps = []
    asr_scaling_nc_all = []
    asr_pixel_nc_all = []
    mixed_count = 0
    diag_correct_count = 0
    per_seed = []

    for seed in seeds:
        seed_dir = os.path.join(tau_dir, f"seed_{seed}")
        psr_path = os.path.join(seed_dir, "per_seed_results.json")
        if not os.path.exists(psr_path):
            print(f"  WARNING: τ={tau_val} seed {seed}: no per_seed_results.json at {psr_path}")
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

        # ASR extraction
        asr_scaling_nc = None
        asr_pixel_nc = None
        asr_scaling_fa = None
        for key, lst in psr.items():
            for e in lst:
                if e.get("seed") == seed:
                    if key == "model_scaling_norm_clip":
                        asr_scaling_nc = e["attack_success_rate"]
                    elif key == "backdoor_pixel_norm_clip":
                        asr_pixel_nc = e["attack_success_rate"]
                    elif key == "model_scaling_fedavg":
                        asr_scaling_fa = e["attack_success_rate"]

        gap = (asr_scaling_nc - asr_pixel_nc) if asr_scaling_nc is not None and asr_pixel_nc is not None else None
        if gap is not None:
            gaps.append(gap)
        if asr_scaling_nc is not None:
            asr_scaling_nc_all.append(asr_scaling_nc)
        if asr_pixel_nc is not None:
            asr_pixel_nc_all.append(asr_pixel_nc)

        per_seed.append({
            "seed": seed,
            "vopd": float(best_vopd),
            "mixed": is_mixed,
            "diag_correct": all_correct,
            "asr_scaling_fa": asr_scaling_fa,
            "asr_scaling_nc": asr_scaling_nc,
            "asr_pixel_nc": asr_pixel_nc,
            "gap": gap,
        })
        sca_str = f"{asr_scaling_nc:.3f}" if asr_scaling_nc is not None else "N/A"
        pix_str = f"{asr_pixel_nc:.3f}" if asr_pixel_nc is not None else "N/A"
        gap_str = f"{gap:.3f}" if gap is not None else "N/A"
        print(f"  τ={tau_val} seed {seed}: VoPD={best_vopd:.4f}, mixed={is_mixed}, "
              f"ASR(sca,NC)={sca_str}, ASR(pix,NC)={pix_str}, gap={gap_str}")

    result = {
        "tau": tau_val,
        "n_seeds": len(vopds),
        "mean_vopd": float(np.mean(vopds)) if vopds else None,
        "std_vopd": float(np.std(vopds)) if vopds else None,
        "mixed_count": mixed_count,
        "diag_correct_count": diag_correct_count,
        "mean_asr_pixel_nc": float(np.mean(asr_pixel_nc_all)) if asr_pixel_nc_all else None,
        "std_asr_pixel_nc": float(np.std(asr_pixel_nc_all)) if asr_pixel_nc_all else None,
        "mean_asr_scaling_nc": float(np.mean(asr_scaling_nc_all)) if asr_scaling_nc_all else None,
        "std_asr_scaling_nc": float(np.std(asr_scaling_nc_all)) if asr_scaling_nc_all else None,
        "mean_gap": float(np.mean(gaps)) if gaps else None,
        "std_gap": float(np.std(gaps)) if gaps else None,
        "per_seed": per_seed,
    }
    return result


print("=== τ sweep analysis ===")
print(f"Analyzing τ ∈ {{1, 5, 10}}, seeds {SEEDS}\n")

all_results = {}
for tau_val in [1.0, 5.0, 10.0]:
    tau_dir = TAU_DIRS[tau_val]
    print(f"--- τ={tau_val} ---")
    r = analyze_tau(tau_val, tau_dir)
    all_results[tau_val] = r

# Print summary table
print("\n" + "="*80)
print("CROSS-τ SUMMARY")
print(f"{'τ':>4} | {'N_seed':>6} | {'MixNE':>5} | {'MeanVoPD':>8} | {'DiagOK':>6} | "
      f"{'ASR(pix,NC)':>11} | {'ASR(sca,NC)':>11} | {'Gap':>6}")
print("-"*80)
for tau_val in [1.0, 5.0, 10.0]:
    r = all_results[tau_val]
    n = r["n_seeds"]
    diag = f"{r['diag_correct_count']}/{n}"
    mixed = f"{r['mixed_count']}/{n}"
    vopd = f"{r['mean_vopd']:.4f}±{r['std_vopd']:.4f}" if r["mean_vopd"] is not None else "N/A"
    asr_pix = f"{r['mean_asr_pixel_nc']:.3f}±{r['std_asr_pixel_nc']:.3f}" if r["mean_asr_pixel_nc"] is not None else "N/A"
    asr_sca = f"{r['mean_asr_scaling_nc']:.3f}±{r['std_asr_scaling_nc']:.3f}" if r["mean_asr_scaling_nc"] is not None else "N/A"
    gap = f"{r['mean_gap']:.3f}±{r['std_gap']:.3f}" if r["mean_gap"] is not None else "N/A"
    print(f"{tau_val:>4.0f} | {n:>6} | {mixed:>5} | {vopd:>8} | {diag:>6} | {asr_pix:>11} | {asr_sca:>11} | {gap:>6}")

# Save machine-readable
sweep_base = os.path.join(base_dir, "results", "cifar10_tau_sweep")
os.makedirs(sweep_base, exist_ok=True)

table_data = {str(k): v for k, v in all_results.items()}
with open(os.path.join(sweep_base, "tau_sweep_table.json"), "w") as f:
    json.dump(table_data, f, indent=2)
print(f"\nSaved: {sweep_base}/tau_sweep_table.json")

# Generate LaTeX table
def fmt_pm(mean, std, digits=3):
    if mean is None:
        return "---"
    return f"{mean:.{digits}f}$\\pm${std:.{digits}f}"


latex_rows = []
for tau_val in [1.0, 5.0, 10.0]:
    r = all_results[tau_val]
    n = r["n_seeds"]
    if n == 0:
        continue
    compl = "\\checkmark" if r["diag_correct_count"] == n else f"{r['diag_correct_count']}/{n}"
    row = (
        f"$\\tau={int(tau_val)}$ & "
        f"{fmt_pm(r['mean_asr_pixel_nc'], r['std_asr_pixel_nc'])} & "
        f"{fmt_pm(r['mean_asr_scaling_nc'], r['std_asr_scaling_nc'])} & "
        f"{fmt_pm(r['mean_gap'], r['std_gap'])} & "
        f"{compl} & "
        f"{r['mixed_count']}/{n}"
        " \\\\"
    )
    latex_rows.append(row)

latex = r"""\begin{table}[t]
\centering
\caption{$\tau$ sweep: NormClip threshold sensitivity (CIFAR-10, $N{=}10$, $K{=}5$, 5 seeds each).
``Compl.'' = complementarity condition satisfied for all seeds;
``Gap'' = $\overline{\text{ASR}(\text{scaling},\text{NClip})} - \overline{\text{ASR}(\text{pixel},\text{NClip})}$;
``Mixed NE'' = seeds with mixed-strategy equilibrium.
$\tau{=}5$ row uses seeds 42--46 from the main experiment.}
\label{tab:tau_sweep}
\begin{tabular}{lccccc}
\toprule
$\tau$ & ASR(pixel, NClip) & ASR(scaling, NClip) & Gap & Compl.\ & Mixed NE \\
\midrule
""" + "\n".join(latex_rows) + r"""
\bottomrule
\end{tabular}
\end{table}"""

with open(os.path.join(sweep_base, "tau_sweep_latex.tex"), "w") as f:
    f.write(latex)
print(f"Saved: {sweep_base}/tau_sweep_latex.tex")
print("\nLaTeX table:")
print(latex)
