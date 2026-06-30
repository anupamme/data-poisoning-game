"""
λ_c sweep on FLTrust menu (Round 44 — primary lever for 8/10).

Reviewer Q2: "FLTrust: a λ_c sweep on that menu — at what cost weight (if any)
does FLTrust enter support and produce realized VoPD > 0? This seems like the
single experiment most likely to yield the missing clean positive FL case."

Approach: REUSE existing FLTrust pilot payoff data (no new compute). U_D depends
on λ_c only via Acc - λ_c * C_D, so we can re-solve Nash equilibria at any λ_c.

Sweep: λ_c ∈ {0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3}
Per λ_c: for each seed, recompute U_D, re-solve NE, record whether FLTrust enters
support, the best VoPD, the equilibrium support structure.

Pre-committed STRONG-positive criterion (8/10 path):
  ≥3/5 seeds at some λ_c ∈ [0, 0.05] have:
    - FLTrust support weight > 0.1, AND
    - Mixed-NE VoPD > 0.05 under metric (i), AND
    - The mixed NE is genuinely mixed (not just pure pixel vs pure FLTrust)

Output:
  results/cifar10_fltrust_pilot/lambda_c_sweep.json
  results/cifar10_fltrust_pilot/lambda_c_sweep_table.tex
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GameConfig
from game_theory.game_solver import GameSolver
from game_theory.payoff_matrix import PayoffMatrix
from defenses.defense_costs import get_defense_cost
from attacks import get_attack

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pilot_dir = os.path.join(base_dir, "results", "cifar10_fltrust_pilot")

attacks = ["no_attack", "backdoor_pixel", "model_scaling", "dba"]
defenses = ["fedavg", "norm_clip", "rfa", "trimmed_mean", "coord_median", "fltrust"]
SEEDS = [42, 43, 44, 45, 46]
LAMBDA_C_VALUES = [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]


def build_payoffs(seed, lambda_c):
    """Build U_A (metric i, primary) and U_D at given λ_c."""
    with open(os.path.join(pilot_dir, f"seed_{seed}", "payoff_results.json")) as f:
        results_raw = json.load(f)

    U_A = np.zeros((len(attacks), len(defenses)))
    U_D = np.zeros((len(attacks), len(defenses)))
    for i, a in enumerate(attacks):
        for j, d in enumerate(defenses):
            r = results_raw[f"{a}_{d}"]
            asr = r["attack_success_rate"]
            acc = r["accuracy"]
            U_A[i, j] = (0 if a == "no_attack" else asr - 0.1 * get_attack(a).cost)
            U_D[i, j] = acc - lambda_c * get_defense_cost(d)
    return U_A, U_D


def analyze_seed(seed, lambda_c):
    """Solve NE at given λ_c for this seed; return support and VoPD."""
    U_A, U_D = build_payoffs(seed, lambda_c)
    pm = PayoffMatrix(attacks=attacks, defenses=defenses,
                      adversary_payoffs=U_A, server_payoffs=U_D)
    eqs = GameSolver(pm).solve_nash()
    if not eqs:
        return None

    best = max(eqs, key=lambda ne: ne.value_of_information(pm))
    vopd = float(best.value_of_information(pm))
    adv = {attacks[k]: float(best.adversary_strategy[k]) for k in range(len(attacks)) if best.adversary_strategy[k] > 0.005}
    srv = {defenses[k]: float(best.server_strategy[k]) for k in range(len(defenses)) if best.server_strategy[k] > 0.005}
    fltrust_weight = srv.get("fltrust", 0.0)
    is_mixed = (len(adv) > 1) or (len(srv) > 1)

    return {
        "vopd": vopd,
        "mixed": is_mixed,
        "fltrust_weight": fltrust_weight,
        "fltrust_in_support": fltrust_weight > 0.01,
        "adv_support": adv,
        "srv_support": srv,
    }


# Sweep
sweep_results = {}
for lc in LAMBDA_C_VALUES:
    print(f"\n=== λ_c = {lc} ===")
    seed_results = []
    for seed in SEEDS:
        r = analyze_seed(seed, lc)
        if r is None:
            continue
        r["seed"] = seed
        seed_results.append(r)
        print(f"  seed {seed}: VoPD={r['vopd']:.4f}, mixed={r['mixed']}, FLTrust weight={r['fltrust_weight']:.3f}")
        if r['fltrust_in_support']:
            print(f"    srv: {r['srv_support']}")
            print(f"    adv: {r['adv_support']}")

    vopds = [r["vopd"] for r in seed_results]
    n_mixed = sum(1 for r in seed_results if r["mixed"])
    n_fltrust = sum(1 for r in seed_results if r["fltrust_in_support"])
    sweep_results[str(lc)] = {
        "lambda_c": lc,
        "n_seeds": len(seed_results),
        "mean_vopd": float(np.mean(vopds)) if vopds else None,
        "std_vopd": float(np.std(vopds)) if vopds else None,
        "n_mixed_NE": n_mixed,
        "n_fltrust_in_support": n_fltrust,
        "per_seed": seed_results,
    }
    print(f"  Summary: mean VoPD={np.mean(vopds):.4f}, mixed NEs={n_mixed}/{len(seed_results)}, FLTrust in support={n_fltrust}/{len(seed_results)}")


# Pre-committed verdict
print("\n\n=== PRE-COMMITTED VERDICT ===")
strong_positive_lc = None
for lc in LAMBDA_C_VALUES:
    if lc > 0.05:
        continue
    sr = sweep_results[str(lc)]
    n_qualifying = sum(
        1 for r in sr["per_seed"]
        if r["fltrust_in_support"] and r["fltrust_weight"] > 0.1
        and r["vopd"] > 0.05 and r["mixed"]
    )
    if n_qualifying >= 3:
        strong_positive_lc = lc
        print(f"STRONG positive at λ_c={lc}: {n_qualifying}/5 qualifying seeds.")
        break

if strong_positive_lc is None:
    # Check MEDIUM
    medium_positive_lc = None
    for lc in LAMBDA_C_VALUES:
        if lc > 0.05:
            continue
        sr = sweep_results[str(lc)]
        if sr["n_fltrust_in_support"] >= 1 and (sr["mean_vopd"] or 0) > 0.02:
            medium_positive_lc = lc
            break
    if medium_positive_lc is not None:
        print(f"MEDIUM positive at λ_c={medium_positive_lc}: FLTrust enters support in some seeds.")
    else:
        # Check if FLTrust ever enters at λ_c=0
        sr0 = sweep_results["0.0"]
        if sr0["n_fltrust_in_support"] >= 1:
            print(f"WEAK: FLTrust enters support only at λ_c=0; {sr0['n_fltrust_in_support']}/5 seeds.")
        else:
            print(f"NONE: FLTrust never enters equilibrium support even at λ_c=0. Cost-vs-benefit explanation incomplete.")

# Save
with open(os.path.join(pilot_dir, "lambda_c_sweep.json"), "w") as f:
    json.dump({"sweep": sweep_results, "strong_positive_lc": strong_positive_lc}, f, indent=2)
print(f"\nSaved: {pilot_dir}/lambda_c_sweep.json")

# LaTeX table
lines = [
    r"\begin{table}[h]",
    r"\caption{$\lambda_c$ sweep on FLTrust menu (CIFAR-10, 5 seeds, metric (i)). We re-solve Nash equilibria at each $\lambda_c$ using the cached payoff matrices. ``FLTrust support'' = number of seeds whose best NE has FLTrust with weight $>0.01$. ``Mixed NEs'' = seeds with a genuinely mixed equilibrium.}",
    r"\label{tab:lambda_c_sweep}",
    r"\centering\small",
    r"\begin{tabular}{lcccc}",
    r"\toprule",
    r"$\lambda_c$ & Mean VoPD $\pm$ std & Mixed NEs & FLTrust in support & Best NE composition example \\",
    r"\midrule",
]

for lc in LAMBDA_C_VALUES:
    sr = sweep_results[str(lc)]
    mean_vopd = sr["mean_vopd"]
    std_vopd = sr["std_vopd"]
    n_mixed = sr["n_mixed_NE"]
    n_fl = sr["n_fltrust_in_support"]
    n_s = sr["n_seeds"]
    # Example from first FLTrust-in-support seed if any
    example = ""
    for r in sr["per_seed"]:
        if r["fltrust_in_support"]:
            srv = ", ".join(f"{k}={v:.2f}" for k, v in r["srv_support"].items())
            adv = ", ".join(f"{k}={v:.2f}" for k, v in r["adv_support"].items())
            example = f"srv\\{{{srv}\\}} vs adv\\{{{adv}\\}}"
            break
    lines.append(f"${lc:.3f}$ & ${mean_vopd:.4f}\\pm{std_vopd:.3f}$ & {n_mixed}/{n_s} & {n_fl}/{n_s} & {example} \\\\")

lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]
latex = "\n".join(lines)
with open(os.path.join(pilot_dir, "lambda_c_sweep_table.tex"), "w") as f:
    f.write(latex)
print(f"Saved: {pilot_dir}/lambda_c_sweep_table.tex")
print("\nLaTeX table:")
print(latex)
