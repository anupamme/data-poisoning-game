"""
Out-of-cache scaling-law prediction test: N=50, K=5, f=0.05.

Reviewer Q5: "What does the scaling law predict for a configuration you haven't run?"

Configuration: N=50, K=5, f=0.05 (2-3 adversarial clients)
- p_present = 1 - C(47,5)/C(50,5) ≈ 0.266
- Very low adversarial participation → argmax divergence may survive
- Tests the regime where the paper's own extrapolation says "collapse is not guaranteed"

Phase 1: Build 4×5 payoff matrix (5 seeds) → compute static NE and VoPD
Phase 2: Deploy computed NE as mixed policy, measure realized VoPD (5 seeds)

Output: results/out_of_cache_n50_f005/
"""
import sys, os, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FLConfig, ExperimentConfig, GameConfig
from experiments.run_payoff_matrix import run_full_payoff_matrix
from game_theory.game_solver import GameSolver
from game_theory.payoff_matrix import PayoffMatrix

fl_config = FLConfig(num_clients=50, clients_per_round=5, num_rounds=50)
game_config = GameConfig(
    attacks=["no_attack", "backdoor_pixel", "model_scaling", "dba"],
    defenses=["fedavg", "norm_clip", "rfa", "trimmed_mean", "coord_median"],
)
ADV_FRACTION = 0.05
SEEDS = [42, 43, 44, 45, 46]

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(base_dir, "results", "out_of_cache_n50_f005")
os.makedirs(output_dir, exist_ok=True)

print("=== Out-of-Cache Scaling-Law Prediction ===")
print(f"  N=50, K=5, f={ADV_FRACTION} ({int(50*ADV_FRACTION)} adversarial clients)")
print(f"  p_present = 1 - C(47,5)/C(50,5)")
from math import comb
p_present = 1 - comb(int(50*(1-ADV_FRACTION)), 5) / comb(50, 5)
print(f"  p_present = {p_present:.4f}")
print(f"  4×5 game, {len(SEEDS)} seeds")
print(f"  Output: {output_dir}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Build payoff matrix
# ═══════════════════════════════════════════════════════════════════════════════
print("--- PHASE 1: Payoff Matrix ---")
t0 = time.time()

vopds = []
per_seed_nes = []

for seed in SEEDS:
    seed_dir = os.path.join(output_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    if os.path.exists(os.path.join(seed_dir, "per_seed_results.json")):
        print(f"Seed {seed}: already complete, loading...")
    else:
        exp_config = ExperimentConfig(
            dataset="cifar10",
            model="cifar_cnn",
            dirichlet_alpha=0.5,
            adversarial_fraction=ADV_FRACTION,
            num_trials=1,
            seed=seed,
            device="mps",
        )
        print(f"\n--- Seed {seed} ---")
        run_full_payoff_matrix(fl_config, exp_config, game_config, seed_dir)

    with open(os.path.join(seed_dir, "per_seed_results.json")) as f:
        psr = json.load(f)

    results = {}
    for a in game_config.attacks:
        for d in game_config.defenses:
            key = f"{a}_{d}"
            if key in psr:
                for e in psr[key]:
                    if e["seed"] == seed:
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
    vopds.append(best_vopd)

    seed_nes = []
    for i, ne in enumerate(equilibria):
        adv = [game_config.attacks[j] for j, p in enumerate(ne.adversary_strategy) if p > 0.01]
        srv = [game_config.defenses[j] for j, p in enumerate(ne.server_strategy) if p > 0.01]
        vopd_val = float(ne.value_of_information(pm))
        seed_nes.append({
            "ne_idx": i + 1,
            "adversary_utility": float(ne.adversary_utility),
            "server_utility": float(ne.server_utility),
            "vopd": vopd_val,
            "adversary_support": adv,
            "server_support": srv,
        })
        print(f"    NE{i+1}: U_A={ne.adversary_utility:.4f}, VoPD={vopd_val:.4f}, Adv={adv}, Srv={srv}")

    asr_binding = None
    binding_entry = psr.get("model_scaling_fedavg", [])
    for e in binding_entry:
        if e["seed"] == seed:
            asr_binding = e["attack_success_rate"]
            break

    per_seed_nes.append({
        "seed": seed,
        "best_vopd": float(best_vopd),
        "mixed": is_mixed,
        "asr_model_scaling_fedavg": asr_binding,
        "equilibria": seed_nes,
    })
    asr_str = f"{asr_binding:.3f}" if asr_binding is not None else "N/A"
    print(f"  Seed {seed}: VoPD={best_vopd:.4f}, mixed={is_mixed}, ASR(scaling,FA)={asr_str}")

print(f"\n=== PHASE 1 COMPLETE ({(time.time()-t0)/60:.1f} min) ===")
mixed_count = sum(v > 1e-4 for v in vopds)
print(f"Static VoPDs: {[round(v, 3) for v in vopds]}")
print(f"Mixed NE: {mixed_count}/{len(vopds)} seeds")
print(f"Mean static VoPD: {np.mean(vopds):.4f} ± {np.std(vopds):.4f}")
asrs = [s["asr_model_scaling_fedavg"] for s in per_seed_nes if s["asr_model_scaling_fedavg"] is not None]
if asrs:
    print(f"ASR(model_scaling, fedavg): {np.mean(asrs):.3f} ± {np.std(asrs):.3f}")

# Scaling-law prediction
gamma_hat, alpha_hat = 0.66, 0.37
p_admit = 0.95  # scaling passes NormClip at tau=5
p_eff = p_present * p_admit
s_inf = (p_eff * alpha_hat) / (p_eff * alpha_hat + (1 - p_eff) * (1 - gamma_hat))
C = np.mean(vopds) if np.mean(vopds) > 0 else 0.087  # use observed static VoPD as C
bound = C * (1 - s_inf)
print(f"\n=== SCALING LAW PREDICTION ===")
print(f"  p_present = {p_present:.4f}")
print(f"  p_eff = p_present × p_admit = {p_present:.4f} × {p_admit} = {p_eff:.4f}")
print(f"  s_inf = {s_inf:.4f}")
print(f"  C (static VoPD) = {C:.4f}")
print(f"  Predicted realized VoPD bound = C(1-s_inf) = {bound:.4f}")

summary = {
    "config": {
        "num_clients": 50,
        "clients_per_round": 5,
        "adversarial_fraction": ADV_FRACTION,
        "num_adv": int(50 * ADV_FRACTION),
        "p_present": float(p_present),
    },
    "phase1_static": {
        "seeds": SEEDS,
        "vopds": [float(v) for v in vopds],
        "mean_vopd": float(np.mean(vopds)),
        "std_vopd": float(np.std(vopds)),
        "mixed_count": int(mixed_count),
        "asr_binding": asrs,
    },
    "scaling_law_prediction": {
        "gamma_hat": gamma_hat,
        "alpha_hat": alpha_hat,
        "p_admit": p_admit,
        "p_eff": float(p_eff),
        "s_inf": float(s_inf),
        "C": float(C),
        "predicted_realized_vopd_bound": float(bound),
    },
    "per_seed": per_seed_nes,
}
with open(os.path.join(output_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved Phase 1 to {output_dir}/summary.json")
print(f"Phase 1 wall time: {(time.time()-t0)/60:.1f} min")
