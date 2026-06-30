"""
τ sweep: NormClip threshold sensitivity at N=10, K=5, 5 seeds.

Tests whether CIFAR-10 complementarity diagnosis survives across τ ∈ {1, 5, 10}.
τ=5 re-uses existing cifar10_10seeds data (seeds 42-46); τ=1 and τ=10 are new runs.

Output: results/cifar10_tau_sweep/tau_{value}/seed_{seed}/
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FLConfig, ExperimentConfig, GameConfig
from experiments.run_payoff_matrix import run_full_payoff_matrix
from game_theory.game_solver import GameSolver
from game_theory.payoff_matrix import PayoffMatrix

fl_config = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
game_config = GameConfig(
    attacks=["no_attack", "backdoor_pixel", "model_scaling", "dba"],
    defenses=["fedavg", "norm_clip", "rfa", "trimmed_mean", "coord_median"],
)
SEEDS = [42, 43, 44, 45, 46]
TAU_VALUES = [1.0, 10.0]  # τ=5 reused from existing data

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sweep_base = os.path.join(base_dir, "results", "cifar10_tau_sweep")

print(f"τ sweep: NormClip threshold sensitivity")
print(f"N=10, K=5, seeds {SEEDS}, τ values (new): {TAU_VALUES}")
print(f"Output base: {sweep_base}\n")


def run_tau(tau_val):
    tau_str = str(int(tau_val)) if tau_val == int(tau_val) else str(tau_val)
    tau_dir = os.path.join(sweep_base, f"tau_{tau_str}")
    os.makedirs(tau_dir, exist_ok=True)

    vopds = []
    per_seed_nes = []

    for seed in SEEDS:
        seed_dir = os.path.join(tau_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        if os.path.exists(os.path.join(seed_dir, "per_seed_results.json")):
            print(f"  τ={tau_val} seed {seed}: already complete, loading...")
        else:
            exp_config = ExperimentConfig(
                dataset="cifar10",
                model="cifar_cnn",
                dirichlet_alpha=0.5,
                adversarial_fraction=0.2,
                num_trials=1,
                seed=seed,
                device="mps",
            )
            print(f"\n  --- τ={tau_val}, Seed {seed} ---")
            run_full_payoff_matrix(fl_config, exp_config, game_config, seed_dir,
                                   norm_clip_tau=tau_val)

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

        complementarity_correct = []
        seed_nes = []
        for i, ne in enumerate(equilibria):
            adv = [game_config.attacks[j] for j, p in enumerate(ne.adversary_strategy) if p > 0.01]
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
            complementarity_correct.append(theorem_predicts_null == actual_null)

            seed_nes.append({
                "ne_idx": i + 1,
                "adversary_utility": float(ne.adversary_utility),
                "server_utility": float(ne.server_utility),
                "vopd": vopd_val,
                "adversary_support": adv,
                "server_support": srv,
                "complementarity_correct": theorem_predicts_null == actual_null,
            })
            print(f"    NE{i+1}: U_A={ne.adversary_utility:.4f}, VoPD={vopd_val:.4f}")
            print(f"      Adv: {adv}  Srv: {srv}")

        # Record ASR for key cells
        asr_scaling_fedavg = None
        asr_scaling_nclip = None
        asr_pixel_nclip = None
        for key, val_list in [("model_scaling_fedavg", None), ("model_scaling_norm_clip", None),
                               ("backdoor_pixel_norm_clip", None)]:
            for e in psr.get(key, []):
                if e["seed"] == seed:
                    if key == "model_scaling_fedavg":
                        asr_scaling_fedavg = e["attack_success_rate"]
                    elif key == "model_scaling_norm_clip":
                        asr_scaling_nclip = e["attack_success_rate"]
                    elif key == "backdoor_pixel_norm_clip":
                        asr_pixel_nclip = e["attack_success_rate"]
                    break

        all_correct = all(complementarity_correct)
        per_seed_nes.append({
            "seed": seed,
            "tau": tau_val,
            "best_vopd": float(best_vopd),
            "mixed": is_mixed,
            "complementarity_all_correct": all_correct,
            "asr_model_scaling_fedavg": asr_scaling_fedavg,
            "asr_model_scaling_nclip": asr_scaling_nclip,
            "asr_pixel_nclip": asr_pixel_nclip,
            "gap_scaling_minus_pixel_nclip": (
                (asr_scaling_nclip - asr_pixel_nclip)
                if asr_scaling_nclip is not None and asr_pixel_nclip is not None else None
            ),
            "equilibria": seed_nes,
        })
        fa_str = f"{asr_scaling_fedavg:.3f}" if asr_scaling_fedavg is not None else "N/A"
        nc_str = f"{asr_scaling_nclip:.3f}" if asr_scaling_nclip is not None else "N/A"
        pix_str = f"{asr_pixel_nclip:.3f}" if asr_pixel_nclip is not None else "N/A"
        print(f"  τ={tau_val} Seed {seed}: VoPD={best_vopd:.4f}, mixed={is_mixed}, "
              f"ASR(scaling,FA)={fa_str}, ASR(scaling,NC)={nc_str}, ASR(pixel,NC)={pix_str}")

    print(f"\n=== τ={tau_val} Summary ===")
    mixed_count = sum(v > 1e-4 for v in vopds)
    diag_correct = sum(s["complementarity_all_correct"] for s in per_seed_nes)
    gaps = [s["gap_scaling_minus_pixel_nclip"] for s in per_seed_nes if s["gap_scaling_minus_pixel_nclip"] is not None]
    print(f"Mixed NE: {mixed_count}/{len(vopds)}, Mean VoPD: {np.mean(vopds):.4f}±{np.std(vopds):.4f}")
    print(f"Diagnostic correct: {diag_correct}/{len(vopds)}")
    if gaps:
        print(f"Mean gap (ASR(scaling,NC)−ASR(pixel,NC)): {np.mean(gaps):.3f}±{np.std(gaps):.3f}")

    summary = {
        "tau": tau_val,
        "num_clients": 10,
        "clients_per_round": 5,
        "seeds": SEEDS,
        "attacks": game_config.attacks,
        "defenses": game_config.defenses,
        "vopds": [float(v) for v in vopds],
        "mixed_count": int(mixed_count),
        "n_seeds": len(vopds),
        "mean_vopd": float(np.mean(vopds)),
        "std_vopd": float(np.std(vopds)),
        "diagnostic_correct": int(diag_correct),
        "mean_gap_scaling_minus_pixel_nclip": float(np.mean(gaps)) if gaps else None,
        "per_seed": per_seed_nes,
    }
    with open(os.path.join(tau_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {tau_dir}/summary.json")
    return summary


all_summaries = {}
for tau_val in TAU_VALUES:
    print(f"\n{'='*60}")
    print(f"Running τ = {tau_val}")
    print(f"{'='*60}")
    all_summaries[tau_val] = run_tau(tau_val)

# Cross-τ summary
print("\n" + "="*60)
print("CROSS-τ SUMMARY (τ=5 from existing data):")
print("τ  | Mixed-NE | Mean VoPD | Diag OK | Mean Gap(scaling-pixel@NClip)")
for tau_val, s in sorted(all_summaries.items()):
    gaps = [p["gap_scaling_minus_pixel_nclip"] for p in s["per_seed"] if p["gap_scaling_minus_pixel_nclip"] is not None]
    gap_str = f"{np.mean(gaps):.3f}" if gaps else "N/A"
    print(f"{tau_val:4.0f} | {s['mixed_count']}/{s['n_seeds']}      | "
          f"{s['mean_vopd']:.4f}    | {s['diagnostic_correct']}/{s['n_seeds']}     | {gap_str}")
