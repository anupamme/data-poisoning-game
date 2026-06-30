"""
Payoff-noise sensitivity analysis for CIFAR-10.
Two tests:
  1. Bootstrap resampling (with replacement from 10 seeds)
  2. Gaussian noise injection at sigma in {0.01, 0.02, 0.05}
Reports: % mixed NE, % VoPD > 0, median/IQR of VoPD, equilibrium support stability.
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game_theory.game_solver import GameSolver
from game_theory.payoff_matrix import PayoffMatrix

RNG = np.random.default_rng(0)

SEEDS = list(range(42, 52))


def build_seed_matrices(base_dir):
    """Load per-seed payoff data from the 10-seed experiment directory."""
    mats = {}
    attacks = defenses = None
    for seed in SEEDS:
        seed_path = os.path.join(base_dir, f"seed_{seed}", "per_seed_results.json")
        payoff_path = os.path.join(base_dir, f"seed_{seed}", "payoff_results.json")
        if not os.path.exists(seed_path):
            print(f"Warning: missing {seed_path}, skipping seed {seed}")
            continue
        with open(seed_path) as f:
            per_seed = json.load(f)
        if attacks is None:
            attack_set, defense_set = set(), set()
            with open(payoff_path) as f:
                payoff = json.load(f)
            for v in payoff.values():
                attack_set.add(v["attack"])
                defense_set.add(v["defense"])
            attacks = sorted(attack_set)
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


def compute_vopd(results, attacks, defenses):
    try:
        pm = PayoffMatrix.from_experiment_results(results, attacks, defenses)
        solver = GameSolver(pm)
        equilibria = solver.solve_nash()
        if not equilibria:
            return 0.0, False, None
        best_vopd = 0.0
        best_mixed = False
        best_support = None
        for ne in equilibria:
            v = ne.value_of_information(pm)
            mixed = (ne.adversary_strategy > 0.01).sum() > 1 or (ne.server_strategy > 0.01).sum() > 1
            if v > best_vopd:
                best_vopd = v
                best_mixed = mixed
                best_support = (
                    [attacks[i] for i, p in enumerate(ne.adversary_strategy) if p > 0.01],
                    [defenses[j] for j, p in enumerate(ne.server_strategy) if p > 0.01],
                )
        return best_vopd, bool(best_mixed), best_support
    except Exception:
        return 0.0, False, None


def run_bootstrap(mats, attacks, defenses, n_resamples=500):
    seeds = list(mats.keys())
    vopds = []
    mixed_count = 0
    for _ in range(n_resamples):
        sampled = RNG.choice(seeds, size=len(seeds), replace=True)
        avg_results = {}
        for a in attacks:
            for d in defenses:
                key = (a, d)
                accs, asrs, worsts = [], [], []
                for seed in sampled:
                    e = mats[seed].get(key)
                    if e:
                        accs.append(e["accuracy"])
                        asrs.append(e["attack_success_rate"])
                        worsts.append(e["worst_class_accuracy"])
                if accs:
                    avg_results[key] = {
                        "accuracy": np.mean(accs),
                        "attack_success_rate": np.mean(asrs),
                        "worst_class_accuracy": np.mean(worsts),
                    }
        v, is_mixed, _ = compute_vopd(avg_results, attacks, defenses)
        vopds.append(v)
        if is_mixed:
            mixed_count += 1
    vopds = np.array(vopds)
    return {
        "pct_mixed_ne": float(100 * mixed_count / n_resamples),
        "pct_vopd_positive": float(100 * (vopds > 1e-4).mean()),
        "median_vopd": float(np.median(vopds)),
        "q25_vopd": float(np.percentile(vopds, 25)),
        "q75_vopd": float(np.percentile(vopds, 75)),
        "mean_vopd": float(vopds.mean()),
    }


def run_noise_injection(base_results, attacks, defenses, sigmas, n_perturb=500):
    results_by_sigma = {}
    for sigma in sigmas:
        vopds = []
        mixed_count = 0
        for _ in range(n_perturb):
            noisy = {}
            for k, v in base_results.items():
                noise_acc = float(np.clip(v["accuracy"] + RNG.normal(0, sigma), 0, 1))
                noise_asr = float(np.clip(v["attack_success_rate"] + RNG.normal(0, sigma), 0, 1))
                noise_worst = float(np.clip(v["worst_class_accuracy"] + RNG.normal(0, sigma), 0, 1))
                noisy[k] = {"accuracy": noise_acc, "attack_success_rate": noise_asr, "worst_class_accuracy": noise_worst}
            v_vopd, is_mixed, _ = compute_vopd(noisy, attacks, defenses)
            vopds.append(v_vopd)
            if is_mixed:
                mixed_count += 1
        vopds = np.array(vopds)
        results_by_sigma[sigma] = {
            "pct_mixed_ne": float(100 * mixed_count / n_perturb),
            "pct_vopd_positive": float(100 * (vopds > 1e-4).mean()),
            "median_vopd": float(np.median(vopds)),
            "q25_vopd": float(np.percentile(vopds, 25)),
            "q75_vopd": float(np.percentile(vopds, 75)),
        }
    return results_by_sigma


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tenseed_dir = os.path.join(base_dir, "results", "cifar10_10seeds")

    print("Loading 10-seed CIFAR-10 results...")
    mats, attacks, defenses = build_seed_matrices(tenseed_dir)
    print(f"Loaded {len(mats)} seeds: {sorted(mats.keys())}")

    # Build averaged base results from all 10 seeds
    base_results = {}
    for a in attacks:
        for d in defenses:
            key = (a, d)
            accs, asrs, worsts = [], [], []
            for seed in sorted(mats.keys()):
                e = mats[seed].get(key)
                if e:
                    accs.append(e["accuracy"])
                    asrs.append(e["attack_success_rate"])
                    worsts.append(e["worst_class_accuracy"])
            if accs:
                base_results[key] = {
                    "accuracy": np.mean(accs),
                    "attack_success_rate": np.mean(asrs),
                    "worst_class_accuracy": np.mean(worsts),
                }

    base_vopd, base_mixed, base_support = compute_vopd(base_results, attacks, defenses)
    print(f"\nBaseline (10-seed average): VoPD={base_vopd:.4f}, mixed={base_mixed}")
    if base_support:
        print(f"  Adversary support: {base_support[0]}")
        print(f"  Server support:    {base_support[1]}")

    print("\n=== Bootstrap resampling (n=500, resample 10 seeds w/ replacement) ===")
    bs = run_bootstrap(mats, attacks, defenses, n_resamples=500)
    print(f"  Mixed NE:     {bs['pct_mixed_ne']:.1f}% of resamples")
    print(f"  VoPD > 0:     {bs['pct_vopd_positive']:.1f}% of resamples")
    print(f"  VoPD median:  {bs['median_vopd']:.4f}  IQR: [{bs['q25_vopd']:.4f}, {bs['q75_vopd']:.4f}]")
    print(f"  VoPD mean:    {bs['mean_vopd']:.4f}")

    print("\n=== Gaussian noise injection ===")
    sigmas = [0.01, 0.02, 0.05]
    noise_results = run_noise_injection(base_results, attacks, defenses, sigmas, n_perturb=500)
    for sigma, r in noise_results.items():
        print(f"  sigma={sigma:.2f}: mixed={r['pct_mixed_ne']:.1f}%, VoPD>0={r['pct_vopd_positive']:.1f}%, median VoPD={r['median_vopd']:.4f}")

    out = {
        "baseline": {"vopd": base_vopd, "mixed": base_mixed},
        "bootstrap": bs,
        "noise": {str(k): v for k, v in noise_results.items()},
    }
    out_path = os.path.join(tenseed_dir, "noise_sensitivity.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")
