"""
Round 46 — compute the 30-seed realized-VoPD statistic from scale_up_30seeds output.

Inputs (read-only):
  results/cifar10_10seeds/seed_{42..71}/per_seed_results.json     -- PS baselines
  results/randomized_defense/summary.json                          -- BR seeds 42-56
  results/randomized_defense/summary_extended.json                 -- BR seeds 57-71

Formula (matching the published -0.0001 +/- 0.083 at n=10):
  full_info_expected_payoff(seed) = 0.26 * max_a pure_ASR(a, fedavg)(seed)
                                  + 0.74 * max_a pure_ASR(a, norm_clip)(seed)
  realized_NE_payoff(seed)        = realized ASR of the *committed* BR attack
                                    (model_scaling -- chosen because mean across seeds
                                    is highest under NE3, the same commitment the
                                    paper's BR adversary makes)
  realized_VoPD(seed)             = full_info_expected - realized_NE
  The BR adversary commits to one attack across all seeds; the published value
  reflects the variance from the seed-45 scaling-failure outlier (realized_scaling=0.70
  vs typical 0.97).

For each n in {10, 15, 20, 25, 30}: report mean, std, #seeds with VoPD <= 0.
"""
import json
import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

PS_ROOT = os.path.join(base_dir, "results", "cifar10_10seeds")
BR_BASE = os.path.join(base_dir, "results", "randomized_defense", "summary.json")
BR_EXT = os.path.join(base_dir, "results", "randomized_defense", "summary_extended.json")
OUTPUT = os.path.join(base_dir, "results", "randomized_defense", "vopd_30seed_analysis.json")

P_FA = 0.26
P_NC = 0.74

ATTACKS = ["model_scaling", "backdoor_pixel"]
DEFENSES = ["fedavg", "norm_clip"]


def load_ps_for_seed(seed):
    """Return dict (attack, defense) -> asr from cached per_seed_results.json."""
    path = os.path.join(PS_ROOT, f"seed_{seed}", "per_seed_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    out = {}
    for attack in ATTACKS:
        for defense in DEFENSES:
            key = f"{attack}_{defense}"
            if key not in data:
                return None
            entry = next((e for e in data[key] if e["seed"] == seed), None)
            if entry is None:
                return None
            out[(attack, defense)] = entry["attack_success_rate"]
    return out


def load_br_realized(seed):
    """Return dict attack -> realized ASR for this seed from BR runs.

    Combines summary.json (seeds 42-56) and summary_extended.json (seeds 57-71).
    Returns None if any required attack data is missing.
    """
    # Check base
    if os.path.exists(BR_BASE):
        with open(BR_BASE) as f:
            base = json.load(f)
        if seed in base.get("seeds", []):
            idx = base["seeds"].index(seed)
            out = {}
            for attack in ATTACKS:
                a = base["attacks"].get(attack, {})
                arr = a.get("realized_asr", {})
                if isinstance(arr, dict):
                    arr = arr.get("per_seed", [])
                if idx < len(arr):
                    out[attack] = arr[idx]
                else:
                    return None
            return out
    if os.path.exists(BR_EXT):
        with open(BR_EXT) as f:
            ext = json.load(f)
        # For the committed-BR formula we only need model_scaling realized.
        out = {}
        for attack in ATTACKS:
            a = ext["attacks"].get(attack, {})
            entry = next((e for e in a.get("per_seed", []) if e["seed"] == seed), None)
            if entry is not None:
                out[attack] = entry["attack_success_rate"]
        if "model_scaling" in out:
            return out
    return None


def compute_vopd(ps, br_realized):
    """Per-seed realized VoPD using the committed-BR formula.

    BR adversary commits to model_scaling (the highest-mean attack under NE3 across
    seeds; mean realized = 0.956 vs pixel = 0.822 over 15 seeds). The committed
    BR's realized payoff per seed is the realized scaling ASR, NOT max over attacks.
    """
    fi_fa = max(ps[(a, "fedavg")] for a in ATTACKS)
    fi_nc = max(ps[(a, "norm_clip")] for a in ATTACKS)
    full_info = P_FA * fi_fa + P_NC * fi_nc
    realized = br_realized["model_scaling"]
    return full_info - realized, full_info, realized


def main():
    per_seed = []
    for seed in range(42, 72):
        ps = load_ps_for_seed(seed)
        if ps is None:
            print(f"  seed {seed}: PS data missing, skip")
            continue
        br = load_br_realized(seed)
        if br is None:
            print(f"  seed {seed}: BR data missing, skip")
            continue
        vopd, fi, rl = compute_vopd(ps, br)
        per_seed.append({
            "seed": seed, "vopd": vopd, "full_info_payoff": fi, "realized_payoff": rl,
            "pure_scaling_fa": ps[("model_scaling", "fedavg")],
            "pure_scaling_nc": ps[("model_scaling", "norm_clip")],
            "pure_pixel_fa": ps[("backdoor_pixel", "fedavg")],
            "pure_pixel_nc": ps[("backdoor_pixel", "norm_clip")],
            "br_realized_scaling": br["model_scaling"],
            "br_realized_pixel": br.get("backdoor_pixel"),
        })
        print(f"  seed {seed}: VoPD = {vopd:+.4f}  (full_info={fi:.3f}, realized={rl:.3f})")

    n_seeds = len(per_seed)
    print(f"\n=== Found {n_seeds} seeds with full PS + BR data ===\n")

    def stats(vals):
        import statistics
        m = statistics.fmean(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        leq_zero = sum(1 for v in vals if v <= 0)
        return m, sd, leq_zero

    summary = {"n_total_seeds": n_seeds, "per_seed": per_seed, "by_n": {}}
    print(f"{'n':>4} {'mean':>10} {'std':>10} {'frac<=0':>10}")
    for n in [10, 15, 20, 25, 30]:
        if n > n_seeds:
            break
        sub = [s["vopd"] for s in per_seed[:n]]
        m, sd, leq = stats(sub)
        print(f"{n:>4} {m:>+10.4f} {sd:>10.4f} {leq}/{n}")
        summary["by_n"][str(n)] = {
            "mean": m, "std": sd, "n_leq_zero": leq, "total": n,
            "frac_leq_zero": leq / n,
        }

    with open(OUTPUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
