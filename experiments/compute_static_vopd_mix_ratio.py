"""
Round 51 — Static VoPD analysis on cached cell payoffs.

For each deterministic mix policy pi over a 2-defense menu {d1, d2} with
weight (p1, p2), and adversary menu {scaling, pixel}:

  U_A(a, pi)            = p1 * U_A(a, d1) + p2 * U_A(a, d2)
  U_A_committed_max(pi) = max_a  U_A(a, pi)
  U_A_full_info(pi)     = p1 * max_a U_A(a, d1) + p2 * max_a U_A(a, d2)
  static_VoPD(pi)       = U_A_full_info(pi) - U_A_committed_max(pi)  (>= 0)

We sweep:
  - menu choices over {fedavg, norm_clip, foolsgold, reputation, rfa,
                       trimmed_mean, coord_median}
  - mix weights at 9 points {0.1, 0.2, ..., 0.9}

Output: results/static_vopd_mix_analysis/summary.json
"""
import json
import os
import sys
from itertools import combinations
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

SEEDS = [42, 43, 44, 45, 46]
ATTACKS = ["model_scaling", "backdoor_pixel", "backdoor_edge_case"]
DEFENSES = ["fedavg", "norm_clip", "foolsgold", "reputation",
            "rfa", "trimmed_mean", "coord_median"]
MIX_WEIGHTS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

out_dir = os.path.join(base_dir, "results", "static_vopd_mix_analysis")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "summary_3attack.json")


def load_cell(seed, attack, defense):
    """Returns (asr, accuracy) for a single (attack, defense) cell, or None."""
    path = os.path.join(base_dir, "results", "cifar10_10seeds",
                        f"seed_{seed}", "per_seed_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    key = f"{attack}_{defense}"
    if key not in d:
        return None
    entry = d[key][0] if isinstance(d[key], list) else d[key]
    return entry["attack_success_rate"], entry["accuracy"]


def compute_static_vopd(d1, d2, p1, seed):
    """Static VoPD for deterministic mix (p1*d1 + (1-p1)*d2), seed-specific."""
    p2 = 1.0 - p1
    cells = {}
    for a in ATTACKS:
        for d in [d1, d2]:
            c = load_cell(seed, a, d)
            if c is None:
                return None
            cells[(a, d)] = c
    # Mix U_A per attack
    U_A_mix = {a: p1 * cells[(a, d1)][0] + p2 * cells[(a, d2)][0] for a in ATTACKS}
    # Pure max per defense
    max_d1 = max(cells[(a, d1)][0] for a in ATTACKS)
    max_d2 = max(cells[(a, d2)][0] for a in ATTACKS)
    full_info = p1 * max_d1 + p2 * max_d2
    committed_max = max(U_A_mix.values())
    vopd = full_info - committed_max
    # Per-defense argmax: which attack is the oracle's best play on each defense
    argmax_d1 = max(ATTACKS, key=lambda a: cells[(a, d1)][0])
    argmax_d2 = max(ATTACKS, key=lambda a: cells[(a, d2)][0])
    return {
        "U_A_mix": {a: U_A_mix[a] for a in ATTACKS},
        "full_info": full_info,
        "committed_max": committed_max,
        "best_committed_attack": max(U_A_mix, key=U_A_mix.get),
        "argmax_per_defense": {d1: argmax_d1, d2: argmax_d2},
        "vopd": vopd,
    }


def main():
    print(f"=== Static VoPD on 2-defense mixes (cached cells, {len(SEEDS)} seeds) ===\n")
    summary = {"seeds": SEEDS, "results": []}

    # Build mean-VoPD landscape over all defense pairs and mix weights
    pair_vopd = {}  # (d1, d2) -> list of mean VoPD per weight
    for d1, d2 in combinations(DEFENSES, 2):
        for p1 in MIX_WEIGHTS:
            seed_results = []
            for s in SEEDS:
                r = compute_static_vopd(d1, d2, p1, s)
                if r is None:
                    seed_results = None
                    break
                seed_results.append(r)
            if seed_results is None:
                continue
            vopds = [r["vopd"] for r in seed_results]
            mean_vopd = float(np.mean(vopds))
            full_infos = [r["full_info"] for r in seed_results]
            mean_full_info = float(np.mean(full_infos))
            committed_maxes = [r["committed_max"] for r in seed_results]
            mean_committed_max = float(np.mean(committed_maxes))
            best_atks = [r["best_committed_attack"] for r in seed_results]
            from collections import Counter
            ba_majority = Counter(best_atks).most_common(1)[0][0]
            entry = {
                "d1": d1, "d2": d2, "p1": p1,
                "mean_vopd": mean_vopd,
                "std_vopd": float(np.std(vopds)),
                "mean_full_info": mean_full_info,
                "mean_committed_max": mean_committed_max,
                "best_committed_attack_majority": ba_majority,
                "per_seed": seed_results,
            }
            summary["results"].append(entry)
            pair_vopd.setdefault((d1, d2), []).append((p1, mean_vopd, ba_majority,
                                                      mean_full_info, mean_committed_max))

    # Identify top-10 highest static VoPD configurations
    sorted_all = sorted(summary["results"], key=lambda r: -r["mean_vopd"])
    print(f"--- Top 10 highest static VoPD across all (menu, mix) ---")
    for r in sorted_all[:10]:
        print(f"  {r['d1']:13s}+{r['d2']:13s} p1={r['p1']:.2f}  "
              f"VoPD={r['mean_vopd']:.3f}±{r['std_vopd']:.3f}  "
              f"full_info={r['mean_full_info']:.3f}  best_committed={r['mean_committed_max']:.3f} "
              f"({r['best_committed_attack_majority']})")

    # Focused report: NC + reputation menu (the planned focus)
    print(f"\n--- NC + reputation menu (planned focus) ---")
    for entry in summary["results"]:
        if entry["d1"] == "norm_clip" and entry["d2"] == "reputation":
            print(f"  NC{int(entry['p1']*100):2d}/rep{int((1-entry['p1'])*100):2d}: "
                  f"VoPD={entry['mean_vopd']:.3f}±{entry['std_vopd']:.3f}  "
                  f"full_info={entry['mean_full_info']:.3f}  committed_max={entry['mean_committed_max']:.3f} "
                  f"({entry['best_committed_attack_majority']})")

    # Identify per-pair max VoPD points
    print(f"\n--- Per-pair max VoPD point ---")
    pair_summary = []
    for (d1, d2), pts in pair_vopd.items():
        best = max(pts, key=lambda x: x[1])
        pair_summary.append((d1, d2, best))
    pair_summary.sort(key=lambda x: -x[2][1])
    for d1, d2, (p1, vopd, ba, fi, cm) in pair_summary[:15]:
        print(f"  {d1:13s}+{d2:13s} max VoPD={vopd:.3f} at p1={p1:.2f}  "
              f"full_info={fi:.3f}  committed_max={cm:.3f} ({ba})")

    # Round 52: identify menus where backdoor_edge_case is argmax of >= 1 defense
    # (the survivor pilot needs the oracle to actually pick edge_case on some round)
    if "backdoor_edge_case" in ATTACKS:
        print(f"\n--- Round 52: menus where edge_case is argmax of >=1 defense (mean across seeds) ---")
        edge_bearing = []
        for entry in summary["results"]:
            # Per-seed argmax_per_defense; majority vote across seeds
            d1, d2 = entry["d1"], entry["d2"]
            argmax_d1_votes = [s["argmax_per_defense"][d1] for s in entry["per_seed"]]
            argmax_d2_votes = [s["argmax_per_defense"][d2] for s in entry["per_seed"]]
            from collections import Counter
            argmax_d1_maj = Counter(argmax_d1_votes).most_common(1)[0][0]
            argmax_d2_maj = Counter(argmax_d2_votes).most_common(1)[0][0]
            if argmax_d1_maj == "backdoor_edge_case" or argmax_d2_maj == "backdoor_edge_case":
                edge_bearing.append({
                    "d1": d1, "d2": d2, "p1": entry["p1"],
                    "mean_vopd": entry["mean_vopd"],
                    "mean_full_info": entry["mean_full_info"],
                    "mean_committed_max": entry["mean_committed_max"],
                    "argmax_d1": argmax_d1_maj,
                    "argmax_d2": argmax_d2_maj,
                })
        edge_bearing.sort(key=lambda r: -r["mean_vopd"])
        for r in edge_bearing[:10]:
            print(f"  {r['d1']:13s}+{r['d2']:13s} p1={r['p1']:.2f}  "
                  f"VoPD={r['mean_vopd']:.3f}  "
                  f"argmax({r['d1']})={r['argmax_d1']}  argmax({r['d2']})={r['argmax_d2']}")
        if not edge_bearing:
            print("  (no menu/mix has edge_case as a per-defense argmax; "
                  "this menu axis does not provide a 3-attack survivor)")

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
