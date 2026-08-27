"""
POST-HOC seed top-up for the matched-d_2 contrasts of the metric-swap suite.

WHY THIS IS A SEPARATE FILE AND A SEPARATE RESULTS DIRECTORY
------------------------------------------------------------
The frozen label test in experiments/pre_registration_metric_swap.md is scored on seeds 42--44,
and non-negotiable #2 of that pre-registration forbids changing the seed count. That test is
already complete and reported (8/8, TN=8) and its scoring MUST NOT move. So nothing here is
written into results/metric_swap/; this suite writes results/metric_swap_topup/ and is reported
as an explicitly post-hoc power analysis of one comparison the frozen suite happened to contain.

WHAT IS BEING TESTED
--------------------
The frozen suite scores each pair against a predicted label, which is confounded: C1 and C2
co-vary across pairs. But a subset of its cells supports a comparison that is NOT confounded --
the same d_2, the same attack, the same standalone baseline, under two upstream transforms that
differ in how much they actually disturb d_2's statistic (measured independently in
results/cos_invariance_check.json). C1 is then identical by construction and only C2 varies.

  d_2         attack   upstream    measured disturbance   n=3 ASR
  krum        scaling  norm_clip   0/9 rounds             0.062
  krum        scaling  rfa         6/9 rounds             0.691    <-- gap +0.629, but p=0.14 at n=3
  reputation  scaling  norm_clip   3/9 rounds             0.792
  reputation  scaling  rfa         9/9 rounds             0.882
  cos_krum    pixel    norm_clip   0/9 rounds             0.300    invariant control:
  cos_krum    pixel    rfa         0/9 rounds             0.273    undisturbed under BOTH

At n=3 the krum/scaling gap is large in mean but not significant (Welch p=0.159, paired p=0.142):
per-seed norm_clip 0.120/0.018/0.047 vs rfa 0.974/0.116/0.982, i.e. two seeds separate cleanly and
seed 43 gives only +0.098. Adding seeds 45 and 46 buys the power to say whether that is a real
effect or seed noise. The direction of the test is fixed here, before the new seeds run: the
prediction is that the disturbed arm is higher, and if the topped-up gap is not significant that is
reported as a null.

6 cells x 2 new seeds = 12 runs (~2.2 h). Protocol otherwise identical to the frozen suite.
Output: results/metric_swap_topup/summary.json
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
from experiments.run_metric_swap_suite import run_one, SEEDS as FROZEN_SEEDS

# (d1, d2, attack) -- the cells carrying the matched-d_2 contrasts, nothing else.
CELLS = [
    ("norm_clip", "krum",       "committed_scaling"),
    ("rfa",       "krum",       "committed_scaling"),
    ("norm_clip", "reputation", "committed_scaling"),
    ("rfa",       "reputation", "committed_scaling"),
    ("norm_clip", "cos_krum",   "committed_pixel"),
    ("rfa",       "cos_krum",   "committed_pixel"),
]
NEW_SEEDS = [45, 46]
# Fixed before the new seeds run: within each (d_2, attack), the arm with the higher measured
# disturbance rate is predicted to have the higher ASR. A non-significant topped-up gap is a null.
DIRECTION = {("krum", "committed_scaling"): "rfa",
             ("reputation", "committed_scaling"): "rfa",
             ("cos_krum", "committed_pixel"): None}   # both undisturbed: no gap predicted

out_dir = os.path.join(base, "results", "metric_swap_topup")
out_path = os.path.join(out_dir, "summary.json")

if __name__ == "__main__":
    assert not set(NEW_SEEDS) & set(FROZEN_SEEDS), "top-up seeds must not overlap the frozen seeds"
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== POST-HOC seed top-up: {len(CELLS)} cells x {len(NEW_SEEDS)} seeds = "
          f"{len(CELLS)*len(NEW_SEEDS)} runs ===")
    print("    frozen label scoring in results/metric_swap/ is NOT modified\n", flush=True)
    cells = {}
    if os.path.exists(out_path):
        try:
            cells = json.load(open(out_path)).get("cells", {})
            print(f"  resuming: {len(cells)} cells\n", flush=True)
        except Exception:
            cells = {}
    t0 = time.time(); done = 0; total = len(CELLS) * len(NEW_SEEDS)
    for d1, d2, atk in CELLS:
        key = f"{d1}_then_{d2}|{atk}"
        ps = cells.get(key, {}).get("per_seed", [])
        have = {r["seed"] for r in ps}
        for seed in NEW_SEEDS:
            if seed in have:
                continue
            t = time.time(); acc, asr = run_one(seed, d1, d2, atk)
            ps.append({"seed": seed, "accuracy": acc, "asr": asr}); done += 1
            print(f"  [{done}/{total}] {d1}->{d2:12s} {atk:18s} s{seed}: "
                  f"acc={acc:.3f} ASR={asr:.3f} ({time.time()-t:.0f}s)", flush=True)
        cells[key] = {"d1": d1, "d2": d2, "attack": atk, "seeds": NEW_SEEDS, "per_seed": ps}
        json.dump({"description": "POST-HOC seed top-up (45, 46) for the matched-d_2 contrasts of "
                                  "the metric-swap suite; frozen n=3 scoring unchanged",
                   "predicted_higher_arm": {f"{k[0]}|{k[1]}": v for k, v in DIRECTION.items()},
                   "cells": cells}, open(out_path, "w"), indent=2)
    print(f"\nWall time: {(time.time()-t0)/3600:.1f} h\nSaved to {out_path}")
    print("Now run: python3 experiments/analyze_metric_swap.py --with-topup")
