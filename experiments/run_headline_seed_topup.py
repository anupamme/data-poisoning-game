"""
Seed top-up for the three headline compositions.

Audit finding: 16 of the 18 development-set pairs in results/all_compositions/summary.json
were run at 3 seeds (42/43/44), not 5. The 24 held-out pairs and all six single-defense
baselines are at 5 seeds (42-46). The three compositions the paper leads with --
FG->RFA (0.045), FG->CM (0.131), Rep->CM (0.352) -- are among the n=3 set, so the paper's
headline numbers rest on 3 seeds while the baselines they are compared against rest on 5.

This adds seeds 45 and 46 to those three pairs, bringing them to the same n=5 protocol,
so that reported confidence intervals are computed on a uniform basis. Existing seeds are
not re-run and existing values are not modified: results are written to a separate file and
merged at analysis time.

Config identical to results/all_compositions: N=10, K=5, f=0.2, alpha=0.5, 50 rounds,
cifar_cnn, both committed attacks.
Output: results/headline_seed_topup/summary.json
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
from config import FLConfig
from experiments.run_all_compositions import run_one

PAIRS = [
    ("foolsgold", "rfa"),
    ("foolsgold", "coord_median"),
    ("reputation", "coord_median"),
]
ATTACKS = ["committed_scaling", "committed_pixel"]
NEW_SEEDS = [45, 46]          # 42/43/44 already exist in results/all_compositions
EXISTING_SEEDS = [42, 43, 44]

out_dir = os.path.join(base, "results", "headline_seed_topup"); os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "summary.json")


if __name__ == "__main__":
    total = len(PAIRS) * len(ATTACKS) * len(NEW_SEEDS)
    print(f"=== Seed top-up: 3 headline pairs, seeds {NEW_SEEDS} ({total} runs) ===")
    print("    brings FG->RFA, FG->CM, Rep->CM from n=3 to the n=5 protocol\n", flush=True)
    cells = {}
    if os.path.exists(out_path):
        try:
            cells = json.load(open(out_path)).get("cells", {}); print(f"  resuming: {len(cells)} cells\n", flush=True)
        except Exception:
            cells = {}
    t0 = time.time(); done = 0
    for d1, d2 in PAIRS:
        for atk in ATTACKS:
            key = f"{d1}_then_{d2}|{atk}"
            existing = {r["seed"]: r for r in cells.get(key, {}).get("per_seed", [])}
            for seed in NEW_SEEDS:
                if seed in existing:
                    done += 1; continue
                t = time.time()
                acc, asr = run_one(seed, d1, d2, atk)
                existing[seed] = {"seed": seed, "accuracy": float(acc), "asr": float(asr)}
                done += 1
                print(f"  [{done}/{total}] {d1}->{d2:13s} {atk:18s} s{seed}: "
                      f"acc={acc:.3f} ASR={asr:.3f} ({time.time()-t:.0f}s)", flush=True)
                cells[key] = {"d1": d1, "d2": d2, "attack": atk,
                              "per_seed": [existing[s] for s in sorted(existing)]}
                json.dump({"description": "Seeds 45/46 for the three headline compositions; "
                                          "merge with results/all_compositions (seeds 42/43/44) for n=5",
                           "new_seeds": NEW_SEEDS, "existing_seeds": EXISTING_SEEDS,
                           "config": {"N": 10, "K": 5, "f": 0.2, "alpha": 0.5, "rounds": 50},
                           "cells": cells}, open(out_path, "w"), indent=2)

    print(f"\nWall time: {(time.time()-t0)/3600:.1f} h\nSaved to {out_path}")
    print("Run experiments/analyze_headline_cis.py to merge and report CIs.")
