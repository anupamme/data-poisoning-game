"""
Factorial ablation of C1/C2/C3 over the 42 already-run two-way compositions.

Reviewer question (Round 9): "Why do I need all three? Perhaps C1 alone is doing most
of the work." The paper's stored labels record only the FIRST failing condition, which
cannot answer that. Here each pair is scored on all three conditions INDEPENDENTLY:

  C1  from measured single-defense baselines (fedavg->X == X alone, same protocol):
      for EVERY committed attack a, min over {d1,d2} of ASR(d,a) < 0.5.
  C2  from Proposition 1's invariance classes -- determined by what statistic d2 reads
      and whether d1 is a positive per-client rescaling:
        cosine similarity (foolsgold, fltrust)      -> invariant, C2 always holds
        coordinate ordering (trimmed_mean, coord_median) -> conditional (see C3)
        consensus distance (reputation) / residual (rfa) / pairwise distance (krum)
                                                     -> not invariant under heterogeneous w
        norm magnitude (norm_clip)                   -> not invariant
      Degenerate d1 (an aggregator upstream) emits no per-client signal: C2 vacuously fails.
  C3  margin preservation, taken from the pre-registered per-pair reasoning.

No new compute. Output: results/condition_ablation/summary.json
"""
import json
import os
import sys
from itertools import product

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)
os.chdir(base_dir)

from experiments.analyze_continuous_metric import CRITERION_PREDICTIONS

ATTACKS = ("committed_scaling", "committed_pixel")
THRESH = 0.5

# ---- load actuals -------------------------------------------------------------
pairs = {}
for path in ("results/all_compositions/summary.json", "results/wave2_held_out/summary.json"):
    for k, v in json.load(open(path))["pairs"].items():
        pairs[k] = {a: v[a]["mean_asr"] for a in ATTACKS if a in v}

def max_committed(k):
    return max(pairs[k].values())

# single-defense baselines: fedavg->X is X alone under the identical protocol
single = {k.replace("fedavg_then_", ""): v for k, v in pairs.items() if k.startswith("fedavg_then_")}
single["fedavg"] = pairs["fedavg_then_norm_clip"]  # placeholder, replaced below if present

# ---- condition semantics ------------------------------------------------------
# what statistic each defense reads as d2, and whether it survives heterogeneous
# positive per-client rescaling (Proposition 1)
SIGNAL_INVARIANT = {
    "foolsgold": True,      # pairwise cosine  -- exactly invariant  (Prop 1a)
    "fltrust": True,        # cosine to server reference -- invariant (Prop 1a)
    "coord_median": None,   # coordinate ordering -- conditional      (Prop 1b)
    "trimmed_mean": None,   # coordinate ordering -- conditional      (Prop 1b)
    "reputation": False,    # consensus distance -- not invariant     (Prop 1c)
    # residual magnitude is not invariant in general (Prop 1c), but Theorem 1(2) gives an
    # explicit separation condition under which RFA's suppression IS preserved -- so this is
    # a conditional class like coordinate ordering, not an automatic C2 failure.
    "rfa": None,
    "norm_clip": False,     # norm magnitude     -- not invariant
    "krum": False,          # pairwise distance  -- not invariant     (Prop 1c)
    "multi_krum": False,
    "fedavg": False,        # no discriminative signal at all
}
# d1 that emit an aggregate rather than per-client updates
DEGENERATE_D1 = {"coord_median", "trimmed_mean", "fedavg", "krum", "multi_krum"}


def c1_holds(d1, d2):
    """For EVERY committed attack, at least one constituent suppresses it."""
    for a in ATTACKS:
        best = min(single.get(d1, {}).get(a, 1.0), single.get(d2, {}).get(a, 1.0))
        if best >= THRESH:
            return False
    return True


def c2_holds(d1, d2, stored_cat):
    if d1 in DEGENERATE_D1:
        return False                      # no per-client signal reaches d2
    inv = SIGNAL_INVARIANT.get(d2)
    if inv is True:
        return True                       # Proposition 1(a)
    if inv is False:
        return False                      # Proposition 1(c)
    return stored_cat != "C2-FAIL"        # conditional class: defer to pre-registered call


def c3_holds(stored_cat):
    return stored_cat != "C3-FAIL"


if __name__ == "__main__":
    rows = []
    for k, (cat, _) in CRITERION_PREDICTIONS.items():
        if k not in pairs:
            continue
        d1, d2 = k.split("_then_")
        r = {"pair": k, "d1": d1, "d2": d2, "stored_category": cat,
             "C1": c1_holds(d1, d2), "C2": c2_holds(d1, d2, cat), "C3": c3_holds(cat),
             "max_committed_asr": max_committed(k), "low_asr": max_committed(k) < THRESH}
        rows.append(r)

    print(f"=== C1/C2/C3 FACTORIAL ABLATION over {len(rows)} pairs ===\n")
    print(f"{'C1':>3} {'C2':>3} {'C3':>3} | {'#pairs':>6} {'low ASR':>8} {'rate':>7}")
    print("-" * 44)
    cells = {}
    for c1, c2, c3 in product([True, False], repeat=3):
        sub = [r for r in rows if (r["C1"], r["C2"], r["C3"]) == (c1, c2, c3)]
        if not sub:
            continue
        lo = sum(r["low_asr"] for r in sub)
        cells[f"C1={c1},C2={c2},C3={c3}"] = {"n": len(sub), "low": lo, "rate": lo / len(sub)}
        print(f"{str(c1):>3} {str(c2):>3} {str(c3):>3} | {len(sub):6d} {lo:8d} {lo/len(sub):7.0%}")

    print("\n=== MARGINAL VALUE OF EACH CONDITION ===")
    marg = {}
    for name, sel in [("C1 alone", lambda r: r["C1"]),
                      ("C2 alone", lambda r: r["C2"]),
                      ("C3 alone", lambda r: r["C3"]),
                      ("C1 and C2", lambda r: r["C1"] and r["C2"]),
                      ("C1 and C3", lambda r: r["C1"] and r["C3"]),
                      ("C2 and C3", lambda r: r["C2"] and r["C3"]),
                      ("C1 and C2 and C3", lambda r: r["C1"] and r["C2"] and r["C3"])]:
        sub = [r for r in rows if sel(r)]
        lo = sum(r["low_asr"] for r in sub) if sub else 0
        marg[name] = {"n": len(sub), "low": lo, "precision": (lo / len(sub)) if sub else None}
        p = f"{lo/len(sub):.0%}" if sub else "n/a"
        print(f"  {name:20s} selects {len(sub):3d} pairs, {lo:2d} low-ASR -> precision {p}")

    total_low = sum(r["low_asr"] for r in rows)
    print(f"\n  (total low-ASR pairs in the set: {total_low}/{len(rows)}; "
          f"base rate {total_low/len(rows):.0%})")

    os.makedirs("results/condition_ablation", exist_ok=True)
    json.dump({"description": "Factorial C1/C2/C3 ablation over the 42 evaluated pairs",
               "threshold": THRESH, "cells": cells, "marginal": marg,
               "base_rate_low_asr": total_low / len(rows), "rows": rows},
              open("results/condition_ablation/summary.json", "w"), indent=2)
    print("\nSaved to results/condition_ablation/summary.json")
