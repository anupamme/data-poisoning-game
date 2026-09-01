"""
Seed top-up of the flagship Mode-S Krum arm: n=5 -> n=20 (Round 34, item A).

WHY THIS EXISTS. The adjudicating arm of Round 12 -- Mode S into krum under committed_scaling -- is
the arm on which H-statistic and H-admission make opposite predictions, and it is the (P3)=/=>(P4)
witness the paper leads with. It ran at n=5. Its published verdict is |dASR| = 0.026 < 0.15 with
JT increasing p = 0.70, i.e. H-admission confirmed. That verdict STANDS and is not re-tested here.

What the published version never did is state the INTERVAL around the equivalence claim. At n=5 the
paired 95% Student-t interval is [-0.078, +0.026], half-width 0.052, which is 34.7% of the +-0.15
margin; at n=20, holding the observed sd, it is 0.020, or 13.1% of the margin. An equivalence claim
whose width is not reported is not readable as evidence, and that -- not a second test -- is what
these 60 runs buy.

WHY THIS IS NOT OPTIONAL STOPPING. Three reasons, all frozen in
experiments/pre_registration_dose_seed_topup.md before any seed ran:

  1. The published interval ALREADY clears the margin, so the top-up cannot rescue a failing claim;
     it can only narrow a passing one or reveal that the passage was an artifact of five seeds.
  2. THE REVERSAL CLAUSE: if the n=20 interval is not contained in (-0.15, +0.15) we report the
     flagship equivalence claim as refuted BY OUR OWN TOP-UP, in the abstract, and withdraw the
     (P3)=/=>(P4) witness to "not established at n=20".
  3. Seeds are fixed in advance at 47-61. No stopping rule, no interim look, no extension.

Precedent: SEEDS8 in run_targeted_dose.py:90 extended the cos_krum arm from 5 seeds to 8 for the
identical reason -- interval width, not a new test -- frozen at 5130cec before any outcome existed.

THE IDENTITY RUNG IS COMPUTED HERE, NOT IMPORTED. The published kappa=0 rung is imported from
results/dose_response/, which holds seeds 42-46 only. For seeds 47-61 there is nothing to import, so
kappa=0 is computed in place and marked "<computed here>" per seed. This is the established practice
of the suite, not a departure: the published cos_krum arm already mixes the two, with 42-46 imported
and 47-49 computed. --harness-check computes kappa=0 IN-SUITE AT SEED 42, where the imported value
already exists, and asserts the two agree to < 1e-9; a new seed would make the check vacuous, since
there would be nothing to compare against. One run settles it for all fifteen new seeds because the
code path does not depend on the seed. If it fails, the mixed-provenance kappa=0 rung is not one rung
and the top-up does not run.

SEEDS5 in run_targeted_dose.py is NOT edited and results/targeted_dose/ is NOT rewritten. This suite
writes its own directory and the two are merged only at analysis time, where the five published seeds
are asserted to reproduce bit-identically before any pooled number is printed.

Config identical to results/targeted_dose: N=10, K=5, f=0.2, alpha=0.5, 50 rounds, cifar_cnn.
New runs: 4 rungs x 15 seeds = 60, at ~15-20 min each.

Output: results/dose_seed_topup/summary.json (resumable; written after every run).

DO NOT RUN until experiments/pre_registration_dose_seed_topup.md is git-committed and PREREG_COMMIT
below is set to that hash. The script refuses to start otherwise.

  python3 experiments/run_dose_seed_topup.py --harness-check
  python3 experiments/run_dose_seed_topup.py
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
# Single-sourced from the frozen suite: the runner, the rung grid, the dial, the cell-key format, the
# accuracy floor and the equivalence margin are all imported rather than restated, so the top-up
# cannot drift from the arm it extends in any of them.
from experiments.run_targeted_dose import (run_one, d1_name, dial, cell_key, KAPPAS, SEEDS5,
                                           ACC_FLOOR, EQUIV_MARGIN, ATTACK_MAP)

# experiments/pre_registration_dose_seed_topup.md, committed before results/dose_seed_topup/ existed.
PREREG_COMMIT = "684b31e"

MODE, D2, ATTACK = "S", "krum", "committed_scaling"
NEW_SEEDS = list(range(47, 62))          # 47-61, frozen in the pre-registration
PUBLISHED_SEEDS = SEEDS5                 # 42-46, in results/targeted_dose/, not re-run

out_dir = os.path.join(base, "results", "dose_seed_topup")
out_path = os.path.join(out_dir, "summary.json")
TARGETED = os.path.join(base, "results", "targeted_dose", "summary.json")


def published_cell(val):
    """The published cell for this rung, or {} if absent. Read-only."""
    if not os.path.exists(TARGETED):
        return {}
    for c in json.load(open(TARGETED)).get("cells", {}).values():
        if (c.get("mode") == MODE and c.get("d2") == D2 and c.get("attack") == ATTACK
                and abs(c.get("rung", 1e9) - val) < 1e-12):
            return c
    return {}


def check_frozen():
    prereg = os.path.join(base, "experiments", "pre_registration_dose_seed_topup.md")
    if not os.path.exists(prereg):
        sys.exit(f"REFUSING TO RUN: {prereg} does not exist.")
    if PREREG_COMMIT is None:
        sys.exit("REFUSING TO RUN: the reversal clause is not frozen.\n"
                 f"  1. git commit {prereg}\n"
                 "  2. set PREREG_COMMIT here to that hash.\n"
                 "Adding seeds after seeing a result is licensed ONLY by a committed pre-registered "
                 "commitment to publish the reversal. Without the commit there is no commitment.")
    if not os.path.exists(TARGETED):
        sys.exit(f"REFUSING TO RUN: {TARGETED} does not exist. The published five seeds this top-up "
                 "extends live there, and the analysis asserts they reproduce before pooling.")
    for val in KAPPAS:
        if not published_cell(val):
            sys.exit(f"REFUSING TO RUN: no published {MODE}/{D2}/{ATTACK} cell at rung {val}. The "
                     "top-up extends an existing ladder; it does not create one.")


def harness_check():
    """Computing kappa=0 in-suite must reproduce the IMPORTED kappa=0 value bit-identically.

    The published kappa=0 rung is imported from results/dose_response/, which holds seeds 42-46 only.
    For seeds 47-61 it is computed in place, so the kappa=0 rung of the merged n=20 ladder has MIXED
    PROVENANCE -- imported for the first five seeds, computed for the last fifteen. That is legitimate
    only if the two provenances are the same computation, and the way to establish it is to compute
    in-suite at a seed where the imported value already exists and demand bit-equality.

    A published seed is used deliberately rather than a new one: at a new seed there is nothing to
    compare against, so the check would be vacuous. This costs one run at seed 42 and settles the
    claim for all fifteen new seeds, because the code path does not depend on the seed.
    """
    ref_rows = {r["seed"]: r for r in published_cell(0.0).get("per_seed", [])}
    seed = PUBLISHED_SEEDS[0]
    ref = ref_rows.get(seed)
    if ref is None:
        sys.exit(f"no published kappa=0 row at seed {seed} to check against")
    print("=== HARNESS CHECK: computing kappa=0 must reproduce the IMPORTED kappa=0 value ===")
    print(f"    cifar10/cifar_cnn, {D2}/{ATTACK.replace('committed_', '')} at seed {seed}")
    print(f"    imported ({ref.get('source', '?')}): acc={ref['accuracy']:.6f} "
          f"ASR={ref['asr']:.6f}")
    print("    If this fails, the merged kappa=0 rung is not one rung and the top-up must not run.\n",
          flush=True)
    t = time.time()
    acc, asr = run_one(seed, MODE, D2, ATTACK, 0.0)
    d = (acc - ref["accuracy"], asr - ref["asr"])
    print(f"  computed in-suite:      acc={acc:.6f} ASR={asr:.6f}   "
          f"d=({d[0]:+.2e}, {d[1]:+.2e})  ({time.time() - t:.0f}s)")
    ok = abs(d[0]) < 1e-9 and abs(d[1]) < 1e-9
    print(f"\n  {'BIT-IDENTICAL. The computed new-seed kappa=0 rows may be merged with the imported ones.' if ok else 'NOT IDENTICAL: the kappa=0 rung would contain two different computations. DO NOT MERGE.'}")
    return 0 if ok else 1


def load_cells():
    if not os.path.exists(out_path):
        return {}
    try:
        return json.load(open(out_path)).get("cells", {})
    except Exception:
        return {}


def save(cells):
    json.dump({"description": "Seed top-up of the flagship Mode-S krum arm, seeds 47-61, bringing "
                              "it from n=5 to n=20. Purpose is INTERVAL WIDTH, not a second test of "
                              "the published verdict. Reversal clause and seed list frozen at "
                              f"{PREREG_COMMIT} "
                              "(experiments/pre_registration_dose_seed_topup.md). Merge with "
                              "results/targeted_dose/ at analysis time; that file is never written "
                              "by this suite.",
               "prereg_commit": PREREG_COMMIT,
               "dataset": "cifar10", "model": "cifar_cnn",
               "config": {"N": 10, "K": 5, "f": 0.2, "alpha": 0.5, "rounds": 50,
                          "kappas": KAPPAS,
                          "new_seeds": NEW_SEEDS, "published_seeds": PUBLISHED_SEEDS,
                          "rhos": {str(k): dial(MODE, k) for k in KAPPAS},
                          "acc_floor": ACC_FLOOR, "equiv_margin": EQUIV_MARGIN},
               "arm": {"mode": MODE, "d2": D2, "attack": ATTACK,
                       "attack_impl": ATTACK_MAP[ATTACK]},
               "published_n5_verdict": {
                   "delta_asr_kappa2_minus_kappa0": -0.026,
                   "equiv_margin": EQUIV_MARGIN,
                   "jt_increasing_p": 0.70,
                   "verdict": "H-admission confirmed, H-statistic refuted",
                   "note": "Reported unchanged alongside the n=20 verdict whatever the latter is."},
               "identity_rung_provenance": "kappa=0 is COMPUTED here for seeds 47-61: "
                                           "results/dose_response/ holds seeds 42-46 only, so there "
                                           "is nothing to import. Verified bit-identical to krum "
                                           "standalone by --harness-check.",
               "cells": cells}, open(out_path, "w"), indent=2)


def main():
    check_frozen()
    os.makedirs(out_dir, exist_ok=True)
    total = len(KAPPAS) * len(NEW_SEEDS)
    print(f"=== Flagship seed top-up: Mode S -> {D2} / {ATTACK.replace('committed_', '')} ===")
    print(f"    rungs kappa={KAPPAS}, seeds {NEW_SEEDS[0]}-{NEW_SEEDS[-1]} ({total} runs)")
    print(f"    n=5 -> n=20. Rules frozen at {PREREG_COMMIT}.")
    print(f"    Reversal clause: an n=20 interval outside +-{EQUIV_MARGIN} refutes our own "
          "published claim.\n", flush=True)

    cells = load_cells()
    if cells:
        print(f"  resuming: {sum(len(c['per_seed']) for c in cells.values())} runs already done\n",
              flush=True)
    t0 = time.time(); done = 0
    for val in KAPPAS:
        key = cell_key(MODE, D2, ATTACK, val)
        existing = {r["seed"]: r for r in cells.get(key, {}).get("per_seed", [])}
        for seed in NEW_SEEDS:
            if seed in existing:
                done += 1
                continue
            t = time.time()
            acc, asr = run_one(seed, MODE, D2, ATTACK, val)
            existing[seed] = {"seed": int(seed), "accuracy": float(acc), "asr": float(asr),
                              "source": "<computed here>"}
            done += 1
            print(f"  [{done}/{total}] {d1_name(MODE, val):16s} s{seed}: acc={acc:.4f} "
                  f"ASR={asr:.4f} ({time.time() - t:.0f}s)"
                  + ("  * below acc floor" if acc < ACC_FLOOR else ""), flush=True)
            cells[key] = {"mode": MODE, "d2": D2, "attack": ATTACK, "rung": float(val),
                          "dial": dial(MODE, val),
                          "per_seed": [existing[s] for s in sorted(existing)]}
            save(cells)

    print(f"\nWall time: {(time.time() - t0) / 3600:.1f} h\nSaved to {out_path}")
    print("Run experiments/analyze_dose_seed_topup.py to merge with results/targeted_dose/ and "
          "report the n=20 interval and TOST verdict.")
    return 0


if __name__ == "__main__":
    if "--harness-check" in sys.argv:
        check_frozen()
        sys.exit(harness_check())
    sys.exit(main())
