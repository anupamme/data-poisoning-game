"""
Seed top-up of the SIGN-REVERSAL cell: n=5 -> n=20, both designs (Round 63).

WHY THIS EXISTS. The paper's centerpiece is one cell -- coord_median under committed_pixel on
CIFAR-10/cifar_cnn -- estimated two ways. The outcome-gated ladder reads dASR = -0.272; the Mode-S
intervention reads +0.098. Both 95% intervals exclude zero and they do not overlap, and that sign
reversal is what the abstract, section 1, section 5, the Conclusion and Figure 1(c) are built on.

BOTH LEGS ARE AT n=5. The n=20 top-up already in the paper (results/dose_seed_topup/) covers a
DIFFERENT cell: the krum/committed_scaling equivalence arm. The centerpiece has never been run beyond
five seeds. That is what these 45 runs fix, and nothing else: the purpose is INTERVAL WIDTH on the
separation between the two designs, not a second test of its direction.

At n=5 the paired half-widths are 0.1496 (confounded) and 0.0802 (controlled). Holding the observed
sds, n=20 gives 0.0564 and 0.0303 -- a factor of 2.65 narrower on each leg. No projected number goes
in the paper; the realized intervals are what is reported.

WHY THIS IS NOT OPTIONAL STOPPING. Frozen in experiments/pre_registration_reversal_seed_topup.md
before any seed ran:

  1. The published claim ALREADY holds at n=5 -- both intervals exclude zero and they do not overlap
     (-0.1225 < +0.0178, a gap of 0.140). So the top-up cannot rescue a failing claim. It can only
     narrow two intervals that already separate, or reveal that the separation was an artifact of five
     seeds. Only the second outcome is news, and it is news against us.
  2. THE DEMOTION CLAUSE: if at n=20 EITHER interval contains zero, OR the two intervals overlap, the
     paper reports the sign reversal as NOT ESTABLISHED AT n=20 and demotes it to a design
     DISAGREEMENT -- in the abstract, section 1, Figure 1(a) and 1(c), section 5 and the Conclusion.
     Naming that downside in advance is the only thing that licenses adding seeds.
  3. Seeds are fixed at 47-61. No stopping rule, no interim look, no extension. If interrupted, the
     analysis reports the n actually reached.

ENDPOINT RUNGS ONLY, kappa in {0, 2}, which is exactly what the frozen primary contrast reads
(LO, HI = "0.0", "2.0" in analyze_comparability.py). The consequence is declared rather than
discovered: the interior rungs 0.5 and 1.0 stay at n=5, so this cell's four-rung Jonckheere-Terpstra
statistics stay at n=5 and stay post hoc, and no four-rung display may print a mixed-n grid without a
per-rung n.

45 RUNS, NOT 60. At kappa=0 the transform returns the update list unwrapped, so dose_kappa0.0 and
doseS_kappa0.0 are the same computation -- and this is already true BIT-FOR-BIT in the published
artifacts: for all five seeds 42-46, results/dose_response/ and results/dose_replication/ carry
identical float reprs at kappa=0 in both ASR and clean accuracy. So the identity rung is computed once
per new seed and shared by both legs: 15 seeds x 3 runs = 45.

--harness-check establishes that rather than inheriting it, in three runs at seed 42 where published
values already exist (a new seed would make the check vacuous, since there would be nothing to compare
against):

  1. confounded kappa=0 in-suite == published dose_response kappa=0
  2. controlled kappa=0 in-suite == published dose_replication kappa=0
  3. confounded kappa=2 in-suite == published dose_response kappa=2

(1) and (2) together are what licenses sharing: each provenance of the shared rung is tied to its own
published value, so all four quantities are equal by transitivity. (3) proves the harness is the same
loop the published cells were run with, not merely the same at the identity. If any of the three
fails, the shared kappa=0 rung would not be one rung and the top-up does not run.

results/dose_response/ and results/dose_replication/ are NOT written, and no existing runner is
edited. This suite writes its own directory; the merge happens at analysis time, where the five
published seeds are asserted to reproduce before any pooled number is printed.

Config is inherited by import from run_comparability_cells, whose run_one() already implements both
designs with the two-token contrast (d1 and adv_mask) and whose own --harness-check already reproduces
this exact cell's published kappa=2 value. Reimplementing the loop here would be a third copy.

DO NOT RUN until experiments/pre_registration_reversal_seed_topup.md is git-committed and
PREREG_COMMIT below is set to that hash. The suite refuses to start otherwise, and also refuses if the
pre-registration has uncommitted edits, because `git log -1` cannot see those.

Output: results/reversal_seed_topup/summary.json (resumable; written after every run).

  python3 experiments/run_reversal_seed_topup.py --harness-check
  python3 experiments/run_reversal_seed_topup.py
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Single-sourced from the frozen comparability suite: the FL loop, the two-token design contrast, the
# cell-key format, the rung grid and the published seed list are all imported rather than restated, so
# the top-up cannot drift from the cell it extends in any of them.
from experiments.run_comparability_cells import (run_one, cell_key,      # noqa: E402
                                                 KAPPAS, SEEDS,
                                                 FL_CONFIG, ADV_FRACTION)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "results", "reversal_seed_topup")
PREREG = "experiments/pre_registration_reversal_seed_topup.md"

# The commit that froze PREREG, made before results/reversal_seed_topup/ existed. `git log` can
# demonstrate that ordering, which is what makes this top-up prospective (non-negotiable 7).
#
# Bumped fb94d3a -> 115b804 for Amendment 1, which corrects a false factual claim in the Scope
# section ("Table 1's rows are n = 5 throughout"; they are 5, 5, 8, 5, 3, 5) and widens the reporting
# obligation to Table 1 and Figure 1(c). No number, rule, seed list, interval definition or demotion
# criterion moved, and no results directory existed at either hash. The amendment was appended, not
# edited in place, so fb94d3a is still the hash that froze everything this runner reads.
#
# Bumped 115b804 -> 1efe5ec for Amendment 2, which fixes the reporting split site by site because the
# table-level split was too coarse to bind one outcome-sensitive class: paper/main.tex:437 and :1693
# evaluate a frozen rule (confirm if Delta > +0.178, refute if Delta < +0.150) against +0.098, so if
# the n = 20 controlled mean lands above +0.178, deciding then whether to update them would be
# deciding on the outcome. Again append-only (68 insertions, 0 deletions), and again no number, rule,
# seed list, interval definition or demotion criterion moved: the amendment adds classes 10-12, which
# constrain how the result is REPORTED, not what is run. fb94d3a remains the hash that froze
# everything this runner reads; both bumps exist so the guard checks the document the paper will cite.
#
# Bumped 1efe5ec -> 89beed0 for Amendment 3, appended after this suite had already started (with zero
# of the 45 runs scored). It corrects one site Amendment 2 classified by its digits rather than by its
# quantity -- main.tex:1655's 0.098 is a Mode A rung ASR level, not a paired difference -- and it moves
# that site OUT of scope. Again append-only, again no number, rule, seed list, interval definition or
# demotion criterion moved. If this suite is interrupted and resumed, the guard checks 89beed0, which
# is the point: a resume must read the same document the paper will cite, not the one that was current
# when the first run started.
PREREG_COMMIT = "89beed0"

DATASET, MODEL, D2, ATTACK = "cifar10", "cifar_cnn", "coord_median", "committed_pixel"
LO, HI = KAPPAS[0], KAPPAS[-1]           # 0.0 and 2.0: the frozen primary contrast, endpoints only
NEW_SEEDS = list(range(47, 62))          # 47-61, frozen in the pre-registration
PUBLISHED_SEEDS = SEEDS                  # 42-46, in the two frozen directories, not re-run

# The two frozen directories this top-up merges with at analysis time. Read-only here.
SRC = {"confounded": os.path.join(BASE, "results", "dose_response", "summary.json"),
       "controlled": os.path.join(BASE, "results", "dose_replication", "summary.json")}

# Inherited verbatim from pre_registration_comparability.md:75 and :149 -- floor on a rung's MEAN, not
# per seed, and "seeds below it are flagged, never excluded". analyze_comparability.py reads no
# accuracy at all, so the flagging is this runner's job and is written into the artifact.
ACC_FLOOR = 0.35

# The published n=5 verdict, recomputed from the two frozen artifacts rather than transcribed from the
# paper, and reported alongside the n=20 verdict whatever the latter turns out to be.
PUBLISHED_N5 = {
    "confounded": {"mean": -0.272067, "sd": 0.120508, "ci95": [-0.421673, -0.122461], "n": 5},
    "controlled": {"mean": +0.098067, "sd": 0.064640, "ci95": [+0.017819, +0.178314], "n": 5},
    "verdict": "Sign reversal: both intervals exclude zero and they do not overlap (gap 0.140).",
    "note": "Reported unchanged alongside the n=20 verdict. If the two n disagree, both appear and "
            "the disagreement is the result.",
}


def published_rows(family, kappa):
    """{seed: row} for the published rung, or {} if absent. Read-only, never written."""
    path = SRC[family]
    if not os.path.exists(path):
        return {}
    cell = json.load(open(path)).get("cells", {}).get(cell_key(family, D2, ATTACK, kappa))
    return {int(r["seed"]): r for r in cell["per_seed"]} if cell else {}


def check_frozen():
    """Refuse to run unless the pre-registration is committed at the recorded hash AND clean.

    The hash check alone is necessary and not sufficient: `git log -1` reports the last commit that
    touched the file, which is unchanged by uncommitted edits to it. That defect let a Round-57
    amendment pass its own gate (run_comparability_cells.py:184-195), so the working-tree check is part
    of the guard here from the start, not an addition after the fact.
    """
    if not os.path.exists(os.path.join(BASE, PREREG)):
        print(f"REFUSING TO RUN: {PREREG} does not exist.")
        return False
    if PREREG_COMMIT in (None, "PENDING"):
        print("REFUSING TO RUN: the demotion clause is not frozen.\n"
              f"  1. git add {PREREG} && git commit\n"
              "  2. set PREREG_COMMIT here to that hash\n"
              "  3. rerun.\n"
              "Adding seeds after seeing a result is licensed ONLY by a committed pre-registered "
              "commitment to publish the demotion. Without the commit there is no commitment.")
        return False
    out = subprocess.run(["git", "log", "-1", "--format=%h", "--", PREREG],
                         cwd=BASE, capture_output=True, text=True, timeout=20)
    actual = out.stdout.strip()
    if not actual or not actual.startswith(PREREG_COMMIT[:7]):
        print(f"REFUSING TO RUN: {PREREG} last touched at {actual or 'UNTRACKED'}, "
              f"but PREREG_COMMIT is {PREREG_COMMIT}.")
        return False
    dirty = subprocess.run(["git", "status", "--porcelain", "--", PREREG],
                           cwd=BASE, capture_output=True, text=True, timeout=20)
    if dirty.stdout.strip():
        print(f"REFUSING TO RUN: {PREREG} has uncommitted changes "
              f"({dirty.stdout.strip().split()[0]}), so it is not frozen at {actual} whatever "
              "`git log` says. Commit it and set PREREG_COMMIT to the new hash.")
        return False
    for family, path in SRC.items():
        if not os.path.exists(path):
            print(f"REFUSING TO RUN: {os.path.relpath(path, BASE)} does not exist. The five "
                  f"published {family} seeds this top-up extends live there, and the analysis asserts "
                  "they reproduce before pooling.")
            return False
        for kappa in (LO, HI):
            if not published_rows(family, kappa):
                print(f"REFUSING TO RUN: no published {family} {D2}/{ATTACK} rung at kappa={kappa}. "
                      "The top-up extends an existing pair of ladders; it does not create them.")
                return False
    print(f"[OK] {PREREG} frozen at {actual}, working tree clean")
    return True


def _one_check(n, label, family, kappa):
    """Recompute one published row in-suite and demand bit-equality to 1e-9."""
    rows = published_rows(family, kappa)
    seed = PUBLISHED_SEEDS[0]
    ref = rows.get(seed)
    if ref is None:
        print(f"  [{n}/3] {label}: SKIPPED, seed {seed} not published")
        return None
    print(f"  [{n}/3] {label}: published acc={ref['accuracy']!r} ASR={ref['asr']!r}", flush=True)
    t0 = time.time()
    acc, asr = run_one(seed, family, D2, ATTACK, kappa, DATASET, MODEL)
    dacc, dasr = acc - ref["accuracy"], asr - ref["asr"]
    ok = abs(dacc) < 1e-9 and abs(dasr) < 1e-9
    print(f"        recomputed acc={acc!r} ASR={asr!r}")
    print(f"        |d| = ({abs(dacc):.3e}, {abs(dasr):.3e})  "
          f"{'OK' if ok else '** MISMATCH'}  ({time.time() - t0:.0f}s)\n", flush=True)
    return ok


def harness_check():
    """Three runs at seed 42. Establishes the shared identity rung instead of assuming it."""
    print(f"=== HARNESS CHECK: {D2}/{ATTACK.replace('committed_', '')} on {DATASET}, "
          f"seed {PUBLISHED_SEEDS[0]} ===")
    print("    (1) and (2) tie each provenance of the SHARED kappa=0 rung to its own published value,")
    print("    so all four quantities are equal by transitivity and the rung may be computed once.")
    print("    (3) proves this is the same loop the published cells were run with, not merely the")
    print("    same at the identity. Any failure means the shared rung is not one rung.\n", flush=True)

    a = _one_check(1, f"confounded kappa={LO} vs results/dose_response/", "confounded", LO)
    b = _one_check(2, f"controlled kappa={LO} vs results/dose_replication/", "controlled", LO)
    c = _one_check(3, f"confounded kappa={HI} vs results/dose_response/", "confounded", HI)

    results = [x for x in (a, b, c) if x is not None]
    if not results:
        print("  ALL THREE SKIPPED: nothing was verified. Treat as a failure, not a pass.")
        return False
    if not all(results):
        print("  ** REFUSING TO CONTINUE. A mismatch here means the merged kappa=0 rung would contain")
        print("     two different computations, or that this harness is not the published one.")
        return False
    if a is None or b is None:
        print("  ** REFUSING TO CONTINUE: the shared-rung check itself did not run, so sharing")
        print("     kappa=0 between the two legs is unverified. Run 60 runs, not 45, or fix the skip.")
        return False
    print(f"  OK, all {len(results)} checks bit-identical to 1e-9. kappa=0 may be shared "
          "between the two legs.\n")
    return True


def load_cells():
    p = os.path.join(OUT, "summary.json")
    return json.load(open(p)).get("cells", {}) if os.path.exists(p) else {}


def save(cells):
    os.makedirs(OUT, exist_ok=True)
    json.dump({
        "description": "Seed top-up of the sign-reversal cell (coord_median / committed_pixel on "
                       "CIFAR-10/cifar_cnn), seeds 47-61, bringing BOTH designs from n=5 to n=20 at "
                       "the endpoint rungs kappa in {0, 2}. Purpose is INTERVAL WIDTH on the "
                       "separation between the two designs, not a second test of its direction. "
                       "Demotion clause and seed list frozen at "
                       f"{PREREG_COMMIT} ({PREREG}). Merge with results/dose_response/ and "
                       "results/dose_replication/ at analysis time; neither is ever written here.",
        "prereg": PREREG, "prereg_commit": PREREG_COMMIT,
        "dataset": DATASET, "model": MODEL,
        "config": {"N": FL_CONFIG.num_clients, "K": FL_CONFIG.clients_per_round,
                   "f": ADV_FRACTION, "alpha": 0.5, "rounds": FL_CONFIG.num_rounds,
                   "kappas_run": [LO, HI], "kappas_frozen_grid": KAPPAS,
                   "new_seeds": NEW_SEEDS, "published_seeds": PUBLISHED_SEEDS,
                   "acc_floor": ACC_FLOOR},
        "arm": {"d2": D2, "attack": ATTACK, "families": ["confounded", "controlled"]},
        "endpoint_only": "kappa=0.5 and kappa=1.0 are NOT run for the new seeds. This cell's "
                         "four-rung trend statistics therefore stay at n=5 and stay post hoc, and no "
                         "four-rung display may print this ladder without a per-rung n.",
        "published_n5_verdict": PUBLISHED_N5,
        "identity_rung_provenance": "kappa=0 is COMPUTED ONCE per new seed and shared by both legs. "
                                    "dose_kappa0.0 and doseS_kappa0.0 are the same computation -- "
                                    "bit-identical in ASR and accuracy across all five published "
                                    "seeds -- and --harness-check re-establishes that at seed 42 "
                                    "against BOTH published directories before any new run. The "
                                    "controlled copy of each shared row is labelled in `source`.",
        "cells": cells}, open(os.path.join(OUT, "summary.json"), "w"), indent=2)


def _record(cells, family, kappa, seed, acc, asr, source):
    key = cell_key(family, D2, ATTACK, kappa)
    cell = cells.setdefault(key, {"d2": D2, "attack": ATTACK, "kappa": kappa, "family": family,
                                 "dataset": DATASET, "model": MODEL, "per_seed": []})
    row = {"seed": int(seed), "accuracy": float(acc), "asr": float(asr), "source": source,
           "below_acc_floor": bool(acc < ACC_FLOOR)}
    # Replace, never append blind. The shared kappa=0 rung is written to two keys from one run, so a
    # crash between the two writes leaves the confounded row present and the controlled one absent;
    # the resume then recomputes and would append a duplicate seed to a rung that already had it,
    # silently inflating n on one leg only.
    cell["per_seed"] = ([r for r in cell["per_seed"] if int(r["seed"]) != int(seed)] + [row])
    cell["per_seed"].sort(key=lambda r: r["seed"])
    cell["mean_asr"] = float(np.mean([r["asr"] for r in cell["per_seed"]]))
    cell["mean_accuracy"] = float(np.mean([r["accuracy"] for r in cell["per_seed"]]))
    cell["n_below_acc_floor"] = sum(1 for r in cell["per_seed"] if r["below_acc_floor"])
    return key


def _has(cells, family, kappa, seed):
    cell = cells.get(cell_key(family, D2, ATTACK, kappa), {})
    return any(int(r["seed"]) == seed for r in cell.get("per_seed", []))


def main():
    if not check_frozen():
        return 1
    if "--harness-check" in sys.argv:
        return 0 if harness_check() else 1

    cells = load_cells()
    print(f"=== Sign-reversal seed top-up: {D2} / {ATTACK.replace('committed_', '')} on {DATASET} ===")
    print(f"    endpoint rungs kappa in {{{LO}, {HI}}}, seeds {NEW_SEEDS[0]}-{NEW_SEEDS[-1]}, "
          "both designs")
    print(f"    n=5 -> n=20 on BOTH legs. Rules frozen at {PREREG_COMMIT}.")
    print(f"    kappa={LO} is computed ONCE per seed and shared, so {len(NEW_SEEDS)} x 3 = "
          f"{len(NEW_SEEDS) * 3} runs, not {len(NEW_SEEDS) * 4}.")
    print("    Demotion clause: an n=20 interval containing zero, or two overlapping intervals, "
          "demotes\n    the sign reversal to a design disagreement in the abstract and in Figure 1.\n",
          flush=True)

    # Per seed: the shared identity rung first, then the two kappa=2 legs. Seed-major so that an
    # interruption leaves COMPLETE seeds on both legs -- equal n, which is the only shape the paired
    # contrast can read -- rather than a long confounded leg with nothing to contrast it against.
    # run_one() re-seeds torch and numpy from `seed` on entry, so no run depends on what ran before it.
    todo = [(seed, family, kappa)
            for seed in NEW_SEEDS
            for (family, kappa) in (("confounded", LO), ("confounded", HI), ("controlled", HI))]
    done = sum(len(c.get("per_seed", [])) for c in cells.values())
    print(f"  resuming: {done} rows already recorded ({len(todo)} runs planned, "
          "shared rows counted once)\n", flush=True)

    t0 = time.time()
    for i, (seed, family, kappa) in enumerate(todo, 1):
        shared = (kappa == LO)
        if _has(cells, family, kappa, seed) and (not shared or _has(cells, "controlled", LO, seed)):
            continue
        t = time.time()
        acc, asr = run_one(seed, family, D2, ATTACK, kappa, DATASET, MODEL)
        _record(cells, family, kappa, seed, acc, asr, "<computed here>")
        if shared:
            # The same computation, recorded under the controlled key so the analyzer needs no special
            # case. Labelled, not silently duplicated: `source` says which leg actually ran it.
            _record(cells, "controlled", kappa, seed, acc, asr,
                    "<shared identity rung: computed once as confounded kappa=0; "
                    "dose_kappa0.0 == doseS_kappa0.0, verified by --harness-check>")
        save(cells)
        print(f"  [{i}/{len(todo)}] s{seed} {family:11s} k={kappa:<4} acc={acc:.4f} ASR={asr:.4f}"
              + ("  * below acc floor" if acc < ACC_FLOOR else "")
              + ("  (shared with controlled)" if shared else "")
              + f" ({time.time() - t:.0f}s)", flush=True)

    print(f"\nWall time: {(time.time() - t0) / 3600:.1f} h")
    print(f"Saved to {os.path.join(OUT, 'summary.json')}")
    print("Count `per_seed` entries in the artifact to judge progress; the [i/N] index above counts "
          "resumed-and-skipped runs and can go backwards across restarts.")
    print("Run experiments/analyze_comparability.py to merge and report both n=20 intervals.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
