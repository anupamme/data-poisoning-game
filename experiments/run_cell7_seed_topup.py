"""
Seed top-up of the CIFAR-100 REPLICATION of the sign reversal: n=5 -> n=20, both designs (Round 71).

WHY THIS EXISTS. Cell 7 of the comparability suite is the sign-reversal cell moved to a third dataset:
coord_median under committed_pixel on CIFAR-100/cifar_cnn. The outcome-gated ladder reads dASR =
-0.213 [-0.284, -0.141]; the Mode-S intervention reads +0.071 [+0.031, +0.110]. Both 95% intervals
exclude zero and they do not overlap, and that is what the paper means when it says the reversal
REPLICATES -- a claim that appears in the abstract, in section 1's table, in section 6, in the
seven-cell table and in the Conclusion.

BOTH LEGS ARE AT n=5. Round 63's top-up (results/reversal_seed_topup/, frozen at 89beed0) took the
CIFAR-10 cell to n=20 on both legs and explicitly declined to extend this one; its Caveat 3 reads "The
reversal's replication on CIFAR-100 stays at n = 5 and is not extended here." That is what these 45
runs fix, and nothing else: the purpose is INTERVAL WIDTH on the separation between the two designs,
not a second test of its direction.

At n=5 the paired half-widths are 0.0715 (confounded) and 0.0394 (controlled). Holding the observed
sds, n=20 gives 0.0270 and 0.0149 -- a factor of 2.65 narrower on each leg. No projected number goes
in the paper; the realized intervals are what is reported.

WHY THIS IS NOT OPTIONAL STOPPING. Frozen in experiments/pre_registration_cell7_seed_topup.md before
any seed ran:

  1. The published claim ALREADY holds at n=5 -- both intervals exclude zero and they do not overlap
     (-0.1414 < +0.0312, a gap of 0.1726). So the top-up cannot rescue a failing claim. It can only
     narrow two intervals that already separate, or reveal that the separation was an artifact of five
     seeds. Only the second outcome is news, and it is news against us.
  2. THE DEMOTION CLAUSE: if at n=20 EITHER interval contains zero, OR the two intervals overlap, the
     paper reports the CIFAR-100 replication as NOT ESTABLISHED AT n=20 and demotes it to a design
     DISAGREEMENT -- in the abstract, section 1's table, Figure 1's forest panel, section 6, the
     seven-cell table and the Conclusion. Naming that downside in advance is the only thing that
     licenses adding seeds.
  3. Seeds are fixed at 47-61. No stopping rule, no interim look, no extension. If interrupted, the
     analysis reports the n actually reached.

ENDPOINT RUNGS ONLY, kappa in {0, 2}, which is exactly what the frozen primary contrast reads
(LO, HI = "0.0", "2.0" in analyze_comparability.py). The consequence is declared rather than
discovered: the interior rungs 0.5 and 1.0 stay at n=5, so this cell's four-rung trend statistics stay
at n=5 and stay post hoc (main.tex:1784's JT p=0.079 and permutation p=0.084 do NOT move while the
endpoint interval in the same sentence does), and no four-rung display may print a mixed-n grid
without a per-rung n.

45 RUNS, NOT 60. At kappa=0 the transform returns the update list unwrapped, so dose_kappa0.0 and
doseS_kappa0.0 are the same computation -- and this is already true BIT-FOR-BIT in the published
artifact: for all five seeds 42-46, the confounded and controlled kappa=0 rows in
results/comparability_cells/summary.json carry identical float reprs in both ASR and clean accuracy.
So the identity rung is computed once per new seed and shared by both legs: 15 seeds x 3 runs = 45.

--harness-check establishes that rather than inheriting it, in three runs at seed 42 where published
values already exist (a new seed would make the check vacuous, since there would be nothing to compare
against):

  1. confounded kappa=0 in-suite == published confounded kappa=0
  2. controlled kappa=0 in-suite == published controlled kappa=0
  3. confounded kappa=2 in-suite == published confounded kappa=2

(1) and (2) together are what licenses sharing: each provenance of the shared rung is tied to its own
published value, so all four quantities are equal by transitivity. (3) proves the harness is the same
loop the published cell was run with, not merely the same at the identity. If any of the three fails,
the shared kappa=0 rung would not be one rung and the top-up does not run.

THE VERDICT IS WRITTEN INTO THE ARTIFACT, NOT ONLY PRINTED. Round 69 found that an existing
--harness-check returns a verdict dict its caller discards, leaving a paper sentence witnessed only by
run stdout that costs hours to reproduce. Here the three comparisons, their deltas and their verdict
are persisted to results/cell7_seed_topup/summary.json, and main() REFUSES to share the identity rung
unless that record exists and passed. So the 45-run shape is earned by a recorded check rather than
asserted by a comment.

results/comparability_cells/ is NOT written, and no existing runner is edited. This suite writes its
own directory; the merge happens at analysis time, where the five published seeds are asserted to
reproduce before any pooled number is printed.

Config is inherited by import from run_comparability_cells, whose run_one() already implements both
designs with the two-token contrast (d1 and adv_mask) and whose cell_key() already carries this cell's
|cifar100 suffix. Reimplementing the loop here would be a third copy.

DO NOT RUN until experiments/pre_registration_cell7_seed_topup.md is git-committed and PREREG_COMMIT
below is set to that hash. The suite refuses to start otherwise, and also refuses if the
pre-registration has uncommitted edits, because `git log -1` cannot see those.

Output: results/cell7_seed_topup/summary.json (resumable; written after every run).

  python3 experiments/run_cell7_seed_topup.py --harness-check
  python3 experiments/run_cell7_seed_topup.py
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Single-sourced from the frozen comparability suite: the FL loop, the two-token design contrast, the
# cell-key format (including this cell's |cifar100 suffix), the rung grid and the published seed list
# are all imported rather than restated, so the top-up cannot drift from the cell it extends.
from experiments.run_comparability_cells import (run_one, cell_key,      # noqa: E402
                                                 KAPPAS, SEEDS,
                                                 FL_CONFIG, ADV_FRACTION)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "results", "cell7_seed_topup")
PREREG = "experiments/pre_registration_cell7_seed_topup.md"

# The commit that froze PREREG, made before results/cell7_seed_topup/ existed. `git log` can
# demonstrate that ordering, which is what makes this top-up prospective (non-negotiable 7).
PREREG_COMMIT = "PENDING"

DATASET, MODEL, D2, ATTACK = "cifar100", "cifar_cnn", "coord_median", "committed_pixel"
SUFFIX = "|cifar100"                     # cell 7's key suffix, per run_comparability_cells.CELLS
LO, HI = KAPPAS[0], KAPPAS[-1]           # 0.0 and 2.0: the frozen primary contrast, endpoints only
NEW_SEEDS = list(range(47, 62))          # 47-61, frozen in the pre-registration
PUBLISHED_SEEDS = SEEDS                  # 42-46, in the frozen directory, not re-run

# The ONE frozen directory this top-up merges with at analysis time -- unlike the CIFAR-10 cell, whose
# two legs live in two directories, both of cell 7's legs live here under the |cifar100 suffix.
# Read-only in this file.
SRC = os.path.join(BASE, "results", "comparability_cells", "summary.json")

# Inherited verbatim from pre_registration_comparability.md:75 and :149 -- floor on a rung's MEAN, not
# per seed, and "seeds below it are flagged, never excluded". analyze_comparability.py reads no
# accuracy at all, so the flagging is this runner's job and is written into the artifact.
#
# The margin here is thin and that is recorded rather than discovered: the published rung means are
# 0.3931, 0.3944 and 0.3947 against a floor of 0.35, so a flag is a live possibility on CIFAR-100 in a
# way it is not on CIFAR-10. A flag is not an exclusion.
ACC_FLOOR = 0.35

# The published n=5 verdict, recomputed from the frozen artifact rather than transcribed from the
# paper, and reported alongside the n=20 verdict whatever the latter turns out to be.
PUBLISHED_N5 = {
    "confounded": {"mean": -0.212929, "sd": 0.057616, "ci95": [-0.284469, -0.141390], "n": 5},
    "controlled": {"mean": +0.070586, "sd": 0.031741, "ci95": [+0.031174, +0.109997], "n": 5},
    "verdict": "Sign reversal replicates: both intervals exclude zero and they do not overlap "
               "(gap 0.1726).",
    "note": "Reported unchanged alongside the n=20 verdict. If the two n disagree, both appear and "
            "the disagreement is the result.",
}

# Amendment 4's two-sided admissibility gate, evaluated ONCE on the published identity rung and passed.
# Recorded here as history, NOT re-applied at n=20: re-applying it would create a licence to discard
# the run on its own outcome (pre_registration_cell7_seed_topup.md, Decision rules).
ADMISSIBILITY = {"gate": "identity-rung mean acc >= 0.35 and mean ASR in [0.15, 0.85] "
                         "(pre_registration_comparability.md Amendment 4)",
                 "identity_rung_mean_asr": 0.7034, "identity_rung_mean_acc": 0.3931,
                 "passed": True,
                 "reapplied_at_n20": False,
                 "why_not": "The gate decided whether an unrun third dataset had headroom. The cell "
                            "is published at n=5; there is nothing left to admit."}


def published_rows(family, kappa):
    """{seed: row} for the published rung, or {} if absent. Read-only, never written."""
    if not os.path.exists(SRC):
        return {}
    cell = json.load(open(SRC)).get("cells", {}).get(cell_key(family, D2, ATTACK, kappa, SUFFIX))
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
    if not os.path.exists(SRC):
        print(f"REFUSING TO RUN: {os.path.relpath(SRC, BASE)} does not exist. The five published "
              "seeds this top-up extends live there, and the analysis asserts they reproduce before "
              "pooling.")
        return False
    for family in ("confounded", "controlled"):
        for kappa in (LO, HI):
            if not published_rows(family, kappa):
                print(f"REFUSING TO RUN: no published {family} {D2}/{ATTACK}{SUFFIX} rung at "
                      f"kappa={kappa}. The top-up extends an existing pair of ladders; it does not "
                      "create them.")
                return False
    print(f"[OK] {PREREG} frozen at {actual}, working tree clean")
    return True


def _one_check(n, label, family, kappa):
    """Recompute one published row in-suite and demand bit-equality to 1e-9.

    Returns a record rather than a bare bool so the verdict can be written into the artifact instead
    of living only in this run's stdout.
    """
    rows = published_rows(family, kappa)
    seed = PUBLISHED_SEEDS[0]
    ref = rows.get(seed)
    rec = {"n": n, "label": label, "family": family, "kappa": kappa, "seed": seed}
    if ref is None:
        print(f"  [{n}/3] {label}: SKIPPED, seed {seed} not published")
        return dict(rec, status="SKIPPED", ok=None)
    print(f"  [{n}/3] {label}: published acc={ref['accuracy']!r} ASR={ref['asr']!r}", flush=True)
    t0 = time.time()
    acc, asr = run_one(seed, family, D2, ATTACK, kappa, DATASET, MODEL)
    dacc, dasr = acc - ref["accuracy"], asr - ref["asr"]
    ok = abs(dacc) < 1e-9 and abs(dasr) < 1e-9
    print(f"        recomputed acc={acc!r} ASR={asr!r}")
    print(f"        |d| = ({abs(dacc):.3e}, {abs(dasr):.3e})  "
          f"{'OK' if ok else '** MISMATCH'}  ({time.time() - t0:.0f}s)\n", flush=True)
    return dict(rec, status="OK" if ok else "MISMATCH", ok=bool(ok),
                published={"accuracy": ref["accuracy"], "asr": ref["asr"]},
                recomputed={"accuracy": float(acc), "asr": float(asr)},
                abs_delta={"accuracy": abs(float(dacc)), "asr": abs(float(dasr))},
                tolerance=1e-9, seconds=round(time.time() - t0, 1))


def harness_check():
    """Three runs at seed 42. Establishes the shared identity rung instead of assuming it.

    Writes its verdict into the artifact so the claim has a witness other than this run's stdout, and
    so main() can refuse to share the identity rung without a recorded pass.
    """
    print(f"=== HARNESS CHECK: {D2}/{ATTACK.replace('committed_', '')} on {DATASET}, "
          f"seed {PUBLISHED_SEEDS[0]} ===")
    print("    (1) and (2) tie each provenance of the SHARED kappa=0 rung to its own published value,")
    print("    so all four quantities are equal by transitivity and the rung may be computed once.")
    print("    (3) proves this is the same loop the published cell was run with, not merely the")
    print("    same at the identity. Any failure means the shared rung is not one rung.\n", flush=True)

    checks = [_one_check(1, f"confounded kappa={LO} vs results/comparability_cells/", "confounded", LO),
              _one_check(2, f"controlled kappa={LO} vs results/comparability_cells/", "controlled", LO),
              _one_check(3, f"confounded kappa={HI} vs results/comparability_cells/", "confounded", HI)]
    a, b, c = (x["ok"] for x in checks)
    results = [x for x in (a, b, c) if x is not None]

    if not results:
        verdict, why = False, ("ALL THREE SKIPPED: nothing was verified. Treated as a failure, not a "
                               "pass.")
    elif not all(results):
        verdict, why = False, ("A mismatch means the merged kappa=0 rung would contain two different "
                               "computations, or that this harness is not the published one.")
    elif a is None or b is None:
        verdict, why = False, ("The shared-rung check itself did not run, so sharing kappa=0 between "
                               "the two legs is unverified. Run 60 runs, not 45, or fix the skip.")
    else:
        verdict, why = True, (f"All {len(results)} checks bit-identical to 1e-9. kappa=0 may be "
                              "shared between the two legs, so 45 runs and not 60.")

    # Persisted BEFORE returning, and on failure as well as success: a check that ran and failed is
    # itself evidence, and discarding it would leave the next run unable to tell "failed" from
    # "never run".
    cells = load_cells()
    save(cells, harness={"passed": bool(verdict), "why": why, "checks": checks,
                         "shared_identity_rung_licensed": bool(verdict),
                         "runs_implied": 45 if verdict else 60})
    print(("  OK, " if verdict else "  ** REFUSING TO CONTINUE. ") + why)
    print(f"  Verdict written to {os.path.relpath(os.path.join(OUT, 'summary.json'), BASE)}\n")
    return verdict


def load_summary():
    p = os.path.join(OUT, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def load_cells():
    return load_summary().get("cells", {})


def save(cells, harness=None):
    """Write the artifact. `harness` is preserved across calls once recorded."""
    os.makedirs(OUT, exist_ok=True)
    if harness is None:
        harness = load_summary().get("harness_check")
    json.dump({
        "description": "Seed top-up of the CIFAR-100 replication of the sign reversal (coord_median / "
                       "committed_pixel on CIFAR-100/cifar_cnn, cell 7 of the comparability suite), "
                       "seeds 47-61, bringing BOTH designs from n=5 to n=20 at the endpoint rungs "
                       "kappa in {0, 2}. Purpose is INTERVAL WIDTH on the separation between the two "
                       "designs, not a second test of its direction. Demotion clause and seed list "
                       f"frozen at {PREREG_COMMIT} ({PREREG}). Merge with "
                       "results/comparability_cells/ at analysis time; it is never written here.",
        "prereg": PREREG, "prereg_commit": PREREG_COMMIT,
        "dataset": DATASET, "model": MODEL,
        "config": {"N": FL_CONFIG.num_clients, "K": FL_CONFIG.clients_per_round,
                   "f": ADV_FRACTION, "alpha": 0.5, "rounds": FL_CONFIG.num_rounds,
                   "kappas_run": [LO, HI], "kappas_frozen_grid": KAPPAS,
                   "new_seeds": NEW_SEEDS, "published_seeds": PUBLISHED_SEEDS,
                   "acc_floor": ACC_FLOOR},
        "arm": {"d2": D2, "attack": ATTACK, "suffix": SUFFIX,
                "families": ["confounded", "controlled"]},
        "endpoint_only": "kappa=0.5 and kappa=1.0 are NOT run for the new seeds. This cell's "
                         "four-rung trend statistics therefore stay at n=5 and stay post hoc -- "
                         "main.tex:1784's JT p=0.079 and permutation p=0.084 do not move while the "
                         "endpoint interval in the same sentence does -- and no four-rung display "
                         "may print this ladder without a per-rung n.",
        "published_n5_verdict": PUBLISHED_N5,
        "admissibility_gate": ADMISSIBILITY,
        "harness_check": harness,
        "identity_rung_provenance": "kappa=0 is COMPUTED ONCE per new seed and shared by both legs. "
                                    "dose_kappa0.0 and doseS_kappa0.0 are the same computation -- "
                                    "bit-identical in ASR and accuracy across all five published "
                                    "seeds -- and --harness-check re-establishes that at seed 42 "
                                    "against BOTH published legs before any new run. The controlled "
                                    "copy of each shared row is labelled in `source`.",
        "cells": cells}, open(os.path.join(OUT, "summary.json"), "w"), indent=2)


def _record(cells, family, kappa, seed, acc, asr, source):
    key = cell_key(family, D2, ATTACK, kappa, SUFFIX)
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
    cell = cells.get(cell_key(family, D2, ATTACK, kappa, SUFFIX), {})
    return any(int(r["seed"]) == seed for r in cell.get("per_seed", []))


def main():
    if not check_frozen():
        return 1
    if "--harness-check" in sys.argv:
        return 0 if harness_check() else 1

    # The 45-run shape is earned by a RECORDED check, not by a comment. Without it there is no
    # evidence that the two legs' kappa=0 rungs are one computation, and sharing would silently
    # fabricate 15 controlled rows.
    harness = load_summary().get("harness_check")
    if not (harness or {}).get("passed"):
        state = "never run" if harness is None else "recorded as FAILED"
        print(f"REFUSING TO RUN: the harness check is {state}, so sharing the kappa=0 rung between "
              "the two legs is unverified.\n"
              f"  python3 {os.path.relpath(__file__, BASE)} --harness-check\n"
              "Sharing an unverified identity rung would write 15 controlled rows that were never "
              "computed as controlled.")
        return 1

    cells = load_cells()
    print(f"=== CIFAR-100 replication seed top-up: {D2} / {ATTACK.replace('committed_', '')} "
          f"on {DATASET} ===")
    print(f"    endpoint rungs kappa in {{{LO}, {HI}}}, seeds {NEW_SEEDS[0]}-{NEW_SEEDS[-1]}, "
          "both designs")
    print(f"    n=5 -> n=20 on BOTH legs. Rules frozen at {PREREG_COMMIT}.")
    print(f"    kappa={LO} is computed ONCE per seed and shared, so {len(NEW_SEEDS)} x 3 = "
          f"{len(NEW_SEEDS) * 3} runs, not {len(NEW_SEEDS) * 4}.")
    print("    Demotion clause: an n=20 interval containing zero, or two overlapping intervals, "
          "demotes\n    the CIFAR-100 replication to a design disagreement in the abstract and in "
          "Figure 1.\n", flush=True)

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
