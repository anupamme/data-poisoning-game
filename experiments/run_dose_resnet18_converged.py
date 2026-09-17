"""
The ResNet18 replication arm re-run at a 4x longer horizon: 200 rounds (Round 69, ninth review).

Rules frozen at ca89f96 (experiments/pre_registration_dose_resnet18_converged.md). ORCHESTRATION ONLY:
`run_one` is imported from experiments/run_targeted_dose.py and the arm's constants from that file and
from experiments/run_dose_resnet18.py, so this arm and its 50-round parent are the same computation by
construction rather than by inspection.

THE QUESTION IS THE REVIEW'S. main.tex:1534 discloses of the published arm that "Clean accuracy is far
below this architecture's own reach: 0.536-0.557 against the 0.66-0.76 the ResNet18 mini-sweep attains
in the same regime, ResNet18 at 50 rounds being far from converged." The review's objection is that a
security conclusion drawn at 54% clean accuracy may not survive further training. ONE THING MOVES: the
round count, 50 -> 200. Dataset, architecture, N, K, f, alpha, attack, defense, mode, rungs and seeds
are carried over unchanged from pre_registration_dose_resnet18.md @ 89046f6.

TWO GATES THAT DECIDE DIFFERENT THINGS, and the separation is the point:
  validity   -- the parent's own frozen gate. Identity-rung mean ASR < SUPPRESS_ASR = 0.5 at mean
                clean accuracy >= ACC_FLOOR = 0.35, and every rung above the floor. Failure means
                VOID, NOT NEGATIVE.
  label      -- CONVERGED_ACC = 0.65, the bottom of the 0.66-0.76 band the paper's own ResNet18
                mini-sweep attains in this regime, rounded DOWN. Failure does not void the arm; it
                fixes the FINDING as "four times the round budget does not bring Krum-aggregated
                ResNet18 to the accuracy this architecture reaches under averaging defenses here."
The kappa=2 leg runs either way: 0.35 decides validity, 0.65 decides only what the arm may be called.

THE INFORMATIVENESS CONDITION, pre-registered because a favourable outcome is partly guaranteed by a
floor. At 50 rounds the identity rung sits at ASR 0.0707. A quantity starting at 0.07 cannot fall by
more than 0.07, so |Delta| < 0.15 is attainable by arithmetic alone. If the identity rung's mean ASR
is below 0.15, the arm is reported as replicating WITH THE MARGIN NOT BINDING, never as a strengthened
replication. The freeze expects this condition to fail and says so.

NO CROSS-HORIZON DELTA. 200 rounds is four times the backdoor injection exposure, so this arm answers
"does the within-arm dissociation survive at a longer horizon" and NOT "what is ResNet18's converged
ASR relative to the 50-round arm". ASR_200(kappa) - ASR_50(kappa) is never computed or reported.

Cost: 40-54 h for 6 runs. The parent arm's own measured rate is 2.23 h/run at 50 rounds
(main.tex:1534(iv) prices its 6 runs at 13.4 h), giving ~8.9 h/run and ~54 h under linear scaling in
rounds; the mini-sweep's 6039 s mean gives ~6.7 h/run and ~40 h. Linear scaling is an assumption, so
the realized wall time is stored per run and reported.

Usage (from the repository root):
    PYTHONPATH=. python3 -m experiments.run_dose_resnet18_converged --harness-check
    PYTHONPATH=. python3 -m experiments.run_dose_resnet18_converged
"""
import json
import os
import subprocess
import sys
import time

import warnings
warnings.filterwarnings("ignore")
import numpy as np

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)

from config import FLConfig                                              # noqa: E402
# Single-sourced from the frozen Round-12 suite. Only the round count differs, and it differs by
# argument (run_one's fl_config) rather than by a fork.
from experiments.run_targeted_dose import (                              # noqa: E402
    run_one, cell_key, dial, KAPPAS, ACC_FLOOR, EQUIV_MARGIN, TOL, FL_CONFIG as FROZEN_CFG,
)
# The parent arm's own constants and its read-only, mixed-n-refusing helpers. Its run_rungs/save/
# verdict are NOT imported: they are bound to results/dose_resnet18/ through module globals, and
# calling them from here would write into a frozen artifact.
from experiments.run_dose_resnet18 import (                              # noqa: E402
    SUPPRESS_ASR, arm_delta, mean_asr, mean_acc, seeds_of,
)

PREREG = "experiments/pre_registration_dose_resnet18_converged.md"
PREREG_COMMIT = "ca89f96"

DATASET, MODEL = "cifar10", "resnet18"
SEEDS3 = [42, 43, 44]                 # the parent arm's three seeds, unchanged
KAPPAS_ARM = [0.0, 2.0]               # ENDPOINT RUNGS ONLY. Same restriction, four times the price.
ROUNDS = 200                          # the one thing this arm changes
CONVERGED_ACC = 0.65                  # label gate: the floor of the mini-sweep's 0.66-0.76 band
INFORMATIVE_ASR = 0.15                # below this the equivalence margin cannot bind on a fall

assert all(any(abs(k - f) < TOL for f in KAPPAS) for k in KAPPAS_ARM), \
    "the arm's rungs must be a subset of the frozen ladder, or the cell keys will not resolve"

# THE HORIZON IS THE ONLY DIFFERENCE, and it is asserted rather than asserted-in-prose. Every other
# FLConfig field must equal the frozen suite's, or this arm varies more than one factor and its
# comparison to the parent means nothing.
CFG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=ROUNDS)
_differs = sorted(k for k in vars(CFG) if getattr(CFG, k) != getattr(FROZEN_CFG, k))
assert _differs == ["num_rounds"], (
    f"this arm must differ from the frozen config in num_rounds ALONE; it differs in {_differs}")
assert FROZEN_CFG.num_rounds == 50 and CFG.num_rounds == 200, (FROZEN_CFG.num_rounds, CFG.num_rounds)

ARM = ("krum", "committed_scaling", "c) not invariant",
       "FLAT if the 50-round negative is a property of the mechanism; RISING if it was specific to "
       "short-horizon training", SEEDS3)
CIFAR_ARM = ("krum", "committed_scaling")
REPLICATE_WITHIN = EQUIV_MARGIN       # 0.15, unchanged -- no new constant

out_dir = os.path.join(base, "results", "dose_resnet18_converged")
out_path = os.path.join(out_dir, "summary.json")
PARENT = os.path.join(base, "results", "dose_resnet18", "summary.json")
TARGETED = os.path.join(base, "results", "targeted_dose", "summary.json")
FEMNIST = os.path.join(base, "results", "dose_femnist", "summary.json")


# --------------------------------------------------------------------------------------------------
# The freeze
# --------------------------------------------------------------------------------------------------
def check_frozen():
    """A hash that nobody checks is a claim, not a freeze. See run_dose_resnet18.check_frozen."""
    path = os.path.join(base, PREREG)
    if not os.path.exists(path):
        sys.exit(f"REFUSING TO RUN: {PREREG} does not exist.")
    if PREREG_COMMIT is None:
        sys.exit("REFUSING TO RUN: the predicted outcome is not frozen.\n"
                 f"  1. git add {PREREG} && git commit\n"
                 "  2. set PREREG_COMMIT here to that hash.")
    try:
        head = subprocess.run(["git", "-C", base, "rev-parse", "--verify", f"{PREREG_COMMIT}^{{commit}}"],
                              capture_output=True, text=True)
        if head.returncode != 0:
            sys.exit(f"REFUSING TO RUN: PREREG_COMMIT {PREREG_COMMIT} does not resolve to a commit in "
                     f"{base}. The freeze names nothing.")
        committed = subprocess.run(["git", "-C", base, "rev-parse", f"{PREREG_COMMIT}:{PREREG}"],
                                   capture_output=True, text=True)
        if committed.returncode != 0:
            sys.exit(f"REFUSING TO RUN: {PREREG} does not exist at commit {PREREG_COMMIT}.")
        working = subprocess.run(["git", "-C", base, "hash-object", PREREG],
                                 capture_output=True, text=True)
        if working.returncode != 0:
            sys.exit(f"REFUSING TO RUN: cannot hash {PREREG} to compare it against the freeze.")
        if committed.stdout.strip() != working.stdout.strip():
            sys.exit(f"REFUSING TO RUN: {PREREG} has been EDITED since it was frozen at "
                     f"{PREREG_COMMIT}.\n"
                     f"  committed blob: {committed.stdout.strip()[:12]}\n"
                     f"  working blob:   {working.stdout.strip()[:12]}\n"
                     "The rules on disk are not the rules on record.")
    except FileNotFoundError:
        sys.exit("REFUSING TO RUN: git is not available, so the freeze cannot be verified. An "
                 "unverifiable freeze is not a freeze.")
    print(f"  freeze verified: {PREREG} @ {PREREG_COMMIT}, blob equal to the working copy")


# --------------------------------------------------------------------------------------------------
# The harness check: BY VALUE against the parent's stored rows, never by hashing an artifact
# --------------------------------------------------------------------------------------------------
def parent_cell(v):
    if not os.path.exists(PARENT):
        return None
    d2, atk = ARM[0], ARM[1]
    return json.load(open(PARENT)).get("cells", {}).get(cell_key("S", d2, atk, v))


def harness_check():
    """Re-run one PUBLISHED 50-round cell through the imported run_one with fl_config=None.

    This proves the import path still reproduces the parent arm before 40-54 h is spent on the 200-
    round one. A value comparison against a stored per-seed row, NOT an md5 of a results file: this
    repository has already shipped one arm that was bit-identical to another because a manipulation
    hook was never called and failed silently. If it fails, the arm does not run.
    """
    d2, atk = ARM[0], ARM[1]
    cell = parent_cell(0.0)
    if not cell or not cell.get("per_seed"):
        sys.exit(f"REFUSING: no stored kappa=0 rows in {PARENT} to check the harness against.")
    stored = cell["per_seed"][0]
    seed = stored["seed"]
    print(f"  harness check: seed {seed}, kappa=0, 50 rounds (fl_config=None -> the frozen config)")
    print(f"    stored: acc={stored['accuracy']:.6f} ASR={stored['asr']:.6f}")
    t = time.time()
    acc, asr = run_one(seed, "S", d2, atk, 0.0, dataset=DATASET, model=MODEL)
    print(f"    fresh : acc={acc:.6f} ASR={asr:.6f}   ({(time.time()-t)/3600:.2f} h)")
    d_acc, d_asr = abs(acc - stored["accuracy"]), abs(asr - stored["asr"])
    print(f"    |dacc| {d_acc:.9e}   |dASR| {d_asr:.9e}   tol 1e-9")
    if d_acc > 1e-9 or d_asr > 1e-9:
        sys.exit("HARNESS CHECK FAILED. The imported run_one no longer reproduces the parent arm, so "
                 "the 200-round cells would not be comparable to the 50-round ones. The arm does not "
                 "run.")
    print("  HARNESS CHECK PASSED: the imported run_one reproduces the frozen row bit-for-bit, and "
          "fl_config=None\n  resolves to the frozen 50-round configuration.")
    return {"seed": seed, "rung": 0.0, "rounds": FROZEN_CFG.num_rounds,
            "abs_dev_acc": d_acc, "abs_dev_asr": d_asr, "tol": 1e-9, "passed": True}


# --------------------------------------------------------------------------------------------------
# Checkpointing. Writes results/dose_resnet18_converged/ and nothing else.
# --------------------------------------------------------------------------------------------------
def load_cells():
    if not os.path.exists(out_path):
        return {}
    try:
        return json.load(open(out_path)).get("cells", {})
    except Exception:
        return {}


def save(cells):
    d2, atk, cls, ordering, seeds = ARM
    c_delta, c_lo, c_hi = arm_delta(TARGETED, *CIFAR_ARM)
    f_delta, f_lo, f_hi = arm_delta(FEMNIST, *CIFAR_ARM)
    p_delta, p_lo, p_hi = arm_delta(PARENT, *CIFAR_ARM)
    os.makedirs(out_dir, exist_ok=True)
    json.dump({"description":
                   "The ResNet18/Mode-S/krum replication arm at a 4x longer horizon: 200 rounds "
                   "instead of 50, everything else carried over from "
                   "pre_registration_dose_resnet18.md @ 89046f6. Answers the ninth review's objection "
                   "to disclosure (ii) of main.tex:1534, that the published arm evaluates security on "
                   f"a model at 0.536-0.557 clean accuracy. Rules frozen at {PREREG_COMMIT} ({PREREG}).",
               "prereg_commit": PREREG_COMMIT,
               "dataset": DATASET, "model": MODEL,
               "config": {"N": CFG.num_clients, "K": CFG.clients_per_round, "f": 0.2, "alpha": 0.5,
                          "rounds": CFG.num_rounds, "rounds_parent": FROZEN_CFG.num_rounds,
                          "differs_from_frozen_config_in": _differs,
                          "kappas_run": KAPPAS_ARM, "kappas_frozen_ladder": KAPPAS,
                          "endpoint_only": True,
                          "endpoint_only_reason":
                              "Same restriction as the 50-round parent at four times the price: "
                              "~6.7-8.9 h/run, so two rungs x 3 seeds is 40-54 h and four rungs would "
                              "be 80-108 h. A two-rung ladder has no trend to test and is never "
                              "displayed as four.",
                          "no_secondary_trend_test": True,
                          "seeds": seeds,
                          "rhos": {str(k): dial("S", k) for k in KAPPAS_ARM},
                          "acc_floor": ACC_FLOOR, "suppress_asr": SUPPRESS_ASR,
                          "converged_acc": CONVERGED_ACC, "informative_asr": INFORMATIVE_ASR,
                          "equiv_margin": EQUIV_MARGIN, "replicate_within": REPLICATE_WITHIN},
               "arm": {"mode": "S", "d2": d2, "attack": atk, "prop1_class": cls,
                       "predicted_outcome": ordering, "seeds": seeds,
                       "varies": "the round count alone"},
               "no_cross_horizon_delta":
                   "200 rounds is four times the backdoor injection exposure, so no Delta is taken "
                   "across horizons. ASR_200(kappa) - ASR_50(kappa) is not computed, not reported and "
                   "not described. The parent's figures appear only as context, labelled with their "
                   "own round count.",
               "parent_50round_comparison": {"d2": CIFAR_ARM[0], "attack": CIFAR_ARM[1],
                                             "mode_s_delta": p_delta, "n_kappa0": p_lo,
                                             "n_kappa2": p_hi, "rounds": FROZEN_CFG.num_rounds},
               "cifar_cnn_comparison": {"d2": CIFAR_ARM[0], "attack": CIFAR_ARM[1],
                                        "mode_s_delta": c_delta, "n_kappa0": c_lo, "n_kappa2": c_hi},
               "femnist_comparison": {"d2": CIFAR_ARM[0], "attack": CIFAR_ARM[1],
                                      "mode_s_delta": f_delta, "n_kappa0": f_lo, "n_kappa2": f_hi},
               "verdict": verdict(cells),
               "cells": cells}, open(out_path, "w"), indent=2)


def run_rungs(cells, todo):
    d2, atk, cls, ordering, _ = ARM
    t0, done = time.time(), 0
    for v, seed in todo:
        key = cell_key("S", d2, atk, v)
        cell = cells.setdefault(key, {"mode": "S", "d2": d2, "attack": atk, "rung": v,
                                      "dial": dial("S", v), "prop1_class": cls,
                                      "dataset": DATASET, "model": MODEL, "rounds": CFG.num_rounds,
                                      "predicted_outcome": ordering, "per_seed": []})
        t = time.time()
        # fl_config is the hook the parent round added, and it is honored in the round loop:
        # run_targeted_dose.run_one:163 sets cfg = fl_config or FL_CONFIG and cfg.num_rounds drives
        # `for rnd in range(cfg.num_rounds)`. No new hook is added here.
        acc, asr = run_one(seed, "S", d2, atk, v, dataset=DATASET, model=MODEL, fl_config=CFG)
        cell["per_seed"] = [r for r in cell["per_seed"] if r["seed"] != seed] + [
            {"seed": seed, "accuracy": acc, "asr": asr, "wall_time_s": time.time() - t}]
        cell["per_seed"].sort(key=lambda r: r["seed"])
        cell["mean_asr"], cell["mean_acc"] = mean_asr(cell), mean_acc(cell)
        cell["std_asr"] = float(np.std([r["asr"] for r in cell["per_seed"]], ddof=0))
        cell["n"] = len(cell["per_seed"])
        cells[key] = cell
        save(cells)
        done += 1
        # [done/total] counts THIS invocation only and is not progress through the arm: a resumed
        # invocation has a smaller total because finished runs are not in its todo list. The count
        # that means something is the number of per_seed rows in the artifact.
        have = sum(len(c.get("per_seed", [])) for c in cells.values())
        print(f"  [{done}/{len(todo)} this invocation] {MODEL} S {d2} kappa={v} s{seed} "
              f"@{CFG.num_rounds}r: acc={acc:.4f} ASR={asr:.4f} ({(time.time()-t)/3600:.2f} h; "
              f"artifact holds {have}/{len(KAPPAS_ARM)*len(SEEDS3)} runs)", flush=True)
    if todo:
        print(f"  ({(time.time() - t0) / 3600:.2f} h this invocation)")
    return done


# --------------------------------------------------------------------------------------------------
# The gates and the primary rule
# --------------------------------------------------------------------------------------------------
def report_gate(cells):
    """The parent's frozen VALIDITY gate, plus this arm's CONVERGENCE LABEL gate, reported apart."""
    d2, atk, _, _, _ = ARM
    lo = cells.get(cell_key("S", d2, atk, 0.0))
    if lo is None or not lo.get("per_seed"):
        print("  gate not scoreable: no kappa=0 rows.")
        return None
    a, acc, n = mean_asr(lo), mean_acc(lo), len(lo["per_seed"])
    valid = bool(a < SUPPRESS_ASR and acc >= ACC_FLOOR)
    converged = bool(acc >= CONVERGED_ACC)
    informative = bool(a >= INFORMATIVE_ASR)
    print(f"\n  kappa=0 (krum alone) @ {CFG.num_rounds} rounds: mean ASR {a:.4f} @ mean acc "
          f"{acc:.4f}, n={n}")
    if valid:
        print(f"  VALIDITY GATE PASSED: ASR {a:.4f} < {SUPPRESS_ASR} at accuracy >= {ACC_FLOOR}. "
              "There is suppression for the ladder to preserve or lose.")
    elif acc < ACC_FLOOR:
        print(f"  VALIDITY GATE FAILED, accuracy: {acc:.4f} < {ACC_FLOOR}. A low ASR on a collapsed "
              "model is not suppression. THE ARM IS VOID, NOT NEGATIVE.")
    else:
        print(f"  VALIDITY GATE FAILED, no suppression: ASR {a:.4f} >= {SUPPRESS_ASR}. THE ARM IS "
              "VOID, NOT NEGATIVE, and it licenses no sentence about horizons.")
    if converged:
        print(f"  CONVERGENCE LABEL GATE PASSED: {acc:.4f} >= {CONVERGED_ACC}, the floor of the "
              "0.66-0.76 band this architecture reaches in this regime. The arm may be described as "
              "evaluated near that band.")
    else:
        print(f"  CONVERGENCE LABEL GATE FAILED: {acc:.4f} < {CONVERGED_ACC}. THE PRE-REGISTERED "
              "FINDING IS: four times the round budget does not bring Krum-aggregated ResNet18 to the "
              "accuracy this architecture reaches under averaging defenses in this regime. The arm is "
              "NOT described as near-converged, and the kappa=2 leg still runs: 0.35 decides "
              "validity, 0.65 decides only the label.")
    if informative:
        print(f"  INFORMATIVENESS CONDITION MET: identity ASR {a:.4f} >= {INFORMATIVE_ASR}, so the "
              f"+-{REPLICATE_WITHIN} margin can bind on a fall and the equivalence test is a real one.")
    else:
        print(f"  INFORMATIVENESS CONDITION NOT MET: identity ASR {a:.4f} < {INFORMATIVE_ASR}, so "
              f"|Delta| < {REPLICATE_WITHIN} is attainable by arithmetic alone. Any replication here "
              "is reported WITH THE MARGIN NOT BINDING and never as a strengthened replication. The "
              "freeze expected this.")
    return {"kappa0_mean_asr": a, "kappa0_mean_acc": acc, "n": n,
            "validity_gate_passed": valid, "suppress_asr": SUPPRESS_ASR, "acc_floor": ACC_FLOOR,
            "convergence_label_gate_passed": converged, "converged_acc": CONVERGED_ACC,
            "convergence_fallback_finding":
                None if converged else
                ("Four times the round budget does not bring Krum-aggregated ResNet18 to the accuracy "
                 "this architecture reaches under averaging defenses in this regime: mean clean "
                 f"accuracy {acc:.4f} at {CFG.num_rounds} rounds against the {CONVERGED_ACC} gate. "
                 "This arm cannot separate an aggregation-rule explanation (Krum selects one update "
                 "per round; the 0.66-0.76 comparison band comes from averaging mixture policies) "
                 "from a horizon explanation, and does not claim to."),
            "informativeness_condition_met": informative, "informative_asr": INFORMATIVE_ASR}


def paired_interval(lo, hi):
    """Paired 95% interval on the per-seed differences, and m* = the smallest containing margin."""
    r_lo = {r["seed"]: r["asr"] for r in lo.get("per_seed", [])}
    r_hi = {r["seed"]: r["asr"] for r in hi.get("per_seed", [])}
    seeds = sorted(set(r_lo) & set(r_hi))
    if sorted(r_lo) != sorted(r_hi) or len(seeds) < 2:
        return None
    from experiments.analyze_headline_cis import t_crit
    d = np.array([r_hi[s] - r_lo[s] for s in seeds])
    n = len(d)
    m, sd = float(d.mean()), float(d.std(ddof=1))
    hw = float(t_crit(n) * sd / np.sqrt(n))
    return {"seeds": seeds, "n": n, "per_seed_diff": [float(x) for x in d], "mean": m, "sd": sd,
            "hw95": hw, "lo": m - hw, "hi": m + hw,
            "m_star": float(max(abs(m - hw), abs(m + hw))),
            "m_star_note": "the smallest margin this arm's interval fits inside; no equivalence claim "
                           f"is made at any margin tighter than the frozen {REPLICATE_WITHIN}"}


def verdict(cells):
    """The frozen primary rule. Returns None unless BOTH endpoints exist on the SAME seed set."""
    d2, atk, _, _, _ = ARM
    lo, hi = cells.get(cell_key("S", d2, atk, 0.0)), cells.get(cell_key("S", d2, atk, 2.0))
    if lo is None or hi is None:
        return None
    s_lo, s_hi = seeds_of(lo), seeds_of(hi)
    if s_lo != s_hi:
        # A Delta across unequal seed sets is not a within-arm contrast, so it is REFUSED rather
        # than reported with a caveat.
        return {"delta": None, "n_kappa0": len(s_lo), "n_kappa2": len(s_hi),
                "seeds_kappa0": s_lo, "seeds_kappa2": s_hi,
                "verdict": "NOT SCOREABLE: the two endpoints were scored on different seed sets, so "
                           "their difference mixes n. Finish the shorter leg on the same seeds."}
    a0, a2 = mean_asr(lo), mean_asr(hi)
    acc0, acc2 = mean_acc(lo), mean_acc(hi)
    delta = a2 - a0
    if delta > REPLICATE_WITHIN:
        v = ("THE NEGATIVE IS HORIZON-SPECIFIC: statistic disturbance moves suppression on ResNet18 "
             "once trained further. The paper's central claim must be scoped to short-horizon "
             "training IN THE BODY, not in a limitation.")
    elif delta < -REPLICATE_WITHIN:
        v = "INDETERMINATE (attenuation-side fall; not scored in our favour)"
    else:
        v = f"THE NEGATIVE SURVIVES A {CFG.num_rounds // FROZEN_CFG.num_rounds}x LONGER HORIZON"
    valid = bool(a0 < SUPPRESS_ASR and acc0 >= ACC_FLOOR)
    floors_ok = bool(acc0 >= ACC_FLOOR and acc2 >= ACC_FLOOR)
    informative = bool(a0 >= INFORMATIVE_ASR)
    if valid and not informative and delta > -REPLICATE_WITHIN:
        v += (" -- WITH THE MARGIN NOT BINDING: the identity rung has less headroom "
              f"({a0:.4f}) than the margin ({REPLICATE_WITHIN}), so this is not a strengthened "
              "replication.")
    return {"rounds": CFG.num_rounds,
            "identity_mean_asr": a0, "kappa2_mean_asr": a2, "delta": delta,
            "n_kappa0": len(s_lo), "n_kappa2": len(s_hi), "seeds": s_lo,
            "replicate_within": REPLICATE_WITHIN,
            "paired_interval": paired_interval(lo, hi),
            "verdict": v if valid else
                       "VOID, NOT NEGATIVE: the kappa=0 validity gate failed, so there was no "
                       "suppression for the ladder to preserve. The primary rule does not apply and "
                       "the Delta below is reported for completeness only.",
            "validity_gate_passed": valid,
            "per_rung_mean_acc": {"0.0": acc0, "2.0": acc2},
            "accuracy_floor_ok": floors_ok,
            "convergence_label_gate_passed": bool(acc0 >= CONVERGED_ACC),
            "informativeness_condition_met": informative,
            "endpoint_only": True, "no_secondary_trend_test": True,
            "rungs_run": KAPPAS_ARM, "rungs_in_frozen_ladder": KAPPAS}


# --------------------------------------------------------------------------------------------------
def main():
    argv = sys.argv[1:]
    d2, atk, cls, ordering, seeds = ARM
    print("=" * 78)
    print(f"  ResNet18 MODE-S/KRUM ARM AT {CFG.num_rounds} ROUNDS  (Round 69, ninth review)")
    print("=" * 78)
    print(f"    {DATASET}/{MODEL}, mode S, {d2}, {atk}, seeds {seeds}, rungs {KAPPAS_ARM}")
    print(f"    ONE factor moves: rounds {FROZEN_CFG.num_rounds} -> {CFG.num_rounds}. Config differs "
          f"from the frozen suite in {_differs} and nothing else.")
    print(f"    ENDPOINT RUNGS ONLY ({KAPPAS_ARM} of the frozen ladder {KAPPAS}); no trend test, and "
          "these two rungs are never displayed as four.")
    print("    NO CROSS-HORIZON DELTA: 200 rounds is 4x the injection exposure, so the 50-round "
          "figures are context only.")
    print(f"    budget 40-54 h for {len(KAPPAS_ARM) * len(seeds)} runs; per-run wall time is stored.")
    check_frozen()

    if "--harness-check" in argv:
        harness_check(); return

    os.makedirs(out_dir, exist_ok=True)
    cells = load_cells()

    # THE VALIDITY GATE IS SCORED FIRST AND ON ITS OWN. kappa=0 runs are stored, not thrown away:
    # doseS_kappa0.0 hands the update list to krum unwrapped, so the kappa=0 rung IS krum alone, and
    # run_one reseeds torch and numpy from its arguments, so the run is the same computation whether
    # it was invoked here or from the main loop. The main loop's resume logic then skips it.
    print("\n=== GATE (scored first): does krum alone suppress scaling on ResNet18 at 200 rounds? ===")
    print(f"    VOID unless mean ASR < {SUPPRESS_ASR} at mean clean accuracy >= {ACC_FLOOR}.")
    print(f"    Separately, the CONVERGENCE LABEL gate is mean clean accuracy >= {CONVERGED_ACC}; "
          "failing it fixes the finding, it does not void the arm.\n", flush=True)
    have0 = {r["seed"] for r in (cells.get(cell_key("S", d2, atk, 0.0)) or {}).get("per_seed", [])}
    run_rungs(cells, [(0.0, s) for s in seeds if s not in have0])
    g = report_gate(cells)

    if g and not g["validity_gate_passed"]:
        save(cells)
        print("\n  The kappa=2 leg is NOT run: the arm is void and there is nothing for the ladder to "
              "preserve.\n  Reported as void, in the paper, next to the arms that were not void.")
        print(f"  written: {out_path}")
        return

    print(f"\n=== LADDER: kappa=2.0 (rho={dial('S', 2.0):.2f}) at {CFG.num_rounds} rounds ===",
          flush=True)
    have2 = {r["seed"] for r in (cells.get(cell_key("S", d2, atk, 2.0)) or {}).get("per_seed", [])}
    run_rungs(cells, [(2.0, s) for s in seeds if s not in have2])
    save(cells)

    v = verdict(cells)
    print("\n" + "=" * 78)
    print("  PRIMARY RULE (frozen): Delta = mean ASR(kappa=2) - mean ASR(kappa=0)")
    print("=" * 78)
    if v is None or v.get("delta") is None:
        print(f"  {(v or {}).get('verdict', 'not scoreable yet')}")
    else:
        pi = v.get("paired_interval") or {}
        print(f"  identity {v['identity_mean_asr']:.4f}  kappa=2 {v['kappa2_mean_asr']:.4f}  "
              f"Delta {v['delta']:+.4f}  (margin +-{REPLICATE_WITHIN}, n={v['n_kappa0']})")
        if pi:
            print(f"  paired 95% CI [{pi['lo']:+.4f}, {pi['hi']:+.4f}]  m* {pi['m_star']:.4f}  "
                  f"per-seed diffs {[round(x, 4) for x in pi['per_seed_diff']]}")
        print(f"  accuracy {v['per_rung_mean_acc']['0.0']:.4f} / "
              f"{v['per_rung_mean_acc']['2.0']:.4f}  (floor {ACC_FLOOR}, label gate {CONVERGED_ACC})")
        print(f"  VERDICT: {v['verdict']}")
        p_delta, p_lo, p_hi = arm_delta(PARENT, *CIFAR_ARM)
        print(f"  context, NOT a contrast: the {FROZEN_CFG.num_rounds}-round parent's own within-arm "
              f"Delta is {p_delta:+.4f} (n={p_lo}). No cross-horizon Delta is taken.")
    print(f"  written: {out_path}")


if __name__ == "__main__":
    main()
