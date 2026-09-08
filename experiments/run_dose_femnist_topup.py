"""Seed top-up for the EMNIST-byclass Mode-S krum arm: n=3 -> n=5. Adds seeds only.

WHY. results/dose_femnist/ is the paper's second-dataset replication of its flagship negative, and it
is the paper's thinnest arm at n=3. The original pre-registration froze n=3 for a compute reason it
states outright ("4 rungs x 3 seeds x ~55 min is already ~11 h"), not a statistical one, and a
reviewer named it: the headline CIFAR-10 evidence runs at n=5 and this replication at n=3, so the two
are not read at equal power. Two more seeds fix that and nothing else.

WHAT IS FROZEN, AND WHY THIS IS NOT A VIOLATION. Rules are copied verbatim from
experiments/pre_registration_dose_femnist_topup.md, which amends the original by ADDING SEEDS 45 AND
46 and changing no decision rule: |Delta| < 0.15 REPLICATED, Delta > +0.15 DATASET-SPECIFIC,
Delta < -0.15 INDETERMINATE and not scored in our favour, accuracy gate ACC_FLOOR at every rung. The
frozen n=3 Delta is -0.017, so n=5 IS A TEST THIS ARM CAN FAIL, and the amendment commits in advance
to reporting a flip as a flip. The paper has this precedent twice: the Krum n=20 top-up, and the
cos_krum n=8 top-up that flipped a C1 input from 0.300 to 0.584 and is reported flipped.

FROZEN ARTIFACTS ARE NOT TOUCHED. results/dose_femnist/summary.json is opened read-only, its md5 is
printed at start and re-checked at exit, and every value this script produces goes to
results/dose_femnist_topup/summary.json. Nothing here rewrites a published number.

TWO GUARDS, BECAUSE THE FAILURE MODE THEY COVER IS SILENT.
  (1) Static: attack.manipulate_update is the ONLY thing separating committed_scaling from
      committed_pixel -- the two share poison_dataset -- and when a different runner omitted that call
      an entire FEMNIST wave came back bit-identical to the pixel arm and looked plausible
      (results/femnist_c1_inputs_scaling_INVALID/). run_targeted_dose.run_one does call it; this
      script asserts the call is still in that source and aborts if it is not.
  (2) Functional, at no extra cost: kappa=0 is identity-then-krum, i.e. krum ALONE, so the new seeds'
      kappa=0 ASR must land near the payoff matrix's krum/model_scaling 0.027. A skipped hook would
      return the pixel arm's 0.660 instead. kappa=0 runs FIRST and the script aborts above
      HOOK_MAX_ASR before spending six more hours.

run_one is IMPORTED from run_targeted_dose, like the arm it tops up, so the two cannot drift apart in
what they compute.

Resumable: every run is checkpointed and never recomputed.

Output: results/dose_femnist_topup/summary.json
Run:    python3 experiments/run_dose_femnist_topup.py
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base)
os.chdir(base)

from experiments.run_targeted_dose import (  # noqa: E402
    run_one, cell_key, dial, KAPPAS, ACC_FLOOR, EQUIV_MARGIN,
)
from experiments.run_dose_femnist import (  # noqa: E402
    DATASET, MODEL, ARM, SEEDS3, REPLICATE_WITHIN, PREREG_COMMIT,
)

# experiments/pre_registration_dose_femnist_topup.md, committed before this ran.
PREREG_COMMIT_TOPUP = "ad5479b"  # the amendment's commit

NEW_SEEDS = [45, 46]            # named in the amendment before any run
ALL_SEEDS = SEEDS3 + NEW_SEEDS  # n=5

# The kappa=0 functional guard. Standalone krum/model_scaling on EMNIST-byclass is 0.027; the pixel
# arm, which a skipped manipulate_update would silently produce, is 0.660.
HOOK_EXPECT_ASR = 0.027
HOOK_PIXEL_ASR = 0.660
HOOK_MAX_ASR = 0.35

FROZEN = os.path.join(base, "results", "dose_femnist", "summary.json")
OUT_DIR = os.path.join(base, "results", "dose_femnist_topup")
OUT = os.path.join(OUT_DIR, "summary.json")

D2, ATTACK, PROP1_CLASS, ORDERING, _ = ARM


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def frozen_cells():
    """The n=3 rows, read-only."""
    return json.load(open(FROZEN))["cells"]


def load():
    if os.path.exists(OUT):
        return json.load(open(OUT))
    return {"description": "Seed top-up (n=3 -> n=5) for the EMNIST-byclass/simple_cnn Mode-S krum "
                           "arm of results/dose_femnist/. Adds seeds 45 and 46 at all four rungs; "
                           "no decision rule changed. Rules frozen in "
                           "experiments/pre_registration_dose_femnist_topup.md.",
            "amends": {"artifact": "results/dose_femnist/summary.json",
                       "prereg": "experiments/pre_registration_dose_femnist.md",
                       "prereg_commit": PREREG_COMMIT},
            "prereg_commit_topup": PREREG_COMMIT_TOPUP,
            "dataset": DATASET, "model": MODEL,
            "config": {"N": 10, "K": 5, "f": 0.2, "alpha": 0.5, "rounds": 50,
                       "kappas": KAPPAS, "frozen_seeds": SEEDS3, "new_seeds": NEW_SEEDS,
                       "seeds_n5": ALL_SEEDS,
                       "rhos": {str(k): dial("S", k) for k in KAPPAS},
                       "acc_floor": ACC_FLOOR, "equiv_margin": EQUIV_MARGIN,
                       "replicate_within": REPLICATE_WITHIN},
            "arm": {"mode": "S", "d2": D2, "attack": ATTACK, "prop1_class": PROP1_CLASS,
                    "predicted_outcome": ORDERING},
            "new_cells": {}}


def save(d):
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(d, open(OUT, "w"), indent=2)


def pooled(new_cells, kappa):
    """All per-seed rows for a rung: the frozen n=3 plus whatever this run has added."""
    key = cell_key("S", D2, ATTACK, kappa)
    rows = list(frozen_cells().get(key, {}).get("per_seed", []))
    rows += new_cells.get(key, {}).get("per_seed", [])
    return sorted(rows, key=lambda r: r["seed"])


def verdict(new_cells, seeds_expected):
    """The frozen primary rule, evaluated on whatever rows exist. Never refits the margin."""
    lo, hi = pooled(new_cells, 0.0), pooled(new_cells, 2.0)
    if len(lo) < seeds_expected or len(hi) < seeds_expected:
        return None
    a0, a2 = float(np.mean([r["asr"] for r in lo])), float(np.mean([r["asr"] for r in hi]))
    delta = a2 - a0
    if delta > REPLICATE_WITHIN:
        v = ("NEGATIVE IS DATASET- OR ARCHITECTURE-SPECIFIC: statistic disturbance moves suppression "
             "on EMNIST-byclass, and the central claim is scoped to CIFAR-10")
    elif delta < -REPLICATE_WITHIN:
        v = "INDETERMINATE (attenuation-side fall; not scored in our favour)"
    else:
        v = "FLAGSHIP NEGATIVE REPLICATED on a second dataset and architecture"
    accs = {}
    for k in KAPPAS:
        rows = pooled(new_cells, k)
        if rows:
            accs[str(k)] = float(np.mean([r["accuracy"] for r in rows]))
    return {"n": len(lo), "identity_mean_asr": a0, "kappa2_mean_asr": a2, "delta": delta,
            "replicate_within": REPLICATE_WITHIN, "verdict": v, "per_rung_mean_acc": accs,
            "accuracy_gate_ok": bool(all(a >= ACC_FLOOR for a in accs.values()))}


def check_frozen():
    prereg = os.path.join(base, "experiments", "pre_registration_dose_femnist_topup.md")
    if not os.path.exists(prereg):
        sys.exit(f"REFUSING TO RUN: {prereg} does not exist.")
    if PREREG_COMMIT_TOPUP is None:
        sys.exit("REFUSING TO RUN: the amendment is not frozen.\n"
                 f"  1. git commit {os.path.relpath(prereg, base)}\n"
                 "  2. set PREREG_COMMIT_TOPUP in this file to that hash.\n"
                 "The frozen n=3 delta is -0.017, so n=5 is a test this arm can fail. An unfrozen "
                 "top-up makes that unfalsifiable, which is the entire point of running it.")
    r = subprocess.run(["git", "cat-file", "-e", f"{PREREG_COMMIT_TOPUP}:experiments/"
                        "pre_registration_dose_femnist_topup.md"], capture_output=True)
    if r.returncode != 0:
        sys.exit(f"REFUSING TO RUN: the amendment is not in commit {PREREG_COMMIT_TOPUP}.")
    if not os.path.exists(FROZEN):
        sys.exit(f"REFUSING TO RUN: {FROZEN} does not exist; there is no n=3 arm to top up.")
    if not os.path.exists(os.path.join(base, "results", "femnist_admission.json")):
        sys.exit("REFUSING TO RUN: results/femnist_admission.json is missing. The arm's premise was "
                 "measured prospectively and this top-up does not re-open it.")
    src = open(os.path.join(base, "experiments", "run_targeted_dose.py")).read()
    if "manipulate_update(" not in src:
        sys.exit("ABORT: run_targeted_dose no longer calls attack.manipulate_update, so "
                 "committed_scaling would silently equal committed_pixel. See "
                 "results/femnist_c1_inputs_scaling_INVALID/.")
    print("  [guard] amendment frozen at "
          f"{PREREG_COMMIT_TOPUP}; run_targeted_dose applies manipulate_update: OK")


def main():
    check_frozen()
    frozen_md5 = md5(FROZEN)
    data = load()
    data["prereg_commit_topup"] = PREREG_COMMIT_TOPUP
    data.setdefault("frozen_artifact_md5", {})["at_start"] = frozen_md5
    new_cells = data["new_cells"]

    n3 = verdict({}, 3)
    todo = [(k, s) for k in KAPPAS for s in NEW_SEEDS
            if not any(r["seed"] == s
                       for r in new_cells.get(cell_key("S", D2, ATTACK, k), {}).get("per_seed", []))]

    print("=== EMNIST-BYCLASS MODE-S KRUM: SEED TOP-UP n=3 -> n=5 ===")
    print(f"    rules frozen in pre_registration_dose_femnist_topup.md @ {PREREG_COMMIT_TOPUP}")
    print(f"    amends results/dose_femnist/ (md5 {frozen_md5}), which is NOT written")
    print(f"    {DATASET}/{MODEL}, mode S {D2}/{ATTACK.replace('committed_', '')}")
    print(f"    adding seeds {NEW_SEEDS} at kappa {KAPPAS}: {len(todo)} runs left of 8, ~56 min each")
    if n3:
        print(f"    frozen n=3: delta = {n3['delta']:+.4f} -> {n3['verdict'].split(':')[0]}")
    print(f"    REPLICATED if |delta| < {REPLICATE_WITHIN:.2f}; the margin does not move\n", flush=True)

    t0, done, guarded = time.time(), 0, False
    for kappa, seed in todo:
        key = cell_key("S", D2, ATTACK, kappa)
        cell = new_cells.setdefault(key, {"mode": "S", "d2": D2, "attack": ATTACK, "rung": kappa,
                                          "dial": dial("S", kappa), "prop1_class": PROP1_CLASS,
                                          "dataset": DATASET, "model": MODEL,
                                          "predicted_outcome": ORDERING, "per_seed": []})
        t = time.time()
        acc, asr = run_one(seed, "S", D2, ATTACK, kappa, dataset=DATASET, model=MODEL)
        cell["per_seed"].append({"seed": seed, "accuracy": acc, "asr": asr})
        cell["per_seed"].sort(key=lambda r: r["seed"])
        done += 1
        save(data)
        print(f"  [{done}/{len(todo)}] kappa={kappa} s{seed}: acc={acc:.4f} ASR={asr:.4f} "
              f"({time.time() - t:.0f}s)", flush=True)

        # Functional hook guard, as soon as kappa=0 has both new seeds and before the other 6 runs.
        rung0 = new_cells.get(cell_key("S", D2, ATTACK, 0.0), {}).get("per_seed", [])
        if not guarded and len(rung0) == len(NEW_SEEDS):
            m = float(np.mean([r["asr"] for r in rung0]))
            data["hook_guard"] = {"rung": 0.0, "new_seed_mean_asr": m,
                                  "expected_krum_alone": HOOK_EXPECT_ASR,
                                  "pixel_value_if_hook_skipped": HOOK_PIXEL_ASR,
                                  "max_allowed": HOOK_MAX_ASR, "passed": bool(m <= HOOK_MAX_ASR)}
            save(data)
            if m > HOOK_MAX_ASR:
                sys.exit(f"ABORT: kappa=0 is identity-then-krum, i.e. krum alone, and the new seeds "
                         f"mean ASR={m:.4f} > {HOOK_MAX_ASR}. Standalone krum on EMNIST-byclass is "
                         f"{HOOK_EXPECT_ASR}; the pixel arm is {HOOK_PIXEL_ASR}. The attack under "
                         f"test is not model scaling. {done} runs done, 0 interpretable.")
            print(f"  [guard] kappa=0 new-seed mean ASR={m:.4f} (krum alone is "
                  f"~{HOOK_EXPECT_ASR}, pixel would be {HOOK_PIXEL_ASR}): scaling hook live\n",
                  flush=True)
            guarded = True

    for k in KAPPAS:
        key = cell_key("S", D2, ATTACK, k)
        if key in new_cells:
            rows = pooled(new_cells, k)
            new_cells[key]["pooled_n5"] = {
                "n": len(rows), "seeds": [r["seed"] for r in rows],
                "mean_asr": float(np.mean([r["asr"] for r in rows])),
                "std_asr": float(np.std([r["asr"] for r in rows], ddof=0)),
                "mean_acc": float(np.mean([r["accuracy"] for r in rows]))}

    n5 = verdict(new_cells, len(ALL_SEEDS))
    data["verdict_n3_frozen"], data["verdict_n5"] = n3, n5
    data["frozen_artifact_md5"]["at_exit"] = md5(FROZEN)
    save(data)

    print("=== PRIMARY (frozen rule, evaluated at n=5) ===")
    if n5 is None:
        print("  incomplete: rerun to finish the remaining seeds.")
    else:
        print(f"  n=3 (frozen): delta = {n3['delta']:+.4f}")
        print(f"  n=5:          delta = {n5['delta']:+.4f} "
              f"({n5['identity_mean_asr']:.4f} -> {n5['kappa2_mean_asr']:.4f})")
        print(f"  {n5['verdict']}")
        if n3 and (abs(n3["delta"]) < REPLICATE_WITHIN) != (abs(n5["delta"]) < REPLICATE_WITHIN):
            print("  *** THE LABEL FLIPPED between n=3 and n=5. The amendment commits to reporting")
            print("      this as a flip, in the same sentence as the count. The margin does not move.")
        if not n5["accuracy_gate_ok"]:
            print(f"  ACCURACY GATE FAILED (a rung below {ACC_FLOOR}): the cell is uninterpretable "
                  "at n=5 and the n=3 verdict is not superseded.")
    if data["frozen_artifact_md5"]["at_exit"] != frozen_md5:
        sys.exit("ABORT: results/dose_femnist/summary.json changed during this run. It is frozen.")
    print(f"\n  frozen artifact md5 unchanged: {frozen_md5}")
    print(f"  Wall time: {(time.time() - t0) / 3600:.2f} h\n  Saved to {os.path.relpath(OUT, base)}")
    print("\n  Secondary JT trend test: python3 experiments/analyze_dose_femnist.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
