"""
Mode M: coordinate masking into krum -- a SECOND transformation class (Round 34, item B).

WHY THIS EXISTS. Every upstream transformation the paper has run -- Round 11's dose_kappa<K>, Round
12's Mode S and Mode A -- is a positive per-client rescaling, u_i -> c_i * u_i with c_i > 0. So a
reader can grant the entire result and still object that the (P3)=/=>(P4) non-implication is an
artifact of scalar multiplication. No existing arm can answer that, because the whole apparatus,
including the bounded-reweighting theorem and the invariance proposition, is built on that family.

Coordinate masking leaves the family. It is not of the form c_i * u_i for any scalar c_i -- it is not
even a linear map with a client-independent matrix, since each benign client draws its own mask -- so
the theorem and the proposition DO NOT COVER IT. That is the design. If the flagship negative
replicates here, the non-implication is a statement about statistic preservation rather than about
rescaling. If it does not, the paper's central claim has a scope condition it did not know about, and
that is what gets reported, in the abstract.

WHAT THE PROSPECTIVE CHANNEL MEASUREMENT ESTABLISHED (results/mask_admission.json, no ASR):

  krum decision change   0.000 / 0.133 / 0.400 / 0.467   across m = 0, 0.2, 0.5, 0.8
  krum ADMISSION change  0.000 / 0.000 / 0.000 / 0.000   exactly zero at every rung
  aggregate displacement 0.000 / 0.532 / 0.888 / 1.095

so this arm has the same premise the CIFAR-10 flagship rests on -- the statistic's decision is
disturbed and the admitted adversarial mass is not -- now outside the rescaling family. The
three-branch eligibility rule that assigned it type GENERALIZATION (frozen prediction FLAT) was
written into measure_admission_mask.py BEFORE its numbers were read.

The two transformation classes are NOT ordered by disturbance; they disturb different things. At its
top rung Mode M moves Krum's aggregate FURTHER than Mode S does (1.095 against 0.892) while
disturbing Krum's DECISION LESS (0.467 against 0.733). Replicating on the second class is therefore
informative rather than redundant.

THE BRANCH THAT WOULD REFUTE US, frozen before any run: heavy structural damage to benign updates
could make Krum start selecting the adversary. Then admission moves, the equivalence claim is void at
that rung, and that is a CONFIRMATION of the admission reading and a refutation of nothing -- but it
is also uninformative about the non-implication, and is reported as uninformative rather than as a
win. Only FLAT-with-admission-unchanged generalizes the negative. A rise with admission unchanged
refutes us.

THE ACCURACY GATE IS A LIVE RISK HERE, unlike in any rescaling arm. When Krum selects a masked benign
client the emitted update has a fraction m of its coordinates zeroed and the survivors inflated by
about 1/sqrt(1-m), for 50 consecutive rounds. A low ASR at m=0.8 under a failed accuracy gate is a
COLLAPSED MODEL, not preserved suppression. If the top rung fails ACC_FLOOR the equivalence verdict is
read at m=0.5 instead and the substitution is disclosed as a substitution -- frozen in advance
precisely so it cannot look like a choice made after seeing the numbers.

THE SCORE-ONLY MASK CONTROL is not optional: without it this arm would be less well instrumented than
the arm it generalizes. score_only=True makes Krum SELECT on the masked stack and EMIT the selected
client's ORIGINAL, unmasked update. Since adversarial updates are unmasked in both conditions, the two
differ only when a benign client is selected, so the control separates the two ways masking can act:
through WHICH client Krum picks, and through the DAMAGE to what it then emits.

THE IDENTITY RUNG IS NOT RE-RUN WHERE A BIT-IDENTICAL RUN EXISTS. At m=0 apply_d1_transform returns
the update list unwrapped and run_one's participant RNG stream does not depend on d1's name, so
m=0 IS the Mode-S kappa=0 run and Krum standalone. It is imported from results/dose_response/ for
seeds 42-46 and from results/dose_seed_topup/ for seeds 47-51, with per-seed provenance recorded.
--harness-check computes it in-suite at a seed where the imported value exists and demands
bit-equality; if that fails the suite does not run.

Config identical to the rest of the paper: N=10, K=5, f=0.2, alpha=0.5, 50 rounds, cifar_cnn.
New runs: main ladder 3 rungs x 10 seeds = 30, score-only control 3 x 10 = 30, total 60.

Output: results/dose_mask/summary.json (resumable; written after every run).

DO NOT RUN until experiments/pre_registration_dose_mask.md is git-committed and PREREG_COMMIT below is
set to that hash. The script refuses to start otherwise.

  python3 experiments/run_dose_mask.py --harness-check
  python3 experiments/run_dose_mask.py
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
# Single-sourced from the frozen suite so the mask arm and the rescaling arms cannot drift apart in
# the runner, the rung naming, the cell-key format, the accuracy floor or the equivalence margin.
from experiments.run_targeted_dose import (run_one, d1_name, dial, cell_key, ACC_FLOOR,
                                           EQUIV_MARGIN, ATTACK_MAP)

# experiments/pre_registration_dose_mask.md, committed before results/dose_mask/ existed.
PREREG_COMMIT = "684b31e"

MODE, D2, ATTACK = "M", "krum", "committed_scaling"
DROPS = [0.0, 0.2, 0.5, 0.8]              # frozen rung grid; identical to measure_admission_mask.py
NEW_DROPS = [d for d in DROPS if d != 0.0]
SEEDS10 = list(range(42, 52))             # 42-51, frozen in the pre-registration
CONTRAST = (0.0, 0.8)                     # primary equivalence contrast; 0.5 only if 0.8 fails ACC
FALLBACK_CONTRAST = (0.0, 0.5)

out_dir = os.path.join(base, "results", "dose_mask")
out_path = os.path.join(out_dir, "summary.json")
MASK_ADM = os.path.join(base, "results", "mask_admission.json")
TARGETED = os.path.join(base, "results", "targeted_dose", "summary.json")
TOPUP = os.path.join(base, "results", "dose_seed_topup", "summary.json")


def identity_rung():
    """{seed: (accuracy, asr, source)} for the m=0 rung, assembled from bit-identical published runs.

    m=0 returns the update list unwrapped, so it is the Mode-S kappa=0 run, which is itself the
    Round-11 dose_kappa0.0 run. Two files can supply it: results/targeted_dose/ carries the imported
    seeds 42-46 and results/dose_seed_topup/ carries the computed seeds 47-61. Both are read
    read-only and neither is written by this suite.
    """
    out = {}
    for path, tag in ((TARGETED, "results/targeted_dose kappa=0"),
                      (TOPUP, "results/dose_seed_topup kappa=0")):
        if not os.path.exists(path):
            continue
        for c in json.load(open(path)).get("cells", {}).values():
            if (c.get("mode") == "S" and c.get("d2") == D2 and c.get("attack") == ATTACK
                    and abs(c.get("rung", 1e9)) < 1e-12):
                for r in c["per_seed"]:
                    out.setdefault(int(r["seed"]),
                                   (float(r["accuracy"]), float(r["asr"]),
                                    f"{tag} (identical computation)"))
    return out


def premise():
    """The channel measurement this arm's frozen prediction rests on. Read-only, never recomputed."""
    if not os.path.exists(MASK_ADM):
        return {}
    d = json.load(open(MASK_ADM))
    s = d.get("summary", {})
    return {
        "source": "results/mask_admission.json",
        "decision_change": {str(m): s.get(f"doseM|{D2}|{m}|decision") for m in DROPS},
        "admission_change": {str(m): s.get(f"doseM|{D2}|{m}|admission") for m in DROPS},
        "agg_displacement": {str(m): s.get(f"doseM|{D2}|{m}|agg_disp") for m in DROPS},
        "arm_type": d.get("eligibility", {}).get("arm_type"),
        "c_adv_max_dev_from_1": {k: v.get("max_dev_from_1")
                                 for k, v in d.get("c_adv_mode_m", {}).items()},
        "benign_norm_worst_dev_float64": d.get("construction_check_float64", {})
                                          .get("worst_benign_abs_dev_from_1"),
        "adversarial_object_identity": d.get("construction_check_float64", {})
                                        .get("adversarial_object_identity_and_exact_ratio"),
        "two_class_comparison": d.get("two_class_comparison", {}),
        "n_degenerate_rung_rounds": d.get("n_degenerate_rung_rounds"),
    }


def check_frozen():
    prereg = os.path.join(base, "experiments", "pre_registration_dose_mask.md")
    if not os.path.exists(prereg):
        sys.exit(f"REFUSING TO RUN: {prereg} does not exist.")
    if PREREG_COMMIT is None:
        sys.exit("REFUSING TO RUN: the FLAT prediction and the refuting branch are not frozen.\n"
                 f"  1. git commit {prereg} together with results/mask_admission.json\n"
                 "  2. set PREREG_COMMIT here to that hash.\n"
                 "An unfrozen run makes the prediction unfalsifiable, which is the entire point.")
    if not os.path.exists(MASK_ADM):
        sys.exit(f"REFUSING TO RUN: {MASK_ADM} does not exist. This arm's type and its FLAT "
                 "prediction were assigned by that prospective channel measurement; without it the "
                 "prediction has no basis.")
    p = premise()
    if p.get("arm_type") != "GENERALIZATION":
        sys.exit(f"REFUSING TO RUN: {MASK_ADM} reports arm_type={p.get('arm_type')!r}, not "
                 "'GENERALIZATION'. The frozen prediction below is only defined for that branch; "
                 "re-read the pre-registration's three-way rule before running anything.")
    if not identity_rung():
        sys.exit("REFUSING TO RUN: no published m=0 (Mode-S kappa=0) rung to import. Run "
                 "experiments/run_dose_seed_topup.py first, or check results/targeted_dose/.")


def harness_check():
    """Computing m=0 in-suite must reproduce the IMPORTED m=0 value bit-identically.

    m=0 returns the update list unwrapped, so doseM_m0.0 -> krum is Krum alone and is the same
    computation as doseS_kappa0.0 -> krum. That is the claim that lets the identity rung be imported
    across two suites, and it is asserted rather than assumed. A seed with an existing imported value
    is used deliberately: at a seed without one the check would be vacuous.
    """
    rows = identity_rung()
    seed = sorted(rows)[0]
    r_acc, r_asr, src = rows[seed]
    print("=== HARNESS CHECK: doseM_m0.0 must BE the imported kappa=0 run, bit-identically ===")
    print(f"    cifar10/cifar_cnn, {D2}/{ATTACK.replace('committed_', '')} at seed {seed}")
    print(f"    imported ({src}): acc={r_acc:.6f} ASR={r_asr:.6f}")
    print("    If this fails, the m=0 rung is not the same run as kappa=0 and must be re-run "
          "in-suite at every seed.\n", flush=True)
    ok = True
    # BOTH ladders import the same identity rung, so both claims are checked. The score-only one
    # holds for the same reason: at m=0 the scoring stack and the raw stack are the same object, so
    # Krum scores on the raw stack and emits the raw selected update, which is Krum alone. That
    # argument is the one run_score_only_control.py verified for Mode S at 35788d9 and it does not
    # depend on the mode -- but it costs one run to assert it here rather than cite it.
    for score_only in (False, True):
        t = time.time()
        acc, asr = run_one(seed, MODE, D2, ATTACK, 0.0, score_only=score_only)
        d = (acc - r_acc, asr - r_asr)
        ok = ok and abs(d[0]) < 1e-9 and abs(d[1]) < 1e-9
        print(f"  in-suite, score_only={str(score_only):5s}: acc={acc:.6f} ASR={asr:.6f}   "
              f"d=({d[0]:+.2e}, {d[1]:+.2e})  ({time.time() - t:.0f}s)", flush=True)
    print(f"\n  {'BIT-IDENTICAL on both ladders. The m=0 import is valid.' if ok else 'NOT IDENTICAL: the m=0 import claim is FALSE and the identity rung must be re-run in-suite.'}")
    return 0 if ok else 1


def load_cells():
    if not os.path.exists(out_path):
        return {}
    try:
        return json.load(open(out_path)).get("cells", {})
    except Exception:
        return {}


def mean_of(cell, field):
    v = [r[field] for r in cell["per_seed"]]
    return float(np.mean(v)) if v else float("nan")


def save(cells):
    json.dump({"description": "Mode M (coordinate masking) into krum: a second, non-rescaling "
                              "transformation class. Adversarial updates pass through "
                              "BIT-IDENTICALLY; benign updates lose a Bernoulli(m) fraction of "
                              "coordinates and are renormalized to their own original L2 norm. NOT "
                              "of the form c_i*u_i, so the bounded-reweighting theorem and the "
                              "invariance proposition do not cover it. Prediction FLAT, frozen at "
                              f"{PREREG_COMMIT} (experiments/pre_registration_dose_mask.md).",
               "prereg_commit": PREREG_COMMIT,
               "dataset": "cifar10", "model": "cifar_cnn",
               "config": {"N": 10, "K": 5, "f": 0.2, "alpha": 0.5, "rounds": 50,
                          "drops": DROPS, "seeds": SEEDS10,
                          "acc_floor": ACC_FLOOR, "equiv_margin": EQUIV_MARGIN,
                          "primary_contrast": list(CONTRAST),
                          "fallback_contrast_if_acc_floor_fails": list(FALLBACK_CONTRAST)},
               "arm": {"mode": MODE, "d2": D2, "attack": ATTACK,
                       "attack_impl": ATTACK_MAP[ATTACK], "arm_type": "GENERALIZATION",
                       "frozen_prediction": "FLAT"},
               "not_a_defense": "doseM_m<m> reads adversary identity: it masks benign clients and "
                                "leaves adversarial ones untouched. It is an instrument for causal "
                                "identification, never a proposed defense.",
               "theory_scope": "thm:bounded_reweight and prop:invariance make NO prediction for "
                               "Mode M, by construction. No result here is a test of either.",
               "frozen_premise_channels": premise(),
               "cells": cells}, open(out_path, "w"), indent=2)


def run_ladder(cells, score_only, total, done, t0):
    """One ladder: the main arm (score_only=False) or the score-only control (score_only=True)."""
    tag = "score_only" if score_only else "full"
    ident = identity_rung()
    for val in DROPS:
        key = cell_key(MODE, D2, ATTACK, val) + ("|score_only" if score_only else "")
        existing = {r["seed"]: r for r in cells.get(key, {}).get("per_seed", [])}
        for seed in SEEDS10:
            if seed in existing:
                done += 1
                continue
            if val == 0.0 and seed in ident:
                # Imported, not run: m=0 is the unwrapped update list, so both ladders' identity
                # rung IS the published kappa=0 run. score_only has nothing to split at m=0.
                acc, asr, src = ident[seed]
                existing[seed] = {"seed": int(seed), "accuracy": acc, "asr": asr, "source": src}
                done += 1
            else:
                t = time.time()
                acc, asr = run_one(seed, MODE, D2, ATTACK, val, score_only=score_only)
                existing[seed] = {"seed": int(seed), "accuracy": float(acc), "asr": float(asr),
                                  "source": "<computed here>"}
                done += 1
                print(f"  [{done}/{total}] {d1_name(MODE, val):14s} {tag:10s} s{seed}: "
                      f"acc={acc:.4f} ASR={asr:.4f} ({time.time() - t:.0f}s)"
                      + ("  * below acc floor" if acc < ACC_FLOOR else ""), flush=True)
            cells[key] = {"mode": MODE, "d2": D2, "attack": ATTACK, "rung": float(val),
                          "dial": dial(MODE, val), "score_only": bool(score_only),
                          "per_seed": [existing[s] for s in sorted(existing)]}
            save(cells)
    return done


def report(cells):
    """Print the frozen rules scored against the runs. The verdict itself is emitted by
    experiments/analyze_dose_mask.py, which also computes the intervals; this is a running summary so
    a long job is readable while it is still going."""
    print("\n=== MODE M LADDERS (frozen prediction: FLAT) ===")
    for score_only in (False, True):
        tag = "score_only" if score_only else "full"
        print(f"  --- {tag} ---")
        for val in DROPS:
            key = cell_key(MODE, D2, ATTACK, val) + ("|score_only" if score_only else "")
            c = cells.get(key)
            if not c or not c["per_seed"]:
                print(f"  m={val:<4} (no runs yet)")
                continue
            a, k = mean_of(c, "asr"), mean_of(c, "accuracy")
            print(f"  m={val:<4} n={len(c['per_seed']):2d} mean ASR={a:.4f} mean acc={k:.4f}"
                  + ("   * RUNG BELOW ACC FLOOR: a low ASR here is a collapsed model"
                     if k < ACC_FLOOR else ""))
    print("\n  The primary contrast is m=0 -> m=0.8 unless the m=0.8 rung mean accuracy falls below "
          f"ACC_FLOOR={ACC_FLOOR}, in which case it is m=0 -> m=0.5 and the substitution is "
          "disclosed. Equivalence margin +-" f"{EQUIV_MARGIN}, every rung < 0.5.")


def main():
    check_frozen()
    os.makedirs(out_dir, exist_ok=True)
    p = premise()
    total = 2 * len(DROPS) * len(SEEDS10)
    print("=== Mode M: coordinate masking -> krum / scaling (second transformation class) ===")
    print(f"    rungs m={DROPS}, seeds {SEEDS10[0]}-{SEEDS10[-1]}, full + score-only ladders")
    print(f"    {2 * len(NEW_DROPS) * len(SEEDS10)} new runs; the m=0 rung is imported at every seed "
          "that has one")
    print(f"    arm type {p.get('arm_type')}, frozen prediction FLAT, rules frozen at "
          f"{PREREG_COMMIT}")
    print(f"    premise: krum admission change {list(p.get('admission_change', {}).values())} "
          "at decision change "
          f"{[None if v is None else round(v, 3) for v in p.get('decision_change', {}).values()]}")
    print("    A rise WITH admission unchanged refutes the generalization and is reported in the "
          "abstract.\n", flush=True)

    cells = load_cells()
    if cells:
        print(f"  resuming: {sum(len(c['per_seed']) for c in cells.values())} rows already present\n",
              flush=True)
    t0 = time.time(); done = 0
    done = run_ladder(cells, False, total, done, t0)
    done = run_ladder(cells, True, total, done, t0)
    report(cells)
    print(f"\nWall time: {(time.time() - t0) / 3600:.1f} h\nSaved to {out_path}")
    print("Run experiments/analyze_dose_mask.py to score the frozen rules and emit the table row.")
    return 0


if __name__ == "__main__":
    if "--harness-check" in sys.argv:
        check_frozen()
        sys.exit(harness_check())
    sys.exit(main())
