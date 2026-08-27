"""
Second-dataset replication arm for the flagship negative: FEMNIST / krum / model-scaling under Mode S.

WHY THIS ARM, AND WHY IT IS THE ONLY ONE. The paper's central negative lives in one cell: Mode S into
Krum under model-scaling, where the ladder changes Krum's decision in 80% of rounds, changes the
admitted adversarial mass in exactly 0%, and does not move suppression. Everything the paper concludes
about statistic disturbance being causally irrelevant to suppression rests on that cell holding on
CIFAR-10, with cifar_cnn. This arm re-runs the SAME cell on a different dataset AND a different
architecture at once -- EMNIST-byclass (62 classes, 1x28x28) with simple_cnn -- which is the cheapest
honest test of whether the negative is a property of the mechanism or of one dataset/architecture pair.

It is the only eligible cell. A Mode-S arm requires standalone d2 to genuinely suppress the attack at
usable accuracy (ASR < 0.5 at accuracy >= ACC_FLOOR); otherwise there is no suppression to preserve or
lose and the ladder measures nothing. On FEMNIST (results/femnist/payoff_results.json) only
krum/model_scaling clears both gates -- ASR 0.027 at accuracy 0.767. FEMNIST/krum/backdoor_pixel is
0.660 (no suppression), and every other FEMNIST defense either fails to suppress model-scaling or
collapses to ~4% accuracy. That leaves exactly one cell, and it happens to be the flagship one.

THE PREMISE WAS ESTABLISHED PROSPECTIVELY, BEFORE THIS RULE WAS WRITTEN.
experiments/measure_admission_femnist.py measured the channels on FEMNIST with no ASR anywhere in
sight, and found the same premise the CIFAR-10 arm rests on (results/femnist_admission.json):

  Mode S krum, FEMNIST:   decision change 0.000 / 0.643 / 0.714 / 0.643   admission change 0.000 at
                          every rung; c_adv exactly 1.0 at every rung (max deviation 0.00e+00)
  Mode S krum, CIFAR-10:  decision change 0.000 / 0.533 / 0.800 / 0.733   admission change 0.000

So the ladder does disturb Krum's decision on FEMNIST and does not touch the admitted adversarial
mass. Had the decision change come out near zero, this arm would have had no disturbance to test --
the cos_krum failure mode -- and it would have been reported as ineligible instead of run.

FROZEN DECISION RULES (verbatim from experiments/pre_registration_dose_femnist.md).

  Primary, on Delta = mean ASR(kappa=2) - mean ASR(kappa=0), n=3, against the CIFAR-10 krum arm's
  published Delta = -0.026 and the suite's existing EQUIV_MARGIN = 0.15 (no new constant):

    |Delta| < 0.15     FLAGSHIP NEGATIVE REPLICATED on a second dataset and architecture. Statistic
                       disturbance is causally irrelevant to suppression here too.
    Delta > +0.15      THE NEGATIVE IS DATASET- OR ARCHITECTURE-SPECIFIC. Statistic disturbance does
                       move suppression on FEMNIST; the paper's central claim is scoped to CIFAR-10
                       and must say so in the body.
    Delta < -0.15      Attenuation-side fall. INDETERMINATE, reported as such and NOT scored in our
                       favour.

  Secondary: Jonckheere-Terpstra across all four rungs -- which is why this arm runs 4 rungs and not
  2. A monotone rise refutes the flat prediction with the attenuation channel already closed by Mode
  S's construction (c_adv == 1 exactly), so it cannot be explained away as a payload effect.

  Accuracy gate: every rung must hold mean clean accuracy >= ACC_FLOOR = 0.35, or the cell is
  uninterpretable and no verdict stands.

  Harness sanity, NOT a hypothesis test: the kappa=0 rung is identity-then-krum, i.e. krum alone, so
  it should land near the FEMNIST payoff matrix's krum/model_scaling figure. That figure is read from
  results/femnist/payoff_results.json at run time, never transcribed. A large discrepancy means the
  two harnesses disagree and the arm is void rather than interesting.

  n=3, declared up front, not chosen after seeing anything: the FEMNIST payoff matrix ran 3 trials
  (experiments/run_femnist.py) and 4 rungs x 3 seeds x ~55 min is already ~11 h. Disclosed in the
  paper next to the result.

NO IDENTITY RUNG IS IMPORTED. run_dose_replication.py could import kappa=0 from the Round-11 CIFAR-10
ladder because that ladder exists; there is no FEMNIST Round-11 ladder, so kappa=0 is run in-suite.
That is 12 new runs, not 9.

THE ARM IS AN INSTRUMENT, NOT A DEFENSE. Mode S reads adversary identity to pin c_adv at 1; no
deployable defense knows which clients are adversarial. Nothing here is proposed for deployment.

run_one is IMPORTED from run_targeted_dose, not copied -- it takes dataset/model parameters whose
defaults are the frozen CIFAR-10 configuration, so this suite and the frozen one cannot drift apart in
what they compute. Config otherwise identical to the rest of the paper: N=10, K=5, f=0.2, alpha=0.5,
50 rounds.

Output: results/dose_femnist/summary.json (resumable; written after every run).

DO NOT RUN until experiments/pre_registration_dose_femnist.md is git-committed and PREREG_COMMIT below
is set to that hash. The script refuses to start otherwise.

  python3 experiments/run_dose_femnist.py --harness-check
  python3 experiments/run_dose_femnist.py
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
# Single-sourced from the frozen Round-12 suite: the same run_one, the same rung naming, the same
# gates and constants. Only dataset/model differ, and they differ by argument rather than by a fork.
from experiments.run_targeted_dose import (
    run_one, cell_key, dial, d1_name, KAPPAS, ACC_FLOOR, EQUIV_MARGIN, TOL,
)

# experiments/pre_registration_dose_femnist.md, committed before results/dose_femnist/ existed.
PREREG_COMMIT = "478b555"

DATASET, MODEL = "femnist", "simple_cnn"
SEEDS3 = [42, 43, 44]          # the FEMNIST payoff matrix's 3 trials; frozen before the run
ARM = ("krum", "committed_scaling", "c) not invariant",
       "FLAT if the CIFAR-10 negative is a property of the mechanism; RISING if it was "
       "dataset- or architecture-specific", SEEDS3)

# The CIFAR-10 comparison. Its rise is COMPUTED from the frozen results file below, not written down.
CIFAR_ARM = ("krum", "committed_scaling")
REPLICATE_WITHIN = EQUIV_MARGIN   # 0.15, the suite's existing equivalence margin -- no new constant

out_dir = os.path.join(base, "results", "dose_femnist")
out_path = os.path.join(out_dir, "summary.json")
TARGETED = os.path.join(base, "results", "targeted_dose", "summary.json")
PAYOFF = os.path.join(base, "results", "femnist", "payoff_results.json")
ADMISSION = os.path.join(base, "results", "femnist_admission.json")


def mean_asr(cell):
    """Mean ASR of a cell recomputed from its per-seed rows, never read from a stored aggregate."""
    rows = cell.get("per_seed", [])
    return float(np.mean([r["asr"] for r in rows])) if rows else float("nan")


def mean_acc(cell):
    rows = cell.get("per_seed", [])
    return float(np.mean([r["accuracy"] for r in rows])) if rows else float("nan")


def cifar_rise():
    """mean ASR(kappa=2) - mean ASR(kappa=0) for the CIFAR-10 Mode-S krum arm, from its per-seed rows.

    Read-only on the frozen suite. This is the number this arm replicates or refutes, so it is
    recomputed at run time rather than quoted, which is also how the pre-registration states it.
    """
    if not os.path.exists(TARGETED):
        return float("nan")
    cells = json.load(open(TARGETED)).get("cells", {})
    d2, atk = CIFAR_ARM
    lo, hi = cells.get(cell_key("S", d2, atk, 0.0)), cells.get(cell_key("S", d2, atk, 2.0))
    if lo is None or hi is None:
        return float("nan")
    return mean_asr(hi) - mean_asr(lo)


def payoff_standalone():
    """(accuracy, ASR) for standalone krum/model_scaling on FEMNIST, from the payoff matrix."""
    if not os.path.exists(PAYOFF):
        return (float("nan"), float("nan"))
    d = json.load(open(PAYOFF))
    for container in (d, d.get("results", {}) if isinstance(d.get("results"), dict) else {}):
        c = container.get("model_scaling_krum") if isinstance(container, dict) else None
        if isinstance(c, dict):
            return (float(c.get("accuracy", float("nan"))),
                    float(c.get("attack_success_rate", float("nan"))))
    return (float("nan"), float("nan"))


def premise():
    """The prospectively measured FEMNIST channels this arm's interpretability rests on."""
    if not os.path.exists(ADMISSION):
        return None
    s = json.load(open(ADMISSION)).get("summary", {})
    d2 = ARM[0]
    return {str(k): {"decision": s.get(f"doseS|{d2}|{k}|decision"),
                     "admission": s.get(f"doseS|{d2}|{k}|admission")} for k in KAPPAS}


def load_cells():
    if not os.path.exists(out_path):
        return {}
    try:
        return json.load(open(out_path)).get("cells", {})
    except Exception:
        return {}


def save(cells):
    d2, atk, cls, ordering, seeds = ARM
    json.dump({"description": "Second-dataset replication arm for the flagship negative: "
                              "FEMNIST/simple_cnn, krum under Mode S (statistic-only, adversary "
                              f"pinned at c=1). Rules frozen at {PREREG_COMMIT} "
                              "(experiments/pre_registration_dose_femnist.md).",
               "prereg_commit": PREREG_COMMIT,
               "dataset": DATASET, "model": MODEL,
               "config": {"N": 10, "K": 5, "f": 0.2, "alpha": 0.5, "rounds": 50,
                          "kappas": KAPPAS, "seeds": seeds,
                          "rhos": {str(k): dial("S", k) for k in KAPPAS},
                          "acc_floor": ACC_FLOOR, "equiv_margin": EQUIV_MARGIN,
                          "replicate_within": REPLICATE_WITHIN},
               "arm": {"mode": "S", "d2": d2, "attack": atk, "prop1_class": cls,
                       "predicted_outcome": ordering, "seeds": seeds},
               "cifar10_comparison": {"d2": CIFAR_ARM[0], "attack": CIFAR_ARM[1],
                                      "published_mode_s_rise": cifar_rise()},
               "femnist_standalone_payoff": dict(zip(("accuracy", "asr"), payoff_standalone())),
               "prospective_premise_femnist": premise(),
               "cells": cells}, open(out_path, "w"), indent=2)


def check_frozen():
    prereg = os.path.join(base, "experiments", "pre_registration_dose_femnist.md")
    if not os.path.exists(prereg):
        sys.exit(f"REFUSING TO RUN: {prereg} does not exist.")
    if PREREG_COMMIT is None:
        sys.exit("REFUSING TO RUN: the predicted outcome is not frozen.\n"
                 f"  1. git commit {prereg}\n"
                 "  2. set PREREG_COMMIT here to that hash.\n"
                 "An unfrozen run makes the prediction unfalsifiable, which is the entire point.")
    if not os.path.exists(ADMISSION):
        sys.exit(f"REFUSING TO RUN: {ADMISSION} does not exist. The arm's premise -- that the Mode-S "
                 "ladder disturbs Krum's decision on FEMNIST without changing admission -- must be "
                 "measured before the ASR run, not assumed.\n"
                 "  python3 experiments/measure_admission_femnist.py")


def harness_check():
    """kappa=0 at one seed must land near standalone krum/model_scaling from the payoff matrix.

    This is a HARNESS check and explicitly not a hypothesis test: doseS_kappa0.0 returns the update
    list unwrapped, so the kappa=0 rung IS krum alone, and if this suite and the payoff matrix
    disagree materially about krum alone on FEMNIST then the two harnesses are not running the same
    protocol and nothing this arm produces can be interpreted. The payoff matrix averages 3 trials at
    its own seeds, so exact agreement is not expected and no tolerance is asserted here -- the number
    is printed for a human to judge, which is what a sanity check is.
    """
    d2, atk, _, _, _ = ARM
    seed = SEEDS3[0]
    p_acc, p_asr = payoff_standalone()
    print("=== HARNESS CHECK: kappa=0 (identity-then-krum) vs the FEMNIST payoff matrix ===")
    print(f"    {DATASET}/{MODEL}, {d2}/{atk.replace('committed_', '')} at seed {seed}")
    print(f"    payoff matrix, 3-trial mean: acc={p_acc:.4f} ASR={p_asr:.4f}\n", flush=True)
    t = time.time()
    acc, asr = run_one(seed, "S", d2, atk, 0.0, dataset=DATASET, model=MODEL)
    print(f"  this suite s{seed}: acc={acc:.4f} ASR={asr:.4f}   "
          f"d=({acc - p_acc:+.4f}, {asr - p_asr:+.4f})  ({time.time() - t:.0f}s)")
    print("\n  Judge by eye: a single seed against a 3-trial mean. A gap of a few points is seed")
    print("  noise; a gap of tens of points means the harnesses disagree and the arm is void.")
    return 0


def verdict(cells):
    """Score the frozen primary rule. Printed here, and recomputed by the analyzer."""
    d2, atk, _, _, _ = ARM
    lo, hi = cells.get(cell_key("S", d2, atk, 0.0)), cells.get(cell_key("S", d2, atk, 2.0))
    if lo is None or hi is None:
        return None
    a0, a2 = mean_asr(lo), mean_asr(hi)
    delta = a2 - a0
    if delta > REPLICATE_WITHIN:
        v = "NEGATIVE IS DATASET- OR ARCHITECTURE-SPECIFIC (statistic disturbance moves suppression)"
    elif delta < -REPLICATE_WITHIN:
        v = "INDETERMINATE (attenuation-side fall; not scored in our favour)"
    else:
        v = "FLAGSHIP NEGATIVE REPLICATED on a second dataset and architecture"
    gates = {str(k): mean_acc(cells[cell_key("S", d2, atk, k)])
             for k in KAPPAS if cell_key("S", d2, atk, k) in cells}
    return {"identity_mean_asr": a0, "kappa2_mean_asr": a2, "delta": delta,
            "replicate_within": REPLICATE_WITHIN, "cifar10_delta": cifar_rise(), "verdict": v,
            "per_rung_mean_acc": gates,
            "accuracy_gate_ok": bool(all(v >= ACC_FLOOR for v in gates.values() if v == v))}


def main():
    check_frozen()
    os.makedirs(out_dir, exist_ok=True)
    if "--harness-check" in sys.argv:
        return harness_check()

    d2, atk, cls, ordering, seeds = ARM
    cells = load_cells()
    todo = [(v, s) for v in KAPPAS for s in seeds
            if not any(r["seed"] == s
                       for r in cells.get(cell_key("S", d2, atk, v), {}).get("per_seed", []))]

    p_acc, p_asr = payoff_standalone()
    print(f"=== DOSE FEMNIST: {len(todo)} new runs (no identity rung to import) ===")
    print(f"    rules frozen in pre_registration_dose_femnist.md @ {PREREG_COMMIT}")
    print(f"    {DATASET}/{MODEL}, mode S {d2}/{atk.replace('committed_', '')}, class {cls}")
    print(f"    kappa {KAPPAS} -> rho {[round(dial('S', k), 2) for k in KAPPAS]}, seeds {seeds}")
    print(f"    standalone krum on FEMNIST (payoff matrix): ASR {p_asr:.3f} @ acc {p_acc:.3f}")
    print(f"    CIFAR-10 Mode-S krum rise = {cifar_rise():+.3f} (recomputed from results/targeted_dose)")
    print(f"    REPLICATED if |delta| < {REPLICATE_WITHIN:.2f}; "
          f"DATASET-SPECIFIC if delta > {REPLICATE_WITHIN:+.2f}")
    pr = premise()
    if pr:
        print("    prospective premise (results/femnist_admission.json), decision/admission by kappa:")
        print("      " + "  ".join(f"{k}:{pr[str(k)]['decision']:.3f}/{pr[str(k)]['admission']:.3f}"
                                   for k in KAPPAS))
    print(flush=True)

    t0, done = time.time(), 0
    for v, seed in todo:
        key = cell_key("S", d2, atk, v)
        cell = cells.setdefault(key, {"mode": "S", "d2": d2, "attack": atk, "rung": v,
                                      "dial": dial("S", v), "prop1_class": cls,
                                      "dataset": DATASET, "model": MODEL,
                                      "predicted_outcome": ordering, "per_seed": []})
        t = time.time()
        acc, asr = run_one(seed, "S", d2, atk, v, dataset=DATASET, model=MODEL)
        cell["per_seed"].append({"seed": seed, "accuracy": acc, "asr": asr})
        cell["per_seed"].sort(key=lambda r: r["seed"])
        cell["mean_asr"], cell["mean_acc"] = mean_asr(cell), mean_acc(cell)
        cell["std_asr"] = float(np.std([r["asr"] for r in cell["per_seed"]], ddof=0))
        cells[key] = cell
        save(cells)
        done += 1
        print(f"  [{done}/{len(todo)}] {DATASET} mode S {d2} kappa={v} s{seed}: "
              f"acc={acc:.3f} ASR={asr:.3f} ({time.time() - t:.0f}s)", flush=True)

    save(cells)
    print("\n=== LADDER (mean ASR @ mean clean accuracy) ===")
    print("  " + " ".join(f"{'kappa=' + str(v):>14s}" for v in KAPPAS))
    row = []
    for v in KAPPAS:
        c = cells.get(cell_key("S", d2, atk, v))
        row.append("na" if c is None else f"{mean_asr(c):.3f}@{mean_acc(c):.2f}"
                   + ("!" if mean_acc(c) < ACC_FLOOR else " "))
    print("  " + " ".join(f"{x:>14s}" for x in row))
    print(f"  '!' = mean clean accuracy < {ACC_FLOOR}: uninterpretable, not suppression.")

    ver = verdict(cells)
    if ver:
        print(f"\n=== PRIMARY (frozen) ===\n  delta = {ver['delta']:+.3f} "
              f"({ver['identity_mean_asr']:.3f} -> {ver['kappa2_mean_asr']:.3f}), "
              f"CIFAR-10 delta = {ver['cifar10_delta']:+.3f}")
        print(f"  {ver['verdict']}")
        if not ver["accuracy_gate_ok"]:
            print(f"  ACCURACY GATE FAILED (some rung below {ACC_FLOOR}): the cell is "
                  "uninterpretable and the verdict does not stand.")
    print("\n  The secondary JT trend test is scored by experiments/analyze_dose_femnist.py")
    print("  against the same frozen rules.")
    print(f"\nWall time: {(time.time() - t0) / 3600:.1f} h\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
