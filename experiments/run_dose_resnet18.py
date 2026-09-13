"""
Second-ARCHITECTURE replication arm for the flagship negative: CIFAR-10 / resnet18 / krum / scaling, Mode S.

WHY THIS ARM AND NOT THE FEMNIST ONE AGAIN. The paper's central negative lives in one cell: Mode S into
Krum under model-scaling, where the ladder changes Krum's decision in most rounds, changes the admitted
adversarial mass in exactly 0%, and does not move suppression. The existing replication
(experiments/run_dose_femnist.py) changed the dataset AND the architecture at once -- EMNIST-byclass with
simple_cnn against CIFAR-10 with cifar_cnn -- which is the cheapest honest test but cannot say which of
the two mattered, and simple_cnn and cifar_cnn are two configurations of the same shallow-convnet family.
This arm holds the dataset, N, K, f, alpha, round count, attack and defense at the frozen CIFAR-10 values
and moves ONLY the architecture, to a different family: an 18-layer residual network with skip
connections and GroupNorm (fl_core/models.py:73-79, the repo's single resnet18 configuration). A flat
result here is attributable to architecture alone.

THE PREMISE WAS ESTABLISHED PROSPECTIVELY, BEFORE THIS RULE WAS WRITTEN.
experiments/measure_admission_resnet18.py measured the channels on ResNet18 with no ASR anywhere in
sight and with no per-rung training, and its verdict is read from results/resnet18_admission.json at run
time rather than transcribed here. The arm is interpretable only if the Mode-S ladder actually disturbs
Krum's decision on this architecture while leaving the admitted adversarial mass alone; had the decision
change come out near zero there would be no disturbance to test, a flat ASR curve would carry no
information, and the arm would be reported as ineligible instead of run. That is the cos_krum failure
mode the paper already discloses.

WHAT THIS ARM HAS THAT THE FEMNIST ARM DID NOT, AND WHAT IT LACKS.
  - It varies one factor. The FEMNIST arm varies two and cannot attribute.
  - It has NO independent cross-check on its own harness. The FEMNIST arm could compare its kappa=0
    rung against a published 3-trial payoff-matrix figure for standalone krum/model_scaling on FEMNIST
    (results/femnist/payoff_results.json). No artifact in results/ contains both resnet18 and krum, so
    there is no such figure here and none is invented. The kappa=0 rung IS the first measurement of
    standalone Krum against model-scaling on this architecture, which is why the gate below is a
    pre-registered admissibility branch rather than a harness-agreement eyeball. This is a genuine
    weakness of this arm relative to its predecessor and is disclosed as one.

FROZEN DECISION RULES (verbatim from experiments/pre_registration_dose_resnet18.md).

  Gate, scored FIRST and on its own: standalone Krum must suppress model-scaling on ResNet18 at usable
  accuracy -- mean ASR at kappa=0 below SUPPRESS_ASR = 0.5 with mean clean accuracy at or above
  ACC_FLOOR = 0.35. If it does not, there is no suppression for the ladder to preserve or lose and the
  arm is VOID, NOT NEGATIVE. A void arm is reported as void and licenses no sentence about
  architecture.

  Primary, on Delta = mean ASR(kappa=2) - mean ASR(kappa=0), n=3, against the frozen CIFAR-10/cifar_cnn
  krum arm's Delta (recomputed at run time from results/targeted_dose) and the suite's existing
  EQUIV_MARGIN = 0.15 (no new constant):

    |Delta| < 0.15     FLAGSHIP NEGATIVE REPLICATED on a second architecture FAMILY. Statistic
                       disturbance is causally irrelevant to suppression here too.
    Delta > +0.15      THE NEGATIVE IS ARCHITECTURE-SPECIFIC. Statistic disturbance does move
                       suppression on ResNet18; the paper's central claim is scoped to the shallow-CNN
                       family and must say so in the body.
    Delta < -0.15      Attenuation-side fall. INDETERMINATE, reported as such and NOT scored in our
                       favour.

  NO SECONDARY TREND TEST. The FEMNIST arm ran four rungs so that a Jonckheere-Terpstra trend test
  could be scored across them. This arm runs the two ENDPOINTS ONLY (kappa=0 and kappa=2), because
  ResNet18 costs 1.63-1.72 h/run at this regime (measured: results/cifar10_mix_ratio_sweep_resnet18
  records wall_time_s 5884-6205 over 15 runs, mean 6039 s, for exactly N=10, K=5, 50 rounds, CIFAR-10,
  resnet18) and four rungs x 3 seeds would be ~20 h. A two-rung ladder has no trend to test and must
  never be displayed or described as a four-rung one. The restriction is printed on every run and
  disclosed in the paper.

  Accuracy gate: every rung must hold mean clean accuracy >= ACC_FLOOR = 0.35, or the cell is
  uninterpretable and no verdict stands. This gate is LIVE, not a formality: in that same sweep 14 of
  15 resnet18 runs reached 0.662-0.766 clean accuracy but the fifteenth collapsed to 0.153, below the
  floor, so this architecture does sometimes collapse at this regime.

  n=3, seeds 42/43/44, declared before the first run and not chosen after seeing anything -- the same
  three seeds the FEMNIST replication froze, so the two replication arms are directly comparable. No
  optional stopping: no interim look, no extension. If interrupted, the n reached is reported.

  BOTH ENDPOINTS AT EQUAL n. Delta is refused rather than printed if the two rungs were scored on
  different seed sets; a top-up that moves only one leg hides a mixed-n comparison in the other.

WHAT THIS ARM DOES NOT LICENSE. It is one cell, one attack, one architecture. It does not widen N, K,
f, alpha, the round count or the attack menu, it says nothing about datasets, and it does not touch the
sign-reversal cell.

THE ARM IS AN INSTRUMENT, NOT A DEFENSE. Mode S reads adversary identity to pin the adversarial
coefficient share at 1; no deployable defense knows which clients are adversarial. Nothing here is
proposed for deployment.

run_one is IMPORTED from run_targeted_dose, not copied -- it takes dataset/model parameters whose
defaults are the frozen CIFAR-10/cifar_cnn configuration, so this suite and the frozen one cannot drift
apart in what they compute. Config otherwise identical to the rest of the paper: N=10, K=5, f=0.2,
alpha=0.5, 50 rounds.

Output: results/dose_resnet18/summary.json (new directory; resumable, written after every run).

DO NOT RUN until experiments/pre_registration_dose_resnet18.md is git-committed and PREREG_COMMIT below
is set to that hash. The script refuses to start otherwise.

  python3 experiments/run_dose_resnet18.py --harness-check   # the kappa=0 gate; its run IS scored
  python3 experiments/run_dose_resnet18.py
"""
import json, os, subprocess, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
# Single-sourced from the frozen Round-12 suite: the same run_one, the same rung naming, the same gates
# and constants. Only dataset/model differ, and they differ by argument rather than by a fork.
from experiments.run_targeted_dose import (
    run_one, cell_key, dial, d1_name, KAPPAS, ACC_FLOOR, EQUIV_MARGIN, TOL,
)

# experiments/pre_registration_dose_resnet18.md, committed before results/dose_resnet18/ existed.
# None means the prediction is not frozen and the script refuses to start; see check_frozen().
# This hash is VERIFIED against git rather than recited: check_frozen() requires the commit to exist,
# to contain the pre-registration, and its blob to equal the working copy byte for byte. A hash that
# nobody checks is a claim, not a freeze, and it would let the rules be edited after the run began.
PREREG_COMMIT = "89046f6"

DATASET, MODEL = "cifar10", "resnet18"
SEEDS3 = [42, 43, 44]                 # the FEMNIST replication's three seeds; frozen before the run

# ENDPOINT RUNGS ONLY. KAPPAS (the frozen four-rung list) is imported and kept so the restriction can
# be stated as a restriction, printed against the full ladder, and audited -- not silently redefined.
KAPPAS_ARM = [0.0, 2.0]
assert all(any(abs(k - f) < TOL for f in KAPPAS) for k in KAPPAS_ARM), \
    "the arm's rungs must be a subset of the frozen ladder, or the cell keys will not resolve"

ARM = ("krum", "committed_scaling", "c) not invariant",
       "FLAT if the CIFAR-10/cifar_cnn negative is a property of the mechanism; RISING if it was "
       "specific to the shallow-CNN family", SEEDS3)

# The comparisons. Both are RECOMPUTED from their frozen files below, never written down here.
CIFAR_ARM = ("krum", "committed_scaling")
REPLICATE_WITHIN = EQUIV_MARGIN       # 0.15, the suite's existing equivalence margin -- no new constant
# The suppression threshold of the gate. Not a new constant either: it is the same 0.5 the FEMNIST
# arm's eligibility rule was written against (run_dose_femnist.py:13, "ASR < 0.5 at accuracy >=
# ACC_FLOOR"), where it decided which FEMNIST cells were admissible at all. Here it is a named
# pre-registered branch instead of prose, because on this architecture nothing on disk answers it.
SUPPRESS_ASR = 0.5

out_dir = os.path.join(base, "results", "dose_resnet18")
out_path = os.path.join(out_dir, "summary.json")
TARGETED = os.path.join(base, "results", "targeted_dose", "summary.json")
FEMNIST = os.path.join(base, "results", "dose_femnist", "summary.json")
ADMISSION = os.path.join(base, "results", "resnet18_admission.json")


def mean_asr(cell):
    """Mean ASR of a cell recomputed from its per-seed rows, never read from a stored aggregate."""
    rows = cell.get("per_seed", [])
    return float(np.mean([r["asr"] for r in rows])) if rows else float("nan")


def mean_acc(cell):
    rows = cell.get("per_seed", [])
    return float(np.mean([r["accuracy"] for r in rows])) if rows else float("nan")


def seeds_of(cell):
    return sorted(r["seed"] for r in (cell or {}).get("per_seed", []))


def arm_delta(cells_path, d2, atk):
    """mean ASR(kappa=2) - mean ASR(kappa=0) for a Mode-S arm in some suite's summary.json.

    Read-only, and BOTH legs are recomputed from their own per-seed rows in the same call, so the
    subtrahend can never be a stale aggregate from a different seed count than the minuend. Returns
    (delta, n_lo, n_hi) and refuses -- delta = nan -- when the two legs were scored on different seed
    sets, because a Delta across unequal seed sets is not a within-arm contrast.
    """
    if not os.path.exists(cells_path):
        return (float("nan"), 0, 0)
    cells = json.load(open(cells_path)).get("cells", {})
    lo, hi = cells.get(cell_key("S", d2, atk, 0.0)), cells.get(cell_key("S", d2, atk, 2.0))
    if lo is None or hi is None:
        return (float("nan"), len(seeds_of(lo)), len(seeds_of(hi)))
    s_lo, s_hi = seeds_of(lo), seeds_of(hi)
    if s_lo != s_hi:
        return (float("nan"), len(s_lo), len(s_hi))
    return (mean_asr(hi) - mean_asr(lo), len(s_lo), len(s_hi))


def premise():
    """The prospectively measured ResNet18 channels this arm's interpretability rests on.

    Reported across the FULL four-rung ladder, because the measurement covers all of it: the arm's
    two-rung restriction is a compute decision about ASR runs and not a claim that the interior
    channels are unmeasured.
    """
    if not os.path.exists(ADMISSION):
        return None
    d = json.load(open(ADMISSION))
    s = d.get("summary", {})
    d2 = ARM[0]
    return {"by_kappa": {str(k): {"decision": s.get(f"doseS|{d2}|{k}|decision"),
                                  "admission": s.get(f"doseS|{d2}|{k}|admission")} for k in KAPPAS},
            "eligibility": d.get("eligibility"),
            "nonfinite_rounds_excluded": len(d.get("nonfinite_rounds_excluded", [])),
            "n_rounds_total": d.get("n_rounds_total")}


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
    json.dump({"description": "Second-ARCHITECTURE replication arm for the flagship negative: "
                              "CIFAR-10/resnet18, krum under Mode S (statistic-only, adversary pinned "
                              "at c=1). Architecture varies alone against the frozen cifar_cnn arm. "
                              f"Rules frozen at {PREREG_COMMIT} "
                              "(experiments/pre_registration_dose_resnet18.md).",
               "prereg_commit": PREREG_COMMIT,
               "dataset": DATASET, "model": MODEL,
               "config": {"N": 10, "K": 5, "f": 0.2, "alpha": 0.5, "rounds": 50,
                          "kappas_run": KAPPAS_ARM, "kappas_frozen_ladder": KAPPAS,
                          "endpoint_only": True,
                          "endpoint_only_reason": "resnet18 costs 1.63-1.72 h/run at this regime; two "
                                                  "rungs x 3 seeds is ~10 h. A two-rung ladder has "
                                                  "no trend to test and is never displayed as four.",
                          "no_secondary_trend_test": True,
                          "seeds": seeds,
                          "rhos": {str(k): dial("S", k) for k in KAPPAS_ARM},
                          "acc_floor": ACC_FLOOR, "suppress_asr": SUPPRESS_ASR,
                          "equiv_margin": EQUIV_MARGIN, "replicate_within": REPLICATE_WITHIN},
               "arm": {"mode": "S", "d2": d2, "attack": atk, "prop1_class": cls,
                       "predicted_outcome": ordering, "seeds": seeds,
                       "varies": "architecture alone"},
               "cifar_cnn_comparison": {"d2": CIFAR_ARM[0], "attack": CIFAR_ARM[1],
                                        "mode_s_delta": c_delta, "n_kappa0": c_lo, "n_kappa2": c_hi},
               "femnist_comparison": {"d2": CIFAR_ARM[0], "attack": CIFAR_ARM[1],
                                      "mode_s_delta": f_delta, "n_kappa0": f_lo, "n_kappa2": f_hi,
                                      "note": "EMNIST-byclass/simple_cnn: a second dataset and a "
                                              "differently configured CNN, not a second family"},
               "standalone_krum_resnet18_prior": None,
               "standalone_krum_resnet18_prior_note":
                   "No artifact in results/ contains both resnet18 and krum, so this arm has no "
                   "independent published figure to cross-check its kappa=0 rung against. That rung "
                   "is the first such measurement and is scored as the pre-registered gate.",
               "prospective_premise_resnet18": premise(),
               "verdict": verdict(cells),
               "cells": cells}, open(out_path, "w"), indent=2)


def check_frozen():
    prereg = os.path.join(base, "experiments", "pre_registration_dose_resnet18.md")
    if not os.path.exists(prereg):
        sys.exit(f"REFUSING TO RUN: {prereg} does not exist.")
    if PREREG_COMMIT is None:
        sys.exit("REFUSING TO RUN: the predicted outcome is not frozen.\n"
                 f"  1. git commit {prereg}\n"
                 "  2. set PREREG_COMMIT here to that hash.\n"
                 "An unfrozen run makes the prediction unfalsifiable, which is the entire point.")

    # THE HASH IS CHECKED, NOT RECITED. Three things must hold, and each has its own failure mode:
    #   (a) the commit resolves -- otherwise the hash is a typo and points at nothing;
    #   (b) the pre-registration exists AT that commit -- otherwise the freeze names a commit that
    #       does not contain the rules;
    #   (c) the committed blob equals the working copy byte for byte -- otherwise the file was edited
    #       after the freeze, which is precisely the move the freeze exists to prevent, and the run
    #       would be scored against rules that are not the ones on record.
    # Read-only: rev-parse and hash-object write nothing. A missing git binary or a non-repository is
    # itself a refusal, because an unverifiable freeze is not a freeze.
    rel = os.path.relpath(prereg, base)
    try:
        head = subprocess.run(["git", "-C", base, "rev-parse", "--verify", f"{PREREG_COMMIT}^{{commit}}"],
                              capture_output=True, text=True)
        if head.returncode != 0:
            sys.exit(f"REFUSING TO RUN: PREREG_COMMIT {PREREG_COMMIT} does not resolve to a commit in "
                     f"{base}. The freeze names nothing.")
        committed = subprocess.run(["git", "-C", base, "rev-parse", f"{PREREG_COMMIT}:{rel}"],
                                   capture_output=True, text=True)
        if committed.returncode != 0:
            sys.exit(f"REFUSING TO RUN: {rel} does not exist at commit {PREREG_COMMIT}. The hash names "
                     "a commit that does not contain the pre-registration.")
        working = subprocess.run(["git", "-C", base, "hash-object", rel],
                                 capture_output=True, text=True)
        if working.returncode != 0:
            sys.exit(f"REFUSING TO RUN: cannot hash {rel} to compare it against the freeze.")
        if committed.stdout.strip() != working.stdout.strip():
            sys.exit(f"REFUSING TO RUN: {rel} has been EDITED since it was frozen at "
                     f"{PREREG_COMMIT}.\n"
                     f"  committed blob: {committed.stdout.strip()[:12]}\n"
                     f"  working blob:   {working.stdout.strip()[:12]}\n"
                     "The rules on disk are not the rules on record. Either revert the file, or commit\n"
                     "the amendment AS an amendment and set PREREG_COMMIT to the new hash -- but a "
                     "post-hoc\n  edit to a decision rule is not an amendment, it is the thing "
                     "pre-registration forbids.")
    except FileNotFoundError:
        sys.exit("REFUSING TO RUN: git is not available, so the freeze cannot be verified. An "
                 "unverifiable freeze is not a freeze.")

    if not os.path.exists(ADMISSION):
        sys.exit(f"REFUSING TO RUN: {ADMISSION} does not exist. The arm's premise -- that the Mode-S "
                 "ladder disturbs Krum's decision on ResNet18 without changing admission -- must be "
                 "measured before the ASR run, not assumed.\n"
                 "  python3 experiments/measure_admission_resnet18.py")
    pr = premise()
    v = ((pr or {}).get("eligibility") or {}).get("verdict", "")
    if not v.startswith("ELIGIBLE"):
        sys.exit("REFUSING TO RUN: the prospective channel measurement did not license this arm.\n"
                 f"  {ADMISSION} says: {v[:200]}\n"
                 "Running anyway would produce an ASR curve nobody can interpret.")


def gate(cells, save_fn):
    """kappa=0 at every arm seed is the pre-registered admissibility branch, and it is a scored rung.

    NOT the FEMNIST arm's harness check. That one compared kappa=0 against a published payoff-matrix
    figure and asserted no tolerance, because its job was to catch two harnesses disagreeing. Here
    nothing on disk measures standalone Krum against model-scaling on ResNet18, so there is nothing to
    agree with; the question this rung answers is the pre-registered one -- does Krum suppress this
    attack on this architecture at usable accuracy at all? -- and its answer decides whether the arm
    is scoreable or VOID.

    Its runs are STORED, not thrown away: doseS_kappa0.0 hands the update list to krum unwrapped, so
    the kappa=0 rung IS krum alone, and run_one reseeds torch and numpy from its arguments, so the
    run is the same computation whether it was invoked here or from the main loop. The main loop's
    resume logic then skips what this scored.
    """
    d2, atk, cls, ordering, seeds = ARM
    print("=== GATE (pre-registered, scored first): does krum alone suppress scaling on ResNet18? ===")
    print(f"    {DATASET}/{MODEL}, {d2}/{atk.replace('committed_', '')} at kappa=0, seeds {seeds}")
    print(f"    VOID unless mean ASR < {SUPPRESS_ASR} at mean clean accuracy >= {ACC_FLOOR}.")
    print("    No prior resnet18/krum figure exists on disk; this rung is the first measurement, so")
    print("    it is scored against the frozen threshold and not eyeballed against a published mean.")
    print(f"    Runs are stored to {out_path} and the main loop will not repeat them.\n", flush=True)
    run_rungs(cells, [(0.0, s) for s in seeds], save_fn)
    return report_gate(cells)


def report_gate(cells):
    d2, atk, _, _, _ = ARM
    lo = cells.get(cell_key("S", d2, atk, 0.0))
    if lo is None or not lo.get("per_seed"):
        print("  gate not scoreable: no kappa=0 rows.")
        return None
    a, acc, n = mean_asr(lo), mean_acc(lo), len(lo["per_seed"])
    ok = bool(a < SUPPRESS_ASR and acc >= ACC_FLOOR)
    print(f"\n  kappa=0 (krum alone): mean ASR {a:.3f} @ mean acc {acc:.3f}, n={n}")
    if ok:
        print(f"  GATE PASSED: krum suppresses model-scaling on ResNet18 (ASR {a:.3f} < "
              f"{SUPPRESS_ASR}) at usable accuracy. There is suppression for the ladder to preserve "
              "or lose, so the primary rule is scoreable.")
    elif acc < ACC_FLOOR:
        print(f"  GATE FAILED, accuracy: mean clean accuracy {acc:.3f} < {ACC_FLOOR}. A low ASR on a "
              "collapsed model is not suppression. THE ARM IS VOID, NOT NEGATIVE.")
    else:
        print(f"  GATE FAILED, no suppression: mean ASR {a:.3f} >= {SUPPRESS_ASR}. Krum does not "
              "suppress model-scaling on this architecture, so there is nothing for the Mode-S "
              "ladder to preserve or lose. THE ARM IS VOID, NOT NEGATIVE, and it licenses no "
              "sentence about architecture.")
    return {"kappa0_mean_asr": a, "kappa0_mean_acc": acc, "n": n, "passed": ok,
            "suppress_asr": SUPPRESS_ASR, "acc_floor": ACC_FLOOR}


def run_rungs(cells, todo, save_fn):
    d2, atk, cls, ordering, _ = ARM
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
        cell["n"] = len(cell["per_seed"])
        cells[key] = cell
        save_fn(cells)
        done += 1
        # [done/total] counts THIS invocation's work only. It is not progress through the arm: a
        # resumed invocation has a smaller total because finished runs are not in its todo list.
        # Count per_seed rows in the artifact to judge how much of the arm exists.
        print(f"  [{done}/{len(todo)} this invocation] {MODEL} mode S {d2} kappa={v} s{seed}: "
              f"acc={acc:.3f} ASR={asr:.3f} ({time.time() - t:.0f}s)", flush=True)
    if todo:
        print(f"  ({(time.time() - t0) / 3600:.2f} h this invocation)")
    return done


def verdict(cells):
    """Score the frozen primary rule. Printed here, and recomputed by the analyzer.

    Returns None unless BOTH endpoints exist on the SAME seed set. A Delta whose two legs were scored
    on different seeds is not a within-arm contrast and is refused rather than reported with a caveat.
    """
    d2, atk, _, _, _ = ARM
    lo, hi = cells.get(cell_key("S", d2, atk, 0.0)), cells.get(cell_key("S", d2, atk, 2.0))
    if lo is None or hi is None:
        return None
    s_lo, s_hi = seeds_of(lo), seeds_of(hi)
    if s_lo != s_hi:
        return {"delta": None, "n_kappa0": len(s_lo), "n_kappa2": len(s_hi),
                "seeds_kappa0": s_lo, "seeds_kappa2": s_hi,
                "verdict": "NOT SCOREABLE: the two endpoints were scored on different seed sets, so "
                           "their difference mixes n. Finish the shorter leg on the same seeds."}
    a0, a2 = mean_asr(lo), mean_asr(hi)
    delta = a2 - a0
    if delta > REPLICATE_WITHIN:
        v = "NEGATIVE IS ARCHITECTURE-SPECIFIC (statistic disturbance moves suppression on ResNet18)"
    elif delta < -REPLICATE_WITHIN:
        v = "INDETERMINATE (attenuation-side fall; not scored in our favour)"
    else:
        v = "FLAGSHIP NEGATIVE REPLICATED on a second architecture family"
    acc0, acc2 = mean_acc(lo), mean_acc(hi)
    gate_ok = bool(a0 < SUPPRESS_ASR and acc0 >= ACC_FLOOR)
    c_delta, c_lo, c_hi = arm_delta(TARGETED, *CIFAR_ARM)
    return {"identity_mean_asr": a0, "kappa2_mean_asr": a2, "delta": delta,
            "n_kappa0": len(s_lo), "n_kappa2": len(s_hi), "seeds": s_lo,
            "replicate_within": REPLICATE_WITHIN,
            "cifar_cnn_delta": c_delta, "cifar_cnn_n": [c_lo, c_hi],
            "verdict": v if gate_ok else
                       "VOID, NOT NEGATIVE: the kappa=0 gate failed, so there was no suppression for "
                       "the ladder to preserve. The primary rule does not apply and the Delta below "
                       "is reported for completeness only.",
            "gate_passed": gate_ok,
            "per_rung_mean_acc": {"0.0": acc0, "2.0": acc2},
            "accuracy_gate_ok": bool(acc0 >= ACC_FLOOR and acc2 >= ACC_FLOOR),
            "endpoint_only": True, "no_secondary_trend_test": True,
            "rungs_run": KAPPAS_ARM, "rungs_in_frozen_ladder": KAPPAS}


def main():
    check_frozen()
    os.makedirs(out_dir, exist_ok=True)
    cells = load_cells()

    d2, atk, cls, ordering, seeds = ARM
    c_delta, c_lo, c_hi = arm_delta(TARGETED, *CIFAR_ARM)
    f_delta, f_lo, f_hi = arm_delta(FEMNIST, *CIFAR_ARM)
    pr = premise()

    print(f"=== DOSE ResNet18: second-ARCHITECTURE arm, architecture varies alone ===")
    print(f"    rules frozen in pre_registration_dose_resnet18.md @ {PREREG_COMMIT}")
    print(f"    {DATASET}/{MODEL}, mode S {d2}/{atk.replace('committed_', '')}, class {cls}")
    print(f"    ENDPOINT RUNGS ONLY: kappa {KAPPAS_ARM} of the frozen ladder {KAPPAS} "
          f"-> rho {[round(dial('S', k), 2) for k in KAPPAS_ARM]}")
    print( "    The interior rungs are NOT run in this arm and there is therefore NO trend test.")
    print(f"    seeds {seeds} (n={len(seeds)}), no optional stopping")
    print(f"    REPLICATED if |delta| < {REPLICATE_WITHIN:.2f}; "
          f"ARCHITECTURE-SPECIFIC if delta > {REPLICATE_WITHIN:+.2f}; "
          f"VOID if kappa=0 ASR >= {SUPPRESS_ASR}")
    print(f"    cifar_cnn Mode-S krum delta = {c_delta:+.3f} (n={c_lo}/{c_hi}, recomputed from "
          "results/targeted_dose)")
    print(f"    EMNIST/simple_cnn delta     = {f_delta:+.3f} (n={f_lo}/{f_hi}, recomputed from "
          "results/dose_femnist)")
    if pr:
        el = (pr.get("eligibility") or {}).get("verdict", "")
        print("    prospective premise (results/resnet18_admission.json), decision/admission by kappa:")
        print("      " + "  ".join(
            f"{k}:{pr['by_kappa'][str(k)]['decision']:.3f}/{pr['by_kappa'][str(k)]['admission']:.3f}"
            for k in KAPPAS))
        print(f"      {el[:110]}")
        print(f"      non-finite measurement rounds excluded: "
              f"{pr['nonfinite_rounds_excluded']}/{pr['n_rounds_total']}")
    print(flush=True)

    gate_state = gate(cells, save) if not any(
        r["seed"] in seeds for r in cells.get(cell_key("S", d2, atk, 0.0), {}).get("per_seed", [])
    ) else report_gate(cells)

    if "--harness-check" in sys.argv:
        print("\n  Gate only (--harness-check). Re-run without the flag to score kappa=2.")
        save(cells)
        return 0
    if gate_state is not None and not gate_state["passed"]:
        print("\n  REFUSING to run kappa=2: the gate failed, so the ladder has no suppression to")
        print("  preserve and a kappa=2 rung would answer no pre-registered question. THE ARM IS")
        print("  VOID. Report it as void; do not report it as a negative.")
        save(cells)
        return 0

    todo = [(v, s) for v in KAPPAS_ARM for s in seeds
            if not any(r["seed"] == s
                       for r in cells.get(cell_key("S", d2, atk, v), {}).get("per_seed", []))]
    print(f"\n=== {len(todo)} run(s) remaining in this arm "
          f"(~{1.68 * len(todo):.1f} h at the measured mean of 1.68 h/run) ===", flush=True)
    run_rungs(cells, todo, save)
    save(cells)

    print("\n=== LADDER, TWO ENDPOINTS ONLY (mean ASR @ mean clean accuracy, n) ===")
    print("  " + " ".join(f"{'kappa=' + str(v):>18s}" for v in KAPPAS_ARM))
    row = []
    for v in KAPPAS_ARM:
        c = cells.get(cell_key("S", d2, atk, v))
        row.append("na" if c is None else
                   f"{mean_asr(c):.3f}@{mean_acc(c):.2f},n={len(c['per_seed'])}"
                   + ("!" if mean_acc(c) < ACC_FLOOR else ""))
    print("  " + " ".join(f"{x:>18s}" for x in row))
    print(f"  '!' = mean clean accuracy < {ACC_FLOOR}: uninterpretable, not suppression.")
    print("  Two rungs. There is no interior and no trend test; do not draw this as a ladder.")

    ver = verdict(cells)
    if ver and ver.get("delta") is None:
        print(f"\n=== PRIMARY (frozen) ===\n  {ver['verdict']}")
    elif ver:
        print(f"\n=== PRIMARY (frozen) ===\n  delta = {ver['delta']:+.3f} "
              f"({ver['identity_mean_asr']:.3f} -> {ver['kappa2_mean_asr']:.3f}), n={ver['n_kappa0']}"
              f" at both endpoints on seeds {ver['seeds']}")
        print(f"  cifar_cnn delta = {ver['cifar_cnn_delta']:+.3f}, "
              f"EMNIST/simple_cnn delta = {f_delta:+.3f}")
        print(f"  {ver['verdict']}")
        if not ver["accuracy_gate_ok"]:
            print(f"  ACCURACY GATE FAILED (some rung below {ACC_FLOOR}): the cell is "
                  "uninterpretable and the verdict does not stand.")
    print(f"\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
