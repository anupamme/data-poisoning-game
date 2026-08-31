"""The emit-only control: Krum scores on the untransformed stack, aggregates the transformed pick.

Rules frozen in experiments/pre_registration_emit_only.md. This script refuses to start until that
file is git-committed and PREREG_COMMIT below is set to its hash.

WHY THIS ARM EXISTS. Mode S opens two channels into a selector: the STATISTIC channel (the rescaling
changes Krum's pairwise distances, so it changes WHICH client is selected -- 0.733 of rounds at
kappa=2) and the MAGNITUDE channel (the selected client's update enters scaled by its own
coefficient -- emitted-aggregate displacement 0.892). run_score_only_control.py closes the second and
keeps the first. This arm is its MIRROR and closes the first: Krum scores the RAW stack, so its
decision is pinned to the identity rung's, and aggregates the TRANSFORMED selected update, so the
only channel still open is what the selected client contributes.

Together the two controls plus the identity rung and the uncontrolled arm are the four cells of a
2x2 factorial. That replaces measure_admission.py's algebraic "re-selection / rescaling" displacement
identity -- which is bookkeeping, not a counterfactual -- with a run decomposition.

WHAT IS RUN. krum / committed_scaling / CIFAR-10 / cifar_cnn, kappa in {0, 0.5, 1, 2}, seeds 42-46.
The kappa=0 rung is IMPORTED, not re-run: at kappa=0 apply_d1_transform returns the stack unwrapped
(the SAME list object), so `scored` and `emitted` are the same object and emit-only at the identity IS
krum alone. --harness-check verifies that at one seed before any new cell is written.

  python3 experiments/run_emit_only_control.py --harness-check   # verify the import claim, 1 seed
  python3 experiments/run_emit_only_control.py

Writes results/emit_only/summary.json only (resumable; written after every run).
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)

# Single-sourced from the frozen Round-12 suite: the same run_one, the same rung naming, the same
# dial, the same gate and the same margin. A copy is how two suites drift apart in what they compute.
from experiments.run_targeted_dose import (  # noqa: E402
    run_one, cell_key, dial, KAPPAS, ACC_FLOOR, EQUIV_MARGIN, SEEDS5,
)

# experiments/pre_registration_emit_only.md, committed before results/emit_only/ exists.
PREREG_COMMIT = "b995f1b"

D2 = "krum"
ATTACK = "committed_scaling"
DATASET, MODEL = "cifar10", "cifar_cnn"

# The two frozen thresholds of the primary rule, verbatim from prereg Section 4. Not derived at run
# time, so no outcome can move them.
INERT_MARGIN = 0.05        # refuted if |dASR_EO| >= this
ADDITIVITY_MARGIN = 0.05   # channels not separable if the residual >= this

TARGETED = os.path.join(base, "results", "targeted_dose", "summary.json")
SCORE_ONLY = os.path.join(base, "results", "score_only", "summary.json")
ADMISSION = os.path.join(base, "results", "admission_measurement.json")
out_dir = os.path.join(base, "results", "emit_only")
out_path = os.path.join(out_dir, "summary.json")


def _arm(path, mode_s_only=True):
    """{kappa: {seed: (accuracy, asr)}} for this cell in a frozen summary. Never transcribed."""
    if not os.path.exists(path):
        return {}
    out = {}
    for c in json.load(open(path)).get("cells", {}).values():
        if c.get("d2") != D2 or c.get("attack") != ATTACK:
            continue
        if mode_s_only and c.get("mode") != "S":
            continue
        out[c["rung"]] = {r["seed"]: (r["accuracy"], r["asr"]) for r in c["per_seed"]}
    return out


def seed_matched_delta(arm, rung):
    """Mean over seeds of (asr at rung - asr at kappa=0), on the seeds present in BOTH.

    Seed-matched because that is what the pre-registration specifies and what the published cells
    report: the contrast is within-seed, so a seed that is absent from either end is dropped from
    both rather than silently widening one mean.
    """
    lo, hi = arm.get(KAPPAS[0], {}), arm.get(rung, {})
    shared = sorted(set(lo) & set(hi))
    if not shared:
        return None, []
    d = [hi[s][1] - lo[s][1] for s in shared]
    return float(np.mean(d)), shared


def premise():
    """The frozen channel measurement this arm reasons about. Read, never re-derived."""
    if not os.path.exists(ADMISSION):
        return {}
    s = json.load(open(ADMISSION)).get("summary", {})
    return {str(k): {"agg_disp": s.get(f"doseS|{D2}|{k}|agg_disp"),
                     "decision": s.get(f"doseS|{D2}|{k}|decision"),
                     "admission": s.get(f"doseS|{D2}|{k}|admission")} for k in KAPPAS}


def identity_rung():
    """kappa=0 imported bit-exactly: emit-only at the identity IS krum alone (see module docstring)."""
    return _arm(TARGETED).get(KAPPAS[0], {})


def mean_asr(cell):
    return float(np.mean([r["asr"] for r in cell["per_seed"]])) if cell["per_seed"] else float("nan")


def mean_acc(cell):
    return float(np.mean([r["accuracy"] for r in cell["per_seed"]])) if cell["per_seed"] else float("nan")


def load_cells():
    if not os.path.exists(out_path):
        return {}
    try:
        return json.load(open(out_path)).get("cells", {})
    except Exception:
        return {}


def save(cells, ver=None):
    json.dump({"description": "Emit-only control: krum scores on the UNTRANSFORMED stack (decision "
                              "pinned to the identity rung) and aggregates the TRANSFORMED selected "
                              "update, closing the statistic channel and leaving only the magnitude "
                              "channel open. The mirror of results/score_only and the fourth cell of "
                              f"the 2x2 factorial. Rules frozen at {PREREG_COMMIT} "
                              "(experiments/pre_registration_emit_only.md).",
               "prereg_commit": PREREG_COMMIT,
               "dataset": DATASET, "model": MODEL,
               "config": {"N": 10, "K": 5, "f": 0.2, "alpha": 0.5, "rounds": 50,
                          "kappas": KAPPAS, "seeds": SEEDS5,
                          "rhos": {str(k): dial("S", k) for k in KAPPAS},
                          "acc_floor": ACC_FLOOR, "equiv_margin": EQUIV_MARGIN,
                          "inert_margin": INERT_MARGIN,
                          "additivity_margin": ADDITIVITY_MARGIN},
               "arm": {"mode": "S", "d2": D2, "attack": ATTACK, "emit_only": True,
                       "seeds": SEEDS5},
               "factorial_comparison": {
                   "full_mode_s_delta": seed_matched_delta(_arm(TARGETED), KAPPAS[-1])[0],
                   "score_only_delta": seed_matched_delta(_arm(SCORE_ONLY), KAPPAS[-1])[0]},
               "frozen_premise_channels": premise(),
               "verdict": ver,
               "cells": cells}, open(out_path, "w"), indent=2)


def check_frozen():
    prereg = os.path.join(base, "experiments", "pre_registration_emit_only.md")
    if not os.path.exists(prereg):
        sys.exit(f"REFUSING TO RUN: {prereg} does not exist.")
    if PREREG_COMMIT is None:
        sys.exit("REFUSING TO RUN: the primary rule is not frozen.\n"
                 f"  1. git commit {prereg}\n"
                 "  2. set PREREG_COMMIT here to that hash.\n"
                 "An unfrozen run makes the prediction unfalsifiable, which is the entire point.")
    for p, why in ((TARGETED, "the uncontrolled arm and the identity rung this arm imports"),
                   (SCORE_ONLY, "the opposite cell of the factorial, without which the additivity "
                                "residual cannot be scored")):
        if not os.path.exists(p):
            sys.exit(f"REFUSING TO RUN: {p} does not exist. It holds {why}.")


def harness_check():
    """kappa=0 with emit_only=True must reproduce the published identity rung BIT-IDENTICALLY.

    At kappa=0 apply_d1_transform returns the update list unwrapped -- the same list object -- so
    `scored` and `emitted` are that one object: Krum scores the raw stack and aggregates the raw
    selected update, which is Krum alone. This is the claim that lets the kappa=0 rung be imported
    rather than re-run, and it is an EXACT claim, so a tolerance is asserted rather than eyeballed.
    """
    seed = SEEDS5[0]
    ref = identity_rung().get(seed)
    if ref is None:
        sys.exit("no published identity rung at this seed to check against")
    r_acc, r_asr = ref
    print("=== HARNESS CHECK: emit-only at kappa=0 must BE krum alone, bit-identically ===")
    print(f"    {DATASET}/{MODEL}, {D2}/{ATTACK.replace('committed_', '')} at seed {seed}")
    print(f"    published identity rung: acc={r_acc:.6f} ASR={r_asr:.6f}\n", flush=True)
    t = time.time()
    acc, asr = run_one(seed, "S", D2, ATTACK, 0.0, dataset=DATASET, model=MODEL, emit_only=True)
    print(f"  emit_only s{seed}:        acc={acc:.6f} ASR={asr:.6f}   "
          f"d=({acc - r_acc:+.2e}, {asr - r_asr:+.2e})  ({time.time() - t:.0f}s)")
    ok = abs(acc - r_acc) < 1e-9 and abs(asr - r_asr) < 1e-9
    print(f"\n  {'BIT-IDENTICAL. The kappa=0 import is valid.' if ok else 'NOT IDENTICAL: the import claim is FALSE and kappa=0 must be re-run in-suite.'}")
    return 0 if ok else 1


def verdict(cells):
    """Score the two frozen predictions of prereg Section 4. Printed here and stored in the summary.

    The primary rung is kappa=2, matched to the endpoint both other cells report. Section 5 permits
    exactly one substitution: if kappa=2 fails the accuracy floor the highest gate-passing rung is
    used instead, and the substitution is recorded as FORCED so it cannot read as a choice.
    """
    arm = {c["rung"]: {r["seed"]: (r["accuracy"], r["asr"]) for r in c["per_seed"]}
           for c in cells.values()}
    if KAPPAS[0] not in arm:
        return None

    def acc_ok(rung):
        c = cells.get(cell_key("S", D2, ATTACK, rung))
        return c is not None and mean_acc(c) >= ACC_FLOOR

    primary, forced = KAPPAS[-1], False
    if not acc_ok(primary):
        passing = [k for k in KAPPAS[1:] if k in arm and acc_ok(k)]
        if not passing:
            return {"primary_rung": None, "verdict": "NO RUNG CLEARS THE ACCURACY FLOOR: the cell "
                    f"is uninterpretable (ACC_FLOOR={ACC_FLOOR}). A low ASR here is a collapsed "
                    "model, not suppression, and nothing is concluded.",
                    "accuracy_gate_ok": False, "primary_rung_forced": True}
        primary, forced = max(passing), True

    d_eo, shared = seed_matched_delta(arm, primary)
    if d_eo is None:
        return None
    d_full, _ = seed_matched_delta(_arm(TARGETED), primary)
    d_so, _ = seed_matched_delta(_arm(SCORE_ONLY), primary)

    inert = abs(d_eo) < INERT_MARGIN
    resid = None if (d_full is None or d_so is None) else float(d_full - (d_so + d_eo))
    separable = resid is not None and abs(resid) < ADDITIVITY_MARGIN

    if inert and separable:
        v = ("BOTH PREDICTIONS HOLD: the magnitude channel is inert on its own and the two channels "
             "are additive. Note prereg limitation 1 -- at n=5 this is WEAK evidence of "
             "separability, not a confirmation.")
    elif not inert:
        v = ("REFUTED: the magnitude channel alone moves ASR by more than the frozen margin. The "
             "score-only control's agreement with the uncontrolled arm was therefore partly "
             "coincidental, and the paper must RE-ATTRIBUTE: the claim narrows to 'both channels "
             "are individually small' and the decision channel is no longer established as the one "
             "doing the work.")
    else:
        v = ("CHANNELS INTERACT: the magnitude channel is inert alone but the additivity residual "
             "exceeds the frozen margin, so NO per-channel decomposition of the uncontrolled arm is "
             "licensed -- including the algebraic one this arm was built to replace.")

    return {"primary_rung": primary, "primary_rung_forced": forced,
            "seeds_matched": shared,
            "d_asr_emit_only": d_eo, "d_asr_full_mode_s": d_full, "d_asr_score_only": d_so,
            "additive_point_prediction": (None if (d_full is None or d_so is None)
                                          else float(d_full - d_so)),
            "additivity_residual": resid,
            "inert_margin": INERT_MARGIN, "additivity_margin": ADDITIVITY_MARGIN,
            "magnitude_channel_inert": inert, "channels_separable": separable,
            "accuracy_gate_ok": all(acc_ok(k) for k in KAPPAS[1:] if k in arm),
            "verdict": v}


def main():
    check_frozen()
    os.makedirs(out_dir, exist_ok=True)
    if "--harness-check" in sys.argv:
        return harness_check()

    cells = load_cells()
    # kappa=0 is imported bit-exactly; --harness-check verifies the claim at one seed.
    key0 = cell_key("S", D2, ATTACK, KAPPAS[0])
    if key0 not in cells:
        imported = identity_rung()
        if not imported:
            sys.exit("no published identity rung to import; refusing to fabricate kappa=0")
        cells[key0] = {"mode": "S", "d2": D2, "attack": ATTACK, "rung": KAPPAS[0],
                       "dial": dial("S", KAPPAS[0]), "emit_only": True,
                       "dataset": DATASET, "model": MODEL,
                       "per_seed": [{"seed": s, "accuracy": a, "asr": r,
                                     "source": "results/targeted_dose kappa=0 (identical "
                                               "computation: emit-only at the identity IS krum)"}
                                    for s, (a, r) in sorted(imported.items())]}
        cells[key0]["mean_asr"] = mean_asr(cells[key0])
        cells[key0]["mean_acc"] = mean_acc(cells[key0])
        save(cells)

    todo = [(v, s) for v in KAPPAS[1:] for s in SEEDS5
            if not any(r["seed"] == s
                       for r in cells.get(cell_key("S", D2, ATTACK, v), {}).get("per_seed", []))]

    d_full, _ = seed_matched_delta(_arm(TARGETED), KAPPAS[-1])
    d_so, _ = seed_matched_delta(_arm(SCORE_ONLY), KAPPAS[-1])
    print(f"=== EMIT-ONLY CONTROL: {len(todo)} new runs (kappa=0 imported bit-exactly) ===")
    print(f"    rules frozen in pre_registration_emit_only.md @ {PREREG_COMMIT}")
    print(f"    {DATASET}/{MODEL}, mode S {D2}/{ATTACK.replace('committed_', '')}, emit_only=True")
    print(f"    kappa {KAPPAS} -> rho {[round(dial('S', k), 2) for k in KAPPAS]}, seeds {SEEDS5}")
    print("    the other three cells of the factorial, recomputed seed-matched from frozen files:")
    print(f"      full Mode S  dASR = {d_full:+.4f}   (results/targeted_dose)")
    print(f"      score-only   dASR = {d_so:+.4f}   (results/score_only)")
    print(f"    => additive point prediction for THIS arm: {d_full - d_so:+.4f}")
    print(f"    REFUTED if |dASR| >= {INERT_MARGIN}; channels not separable if "
          f"|residual| >= {ADDITIVITY_MARGIN}")
    pr = premise()
    if pr and all(pr[str(k)]["agg_disp"] is not None for k in KAPPAS):
        print("    frozen premise (results/admission_measurement.json), agg_disp/decision/admission:")
        print("      " + "  ".join(
            f"{k}:{pr[str(k)]['agg_disp']:.3f}/{pr[str(k)]['decision']:.3f}/"
            f"{pr[str(k)]['admission']:.3f}" for k in KAPPAS))
        print("      the statistic channel this arm closes is the decision column")
    print("    NOTE prereg Section 5: the emitted update is the RESCALED one, so the accuracy floor "
          f"({ACC_FLOOR}) is a live risk at kappa=2 in a way it is not for the other three cells.")
    print(flush=True)

    t0, done = time.time(), 0
    for v, seed in todo:
        key = cell_key("S", D2, ATTACK, v)
        cell = cells.setdefault(key, {"mode": "S", "d2": D2, "attack": ATTACK, "rung": v,
                                      "dial": dial("S", v), "emit_only": True,
                                      "dataset": DATASET, "model": MODEL, "per_seed": []})
        t = time.time()
        acc, asr = run_one(seed, "S", D2, ATTACK, v, dataset=DATASET, model=MODEL, emit_only=True)
        cell["per_seed"].append({"seed": seed, "accuracy": acc, "asr": asr})
        cell["per_seed"].sort(key=lambda r: r["seed"])
        cell["mean_asr"], cell["mean_acc"] = mean_asr(cell), mean_acc(cell)
        cell["std_asr"] = float(np.std([r["asr"] for r in cell["per_seed"]], ddof=0))
        cells[key] = cell
        save(cells)
        done += 1
        print(f"  [{done}/{len(todo)}] emit-only {D2} kappa={v} s{seed}: "
              f"acc={acc:.3f} ASR={asr:.3f} ({time.time() - t:.0f}s)"
              + ("  * below acc floor" if acc < ACC_FLOOR else ""), flush=True)

    print("\n=== LADDER (mean ASR @ mean clean accuracy) ===")
    print("  " + " ".join(f"{'kappa=' + str(v):>14s}" for v in KAPPAS))
    row = []
    for v in KAPPAS:
        c = cells.get(cell_key("S", D2, ATTACK, v))
        row.append("na" if c is None else f"{mean_asr(c):.3f}@{mean_acc(c):.2f}"
                   + ("!" if mean_acc(c) < ACC_FLOOR else " "))
    print("  " + " ".join(f"{x:>14s}" for x in row))
    print(f"  '!' = mean clean accuracy < {ACC_FLOOR}: uninterpretable, not suppression.")

    ver = verdict(cells)
    save(cells, ver)
    if ver and ver.get("primary_rung") is not None:
        print(f"\n=== PRIMARY (frozen) ===\n  primary rung kappa={ver['primary_rung']}"
              + ("  [FORCED by the accuracy floor, not chosen]" if ver["primary_rung_forced"] else ""))
        print(f"  dASR emit-only = {ver['d_asr_emit_only']:+.4f}  "
              f"(margin {INERT_MARGIN}; predicted {ver['additive_point_prediction']:+.4f})")
        print(f"  full Mode S {ver['d_asr_full_mode_s']:+.4f} = score-only "
              f"{ver['d_asr_score_only']:+.4f} + emit-only {ver['d_asr_emit_only']:+.4f} + "
              f"residual {ver['additivity_residual']:+.4f}  (margin {ADDITIVITY_MARGIN})")
        print(f"  {ver['verdict']}")
        if not ver["accuracy_gate_ok"]:
            print(f"  ACCURACY GATE: some rung is below {ACC_FLOOR} and is reported as "
                  "uninterpretable rather than as suppression.")
    elif ver:
        print(f"\n=== PRIMARY (frozen) ===\n  {ver['verdict']}")
    print(f"\nWall time: {(time.time() - t0) / 3600:.1f} h\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
