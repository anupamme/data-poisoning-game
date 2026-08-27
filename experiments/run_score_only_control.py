"""The score-only control: Krum scores on the transformed stack, aggregates the untransformed pick.

Rules frozen in experiments/pre_registration_score_only.md. This script refuses to start until that
file is git-committed and PREREG_COMMIT below is set to that hash.

WHY THIS EXISTS. Mode S pins every adversary's coefficient at c_adv = 1.0 exactly, closing the
ATTENUATION channel by construction. It does not hold the benign coefficients fixed, so the dose also
moves the MAGNITUDE of the update that actually reaches the model: the selected client's update enters
scaled by its own coefficient. That displacement is already measured and already published -- the
`Delta agg.` column of the paper's channel table is ||agg(T(U)) - agg(U)|| / ||agg(U)||, and it reads
0.892 on this cell at kappa=2.

This arm closes that channel. Krum SCORES on the transformed stack -- statistic and decision disturbed
exactly as before -- and AGGREGATES the untransformed selected update, so what enters training is
bit-for-bit what a client produced. The dose can change WHICH client contributes and nothing about
WHAT that client contributes. It is the reviewer-suggested control in its own terms: hold the selected
benign update's magnitude and direction fixed while perturbing the selection statistic independently.

WHAT IS RUN. krum / committed_scaling / CIFAR-10 / cifar_cnn, kappa in {0, 0.5, 1, 2}, seeds 42-46.
The kappa=0 rung is IMPORTED, not re-run: at kappa=0 apply_d1_transform returns the stack unwrapped,
so score-only Krum IS Krum alone and the rung is bit-identical to the published identity rung.
15 new runs at ~730 s each, about 3 h.

  python3 experiments/run_score_only_control.py --harness-check   # verify the import claim, 1 seed
  python3 experiments/run_score_only_control.py --cos-krum-check  # the Section 5 bit-identity test
  python3 experiments/run_score_only_control.py

Reads results/targeted_dose/summary.json and results/admission_measurement.json, both READ-ONLY.
Writes results/score_only/summary.json only (resumable; written after every run).
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
# Single-sourced from the frozen Round-12 suite: the same run_one, the same rung naming, the same
# dial, the same gate and the same margin. A copy is how two suites drift apart in what they compute.
from experiments.run_targeted_dose import (  # noqa: E402
    run_one, cell_key, dial, d1_name, KAPPAS, ACC_FLOOR, EQUIV_MARGIN, SEEDS5,
)

# experiments/pre_registration_score_only.md, committed before results/score_only/ existed.
PREREG_COMMIT = "35788d9"

D2 = "krum"
ATTACK = "committed_scaling"
DATASET, MODEL = "cifar10", "cifar_cnn"
COS_ARM = "cos_krum"          # Section 5 of the pre-registration: an assertion, not a five-seed arm

TARGETED = os.path.join(base, "results", "targeted_dose", "summary.json")
ADMISSION = os.path.join(base, "results", "admission_measurement.json")
out_dir = os.path.join(base, "results", "score_only")
out_path = os.path.join(out_dir, "summary.json")


def published_arm():
    """{kappa: {seed: (accuracy, asr)}} for the uncontrolled Mode-S arm. Never transcribed."""
    if not os.path.exists(TARGETED):
        return {}
    cells = json.load(open(TARGETED)).get("cells", {})
    out = {}
    for c in cells.values():
        if c.get("mode") == "S" and c.get("d2") == D2 and c.get("attack") == ATTACK:
            out[c["rung"]] = {r["seed"]: (r["accuracy"], r["asr"]) for r in c["per_seed"]}
    return out


def published_delta():
    """The published Mode-S krum rise, recomputed from per-seed rows at run time."""
    arm = published_arm()
    if not (KAPPAS[0] in arm and KAPPAS[-1] in arm):
        return None
    lo = np.mean([a for _, a in arm[KAPPAS[0]].values()])
    hi = np.mean([a for _, a in arm[KAPPAS[-1]].values()])
    return float(hi - lo)


def premise():
    """The frozen channel measurement this arm reasons about. Read, never re-derived."""
    if not os.path.exists(ADMISSION):
        return {}
    s = json.load(open(ADMISSION)).get("summary", {})
    return {str(k): {"agg_disp": s.get(f"doseS|{D2}|{k}|agg_disp"),
                     "decision": s.get(f"doseS|{D2}|{k}|decision"),
                     "admission": s.get(f"doseS|{D2}|{k}|admission")} for k in KAPPAS}


def identity_rung():
    """kappa=0 imported bit-exactly: score-only at the identity IS krum alone (see docstring)."""
    arm = published_arm()
    return arm.get(KAPPAS[0], {})


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


def save(cells):
    json.dump({"description": "Score-only control: krum scores on the transformed stack and "
                              "aggregates the UNTRANSFORMED selected update, closing the magnitude "
                              "channel that Mode S leaves open. Rules frozen at "
                              f"{PREREG_COMMIT} (experiments/pre_registration_score_only.md).",
               "prereg_commit": PREREG_COMMIT,
               "dataset": DATASET, "model": MODEL,
               "config": {"N": 10, "K": 5, "f": 0.2, "alpha": 0.5, "rounds": 50,
                          "kappas": KAPPAS, "seeds": SEEDS5,
                          "rhos": {str(k): dial("S", k) for k in KAPPAS},
                          "acc_floor": ACC_FLOOR, "equiv_margin": EQUIV_MARGIN},
               "arm": {"mode": "S", "d2": D2, "attack": ATTACK, "score_only": True,
                       "seeds": SEEDS5},
               "uncontrolled_comparison": {"d2": D2, "attack": ATTACK,
                                           "published_mode_s_delta": published_delta()},
               "frozen_premise_channels": premise(),
               "cells": cells}, open(out_path, "w"), indent=2)


def check_frozen():
    prereg = os.path.join(base, "experiments", "pre_registration_score_only.md")
    if not os.path.exists(prereg):
        sys.exit(f"REFUSING TO RUN: {prereg} does not exist.")
    if PREREG_COMMIT is None:
        sys.exit("REFUSING TO RUN: the primary rule is not frozen.\n"
                 f"  1. git commit {prereg}\n"
                 "  2. set PREREG_COMMIT here to that hash.\n"
                 "An unfrozen run makes the prediction unfalsifiable, which is the entire point.")
    if not os.path.exists(TARGETED):
        sys.exit(f"REFUSING TO RUN: {TARGETED} does not exist. The uncontrolled arm this control is "
                 "compared against, and the identity rung it imports, both live there.")


def harness_check():
    """kappa=0 with score_only=True must reproduce the published identity rung BIT-IDENTICALLY.

    At kappa=0 apply_d1_transform returns the update list unwrapped, so the transformed and raw
    stacks are the same object: Krum scores on the raw stack and aggregates the raw selected update,
    which is Krum alone. This is the claim that lets the kappa=0 rung be imported rather than re-run,
    and unlike the FEMNIST arm's sanity check it is an EXACT claim, so a tolerance is asserted.
    """
    seed = SEEDS5[0]
    ref = identity_rung().get(seed)
    if ref is None:
        sys.exit("no published identity rung at this seed to check against")
    r_acc, r_asr = ref
    print("=== HARNESS CHECK: score-only at kappa=0 must BE krum alone, bit-identically ===")
    print(f"    {DATASET}/{MODEL}, {D2}/{ATTACK.replace('committed_', '')} at seed {seed}")
    print(f"    published identity rung: acc={r_acc:.6f} ASR={r_asr:.6f}\n", flush=True)
    t = time.time()
    acc, asr = run_one(seed, "S", D2, ATTACK, 0.0, dataset=DATASET, model=MODEL, score_only=True)
    print(f"  score_only s{seed}:       acc={acc:.6f} ASR={asr:.6f}   "
          f"d=({acc - r_acc:+.2e}, {asr - r_asr:+.2e})  ({time.time() - t:.0f}s)")
    ok = abs(acc - r_acc) < 1e-9 and abs(asr - r_asr) < 1e-9
    print(f"\n  {'BIT-IDENTICAL. The kappa=0 import is valid.' if ok else 'NOT IDENTICAL: the import claim is FALSE and kappa=0 must be re-run in-suite.'}")
    return 0 if ok else 1


def cos_krum_check():
    """Section 5 of the pre-registration: score-only cos_krum must be bit-identical to kappa=0.

    cos_krum's statistic is exactly invariant under positive per-client rescaling, and the paper
    measures 0/120 rounds changing selection at a maximum relative score drift of 1.1e-4. Under
    score-only the selection is therefore unchanged AND the aggregated update is untransformed, so
    the whole trajectory must be identical to the identity rung. This is a TEST of the paper's claim
    that the uncontrolled cos_krum arm's 0.173 fall travels through the MAGNITUDE channel -- the only
    channel left open there. One seed, because the prediction is identity rather than a mean.
    """
    seed = SEEDS5[0]
    print("=== COS_KRUM CHECK: score-only must be bit-identical across rungs (prereg Section 5) ===")
    print(f"    {DATASET}/{MODEL}, {COS_ARM}/{ATTACK.replace('committed_', '')} at seed {seed}")
    print("    Predicted: identical at every rung. If not, the magnitude attribution of the")
    print("    uncontrolled arm's 0.173 fall is WRONG and is withdrawn.\n", flush=True)
    ref, rows = None, []
    for v in KAPPAS:
        t = time.time()
        acc, asr = run_one(seed, "S", COS_ARM, ATTACK, v, dataset=DATASET, model=MODEL,
                           score_only=True)
        if ref is None:
            ref = (acc, asr)
        d = (acc - ref[0], asr - ref[1])
        rows.append({"kappa": v, "accuracy": acc, "asr": asr,
                     "d_accuracy": d[0], "d_asr": d[1]})
        print(f"  kappa={v:<4} acc={acc:.6f} ASR={asr:.6f}   d=({d[0]:+.2e}, {d[1]:+.2e})  "
              f"({time.time() - t:.0f}s)", flush=True)
    ok = all(abs(r["d_asr"]) < 1e-9 and abs(r["d_accuracy"]) < 1e-9 for r in rows)
    print(f"\n  {'BIT-IDENTICAL at every rung: the magnitude attribution is confirmed.' if ok else 'NOT IDENTICAL: the magnitude attribution is REFUTED and must be withdrawn.'}")
    os.makedirs(out_dir, exist_ok=True)
    json.dump({"description": "Prereg Section 5: score-only cos_krum bit-identity check, 1 seed.",
               "prereg_commit": PREREG_COMMIT, "d2": COS_ARM, "attack": ATTACK, "seed": seed,
               "rungs": rows, "bit_identical": ok},
              open(os.path.join(out_dir, "cos_krum_check.json"), "w"), indent=2)
    return 0


def verdict(cells):
    """Score the frozen primary rule. Printed here, and recomputed by the analyzer."""
    lo = cells.get(cell_key("S", D2, ATTACK, KAPPAS[0]))
    hi = cells.get(cell_key("S", D2, ATTACK, KAPPAS[-1]))
    if lo is None or hi is None:
        return None
    m0, m2 = mean_asr(lo), mean_asr(hi)
    delta = float(m2 - m0)
    if abs(delta) < EQUIV_MARGIN:
        v = "THE NEGATIVE SURVIVES WITH THE MAGNITUDE CHANNEL CLOSED"
    elif delta > EQUIV_MARGIN:
        v = ("THE PUBLISHED FLAT RESULT WAS PARTLY AN ARTIFACT OF THE MAGNITUDE CHANNEL: "
             "the central claim narrows to 'with benign magnitudes free to move'")
    else:
        v = "INDETERMINATE (fall with both channels closed), not scored in our favour"
    gate_ok = all(mean_acc(c) >= ACC_FLOOR
                  for c in (cells.get(cell_key("S", D2, ATTACK, k)) for k in KAPPAS)
                  if c is not None)
    return {"delta": delta, "identity_mean_asr": m0, "kappa2_mean_asr": m2, "verdict": v,
            "refuting_threshold": m0 + EQUIV_MARGIN, "accuracy_gate_ok": gate_ok,
            "published_mode_s_delta": published_delta()}


def main():
    check_frozen()
    os.makedirs(out_dir, exist_ok=True)
    if "--harness-check" in sys.argv:
        return harness_check()
    if "--cos-krum-check" in sys.argv:
        return cos_krum_check()

    cells = load_cells()
    # kappa=0 is imported bit-exactly; --harness-check verifies the claim at one seed.
    key0 = cell_key("S", D2, ATTACK, KAPPAS[0])
    if key0 not in cells:
        imported = identity_rung()
        if not imported:
            sys.exit("no published identity rung to import; refusing to fabricate kappa=0")
        cells[key0] = {"mode": "S", "d2": D2, "attack": ATTACK, "rung": KAPPAS[0],
                       "dial": dial("S", KAPPAS[0]), "score_only": True,
                       "dataset": DATASET, "model": MODEL,
                       "per_seed": [{"seed": s, "accuracy": a, "asr": r,
                                     "source": "results/targeted_dose kappa=0 (identical "
                                               "computation: score-only at the identity IS krum)"}
                                    for s, (a, r) in sorted(imported.items())]}
        cells[key0]["mean_asr"] = mean_asr(cells[key0])
        cells[key0]["mean_acc"] = mean_acc(cells[key0])
        save(cells)

    todo = [(v, s) for v in KAPPAS[1:] for s in SEEDS5
            if not any(r["seed"] == s
                       for r in cells.get(cell_key("S", D2, ATTACK, v), {}).get("per_seed", []))]

    pd_ = published_delta()
    print(f"=== SCORE-ONLY CONTROL: {len(todo)} new runs (kappa=0 imported bit-exactly) ===")
    print(f"    rules frozen in pre_registration_score_only.md @ {PREREG_COMMIT}")
    print(f"    {DATASET}/{MODEL}, mode S {D2}/{ATTACK.replace('committed_', '')}, score_only=True")
    print(f"    kappa {KAPPAS} -> rho {[round(dial('S', k), 2) for k in KAPPAS]}, seeds {SEEDS5}")
    print(f"    uncontrolled Mode-S delta = {pd_:+.3f} (recomputed from results/targeted_dose)")
    m0 = mean_asr(cells[key0])
    print(f"    identity rung {m0:.4f}; REFUTED if mean ASR(kappa=2) > {m0 + EQUIV_MARGIN:.4f}, "
          f"survives if |delta| < {EQUIV_MARGIN}")
    pr = premise()
    if pr:
        print("    frozen premise (results/admission_measurement.json), agg_disp/decision/admission:")
        print("      " + "  ".join(
            f"{k}:{pr[str(k)]['agg_disp']:.3f}/{pr[str(k)]['decision']:.3f}/"
            f"{pr[str(k)]['admission']:.3f}" for k in KAPPAS))
        print("      the magnitude channel this arm closes is the agg_disp column")
    print(flush=True)

    t0, done = time.time(), 0
    for v, seed in todo:
        key = cell_key("S", D2, ATTACK, v)
        cell = cells.setdefault(key, {"mode": "S", "d2": D2, "attack": ATTACK, "rung": v,
                                      "dial": dial("S", v), "score_only": True,
                                      "dataset": DATASET, "model": MODEL, "per_seed": []})
        t = time.time()
        acc, asr = run_one(seed, "S", D2, ATTACK, v, dataset=DATASET, model=MODEL, score_only=True)
        cell["per_seed"].append({"seed": seed, "accuracy": acc, "asr": asr})
        cell["per_seed"].sort(key=lambda r: r["seed"])
        cell["mean_asr"], cell["mean_acc"] = mean_asr(cell), mean_acc(cell)
        cell["std_asr"] = float(np.std([r["asr"] for r in cell["per_seed"]], ddof=0))
        cells[key] = cell
        save(cells)
        done += 1
        print(f"  [{done}/{len(todo)}] score-only {D2} kappa={v} s{seed}: "
              f"acc={acc:.3f} ASR={asr:.3f} ({time.time() - t:.0f}s)", flush=True)

    save(cells)
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
    if ver:
        print(f"\n=== PRIMARY (frozen) ===\n  delta = {ver['delta']:+.3f} "
              f"({ver['identity_mean_asr']:.3f} -> {ver['kappa2_mean_asr']:.3f}), margin "
              f"{EQUIV_MARGIN}; uncontrolled Mode-S delta = {ver['published_mode_s_delta']:+.3f}")
        print(f"  {ver['verdict']}")
        if not ver["accuracy_gate_ok"]:
            print(f"  ACCURACY GATE FAILED (some rung below {ACC_FLOOR}): the cell is "
                  "uninterpretable and the verdict does not stand.")
    print("\n  The secondary JT trend test is scored by experiments/analyze_score_only_control.py")
    print("  against the same frozen rules.")
    print(f"\nWall time: {(time.time() - t0) / 3600:.1f} h\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
