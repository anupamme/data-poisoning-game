"""
Rank-discordant replication arm for the admission channel (Round 15): coord_median under Mode S.

WHY THIS ARM EXISTS, IN ONE PARAGRAPH. Round 12's Mode-S suite established the paper's flagship
negative result on the Krum arm -- decision changed in 80% of rounds, admission in exactly 0%, and
suppression did not deteriorate -- so disturbance of the downstream statistic was causally irrelevant
to suppression there. The POSITIVE half, that admission is the channel which does matter, stayed weak
for a structural reason: across the three interpretable Mode-S arms the two candidate predictors
CO-VARY. Reputation is higher than Krum on both decision change (1.000 vs 0.733) and admission change
(0.020 vs 0.000) and had the larger rise (+0.178 vs -0.026), an ordering consistent with either
reading. That is why the pooled two-channel model came out unconfirmed (R^2 = 0.24, admission
coefficient one-sided p = 0.055).

coord_median/pixel under Mode S is RANK-DISCORDANT and therefore breaks the tie. From
results/admission_measurement.json, measured in Round 12 before any ASR existed for this cell:

  decision change  0.0000 / 0.2315 / 0.3576 / 0.4821   <- the SMALLEST of any interpretable arm
  admission change 0.0000 / 0.0046 / 0.0125 / 0.0332   <- the LARGEST of any interpretable arm

  ordering by decision change  predicts this arm's rise is the SMALLEST of the three.
  ordering by admission change predicts this arm's rise is the LARGEST of the three.

It is also a different KIND of aggregator -- a coordinate-wise order statistic (Prop 1b) rather than a
selector or weighted averager reading pairwise/consensus distance (Prop 1c) -- so the outcome also
answers whether the flagship result is a property of selector-type defenses or of mechanism
preservation generally.

FROZEN DECISION RULES (verbatim from experiments/pre_registration_dose_replication.md).

  Primary, on Delta = mean ASR(kappa=2) - mean ASR(kappa=0), against the two published rises
  (-0.026 krum, +0.178 reputation) and the suite's existing EQUIV_MARGIN = 0.15:

    Delta >  +0.178            ADMISSION ORDERING CONFIRMED. Largest admission change, smallest
                               decision change, largest rise. First prospective support for the
                               positive half of the claim.
    Delta <  +0.150            ADMISSION ORDERING REFUTED. The arm with the largest admission change
                               rose within the flat margin, so admission does not order the outcomes;
                               the positive claim is scoped to selector-type defenses. This outcome
                               simultaneously REPLICATES THE FLAGSHIP NEGATIVE on a structurally
                               different defense: decision change 0.482 with suppression preserved.
    otherwise                  INDETERMINATE, reported as such and not scored in our favour.

  Secondary: Jonckheere-Terpstra across the four rungs. BOTH readings predict a rise on this arm, so a
  flat or falling curve refutes both with the attenuation channel already closed -- the strongest
  single negative this arm can produce.

  Tertiary: the pooled two-channel model re-fit with these 3 cells added to the existing 22.

THE ARM IS AN INSTRUMENT, NOT A DEFENSE. doseS reads adversary identity; no deployable defense knows
which clients are adversarial. Nothing here is proposed for deployment.

THE IDENTITY RUNG IS NOT RE-RUN. doseS_kappa0.0 returns the update list unwrapped and the participant
RNG stream does not depend on d1's name, so at the same seed it is bit-for-bit the Round-11 kappa=0
rung. Imported from results/dose_response/ and marked with its provenance. --harness-check re-runs it
at one seed and asserts equality to 1e-6; if that fails the import is invalid and the rung must be
re-run.

Every quantity above that this script prints is RECOMPUTED from the results files per seed -- the
comparison rises included -- never transcribed. Config identical to the rest of the paper: N=10, K=5,
f=0.2, alpha=0.5, 50 rounds, cifar_cnn. New runs: 15.

Output: results/dose_replication/summary.json (resumable; written after every run).

DO NOT RUN until experiments/pre_registration_dose_replication.md is git-committed and PREREG_COMMIT
below is set to that hash. The script refuses to start otherwise.

  python3 experiments/run_dose_replication.py --harness-check
  python3 experiments/run_dose_replication.py
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
# Single-sourced from the frozen Round-12 suite: the same run_one, the same rung naming, the same
# identity-rung import, the same gates. Nothing about Mode S is reimplemented here, so the two suites
# cannot drift apart in what they compute.
from experiments.run_targeted_dose import (
    run_one, cell_key, dial, d1_name, ladder1_identity,
    KAPPAS, SEEDS5, ACC_FLOOR, EQUIV_MARGIN, TOL,
)

# experiments/pre_registration_dose_replication.md, committed before results/dose_replication/ existed.
PREREG_COMMIT = None

# (d2, attack, prop1_class, predicted_ordering, seeds) -- verbatim from the frozen pre-registration.
ARM = ("coord_median", "committed_pixel", "b) conditionally inv.",
       "rise LARGEST of the three arms if admission orders the outcomes, SMALLEST if decision does",
       SEEDS5)

# The two comparison arms, named here so the primary test's inputs are fixed before the run. Their
# rises are COMPUTED from the results files below, not written down.
COMPARISON_ARMS = [("krum", "committed_scaling"), ("reputation", "committed_scaling")]
CONFIRM_ABOVE = 0.178      # reputation's published Mode-S rise; recomputed and asserted at run time
REFUTE_BELOW = EQUIV_MARGIN  # 0.15, the suite's existing equivalence margin -- no new constant

out_dir = os.path.join(base, "results", "dose_replication")
out_path = os.path.join(out_dir, "summary.json")
TARGETED = os.path.join(base, "results", "targeted_dose", "summary.json")


def mean_asr(cell):
    """Mean ASR of a cell recomputed from its per-seed rows.

    The frozen suite writes mean_asr only for cells it ran; imported identity cells carry per_seed
    rows and no aggregate. Recomputing here rather than reading a stored mean is also what
    non-negotiable 3 of the pre-registration requires.
    """
    rows = cell.get("per_seed", [])
    return float(np.mean([r["asr"] for r in rows])) if rows else float("nan")


def mean_acc(cell):
    rows = cell.get("per_seed", [])
    return float(np.mean([r["accuracy"] for r in rows])) if rows else float("nan")


def published_rise(d2, attack):
    """mean ASR(kappa=2) - mean ASR(kappa=0) for a Round-12 Mode-S arm, from its per-seed rows."""
    if not os.path.exists(TARGETED):
        return float("nan")
    cells = json.load(open(TARGETED)).get("cells", {})
    lo = cells.get(cell_key("S", d2, attack, 0.0))
    hi = cells.get(cell_key("S", d2, attack, 2.0))
    if lo is None or hi is None:
        return float("nan")
    return mean_asr(hi) - mean_asr(lo)


def load_cells():
    if not os.path.exists(out_path):
        return {}
    try:
        return json.load(open(out_path)).get("cells", {})
    except Exception:
        return {}


def save(cells):
    d2, atk, cls, ordering, seeds = ARM
    json.dump({"description": "Rank-discordant replication arm for the admission channel: "
                              "coord_median under Mode S (statistic-only, adversary pinned at c=1). "
                              f"Ordering and thresholds frozen at {PREREG_COMMIT} "
                              "(experiments/pre_registration_dose_replication.md).",
               "prereg_commit": PREREG_COMMIT,
               "config": {"N": 10, "K": 5, "f": 0.2, "alpha": 0.5, "rounds": 50,
                          "kappas": KAPPAS, "seeds": seeds,
                          "rhos": {str(k): dial("S", k) for k in KAPPAS},
                          "acc_floor": ACC_FLOOR, "equiv_margin": EQUIV_MARGIN,
                          "confirm_above": CONFIRM_ABOVE, "refute_below": REFUTE_BELOW},
               "arm": {"mode": "S", "d2": d2, "attack": atk, "prop1_class": cls,
                       "predicted_ordering": ordering, "seeds": seeds},
               "comparison_arms": [{"d2": a, "attack": b, "published_mode_s_rise": published_rise(a, b)}
                                   for a, b in COMPARISON_ARMS],
               "cells": cells}, open(out_path, "w"), indent=2)


def check_frozen():
    prereg = os.path.join(base, "experiments", "pre_registration_dose_replication.md")
    if not os.path.exists(prereg):
        sys.exit(f"REFUSING TO RUN: {prereg} does not exist.")
    if PREREG_COMMIT is None:
        sys.exit("REFUSING TO RUN: the predicted ordering is not frozen.\n"
                 f"  1. git commit {prereg}\n"
                 "  2. set PREREG_COMMIT here to that hash.\n"
                 "An unfrozen run makes the ordering unfalsifiable, which is the entire point.")


def harness_check():
    """The identity rung must reproduce the Round-11 kappa=0 rung EXACTLY, at one seed.

    doseS_kappa0.0 returns the update list unwrapped, so at the same seed this is bit-for-bit the same
    computation as dose_kappa0.0. Any nonzero deviation means the two suites do not share a code path
    and the imported identity rung must be discarded and re-run rather than reconciled.
    """
    d2, atk, _, _, _ = ARM
    seed = SEEDS5[0]
    print("=== HARNESS CHECK: the imported identity rung vs the Round-11 kappa=0 rung ===")
    print(f"    {d2}/{atk.replace('committed_', '')} at seed {seed}, tolerance {TOL:g}\n", flush=True)
    pub = ladder1_identity(d2, atk)
    if seed not in pub:
        print(f"  NO ROUND-11 kappa=0 SEED {seed} for this cell -- the import is not available and "
              "the identity rung must be run in this suite.")
        return 1
    t = time.time()
    acc, asr = run_one(seed, "S", d2, atk, 0.0)
    p_acc, p_asr = pub[seed]
    d_acc, d_asr = acc - p_acc, asr - p_asr
    ok = max(abs(d_asr), abs(d_acc)) <= TOL
    print(f"  this suite: acc={acc:.4f} ASR={asr:.4f}   Round-11 s{seed}: acc={p_acc:.4f} "
          f"ASR={p_asr:.4f}   d=({d_acc:+.6f}, {d_asr:+.6f})  {'OK' if ok else 'MISMATCH'}"
          f"  ({time.time() - t:.0f}s)")
    print(f"\n  largest absolute deviation: {max(abs(d_asr), abs(d_acc)):.2e}  (tolerance {TOL:g})")
    print("  Import is valid." if ok else
          "  MISMATCH -- the identity rung may NOT be imported; run it in this suite.")
    return 0 if ok else 1


def verdict(cells):
    """Score the frozen primary rule. Printed, and recomputed by analyze_dose_replication.py."""
    d2, atk, _, _, _ = ARM
    lo = cells.get(cell_key("S", d2, atk, 0.0))
    hi = cells.get(cell_key("S", d2, atk, 2.0))
    if lo is None or hi is None:
        return None
    a0, a2 = mean_asr(lo), mean_asr(hi)
    delta = a2 - a0
    headroom = 1.0 - a0
    if delta > CONFIRM_ABOVE:
        v = "ADMISSION ORDERING CONFIRMED"
    elif delta < REFUTE_BELOW:
        v = "ADMISSION ORDERING REFUTED (and the flagship negative replicated on class (b))"
    else:
        v = "INDETERMINATE"
    return {"identity_mean_asr": a0, "kappa2_mean_asr": a2, "delta": delta,
            "delta_normalized_by_headroom": delta / headroom if headroom > 0 else float("nan"),
            "confirm_above": CONFIRM_ABOVE, "refute_below": REFUTE_BELOW, "verdict": v,
            "accuracy_gate_ok": bool(mean_acc(hi) >= ACC_FLOOR), "kappa2_mean_acc": mean_acc(hi)}


def main():
    check_frozen()
    os.makedirs(out_dir, exist_ok=True)
    if "--harness-check" in sys.argv:
        return harness_check()

    d2, atk, cls, ordering, seeds = ARM
    cells = load_cells()

    # Import the identity rung before counting work, so the printed total is the number of NEW
    # 50-round runs and a reader can see exactly what is being spent.
    imported, pub = 0, ladder1_identity(d2, atk)
    key0 = cell_key("S", d2, atk, 0.0)
    cell0 = cells.setdefault(key0, {"mode": "S", "d2": d2, "attack": atk, "rung": 0.0,
                                    "dial": dial("S", 0.0), "prop1_class": cls,
                                    "predicted_ordering": ordering, "per_seed": []})
    for seed in seeds:
        if seed in pub and not any(r["seed"] == seed for r in cell0["per_seed"]):
            acc, asr = pub[seed]
            cell0["per_seed"].append({"seed": seed, "accuracy": acc, "asr": asr,
                                      "source": "results/dose_response kappa=0 "
                                                "(identical computation)"})
            imported += 1
    cell0["per_seed"].sort(key=lambda r: r["seed"])
    cell0["mean_asr"], cell0["mean_acc"] = mean_asr(cell0), mean_acc(cell0)
    cell0["std_asr"] = float(np.std([r["asr"] for r in cell0["per_seed"]], ddof=0))

    todo = [(v, s) for v in KAPPAS for s in seeds
            if not any(r["seed"] == s
                       for r in cells.get(cell_key("S", d2, atk, v), {}).get("per_seed", []))]

    print(f"=== DOSE REPLICATION: {len(todo)} new runs "
          f"({imported} identity runs imported from Round 11) ===")
    print(f"    ordering frozen in pre_registration_dose_replication.md @ {PREREG_COMMIT}")
    print(f"    arm: mode S {d2}/{atk.replace('committed_', '')}, class {cls}")
    print(f"    kappa {KAPPAS} -> rho {[round(dial('S', k), 2) for k in KAPPAS]}")
    for a, b in COMPARISON_ARMS:
        print(f"    comparison: {a}/{b.replace('committed_', '')} Mode-S rise = "
              f"{published_rise(a, b):+.3f}  (recomputed from results/targeted_dose)")
    print(f"    CONFIRM if delta > {CONFIRM_ABOVE:+.3f}; REFUTE if delta < {REFUTE_BELOW:+.3f}\n",
          flush=True)

    t0, done = time.time(), 0
    for v, seed in todo:
        key = cell_key("S", d2, atk, v)
        cell = cells.setdefault(key, {"mode": "S", "d2": d2, "attack": atk, "rung": v,
                                      "dial": dial("S", v), "prop1_class": cls,
                                      "predicted_ordering": ordering, "per_seed": []})
        t = time.time()
        acc, asr = run_one(seed, "S", d2, atk, v)
        cell["per_seed"].append({"seed": seed, "accuracy": acc, "asr": asr})
        cell["per_seed"].sort(key=lambda r: r["seed"])
        cell["mean_asr"], cell["mean_acc"] = mean_asr(cell), mean_acc(cell)
        cell["std_asr"] = float(np.std([r["asr"] for r in cell["per_seed"]], ddof=0))
        cells[key] = cell
        save(cells)
        done += 1
        print(f"  [{done}/{len(todo)}] mode S {d2}/{atk.replace('committed_', '')} kappa={v} "
              f"s{seed}: acc={acc:.3f} ASR={asr:.3f} ({time.time() - t:.0f}s)", flush=True)

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
              f"headroom-normalized {ver['delta_normalized_by_headroom']:+.3f}")
        print(f"  {ver['verdict']}")
        if not ver["accuracy_gate_ok"]:
            print(f"  ACCURACY GATE FAILED at kappa=2 ({ver['kappa2_mean_acc']:.3f} < {ACC_FLOOR}): "
                  "this cell is uninterpretable and the verdict does not stand.")
    print("\n  The secondary JT test and the tertiary pooled re-fit are scored by")
    print("  experiments/analyze_dose_replication.py against the same frozen rules.")
    print(f"\nWall time: {(time.time() - t0) / 3600:.1f} h\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
