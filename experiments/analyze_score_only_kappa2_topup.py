"""Arm A: score the score-only magnitude control at n=20 against its own frozen rule.

WHAT THIS SCORES. experiments/pre_registration_score_only_kappa2_n20.md (frozen alone at 15d02e4,
carrying the margin, floor and verdict labels forward unchanged from 35788d9) froze

    Delta = mean over the paired seeds of [ ASR_s(kappa=2, score-only) - ASR_s(kappa=0) ]

on cifar10/cifar_cnn, krum, committed_scaling, seeds 42-61, n=20, decided by |Delta| against the
margin of 0.15 -- THE RULE IS ON THE MEAN, as frozen -- with the paired two-sided 95% Student-t
interval reported beside it and its relation to BOTH +/-0.15 AND zero stated, because an interval can
sit inside the margin and still exclude zero.

BOTH LEGS ARE AT THE SAME 20 SEEDS, AND THIS FILE ASSERTS IT RATHER THAN ASSUMING IT. A top-up that
moves only the minuend hides a mixed-n comparison in the subtrahend, which is how a Delta acquires a
spurious effect on a bit-exact no-op. The four legs, each read with its own subscripted keys and never
a defaulted .get:

    kappa=2, seeds 47-61  results/score_only_kappa2_topup/  <- computed by Arm A
    kappa=2, seeds 42-46  results/score_only/               <- published
    kappa=0, seeds 47-61  results/dose_seed_topup/          <- imported; score-only at the identity IS
    kappa=0, seeds 42-46  results/score_only/                  krum alone, asserted by --harness-check

THE VERDICT LITERALS ARE READ FROM THE ARTIFACT, NOT RETYPED HERE. results/score_only_kappa2_topup/
summary.json carries verdict_labels_frozen; this file selects among them and prints the selected one
verbatim beside the measured direction. If a label is right in its stated label and wrong in its
stated mechanism, both are printed and neither is amended -- the freeze is not edited after the
numbers exist.

    PYTHONPATH=. python3 -m experiments.analyze_score_only_kappa2_topup
"""

import os
import sys
import json

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
# t_crit rather than a literal table: the small tables in this repository stop at df=9 and this arm
# needs t_19. analyze_headline_cis is import-safe (module level is constants only).
from experiments.analyze_headline_cis import t_crit  # noqa: E402

ARM = "results/score_only_kappa2_topup/summary.json"
PUB = "results/score_only/summary.json"
BASE_TOPUP = "results/dose_seed_topup/summary.json"

CELL_K2_TOPUP = "doseS_kappa2.0_then_krum|committed_scaling|score_only"
CELL_K2_PUB = "doseS_kappa2.0_then_krum|committed_scaling"
CELL_K0 = "doseS_kappa0.0_then_krum|committed_scaling"

SEEDS = list(range(42, 62))


def load(rel):
    p = os.path.join(BASE, rel)
    if not os.path.exists(p):
        sys.exit(f"REFUSING TO SCORE: {rel} is absent. This analyzer recomputes nothing.")
    return json.load(open(p))


def leg(doc, cell, rel):
    """seed -> (asr, accuracy) for one cell, keys named at the call site and never defaulted."""
    cells = doc.get("cells", {})
    if cell not in cells:
        sys.exit(f"REFUSING TO SCORE: {rel} has no cell {cell!r}. Present: {sorted(cells)}")
    out = {}
    for r in cells[cell]["per_seed"]:
        out[int(r["seed"])] = (float(r["asr"]), float(r["accuracy"]))
    return out


def ci(vals, label):
    v = np.asarray(vals, dtype=float)
    n = len(v)
    mean = float(v.mean())
    sd = float(v.std(ddof=1))
    half = float(t_crit(n) * sd / np.sqrt(n))
    return {"label": label, "n": n, "mean": mean, "sd": sd, "half_width": half,
            "t_crit": float(t_crit(n)), "lo": mean - half, "hi": mean + half}


def show(c):
    print(f"  {c['label']:<34s} n={c['n']:2d}  mean {c['mean']:+.6f}  sd {c['sd']:.6f}  "
          f"95% CI [{c['lo']:+.4f}, {c['hi']:+.4f}]  (t_{c['n'] - 1}={c['t_crit']:.3f})")


def main():
    arm = load(ARM)
    pub = load(PUB)
    bas = load(BASE_TOPUP)

    margin = float(arm["config"]["margin"])
    acc_floor = float(arm["config"]["acc_floor"])
    labels = arm["verdict_labels_frozen"]
    n5 = arm["published_n5_contrast"]

    print("=== ARM A: the score-only magnitude control at n=20 ===")
    print(f"    {arm['dataset']}/{arm['model']}, {arm['arm']['d2']}/{arm['arm']['attack']}, "
          f"kappa=0 -> kappa={arm['config']['contrast_rung']}")
    print(f"    frozen at {arm['prereg_commit']} ({arm['prereg']}), parent "
          f"{arm['parent_prereg_commit']}")
    print(f"    margin {margin} ON THE MEAN as frozen; ACC_FLOOR {acc_floor} on the rung mean\n")

    # --- assemble both legs at the same seeds -------------------------------------------------
    k2 = dict(leg(arm, CELL_K2_TOPUP, ARM))          # 47-61, computed by this arm
    k2.update(leg(pub, CELL_K2_PUB, PUB))            # 42-46, published
    k0 = dict(leg(bas, CELL_K0, BASE_TOPUP))         # 47-61, imported
    k0.update(leg(pub, CELL_K0, PUB))                # 42-46, published

    print("=== BOTH LEGS, SEED SETS ASSERTED RATHER THAN ASSUMED ===")
    print(f"  kappa=2 seeds: n={len(k2)}  {min(k2)}-{max(k2)}")
    print(f"  kappa=0 seeds: n={len(k0)}  {min(k0)}-{max(k0)}")
    want = set(SEEDS)
    if set(k2) != want or set(k0) != want:
        sys.exit("REFUSING TO SCORE: the two legs are not on the same 20 seeds.\n"
                 f"  kappa=2 only: {sorted(set(k2) - set(k0))}\n"
                 f"  kappa=0 only: {sorted(set(k0) - set(k2))}\n"
                 f"  missing from 42-61: {sorted(want - (set(k2) & set(k0)))}\n"
                 "  A Delta needs BOTH legs at the same seeds; a mixed-n subtrahend is not a "
                 "paired contrast.")
    print("  [OK] both legs on exactly seeds 42-61, paired, n=20\n")

    # --- the accuracy gate, before any verdict ------------------------------------------------
    acc_k2 = float(np.mean([k2[s][1] for s in SEEDS]))
    acc_k0 = float(np.mean([k0[s][1] for s in SEEDS]))
    print("=== GATE: ACCURACY FLOOR ON THE RUNG MEAN ===")
    print(f"  mean clean accuracy  kappa=2 {acc_k2:.4f}   kappa=0 {acc_k0:.4f}   floor {acc_floor}")
    if acc_k2 < acc_floor:
        print(f"\n  GATE FAILED: kappa=2 mean accuracy {acc_k2:.4f} < {acc_floor}. The contrast is "
              "UNINTERPRETABLE and\n  NO VERDICT STANDS. It is not substituted onto another rung.")
        return 1
    print(f"  [OK] clears by {acc_k2 - acc_floor:+.4f}\n")

    # --- the primary contrast ------------------------------------------------------------------
    d = [k2[s][0] - k0[s][0] for s in SEEDS]
    c20 = ci(d, "Delta (n=20, paired)")
    c15 = ci([k2[s][0] - k0[s][0] for s in range(47, 62)], "  new seeds 47-61 alone")
    c05 = ci([k2[s][0] - k0[s][0] for s in range(42, 47)], "  published seeds 42-46 alone")

    print("=== PRIMARY: Delta = ASR(kappa=2, score-only) - ASR(kappa=0) ===")
    show(c20)
    show(c15)
    show(c05)
    print(f"\n  mean ASR  kappa=2 {np.mean([k2[s][0] for s in SEEDS]):.6f}   "
          f"kappa=0 {np.mean([k0[s][0] for s in SEEDS]):.6f}")

    # --- the frozen rule, on the mean ----------------------------------------------------------
    m = c20["mean"]
    if m > margin:
        key, direction = "refuted", f"Delta {m:+.6f} > +{margin}"
    elif m < -margin:
        key, direction = "indeterminate", f"Delta {m:+.6f} < -{margin}"
    else:
        key, direction = "flat", f"|Delta| = {abs(m):.6f} < {margin}"

    print("\n=== THE FROZEN VERDICT LITERAL, PRINTED BESIDE THE MEASURED DIRECTION ===")
    print(f"  measured: {direction}")
    print(f"  label:    {key}")
    print(f"  literal:  {labels[key]}")

    # Both relations stated: an interval can sit inside the margin and still exclude zero.
    inside = (c20["lo"] > -margin) and (c20["hi"] < margin)
    excl0 = (c20["lo"] > 0) or (c20["hi"] < 0)
    print("\n=== THE INTERVAL'S RELATION TO BOTH +/-0.15 AND ZERO, BOTH STATED ===")
    print(f"  interval [{c20['lo']:+.4f}, {c20['hi']:+.4f}] lies entirely inside +/-{margin}: "
          f"{'YES' if inside else 'NO'}")
    print(f"  interval excludes zero: {'YES' if excl0 else 'NO'}"
          + ("" if not excl0 else "  <- a real change, inside the margin: report both, not one"))
    print(f"  projected half-width at freeze time was 0.0319; realized {c20['half_width']:.4f}")

    # --- the published n=5 verdict, unchanged, beside it ---------------------------------------
    print("\n=== THE PUBLISHED n=5 VERDICT, REPORTED UNCHANGED ALONGSIDE ===")
    print(f"  n=5 mean {n5['mean']:+.6f}  sd {n5['sd']:.6f}  verdict: {n5['verdict']}")
    print(f"  {n5['half_width_is_t19_not_t4']}")
    recomputed5 = c05["mean"]
    if abs(recomputed5 - n5["mean"]) > 1e-12:
        print(f"  NOTE: recomputed 42-46 mean {recomputed5:+.9f} != recorded {n5['mean']:+.9f}")
    else:
        print(f"  [OK] recomputed from the artifacts at {recomputed5:+.9f}, bit-equal to the record")

    # --- the reversal clause -------------------------------------------------------------------
    n5_inside = abs(n5["mean"]) < margin
    print("\n=== THE REVERSAL CLAUSE ===")
    if n5_inside and key != "flat":
        print("  THE PUBLISHED VERDICT DID NOT SURVIVE OUR OWN TOP-UP: the n=5 mean was inside "
              f"+/-{margin} and the n=20 mean is not.\n  This is reported at the site where that "
              "verdict appears, with the n=5 numbers retained beside it.")
    elif n5_inside and key == "flat":
        moved_in = abs(m) < abs(n5["mean"])
        print(f"  The published verdict survives the top-up. The n=20 mean moved "
              f"{'FURTHER INSIDE' if moved_in else 'nearer to'} the margin "
              f"({abs(n5['mean']):.6f} -> {abs(m):.6f}), which is reported too.")
    else:
        print("  The n=5 mean was already outside the margin; see the artifact's own record.")

    # --- the scope clauses that must travel with the number ------------------------------------
    print("\n=== SCOPE CLAUSES THAT TRAVEL WITH THIS NUMBER (from the artifact, verbatim) ===")
    for k in ("can_certify", "ladder_remains_at_n5", "oracle_scope", "trajectory_not_closed",
              "selectors_only", "departure_from_parent_non_negotiable"):
        print(f"\n  [{k}]\n    {arm[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
