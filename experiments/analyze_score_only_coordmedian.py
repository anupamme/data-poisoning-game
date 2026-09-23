"""Arm B: score the magnitude control on the SIGN-REVERSAL cell against its frozen rule.

WHAT THIS SCORES. experiments/pre_registration_score_only_coordmedian.md (frozen alone at 16a17b2)
froze two estimands on cifar10/cifar_cnn, coord_median, committed_pixel, seeds 42-61, n=20:

    primary    Delta_B   = mean_s [ ASR_s(kappa=2, score-only CM) - ASR_s(kappa=0) ]
    secondary  Delta_mag = mean_s [ ASR_s(kappa=2, uncontrolled Mode S) - ASR_s(kappa=2, score-only) ]

Delta_mag is a direct estimate of the magnitude channel's contribution to the headline rise on the
headline cell. Both arms share the kappa=0 term exactly, so it cancels and the difference of the two
contrasts is itself a paired quantity.

NO MARGIN TEST IS RUN HERE AND NONE IS IMPORTED, and that is the freeze's instruction, not an
omission: the +/-0.15 practical-equivalence margin scores arms whose claim is a NON-INCREASE, whereas
this arm's claim is about the SIGN AND SIZE of a rise. The decision is on the interval's relation to
zero and to the published interval. Importing a margin afterwards would be choosing a rule after the
numbers. This differs from Arm A, which IS a margin arm; the two must not be scored alike.

THE kappa=0 LEG IS IMPORTED THROUGH THE RUNNER'S OWN published(), NOT REIMPLEMENTED. A second
implementation of "the published leg" is a second chance to read a different cell key. published()
also returns the uncontrolled kappa=2 leg, which is exactly the secondary estimand's other term.

THE VERDICT LITERALS ARE PARSED FROM THE FROZEN PRE-REGISTRATION, not retyped, so they cannot drift
from the freeze. If the three decision rows are not found the analyzer refuses to score rather than
inventing a label.

    PYTHONPATH=. python3 -m experiments.analyze_score_only_coordmedian
"""

import os
import re
import sys
import json

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from experiments.analyze_headline_cis import t_crit                      # noqa: E402
# Imported, never retyped: published() is the runner's own reader for the frozen Mode S legs.
from experiments.run_score_only_coordmedian import (published, SEEDS, KAPPA,  # noqa: E402
                                                    ACC_FLOOR, D2, ATTACK, OUT,
                                                    PREREG, PREREG_COMMIT)

CELL = f"doseS_kappa{KAPPA}_then_{D2}|{ATTACK}|score_only"


def ci(vals, label):
    v = np.asarray(vals, dtype=float)
    n = len(v)
    mean = float(v.mean())
    sd = float(v.std(ddof=1))
    half = float(t_crit(n) * sd / np.sqrt(n))
    return {"label": label, "n": n, "mean": mean, "sd": sd, "half_width": half,
            "t_crit": float(t_crit(n)), "lo": mean - half, "hi": mean + half}


def show(c):
    print(f"  {c['label']:<40s} n={c['n']:2d}  mean {c['mean']:+.6f}  sd {c['sd']:.6f}  "
          f"95% CI [{c['lo']:+.4f}, {c['hi']:+.4f}]  (t_{c['n'] - 1}={c['t_crit']:.3f})")


def frozen_literals():
    """The three decision-rule verdicts, parsed from the frozen prereg rather than transcribed."""
    p = os.path.join(BASE, PREREG)
    if not os.path.exists(p):
        sys.exit(f"REFUSING TO SCORE: {PREREG} is absent; its labels are the decision rule.")
    out = {}
    for line in open(p):
        if not line.startswith("|") or "**" not in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 2:
            continue
        outcome, verdict = cells
        m = re.search(r"\*\*(.+?)\*\*", verdict)
        if not m:
            continue
        low = outcome.lower()
        if "> 0" in low:
            out["positive"] = (outcome, m.group(1))
        elif "contains zero" in low:
            out["zero"] = (outcome, m.group(1))
        elif "< 0" in low:
            out["negative"] = (outcome, m.group(1))
    if set(out) != {"positive", "zero", "negative"}:
        sys.exit(f"REFUSING TO SCORE: parsed {sorted(out)} from {PREREG}, expected exactly the "
                 "three decision rows. Read the freeze and fix the parser, never the freeze.")
    return out


def overlaps(a, b):
    return not (a[1] < b[0] or b[1] < a[0])


def main():
    p = os.path.join(OUT, "summary.json")
    if not os.path.exists(p):
        sys.exit(f"REFUSING TO SCORE: {p} is absent.")
    doc = json.load(open(p))
    cells = doc.get("cells", {})
    if CELL not in cells:
        sys.exit(f"REFUSING TO SCORE: no cell {CELL!r}. Present: {sorted(cells)}")

    labels = frozen_literals()
    pubcmp = doc["published_comparison"]
    pub_ctl = (pubcmp["controlled_n20_ci95"][0], pubcmp["controlled_n20_ci95"][1])
    pub_cfd = (pubcmp["confounded_n20_ci95"][0], pubcmp["confounded_n20_ci95"][1])

    print("=== ARM B: the magnitude control on the SIGN-REVERSAL cell, n=20 ===")
    print(f"    {doc['dataset']}/{doc['model']}, {D2}/{ATTACK}, kappa=0 -> kappa={KAPPA}")
    print(f"    frozen at {PREREG_COMMIT} ({PREREG})")
    print(f"    ACC_FLOOR {ACC_FLOOR} on the kappa={KAPPA} mean. NO MARGIN TEST: this arm's claim is "
          "a SIGN AND SIZE,\n    not a non-increase, so the decision is on the interval vs zero and "
          "vs the published interval.\n")

    # --- the three legs, all at the same 20 seeds ---------------------------------------------
    # Every leg is destructured into NAMED dicts at the point of the read. published() returns
    # (accuracy, asr) and this file's own cell is read as (asr, accuracy); a positional tuple shared
    # between two conventions is how accuracy gets scored as ASR, so no tuple survives past here.
    so_asr = {int(r["seed"]): float(r["asr"]) for r in cells[CELL]["per_seed"]}
    so_acc = {int(r["seed"]): float(r["accuracy"]) for r in cells[CELL]["per_seed"]}
    pub = published()
    if 0.0 not in pub or KAPPA not in pub:
        sys.exit(f"REFUSING TO SCORE: published() returned rungs {sorted(pub)}; both 0.0 and "
                 f"{KAPPA} are required.")
    k0_acc = {s: v[0] for s, v in pub[0.0].items()}
    k0_asr = {s: v[1] for s, v in pub[0.0].items()}
    unc_acc = {s: v[0] for s, v in pub[KAPPA].items()}
    unc_asr = {s: v[1] for s, v in pub[KAPPA].items()}
    so, k0, unc = so_asr, k0_asr, unc_asr

    print("=== THREE LEGS, SEED SETS ASSERTED RATHER THAN ASSUMED ===")
    print(f"  score-only kappa={KAPPA}:  n={len(so):2d}")
    print(f"  imported   kappa=0:    n={len(k0):2d}")
    print(f"  uncontrolled kappa={KAPPA}: n={len(unc):2d}")
    want = set(SEEDS)
    common = set(so) & set(k0)
    if common != want:
        sys.exit("REFUSING TO SCORE: the primary's two legs are not both on seeds 42-61.\n"
                 f"  score-only only: {sorted(set(so) - set(k0))}\n"
                 f"  kappa=0 only:    {sorted(set(k0) - set(so))}\n"
                 f"  missing: {sorted(want - common)}\n"
                 "  A Delta needs BOTH legs at the same seeds.")
    print(f"  [OK] primary paired on exactly seeds {min(want)}-{max(want)}, n={len(want)}\n")

    # --- GATE: the import must reproduce the PUBLISHED delta it is scored against --------------
    # The two published legs this analyzer imports are exactly the legs the recorded
    # controlled_n20_delta was computed from, so their paired mean MUST reproduce it. This is the
    # cheapest decisive test that the right field was read from the right rung: an accuracy/ASR
    # transposition, a wrong cell key or a wrong rung all break it, and none of them is visible in
    # a value that merely looks like a probability.
    recon = float(np.mean([unc_asr[s] - k0_asr[s] for s in SEEDS]))
    print("=== GATE: THE IMPORTED LEGS MUST REPRODUCE THE PUBLISHED CONTROLLED DELTA ===")
    print(f"  recomputed mean[ASR(kappa={KAPPA}, Mode S) - ASR(kappa=0)] = {recon:+.6f}")
    print(f"  recorded  published_comparison.controlled_n20_delta       = "
          f"{pubcmp['controlled_n20_delta']:+.6f}")
    if abs(recon - pubcmp["controlled_n20_delta"]) > 1e-6:
        sys.exit(f"\n  REFUSING TO SCORE: the imported legs do not reproduce the published delta "
                 f"(off by {recon - pubcmp['controlled_n20_delta']:+.6f}).\n"
                 "  Either a wrong field, a wrong cell key or a wrong rung was read. Fix the read, "
                 "never the freeze.")
    print(f"  [OK] agrees to {abs(recon - pubcmp['controlled_n20_delta']):.2e}\n")

    # --- the accuracy gate, before any verdict ------------------------------------------------
    acc = float(np.mean([so_acc[s] for s in SEEDS]))
    print("=== GATE: ACCURACY FLOOR ON THE RUNG MEAN ===")
    print(f"  mean clean accuracy at kappa={KAPPA} under the control: {acc:.4f}   floor {ACC_FLOOR}")
    if acc < ACC_FLOOR:
        print(f"\n  GATE FAILED: {acc:.4f} < {ACC_FLOOR}. The contrast is UNINTERPRETABLE, NO VERDICT "
              "STANDS,\n  and it is NOT substituted onto another rung.")
        return 1
    print(f"  [OK] clears by {acc - ACC_FLOOR:+.4f}\n")

    # --- primary -------------------------------------------------------------------------------
    dB = ci([so_asr[s] - k0_asr[s] for s in SEEDS], "Delta_B (primary)")
    print("=== PRIMARY: Delta_B = ASR(kappa=2, score-only CM) - ASR(kappa=0) ===")
    show(dB)
    print(f"  mean ASR  score-only kappa={KAPPA} {np.mean([so_asr[s] for s in SEEDS]):.6f}   "
          f"kappa=0 {np.mean([k0_asr[s] for s in SEEDS]):.6f}")

    # --- secondary -----------------------------------------------------------------------------
    sec_seeds = sorted(set(so_asr) & set(unc_asr))
    dM = ci([unc_asr[s] - so_asr[s] for s in sec_seeds], "Delta_mag (secondary)")
    print(f"\n=== SECONDARY: Delta_mag = ASR(kappa={KAPPA}, uncontrolled) - ASR(kappa={KAPPA}, "
          "score-only) ===")
    print(f"  the magnitude channel's contribution to the headline rise, n={len(sec_seeds)} "
          f"({min(sec_seeds)}-{max(sec_seeds)})")
    show(dM)
    print(f"  mean ASR  uncontrolled kappa={KAPPA} {np.mean([unc_asr[s] for s in sec_seeds]):.6f}")

    # --- the frozen decision, on the interval's relation to zero -------------------------------
    excl0 = (dB["lo"] > 0) or (dB["hi"] < 0)
    if dB["mean"] > 0 and excl0:
        key = "positive"
    elif not excl0:
        key = "zero"
    else:
        key = "negative"
    outcome, literal = labels[key]

    print("\n=== THE FROZEN VERDICT LITERAL, PRINTED BESIDE THE MEASURED DIRECTION ===")
    print(f"  measured: Delta_B {dB['mean']:+.6f}, 95% CI [{dB['lo']:+.4f}, {dB['hi']:+.4f}], "
          f"{'EXCLUDES' if excl0 else 'CONTAINS'} zero")
    print(f"  frozen row matched: {outcome}")
    print(f"  literal:  {literal}")

    # --- relation to BOTH published intervals, as the freeze requires --------------------------
    print("\n=== RELATION TO THE PUBLISHED n=20 INTERVALS (both stated) ===")
    print(f"  published CONTROLLED (Mode S, magnitude OPEN): {pubcmp['controlled_n20_delta']:+.6f} "
          f"[{pub_ctl[0]:+.4f}, {pub_ctl[1]:+.4f}]")
    print(f"  published CONFOUNDED (outcome-gated):          {pubcmp['confounded_n20_delta']:+.6f} "
          f"[{pub_cfd[0]:+.4f}, {pub_cfd[1]:+.4f}]")
    ov_ctl = overlaps((dB["lo"], dB["hi"]), pub_ctl)
    ov_cfd = overlaps((dB["lo"], dB["hi"]), pub_cfd)
    print(f"  Delta_B's interval overlaps the controlled interval: {'YES' if ov_ctl else 'NO'}"
          + ("" if ov_ctl else "  <- compatible with zero-or-not is NOT the same as compatible "
                              "with the uncontrolled arm"))
    print(f"  Delta_B's interval overlaps the confounded interval: {'YES' if ov_cfd else 'NO'}")

    # --- the two-part reversal re-adjudication, explicit and not by implication ----------------
    part_i = excl0 and dB["mean"] > 0
    part_ii = not ov_cfd
    print("\n=== THE SIGN-REVERSAL CLAIM, RE-ADJUDICATED EXPLICITLY ===")
    print(f"  (i)  Delta_B's interval excludes zero on the POSITIVE side: "
          f"{'YES' if part_i else 'NO'}")
    print(f"  (ii) Delta_B's interval FAILS TO OVERLAP the confounded interval: "
          f"{'YES' if part_ii else 'NO'}")
    if part_i and part_ii:
        print("  (i) AND (ii) BOTH HOLD: the published reversal REPRODUCES with the magnitude "
              "channel closed.")
    else:
        print("  (i) and (ii) do NOT both hold, so the published reversal DOES NOT reproduce with "
              "the magnitude\n  channel closed. Only (i) and (ii) together would, and the freeze "
              "says so in advance.")

    # --- how much of the rise the magnitude channel carries -----------------------------------
    print("\n=== HOW MUCH OF THE PUBLISHED RISE THE MAGNITUDE CHANNEL CARRIES ===")
    print(f"  published controlled rise {pubcmp['controlled_n20_delta']:+.6f}")
    print(f"  Delta_B (magnitude closed) {dB['mean']:+.6f}")
    print(f"  Delta_mag (measured)       {dM['mean']:+.6f}   "
          f"[{dM['lo']:+.4f}, {dM['hi']:+.4f}]")
    resid = pubcmp["controlled_n20_delta"] - dB["mean"]
    print(f"  controlled - Delta_B       {resid:+.6f}  (the identity rung cancels, so this is "
          "Delta_mag up to the\n                             seed sets differing; they agree to "
          f"{abs(resid - dM['mean']):.2e})")
    print("  The freeze's reachability note: the refuting branch needed the magnitude channel to "
          "account for\n  roughly 0.095 of the 0.125 rise.")

    # --- caveats that travel with the number ---------------------------------------------------
    print("\n=== CLAUSES THAT TRAVEL WITH THIS NUMBER (from the artifact, verbatim) ===")
    for k in ("rung_provenance", "estimands"):
        v = doc[k]
        if isinstance(v, dict):
            for kk, vv in v.items():
                print(f"\n  [{k}.{kk}]\n    {vv}")
        else:
            print(f"\n  [{k}]\n    {v}")
    print("\n  [magnitude closed, trajectory NOT closed]\n    Which client supplies each coordinate "
          "still changes across rungs, so the model trajectory still\n    diverges. This arm closes "
          "magnitude, not trajectory.")
    print("\n  [no transfer from the Krum arm]\n    The Krum emit-only arm's 'magnitude established "
          "inert' does not transfer here and this arm's\n    result does not transfer back: a "
          "different aggregator, a coordinate-wise rather than\n    per-client channel, and the "
          "freeze refuses the transfer in both directions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
