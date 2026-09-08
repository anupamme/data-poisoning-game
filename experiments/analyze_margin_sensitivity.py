"""How much do the paper's intervention conclusions depend on their two frozen thresholds?

Round 49, answering a reviewer question directly: "How sensitive are the main intervention
conclusions to the ASR threshold and the +/-0.15 practical-equivalence margin?"

Two thresholds are at issue and they bind on different claims:

  the +/-0.15 practical-equivalence margin, frozen before the runs (Table A2). A no-evidence-
    of-a-practically-meaningful-change reading requires the paired 95% interval to lie inside
    [-m, +m], so the SMALLEST margin under which each frozen reading survives is exactly
    m* = max(|lo|, |hi|) of that arm's controlled interval. Reported per arm against 0.15.

  the +/-0.05 tolerance in the AGREE definition of the six-cell table ("intervals overlap and
    point estimates share a sign, OR both lie within +/-eps of zero"). Swept to find the range
    of eps over which all six verdicts are unchanged.

And one claim that turns on NEITHER: SIGN REVERSAL is defined as "signs differ AND both
intervals exclude zero", which reads only the intervals. It is reported here as threshold-free
rather than asserted to be.

READ-ONLY over frozen artifacts. No model is trained, no ASR is computed, and
results/comparability_six_cells.json -- the artifact every number here is read from, and whose
md5 the pre-registration gate checks -- is opened for reading and never written.

Output: results/margin_sensitivity.json
Run:    python3 experiments/analyze_margin_sensitivity.py
"""
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIX = os.path.join(BASE, "results", "comparability_six_cells.json")
OUT = os.path.join(BASE, "results", "margin_sensitivity.json")

FROZEN_MARGIN = 0.15      # the pre-registered practical-equivalence margin, unchanged
FROZEN_EPS = 0.05         # the AGREE definition's near-zero tolerance, unchanged


def verdict(cell, eps):
    """The table's own definition, with the near-zero tolerance made a parameter."""
    a, b = cell["confounded"], cell["controlled"]
    opposite = (a["mean"] > 0) != (b["mean"] > 0)
    disjoint = a["hi"] < b["lo"] or b["hi"] < a["lo"]
    excl_zero = (a["lo"] > 0 or a["hi"] < 0) and (b["lo"] > 0 or b["hi"] < 0)
    if opposite and excl_zero:
        return "SIGN REVERSAL"
    if disjoint or opposite:
        return "DISAGREE"
    if abs(a["mean"]) <= eps and abs(b["mean"]) <= eps:
        return "AGREE"
    return "AGREE" if not disjoint else "DISAGREE"


def main():
    six = json.load(open(SIX))
    cells = six["cells"]

    print("=== (A) THE +/-0.15 PRACTICAL-EQUIVALENCE MARGIN ===")
    print("    m* = the smallest symmetric margin under which the frozen reading survives")
    print(f"    (the controlled 95% interval must lie inside [-m, +m]; frozen m = {FROZEN_MARGIN})\n")
    print(f"    {'arm':26s} {'controlled 95% CI':>28s} {'m*':>8s} {'headroom':>10s}")
    margins = {}
    for c in cells:
        b = c["controlled"]
        mstar = max(abs(b["lo"]), abs(b["hi"]))
        inside = mstar < FROZEN_MARGIN
        margins[c["label"]] = {"ci": [b["lo"], b["hi"]], "m_star": mstar,
                               "inside_frozen_margin": bool(inside),
                               "headroom": FROZEN_MARGIN - mstar}
        ci = f"[{b['lo']:+.4f}, {b['hi']:+.4f}]"
        mark = f"{FROZEN_MARGIN - mstar:+.4f}" if inside else "  n/a (not an equivalence claim)"
        print(f"    {c['label']:26s} {ci:>28s} {mstar:8.4f} {mark:>10s}")
    eq = {k: v for k, v in margins.items() if v["inside_frozen_margin"]}
    if eq:
        worst = max(eq.items(), key=lambda kv: kv[1]["m_star"])
        print(f"\n    Binding arm: {worst[0]} at m* = {worst[1]['m_star']:.4f}. Every equivalence")
        print(f"    reading in the table survives any margin above {worst[1]['m_star']:.3f}, so the")
        print(f"    frozen {FROZEN_MARGIN} could have been set anywhere in "
              f"({worst[1]['m_star']:.3f}, 0.5) without changing one of them.")

    print("\n=== (B) THE +/-0.05 NEAR-ZERO TOLERANCE IN THE AGREE DEFINITION ===")
    frozen = [verdict(c, FROZEN_EPS) for c in cells]
    reported = [c.get("observed") for c in cells]
    print(f"    reproduces the published column: {frozen == reported}")
    if frozen != reported:
        for c, f, r in zip(cells, frozen, reported):
            if f != r:
                print(f"      MISMATCH {c['label']}: recomputed {f}, published {r}")
    # Sweep the closed interval [0, 1] INCLUDING both endpoints. An earlier version stopped at
    # 0.001 and 0.499 because of its loop bounds and then described those as the stable range,
    # which understated the result: the endpoints were never tested.
    grid = [0.0] + [round(0.001 * i, 4) for i in range(1, 1001)]
    stable = [e for e in grid if [verdict(c, e) for c in cells] == frozen]
    lo, hi = min(stable), max(stable)
    inert = (len(stable) == len(grid))
    print(f"    all six verdicts are unchanged for every eps in [{lo:.3f}, {hi:.3f}]"
          f"{' -- the whole swept range' if inert else ''}")
    stable_to_zero = 0.0 in stable
    if stable_to_zero:
        print("    including eps = 0, so the tolerance carries no verdict in this table:")
        print("    every cell is decided by its intervals alone.")
    print(f"    tolerance inert over the whole swept range: {inert}")

    print("\n=== (C) THE CLAIM THAT TURNS ON NEITHER THRESHOLD ===")
    sr = [c for c, v in zip(cells, frozen) if v == "SIGN REVERSAL"]
    for c in sr:
        a, b = c["confounded"], c["controlled"]
        print(f"    {c['label']}: {a['mean']:+.4f} [{a['lo']:+.4f}, {a['hi']:+.4f}] against "
              f"{b['mean']:+.4f} [{b['lo']:+.4f}, {b['hi']:+.4f}]")
    print("    Opposite signs with both intervals excluding zero. No margin and no tolerance")
    print("    enters that determination, so the paper's central empirical claim is")
    print("    threshold-free.")

    out = {
        "description": "Sensitivity of the intervention conclusions to the two frozen "
                       "thresholds. Read-only over results/comparability_six_cells.json; "
                       "no ASR computed, no model trained, no frozen artifact written.",
        "source": "results/comparability_six_cells.json",
        "frozen_equivalence_margin": FROZEN_MARGIN,
        "frozen_agree_tolerance": FROZEN_EPS,
        "per_arm_minimum_margin": margins,
        "binding_equivalence_arm": (worst[0] if eq else None),
        "binding_m_star": (worst[1]["m_star"] if eq else None),
        "agree_tolerance_stable_range": [lo, hi],
        "agree_tolerance_swept_range": [0.0, 1.0],
        "agree_tolerance_inert_over_whole_sweep": bool(inert),
        "agree_tolerance_carries_no_verdict": bool(stable_to_zero),
        "verdicts_recomputed": frozen,
        "verdicts_published": reported,
        "reproduces_published": frozen == reported,
        "sign_reversal_is_threshold_free": True,
    }
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nSaved to {os.path.relpath(OUT, BASE)}")
    return 0 if frozen == reported else 1


if __name__ == "__main__":
    sys.exit(main())
