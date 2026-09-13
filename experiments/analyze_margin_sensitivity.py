"""How much do the paper's intervention conclusions depend on their two frozen thresholds?

Round 49, answering a reviewer question directly: "How sensitive are the main intervention
conclusions to the ASR threshold and the +/-0.15 practical-equivalence margin?"

Two thresholds are at issue and they bind on different claims:

  the +/-0.15 practical-equivalence margin, frozen before the runs (Table A2). A no-evidence-
    of-a-practically-meaningful-change reading requires the paired 95% interval to lie inside
    [-m, +m], so the SMALLEST margin under which each frozen reading survives is exactly
    m* = max(|lo|, |hi|) of that arm's controlled interval. Reported per arm against 0.15.

  the +/-0.05 tolerance in the AGREE definition of the comparability table ("intervals overlap
    and point estimates share a sign, OR both lie within +/-eps of zero"). Swept to find the range
    of eps over which every verdict is unchanged. HOW MANY CELLS: ask the artifact. This docstring
    said "six-cell" and "all six" while a seventh was being added.

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
sys.path.insert(0, BASE)   # so paired_ci95 can be IMPORTED below rather than restated here
SIX = os.path.join(BASE, "results", "comparability_six_cells.json")
OUT = os.path.join(BASE, "results", "margin_sensitivity.json")

# ONE ARM IN THE PAPER CARRIES AN EQUIVALENCE READING AND CANNOT BE IN THE ARTIFACT ABOVE.
# The ResNet18 architecture-alone arm (results/dose_resnet18/, pre-registered at 89046f6) has only the
# controlled leg -- no outcome-gated twin -- so it has no comparability cell and this ladder's universe
# excludes it structurally. That is a scope fact, not a filter, but left unprinted it would make part
# (D)'s own header ("an arm is scored only where the paper makes an equivalence reading") read as a
# claim about every such arm in the paper, which it is not. Its m* is derived below from its own frozen
# artifact, read-only, with the ladder's own paired_ci95, and printed beside the ladder. It is
# deliberately NOT eligible to become the binding arm: the supplement's binding number is a statement
# about the comparability design, and PUB_BINDING guards exactly that.
RESNET18 = os.path.join(BASE, "results", "dose_resnet18", "summary.json")
RESNET18_LEGS = ("doseS_kappa0.0_then_krum|committed_scaling",
                 "doseS_kappa2.0_then_krum|committed_scaling")

FROZEN_MARGIN = 0.15      # the pre-registered practical-equivalence margin, unchanged
FROZEN_EPS = 0.05         # the AGREE definition's near-zero tolerance, unchanged

# What the supplement already prints for part (A), asserted here so a later cell cannot move it in
# silence. A new cell may legitimately ADD an equivalence arm, and if one ever binds harder than this
# the assertion fires and the supplement is edited deliberately; what it may not do is move this number
# unnoticed. Checked, never printed as a result.
PUB_BINDING = ("krum / scaling, EMNIST", 0.0343, 0.1157)
PUB_BINDING_TOL = 5e-4    # the supplement prints four decimals

# The twenty-first review's §21.3 asks for pass/fail at these four margins by name. The frozen
# 0.15 is one rung of it and is not treated specially here.
MARGIN_LADDER = (0.05, 0.10, 0.15, 0.20)


def out_of_artifact_arm():
    """m* for the one paper arm this ladder's artifact cannot hold. None if it is not on disk yet."""
    if not os.path.exists(RESNET18):
        return None
    # The ladder's own interval, imported so this row and the comparability rows cannot be computed
    # two different ways: same paired Student-t, same ddof, same t_crit.
    from experiments.build_comparability_table import paired_ci95
    cells = json.load(open(RESNET18)).get("cells", {})
    legs = [cells.get(k) or {} for k in RESNET18_LEGS]
    if not all(leg.get("per_seed") for leg in legs):
        return None
    lo = {r["seed"]: r["asr"] for r in legs[0]["per_seed"]}
    hi = {r["seed"]: r["asr"] for r in legs[1]["per_seed"]}
    # Both legs on the SAME seeds, asserted rather than assumed: a top rung scored against an identity
    # rung with a different seed set is a mixed-n difference wearing a paired interval's clothes.
    assert sorted(lo) == sorted(hi), f"legs disagree on seeds: {sorted(lo)} vs {sorted(hi)}"
    seeds = sorted(lo)
    ci = paired_ci95([hi[s] for s in seeds], [lo[s] for s in seeds])
    return {"label": "krum / scaling, resnet18", "seeds": seeds, "n": ci["n"],
            "mean": ci["mean"], "lo": ci["lo"], "hi": ci["hi"],
            "m_star": max(abs(ci["lo"]), abs(ci["hi"])),
            "ci_contains_zero": not ci["excludes_zero"],
            "in_ladder_artifact": False,
            "reason_absent": "controlled leg only; no outcome-gated twin, so no comparability cell"}


def tex_label(label):
    """An arm label as the paper's tables write it: monospaced defense name, escaped underscore."""
    name, _, rest = label.partition(" / ")
    name = "\\texttt{%s}" % name.replace("_", "\\_") if "_" in name else name
    return f"{name} / {rest}" if rest else name


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


def complete(c):
    """A cell this script may read. Both designs present, a published verdict, and every frozen seed in.

    All three conjuncts are load-bearing and the third was added after this script silently read a
    ladder that was still running. With cell 7 mid-run it did three wrong things at once: it named that
    cell the new BINDING equivalence arm at m* = 0.1306 (headroom +0.019, against the +0.116 the
    supplement reports), it listed it in part (C) as a SIGN REVERSAL off four controlled seeds against
    five confounded ones, and it reported `reproduces_published: False` because the published verdict
    was correctly null. A sensitivity analysis that reads a partial run does not merely add a row; it
    can MOVE the binding arm, which is the one number in part (A) the supplement quotes.
    """
    return (c.get("confounded") and c.get("controlled") and c.get("observed")
            and not c.get("missing_frozen_seeds"))


def main():
    six = json.load(open(SIX))
    allcells = six["cells"]
    cells = [c for c in allcells if complete(c)]
    skipped = [c for c in allcells if not complete(c)]
    print(f"Cells in the artifact: {len(allcells)}; scored here: {len(cells)}.")
    if skipped:
        # Announced, never silent: a skipped cell must be visible as a skip and not as an absence.
        for c in skipped:
            why = ("frozen seeds still missing " + json.dumps(c["missing_frozen_seeds"])
                   if c.get("missing_frozen_seeds") else "no published verdict")
            print(f"  SKIPPED {c['label']}: {why}. Re-run this script when that ladder finishes.")
    if not cells:
        print("No complete cell to score. Refusing to emit a sensitivity range.")
        return 1
    print()

    print("=== (A) THE +/-0.15 PRACTICAL-EQUIVALENCE MARGIN ===")
    print("    m* = the smallest symmetric margin under which the frozen reading survives")
    print(f"    (the controlled 95% interval must lie inside [-m, +m]; frozen m = {FROZEN_MARGIN})\n")
    print(f"    {'arm':32s} {'controlled 95% CI':>28s} {'m*':>8s} {'headroom':>10s}")
    margins = {}
    for c in cells:
        b = c["controlled"]
        mstar = max(abs(b["lo"]), abs(b["hi"]))
        # An arm enters this part only if the paper makes an equivalence reading about it, and it makes
        # one only where the effect is NOT detected. Both conjuncts are needed and the second was added
        # after cell 7 landed: its controlled leg is +0.071 [+0.031, +0.110], which lies inside
        # [-0.15, +0.15] and so passes a TOST at the frozen margin while ALSO excluding zero. That is a
        # real and well-known TOST outcome -- significant and practically equivalent at once -- but the
        # paper reports that arm as a detected rise inside a SIGN REVERSAL, never as "no evidence of a
        # practically meaningful change". Scoring it here would have named it the binding equivalence arm
        # at m* = 0.1102 and moved part (A)'s headline from +0.1157 to +0.0398 on the strength of an
        # equivalence reading the paper does not make. The conjunct is a no-op on the published set:
        # both arms that qualified before contain zero, so the binding arm and every printed number are
        # unchanged, which is the assertion at the bottom of this run.
        contains_zero = b["lo"] <= 0.0 <= b["hi"]
        inside = mstar < FROZEN_MARGIN and contains_zero
        margins[c["label"]] = {"ci": [b["lo"], b["hi"]], "m_star": mstar,
                               "ci_contains_zero": bool(contains_zero),
                               "inside_frozen_margin": bool(inside),
                               "headroom": FROZEN_MARGIN - mstar}
        ci = f"[{b['lo']:+.4f}, {b['hi']:+.4f}]"
        if inside:
            mark = f"{FROZEN_MARGIN - mstar:+.4f}"
        elif not contains_zero and mstar < FROZEN_MARGIN:
            mark = "  n/a (effect detected; no equivalence reading is made)"
        else:
            mark = "  n/a (not an equivalence claim)"
        print(f"    {c['label']:32s} {ci:>28s} {mstar:8.4f} {mark:>10s}")
    eq = {k: v for k, v in margins.items() if v["inside_frozen_margin"]}
    binding_moved = None
    if eq:
        worst = max(eq.items(), key=lambda kv: kv[1]["m_star"])
        print(f"\n    Binding arm: {worst[0]} at m* = {worst[1]['m_star']:.4f}. Every equivalence")
        print(f"    reading in the table survives any margin above {worst[1]['m_star']:.3f}, so the")
        print(f"    frozen {FROZEN_MARGIN} could have been set anywhere in "
              f"({worst[1]['m_star']:.3f}, 0.5) without changing one of them.")
        lbl, m_pub, h_pub = PUB_BINDING
        if worst[0] != lbl or abs(worst[1]["m_star"] - m_pub) > PUB_BINDING_TOL:
            binding_moved = (f"{worst[0]} at m* = {worst[1]['m_star']:.4f}, headroom "
                             f"{FROZEN_MARGIN - worst[1]['m_star']:+.4f}")
            print(f"\n    ** THE BINDING ARM MOVED ** The supplement prints {lbl} at m* = {m_pub:.4f} "
                  f"(headroom {h_pub:+.4f}).")
            print(f"    This run says {binding_moved}. Part (A) of the supplement is now stale and must")
            print("    be edited deliberately; nothing here rewrites it.")

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
    print(f"    all {len(cells)} verdicts are unchanged for every eps in [{lo:.3f}, {hi:.3f}]"
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

    # Part (D) answers the twenty-first review's §21.3 in its own form: "show pass/fail at
    # +/-0.05, 0.10, 0.15, 0.20". Nothing is recomputed for it -- every cell is a comparison
    # between a ladder rung and the m* part (A) already derived from the same interval.
    #
    # Two things it must not do. It must not print a verdict for an arm the paper makes no
    # equivalence reading about: those arms report a DETECTED effect, and a checkmark against a
    # margin would read as "practically equivalent" where the paper says the opposite. They are
    # marked absent, per part (A)'s ci_contains_zero conjunct. And it must not let a reader think
    # the absence hides a flip, so the loud_absences line below names the arms whose m* happens to
    # fall inside the ladder's own span: coord_median/pixel at 0.1547 (0.1783 before its endpoint
    # top-up to n=20) and its CIFAR-100 twin at
    # 0.1100 would appear to flip between rungs if that conjunct were dropped, which is exactly the
    # error the conjunct exists to prevent.
    print("\n=== (D) THE REVIEWER'S MARGIN LADDER ===")
    print(f"    pass = the controlled 95% CI lies inside [-m, +m], at m in "
          f"{', '.join(f'{m:.2f}' for m in MARGIN_LADDER)}")
    print("    An arm is scored only where the paper MAKES an equivalence reading, i.e. where its")
    print("    interval contains zero. The others report a detected effect and are marked absent.\n")
    header = "    " + f"{'arm':34s}" + "".join(f"{m:>8.2f}" for m in MARGIN_LADDER) + f"{'m*':>10s}"
    print(header)
    ladder, latex = {}, []
    for c in cells:
        lbl = c["label"]
        mg = margins[lbl]
        if mg["ci_contains_zero"]:
            row = {f"{m:.2f}": bool(mg["m_star"] < m) for m in MARGIN_LADDER}
            cells_txt = "".join(f"{('pass' if row[f'{m:.2f}'] else 'FAIL'):>8s}"
                                for m in MARGIN_LADDER)
            tex = "".join(" & " + ("\\checkmark" if row[f"{m:.2f}"] else "$\\times$")
                          for m in MARGIN_LADDER)
        else:
            row = {f"{m:.2f}": None for m in MARGIN_LADDER}
            cells_txt = "".join(f"{'---':>8s}" for m in MARGIN_LADDER)
            tex = " & ---" * len(MARGIN_LADDER)
        ladder[lbl] = row
        print(f"    {lbl:34s}{cells_txt}{mg['m_star']:>10.4f}")
        latex.append(f"{tex_label(lbl)}{tex} & ${mg['m_star']:.3f}$ \\\\")
    scored = [l for l, r in ladder.items() if r[f"{MARGIN_LADDER[0]:.2f}"] is not None]
    flips = [l for l in scored
             if len({ladder[l][f'{m:.2f}'] for m in MARGIN_LADDER}) > 1]
    print(f"\n    {len(scored)} arms scored; verdicts that change anywhere on the ladder: "
          f"{len(flips)}{' (' + ', '.join(flips) + ')' if flips else ''}")
    loud_absences = [(l, margins[l]["m_star"]) for l in ladder
                     if l not in scored
                     and MARGIN_LADDER[0] < margins[l]["m_star"] < MARGIN_LADDER[-1]]
    if loud_absences:
        print("    Absent arms whose m* lies inside the ladder's own span, named so the absence")
        print("    cannot be mistaken for a hidden flip:")
        for l, ms in loud_absences:
            print(f"      {l}: m* = {ms:.4f}, reported as a detected effect, not an equivalence")

    # The arm this artifact structurally cannot hold, printed here so part (D)'s scope is on the page
    # rather than in a docstring. It does not enter `scored`, `flips`, or the binding-arm search.
    outside = out_of_artifact_arm()
    if outside:
        inside = outside["m_star"] < FROZEN_MARGIN
        print("\n    NOT IN THIS ARTIFACT, and it carries an equivalence reading:")
        print(f"      {outside['label']}: m* = {outside['m_star']:.4f} at n = {outside['n']} "
              f"({outside['lo']:+.4f}, {outside['hi']:+.4f})")
        print(f"      {outside['reason_absent']}, so the ladder above is a statement about the")
        print("      comparability design and NOT about every equivalence reading in the paper.")
        print(f"      Its m* is {'inside' if inside else 'WIDER THAN'} the frozen "
              f"{FROZEN_MARGIN} margin, so for that arm the frozen margin "
              f"{'is not' if inside else 'IS'} load-bearing.")
        if not inside:
            print("      Any sentence claiming every equivalence reading survives a margin below")
            print(f"      {outside['m_star']:.3f} is FALSE while this arm is in the paper.")

    print("\n    LaTeX rows (paste into the ladder table; this script writes no .tex file):")
    for r in latex:
        print("      " + r)

    out = {
        "description": "Sensitivity of the intervention conclusions to the two frozen "
                       "thresholds. Read-only over results/comparability_six_cells.json; "
                       "no ASR computed, no model trained, no frozen artifact written.",
        "source": "results/comparability_six_cells.json",
        "n_cells_in_artifact": len(allcells),
        "n_cells_scored": len(cells),
        "cells_skipped": [c["label"] for c in skipped],
        "frozen_equivalence_margin": FROZEN_MARGIN,
        "frozen_agree_tolerance": FROZEN_EPS,
        "per_arm_minimum_margin": margins,
        "binding_equivalence_arm": (worst[0] if eq else None),
        "binding_m_star": (worst[1]["m_star"] if eq else None),
        "binding_arm_matches_supplement": binding_moved is None,
        "binding_arm_moved_to": binding_moved,
        "equivalence_arm_requires_ci_containing_zero": True,
        "agree_tolerance_stable_range": [lo, hi],
        "agree_tolerance_swept_range": [0.0, 1.0],
        "agree_tolerance_inert_over_whole_sweep": bool(inert),
        "agree_tolerance_carries_no_verdict": bool(stable_to_zero),
        "verdicts_recomputed": frozen,
        "verdicts_published": reported,
        "reproduces_published": frozen == reported,
        "sign_reversal_is_threshold_free": True,
        "margin_ladder": list(MARGIN_LADDER),
        "margin_ladder_per_arm": ladder,
        "margin_ladder_arms_scored": scored,
        "margin_ladder_verdicts_that_change": flips,
        "margin_ladder_absent_arms_with_m_star_inside_span":
            {l: ms for l, ms in loud_absences},
        # Scope of the ladder itself, recorded so a reader of the JSON cannot mistake it for global.
        "equivalence_arm_outside_this_artifact": outside,
        "ladder_covers_every_equivalence_reading_in_the_paper": outside is None,
    }
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nSaved to {os.path.relpath(OUT, BASE)}")
    return 0 if frozen == reported else 1


if __name__ == "__main__":
    sys.exit(main())
