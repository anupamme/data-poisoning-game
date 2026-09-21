"""Is the LOWER leg of each equivalence claim a test, or is it satisfied by arithmetic?

Round 76, answering the sixteenth review's objection to the paper's equivalence language directly.

The paper reads practical equivalence off a two-sided TOST against a frozen +/-0.15 margin: the paired
interval must lie inside [-0.15, +0.15]. That is two claims, "the effect does not rise past +0.15" and
"the effect does not fall past -0.15", and the second one can be unfalsifiable. ASR is bounded below by
0, so on a cell whose identity rung already sits at mean ASR a, the paired mean difference obeys

    mean_i (asr_dosed_i - asr_identity_i)  >=  -mean_i asr_identity_i  =  -a

for ANY dosed outcome whatsoever, including a perfect defense. So when a < 0.15 the lower leg is
satisfied before a single run happens: it carries no information, only the upper leg is a test, and the
honest form of the claim is a one-sided non-increase reported with its 95% upper bound.

This script measures `a` for every arm the paper makes an equivalence reading about, and reports which
lower legs are tests. It does NOT weaken, re-derive or re-margin any published interval; every interval
here is recomputed with the same imported estimator its own table uses and asserted equal to the
published one before anything is printed.

WHICH ARMS, IN THREE POPULATIONS THAT ARE NEVER POOLED. The paper scores its Mode S arms in two places
under two conventions, and one arm can appear in both at different seed blocks, so a single pooled count
would double-count it. A third population exists because two of the paper's equivalence claims are about
arms neither of those two places scores.

  GROUP A -- the comparability ladder's cells, both designs each, at the seed block that ladder scores
    (results/comparability_six_cells.json), plus the one equivalence arm that artifact structurally
    cannot hold: the ResNet18 architecture arm has no outcome-gated twin, so it has no comparability
    cell and is imported from analyze_margin_sensitivity.out_of_artifact_arm(). This is the population
    App. G's margin ladder reasons about, at 95%.

  GROUP B -- every Mode S arm in the paper at the seed counts already on disk, which is App. F's
    Table 4 (tab:tost). The arm list, the rung constants and the pairing are IMPORTED from
    experiments/analyze_tost_existing.py, that table's own emitter, rather than relisted: two lists of
    "the paper's equivalence arms" that can drift apart is how one of them gets forgotten. TOST there
    decides on the 90% interval; the reachability bound is a statement about the paired mean and so is
    convention-free, and both intervals are reported.

  GROUP C -- the two equivalence claims whose arm neither list above holds: Mode M, the coordinate-
    masking ladder that is the paper's one non-rescaling transformation class (n=10), and the 200-round
    ResNet18 replication (n=3). These were found by auditing the PROSE rather than the artifacts, with
    experiments/audit_equivalence_claims.py, which enumerates every paragraph in both documents that
    makes an equivalence claim. An emitter whose population comes only from other emitters inherits
    their blind spots; this group is what that audit turned up.

AND THEN A PAPER-WIDE COUNT OVER DISTINCT ARMS, which is the only count a response letter may quote. The
three groups hold 8 claim-rows but only 5 distinct arms, because the flagship and its EMNIST replication
each appear in more than one group at more than one seed block. Quoting a row count as an arm count would
overstate the finding.

THE SAME CELL CAN CARRY TWO REFERENTS and they are not interchangeable between sentences. The flagship
appears as group B's n=5 ladder arm (identity 0.0618, 95% upper bound +0.026) and as group A's n=20
top-up (identity 0.0439, upper bound +0.012).

WHICH SEEDS. Within each group the identity mean is taken over exactly the seed block that group's
SCORED contrast is computed on, because the bound above is a statement about that specific paired mean.
An identity mean over a different seed set bounds a different quantity.

READ-ONLY. No model is trained, no ASR is computed, and every artifact this reads is opened for reading
and never written.

Output: results/margin_reachability.json   (a new path; no existing artifact is touched)
Run:    PYTHONPATH=. python3 -m experiments.measure_margin_reachability
"""
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

# Every constant, arm list and estimator below is IMPORTED. The margin is the paper's frozen margin,
# not a parameter of this script; t_crit is the same Student-t the published intervals used (never
# T95[n-1], whose literal table stops at df = 9); the cell lists and the seed-merge rules belong to the
# two tables themselves, so this script cannot disagree with either about which seeds an arm was scored
# on or about which arms there are.
from experiments.run_targeted_dose import ACC_FLOOR, EQUIV_MARGIN                 # noqa: E402
from experiments.analyze_headline_cis import t_crit                               # noqa: E402
from experiments.analyze_comparability import CELLS, LO, HI, series               # noqa: E402
from experiments.analyze_margin_sensitivity import (                              # noqa: E402
    RESNET18, RESNET18_LEGS, out_of_artifact_arm)
# App. F's Table 4 in full: its arms, its rungs, its pooled-source merge and its TOST.
from experiments.analyze_tost_existing import (                                   # noqa: E402
    ARMS as TOST_ARMS, load as tost_load, paired as tost_paired, tost, MARGIN as TOST_MARGIN)

SIX = os.path.join(BASE, "results", "comparability_six_cells.json")
OUT = os.path.join(BASE, "results", "margin_reachability.json")

# GROUP C: the arms that carry an equivalence claim in the paper and sit in NEITHER of the two lists
# above. Found by auditing the prose rather than the artifacts -- experiments/audit_equivalence_claims.py
# enumerates every paragraph making an equivalence claim, and these two were claims whose arm no emitter
# here scored. They need an explicit row because no existing script lists them, so each one asserts the
# published interval it is quoted by; a row whose recomputation drifts from the printed value aborts the
# whole report rather than being quietly rescored.
#   (label, results dir, key format, identity rung, top rung, published (mean, ci) or None, quoted at)
GROUP_C = [
    ("Mode M / coordinate masking", "dose_mask", "doseM_m{r}_then_krum|committed_scaling",
     "0.0", "0.8", (-0.016, (-0.051, 0.019)),
     "main.tex:1882, the second transformation class at n=10"),
    ("krum / scaling, resnet18, 200 rounds", "dose_resnet18_converged",
     "doseS_kappa{r}_then_krum|committed_scaling", "0.0", "2.0", (None, (-0.182, 0.138)),
     "main.tex:1839, the longer-horizon replication at n=3"),
]
# Emit-only is deliberately NOT here: its verdict is scored against the separate frozen inert and
# additivity margins, not against EQUIV_MARGIN, so it makes no claim this bound speaks to.

# The published intervals this script must reproduce before it is allowed to say anything about their
# legs. Group A's come from the artifact itself; group B's are the ROUNDED values main.tex prints in
# Table 4, keyed by the emitter's own arm label so a renamed arm fails loudly instead of going
# unchecked. PUB_TOL is a display tolerance on a printed 3-decimal number, not a statistical one, and
# it is not a substitute for group A's exact CI_TOL assertion.
CI_TOL = 1e-9
PUB_TOL = 5e-4
TOST_PUBLISHED = {
    "Krum / scaling (flagship)": (-0.026, (-0.078, 0.026), "main.tex:698, :720, :940, tab:tost"),
    "Krum / scaling (score-only)": (-0.023, (-0.108, 0.061), "main.tex:1847, tab:tost"),
    "Krum / scaling (EMNIST-byclass)": (-0.017, (-0.033, -0.001), "main.tex:940, :1820, tab:tost"),
    "Krum / scaling (EMNIST-byclass, $n{=}5$)": (-0.013, (-0.034, 0.009), "tab:tost, the pooled row"),
    "Krum / scaling (ResNet18)": (-0.041, (-0.183, 0.101), "main.tex:940, :1830, tab:tost"),
    "Reputation / scaling": (0.178, (-0.112, 0.469), "tab:tost, a positive control"),
    "cos_krum / pixel": (-0.425, (-0.732, -0.118), "tab:tost, a positive control"),
    "coord_median / pixel": (0.098, (0.018, 0.178), "main.tex:1746, tab:tost"),
    "coord_median / pixel (CIFAR-100)": (0.071, (0.031, 0.110), "main.tex:853, :3141, tab:tost"),
}


def leg_block(sources, d2, atk):
    """(identity {seed: asr}, dosed {seed: asr}, scored seed block) for one design of one cell.

    The block is the SCORED one: the seeds this design carries at BOTH endpoint rungs, at its own full
    n. That is `unpaired_*` in analyze_comparability.contrast, which is the frozen convention and the
    only one under which the published cells reproduce.
    """
    lo = series(sources, d2, atk, LO)
    hi = series(sources, d2, atk, HI)
    return lo, hi, sorted(set(lo) & set(hi))


def ci95(lo, hi, seeds):
    d = np.array([hi[s] - lo[s] for s in seeds], dtype=float)
    n = len(d)
    m = float(d.mean())
    hw = t_crit(n) * float(d.std(ddof=1) / np.sqrt(n))
    return {"mean": m, "n": n, "lo": m - hw, "hi": m + hw}


def reachability(identity, seeds, ci):
    """The whole point of the script, for one (arm, design).

    `fall_available` is the largest fall the paired mean could possibly take on this seed block, which
    is the identity mean itself because ASR cannot go below zero. If it does not exceed the margin, the
    lower TOST leg is satisfied by arithmetic for every conceivable outcome and is not a test.
    """
    a = float(np.mean([identity[s] for s in seeds]))
    return {
        "identity_mean_asr": a,
        "fall_available": a,
        "margin": EQUIV_MARGIN,
        "lower_leg_is_a_test": bool(a > EQUIV_MARGIN),
        "lower_leg_slack": a - EQUIV_MARGIN,
        # What survives when the lower leg does not: the one-sided reading, and its bound.
        "upper_bound": ci["hi"],
        "upper_bound_below_margin": bool(ci["hi"] < EQUIV_MARGIN),
        "ci_contains_zero": bool(ci["lo"] <= 0.0 <= ci["hi"]),
        "two_sided_inside_margin": bool(max(abs(ci["lo"]), abs(ci["hi"])) < EQUIV_MARGIN),
    }


def _fmt(row, label, design):
    return (f"    {label:34s} {design:11s} {row['identity_mean_asr']:7.4f} "
            f"{row['fall_available']:7.4f} "
            f"{('TEST' if row['lower_leg_is_a_test'] else 'arithmetic'):>11s} "
            f"[{row['ci'][0]:+.4f},{row['ci'][1]:+.4f}] "
            f"{('yes' if row['two_sided_inside_margin'] and row['ci_contains_zero'] else '-'):>7s}")


def group_a(mismatches):
    """The comparability ladder's cells, both designs, plus the arm that ladder cannot hold."""
    rows = []
    published = {c["label"]: c for c in json.load(open(SIX))["cells"]}
    for label, d2, atk, conf_src, ctrl_src, training, _adm in CELLS:
        if label not in published:
            print(f"    SKIPPED {label}: not in {os.path.relpath(SIX, BASE)} yet.")
            continue
        for design, sources in (("confounded", conf_src), ("controlled", ctrl_src)):
            ident, dosed, seeds = leg_block(sources, d2, atk)
            if len(seeds) < 2:
                print(f"    SKIPPED {label} / {design}: {len(seeds)} paired seed(s).")
                continue
            ci = ci95(ident, dosed, seeds)
            pub = published[label][design]
            # The identity mean is only attached to the right leg if the interval recomputed from the
            # same two legs IS the published one. Asserted, never assumed: this is what stops a
            # reachability number from being quoted beside an interval it does not belong to.
            for k in ("mean", "lo", "hi", "n"):
                if abs(ci[k] - pub[k]) > CI_TOL:
                    mismatches.append((label, design, k, ci[k], pub[k]))
            row = reachability(ident, seeds, ci)
            row.update({"arm": label, "design": design, "seeds": seeds, "n": ci["n"],
                        "mean": ci["mean"], "ci": [ci["lo"], ci["hi"]], "group": "A",
                        "training": bool(training), "in_comparability_artifact": True,
                        "reproduces_published": not any(m[0] == label and m[1] == design
                                                        for m in mismatches)})
            rows.append(row)
            print(_fmt(row, label, design))

    outside = out_of_artifact_arm()
    if outside is not None:
        legs = [json.load(open(RESNET18))["cells"][k] for k in RESNET18_LEGS]
        ident = {int(r["seed"]): float(r["asr"]) for r in legs[0]["per_seed"]}
        row = reachability(ident, list(outside["seeds"]), outside)
        row.update({"arm": "krum / scaling, resnet18", "design": "controlled",
                    "seeds": list(outside["seeds"]), "n": outside["n"],
                    "mean": outside["mean"], "ci": [outside["lo"], outside["hi"]], "group": "A",
                    "in_comparability_artifact": False,
                    "reason_absent": outside["reason_absent"]})
        rows.append(row)
        print(_fmt(row, "krum / scaling, resnet18", "controlled")
              + f"\n      {outside['reason_absent']}.")
    return rows


def group_b(mismatches):
    """App. F's Table 4 in full: every Mode S arm in the paper at the seed counts on disk.

    The arm list, the rungs and the pooled-source merge are analyze_tost_existing's own, so this cannot
    score a different set of arms, a different pair of rungs or a different seed block than the table
    it is attached to. The identity mean is read from the same `paired` call that produces the
    difference, so a subtrahend can never come from a different seed count than its minuend.
    """
    rows = []
    for arm in TOST_ARMS:
        label, dirname, keyfmt = arm[0], arm[1], arm[2]
        extra = arm[3] if len(arm) > 3 else ()
        cells = tost_load(dirname)
        if cells is None:
            print(f"    SKIPPED {label}: results/{dirname}/summary.json absent.")
            continue
        p = tost_paired(cells, keyfmt, extra)
        if p is None or len(p["seeds"]) < 2:
            print(f"    SKIPPED {label}: no paired endpoint block.")
            continue
        t = tost(p["diff"])
        ident = {s: float(a) for s, a in zip(p["seeds"], p["asr_lo"])}
        ci = {"mean": t["mean"], "n": t["n"], "lo": t["ci95"][0], "hi": t["ci95"][1]}
        if label in TOST_PUBLISHED:
            pub_mean, pub_ci, where = TOST_PUBLISHED[label]
            for k, want in (("mean", pub_mean), ("lo", pub_ci[0]), ("hi", pub_ci[1])):
                if abs(ci[k] - want) > PUB_TOL:
                    mismatches.append((label, "tab:tost", k, ci[k], want))
        else:
            # A new arm in the table's own list that this script has no printed value for would
            # otherwise be scored silently against nothing. Say so rather than let it pass.
            where = None
            mismatches.append((label, "tab:tost", "published_value", None,
                               "MISSING from TOST_PUBLISHED"))
        row = reachability(ident, list(p["seeds"]), ci)
        row.update({"arm": label, "design": "tab:tost", "seeds": list(map(int, p["seeds"])),
                    "n": t["n"], "mean": t["mean"], "ci": list(t["ci95"]),
                    "ci90_tost": list(t["ci90"]), "p_tost": t["p_tost"],
                    "p_lower_leg": t["p_lower"], "p_upper_leg": t["p_upper"],
                    # tost() returns p_tost = max(p_lower, p_upper), so ONE of the two legs sets the
                    # published p. On an arm whose lower leg is arithmetic that max is routinely
                    # attained by the arithmetic leg, which means the published p is set by the
                    # hypothesis ASR's floor already excludes, and the p for the claim actually being
                    # made -- no meaningful RISE -- is the smaller of the two. Recorded per arm rather
                    # than argued, because it decides which number a relabelled sentence may print.
                    "p_tost_attained_by": ("lower" if t["p_lower"] >= t["p_upper"] else "upper"),
                    "p_tost_set_by_the_arithmetic_leg":
                        bool(t["p_lower"] >= t["p_upper"] and not row["lower_leg_is_a_test"]),
                    "tost_equivalent": bool(t["equivalent"]), "detected": bool(t["detected"]),
                    "group": "B", "source": f"results/{dirname}/summary.json", "quoted_at": where,
                    "published_rounded": (None if label not in TOST_PUBLISHED else
                                          {"mean": TOST_PUBLISHED[label][0],
                                           "ci": list(TOST_PUBLISHED[label][1])}),
                    "reproduces_published": not any(m[0] == label for m in mismatches),
                    # The bound is a statement about the paired MEAN, so it is the same statement
                    # under either interval convention: on an arm whose lower leg is arithmetic at
                    # 95%, the 90% lower leg is arithmetic too, a fortiori, being the narrower
                    # interval. Recorded so App. F's 90% convention is not left looking unexamined.
                    "lower_leg_is_a_test_under_either_convention":
                        row["lower_leg_is_a_test"]})
        rows.append(row)
        print(_fmt(row, label, f"n={t['n']}"))
    return rows


def group_c(mismatches):
    """The equivalence claims whose arm neither of the two lists above scores."""
    rows = []
    for label, dirname, keyfmt, r_lo, r_hi, pub, where in GROUP_C:
        cells = tost_load(dirname)
        if cells is None:
            print(f"    SKIPPED {label}: results/{dirname}/summary.json absent.")
            continue
        L = {r["seed"]: float(r["asr"]) for r in (cells.get(keyfmt.format(r=r_lo)) or {}).get(
            "per_seed", [])}
        H = {r["seed"]: float(r["asr"]) for r in (cells.get(keyfmt.format(r=r_hi)) or {}).get(
            "per_seed", [])}
        seeds = sorted(set(L) & set(H))
        if len(seeds) < 2:
            print(f"    SKIPPED {label}: {len(seeds)} paired seed(s).")
            continue
        ci = ci95(L, H, seeds)
        pub_mean, pub_ci = pub
        for k, want in (("mean", pub_mean), ("lo", pub_ci[0]), ("hi", pub_ci[1])):
            if want is not None and abs(ci[k] - want) > PUB_TOL:
                mismatches.append((label, "group C", k, ci[k], want))
        row = reachability(L, seeds, ci)
        row.update({"arm": label, "design": f"n={ci['n']}", "seeds": seeds, "n": ci["n"],
                    "mean": ci["mean"], "ci": [ci["lo"], ci["hi"]], "group": "C",
                    "source": f"results/{dirname}/summary.json", "quoted_at": where,
                    "rungs": [r_lo, r_hi],
                    "published_rounded": {"mean": pub_mean, "ci": list(pub_ci)},
                    "reproduces_published": not any(m[0] == label for m in mismatches)})
        rows.append(row)
        print(_fmt(row, label, f"n={ci['n']}") + f"\n      {where}.")
    return rows


# A claim row's label -> the distinct ARM it is about, so the paper-wide count is over arms and not over
# arm-rows. Three of the five arms are scored more than once, under two naming conventions and at
# different seed blocks, and a letter that quoted the row count would overstate the finding.
CANON = {
    "krum / scaling": "krum / scaling (the flagship cell)",
    "Krum / scaling (flagship)": "krum / scaling (the flagship cell)",
    "Krum / scaling (score-only)": "krum / scaling, score-only control",
    "krum / scaling, EMNIST": "krum / scaling, EMNIST-byclass",
    "Krum / scaling (EMNIST-byclass)": "krum / scaling, EMNIST-byclass",
    "Krum / scaling (EMNIST-byclass, $n{=}5$)": "krum / scaling, EMNIST-byclass",
    "krum / scaling, resnet18": "krum / scaling, ResNet18",
    "Krum / scaling (ResNet18)": "krum / scaling, ResNet18",
    "krum / scaling, resnet18, 200 rounds": "krum / scaling, ResNet18, 200 rounds",
    "Mode M / coordinate masking": "Mode M, coordinate masking",
    "reputation / scaling": "reputation / scaling",
    "Reputation / scaling": "reputation / scaling",
    "cos_krum / pixel": "cos_krum / pixel",
    "coord_median / pixel": "coord_median / pixel",
    "coord_median / pixel, CIFAR-100": "coord_median / pixel, CIFAR-100",
    "coord_median / pixel (CIFAR-100)": "coord_median / pixel, CIFAR-100",
    "coord_median / scaling": "coord_median / scaling",
}


def paper_wide(all_rows):
    """The de-duplicated count: distinct ARMS carrying an equivalence claim, and how many are arithmetic.

    This is the number a response letter may quote. The three groups' counts may not be added together.
    """
    unknown = sorted({r["arm"] for r in all_rows if r["arm"] not in CANON})
    if unknown:
        sys.exit("REFUSING TO REPORT a paper-wide count: no CANON entry for " + "; ".join(unknown)
                 + ". A new arm must be mapped to its distinct-arm name first.")
    claims, arith, keep = {}, {}, {}
    for r in all_rows:
        if not (r["ci_contains_zero"] and r["two_sided_inside_margin"]):
            continue
        k = CANON[r["arm"]]
        claims.setdefault(k, []).append(f"{r['group']}/{r['design']} n={r['n']}")
        (arith if not r["lower_leg_is_a_test"] else keep).setdefault(k, []).append(r["group"])
    # An arm counts as keeping the two-sided reading only if NO seed block of it has an arithmetic
    # lower leg. The conservative direction: one unreachable block is enough to require the qualifier.
    kept = [k for k in claims if k not in arith]
    print("\n=== THE PAPER-WIDE COUNT, OVER DISTINCT ARMS, THE ONLY ONE A LETTER MAY QUOTE ===")
    print(f"    {len(all_rows)} arm-rows scored across the three groups; "
          f"{sum(len(v) for v in claims.values())} of them carry an equivalence claim,")
    print(f"    and those collapse to {len(claims)} DISTINCT arms:")
    for k in sorted(claims):
        mark = "arithmetic lower leg" if k in arith else "LOWER LEG IS A TEST, keeps two sides"
        print(f"      {k:42s} {mark}   [{', '.join(claims[k])}]")
    print(f"    {len(arith)} of {len(claims)} distinct arms rest on an arithmetically satisfied lower "
          f"leg.")
    print(f"    Keeping a genuinely two-sided reading: {', '.join(sorted(kept)) or 'NONE'}.")
    return {"n_arm_rows": len(all_rows),
            "n_claim_rows": sum(len(v) for v in claims.values()),
            "distinct_arms_with_an_equivalence_claim": sorted(claims),
            "where_each_is_scored": {k: sorted(v) for k, v in claims.items()},
            "n_distinct_arms": len(claims),
            "n_distinct_arms_with_arithmetic_lower_leg": len(arith),
            "distinct_arms_with_arithmetic_lower_leg": sorted(arith),
            "distinct_arms_keeping_the_two_sided_reading": sorted(kept),
            "the_number_to_quote": f"{len(arith)} of {len(claims)}",
            "why_not_the_row_count": "the flagship and its EMNIST replication are each scored in more "
                                     "than one group at more than one seed block, so the row count "
                                     "overstates the number of arms"}


def finding(name, rows):
    """The finding for ONE population. Never pooled across groups: one arm can appear in both."""
    # "Makes an equivalence reading" is the comparability ladder's own conjunct, imported in spirit
    # from analyze_margin_sensitivity: the interval contains zero AND lies inside the frozen margin. An
    # arm whose interval excludes zero reports a DETECTED effect and the paper makes no equivalence
    # claim about it, so it is not scored as a claim -- which is why the lists below are not
    # complements of each other.
    claims = [r for r in rows if r["ci_contains_zero"] and r["two_sided_inside_margin"]]
    untestable = [r for r in claims if not r["lower_leg_is_a_test"]]
    testable = [r for r in rows if r["lower_leg_is_a_test"]]
    # The wider population the appendix RULE has to cover: any arm whose lower side is arithmetic,
    # whether or not the sentence quoting it passes the equivalence conjunct. A sentence can read
    # "inside the frozen margin" off an arm whose interval excludes zero and still be leaning on a
    # lower leg that arithmetic already gave it.
    arithmetic_any = [r for r in rows if not r["lower_leg_is_a_test"]]

    print(f"\n=== THE FINDING, GROUP {name} ===")
    print(f"    arms scored                              : {len(rows)}")
    print(f"    of those, making an equivalence reading  : {len(claims)}")
    print(f"    of THOSE, whose lower leg is arithmetic  : {len(untestable)}")
    for r in untestable:
        print(f"      {r['arm']} / {r['design']}: identity {r['identity_mean_asr']:.4f} < "
              f"{EQUIV_MARGIN}, largest possible fall {r['fall_available']:.4f}, one-sided bound "
              f"{r['upper_bound']:+.4f}")
    keep = [f"{r['arm']} / {r['design']}" for r in claims if r["lower_leg_is_a_test"]]
    if claims and not untestable:
        print("    Every equivalence claim in this group has a testable lower leg. Nothing to "
              "relabel.")
    elif untestable and not keep:
        print("    EVERY equivalence claim in this group rests on an arithmetically satisfied lower")
        print("    leg. Each is reported as a one-sided non-increase with its 95% upper bound.")
    elif untestable:
        print(f"    {len(untestable)} of {len(claims)} equivalence claims in this group rest on an "
              "arithmetically")
        print("    satisfied lower leg and are relabelled as one-sided non-increases. The other "
              f"{len(keep)}")
        print(f"    KEEP the two-sided reading: {'; '.join(keep)}.")
        print("    So 'every equivalence claim' is FALSE of this group and may not be written.")
    print(f"    arms with an ARITHMETIC lower leg, any wording : {len(arithmetic_any)} of {len(rows)}")
    return {
        "n_scored": len(rows),
        "equivalence_claims": [f"{r['arm']} / {r['design']}" for r in claims],
        "equivalence_claims_with_untestable_lower_leg":
            [f"{r['arm']} / {r['design']}" for r in untestable],
        "equivalence_claims_keeping_the_two_sided_reading": keep,
        "every_equivalence_claim_has_an_untestable_lower_leg":
            bool(claims) and len(untestable) == len(claims),
        "lower_leg_is_a_test": [f"{r['arm']} / {r['design']}" for r in testable],
        "one_sided_reading_per_equivalence_claim":
            {f"{r['arm']} / {r['design']}": {"upper_bound": r["upper_bound"],
                                             "below_margin": r["upper_bound_below_margin"]}
             for r in claims},
        "arithmetic_lower_leg_any_wording":
            [{"arm": f"{r['arm']} / {r['design']}", "n": r["n"],
              "identity_mean_asr": r["identity_mean_asr"], "upper_bound": r["upper_bound"],
              "ci_contains_zero": r["ci_contains_zero"],
              "two_sided_inside_margin": r["two_sided_inside_margin"],
              "quoted_at": r.get("quoted_at")}
             for r in arithmetic_any],
        "n_arithmetic_lower_leg_any_wording": len(arithmetic_any),
    }


def p_leg_attribution(rows):
    """Which of the two legs sets the published TOST p, on the arms App. F marks equivalent?

    A separate question from reachability, and sharper. `tost()` returns p_tost = max(p_lower,
    p_upper), so exactly one leg sets the number the paper prints. Where the lower leg is arithmetic,
    that p tests a hypothesis ASR's floor has already excluded -- and because it is the MAXIMUM, the p
    for the claim actually being made, no meaningful rise, is the SMALLER of the two. So the one-sided
    relabelling is not a retreat on these arms: it reports a smaller p than the one already published.
    Printed per arm so no sentence has to assert this in general.
    """
    eq = [r for r in rows if r.get("tost_equivalent")]
    by_arith = [r for r in eq if r["p_tost_set_by_the_arithmetic_leg"]]
    print("\n=== WHICH LEG SETS THE PUBLISHED TOST p, ON THE ARMS App. F MARKS EQUIVALENT? ===")
    print("    p_tost = max(p_lower, p_upper), so one leg sets the printed number.")
    print(f"    {'arm':42s} {'lower leg p':>12s} {'upper leg p':>12s} {'printed':>9s} {'set by':>8s}")
    for r in eq:
        print(f"    {r['arm']:42s} {r['p_lower_leg']:12.4f} {r['p_upper_leg']:12.4f} "
              f"{r['p_tost']:9.4f} "
              f"{(r['p_tost_attained_by'] + ('*' if r['p_tost_set_by_the_arithmetic_leg'] else '')):>8s}")
    print(f"    * = set by a leg that is satisfied by arithmetic: {len(by_arith)} of {len(eq)} arms")
    if by_arith:
        print("    On each starred arm the published p is the ARITHMETIC leg's, and the one-sided p for")
        print("    no meaningful rise is smaller. Relabelling reports a stronger number, not a weaker one.")
    return {
        "what": "p_tost = max(p_lower, p_upper); which leg attains the max, on the arms tab:tost "
                "marks equivalent at the frozen margin",
        "n_equivalent_arms": len(eq),
        "n_whose_p_is_set_by_the_arithmetic_leg": len(by_arith),
        "per_arm": [{"arm": r["arm"], "n": r["n"], "p_lower_leg": r["p_lower_leg"],
                     "p_upper_leg": r["p_upper_leg"], "p_tost_printed": r["p_tost"],
                     "attained_by": r["p_tost_attained_by"],
                     "lower_leg_is_a_test": r["lower_leg_is_a_test"],
                     "quoted_at": r.get("quoted_at")} for r in eq],
        "consequence": "on every arm where the max is attained by an arithmetic lower leg, the "
                       "one-sided p for no meaningful RISE is strictly smaller than the published "
                       "p_tost, so the relabelling strengthens the reported evidence",
        "not_a_revision": "no published p is changed anywhere; p_tost stays as printed, and the "
                          "upper-leg p is reported beside it as the separate quantity it is",
    }


def main():
    print("=== IS THE LOWER TOST LEG A TEST? ===")
    print(f"    frozen margin {EQUIV_MARGIN}; ASR is bounded below by 0, so the paired mean cannot")
    print("    fall further than the identity rung's mean. Where that mean is below the margin, the")
    print("    lower leg is satisfied for every possible outcome and only the upper leg is a test.")
    print(f"    (Accuracy floor {ACC_FLOOR} is unchanged and not read here: this is an arithmetic")
    print("    statement about the ASR range, not an admissibility judgement.)")
    if TOST_MARGIN != EQUIV_MARGIN:
        sys.exit(f"REFUSING TO REPORT: App. F's margin {TOST_MARGIN} and the dose suite's "
                 f"{EQUIV_MARGIN} disagree. One margin, two values, is not a thing this script can "
                 "reason about.")
    hdr = (f"    {'arm':34s} {'design/n':11s} {'ident':>7s} {'fall':>7s} {'lower leg':>11s} "
           f"{'95% CI':>20s} {'equiv?':>7s}")

    mismatches = []
    print("\n  GROUP A: the comparability ladder's cells at 95%, both designs, plus the one "
          "equivalence arm\n  that artifact structurally cannot hold.")
    print(hdr)
    rows_a = group_a(mismatches)

    print("\n  GROUP B: App. F's Table 4 -- every Mode S arm in the paper at the seed counts already "
          "on disk.\n  Arms, rungs and pairing imported from analyze_tost_existing.py, that table's "
          "own emitter. One arm\n  can appear in both groups at different seed blocks, so the two "
          "groups are NEVER pooled.")
    print(hdr)
    rows_b = group_b(mismatches)

    print("\n  GROUP C: the two equivalence claims whose arm neither list above scores, found by "
          "auditing the\n  prose rather than the artifacts (experiments/audit_equivalence_claims.py).")
    print(hdr)
    rows_c = group_c(mismatches)

    # The ResNet18 arm is the one arm that legitimately appears in BOTH populations at the SAME seed
    # block: group A imports it because the comparability artifact cannot hold it, and group B has it
    # because tab:tost lists it. Two scripts, two estimators, one arm -- so instead of suppressing the
    # duplicate, it is used as a cross-check that the two agree exactly. If they ever diverge, one of
    # the two tables is quoting an interval the other cannot reproduce, and that must be loud.
    xa = next((r for r in rows_a if r["arm"] == "krum / scaling, resnet18"), None)
    xb = next((r for r in rows_b if r["arm"] == "Krum / scaling (ResNet18)"), None)
    if xa and xb:
        same = (xa["n"] == xb["n"] and xa["seeds"] == xb["seeds"]
                and all(abs(xa[k] - xb[k]) <= CI_TOL for k in ("mean", "identity_mean_asr"))
                and all(abs(a - b) <= CI_TOL for a, b in zip(xa["ci"], xb["ci"])))
        print(f"\n    CROSS-CHECK, the one arm in both populations at the same seed block: ResNet18 "
              f"via\n    analyze_margin_sensitivity and via analyze_tost_existing agree exactly: "
              f"{'yes' if same else 'NO'}.")
        if not same:
            mismatches.append(("krum / scaling, resnet18", "A vs B", "cross_check",
                               {k: xa[k] for k in ("n", "mean", "ci", "identity_mean_asr")},
                               {k: xb[k] for k in ("n", "mean", "ci", "identity_mean_asr")}))

    if mismatches:
        print("\n    ** RECOMPUTED VALUE DOES NOT MATCH THE PUBLISHED ONE **")
        for label, design, k, got, want in mismatches:
            print(f"      {label} / {design}: {k} recomputed {got!r}, published {want!r}")
        print("    Refusing to report reachability against intervals this script cannot reproduce.")
        json.dump({"description": "ABORTED: recomputed values disagree with the published ones.",
                   "mismatches": mismatches}, open(OUT, "w"), indent=2)
        return 1

    fa, fb = finding("A", rows_a), finding("B", rows_b)
    fc = finding("C", rows_c)
    pw = paper_wide(rows_a + rows_b + rows_c)
    pleg = p_leg_attribution(rows_b)

    print("\n=== THE TWO REFERENTS, PRINTED SO THEY CANNOT BE SWAPPED ===")
    for g, rs, want in (("A", rows_a, "krum / scaling"), ("B", rows_b, "Krum / scaling (flagship)")):
        for r in rs:
            if r["arm"] == want and r["design"] in ("controlled", "tab:tost"):
                print(f"    group {g}: {r['arm']} at n={r['n']}, identity "
                      f"{r['identity_mean_asr']:.4f}, 95% upper bound {r['upper_bound']:+.4f}")

    out = {
        "description": "Is the lower leg of each two-sided equivalence claim a test, or is it "
                       "satisfied by the arithmetic of a bounded outcome? Read-only over frozen "
                       "artifacts; no ASR computed, no model trained, no existing artifact written.",
        "sources": ["results/comparability_six_cells.json", "results/dose_resnet18/summary.json",
                    "every results/<dir>/summary.json named in analyze_tost_existing.ARMS"],
        "frozen_equivalence_margin": EQUIV_MARGIN,
        "frozen_acc_floor": ACC_FLOOR,
        "bound": "mean_i(asr_dosed_i - asr_identity_i) >= -mean_i(asr_identity_i), because ASR >= 0",
        "populations": {
            "A": "the comparability ladder's cells at 95%, both designs each, at the seed block that "
                 "ladder scores, plus the ResNet18 architecture arm, which has no outcome-gated twin "
                 "and so no comparability cell",
            "B": "App. F's Table 4: every Mode S arm in the paper at the seed counts already on disk, "
                 "with the arm list, rungs and pairing imported from analyze_tost_existing.py",
            "C": "the two equivalence claims whose arm neither list scores: Mode M, a second "
                 "transformation class at n=10, and the 200-round ResNet18 replication at n=3. Found "
                 "by auditing the prose, not the artifacts",
            "never_pooled": "one arm can appear in both groups at different seed blocks, so a pooled "
                            "count would double-count it"},
        "seed_block_convention": "within each group, the seed block that group's own scored contrast "
                                 "uses: unpaired_* in analyze_comparability.contrast for A, and "
                                 "analyze_tost_existing.paired for B",
        "two_referents_warning": "the flagship cell appears twice, as group B's n=5 ladder arm "
                                 "(identity 0.0618, 95% upper bound +0.026) and as group A's n=20 "
                                 "top-up (identity 0.0439, upper bound +0.012). They are different "
                                 "seed blocks of the same cell and their numbers are not "
                                 "interchangeable between sentences.",
        "all_recomputed_values_match_published": True,
        "rule": "On any arm whose identity-rung mean ASR is below the margin, the lower TOST leg is "
                "satisfied by arithmetic for every possible outcome and is not a test; the claim is "
                "reported as a pre-registered non-increase with its 95% upper bound. Arms whose "
                "identity mean exceeds the margin keep the two-sided reading and say so.",
        "may_not_be_written": "'every equivalence claim' -- coord_median / scaling, outcome-gated, "
                              "keeps a genuinely two-sided reading, so the universal is false of the "
                              "paper; and no group's count may be added to another's",
        "paper_wide": pw,
        "which_leg_sets_the_published_p": pleg,
        "group_A": fa, "group_B": fb, "group_C": fc,
        "rows": rows_a + rows_b + rows_c,
    }
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nSaved to {os.path.relpath(OUT, BASE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
