"""
Score the flagship Mode-S Krum arm at the topped-up seed count, against the rules frozen in
`experiments/pre_registration_dose_seed_topup.md` (committed at 684b31e).

WHAT THIS DOES AND DOES NOT DECIDE
The published n=5 verdict (|Delta| = 0.026 < 0.15, JT increasing p = 0.70) **stands as reported and is
not revisited**; the prereg says so and non-negotiable 5 requires it to be printed beside whatever the
larger n says. What the top-up buys is the interval's WIDTH, which the published version never stated.

THE REVERSAL CLAUSE IS SCORED HERE, NOT NARRATED
If the primary 95% interval is not contained in (-0.15, +0.15), or any rung mean reaches 0.5, this
script prints EQUIVALENCE REFUTED and says that the finding goes in the abstract and the (P3)=/=>(P4)
witness is withdrawn. Pre-committing to that is what licensed adding seeds at all, so the code path
exists whether or not it fires.

PARTIAL RUNS ARE SCORED AT THE n ACTUALLY REACHED
Prereg point 3: "If the runs are interrupted, the analysis reports the n actually reached and the
interval at that n; it does not resume until a threshold is crossed." So this prints the realized n per
rung and never waits for 20, and never silently pools rungs measured at different n.

t_crit IS IMPORTED, NEVER `T95[n-1]`
`analyze_headline_cis.T95` is a literal table that stops at **df = 9**, so reading it directly raises
KeyError the moment this arm reaches n = 11. `t_crit` prefers scipy and falls back to that table only
for small df, which is why every interval in this round goes through it.

Reads results/targeted_dose/ (seeds 42-46, published, never written) and results/dose_seed_topup/
(seeds 47-61). Writes nothing.

Run: python3 experiments/analyze_dose_seed_topup.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.analyze_dose_response import jonckheere            # noqa: E402  the canonical test
from experiments.analyze_headline_cis import t_crit                 # noqa: E402  NOT T95[n-1]
from experiments.analyze_tost_existing import tost                # noqa: E402

try:
    from scipy import stats as sps
except Exception:
    sps = None

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLISHED = os.path.join(BASE, "results", "targeted_dose", "summary.json")
TOPUP = os.path.join(BASE, "results", "dose_seed_topup", "summary.json")

# All three carried forward unchanged from 5130cec; none is a new threshold.
MARGIN, ASR_CEILING, ACC_FLOOR = 0.15, 0.5, 0.35

KAPPAS = ["0.0", "0.5", "1.0", "2.0"]
KEY = "doseS_kappa{r}_then_krum|committed_scaling"
PUBLISHED_SEEDS = [42, 43, 44, 45, 46]
TOPUP_SEEDS = list(range(47, 62))

# The published n=5 numbers, transcribed from the pre-registration's own table so that the merge can be
# checked against the document rather than against itself. Asserted, never printed as a result.
PUB_RUNG_MEANS = {"0.0": 0.0618, "0.5": 0.0161, "1.0": 0.0229, "2.0": 0.0357}
PUB_PER_SEED = {
    "0.0": [0.1149, 0.0209, 0.0469, 0.0567, 0.0694],
    "2.0": [0.0183, 0.0276, 0.0440, 0.0510, 0.0374],
}
PUB_PAIRED_MEAN, PUB_PAIRED_SD = -0.0261, 0.0419
PUB_TOL = 5e-4                     # the prereg prints 4 decimals
EXACT_TOL = 1e-12                  # merged published seeds must be the SAME floats, not close ones

# Carried forward from 5130cec, and NOT recomputed at n=20: the channel measurement was not extended,
# so this is reported as an inherited figure rather than a re-measurement.
DEGENERATE_MODE_S = (24, 120)


def cells(path):
    if not os.path.exists(path):
        return {}
    return json.load(open(path)).get("cells", {})


def merge():
    """{kappa: {seed: (asr, accuracy, provenance)}}, published seeds first and never overwritten."""
    pub, top = cells(PUBLISHED), cells(TOPUP)
    out = {}
    for k in KAPPAS:
        rows = {}
        for src, tag in ((pub, "published"), (top, "top-up")):
            c = src.get(KEY.format(r=k))
            if not c:
                continue
            for r in c["per_seed"]:
                s = int(r["seed"])
                if s in rows:                       # published wins; the top-up never rewrites it
                    continue
                rows[s] = (float(r["asr"]), float(r["accuracy"]), tag)
        out[k] = rows
    return out


def check_published_reproduce(m):
    """Non-negotiable 5: the five published seeds reproduce before ANY pooled number is printed.

    Two checks, because they can fail independently: (i) the merged rows for seeds 42-46 are the
    identical floats stored in results/targeted_dose/, so the merge cannot have shifted a value; and
    (ii) the n=5 quantities recomputed from those rows match the pre-registration's printed table, so
    the arm being extended is the arm that was published.
    """
    problems = []
    pub = cells(PUBLISHED)
    for k in KAPPAS:
        c = pub.get(KEY.format(r=k))
        if not c:
            problems.append(f"kappa={k}: absent from results/targeted_dose/")
            continue
        for r in c["per_seed"]:
            s = int(r["seed"])
            if s not in m[k]:
                problems.append(f"kappa={k} seed {s}: published row lost in the merge")
            elif abs(m[k][s][0] - float(r["asr"])) > EXACT_TOL:
                problems.append(f"kappa={k} seed {s}: merged {m[k][s][0]!r} != published {r['asr']!r}")

    for k, want in PUB_RUNG_MEANS.items():
        got = float(np.mean([m[k][s][0] for s in PUBLISHED_SEEDS if s in m[k]]))
        if abs(got - want) > PUB_TOL:
            problems.append(f"kappa={k}: n=5 mean {got:.4f} != pre-registered {want:.4f}")
    for k, want in PUB_PER_SEED.items():
        got = [m[k][s][0] for s in PUBLISHED_SEEDS if s in m[k]]
        if len(got) == len(want) and max(abs(a - b) for a, b in zip(got, want)) > PUB_TOL:
            problems.append(f"kappa={k}: per-seed {['%.4f' % g for g in got]} != pre-registered {want}")

    d5 = np.array([m["2.0"][s][0] - m["0.0"][s][0] for s in PUBLISHED_SEEDS
                   if s in m["2.0"] and s in m["0.0"]], dtype=float)
    if len(d5) == 5:
        if abs(float(d5.mean()) - PUB_PAIRED_MEAN) > PUB_TOL:
            problems.append(f"paired mean {d5.mean():.4f} != pre-registered {PUB_PAIRED_MEAN}")
        if abs(float(d5.std(ddof=1)) - PUB_PAIRED_SD) > PUB_TOL:
            problems.append(f"paired sd {d5.std(ddof=1):.4f} != pre-registered {PUB_PAIRED_SD}")
    else:
        problems.append(f"only {len(d5)} of 5 published paired differences available")
    return problems


def unpaired(a, b):
    """Secondary interval: difference of rung means, pooled sd, df = n_a + n_b - 2."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    sp2 = (((na - 1) * a.var(ddof=1)) + ((nb - 1) * b.var(ddof=1))) / (na + nb - 2)
    se = np.sqrt(sp2 * (1.0 / na + 1.0 / nb))
    m = float(b.mean() - a.mean())
    # t_crit takes n and uses n-1 df, so pass the equivalent n for df = na+nb-2.
    hw = t_crit(na + nb - 1) * se
    return {"mean": m, "se": float(se), "ci95": (m - hw, m + hw), "df": na + nb - 2}


def main():
    m = merge()
    if not any(m.values()):
        print("No data: neither results/targeted_dose/ nor results/dose_seed_topup/ has this cell.")
        return 1

    problems = check_published_reproduce(m)
    if problems:
        print("REFUSING TO PRINT ANY POOLED NUMBER -- the published seeds do not reproduce:")
        for p in problems:
            print(f"  {p}")
        return 1
    print("Published seeds 42-46 reproduce exactly (merged floats identical to "
          "results/targeted_dose/,\n  and the recomputed n=5 table matches the pre-registration). "
          "Proceeding.\n")

    print("=== FLAGSHIP Mode-S Krum / committed_scaling, seed top-up (item A) ===")
    print("Rules frozen at 684b31e; margin, ceiling and accuracy floor carried forward from 5130cec.\n")

    # Per-rung state at the n actually reached.
    print(f"  {'rung':6s} {'n':>3s} {'mean ASR':>9s} {'mean acc':>9s}   provenance / flags")
    groups, rung_ok = [], True
    for k in KAPPAS:
        rows = m[k]
        seeds = sorted(rows)
        asr = np.array([rows[s][0] for s in seeds], dtype=float)
        acc = np.array([rows[s][1] for s in seeds], dtype=float)
        groups.append(asr)
        n_pub = sum(1 for s in seeds if rows[s][2] == "published")
        flags = []
        if asr.mean() >= ASR_CEILING:
            flags.append(f"RUNG MEAN >= {ASR_CEILING} CEILING")
            rung_ok = False
        if acc.mean() < ACC_FLOOR:
            flags.append(f"RUNG MEAN ACC BELOW {ACC_FLOOR} FLOOR")
            rung_ok = False
        low = [s for s in seeds if rows[s][1] < ACC_FLOOR]
        if low:
            # Flagged, not excluded: the floor applies to a rung's MEAN, per the frozen convention.
            flags.append(f"seeds below floor (flagged, NOT excluded): {low}")
        print(f"  {k:6s} {len(seeds):3d} {asr.mean():9.4f} {acc.mean():9.4f}   "
              f"{n_pub} published + {len(seeds) - n_pub} top-up"
              + ("   ** " + "; ".join(flags) if flags else ""))

    # Primary: the paired interval on the seeds present at BOTH extreme rungs.
    common = sorted(set(m["0.0"]) & set(m["2.0"]))
    d = np.array([m["2.0"][s][0] - m["0.0"][s][0] for s in common], dtype=float)
    n = len(d)
    print(f"\n  PRIMARY (paired, pre-registered as primary even though it was the wider at n=5)")
    print(f"    n = {n} (seeds {common[0]}-{common[-1]}), of a planned 20")
    if n < 2:
        print("    too few paired seeds to form an interval; nothing further is scored.")
        return 0
    r = tost(d, MARGIN)
    hw95 = t_crit(n) * r["se"]
    print(f"    paired Delta = {r['mean']:+.4f}   sd {r['sd']:.4f}   se {r['se']:.4f}")
    print(f"    95% CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]   half-width {hw95:.4f} "
          f"= {100 * hw95 / MARGIN:.1f}% of the +-{MARGIN} margin")
    print(f"    90% CI [{r['ci90'][0]:+.4f}, {r['ci90'][1]:+.4f}]   (the interval TOST decides on)")
    if sps is not None:
        t_lo, t_hi = (r["mean"] + MARGIN) / r["se"], (r["mean"] - MARGIN) / r["se"]
        print(f"    TOST: t_lower {t_lo:+.3f} p {r['p_lower']:.4g} | "
              f"t_upper {t_hi:+.3f} p {r['p_upper']:.4g} | TOST p = {r['p_tost']:.4g}")

    sec = unpaired(np.array([m["0.0"][s][0] for s in sorted(m["0.0"])]),
                   np.array([m["2.0"][s][0] for s in sorted(m["2.0"])]))
    if sec:
        print(f"\n  SECONDARY (unpaired, the form the frozen Round-12 rule is literally written in)")
        print(f"    difference of rung means {sec['mean']:+.4f}, df {sec['df']}, "
              f"95% CI [{sec['ci95'][0]:+.4f}, {sec['ci95'][1]:+.4f}]")

    # The frozen verdict: containment of the PRIMARY 95% interval, and every rung mean below ceiling.
    contained = r["ci95"][0] > -MARGIN and r["ci95"][1] < MARGIN
    print(f"\n  EQUIVALENCE VERDICT at n = {n}")
    if contained and rung_ok:
        print(f"    CONFIRMED: [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] is contained in "
              f"(-{MARGIN}, +{MARGIN}) and every rung mean clears the ceiling and the floor.")
        print("    The 95% criterion is strictly more conservative than TOST at alpha=0.05, so this")
        print("    also implies both one-sided tests reject.")
    else:
        print(f"    ** REFUTED **: "
              + ("the primary interval is NOT contained in the margin"
                 if not contained else "a rung mean failed the ceiling or the accuracy floor"))
        print("    Per the reversal clause frozen at 684b31e, this is reported as a refutation of our")
        print("    OWN published claim, IN THE ABSTRACT, and the (P3)=/=>(P4) witness is withdrawn to")
        print(f"    'not established at n = {n}'.")

    # H-statistic retest, in the pre-registered direction.
    if all(len(g) >= 2 for g in groups):
        J, z, p_inc, p_dec, p_perm = jonckheere(groups)
        print(f"\n  H-STATISTIC RETEST (Jonckheere-Terpstra increasing across kappa "
              f"{'/'.join(KAPPAS)})")
        sizes = [len(g) for g in groups]
        if len(set(sizes)) > 1:
            # The runner fills rung-major, so mid-run the low rungs carry more seeds than the high
            # ones. JT's E[J]/V[J] do account for unequal group sizes, so the test is valid -- but it
            # is then a trend test on a PARTIALLY FILLED ladder, which is not the pre-registered
            # n=20-per-rung test. Said out loud so a mid-run number is never quoted as the retest.
            print(f"    ** PARTIAL LADDER: rung n = {sizes}, not all equal. Valid as computed, but")
            print("       this is NOT the pre-registered retest and must not be quoted as it.")
        print(f"    J = {J:.1f}, z = {z:+.3f}, p_increasing = {p_inc:.4f} "
              f"(permutation {p_perm:.4f}), p_decreasing = {p_dec:.4f}")
        if p_inc < 0.05:
            print("    ** p < 0.05: H-statistic CONFIRMED at this n and the published refutation is")
            print("       WITHDRAWN. This is a reversal and goes in the abstract.")
        else:
            print("    p >= 0.05: the published refutation of H-statistic stands at this n.")

    # Non-negotiable 5, and the inherited caveat.
    print(f"\n  PUBLISHED n=5 VERDICT, reported alongside as required: "
          f"|Delta| = 0.026 < {MARGIN}, JT increasing p = 0.70,")
    print("    i.e. H-admission confirmed, H-statistic refuted. If the two disagree, both appear and")
    print("    the disagreement is the result.")
    print(f"  INHERITED CAVEAT, not re-measured at this n: Mode S has nothing to impose in "
          f"{DEGENERATE_MODE_S[0]}/{DEGENERATE_MODE_S[1]}")
    print("    measured rung-rounds at the frozen configuration; the channel measurement was not")
    print("    extended with the seeds, so this is carried forward from 5130cec, not re-measured.")

    print("\n--- LaTeX (paired interval and TOST at the realized n) ---")
    print(f"$n{{=}}{n}$ & ${r['mean']:+.3f}$ & $[{r['ci95'][0]:+.3f},\\,{r['ci95'][1]:+.3f}]$ & "
          f"$[{r['ci90'][0]:+.3f},\\,{r['ci90'][1]:+.3f}]$ & "
          f"{'equivalent' if contained and rung_ok else 'REFUTED'} \\\\")
    return 0


if __name__ == "__main__":
    sys.exit(main())
