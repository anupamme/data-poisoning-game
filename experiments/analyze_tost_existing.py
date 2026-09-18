"""
TOST equivalence tests at the EXISTING seed counts, for every Mode-S arm the paper reports.

WHY THIS EXISTS, AND WHY IT IS NOT THE SEED TOP-UP
The paper's pre-registered equivalence rule is `|mean ASR(top) - mean ASR(identity)| < 0.15` and
`every rung < 0.5` (pre_registration_targeted_dose.md, frozen at 5130cec). That rule scores a POINT
estimate against a margin. It is not an equivalence TEST: a point estimate inside a margin is
consistent both with a genuinely negligible effect and with an effect the sample is too small to
locate at all. Those are different claims and the paper has only ever made the first.

This script scores the same frozen margin the other way round, with two one-sided t tests (TOST):
equivalence is declared only if the 90% interval on the paired difference lies entirely inside
(-0.15, +0.15), which at alpha=0.05 is exactly max(p_lower, p_upper) < 0.05. Reporting it this way
makes the sample size visible instead of implicit, and the honest result at n=3-5 is that most arms
do NOT pass -- not because the effect is large but because the interval is wider than the margin.

That is the argument FOR the seed top-up rather than a result of it, and it is reported as such:
this script reads only frozen artifacts, adds no runs, and revises no published verdict. The
published n=5 flagship verdict (|Delta|=0.026 < 0.15, p_up=0.70) stands exactly as reported; what
this adds is the width the point estimate never showed.

Paired, not two-sample: every rung of an arm runs at the same seeds, so each per-seed difference
holds one data partition fixed and the seed-to-seed spread that dominates this suite drops out of
the contrast. Same choice, and the same reason, as build_comparability_table.py's paired_ci95.

ONE ARM IS SCORED AT TWO SEED COUNTS, DELIBERATELY. The EMNIST-byclass row appears twice: once on
the frozen n=3 of results/dose_femnist/, and once at n=5 pooling seeds 45--46, which already exist
in results/comparability_cells/ from the comparability pre-registration's Amendment 2 (1026a96).
Frozen rows win the merge and nothing is written back. Reporting both is the amendment's own
commitment; reporting only the larger one would replace a published number silently, and reporting
only the smaller one would hide seeds the repository already holds.

Run: python3 experiments/analyze_tost_existing.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# t_crit, not the bare T95 table it falls back on: that table stops at df=9, so reading it directly
# would raise the moment an arm goes past n=10. Imported rather than restated so this script, the
# comparability table, the figure panel and the caption cannot quote four differently-derived
# intervals for the same quantity.
from experiments.analyze_headline_cis import t_crit  # noqa: E402

try:
    from scipy import stats as sps
except Exception:                                     # scipy is optional everywhere else in this repo
    sps = None

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The frozen equivalence margin and the frozen absolute ceiling, both from
# experiments/pre_registration_targeted_dose.md (5130cec). Neither is chosen here.
MARGIN = 0.15
ASR_CEILING = 0.5
ACC_FLOOR = 0.35

# Every Mode-S arm in the paper, with the directory its frozen artifact lives in. The identity rung
# is kappa=0 in all of them and is bit-identical to d2 standalone by construction.
ARMS = [
    ("Krum / scaling (flagship)",       "targeted_dose",   "doseS_kappa{r}_then_krum|committed_scaling"),
    ("Krum / scaling (score-only)",     "score_only",      "doseS_kappa{r}_then_krum|committed_scaling"),
    ("Krum / scaling (EMNIST-byclass)", "dose_femnist",    "doseS_kappa{r}_then_krum|committed_scaling"),
    # The architecture-alone arm (pre_registration_dose_resnet18.md, 89046f6). Listed for the same
    # reason as cell 7 below: the comment above says EVERY Mode-S arm in the paper and this is one.
    # It is the arm this script exists for. Its frozen rule is the POINT-estimate rule (|Delta| <
    # 0.15), which it passes at -0.041; the TOST row here is post hoc for this arm, as for every other
    # row, and it is the row that shows what n=3 on two endpoint rungs actually buys. Two rungs only,
    # so IDENTITY_RUNG/TOP_RUNG are the only rungs it has -- nothing is dropped by pairing them.
    ("Krum / scaling (ResNet18)",        "dose_resnet18",   "doseS_kappa{r}_then_krum|committed_scaling"),
    ("Reputation / scaling",            "targeted_dose",   "doseS_kappa{r}_then_reputation|committed_scaling"),
    ("cos_krum / pixel",                "targeted_dose",   "doseS_kappa{r}_then_cos_krum|committed_pixel"),
    ("coord_median / pixel",            "dose_replication", "doseS_kappa{r}_then_coord_median|committed_pixel"),
    # The same EMNIST-byclass cell at n=5, on seeds that already exist rather than on new runs:
    # 45 and 46 of this ladder were run under the comparability pre-registration's Amendment 2
    # (1026a96, read with Amendment 3 f16083b) and live in results/comparability_cells/ under an
    # |emnist key suffix. Frozen rows win, results/dose_femnist/ is never written, and
    # analyze_dose_femnist.py --pooled scores the primary rule on the identical merge, so the two
    # scripts cannot report different n=5 readings of one arm. Both rows are printed because the
    # amendment's commitment is to report both seed counts, never to replace the frozen one.
    ("Krum / scaling (EMNIST-byclass, $n{=}5$)", "dose_femnist",
     "doseS_kappa{r}_then_krum|committed_scaling",
     [("comparability_cells", "doseS_kappa{r}_then_krum|committed_scaling|emnist")]),
    # Cell 7 of the comparability table (Amendment 4, 9b8a395): the same coord_median/pixel arm on
    # CIFAR-100. Listed because the header above says EVERY Mode-S arm in the paper, and this is one;
    # omitting it would make that comment false and leave a reader to wonder whether the arm was
    # scored and dropped. It is scored here and it does NOT pass: the arm is a detected rise inside a
    # sign reversal, so the row's equivalence verdict is expected to fail and is reported as a
    # failure, not as support. The paper makes no equivalence claim about it, which is why
    # analyze_margin_sensitivity.py excludes it from the binding-arm search on the separate ground
    # that its interval does not contain zero.
    # Round 71's top-up (pre_registration_cell7_seed_topup.md) adds seeds 47--61 to this arm's two
    # endpoint rungs. The 4th element is the same pooled-source mechanism the EMNIST row above uses,
    # and it is required rather than automatic: `load` reads ONE directory, so without it this row
    # would keep scoring at n=5 while the comparability table beside it printed n=20 -- one arm
    # reported at two n by two scripts, with nothing on the page saying why. Frozen rows still win
    # (dedup is first-source, and results/comparability_cells/ is listed first), so the published five
    # seeds keep their published values.
    ("coord_median / pixel (CIFAR-100)", "comparability_cells",
     "doseS_kappa{r}_then_coord_median|committed_pixel|cifar100",
     [("cell7_seed_topup", "doseS_kappa{r}_then_coord_median|committed_pixel|cifar100")]),
]
IDENTITY_RUNG, TOP_RUNG = "0.0", "2.0"


def load(dirname):
    p = os.path.join(BASE, "results", dirname, "summary.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p)).get("cells", {})


def rows_of(cells, keyfmt, rung):
    return {r["seed"]: r for r in (cells.get(keyfmt.format(r=rung)) or {}).get("per_seed", [])}


def paired(cells, keyfmt, extra=()):
    """Per-seed (asr_top - asr_identity), plus the accuracies, on the seeds present in BOTH rungs.

    `extra` is a list of (dirname, keyfmt) consulted after `cells`, deduplicated by seed with the
    FIRST source winning, so a pooled row can only ever fill a seed the authoritative artifact does
    not have.
    """
    L, H = rows_of(cells, keyfmt, IDENTITY_RUNG), rows_of(cells, keyfmt, TOP_RUNG)
    if not L or not H:
        return None
    for dirname, kf in extra:
        c = load(dirname)
        if c is None:
            continue
        for dst, src in ((L, rows_of(c, kf, IDENTITY_RUNG)), (H, rows_of(c, kf, TOP_RUNG))):
            for s, r in src.items():
                dst.setdefault(s, r)
    seeds = sorted(set(L) & set(H))
    if not seeds:
        return None
    return {
        "seeds": seeds,
        "diff": np.array([H[s]["asr"] - L[s]["asr"] for s in seeds], dtype=float),
        "asr_lo": np.array([L[s]["asr"] for s in seeds], dtype=float),
        "asr_hi": np.array([H[s]["asr"] for s in seeds], dtype=float),
        "acc": np.array([L[s]["accuracy"] for s in seeds] + [H[s]["accuracy"] for s in seeds]),
    }


def t_crit90(n):
    """One-sided 95% / two-sided 90% critical value, which is the interval TOST actually uses."""
    if sps is not None:
        return float(sps.t.ppf(0.95, n - 1))
    # df 1..29, two-sided 90%. Only reached if scipy is missing; kept short deliberately, and it
    # raises rather than extrapolating past its end.
    tbl = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860,
           9: 1.833, 10: 1.812, 11: 1.796, 12: 1.782, 13: 1.771, 14: 1.761, 15: 1.753,
           16: 1.746, 17: 1.740, 18: 1.734, 19: 1.729, 20: 1.725, 21: 1.721, 22: 1.717,
           23: 1.714, 24: 1.711, 25: 1.708, 26: 1.706, 27: 1.703, 28: 1.701, 29: 1.699}
    return tbl[n - 1]


def tost(d, margin=MARGIN):
    """Two one-sided t tests of |mean(d)| < margin, plus the 90% interval that decides them."""
    n = len(d)
    m, sd = float(d.mean()), float(d.std(ddof=1)) if n > 1 else float("nan")
    if n < 2:
        return {"n": n, "note": "no test at n<2"}
    se = sd / np.sqrt(n)
    # H01: mean <= -margin (rejected by a RIGHT tail); H02: mean >= +margin (rejected by a LEFT tail).
    t_lo, t_hi = (m + margin) / se, (m - margin) / se
    if sps is not None:
        p_lo = float(sps.t.sf(t_lo, n - 1))
        p_hi = float(sps.t.cdf(t_hi, n - 1))
    else:
        p_lo = p_hi = float("nan")
    hw90 = t_crit90(n) * se
    hw95 = t_crit(n) * se
    lo90, hi90 = m - hw90, m + hw90
    lo95, hi95 = m - hw95, m + hw95
    equiv = bool(lo90 > -margin and hi90 < margin)
    return {"n": n, "mean": m, "sd": sd, "se": float(se),
            "ci90": (lo90, hi90), "ci95": (lo95, hi95),
            "p_lower": p_lo, "p_upper": p_hi,
            "p_tost": (max(p_lo, p_hi) if sps is not None else float("nan")),
            "equivalent": equiv,
            # The distinction that is the whole point of the script.
            "point_inside_margin": bool(abs(m) < margin),
            # An interval can lie inside the margin AND exclude zero: equivalence and detection are
            # not exclusive, and a bare checkmark on such an arm reads as "no effect" when the paper
            # reports a measured one. Scored and marked separately so the table cannot say otherwise.
            "detected": not (lo95 <= 0.0 <= hi95)}


def main():
    print(f"TOST against the frozen +/-{MARGIN} margin, at the seed counts already on disk.")
    print("Equivalence requires the 90% interval on the PAIRED difference inside the margin.")
    print("Reads frozen artifacts only; adds no runs; revises no published verdict.\n")

    rows, n_equiv, n_inside_only, n_both = [], 0, 0, 0
    for label, dirname, keyfmt, *rest in ARMS:
        extra = rest[0] if rest else ()
        cells = load(dirname)
        if cells is None:
            print(f"{label:34s} results/{dirname}/summary.json ABSENT -- skipped\n")
            continue
        p = paired(cells, keyfmt, extra)
        if p is None:
            print(f"{label:34s} rungs {IDENTITY_RUNG}/{TOP_RUNG} not both present -- skipped\n")
            continue
        r = tost(p["diff"])
        rows.append((label, dirname, p, r))

        print(f"{label}  [results/{dirname}"
              + ("".join(f" + results/{d}" for d, _ in extra) if extra else "") + "]")
        print(f"  seeds {p['seeds'][0]}-{p['seeds'][-1]} (n={r['n']})   "
              f"identity mean {p['asr_lo'].mean():.4f} -> top mean {p['asr_hi'].mean():.4f}")
        print(f"  paired Delta = {r['mean']:+.4f}  (sd {r['sd']:.4f}, se {r['se']:.4f})")
        print(f"  95% CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]   "
              f"90% CI (TOST) [{r['ci90'][0]:+.4f}, {r['ci90'][1]:+.4f}]")
        if sps is not None:
            print(f"  TOST p = max({r['p_lower']:.4f}, {r['p_upper']:.4f}) = {r['p_tost']:.4f}")
        verdict = ("EQUIVALENT at the frozen margin" if r["equivalent"] else
                   "NOT established -- interval wider than the margin")
        print(f"  -> {verdict}")
        if r["equivalent"] and r["detected"]:
            n_both += 1
            print("     AND THE EFFECT IS DETECTED: the 95% interval excludes zero, so this arm is"
                  "\n           practically equivalent and statistically nonzero at once. The"
                  " checkmark here\n           does NOT mean 'no effect'; it means 'no effect larger"
                  f" than {MARGIN}'.")
        if r["point_inside_margin"] and not r["equivalent"]:
            n_inside_only += 1
            print(f"     note: the point estimate {r['mean']:+.4f} IS inside +/-{MARGIN}, which is what"
                  f" the frozen\n           rule scores; equivalence is a stronger claim and n={r['n']}"
                  " does not support it.")
        if r["equivalent"]:
            n_equiv += 1
        # The margin never stands alone in the frozen rule, so the ceiling is scored beside it.
        worst = max(p["asr_lo"].mean(), p["asr_hi"].mean())
        print(f"     rung means below the frozen {ASR_CEILING} ceiling: "
              f"{worst < ASR_CEILING} (worst {worst:.4f}); "
              f"mean accuracy {p['acc'].mean():.4f} "
              f"({'above' if p['acc'].mean() >= ACC_FLOOR else 'BELOW'} the {ACC_FLOOR} floor)")
        print()

    print("--- SUMMARY ---")
    print(f"  arms scored: {len(rows)}")
    print(f"  equivalence ESTABLISHED by TOST at existing n: {n_equiv}")
    print(f"  point estimate inside margin but TOST inconclusive: {n_inside_only}")
    print(f"  equivalent AND detected (95% interval excludes zero): {n_both}")
    if n_inside_only:
        print("\n  This is the honest reading and the argument for the seed top-up: at n=3-8 the")
        print("  frozen point-estimate rule passes while the equivalence test does not, because the")
        print("  interval is wider than the margin. The paper should report the width, not upgrade")
        print("  the point estimate to an equivalence claim it does not support.")

    # LaTeX, emitted rather than transcribed, so the paper cannot disagree with this script.
    print("\n--- LaTeX rows (arm, n, paired Delta, 95% CI, 90% CI, TOST verdict) ---")
    for label, _, p, r in rows:
        # The table's own vocabulary, so a row can be checked against it without translation.
        v = ("equivalent$^{\\dagger}$" if r["equivalent"] and r["detected"]
             else "equivalent" if r["equivalent"] else "not est.")
        print(f"{label.replace('_', chr(92) + '_')} & ${r['n']}$ & ${r['mean']:+.3f}$ & "
              f"$[{r['ci95'][0]:+.3f},\\,{r['ci95'][1]:+.3f}]$ & "
              f"$[{r['ci90'][0]:+.3f},\\,{r['ci90'][1]:+.3f}]$ & {v} \\\\")
    return 0


if __name__ == "__main__":
    sys.exit(main())
