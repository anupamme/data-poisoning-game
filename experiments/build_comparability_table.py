"""The confounded ladder vs the Mode-S instrument, side by side, on the ONE arm that has both.

WHY THIS EXISTS. The paper reports a sign reversal -- the same aggregator, the same attack, the same
seeds, the same nominal dose, and an ASR effect of -0.273 under the confounded ladder against +0.125
under the Mode-S instrument at n=20 (-0.272 against +0.098 at the frozen n=5, reproduced below)
-- and asks the reader to attribute the difference to the adversarial
coefficient channel. Until now that argument was spread across two tables and a paragraph, so a reader
could not check the one thing that makes it an argument rather than an assertion: that the two designs
are matched on everything else. This script puts the two columns next to each other, recomputes every
cell from per-seed and per-round rows, and asserts the shared-origin identity.

WHAT THE COMPARISON RESTS ON, and what it does not:

  matched      the aggregator (coord_median), the attack (committed_pixel), the seeds (42-61 at the
               endpoint rungs, 42-46 at the interior ones -- identical on both legs either way), the
               nominal dose (kappa 0 -> 2, rho 1 -> 54.6), the identity rung ITSELF (the two designs
               share their kappa=0 cell numerically, asserted below to 1e-12), and -- at the top rung --
               the displacement of the emitted aggregate, the decision-change rate, and the mean clean
               accuracy. Matched accuracy is what rules out "the ladder's fall is an accuracy collapse".
  differing    the adversarial coefficient channel: the ladder lets the adversaries' coefficients be
               drawn from the same widening spread as everyone's, so their share of the coefficient mass
               falls from 0.267 to 0.187 and Delta_c drifts to -0.336; Mode S pins c_adv = 1.0 exactly,
               holding the share constant and Delta_c at 0.
  NOT matched  aggregate adversarial influence Lambda_a moves in BOTH designs, 0.081 in the ladder
               against 0.033 in the instrument. Neither column is a clean statistic-only intervention:
               what Mode S pins exactly is the adversarial COEFFICIENT share, and for a coordinate-wise
               order statistic that does not force Lambda_a to be constant, because which client attains
               the median per coordinate still moves. The ladder's movement is 2.4x the instrument's and
               in the same direction. This is disclosed in the table itself rather than in a footnote,
               and it is why the contrast is stated as "differ chiefly in the coefficient channel"
               rather than "differ only in it".

               VOCABULARY, so this table cannot contradict the channel table. The frozen field
               results/admission_measurement.json '<family>|coord_median|<rung>|admission' is the change
               in coord_median_adv_argmedian_frac -- the adversarial share of selected coordinates --
               which is exactly build_channel_table.py's Delta Lambda_a column (0.033 in both scripts),
               NOT its Delta adm. column. That column counts rounds in which the SUPPORT of the
               adversarial mass changes -- whether any adversarial input enters at all -- and reads
               0.000 for every row. So this row is labelled Delta Lambda_a: what moves is how much
               weight the adversary's direction carries, not whether it enters. For a selector the mass
               is binary and the two coincide.

This is a two-column contrast on one arm, not a controlled experiment over many. It licenses the claim
that the adversarial coefficient channel is where the two designs differ; it does not by itself measure
that channel's effect size, and no p-value is attached to the difference between the two columns.

THE TOP-UP, and why this table is deliberately mixed-n (Round 63). The pre-registered top-up
experiments/pre_registration_reversal_seed_topup.md extends THIS cell from n=5 to n=20 on both legs,
seeds 47-61, at the ENDPOINT rungs only (kappa=0 and kappa=2), which is exactly what the ASR effect
row reads. Three consequences, each visible in the emitted table rather than left to a reader:

  ASR, mean clean accuracy, the ASR effect and its interval are n=20. Both legs are recomputed over
  the identical seed set -- never one leg topped up against a five-seed subtrahend -- and the two
  designs still share their kappa=0 cell numerically at 1e-12, now over 20 seeds instead of 5.

  The Jonckheere-Terpstra trend p-values stay at n=5, on seeds 42-46 across all FOUR rungs. The
  pre-registration fixes this: the interior rungs were not topped up, so a four-rung trend statistic
  at mixed n would be a new test on new data, and the published p_down=0.0015 / p_up=0.069 are the
  n=5 statistics. They are printed here, labelled n=5, and asserted to reproduce bit-for-bit.

  The mechanism rows -- aggregate displacement, decision change, dLambda_a, the coefficient share and
  Delta_c -- stay at n=5 because the top-up runner records accuracy and ASR per seed and nothing else.
  So the "matched on everything else" argument is an n=5 argument about the mechanism and an n=20
  argument about the outcome. Every row therefore carries its own n in its label; no row inherits an
  n from the seeds row, and the caption says which rows are which.

SOURCES, all read-only:
  results/dose_response/summary.json      ASR for the confounded ladder (Round 11)
  results/dose_replication/summary.json   ASR for the Mode-S arm (Round 15)
  results/reversal_seed_topup/summary.json  seeds 47-61 at the endpoint rungs, both legs (Round 63)
  results/admission_measurement.json      aggregate displacement / decision / admission, both families
  results/coefficient_targeting.json       Delta_c and the adversarial coefficient share, both families

Output: results/comparability_table.json plus LaTeX on stdout.
Run: python3 experiments/build_comparability_table.py
"""
import json, os, sys
import numpy as np
try:                                # optional, exactly as in the scorers this script imports from
    import scipy.stats as sps
except ImportError:                 # pragma: no cover
    sps = None

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
# The same trend test the frozen scorers use, so this table and the papers cannot disagree about
# the quoted p-values: the published p_down = 0.0015 and p_up = 0.069 are Jonckheere-Terpstra over all
# FOUR rungs, not two-rung t-tests. The two-rung Welch statistic is computed too, but it is recorded in
# the JSON as a post-hoc diagnostic and deliberately kept out of the LaTeX so no new number enters the
# papers.
from experiments.analyze_dose_response import jonckheere  # noqa: E402
from experiments.analyze_targeted_dose import welch_less  # noqa: E402
# t_crit, not the bare T95 table it falls back on: the table stops at df=9, so reading it directly
# would raise the moment a top-up takes an arm past n=10. Imported rather than restated so the figure
# panel, this table, the caption and the letter cannot quote four differently-derived intervals.
from experiments.analyze_headline_cis import t_crit  # noqa: E402

R = os.path.join(base, "results")
LADDER = os.path.join(R, "dose_response", "summary.json")
MODES = os.path.join(R, "dose_replication", "summary.json")
# Both legs of the top-up live in ONE file, keyed by the same cell names the frozen suites use, so it
# merges into either design without a per-design path. Required, not optional: if it is missing this
# script exits, because the paper now quotes n=20 and a silent fall back to n=5 would reproduce the
# old numbers under the new labels.
TOPUP = os.path.join(R, "reversal_seed_topup", "summary.json")
FROZEN_N = 5            # the seed block the mechanism rows and the trend tests are frozen at
ADM = os.path.join(R, "admission_measurement.json")
COEF = os.path.join(R, "coefficient_targeting.json")
OUT = os.path.join(R, "comparability_table.json")

ARM, ATTACK = "coord_median", "committed_pixel"
KAPPAS = [0.0, 0.5, 1.0, 2.0]
LO, HI = KAPPAS[0], KAPPAS[-1]
SHARED_TOL = 1e-12      # the two designs must share their identity cell NUMERICALLY, not approximately

# (column label, family prefix in the channel files, ASR source, the d1 name pattern)
DESIGNS = [("Confounded ladder", "dose", LADDER, "dose_kappa{}"),
           ("Instrument (Mode S)", "doseS", MODES, "doseS_kappa{}")]


def asr_rungs(path, pattern):
    """{kappa: (per-seed asr list, per-seed acc list, seeds)} recomputed from per-seed rows.

    The frozen suite supplies all four rungs at seeds 42-46; the top-up supplies seeds 47-61 at the
    endpoint rungs only. Merging by seed rather than by concatenation, so a seed present in both files
    can never be counted twice, and re-sorting afterwards, so the per-seed lists of the two rungs are
    in the same seed order and the paired difference pairs like with like.
    """
    if not os.path.exists(path):
        return {}
    cells = json.load(open(path))["cells"]
    top = json.load(open(TOPUP))["cells"] if os.path.exists(TOPUP) else {}
    out = {}
    for k in KAPPAS:
        name = f"{pattern.format(k)}_then_{ARM}|{ATTACK}"
        c = cells.get(name)
        if c is None:
            continue
        by_seed = {int(r["seed"]): r for r in c["per_seed"]}
        for r in (top.get(name) or {}).get("per_seed", []):
            if int(r["seed"]) in by_seed:
                sys.exit(f"seed {r['seed']} is in both {os.path.relpath(path, base)} and the top-up "
                         f"for {name}; the top-up must add seeds, never restate them")
            by_seed[int(r["seed"])] = r
        seeds = sorted(by_seed)
        out[k] = ([by_seed[s]["asr"] for s in seeds], [by_seed[s]["accuracy"] for s in seeds], seeds)
    return out


def frozen_subset(rung):
    """The (asr, acc, seeds) triple restricted to the pre-top-up seed block.

    Used for the trend tests and for the reproduction check on the published n=5 pair. Selecting by
    RANK (the lowest FROZEN_N seeds) rather than by the literal seed numbers, so this stays correct if
    a future top-up chooses a non-contiguous block; the assertion that the block is 42-46 is made
    once, explicitly, in main().
    """
    asr, acc, seeds = rung
    keep = set(sorted(seeds)[:FROZEN_N])
    idx = [i for i, s in enumerate(seeds) if s in keep]
    return ([asr[i] for i in idx], [acc[i] for i in idx], [seeds[i] for i in idx])


def paired_t(hi, lo, alternative):
    """One-sided paired t on the per-seed endpoint difference, or (nan, nan) without scipy.

    This is the ENDPOINT test, at the same n and on the same differences as paired_ci95, which is why
    it can be printed beside that interval. It is NOT the Jonckheere-Terpstra trend test: JT reads all
    four rungs and stays at n=5 by pre-registration, and conflating the two would put two different
    n's inside one parenthesis. Both are emitted, each labelled with its own n.
    """
    if sps is None:
        return float("nan"), float("nan")
    t, p = sps.ttest_rel(hi, lo, alternative=alternative)
    return float(t), float(p)


def channel(family, kappa, field):
    if not os.path.exists(ADM):
        return None
    return json.load(open(ADM))["summary"].get(f"{family}|{ARM}|{kappa}|{field}")


def coeff(family, kappa, field):
    if not os.path.exists(COEF):
        return None
    return json.load(open(COEF))["summary"].get(f"{family}|{kappa}|{field}")


def share(family, kappa):
    if not os.path.exists(ADM):
        return None
    return json.load(open(ADM))["adv_coeff_share"].get(f"{family}|{kappa}")


def fmt(v, p=3, signed=False):
    if v is None:
        return "---"
    if abs(v) and abs(v) < 10 ** -(p + 1):
        return f"{v:.1e}"
    return f"{v:+.{p}f}" if signed else f"{v:.{p}f}"


def paired_ci95(hi, lo):
    """Two-sided 95% Student-t interval on the PER-SEED ASR difference, top rung minus identity.

    Paired, not two-sample, and that is the design rather than a modelling choice: both rungs run at
    the same seeds, so each difference holds one data partition fixed and the seed-to-seed spread that
    dominates this suite drops out of the contrast. Its mean is identically mean(hi) - mean(lo), so the
    interval is centred on the asr_effect already reported and cannot disagree with it.

    This is what the sign-reversal claim needs and the published table never printed: two point
    estimates of opposite sign are only a reversal if their intervals say so.
    """
    d = np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)
    n = len(d)
    if n < 2:
        return {"n": n, "note": "no interval at n<2"}
    m, sd = float(d.mean()), float(d.std(ddof=1))
    hw = float(t_crit(n) * sd / np.sqrt(n))
    return {"n": n, "mean": m, "sd": sd, "t_crit": float(t_crit(n)), "half_width": hw,
            "lo": m - hw, "hi": m + hw, "excludes_zero": bool((m - hw) * (m + hw) > 0),
            "per_seed": [float(x) for x in d]}


def tex_exp(v, p=0):
    """A float in scientific notation as LaTeX math, so the emitted table needs no hand-editing."""
    m, e = f"{v:.{p}e}".split("e")
    return rf"{m}\mathrm{{e}}{{-}}{abs(int(e)):02d}" if int(e) < 0 else \
        rf"{m}\mathrm{{e}}{{+}}{int(e):02d}"


def tex_p(v, p=1):
    """A p-value as $m{\\times}10^{-e}$, the form the papers already use for small p-values.

    Deliberately NOT tex_exp's 'e-09' style: that reads back a machine zero in the Delta_c row, where
    the point is that it is floating-point noise, whereas a p-value is quoted for a human. One
    convention per kind of number, so no reader has to decide whether two notations mean two things.
    """
    m, e = f"{v:.{p}e}".split("e")
    return rf"{m}{{\times}}10^{{{int(e)}}}" if int(e) else m


def main():
    for p in (LADDER, MODES, TOPUP, ADM, COEF):
        if not os.path.exists(p):
            sys.exit(f"missing {os.path.relpath(p, base)} -- this table only assembles frozen "
                     "artifacts and computes nothing new")

    cols = []
    for label, family, src, pattern in DESIGNS:
        a = asr_rungs(src, pattern)
        if LO not in a or HI not in a:
            sys.exit(f"{label}: the {ARM}/{ATTACK} arm is missing its identity or top rung")
        lo_asr, lo_acc, seeds = a[LO]
        hi_asr, hi_acc, seeds_hi = a[HI]
        if seeds != seeds_hi:
            sys.exit(f"{label}: rung seeds differ ({seeds} vs {seeds_hi}); the arm is not seed-matched")
        # The endpoint contrast is n=20; the trend test is n=5. Both subsets are taken here rather
        # than downstream, so no later line can read the wrong one by accident.
        f_lo, f_hi = frozen_subset(a[LO]), frozen_subset(a[HI])
        if f_lo[2] != list(range(42, 42 + FROZEN_N)):
            sys.exit(f"{label}: the frozen block is {f_lo[2]}, not 42--{41 + FROZEN_N}; the published "
                     "n=5 pair cannot be reproduced from it")
        t, p_welch_down = welch_less(hi_asr, lo_asr)   # post-hoc two-rung diagnostic, JSON only
        # The published trend test: JT across all four rungs, in the frozen rung order, AT n=5. The
        # interior rungs were not topped up, so this is the only n at which all four rungs exist on a
        # common seed set, and the pre-registration fixes it there regardless.
        J, z, p_up, p_down, p_perm_up = jonckheere(
            [frozen_subset(a[k])[0] for k in KAPPAS if k in a])
        # The same JT over whatever seeds each rung has, kept in the JSON ONLY, as the mixed-n
        # diagnostic it is. It is not quoted anywhere and must not be: its four groups have n's
        # [20, 5, 5, 20], so its null distribution is not the one the frozen test was specified under.
        Jm, zm, p_up_m, p_down_m, _ = jonckheere([a[k][0] for k in KAPPAS if k in a])
        cols.append({
            "label": label, "family": family, "arm": ARM, "attack": ATTACK, "seeds": seeds,
            "n": len(seeds), "frozen_seeds": f_lo[2], "frozen_n": len(f_lo[2]),
            "asr_source": os.path.relpath(src, base),
            "topup_source": os.path.relpath(TOPUP, base),
            "kappa_range": [LO, HI], "rho_range": [float(np.exp(2 * LO)), float(np.exp(2 * HI))],
            "asr_identity": float(np.mean(lo_asr)), "asr_top": float(np.mean(hi_asr)),
            "asr_identity_per_seed": lo_asr, "asr_top_per_seed": hi_asr,
            "asr_effect": float(np.mean(hi_asr) - np.mean(lo_asr)),
            "asr_effect_ci95": paired_ci95(hi_asr, lo_asr),
            "acc_identity": float(np.mean(lo_acc)), "acc_top": float(np.mean(hi_acc)),
            # The endpoint p at the SAME n as the interval beside it, in the direction this leg's
            # effect actually has. This is the p the table prints from Round 63 on.
            "endpoint_t": paired_t(hi_asr, lo_asr,
                                   "less" if np.mean(hi_asr) < np.mean(lo_asr) else "greater")[0],
            "endpoint_p": paired_t(hi_asr, lo_asr,
                                   "less" if np.mean(hi_asr) < np.mean(lo_asr) else "greater")[1],
            "endpoint_p_alternative": "less" if np.mean(hi_asr) < np.mean(lo_asr) else "greater",
            # The published n=5 pair, recomputed, so the top-up cannot rewrite what was published.
            "asr_effect_frozen_n": float(np.mean(f_hi[0]) - np.mean(f_lo[0])),
            "asr_effect_frozen_ci95": paired_ci95(f_hi[0], f_lo[0]),
            "acc_top_frozen_n": float(np.mean(f_hi[1])),
            "jt_J": J, "jt_z": z, "jt_p_rise": p_up, "jt_p_fall": p_down,
            "jt_p_rise_perm": p_perm_up, "jt_n": len(f_lo[0]),
            "jt_note": ("Jonckheere-Terpstra over all four rungs at n=5 (seeds 42-46), the frozen "
                        "test and the published p-values; the interior rungs were not topped up"),
            "jt_mixed_n_diagnostic": {"J": Jm, "z": zm, "p_rise": p_up_m, "p_fall": p_down_m,
                                      "n_per_rung": [len(a[k][0]) for k in KAPPAS if k in a],
                                      "note": "JSON only, never quoted: unequal group sizes across "
                                              "rungs, not the specified test"},
            "welch_t_two_rung": t, "welch_p_fall_two_rung": p_welch_down,
            "welch_note": "post-hoc two-rung diagnostic; the papers quote the JT p-values",
            "agg_disp_top": channel(family, HI, "agg_disp"),
            "decision_top": channel(family, HI, "decision"),
            "admission_top": channel(family, HI, "admission"),
            "share_identity": share(family, LO), "share_top": share(family, HI),
            "delta_c_identity": coeff(family, LO, "delta_c"),
            "delta_c_top": coeff(family, HI, "delta_c"),
            "all_rungs": {str(k): {"asr": float(np.mean(a[k][0])), "acc": float(np.mean(a[k][1])),
                                   # Per rung, because it is [20, 5, 5, 20] and no display of this
                                   # ladder may print the grid without it.
                                   "n": len(a[k][0]), "seeds": a[k][2],
                                   "agg_disp": channel(family, k, "agg_disp"),
                                   "decision": channel(family, k, "decision"),
                                   "admission": channel(family, k, "admission"),
                                   "share": share(family, k),
                                   "delta_c": coeff(family, k, "delta_c")}
                          for k in KAPPAS if k in a},
        })

    L, S = cols[0], cols[1]

    print("=== THE CONFOUNDED LADDER vs THE MODE-S INSTRUMENT "
          f"({ARM}, {ATTACK.replace('committed_', '')}, seeds {L['seeds']}) ===")
    print("    Assembled read-only from frozen artifacts. Nothing is trained and no ASR is "
          "recomputed here.\n")
    w = 26
    rows = [
        ("downstream $d_2$", ARM.replace("_", " "), ARM.replace("_", " ")),
        ("attack", ATTACK.replace("committed_", ""), ATTACK.replace("committed_", "")),
        ("seeds (endpoints)", f"{L['seeds'][0]}-{L['seeds'][-1]} (n={L['n']})",
         f"{S['seeds'][0]}-{S['seeds'][-1]} (n={S['n']})"),
        ("seeds (mechanism, trend)", f"42-46 (n={L['frozen_n']})", f"42-46 (n={S['frozen_n']})"),
        ("nominal dose", f"kappa {LO}->{HI}, rho 1->{L['rho_range'][1]:.1f}",
         f"kappa {LO}->{HI}, rho 1->{S['rho_range'][1]:.1f}"),
        ("benign coefficients", "mean 1 over ALL clients", "mean 1 over BENIGN clients"),
        ("adversarial coefficients", "same spread (attenuated)", "pinned c=1.0 exactly"),
        ("adv. coeff share  [n=5]", f"{fmt(L['share_identity'])} -> {fmt(L['share_top'])}",
         f"{fmt(S['share_identity'], 6)} -> {fmt(S['share_top'], 6)}"),
        ("Delta_c  [n=5]", f"{fmt(L['delta_c_identity'], signed=True)} -> "
                           f"{fmt(L['delta_c_top'], signed=True)}",
         f"{fmt(S['delta_c_identity'], signed=True)} -> {fmt(S['delta_c_top'], signed=True)}"),
        ("aggregate displ.  [n=5]", f"0.000 -> {fmt(L['agg_disp_top'])}",
         f"0.000 -> {fmt(S['agg_disp_top'])}"),
        ("decision change  [n=5]", f"0.000 -> {fmt(L['decision_top'])}",
         f"0.000 -> {fmt(S['decision_top'])}"),
        ("dLambda_a  [n=5]", f"0.000 -> {fmt(L['admission_top'])}",
         f"0.000 -> {fmt(S['admission_top'])}"),
        (f"mean clean acc  [n={L['n']}]", f"{fmt(L['acc_identity'])} -> {fmt(L['acc_top'])}",
         f"{fmt(S['acc_identity'])} -> {fmt(S['acc_top'])}"),
        (f"ASR  [n={L['n']}]", f"{fmt(L['asr_identity'])} -> {fmt(L['asr_top'])}",
         f"{fmt(S['asr_identity'])} -> {fmt(S['asr_top'])}"),
        (f"ASR EFFECT  [n={L['n']}]",
         f"{fmt(L['asr_effect'], signed=True)}  (endpoint p={L['endpoint_p']:.2e})",
         f"{fmt(S['asr_effect'], signed=True)}  (endpoint p={S['endpoint_p']:.2e})"),
        ("  same, at frozen n=5", f"{fmt(L['asr_effect_frozen_n'], signed=True)}",
         f"{fmt(S['asr_effect_frozen_n'], signed=True)}"),
        ("JT trend z  [n=5]", f"{L['jt_z']:+.3f}", f"{S['jt_z']:+.3f}"),
        ("JT trend p  [n=5]", f"p_fall={L['jt_p_fall']:.4f}", f"p_rise={S['jt_p_rise']:.3f}"),
    ]
    print(f"  {'':26s} {L['label']:>30s} {S['label']:>30s}")
    print("  " + "-" * 88)
    for name, a, b in rows:
        print(f"  {name:26s} {a:>30s} {b:>30s}")

    print("\n=== ASSERTIONS ===")
    same_seeds = L["seeds"] == S["seeds"]
    gap = abs(L["asr_identity"] - S["asr_identity"])
    per_seed_gap = max(abs(x - y) for x, y in zip(L["asr_identity_per_seed"],
                                                 S["asr_identity_per_seed"])) if same_seeds else None
    print(f"  seed sets identical: {same_seeds} (n={L['n']}, {L['seeds'][0]}--{L['seeds'][-1]})")
    # The published n=5 pair must survive the top-up as an arithmetic fact about a subset. If it does
    # not, either the merge double-counted a seed or the frozen files moved, and either way the paper's
    # Class B sites -- which still quote -0.272 / +0.098 against thresholds frozen before these runs --
    # would be quoting a number this repo no longer produces.
    PUBLISHED_N5 = {"dose": -0.272, "doseS": +0.098}
    ok_frozen = True
    for c in cols:
        want = PUBLISHED_N5[c["family"]]
        got = c["asr_effect_frozen_n"]
        good = abs(got - want) < 5e-4
        ok_frozen &= good
        print(f"  published n=5 effect reproduces on the frozen subset ({c['family']}): "
              f"{got:+.4f} vs {want:+.3f}  {'OK' if good else '<- DOES NOT REPRODUCE'}")
    if not ok_frozen:
        print("  ! The n=5 pair the paper still quotes at its pre-registered sites does not "
              "reproduce. Fix this before reading anything below.")
    print(f"  the two designs share their identity cell: mean gap {gap:.2e}"
          + (f", worst per-seed gap {per_seed_gap:.2e}" if per_seed_gap is not None else "")
          + f"  (tolerance {SHARED_TOL:g})  {'OK' if gap < SHARED_TOL else '<- NOT SHARED'}")
    ok_shared = gap < SHARED_TOL and (per_seed_gap is None or per_seed_gap < SHARED_TOL)
    if not ok_shared:
        print("  ! The identity cell is NOT shared, so the two columns do not start from the same "
              "point and the contrast as stated does not hold.")
    for nm, key, tol in (("aggregate displacement", "agg_disp_top", 0.10),
                         ("decision change", "decision_top", 0.10),
                         ("mean clean accuracy", "acc_top", 0.05)):
        d = abs((L[key] or 0) - (S[key] or 0))
        print(f"  matched at the top rung on {nm:24s}: |{fmt(L[key])} - {fmt(S[key])}| = {d:.3f} "
              f"{'(matched)' if d <= tol else f'(NOT matched within {tol})'}")
    d_share = abs((L["share_top"] or 0) - (S["share_top"] or 0))
    d_dc = abs((L["delta_c_top"] or 0) - (S["delta_c_top"] or 0))
    print(f"  differing on the adversarial coefficient channel: share gap {d_share:.3f}, "
          f"Delta_c gap {d_dc:.3f}")
    print(f"  trend tests (Jonckheere-Terpstra over all four rungs AT n={L['jt_n']}, the frozen test "
          f"and the published p-values):\n    ladder z={L['jt_z']:+.3f}, p_fall={L['jt_p_fall']:.6f}; "
          f"instrument z={S['jt_z']:+.3f}, p_rise={S['jt_p_rise']:.4f}")
    print(f"    endpoint paired t at n={L['n']} (the p the table now prints, same n and same "
          f"differences as the interval beside it):\n      ladder t={L['endpoint_t']:+.3f}, "
          f"p_fall={L['endpoint_p']:.2e}; instrument t={S['endpoint_t']:+.3f}, "
          f"p_rise={S['endpoint_p']:.2e}")
    print(f"    mixed-n JT over rungs of size {L['jt_mixed_n_diagnostic']['n_per_rung']} is computed "
          "into the JSON and quoted NOWHERE: unequal groups, not the specified test.")
    print(f"    post-hoc two-rung Welch (JSON only, not in the papers): ladder "
          f"p_fall={L['welch_p_fall_two_rung']:.4f}")
    print(f"  sign reversal: ladder {fmt(L['asr_effect'], signed=True)} vs instrument "
          f"{fmt(S['asr_effect'], signed=True)} -- "
          f"{'opposite signs' if L['asr_effect'] * S['asr_effect'] < 0 else 'SAME SIGN'}")
    # Two point estimates of opposite sign are a reversal only if the intervals agree, so they are
    # printed next to the signs rather than left in the JSON for a reader to assemble.
    cl, cs = L["asr_effect_ci95"], S["asr_effect_ci95"]
    print(f"    paired 95% t intervals (n={cl['n']}, t*={cl['t_crit']:.3f}): "
          f"ladder [{cl['lo']:+.4f}, {cl['hi']:+.4f}] (sd {cl['sd']:.4f}); "
          f"instrument [{cs['lo']:+.4f}, {cs['hi']:+.4f}] (sd {cs['sd']:.4f})")
    both_excl = cl["excludes_zero"] and cs["excludes_zero"]
    disjoint = cl["hi"] < cs["lo"] or cs["hi"] < cl["lo"]
    print(f"    both exclude zero: {both_excl}; intervals disjoint: {disjoint}"
          + ("" if both_excl and disjoint else
             "   <-- the reversal is NOT interval-separated; report the intervals, not the signs"))
    print(f"  NON-MATCH, disclosed: adversarial influence Lambda_a moves in BOTH designs -- ladder "
          f"{fmt(L['admission_top'])} vs instrument {fmt(S['admission_top'])} "
          f"({L['admission_top'] / max(S['admission_top'], 1e-12):.1f}x). Mode S pins the "
          f"coefficient share\n    exactly, which for a coordinate-wise order statistic does not "
          "force admitted mass constant. Neither column is a clean statistic-only intervention.")

    print("\n=== BOTH FULL LADDERS, for the appendix ===")
    for c in cols:
        print(f"\n  -- {c['label']} ({c['family']}) --")
        # n per rung, first column after kappa, because this ladder is [20, 5, 5, 20] and a grid
        # without it reads as one uniform sample.
        print(f"  {'kappa':>6} {'n':>3} {'ASR':>7} {'acc':>7} {'aggDisp':>8} {'dDec':>7} {'dAdm':>7} "
              f"{'share':>9} {'Delta_c':>9}")
        for k in KAPPAS:
            r = c["all_rungs"].get(str(k))
            if r is None:
                continue
            print(f"  {k:>6} {r['n']:>3} {r['asr']:7.3f} {r['acc']:7.3f} {fmt(r['agg_disp']):>8} "
                  f"{fmt(r['decision']):>7} {fmt(r['admission']):>7} {fmt(r['share'], 6):>9} "
                  f"{fmt(r['delta_c'], signed=True):>9}")
        print("       (ASR and acc are per-rung means at that rung's own n; every other column is "
              "the frozen n=5 measurement)")

    # LaTeX. Emitted here so the paper's table cannot drift from the numbers above.
    print("\n=== LATEX (copy verbatim; regenerate rather than edit) ===\n")
    tex = [r"\begin{tabular}{lcc}", r"\toprule",
           r"& Confounded ladder & Instrument (Mode~S) \\",
           r"& \texttt{dose\_kappa} & \texttt{doseS\_kappa} \\", r"\midrule",
           rf"downstream $d_2$ & \texttt{{coord\_median}} & \texttt{{coord\_median}} \\",
           rf"attack & committed pixel & committed pixel \\",
           rf"seeds & {L['seeds'][0]}--{L['seeds'][-1]} & {S['seeds'][0]}--{S['seeds'][-1]} \\",
           rf"nominal dose & $\kappa\,{LO}\!\to\!{HI}$, $\rho\,1\!\to\!{L['rho_range'][1]:.1f}$ "
           rf"& $\kappa\,{LO}\!\to\!{HI}$, $\rho\,1\!\to\!{S['rho_range'][1]:.1f}$ \\",
           r"\midrule",
           r"benign coefficients & mean $1$ over \emph{all} clients "
           r"& mean $1$ over \emph{benign} clients \\",
           r"adversarial coefficients & drawn from the same spread & pinned $c=1.0$ exactly \\",
           # Every row carries its own n. The mechanism rows are frozen at n=5 because the top-up
           # records accuracy and ASR only; the outcome rows are n=20. A reader must not have to
           # infer either from the seeds row.
           rf"adv.\ coefficient share ($n{{=}}{L['frozen_n']}$) & ${fmt(L['share_identity'])} \to "
           rf"\mathbf{{{fmt(L['share_top'])}}}$ & ${fmt(S['share_identity'], 6)}$, constant \\",
           rf"$\Delta_c$ ($n{{=}}{L['frozen_n']}$) & ${fmt(L['delta_c_identity'], signed=True)} \to "
           rf"\mathbf{{{fmt(L['delta_c_top'], signed=True)}}}$ & $0$ by construction "
           rf"(read back ${tex_exp(abs(S['delta_c_top']))}$) \\",
           r"\midrule",
           rf"aggregate displacement ($n{{=}}{L['frozen_n']}$) & $0.000 \to "
           rf"{fmt(L['agg_disp_top'])}$ & $0.000 \to {fmt(S['agg_disp_top'])}$ \\",
           rf"decision change ($n{{=}}{L['frozen_n']}$) & $0.000 \to {fmt(L['decision_top'])}$ "
           rf"& $0.000 \to {fmt(S['decision_top'])}$ \\",
           rf"adversarial influence $\Delta\Lambda_a$ ($n{{=}}{L['frozen_n']}$) "
           rf"& $0.000 \to \mathbf{{{fmt(L['admission_top'])}}}$ "
           rf"& $0.000 \to \mathbf{{{fmt(S['admission_top'])}}}$ \\",
           rf"mean clean accuracy ($n{{=}}{L['n']}$) & ${fmt(L['acc_identity'])} \to "
           rf"{fmt(L['acc_top'])}$ & ${fmt(S['acc_identity'])} \to {fmt(S['acc_top'])}$ \\",
           rf"ASR ($n{{=}}{L['n']}$) & ${fmt(L['asr_identity'])} \to {fmt(L['asr_top'])}$ "
           rf"& ${fmt(S['asr_identity'])} \to {fmt(S['asr_top'])}$ \\",
           r"\midrule",
           # The p beside each effect is the ENDPOINT paired t at the effect's own n, not the JT trend
           # p: the trend test stays at n=5 by pre-registration, and printing it here would put two
           # n's in one parenthesis. The JT values keep their home in the appendix disclosure.
           rf"\textbf{{ASR effect}} ($n{{=}}{L['n']}$) "
           rf"& $\mathbf{{{fmt(L['asr_effect'], signed=True)}}}$ "
           rf"($p_{{\downarrow}}={tex_p(L['endpoint_p'])}$) "
           rf"& $\mathbf{{{fmt(S['asr_effect'], signed=True)}}}$ "
           rf"($p_{{\uparrow}}={tex_p(S['endpoint_p'])}$) \\",
           rf"\quad paired $95\%$ interval & ${{[{L['asr_effect_ci95']['lo']:+.3f},"
           rf"{L['asr_effect_ci95']['hi']:+.3f}]}}$ "
           rf"& ${{[{S['asr_effect_ci95']['lo']:+.3f},{S['asr_effect_ci95']['hi']:+.3f}]}}$ \\",
           r"\bottomrule", r"\end{tabular}"]
    print("\n".join(tex))

    json.dump({"description": "The confounded ladder vs the Mode-S instrument on the one arm that has "
                              "both. Assembled read-only from frozen artifacts; nothing is trained.",
               "arm": ARM, "attack": ATTACK, "kappas": KAPPAS,
               "sources": [os.path.relpath(p, base) for p in (LADDER, MODES, ADM, COEF)],
               "columns": cols,
               "assertions": {"seed_sets_identical": same_seeds,
                              "n": L["n"], "frozen_n": L["frozen_n"],
                              "published_n5_pair_reproduces": bool(ok_frozen),
                              "published_n5_pair": PUBLISHED_N5,
                              "mixed_n_rows": ("aggregate displacement, decision change, "
                                               "dLambda_a, adv. coefficient share and Delta_c are "
                                               "n=5; ASR, mean clean accuracy and the ASR effect "
                                               "are n=20; the JT trend p-values are n=5"),
                              "identity_cell_mean_gap": gap,
                              "identity_cell_worst_per_seed_gap": per_seed_gap,
                              "shared_tolerance": SHARED_TOL,
                              "identity_cell_shared": ok_shared,
                              "sign_reversal": bool(L["asr_effect"] * S["asr_effect"] < 0),
                              # The interval-level form of the same claim, kept separate from the
                              # sign-level one: signs can differ while intervals overlap, and it is
                              # the conjunction the figure and the caption assert.
                              "both_effects_exclude_zero": both_excl,
                              "effect_intervals_disjoint": disjoint,
                              "sign_reversal_interval_separated": bool(both_excl and disjoint),
                              "top_rung_gaps": {
                                  "agg_disp": abs((L["agg_disp_top"] or 0) - (S["agg_disp_top"] or 0)),
                                  "decision": abs((L["decision_top"] or 0) - (S["decision_top"] or 0)),
                                  "clean_accuracy": abs(L["acc_top"] - S["acc_top"]),
                                  "adv_coeff_share": d_share, "delta_c": d_dc}},
               "disclosed_non_match": (
                   f"aggregate adversarial influence Lambda_a moves in both designs: "
                   f"{L['admission_top']:.3f} in the ladder against {S['admission_top']:.3f} in the "
                   "instrument. Mode S pins the adversarial COEFFICIENT share exactly; for a "
                   "coordinate-wise order statistic that does not force Lambda_a to be constant. "
                   "Neither column is a clean statistic-only intervention, so the contrast is 'differ "
                   "chiefly in the coefficient channel', not 'only in it'. This is the same quantity as "
                   "build_channel_table.py's Delta Lambda_a column, not its Delta adm. column (support "
                   "of the adversarial mass), which reads 0.000 for every row."),
               "scope": ("A two-column contrast on one arm. It licenses the claim that the "
                         "adversarial coefficient channel is where the two designs differ; it does "
                         "not measure that channel's effect size, and no p-value is attached to the "
                         "difference between the two columns."),
               "latex": "\n".join(tex)}, open(OUT, "w"), indent=1)
    print(f"\nWrote {OUT}")
    return 0 if ok_shared else 1


if __name__ == "__main__":
    sys.exit(main())
