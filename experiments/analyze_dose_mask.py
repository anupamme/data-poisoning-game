"""
Score the Mode-M coordinate-masking arm against the rules frozen in
`experiments/pre_registration_dose_mask.md` (committed at 684b31e), and emit its `tab:channels` row.

WHAT THIS ARM CAN AND CANNOT ESTABLISH
Only one outcome generalizes the paper's negative: **FLAT with admission unchanged.** The prereg says
so in as many words. A rise with admission unchanged REFUTES us and becomes a scope condition in the
abstract. A rise with admission changed is uninformative about (P3)=/=>(P4) and is reported as
uninformative, not as a win. All three branches are coded below, so the script cannot only be run to
confirm.

THE SUBSTITUTION CONTINGENCY IS DECIDED BY THE ACCURACY GATE, NOT BY THE ASR
The primary contrast is m=0 -> m=0.8. If the m=0.8 rung's MEAN clean accuracy falls below 0.35 the
frozen rule reads the contrast on m=0.5 instead, because a low ASR under a collapsed model is not
preserved suppression. The switch is driven only by accuracy and is printed as a substitution whenever
it fires -- it was frozen in advance precisely so it cannot look like a post-hoc choice.

A PARTIAL LADDER IS NOT THE PRE-REGISTERED VERDICT
Non-negotiable 2 fixes seeds 42-51 and forbids truncating the list by inspection. So at n < 10 every
verdict below is stamped PROVISIONAL: it is the number at the n reached, and it is not the frozen
verdict. Nothing here waits for a threshold to be crossed.

THE ADMISSION PREMISE IS INHERITED, NOT RE-MEASURED HERE
"Admission unchanged" comes from `results/mask_admission.json` -- 5 seeds x 3 live rounds of the
transform's channel, not a per-round guarantee inside each 50-round ASR run, because `run_one` does not
log selections. That is exactly the standing of the flagship's own admission numbers. Stated, not
smoothed over.

t_crit IS IMPORTED, NEVER `T95[n-1]`, because that literal table stops at df = 9 and this arm's n = 10.

Reads results/dose_mask/, results/mask_admission.json, and (for the identity bit-identity check)
results/targeted_dose/ and results/dose_seed_topup/. Writes nothing. Adds no runs.

Run: python3 experiments/analyze_dose_mask.py
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.analyze_dose_response import jonckheere      # noqa: E402  the canonical test
from experiments.analyze_headline_cis import t_crit           # noqa: E402  NOT T95[n-1]
from experiments.analyze_tost_existing import load, tost      # noqa: E402

try:
    from scipy import stats as sps
except Exception:
    sps = None

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASK_ADM = os.path.join(BASE, "results", "mask_admission.json")

PREREG = "experiments/pre_registration_dose_mask.md"
PREREG_COMMIT = "684b31e"

# All carried forward unchanged from 5130cec; none is a new threshold.
MARGIN, ASR_CEILING, ACC_FLOOR = 0.15, 0.5, 0.35

DROPS = ["0.0", "0.2", "0.5", "0.8"]
IDENTITY = "0.0"
PRIMARY_TOP, FALLBACK_TOP = "0.8", "0.5"
KEY = "doseM_m{r}_then_krum|committed_scaling"
SEEDS10 = list(range(42, 52))
N_PLANNED = 10

# The frozen channel figures for krum, transcribed from the prereg's own table so the artifact can be
# checked against the document. Asserted, never printed as a result.
DOC_DECISION = {"0.0": 0.000, "0.2": 0.133, "0.5": 0.400, "0.8": 0.467}
DOC_ADMISSION = {"0.0": 0.000, "0.2": 0.000, "0.5": 0.000, "0.8": 0.000}
DOC_TOL = 5e-4
ADM_FLAT_TOL = 0.05          # the prereg's own GENERALIZATION threshold on max admission change

BITCHECK_TOL = 1e-9          # the prereg's own tolerance for the imported identity rung
DEGENERATE_MODE_M = (24, 120)


def prereg_frozen():
    """(ok, actual_hash): a prediction resting on an uncommitted document is not pre-registered."""
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%h", "--", PREREG],
                             cwd=BASE, capture_output=True, text=True, timeout=20)
        h = out.stdout.strip() or None
    except Exception:
        h = None
    return (h is not None and h.startswith(PREREG_COMMIT[:7])), h


def rows(cells, rung, score_only):
    """{seed: (asr, accuracy, source)} for one rung of one ladder."""
    key = KEY.format(r=rung) + ("|score_only" if score_only else "")
    c = cells.get(key)
    if not c:
        return {}
    return {int(r["seed"]): (float(r["asr"]), float(r["accuracy"]), r.get("source", "?"))
            for r in c["per_seed"]}


def ladder(cells, score_only):
    return {m: rows(cells, m, score_only) for m in DROPS}


def paired(lad, lo, hi):
    """Per-seed ASR(hi) - ASR(lo) on seeds present in BOTH rungs, plus the seed list."""
    common = sorted(set(lad[lo]) & set(lad[hi]))
    if len(common) < 2:
        return None, common
    d = np.array([lad[hi][s][0] - lad[lo][s][0] for s in common], dtype=float)
    return d, common


# Every file that can supply the m=0 rung, checked against ALL of them rather than against whichever
# one the runner happened to read. The chain is Round 11 -> Mode S -> the top-up -> here.
#
# A discrepancy worth recording: the pre-registration says the identity rung is imported "from
# `results/dose_response/` for seeds 42-46", while `run_dose_mask.identity_rung()` reads
# `results/targeted_dose/`. The two are the SAME VALUES -- targeted_dose itself imported seeds 42-46
# from dose_response -- so the prereg names the ultimate origin and the runner reads the intermediate
# carrier. Immaterial, and verified below rather than argued: the check spans all three directories, so
# the prereg's named source is checked directly.
IDENTITY_SOURCES = [
    ("dose_response",    "dose_kappa0.0_then_krum|committed_scaling"),    # Round 11, the prereg's name
    ("targeted_dose",    "doseS_kappa0.0_then_krum|committed_scaling"),   # what the runner reads
    ("dose_seed_topup",  "doseS_kappa0.0_then_krum|committed_scaling"),   # seeds 47+
]


def check_identity_import(lad):
    """The m=0 rung claims to BE the kappa=0 run. Verified here rather than cited.

    The prereg's argument is that m=0 returns the update list unwrapped and `run_one`'s participant RNG
    stream does not depend on d1's name, so the m=0 rung is the same computation as the Mode-S kappa=0
    rung, which is itself Round 11's `dose_kappa0.0` run. Any seed present in more than one of those
    places must therefore agree to floating noise -- including seeds the runner COMPUTED in place rather
    than importing, which is exactly the case the citation does not cover.
    """
    seen, contributors = {}, []
    for dirname, key in IDENTITY_SOURCES:
        c = load(dirname)
        cell = c.get(key) if c else None
        if not cell:
            continue
        contributors.append(dirname)
        for r in cell["per_seed"]:
            seen.setdefault(int(r["seed"]), []).append((dirname, float(r["asr"])))

    problems = []
    # First, the sources against each other: a seed carried by two files must carry one value. Checked
    # rather than assumed, because the whole import argument rests on these being one computation.
    for s, vals in sorted(seen.items()):
        for dirname, v in vals[1:]:
            if abs(v - vals[0][1]) > BITCHECK_TOL:
                problems.append(f"kappa=0 seed {s} diverges between sources: "
                                f"{vals[0][0]} {vals[0][1]!r} vs {dirname} {v!r}")
    ref = {s: vals[0][1] for s, vals in seen.items()}

    checked = computed = 0
    for s, (asr, _acc, src) in sorted(lad[IDENTITY].items()):
        if s not in ref:
            continue
        checked += 1
        if src == "<computed here>":
            computed += 1
        if abs(asr - ref[s]) > BITCHECK_TOL:
            problems.append(f"m=0 seed {s} ({src}): {asr!r} vs kappa=0 {ref[s]!r}")
    return problems, checked, computed, contributors


def admission_premise():
    """(arm_type, decision, admission, problems) for krum, read from the frozen channel measurement."""
    if not os.path.exists(MASK_ADM):
        return None, {}, {}, [f"{MASK_ADM} absent: the GENERALIZATION premise cannot be checked"]
    j = json.load(open(MASK_ADM))
    s = j.get("summary", {})
    dec = {m: s.get(f"doseM|krum|{m}|decision") for m in DROPS}
    adm = {m: s.get(f"doseM|krum|{m}|admission") for m in DROPS}
    problems = []
    for m in DROPS:
        for name, got, doc in (("decision", dec[m], DOC_DECISION[m]),
                               ("admission", adm[m], DOC_ADMISSION[m])):
            if got is None:
                problems.append(f"krum m={m}: {name} missing from the artifact")
            elif abs(got - doc) > DOC_TOL:
                problems.append(f"krum m={m} {name}: artifact {got:.6f} vs pre-registration {doc:.3f}")
    return j.get("eligibility", {}).get("arm_type"), dec, adm, problems


def interval(d):
    r = tost(d, MARGIN)
    r["hw95"] = t_crit(len(d)) * r["se"]
    r["contained"] = r["ci95"][0] > -MARGIN and r["ci95"][1] < MARGIN
    return r


def print_ladder(lad, tag):
    """Per-rung state at the n actually reached; returns (rung means, acc means, ceiling/floor ok)."""
    print(f"  --- {tag} ladder ---")
    print(f"  {'m':5s} {'n':>3s} {'mean ASR':>9s} {'mean acc':>9s}   provenance / flags")
    groups, asr_mean, acc_mean, ok = [], {}, {}, True
    for m in DROPS:
        r = lad[m]
        seeds = sorted(r)
        if not seeds:
            print(f"  {m:5s} {0:3d} {'--':>9s} {'--':>9s}   (no runs yet)")
            groups.append(np.array([]))
            continue
        asr = np.array([r[s][0] for s in seeds], dtype=float)
        acc = np.array([r[s][1] for s in seeds], dtype=float)
        groups.append(asr)
        asr_mean[m], acc_mean[m] = float(asr.mean()), float(acc.mean())
        n_imp = sum(1 for s in seeds if r[s][2] != "<computed here>")
        flags = []
        if asr.mean() >= ASR_CEILING:
            flags.append(f"RUNG MEAN >= {ASR_CEILING} CEILING")
            ok = False
        if acc.mean() < ACC_FLOOR:
            # Named in advance: at m=0.8 the emitted update is heavily sparsified and amplified, so a
            # low ASR under a failed floor is a collapsed model and is uninterpretable.
            flags.append(f"RUNG BELOW {ACC_FLOOR} ACC FLOOR -- a low ASR here is a collapsed model")
            ok = False
        low = [s for s in seeds if r[s][1] < ACC_FLOOR]
        if low:
            flags.append(f"seeds below floor (flagged, NOT excluded): {low}")
        print(f"  {m:5s} {len(seeds):3d} {asr.mean():9.4f} {acc.mean():9.4f}   "
              f"{len(seeds) - n_imp} computed + {n_imp} imported"
              + ("   ** " + "; ".join(flags) if flags else ""))
    return groups, asr_mean, acc_mean, ok


def main():
    cells = load("dose_mask")
    if cells is None:
        print("results/dose_mask/summary.json absent: the arm has not been launched.")
        return 1

    ok, actual = prereg_frozen()
    state = f"committed at {actual}" if actual else "UNTRACKED -- not a pre-registration"
    print(f"[{'OK' if ok else 'MISMATCH'}] rules frozen in {PREREG}: recorded {PREREG_COMMIT}, {state}")
    if not ok:
        print("  REFUSING TO SCORE: the rules being scored are not demonstrably prospective.")
        return 1

    arm_type, dec, adm, adm_problems = admission_premise()
    if adm_problems:
        print("\nCHANNEL ARTIFACT/PRE-REGISTRATION DRIFT -- refusing to score:")
        for p in adm_problems:
            print(f"  {p}")
        return 1
    max_adm = max(adm[m] for m in DROPS)
    print(f"[OK] channel premise reproduces: krum decision "
          f"{'/'.join(f'{dec[m]:.3f}' for m in DROPS)}, admission "
          f"{'/'.join(f'{adm[m]:.3f}' for m in DROPS)}")
    print(f"     arm_type = {arm_type}, max admission change {max_adm:.4f} "
          f"<= {ADM_FLAT_TOL} threshold: {max_adm <= ADM_FLAT_TOL}")
    admission_unchanged = max_adm <= ADM_FLAT_TOL
    if not admission_unchanged:
        print("     ** ADMISSION MOVED: per the frozen branch, a rise in ASR here is a confirmation of")
        print("        the admission reading and a refutation of nothing, and is UNINFORMATIVE about")
        print("        the (P3)=/=>(P4) non-implication. It is not scored as generalization.")

    full = ladder(cells, False)
    so = ladder(cells, True)

    bit_problems, bit_checked, bit_computed, bit_srcs = check_identity_import(full)
    if bit_problems:
        print("\nIDENTITY-RUNG BIT-IDENTITY FAILED -- refusing to score "
              f"(tolerance {BITCHECK_TOL}):")
        for p in bit_problems:
            print(f"  {p}")
        print("  The m=0 rung is claimed to BE the kappa=0 run; if it is not, the ladder has no anchor.")
        return 1
    print(f"[OK] identity rung: {bit_checked} seeds agree with the kappa=0 rung to {BITCHECK_TOL} "
          f"({bit_computed} computed in place, not imported)")
    print(f"     chain checked across {', '.join(bit_srcs)} -- including the source the "
          f"pre-registration names")

    print("\n=== MODE M, coordinate masking: krum / committed_scaling (item B) ===")
    print(f"Frozen prediction: FLAT. Margin +-{MARGIN}, ceiling {ASR_CEILING}, floor {ACC_FLOOR}, "
          f"seeds 42-51.\n")

    g_full, asr_full, acc_full, ok_full = print_ladder(full, "full")
    print()
    g_so, asr_so, acc_so, ok_so = print_ladder(so, "score-only")

    # The substitution is decided by the ACCURACY GATE on the top rung, never by its ASR.
    top = PRIMARY_TOP
    if PRIMARY_TOP in acc_full and acc_full[PRIMARY_TOP] < ACC_FLOOR:
        top = FALLBACK_TOP
        print(f"\n  ** SUBSTITUTION, disclosed as one: the m={PRIMARY_TOP} rung's mean accuracy "
              f"{acc_full[PRIMARY_TOP]:.4f} < {ACC_FLOOR},")
        print(f"     so the equivalence contrast is read on m={FALLBACK_TOP} against m=0, per the "
              f"contingency frozen at {PREREG_COMMIT}. The switch is driven by accuracy alone.")
    elif PRIMARY_TOP not in acc_full:
        print(f"\n  the m={PRIMARY_TOP} rung has no runs yet; the primary contrast is not yet formable.")

    verdicts = {}
    for tag, lad in (("full", full), ("score-only", so)):
        d, common = paired(lad, IDENTITY, top)
        print(f"\n  EQUIVALENCE, {tag} ladder, m=0 -> m={top}")
        if d is None:
            print(f"    only {len(common)} paired seed(s); no interval is formed and no verdict is "
                  f"scored.")
            verdicts[tag] = None
            continue
        n = len(d)
        r = interval(d)
        print(f"    n = {n} of a planned {N_PLANNED} (seeds {common[0]}-{common[-1]})")
        print(f"    paired Delta = {r['mean']:+.4f}   sd {r['sd']:.4f}   se {r['se']:.4f}")
        print(f"    95% CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]   half-width {r['hw95']:.4f} "
              f"= {100 * r['hw95'] / MARGIN:.1f}% of the +-{MARGIN} margin")
        print(f"    90% CI [{r['ci90'][0]:+.4f}, {r['ci90'][1]:+.4f}]   (the interval TOST decides on)")
        if sps is not None:
            print(f"    TOST: p_lower {r['p_lower']:.4g} | p_upper {r['p_upper']:.4g} | "
                  f"TOST p = {r['p_tost']:.4g}")
        verdicts[tag] = {"n": n, "r": r, "ok": ok_full if tag == "full" else ok_so}

    # Jonckheere-Terpstra in the pre-registered direction, on the full grid.
    jt = {}
    for tag, groups in (("full", g_full), ("score-only", g_so)):
        if not all(len(g) >= 2 for g in groups):
            continue
        J, z, p_inc, p_dec, p_perm = jonckheere(groups)
        jt[tag] = p_inc
        sizes = [len(g) for g in groups]
        print(f"\n  H-M-STATISTIC, {tag} ladder (JT increasing across m {'/'.join(DROPS)})")
        if len(set(sizes)) > 1:
            print(f"    ** PARTIAL LADDER: rung n = {sizes}, not all equal. Valid as computed, but not")
            print("       the pre-registered test and it must not be quoted as it.")
        print(f"    J = {J:.1f}, z = {z:+.3f}, p_increasing = {p_inc:.4f} "
              f"(permutation {p_perm:.4f}), p_decreasing = {p_dec:.4f}")

    # The frozen three-branch verdict, on the full ladder.
    v = verdicts.get("full")
    print(f"\n  VERDICT, full ladder")
    if v is None:
        print("    not yet formable.")
    else:
        provisional = v["n"] < N_PLANNED
        stamp = f"PROVISIONAL at n = {v['n']}" if provisional else f"at the frozen n = {N_PLANNED}"
        flat = v["r"]["contained"] and v["ok"]
        rising = jt.get("full") is not None and jt["full"] < 0.05
        if rising and flat:
            # Both criteria fire: a significant trend whose total displacement is still inside the
            # margin. The frozen H-M-statistic clause resolves it -- it says in as many words that a
            # JT rejection refutes H-M-admission -- and the coincidence is disclosed rather than used
            # to pick the friendlier branch.
            print(f"    ** BOTH CRITERIA FIRED ({stamp}): JT p = {jt['full']:.4f} < 0.05 AND the 95%")
            print(f"       interval is contained in (-{MARGIN}, +{MARGIN}). The frozen rule resolves")
            print("       this in favour of the trend test: H-M-STATISTIC CONFIRMED, H-M-ADMISSION")
            print("       REFUTED. The containment is disclosed, not used to select the branch.")
            print("       Reported as a SCOPE CONDITION on the central claim, IN THE ABSTRACT.")
        elif rising:
            print(f"    ** H-M-STATISTIC CONFIRMED, H-M-ADMISSION REFUTED ({stamp}): JT increasing "
                  f"p = {jt['full']:.4f} < 0.05.")
            if admission_unchanged:
                print("       Admission was unchanged at every rung, so this REFUTES US: the")
                print("       (P3)=/=>(P4) non-implication does not generalize beyond scalar")
                print("       rescaling, and the central claim acquires a scope condition naming the")
                print("       transformation class. Reported IN THE ABSTRACT.")
            else:
                print("       Admission MOVED, so this is UNINFORMATIVE about the non-implication and")
                print("       is reported as uninformative, not as a refutation and not as support.")
        elif flat and admission_unchanged:
            print(f"    H-M-ADMISSION CONFIRMED ON A SECOND TRANSFORMATION CLASS ({stamp}): "
                  f"[{v['r']['ci95'][0]:+.4f}, {v['r']['ci95'][1]:+.4f}]")
            print(f"       is contained in (-{MARGIN}, +{MARGIN}), every rung mean clears the ceiling")
            print("       and the floor, and admission is unchanged at every rung. This is the ONLY")
            print("       outcome that generalizes the negative. The 95% criterion is strictly more")
            print("       conservative than TOST at alpha=0.05.")
        elif flat:
            print(f"    FLAT but admission MOVED ({stamp}): uninformative about the non-implication,")
            print("       reported as uninformative.")
        else:
            reason = ("the 95% interval is not contained in the margin"
                      if not v["r"]["contained"] else
                      "a rung mean failed the ceiling or the accuracy floor")
            print(f"    INDETERMINATE ({stamp}): {reason}, and JT increasing does not reject.")
            print("       Neither confirmation nor refutation criterion is met. Per non-negotiable 1")
            print("       this is NOT scored in our favour.")
        if provisional:
            print(f"    Non-negotiable 2 fixes seeds 42-51: this is the number at n = {v['n']}, not the")
            print("       pre-registered verdict, and the seed list is not truncated by inspection.")

    # Localization: the control's only job.
    vf, vs = verdicts.get("full"), verdicts.get("score-only")
    if vf and vs:
        gap = vs["r"]["mean"] - vf["r"]["mean"]
        print(f"\n  LOCALIZATION (the score-only control's only job)")
        print(f"    full {vf['r']['mean']:+.4f} (n={vf['n']}) vs score-only {vs['r']['mean']:+.4f} "
              f"(n={vs['n']}); difference {gap:+.4f}")
        print("    Adversarial updates are unmasked in BOTH conditions, so the ladders differ only")
        print("    when a benign client is selected. A difference localizes the effect to the emitted")
        print("    content; agreement localizes it to which client Krum picks.")
        if vf["r"]["contained"] and vs["r"]["contained"]:
            print("    Both ladders FLAT, which is the frozen prediction.")
        elif vf["r"]["contained"] != vs["r"]["contained"]:
            print("    ** The ladders DISAGREE on containment: reported as a localization result.")

    print(f"\n  CAVEATS, recorded in advance and re-reported here")
    print(f"    Mode M has nothing to impose in {DEGENERATE_MODE_M[0]}/{DEGENERATE_MODE_M[1]} measured "
          f"rung-rounds (a round with no")
    print("      adversary, or with fewer than two benign participants). Identical to Mode S's figure.")
    print("    A mask is ONE non-rescaling transformation, not the class of them. Rotations,")
    print("      projections, quantization and sign compression are untested. The supportable claim is")
    print("      'the negative replicates on a second, structurally different transformation class'.")
    print("    Admission is a 5-seed x 3-round channel measurement of the TRANSFORM, not a per-round")
    print("      guarantee inside each 50-round ASR run. Same standing as the flagship's own numbers.")
    print("    Mode M is an instrument, not a defense: it reads adversary identity.")

    # LaTeX, emitted rather than transcribed. One row per ladder, for tab:channels.
    print("\n--- LaTeX rows (m=0 -> m={} contrast) ---".format(top))
    for tag in ("full", "score-only"):
        v2 = verdicts.get(tag)
        if not v2:
            print(f"Mode M, {tag} & --- & --- & --- & --- \\\\")
            continue
        r = v2["r"]
        flat = r["contained"] and v2["ok"]
        print(f"Mode M, {tag} & $n{{=}}{v2['n']}$ & ${r['mean']:+.3f}$ & "
              f"$[{r['ci95'][0]:+.3f},\\,{r['ci95'][1]:+.3f}]$ & "
              f"{'equivalent' if flat else 'not est.'} \\\\")
    return 0


if __name__ == "__main__":
    sys.exit(main())
