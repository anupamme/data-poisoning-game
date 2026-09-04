"""
Re-score the score-only / emit-only 2x2 at the topped-up seed count, against the rules frozen in
`experiments/pre_registration_emit_only_topup.md` (committed at 684b31e).

THIS IS A DISCLOSED POST-HOC EXTENSION, AND IT SAYS SO AT EVERY VERDICT
`pre_registration_emit_only.md` (b995f1b) froze "No seed addition. n=5, seeds 42-46, matching the arm
it is compared against." The top-up departs from that literal clause. The clause's stated RATIONALE is
comparability, and it is honoured -- all three cells are extended at the same rung on the same seeds --
but the text is still departed from, so no number here is described as prospective, and the quoted
clause is printed beside the result.

IT CANNOT CERTIFY SEPARABILITY, AND THAT WAS KNOWN BEFORE THE RUNS
The projected 95% half-widths at n=20 are 0.061 (dASR_EO) and 0.057 (residual), both WIDER than the
0.05 margin itself. So even a point estimate of exactly zero could not put a 95% interval inside
(-0.05, +0.05) at this variance. A pass of the frozen point-estimate rule is therefore printed as
"consistent with separability, NOT established" and never as SEPARABLE-full-stop, with the interval
beside the point estimate every time.

WHAT IT CAN DECIDE IS THE REVIEWER'S ACTUAL COMPLAINT
"One seed carries the whole result" is a question about the distribution. So the deliverables are the
realized sd against 0.1297 at n=5, the count of seed-42-like draws in 20, and the full per-seed list
printed rather than summarized. If the mode recurs the interaction is real and the published verdict is
vindicated; that is the outcome that does NOT flatter the cleaner story, which is why both are coded.

THE RESIDUAL IS ONLY EVER FORMED ON SEEDS SHARED BY ALL THREE CELLS
Prereg: "It is never assembled from cells with different seed sets." The shared n is computed and
printed. The `full` cell comes from item A, so a partial item A shrinks the residual's n, not its
validity.

NO SEED IS DROPPED, AND THE ACCURACY FLOOR DOES NOT GET A SECOND SUBSTITUTION
b995f1b already spent its one substitution moving the primary contrast from kappa=2 to kappa=1. If the
emit-only kappa=1 rung mean falls below 0.35 at the larger n, the contrast is reported UNINTERPRETABLE
rather than moved to another rung, because a second move would be rung shopping.

Reads results/emit_only/, results/score_only/, results/targeted_dose/, results/dose_seed_topup/ and
results/emit_only_topup/. Writes nothing. Adds no runs.

Run: python3 experiments/analyze_emit_only_topup.py
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imported, not reimplemented: the published-reproduction check must run through the SAME code path
# that emitted the published secondaries, or it is not a reproduction check.
from experiments.analyze_emit_only_factorial import deltas, load_cells, per_seed  # noqa: E402
from experiments.analyze_headline_cis import t_crit                              # noqa: E402
from experiments.analyze_tost_existing import tost                               # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PREREG = "experiments/pre_registration_emit_only_topup.md"
PREREG_COMMIT = "684b31e"
DEPARTURE = ("pre_registration_emit_only.md at b995f1b: 'No seed addition. n=5, seeds 42-46, "
             "matching the arm it is compared against.'")

# Carried forward unchanged from b995f1b. Nothing here is a new threshold.
INERT_MARGIN = 0.05
RESID_MARGIN = 0.05
ACC_FLOOR = 0.35
CONTRAST_RUNG = 1.0          # forced by the floor contingency at b995f1b, before this document existed
OUTLIER_THRESHOLD = 0.15     # the descriptive count the review actually asked for

PUBLISHED_SEEDS = [42, 43, 44, 45, 46]
NEW_SEEDS = list(range(47, 62))
N_PLANNED = 20

# Base key (published dirs) and the top-up's suffixed keys.
KEY = "doseS_kappa{k}_then_krum|committed_scaling"
TOPUP_KEY = "doseS_kappa{k}_then_krum|committed_scaling|{tag}"

# Every file that can supply one cell of the factorial. `full` at the new seeds is item A's output.
SOURCES = {
    "full":       [("targeted_dose", KEY), ("dose_seed_topup", KEY)],
    "score_only": [("score_only", KEY), ("emit_only_topup", TOPUP_KEY)],
    "emit_only":  [("emit_only", KEY), ("emit_only_topup", TOPUP_KEY)],
}
# At kappa=0 all three cells are ONE computation (the transform returns the update list unwrapped and
# the score_only/emit_only split has nothing to split), which is why they are equal seed-for-seed today.
BASELINE_SOURCES = [("emit_only", KEY), ("score_only", KEY), ("targeted_dose", KEY),
                    ("dose_seed_topup", KEY)]

# The published n=5 table, transcribed from the pre-registration so the merge is checked against the
# document and not against itself. Asserted, never printed as a result.
PUB_MEAN = {"full": -0.038867, "score_only": -0.029978, "emit_only": +0.045578,
            "residual": -0.054467}
PUB_SD = {"full": 0.0455, "score_only": 0.0638, "emit_only": 0.1297, "residual": 0.1222}
PUB_TOL = 5e-4
EXACT_TOL = 1e-9             # the tolerance the frozen factorial check already uses

# Projected at the freeze, from the n=5 sds. Printed for comparison with what is realized.
PROJECTED_HW = {"emit_only": 0.061, "residual": 0.057}


def prereg_frozen():
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%h", "--", PREREG],
                             cwd=BASE, capture_output=True, text=True, timeout=20)
        h = out.stdout.strip() or None
    except Exception:
        h = None
    return (h is not None and h.startswith(PREREG_COMMIT[:7])), h


def cells_of(dirname):
    p = os.path.join(BASE, "results", dirname, "summary.json")
    if not os.path.exists(p):
        return None
    return load_cells(p)          # the frozen factorial's loader; it takes a PATH, not a dict


def rows(sources, kappa, tag):
    """{seed: (asr, acc, dirname)} merged across sources; the FIRST source to carry a seed wins."""
    out = {}
    for dirname, keyfmt in sources:
        c = cells_of(dirname)
        if not c:
            continue
        key = keyfmt.format(k=kappa, tag=tag)
        cell = c.get(key)
        if not cell:
            continue
        for r in cell["per_seed"]:
            out.setdefault(int(r["seed"]),
                           (float(r["asr"]), float(r["accuracy"]), dirname))
    return out


def check_baseline_agrees():
    """All three cells' kappa=0 rung is one computation. Verified, not cited."""
    seen = {}
    for dirname, keyfmt in BASELINE_SOURCES:
        c = cells_of(dirname)
        cell = c.get(keyfmt.format(k=0.0, tag="")) if c else None
        if not cell:
            continue
        for r in cell["per_seed"]:
            seen.setdefault(int(r["seed"]), []).append((dirname, float(r["asr"])))
    problems = []
    for s, vals in sorted(seen.items()):
        for dirname, v in vals[1:]:
            if abs(v - vals[0][1]) > EXACT_TOL:
                problems.append(f"kappa=0 seed {s}: {vals[0][0]} {vals[0][1]!r} vs {dirname} {v!r}")
    return problems, {s: vals[0][1] for s, vals in seen.items()}, sorted(seen)


def check_published_reproduce():
    """Non-negotiable 3, run through the frozen factorial's own code path before anything is pooled."""
    problems = []
    eo, full, so = (cells_of(d) for d in ("emit_only", "targeted_dose", "score_only"))
    if not all((eo, full, so)):
        return ["one of results/emit_only/, results/targeted_dose/, results/score_only/ is absent"], {}
    d = {}
    try:
        d["emit_only"] = deltas(eo, CONTRAST_RUNG, PUBLISHED_SEEDS)
        d["full"] = deltas(full, CONTRAST_RUNG, PUBLISHED_SEEDS)
        d["score_only"] = deltas(so, CONTRAST_RUNG, PUBLISHED_SEEDS)
    except (KeyError, TypeError) as e:
        return [f"published cells not readable through the frozen code path: {e!r}"], {}
    d["residual"] = d["full"] - (d["score_only"] + d["emit_only"])

    # Against the frozen artifact's own verdict block, at the tolerance it already uses.
    v = json.load(open(os.path.join(BASE, "results", "emit_only", "summary.json"))).get("verdict", {})
    for name, field in (("emit_only", "d_asr_emit_only"), ("full", "d_asr_full_mode_s"),
                        ("score_only", "d_asr_score_only"), ("residual", "additivity_residual")):
        want = v.get(field)
        if want is None:
            problems.append(f"{field}: absent from the frozen verdict block")
        elif abs(float(d[name].mean()) - float(want)) > EXACT_TOL:
            problems.append(f"{field}: recomputed {d[name].mean()!r} != frozen {want!r}")
    # And against the pre-registration's printed table, which is an independent transcription.
    for name in ("full", "score_only", "emit_only", "residual"):
        if abs(float(d[name].mean()) - PUB_MEAN[name]) > PUB_TOL:
            problems.append(f"{name}: n=5 mean {d[name].mean():.6f} != pre-registered {PUB_MEAN[name]}")
        if abs(float(d[name].std(ddof=1)) - PUB_SD[name]) > PUB_TOL:
            problems.append(f"{name}: n=5 sd {d[name].std(ddof=1):.4f} != pre-registered {PUB_SD[name]}")
    return problems, d


def interval(d, margin):
    r = tost(d, margin)
    r["hw95"] = t_crit(len(d)) * r["se"]
    r["contained"] = r["ci95"][0] > -margin and r["ci95"][1] < margin
    r["point_inside"] = abs(r["mean"]) < margin
    return r


def qualify(r, margin, label):
    """The frozen qualifier: a point-estimate pass with an interval wider than the margin is
    'consistent with, not established'. Printed at the site of the claim, every time."""
    if r["point_inside"] and r["contained"]:
        return f"{label} ESTABLISHED (interval contained in +-{margin})"
    if r["point_inside"]:
        return (f"consistent with {label}, NOT established "
                f"(half-width {r['hw95']:.4f} exceeds the +-{margin} margin)")
    return f"{label} does NOT hold at this n (point estimate outside the margin)"


def main():
    ok, actual = prereg_frozen()
    state = f"committed at {actual}" if actual else "UNTRACKED -- not a pre-registration"
    print(f"[{'OK' if ok else 'MISMATCH'}] rules frozen in {PREREG}: recorded {PREREG_COMMIT}, {state}")
    if not ok:
        print("  REFUSING TO SCORE: the rules being scored are not demonstrably prospective.")
        return 1
    print(f"[!] DISCLOSED DEPARTURE from {DEPARTURE}")
    print("    Rationale honoured (all three cells, same seeds, same rung); literal text departed from.")
    print("    Nothing below is prospective, and the published n=5 verdict stands as reported.")

    problems, _published = check_published_reproduce()   # the deltas themselves are recomputed below
    if problems:
        print("\nREFUSING TO PRINT ANY POOLED NUMBER -- the published seeds do not reproduce:")
        for p in problems:
            print(f"  {p}")
        return 1
    print(f"[OK] all four published quantities reproduce from per-seed cells at <{EXACT_TOL}, and match")
    print("     the pre-registration's printed table independently.")

    base_problems, baseline, base_seeds = check_baseline_agrees()
    if base_problems:
        print(f"\nKAPPA=0 BASELINE DIVERGES ACROSS CELLS -- refusing to score (tolerance {EXACT_TOL}):")
        for p in base_problems:
            print(f"  {p}")
        print("  The three cells share one identity computation; if they do not, the residual has no")
        print("  common baseline and the factorial is not a factorial.")
        return 1
    print(f"[OK] kappa=0 baseline: {len(base_seeds)} seeds agree across every cell that carries them")

    print(f"\n=== 2x2 FACTORIAL AT THE TOPPED-UP n: krum / committed_scaling, "
          f"kappa=0 -> kappa={CONTRAST_RUNG:g} (item C) ===")
    print(f"Contrast rung forced by the b995f1b accuracy floor, before this top-up existed. "
          f"Margins {INERT_MARGIN}/{RESID_MARGIN}, floor {ACC_FLOOR}.\n")

    # Per-cell state at the n actually reached.
    cellrows, accmean = {}, {}
    print(f"  {'cell':12s} {'n':>3s} {'mean ASR':>9s} {'mean acc':>9s}   provenance")
    for tag in ("full", "score_only", "emit_only"):
        r = rows(SOURCES[tag], CONTRAST_RUNG, tag)
        cellrows[tag] = r
        if not r:
            print(f"  {tag:12s} {0:3d} {'--':>9s} {'--':>9s}   (no runs yet)")
            continue
        seeds = sorted(r)
        asr = np.array([r[s][0] for s in seeds])
        acc = np.array([r[s][1] for s in seeds])
        accmean[tag] = float(acc.mean())
        src = {}
        for s in seeds:
            src[r[s][2]] = src.get(r[s][2], 0) + 1
        flag = ""
        if acc.mean() < ACC_FLOOR:
            flag = f"   ** RUNG MEAN BELOW {ACC_FLOOR} FLOOR"
        print(f"  {tag:12s} {len(seeds):3d} {asr.mean():9.4f} {acc.mean():9.4f}   "
              + ", ".join(f"{k}:{n}" for k, n in sorted(src.items())) + flag)

    # The residual only ever forms on seeds shared by ALL THREE cells, plus the baseline.
    shared = sorted(set(baseline) & set(cellrows.get("full", {}))
                    & set(cellrows.get("score_only", {})) & set(cellrows.get("emit_only", {})))
    print(f"\n  seeds shared by all three cells AND the kappa=0 baseline: n = {len(shared)} "
          f"of a planned {N_PLANNED}")
    if len(shared) < 2:
        print("    fewer than 2 shared seeds: no interval is formed and no verdict is scored.")
        print("    (item A supplies the `full` cell at seeds 47-61; a partial item A shrinks this n.)")
        return 0
    print(f"    {shared}")

    dd = {tag: np.array([cellrows[tag][s][0] - baseline[s] for s in shared])
          for tag in ("full", "score_only", "emit_only")}
    dd["residual"] = dd["full"] - (dd["score_only"] + dd["emit_only"])
    n = len(shared)
    provisional = n < N_PLANNED

    # The accuracy gate, applied per cell, with NO second substitution.
    eo_acc_shared = float(np.mean([cellrows["emit_only"][s][1] for s in shared]))
    uninterpretable = eo_acc_shared < ACC_FLOOR
    if uninterpretable:
        print(f"\n  ** ACCURACY GATE FAILED: the emit-only kappa={CONTRAST_RUNG:g} rung mean accuracy "
              f"is {eo_acc_shared:.4f} < {ACC_FLOOR}")
        print("     over the shared seeds. Per the frozen rule the contrast is UNINTERPRETABLE and is")
        print("     NOT substituted onto another rung: b995f1b already spent its one substitution")
        print("     moving the primary from kappa=2 to kappa=1, and a second move would be rung")
        print("     shopping. The numbers below are printed, and none of them is a verdict.")

    print(f"\n  {'quantity':12s} {'mean':>10s} {'sd':>8s} {'95% CI':>22s} {'half-width':>11s} "
          f"{'n=5 sd':>8s}")
    res = {}
    for tag in ("full", "score_only", "emit_only", "residual"):
        margin = INERT_MARGIN if tag == "emit_only" else RESID_MARGIN
        r = interval(dd[tag], margin)
        res[tag] = r
        print(f"  {tag:12s} {r['mean']:>+10.6f} {r['sd']:>8.4f} "
              f"{'[%+.4f, %+.4f]' % r['ci95']:>22s} {r['hw95']:>11.4f} {PUB_SD[tag]:>8.4f}")

    # The two frozen rules, each with its qualifier at the site of the claim.
    print(f"\n  FROZEN RULE 1: magnitude channel inert, |dASR_EO| < {INERT_MARGIN}")
    print(f"    |{res['emit_only']['mean']:+.6f}| -> "
          f"{qualify(res['emit_only'], INERT_MARGIN, 'INERTNESS')}")
    print(f"  FROZEN RULE 2: channels separable, |residual| < {RESID_MARGIN}")
    print(f"    |{res['residual']['mean']:+.6f}| -> "
          f"{qualify(res['residual'], RESID_MARGIN, 'SEPARABILITY')}")

    inert = res["emit_only"]["point_inside"]
    separable = res["residual"]["point_inside"]
    stamp = (f"PROVISIONAL at n = {n}" if provisional else f"at the frozen n = {N_PLANNED}")

    print(f"\n  VERDICT ({stamp})")
    if uninterpretable:
        print("    UNINTERPRETABLE: the accuracy gate failed, so no verdict is scored at this n.")
    elif separable and inert:
        # The reversal clause, in the direction that flatters the cleaner story -- which is exactly
        # why it is stated as a failure of our own published verdict rather than as a new finding.
        print("    ** THE PUBLISHED 'CHANNELS INTERACT' VERDICT DID NOT SURVIVE OUR OWN TOP-UP.")
        print(f"       The residual is {res['residual']['mean']:+.6f}, inside "
              f"(-{RESID_MARGIN}, +{RESID_MARGIN}).")
        print("       Reported in the section where the verdict appears, with the n=5 numbers retained")
        print("       beside it, and only as 'consistent with separability' unless the interval above")
        print("       is contained in the margin -- which was projected in advance to be impossible at")
        print(f"       this variance and n (projected half-widths {PROJECTED_HW}).")
    elif not separable and inert:
        further = abs(res["residual"]["mean"]) > abs(PUB_MEAN["residual"])
        print(f"    CHANNELS INTERACT, CONFIRMED ({stamp}): residual {res['residual']['mean']:+.6f} is")
        print(f"       outside +-{RESID_MARGIN} while |dASR_EO| = {abs(res['emit_only']['mean']):.6f} "
              f"< {INERT_MARGIN}.")
        scale = (f"at {n // 5}x the published sample size" if n >= 10
                 else f"at n = {n}, the published sample size" if n == 5
                 else f"at n = {n}")
        print(f"       This reproduces the published verdict {scale}, which is the strongest thing")
        print("       this run can deliver precisely because it is NOT the cleaner story.")
        if further:
            print(f"       The residual moved FURTHER outside the margin than the published "
                  f"{PUB_MEAN['residual']:+.6f}; reported as such.")
    else:
        print(f"    NEITHER FROZEN RULE HOLDS ({stamp}): |dASR_EO| = "
              f"{abs(res['emit_only']['mean']):.6f} is not")
        print(f"       below {INERT_MARGIN}, so the magnitude channel is not inert at this n and the")
        print("       residual cannot be read as a statement about separability alone. Reported as")
        print("       such, and not scored in our favour.")
    if provisional:
        print(f"    Non-negotiable 2 fixes seeds 47-61: this is the number at n = {n}, not the")
        print(f"       n = {N_PLANNED} result, and no seed is dropped or filtered.")

    # THE ACTUAL DELIVERABLE: the distribution, which is what "one seed carries it" is a claim about.
    print(f"\n  DISTRIBUTION OF dASR_EO -- the deliverable, not the margin")
    print(f"    realized sd {res['emit_only']['sd']:.4f} at n={n}, against "
          f"{PUB_SD['emit_only']:.4f} at n=5")
    big = [s for s, v in zip(shared, dd["emit_only"]) if v > OUTLIER_THRESHOLD]
    small = [s for s, v in zip(shared, dd["residual"]) if v < -OUTLIER_THRESHOLD]
    print(f"    seeds with dASR_EO > +{OUTLIER_THRESHOLD}: {len(big)} of {n}  {big}")
    print(f"    seeds with residual < -{OUTLIER_THRESHOLD}: {len(small)} of {n}  {small}")
    if 42 in shared:
        i42 = shared.index(42)
        print(f"    seed 42: dASR_EO {dd['emit_only'][i42]:+.4f}, residual {dd['residual'][i42]:+.4f} "
              f"(the published knife edge)")
    if len(big) <= 1:
        print("    If this is the whole of it, seed 42 was an outlier and the published verdict was")
        print("    driven by one draw. That reading needs the full n; it is not asserted here.")
    else:
        print("    The mode RECURS: the emit-only channel is genuinely bimodal across data partitions")
        print("    and the interaction is real. This VINDICATES the published verdict.")

    print(f"\n  PER-SEED, printed rather than summarized (no seed is excluded from anything)")
    print(f"    {'seed':>6} {'dASR_EO':>11} {'dASR_SO':>11} {'dASR_full':>11} {'residual':>11} "
          f"{'EO acc':>8}")
    for i, s in enumerate(shared):
        acc = cellrows["emit_only"][s][1]
        print(f"    {s:>6} {dd['emit_only'][i]:>+11.6f} {dd['score_only'][i]:>+11.6f} "
              f"{dd['full'][i]:>+11.6f} {dd['residual'][i]:>+11.6f} {acc:>8.4f}"
              + ("  * below floor (flagged, not excluded)" if acc < ACC_FLOOR else ""))

    print(f"\n  CAVEATS, recorded in advance")
    print(f"    Projected 95% half-widths at n=20 were {PROJECTED_HW['emit_only']} (dASR_EO) and "
          f"{PROJECTED_HW['residual']} (residual),")
    print(f"      both wider than the {INERT_MARGIN} margin. The top-up was authorized knowing it")
    print("      cannot certify inertness or separability by containment.")
    print("    The LADDERS and the Jonckheere-Terpstra trend test stay at n=5 and are NOT restated")
    print("      here; every reported number carries its own n and no row pools the two.")
    print(f"    n={n} is {n} data partitions of one dataset, one aggregator, one attack, one synthetic")
    print("      instrument. Nothing here widens the factorial's scope.")
    print("    Seed 42 is not excluded from anything. The published post-hoc exclusion (residual")
    print("      -0.003444 without it) stays labelled a statement about power, not a verdict.")

    print(f"\n  PUBLISHED n=5 VERDICT, reported alongside as non-negotiable 5 requires:")
    print(f"    CHANNELS INTERACT -- dASR_EO {PUB_MEAN['emit_only']:+.6f}, residual "
          f"{PUB_MEAN['residual']:+.6f}, missing the")
    print(f"    {RESID_MARGIN} margin by {abs(PUB_MEAN['residual']) - RESID_MARGIN:.4f}. "
          f"Seed 42 contributed +0.2721 of the mean dASR_EO.")

    print(f"\n--- LaTeX (kappa=0 -> kappa={CONTRAST_RUNG:g}, n={n}) ---")
    for tag in ("emit_only", "residual"):
        margin = INERT_MARGIN if tag == "emit_only" else RESID_MARGIN
        r = res[tag]
        note = ("established" if r["point_inside"] and r["contained"]
                else "consistent, not est." if r["point_inside"] else "not held")
        print(f"{tag.replace('_', '-')} & $n{{=}}{n}$ & ${r['mean']:+.3f}$ & "
              f"$[{r['ci95'][0]:+.3f},\\,{r['ci95'][1]:+.3f}]$ & ${r['sd']:.3f}$ & {note} \\\\")
    return 0


if __name__ == "__main__":
    sys.exit(main())
