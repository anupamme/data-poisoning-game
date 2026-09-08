"""Score the FEMNIST second-dataset replication of the flagship Mode-S negative.

Scores exactly the rules frozen in experiments/pre_registration_dose_femnist.md (committed 478b555)
and nothing else:

  primary    Delta = mean ASR(kappa=2) - mean ASR(kappa=0), n=3, against EQUIV_MARGIN = 0.15
             |Delta| < 0.15  -> replicated on a second dataset and architecture
             Delta > +0.15   -> the negative is dataset- or architecture-specific
             Delta < -0.15   -> attenuation-side fall, INDETERMINATE, not scored in our favour
  secondary  Jonckheere-Terpstra across all four rungs, specified to detect a monotone RISE
  gate       every rung needs mean clean accuracy >= ACC_FLOOR = 0.35
  sanity     the kappa=0 rung is krum alone, so it should land near the payoff matrix's figure

Every constant, the rung builder, the interval and the trend test are imported from the frozen
suite's own scorers rather than re-derived here, so this arm and the CIFAR-10 arm compute the same
quantities by construction. The CIFAR-10 comparison Delta and the standalone payoff figure are
recomputed from their artifacts per seed at run time; neither is transcribed.

--pooled scores the SAME rules at n=5, and the two extra seeds were not run for it. Seeds 45 and 46
of this exact ladder (same d2, attack, dataset, model and four rungs) already exist in
results/comparability_cells/, run under the comparability pre-registration's Amendment 2 (1026a96,
read with Amendment 3 f16083b) because the six-cell scoring would otherwise have compared n=5
against a frozen n=3 on non-identical seed sets. So the n=5 reading of this arm is already in the
paper, in tab:sixcell's controlled column and in the supplement's margin-sensitivity m*: what was
missing is the arm's OWN section saying so. Rows are merged by seed with the FROZEN DIRECTORY
WINNING, results/dose_femnist/ is never written, and every pooled-in row is tagged with the
directory it came from. results/dose_femnist_topup/ (an independent recomputation of seeds 45--46
through this suite's own runner, amendment ad5479b) is pooled in only where a seed is absent from
both, i.e. never for 45--46; it is instead printed as a CROSS-HARNESS comparison, since two runners
agreeing to the digit on the same seed is a harness identity check and two runners disagreeing is a
finding about the suites. The confounded ladder is excluded by family, so it can never pool in.

Read-only on every frozen artifact. Writes results/dose_femnist/scored.json, or
results/dose_femnist_pooled_scored.json under --pooled, which is outside the frozen directory.
"""
import argparse
import copy
import json
import os
import sys

import numpy as np

try:
    from scipy import stats as sps
except Exception:
    sps = None

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base)
# single-sourced: same rung builder, same interval, same gate, same trend test as the frozen suite
from experiments.analyze_targeted_dose import (  # noqa: E402
    ACC_FLOOR, EQUIV_MARGIN, KAPPAS, rungs_of, show,
)
from experiments.analyze_dose_response import jonckheere  # noqa: E402
# the payoff-matrix figure and the prospectively measured premise are read by the runner's own
# readers, so the analyzer and the run cannot disagree about either
from experiments.run_dose_femnist import payoff_standalone, premise  # noqa: E402

FEMNIST = os.path.join(base, "results", "dose_femnist", "summary.json")
TARGETED = os.path.join(base, "results", "targeted_dose", "summary.json")
PAYOFF = os.path.join(base, "results", "femnist", "payoff_results.json")
ADM_FEMNIST = os.path.join(base, "results", "femnist_admission.json")
OUT = os.path.join(base, "results", "dose_femnist", "scored.json")

# --pooled sources, in the order they are consulted. The frozen directory is always consulted first
# and always wins, so neither of these can displace a published per-seed value.
COMPARABILITY = os.path.join(base, "results", "comparability_cells", "summary.json")
TOPUP = os.path.join(base, "results", "dose_femnist_topup", "summary.json")
POOLED_OUT = os.path.join(base, "results", "dose_femnist_pooled_scored.json")

D2 = "krum"
ATTACK = "committed_scaling"
PREREG_COMMIT = "478b555"
DATASET_KEY = "femnist"          # the JSON key; the arm is EMNIST-byclass in all prose
POOLED_N = 5


def verdict_of(delta):
    """The frozen primary rule. One definition, so the frozen and pooled readings cannot drift."""
    if abs(delta) < EQUIV_MARGIN:
        return "REPLICATED on a second dataset and architecture"
    if delta > EQUIV_MARGIN:
        return "DATASET- OR ARCHITECTURE-SPECIFIC: the negative does not replicate"
    return "INDETERMINATE (attenuation-side fall), not scored in our favour"


def arm_rows(path, container):
    """Per-kappa per-seed rows for THIS arm out of another suite's artifact, tagged with its source.

    Selection is on the arm's identity and not on the key spelling: same d2, same committed attack,
    same dataset, and Mode S, which this suite spells `mode: S` and the comparability suite spells
    `family: controlled`. The confounded ladder shares (d2, attack, dataset) and differs only in
    that field, so dropping the test would silently pool the very thing the instrument removes.
    """
    if not os.path.exists(path):
        return {}, None
    doc = json.load(open(path))
    src = os.path.basename(os.path.dirname(path))
    out = {}
    for c in list(doc.get("cells", {}).values()) + list(doc.get("new_cells", {}).values()):
        if c.get("d2") != D2 or c.get("attack") != ATTACK:
            continue
        if c.get("dataset") != DATASET_KEY:
            continue
        if c.get("mode") != "S" and c.get("family") != "controlled":
            continue
        kap = float(c["rung"] if "rung" in c else c["kappa"])
        for r in c["per_seed"]:
            out.setdefault(kap, []).append(dict(r, source=src))
    return out, doc.get("prereg_commit") or doc.get("prereg_commit_topup")


def pooled_cells(frozen, sources):
    """Frozen rows plus rows from `sources` in order, deduplicated by seed. FROZEN WINS, always.

    Returns (cells, added, collisions): `added` names every pooled-in seed and where it came from,
    `collisions` every seed a later source also holds, which is what makes the cross-harness
    comparison possible instead of silently dropping the second value.
    """
    cells = copy.deepcopy(frozen)
    added, collisions = {}, {}
    for extra in sources:
        for c in cells.values():
            if c.get("mode") != "S" or c.get("d2") != D2 or c.get("attack") != ATTACK:
                continue
            kap = float(c["rung"])
            have = {r["seed"]: r for r in c["per_seed"]}
            for r in extra.get(kap, []):
                if r["seed"] in have:
                    collisions.setdefault(kap, []).append((r["seed"], have[r["seed"]], r))
                    continue
                c["per_seed"].append(r)
                have[r["seed"]] = r
                added.setdefault(kap, []).append((r["seed"], r["source"]))
            c["per_seed"].sort(key=lambda x: x["seed"])
    return cells, added, collisions


def cifar10_delta():
    """The published CIFAR-10 Mode-S krum rise, recomputed from per-seed rows. Never transcribed."""
    if not os.path.exists(TARGETED):
        return None
    r = rungs_of(json.load(open(TARGETED))["cells"], "S", D2, KAPPAS)
    return None if r is None else float(r[-1]["mean"] - r[0]["mean"])


def main():
    if not os.path.exists(FEMNIST):
        sys.exit("results/dose_femnist/summary.json missing -- run experiments/run_dose_femnist.py")
    doc = json.load(open(FEMNIST))
    if doc.get("prereg_commit") != PREREG_COMMIT:
        sys.exit(f"prereg commit mismatch: {doc.get('prereg_commit')} != {PREREG_COMMIT}")
    rungs = rungs_of(doc["cells"], "S", D2, KAPPAS)
    if rungs is None:
        sys.exit("incomplete ladder: not every frozen rung is present")

    print(f"=== FEMNIST MODE-S REPLICATION: {D2}/{ATTACK}, rules frozen @ {PREREG_COMMIT} ===")
    print(f"    {doc['dataset']}/{doc['model']}, seeds {rungs[0]['seeds']}, "
          f"n={[len(r['asrs']) for r in rungs]} per rung")
    show(f"doseS -> {D2}", rungs)

    ns = [len(r["asrs"]) for r in rungs]
    if any(n != 3 for n in ns):
        print(f"\n  !! PARTIAL SEEDS {ns}: the frozen rule is stated at n=3. No verdict stands.")
        sys.exit(1)

    # --- accuracy gate, before any verdict ---
    print("\n=== ACCURACY GATE (every rung needs mean clean accuracy >= "
          f"{ACC_FLOOR}) ===")
    for r in rungs:
        print(f"  kappa={r['rung']:<4} acc {r['acc']:.3f}  "
              f"{'FAIL -- uninterpretable' if r['gated'] else 'pass'}")
    gated = [r for r in rungs if r["gated"]]
    if gated:
        print("\n  NO VERDICT STANDS: a low ASR at collapsed accuracy is not suppression.")
        sys.exit(1)
    print("  all rungs pass, with margin; the gate does not bind on this cell")

    # --- primary ---
    delta = float(rungs[-1]["mean"] - rungs[0]["mean"])
    c10 = cifar10_delta()
    verdict = verdict_of(delta)
    print(f"\n=== PRIMARY (frozen) ===")
    print(f"  Delta = {delta:+.3f}  ({rungs[0]['mean']:.3f} -> {rungs[-1]['mean']:.3f}), "
          f"margin {EQUIV_MARGIN}")
    print(f"  CIFAR-10 Mode-S {D2} Delta = {c10:+.3f} (recomputed from results/targeted_dose)"
          if c10 is not None else "  CIFAR-10 comparison unavailable")
    print(f"  VERDICT: {verdict}")
    print("  wording is fixed by the freeze: 'replicated on a second dataset and architecture',"
          " never 'generalizes'")

    # --- the headroom asymmetry, stated with the verdict rather than after it ---
    identity = float(rungs[0]["mean"])
    print(f"\n=== HEADROOM OF THE TWO BRANCHES (not a test; a disclosure) ===")
    print(f"  identity rung ASR is {identity:.3f}, so a fall cannot exceed {identity:.3f} in"
          f" magnitude and the")
    print(f"  'Delta < -{EQUIV_MARGIN}' indeterminate branch was unreachable on this cell.")
    print(f"  The branch that would REFUTE us was fully reachable: it needed mean ASR(kappa=2)"
          f" > {identity + EQUIV_MARGIN:.3f},")
    print(f"  against an observed {rungs[-1]['mean']:.3f} and a ceiling of 1.0. Equivalence here is"
          " to the identity")
    print("  rung, not to a low absolute ASR.")

    # --- secondary ---
    J, z, p_inc, p_dec, p_perm = jonckheere([r["asrs"] for r in rungs])
    print(f"\n=== SECONDARY: Jonckheere-Terpstra over four rungs (frozen) ===")
    print(f"  J={J:.1f}  z={z:+.3f}  p(increasing)={p_inc:.3f}  p(decreasing)={p_dec:.3f}  "
          f"p_perm(increasing)={p_perm:.4f}")
    print("  The frozen rule specifies this test to detect a monotone RISE, which would have refuted")
    print("  the flat prediction with the attenuation channel already closed by construction.")
    if z > 0:
        print(f"  Observed trend is upward (z={z:+.3f}); p(increasing)={p_inc:.3f}.")
    else:
        print(f"  Observed trend is DOWNWARD (z={z:+.3f}), which is not the direction this test was")
        print("  specified to detect. It is reported, and it is NOT scored as support for flatness.")
    print(f"  Pre-registered caveat, quoted: at n=3 the JT test is weak; a null from it is not")
    print("  evidence of flatness and is not reported as such.")

    # --- sanity, explicitly not a hypothesis test ---
    p_acc, p_asr = payoff_standalone()
    print(f"\n=== HARNESS SANITY (not a hypothesis test) ===")
    if p_asr is None or p_asr != p_asr:
        print("  payoff matrix figure unavailable")
    else:
        print(f"  kappa=0 rung (krum alone) {rungs[0]['mean']:.4f} @ {rungs[0]['acc']:.3f}   "
              f"payoff matrix {p_asr:.4f} @ {p_acc:.3f}")
        gap = abs(rungs[0]["mean"] - p_asr)
        print(f"  gap {gap:.10f} ASR, {abs(rungs[0]['acc'] - p_acc):.10f} accuracy")
        if gap < 1e-9:
            print("  BIT-IDENTICAL. run_femnist.py ran 3 trials from base seed 42 and this rung runs")
            print("  seeds 42/43/44, doseS_kappa0.0 returns the stack unwrapped so the rung IS krum")
            print("  alone, and the pipeline is deterministic given a seed. This is the strongest")
            print("  form of the pre-registered check -- the two harnesses run the same protocol, not")
            print("  merely a similar one. It is NOT independent corroboration of the figure, and is")
            print("  not reported as such: the same computation twice cannot corroborate itself.")
        elif gap < 0.05:
            print("  consistent: seed noise against a 3-trial mean, the harnesses agree")
        else:
            print("  TENS OF POINTS: the harnesses disagree and the arm is VOID rather than"
                  " interesting")

    # --- the prospectively measured premise, restated as measured ---
    prem = premise()
    if prem:
        print("\n=== PREMISE, MEASURED BEFORE THIS RULE WAS WRITTEN ===")
        print("  (results/femnist_admission.json, computed with no ASR anywhere and no model")
        print("   trained per rung; the arm would have been reported ineligible had the decision")
        print("   change come out near zero, which is the cos_krum failure mode)")
        print("  kappa   decision change   admission change")
        for k in KAPPAS:
            r = prem.get(str(k), {})
            d_, a_ = r.get("decision"), r.get("admission")
            if d_ is not None:
                print(f"  {k:<6}  {d_:<16.3f}  {a_:.3f}")

    json.dump(dict(prereg_commit=PREREG_COMMIT, dataset=doc["dataset"], model=doc["model"],
                   d2=D2, attack=ATTACK, n_per_rung=ns,
                   rungs=[dict(kappa=r["rung"], asrs=r["asrs"], accs=r["accs"],
                               mean_asr=r["mean"], lo=r["lo"], hi=r["hi"], mean_acc=r["acc"],
                               gated=r["gated"]) for r in rungs],
                   delta=delta, equiv_margin=EQUIV_MARGIN, verdict=verdict,
                   cifar10_delta=c10, acc_floor=ACC_FLOOR,
                   jt=dict(J=float(J), z=z, p_increasing=p_inc, p_decreasing=p_dec,
                           p_perm_increasing=p_perm),
                   identity_asr=identity,
                   refuting_threshold=identity + EQUIV_MARGIN,
                   payoff_standalone=dict(asr=p_asr, accuracy=p_acc),
                   prospective_premise=prem),
              open(OUT, "w"), indent=1)
    print(f"\nWrote {OUT}")


def pooled_main():
    """The same frozen rules at n=5, on seeds that already exist. Adds no run and rewrites nothing."""
    if not os.path.exists(FEMNIST):
        sys.exit("results/dose_femnist/summary.json missing -- nothing to pool onto")
    doc = json.load(open(FEMNIST))
    if doc.get("prereg_commit") != PREREG_COMMIT:
        sys.exit(f"prereg commit mismatch: {doc.get('prereg_commit')} != {PREREG_COMMIT}")

    comp, comp_commit = arm_rows(COMPARABILITY, "cells")
    top, top_commit = arm_rows(TOPUP, "new_cells")
    cells, added, collisions = pooled_cells(doc["cells"], [comp, top])

    frozen_rungs = rungs_of(doc["cells"], "S", D2, KAPPAS)
    rungs = rungs_of(cells, "S", D2, KAPPAS)
    if frozen_rungs is None or rungs is None:
        sys.exit("incomplete ladder: not every frozen rung is present")

    print(f"=== EMNIST-BYCLASS MODE-S REPLICATION AT n={POOLED_N}: {D2}/{ATTACK} ===")
    print(f"    primary rule, margin and gate unchanged, from {PREREG_COMMIT}; only the seed count moves")
    print(f"    frozen  results/dose_femnist/          seeds {frozen_rungs[0]['seeds']} @ {PREREG_COMMIT}")
    print(f"    pooled  results/comparability_cells/   @ {comp_commit} (amendment 2, read with f16083b)")
    if top:
        print(f"    also on disk: results/dose_femnist_topup/ @ {top_commit}, "
              "an independent recomputation -- compared below, never pooled over a frozen row")
    for k in sorted(added):
        print(f"      kappa={k}: added " + ", ".join(f"seed {s} from {d}" for s, d in added[k]))
    if not added:
        print("      NOTHING POOLED IN: every seed already sits in the frozen directory.")
    show(f"doseS -> {D2} (n={POOLED_N})", rungs)

    ns = [len(r["asrs"]) for r in rungs]
    if any(n != POOLED_N for n in ns):
        print(f"\n  !! UNEQUAL OR PARTIAL RUNGS {ns}: an endpoint contrast at n={POOLED_N} against a "
              f"middle rung at n=3 is the defect the comparability amendment existed to avoid.")
        print("     No pooled verdict stands.")
        sys.exit(1)

    print(f"\n=== ACCURACY GATE (every rung needs mean clean accuracy >= {ACC_FLOOR}) ===")
    for r in rungs:
        print(f"  kappa={r['rung']:<4} acc {r['acc']:.3f}  "
              f"{'FAIL -- uninterpretable' if r['gated'] else 'pass'}")
    if [r for r in rungs if r["gated"]]:
        print(f"\n  NO VERDICT STANDS at n={POOLED_N}, and per the amendment the n=3 verdict is then")
        print("  reported as NOT SUPERSEDED rather than as confirmed.")
        sys.exit(1)
    print("  all rungs pass; the gate does not bind at either seed count")

    d3 = float(frozen_rungs[-1]["mean"] - frozen_rungs[0]["mean"])
    d5 = float(rungs[-1]["mean"] - rungs[0]["mean"])
    print("\n=== PRIMARY (frozen rule, both seed counts, as the amendment requires) ===")
    print(f"  n=3 (frozen) Delta = {d3:+.4f}  ({frozen_rungs[0]['mean']:.4f} -> {frozen_rungs[-1]['mean']:.4f})"
          f"  -> {verdict_of(d3)}")
    print(f"  n={POOLED_N}          Delta = {d5:+.4f}  ({rungs[0]['mean']:.4f} -> {rungs[-1]['mean']:.4f})"
          f"  -> {verdict_of(d5)}")
    c10 = cifar10_delta()
    if c10 is not None:
        print(f"  CIFAR-10 Mode-S {D2} Delta = {c10:+.3f} (recomputed from results/targeted_dose)")
    flipped = (abs(d3) < EQUIV_MARGIN) != (abs(d5) < EQUIV_MARGIN)
    if flipped:
        print("  *** THE LABEL FLIPPED between n=3 and n=5, and is reported as a flip in the same")
        print("      sentence as the count. The margin does not move.")
    else:
        print(f"  Label unchanged by two more seeds; the margin ({EQUIV_MARGIN}) does not move either way.")

    # Per-seed, because a mean that survives can still hide a seed that reverses.
    lo = dict(zip(rungs[0]["seeds"], rungs[0]["asrs"]))     # rungs_of sorts per_seed, so these align
    hi = dict(zip(rungs[-1]["seeds"], rungs[-1]["asrs"]))
    diffs = {s: hi[s] - lo[s] for s in sorted(set(lo) & set(hi))}
    print("\n=== PER-SEED PAIRED DIFFERENCES (kappa=2 minus kappa=0) ===")
    print("  " + "  ".join(f"s{s}={d:+.4f}" for s, d in diffs.items()))
    rev = [s for s, d in diffs.items() if (d > 0) != (d3 > 0)]
    if rev:
        print(f"  {len(rev)} of {len(diffs)} seeds carry the OPPOSITE sign to the frozen mean: {rev}.")
        print("  Disclosed with the count: the arm's mean is inside the margin, not every seed is on")
        print("  one side of zero, and a 5-seed mean is not evidence that it is.")

    J, z, p_inc, p_dec, p_perm = jonckheere([r["asrs"] for r in rungs])
    Jf, zf, pf_inc, pf_dec, pf_perm = jonckheere([r["asrs"] for r in frozen_rungs])
    print(f"\n=== SECONDARY: Jonckheere-Terpstra over four rungs (frozen rule, both counts) ===")
    print(f"  n=3  J={Jf:.1f}  z={zf:+.3f}  p(increasing)={pf_inc:.3f}  p_perm(inc)={pf_perm:.4f}")
    print(f"  n={POOLED_N}  J={J:.1f}  z={z:+.3f}  p(increasing)={p_inc:.3f}  p_perm(inc)={p_perm:.4f}")
    print("  Specified to detect a monotone RISE, the outcome that would have refuted flatness with")
    print("  attenuation closed by construction. A null from it is NOT scored as support for")
    print("  flatness at either seed count, and two more seeds do not change that.")

    if collisions:
        print("\n=== CROSS-HARNESS COMPARISON (not a test, and not corroboration) ===")
        print("  Seeds held by both a frozen/pooled source and a second suite. The two runners are")
        print("  distinct code paths (run_targeted_dose.run_one against run_comparability_cells'),")
        print("  so agreement to the digit shows they execute the same protocol; disagreement is a")
        print("  finding about the suites and not about this arm.")
        worst = 0.0
        for k in sorted(collisions):
            for s, a, b in collisions[k]:
                gap = abs(a["asr"] - b["asr"])
                worst = max(worst, gap)
                print(f"  kappa={k:<4} seed {s}: {a.get('source', 'frozen')} {a['asr']:.10f}  vs  "
                      f"{b['source']} {b['asr']:.10f}   gap {gap:.3e}")
        print(f"  largest gap {worst:.3e}: " + ("BIT-IDENTICAL protocols" if worst < 1e-9 else
              "the two harnesses DISAGREE and that is reportable in itself"))
    elif top:
        print("\n  (results/dose_femnist_topup/ holds no seed that overlaps a pooled row yet.)")

    out = dict(rules_from=PREREG_COMMIT, pooled_from=dict(comparability_cells=comp_commit,
                                                          dose_femnist_topup=top_commit),
               dataset=doc["dataset"], model=doc["model"], d2=D2, attack=ATTACK,
               frozen_wins=True, n_per_rung=ns,
               added={str(k): [dict(seed=s, source=d) for s, d in v] for k, v in added.items()},
               rungs=[dict(kappa=r["rung"], seeds=r["seeds"], asrs=r["asrs"], accs=r["accs"],
                           mean_asr=r["mean"], lo=r["lo"], hi=r["hi"], mean_acc=r["acc"],
                           gated=r["gated"], imported=r["imported"]) for r in rungs],
               delta_n3=d3, delta_n5=d5, equiv_margin=EQUIV_MARGIN,
               verdict_n3=verdict_of(d3), verdict_n5=verdict_of(d5), label_flipped=flipped,
               per_seed_paired_diff={str(s): d for s, d in diffs.items()},
               sign_reversing_seeds=rev,
               jt_n3=dict(J=float(Jf), z=zf, p_increasing=pf_inc, p_perm_increasing=pf_perm),
               jt_n5=dict(J=float(J), z=z, p_increasing=p_inc, p_perm_increasing=p_perm),
               cross_harness=[dict(kappa=k, seed=s, a=a["asr"], a_source=a.get("source", "frozen"),
                                   b=b["asr"], b_source=b["source"], gap=abs(a["asr"] - b["asr"]))
                              for k in sorted(collisions) for s, a, b in collisions[k]])
    json.dump(out, open(POOLED_OUT, "w"), indent=1)
    print(f"\nWrote {POOLED_OUT} (the frozen directory is not written)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooled", action="store_true",
                    help="score the same frozen rules at n=5 on seeds that already exist")
    if ap.parse_args().pooled:
        sys.exit(pooled_main())
    main()
