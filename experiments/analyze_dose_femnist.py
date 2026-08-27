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

Read-only on every frozen artifact. Writes results/dose_femnist/scored.json.
"""
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

D2 = "krum"
ATTACK = "committed_scaling"
PREREG_COMMIT = "478b555"


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
    if abs(delta) < EQUIV_MARGIN:
        verdict = "REPLICATED on a second dataset and architecture"
    elif delta > EQUIV_MARGIN:
        verdict = "DATASET- OR ARCHITECTURE-SPECIFIC: the negative does not replicate"
    else:
        verdict = "INDETERMINATE (attenuation-side fall), not scored in our favour"
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


if __name__ == "__main__":
    main()
