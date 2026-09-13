"""
Second-architecture eligibility check for the Mode-S replication arm: ResNet18, before any ASR exists.

WHY THIS RUNS FIRST. The planned arm is CIFAR-10 / resnet18 / krum / model_scaling under Mode S -- the
exact cell that produced the paper's flagship negative on cifar_cnn (decision changed in 73% of rounds
at kappa=2, admission in 0%, suppression unharmed). That arm is only informative if its PREMISE holds
on the new architecture: the Mode-S ladder must actually disturb Krum's decision on ResNet18, and must
leave the admitted adversarial mass alone. If the ladder does not move Krum's decision here, the arm
has no disturbance to test and a flat ASR curve would be uninformative rather than a replication --
which is exactly the cos_krum failure mode the paper already discloses. Establishing the premise BEFORE
the hypothesis is frozen is the difference between a prospective test and a post-hoc rationalization.

WHAT THIS ARM ISOLATES, AND WHY IT IS NOT THE FEMNIST ARM AGAIN. The existing replication
(experiments/measure_admission_femnist.py, experiments/run_dose_femnist.py) changed dataset AND
architecture at once: EMNIST-byclass with simple_cnn against CIFAR-10 with cifar_cnn. That is the
cheapest single test but it cannot say which of the two mattered, and simple_cnn and cifar_cnn are two
configurations of the same shallow-convnet family. Here the dataset, N, K, f, alpha, round count,
attack and defense are all held at the frozen CIFAR-10 values and ONLY the architecture moves, to a
different family: an 18-layer residual network with skip connections and GroupNorm. So a flat result
here is attributable to architecture alone, and the phrase "a second architecture" is earned by this
arm rather than by the FEMNIST one.

This script computes NO ASR and trains no model per rung: one raw update stack per (seed, round) is
shared by every transform, so all thirteen rungs cost the same ROUNDS_ELIG rounds of local training.

WHAT IS FROZEN BY THIS SCRIPT: nothing. It measures the channels and the instrument's own construction
assertions on ResNet18. The frozen artifact results/admission_measurement.json is NEVER read for
writing and never modified; this writes its own file.

THE INSTRUMENT'S CONSTRUCTION ASSERTIONS, re-checked on the new architecture (they are properties of
the transform, not of cifar_cnn, so they must hold exactly here too):
  - Mode S: adversarial coefficient share CONSTANT across rungs, c_adv == 1.0 exactly at every rung.
  - Mode A: adversarial coefficient share monotone in nu.
If either fails on ResNet18 the instrument is not doing on this architecture what it does on
cifar_cnn and the arm must not run.

WHICH resnet18. fl_core/models.py:73-79 has exactly one: torchvision resnet18 with a CIFAR stem
(conv1 3x3 stride 1, maxpool replaced by Identity) and every BatchNorm2d replaced by GroupNorm(8).
There is no second configuration to disambiguate, and the substitution is not cosmetic -- BatchNorm's
cross-sample statistics are not well defined under non-IID federated averaging, which is why the
repo's other resnet18 arms use the same builder.

Config matches the planned arm exactly: N=10, K=5, f=0.2, alpha=0.5, resnet18, seeds 42/43/44
(the same three seeds the FEMNIST replication froze, so the two replication arms are comparable).

Output: results/resnet18_admission.json
Run: python3 experiments/measure_admission_resnet18.py
"""
import json, math, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
# Single-sourced from the frozen CIFAR-10 measurement: the same measure(), the same rung list, the
# same decision/admission field names, the same rate(). Nothing about the channels is reimplemented,
# so the two architectures cannot be measured by two different definitions.
from experiments.measure_admission import (
    measure, rate, RUNGS, KAPPAS, NUS, ROUNDS, DECISION_KEY, ADMISSION_KEY, SHARE_TOL,
)

OUT = os.path.join(base, "results", "resnet18_admission.json")
FROZEN = os.path.join(base, "results", "admission_measurement.json")

DATASET, MODEL = "cifar10", "resnet18"
SEEDS3 = [42, 43, 44]              # the seeds the planned ASR arm will use
ROUNDS_ELIG = 6                    # more rounds than the frozen cifar_cnn measurement's 3, for the
                                   # same reason the FEMNIST check used 6: plain-fedavg advancement
                                   # can go non-finite under a scaling adversary, and the premise
                                   # needs headroom to rest on a comparable number of live rounds.
                                   # Whether ResNet18 overflows at this depth is measured below, not
                                   # assumed either way.
ARM, ATTACK = "krum", "committed_scaling"


def main():
    print("=== ResNet18 channel measurement (no ASR, no per-rung training) ===")
    print(f"    dataset={DATASET} model={MODEL}  seeds={SEEDS3} x {ROUNDS_ELIG} rounds x "
          f"{len(RUNGS)} rungs")
    print(f"    arm under consideration: mode S {ARM}/{ATTACK.replace('committed_', '')}")
    print("    architecture varies ALONE: dataset, N, K, f, alpha, attack and defense are the")
    print("    frozen CIFAR-10 values, so a result here is attributable to architecture.")
    print("    PROSPECTIVE: this runs before any hypothesis about ResNet18 ASR is frozen.\n",
          flush=True)

    # --reemit recomputes every printed verdict from the rows ALREADY in the artifact instead of
    # re-running the measurement. The rows are the measurement; the verdicts are functions of them, so
    # a verdict that had to be corrected can be corrected without spending an hour of compute and
    # without any opportunity to alter what was measured. It refuses to invent rows if none exist.
    if "--reemit" in sys.argv:
        if not os.path.exists(OUT):
            sys.exit(f"--reemit needs {OUT}, which does not exist. Run the measurement first.")
        all_rows = json.load(open(OUT))["per_round_including_nonfinite"]
        print(f"    --reemit: {len(all_rows)} stored rows re-read from {OUT}; nothing re-measured.\n",
              flush=True)
    else:
        all_rows = measure(ATTACK, dataset=DATASET, model=MODEL, seeds=SEEDS3, rounds=ROUNDS_ELIG)

    # NON-FINITE ROUNDS, and why they would be this harness's artifact rather than the arm's.
    # measure() advances its server with the DEFAULT aggregation, which is plain fedavg
    # (fl_core/federated.py:44-47) -- it has to, because the point is to hand every rung the same raw
    # update stack, not to run any one defense. Plain averaging does not filter the model_scaling
    # adversary, so the global model's scale compounds each round and can overflow float32; that is
    # what happened on FEMNIST/simple_cnn from round 2 of seed 42 onward. Once the global model is
    # non-finite so is every local update, every distance, and every coefficient, and no decision or
    # admission statistic computed on that round means anything. The arm this check licenses does NOT
    # have this property: there d2 = krum aggregates, and krum discards the scaled update instead of
    # averaging it in. Whether ResNet18 overflows within ROUNDS_ELIG rounds is reported below and not
    # predicted here: the frozen cifar_cnn measurement has 0/390 such rows but covers only 3 rounds,
    # so neither a zero nor a non-zero count here would be a surprise.
    # Such rounds are EXCLUDED from every statistic rather than averaged over: averaging a boolean
    # "did the decision change" over a round whose distances are all nan silently records "no change"
    # and biases the disturbance rate downward.
    rows = [r for r in all_rows if math.isfinite(r["c_mean"])]
    dropped = sorted({(r["seed"], r["round"]) for r in all_rows if not math.isfinite(r["c_mean"])})
    n_rounds = len(SEEDS3) * ROUNDS_ELIG
    print(f"\n=== NON-FINITE ROUNDS EXCLUDED: {len(dropped)}/{n_rounds} "
          f"(seed, round) = {dropped} ===")
    print("    measure() advances its server with plain fedavg so that every rung sees the same raw")
    print("    stack; fedavg does not filter the scaling adversary, so the global model can overflow")
    print("    and take the whole stack non-finite. No decision or admission statistic on such a")
    print("    round is meaningful. The ASR arm this check licenses aggregates with krum, which")
    print("    discards the scaled update instead of averaging it in, so it does not inherit this.")
    if len(dropped) >= n_rounds / 2:
        print("    MORE THAN HALF THE ROUNDS ARE NON-FINITE -- this cell is not measurable and the")
        print("    arm must not be run.")

    summary = {}

    print("\n=== DECISION CHANGE / ADMISSION CHANGE, all four arms, ResNet18 ===")
    print("  (left = the statistic-level disturbance Round 11 put on its abscissa;")
    print("   right = the adversarial mass actually admitted. The arm needs left > 0, right ~ 0.)\n")
    for family, vals in (("dose", KAPPAS), ("doseS", KAPPAS), ("doseA", NUS)):
        dial = "nu" if family == "doseA" else "kappa"
        print(f"  --- {family} ---")
        print(f"  {'arm':26s} " + " ".join(f"{dial + '=' + str(v):>15s}" for v in vals))
        for arm in ("krum", "reputation", "cos_krum", "coord_median"):
            cols = []
            for v in vals:
                sub = [r for r in rows if r["family"] == family and abs(r["rung"] - v) < 1e-12]
                d, a = rate(sub, DECISION_KEY[arm]), rate(sub, ADMISSION_KEY[arm])
                summary[f"{family}|{arm}|{v}|decision"] = d
                summary[f"{family}|{arm}|{v}|admission"] = a
                summary[f"{family}|{arm}|{v}|agg_disp"] = rate(sub, f"agg_disp_{arm}")
                cols.append(f"{d:.3f}/{a:.3f}")
            print(f"  {arm:26s} " + " ".join(f"{c:>15s}" for c in cols))
        print()

    # The eligibility verdict, stated as a rule before the numbers are seen rather than after. The
    # three branches and the two thresholds are copied from the FEMNIST check unchanged, so this arm
    # is held to the rule its predecessor was held to and not to one chosen after seeing these rows.
    d_hi = summary[f"doseS|{ARM}|{KAPPAS[-1]}|decision"]
    a_hi = summary[f"doseS|{ARM}|{KAPPAS[-1]}|admission"]
    print("=== ELIGIBILITY OF THE PLANNED ARM ===")
    print(f"  mode S {ARM} at kappa={KAPPAS[-1]}: decision change {d_hi:.3f}, "
          f"admission change {a_hi:.3f}")
    if d_hi < 0.10:
        verdict = ("INELIGIBLE: the Mode-S ladder barely disturbs this defense's decision on "
                   "ResNet18, so a flat ASR curve would carry no information about C2. This is the "
                   "cos_krum failure mode and the arm must not be run as a test of the negative.")
    elif a_hi > 0.05:
        verdict = ("CONFOUNDED: admission moves too, so a flat or rising ASR curve could not be "
                   "attributed to statistic disturbance alone. The arm is not the cifar_cnn arm's "
                   "analogue and must not be run as a replication.")
    else:
        verdict = ("ELIGIBLE: the decision is disturbed and the admitted adversarial mass is not, "
                   "which is the same premise the cifar_cnn flagship arm rests on. A flat ASR curve "
                   "here replicates the negative; a rise refutes it.")
    print(f"  {verdict}")
    print("  NOTE: eligibility here is about the CHANNELS only. Whether krum standalone actually")
    print("  suppresses model-scaling on ResNet18 at usable accuracy is an ASR question this script")
    print("  does not answer, and it is the kappa=0 rung of the arm itself. The pre-registration")
    print("  carries that as a separate void branch.")

    # THE SHARE ASSERTION NEEDS A DISCRIMINATOR, NOT JUST A TOLERANCE, AND HERE IS WHY.
    # SHARE_TOL = 1e-6 is an ABSOLUTE tolerance, imported unchanged and calibrated on cifar_cnn, whose
    # own Mode-S spread is 3.5e-07. The share is a RATIO whose denominator sums float32 coefficients
    # over every parameter, so its read-back noise grows with model size, and resnet18 has roughly two
    # orders of magnitude more parameters than cifar_cnn. An absolute threshold set on the small model
    # can therefore be crossed by noise alone on the large one, and reporting "confounded" off that
    # crossing would be a false positive -- while quietly widening the tolerance to make the arm pass
    # would be the opposite sin. So the tolerance is NOT touched; a second, substantive discriminator
    # is added and BOTH are printed:
    #   MONOTONICITY IN THE RUNG. A dose confound is monotone in the dial by construction -- that is
    #   what a dose is, and the `dose` family exhibits it. Float32 read-back noise is not. So a spread
    #   above tolerance that is non-monotone, and orders of magnitude below the smallest signal the
    #   instrument must resolve, is noise; a spread that is monotone is a confound at any magnitude.
    # Both quantities go in the artifact so the verdict is re-derivable and not taken on trust.
    print("\n=== INSTRUMENT CONSTRUCTION ASSERTIONS ON ResNet18 ===")
    share, share_diag = {}, {}
    frozen_share = (json.load(open(FROZEN)).get("adv_coeff_share", {})
                    if os.path.exists(FROZEN) else {})
    for family, vals in (("dose", KAPPAS), ("doseS", KAPPAS), ("doseA", NUS)):
        seq = []
        for v in vals:
            sub = [r for r in rows if r["family"] == family and abs(r["rung"] - v) < 1e-12
                   and r["n_adv_in_round"] > 0]
            s = float(np.mean([r["adv_coeff_share"] for r in sub])) if sub else float("nan")
            share[f"{family}|{v}"] = s
            seq.append(s)
        spread = float(max(seq) - min(seq))
        rel = spread / seq[0] if seq[0] else float("nan")
        diffs = [b - a for a, b in zip(seq, seq[1:])]
        monotone = all(d >= 0 for d in diffs) or all(d <= 0 for d in diffs)
        fz = [v for k, v in frozen_share.items() if k.startswith(family + "|")]
        fz_spread = (max(fz) - min(fz)) if fz else float("nan")
        if family == "doseA":
            ok = f"varies by {spread:.4f}, monotone={monotone} <- INTENDED (this is the payload dial)"
        elif family == "dose":
            ok = (f"varies by {spread:.4f}, monotone={monotone} <- ROUND 11's REPORTED CONFOUND: "
                  "this ladder moves the adversary's weight as well as the statistic, which is the "
                  "confound the paper reports, not a new defect")
        elif spread < SHARE_TOL:
            ok = f"CONSTANT (spread {spread:.2e} < {SHARE_TOL:.0e}, float32 norm read-back)"
        elif monotone:
            ok = (f"MONOTONE spread {spread:.2e} >= {SHARE_TOL:.0e} <- INSTRUMENT CONFOUNDED ON THIS "
                  "ARCHITECTURE: the share tracks the dial, so a Mode-S result here could not be "
                  "attributed to statistic disturbance alone. THE ARM MUST NOT RUN.")
        else:
            ok = (f"NON-MONOTONE spread {spread:.2e} >= {SHARE_TOL:.0e} (rel {rel:.1e}); cifar_cnn's "
                  f"own spread is {fz_spread:.2e} <- ABOVE THE IMPORTED TOLERANCE AND DISCLOSED AS "
                  "SUCH: not monotone in the rung, so it is read-back noise and not a dose. The "
                  "tolerance is absolute and was calibrated on a model two orders of magnitude "
                  "smaller; c_adv below is the assertion that is pinned by construction.")
        share_diag[family] = {"spread": spread, "relative_spread": rel, "monotone": bool(monotone),
                              "per_rung": seq, "cifar_cnn_spread": fz_spread,
                              "share_tol": SHARE_TOL, "verdict": ok}
        print(f"  {family:6s} " + "  ".join(f"{v:>5}:{s:.4f}" for v, s in zip(vals, seq)) + f"   {ok}")
    ds = share_diag["doseS"]
    print(f"\n  SCALE OF THE MODE-S DEVIATION AGAINST THE SIGNALS THE INSTRUMENT MUST RESOLVE:")
    print(f"    Mode S (must not move):        {ds['spread']:.3e}")
    print(f"    dose, the reported confound:   {share_diag['dose']['spread']:.3e} "
          f"({share_diag['dose']['spread'] / ds['spread']:.0f}x larger)")
    print(f"    Mode A, the intended dial:     {share_diag['doseA']['spread']:.3e} "
          f"({share_diag['doseA']['spread'] / ds['spread']:.0f}x larger)")

    print("\n  c_adv under mode S (must be exactly 1.0 at every rung):")
    cadv = {}
    for v in KAPPAS:
        sub = [r for r in rows if r["family"] == "doseS" and abs(r["rung"] - v) < 1e-12
               and r["n_adv_in_round"] > 0]
        lo = min(r["c_adv_min"] for r in sub); hi = max(r["c_adv_max"] for r in sub)
        dev = max(abs(lo - 1), abs(hi - 1))
        cadv[str(v)] = {"min": lo, "max": hi, "max_dev_from_1": dev}
        print(f"    kappa={v}: c_adv in [{lo:.12f}, {hi:.12f}]   max deviation from 1: {dev:.2e}")

    ndegen = sum(r["degenerate"] for r in rows if r["family"] == "doseS")
    ntot = sum(1 for r in rows if r["family"] == "doseS")
    print(f"\n  degenerate rung-rounds (0 adversaries or <2 benign): {ndegen}/{ntot} "
          f"({100.0 * ndegen / max(ntot, 1):.1f}%)")

    # Side-by-side with the frozen cifar_cnn numbers, READ ONLY, so the two architectures' premises
    # can be compared before the arm is frozen. Same dataset on both sides, unlike the FEMNIST
    # comparison, so this table is a clean architecture contrast. The frozen file is never written.
    cross = None
    if os.path.exists(FROZEN):
        fz = json.load(open(FROZEN))["summary"]
        print("\n=== SAME CELL, SAME DATASET, TWO ARCHITECTURES "
              "(frozen cifar_cnn file read, never modified) ===")
        print(f"  {'kappa':>6s}  {'cifar_cnn dec/adm':>20s}  {'resnet18 dec/adm':>20s}")
        cross = {}
        for v in KAPPAS:
            cd = fz.get(f"doseS|{ARM}|{v}|decision", float("nan"))
            ca = fz.get(f"doseS|{ARM}|{v}|admission", float("nan"))
            rd = summary[f"doseS|{ARM}|{v}|decision"]
            ra = summary[f"doseS|{ARM}|{v}|admission"]
            cross[str(v)] = {"cifar_cnn_decision": cd, "cifar_cnn_admission": ca,
                             "resnet18_decision": rd, "resnet18_admission": ra}
            print(f"  {v:>6}  {cd:9.3f}/{ca:<10.3f}  {rd:9.3f}/{ra:<10.3f}")

    json.dump({
        "description": "ResNet18 channel measurement: prospective eligibility check for the Mode-S "
                       f"{ARM}/{ATTACK} replication arm. Architecture varies alone; dataset, N, K, "
                       "f, alpha, attack and defense are the frozen CIFAR-10 values. No ASR. "
                       "Freezes nothing. results/admission_measurement.json is read for comparison "
                       "and never written.",
        "dataset": DATASET, "model": MODEL, "seeds": SEEDS3, "rounds_per_seed": ROUNDS_ELIG,
        "frozen_measurement_rounds": ROUNDS,
        "kappas": KAPPAS, "nus": NUS, "arm": ARM, "attack": ATTACK,
        "eligibility": {"decision_change_at_max_kappa": d_hi,
                        "admission_change_at_max_kappa": a_hi, "verdict": verdict,
                        "scope": "channels only; standalone suppression at usable accuracy is the "
                                 "arm's own kappa=0 rung and is a separate pre-registered branch"},
        "nonfinite_rounds_excluded": [list(x) for x in dropped],
        "n_rounds_total": n_rounds,
        "per_round": rows,
        "per_round_including_nonfinite": all_rows,
        "summary": summary,
        "adv_coeff_share": share,
        "adv_coeff_share_diagnostic": share_diag,
        "c_adv_mode_s": cadv,
        "n_degenerate_modeS_rung_rounds": int(ndegen),
        "cross_architecture_mode_s": cross,
    }, open(OUT, "w"), indent=1)
    print(f"\nWrote {OUT}")
    print("results/admission_measurement.json was not modified.")


if __name__ == "__main__":
    main()
