"""
Second-dataset eligibility check for the Mode-S replication arm: FEMNIST, before any ASR exists.

WHY THIS RUNS FIRST. The planned arm is FEMNIST / krum / model_scaling under Mode S -- the exact cell
that produced the paper's flagship negative on CIFAR-10 (decision changed in 80% of rounds, admission
in 0%, suppression unharmed). That arm is only informative if its PREMISE holds on the new dataset:
the Mode-S ladder must actually disturb Krum's decision on FEMNIST, and must leave the admitted
adversarial mass alone. If the ladder does not move Krum's decision here, the arm has no disturbance
to test and a flat ASR curve would be uninformative rather than a replication -- which is exactly the
cos_krum failure mode the paper already discloses. Establishing the premise BEFORE the hypothesis is
frozen is the difference between a prospective test and a post-hoc rationalization.

This script computes NO ASR and trains no model per rung: one raw update stack per (seed, round) is
shared by every transform, so all thirteen rungs cost the same three rounds of local training.

WHAT IS FROZEN BY THIS SCRIPT: nothing. It measures the channels and the instrument's own
construction assertions on FEMNIST. The frozen artifact results/admission_measurement.json is NEVER
read for writing and never modified; this writes its own file.

THE INSTRUMENT'S CONSTRUCTION ASSERTIONS, re-checked on the new dataset (they are properties of the
transform, not of CIFAR-10, so they must hold exactly here too):
  - Mode S: adversarial coefficient share CONSTANT across rungs, c_adv == 1.0 exactly at every rung.
  - Mode A: adversarial coefficient share monotone in nu.
If either fails on FEMNIST the instrument is not doing on this dataset what it does on CIFAR-10 and
the arm must not run.

Config matches the planned arm exactly: N=10, K=5, f=0.2, alpha=0.5, simple_cnn, seeds 42/43/44
(the FEMNIST payoff matrix ran 3 trials; see experiments/run_femnist.py).

Output: results/femnist_admission.json
Run: python3 experiments/measure_admission_femnist.py
"""
import json, math, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
# Single-sourced from the frozen CIFAR-10 measurement: the same measure(), the same rung list, the
# same decision/admission field names, the same rate(). Nothing about the channels is reimplemented,
# so the two datasets cannot be measured by two different definitions.
from experiments.measure_admission import (
    measure, rate, RUNGS, KAPPAS, NUS, ROUNDS, DECISION_KEY, ADMISSION_KEY, SHARE_TOL,
)

OUT = os.path.join(base, "results", "femnist_admission.json")
FROZEN = os.path.join(base, "results", "admission_measurement.json")

DATASET, MODEL = "femnist", "simple_cnn"
SEEDS3 = [42, 43, 44]              # the seeds the planned ASR arm will use
ROUNDS_ELIG = 6                    # more rounds than the CIFAR-10 measurement used: on FEMNIST
                                   # some rounds are non-finite and get dropped, so the premise
                                   # needs headroom to rest on a comparable number of live rounds
ARM, ATTACK = "krum", "committed_scaling"


def main():
    print("=== FEMNIST channel measurement (no ASR, no per-rung training) ===")
    print(f"    dataset={DATASET} model={MODEL}  seeds={SEEDS3} x {ROUNDS_ELIG} rounds x "
          f"{len(RUNGS)} rungs")
    print(f"    arm under consideration: mode S {ARM}/{ATTACK.replace('committed_', '')}")
    print("    PROSPECTIVE: this runs before any hypothesis about FEMNIST ASR is frozen.\n",
          flush=True)

    all_rows = measure(ATTACK, dataset=DATASET, model=MODEL, seeds=SEEDS3, rounds=ROUNDS_ELIG)

    # NON-FINITE ROUNDS, and why they are this harness's artifact rather than the arm's.
    # measure() advances its server with the DEFAULT aggregation, which is plain fedavg
    # (fl_core/federated.py:44-47) -- it has to, because the point is to hand every rung the same raw
    # update stack, not to run any one defense. Plain averaging does not filter the model_scaling
    # adversary, so the global model's scale compounds each round, and on FEMNIST/simple_cnn it
    # overflows float32 from round 2 of seed 42 onward. Once the global model is non-finite so is
    # every local update, every distance, and every coefficient, and no decision or admission
    # statistic computed on that round means anything. The arm this check licenses does NOT have this
    # property: there d2 = krum aggregates, and krum discards the scaled update instead of averaging
    # it in. The frozen CIFAR-10 measurement has 0/390 such rows, but it covers only 3 rounds, so the
    # honest statement is that FEMNIST/simple_cnn overflows within 6 rounds where CIFAR-10/cifar_cnn
    # does not within 3 -- not that CIFAR-10 would never overflow at this depth.
    # Such rounds are EXCLUDED from every statistic rather than averaged over: averaging a boolean
    # "did the decision change" over a round whose distances are all nan silently records "no change"
    # and biases the disturbance rate downward.
    rows = [r for r in all_rows if math.isfinite(r["c_mean"])]
    dropped = sorted({(r["seed"], r["round"]) for r in all_rows if not math.isfinite(r["c_mean"])})
    n_rounds = len(SEEDS3) * ROUNDS_ELIG
    print(f"\n=== NON-FINITE ROUNDS EXCLUDED: {len(dropped)}/{n_rounds} "
          f"(seed, round) = {dropped} ===")
    print("    measure() advances its server with plain fedavg so that every rung sees the same raw")
    print("    stack; fedavg does not filter the scaling adversary, so the global model overflows and")
    print("    the whole stack goes non-finite. No decision or admission statistic on such a round is")
    print("    meaningful. The ASR arm this check licenses aggregates with krum, which discards the")
    print("    scaled update instead of averaging it in, so it does not inherit this.")
    if len(dropped) >= n_rounds / 2:
        print("    MORE THAN HALF THE ROUNDS ARE NON-FINITE -- this cell is not measurable and the")
        print("    arm must not be run.")

    summary = {}

    print("\n=== DECISION CHANGE / ADMISSION CHANGE, all four arms, FEMNIST ===")
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

    # The eligibility verdict, stated as a rule before the numbers are seen rather than after.
    d_hi = summary[f"doseS|{ARM}|{KAPPAS[-1]}|decision"]
    a_hi = summary[f"doseS|{ARM}|{KAPPAS[-1]}|admission"]
    print("=== ELIGIBILITY OF THE PLANNED ARM ===")
    print(f"  mode S {ARM} at kappa={KAPPAS[-1]}: decision change {d_hi:.3f}, "
          f"admission change {a_hi:.3f}")
    if d_hi < 0.10:
        verdict = ("INELIGIBLE: the Mode-S ladder barely disturbs this defense's decision on "
                   "FEMNIST, so a flat ASR curve would carry no information about C2. This is the "
                   "cos_krum failure mode and the arm must not be run as a test of the negative.")
    elif a_hi > 0.05:
        verdict = ("CONFOUNDED: admission moves too, so a flat or rising ASR curve could not be "
                   "attributed to statistic disturbance alone. The arm is not the CIFAR-10 arm's "
                   "analogue and must not be run as a replication.")
    else:
        verdict = ("ELIGIBLE: the decision is disturbed and the admitted adversarial mass is not, "
                   "which is the same premise the CIFAR-10 flagship arm rests on. A flat ASR curve "
                   "here replicates the negative; a rise refutes it.")
    print(f"  {verdict}")

    print("\n=== INSTRUMENT CONSTRUCTION ASSERTIONS ON FEMNIST ===")
    share = {}
    for family, vals in (("dose", KAPPAS), ("doseS", KAPPAS), ("doseA", NUS)):
        seq = []
        for v in vals:
            sub = [r for r in rows if r["family"] == family and abs(r["rung"] - v) < 1e-12
                   and r["n_adv_in_round"] > 0]
            s = float(np.mean([r["adv_coeff_share"] for r in sub])) if sub else float("nan")
            share[f"{family}|{v}"] = s
            seq.append(s)
        spread = float(max(seq) - min(seq))
        ok = (f"CONSTANT (spread {spread:.2e} < {SHARE_TOL:.0e})" if spread < SHARE_TOL else
              f"varies by {spread:.4f}" + (" <- INTENDED" if family == "doseA" else
                                           " <- INSTRUMENT CONFOUNDED ON THIS DATASET"))
        print(f"  {family:6s} " + "  ".join(f"{v:>5}:{s:.4f}" for v, s in zip(vals, seq)) + f"   {ok}")

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

    # Side-by-side with the frozen CIFAR-10 numbers, READ ONLY, so the two datasets' premises can be
    # compared before the arm is frozen. The frozen file is never written.
    cross = None
    if os.path.exists(FROZEN):
        fz = json.load(open(FROZEN))["summary"]
        print("\n=== SAME CELL, TWO DATASETS (frozen CIFAR-10 file read, never modified) ===")
        print(f"  {'kappa':>6s}  {'CIFAR-10 dec/adm':>20s}  {'FEMNIST dec/adm':>20s}")
        cross = {}
        for v in KAPPAS:
            cd = fz.get(f"doseS|{ARM}|{v}|decision", float("nan"))
            ca = fz.get(f"doseS|{ARM}|{v}|admission", float("nan"))
            fd = summary[f"doseS|{ARM}|{v}|decision"]
            fa = summary[f"doseS|{ARM}|{v}|admission"]
            cross[str(v)] = {"cifar10_decision": cd, "cifar10_admission": ca,
                             "femnist_decision": fd, "femnist_admission": fa}
            print(f"  {v:>6}  {cd:9.3f}/{ca:<10.3f}  {fd:9.3f}/{fa:<10.3f}")

    json.dump({
        "description": "FEMNIST channel measurement: prospective eligibility check for the Mode-S "
                       f"{ARM}/{ATTACK} replication arm. No ASR. Freezes nothing. "
                       "results/admission_measurement.json is read for comparison and never written.",
        "dataset": DATASET, "model": MODEL, "seeds": SEEDS3, "rounds_per_seed": ROUNDS_ELIG,
        "kappas": KAPPAS, "nus": NUS, "arm": ARM, "attack": ATTACK,
        "eligibility": {"decision_change_at_max_kappa": d_hi,
                        "admission_change_at_max_kappa": a_hi, "verdict": verdict},
        "nonfinite_rounds_excluded": [list(x) for x in dropped],
        "n_rounds_total": n_rounds,
        "per_round": rows,
        "per_round_including_nonfinite": all_rows,
        "summary": summary,
        "adv_coeff_share": share,
        "c_adv_mode_s": cadv,
        "n_degenerate_modeS_rung_rounds": int(ndegen),
        "cross_dataset_mode_s": cross,
    }, open(OUT, "w"), indent=1)
    print(f"\nWrote {OUT}")
    print("results/admission_measurement.json was not modified.")


if __name__ == "__main__":
    main()
