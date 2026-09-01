"""
Channel measurement for the coordinate-masking family (Mode M), before any ASR exists.

WHY THIS RUNS FIRST. Every controlled intervention this paper has run is a positive per-client
rescaling, `u_i -> c_i * u_i` with `c_i > 0`. That is exactly the hypothesis of the invariance
proposition and of the bounded-reweighting theorem, so the paper's central negative -- decision
preserved, admission preserved, suppression NOT preserved, and its mirror -- is open to a reading it
cannot answer from inside that family: the non-implication might be a property of scalar
multiplication rather than of statistic preservation. Mode M leaves the family. Benign updates lose a
Bernoulli(m) fraction of their coordinates and are renormalized to their own original L2 norm;
adversarial updates are passed through bit-identically. That is not `c_i * u_i` for any scalar `c_i`,
so the theorem does not cover it, which is the point rather than a gap.

An arm is only informative if its PREMISE holds, and the premise is a statement about channels, not
about ASR. So this script establishes, before any hypothesis about masked ASR is frozen:

  1. Does the mask ladder actually disturb Krum's decision? If it does not, a flat ASR curve carries
     no information -- that is the cos_krum failure mode the paper already discloses.
  2. Does it move the admitted adversarial mass? Unlike the second-dataset check, a YES here is NOT
     disqualifying. It selects which of two genuinely different arms this is, and the choice is made
     here rather than after the outcome:
       admission ~ 0  -> the arm generalizes the flagship negative to a non-rescaling transform, and
                         the frozen prediction is FLAT.
       admission > 0  -> the arm tests the admission reading itself, and the frozen prediction is
                         that ASR moves WITH admission. This CONFIRMS the paper's reading of its own
                         mechanism and refutes nothing, but it must be claimed before the fact.
  3. Do the construction assertions hold on real update stacks, through the shipped code path?
       c_adv == 1.0 with maximum deviation exactly 0.00e+00 -- and here that is a bit-level claim
         rather than an exact-arithmetic one, because the adversarial entries of the returned list
         ARE the input objects, so their norms are the same float32 numbers and the read-back ratio
         is exactly 1.0. Mode S can only claim this in exact arithmetic (measured 3.5e-07).
       rho_realized == 1.0 to float32 noise, i.e. EVERY client's norm is preserved, not just the
         adversary's. Mode S reaches rho = 54.6 at its top rung. So Mode M closes the per-client
         magnitude channel that Mode S leaves open, and disturbs direction alone.

This script computes NO ASR and trains no model per rung: one raw update stack per (seed, round) is
shared by every rung, so all four cost the same three rounds of local training. It freezes nothing.
The frozen artifact results/admission_measurement.json is read for comparison and NEVER written.

A free cross-class prediction, which is the reason to measure all four aggregators rather than only
Krum: cos_krum's statistic is EXACTLY invariant under positive rescaling (0/120 rounds change
selection, measured), so under Modes S and A its decision cannot move. Masking changes cosine
distances, so it must. If that shows up here, one transformation class reclassifies an aggregator the
other class cannot, which is the strongest thing a two-class prediction matrix can contain.

Config matches the frozen CIFAR-10 measurement exactly -- N=10, K=5, f=0.2, alpha=0.5, cifar_cnn,
seeds 42-46, 3 live rounds, both committed attacks -- so the mask columns are directly comparable to
the Mode-S columns rather than merely adjacent to them.

Output: results/mask_admission.json
Run: python3 experiments/measure_admission_mask.py
"""
import json, math, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
from fl_core import get_federated_dataset, get_model, FederatedServer, FederatedClient  # noqa: E402
from attacks import get_attack  # noqa: E402
from experiments.run_all_compositions import apply_d1_transform  # noqa: E402
# Single-sourced from the frozen CIFAR-10 measurement: the same measure(), the same decision and
# admission field names, the same rate(), the same aggregate-displacement definition. Only the rung
# list is new, and it is passed in rather than edited into that module, so the default call still
# reproduces results/admission_measurement.json exactly.
from experiments.measure_admission import (  # noqa: E402
    measure, rate, KAPPAS, SEEDS, ROUNDS, ARM_ATTACK, DECISION_KEY, ADMISSION_KEY,
)

OUT = os.path.join(base, "results", "mask_admission.json")
FROZEN = os.path.join(base, "results", "admission_measurement.json")

# The mask ladder. Four rungs so the table shape matches every other arm in the paper, m=0 first so
# the identity is measured by the same code path as the rest rather than assumed.
DROPS = [0.0, 0.2, 0.5, 0.8]
RUNGS_M = [("doseM", m, f"doseM_m{m}") for m in DROPS]

ARM, ATTACK = "krum", "committed_scaling"      # the cell that produced the flagship negative
COS_ARM = "cos_krum"                           # the cross-class prediction
DECISION_FLOOR = 0.10                          # below this there is no disturbance to test
ADMISSION_FLAT = 0.05                          # above this the arm is an admission test, not a
                                               # generalization of the flagship negative

# TWO TOLERANCES, AND WHY THERE HAVE TO BE TWO. measure() reads every client's coefficient back from
# the float32 norms of the transformed stack (measure_admission.py:148) rather than from the builder,
# which is deliberate: that is how it verifies the SHIPPED code path applied what the docstring
# claims. It therefore cannot resolve the construction's actual precision. flatten() concatenates
# ~1.1e6 float32 coordinates and takes a float32 norm, and a masked-and-rescaled update is sparser
# with correspondingly larger surviving entries, so the two norms in the ratio do not accumulate the
# same rounding: the measured read-back deviation grows with the drop rate and reaches 1.4e-04 at
# m=0.8. That is a property of the measurement, not of the transform, and it is exactly the situation
# SHARE_TOL documents for Mode S at 3.5e-07.
# So the norm-preservation claim is checked TWICE, and only the second one is the claim:
#   READBACK_TOL  bounds the float32 read-back, and controls a printed verdict only.
#   EXACT_TOL     bounds the float64 construction check below, which recomputes both norms in
#                 float64 on the CPU from the same live update stacks. That is the assertion the
#                 arm rests on, and it holds at ~3e-08 -- the float32 rounding of the rescale
#                 product itself, which is the floor for any transform that emits float32.
READBACK_TOL = 2e-4
EXACT_TOL = 1e-6
CHECK_SEED, CHECK_ROUNDS = SEEDS[0], 2


def construction_check():
    """Norm preservation and adversarial bit-identity in float64, on live update stacks.

    Two claims, both of which the float32 read-back in measure() is too coarse to establish:

      benign      ||T(u_j)|| / ||u_j|| == 1 to float64 precision, for every benign j and every rung.
      adversarial T(u_i) is the SAME OBJECT as u_i, so the ratio is exactly 1.0 and torch.equal
                  holds on every tensor. Mode S can only claim c_adv = 1 in exact arithmetic; here
                  it is object identity, which is a strictly stronger statement.

    Deliberately NOT folded into measure(): that function's float32 read-back is the check that the
    shipped code path applied what the builder documents, and replacing it with a float64 one would
    weaken it. These are different claims and they get different checks.
    """
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(CHECK_SEED); np.random.seed(CHECK_SEED)
    cd, _, nc = get_federated_dataset("cifar10", 10, 0.5, CHECK_SEED)
    srv = FederatedServer(get_model("cifar_cnn", nc), dev)
    atk = get_attack("model_scaling")
    adv = set(range(2))
    cl = [FederatedClient(i, atk.poison_dataset(cd[i]) if i in adv else cd[i], dev)
          for i in range(10)]
    rows, worst_ben, ident_ok = [], 0.0, True
    for rnd in range(CHECK_ROUNDS):
        pids = np.random.choice(10, 5, replace=False)
        ups = []
        for cid in pids:
            u = cl[cid].train(srv.global_model, 1, 0.01, 64)
            if cid in adv:
                u = atk.manipulate_update(u, srv.global_model)
            ups.append(u)
        am = [bool(cid in adv) for cid in pids]
        for m in DROPS[1:]:
            t = apply_d1_transform(ups, f"doseM_m{m}", tau=5.0, dose_key=(CHECK_SEED, rnd),
                                   adv_mask=am)
            for i, is_adv in enumerate(am):
                a = torch.cat([ups[i][k].flatten().detach().cpu().double() for k in ups[i]])
                b = torch.cat([t[i][k].flatten().detach().cpu().double() for k in t[i]])
                ratio = float(b.norm() / a.norm().clamp(min=1e-300))
                same = all(t[i][k] is ups[i][k] for k in ups[i])
                if is_adv:
                    ident_ok = ident_ok and same and ratio == 1.0
                else:
                    worst_ben = max(worst_ben, abs(ratio - 1.0))
                rows.append({"round": rnd, "m": m, "row": i, "adversarial": is_adv,
                             "norm_ratio_float64": ratio, "object_identity": same})
        srv.apply_update(srv.aggregate(ups))
    return {"rounds": CHECK_ROUNDS, "seed": CHECK_SEED, "drops": DROPS[1:],
            "worst_benign_abs_dev_from_1": worst_ben,
            "adversarial_object_identity_and_exact_ratio": ident_ok,
            "exact_tol": EXACT_TOL, "passes": bool(worst_ben < EXACT_TOL and ident_ok),
            "per_row": rows}


def main():
    print("=== MODE M (coordinate masking) channel measurement: no ASR, no per-rung training ===")
    print(f"    {len(SEEDS)} seeds x {ROUNDS} live rounds x {len(RUNGS_M)} rungs x 2 attacks, "
          "every rung on the same raw updates")
    print(f"    drop rates m = {DROPS}")
    print(f"    arm under consideration: mode M {ARM}/{ATTACK.replace('committed_', '')}")
    print("    PROSPECTIVE: this runs before any hypothesis about masked ASR is frozen.\n",
          flush=True)

    all_rows = []
    for atk in ("committed_scaling", "committed_pixel"):
        print(f"-- {atk} --", flush=True)
        all_rows += measure(atk, rungs=RUNGS_M)

    # Non-finite rounds are excluded rather than averaged over, for the reason
    # measure_admission_femnist.py documents at length: measure() advances its server with plain
    # fedavg so that every rung sees the same raw stack, and plain averaging does not filter the
    # model_scaling adversary. Averaging a boolean "did the decision change" over a round whose
    # distances are all nan silently records "no change" and biases the disturbance rate downward.
    rows = [r for r in all_rows if math.isfinite(r["c_mean"])]
    dropped = sorted({(r["seed"], r["round"], r["attack"])
                      for r in all_rows if not math.isfinite(r["c_mean"])})
    print(f"\n=== NON-FINITE ROUNDS EXCLUDED: {len(dropped)} of "
          f"{len(SEEDS) * ROUNDS * 2} (seed, round, attack) ===")
    if dropped:
        print(f"    {dropped}")

    summary = {}
    print("\n=== DECISION CHANGE / ADMISSION CHANGE UNDER MASKING ===")
    print("  (left = statistic-level disturbance, the quantity a decision-level reading uses;")
    print("   right = the adversarial mass actually admitted, which is what the mechanism claim")
    print("   is about. Each arm is measured on its own committed attack, as in the frozen file.)\n")
    print(f"  {'arm':26s} " + " ".join(f"{'m=' + str(v):>15s}" for v in DROPS))
    for arm in ("krum", "reputation", "cos_krum", "coord_median"):
        cols = []
        for v in DROPS:
            sub = [r for r in rows if r["attack"] == ARM_ATTACK[arm] and abs(r["rung"] - v) < 1e-12]
            d, a = rate(sub, DECISION_KEY[arm]), rate(sub, ADMISSION_KEY[arm])
            summary[f"doseM|{arm}|{v}|decision"] = d
            summary[f"doseM|{arm}|{v}|admission"] = a
            summary[f"doseM|{arm}|{v}|agg_disp"] = rate(sub, f"agg_disp_{arm}")
            cols.append(f"{d:.3f}/{a:.3f}")
        print(f"  {arm:26s} " + " ".join(f"{c:>15s}" for c in cols))

    print("\n=== AGGREGATE DISPLACEMENT ||agg(T(U)) - agg(U)|| / ||agg(U)|| ===")
    print(f"  {'arm':26s} " + " ".join(f"{'m=' + str(v):>15s}" for v in DROPS))
    for arm in ("krum", "reputation", "cos_krum", "coord_median"):
        print(f"  {arm:26s} "
              + " ".join(f"{summary[f'doseM|{arm}|{v}|agg_disp']:>15.3f}" for v in DROPS))

    # --- The eligibility verdict, as a rule stated before the numbers are read ---
    d_hi = summary[f"doseM|{ARM}|{DROPS[-1]}|decision"]
    a_hi = summary[f"doseM|{ARM}|{DROPS[-1]}|admission"]
    a_all = [summary[f"doseM|{ARM}|{v}|admission"] for v in DROPS]
    print("\n=== ELIGIBILITY AND ARM TYPE (rule fixed above, in DECISION_FLOOR/ADMISSION_FLAT) ===")
    print(f"  mode M {ARM} at m={DROPS[-1]}: decision change {d_hi:.3f}, admission change {a_hi:.3f}")
    print(f"  admission across the ladder: " + "  ".join(f"m={v}:{a:.3f}"
                                                         for v, a in zip(DROPS, a_all)))
    if d_hi < DECISION_FLOOR:
        arm_type = "INELIGIBLE"
        verdict = (f"INELIGIBLE: masking barely disturbs {ARM}'s decision (< {DECISION_FLOOR}), so a "
                   "flat ASR curve would carry no information about statistic preservation. This is "
                   "the cos_krum failure mode and the arm must not be run as a test of the negative.")
    elif max(a_all) <= ADMISSION_FLAT:
        arm_type = "GENERALIZATION"
        verdict = ("ELIGIBLE AS A GENERALIZATION TEST: the decision is disturbed and the admitted "
                   "adversarial mass is not, which is the same premise the CIFAR-10 flagship arm "
                   "rests on -- now outside the scalar-rescaling family. FLAT replicates the "
                   "negative on a second transformation class; a rise refutes it.")
    else:
        arm_type = "ADMISSION"
        verdict = ("ELIGIBLE AS AN ADMISSION TEST: masking moves the admitted adversarial mass, so "
                   "this is NOT an analogue of the flagship arm and must not be reported as one. It "
                   "tests the paper's own reading instead: if admission is what governs suppression, "
                   "ASR must move WITH admission here. A flat curve at moved admission would refute "
                   "that reading.")
    print(f"  {verdict}")

    # --- Construction assertions, on real stacks, through the shipped code path ---
    print("\n=== CONSTRUCTION ASSERTIONS (measured, not asserted in a docstring) ===")
    print("  c_adv must be EXACTLY 1.0: the adversarial entries of the returned list are the input")
    print("  objects, so their float32 norms are identical and the read-back ratio is exactly 1.")
    cadv = {}
    for v in DROPS:
        sub = [r for r in rows if abs(r["rung"] - v) < 1e-12 and r["n_adv_in_round"] > 0]
        lo = min(r["c_adv_min"] for r in sub); hi = max(r["c_adv_max"] for r in sub)
        dev = max(abs(lo - 1.0), abs(hi - 1.0))
        cadv[str(v)] = {"min": lo, "max": hi, "max_dev_from_1": dev}
        print(f"    m={v}: c_adv in [{lo:.12f}, {hi:.12f}]   max deviation from 1: {dev:.2e}"
              + ("   BIT-EXACT" if dev == 0.0 else "   NOT BIT-EXACT"))

    print("\n  EVERY client's norm must be preserved, so rho_realized = max(c)/min(c) must be 1.0.")
    print("  This is the channel Mode S leaves open (rho = 54.6 at its top rung) and Mode M closes.")
    print(f"  Read back through float32 flatten(), so bounded by READBACK_TOL={READBACK_TOL:g}; the")
    print("  construction's own precision is the separate float64 check below.")
    norms = {}
    for v in DROPS:
        sub = [r for r in rows if abs(r["rung"] - v) < 1e-12 and not r["degenerate"]]
        rr = [r["rho_realized"] for r in sub]; cm = [r["c_mean"] for r in sub]
        dev = max(abs(max(rr) - 1.0), abs(min(rr) - 1.0))
        norms[str(v)] = {"rho_min": min(rr), "rho_max": max(rr), "max_abs_dev_from_1": dev,
                         "c_mean_min": min(cm), "c_mean_max": max(cm)}
        print(f"    m={v}: rho in [{min(rr):.8f}, {max(rr):.8f}]   mean(c) in "
              f"[{min(cm):.8f}, {max(cm):.8f}]   dev {dev:.2e}   "
              + ("norm-preserving to read-back precision" if dev < READBACK_TOL
                 else f"EXCEEDS READBACK_TOL {READBACK_TOL:g}"))

    print("\n  The claim itself, recomputed in float64 on live stacks (construction_check):")
    cc = construction_check()
    print(f"    benign  worst |ratio - 1| over {len(cc['per_row'])} client-rungs: "
          f"{cc['worst_benign_abs_dev_from_1']:.2e}   (tol {EXACT_TOL:g})")
    print(f"    adversary object identity AND exactly-1.0 ratio at every rung: "
          f"{cc['adversarial_object_identity_and_exact_ratio']}")
    print(f"    {'CONSTRUCTION VERIFIED' if cc['passes'] else 'CONSTRUCTION FAILS -- do not freeze'}")

    # --- The cross-class prediction, which only two families together can make ---
    cross = None
    if os.path.exists(FROZEN):
        fz = json.load(open(FROZEN))["summary"]
        print("\n=== TWO TRANSFORMATION CLASSES, SAME AGGREGATORS ===")
        print("  (frozen rescaling file read, never modified. Mode S is indexed by kappa and Mode M")
        print("   by m: the rungs are NOT comparable in dose, only in whether the channel moved.)\n")
        print(f"  {'arm':14s} {'rescaling (Mode S) decision':>30s} {'masking (Mode M) decision':>28s}")
        cross = {}
        for arm in ("krum", "reputation", "cos_krum", "coord_median"):
            s_hi = fz.get(f"doseS|{arm}|{KAPPAS[-1]}|decision", float("nan"))
            m_hi = summary[f"doseM|{arm}|{DROPS[-1]}|decision"]
            s_ad = fz.get(f"doseS|{arm}|{KAPPAS[-1]}|admission", float("nan"))
            m_ad = summary[f"doseM|{arm}|{DROPS[-1]}|admission"]
            cross[arm] = {"modeS_decision_at_max": s_hi, "modeM_decision_at_max": m_hi,
                          "modeS_admission_at_max": s_ad, "modeM_admission_at_max": m_ad,
                          "reclassified": bool(s_hi < 1e-12 and m_hi >= 1e-12)}
            flag = "  <- RECLASSIFIED by the second class" if cross[arm]["reclassified"] else ""
            print(f"  {arm:14s} {s_hi:>30.3f} {m_hi:>28.3f}{flag}")
        print(f"\n  {COS_ARM} is the one the theory names: exactly invariant under positive")
        print("  rescaling, and masking changes cosine distances, so its decision MUST move here.")
        print(f"  Measured: Mode S {cross[COS_ARM]['modeS_decision_at_max']:.3f} -> "
              f"Mode M {cross[COS_ARM]['modeM_decision_at_max']:.3f}")

    ndegen = sum(r["degenerate"] for r in rows)
    print(f"\n  degenerate rung-rounds (0 adversaries or <2 benign): {ndegen}/{len(rows)} "
          f"({100.0 * ndegen / max(len(rows), 1):.1f}%)")

    json.dump({
        "description": "Mode M (coordinate masking) channel measurement: prospective eligibility "
                       f"check and arm-type selection for the mode M {ARM}/{ATTACK} arm. No ASR. "
                       "Freezes nothing. results/admission_measurement.json is read for the "
                       "two-class comparison and never written.",
        "family": "doseM", "drops": DROPS, "seeds": SEEDS, "rounds_per_seed": ROUNDS,
        "dataset": "cifar10", "model": "cifar_cnn",
        "arm": ARM, "attack": ATTACK, "arm_attack": ARM_ATTACK,
        "rules": {"decision_floor": DECISION_FLOOR, "admission_flat": ADMISSION_FLAT,
                  "readback_tol": READBACK_TOL, "exact_tol": EXACT_TOL},
        "eligibility": {"decision_change_at_max_drop": d_hi,
                        "admission_change_at_max_drop": a_hi,
                        "admission_across_ladder": dict(zip(map(str, DROPS), a_all)),
                        "arm_type": arm_type, "verdict": verdict},
        "nonfinite_rounds_excluded": [list(x) for x in dropped],
        "per_round": rows,
        "per_round_including_nonfinite": all_rows,
        "summary": summary,
        "c_adv_mode_m": cadv,
        "norm_preservation_float32_readback": norms,
        "construction_check_float64": cc,
        "two_class_comparison": cross,
        "n_degenerate_rung_rounds": int(ndegen),
    }, open(OUT, "w"), indent=1)
    print(f"\nWrote {OUT}")
    print("results/admission_measurement.json was not modified.")


if __name__ == "__main__":
    main()
