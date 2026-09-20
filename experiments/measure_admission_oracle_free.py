"""
Channel measurement for the ORACLE-FREE arm, before any ASR exists.

WHY THIS RUNS FIRST. A review named one condition for raising its score: a non-floor-effect causal
intervention with more seeds and no dependence on oracle adversary identity. Every published instance
of either leg of the channel dissociation is a doseS/doseA/doseM rung, and apply_d1_transform RAISES
for those without adv_mask -- "refusing to run a targeted dose without knowing which participants are
adversarial" -- so every published instance is an oracle instance. This arm keeps the paper's own
pre-registered factorial controls (score_only / emit_only) and changes only the transform: rfa as d1
runs the Weiszfeld iteration and rescales each client by its final Weiszfeld weight, computed from the
update stack alone.

An arm is only informative if its PREMISE holds, and the premise is a statement about channels rather
than about ASR. So this script establishes, before any hypothesis about oracle-free ASR is frozen:

  1. Does rfa actually disturb Krum's decision on this cell? If it does not, a flat ASR curve carries
     no information -- the cos_krum failure mode the paper already discloses -- and the 16-hour
     ladder must not run at all.
  2. Does it move the admitted adversarial mass? A YES is not disqualifying; it selects which of two
     genuinely different arms this is, and the choice is made here rather than after the outcome.
  3. Is the composition really oracle-free, through the shipped code path? Asserted by running
     apply_d1_transform twice on the same live stack, once with the true adv_mask and once with
     adv_mask=None, and demanding BIT-IDENTICAL output. That is a proof rather than a reading of the
     branch structure.
  4. Do the two controls do what their pre-registrations say, at the bit level? emit_only must score
     the RAW stack, so its decision is pinned to the identity arm's; score_only must emit the RAW
     selected update. Both are checked with torch.equal against independently constructed aggregates,
     because the standing failure mode in this repository is a hook that is never called while the
     run passes silently.

The host cell is krum under committed_pixel, and that choice is the answer to the review's other
objection. Recomputed read-only from results/admission_measurement.json's per_round rows, the same
Krum under the two committed attacks:

    committed_scaling   baseline krum_admits_adv 0.0000, nonzero in  0 of 12 adversary rounds
    committed_pixel     baseline krum_admits_adv 0.3333, nonzero in  4 of 12 adversary rounds

so under the pixel backdoor admission has somewhere to fall and an unchanged admission is a
measurement rather than a ceiling artifact. This script recomputes that baseline from its own rows and
must reproduce it, which is its harness check.

The three-way eligibility rule and its two constants are IMPORTED from measure_admission_mask.py, not
restated, so this arm is typed by literally the numbers Mode M was typed by. The rule and this arm's
frozen predictions are in experiments/pre_registration_oracle_free_channels.md; the rule was written
there before any number below was read.

This script computes NO ASR and trains no model per rung: one raw update stack per (seed, round) is
shared by both rungs. It freezes nothing. results/admission_measurement.json is read for the frozen
comparison and NEVER written.

Output: results/oracle_free_admission.json
Run:    PYTHONPATH=. python3 -m experiments.measure_admission_oracle_free
"""
import json, math, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
from fl_core import get_federated_dataset, get_model, FederatedServer, FederatedClient  # noqa: E402
from attacks import get_attack  # noqa: E402
from experiments.run_all_compositions import apply_d1_transform, generic_compose  # noqa: E402
# Single-sourced from the frozen CIFAR-10 measurement: the same measure(), the same decision and
# admission field names, the same rate(), the same seeds and live-round count. Only the rung list is
# new, and it is passed in rather than edited into that module, so the default call still reproduces
# results/admission_measurement.json exactly.
from experiments.measure_admission import (  # noqa: E402
    measure, rate, admission, flatten, KAPPAS, SEEDS, ROUNDS, ARM_ATTACK,
    DECISION_KEY, ADMISSION_KEY, SHARE_TOL,
)
from experiments.verify_cos_invariance import krum_selection  # noqa: E402
# The eligibility rule is Mode M's rule, imported rather than re-typed, so the two prospective
# measurements cannot drift apart in what counts as "the decision moved" or "admission is flat".
from experiments.measure_admission_mask import DECISION_FLOOR, ADMISSION_FLAT  # noqa: E402

# experiments/pre_registration_oracle_free_channels.md, committed before either output path existed.
PREREG_COMMIT = "4e7090b"

ARM, ATTACK = "krum", "committed_pixel"        # the non-floor cell; see the table above
# (family, rung value, d1 name). The "rung" here is not a dose: 0.0 is the identity (fedavg returns
# the update list unwrapped) and 1.0 is rfa applied once. rfa has no dial, which is why this arm is
# two arms plus two controls rather than a ladder, and why no monotone shape is predicted anywhere.
RUNGS_OF = [("oracleFree", 0.0, "fedavg"), ("oracleFree", 1.0, "rfa")]
D1_TRANSFORM = "rfa"

# The two bit-level checks train their own short run. 5 rounds rather than 2 because the first pass
# reported "controls select different clients in 0 of 2 rounds", which is a true statement about two
# early rounds of a fresh model and reads like "the two controls are the same computation" -- while the
# 15-row table above shows Krum's decision moving in over half of all rounds. The divergence receipt is
# that table; this field must not be readable as its contradiction.
CHECK_SEED, CHECK_ROUNDS = SEEDS[0], 5
EXACT_TOL = 1e-6

OUT = os.path.join(base, "results", "oracle_free_admission.json")
FROZEN = os.path.join(base, "results", "admission_measurement.json")
PREREG = os.path.join(base, "experiments", "pre_registration_oracle_free_channels.md")


def check_frozen():
    """Refuse to measure until the rule that types this arm is committed and unmodified."""
    if not os.path.exists(PREREG):
        sys.exit(f"REFUSING TO RUN: {PREREG} does not exist.")
    if PREREG_COMMIT is None:
        sys.exit("REFUSING TO RUN: the three-way eligibility rule is not frozen.\n"
                 f"  1. git commit {PREREG} alone\n"
                 "  2. set PREREG_COMMIT here to that hash.\n"
                 "An unfrozen rule can be rewritten around whatever this script prints, which is the "
                 "entire thing the freeze exists to prevent.")
    # PREREG_COMMIT is None is only half a guard: the moment the constant is set, the only test that
    # ever fires stops firing. So the hash is checked against the log and the tree against porcelain.
    import subprocess
    try:
        log = subprocess.run(["git", "log", "-1", "--format=%h", "--", PREREG], cwd=base,
                             capture_output=True, text=True, timeout=30).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", PREREG], cwd=base,
                               capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:                                    # a missing git is not a licence to run
        sys.exit(f"REFUSING TO RUN: cannot verify the freeze ({e}).")
    if not log.startswith(PREREG_COMMIT[:7]) and not PREREG_COMMIT.startswith(log[:7]):
        sys.exit(f"REFUSING TO RUN: {PREREG} was last committed at {log!r}, not {PREREG_COMMIT!r}. "
                 "Either the pre-registration was amended after the freeze or PREREG_COMMIT is "
                 "stale. Both are disqualifying.")
    if dirty:
        sys.exit(f"REFUSING TO RUN: {PREREG} has uncommitted modifications ({dirty!r}). The frozen "
                 "document and the document on disk must be the same bytes.")


def oracle_free_check():
    """THE claim this arm rests on: the transform cannot read adversary identity.

    Proved rather than read off the branch structure. apply_d1_transform is called twice on the same
    live update stack, once with the true adv_mask and once with adv_mask=None, and every tensor of
    every client must be bit-identical. If the rfa branch consulted adversary identity anywhere -- or
    if a future edit made it -- the two calls would differ and this check would fail.

    The dose families cannot pass this check by construction: they RAISE without adv_mask. That
    asymmetry is the point, and it is asserted here too, so the artifact records both halves.
    """
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(CHECK_SEED); np.random.seed(CHECK_SEED)
    cd, _, nc = get_federated_dataset("cifar10", 10, 0.5, CHECK_SEED)
    srv = FederatedServer(get_model("cifar_cnn", nc), dev)
    atk = get_attack("backdoor_pixel")
    adv = set(range(2))
    cl = [FederatedClient(i, atk.poison_dataset(cd[i]) if i in adv else cd[i], dev)
          for i in range(10)]
    rows, identical, dose_raises = [], True, None
    for rnd in range(CHECK_ROUNDS):
        pids = np.random.choice(10, 5, replace=False)
        ups = []
        for cid in pids:
            u = cl[cid].train(srv.global_model, 1, 0.01, 64)
            if cid in adv:
                u = atk.manipulate_update(u, srv.global_model)
            ups.append(u)
        am = [bool(cid in adv) for cid in pids]
        with_mask = apply_d1_transform(ups, D1_TRANSFORM, tau=5.0, dose_key=(CHECK_SEED, rnd),
                                      adv_mask=am)
        no_mask = apply_d1_transform(ups, D1_TRANSFORM, tau=5.0, dose_key=(CHECK_SEED, rnd),
                                     adv_mask=None)
        same = all(torch.equal(with_mask[i][k], no_mask[i][k])
                   for i in range(len(ups)) for k in ups[i])
        # The transform must also actually DO something: an inert transform would pass the
        # mask-independence check trivially and then produce a null result that looks like a finding.
        moved = not all(torch.equal(with_mask[i][k], ups[i][k])
                        for i in range(len(ups)) for k in ups[i])
        ratios = []
        for i in range(len(ups)):
            a = torch.cat([ups[i][k].flatten().detach().cpu().double() for k in ups[i]])
            b = torch.cat([with_mask[i][k].flatten().detach().cpu().double() for k in with_mask[i]])
            ratios.append(float(b.norm() / a.norm().clamp(min=1e-300)))
        identical = identical and same and moved
        rows.append({"round": rnd, "mask_independent": same, "transform_moved_the_stack": moved,
                     "per_client_norm_ratio_float64": ratios,
                     "n_adv_in_round": int(sum(am))})
        if dose_raises is None:
            try:
                apply_d1_transform(ups, "doseS_kappa2.0", tau=5.0, dose_key=(CHECK_SEED, rnd),
                                   adv_mask=None)
                dose_raises = False
            except ValueError:
                dose_raises = True
        srv.apply_update(srv.aggregate(ups))
    return {"d1": D1_TRANSFORM, "rounds": CHECK_ROUNDS, "seed": CHECK_SEED,
            "mask_independent_and_non_inert": bool(identical),
            "published_dose_raises_without_adv_mask": dose_raises,
            "per_round": rows,
            "claim": "apply_d1_transform('rfa', adv_mask=None) is bit-identical to the same call "
                     "with the true adv_mask on live stacks, and is not the identity. The published "
                     "doseS transform raises without adv_mask. So this arm's transform is "
                     "oracle-free through the shipped code path and the published one is not."}


def control_check():
    """The two controls, asserted at the bit level against independently built aggregates.

    emit_only must SCORE the raw stack, so its selection is the identity arm's selection and neither
    Krum's decision nor its admitted adversarial mass can move. score_only must EMIT the raw selected
    update, so the magnitude channel is closed. Both are claims about the shipped code path, and the
    standing failure mode here is a manipulation hook that is never called while the run passes
    silently, so each is checked with torch.equal against an aggregate this function builds itself
    from the selection it expects -- not against generic_compose's own reasoning.

    Also recorded per round: whether the two controls select DIFFERENT clients. If they never do, the
    two ladders are the same computation and the arm has no factorial.
    """
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(CHECK_SEED); np.random.seed(CHECK_SEED)
    cd, _, nc = get_federated_dataset("cifar10", 10, 0.5, CHECK_SEED)
    srv = FederatedServer(get_model("cifar_cnn", nc), dev)
    atk = get_attack("backdoor_pixel")
    adv = set(range(2))
    cl = [FederatedClient(i, atk.poison_dataset(cd[i]) if i in adv else cd[i], dev)
          for i in range(10)]
    rows, emit_ok, score_ok = [], True, True
    for rnd in range(CHECK_ROUNDS):
        pids = np.random.choice(10, 5, replace=False)
        ups = []
        for cid in pids:
            u = cl[cid].train(srv.global_model, 1, 0.01, 64)
            if cid in adv:
                u = atk.manipulate_update(u, srv.global_model)
            ups.append(u)
        t = apply_d1_transform(ups, D1_TRANSFORM, tau=5.0, dose_key=(CHECK_SEED, rnd), adv_mask=None)
        sel_raw, _ = krum_selection(flatten(ups), False)
        sel_tr, _ = krum_selection(flatten(t), False)
        # Built here, from the selection each control is DEFINED to use.
        want_emit = srv.aggregate([t[sel_raw]], method="fedavg")
        want_score = srv.aggregate([ups[sel_tr]], method="fedavg")
        got_emit = generic_compose(srv, ups, D1_TRANSFORM, "krum", tau=5.0, dose_key=None,
                                  adv_mask=None, emit_only=True)
        got_score = generic_compose(srv, ups, D1_TRANSFORM, "krum", tau=5.0, dose_key=None,
                                   adv_mask=None, score_only=True)
        e = all(torch.equal(want_emit[k], got_emit[k]) for k in want_emit)
        s = all(torch.equal(want_score[k], got_score[k]) for k in want_score)
        emit_ok, score_ok = emit_ok and e, score_ok and s
        rows.append({"round": rnd, "sel_raw": int(sel_raw), "sel_transformed": int(sel_tr),
                     "controls_select_different_clients": bool(sel_raw != sel_tr),
                     "emit_only_scores_the_raw_stack": bool(e),
                     "score_only_emits_the_raw_update": bool(s)})
        srv.apply_update(srv.aggregate(ups))
    return {"rounds": CHECK_ROUNDS, "seed": CHECK_SEED,
            "emit_only_decision_pinned_bit_exact": bool(emit_ok),
            "score_only_emits_raw_selected_update_bit_exact": bool(score_ok),
            "n_rounds_where_controls_diverge": sum(r["controls_select_different_clients"]
                                                   for r in rows),
            "divergence_receipt_is_the_main_table_not_this_field":
                "This function trains its own short run to assert BIT-EQUALITY of the two controls' "
                f"aggregates; its {CHECK_ROUNDS} rounds are too few to estimate how often the two "
                "selections differ. The rate at which rfa moves Krum's decision is the "
                "oracleFree|krum|1.0|decision entry of `summary`, measured over 5 seeds x 3 rounds. A "
                "zero here is not evidence that the two controls are the same computation.",
            "per_round": rows,
            "claim": "emit_only's aggregate equals fedavg over the RAW-stack selection's TRANSFORMED "
                     "update, and score_only's equals fedavg over the TRANSFORMED-stack selection's "
                     "RAW update, both bit-exactly. So emit_only's decision is pinned to the "
                     "identity arm's by construction and score_only closes the magnitude channel."}


def baseline_admission(rows):
    """The non-floor receipt, recomputed from this script's own rows rather than quoted.

    base_krum_admits_adv is a property of the RAW stack, so it is identical at every rung; it is read
    at the identity rung alone so no adversary round is counted twice. Restricted to rounds that
    actually contain an adversary, which is the convention build_channel_table.py's cross-check uses.
    """
    sub = [r for r in rows if abs(r["rung"]) < 1e-12 and r["n_adv_in_round"] > 0]
    vals = [float(r["base_krum_admits_adv"]) for r in sub]
    return {"attack": ATTACK, "n_adversary_rounds": len(vals),
            "n_nonzero": int(sum(1 for v in vals if v > 0.0)),
            "mean_base_krum_admits_adv": float(np.mean(vals)) if vals else float("nan"),
            "per_round": vals}


def frozen_cross_check():
    """Apply this arm's rule to the PUBLISHED configuration before condemning anything with it.

    Round 73's heterogeneity gate condemned three new concentrations at a tolerance the
    already-published concentration also failed, and the verdict it emitted asserted a mechanism the
    measurement did not support. So the published doseS/krum arm is scored by the same rule first. If
    the rule as implemented would type the published arm INELIGIBLE, the instrument is wrong and this
    arm's verdict is INDETERMINATE rather than a condemnation.
    """
    if not os.path.exists(FROZEN):
        return {"available": False}
    fz = json.load(open(FROZEN)).get("summary", {})
    d = fz.get(f"doseS|{ARM}|{KAPPAS[-1]}|decision")
    a = fz.get(f"doseS|{ARM}|{KAPPAS[-1]}|admission")
    if d is None or a is None:
        return {"available": False}
    would = ("INELIGIBLE" if d < DECISION_FLOOR
             else "GENERALIZATION" if abs(a) <= ADMISSION_FLAT else "ADMISSION")
    return {"available": True, "source": "results/admission_measurement.json",
            "published_arm": f"doseS kappa={KAPPAS[-1]} -> {ARM} under {ARM_ATTACK[ARM]}",
            "decision_change": d, "admission_change": a,
            "would_be_typed": would, "rule_condemns_the_published_arm": bool(would == "INELIGIBLE")}


def main():
    check_frozen()
    print("=== ORACLE-FREE channel measurement: no ASR, no per-rung training ===")
    print(f"    rule frozen at {PREREG_COMMIT} "
          "(experiments/pre_registration_oracle_free_channels.md)")
    print(f"    host cell: {ARM} under {ATTACK.replace('committed_', '')}, "
          f"upstream d1 = {D1_TRANSFORM} (Weiszfeld weights from the stack alone)")
    print(f"    {len(SEEDS)} seeds x {ROUNDS} live rounds x {len(RUNGS_OF)} rungs, "
          "both rungs on the same raw updates")
    print(f"    eligibility rule imported from measure_admission_mask.py: "
          f"decision_floor={DECISION_FLOOR}, admission_flat={ADMISSION_FLAT}")
    print("    PROSPECTIVE: this runs before any oracle-free ASR exists, and it can stop the "
          "16-hour ladder.\n", flush=True)

    all_rows = measure(ATTACK, rungs=RUNGS_OF)
    # Non-finite rounds are excluded rather than averaged over, the same way its two siblings do it
    # (measure_admission_femnist.py:81, measure_admission_mask.py:177), because averaging a boolean
    # "did the decision change" over a round whose distances are all nan silently records "no
    # change" -- a null that looks like a measurement. Under the pixel backdoor there is no norm
    # blowup, so this is expected to drop nothing here; it is kept so that a surprise is visible
    # rather than absorbed.
    rows = [r for r in all_rows if math.isfinite(r["c_mean"])]
    dropped = sorted({(r["seed"], r["round"]) for r in all_rows if not math.isfinite(r["c_mean"])})
    print(f"\n=== NON-FINITE ROUNDS EXCLUDED: {len(dropped)} of {len(SEEDS) * ROUNDS} "
          "(seed, round) ===")
    if dropped:
        print(f"    {dropped}")

    summary = {}
    print("\n=== DECISION CHANGE / ADMISSION CHANGE UNDER AN ORACLE-FREE TRANSFORM ===")
    print("  (left = statistic-level disturbance, the quantity a decision-level reading uses;")
    print("   right = the adversarial mass actually admitted, which is what the mechanism claim is")
    print("   about. rung 0.0 is the identity: fedavg returns the stack unwrapped, so BOTH numbers")
    print("   must be exactly 0.000 there or this measurement is broken.)\n")
    print(f"  {'arm':26s} " + " ".join(f"{('rung=' + str(v)):>15s}" for _, v, _ in RUNGS_OF))
    for arm in ("krum", "reputation", "cos_krum", "coord_median"):
        cols = []
        for _, v, _ in RUNGS_OF:
            sub = [r for r in rows if abs(r["rung"] - v) < 1e-12]
            d, a = rate(sub, DECISION_KEY[arm]), rate(sub, ADMISSION_KEY[arm])
            summary[f"oracleFree|{arm}|{v}|decision"] = d
            summary[f"oracleFree|{arm}|{v}|admission"] = a
            summary[f"oracleFree|{arm}|{v}|agg_disp"] = rate(sub, f"agg_disp_{arm}")
            cols.append(f"{d:.3f}/{a:.3f}")
        print(f"  {arm:26s} " + " ".join(f"{c:>15s}" for c in cols)
              + ("   <- the host arm" if arm == ARM else ""))
    print("\n  Every aggregator is measured, not only the host, because a transform that moves one")
    print("  aggregator's decision and not another's is a fact about the transform worth recording.")
    print("  Only the host arm's numbers type this arm; the rest are diagnostics.")

    print("\n=== AGGREGATE DISPLACEMENT ||agg(T(U)) - agg(U)|| / ||agg(U)|| ===")
    print(f"  {'arm':26s} " + " ".join(f"{('rung=' + str(v)):>15s}" for _, v, _ in RUNGS_OF))
    for arm in ("krum", "reputation", "cos_krum", "coord_median"):
        print(f"  {arm:26s} " + " ".join(
            f"{summary[f'oracleFree|{arm}|{v}|agg_disp']:>15.3f}" for _, v, _ in RUNGS_OF))

    # --- The identity rung must be the exact identity, tested with == and not a tolerance ---
    ident_sub = [r for r in rows if abs(r["rung"]) < 1e-12]
    id_dec = [r[DECISION_KEY[ARM]] for r in ident_sub]
    id_adm = [r[ADMISSION_KEY[ARM]] for r in ident_sub]
    id_c = [(r["c_adv_min"], r["c_adv_max"], r["c_mean"], r["rho_realized"]) for r in ident_sub]
    ident_exact = (all(float(x) == 0.0 for x in id_dec) and all(float(x) == 0.0 for x in id_adm))
    print("\n=== IDENTITY RUNG EXACTNESS (fedavg returns the stack unwrapped) ===")
    print(f"  decision changes over {len(id_dec)} rows: {sum(float(x) != 0.0 for x in id_dec)} "
          f"nonzero;  admission changes: {sum(float(x) != 0.0 for x in id_adm)} nonzero")
    print(f"  rho_realized at the identity: "
          f"[{min(c[3] for c in id_c):.12f}, {max(c[3] for c in id_c):.12f}]")
    print(f"  {'EXACT: the measurement reproduces its own baseline.' if ident_exact else 'NOT EXACT -- the measurement is broken and no number from it may be used.'}")

    # --- The non-floor receipt, recomputed rather than quoted ---
    ba = baseline_admission(rows)
    print("\n=== THE BASELINE IS NOT A FLOOR (recomputed from this script's own rows) ===")
    print(f"  {ARM} under {ATTACK.replace('committed_', '')}: baseline admitted adversarial mass "
          f"{ba['mean_base_krum_admits_adv']:.4f}, nonzero in {ba['n_nonzero']} of "
          f"{ba['n_adversary_rounds']} adversary rounds")
    print("  The frozen committed_scaling row of the same aggregator is 0.0000 in 0 of 12, which is")
    print("  the floor the review objects to. This cell is off that floor by measurement.")

    # --- Oracle-freeness and the two controls, through the shipped code path ---
    print("\n=== ORACLE-FREENESS, PROVED ON LIVE STACKS ===", flush=True)
    ofc = oracle_free_check()
    print(f"  apply_d1_transform('{D1_TRANSFORM}') with adv_mask=None vs the true mask, bit-identical "
          f"and non-inert: {ofc['mask_independent_and_non_inert']}")
    print(f"  the published doseS transform raises without adv_mask: "
          f"{ofc['published_dose_raises_without_adv_mask']}")
    print("\n=== THE TWO CONTROLS, ASSERTED WITH torch.equal ===", flush=True)
    cc = control_check()
    print(f"  emit_only scores the RAW stack (decision pinned to the identity arm): "
          f"{cc['emit_only_decision_pinned_bit_exact']}")
    print(f"  score_only emits the RAW selected update (magnitude channel closed): "
          f"{cc['score_only_emits_raw_selected_update_bit_exact']}")
    print(f"  rounds where the two controls select different clients: "
          f"{cc['n_rounds_where_controls_diverge']} of {cc['rounds']}")

    # --- The published configuration is scored by this rule BEFORE anything is condemned ---
    fx = frozen_cross_check()
    print("\n=== THE RULE, APPLIED TO THE PUBLISHED ARM FIRST ===")
    if fx.get("available"):
        print(f"  {fx['published_arm']}: decision {fx['decision_change']:.3f}, "
              f"admission {fx['admission_change']:.3f}  ->  would be typed {fx['would_be_typed']}")
        print("  A rule that condemns the already-published arm cannot discriminate this one.")
    else:
        print("  results/admission_measurement.json not readable; cross-check UNAVAILABLE.")

    # --- The three-way eligibility verdict, on the rule frozen before these numbers were read ---
    d_hi = summary[f"oracleFree|{ARM}|1.0|decision"]
    a_hi = summary[f"oracleFree|{ARM}|1.0|admission"]
    blockers = []
    if not ident_exact:
        blockers.append("the identity rung is not the exact identity")
    if not ofc["mask_independent_and_non_inert"]:
        blockers.append("the transform is not mask-independent, or is inert")
    if not ofc["published_dose_raises_without_adv_mask"]:
        blockers.append("the published dose transform does not raise without adv_mask, so the "
                        "oracle-free contrast this arm rests on is not what it claims")
    if not (cc["emit_only_decision_pinned_bit_exact"]
            and cc["score_only_emits_raw_selected_update_bit_exact"]):
        blockers.append("a control does not do what its pre-registration says, at the bit level")
    if fx.get("rule_condemns_the_published_arm"):
        blockers.append("the eligibility rule as implemented condemns the published arm")

    print("\n=== ELIGIBILITY AND ARM TYPE "
          "(rule frozen in the pre-registration, constants imported) ===")
    print(f"  {D1_TRANSFORM} -> {ARM} under {ATTACK.replace('committed_', '')}: "
          f"decision change {d_hi:.3f}, admission change {a_hi:.3f}")
    if blockers:
        arm_type = "INDETERMINATE"
        verdict = ("INDETERMINATE: a structural check failed, so no eligibility verdict is emitted "
                   "and the ladder does not run. " + "; ".join(blockers) + ".")
    elif d_hi < DECISION_FLOOR:
        arm_type = "INELIGIBLE"
        verdict = (f"INELIGIBLE: an oracle-free rfa barely disturbs {ARM}'s decision "
                   f"(< {DECISION_FLOOR}), so a flat ASR curve would carry no information about "
                   "statistic preservation. This is the cos_krum failure mode and the 60-run ladder "
                   "must not be run as a test of the negative.")
    elif abs(a_hi) <= ADMISSION_FLAT:
        arm_type = "GENERALIZATION"
        verdict = ("ELIGIBLE AS AN ORACLE-FREE GENERALIZATION TEST: the decision is disturbed and "
                   "the admitted adversarial mass is not, which is the premise the flagship arm "
                   "rests on -- now with a transform that reads no adversary identity, on a cell "
                   "whose baseline admission is not a floor. The pre-registered predictions are in "
                   "force: leg 1 flat inside the margin, leg 2 nonzero.")
    else:
        arm_type = "ADMISSION"
        verdict = ("ELIGIBLE AS AN ADMISSION TEST, NOT AS AN ORACLE-FREE GENERALIZATION: an "
                   "oracle-free rfa moves the admitted adversarial mass, so leg 1's premise fails "
                   "and the review's leg 1 is UNANSWERED by this arm. It tests the paper's own "
                   "reading instead: if admission governs suppression, ASR must move WITH admission "
                   "here. It may not be reported as an oracle-free analogue of the flagship arm.")
    print(f"  {verdict}")

    ndegen = sum(r["degenerate"] for r in rows)
    print(f"\n  degenerate rung-rounds (0 adversaries or <2 benign): {ndegen}/{len(rows)} "
          f"({100.0 * ndegen / max(len(rows), 1):.1f}%)")

    json.dump({
        "description": "Oracle-free channel measurement: prospective eligibility check and arm-type "
                       f"selection for the {D1_TRANSFORM} -> {ARM} arm under {ATTACK}, with the "
                       "paper's own score_only/emit_only controls. No ASR. Freezes nothing. "
                       "results/admission_measurement.json is read for the frozen comparison and "
                       "never written.",
        "prereg_commit": PREREG_COMMIT,
        "prereg": "experiments/pre_registration_oracle_free_channels.md",
        "family": "oracleFree", "rungs": [{"family": f, "rung": v, "d1": d} for f, v, d in RUNGS_OF],
        "seeds": SEEDS, "rounds_per_seed": ROUNDS, "dataset": "cifar10", "model": "cifar_cnn",
        "arm": ARM, "attack": ATTACK, "d1": D1_TRANSFORM, "arm_attack": ARM_ATTACK,
        "rules": {"decision_floor": DECISION_FLOOR, "admission_flat": ADMISSION_FLAT,
                  "share_tol": SHARE_TOL, "exact_tol": EXACT_TOL,
                  "rule_source": "experiments/measure_admission_mask.py (imported, not restated)"},
        "eligibility": {"decision_change": d_hi, "admission_change": a_hi,
                        "arm_type": arm_type, "verdict": verdict,
                        "structural_blockers": blockers},
        "identity_rung_exactness": {"n_rows": len(ident_sub),
                                    "n_decision_nonzero": sum(float(x) != 0.0 for x in id_dec),
                                    "n_admission_nonzero": sum(float(x) != 0.0 for x in id_adm),
                                    "exact": bool(ident_exact)},
        "baseline_admission_not_a_floor": ba,
        "oracle_free_check": ofc,
        "control_check": cc,
        "published_arm_cross_check": fx,
        "nonfinite_rounds_excluded": [list(x) for x in dropped],
        "per_round": rows,
        "per_round_including_nonfinite": all_rows,
        "summary": summary,
        "n_degenerate_rung_rounds": int(ndegen),
    }, open(OUT, "w"), indent=1)
    print(f"\nWrote {OUT}")
    print("results/admission_measurement.json was not modified.")
    print(f"\nArm type: {arm_type}. "
          + ("Do NOT run experiments/run_oracle_free_channels.py."
             if arm_type in ("INELIGIBLE", "INDETERMINATE")
             else "experiments/run_oracle_free_channels.py may now run."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
