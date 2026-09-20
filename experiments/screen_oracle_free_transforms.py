"""Screen every deployed oracle-free d1 family for a clean statistic-channel probe.

WHAT THIS IS. App. D.7 measured the statistic channel without adversary identity and then disclosed,
post hoc, that its transform does not hold the adversarial coefficient share fixed: rfa moves the
share in 12 of 12 adversary rounds, by up to 0.0859, against a frozen tolerance of 1e-6 that Mode S
holds to 3.5e-07. So that arm moves two channels at once and its refutation does not transfer to the
published Mode S result. This script asks the prior question App. D.7 never asked: across the WHOLE
deployed oracle-free class, is there a family that moves the decision while leaving the coefficient
share fixed to the paper's own tolerance?

WHAT THIS IS NOT. This is a screen that selects an instrument, not a test that returns a verdict on
any hypothesis about suppression. It measures no ASR and trains no ladder. Nothing here is scored
against a prediction, and no outcome of this screen is reported as evidence for or against the
paper's reading. Its only job is to decide which transform the decomposition arm should use, and to
record that decision's basis before any ASR exists. That is the same order App. D.7's premise check
ran in -- premise first, outcome second -- and the same order measure_admission_mask.py uses to
decide whether Mode M is eligible at all.

WHY IT NEEDS NO FREEZE OF ITS OWN, AND WHAT WOULD BE WRONG IF IT DID. A screen whose thresholds are
chosen after its numbers are read chooses the instrument to suit the answer. This screen cannot do
that, because it invents no threshold: all three are imported from constants already frozen and
already used to type published arms.

    DECISION_FLOOR = 0.10   measure_admission_mask.py -- below this there is no disturbance to test
    ADMISSION_FLAT = 0.05   measure_admission_mask.py -- above this the arm is an admission test
    SHARE_TOL      = 1e-6   measure_admission.py     -- Mode S's own share-constancy tolerance

Importing the third is the substantive choice. App. D.7's pre-registration recorded SHARE_TOL in its
rules block and its runner recorded adv_coeff_share every round, and the gate was still not written;
that omission is the defect App. D.7's own text and letter own. Writing the gate here with the same
constant, rather than a laxer one picked to let a family through, is what makes a PASS meaningful and
an empty result reportable. A new tolerance appears nowhere in this file.

THE EXPECTED OUTCOME IS EMPTY, AND EMPTY IS THE RESULT. App. D.6 proves that no member of the
positive per-client rescaling class is simultaneously share-neutral and informative: oracle-freeness
forces share preservation to hold for EVERY adversary set, which forces c constant, which leaves
Krum's argmin invariant. So this screen is that impossibility's quantitative shadow, measured across
the deployed class rather than argued for one member. If every family fails, that is the finding, it
costs one pass instead of a ladder, and it is what licenses constructing a boundary transform
instead. The per-family graded quantities are reported either way, including the tolerance at which
each family WOULD pass, because a screen that prints only PASS/FAIL is not informative about how far
from neutral each family sits.

THE CELL. Krum under the committed pixel backdoor, which is App. D.7's cell rather than
ARM_ATTACK's default pairing of krum with committed_scaling. The reason is the admission floor:
under committed_scaling this Krum admits baseline adversarial mass 0.000 in 0 of 12 adversary
rounds, so admission has nowhere to fall and its flatness is a ceiling artifact rather than a
measurement; under committed_pixel it admits 0.333, nonzero in 4 of 12. The floor belongs to the
attack, not to the aggregator, which is why the cell is chosen by the attack.

OUTPUT. results/oracle_free_screen.json, a path that does not exist before this script runs. No
existing results directory is written, no existing runner or pre-registration is edited, and
measure_admission.py is reused through its own `rungs=` parameter rather than modified -- the same
hook measure_admission_mask.py and measure_admission_oracle_free.py already use, so the default call
still reproduces results/admission_measurement.json exactly.

Run from the repository root:  PYTHONPATH=. python3 -m experiments.screen_oracle_free_transforms
"""

import json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
from fl_core import get_federated_dataset, get_model, FederatedServer, FederatedClient  # noqa: E402
from attacks import get_attack  # noqa: E402
from experiments.run_all_compositions import apply_d1_transform  # noqa: E402
# Single-sourced from the frozen CIFAR-10 measurement: the same measure(), the same decision and
# admission field names, the same rate(), the same seeds and live-round count, the same share
# tolerance. Only the rung list is new, and it is passed in rather than edited into that module.
from experiments.measure_admission import (  # noqa: E402
    measure, rate, SEEDS, ROUNDS, DECISION_KEY, ADMISSION_KEY, SHARE_TOL,
)
# The eligibility rule is Mode M's rule, imported rather than re-typed, so the screen and the two
# prospective measurements cannot drift apart in what counts as "the decision moved" or "admission
# is flat".
from experiments.measure_admission_mask import DECISION_FLOOR, ADMISSION_FLAT  # noqa: E402

ARM, ATTACK = "krum", "committed_pixel"        # App. D.7's non-floor cell; see the docstring

# Every oracle-free d1 family apply_d1_transform actually implements as a per-client rescaling.
#
# Excluded, with the reason in each case rather than by omission:
#   dose / doseS / doseA / doseM  RAISE without adv_mask. They are the oracle-bound families this
#                                 screen exists to find a replacement for, and that asymmetry is
#                                 asserted below rather than assumed.
#   fltrust                       raises unless the caller supplies a server carrying a
#                                 clean_holdout_dataset, which measure() does not pass. It is
#                                 therefore unreachable through this loop. Not a silent drop: it is
#                                 recorded in the artifact as unreachable-by-construction, and it is
#                                 the one family whose d2 form cancels a positive rescaling on BOTH
#                                 channels, which is why the appendix discusses it separately.
#   trimmed_mean / coord_median / krum / multi_krum
#                                 pass through unchanged as d1 -- they emit one aggregate, not a
#                                 per-client stack -- so as upstream stages they are the identity
#                                 and cannot move any decision.
FAMILIES = ["norm_clip", "reputation", "foolsgold", "rfa"]

# (family, rung value, d1 name). The rung value is not a dose: 0.0 is the identity and 1.0 is the
# family applied once. None of these families has a dial, which is why this is a screen over
# families rather than a ladder, and why no monotone shape is predicted or read anywhere. Rows are
# grouped by their `d1` field, not by (family, rung), since every family shares rung 1.0.
RUNGS_SCREEN = ([("ofScreen", 0.0, "fedavg")]
                + [("ofScreen", 1.0, f) for f in FAMILIES])

CHECK_SEED, CHECK_ROUNDS = SEEDS[0], 5
OUT = os.path.join(base, "results", "oracle_free_screen.json")


def oracle_free_check_all():
    """Prove per family that the transform cannot read adversary identity.

    Generalizes measure_admission_oracle_free.oracle_free_check over the screened families. Each
    family's apply_d1_transform is called twice on the same live stack, once with the true adv_mask
    and once with adv_mask=None, and every tensor of every client must be bit-identical. A family
    that consulted adversary identity anywhere -- or that a future edit made consult it -- fails
    here rather than being trusted on its branch structure.

    Two failure modes are separated, because they need opposite treatment:

      mask_independent      the transform ignores identity. Required.
      transform_moved       the transform is not the identity on this stack. Also required: an inert
                            transform passes the identity check trivially and then produces a null
                            result that reads like a finding. norm_clip at tau=5.0 is the live
                            candidate for inertness, since it rescales nothing when every client
                            norm is already below tau, so this is measured rather than assumed.

    The published dose transform's REFUSAL without adv_mask is asserted in the same loop, so both
    halves of the oracle-free/oracle-bound contrast are witnessed by one measurement.
    """
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(CHECK_SEED); np.random.seed(CHECK_SEED)
    cd, _, nc = get_federated_dataset("cifar10", 10, 0.5, CHECK_SEED)
    srv = FederatedServer(get_model("cifar_cnn", nc), dev)
    atk = get_attack("backdoor_pixel")
    adv = set(range(2))
    cl = [FederatedClient(i, atk.poison_dataset(cd[i]) if i in adv else cd[i], dev)
          for i in range(10)]
    per_family = {f: {"rounds": [], "mask_independent": True, "moved_every_round": True}
                  for f in FAMILIES}
    dose_raises = None
    for rnd in range(CHECK_ROUNDS):
        pids = np.random.choice(10, 5, replace=False)
        ups = []
        for cid in pids:
            u = cl[cid].train(srv.global_model, 1, 0.01, 64)
            if cid in adv:
                u = atk.manipulate_update(u, srv.global_model)
            ups.append(u)
        am = [bool(cid in adv) for cid in pids]
        for fam in FAMILIES:
            with_mask = apply_d1_transform(ups, fam, tau=5.0, dose_key=(CHECK_SEED, rnd),
                                           adv_mask=am)
            no_mask = apply_d1_transform(ups, fam, tau=5.0, dose_key=(CHECK_SEED, rnd),
                                         adv_mask=None)
            same = all(torch.equal(with_mask[i][k], no_mask[i][k])
                       for i in range(len(ups)) for k in ups[i])
            moved = not all(torch.equal(with_mask[i][k], ups[i][k])
                            for i in range(len(ups)) for k in ups[i])
            ratios = []
            for i in range(len(ups)):
                a = torch.cat([ups[i][k].flatten().detach().cpu().double() for k in ups[i]])
                b = torch.cat([with_mask[i][k].flatten().detach().cpu().double()
                               for k in with_mask[i]])
                ratios.append(float(b.norm() / a.norm().clamp(min=1e-300)))
            d = per_family[fam]
            d["mask_independent"] = bool(d["mask_independent"] and same)
            d["moved_every_round"] = bool(d["moved_every_round"] and moved)
            d["rounds"].append({"round": rnd, "mask_independent": same,
                                "transform_moved_the_stack": moved,
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
    # fltrust is recorded as unreachable rather than dropped in silence, and the claim is asserted
    # rather than quoted from the branch comment.
    try:
        apply_d1_transform(ups, "fltrust", tau=5.0, dose_key=(CHECK_SEED, 0), adv_mask=None)
        fltrust_raises = False
    except ValueError:
        fltrust_raises = True
    return {"rounds": CHECK_ROUNDS, "seed": CHECK_SEED,
            "per_family": per_family,
            "published_dose_raises_without_adv_mask": dose_raises,
            "fltrust_as_d1_raises_without_clean_holdout": fltrust_raises,
            "claim": "For each screened family, apply_d1_transform(fam, adv_mask=None) is "
                     "bit-identical to the same call with the true adv_mask on live stacks, and is "
                     "not the identity. The published doseS transform raises without adv_mask, and "
                     "fltrust as d1 raises without a clean holdout. So every screened family is "
                     "oracle-free through the shipped code path and the published dose families "
                     "are not."}


def clip_binding_check():
    """Record whether tau=5.0 binds at all on this cell, so an inert norm_clip is not misread.

    norm_clip's coefficient is c_i = min(1, tau/||u_i||), which is EXACTLY 1 for every client whose
    norm is already below tau. On this cell it is: the screen therefore finds norm_clip bit-identical
    to the identity, share-neutral to 0.0 and uninformative, which is not a property of norm_clip but
    of tau=5.0 against these update norms. Without this measurement the artifact would read as "the
    clip is inert", a claim about the defense; with it the claim is "the clip does not bind here", a
    claim about a hyperparameter at one operating point. tau=5.0 is the value the composition suite
    and measure() both pass, so it is the deployed operating point rather than one chosen here.

    This is also why the screen's empty result is not an artifact of a badly chosen tau. Lowering tau
    until the clip binds would make it informative AND move the share, which is the trade the
    impossibility describes; it would not produce an eligible family.
    """
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(CHECK_SEED); np.random.seed(CHECK_SEED)
    cd, _, nc = get_federated_dataset("cifar10", 10, 0.5, CHECK_SEED)
    srv = FederatedServer(get_model("cifar_cnn", nc), dev)
    atk = get_attack("backdoor_pixel")
    adv = set(range(2))
    cl = [FederatedClient(i, atk.poison_dataset(cd[i]) if i in adv else cd[i], dev)
          for i in range(10)]
    rows, any_bind = [], False
    for rnd in range(CHECK_ROUNDS):
        pids = np.random.choice(10, 5, replace=False)
        ups = []
        for cid in pids:
            u = cl[cid].train(srv.global_model, 1, 0.01, 64)
            if cid in adv:
                u = atk.manipulate_update(u, srv.global_model)
            ups.append(u)
        norms = [float(torch.cat([u[k].flatten().float() for k in u]).norm().item()) for u in ups]
        binds = [bool(nm > 5.0) for nm in norms]
        any_bind = any_bind or any(binds)
        rows.append({"round": rnd, "tau": 5.0, "client_norms": norms,
                     "n_clients_above_tau": int(sum(binds)),
                     "max_norm": max(norms), "min_norm": min(norms)})
        srv.apply_update(srv.aggregate(ups))
    return {"tau": 5.0, "rounds": CHECK_ROUNDS, "clip_binds_on_any_client": bool(any_bind),
            "max_client_norm_seen": float(max(r["max_norm"] for r in rows)),
            "per_round": rows,
            "claim": ("norm_clip's inertness on this cell is a fact about tau=5.0 against these "
                      "update norms, not a property of the defense. Every client norm is below tau, "
                      "so c_i == 1 exactly for every client and the transform is the identity.")}


def share_stats(sub):
    """Per-family coefficient-share movement against the identity rung, adversary rounds only.

    The share is nan on a round that sampled no adversary, and Mode S's own tolerance is about
    adversary rounds, so those rounds are excluded here rather than averaged as zeros. The identity
    rung's share is exactly n_adv/K because FedAvg gives every client coefficient 1, which is why
    the comparison is against the measured identity row rather than against that closed form.
    """
    ident = {(r["seed"], r["round"]): r for r in sub if r["d1"] == "fedavg"}
    out = {}
    for fam in FAMILIES:
        deltas, rows = [], []
        for r in sub:
            if r["d1"] != fam:
                continue
            key = (r["seed"], r["round"])
            if key not in ident or r["n_adv_in_round"] == 0:
                continue
            a, b = ident[key]["adv_coeff_share"], r["adv_coeff_share"]
            if any(np.isnan(float(x)) for x in (a, b)):
                continue
            deltas.append(float(b) - float(a))
            rows.append({"seed": r["seed"], "round": r["round"],
                         "n_adv_in_round": r["n_adv_in_round"],
                         "share_identity": float(a), "share_family": float(b),
                         "delta": float(b) - float(a),
                         "rho_realized": float(r["rho_realized"]),
                         "decision_moved": bool(r[DECISION_KEY[ARM]])})
        n_moved = sum(1 for d in deltas if abs(d) > SHARE_TOL)
        out[fam] = {
            "n_adversary_rounds": len(deltas),
            "n_rounds_share_moved_beyond_tol": n_moved,
            "max_abs_delta": float(max((abs(d) for d in deltas), default=float("nan"))),
            "mean_delta": float(np.mean(deltas)) if deltas else float("nan"),
            "min_delta": float(min(deltas)) if deltas else float("nan"),
            "per_round": rows,
        }
    return out


def main():
    print("=== Screening the deployed oracle-free d1 class for a clean statistic probe ===")
    print(f"cell: {ARM} under {ATTACK}   families: {', '.join(FAMILIES)}")
    print(f"thresholds, all imported: DECISION_FLOOR={DECISION_FLOOR} ADMISSION_FLAT="
          f"{ADMISSION_FLAT} SHARE_TOL={SHARE_TOL:g}")
    if os.path.exists(OUT):
        print(f"NOTE: {OUT} exists and will be rewritten by this run.")

    print("\n-- oracle-freeness, asserted per family on live stacks --")
    ofc = oracle_free_check_all()
    for fam in FAMILIES:
        d = ofc["per_family"][fam]
        print(f"   {fam:12s} mask_independent={d['mask_independent']}  "
              f"moved_every_round={d['moved_every_round']}")
    print(f"   published doseS raises without adv_mask : "
          f"{ofc['published_dose_raises_without_adv_mask']}")
    print(f"   fltrust as d1 raises without holdout    : "
          f"{ofc['fltrust_as_d1_raises_without_clean_holdout']}")

    print("\n-- does tau=5.0 bind at all on this cell? (so an inert clip is not misread) --")
    clip = clip_binding_check()
    print(f"   clip binds on any client: {clip['clip_binds_on_any_client']}   "
          f"max client norm seen {clip['max_client_norm_seen']:.4f} against tau 5.0")

    print(f"\n-- channel measurement, {len(SEEDS)} seeds x {ROUNDS} live rounds, one training pass "
          "shared by every rung --")
    rows = measure(ATTACK, seeds=SEEDS, rounds=ROUNDS, rungs=RUNGS_SCREEN)
    shares = share_stats(rows)

    dec_key, adm_key = DECISION_KEY[ARM], ADMISSION_KEY[ARM]
    table, eligible = {}, []
    for fam in FAMILIES:
        sub = [r for r in rows if r["d1"] == fam]
        dec = rate(sub, dec_key)
        adm = rate(sub, adm_key)
        sh = shares[fam]
        of = ofc["per_family"][fam]
        # Every leg is reported, and the verdict is the conjunction. A family that fails one leg is
        # not thereby uninteresting: the graded quantities below are what B2's construction is
        # calibrated against.
        legs = {
            "oracle_free": bool(of["mask_independent"]),
            "non_inert": bool(of["moved_every_round"]),
            "decision_moves": bool(dec >= DECISION_FLOOR),
            "admission_flat": bool(adm <= ADMISSION_FLAT),
            "share_neutral": bool(sh["n_rounds_share_moved_beyond_tol"] == 0),
        }
        ok = all(legs.values())
        if ok:
            eligible.append(fam)
        # The tolerance at which this family WOULD be share-neutral. Reported because the screen's
        # question is quantitative: App. D.6 forbids EXACT neutrality, and how far each deployed
        # family sits from it is what says whether a constructed boundary transform is needed.
        table[fam] = {
            "decision_change_rate": dec, "admission_change_rate": adm,
            "share": {k: v for k, v in sh.items() if k != "per_round"},
            "would_be_share_neutral_at_tol": sh["max_abs_delta"],
            "legs": legs, "eligible": ok,
        }
        print(f"   {fam:12s} decision={dec:.4f}  |admission|={adm:.4f}  "
              f"share moved {sh['n_rounds_share_moved_beyond_tol']}/{sh['n_adversary_rounds']} "
              f"(max |d|={sh['max_abs_delta']:.4g})  -> "
              f"{'ELIGIBLE' if ok else 'fails: ' + ','.join(k for k, v in legs.items() if not v)}")

    # The identity rung is its own control: it must move nothing, and it goes through the same code
    # path as every family, so a screen that silently measured nothing would fail here first.
    ident = [r for r in rows if r["d1"] == "fedavg"]
    ident_ctrl = {
        "n_rows": len(ident),
        "decision_change_rate": rate(ident, dec_key),
        "admission_change_rate": rate(ident, adm_key),
        "max_abs_rho_minus_one": float(max((abs(r["rho_realized"] - 1.0) for r in ident),
                                           default=float("nan"))),
        "all_flat": bool(all((not r[dec_key]) and abs(float(r[adm_key])) == 0.0 for r in ident)),
    }
    print(f"\n   identity control (fedavg): {ident_ctrl['n_rows']} rows, all_flat="
          f"{ident_ctrl['all_flat']}, max|rho-1|={ident_ctrl['max_abs_rho_minus_one']:.3g}")

    verdict = ("ELIGIBLE_FAMILY_FOUND" if eligible else "NO_ELIGIBLE_FAMILY")
    print(f"\n=== SCREEN VERDICT: {verdict} ===")
    if eligible:
        print(f"    {', '.join(eligible)} move Krum's decision at a coefficient share constant to "
              f"{SHARE_TOL:g}.")
    else:
        print("    No deployed oracle-free family is both informative and share-neutral at the "
              "paper's own tolerance.")
        print("    This is App. D.6's impossibility measured across the deployed class rather than "
              "argued for one member,")
        print("    and it is what licenses constructing a boundary transform instead. The screen "
              "cost one pass, not a ladder.")

    out = {
        "description": (
            "Screen of every deployed oracle-free d1 family for a clean statistic-channel probe: "
            "does any family move Krum's decision while holding the adversarial coefficient share "
            "constant to Mode S's own tolerance? Selects an instrument; measures no ASR; returns no "
            "verdict on any hypothesis about suppression. Every threshold is imported from an "
            "already-frozen constant, so none was chosen after reading these numbers."),
        "cell": {"arm": ARM, "attack": ATTACK,
                 "why_this_attack": (
                     "Krum under committed_scaling admits baseline adversarial mass 0.000 in 0 of "
                     "12 adversary rounds, so admission flatness there is a ceiling artifact; "
                     "under committed_pixel it admits 0.333, nonzero in 4 of 12. The floor belongs "
                     "to the attack, not the aggregator.")},
        "config": {"seeds": list(SEEDS), "rounds": ROUNDS, "families": list(FAMILIES),
                   "rungs": [[f, v, d] for f, v, d in RUNGS_SCREEN],
                   "decision_key": dec_key, "admission_key": adm_key},
        "thresholds": {
            "DECISION_FLOOR": DECISION_FLOOR, "ADMISSION_FLAT": ADMISSION_FLAT,
            "SHARE_TOL": SHARE_TOL,
            "provenance": (
                "DECISION_FLOOR and ADMISSION_FLAT imported from experiments/"
                "measure_admission_mask.py; SHARE_TOL imported from experiments/"
                "measure_admission.py. No threshold is defined in this script. App. D.7's "
                "pre-registration recorded SHARE_TOL and its runner recorded adv_coeff_share every "
                "round, and the gate was still not written; this screen writes that gate with the "
                "same constant rather than a laxer one.")},
        "excluded_families": {
            "dose|doseS|doseA|doseM": "raise without adv_mask; the oracle-bound families",
            "fltrust": ("raises unless the caller supplies a server with a clean_holdout_dataset, "
                        "which measure() does not pass, so it is unreachable through this loop"),
            "trimmed_mean|coord_median|krum|multi_krum": (
                "pass through unchanged as d1, so they are the identity upstream")},
        "oracle_free_check": ofc,
        "clip_binding_check": clip,
        "identity_control": ident_ctrl,
        "per_family": table,
        "eligible_families": eligible,
        "verdict": verdict,
        "what_this_does_not_establish": (
            "An empty result is a statement about the four families apply_d1_transform implements "
            "at this cell, on 5 seeds and 3 live rounds, at Mode S's tolerance. It is not a claim "
            "that no oracle-free share-neutral informative transform exists in general; App. D.6 "
            "makes that claim, for the positive per-client rescaling class, and by proof rather "
            "than by this measurement. Neither this screen nor that proof forbids APPROXIMATE "
            "uniform neutrality, which is what a constructed boundary transform targets."),
        "per_round": rows,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}  ({len(rows)} rows)")
    return out


if __name__ == "__main__":
    main()
