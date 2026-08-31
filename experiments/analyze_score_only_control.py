"""Score the score-only control against the rules frozen in pre_registration_score_only.md.

Scores exactly those rules and nothing else:

  primary    Delta = mean ASR(kappa=2) - mean ASR(kappa=0), n=5, against EQUIV_MARGIN = 0.15
             |Delta| < 0.15  -> the negative survives with the magnitude channel closed
             Delta > +0.15   -> the published flat result was partly a magnitude-channel artifact,
                                and the central claim narrows to "with benign magnitudes free to move"
             Delta < -0.15   -> a fall with both channels closed, INDETERMINATE, not in our favour
  secondary  Jonckheere-Terpstra across all four rungs, specified to detect a monotone RISE
  gate       every rung needs mean clean accuracy >= ACC_FLOOR = 0.35
  sanity     kappa=0 is score-only Krum at the identity, which IS krum alone, so that rung is
             imported from results/targeted_dose and must be bit-identical (the runner's
             --harness-check verifies the claim at one seed)

Every constant, the rung builder, the interval and the trend test are imported from the frozen suite's
own scorers rather than re-derived, so this arm and the uncontrolled arm compute the same quantities by
construction. The uncontrolled Mode-S Delta and the frozen channel premise are read by the runner's own
readers, so the analyzer and the run cannot disagree about either.

Read-only on every frozen artifact. Writes results/score_only/scored.json.
"""
import json
import os
import sys

import numpy as np

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base)
# single-sourced: same rung builder, same interval, same gate, same margin, same trend test
from experiments.analyze_targeted_dose import (  # noqa: E402
    ACC_FLOOR, EQUIV_MARGIN, KAPPAS, rungs_of, show,
)
from experiments.analyze_dose_response import jonckheere  # noqa: E402
# the uncontrolled comparison and the frozen premise come from the runner's readers, not from copies
from experiments.run_score_only_control import (  # noqa: E402
    ATTACK, D2, PREREG_COMMIT, premise, published_delta,
)

SCORE_ONLY = os.path.join(base, "results", "score_only", "summary.json")
COS_CHECK = os.path.join(base, "results", "score_only", "cos_krum_check.json")
OUT = os.path.join(base, "results", "score_only", "scored.json")


def cos_uncontrolled(attack, seed):
    """The published Mode-S cos_krum ladder the corollary reasons about: arm mean and one seed.

    Read here rather than asserted, because bit-identity at a seed only speaks to the ARM's published
    fall if that seed actually falls. Seed 42 does not, and an artifact that claims otherwise would
    put an overstatement into the paper.
    """
    path = os.path.join(base, "results", "targeted_dose", "summary.json")
    if not os.path.exists(path):
        return None
    out = {}
    for c in json.load(open(path)).get("cells", {}).values():
        if c.get("mode") == "S" and c.get("d2") == "cos_krum" and c.get("attack") == attack:
            rows = c["per_seed"]
            per = {r["seed"]: r["asr"] for r in rows}
            out[c["rung"]] = {"mean_asr": float(np.mean([r["asr"] for r in rows])),
                              "n_seeds": len(rows), "seed_asr": per.get(seed)}
    if not out or any(v["seed_asr"] is None for v in out.values()):
        return None
    ks = sorted(out)
    return {"rungs": out, "arm_delta": out[ks[-1]]["mean_asr"] - out[ks[0]]["mean_asr"],
            "seed_delta": out[ks[-1]]["seed_asr"] - out[ks[0]]["seed_asr"],
            "n_seeds": out[ks[0]]["n_seeds"]}


def cos_corollary():
    """Prereg Section 5: the one-seed bit-identity test of the magnitude attribution."""
    if not os.path.exists(COS_CHECK):
        return None
    c = json.load(open(COS_CHECK))
    ident = c.get("bit_identical")
    unc = cos_uncontrolled(c.get("attack", "committed_pixel"), c.get("seed"))
    asrs = [r["asr"] for r in c["rungs"]]
    vacuous = bool(ident and max(asrs) == 0.0)
    # Does the checked seed move at all under the uncontrolled dose, and in the arm's direction?
    same_dir = None if unc is None else bool(
        unc["seed_delta"] * unc["arm_delta"] > 0 and abs(unc["seed_delta"]) > 1e-12)
    if not ident:
        interp = ("NOT bit-identical: the magnitude attribution of the published fall is REFUTED and "
                  "is withdrawn in the body, in the same paragraph and with the same prominence")
    elif vacuous:
        interp = ("bit-identical but VACUOUS: ASR is 0.000 at every rung, so no movement exists in "
                  "this run to attribute to any channel. Do not report as confirming the attribution")
    elif same_dir is False:
        interp = ("bit-identical, so score-only removes ALL of this seed's uncontrolled movement "
                  f"({unc['seed_delta']:+.3f}) and the invariance claim holds. But this seed moves "
                  f"OPPOSITE to the arm mean ({unc['arm_delta']:+.3f} over {unc['n_seeds']} seeds), "
                  "so the check corroborates the CHANNEL and not the arm's published fall; "
                  "attributing that fall would need the arm's other seeds, which the "
                  "pre-registration fixed at one and which are not added after seeing the data")
    else:
        interp = ("bit-identical, and the checked seed moves with the arm, so the attribution of the "
                  "published fall to the magnitude channel is corroborated at this seed")
    return {"ran": True, "bit_identical": ident, "seed": c.get("seed"),
            "attack": c.get("attack"), "vacuous_all_zero_asr": vacuous,
            "rungs_below_acc_floor": c.get("rungs_below_acc_floor"),
            "max_abs_d_asr": max(abs(r["d_asr"]) for r in c["rungs"]),
            "max_abs_d_accuracy": max(abs(r["d_accuracy"]) for r in c["rungs"]),
            "uncontrolled": unc, "checked_seed_moves_with_arm": same_dir,
            "interpretation": interp}


def main():
    if not os.path.exists(SCORE_ONLY):
        sys.exit(f"missing {SCORE_ONLY} -- run experiments/run_score_only_control.py first")
    doc = json.load(open(SCORE_ONLY))
    cells = doc["cells"]

    r = rungs_of(cells, "S", D2, KAPPAS)
    if r is None:
        sys.exit("the score-only arm is incomplete: not all four rungs are present")

    print("=== SCORE-ONLY CONTROL: statistic and decision disturbed, magnitude held fixed ===")
    print(f"    rules frozen in experiments/pre_registration_score_only.md @ {PREREG_COMMIT}")
    print(f"    {doc['dataset']}/{doc['model']}, mode S {D2}/{ATTACK.replace('committed_', '')}, "
          f"score_only=True")
    print(f"    kappa {KAPPAS}, seeds {r[0]['seeds']}")
    print(f"    '*' = mean clean accuracy < {ACC_FLOOR} (uninterpretable, not suppression)\n")
    show(f"score-only {D2}", r)
    if r[0]["imported"]:
        print(f"  {'':24s} kappa=0 imported bit-exactly: score-only at the identity IS krum alone")
    print()

    delta = float(r[-1]["mean"] - r[0]["mean"])
    pub = published_delta()
    equivalent = abs(delta) < EQUIV_MARGIN
    if equivalent:
        verdict = "THE NEGATIVE SURVIVES WITH THE MAGNITUDE CHANNEL CLOSED"
        reading = ("Substantial statistic and decision disturbance does not produce a corresponding "
                   "change in suppression even when the magnitude and direction of the aggregated "
                   "update are held exactly fixed. The central claim is instrumented against two "
                   "independent confounds -- attenuation (Mode S) and magnitude (here).")
    elif delta > EQUIV_MARGIN:
        verdict = "THE PUBLISHED FLAT RESULT WAS PARTLY AN ARTIFACT OF THE MAGNITUDE CHANNEL"
        reading = ("With magnitude closed, statistic disturbance does move suppression. The central "
                   "claim narrows to 'with benign magnitudes free to move', stated in the body next "
                   "to the result rather than in a limitation.")
    else:
        verdict = "INDETERMINATE: a fall with both the attenuation and magnitude channels closed"
        reading = ("Not scored in our favour. Reported as indeterminate; an equivalence margin is not "
                   "satisfied by a fall.")

    print("=== PRIMARY (frozen) ===")
    print(f"  Delta = {delta:+.4f}  ({r[0]['mean']:.4f} -> {r[-1]['mean']:.4f}), "
          f"margin +/-{EQUIV_MARGIN}")
    print(f"  refuting threshold: mean ASR(kappa=2) > {r[0]['mean'] + EQUIV_MARGIN:.4f}")
    if pub is not None:
        print(f"  uncontrolled Mode-S Delta = {pub:+.4f} (recomputed from results/targeted_dose)")
    print(f"  {verdict}\n  {reading}\n")

    gated = [x["rung"] for x in r if x["gated"]]
    if gated:
        print(f"  ACCURACY GATE FAILED at kappa {gated}: mean clean accuracy below {ACC_FLOOR}. "
              "The cell is uninterpretable and no verdict stands.\n")

    J, z, p_up, p_down, p_perm = jonckheere([x["asrs"] for x in r])
    print("=== SECONDARY (frozen): Jonckheere-Terpstra, specified to detect a monotone RISE ===")
    print(f"  J = {J:.1f}, z = {z:+.3f}, p(rise) = {p_up:.4f}, permutation p(rise) = {p_perm:.4f}")
    print("  Pre-registered caveat: at n=5 the JT test is weak; a null is NOT evidence of flatness "
          "and is not reported as such.")
    if z < 0:
        print("  The observed trend is downward, which is not the direction this test was specified "
              "to detect: reported, not scored.\n")
    else:
        print()

    cos = cos_corollary()
    print("=== COROLLARY (prereg Section 5): score-only cos_krum bit-identity ===")
    if cos is None:
        print("  NOT RUN. python3 experiments/run_score_only_control.py --cos-krum-check\n")
    else:
        print(f"  seed {cos['seed']} on {cos['attack']}: max |d ASR| = {cos['max_abs_d_asr']:.2e}, "
              f"max |d acc| = {cos['max_abs_d_accuracy']:.2e}")
        u = cos["uncontrolled"]
        if u is not None:
            print(f"  uncontrolled at this seed: {u['seed_delta']:+.3f};  arm mean over "
                  f"{u['n_seeds']} seeds: {u['arm_delta']:+.3f}")
        print(f"  {cos['interpretation']}\n")

    pr = premise()
    print("=== THE CHANNEL THIS ARM CLOSES (frozen, results/admission_measurement.json) ===")
    print("  kappa   aggregate displacement   decision change   admission change")
    for k in KAPPAS:
        q = pr.get(str(k), {})
        f = lambda v: "na" if v is None else f"{v:.3f}"  # noqa: E731
        print(f"  {k:<6}  {f(q.get('agg_disp')):>22}  {f(q.get('decision')):>16}  "
              f"{f(q.get('admission')):>16}")
    print("  Mode S closes the adversarial-attenuation channel (admission pinned at 0.000);")
    print("  this arm additionally closes the aggregate-displacement channel, which the")
    print("  uncontrolled arm leaves open at 0.892 by the top rung.\n")

    scored = {
        "description": "Score-only control scored against experiments/pre_registration_score_only.md.",
        "prereg_commit": PREREG_COMMIT,
        "dataset": doc["dataset"], "model": doc["model"],
        "arm": {"mode": "S", "d2": D2, "attack": ATTACK, "score_only": True},
        "rungs": [{"kappa": x["rung"], "mean_asr": x["mean"], "ci": [x["lo"], x["hi"]],
                   "mean_accuracy": x["acc"], "asrs": x["asrs"], "accs": x["accs"],
                   "seeds": x["seeds"], "gated": x["gated"], "imported": x["imported"]}
                  for x in r],
        "primary": {"delta": delta, "equiv_margin": EQUIV_MARGIN, "equivalent": equivalent,
                    "refuting_threshold": float(r[0]["mean"] + EQUIV_MARGIN),
                    "uncontrolled_mode_s_delta": pub,
                    "verdict": verdict, "reading": reading},
        "secondary_jt": {"J": J, "z": z, "p_rise": p_up, "p_rise_perm": p_perm,
                         "caveat": "n=5; a null is not evidence of flatness",
                         "direction_as_specified": bool(z >= 0)},
        "accuracy_gate": {"floor": ACC_FLOOR, "failed_rungs": gated, "ok": not gated},
        "cos_krum_corollary": cos,
        "frozen_premise_channels": pr,
        "limitations": [
            "Selectors only: coord_median and reputation emit no single selected client, so the "
            "control is undefined there. The coord_median arm carrying the +0.098 rise is NOT "
            "magnitude-controlled.",
            "Score-only Krum is an instrument, not a defense: it aggregates an update the scoring "
            "stage did not see.",
            "One cell: one dataset, one architecture, one attack, one defense, n=5.",
            "Closing the magnitude channel does not hold the model trajectory fixed, because which "
            "client is selected still changes -- by design.",
        ],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(scored, open(OUT, "w"), indent=2)
    print(f"Saved {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
