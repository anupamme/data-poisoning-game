"""
Seed top-up of the score-only magnitude control at its OWN frozen primary contrast: n=5 -> n=20.

WHY THIS ARM. The score-only control is the paper's only instrument that closes the MAGNITUDE channel
while leaving the STATISTIC channel open, so it is the one cell carrying "substantial statistic and
decision disturbance does not move suppression even with the aggregated update held fixed". It is
reported in the body and in tab:channels at n=5, while the paper's headline reversal is at n=20. The
load-bearing control is the paper's smallest seed count, and that asymmetry is what this suite fixes.

WHAT IS ALREADY DONE, so this does not re-run it. results/emit_only_topup/ took the score-only AND
emit-only cells to n=20 -- but only at the FACTORIAL's contrast rung kappa=1.0, which was itself a floor
substitution (the EMIT-only cell's mean clean accuracy at kappa=2 is 0.293 < ACC_FLOOR). The score-only
arm's own frozen primary contrast is kappa=0 -> kappa=2, its accuracy at kappa=2 is 0.6299, and that
contrast exists only at n=5. This suite extends that one contrast and nothing else.

IT STARTS WITH A DEPARTURE FROM A FROZEN CLAUSE, disclosed rather than smuggled.
experiments/pre_registration_score_only.md, frozen at 35788d9, fixes the primary rule at

    "On Delta = mean ASR(kappa=2) - mean ASR(kappa=0), n = 5 seeds {42, 43, 44, 45, 46}"

and declares as a limitation "One cell. One dataset, one architecture, one attack, one defense, n = 5."

The n=5 there was a COMPARABILITY commitment, not a power commitment -- the control had to share seeds
with the uncontrolled arm it is compared against. That rationale is honoured in full and in fact
RESTORED: the uncontrolled Mode-S arm is already at n=20 on this cell and these seeds
(results/dose_seed_topup/), so it is the control that now lags. The literal text is still departed from.
So: the published n=5 verdict and its Delta=-0.023 are reported unchanged wherever they appear, the n=20
result is labelled a DISCLOSED POST-HOC EXTENSION and never described as prospective, and 35788d9's n=5
text is quoted in the paper at the site where the n=20 number is reported.

WHAT IS PUBLISHED, recomputed per seed at run time rather than transcribed. krum / committed_scaling /
CIFAR-10 / cifar_cnn, seeds 42-46, from results/score_only/summary.json:

    kappa=0   mean ASR 0.06176 (sd 0.03464)   mean acc 0.5648
    kappa=2   mean ASR 0.03853 (sd 0.03780)   mean acc 0.6299
    paired d  -0.1144 / +0.0423 / +0.0402 / -0.0173 / -0.0669
              mean -0.02322, sd 0.06816, 95% CI at n=5 [-0.1078, +0.0614]

against EQUIV_MARGIN = 0.15. Published verdict: THE NEGATIVE SURVIVES WITH THE MAGNITUDE CHANNEL CLOSED.

WHAT n=20 CAN DECIDE -- and unlike the emit-only top-up, THIS CONTRAST IS POWERED FOR ITS MARGIN, which
is stated in advance so a pass cannot later be oversold or a failure explained away. Holding the observed
paired sd of 0.06816, the projected 95% half-width at n=20 (t19=2.093) is 0.0319 and the projected
interval holding the point estimate is [-0.0551, +0.0087]: inside +/-0.15 with room to spare, and
containing zero. So a pass at n=20 is a real certification against the frozen margin, and the arm is
genuinely falsifiable -- the refuting branch needs mean ASR(kappa=2) > 0.2118 against an identity rung of
0.0618, i.e. 3.4x the baseline against a ceiling of 1.0, which is reachable.

DESIGN. Seeds 47-61, rung kappa=2.0 ONLY. kappa=0.5 and kappa=1.0 are NOT run for the new seeds: the
score-only ladder across kappa and the Jonckheere-Terpstra trend test stay at n=5 and are not restated at
n=20. After this top-up the honest shape is kappa=0 and kappa=2 at n=20 with the interior at n=5, and no
display may print the ladder without a per-rung n.

    kappa=2 score_only x 15 = 15 new runs.

THE kappa=0 BASELINE IS IMPORTED, AND THE IMPORT'S WEAK POINT IS DISCLOSED HERE RATHER THAN DISCOVERED
LATER. At kappa=0 apply_d1_transform returns the update list unwrapped, generic_compose's score_only
branch has nothing to split, and run_one's participant RNG stream does not depend on d1's name -- so
score-only Krum at kappa=0 IS Krum alone. run_score_only_control.py:21 states it and :164 asserts it. For
seeds 47-61 the baseline is results/dose_seed_topup/'s kappa=0 cell, whose identity_rung_provenance
records it as computed there and "Verified bit-identical to krum standalone by --harness-check."
--harness-check re-asserts it HERE at ONE NEW SEED (47), not at fifteen, because asserting at all fifteen
costs the fifteen runs the import exists to save. If that single assertion fails at 1e-9 the import is
VOID, the arm owes all 30 runs, and the failure is reported -- the tolerance is not loosened.

Config identical to results/score_only: N=10, K=5, f=0.2, alpha=0.5, 50 rounds, cifar_cnn.
results/score_only/, results/emit_only/, results/targeted_dose/, results/dose_seed_topup/ and
results/emit_only_topup/ are read and NEVER written here.

Output: results/score_only_kappa2_topup/summary.json (resumable; written after every run).

DO NOT RUN until experiments/pre_registration_score_only_kappa2_n20.md is git-committed and PREREG_COMMIT
below is set to that hash, with the file clean in the working tree. The script refuses to start otherwise.

  python3 experiments/run_score_only_kappa2_topup.py --harness-check
  python3 experiments/run_score_only_kappa2_topup.py
"""
import json, os, subprocess, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
from experiments.run_targeted_dose import (run_one, d1_name, dial, cell_key, SEEDS5, ACC_FLOOR,
                                           EQUIV_MARGIN, ATTACK_MAP)

# experiments/pre_registration_score_only_kappa2_n20.md, committed before results/ existed for this arm.
PREREG = "experiments/pre_registration_score_only_kappa2_n20.md"
PREREG_COMMIT = "15d02e4"
# The frozen suite this one extends, and the clause it departs from. Quoted, not paraphrased.
PARENT_PREREG_COMMIT = "35788d9"
DEPARTURE = ("pre_registration_score_only.md at 35788d9: 'On Delta = mean ASR(kappa=2) - mean "
             "ASR(kappa=0), n = 5 seeds {42, 43, 44, 45, 46}', and 'One cell. One dataset, one "
             "architecture, one attack, one defense, n = 5.' This suite departs from the n=5 in both. "
             "That n was a COMPARABILITY commitment -- the control had to share seeds with the "
             "uncontrolled arm -- and the uncontrolled arm is already at n=20 on these seeds, so "
             "extending the control RESTORES the seed match rather than breaking it. The literal text "
             "is still departed from. The n=20 result is a DISCLOSED POST-HOC EXTENSION, never "
             "prospective, and the published n=5 verdict is reported unchanged beside it.")

MODE, D2, ATTACK = "S", "krum", "committed_scaling"
CONTRAST_RUNG = 2.0          # frozen at 35788d9 as this arm's OWN primary; acc 0.6299, clears the floor
BASELINE_RUNG = 0.0
NEW_SEEDS = list(range(47, 62))
PUBLISHED_SEEDS = SEEDS5
HARNESS_SEED = 47            # frozen: ONE new seed, named here so it cannot be chosen after the fact
LADDER = ("score_only", {"score_only": True})

# Disclosed in advance, from the published paired sd of 0.06816 (t19 = 2.093). This arm CAN certify
# against its own margin, which the emit-only top-up could not against its 0.05.
PROJECTED_HALF_WIDTH_N20 = 0.0319
T_CRIT_N20 = 2.093
# Carried forward unchanged from 35788d9. The refuting branch needs mean ASR(kappa=2) above this.
REFUTING_ASR_THRESHOLD = 0.2118

VERDICTS = {
    "flat": ("THE NEGATIVE SURVIVES WITH THE MAGNITUDE CHANNEL CLOSED (n=20). Substantial statistic and "
             "decision disturbance does not produce a corresponding change in suppression even when the "
             "aggregated update's magnitude and direction are held exactly fixed, at the same seed count "
             "as the headline reversal."),
    "refuted": ("THE PUBLISHED FLAT RESULT WAS PARTLY AN ARTIFACT OF THE MAGNITUDE CHANNEL, and n=5 "
                "concealed it. The paper's central claim narrows to 'with benign magnitudes free to "
                "move', stated in the body beside the result and in the abstract, not in a limitation."),
    "indeterminate": ("A fall with attenuation already closed and magnitude also closed. INDETERMINATE, "
                      "reported as such and NOT scored in our favour."),
}

out_dir = os.path.join(base, "results", "score_only_kappa2_topup")
out_path = os.path.join(out_dir, "summary.json")
SCORE = os.path.join(base, "results", "score_only", "summary.json")
TOPUP = os.path.join(base, "results", "dose_seed_topup", "summary.json")


def rung_rows(path, val, require_no_flag=True):
    """{seed: (accuracy, asr)} for this arm's rung of a published summary. Read-only.

    Mode, d2 and attack are all matched, not just the rung: results/targeted_dose/ holds five arms and
    two modes and Mode A's rung values overlap Mode S's, so matching on the rung alone would silently
    pick up the wrong cell. require_no_flag additionally skips any cell carrying a 'ladder' key, so a
    score_only/emit_only cell can never be mistaken for an uncontrolled one.
    """
    if not os.path.exists(path):
        return {}
    for c in json.load(open(path)).get("cells", {}).values():
        if require_no_flag and c.get("ladder") is not None:
            continue
        if (c.get("mode", MODE) == MODE and c.get("d2") == D2 and c.get("attack") == ATTACK
                and abs(c.get("rung", 1e9) - val) < 1e-12):
            return {int(r["seed"]): (float(r["accuracy"]), float(r["asr"])) for r in c["per_seed"]}
    return {}


def baseline_rows():
    """{seed: (accuracy, asr, source)} for the shared kappa=0 identity rung.

    Published seeds come from results/score_only/ (whose kappa=0 rung is itself the imported identity
    rung); new seeds from results/dose_seed_topup/, where kappa=0 is plain Mode S and therefore the same
    computation as score-only at kappa=0. That equality is the import, and --harness-check asserts it.
    """
    out = {}
    for path, tag in ((SCORE, "results/score_only kappa=0 (imported identity rung)"),
                      (TOPUP, "results/dose_seed_topup kappa=0 (Mode S = score-only at kappa=0)")):
        for s, (acc, asr) in rung_rows(path, BASELINE_RUNG, require_no_flag=False).items():
            out.setdefault(s, (acc, asr, tag))
    return out


def stat(a):
    a = np.asarray(a, dtype=float)
    n = len(a)
    sd = float(a.std(ddof=1)) if n > 1 else float("nan")
    hw = float(T_CRIT_N20 * sd / np.sqrt(n)) if n > 1 else float("nan")
    return {"n": n, "mean": float(a.mean()), "sd": sd, "half_width_t19": hw,
            "ci95": [float(a.mean() - hw), float(a.mean() + hw)],
            "per_seed": [float(x) for x in a]}


def published_contrast():
    """The n=5 verdict, recomputed per seed from results/score_only/ rather than transcribed."""
    b = {s: v[1] for s, v in baseline_rows().items() if s in PUBLISHED_SEEDS}
    k2 = rung_rows(SCORE, CONTRAST_RUNG, require_no_flag=False)
    seeds = sorted(set(b) & set(k2))
    if not seeds:
        return {}
    d = np.array([k2[s][1] - b[s] for s in seeds])
    s = stat(d)
    s.update({"seeds": seeds, "margin": EQUIV_MARGIN,
              "verdict": ("THE NEGATIVE SURVIVES WITH THE MAGNITUDE CHANNEL CLOSED"
                          if abs(d.mean()) < EQUIV_MARGIN else "see analyzer"),
              "note": "Reported unchanged alongside the n=20 result whatever the latter is.",
              "half_width_is_t19_not_t4": "The half_width field uses t19 for comparability with the "
                                          "n=20 row; the published n=5 interval is [-0.1078, +0.0614] "
                                          "at t4=2.776 and is what the paper quotes."})
    return s


def check_frozen():
    prereg = os.path.join(base, PREREG)
    if not os.path.exists(prereg):
        sys.exit(f"REFUSING TO RUN: {prereg} does not exist.")
    if PREREG_COMMIT is None:
        sys.exit(f"REFUSING TO RUN: the verdict rules and the power disclosure are not frozen.\n"
                 f"  1. git commit {PREREG}\n  2. set PREREG_COMMIT here to that hash.\n"
                 "This suite departs from a frozen clause of its parent. That is only defensible if "
                 "the departure, its rationale and its limits are committed before any seed runs.")
    # PREREG_COMMIT is None is only HALF a guard: once the constant is set, the only test that ever
    # fires stops firing. So also assert the file is committed AT that hash and clean in the tree.
    try:
        in_commit = subprocess.run(["git", "cat-file", "-e", f"{PREREG_COMMIT}:{PREREG}"],
                                   cwd=base, capture_output=True).returncode == 0
        dirty = subprocess.run(["git", "status", "--porcelain", "--", PREREG],
                               cwd=base, capture_output=True, text=True).stdout.strip()
    except Exception as e:
        sys.exit(f"REFUSING TO RUN: cannot verify the freeze with git ({e}).")
    if not in_commit:
        sys.exit(f"REFUSING TO RUN: {PREREG} is not present at {PREREG_COMMIT}. PREREG_COMMIT names a "
                 "commit that does not contain the pre-registration, so nothing is actually frozen.")
    if dirty:
        sys.exit(f"REFUSING TO RUN: {PREREG} is modified in the working tree ({dirty!r}). The frozen "
                 "rules must be the committed ones, not an edited copy.")
    if not os.path.exists(SCORE):
        sys.exit(f"REFUSING TO RUN: {SCORE} does not exist. The published n=5 cell this top-up extends "
                 "lives there, and the analysis asserts it reproduces before pooling.")
    if not rung_rows(SCORE, CONTRAST_RUNG, require_no_flag=False):
        sys.exit(f"REFUSING TO RUN: no published score-only cell at kappa={CONTRAST_RUNG}. The contrast "
                 "rung is frozen; it is not relocated to whatever rung exists.")
    missing = [s for s in NEW_SEEDS if s not in baseline_rows()]
    if missing:
        sys.exit(f"REFUSING TO RUN: no kappa=0 identity rung to import at seeds {missing}. The "
                 "baseline is imported by design; it is not silently recomputed at a different n.")


def harness_check():
    """score_only at kappa=0 must reproduce the imported identity rung bit-identically, at seed 47.

    At kappa=0 apply_d1_transform returns the update list unwrapped, so there is no transformed stack to
    score on: score_only collapses to plain Krum. That equality IS the import of the kappa=0 baseline at
    the new seeds, so it is asserted rather than assumed -- and asserted at a NEW seed, because the
    published seeds' equality is already visible in the published data and proves nothing about 47-61.
    """
    rows = baseline_rows()
    if HARNESS_SEED not in rows:
        sys.exit(f"no imported kappa=0 row at seed {HARNESS_SEED} to check against")
    r_acc, r_asr, src = rows[HARNESS_SEED]
    print("=== HARNESS CHECK: score_only at kappa=0 must BE the imported identity rung ===")
    print(f"    cifar10/cifar_cnn, {D2}/{ATTACK.replace('committed_', '')} at seed {HARNESS_SEED}")
    print(f"    imported ({src}): acc={r_acc:.6f} ASR={r_asr:.6f}\n", flush=True)
    t = time.time()
    acc, asr = run_one(HARNESS_SEED, MODE, D2, ATTACK, BASELINE_RUNG, **LADDER[1])
    d = (acc - r_acc, asr - r_asr)
    ok = abs(d[0]) < 1e-9 and abs(d[1]) < 1e-9
    print(f"  score_only  acc={acc:.6f} ASR={asr:.6f}   d=({d[0]:+.2e}, {d[1]:+.2e})  "
          f"({time.time() - t:.0f}s)", flush=True)
    print("\n  " + ("BIT-IDENTICAL. The kappa=0 import is valid at the new seeds."
                    if ok else
                    "NOT IDENTICAL: the import is VOID. This arm owes all 30 runs, and the failure is "
                    "reported rather than explained away. The tolerance is NOT loosened."))
    return 0 if ok else 1


def load_cells():
    if not os.path.exists(out_path):
        return {}
    try:
        return json.load(open(out_path)).get("cells", {})
    except Exception:
        return {}


def save(cells):
    json.dump({"description": "Seed top-up of the score-only magnitude control at its OWN frozen "
                              f"primary contrast kappa=0 -> kappa={CONTRAST_RUNG}, seeds 47-61, "
                              "bringing the magnitude-closed negative from n=5 to n=20. A DISCLOSED "
                              f"POST-HOC EXTENSION of {PARENT_PREREG_COMMIT}, not a prospective test. "
                              f"Limits frozen at {PREREG_COMMIT} ({PREREG}).",
               "prereg": PREREG,
               "prereg_commit": PREREG_COMMIT,
               "parent_prereg_commit": PARENT_PREREG_COMMIT,
               "departure_from_parent_non_negotiable": DEPARTURE,
               "dataset": "cifar10", "model": "cifar_cnn",
               "config": {"N": 10, "K": 5, "f": 0.2, "alpha": 0.5, "rounds": 50,
                          "contrast_rung": CONTRAST_RUNG, "baseline_rung": BASELINE_RUNG,
                          "new_seeds": NEW_SEEDS, "published_seeds": PUBLISHED_SEEDS,
                          "rho_at_contrast": dial(MODE, CONTRAST_RUNG),
                          "acc_floor": ACC_FLOOR, "margin": EQUIV_MARGIN,
                          "harness_seed": HARNESS_SEED},
               "arm": {"mode": MODE, "d2": D2, "attack": ATTACK,
                       "attack_impl": ATTACK_MAP[ATTACK],
                       "ladder_run_here": LADDER[0],
                       "channels": "statistic channel OPEN, magnitude channel CLOSED",
                       "baseline_source": "results/score_only/ (42-46) + results/dose_seed_topup/ "
                                          "(47-61); imported, not recomputed"},
               "can_certify": "UNLIKE results/emit_only_topup/, this contrast is powered for its "
                              f"margin: projected 95% half-width at n=20 is {PROJECTED_HALF_WIDTH_N20} "
                              f"against a margin of {EQUIV_MARGIN}. A pass is a real certification "
                              "against the frozen margin, reported with its interval at every site.",
               "falsifiable": f"The refuting branch needs mean ASR(kappa={CONTRAST_RUNG}) > "
                              f"{REFUTING_ASR_THRESHOLD} against an identity rung of 0.0618, i.e. 3.4x "
                              "the baseline against a ceiling of 1.0. Reachable.",
               "ladder_remains_at_n5": "The score-only ladder across kappa and the Jonckheere-Terpstra "
                                       "trend test are NOT restated at n=20. Only the frozen primary "
                                       "contrast is extended. After this top-up the honest shape is "
                                       "kappa=0 and kappa=2 at n=20 with the interior at n=5, and no "
                                       "display may print the ladder without a per-rung n.",
               "oracle_scope": "Closing the magnitude channel does NOT remove the oracle. Mode S still "
                               "reads adversary identity to pin c_adv=1, at n=20 exactly as at n=5. "
                               "This arm licenses no statement about oracle-free behaviour in either "
                               "direction.",
               "trajectory_not_closed": "Which client is selected still changes across rungs, so the "
                                        "model trajectory still diverges. 35788d9's clause stands: the "
                                        "claim is about magnitude, not trajectory. n=20 does not widen "
                                        "it.",
               "selectors_only": "generic_compose raises for non-selectors, so the coord_median arms -- "
                                 "including the cell carrying the paper's headline reversal -- are NOT "
                                 "controlled for magnitude by this arm. See "
                                 "experiments/pre_registration_score_only_coordmedian.md; no result "
                                 "here transfers to it.",
               "verdict_labels_frozen": VERDICTS,
               "published_n5_contrast": published_contrast(),
               "cells": cells}, open(out_path, "w"), indent=2)


def main():
    check_frozen()
    os.makedirs(out_dir, exist_ok=True)
    pub = published_contrast()
    label, flags = LADDER
    total = len(NEW_SEEDS)
    print("=== Score-only magnitude control: seed top-up at its own frozen primary contrast ===")
    print(f"    {D2}/{ATTACK.replace('committed_', '')}, kappa={CONTRAST_RUNG} "
          f"(rho={dial(MODE, CONTRAST_RUNG):.2f}), seeds {NEW_SEEDS[0]}-{NEW_SEEDS[-1]} "
          f"({total} runs)")
    print(f"    n=5 -> n=20 on the magnitude-closed negative. Limits frozen at {PREREG_COMMIT}.")
    if pub:
        print(f"    published n={pub['n']}: Delta={pub['mean']:+.6f} (sd {pub['sd']:.5f}), "
              f"margin {EQUIV_MARGIN}, verdict: {pub['verdict']}")
    print(f"    DISCLOSED POST-HOC EXTENSION of {PARENT_PREREG_COMMIT}, which fixed n=5.")
    print(f"    This contrast IS powered for its margin: projected 95% half-width "
          f"{PROJECTED_HALF_WIDTH_N20} < {EQUIV_MARGIN}. A pass certifies; the refuting branch needs "
          f"mean ASR > {REFUTING_ASR_THRESHOLD} and is reachable.\n", flush=True)

    cells = load_cells()
    if cells:
        print(f"  resuming: {sum(len(c['per_seed']) for c in cells.values())} runs already done\n",
              flush=True)
    t0 = time.time(); done = 0
    key = cell_key(MODE, D2, ATTACK, CONTRAST_RUNG) + f"|{label}"
    existing = {r["seed"]: r for r in cells.get(key, {}).get("per_seed", [])}
    for seed in NEW_SEEDS:
        if seed in existing:
            done += 1
            continue
        t = time.time()
        acc, asr = run_one(seed, MODE, D2, ATTACK, CONTRAST_RUNG, **flags)
        existing[seed] = {"seed": int(seed), "accuracy": float(acc), "asr": float(asr),
                          "source": "<computed here>"}
        done += 1
        print(f"  [{done}/{total}] {d1_name(MODE, CONTRAST_RUNG):16s} {label:11s} s{seed}: "
              f"acc={acc:.4f} ASR={asr:.4f} ({time.time() - t:.0f}s)"
              + ("  * below acc floor" if acc < ACC_FLOOR else ""), flush=True)
        cells[key] = {"mode": MODE, "d2": D2, "attack": ATTACK, "rung": float(CONTRAST_RUNG),
                      "dial": dial(MODE, CONTRAST_RUNG), "ladder": label,
                      "per_seed": [existing[s] for s in sorted(existing)]}
        save(cells)

    print(f"\nWall time: {(time.time() - t0) / 3600:.1f} h\nSaved to {out_path}")
    print("Run experiments/analyze_score_only_kappa2_topup.py, which pools these with the published "
          "five, applies the frozen margin rule to the mean, prints the interval beside it, and reports "
          "the n=5 verdict unchanged alongside.")
    return 0


if __name__ == "__main__":
    if "--harness-check" in sys.argv:
        check_frozen()
        sys.exit(harness_check())
    sys.exit(main())
