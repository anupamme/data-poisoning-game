"""
Seed top-up of the score-only / emit-only factorial at its contrast rung: n=5 -> n=20 (item C).

IT STARTS WITH A DEPARTURE FROM A FROZEN NON-NEGOTIABLE, and the departure is disclosed rather than
smuggled. experiments/pre_registration_emit_only.md, frozen at b995f1b, says:

    "No seed addition. n=5, seeds 42-46, matching the arm it is compared against."

Two things are true about that:

  1. The clause's STATED RATIONALE is comparability -- the seeds must match "the arm it is compared
     against", because the additivity residual is computed ACROSS the three factorial cells and a
     residual assembled from cells with different n is not a residual. Adding seeds to the emit-only
     cell alone would violate that rationale. This top-up extends ALL THREE cells at the contrast
     rung to the same n=20 on the same seeds, so the rationale is honoured in full.
  2. The LITERAL TEXT is still departed from, and no reading of it makes that disappear. So: the
     published n=5 verdict is reported unchanged wherever it appears, this suite's n=20 result is
     labelled a DISCLOSED POST-HOC EXTENSION of a frozen suite and never described as prospective,
     and b995f1b's clause is quoted in the paper at the site where the n=20 number is reported.

WHAT IS PUBLISHED. krum / committed_scaling / CIFAR-10, seeds 42-46. The frozen primary contrast is
the kappa=0 -> kappa=2 endpoint with a floor contingency also frozen at b995f1b; the emit-only cell's
mean clean accuracy at kappa=2 is 0.293, below ACC_FLOOR=0.35, so the contingency fired and the
primary contrast is kappa=0 -> kappa=1. Seed-matched mean changes at that contrast:

  dASR_full  -0.038867 (sd 0.0455)     dASR_SO  -0.029978 (sd 0.0638)
  dASR_EO    +0.045578 (sd 0.1297)     residual -0.054467 (sd 0.1222)

against frozen margins |dASR_EO| < 0.05 (magnitude channel inert) and |residual| < 0.05 (channels
separable). The published verdict is CHANNELS INTERACT: the residual misses its margin by 0.0045.
SEED 42 IS THE WHOLE OF IT -- it contributes +0.2721 of the +0.0456 mean dASR_EO, i.e. 119% of the
mean, and -0.2586 of the -0.0545 residual.

WHAT n=20 CAN DECIDE, AND WHAT IT CANNOT. Both frozen before any run, because SEPARABLE is the
cleaner story and a top-up that could flip the verdict toward it is a top-up with an interest.

  IT CANNOT CERTIFY. Holding the observed sds, the projected 95% half-widths at n=20 are 0.061 for
  dASR_EO and 0.057 for the residual -- both WIDER THAN THE 0.05 MARGIN ITSELF. Even a point estimate
  of exactly zero could not put a 95% interval inside the margin at this variance and this n. So a
  pass of the frozen point-estimate rule at n=20 is reported as "consistent with separability, NOT
  established", never as SEPARABLE-full-stop, and the interval is printed next to the point estimate
  every time.

  IT CAN DECIDE THE REVIEWER'S ACTUAL COMPLAINT. "One seed carries the whole result" is a question
  about the distribution, and 15 more seeds answer it: the realized sd against 0.1297, the COUNT of
  seeds with dASR_EO > +0.15, the count with residual < -0.15, and the full per-seed distribution
  printed rather than summarized. If the sd collapses and 42 is alone in 20 seeds, the published
  verdict rode one draw. If the mode recurs, the emit-only channel is genuinely bimodal across data
  partitions and the interaction is real -- and THAT outcome vindicates the published verdict, which
  is why the descriptive pre-commitment is not directional.

DESIGN. Seeds 47-61, contrast rung kappa=1.0 ONLY. The additivity residual is a function of that
contrast alone, so extending kappa=0.5 and kappa=2 buys nothing for the quantity under discussion.
The score-only and emit-only LADDERS, and the Jonckheere-Terpstra trend test the score-only arm
reports, stay at n=5 and are not restated at n=20; every reported number carries its own n.

  kappa=1 score_only x 15 + kappa=1 emit_only x 15 = 30 new runs.

The kappa=0 baseline is IMPORTED: at kappa=0 apply_d1_transform returns the update list unwrapped,
generic_compose's score_only/emit_only split has nothing to split, and run_one's participant RNG
stream does not depend on d1's name -- which is why all three published cells' identity rungs are
equal seed for seed today. For seeds 47-61 it comes from results/dose_seed_topup/ (item A).
The dASR_full cell at those seeds is item A's output and is not recomputed here.

DEPENDENCY, stated because it can bite: the residual at n=20 needs all three cells at the same seeds.
If item A is incomplete when the analysis runs, the residual is reported at the n actually SHARED by
all three cells, and that n is printed. It is never assembled from cells with different seed sets.

Config identical to results/emit_only and results/score_only: N=10, K=5, f=0.2, alpha=0.5, 50 rounds,
cifar_cnn. results/emit_only/, results/score_only/ and results/targeted_dose/ are never written here.

Output: results/emit_only_topup/summary.json (resumable; written after every run).

DO NOT RUN until experiments/pre_registration_emit_only_topup.md is git-committed and PREREG_COMMIT
below is set to that hash. The script refuses to start otherwise.

  python3 experiments/run_emit_only_topup.py --harness-check
  python3 experiments/run_emit_only_topup.py
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
from experiments.run_targeted_dose import (run_one, d1_name, dial, cell_key, SEEDS5, ACC_FLOOR,
                                           ATTACK_MAP)

# experiments/pre_registration_emit_only_topup.md, committed before results/emit_only_topup/ existed.
PREREG_COMMIT = "684b31e"
# The frozen suite this one extends, and the clause it departs from. Quoted, not paraphrased.
PARENT_PREREG_COMMIT = "b995f1b"
DEPARTURE = ("pre_registration_emit_only.md at b995f1b: 'No seed addition. n=5, seeds 42-46, "
             "matching the arm it is compared against.' This suite departs from that clause. Its "
             "stated rationale -- comparability across the factorial's cells -- is honoured by "
             "extending ALL THREE cells at the contrast rung to the same seeds; its literal text is "
             "not. The n=20 result is a DISCLOSED POST-HOC EXTENSION, never prospective, and the "
             "published n=5 verdict is reported unchanged beside it.")

MODE, D2, ATTACK = "S", "krum", "committed_scaling"
CONTRAST_RUNG = 1.0          # frozen: kappa=2 failed the emit-only accuracy floor (0.293 < 0.35)
BASELINE_RUNG = 0.0
NEW_SEEDS = list(range(47, 62))
PUBLISHED_SEEDS = SEEDS5
INERT_MARGIN = 0.05         # carried forward unchanged from b995f1b
ADDITIVITY_MARGIN = 0.05    # carried forward unchanged from b995f1b
# Descriptive pre-commitment: how many seeds look like seed 42. Frozen so the count cannot be chosen
# after the distribution is seen.
OUTLIER_THRESHOLD = 0.15

out_dir = os.path.join(base, "results", "emit_only_topup")
out_path = os.path.join(out_dir, "summary.json")
EMIT = os.path.join(base, "results", "emit_only", "summary.json")
SCORE = os.path.join(base, "results", "score_only", "summary.json")
TOPUP = os.path.join(base, "results", "dose_seed_topup", "summary.json")

# (label, flag kwargs). The full cell is item A's; only these two are run here.
LADDERS = [("score_only", {"score_only": True}), ("emit_only", {"emit_only": True})]


def rung_rows(path, val):
    """{seed: (accuracy, asr)} for this arm's rung of a published summary. Read-only.

    Mode, d2 and attack are all matched, not just the rung: results/targeted_dose/ holds five arms
    and two modes, and Mode A's rung values overlap Mode S's, so matching on the rung alone would
    silently pick up the wrong cell.
    """
    if not os.path.exists(path):
        return {}
    for c in json.load(open(path)).get("cells", {}).values():
        if (c.get("mode", MODE) == MODE and c.get("d2") == D2 and c.get("attack") == ATTACK
                and abs(c.get("rung", 1e9) - val) < 1e-12):
            return {int(r["seed"]): (float(r["accuracy"]), float(r["asr"])) for r in c["per_seed"]}
    return {}


def baseline_rows():
    """{seed: (accuracy, asr, source)} for the shared kappa=0 identity rung across all three cells.

    Published seeds come from results/emit_only/ (which imported them from the Round-11 ladder and is
    equal to the other two cells seed for seed); new seeds come from item A.
    """
    out = {}
    for path, tag in ((EMIT, "results/emit_only kappa=0 (imported identity)"),
                      (TOPUP, "results/dose_seed_topup kappa=0")):
        for s, (acc, asr) in rung_rows(path, BASELINE_RUNG).items():
            out.setdefault(s, (acc, asr, f"{tag} (identical computation)"))
    return out


def published_contrast():
    """The n=5 verdict, recomputed per seed from the three published files rather than transcribed."""
    b = {s: v[1] for s, v in baseline_rows().items() if s in PUBLISHED_SEEDS}
    cells = {"full": rung_rows(os.path.join(base, "results", "targeted_dose", "summary.json"),
                               CONTRAST_RUNG),
             "score_only": rung_rows(SCORE, CONTRAST_RUNG),
             "emit_only": rung_rows(EMIT, CONTRAST_RUNG)}
    seeds = sorted(set(b) & set.intersection(*[set(c) for c in cells.values()]))
    if not seeds:
        return {}
    d = {k: np.array([cells[k][s][1] - b[s] for s in seeds]) for k in cells}
    res = d["full"] - (d["score_only"] + d["emit_only"])
    def stat(a):
        return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else float("nan"),
                "per_seed": [float(x) for x in a]}
    return {"n": len(seeds), "seeds": seeds, "contrast_rung": CONTRAST_RUNG,
            "d_asr_full": stat(d["full"]), "d_asr_score_only": stat(d["score_only"]),
            "d_asr_emit_only": stat(d["emit_only"]), "additivity_residual": stat(res),
            "inert_margin": INERT_MARGIN, "additivity_margin": ADDITIVITY_MARGIN,
            "verdict": ("CHANNELS INTERACT"
                        if abs(res.mean()) >= ADDITIVITY_MARGIN
                        and abs(d["emit_only"].mean()) < INERT_MARGIN else "see analyzer"),
            "note": "Reported unchanged alongside the n=20 result whatever the latter is."}


def check_frozen():
    prereg = os.path.join(base, "experiments", "pre_registration_emit_only_topup.md")
    if not os.path.exists(prereg):
        sys.exit(f"REFUSING TO RUN: {prereg} does not exist.")
    if PREREG_COMMIT is None:
        sys.exit("REFUSING TO RUN: the reversal clause and the 'cannot certify' limit are not "
                 f"frozen.\n  1. git commit {prereg}\n"
                 "  2. set PREREG_COMMIT here to that hash.\n"
                 "This suite departs from a frozen non-negotiable of its parent. That is only "
                 "defensible if the departure, its rationale and its limits are themselves committed "
                 "before any seed runs.")
    for path in (EMIT, SCORE):
        if not os.path.exists(path):
            sys.exit(f"REFUSING TO RUN: {path} does not exist. The published n=5 cells this top-up "
                     "extends live there, and the analysis asserts they reproduce before pooling.")
    if not baseline_rows():
        sys.exit("REFUSING TO RUN: no kappa=0 identity rung to import.")
    if not rung_rows(EMIT, CONTRAST_RUNG):
        sys.exit(f"REFUSING TO RUN: no published emit-only cell at kappa={CONTRAST_RUNG}. The "
                 "contrast rung is frozen; it is not relocated to whatever rung exists.")


def harness_check():
    """kappa=0 with each flag set must reproduce the imported identity rung bit-identically.

    At kappa=0 apply_d1_transform returns the update list unwrapped, so there is no transformed stack
    to score on and no untransformed one to emit instead: both flags collapse to plain Krum. This is
    the claim that lets all three cells share one baseline, which is why they are equal seed for seed
    in the published data, and it is asserted here rather than assumed.
    """
    rows = baseline_rows()
    seed = PUBLISHED_SEEDS[0]
    if seed not in rows:
        sys.exit(f"no imported kappa=0 row at seed {seed} to check against")
    r_acc, r_asr, src = rows[seed]
    print("=== HARNESS CHECK: both flags at kappa=0 must BE the imported identity rung ===")
    print(f"    cifar10/cifar_cnn, {D2}/{ATTACK.replace('committed_', '')} at seed {seed}")
    print(f"    imported ({src}): acc={r_acc:.6f} ASR={r_asr:.6f}\n", flush=True)
    ok = True
    for label, flags in LADDERS:
        t = time.time()
        acc, asr = run_one(seed, MODE, D2, ATTACK, BASELINE_RUNG, **flags)
        d = (acc - r_acc, asr - r_asr)
        ok = ok and abs(d[0]) < 1e-9 and abs(d[1]) < 1e-9
        print(f"  {label:11s} acc={acc:.6f} ASR={asr:.6f}   d=({d[0]:+.2e}, {d[1]:+.2e})  "
              f"({time.time() - t:.0f}s)", flush=True)
    print(f"\n  {'BIT-IDENTICAL on both flags. The three cells share one baseline and the import is valid.' if ok else 'NOT IDENTICAL: the three cells do NOT share a baseline and the residual is not a residual. DO NOT RUN.'}")
    return 0 if ok else 1


def load_cells():
    if not os.path.exists(out_path):
        return {}
    try:
        return json.load(open(out_path)).get("cells", {})
    except Exception:
        return {}


def save(cells):
    json.dump({"description": "Seed top-up of the score-only / emit-only factorial at its frozen "
                              f"contrast rung kappa={CONTRAST_RUNG}, seeds 47-61, bringing the "
                              "additivity residual from n=5 to n=20. A DISCLOSED POST-HOC EXTENSION "
                              f"of {PARENT_PREREG_COMMIT}, not a prospective test. Limits frozen at "
                              f"{PREREG_COMMIT} "
                              "(experiments/pre_registration_emit_only_topup.md).",
               "prereg_commit": PREREG_COMMIT,
               "parent_prereg_commit": PARENT_PREREG_COMMIT,
               "departure_from_parent_non_negotiable": DEPARTURE,
               "dataset": "cifar10", "model": "cifar_cnn",
               "config": {"N": 10, "K": 5, "f": 0.2, "alpha": 0.5, "rounds": 50,
                          "contrast_rung": CONTRAST_RUNG, "baseline_rung": BASELINE_RUNG,
                          "new_seeds": NEW_SEEDS, "published_seeds": PUBLISHED_SEEDS,
                          "rho_at_contrast": dial(MODE, CONTRAST_RUNG),
                          "acc_floor": ACC_FLOOR,
                          "inert_margin": INERT_MARGIN,
                          "additivity_margin": ADDITIVITY_MARGIN,
                          "outlier_threshold": OUTLIER_THRESHOLD},
               "arm": {"mode": MODE, "d2": D2, "attack": ATTACK,
                       "attack_impl": ATTACK_MAP[ATTACK],
                       "ladders_run_here": [l for l, _ in LADDERS],
                       "full_cell_source": "results/dose_seed_topup/ (item A); not recomputed here"},
               "cannot_certify": "Projected 95% half-widths at n=20, holding the published sds, are "
                                 "0.061 (dASR_EO) and 0.057 (residual), both wider than the 0.05 "
                                 "margin. A pass of the point-estimate rule is therefore reported as "
                                 "'consistent with separability, not established'.",
               "ladders_remain_at_n5": "The score-only and emit-only ladders across kappa, and the "
                                       "Jonckheere-Terpstra trend test, are NOT restated at n=20. "
                                       "Only the frozen contrast rung is extended.",
               "published_n5_contrast": published_contrast(),
               "cells": cells}, open(out_path, "w"), indent=2)


def main():
    check_frozen()
    os.makedirs(out_dir, exist_ok=True)
    pub = published_contrast()
    total = len(LADDERS) * len(NEW_SEEDS)
    print("=== Factorial seed top-up: score-only and emit-only at the frozen contrast rung ===")
    print(f"    {D2}/{ATTACK.replace('committed_', '')}, kappa={CONTRAST_RUNG} "
          f"(rho={dial(MODE, CONTRAST_RUNG):.2f}), seeds {NEW_SEEDS[0]}-{NEW_SEEDS[-1]} "
          f"({total} runs)")
    print(f"    n=5 -> n=20 on the additivity residual. Limits frozen at {PREREG_COMMIT}.")
    if pub:
        print(f"    published n={pub['n']}: dASR_EO={pub['d_asr_emit_only']['mean']:+.6f} "
              f"(sd {pub['d_asr_emit_only']['sd']:.4f}), "
              f"residual={pub['additivity_residual']['mean']:+.6f} "
              f"(sd {pub['additivity_residual']['sd']:.4f}), margin {ADDITIVITY_MARGIN}")
    print("    DISCLOSED POST-HOC EXTENSION of " + PARENT_PREREG_COMMIT + ", which said "
          "'No seed addition.'")
    print("    n=20 cannot certify separability (projected half-widths 0.061 / 0.057 > 0.05); it "
          "resolves whether seed 42 is alone.\n", flush=True)

    cells = load_cells()
    if cells:
        print(f"  resuming: {sum(len(c['per_seed']) for c in cells.values())} runs already done\n",
              flush=True)
    t0 = time.time(); done = 0
    for label, flags in LADDERS:
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
    print("Run experiments/analyze_emit_only_topup.py, which needs item A's kappa=0 and kappa=1 rows "
          "at these seeds and reports the residual at the n actually shared by all three cells.")
    return 0


if __name__ == "__main__":
    if "--harness-check" in sys.argv:
        check_frozen()
        sys.exit(harness_check())
    sys.exit(main())
