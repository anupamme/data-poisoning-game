"""
POST-HOC PRECISION EXTENSION of the Mode-S Krum flagship contrast. NOT a pre-registered run.

READ THIS FIRST. These seeds were chosen AFTER the n=5 result was known, at a reviewer's request
for more seeds on the paper's flagship cell. That makes this run post-hoc by construction, so it is
declared as such rather than folded into the frozen suite:

  * The PRE-REGISTERED VERDICT REMAINS THE n=5 ONE. results/targeted_dose/summary.json is the scored
    artifact and this script never writes to it; it writes to a sibling directory. Fourteen files in
    this repo read the frozen path by exact name, including the body figure's emitter, so overwriting
    it would silently rewrite published numbers.
  * This is a PRECISION EXTENSION, NOT A RE-TEST. It cannot change whether the arm confirmed,
    refuted, or came out indeterminate. It reports how tight the flagship Delta ASR estimate is when
    the same cell is measured at more seeds, and nothing else.
  * The THRESHOLDS ARE CARRIED OVER UNCHANGED (ACC_FLOOR = 0.35, EQUIV_MARGIN = 0.15), imported from
    run_targeted_dose.py rather than restated. Re-choosing a margin after seeing an outcome is the
    one move that would make the original test unfalsifiable in hindsight.
  * The precedent for declaring extra seeds is already in the frozen suite: cos_krum runs at n=8
    because its identity rung was bimodal, and those seeds were frozen BEFORE any outcome existed
    (run_targeted_dose.py, SEEDS8). This extension is the weaker, post-hoc version of that move and
    is labelled accordingly wherever it is reported.

WHAT IS RUN. One arm, one attack, two rungs, ten new seeds: d2 = krum, committed_scaling, mode S,
kappa in {0.0, 2.0}, seeds 47-56 -> 20 runs. kappa=2.0 is the top of the dose (rho = 54.60) and
kappa=0.0 is the identity rung it is contrasted against, so the two rungs are exactly the endpoints
of the |Delta ASR| the paper reports inside the +-0.15 equivalence margin. No other arm, rung,
attack, aggregator, dataset or mode is touched.

WHY kappa=0.0 IS RUN RATHER THAN IMPORTED. The frozen suite imports its identity rung from
results/dose_response/ because that ladder already held seeds 42-46 for this cell and the two runs
are the same computation (apply_d1_transform returns the update list unwrapped at kappa=0, and
run_one's participant RNG stream does not depend on d1's name -- asserted bit-exactly by
run_targeted_dose.py --harness-check). Seeds 47-56 do not exist in that ladder, so here there is
nothing to import and the rung is computed directly. Same computation, no provenance shortcut.

EVERYTHING IS IMPORTED FROM THE FROZEN SUITE -- run_one, d1_name, dial, the FL config, the
adversarial fraction and both thresholds. Nothing is reimplemented. A copy is how two suites drift
apart in what they compute.

The frozen artifact's SHA-256 is asserted before the first run and after the last, so a claim that
it was untouched is checked rather than promised.

Output: results/targeted_dose_seed_extension/summary.json (resumable; written after every run).

  python3 experiments/run_targeted_dose_seed_extension.py --one     # time a single run first
  python3 experiments/run_targeted_dose_seed_extension.py
"""
import hashlib, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
# Single-sourced from the scored suite: same runner, same config, same thresholds, same rung naming.
from experiments.run_targeted_dose import (  # noqa: E402
    ACC_FLOOR, ADV_FRACTION, EQUIV_MARGIN, FL_CONFIG, PREREG_COMMIT,
    cell_key, d1_name, dial, run_one,
)

D2, ATTACK, MODE = "krum", "committed_scaling", "S"
RUNGS = [0.0, 2.0]                       # the two endpoints of the reported |Delta ASR|
SEEDS_PREREG = [42, 43, 44, 45, 46]      # for reference only; never re-run here
SEEDS_NEW = [47, 48, 49, 50, 51, 52, 53, 54, 55, 56]

FROZEN = os.path.join(base, "results", "targeted_dose", "summary.json")
FROZEN_SHA = None                        # captured at start, re-asserted at end

out_dir = os.path.join(base, "results", "targeted_dose_seed_extension")
out_path = os.path.join(out_dir, "summary.json")


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load():
    if not os.path.exists(out_path):
        return {}
    try:
        return json.load(open(out_path)).get("runs", {})
    except Exception:
        return {}


def save(runs):
    os.makedirs(out_dir, exist_ok=True)
    json.dump({"description":
               "POST-HOC precision extension of the Mode-S krum/committed_scaling flagship contrast: "
               "10 additional seeds at the two rungs the reported |Delta ASR| is taken between. "
               "Chosen after the n=5 outcome was known, at a reviewer's request. The pre-registered "
               "verdict remains the n=5 one in results/targeted_dose/summary.json, which this run "
               "does not modify. Thresholds carried over unchanged, not re-chosen.",
               "status": "post_hoc_precision_extension",
               "scored_artifact": "results/targeted_dose/summary.json",
               "scored_artifact_sha256": FROZEN_SHA,
               "prereg_commit_of_scored_suite": PREREG_COMMIT,
               "arm": {"mode": MODE, "d2": D2, "attack": ATTACK, "rungs": RUNGS,
                       "rhos": {str(k): dial(MODE, k) for k in RUNGS}},
               "seeds_prereg_not_rerun": SEEDS_PREREG,
               "seeds_added_post_hoc": SEEDS_NEW,
               "config": {"N": FL_CONFIG.num_clients, "K": FL_CONFIG.clients_per_round,
                          "f": ADV_FRACTION, "alpha": 0.5, "rounds": FL_CONFIG.num_rounds,
                          "acc_floor": ACC_FLOOR, "equiv_margin": EQUIV_MARGIN},
               "runs": runs}, open(out_path, "w"), indent=2)


def frozen_per_seed():
    """{kappa: {seed: asr}} for the SCORED cells, read-only, so the report can show all three ns."""
    if not os.path.exists(FROZEN):
        return {}
    cells = json.load(open(FROZEN)).get("cells", {})
    out = {}
    for k in RUNGS:
        c = cells.get(cell_key(MODE, D2, ATTACK, k))
        if c:
            out[k] = {r["seed"]: float(r["asr"]) for r in c["per_seed"]}
    return out


def t_ci(xs):
    """Student-t 95% interval on the mean, the same form the paper's bars use."""
    xs = np.asarray(xs, dtype=float)
    n = len(xs)
    m = float(xs.mean())
    if n < 2:
        return m, float("nan"), float("nan")
    se = float(xs.std(ddof=1) / np.sqrt(n))
    try:
        from scipy import stats
        t = float(stats.t.ppf(0.975, n - 1))
    except Exception:                                  # no scipy: table lookup, df 1..30 then normal
        TAB = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
               9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
               16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
               23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
               30: 2.042}
        t = TAB.get(n - 1, 1.960)
    return m, m - t * se, m + t * se


def delta_ci(hi_rung, lo_rung):
    """Delta ASR = top rung - identity rung, SEED-MATCHED and paired.

    Paired, not two-sample: the paper's contrast is a seed-matched intervention on one cell, so the
    seed is a block and the difference is taken within it. Only seeds present at BOTH rungs count.
    """
    seeds = sorted(set(hi_rung) & set(lo_rung))
    d = [hi_rung[s] - lo_rung[s] for s in seeds]
    m, lo, hi = t_ci(d)
    return len(seeds), m, lo, hi, seeds


def report(runs):
    """Print all three estimates. The appendix text is transcribed from THIS output, never memory."""
    frozen = frozen_per_seed()
    new = {k: {} for k in RUNGS}
    for key, r in runs.items():
        new[float(r["kappa"])][int(r["seed"])] = float(r["asr"])

    print("\n=== FLAGSHIP CONTRAST: Mode S, krum / committed_scaling, ASR(kappa=2) - ASR(kappa=0) ===")
    print("    Seed-matched paired differences. rho = 54.60 at the top rung, 1.00 at the identity.\n")
    rows = [("pre-registered (n=5, SCORED)", frozen.get(2.0, {}), frozen.get(0.0, {})),
            ("post-hoc extension only", new[2.0], new[0.0]),
            ("pooled (pre-reg + post-hoc)", {**frozen.get(2.0, {}), **new[2.0]},
             {**frozen.get(0.0, {}), **new[0.0]})]
    print(f"  {'estimate':30s} {'n':>3s} {'mean ASR hi':>12s} {'mean ASR lo':>12s} "
          f"{'Delta':>8s} {'95% CI':>20s}")
    print("  " + "-" * 92)
    for lab, hi, lo in rows:
        if not hi or not lo:
            print(f"  {lab:30s} {'--':>3s}   not yet measured")
            continue
        n, m, l, h = delta_ci(hi, lo)[:4]
        mh = float(np.mean([hi[s] for s in sorted(set(hi) & set(lo))]))
        ml = float(np.mean([lo[s] for s in sorted(set(hi) & set(lo))]))
        print(f"  {lab:30s} {n:3d} {mh:12.4f} {ml:12.4f} {m:+8.4f} "
              f"{'[' + f'{l:+.4f}, {h:+.4f}' + ']':>20s}")

    hi_all = {**frozen.get(2.0, {}), **new[2.0]}
    lo_all = {**frozen.get(0.0, {}), **new[0.0]}
    if hi_all and lo_all:
        n, m, l, h = delta_ci(hi_all, lo_all)[:4]
        inside = abs(l) <= EQUIV_MARGIN and abs(h) <= EQUIV_MARGIN
        print(f"\n  equivalence margin +-{EQUIV_MARGIN} (carried over, NOT re-chosen)")
        print(f"  pooled n={n}: |Delta| = {abs(m):.4f}, CI {'INSIDE' if inside else 'NOT INSIDE'} "
              f"the margin")
        if not inside:
            print("  >>> The pooled interval leaves the margin. REPORT THIS AS THE FINDING. The "
                  "pre-registered\n      n=5 verdict still stands as the scored result; this "
                  "extension says the estimate is less\n      precise than n=5 suggested, and that "
                  "is what the appendix must say.")
    print("\n  The SCORED verdict is the n=5 row. The other two rows are post-hoc precision, "
          "declared as such.")


def main():
    global FROZEN_SHA
    if not os.path.exists(FROZEN):
        sys.exit(f"REFUSING TO RUN: {FROZEN} is missing; there is nothing to extend.")
    FROZEN_SHA = sha256(FROZEN)
    print(f"scored artifact SHA-256 at start: {FROZEN_SHA}")

    only_one = "--one" in sys.argv
    runs = load()
    todo = [(k, s) for k in RUNGS for s in SEEDS_NEW
            if f"{d1_name(MODE, k)}|{D2}|{ATTACK}|{s}" not in runs]
    if only_one:
        todo = todo[:1]
    print(f"{len(runs)} run(s) already on disk; {len(todo)} to run"
          f"{' (--one: timing a single run)' if only_one else ''}.\n")

    for i, (k, seed) in enumerate(todo, 1):
        key = f"{d1_name(MODE, k)}|{D2}|{ATTACK}|{seed}"
        t0 = time.time()
        acc, asr = run_one(seed, MODE, D2, ATTACK, k)
        wall = time.time() - t0
        runs[key] = {"mode": MODE, "d2": D2, "attack": ATTACK, "kappa": k,
                     "rho": dial(MODE, k), "seed": seed, "accuracy": acc, "asr": asr,
                     "wall_time_s": wall, "post_hoc": True}
        save(runs)
        print(f"  [{i}/{len(todo)}] kappa={k} seed={seed}  acc={acc:.4f}  asr={asr:.4f}  "
              f"{wall/60:.1f} min"
              + ("  <<< BELOW ACC FLOOR" if acc < ACC_FLOOR else ""))
        if only_one:
            print(f"\n  one run took {wall/60:.1f} min -> 20 runs ~ {20*wall/3600:.2f} h "
                  f"({19*wall/3600:.2f} h remaining).")

    if sha256(FROZEN) != FROZEN_SHA:
        sys.exit("FROZEN ARTIFACT CHANGED DURING THIS RUN. This script must never write it.")
    print(f"scored artifact SHA-256 at end:   {FROZEN_SHA}  (unchanged)")
    report(runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
