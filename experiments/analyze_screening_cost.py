"""
Run-level cost accounting for the screen, replacing the pair-level "95.2%" claim.

WHY THIS EXISTS. The paper previously reported that C1&C2&C3 admits 2 of 42 ordered pairs, so
"40/42 = 95.2% of ordered compositions need no composition-level evaluation". That number is
misleading in a specific way: it counts PAIRS and silently ignores the standalone evaluations C1
itself requires. C1 is an empirical gate -- it is read off measured single-defense ASR -- so a
screening protocol does not start from zero runs. The honest unit is the RUN.

WHAT IS COUNTED. A "cell" is one (composition, attack) pair; a "run" is one 50-round federated
training at one seed. With D defenses there are D(D-1) ordered pairs.

  unscreened   evaluate every ordered pair on every committed attack
               = D(D-1) * |attacks| cells                     -- QUADRATIC in D
  screened     evaluate every defense standalone (the C1 input), then only the admitted pairs
               = (D-1) * |attacks| + |admitted| * |attacks|    -- LINEAR in D plus the admitted set

The standalone term is (D-1), not D, because fedavg-as-upstream IS the standalone protocol
(fedavg_then_X == X alone, the same identity used by analyze_condition_ablation.py) and fedavg's own
standalone number is the undefended baseline, which any evaluation has already.

The admitted set is recomputed here from results/condition_ablation/summary.json under the criterion
this round recommends, C1&C2, and under the frozen C1&C2&C3 -- they coincide, which is why the
demotion changes no number.

TIMING IS NOW MEASURED, AND THE OLD CAVEAT WAS WRONG. Through Round 51 this script and the paper
both said "no per-run timing telemetry exists in any results file", and the GPU-hour figures were an
assumption at MIN_PER_RUN = 11.0. No results JSON does carry a wall-clock field -- but the RUNNER
LOGS in results/*.log print per-run seconds on every line ("s45: acc=... ASR=... (1231s)"), and
results/*.log are results files. So the caveat was true of one artifact type and false as stated.
The distribution over the pair-evaluation logs (PAIR_EVAL_LOGS below, n=228) has median 11.1 min,
which is why the assumed 11.0 held up; the mean is 13.1 min because the tail is long (max 30.6),
so both are reported and the median carries the headline. EMNIST arms run 52 min/run and are
excluded, because the 420/80 run counts this file accounts for are CIFAR-10 pair evaluations.

  Lesson worth keeping: an assumption that turns out to be right is still worth replacing with the
  measurement, because the paper was paying a credibility cost for a number it could have measured.

ONE CAVEAT REMAINS. The standalone baselines are not pure screening overhead -- they have
independent value (they are the single-defense results the paper reports anyway). Counting them as
screening cost is the conservative choice.

No new compute. Output: results/screening_cost.json
"""
import json
import os
import re
import statistics
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)
os.chdir(base_dir)

ATTACKS = ("committed_scaling", "committed_pixel")
SEEDS_PER_CELL = 5      # the headline protocol's seed count
MIN_PER_RUN_ASSUMED = 11.0   # the pre-Round-52 assumption, kept only for the comparison

# The logs whose runs are CIFAR-10 pair evaluations at the headline config, which is what the
# 420-run and 80-run counts below are counting. Dose ladders and the comparability harness run
# heavier configs and the femnist arms are a different dataset, so neither belongs in this pool.
PAIR_EVAL_LOGS = ("results/prospective_suite_run.log", "results/prospective_pilot_run.log",
                  "results/metric_swap_suite.log", "results/metric_swap_baselines.log",
                  "results/run_wave3_emergent.log", "results/headline_seed_topup.log")
RUN_SECONDS = re.compile(r"\((\d{2,5})s\)")

ABLATION = "results/condition_ablation/summary.json"
PAIR_FILES = ("results/all_compositions/summary.json", "results/wave2_held_out/summary.json")


def measured_minutes_per_run():
    """Per-run wall clock, measured from the runner logs' own per-run seconds.

    Returns None if no log carries timings, so the caller degrades to the assumption rather than
    inventing one.
    """
    secs, per_log = [], {}
    for path in PAIR_EVAL_LOGS:
        if not os.path.exists(path):
            continue
        v = [int(x) for x in RUN_SECONDS.findall(open(path, errors="ignore").read())]
        if v:
            per_log[os.path.basename(path)] = {"runs": len(v),
                                               "median_min": round(statistics.median(v) / 60, 2)}
            secs += v
    if not secs:
        return None
    secs.sort()
    return {
        "source": "per-run seconds printed by the runners into results/*.log",
        "logs": per_log,
        "runs_timed": len(secs),
        "median_min": statistics.median(secs) / 60,
        "mean_min": statistics.mean(secs) / 60,
        "iqr_min": [secs[len(secs) // 4] / 60, secs[3 * len(secs) // 4] / 60],
        "min_min": secs[0] / 60,
        "max_min": secs[-1] / 60,
        "excluded": "EMNIST-byclass arms (52 min/run) and the dose ladders, which are not "
                    "CIFAR-10 pair evaluations at the headline config",
    }


def load_rows():
    return json.load(open(ABLATION))["rows"]


def admitted(rows, conditions):
    return sorted(r["pair"] for r in rows if all(r[c] for c in conditions))


def actual_runs_spent():
    """Composition-level runs actually spent in this paper, counted per seed from the results files.

    This is what we DID, not the counterfactual: seed counts per cell vary (3 for most tiers, 5 for
    the headline), so it is reported alongside the fixed-budget counterfactual rather than folded
    into it.
    """
    runs, cells = 0, 0
    seen = set()
    for path in PAIR_FILES:
        for k, v in json.load(open(path))["pairs"].items():
            if k in seen:
                continue
            seen.add(k)
            for a in ATTACKS:
                if a in v:
                    cells += 1
                    runs += len(v[a]["per_seed"])
    return {"pairs_with_results": len(seen), "cells": cells, "runs": runs}


def main():
    rows = load_rows()
    defenses = sorted({r["d1"] for r in rows} | {r["d2"] for r in rows})
    D = len(defenses)
    n_pairs = len(rows)
    assert n_pairs == D * (D - 1), f"{n_pairs} rows but D(D-1) = {D * (D - 1)}"

    timing = measured_minutes_per_run()
    mpr = timing["median_min"] if timing else MIN_PER_RUN_ASSUMED
    mpr_mean = timing["mean_min"] if timing else MIN_PER_RUN_ASSUMED

    adm_c12 = admitted(rows, ("C1", "C2"))
    adm_c123 = admitted(rows, ("C1", "C2", "C3"))

    A = len(ATTACKS)
    S = SEEDS_PER_CELL
    n_standalone = D - 1                       # fedavg-as-d1 is the standalone protocol

    unscreened_cells = n_pairs * A
    screened_cells = n_standalone * A + len(adm_c12) * A
    unscreened_runs = unscreened_cells * S
    screened_runs = screened_cells * S

    out = {
        "description": "Run-level cost of the screen. Replaces the pair-level 95.2% figure, which "
                       "ignored the standalone evaluations C1 requires. Every number here is "
                       "computed from the results files, and the GPU-hour figures are now MEASURED "
                       "from the per-run seconds the runners print into results/*.log rather than "
                       "assumed at 11.0 min/run.",
        "defenses": defenses,
        "n_defenses": D,
        "n_ordered_pairs": n_pairs,
        "attacks": list(ATTACKS),
        "seeds_per_cell": S,
        "admitted": {"C1_and_C2": adm_c12, "C1_and_C2_and_C3": adm_c123,
                     "identical": adm_c12 == adm_c123},
        "cells": {"unscreened": unscreened_cells, "screened": screened_cells,
                  "standalone_component": n_standalone * A,
                  "admitted_component": len(adm_c12) * A},
        "runs": {"unscreened": unscreened_runs, "screened": screened_runs,
                 "avoided": unscreened_runs - screened_runs,
                 "fraction_avoided": (unscreened_runs - screened_runs) / unscreened_runs,
                 "standalone_component": n_standalone * A * S,
                 "admitted_component": len(adm_c12) * A * S},
        "pair_level_figure_for_contrast": {
            "pairs_not_composition_evaluated": n_pairs - len(adm_c12),
            "fraction": (n_pairs - len(adm_c12)) / n_pairs,
            "note": "This is the old 95.2%-style figure. It is a pair count and excludes the "
                    "standalone runs C1 needs, so it overstates the saving."},
        "gpu_hours": {
            "minutes_per_run_measured_median": mpr,
            "minutes_per_run_assumed_before_round52": MIN_PER_RUN_ASSUMED,
            "timing": timing,
            "unscreened": unscreened_runs * mpr / 60.0,
            "screened": screened_runs * mpr / 60.0,
            "saved": (unscreened_runs - screened_runs) * mpr / 60.0,
            "saved_at_mean": (unscreened_runs - screened_runs) * mpr_mean / 60.0,
            "caveat": "Measured, not assumed: per-run seconds come from the runner logs. The median "
                      "carries the figure and the mean is reported beside it because the per-run "
                      "distribution has a long right tail. Hardware is the single machine the paper "
                      "describes, so these are wall-clock hours on it and not a portable cost."},
        "scaling": {
            "unscreened": "|attacks| * seeds * D(D-1)  -- quadratic in D",
            "screened": "|attacks| * seeds * ((D-1) + |admitted|)  -- linear in D plus the "
                        "admitted set",
            "note": "The screen removes the quadratic term. It does not remove experimentation: "
                    "C1 is an empirical gate, so the standalone runs are unavoidable."},
        "actually_spent_in_this_paper": actual_runs_spent(),
    }

    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/screening_cost.json", "w"), indent=2)

    print("=== SCREENING COST, PER RUN ===")
    print(f"  {D} defenses -> {n_pairs} ordered pairs, {A} committed attacks, {S} seeds/cell")
    print(f"  admitted by C1&C2:      {adm_c12}")
    print(f"  admitted by C1&C2&C3:   {adm_c123}"
          f"   {'(identical -- the C3 demotion changes no number)' if adm_c12 == adm_c123 else '(DIFFER)'}")
    print()
    print(f"  unscreened: {unscreened_cells:>3d} cells = {unscreened_runs:>4d} runs")
    print(f"  screened:   {screened_cells:>3d} cells = {screened_runs:>4d} runs"
          f"  ({n_standalone * A * S} standalone + {len(adm_c12) * A * S} admitted-pair)")
    print(f"  avoided:    {unscreened_runs - screened_runs:>4d} runs "
          f"= {100 * out['runs']['fraction_avoided']:.1f}% of runs")
    print(f"  old pair-level figure for contrast: "
          f"{100 * out['pair_level_figure_for_contrast']['fraction']:.1f}% of PAIRS "
          f"(overstates: excludes C1's standalone runs)")
    print()
    g = out["gpu_hours"]
    if timing:
        print(f"  measured per-run wall clock over {timing['runs_timed']} logged pair-evaluation "
              f"runs: median {timing['median_min']:.2f} min, mean {timing['mean_min']:.2f}, "
              f"IQR {timing['iqr_min'][0]:.1f}--{timing['iqr_min'][1]:.1f}, "
              f"range {timing['min_min']:.1f}--{timing['max_min']:.1f}")
    else:
        print(f"  NO LOG TIMINGS FOUND; falling back to the {MIN_PER_RUN_ASSUMED:g} min/run assumption")
    print(f"  GPU-hours at the measured median {mpr:.2f} min/run: "
          f"{g['unscreened']:.0f} h -> {g['screened']:.0f} h, saving {g['saved']:.0f} h "
          f"({g['saved_at_mean']:.0f} h at the mean)")
    print(f"  for contrast, the pre-Round-52 assumption of {MIN_PER_RUN_ASSUMED:g} min/run gave "
          f"{(unscreened_runs - screened_runs) * MIN_PER_RUN_ASSUMED / 60.0:.1f} h saved")
    print()
    sp = out["actually_spent_in_this_paper"]
    print(f"  actually spent on composition cells: {sp['runs']} runs over {sp['cells']} cells "
          f"({sp['pairs_with_results']} pairs; seed counts vary by tier)")
    print("\nSaved to results/screening_cost.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
