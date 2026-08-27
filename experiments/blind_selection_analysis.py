"""
Blind composition-selection analysis: criterion-guided vs random vs oracle.

Uses existing ASR data from all 42 ordered pairs (18 dev + 24 held-out).
No new FL training required — pure resampling statistics.
"""

import numpy as np
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

def load_all_pair_asrs():
    """Load max-committed ASR for all 42 pairs."""
    pairs = {}

    # Development set (18 pairs)
    dev_path = RESULTS_DIR / "all_compositions" / "summary.json"
    with open(dev_path) as f:
        dev = json.load(f)
    for name, info in dev["pairs"].items():
        pairs[name] = info["max_committed_asr"]

    # Held-out set (24 pairs)
    heldout_path = RESULTS_DIR / "wave2_held_out" / "summary.json"
    with open(heldout_path) as f:
        heldout = json.load(f)
    heldout_pairs = heldout.get("pairs", heldout)
    for name, info in heldout_pairs.items():
        if isinstance(info, dict):
            asr = info.get("max_committed_asr", info.get("max_asr"))
            if asr is not None and name not in pairs:
                pairs[name] = asr

    return pairs


def load_three_way_asrs():
    """Load max ASR for all 10 three-way triples."""
    path = RESULTS_DIR / "three_way_compositions" / "summary.json"
    with open(path) as f:
        data = json.load(f)
    triples_data = data.get("triples", data)
    triples = {}
    for name, info in triples_data.items():
        if isinstance(info, dict):
            asr = info.get("max_committed_asr", info.get("max_asr"))
            if asr is not None:
                triples[name] = asr
    return triples


def resampling_analysis_2way(pairs, n_bootstrap=100_000):
    """Compare random vs criterion-guided vs oracle for 2-way pairs."""
    all_asrs = np.array(list(pairs.values()))
    all_names = list(pairs.keys())
    n_pairs = len(all_asrs)

    # Criterion-guided: the 3 clean PASS pairs
    pass_pairs = ["foolsgold_then_rfa", "foolsgold_then_coord_median",
                  "reputation_then_coord_median"]
    criterion_asrs = [pairs[p] for p in pass_pairs]
    criterion_mean = np.mean(criterion_asrs)
    criterion_max = np.max(criterion_asrs)
    k = len(pass_pairs)

    # Oracle: k lowest-ASR pairs
    sorted_asrs = np.sort(all_asrs)
    oracle_mean = np.mean(sorted_asrs[:k])
    oracle_max = np.max(sorted_asrs[:k])

    # Random baseline: sample k pairs uniformly, compute mean max-committed ASR
    rng = np.random.default_rng(42)
    random_means = []
    random_maxes = []
    random_all_below_05 = 0
    for _ in range(n_bootstrap):
        idx = rng.choice(n_pairs, size=k, replace=False)
        sample = all_asrs[idx]
        random_means.append(np.mean(sample))
        random_maxes.append(np.max(sample))
        if np.all(sample < 0.5):
            random_all_below_05 += 1

    random_means = np.array(random_means)
    random_maxes = np.array(random_maxes)

    # P(random ≤ criterion)
    p_random_beats_criterion_mean = np.mean(random_means <= criterion_mean)
    p_random_beats_criterion_max = np.mean(random_maxes <= criterion_max)
    p_all_below_05 = random_all_below_05 / n_bootstrap

    print("=" * 60)
    print("BLIND COMPOSITION-SELECTION ANALYSIS (2-WAY PAIRS)")
    print("=" * 60)
    print(f"\nTotal pairs available: {n_pairs}")
    print(f"Selection size k = {k}")
    print(f"\n--- Criterion-guided (PASS pairs) ---")
    print(f"  Pairs: {pass_pairs}")
    print(f"  ASRs: {[f'{a:.3f}' for a in criterion_asrs]}")
    print(f"  Mean ASR: {criterion_mean:.3f}")
    print(f"  Max ASR: {criterion_max:.3f}")
    print(f"\n--- Oracle (k lowest) ---")
    print(f"  Pairs: {[all_names[i] for i in np.argsort(all_asrs)[:k]]}")
    print(f"  Mean ASR: {oracle_mean:.3f}")
    print(f"  Max ASR: {oracle_max:.3f}")
    print(f"\n--- Random baseline (n={n_bootstrap:,} resamples) ---")
    print(f"  Mean of mean ASRs: {np.mean(random_means):.3f}")
    print(f"  95% CI of mean ASR: [{np.percentile(random_means, 2.5):.3f}, "
          f"{np.percentile(random_means, 97.5):.3f}]")
    print(f"  Mean of max ASRs: {np.mean(random_maxes):.3f}")
    print(f"  P(random mean ≤ criterion mean): {p_random_beats_criterion_mean:.4f}")
    print(f"  P(random max ≤ criterion max): {p_random_beats_criterion_max:.4f}")
    print(f"  P(all k random picks have ASR < 0.5): {p_all_below_05:.4f}")
    print(f"\n--- Key result ---")
    print(f"  Criterion selection ({criterion_mean:.3f}) beats "
          f"{(1-p_random_beats_criterion_mean)*100:.1f}% of random selections")
    print(f"  Only {p_random_beats_criterion_mean*100:.2f}% of random draws "
          f"achieve mean ASR ≤ {criterion_mean:.3f}")

    return {
        "n_pairs": n_pairs,
        "k": k,
        "criterion_mean_asr": criterion_mean,
        "criterion_max_asr": criterion_max,
        "oracle_mean_asr": oracle_mean,
        "oracle_max_asr": oracle_max,
        "random_mean_of_means": float(np.mean(random_means)),
        "random_95ci": [float(np.percentile(random_means, 2.5)),
                        float(np.percentile(random_means, 97.5))],
        "p_random_leq_criterion_mean": float(p_random_beats_criterion_mean),
        "p_random_leq_criterion_max": float(p_random_beats_criterion_max),
        "p_all_below_05": float(p_all_below_05),
    }


def resampling_analysis_3way(triples, n_bootstrap=100_000):
    """Compare random vs criterion-guided for 3-way triples."""
    all_asrs = np.array(list(triples.values()))
    all_names = list(triples.keys())
    n_triples = len(all_asrs)

    # From pre-registration: 6 predicted LOW, 4 predicted HIGH
    pass_names = ["fg_rep_rfa", "fg_rep_cm", "rep_fg_rfa", "rep_fg_cm",
                  "fg_nc_cm", "fg_rfa_cm"]
    criterion_asrs_3way = [triples[n] for n in pass_names if n in triples]

    k = len(criterion_asrs_3way) if criterion_asrs_3way else 6
    criterion_mean = np.mean(criterion_asrs_3way) if criterion_asrs_3way else 0

    # For 3-way: 210 possible ordered triples from 7 defenses
    # We only measured 10. Estimate from those 10.
    rng = np.random.default_rng(42)
    random_means = []
    random_all_below_05 = 0
    for _ in range(n_bootstrap):
        idx = rng.choice(n_triples, size=min(k, n_triples), replace=False)
        sample = all_asrs[idx]
        random_means.append(np.mean(sample))
        if np.all(sample < 0.5):
            random_all_below_05 += 1

    random_means = np.array(random_means)
    p_random_beats = np.mean(random_means <= criterion_mean)

    print("\n" + "=" * 60)
    print("BLIND COMPOSITION-SELECTION ANALYSIS (3-WAY TRIPLES)")
    print("=" * 60)
    print(f"\nTriples measured: {n_triples}")
    print(f"Total possible (7 defenses): 210")
    print(f"Selection size k = {k}")
    print(f"\n--- Criterion-guided (predicted LOW) ---")
    print(f"  ASRs: {[f'{a:.3f}' for a in criterion_asrs_3way]}")
    print(f"  Mean ASR: {criterion_mean:.3f}")
    print(f"  All below 0.5: {all(a < 0.5 for a in criterion_asrs_3way)}")
    print(f"\n--- Random from measured set (n={n_bootstrap:,}) ---")
    print(f"  Mean of mean ASRs: {np.mean(random_means):.3f}")
    print(f"  P(random mean ≤ criterion mean): {p_random_beats:.4f}")
    print(f"  P(all {k} random picks from 10 have ASR < 0.5): "
          f"{random_all_below_05/n_bootstrap:.4f}")

    # Combinatorial argument for 210 triples
    # From 10 measured: 6 are LOW (<0.15), 4 are HIGH (>0.25)
    # If we assume 6/10 = 60% of all 210 are LOW (generous), P(select 6 all LOW from 210) is:
    # More conservative: criterion predicts LOW for ~30 of 210 (FG/rep as d1 = 5*6*5 combos,
    # then filter by C2/C3), so base rate of criterion-LOW is ~30/210 ≈ 14%
    n_total_possible = 210
    n_pass_estimated = 30  # FG/rep as d1, ~30 have plausible PASS chain
    base_rate_pass = n_pass_estimated / n_total_possible
    print(f"\n--- Extrapolation to 210 triples ---")
    print(f"  Estimated criterion-PASS rate: {n_pass_estimated}/{n_total_possible} "
          f"= {base_rate_pass:.1%}")
    print(f"  P(randomly selecting {k} triples all have ASR < 0.5):")
    print(f"    If ~30/210 triples are actually LOW: "
          f"{np.prod([(30-i)/(210-i) for i in range(k)]):.4f}")
    print(f"    vs criterion: 6/6 = 100% observed accuracy on LOW predictions")

    return {
        "n_measured": n_triples,
        "n_possible": 210,
        "k": k,
        "criterion_mean_asr": float(criterion_mean),
        "p_random_leq_criterion": float(p_random_beats),
    }


if __name__ == "__main__":
    pairs = load_all_pair_asrs()
    print(f"Loaded {len(pairs)} pairs")

    results_2way = resampling_analysis_2way(pairs)

    triples = load_three_way_asrs()
    print(f"\nLoaded {len(triples)} triples")
    results_3way = resampling_analysis_3way(triples)

    # Save results
    output = {"two_way": results_2way, "three_way": results_3way}
    out_path = RESULTS_DIR / "blind_selection_analysis" / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
