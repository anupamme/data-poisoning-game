"""
Analyze composability of defense pairs using the Signal-Preservation
Composability Criterion (Theorem 2).

Reads:
  - results/all_compositions/summary.json (empirical composition ASR for all 21 pairs)
  - results/pure_defense_baselines/summary.json (pure defense ASR baselines)

Applies the criterion to predict which compositions work well vs poorly,
then reports the confusion matrix.
"""

import json
import os
import sys
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

# ==============================================================================
# Defense type taxonomy
# ==============================================================================

DEFENSE_TYPES = {
    "reputation": "weighting",
    "foolsgold": "weighting",
    "norm_clip": "clipping",
    "trimmed_mean": "rank_based",
    "coord_median": "rank_based",
    "rfa": "geometric",
    "fedavg": "noop",
}

# ==============================================================================
# Signal-Preservation Matrix
# ==============================================================================
# For each (d1_type, d2_type) pair, does d2's transformation preserve d1's
# suppression signal?
#
# Values: "PASS", "FAIL", "DEGENERATE"
# DEGENERATE means d1 produces a single aggregate (rank-based) or does nothing
# (noop), so the composition degenerates to just d2.

SIGNAL_PRESERVATION = {
    # d1=Weighting: keys on distance-from-consensus
    ("weighting", "clipping"): "FAIL",        # clipping normalizes magnitudes -> equidistant
    ("weighting", "rank_based"): "PASS",      # trimming coordinates doesn't change consensus distance
    ("weighting", "geometric"): "PASS",       # reweighting doesn't modify update vectors
    ("weighting", "noop"): "PASS",            # no modification
    ("weighting", "weighting"): "PASS",       # second weighting modifies weights, not vectors

    # d1=Clipping: keys on update L2 norm
    ("clipping", "weighting"): "FAIL",        # all clipped to same norm -> equidistant from consensus
    ("clipping", "rank_based"): "PASS",       # rank statistics unaffected by uniform scaling
    ("clipping", "geometric"): "PASS",        # partial, treated as PASS
    ("clipping", "noop"): "PASS",             # no modification
    ("clipping", "clipping"): "PASS",         # second clip is redundant but harmless

    # d1=Rank-based: produces single aggregate, can't meaningfully compose as d1
    ("rank_based", "weighting"): "DEGENERATE",
    ("rank_based", "rank_based"): "DEGENERATE",
    ("rank_based", "geometric"): "DEGENERATE",
    ("rank_based", "noop"): "DEGENERATE",
    ("rank_based", "clipping"): "DEGENERATE",

    # d1=No-op: no transformation
    ("noop", "weighting"): "DEGENERATE",
    ("noop", "rank_based"): "DEGENERATE",
    ("noop", "geometric"): "DEGENERATE",
    ("noop", "noop"): "DEGENERATE",
    ("noop", "clipping"): "DEGENERATE",

    # d1=Geometric: keys on distance-from-geometric-median
    ("geometric", "weighting"): "PASS",       # weighting doesn't modify vectors
    ("geometric", "clipping"): "PASS",        # partial, treated as PASS
    ("geometric", "rank_based"): "PASS",      # rank stats preserve geometric median structure
    ("geometric", "noop"): "PASS",            # no modification
    ("geometric", "geometric"): "PASS",       # second geometric re-weights independently
}

# ==============================================================================
# Thresholds
# ==============================================================================

# C1 threshold: pure ASR below this means the defense suppresses the attack
SUPPRESSION_THRESHOLD = 0.3

# Actual outcome threshold: max_committed_asr below this means composition "works"
WORKS_THRESHOLD = 0.5

# Attack types we check for committed attacks
COMMITTED_ATTACKS = ["committed_scaling", "committed_pixel"]


def load_json(path):
    """Load a JSON file, returning None if not found."""
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def compute_pure_defense_asr(baselines_data):
    """
    From pure defense baselines, compute mean ASR for each defense x attack.

    Returns:
        dict: {defense_name: {attack_name: mean_asr}}
    """
    result = {}
    defenses = baselines_data.get("defenses", {})

    for defense_name, defense_data in defenses.items():
        result[defense_name] = {}
        adversary_policies = defense_data.get("adversary_policies", {})

        for attack_name, attack_data in adversary_policies.items():
            per_seed = attack_data.get("per_seed", [])
            if per_seed:
                asrs = [s.get("asr", 0.0) for s in per_seed if "asr" in s]
                if asrs:
                    result[defense_name][attack_name] = float(np.mean(asrs))
                else:
                    result[defense_name][attack_name] = None
            else:
                result[defense_name][attack_name] = None

    return result


def get_defense_type(defense_name):
    """Get the defense type category for a defense."""
    return DEFENSE_TYPES.get(defense_name, "unknown")


def check_signal_preservation(d1_name, d2_name):
    """
    Check signal preservation for d1 -> d2 composition.

    Returns: "PASS", "FAIL", or "DEGENERATE"
    """
    d1_type = get_defense_type(d1_name)
    d2_type = get_defense_type(d2_name)

    key = (d1_type, d2_type)
    return SIGNAL_PRESERVATION.get(key, "UNKNOWN")


def predict_composition(d1_name, d2_name, pure_asr, attack_name):
    """
    Apply the Composability Criterion for a specific composition and attack.

    Returns:
        (prediction, reason)
        prediction: "LOW" (works) or "HIGH" (fails)
        reason: explanation string
    """
    # Get pure ASR for each defense individually against this attack
    d1_asr = pure_asr.get(d1_name, {}).get(attack_name, None)
    d2_asr = pure_asr.get(d2_name, {}).get(attack_name, None)

    # If we don't have baseline data, can't predict
    if d1_asr is None and d2_asr is None:
        return "UNKNOWN", "missing baseline data for both defenses"

    # C1: At least one defense individually suppresses the attack
    d1_suppresses = d1_asr is not None and d1_asr < SUPPRESSION_THRESHOLD
    d2_suppresses = d2_asr is not None and d2_asr < SUPPRESSION_THRESHOLD
    c1_holds = d1_suppresses or d2_suppresses

    if not c1_holds:
        return "HIGH", (
            f"C1 fails: neither defense suppresses "
            f"(d1={d1_asr:.3f}, d2={d2_asr:.3f})"
            if d1_asr is not None and d2_asr is not None
            else "C1 fails: insufficient suppression"
        )

    # C2: Signal preservation
    signal_status = check_signal_preservation(d1_name, d2_name)

    if signal_status == "FAIL":
        return "HIGH", (
            f"C2 fails: {get_defense_type(d2_name)} destroys "
            f"{get_defense_type(d1_name)}'s signal"
        )

    if signal_status == "DEGENERATE":
        # Composition degenerates to d2 only
        # Prediction depends on whether d2 alone suppresses
        if d2_suppresses:
            return "LOW", f"DEGENERATE: reduces to d2 ({d2_name}), which suppresses"
        else:
            return "HIGH", f"DEGENERATE: reduces to d2 ({d2_name}), which does NOT suppress"

    if signal_status == "PASS":
        return "LOW", "C1+C2 both hold: composition preserves suppression signal"

    # Unknown combination
    return "UNKNOWN", f"unknown type combination: {get_defense_type(d1_name)} -> {get_defense_type(d2_name)}"


def predict_pair(d1_name, d2_name, pure_asr):
    """
    Predict whether a composition pair works (low max ASR) across all attacks.

    Returns:
        (overall_prediction, per_attack_details)
        overall_prediction: "LOW" if ALL attacks predicted LOW, else "HIGH"
    """
    per_attack = {}
    any_high = False

    for attack in COMMITTED_ATTACKS:
        pred, reason = predict_composition(d1_name, d2_name, pure_asr, attack)
        per_attack[attack] = {"prediction": pred, "reason": reason}
        if pred == "HIGH":
            any_high = True

    overall = "HIGH" if any_high else "LOW"
    return overall, per_attack


def main():
    # Paths
    compositions_path = os.path.join(base_dir, "results", "all_compositions", "summary.json")
    baselines_path = os.path.join(base_dir, "results", "pure_defense_baselines", "summary.json")

    # Load data
    compositions_data = load_json(compositions_path)
    baselines_data = load_json(baselines_path)

    if compositions_data is None:
        print(f"WARNING: Could not load {compositions_path}")
        print("Cannot proceed without composition results.")
        sys.exit(1)

    if baselines_data is None:
        print(f"WARNING: Could not load {baselines_path}")
        print("Cannot proceed without pure defense baselines.")
        sys.exit(1)

    # Compute pure defense ASR
    pure_asr = compute_pure_defense_asr(baselines_data)

    print("=" * 80)
    print("SIGNAL-PRESERVATION COMPOSABILITY CRITERION ANALYSIS (Theorem 2)")
    print("=" * 80)
    print()

    # Print pure defense baselines summary
    print("-" * 60)
    print("Pure Defense Baselines (mean ASR per defense x attack)")
    print("-" * 60)
    print(f"{'Defense':<15} {'committed_scaling':<20} {'committed_pixel':<20}")
    print("-" * 60)
    for defense in sorted(pure_asr.keys()):
        scaling = pure_asr[defense].get("committed_scaling", None)
        pixel = pure_asr[defense].get("committed_pixel", None)
        scaling_str = f"{scaling:.3f}" if scaling is not None else "N/A"
        pixel_str = f"{pixel:.3f}" if pixel is not None else "N/A"
        print(f"{defense:<15} {scaling_str:<20} {pixel_str:<20}")
    print()

    # Process each pair
    pairs_data = compositions_data.get("pairs", {})
    results = []

    print("-" * 80)
    print("Per-Pair Predictions")
    print("-" * 80)
    print(f"{'Pair':<35} {'Predicted':<10} {'Actual ASR':<12} {'Outcome':<10} {'Correct':<8}")
    print("-" * 80)

    for pair_name, pair_info in sorted(pairs_data.items()):
        d1 = pair_info.get("d1", "")
        d2 = pair_info.get("d2", "")
        actual_max_asr = pair_info.get("max_committed_asr", None)

        if actual_max_asr is None:
            print(f"{pair_name:<35} {'???':<10} {'N/A':<12} {'N/A':<10} {'N/A':<8}")
            continue

        # Predict
        prediction, per_attack = predict_pair(d1, d2, pure_asr)

        # Actual outcome
        actual_works = actual_max_asr < WORKS_THRESHOLD
        actual_label = "LOW" if actual_works else "HIGH"

        # Correct?
        predicted_works = prediction == "LOW"
        correct = predicted_works == actual_works

        results.append({
            "pair_name": pair_name,
            "d1": d1,
            "d2": d2,
            "prediction": prediction,
            "predicted_works": predicted_works,
            "actual_max_asr": actual_max_asr,
            "actual_works": actual_works,
            "correct": correct,
            "per_attack": per_attack,
        })

        correct_str = "YES" if correct else "NO ***"
        print(
            f"{pair_name:<35} {prediction:<10} {actual_max_asr:<12.3f} "
            f"{actual_label:<10} {correct_str:<8}"
        )

    print()

    # ==============================================================================
    # Confusion matrix
    # ==============================================================================
    if not results:
        print("No results to analyze.")
        return

    # TP = predicted LOW and actual LOW (correctly predicted works)
    # FP = predicted LOW but actual HIGH (incorrectly predicted works)
    # TN = predicted HIGH and actual HIGH (correctly predicted fails)
    # FN = predicted HIGH but actual LOW (incorrectly predicted fails)
    tp = sum(1 for r in results if r["predicted_works"] and r["actual_works"])
    fp = sum(1 for r in results if r["predicted_works"] and not r["actual_works"])
    tn = sum(1 for r in results if not r["predicted_works"] and not r["actual_works"])
    fn = sum(1 for r in results if not r["predicted_works"] and r["actual_works"])

    total = len(results)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    print()
    print(f"                    Actual LOW    Actual HIGH")
    print(f"  Predicted LOW       TP={tp:<4}       FP={fp:<4}")
    print(f"  Predicted HIGH      FN={fn:<4}       TN={tn:<4}")
    print()
    print(f"  Total pairs evaluated: {total}")
    print(f"  Accuracy:  {accuracy:.3f} ({tp+tn}/{total})")
    print(f"  Precision: {precision:.3f} (of predicted LOW, how many actually LOW)")
    print(f"  Recall:    {recall:.3f} (of actual LOW, how many predicted LOW)")
    print(f"  F1 Score:  {f1:.3f}")
    print()

    # ==============================================================================
    # Misclassified pairs
    # ==============================================================================
    misclassified = [r for r in results if not r["correct"]]

    if misclassified:
        print("=" * 60)
        print("MISCLASSIFIED PAIRS")
        print("=" * 60)
        print()
        for r in misclassified:
            print(f"  {r['pair_name']}:")
            print(f"    d1={r['d1']} ({get_defense_type(r['d1'])}), "
                  f"d2={r['d2']} ({get_defense_type(r['d2'])})")
            print(f"    Predicted: {r['prediction']}, Actual max ASR: {r['actual_max_asr']:.3f}")
            signal = check_signal_preservation(r['d1'], r['d2'])
            print(f"    Signal preservation: {signal}")
            for attack, detail in r["per_attack"].items():
                print(f"      {attack}: {detail['prediction']} - {detail['reason']}")
            print()
    else:
        print("No misclassified pairs — perfect prediction!")
        print()

    # ==============================================================================
    # Summary by signal preservation category
    # ==============================================================================
    print("=" * 60)
    print("BREAKDOWN BY SIGNAL PRESERVATION CATEGORY")
    print("=" * 60)
    print()

    categories = {"PASS": [], "FAIL": [], "DEGENERATE": []}
    for r in results:
        cat = check_signal_preservation(r["d1"], r["d2"])
        if cat in categories:
            categories[cat].append(r)

    for cat_name, cat_results in categories.items():
        if not cat_results:
            continue
        n_works = sum(1 for r in cat_results if r["actual_works"])
        n_total = len(cat_results)
        mean_asr = np.mean([r["actual_max_asr"] for r in cat_results])
        print(f"  {cat_name} ({n_total} pairs):")
        print(f"    Actually work (ASR<{WORKS_THRESHOLD}): {n_works}/{n_total}")
        print(f"    Mean max ASR: {mean_asr:.3f}")
        print()


if __name__ == "__main__":
    main()
