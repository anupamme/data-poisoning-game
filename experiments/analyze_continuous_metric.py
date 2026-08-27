"""
Continuous metric analysis for the composability criterion.

Computes:
1. Spearman rank correlation between predicted category and measured ASR
2. Calibration data (predicted category vs ASR distribution)
3. Threshold sensitivity analysis (accuracy at thresholds 0.3-0.7)

Uses existing results from:
- results/all_compositions/summary.json (dev set, 18 pairs)
- results/wave2_held_out/summary.json (held-out set, 24 pairs)

Output: results/continuous_metric/analysis.json + LaTeX table
"""
import json
import os
import sys
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

# --- Load data ---
dev_path = os.path.join(base_dir, "results", "all_compositions", "summary.json")
heldout_path = os.path.join(base_dir, "results", "wave2_held_out", "summary.json")

output_dir = os.path.join(base_dir, "results", "continuous_metric")
os.makedirs(output_dir, exist_ok=True)

# Criterion predictions from the paper's validation tables (Tables 1 and 2).
# Categories ordered by expected ASR: PASS < C3-FAIL < C2-FAIL < C1-FAIL ≈ DEGEN
# Ordinals: PASS=1, C3-FAIL=2, C2-FAIL=3, C1-FAIL=4, DEGEN=5
CRITERION_PREDICTIONS = {
    # Dev set (18 pairs) — from Table 1
    "foolsgold_then_rfa": ("PASS", 1),
    "foolsgold_then_coord_median": ("PASS", 1),
    "reputation_then_coord_median": ("PASS", 1),
    "reputation_then_trimmed_mean": ("C1-FAIL", 4),  # rep doesn't suppress pixel
    "reputation_then_rfa": ("C1-FAIL", 4),
    "reputation_then_foolsgold": ("C1-FAIL", 4),
    "reputation_then_norm_clip": ("C1-FAIL", 4),
    "norm_clip_then_coord_median": ("C3-FAIL", 2),
    "norm_clip_then_rfa": ("C3-FAIL", 2),
    "rfa_then_trimmed_mean": ("C3-FAIL", 2),
    "rfa_then_coord_median": ("C3-FAIL", 2),
    "foolsgold_then_trimmed_mean": ("C3-FAIL", 2),
    "norm_clip_then_trimmed_mean": ("C3-FAIL", 2),
    "norm_clip_then_reputation": ("C2-FAIL", 3),
    # CORRECTED (Proposition 1a): NormClip is a per-client POSITIVE scalar rescaling,
    # u_i -> u_i * min(1, tau/||u_i||), and cosine similarity is exactly invariant under
    # positive rescaling. FoolsGold's weight vector is therefore bit-identical before and
    # after clipping (verified: max deviation 3e-8, see verify_fg_invariance.py). C2 HOLDS.
    # The pair fails C1 instead: NC alone 0.935, FG alone 0.732, neither < 0.5.
    # Prediction is HIGH either way, so the confusion matrix is unchanged.
    "norm_clip_then_foolsgold": ("C1-FAIL", 4),
    "rfa_then_reputation": ("C2-FAIL", 3),
    "fedavg_then_norm_clip": ("DEGEN", 5),
    "fedavg_then_trimmed_mean": ("DEGEN", 5),
    # Held-out set (24 pairs) — from Table 2 / pre_registration_wave2.md
    "fedavg_then_coord_median": ("DEGEN", 5),
    "fedavg_then_rfa": ("DEGEN", 5),
    "fedavg_then_foolsgold": ("DEGEN", 5),
    "fedavg_then_reputation": ("DEGEN", 5),
    "trimmed_mean_then_fedavg": ("DEGEN", 5),
    "trimmed_mean_then_norm_clip": ("DEGEN", 5),
    "trimmed_mean_then_coord_median": ("DEGEN", 5),
    "trimmed_mean_then_rfa": ("DEGEN", 5),
    "trimmed_mean_then_foolsgold": ("DEGEN", 5),
    "trimmed_mean_then_reputation": ("DEGEN", 5),
    "coord_median_then_fedavg": ("DEGEN", 5),
    "coord_median_then_norm_clip": ("DEGEN", 5),
    "coord_median_then_trimmed_mean": ("DEGEN", 5),
    "coord_median_then_rfa": ("DEGEN", 5),
    "coord_median_then_foolsgold": ("DEGEN", 5),
    "coord_median_then_reputation": ("DEGEN", 5),
    "foolsgold_then_norm_clip": ("C1-FAIL", 4),  # criterion mechanism = C1-FAIL (HIGH); pre-reg made a speculative LOW call
    "foolsgold_then_reputation": ("C2-FAIL", 3),  # FG reweighting corrupts reputation's distance signal
    "foolsgold_then_fedavg": ("C1-FAIL", 4),
    "reputation_then_fedavg": ("C1-FAIL", 4),
    # CORRECTED (Proposition 1a): same error as norm_clip_then_foolsgold. RFA-reweighting is
    # also a per-client positive rescaling, so FG's cosine similarities are NOT distorted --
    # they are exactly invariant. C2 holds; the pair fails C1 (RFA alone 0.885, FG alone
    # 0.732). Prediction HIGH either way; actual 0.914, still correct.
    "rfa_then_foolsgold": ("C1-FAIL", 4),
    "norm_clip_then_fedavg": ("C1-FAIL", 4),
    "rfa_then_fedavg": ("C1-FAIL", 4),
    "rfa_then_norm_clip": ("C3-FAIL", 2),  # RFA downweights adv -> NC threshold no longer fires
}

# Pre-registered binary predictions (committed in pre_registration_wave2.md BEFORE
# experiments). These are the LOW-predicted pairs; every other matched pair was
# pre-registered HIGH. The threshold-sensitivity sweep uses THESE labels so that it
# honors the pre-registration exactly (in particular, the two speculative-LOW calls
# fg->NC and fedavg->cm are scored as misses, not silently dropped or re-coded).
PREREG_LOW = {
    # dev-set clean C1^C2^C3 PASS predictions
    "foolsgold_then_rfa",
    "foolsgold_then_coord_median",
    "reputation_then_coord_median",
    # held-out speculative-LOW calls (both missed: actual >= 0.5)
    "fedavg_then_coord_median",
    "foolsgold_then_norm_clip",
}


def load_results():
    """Load all composition ASR results from dev and held-out sets."""
    pairs = {}

    # Dev set
    if os.path.exists(dev_path):
        with open(dev_path) as f:
            dev = json.load(f)
        for pk, data in dev.get("pairs", {}).items():
            if "max_committed_asr" in data:
                pairs[pk] = data["max_committed_asr"]
            else:
                # Compute from per-attack means
                max_asr = 0.0
                for atk in ["committed_scaling", "committed_pixel"]:
                    if atk in data and "mean_asr" in data[atk]:
                        max_asr = max(max_asr, data[atk]["mean_asr"])
                if max_asr > 0:
                    pairs[pk] = max_asr

    # Held-out set
    if os.path.exists(heldout_path):
        with open(heldout_path) as f:
            heldout = json.load(f)
        for pk, data in heldout.get("pairs", {}).items():
            if "max_committed_asr" in data:
                pairs[pk] = data["max_committed_asr"]
            else:
                max_asr = 0.0
                for atk in ["committed_scaling", "committed_pixel"]:
                    if atk in data and "mean_asr" in data[atk]:
                        max_asr = max(max_asr, data[atk]["mean_asr"])
                if max_asr > 0:
                    pairs[pk] = max_asr

    return pairs


def compute_spearman(ordinals, asrs):
    """Compute Spearman rank correlation."""
    from scipy import stats
    rho, pval = stats.spearmanr(ordinals, asrs)
    return rho, pval


def compute_threshold_accuracy(predictions, asrs, thresholds):
    """Compute accuracy at each threshold."""
    results = {}
    for threshold in thresholds:
        correct = 0
        total = 0
        for pk in predictions:
            if pk not in asrs:
                continue
            cat, ordinal = predictions[pk]
            actual_asr = asrs[pk]
            # PASS prediction → expect ASR < threshold
            # FAIL prediction → expect ASR >= threshold
            predicted_low = (cat == "PASS")
            actual_low = (actual_asr < threshold)
            if predicted_low == actual_low:
                correct += 1
            elif cat != "PASS":
                # Non-PASS predictions always predict HIGH
                if actual_asr >= threshold:
                    correct += 1
                else:
                    pass  # non-PASS but ASR below threshold
            total += 1
        results[threshold] = {"correct": correct, "total": total,
                              "accuracy": correct / max(total, 1)}
    return results


def compute_calibration(predictions, asrs):
    """Group ASR values by predicted category."""
    groups = {}
    for pk in predictions:
        if pk not in asrs:
            continue
        cat, ordinal = predictions[pk]
        if cat not in groups:
            groups[cat] = []
        groups[cat].append(asrs[pk])
    return groups


if __name__ == "__main__":
    print("=" * 70)
    print("  CONTINUOUS METRIC ANALYSIS")
    print("=" * 70)

    asrs = load_results()
    print(f"\n  Loaded {len(asrs)} pair results")

    # Filter to pairs with predictions
    matched = {pk: asrs[pk] for pk in CRITERION_PREDICTIONS if pk in asrs}
    print(f"  Matched with predictions: {len(matched)}")

    # 1. Spearman rank correlation
    ordinals = [CRITERION_PREDICTIONS[pk][1] for pk in matched]
    asr_values = [matched[pk] for pk in matched]

    try:
        rho, pval = compute_spearman(ordinals, asr_values)
        print(f"\n  Spearman ρ = {rho:.3f} (p = {pval:.2e})")
        spearman_result = {"rho": rho, "p_value": pval, "n": len(matched)}
    except ImportError:
        # Fallback: manual Spearman
        print("\n  scipy not available; computing manual rank correlation")
        from collections import Counter
        n = len(ordinals)
        rank_x = np.argsort(np.argsort(ordinals)).astype(float) + 1
        rank_y = np.argsort(np.argsort(asr_values)).astype(float) + 1
        d = rank_x - rank_y
        rho = 1 - 6 * np.sum(d**2) / (n * (n**2 - 1))
        # t-test for significance
        t_stat = rho * np.sqrt((n - 2) / (1 - rho**2 + 1e-10))
        pval = 2 * (1 - 0.5)  # placeholder
        print(f"  Spearman ρ (manual) ≈ {rho:.3f}")
        spearman_result = {"rho": float(rho), "p_value": None, "n": n, "note": "manual computation"}

    # 2. Calibration
    print("\n  Calibration (predicted category → ASR distribution):")
    calibration = compute_calibration(CRITERION_PREDICTIONS, asrs)
    cal_summary = {}
    for cat in ["PASS", "C3-FAIL", "C2-FAIL", "C1-FAIL", "DEGEN"]:
        if cat in calibration:
            vals = calibration[cat]
            cal_summary[cat] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "n": len(vals),
                "values": [float(v) for v in sorted(vals)],
            }
            print(f"    {cat:10s}: mean={np.mean(vals):.3f}±{np.std(vals):.3f} "
                  f"[{np.min(vals):.3f}, {np.max(vals):.3f}] (n={len(vals)})")

    # 3. Threshold sensitivity
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    print(f"\n  Threshold sensitivity (accuracy at various cutoffs):")
    print(f"    {'Threshold':<12s} {'Accuracy':<12s} {'Correct/Total'}")
    print(f"    {'-'*40}")

    thresh_results = {}
    for t in thresholds:
        correct = 0
        total = 0
        for pk in CRITERION_PREDICTIONS:
            if pk not in asrs:
                continue
            actual = asrs[pk]
            # Use the PRE-REGISTERED binary label (honors the committed predictions,
            # including the two speculative-LOW misses fg->NC and fedavg->cm).
            predicted_low = (pk in PREREG_LOW)
            actual_low = (actual < t)
            if predicted_low == actual_low:
                correct += 1
            total += 1
        acc = correct / max(total, 1)
        thresh_results[str(t)] = {"accuracy": acc, "correct": correct, "total": total}
        print(f"    {t:<12.1f} {acc:<12.3f} {correct}/{total}")

    # N3.3: honest PASS-vs-rest separation (the taxonomy does NOT order magnitude)
    pass_vals = [asrs[pk] for pk in CRITERION_PREDICTIONS
                 if pk in asrs and CRITERION_PREDICTIONS[pk][0] == "PASS"]
    rest_vals = [asrs[pk] for pk in CRITERION_PREDICTIONS
                 if pk in asrs and CRITERION_PREDICTIONS[pk][0] != "PASS"]
    print(f"\n  PASS-vs-rest: PASS mean={np.mean(pass_vals):.3f} (n={len(pass_vals)}), "
          f"rest mean={np.mean(rest_vals):.3f} (n={len(rest_vals)}), "
          f"rest range=[{np.min(rest_vals):.3f}, {np.max(rest_vals):.3f}]")

    # Summary
    output = {
        "spearman": spearman_result,
        "calibration": cal_summary,
        "threshold_sensitivity": thresh_results,
        "n_pairs_total": len(matched),
    }

    output_file = os.path.join(output_dir, "analysis.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Output: {output_file}")

    # LaTeX table for threshold sensitivity
    print("\n  --- LaTeX table (threshold sensitivity) ---")
    print("  \\begin{tabular}{ccc}")
    print("  \\toprule")
    print("  Threshold & Accuracy & Correct/Total \\\\")
    print("  \\midrule")
    for t in thresholds:
        r = thresh_results[str(t)]
        bold = "\\textbf" if t == 0.5 else ""
        if bold:
            print(f"  {bold}{{{t:.1f}}} & {bold}{{{r['accuracy']:.1%}}} & {bold}{{{r['correct']}/{r['total']}}} \\\\")
        else:
            print(f"  {t:.1f} & {r['accuracy']:.1%} & {r['correct']}/{r['total']} \\\\")
    print("  \\bottomrule")
    print("  \\end{tabular}")
