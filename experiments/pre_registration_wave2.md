# Pre-Registration: Wave 2 Composability Criterion Predictions

**Date:** 2026-07-18
**Author:** Anupam Mediratta
**Purpose:** Pre-registered predictions for 24 held-out defense pairs, to be evaluated AFTER this file is committed.

## Background

The Signal-Preservation Composability Criterion (C1∧C2∧C3) was developed on an 18-pair "development set" achieving 16/18 = 89% accuracy. This document pre-registers predictions for the remaining 24 ordered pairs from our 7-defense menu {fedavg, norm_clip, trimmed_mean, coord_median, rfa, foolsgold, reputation} before running experiments. The threshold for LOW vs HIGH is max-committed ASR < 0.5 vs ≥ 0.5.

## Pure Defense Suppression Profiles (C1 assessment basis)

| Defense | Scaling ASR | Pixel ASR | Suppresses Scaling? | Suppresses Pixel? |
|---------|-------------|-----------|--------------------|--------------------|
| fedavg | 0.667 | 0.768 | NO | NO |
| norm_clip | 0.936 | 0.768 | NO | NO |
| trimmed_mean | 0.846 | 0.616 | NO | NO |
| coord_median | 0.450 | 0.377 | YES (borderline) | YES |
| rfa | 0.870 | 0.819 | NO | NO |
| foolsgold | ~0.50* | ~0.50* | PARTIAL | PARTIAL |
| reputation | 0.017 | 0.842 | YES | NO |

*FoolsGold pure baseline not measured directly; inferred from FG→RFA (0.045/0.038) and FG→CM (0.131/0.111) compositions suggesting FG partially suppresses both attacks.

## Defense Type Classification

- **Type A (per-client weighting):** reputation, foolsgold — compute weights, scale updates
- **Type B (per-client clipping):** norm_clip — clip each update to norm bound
- **Type C (aggregating, rank-based):** trimmed_mean, coord_median — coordinate-wise statistics
- **Type D (aggregating, geometric):** rfa — geometric median (iterative reweighting)
- **Type E (no-op):** fedavg — simple averaging

**Key principle:** Types C, D, E as d1 produce a SINGLE aggregate vector (no per-client transformation preserved for d2). Composition degenerates to d2 alone → DEGEN (predicted HIGH unless d2 alone suppresses).

## Predictions for 24 Held-Out Pairs

### Category: DEGEN (d1 is an aggregator with no per-client transform)
These defenses as d1 cannot provide per-client signal to d2.

| # | Pair | C1 | C2 | C3 | Predicted | Reasoning |
|---|------|----|----|-----|-----------|-----------|
| 1 | coord_median → fedavg | N/A | N/A | N/A | HIGH (DEGEN) | CM aggregates to single point; fedavg averages that single point = CM alone. Neither suppresses scaling. |
| 2 | coord_median → foolsgold | N/A | N/A | N/A | HIGH (DEGEN) | CM produces single aggregate; FG sees 1 "update" → no similarity signal. Degenerates to CM alone. CM has scaling ASR 0.450 (borderline) and pixel 0.377; max = 0.450. **Borderline — might be LOW.** |
| 3 | coord_median → norm_clip | N/A | N/A | N/A | HIGH (DEGEN) | CM aggregate → NC clips single vector (no-op if within τ). = CM alone. Max 0.450/0.377 → 0.450 borderline. **Predicted HIGH but borderline.** |
| 4 | coord_median → reputation | N/A | N/A | N/A | HIGH (DEGEN) | CM → single aggregate → reputation has 1 client → no weighting. = CM alone. Borderline. |
| 5 | coord_median → rfa | N/A | N/A | N/A | HIGH (DEGEN) | CM → single point → RFA geometric median of 1 = identity. = CM alone. Borderline. |
| 6 | coord_median → trimmed_mean | N/A | N/A | N/A | HIGH (DEGEN) | CM → single point → TM of 1 = identity. = CM alone. Max 0.450 borderline. |
| 7 | trimmed_mean → coord_median | N/A | N/A | N/A | HIGH (DEGEN) | TM aggregates to single point → CM of 1 = identity. = TM alone. scaling 0.846 > 0.5. |
| 8 | trimmed_mean → fedavg | N/A | N/A | N/A | HIGH (DEGEN) | TM alone. Scaling 0.846. |
| 9 | trimmed_mean → foolsgold | N/A | N/A | N/A | HIGH (DEGEN) | TM produces single aggregate → FG cannot operate. = TM alone. |
| 10 | trimmed_mean → norm_clip | N/A | N/A | N/A | HIGH (DEGEN) | TM alone. |
| 11 | trimmed_mean → reputation | N/A | N/A | N/A | HIGH (DEGEN) | TM alone. |
| 12 | trimmed_mean → rfa | N/A | N/A | N/A | HIGH (DEGEN) | TM alone. |

**IMPORTANT NOTE on coord_median DEGEN pairs (#2-6):** The `apply_d1_transform()` function in `run_all_compositions.py` passes through updates unchanged for coord_median and trimmed_mean as d1 (lines 217-220: "These aggregate to a single point per-coordinate; cannot meaningfully transform per-client. Pass through unchanged"). This means composition of CM→d2 actually equals d2 alone (not CM alone), because d1's transform is identity and d2 does the aggregation. This changes predictions:
- CM→FG = FG alone (scaling ~0.5, pixel ~0.5) → HIGH
- CM→NC = NC alone (scaling 0.936, pixel 0.768) → HIGH
- CM→rep = rep alone (scaling 0.017, pixel 0.842) → HIGH (pixel dominates)
- CM→RFA = RFA alone (scaling 0.870, pixel 0.819) → HIGH
- CM→TM = TM alone (scaling 0.846, pixel 0.616) → HIGH
- CM→fedavg = fedavg alone (scaling 0.667, pixel 0.768) → HIGH
- TM→X = same logic → all HIGH

**Revised DEGEN predictions: ALL HIGH.**

### Category: DEGEN (d1 = fedavg, remaining untested)

| # | Pair | Predicted | Reasoning |
|---|------|-----------|-----------|
| 13 | fedavg → coord_median | HIGH (DEGEN) | fedavg passes through → CM aggregates. CM alone: scaling 0.450, pixel 0.377 → max 0.450. **Borderline — might be LOW (0.450 < 0.5).** |
| 14 | fedavg → foolsgold | HIGH (DEGEN) | fedavg → FG alone. FG ~0.5 both? Uncertain. Predicted HIGH. |
| 15 | fedavg → reputation | HIGH (DEGEN) | fedavg → rep alone. Pixel 0.842 >> 0.5. HIGH. |
| 16 | fedavg → rfa | HIGH (DEGEN) | fedavg → RFA alone. Scaling 0.870. HIGH. |

### Category: Per-client weighting d1 with non-reputation d2

| # | Pair | C1 | C2 | C3 | Predicted | Reasoning |
|---|------|----|----|-----|-----------|-----------|
| 17 | foolsgold → fedavg | Partial | N/A | N/A | HIGH | FG weights → fedavg averages weighted updates. FG partially suppresses both but fedavg has no additional mechanism. Without d2 adding suppression, FG's partial downweighting alone may not bring max below 0.5. Predicted HIGH. |
| 18 | foolsgold → norm_clip | Partial | ✓ | ? | LOW? | FG weights → NC clips. FG partially suppresses; NC clips provide an additional bound. If FG brings adversarial updates partially down and NC further clips them, might get LOW. But NC alone doesn't suppress (0.936/0.768). **Uncertain — predicted LOW (0.3-0.5) based on FG→RFA analogy.** |
| 19 | foolsgold → reputation | Partial | ✗ | — | HIGH (C2-FAIL) | FG rescales → reputation sees equalized (FG-weighted) norms. FG's heterogeneous rescaling compresses consensus-distance signal. Reputation's distance metric is corrupted by FG's reweighting. Predicted HIGH. |

### Category: RFA as d1 (remaining)

| # | Pair | C1 | C2 | C3 | Predicted | Reasoning |
|---|------|----|----|-----|-----------|-----------|
| 20 | rfa → fedavg | — | — | — | HIGH (DEGEN-like) | RFA reweights → fedavg averages. RFA's weights are inverse-distance from geometric median. This is similar to RFA alone since averaging weighted updates ≈ RFA aggregate. But the generic_compose function passes through RFA reweighted updates → fedavg averages them = weighted average. RFA alone is 0.870/0.819. HIGH. |
| 21 | rfa → foolsgold | ✗ | ✗ | — | HIGH (C2-FAIL) | RFA reweights → FG sees reweighted updates. FG's cosine-similarity is scale-sensitive to heterogeneous rescaling. RFA assigns very different weights to different clients → FG's similarity computation is distorted. HIGH. |
| 22 | rfa → norm_clip | ✗ | ✓ | ✗ | HIGH (C3-FAIL) | RFA reweights → NC clips. RFA downweights adversarial clients (lower magnitude) → NC's norm threshold doesn't fire on already-small updates. Same C3 mechanism as NC→TM. HIGH. |

### Category: NC as d1 (remaining)

| # | Pair | C1 | C2 | C3 | Predicted | Reasoning |
|---|------|----|----|-----|-----------|-----------|
| 23 | norm_clip → fedavg | — | — | — | HIGH (DEGEN-like) | NC clips → fedavg averages clipped. NC alone doesn't suppress (0.936/0.768). Clipping + averaging has no additional mechanism. HIGH. |

### Category: Reputation as d1 (remaining)

| # | Pair | C1 | C2 | C3 | Predicted | Reasoning |
|---|------|----|----|-----|-----------|-----------|
| 24 | reputation → fedavg | C1-FAIL (pixel) | — | — | HIGH | Rep kills scaling (0.017) but pixel passes (0.842). FedAvg has no suppression mechanism. Max = pixel ASR ~0.84. HIGH. |

## Summary of Predictions

| Category | Count | Predicted |
|----------|-------|-----------|
| DEGEN (all HIGH) | 16 | HIGH |
| C2-FAIL | 2 (fg→rep, rfa→fg) | HIGH |
| C3-FAIL | 1 (rfa→NC) | HIGH |
| C1-FAIL | 1 (rep→fedavg) | HIGH |
| Uncertain/borderline | 2 (fg→NC, fedavg→CM) | LOW?, HIGH? |
| Clearly HIGH | 2 (fg→fedavg, NC→fedavg) | HIGH |

**Overall prediction: 22 HIGH, 1 LOW (fg→NC), 1 borderline (fedavg→CM might be LOW if CM suppresses to <0.5).**

### Specific predictions for scoring:
- **Predicted LOW (< 0.5):** foolsgold → norm_clip (pair #18)
- **Predicted borderline (might be LOW):** fedavg → coord_median (#13) — max(0.450, 0.377) = 0.450 < 0.5
- **All others:** HIGH (≥ 0.5)

### Final binary predictions (for accuracy calculation):
| Pair | Predicted |
|------|-----------|
| coord_median → fedavg | HIGH |
| coord_median → foolsgold | HIGH |
| coord_median → norm_clip | HIGH |
| coord_median → reputation | HIGH |
| coord_median → rfa | HIGH |
| coord_median → trimmed_mean | HIGH |
| fedavg → coord_median | **LOW** |
| fedavg → foolsgold | HIGH |
| fedavg → reputation | HIGH |
| fedavg → rfa | HIGH |
| foolsgold → fedavg | HIGH |
| foolsgold → norm_clip | **LOW** |
| foolsgold → reputation | HIGH |
| norm_clip → fedavg | HIGH |
| reputation → fedavg | HIGH |
| rfa → fedavg | HIGH |
| rfa → foolsgold | HIGH |
| rfa → norm_clip | HIGH |
| trimmed_mean → coord_median | HIGH |
| trimmed_mean → fedavg | HIGH |
| trimmed_mean → foolsgold | HIGH |
| trimmed_mean → norm_clip | HIGH |
| trimmed_mean → reputation | HIGH |
| trimmed_mean → rfa | HIGH |

**Total LOW predictions: 2 (fedavg→CM, fg→NC)**
**Total HIGH predictions: 22**

## Methodology
- Seeds: [42, 43, 44, 45, 46] (5 seeds per pair per attack)
- Attacks: committed_scaling, committed_pixel
- FL config: N=10, K=5, f=0.2, 50 rounds, cifar_cnn
- Metric: max(mean_scaling_ASR, mean_pixel_ASR) across seeds
- Threshold: LOW < 0.5, HIGH ≥ 0.5
- This file will be committed BEFORE experiments are run.

---

# CORRECTION NOTE (appended 2026-08-17)

**Nothing above this line has been altered.** The original predictions stand as
pre-registered on 2026-07-18 (commit `7e4d9b1`). This note records a mathematical error
found in the *reasoning* of two entries, and its (nil) effect on the scored outcome.

## The error

Entry #21 (`rfa → foolsgold`) justifies its C2-FAIL call with "FG's cosine-similarity is
scale-sensitive to heterogeneous rescaling." The same reasoning was applied in the
development set to `norm_clip → foolsgold`. **This is false.** Both RFA-reweighting and
NormClip are per-client *positive scalar* rescalings, `T(u_i) = c_i u_i` with `c_i > 0`
(NormClip: `c_i = min(1, tau/||u_i||)`). Cosine similarity is exactly invariant under
positive rescaling, so FoolsGold's entire weight vector is unchanged:

    F.normalize(c_i * u_i) == F.normalize(u_i)  =>  sim matrix, max_sim, weights all identical

Verified numerically in the shipped implementation by `experiments/verify_fg_invariance.py`:
maximum weight deviation `3e-8` (float32 rounding) across 1000 synthetic trials and live
FL rounds in which 1–2 of 5 clients were actually clipped (raw norms up to 24.5 vs tau=5).
Result written to `results/fg_invariance_check.json`.

## Corrected classification

Both pairs satisfy C2. They fail **C1** instead — no constituent suppresses:

| pair | old category | corrected | C1 evidence | predicted | actual | correct? |
|---|---|---|---|---|---|---|
| `norm_clip → foolsgold` | C2-FAIL | **C1-FAIL** | NC alone 0.935, FG alone 0.732 | HIGH | 0.887 | yes |
| `rfa → foolsgold` | C2-FAIL | **C1-FAIL** | RFA alone 0.885, FG alone 0.732 | HIGH | 0.914 | yes |

**The predictions do not change** (HIGH in both cases) and both remain correct, so the
confusion matrix and all reported accuracy figures are unaffected. Only the failure-mode
labels move. Category tally over 42 pairs: DEGEN 18, C1-FAIL 11, C2-FAIL 3, C3-FAIL 7,
PASS 3.

A consequence worth noting: after the correction, **all three surviving C2 failures are
`· → reputation`** (nc→rep 0.819, rfa→rep 0.882, fg→rep 0.798). Reputation's discriminative
property is consensus distance (L2 from the coordinate-wise median), which is *not*
invariant under heterogeneous positive rescaling — so this is exactly the class the
invariance analysis predicts should fail. The taxonomy is more uniform after the fix than
before it.

## Scoring provenance (also clarified, no data changed)

The 24 pairs above are genuinely held out and pre-registered. They contain **no
predicted-PASS pair**, so they test specificity only; scored against the labels above the
result is TP=0, FP=2, TN=22, FN=0 (22/24). The two misses are the entries hedged above as
speculative: `foolsgold → norm_clip` (#18, "Uncertain — predicted LOW") measured 0.927, and
`fedavg → coord_median` (#13, "borderline — might be LOW") measured 0.519. Both are scored
as misses, not dropped. The 18-pair development set is where the criterion was developed
and is reported as fit, not prediction. PASS-direction predictions are pre-registered
separately in `pre_registration_oos_pass.md`.
