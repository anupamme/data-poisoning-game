# Pre-Registration: 3-Way Defense Composition Experiment

**Date:** 2026-07-23
**Author:** Anupam Mediratta
**Purpose:** Pre-registered predictions for 10 ordered defense triples (d1->d2->d3), to be evaluated AFTER this file is committed.

## Background

The Signal-Preservation Composability Criterion (C1 ^ C2 ^ C3) was developed and validated on 2-way compositions (42 ordered pairs from a 7-defense menu). This document extends the criterion to 3-way compositions (ordered triples) where exhaustive search is infeasible: 7 defenses yield P(7,3) = 210 ordered triples, making criterion-guided selection essential.

We pre-register 10 triples: 5 criterion-selected (predicted PASS/LOW) and 5 randomly selected (predicted FAIL/HIGH).

## Defense Menu

{fedavg, norm_clip, trimmed_mean, coord_median, rfa, foolsgold, reputation}

## 3-Way Composition Semantics

For a triple (d1 -> d2 -> d3):
- d1 applies its transform to per-client updates
- d2 applies its transform to d1's output
- d3 performs the final aggregation

The `apply_d1_transform()` function handles:
- **Weighting defenses** (reputation, foolsgold): per-client reweight
- **Clipping** (norm_clip): clip each update norm
- **Aggregating/no-op** (trimmed_mean, coord_median, rfa, fedavg): pass through (degenerate as d1 or d2)

## 3-Way Criterion Extension (C1 ^ C2' ^ C3')

For a triple (d1 -> d2 -> d3), the criterion requires ALL of:

1. **C1:** At least one of the trio individually suppresses the attack (max ASR < 0.5 in isolation)
2. **C2' (chained signal preservation):**
   - d1's transform preserves d2's transform signal (so d2 still functions correctly)
   - The combined d1->d2 output preserves d3's admission/aggregation signal
3. **C3' (chained separation preservation):**
   - After both d1 and d2 transforms, adversarial-benign separation still exceeds d3's rejection threshold

## Key 2-Way Results Informing Predictions

| Pair | Outcome | Mechanism |
|------|---------|-----------|
| fg -> rfa | PASS (0.045) | FG reweights bounded; RFA gmedian robust to bounded reweight |
| fg -> cm | PASS (0.131) | FG reweights bounded; CM rank robust to moderate reweight |
| rep -> cm | PASS (0.352) | Rep reweights; CM rank preserved |
| NC -> rep | C2-FAIL | NC equalizes norms, destroys rep's distance signal |
| FG -> TM | C3-FAIL | FG compresses adversarial coords below TM threshold |
| rep -> rfa | C1-FAIL | Rep doesn't suppress pixel; RFA doesn't suppress pixel |
| RFA -> rep | C2-FAIL | RFA reweight distorts rep's distance signal |

## Bounded Reweight Theorem

FoolsGold's weight ratio rho is bounded. For RFA at f=0.2 the margin requires rho < n_b/n_a = 4. FG's bounded reweighting preserves most downstream signals, making FG a reliable d1 or d2 in compositions.

## Defense Type Classification

- **Type A (per-client weighting):** reputation, foolsgold -- compute weights, scale updates
- **Type B (per-client clipping):** norm_clip -- clip each update to norm bound
- **Type C (aggregating, rank-based):** trimmed_mean, coord_median -- coordinate-wise statistics
- **Type D (aggregating, geometric):** rfa -- geometric median (iterative reweighting)
- **Type E (no-op):** fedavg -- simple averaging

## Predictions

### Criterion-Selected Triples (Predicted PASS / LOW)

| # | Triple (d1 -> d2 -> d3) | C1 | C2' | C3' | Predicted | Reasoning |
|---|--------------------------|----|----|------|-----------|-----------|
| 1 | FG -> rep -> RFA | PASS | PASS | PASS | **LOW** | FG reweights (bounded rho, preserves per-client structure). Rep reweights (bounded, preserves per-client structure). RFA aggregates via gmedian. C1: FG + rep together suppress scaling; rep suppresses scaling alone (0.017). C2': FG's bounded reweight preserves rep's distance signal (unlike NC which equalizes); rep's bounded reweight preserves RFA's geometric structure. C3': Combined reweight ratio rho_total <= rho_FG x rho_rep still within RFA's rejection radius margin. |
| 2 | FG -> rep -> CM | PASS | PASS | PASS | **LOW** | FG reweights (bounded). Rep reweights (bounded). CM aggregates by coordinate-wise median. C1: Rep suppresses scaling (0.017). C2': FG preserves rep's distance signal; rep's reweight preserves CM's rank ordering (rank is robust to bounded multiplicative scaling). C3': Combined bounded reweight does not break CM's rank-based rejection. Analogous to fg->cm (0.131) with rep providing additional suppression. |
| 3 | rep -> FG -> RFA | PASS | PASS | PASS | **LOW** | Rep reweights (changes magnitudes, not directions). FG reweights (bounded). RFA aggregates via gmedian. C1: FG suppresses (partial); rep suppresses scaling. C2': Rep preserves FG's cosine similarity signal (cosine is direction-based; rep's multiplicative reweight does not change direction). FG's bounded reweight preserves RFA's geometric structure. C3': Combined reweight ratio bounded; RFA's margin maintained. |
| 4 | rep -> FG -> CM | PASS | PASS | PASS | **LOW** | Rep reweights. FG reweights (bounded). CM aggregates by coordinate-wise median. C1: Rep suppresses scaling. C2': Rep preserves FG's cosine signal (direction unchanged). FG's bounded reweight preserves CM's rank. C3': Rank-based rejection robust to combined bounded reweight. Same logic as #3 but with CM final aggregation. |
| 5 | FG -> NC -> CM | PASS | PASS | PASS (borderline) | **LOW** | FG reweights (bounded). NC clips each update's norm. CM aggregates by coordinate-wise median. C1: FG partially suppresses; CM suppresses (0.450 scaling, 0.377 pixel). C2': FG's bounded reweight preserves NC's norm threshold applicability (adversarial updates still have larger norms after bounded rescaling). NC's clipping preserves CM's rank ordering (clipping truncates magnitude but preserves relative coordinate ordering for non-extreme updates). C3': After FG reweight + NC clip, adversarial-benign separation in coordinate space should still exceed CM's rejection threshold, though NC's compression makes this borderline. **Note:** This is the weakest of the 5 -- NC's clipping may compress separation below CM's threshold. |

### Randomly-Selected Triples (Predicted FAIL / HIGH)

| # | Triple (d1 -> d2 -> d3) | C1 | C2' | C3' | Predicted | Reasoning |
|---|--------------------------|----|----|------|-----------|-----------|
| 6 | NC -> FG -> TM | PASS | ? | FAIL | **HIGH** | NC clips (equalizes norms) -> FG sees equalized updates (cosine similarity now measures pure direction, not magnitude-weighted direction) -> TM aggregates. NC's clipping may actually help FG's cosine computation (removes magnitude noise), but FG->TM is a known C3-FAIL at 2-way (FG compresses adversarial coords below TM's trimming threshold). The upstream NC clip does not fix this downstream C3 failure -- it may even worsen it by further compressing adversarial signal before FG's additional compression. |
| 7 | RFA -> NC -> rep | FAIL | FAIL | -- | **HIGH** | RFA reweights -> NC clips reweighted updates -> rep sees clipped updates. Chain of two signal-destroying transforms. RFA's heterogeneous reweighting distorts NC's norm threshold (C2-FAIL for NC), and NC's clipping then equalizes norms destroying rep's distance signal (NC->rep is known C2-FAIL). Even if RFA's reweight were benign, NC->rep alone breaks the chain. Double signal destruction. |
| 8 | FG -> RFA -> CM | PASS | PASS | PASS (borderline) | **LOW** | FG reweights (bounded rho_FG). RFA reweights (inverse-distance from gmedian, bounded rho_RFA). CM aggregates by coordinate-wise median. Both FG and RFA are bounded reweights; their composition is also bounded (rho_total <= rho_FG x rho_RFA). CM's rank is preserved if total rho < r*. C1: FG partially suppresses; CM suppresses scaling. **Note:** More aggressive total reweighting makes this borderline -- if rho_total exceeds CM's rank-preservation bound, C3' fails. Predicted LOW but with higher uncertainty than triples #1-4. |
| 9 | fedavg -> NC -> rep | FAIL | FAIL | -- | **HIGH** | Fedavg passes through unchanged (degenerate d1). This degenerates to NC->rep (2-way), which is a known C2-FAIL: NC equalizes norms, destroying reputation's distance-based signal. Rep sees equalized updates and cannot distinguish adversarial from benign. Pixel ASR will remain high. |
| 10 | NC -> rep -> TM | PASS | FAIL | -- | **HIGH** | NC clips (equalizes norms) -> rep sees equalized norms (known C2-FAIL: NC destroys rep's distance signal) -> rep's degenerate/broken reweighting fed to TM. Since the NC->rep link is broken (C2-FAIL), reputation cannot meaningfully reweight, and TM receives effectively unfiltered updates. TM alone does not suppress (scaling 0.846). The chain is broken at the first link. |

## Summary of Predictions

| Category | Triples | Count | Predicted |
|----------|---------|-------|-----------|
| Criterion-selected (PASS) | #1, #2, #3, #4, #5 | 5 | LOW (< 0.5) |
| Random / likely FAIL | #6, #7, #9, #10 | 4 | HIGH (>= 0.5) |
| Random / borderline PASS | #8 | 1 | LOW (borderline) |

**Total LOW predictions: 6** (triples #1-5, #8)
**Total HIGH predictions: 4** (triples #6, #7, #9, #10)

### Final Binary Predictions (for accuracy calculation):

| # | Triple | Predicted |
|---|--------|-----------|
| 1 | FG -> rep -> RFA | **LOW** |
| 2 | FG -> rep -> CM | **LOW** |
| 3 | rep -> FG -> RFA | **LOW** |
| 4 | rep -> FG -> CM | **LOW** |
| 5 | FG -> NC -> CM | **LOW** |
| 6 | NC -> FG -> TM | **HIGH** |
| 7 | RFA -> NC -> rep | **HIGH** |
| 8 | FG -> RFA -> CM | **LOW** |
| 9 | fedavg -> NC -> rep | **HIGH** |
| 10 | NC -> rep -> TM | **HIGH** |

## Methodology

- **Seeds:** [42, 43, 44, 45, 46] (5 seeds per triple per attack)
- **Attacks:** committed_scaling, committed_pixel
- **FL config:** N=10, K=5, f=0.2, 50 rounds, cifar_cnn
- **Metric:** max(mean_scaling_ASR, mean_pixel_ASR) across seeds
- **Threshold:** LOW < 0.5, HIGH >= 0.5
- **This file will be committed BEFORE experiments are run.**

## Why 3-Way Matters

With 7 defenses and 210 possible ordered triples, exhaustive evaluation is computationally expensive. The criterion enables practitioners to:
1. **Filter** the 210 triples to a tractable set of predicted-PASS compositions
2. **Avoid** triples with known signal-destruction chains (e.g., NC->rep->anything)
3. **Compose** defenses with confidence that chained transforms preserve downstream signals

If the criterion achieves high accuracy on these 10 triples (>= 8/10), it demonstrates scalability from 2-way to 3-way without retraining, supporting the paper's central claim that signal-preservation is a composable, predictive property.
