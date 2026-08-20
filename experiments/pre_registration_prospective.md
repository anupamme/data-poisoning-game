# Pre-Registration: Prospective Composition Test

**Date:** 2026-08-20
**Author:** Anupam Mediratta
**Status:** Predictions below are FROZEN. This file is committed to version control BEFORE any
composition experiment in `results/prospective_suite/` is run.

## Why this exists

Prior validation had a structural weakness a reviewer correctly identified: the 42-pair sets were
used to develop the criterion (18) or contained no predicted-PASS pair (24), and the 7
out-of-sample predictions were **all positive** with FAIL controls that turned out uninformative
because DBA failed to embed even against FedAvg. So the framework had never been shown to
discriminate prospectively.

This suite fixes that. It uses **three defenses never in the paper's 7-defense menu** — `fltrust`,
`krum`, `multi_krum` — opening pairs on which the criterion was not developed, and it registers
predictions in **both** directions with a sanity gate that the DBA experiment lacked.

## Sanity gate (PASSED — measured in Phase 0, before this file)

A predicted-FAIL control is only informative if the attack embeds against no defense at all.

| attack | FedAvg ASR | informative? |
|---|---|---|
| committed_scaling | 0.667 | YES (>= 0.5) |
| committed_pixel | 0.768 | YES (>= 0.5) |

Both clear the threshold, so FAIL predictions in this suite are falsifiable.

## Phase 0 inputs (measured BEFORE freezing; `results/prospective_pilot/summary.json`)

Single-defense baselines are **inputs to C1**, not outcomes, so they are measured first.

| defense | scaling | pixel | max-committed | suppresses both? |
|---|---|---|---|---|
| fltrust | 0.048 | 0.046 | **0.048** | YES |
| krum | 0.061 | 0.583 | 0.583 | no (pixel) |
| multi_krum | 0.575 | 0.783 | 0.783 | no |

Existing menu, same protocol: foolsgold 0.732, reputation 0.842, norm_clip 0.935, rfa 0.885,
coord_median 0.519, trimmed_mean 0.846.

**FLTrust regime** (`results/fltrust_regime.json`, 24 adversary-present rounds): rho mean 69.8
(max 413), r = 0.034, **S*r = 0.34**. This places FLTrust *between* the two safe regimes —
FoolsGold sits at S*r = 4.8 (regime A, ordering), Reputation at 0.017 (regime B, attenuation).
FLTrust is the closest case we have measured to the S*r ~ 1 hazard zone. We record this
explicitly because it makes several predictions below genuinely uncertain rather than routine.

## Conditions used

- **C1** (per-attack): for EVERY committed attack a, `min_d ASR(d,a) < 0.5`.
- **C2**: by Proposition 1 — cosine-similarity signals (foolsgold, fltrust) are invariant under
  positive per-client rescaling; consensus distance (reputation), norm magnitude (norm_clip) and
  pairwise distance (krum, multi_krum) are not; coordinate ordering and RFA residual are
  conditional.
- **C3**: margin/count. TrimmedMean budget is `floor(0.2*5) = 1`, exceeded whenever n_a = 2
  (22.2% of rounds under hypergeometric sampling).
- **DEGEN**: an aggregator as d1 emits no per-client transform, so the pair reduces to d2 alone.
  Its predicted label is therefore **d2's own label**, which may be LOW — that is not a
  framework success and is scored separately.

## FROZEN PREDICTIONS (20 pairs)

Confidence is registered per pair. "uncertain" means the framework's own quantities do not
clearly resolve the case; those are the informative ones.

### A. Predicted LOW — certified C1 ^ C2 ^ C3 (6)

| # | pair | C1 | C2 | C3 | pred | confidence | reasoning |
|---|---|---|---|---|---|---|---|
| 1 | fltrust -> rfa | Y (0.048) | Y | Y | LOW | high | FLTrust is a positive rescaling; RFA preserved under Thm 1(2) separation; Lemma 1 covers its ReLU zeros |
| 2 | foolsgold -> fltrust | Y (0.048) | Y | Y | LOW | high | Prop 1(a): FLTrust reads cosine to a server reference, exactly invariant under FG's rescaling |
| 3 | reputation -> fltrust | Y (0.048) | Y | Y | LOW | high | same invariance argument |
| 4 | norm_clip -> fltrust | Y (0.048) | Y | Y | LOW | high | same; NormClip is c_i = min(1, tau/||u_i||) > 0 |
| 5 | rfa -> fltrust | Y (0.048) | Y | Y | LOW | high | same; RFA-reweighting is positive per-client |
| 6 | fltrust -> coord_median | Y (0.048) | Y | Y | LOW | **uncertain** | S*r = 0.34: adversary is NOT extreme (so rank does not discard it) but payload is attenuated to ~3%. Nearest case to the hazard zone; genuinely could fail |

### B. Predicted HIGH — C2 failure with C1 satisfied (4) — the clean mechanism test

These isolate C2: C1 holds by FLTrust's own strength, so a failure is attributable to the
upstream transform destroying the downstream signal, nothing else.

| # | pair | C1 | C2 | pred | confidence | reasoning |
|---|---|---|---|---|---|---|
| 7 | fltrust -> reputation | Y | **N** | HIGH | high | consensus distance not invariant under heterogeneous rescaling (Prop 1c) |
| 8 | fltrust -> norm_clip | Y | **N** | HIGH | medium | norm magnitude destroyed by rescaling; but FLTrust alone is 0.048, so d2 may simply not matter |
| 9 | fltrust -> krum | Y | **N** | HIGH | medium | Krum ranks pairwise distances; heterogeneous rescaling reorders them |
| 10 | fltrust -> multi_krum | Y | **N** | HIGH | medium | same |

*Note on 8–10: FLTrust alone reaches 0.048, so if these come out LOW it may mean the downstream
stage is irrelevant rather than that C2 was preserved. We register HIGH as the criterion's
prediction and will report this confound explicitly either way.*

### C. Predicted HIGH — C3 / count budget (1)

| # | pair | C1 | C2 | C3 | pred | confidence | reasoning |
|---|---|---|---|---|---|---|---|
| 11 | fltrust -> trimmed_mean | Y | Y | **N** | HIGH | **uncertain** | TM budget exceeded in 22% of rounds. But at S*r = 0.34 the adversary is not extreme, so it may not occupy the trimmed tails at all — unlike FG->TM (S*r = 4.8) which did fail |

### D. Predicted HIGH — C1 failure (4)

| # | pair | C1 | pred | confidence | reasoning |
|---|---|---|---|---|---|
| 12 | foolsgold -> krum | **N** (pixel: min(0.732, 0.583) = 0.583) | HIGH | high | no constituent suppresses pixel; C2 also fails |
| 13 | reputation -> krum | **N** (pixel: min(0.842, 0.583) = 0.583) | HIGH | high | same |
| 14 | norm_clip -> multi_krum | **N** (pixel: min(0.935, 0.783) = 0.783) | HIGH | high | same |
| 15 | rfa -> multi_krum | **N** (pixel: min(0.845, 0.783) = 0.783) | HIGH | high | same |

### E. DEGEN — reduces to d2 alone (5); scored separately from certified predictions

| # | pair | reduces to | pred | confidence | reasoning |
|---|---|---|---|---|---|
| 16 | krum -> rfa | rfa alone (0.885) | HIGH | high | selection emits no per-client transform |
| 17 | krum -> coord_median | cm alone (0.519) | HIGH | medium | borderline: cm alone is 0.519, barely above threshold |
| 18 | multi_krum -> rfa | rfa alone (0.885) | HIGH | high | same |
| 19 | krum -> fltrust | fltrust alone (0.048) | **LOW** | high | degenerate but LOW — because d2 is strong, NOT because composition worked |
| 20 | multi_krum -> fltrust | fltrust alone (0.048) | **LOW** | high | same |

## Summary of frozen predictions

| category | n | predicted LOW | predicted HIGH |
|---|---|---|---|
| certified C1^C2^C3 | 6 | 6 | 0 |
| C2-failure (C1 held) | 4 | 0 | 4 |
| C3 / count budget | 1 | 0 | 1 |
| C1-failure | 4 | 0 | 4 |
| DEGEN (scored separately) | 5 | 2 | 3 |
| **total** | **20** | **8** | **12** |

Of these, **2 are registered as uncertain** (#6, #11) — both hinge on FLTrust's S*r = 0.34
sitting between the safe regimes.

## Scoring rules (fixed now)

1. Threshold: max-committed ASR < 0.5 = LOW.
2. Config: N=10, K=5, f=0.2, alpha=0.5, 50 rounds, cifar_cnn, seeds 42/43/44, both committed
   attacks. Identical to the rest of the paper.
3. The headline confusion matrix covers the **15 non-DEGEN pairs**. DEGEN pairs are reported
   separately because their outcome is determined by d2 alone, not by composition.
4. Predictions are **not revised** after seeing results. Misses are reported as misses.
5. If a predicted-FAIL pair comes out LOW, we check whether d2 alone already achieves LOW before
   attributing anything to the composition.

*Committed to git before `results/prospective_suite/` exists. The commit hash is the timestamp.*
