# Pre-Registration: Out-of-Sample PASS-Direction Validation

**Date:** 2026-07-22
**Author:** Anupam Mediratta
**Purpose:** Pre-registered predictions for out-of-sample validation of the Signal-Preservation Composability Criterion in the PASS direction — predicting which compositions SUPPRESS backdoors under novel conditions not used during criterion development.

## Motivation

Reviewer concern (W1): All three clean C1∧C2∧C3 PASS predictions (fg→rfa 0.045, fg→cm 0.131, rep→cm 0.352) were validated only against committed_scaling and committed_pixel attacks at N=10/cifar_cnn — the same setting used to develop the criterion. This pre-registration commits PASS-direction predictions under:
- **A.** A categorically different attack family (DBA — distributed backdoor) at higher adversarial fraction (f=0.4)
- **B.** A different scale and architecture (N=100, K=10, f=0.05, ResNet18)
- **C.** An independent composition-dominance prediction (fg→cm vs temporal fg/cm mixing)

These predictions are committed to version control BEFORE experiments are run.

---

## A: DBA Attack Transfer (f=0.4, N=10, K=5, cifar_cnn, 50 rounds)

### Why DBA is a meaningful out-of-sample test
DBA (Distributed Backdoor Attack) embeds sub-triggers across multiple adversarial clients at four image corners, requiring inter-client coordination. This is mechanistically different from:
- model_scaling: single centralized trigger + magnitude amplification
- backdoor_pixel: single centralized trigger, no update manipulation

DBA tests whether the criterion's signal-preservation logic generalizes beyond magnitude-based attacks to coordination-based attacks.

### Why f=0.4
DBA at f=0.2 is ineffective (ASR≈0.018 even against fedavg) because co-sampling of multiple adversaries per round is too rare at K=5. At f=0.4 (4 adversaries/10 clients), expected adversarial co-participation = K×f = 2.0, enabling DBA's distributed mechanism to function.

### Predictions

| Pair | Predicted | Reasoning |
|------|-----------|-----------|
| fg→rfa | PASS (ASR < 0.5) | FoolsGold: DBA clients train on similar sub-trigger patterns → high cosine similarity → downweighted. RFA: geometric median is robust to outlier positioning regardless of trigger geometry. C1∧C2∧C3 hold. |
| fg→cm | PASS (ASR < 0.5) | FoolsGold filters high-similarity DBA clients. CoordMedian's rank selection is trigger-geometry-agnostic. C1∧C2∧C3 hold. |
| rep→cm | PASS (ASR < 0.5) | Reputation: DBA clients produce updates far from median → downweighted. CoordMedian: rank-based, trigger pattern irrelevant. C1∧C2∧C3 hold. |
| nc→rep | FAIL (ASR ≥ 0.5) | NormClip does not suppress DBA (clipping doesn't prevent distributed embedding). Reputation applied after clipping has compressed the signal → C2-FAIL. |
| rfa→rep | FAIL (ASR ≥ 0.5) | RFA reweighting + reputation after loses tracking signal → C2-FAIL. |
| fg→tm | FAIL (ASR ≥ 0.5) | FoolsGold compresses boundaries → TrimmedMean admission thresholds fail on compressed updates → C3-FAIL. |

### Sanity check
- Include DBA vs fedavg (no composition) at f=0.4 to establish baseline. If DBA vs fedavg < 0.3, DBA is too weak to differentiate and experiment should be re-run at f=0.6.

### Methodology
- Seeds: [42, 43, 44, 45, 46]
- Metric: mean ASR across 5 seeds for each pair
- Decision rule: PASS if mean ASR < 0.5, FAIL if mean ASR ≥ 0.5
- Composition: apply d1 transform then d2 aggregate (same as `generic_compose()`)
- NormClip tau: 5.0 (standard)

---

## B: Scale Transfer (N=100, K=10, f=0.05, ResNet18, 50 rounds)

### Why scale transfer tests PASS generalization
At N=100, f=0.05 → 5 adversaries among 100 clients. Per round: expected adversarial participation = K×f = 0.5 adversaries. This means many rounds have ZERO adversaries participating — a radically different dynamic from N=10, f=0.2 where ~1 adversary participates per round. Additionally, ResNet18 is a fundamentally different architecture from cifar_cnn (4-layer CNN).

### Predictions

| Pair | Attack | Predicted | Reasoning |
|------|--------|-----------|-----------|
| fg→rfa | committed_scaling | PASS (ASR < 0.5) | FoolsGold's similarity filtering should be MORE effective with 10 clients per round providing diverse gradients. RFA's geometric median becomes more robust with larger sample. Adversarial participation is diluted. |
| fg→rfa | committed_pixel | PASS (ASR < 0.5) | Same logic. Pixel trigger without scaling amplification is even weaker. |
| fg→cm | committed_scaling | PASS (ASR < 0.5) | FoolsGold filters + CoordMedian rank selection. More honest clients strengthen both mechanisms. |
| fg→cm | committed_pixel | PASS (ASR < 0.5) | Same. CoordMedian's rank selection unaffected by architecture. |

### Methodology
- Seeds: [42, 43, 44, 45, 46]
- Model: ResNet18 (torchvision-style, adapted for CIFAR-10 32×32)
- Data: CIFAR-10, Dirichlet α=0.5
- Metric: mean ASR across 5 seeds
- Decision rule: PASS if mean ASR < 0.5

---

## C: fg→cm Composition Dominance (Independent Survivor)

### Why this is independent of the 189-config scan
The rep+tm survivor (§8.2) was identified via payoff-matrix scan across 189 configurations. fg→cm is identified SOLELY by criterion PASS status — it was never part of any game-theoretic scan or payoff-matrix search. This provides an independent test of composition-dominates-mixing.

### Predictions

| Condition | Predicted max committed ASR | Reasoning |
|-----------|----------------------------|-----------|
| fg→cm composed | < 0.2 | Dev-set result: 0.131. Composition always applies both defenses every round. |
| fg/cm 50/50 mixed | > 0.4 | In mixed mode, rounds with only cm have moderate ASR (~0.4 for scaling); rounds with only fg have moderate ASR. Temporal mixing means each round is protected by only ONE defense. |

**Composition dominates mixing:** YES (composed max ASR < mixed max ASR)

**VoPD in mixed condition:** Expect > 0 — oracle can observe which defense is drawn and adapt.

### Oracle table for mixed condition
- When fg is drawn: play model_scaling (FG has less filtering power vs scaling-only than vs pixel)
- When cm is drawn: play model_scaling (CM scaling ASR 0.450 > pixel ASR 0.377)

### Methodology
- N=10, K=5, f=0.2, cifar_cnn, 50 rounds
- Seeds: [42, 43, 44, 45, 46]
- Adversary policies: committed_scaling, committed_pixel, oracle
- Composed condition: always apply fg→cm (generic_compose)
- Mixed condition: per-round random selection from {foolsgold, coord_median} with prob (0.5, 0.5)
- Oracle: picks best attack per defense drawn (model_scaling for both fg and cm)
- Metric: mean ASR across 5 seeds for each policy × condition

---

## Summary of Pre-Registered Predictions

| Experiment | # PASS predictions | # FAIL controls |
|------------|-------------------|-----------------|
| A: DBA transfer | 3 (fg→rfa, fg→cm, rep→cm) | 3 (nc→rep, rfa→rep, fg→tm) |
| B: Scale transfer | 4 (fg→rfa×2, fg→cm×2) | — |
| C: Dominance | 1 (composed < mixed) | — |
| **Total PASS** | **7** (partially overlapping) | **3** |

If all PASS predictions validate: combined PASS accuracy = (3 dev + 7 OOS) / (3 dev + 7 OOS) = 10/10 in PASS direction.

---

*This file committed to git before any experiments are run. Commit hash serves as pre-registration timestamp.*
