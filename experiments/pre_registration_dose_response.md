# Pre-registration: dose–response test of C2

**Status: frozen before any write to `results/dose_response/`.** The predicted *shape* and the
decision rule for each arm below are reproduced verbatim in the `ARMS` list of
`experiments/run_dose_response.py`, which refuses to start until this file is committed and its
hash is recorded in `PREREG_COMMIT`.

## Why this design, and why the previous two could not work

Proposition (testability boundary): if `(d1 → d2)` satisfies C0 ∧ C1 and `a*` is a committed attack
witnessing C0, then `ASR(d2, a*) < 0.5` — the downstream defense suppresses `a*` on its own. So in
a **cross-defense** contrast, C1 and C2 co-vary by construction: **mechanism–effectiveness
confounding**. Two pre-registered suites failed for exactly this reason, the second
(`pre_registration_metric_swap.md`) after being built specifically to escape it.

The proposition constrains cross-defense designs only. It says nothing about holding `d2` **and**
the attack fixed and varying the *dose* of upstream disturbance. Along such a ladder standalone
effectiveness — the C1 input — is **numerically the same quantity at every rung**, so the confound
is impossible rather than merely matched. The proposition therefore stops being the reason C2
cannot be validated and becomes the derivation of the only design that can validate it.

This is the reviewer-named template made continuous: *same downstream defense + same attack + same
standalone effectiveness + controlled and independently measured disturbance of its statistic.*

## The controlled upstream transform

`d1 = dose_kappa<κ>` (`experiments/run_all_compositions.py`, `dose_coefficients`). For the `K`
participants of a round:

```
c_j  ∝  exp( κ · (2j/(K−1) − 1) ),   normalized so mean_j(c_j) = 1
```

- Exactly the invariance proposition's form `T(u_i) = c_i u_i`, `c_i > 0`, so the ladder is inside
  the theory's scope at every rung.
- **The dose is ρ.** `max(c)/min(c) = exp(2κ)` exactly — the same upstream weight ratio the theorem
  and the two mechanism-preserving regimes are stated in, so κ is a dial on the theory's own
  quantity. Verified before this freeze by reading ρ back off the *transformed update norms*
  (`results/dose_disturbance.json`): realized ρ matches nominal to 3 decimals at every rung and
  `mean(c)` stays inside `[0.99999, 1.00002]`.
- **`κ = 0` gives `c ≡ 1`**, so the transform is the identity and the composition *is* `d2`
  standalone. `apply_d1_transform` short-circuits and returns the update list unwrapped, so the
  `κ = 0` rung is bit-identical to standalone by construction, not by rounding.
- **Mean-1 (not sum-1, not max-1) normalization** holds the aggregate's scale fixed. This is the
  design choice that separates what we are measuring — disturbance of `d2`'s *statistic* (C2) —
  from the magnitude-contraction channel (annihilation lemma) that costs accuracy. Without it the
  ladder would confound C2 with attenuation.
- `j` is the client's position in a **`(seed, round)`-keyed pseudorandom permutation, not its client
  id.** Assigning by id would put the adversaries (ids `0…1`) at a fixed end of the ladder and
  systematically attenuate them, manufacturing a success. The permutation depends only on
  `(seed, round)`, so **every arm at a given seed and round receives the identical coefficient
  vector** — the arms differ in `d2` and nothing else.
- **Not a defense.** `dose_kappa<κ>` is an instrument for varying one quantity. It is never
  presented, evaluated, or recommended as a defense.

**Frozen grid:** `κ ∈ {0, 0.5, 1.0, 2.0}` ⟹ `ρ ∈ {1.00, 2.72, 7.39, 54.60}`. Four rungs spanning a
≈55× ratio, chosen to fit the available compute at n=5 with uniform provenance. **No rung is added
or removed after any result is seen**, in either direction.

## Phase 0, part 1 — standalone baselines (measured before this freeze)

Arms were selected by the paper's own power rule, fixed before this freeze: the ladder is
informative only where standalone `d2` genuinely suppresses the attack (ASR < 0.5 at clean accuracy
≥ 0.35). A baseline already above 0.5 has no suppression left to lose, so a flat curve there would
be a ceiling effect rather than preservation. Applying that rule to the measured baselines leaves
exactly four (`d2`, attack) cells, and they cover three invariance classes:

| # | arm (`d2` / attack) | class | statistic | standalone ASR (acc) | source |
|---|---|---|---|---|---|
| 1 | `krum` / scaling | (c) not invariant | summed pairwise Euclidean distance | 0.061 (0.577), n=3 | `results/prospective_pilot/summary.json` |
| 2 | `reputation` / scaling | (c) not invariant | Euclidean distance to client median | 0.017 (0.777), n=5 | `results/pure_defense_baselines/summary.json` |
| 3 | `cos_krum` / pixel | **(a) exactly invariant** | pairwise cosine | 0.300 (0.598), n=3 | `results/metric_swap_baselines/summary.json` |
| 4 | `coord_median` / pixel | (b) conditionally invariant | coordinate ordering | 0.443 (0.767), n=5 | `results/pure_defense_baselines/summary.json` |

`cos_krum` is a **mechanism-isolating ablation, not a proposed defense** (same standing as in
`pre_registration_metric_swap.md`). Its exact invariance was verified in code before it was ever
claimed: `results/cos_invariance_check.json`.

The `κ = 0` rung is **run fresh at all five seeds for every arm** rather than reused from this
table, so every rung of every curve comes from one code path at one seed set. The published
baselines then serve as an *independent cross-check* of the `κ = 0` rung
(`run_dose_response.py --harness-check`, per-seed, tolerance 1e-6), not as data in it. This also
removes the mixed-n problem visible above: the ladder itself is n=5 uniformly.

## Phase 0, part 2 — independently measured disturbance (measured before this freeze)

The abscissa must be measured, not assumed. `experiments/measure_dose_disturbance.py` pushes live
FL updates through the **shipped** `apply_d1_transform` and records what each rung does to each
arm's statistic — 5 seeds × 3 rounds = 15 rounds per attack, every κ evaluated on the *same* raw
updates so that trajectory divergence (the ladder's outcome) cannot contaminate the abscissa. **No
ASR is computed by this script**, which is why it can legitimately run before the freeze.

Measured, from `results/dose_disturbance.json`:

| arm | class | what is counted | κ=0 | κ=0.5 | κ=1.0 | κ=2.0 |
|---|---|---|---|---|---|---|
| 1 `krum` / scaling | (c) | rounds the selected client changes | 0/15 | 9/15 | 11/15 | 13/15 |
| 2 `reputation` / scaling | (c) | rounds the weight ordering changes | 0/15 | 13/15 | 13/15 | 14/15 |
| 3 `cos_krum` / pixel | **(a)** | rounds the selected client changes | **0/15** | **0/15** | **0/15** | **0/15** |
| 4 `coord_median` / pixel | (b) | mean fraction of *coordinates* whose median-attaining client changes | 0.000 | 0.251 | 0.378 | 0.487 |

Three things this establishes, all before any ASR exists:

1. **There is a dose.** Disturbance is 0 at the identity rung and rises monotonically with κ in
   every non-invariant arm. Without this there would be no experiment.
2. **The class-(a) invariance claim holds in the shipped code.** `cos_krum`'s selection is unchanged
   in 0 of 120 (arm, κ, seed, round) cells at ρ up to 54.6. The cosine *scores* move by at most
   1.1e-4 in relative terms — float32 normalization of a ~1.1M-dim vector, recorded rather than
   asserted, since asserting on it would test the floating-point unit and not the mechanism.
3. **Class (b) is graded, not binary,** and its disturbance is roughly half that of the class-(c)
   arms at every rung (0.487 vs 0.867/0.933 at κ=2.0). That asymmetry drives a frozen prediction
   below.

**What we deliberately do *not* build a prediction on.** Theorem 1(1)'s ordering-survival condition
is `ρ < S` with `S` the attack's coordinate extremeness, which would give arm 4 a crossing rung
fixed by a measured `S`. We measured `S` and it is **not a stable per-attack constant**: on
model-scaling it ranges 0.00–12.59 across 12 rounds (median 0.57), on the pixel backdoor 0.61–1.63
(median 0.92). The paper's `S = 10` is a stipulated illustrative value, not a measurement, and we do
not treat it as one. **No frozen prediction here uses `S`.** It is reported as a diagnostic, and the
instability is itself reported as a limitation of the theorem's operational checkability.

## Frozen predictions

All predictions are on **per-attack** ASR (the arm's own committed attack), n=5, with Student-t 95%
intervals, subject to the 0.35 clean-accuracy gate. Trend tests are **Jonckheere–Terpstra**,
one-sided for increasing trend, across the 4 ordered rungs × 5 seeds; α = 0.05. Direction is fixed
here, before any run, which is what licenses the one-sided form.

### Primary (pooled): ASR rise tracks measured disturbance

Across all 16 (arm, κ) cells, the increase in ASR over the arm's own κ=0 rung is predicted to be
**positively rank-correlated with the measured disturbance rate** in the Phase-0 table above.
Spearman ρ, one-sided, α = 0.05. This is the pooled dose–response claim and the one the design is
built for: the predictor is a *measured property of the statistic*, not the identity of the defense
and not its standalone effectiveness, both of which are held fixed within every arm.

- **Confirmed if** Spearman ρ > 0 at p < 0.05.
- **Refuted if** not, or if the correlation is negative.

### Secondary (per-arm shapes)

| # | arm | class | measured disturbance at κ=2.0 | **predicted shape** | confirmed if | **refuted if** |
|---|---|---|---|---|---|---|
| 1 | `krum` / scaling | (c) | 0.867 | **monotone rise** | JT increasing, p < 0.05 | no increasing trend at p < 0.05 |
| 2 | `reputation` / scaling | (c) | 0.933 | **monotone rise** | JT increasing, p < 0.05 | as above |
| 3 | `cos_krum` / pixel | **(a)** | **0.000** | **FLAT** | \|mean ASR(κ=2.0) − mean ASR(κ=0)\| < **0.15** *and* every rung < 0.5 | JT increasing at p < 0.05, or any rung ≥ 0.5 |
| 4 | `coord_median` / pixel | (b) | 0.487 | **monotone rise, shallower than arms 1–2** | JT increasing, p < 0.05, *and* its κ=2.0 rise smaller than both class-(c) arms' | no increasing trend, or a rise steeper than both class-(c) arms |

Arm 4's shape is **monotone rise, not threshold.** That is a change from the design sketch, made on
the strength of the Phase-0 disturbance measurement — which contains no ASR — and recorded here
before the freeze rather than after an outcome: its measured disturbance is graded and nonzero from
the first nonzero rung, so a flat-then-break threshold is not what the instrument delivers. Its
class-(b) signature is the *shallower slope*, which is why that comparison is part of the criterion.

**The 0.15 equivalence margin for arm 3 is fixed now and justified now**, not chosen to fit an
outcome: it is under a quarter of the +0.629 within-defense gap already measured for Krum, and
`0.300 + 0.15 = 0.450 < 0.5`, so "within the margin" and "still suppressing" coincide by design for
this arm. A failure to reject a trend is *not* treated as confirmation; confirmation requires the
equivalence bound to hold.

**Why arm 3 is a genuine falsifiable positive test, unlike the `·→FLTrust` pairs.** `cos_krum`'s
*selection* is bit-identical at every κ — now measured, 0/120 — but the aggregate it emits is
`c_j u_j` for the selected client `j`, so the 50-round trajectory is **not** bit-identical and
suppression can still be lost through the magnitude channel, as `rfa→cos_krum` already did
(Δ = −0.027). The claim under test is "selection preserved ⟹ suppression preserved, at ρ up to
54.6", and it can fail. Symmetrically, arms 1–2 can refute the framework's central fragility claim
by staying flat.

## Caveats recorded in advance, not discovered afterwards

1. **Arms 1–2 are on model-scaling and arms 3–4 on the pixel backdoor.** Not a free choice: on
   model-scaling *no* invariant `d2` suppresses at usable accuracy — that is the finding of the
   metric-swap round (`cos_krum` 0.078 at 0.146 accuracy), and it is why `cos_krum`/scaling is not
   an arm here. "Same attack" is required *within* each ladder, which is where the causal claim
   lives; across arms we compare **curve shapes and slopes**, never levels. Both committed attacks
   are covered by the suite as a whole.
2. **Arm 4 has thin headroom.** `coord_median`/pixel sits only 0.057 below the 0.5 threshold, so
   there is little room for its rise to be measured before it saturates. Reported as a continuous
   curve with intervals; the binary above/below-0.5 reading is weak evidence either way.
3. **Dose is uncorrelated with adversary status by construction, not adversary-adversarial.** A
   dose assignment that deliberately up-weights adversaries would be a strictly harder test and is
   **not** run here. Stated as a limitation; no claim is made about it.
4. **A synthetic upstream transform is not a real defense.** Ecological validity is tested
   separately, by plotting the two *real* upstream transforms already measured on these same cells
   (`norm_clip`, `rfa` from `results/metric_swap/summary.json`) as anchor points on the same
   measured-disturbance axis, using their rates from `results/cos_invariance_check.json`
   (`norm_clip→krum` 0/9 → 0.062; `rfa→krum` 6/9 → 0.691; `norm_clip→reputation` 3/9 → 0.792;
   `rfa→reputation` 9/9 → 0.882; `norm_clip→cos_krum` 0/9 → 0.300; `rfa→cos_krum` 0/9 → 0.273).
   If they fall far off the synthetic curve at matched measured disturbance, that is a finding about
   the instrument and is reported as one.
5. **Disturbance is measured on FedAvg trajectories, one shared set of raw updates per round.**
   That is what makes it a clean abscissa, but it means the rates in the Phase-0 table are not
   re-measured along each arm's own 50-round trajectory. Stated, not corrected.

## Protocol

Unchanged from the rest of the paper: CIFAR-10, `cifar_cnn`, N=10, K=5, f=0.2, Dirichlet α=0.5,
50 rounds. Scored on **per-attack** mean ASR at threshold **0.5**, with the **0.35** clean-accuracy
gate applied to every cell. Seeds **42, 43, 44, 45, 46** (n=5). `d2` is reached via
`server.aggregate(..., method=d2)`; `d1` is the dose transform above.

Runs: 4 arms × 4 rungs × 5 seeds = **80**.

**Sanity gate.** Undefended FedAvg reaches 0.667 on scaling and 0.768 on pixel, both ≥ 0.5, so
cells that come out high are informative rather than vacuous. One caveat stated up front: the
scaling sanity baseline sits at 0.100 clean accuracy, so the undefended scaling regime is itself a
collapsed one.

**Harness check, run before the ladder.** `κ = 0` at seed 42 must reproduce each arm's published
standalone value *at that seed*, to 1e-6. `κ = 0` is the exact identity and the participant RNG
stream does not depend on `d1`, so any discrepancy is a harness bug or a protocol difference between
the suites those baselines came from. Either way it is resolved before Phase 2. Because the ladder's
own `κ = 0` rung is measured fresh, a mismatch does not contaminate the curves — it means the
published baseline is not the same quantity and must not be cited as though it were.

## What each outcome would mean

- **Pooled correlation positive; arms 1–2 and 4 rise, arm 3 flat:** measured disturbance of the
  downstream statistic — not defense identity, not standalone effectiveness — controls whether
  suppression survives composition. This is the prospective positive test for C2 that the previous
  two suites could not contain, and it is the result the design is built to be able to fail at.
- **Arm 3 rises:** C2's invariance mechanism is refuted for this construction, with no appeal to
  "a different defense" available, because `d2` and the attack are held fixed. **That becomes the
  round's headline finding and is written as such.**
- **Arms 1–2 flat:** the framework's central claim that Euclidean statistics are fragile under
  upstream rescaling is refuted at ρ up to 54.6. Also written as such.
- **Pooled correlation absent while individual arms move:** the mechanism is defense-specific and
  measured disturbance is not a transferable predictor. Reported as a limit on the framework's
  generality.

**What this does not establish.** The ladder addresses the causal validation of C2 only. It does
**not** address recall, and it does **not** make the criterion a discovery tool — fg→rfa remains a
false negative of our own rule, and the ladder says nothing about synergy between two defenses that
individually fail. Those limitations stand unchanged.

## Non-negotiables

1. No predicted shape, decision rule, equivalence margin, or refutation criterion above is revised
   after seeing an outcome. A refutation is reported as a refutation.
2. No switching of committed attack per arm, κ grid, threshold, seed count, or accuracy gate.
3. Every reported number is recomputed from `results/dose_response/summary.json` and
   `results/dose_disturbance.json`, per seed, never transcribed.
4. `dose_kappa<κ>` is never presented as a defense; `cos_krum` is never presented as a proposed
   defense.
5. This file is committed before `results/dose_response/` is written; if that ordering cannot be
   demonstrated from `git log`, the suite is reported as non-prospective. The Phase-0 measurements
   above (`results/dose_disturbance.json`) legitimately predate it because they compute no ASR;
   that file is committed together with this one so the ordering is auditable.
