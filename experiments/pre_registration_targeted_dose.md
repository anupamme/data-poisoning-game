# Pre-registration: targeted-dose test of C2 (Round 12)

**Status: frozen before any write to `results/targeted_dose/`.** Every predicted shape and decision
rule below is reproduced verbatim in the `ARMS_S` / `ARMS_A` tables of
`experiments/run_targeted_dose.py`, which refuses to start until this file is committed and its hash
is recorded in `PREREG_COMMIT`.

## What Round 11 established, and the one thing it could not

The dose ladder (`pre_registration_dose_response.md`, frozen at `e711a95`) ran 80 runs and its
**primary prediction was refuted**: pooled Spearman of ASR rise against measured decision change
`ρ_s = +0.351`, one-sided `p = 0.091`, against a frozen criterion of `p < 0.05`. One arm confirmed
(reputation, `0.017 → 0.779`, `p = 1.7×10⁻⁶`), two were refuted, one was indeterminate, and **in
three of four arms ASR fell as the dose rose**. All of that stands and none of it is revisited here.

The paper reported, as a limitation of its own instrument, that mean-1 normalization fixes the
aggregate's scale but not any client's *relative* weight, so a wide spread mostly dilutes whichever
client carries the poison. That was a hypothesis. It is now a measurement
(`results/admission_measurement.json`, no ASR computed): under `dose_kappa<κ>` the adversary's share
of the total coefficient mass **falls monotonically 0.2667 → 0.2387 → 0.2154 → 0.1866** across the
rungs. The Round-11 ladder demonstrably varied the payload channel as well as the statistic.

A second Round-11 result forced a conceptual question rather than an instrumental one. Krum's
**decision** changed in 86.7% of rounds at `ρ = 54.6` and its **suppression did not deteriorate at
all**. Measuring the same rungs at the level of *what the aggregator admits* rather than *whether
its decision moved* shows why that is not a paradox: at `κ = 2.0` Krum's decision changed 0.867 of
the time but the adversary's **admission** changed only 0.133 of the time. A selection that flips
from one benign client to another disturbs the statistic maximally and the mechanism not at all.

## Two candidate readings of C2, and why this suite exists

The paper has been sliding between four distinct claims. They are separated here and given names:

| level | claim | Round-11 status |
|---|---|---|
| statistic invariance | `S(T(U)) = S(U)` | `cos_krum`: exact, 0/120 |
| decision invariance | `d₂`'s selection/ordering unchanged | Krum: fails at 0.867 |
| **admission invariance** | the adversarial input/mass `d₂` admits is unchanged | Krum: 0.133 |
| suppression preservation | ASR stays below threshold | Krum: holds |

- **H-statistic** (C2 as written): what matters is disturbance of `d₂`'s statistic, read off its
  decisions. This is the hypothesis Round 11 tested and could not confirm.
- **H-admission** (the refinement): what matters is whether the transform changes the *adversarial*
  input `d₂` admits. Benign-to-benign churn is not mechanism destruction.

**H-admission is adopted after seeing a Round-11 outcome and is therefore retrospective**, exactly
as C0 was. It is recorded here, before this suite runs, so that it gets one genuinely prospective
test rather than being introduced afterwards as an explanation.

### The post-hoc re-analysis does NOT support H-admission, and that is recorded here

`experiments/analyze_admission.py` re-scores Round 11's own 16 cells against both predictors. The
replay reproduces the frozen Round-11 abscissa exactly (Krum 0.600/0.733/0.867, reputation
0.867/0.867/0.933, `cos_krum` 0.000, `coord_median` 0.251/0.378/0.487) and reproduces its primary
statistic exactly (`ρ_s = +0.351`, `p = 0.0915`). Against that:

| pooled predictor of ASR rise, n = 16 | `ρ_s` | one-sided `p` |
|---|---|---|
| decision change (Round 11's frozen predictor) | **+0.351** | 0.0915 |
| admission change (the refinement) | **+0.271** | 0.1550 |
| change in adversarial coefficient share | **+0.354** | 0.0894 |

**As a single pooled predictor the refinement is worse than the thing it proposes to replace.** The
only support is a two-channel model, `rise ~ admission + Δshare`, `R² = 0.494` against `R² = 0.402`
for the same model built on decision change, with both coefficients positive as the reading
requires (`+5.31` admission, `+5.64` Δshare) — at `n = 16`, which is not a resolution. This suite is
therefore a **test** of H-admission, not a confirmation of it, and the null above is reported in the
paper whatever the outcome here.

## The two instruments

Both are synthetic `d₁`s that read adversary identity. They are **instruments for causal
identification, not defenses**: no deployable defense knows which clients are adversarial. Neither
is proposed for deployment, and `cos_krum` remains a mechanism-isolating ablation.

### Mode S — statistic-only. `d1 = doseS_kappa<κ>`

For the `K` participants of a round, `A` adversarial and `B` benign:

```
c_i = 1                                             for i ∈ A
c_j ∝ exp( κ · (2j/(|B|−1) − 1) ),  mean_B(c) = 1   for j ∈ B
```

`j` is the position in a `(seed, round)`-keyed permutation **of B**.

- **The adversary's coefficient is exactly 1.0 at every rung**, so Lemma 1's attenuation channel is
  closed by construction. Verified before this freeze: `c_adv ∈ [1.000000000000, 1.000000000000]`,
  maximum deviation from 1 exactly `0.00e+00`, at every κ.
- **The adversarial coefficient share is constant across rungs**: `0.266667` at all four κ, total
  spread `3.5×10⁻⁷` — float32 read-back noise off ~1.1M-dimensional norms, against the `8.0×10⁻²`
  the Round-11 instrument moved it by. *This is the assertion the whole design rests on.*
- `mean_i(c_i) = 1` over all participants exactly, so aggregate scale is held as in Round 11.
- `ρ = max(c)/min(c) = exp(2κ)` exactly: realized `[54.5966, 54.5997]` against nominal `54.5982` at
  κ = 2, `mean(c) ∈ [0.99999, 1.00001]`. The dose is the theorem's own quantity, matched rung for
  rung to Round 11.

### Mode A — payload-only. `d1 = doseA_nu<ν>`

```
c_i = γ·s  for i ∈ A,   c_j = s  for j ∈ B,   γ = exp(ν),   s = K / (K + |A|(γ−1))
```

- Benign clients share a **common** coefficient, so their relative weighting is untouched; what
  moves is the adversary's share of the aggregate. Measured before this freeze:
  `0.0494 → 0.1218 → 0.2667 → 0.4845 → 0.7096` across `ν = −2, −1, 0, +1, +2`.
- `s > 0` for every `γ > 0`, so every coefficient stays strictly positive and the transform stays
  inside the invariance proposition's hypothesis. `mean(c) = 1` exactly.
- Realized adversary-to-benign ratio is exactly `γ`; verified `ρ` read-back `[7.3890, 7.3894]`
  against nominal `7.3891` at `|ν| = 2`.

## Arms

Selected by the paper's own two rules, applied before any ASR exists: **(1)** standalone `d₂` must
genuinely suppress the attack (ASR `< 0.5` at clean accuracy `≥ 0.35`), or there is no suppression
to lose; **(2)** the instrument must actually move that arm's quantity, or there is no dose and no
experiment.

### Mode S — κ ∈ {0, 0.5, 1.0, 2.0}, i.e. ρ ∈ {1.00, 2.72, 7.39, 54.60}

| arm (`d₂` / attack) | class | measured decision change (κ=0.5/1/2) | measured admission change | **H-statistic predicts** | **H-admission predicts** |
|---|---|---|---|---|---|
| `krum` / scaling | (c) | 0.533 / 0.800 / 0.733 | **0.000 / 0.000 / 0.000** | **monotone rise** | **flat** |
| `reputation` / scaling | (c) | 0.933 / 1.000 / 1.000 | 0.002 / 0.008 / 0.020 | **rise ≥ Round 11's** | **rise ≪ Round 11's** |
| `cos_krum` / pixel | (a) | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | **flat** | **flat** |

**The Krum arm is the adjudicating arm**: the two hypotheses make *opposite* predictions on it, and
one of them must fall. Its decision change is 0.53–0.80 while its admission change is exactly zero
at every rung.

Seeds 42–46 (`n = 5`) for the two class-(c) arms. **`cos_krum` runs at `n = 8`** (seeds 42–49),
frozen here and applying to that arm only: its Round-11 identity rung was bimodal across seeds
(0.003–0.864, 95% CI [0.058, 0.922]), so the equivalence margin its FLAT prediction is tested
against never had the power that test presumed.

### Mode A — ν ∈ {−2, −1, 0, +1, +2}, i.e. γ ∈ {0.135, 0.368, 1.000, 2.718, 7.389}

| arm (`d₂` / attack) | class | measured *absolute* adversarial admission across ν | prediction |
|---|---|---|---|
| `reputation` / scaling | (c) | 0.2115 / 0.0552 / 0.0014 / 0.0000 / 0.0000 | **single-peaked in γ, maximal at or adjacent to ν = 0** |
| `coord_median` / pixel | (b) | 0.3742 / 0.3527 / 0.2855 / 0.1912 / 0.1239 | **single-peaked in γ, maximal at or adjacent to ν = 0** |

This is Theorem 1's two mechanism-preserving regimes stated as one falsifiable shape. Regime A: an
adversary kept extreme enough is discarded, so ASR falls as γ grows — and the measurement confirms
the *mechanism* in advance, admission dropping to 0.000 (reputation) and 0.124 (coord_median) at
γ = 7.39. Regime B: an adversary attenuated far enough carries no payload, so ASR falls as γ
shrinks — **note that admission moves the other way here**, rising to 0.2115 and 0.3742 at
γ = 0.135, so the prediction is specifically that the 7.4× payload shrinkage dominates the increased
admission. A monotone curve in either direction refutes the two-regime picture.

**`krum`/scaling was considered for Mode A and dropped before the freeze**, on rule (2): its
measured admission is `0.0833 / 0.0000 / 0.0000 / 0.0000 / 0.0000` across ν — Krum discards the
scaling adversary at every γ ≥ 0.368, so there is no dose to give it. Recorded rather than quietly
omitted.

## Decision rules

All tests one-sided at α = 0.05, Jonckheere–Terpstra across the arm's rungs in the order given, with
a fixed-seed permutation p-value reported alongside the normal approximation.

**Mode S, `krum` (adjudicating).**
- JT *increasing* `p < 0.05` ⟹ **H-statistic confirmed, H-admission refuted**.
- `|mean ASR(κ=2) − mean ASR(κ=0)| < 0.15` **and** every rung `< 0.5` ⟹ **H-admission confirmed,
  H-statistic refuted**.
- Neither ⟹ **indeterminate**, reported as such and not scored in our favour.

**Mode S, `reputation` (quantitative).** Its Round-11 rise at κ=2 was `+0.762` (`0.017 → 0.779`) at
an admission change of 0.152; Mode S reaches an admission change of only 0.020 at the same κ while
its *decision* change is larger (1.000 vs 0.933).
- Mode-S rise at κ=2 significantly **smaller** than Round 11's (one-sided Welch, `p < 0.05`)
  ⟹ **H-admission confirmed**.
- Mode-S rise at κ=2 not significantly smaller, **and** JT increasing `p < 0.05`
  ⟹ **H-statistic confirmed**.
- Neither ⟹ indeterminate.

**Mode S, `cos_krum` (preservation control, `n = 8`).** Both hypotheses predict flat.
- Confirmed: `|mean ASR(κ=2) − mean ASR(κ=0)| < 0.15` **and** every rung `< 0.5`.
- Refuted: JT increasing `p < 0.05` **or** any rung `≥ 0.5`.
- Neither ⟹ indeterminate. A rise here refutes **both** hypotheses and the practical content of the
  invariance proposition, with the magnitude channel already closed — the strongest single negative
  this suite can produce.

**Mode A, both arms.**
- Confirmed: JT *decreasing* at `p < 0.05` on `{ν=0, +1, +2}` **and** JT *decreasing* at `p < 0.05`
  on `{ν=0, −1, −2}`.
- Refuted: JT *increasing* at `p < 0.05` on **either** side.
- Neither ⟹ indeterminate. The arg-max rung is reported in every case.

**Pooled secondary, frozen now.** Across all Mode-S and Mode-A cells jointly, ASR rise is regressed
on measured admission change and on change in adversarial coefficient share. The two-channel reading
is confirmed only if **both** coefficients are positive with the admission coefficient's one-sided
`p < 0.05`. This is the prospective version of the `R² = 0.494` post-hoc model above.

**Accuracy gate.** Any cell with mean clean accuracy `< 0.35` is uninterpretable: a low ASR there is
a collapsed model, not suppression. Reported per cell and flagged wherever it feeds a verdict.
`cos_krum` failed this gate at κ = 2 in Round 11 (0.220) and may fail it again.

## Caveats recorded in advance

1. **Mode S has nothing to impose in 20.0% of rung-rounds** (24/120 measured): a round with no
   adversary, or with fewer than two benign participants, admits no benign dispersion. The effective
   dose is diluted accordingly and the fraction is reported with the results.
2. **Mode A uniformly rescales the benign clients** by `s ∈ [0.28, 0.74]` at `|ν| = 2`. A uniform
   rescale leaves the *relative* benign weighting — and therefore the statistic among the benign
   majority — unchanged, but it does change the magnitude of what a selection aggregator outputs.
   This is disclosed as a residual channel that Mode A does not close.
3. **The post-hoc re-analysis does not support H-admission** (table above). Recorded before the run
   so it cannot later be presented as if the refinement arrived with support.
4. Class-(a)/(b) arms are on the pixel backdoor and class-(c) arms on model-scaling, as in Round 11
   and for the same reason: on model-scaling no invariant `d₂` suppresses at usable accuracy. "Same
   attack" is required *within* each arm, which is where the causal claim lives; across arms we
   compare shapes and slopes, never levels.
5. `coord_median`/pixel sits only 0.057 below the 0.5 threshold standalone, so its above/below
   reading is weak either way; the continuous curve is what is reported.
6. Neither mode addresses recall or the discovery limitation. `fg→rfa` remains a false negative of
   our own rule.

## Non-negotiables

1. No predicted shape, decision rule, equivalence margin, or refutation criterion above is revised
   after seeing an outcome. A refutation is reported as a refutation, and an arm meeting neither
   criterion is reported indeterminate rather than scored in our favour.
2. No switching of committed attack per arm, κ or ν grid, threshold, seed count, or accuracy gate.
3. Every reported number is recomputed from `results/targeted_dose/summary.json` and
   `results/admission_measurement.json`, per seed, never transcribed.
4. `doseS_kappa<κ>` and `doseA_nu<ν>` are never presented as defenses; `cos_krum` is never presented
   as a proposed defense.
5. **Nothing in Round 11 is rescored.** Its pooled refutation, its two refuted arms and its
   indeterminate arm stand exactly as reported, whatever this suite shows.
6. This file is committed before `results/targeted_dose/` is written; if that ordering cannot be
   demonstrated from `git log`, the suite is reported as non-prospective. The Phase-0 measurements
   (`results/admission_measurement.json`) legitimately predate it because they compute no ASR; that
   file is committed together with this one so the ordering is auditable.
