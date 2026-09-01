# Pre-registration: seed top-up of the flagship Mode-S Krum arm (Round 34, item A)

**Status: frozen before any write to `results/dose_seed_topup/`.** The seed list, the primary
interval, the equivalence criterion and the reversal clause below are reproduced verbatim in
`experiments/run_dose_seed_topup.py`, which refuses to start until this file is committed and its
hash is recorded in `PREREG_COMMIT`.

## What is being extended, and what is not

The adjudicating arm of Round 12 — Mode S (`doseS_kappa<κ>`) into `krum` under `committed_scaling` —
was frozen at `5130cec` and run at `n = 5` (seeds 42–46). Its published result is
`|mean ASR(κ=2) − mean ASR(κ=0)| = 0.026 < 0.15` with every rung far below 0.5 and Jonckheere–
Terpstra *increasing* `p = 0.70`, i.e. **H-admission confirmed, H-statistic refuted** on the one arm
where the two hypotheses make opposite predictions.

**That verdict stands as reported and is not revisited.** This document does not re-test it. It adds
seeds so that the interval around the equivalence claim can be *stated*, which the published version
never did: an equivalence claim at `n = 5` whose width is not reported is not readable as evidence,
and no reviewer should have to take a point estimate of 0.026 on faith.

Per-seed ASR as published, seeds 42–46 in order:

| rung | ρ | per-seed ASR | mean | mean accuracy |
|---|---|---|---|---|
| κ = 0.0 | 1.00 | 0.1149 / 0.0209 / 0.0469 / 0.0567 / 0.0694 | 0.0618 | 0.5648 |
| κ = 0.5 | 2.72 | — | 0.0161 | — |
| κ = 1.0 | 7.39 | — | 0.0229 | — |
| κ = 2.0 | 54.60 | 0.0183 / 0.0276 / 0.0440 / 0.0510 / 0.0374 | 0.0357 | — |

Paired per-seed difference `ASR(κ=2) − ASR(κ=0)`: mean **−0.0261**, sd **0.0419**, so the two-sided
95% Student-t interval at `n = 5` (`t₄ = 2.776`) is **[−0.078, +0.026]**, half-width **0.052**, which
is **34.7% of the ±0.15 margin**. Holding that sd, `n = 20` (`t₁₉ = 2.093`) gives a half-width of
**0.020**, i.e. **13.1% of the margin**. That factor-of-2.7 reduction in half-width is the entire
purpose of the run.

## Why this is not optional stopping, stated before any new seed runs

Adding seeds after seeing a result is exactly the practice this paper condemns elsewhere, so the
licence has to be earned in advance and in writing.

1. **The published interval already clears the margin.** `[−0.078, +0.026] ⊂ (−0.15, +0.15)` at
   `n = 5`. The top-up therefore cannot rescue a claim that is currently failing — there is no such
   claim. It can only narrow an interval that already passes, or reveal at larger `n` that the
   passage was an artifact of five seeds. Only the second outcome is news, and it is news against us.
2. **The reversal clause.** If the `n = 20` interval is **not** contained in `(−0.15, +0.15)`, we
   report the flagship equivalence claim as **refuted by our own top-up**, in the abstract, and the
   (P3)⇏(P4) witness is withdrawn to "not established at `n = 20`". Pre-committing to publish a
   reversal is the only thing that licenses adding seeds, and it is the reason this file exists
   before the runs rather than alongside them.
3. **The seeds are fixed here**, contiguously, with no look-ahead: **47–61**, 15 new seeds, `n = 20`
   total. No stopping rule, no interim look, no extension of this list. If the runs are interrupted,
   the analysis reports the `n` actually reached and the interval at that `n`; it does not resume
   until a threshold is crossed.
4. **Precedent, not innovation.** `SEEDS8` in `run_targeted_dose.py:90` extended the `cos_krum` arm
   from 5 seeds to 8 for the identical reason — interval width, not a second test — and was frozen
   at `5130cec` before any outcome existed. This is the same move on the arm that turned out to
   matter most.

## The primary quantity, chosen in advance and deliberately the wider of the two

Both of the following are pre-registered and **both are reported**:

- **Primary: the paired interval.** `d_s = ASR_s(κ=2) − ASR_s(κ=0)` per seed `s`; the two-sided 95%
  Student-t interval on `mean(d)` with `n − 1` degrees of freedom. Pairing is real by construction:
  a seed fixes the Dirichlet partition, the model initialization and the participant sampling
  stream, so the two rungs at a given seed differ only in `d₁`.
- **Secondary: the unpaired interval** on the difference of rung means, pooled sd, `2n − 2` degrees
  of freedom. This is the form the frozen Round-12 rule is literally written in
  (`|mean ASR(κ=2) − mean ASR(κ=0)|`).

**At `n = 5` the paired interval is the *wider* of the two** (half-width 0.052 against 0.038), and it
is nominated as primary anyway. This is recorded now so that the choice cannot be read afterwards as
having selected whichever interval came out narrower.

**Equivalence verdict.** Containment of the **two-sided 95%** interval in `(−0.15, +0.15)`. This is
strictly more conservative than TOST at `α = 0.05`, which is equivalent to containment of the 90%
interval; a 95% interval inside the margin therefore implies both one-sided TOST tests reject at
`α = 0.05`. Both are reported: the interval, and the two one-sided `t` statistics with their
`p`-values, so the TOST reading the reviewer asked for is legible and the conservative criterion is
the one that decides.

## Decision rules

Margin, floor and test direction are all **carried forward unchanged** from `5130cec`. Nothing here
is a new threshold.

- **Equivalence confirmed at `n = 20`:** the primary 95% interval is contained in `(−0.15, +0.15)`
  **and** every rung mean is `< 0.5`.
- **Equivalence refuted at `n = 20`:** the primary interval is not contained in `(−0.15, +0.15)`,
  **or** any rung mean is `≥ 0.5`. Reported as a refutation of our own published claim, in the
  abstract.
- **H-statistic retest:** Jonckheere–Terpstra *increasing* across κ ∈ {0, 0.5, 1, 2} at `n = 20`,
  one-sided α = 0.05, normal approximation plus a fixed-seed permutation `p`-value, using the
  canonical `jonckheere` in `analyze_dose_response.py`. `p < 0.05` ⟹ H-statistic confirmed at
  `n = 20` and the published refutation is withdrawn.
- **Neither ⟹ indeterminate**, reported as such and not scored in our favour.
- **Accuracy gate.** Unchanged: `ACC_FLOOR = 0.35` on a **rung's mean** clean accuracy, per the
  existing convention. Per-seed accuracies are reported for every new run, and any individual seed
  below the floor is flagged even where its rung mean clears it.

## The one computational difference between the new seeds and the old, disclosed here

The published κ = 0 rung is **imported** from `results/dose_response/` rather than re-run: at
`κ = 0` `apply_d1_transform` returns the update list unwrapped, and `run_one`'s participant RNG
stream does not depend on `d₁`'s name, so the Round-11 `dose_kappa0.0` run *is* the Mode-S identity
run. `results/dose_response/` holds seeds 42–46 only, so **for seeds 47–61 the identity rung is
computed here**, and each new row records `<computed here>` as its provenance.

This is the established practice of this suite, not a departure from it: the published `cos_krum`
arm already mixes the two, with seeds 42–46 marked `results/dose_response kappa=0 (identical
computation)` and seeds 47–49 computed in place.

The bit-equality claim behind the mixed provenance is re-checked rather than assumed:
`run_dose_seed_topup.py --harness-check` computes `κ = 0` **in-suite at seed 42, where the imported
value already exists**, and asserts the two agree to `< 1e-9`. A published seed is used deliberately;
at a new seed there is nothing to compare against and the check would be vacuous. One run settles it
for all fifteen new seeds, because the code path does not depend on the seed. **If that assert fails
the top-up does not run**, because the mixed-provenance κ = 0 rung would then not be one rung.

## Caveats recorded in advance

1. **The sd is estimated from five seeds.** The projected half-width of 0.020 at `n = 20` assumes
   `sd = 0.0419` carries; the realized sd may be larger, and the realized interval is what is
   reported. No projected number appears in the paper.
2. **Mode S has nothing to impose in 20.0% of rung-rounds** (24/120 measured at the frozen
   configuration): a round with no adversary, or with fewer than two benign participants, admits no
   benign dispersion. Carried forward from `5130cec` and re-reported at `n = 20`.
3. **Narrowing an equivalence interval does not widen the claim.** At `n = 20` this remains one
   aggregator, one attack, one dataset, one synthetic instrument that reads adversary identity. The
   top-up buys precision on the (P3)⇏(P4) witness and nothing else; it is not evidence about any
   other arm, and Mode S is not a defense.
4. **Seeds 47–49 are already used by the `cos_krum` arm** under `SEEDS8`. Different arm, different
   cell key, no collision — recorded because the overlap looks like one.
5. `results/targeted_dose/summary.json` is **not written** by this suite. The top-up writes its own
   directory and the two are merged only at analysis time, where the five published seeds are
   asserted to reproduce bit-identically before any pooled number is printed.

## Non-negotiables

1. No interval definition, equivalence margin, accuracy floor, test direction or refutation
   criterion above is revised after seeing an outcome.
2. The seed list is 47–61. It is not extended, truncated by inspection, or filtered.
3. Every reported number is recomputed per seed from `results/targeted_dose/summary.json` and
   `results/dose_seed_topup/summary.json`, never transcribed.
4. `SEEDS5` in `run_targeted_dose.py` is not edited, and `results/targeted_dose/` is not rewritten.
5. **The published `n = 5` verdict is reported alongside the `n = 20` verdict, whatever the latter
   is.** If they disagree, both appear, and the disagreement is the result.
6. This file is committed before `results/dose_seed_topup/` is written; if that ordering cannot be
   demonstrated from `git log`, the top-up is reported as non-prospective.
