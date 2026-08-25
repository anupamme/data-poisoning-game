# Pre-registration: rank-discordant replication arm for the admission channel (Round 15)

**Status: frozen before any write to `results/dose_replication/`.** The predicted ordering and every
decision rule below are reproduced verbatim in `experiments/run_dose_replication.py`, which refuses to
start until this file is committed and its hash is recorded in `PREREG_COMMIT`. This suite adds **one
arm** and **15 new 50-round runs**. It rescores nothing.

## Why one more arm, and why this one

Round 12's targeted suite produced the paper's flagship negative result on the Krum arm: under Mode S
the adversary's coefficient is pinned at exactly 1.0, Krum's **decision** changed in 80% of rounds,
its **admission** of adversarial mass changed in exactly 0% of rounds, and its suppression did not
deteriorate. Statistic disturbance was causally irrelevant to suppression there. That result stands.

The *positive* half — that admission is the channel that does matter — is the paper's weakest claim,
and it is weak for a structural reason that this arm is designed to fix. Across the three
interpretable Mode-S arms already run, the two candidate predictors **co-vary**, so neither the
existing arms nor any contrast among them can order the outcomes:

| arm (`d₂` / attack) | Prop-1 class | decision change @κ=2 | admission change @κ=2 | observed Mode-S rise, κ=0→2 |
|---|---|---|---|---|
| `krum` / scaling | (c) | 0.733 | **0.000** | `0.062 → 0.036` = **−0.026** |
| `reputation` / scaling | (c) | 1.000 | 0.020 | `0.017 → 0.195` = **+0.178** |
| `cos_krum` / pixel | (a) | 0.000 | 0.000 | fell to acc 0.223 — fails the accuracy gate |

Reputation is higher than Krum on **both** predictors and had the larger rise. That ordering is
consistent with either reading, which is exactly why the pooled two-channel model came out
unconfirmed (`R² = 0.24`, admission coefficient one-sided `p = 0.055`).

`coord_median` / pixel under Mode S breaks the tie, because it is **rank-discordant**: it has the
*smallest* decision change of any interpretable arm and the *largest* admission change. From
`results/admission_measurement.json`, computed in Round 12 and **before any ASR existed for this
cell** (`experiments/measure_admission.py` trains no model to convergence and computes no ASR, by
design, precisely so its numbers can precede a freeze):

| κ | ρ | decision change | admission change | aggregate displacement |
|---|---|---|---|---|
| 0.0 | 1.00 | 0.0000 | 0.0000 | 0.00 |
| 0.5 | 2.72 | 0.2315 | 0.0046 | 0.26 |
| 1.0 | 7.39 | 0.3576 | 0.0125 | 0.44 |
| 2.0 | 54.60 | **0.4821** | **0.0332** | 0.67 |

So the two readings make **opposite predictions about where this arm's rise falls** relative to the
two arms already run:

- ordering by **decision change** — reputation (1.000) > krum (0.733) > **coord_median (0.482)** —
  predicts this arm's rise is the **smallest** of the three.
- ordering by **admission change** — **coord_median (0.033)** > reputation (0.020) > krum (0.000) —
  predicts this arm's rise is the **largest** of the three.

It is also the only viable arm that is a different *kind* of aggregator. Krum and Reputation are
selectors and weighted averagers reading pairwise or consensus distance (Prop 1c). Coordinate median
is a coordinate-wise order statistic (Prop 1b, conditionally invariant). Whatever this arm shows
therefore also answers whether the flagship result is a property of selector-type defenses or of
mechanism preservation generally.

## Arm

One arm, selected by the paper's two pre-existing rules applied before any ASR for this cell exists:
**(1)** standalone `d₂` genuinely suppresses the attack — `coord_median` / pixel is `0.443` at clean
accuracy `0.767`, so there is suppression to lose; **(2)** the instrument moves that arm's quantity —
decision change reaches 0.482 and admission change 0.033, both nonzero.

| `d₁` | `d₂` | attack | rungs | seeds |
|---|---|---|---|---|
| `doseS_kappa<κ>` | `coord_median` | `committed_pixel` | κ ∈ {0, 0.5, 1.0, 2.0} | 42–46 (`n = 5`) |

Nothing about Mode S is changed: adversaries pinned at `c = 1.0` exactly, benign spread over
`ρ = exp(2κ)` normalized to mean 1 among the benign, adversarial coefficient share constant at
`0.266667` across rungs (spread `3.5×10⁻⁷`). The attenuation channel is closed by construction.

**The κ = 0 rung is imported, not re-run**, from `results/dose_response/summary.json`:
`doseS_kappa0.0` returns the update list unwrapped, and the participant RNG stream does not depend on
`d₁`'s name, so at the same seed it is bit-for-bit the same computation as the Round-11 `κ = 0` rung.
`--harness-check` re-runs it at seed 42 and asserts equality to `1e-6`; if that fails the import is
invalid and the identity rung must be re-run. Identity rung as imported: mean `0.443`, sd `0.129`,
per-seed `[0.395, 0.287, 0.449, 0.677, 0.405]`, mean clean accuracy `0.767`. It is **unimodal** — the
defect that cost the `cos_krum` arm its power (identity spread 0.003–0.864) is not present here.

**New compute: 15 runs** (3 rungs × 5 seeds), ≈2.8 h wall clock at ~11 min/run on the hardware
described in the paper.

## Frozen decision rules

Primary quantity: `Δ = mean ASR(κ=2) − mean ASR(κ=0)`, the same rise statistic used for the existing
Mode-S arms. Comparison values are the two already-published rises, `−0.026` (krum) and `+0.178`
(reputation), and the suite's existing equivalence margin `EQUIV_MARGIN = 0.15`. No new constant is
introduced.

**Primary — the rank-discordance test.**
- `Δ > +0.178` ⟹ **admission ordering confirmed.** The arm with the largest admission change and the
  smallest decision change produced the largest rise. This is the paper's first prospective support
  for the positive half of the claim.
- `Δ < +0.150` ⟹ **admission ordering refuted.** The arm with the largest admission change produced a
  rise within the flat margin, so admission does not order the outcomes; the positive claim is scoped
  to selector-type defenses and written that way. Note this outcome simultaneously **replicates the
  flagship negative** on a structurally different defense: a decision change of 0.482 with
  suppression preserved.
- `+0.150 ≤ Δ ≤ +0.178` ⟹ **indeterminate**, reported as such and not scored in our favour.

**Secondary — within-arm shape.** Jonckheere–Terpstra across κ ∈ {0, 0.5, 1.0, 2.0}, one-sided at
α = 0.05, with a fixed-seed permutation p-value alongside the normal approximation. **Both** readings
predict a rise here, since both predictors move on this arm; a flat or falling curve therefore refutes
both, with the attenuation channel already closed, and is the strongest single negative this arm can
produce. Reported whichever way it comes out.

**Tertiary — the pooled two-channel model, re-fit.** `rise ~ admission change + Δ(adversarial
coefficient share)` is re-fit with these 3 new cells added to the existing 22. Pre-specified: the
two-channel reading is confirmed only if both coefficients are positive **and** the admission
coefficient's one-sided `p < 0.05`. It is currently `R² = 0.24`, `p = 0.055`. Three cells may not move
it. Whether it moves or not is reported.

**Accuracy gate, unchanged.** Any cell with mean clean accuracy `< 0.35` is uninterpretable — a low
ASR there is a collapsed model, not suppression. The confounded companion ladder reached accuracy
0.690 at κ = 2 on this cell, so the gate is not expected to bind, but it is applied.

## Disclosures recorded in advance

1. **The confounded companion ladder on this exact cell is already known, and it falls.**
   `results/dose_response/summary.json` has `dose_kappa<κ>` → `coord_median` / pixel at
   `0.443 → 0.428 → 0.327 → 0.170`, a monotone fall against a frozen prediction of monotone rise —
   one of Round 11's refuted arms. The prediction above is prospective with respect to Mode S, but it
   is not formed in ignorance of that fall. Mode S closes the attenuation channel that explains it,
   which is why the two ladders are expected to differ; that expectation is itself part of what is
   being tested.
2. **This arm was available in Round 12 and was not run.** It is being added now, with predictions
   frozen from the same pre-ASR instrument file Round 12 used. It is not a new measurement introduced
   to rescue a result — but it is an arm chosen after seeing that the existing arms could not order
   the outcomes, and that is the honest description of why it exists.
3. **The primary test compares magnitudes across arms that use different attacks.** `coord_median`
   runs on the pixel backdoor and the two comparison arms on model-scaling, for the reason recorded in
   Round 12: on model-scaling no class-(a)/(b) `d₂` suppresses at usable accuracy. The existing suite
   restricted cross-arm comparison to shapes and slopes; this test compares rise magnitudes, which is
   stronger and could reflect the attack rather than the defense. **Both the raw rise and the rise
   normalized by the arm's headroom `1 − ASR(κ=0)` are reported**, and the primary verdict is stated
   with this caveat attached wherever it appears.
4. **Headroom is smaller on this arm.** Identity ASR is 0.443, so the maximum achievable rise is
   0.557 against 0.938 (krum) and 0.983 (reputation). The `+0.178` confirmation threshold is well
   inside that range, but the test is conservative for the admission reading on this ground.
5. **`fltrust` was considered and dropped before the freeze.** It suppresses both attacks standalone
   (0.048 scaling, 0.046 pixel) and would otherwise be the natural class-(a) control that `cos_krum`
   failed to be. It is dropped because `FederatedServer._fltrust` weights clients by cosine to the
   server reference **and renormalizes every client update to the server gradient's norm**, so its
   aggregate is invariant to positive per-client rescaling in both the weights and the magnitudes: the
   arm would be flat by construction and could not fail. `multi_krum` (0.575 scaling at accuracy
   0.139, 0.783 pixel) and `trimmed_mean` (0.846, 0.616) were dropped on rule (1) — neither suppresses
   either attack, so neither has suppression to lose. Recorded rather than quietly omitted.
6. **Mode S has nothing to impose in 20.0% of rung-rounds** (24/120 measured in Round 12): a round
   with no adversary, or with fewer than two benign participants, admits no benign dispersion. The
   effective dose is diluted accordingly and the fraction is reported with the results.
7. **This arm does not address recall or the discovery limitation.** `fg→rfa` remains a false negative
   of our own rule.

## Non-negotiables

1. No predicted ordering, threshold, margin, or refutation criterion above is revised after seeing an
   outcome. An arm meeting neither criterion is reported indeterminate, not scored in our favour.
2. No switching of attack, κ grid, seed count, or accuracy gate.
3. Every reported number is recomputed from `results/dose_replication/summary.json`,
   `results/targeted_dose/summary.json` and `results/admission_measurement.json`, per seed, never
   transcribed.
4. `doseS_kappa<κ>` is never presented as a defense. It reads adversary identity and is an instrument
   for causal identification only.
5. **Nothing in Round 11 or Round 12 is rescored.** The refuted pooled ladder
   (`ρ_s = +0.351`, `p = 0.091`), the unconfirmed pooled two-channel model (`R² = 0.24`,
   `p = 0.055`), the indeterminate `cos_krum` arm and every frozen PASS/FAIL label stand exactly as
   reported, whatever this arm shows.
6. This file is committed before `results/dose_replication/` is written; if that ordering cannot be
   demonstrated from `git log`, the arm is reported as non-prospective.
