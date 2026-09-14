# Pre-registration: seed top-up of the adaptive FoolsGold->RFA arm (Round 69, ninth review)

**Status: frozen before any write to `results/criterion_aware_topup/`.** The seed list, the reused
denominator, the primary interval, the ratio-relabelling rule and the body-number clause below are
reproduced in `experiments/run_criterion_aware_seed_extension.py`, which refuses to start until this file is
committed and its hash is recorded in `PREREG_COMMIT`.

## The claim being extended, and the exact reason it needs extending

`main.tex:489` (body, section 4) and `main.tex:2222` (appendix) both report the whitebox composition-aware
adversary against FoolsGold->RFA as

> raises FG$\to$RFA to ASR $0.167 \pm 0.069$, $3.7\times$ its seed-matched $0.045$

and `:2222` adds that this is *"the ratio every other section of this paper quotes"*. That ratio is a
**5-seed** quantity: `0.1668 / 0.0452` on seeds 42--46, from
`results/criterion_aware_adversary/summary.json`.

The paper already knows those five seeds are unrepresentative and says so, at `main.tex:2189`, about this
very composition:

> the per-seed distribution is right-skewed (median $0.060$, mean $0.093$, sample skew $+2.07$, $8/30$ seeds
> at or above $0.10$, maximum $0.380$), and seeds 42--46 fall in its left tail, so the $n{=}5$ value we first
> led with understated the pair by a factor of $1.8$

and the same sentence already prints the pixel arm's own `n = 30` mean: **`0.064` at `n = 30`** against
`0.045` at `n = 5`. So the paper simultaneously (a) discloses that seeds 42--46 sit in the left tail of the
denominator's distribution and (b) leads a body number with a ratio computed on exactly those five seeds.
**That is the defect this arm exists to repair.** It is our defect, not the reviewer's catch: the ninth
review asks only for wider replication of the adaptive result.

## What is run, and what is reused rather than run

**Run here: the numerator only.** Condition `ca_eps1_decorr` (`eps = 1.0`, `decorrelate = True`) at seeds
**47--71**, 25 new runs, ~0.73 h/run at the frozen configuration, ~18 h. Written to
`results/criterion_aware_topup/` and nowhere else.

**Reused, not re-run: the denominator.** The base pixel leg is taken from
`results/fg_rfa_flagship/summary.json`, `base_composition/committed_pixel/per_seed`, which **already spans
seeds 42--71** (`n = 30`, mean `0.0644`, sd `0.0404`). Spending 25 runs to recompute a leg that exists at the
same 30 seeds would buy nothing.

That reuse is a claim about two different runners agreeing, so it is **proved by value before it is used**,
not assumed. `run_fg_rfa_flagship.py`'s `committed_pixel` and `run_criterion_aware_adversary.py`'s
`committed_pixel` are compared per seed on the five seeds where both exist (42--46):

| | worst absolute deviation over seeds 42--46 |
|---|---|
| ASR | **0.000000000** |
| clean accuracy | **0.000000000** |

All five rows are bit-identical across the two runners. The code path explains why, and the explanation is
checkable rather than atmospheric: in `run_criterion_aware_adversary.run_one` (`:180`) the only consumer of
the numpy stream is `np.random.choice` for participant sampling at `:203`, which executes **before** the
`label` branch at `:216`; `criterion_aware_updates` (`:132`--`:177`) draws no randomness at all; and the
per-round count of torch RNG draws does not depend on the label, because the participant list, the epoch
count and the batch size are label-independent. So a seed fixes the Dirichlet partition, the model
initialization and the participation sequence identically for both legs.

**`results/criterion_aware_adversary/` and `results/fg_rfa_flagship/` are read-only here.** Neither is
rewritten, and the top-up's own directory is merged with them only at analysis time, where the five
overlapping seeds are re-asserted bit-identical before any pooled number is printed.

## Both legs are recomputed at the 30 shared seeds, and the ratio is expected to fall

A top-up that moves only the minuend hides a mixed-`n` comparison inside the subtrahend. So the analysis
recomputes **both** legs at the 30 shared seeds, and the frozen arithmetic below says what to expect.

| quantity | at `n = 5` (published) | at `n = 30` (this arm) |
|---|---|---|
| adaptive `ca_eps1_decorr` mean ASR | `0.1668` | to be measured |
| base `committed_pixel` mean ASR | `0.0452` | **`0.0644`** (already on disk) |
| ratio | **`3.687x`**, printed as `3.7\times` | to be recomputed |

**Even if the adaptive mean is completely unchanged at `0.1668`, the ratio at `n = 30` is `2.588x`**, because
the denominator alone rises by 42%. If the adaptive mean also regresses off a favourable 5-seed block, it
falls further. The published `3.7\times` is therefore expected to move **down**, and this file is committed
before the 25 runs so that the movement cannot be read afterwards as a reluctant concession.

**Pre-registered consequence for the paper.** Whatever the outcome:

1. `main.tex:489` and `main.tex:2222` are rewritten to the `n = 30` ratio, and `:2222`'s self-description
   *"the ratio every other section of this paper quotes"* is re-verified by grep against every site that
   quotes it, because a self-counting claim has no emitter and no build can see it go stale.
2. The `n = 5` ratio is **reported alongside** the `n = 30` ratio, with the reason the two differ named as
   the denominator's left-tail seeds and not as adaptive-attack variance.
3. If the ratio falls, the response letter **leads** with the fall rather than burying it.
4. No number is transcribed. Every reported figure is recomputed per seed from the two artifacts.

## The primary quantity, fixed now

- **Primary: the paired difference.** `d_s = ASR_s(ca_eps1_decorr) - ASR_s(committed_pixel)` per seed `s`,
  over the 30 shared seeds; two-sided 95% Student-t interval on `mean(d)` with `n - 1` degrees of freedom.
  Pairing is exact by construction for the reason given above, and empirically witnessed by the
  `0.000000000` cross-runner agreement.
  At `n = 5` this is `mean +0.1215`, `sd 0.0734`, interval `[+0.0304, +0.2127]`.
- **Secondary: the unpaired difference of means**, pooled sd, `2n - 2` degrees of freedom. Reported because
  the published claim is literally written as a ratio of two means.
- **Descriptive only: the ratio.** A ratio of means has no interval we are willing to defend at these
  sample sizes, so it is reported as a point figure with both `n` values attached and is never given a
  confidence statement.

**The paired interval is nominated as primary before the runs**, and it is the quantity that decides, so the
choice cannot afterwards be read as having selected whichever form came out more favourable.

## Decision rules

- **Adaptive gain confirmed at `n = 30`:** the primary 95% interval excludes zero and lies above it.
- **Adaptive gain refuted at `n = 30`:** the primary interval contains zero. Then the body's
  *"Inherited robustness is not adaptive robustness"* paragraph at `:489` keeps its scope claim but loses its
  quantitative witness, and that withdrawal is reported in the response letter and in the appendix.
- **Neither the confirmation nor the refutation licenses a re-ranking claim.** The other five adaptive
  conditions stay at `n = 5`. No statement of the form "condition X is the strongest adaptive attack" is
  made at `n = 30`, because only one condition is measured there.
- **Accuracy floor.** `ACC_FLOOR = 0.35` on the arm's mean clean accuracy, the existing convention. The
  published legs clear it (base `n = 30` mean `0.4918`, minimum `0.4356`; adaptive `n = 5` mean `0.5620`,
  minimum `0.5470`). Per-seed accuracies are recorded for all 25 new runs and any individual seed below the
  floor is flagged even where the mean clears it. If the arm's mean falls below `0.35`, the arm is reported
  as **uninterpretable for ASR** rather than as a gain, per the paper's own rule at `main.tex:1709`.
- **No stopping rule and no interim look.** The seed list is 47--71, contiguous, fixed here. If the runs are
  interrupted, the analysis reports the `n` actually reached and the interval at that `n`; it does not resume
  until a threshold is crossed. Progress is counted from `per_seed` entries in the artifact, never from a
  `[i/N]` log position, which counts resumed-and-skipped runs and can move backwards across restarts.

## The harness check, run before any new seed

`run_criterion_aware_seed_extension.py --harness-check` re-runs **seed 42 under `ca_eps1_decorr`** through
the **imported** `run_one` and asserts agreement with the stored row in
`results/criterion_aware_adversary/summary.json` to `< 1e-9`. A published seed is used deliberately: at a new
seed there is nothing to compare against and the check would be vacuous. One run settles it for all 25 new
seeds, because the code path does not depend on the seed.

**If that assertion fails, the top-up does not run.** This is a value comparison against stored per-seed
rows, not an md5 of an artifact file, because this repository has already shipped one arm that was
bit-identical to another for the wrong reason -- a manipulation hook that was never called and failed
silently.

## Caveats recorded in advance

1. **This is one condition, one composition, one attack, one dataset.** The arm buys precision on the single
   adaptive datapoint the body quotes. It is not evidence about any other composition, and it does not turn
   the stress test into a security evaluation.
2. **The adversary adapts to the composition, not to the evaluation instrument.** Mode S reads adversary
   identity and does not exist in deployment, so no adaptive adversary can target it. That scope statement is
   unchanged by this arm.
3. **The base leg's `sd` is 3.1x the adaptive leg's at `n = 5`** (`0.0404` at `n = 30` against `0.0132` on
   seeds 42--46). The realized paired `sd` at `n = 30` may therefore be larger than `0.0734`, and the
   realized interval is what is reported. No projected interval appears in the paper: the `0.0274` half-width
   that `sd = 0.0734` would give at `n = 30` is stated here, in this file, and nowhere else.
4. **Seeds 47--71 are already used by the base leg** of `results/fg_rfa_flagship/`. That is the point -- the
   comparison is seed-matched -- but it is recorded because the overlap looks like a collision.
5. **A ratio computed from a right-skewed denominator is fragile in both directions.** The `n = 30`
   denominator has median `0.0517` against mean `0.0644`; we report the mean-based ratio because that is what
   the published claim is, and we report the denominator's skew beside it rather than switching to the
   statistic that flatters the arm.

## Non-negotiables

1. No interval definition, accuracy floor, primary quantity or refutation criterion above is revised after
   seeing an outcome.
2. The seed list is 47--71 and the condition is `ca_eps1_decorr`. Neither is extended, truncated by
   inspection, or filtered.
3. `results/criterion_aware_adversary/`, `results/fg_rfa_flagship/` and every other existing `results/`
   directory are **not written**. This arm writes `results/criterion_aware_topup/` only.
4. `experiments/run_criterion_aware_adversary.py` is imported, never edited and never copied: the extension
   calls its `run_one`, so the physics cannot drift between the frozen seeds and the new ones.
5. **The published `n = 5` figures are reported alongside the `n = 30` figures, whatever the latter are.** If
   they disagree, both appear, and the disagreement is the result.
6. This file is committed before `results/criterion_aware_topup/` is written; if that ordering cannot be
   demonstrated from `git log`, the arm is reported as non-prospective.
