# Pre-registration: matched metric-swap ablation

**Status: frozen before any write to `results/metric_swap/`.** Predicted labels below are
reproduced verbatim in the `PAIRS` list of `experiments/run_metric_swap_suite.py`, which refuses
to start until this file is committed and its hash is recorded in `PREREG_COMMIT`.

## Purpose

Break **mechanism--effectiveness confounding**: in a fixed defense menu the invariant-signal `d2`
tends also to be the strong-standalone `d2`, so C1 and C2 co-vary and neither can be credited. The
testability proposition shows this is forced in scope (C0 ∧ C1 ⟹ `ASR(d2, a*) < 0.5`), so we do not
try to escape it by choosing a different defense. Instead we hold the suppression *principle* fixed
and vary only the invariance class of the statistic `d2` reads, using two minimal-change mirrors of
defenses already in the menu.

`cos_reputation` and `cos_krum` are **mechanism-isolating ablations, not proposed defenses.**

## Phase 0 inputs, measured BEFORE this freeze

C0 and C1 take standalone effectiveness as an *input*, so it is measured first.
Source: `results/metric_swap_baselines/summary.json` (the two mirrors, n=3, seeds 42--44) and
`results/wave2_held_out/summary.json` / `results/all_compositions/summary.json` /
`results/prospective_pilot/summary.json` (the rest, same protocol).

| defense | scaling ASR (acc) | pixel ASR (acc) | suppresses both? |
|---|---|---|---|
| norm_clip (`d1`) | 0.935 (0.74) | 0.806 (0.79) | no — neither |
| rfa (`d1`) | 0.885 (0.78) | 0.845 (0.78) | no — neither |
| reputation | **0.017 (0.78)** | 0.842 (0.77) | no — pixel |
| cos_reputation | **0.982 (0.23)** | 0.754 (0.80) | no — neither |
| krum | **0.061 (0.58)** | 0.583 (0.49) | no — pixel |
| cos_krum | 0.078 (**0.15**) | **0.300 (0.60)** | no — scaling (see gate) |

Per-seed values for the two mirrors (n=3):

- `cos_reputation` scaling ASR 1.000 / 0.982 / 0.965, acc 0.100 / 0.336 / 0.252
- `cos_reputation` pixel   ASR 0.838 / 0.660 / 0.766, acc 0.790 / 0.803 / 0.793
- `cos_krum` scaling       ASR 0.000 / 0.214 / 0.021, acc 0.122 / 0.212 / 0.103
- `cos_krum` pixel         ASR 0.003 / 0.269 / 0.630, acc 0.528 / 0.620 / 0.647

### The pre-registered contingency fired, on both arms, for two different reasons

The contingency written before Phase 0 ran: *both mirrors are magnitude-blind and the Euclidean
originals may be catching model-scaling via magnitude; if a mirror fails to suppress an attack
standalone (mean ASR ≥ 0.5) then C1 fails, the prediction is HIGH, and that arm yields no positive
test — recorded as such, with no switching of attack, threshold or seed count.* It fired:

1. **`cos_reputation`: outright.** 0.017 → 0.982 on model-scaling. Removing magnitude does not
   merely weaken the defense, it destroys it.
2. **`cos_krum`: via the accuracy gate.** 0.078 passes C1's numeric threshold but at **0.146 clean
   accuracy**. This is the hollow-suppression artifact already documented for FoolsGold
   (0.200 @ 0.10) and fg→krum (0.078 @ 0.099). The mechanism is visible in the baselines:
   undefended FedAvg under model-scaling is itself at 0.100 accuracy, so krum's 0.577 is
   *restoring* utility while `cos_krum`'s 0.146 is failing to. **Magnitude information is what
   lets this defense family handle a magnitude attack.**

On the pixel backdoor — magnitude-neutral, so the mirror survives — `cos_krum` reaches 0.300 at
0.598 accuracy, genuinely suppressing, while its control krum is at 0.583 and **fails C1**. So the
arms are unmatched on C1 in the opposite direction.

**Consequence, recorded before the run: this design does not disentangle C1 from C2.** The confound
re-materialized inside an experiment built specifically to break it, in two independent ways. We
report that as the outcome. We do **not** switch to a pixel-only committed-attack protocol, which
would make `cos_krum`'s C1 pass and manufacture two predicted-LOW pairs — the contingency forbids
switching the attack set, and doing so after seeing these inputs would be attack-shopping.

## Protocol

Unchanged from the rest of the paper: CIFAR-10, `cifar_cnn`, N=10, K=5, f=0.2, Dirichlet α=0.5,
50 rounds, seeds 42/43/44, both committed attacks (model-scaling, pixel backdoor). Scored on
**max-committed ASR** at threshold **0.5**. `d2` is reached via `server.aggregate(..., method=d2)`;
`d1` ∈ {norm_clip (τ=5), rfa} is applied as a per-client transform.

The two reputation controls were already measured under exactly this protocol and these seeds, so
they are reused from `results/all_compositions/summary.json` rather than re-run (marked `reuse`).
New runs: 6 pairs × 2 attacks × 3 seeds = 36.

**Sanity gate.** FedAvg reaches 0.667 on scaling and 0.768 on pixel, both ≥ 0.5, so predicted-HIGH
cells are informative rather than vacuous as in the DBA battery. One caveat stated up front: the
scaling sanity baseline sits at 0.100 clean accuracy, so the undefended scaling regime is itself a
collapsed one; per-seed FedAvg scaling ASR is 0.0 / 1.0 / 1.0.

**Accuracy gate.** Any cell below 0.35 mean clean accuracy is reported as uninterpretable, not as
suppression, whichever direction its ASR points.

## Condition evaluation

C0 holds for all 8 pairs: neither norm_clip nor rfa suppresses either committed attack, so `d2` is
load-bearing and no pair can inherit suppression from `d1`.

C2 is assigned by the invariance proposition and **verified in code before being claimed**
(`results/cos_invariance_check.json`): over 18 live rounds `cos_reputation`'s weight ordering never
changes (0/18) and `cos_krum` selects the same client 18/18, while reputation's ordering changes
12/18 and krum's selection 6/18.

C3 is not the binding condition for any pair here: none of the four `d2` is a coordinate-rank
aggregator, so no margin-compression label applies. Recorded as n/a, following the treatment of
·→krum pairs in `pre_registration_prospective.md`.

C1 is the binding condition for **all eight pairs**, which is the finding.

## Frozen predictions

| # | pair | C0 | C1 | C2 | first failure | **predicted** | confidence |
|---|---|---|---|---|---|---|---|
| 1 | norm_clip → reputation | hold | **fail** (pixel 0.842) | fail | C1 | **HIGH** | high |
| 2 | norm_clip → cos_reputation | hold | **fail** (both: 0.982 / 0.754) | hold | C1 | **HIGH** | high |
| 3 | rfa → reputation | hold | **fail** (pixel 0.842) | fail | C1 | **HIGH** | high |
| 4 | rfa → cos_reputation | hold | **fail** (both: 0.982 / 0.754) | hold | C1 | **HIGH** | high |
| 5 | norm_clip → krum | hold | **fail** (pixel 0.583) | fail | C1 | **HIGH** | high |
| 6 | norm_clip → cos_krum | hold | **fail** (scaling, accuracy gate) | hold | C1 | **HIGH** | *low — see split* |
| 7 | rfa → krum | hold | **fail** (pixel 0.583) | fail | C1 | **HIGH** | high |
| 8 | rfa → cos_krum | hold | **fail** (scaling, accuracy gate) | hold | C1 | **HIGH** | *low — see split* |

**Zero predicted-LOW pairs.** Like Tier 2 and the non-FLTrust arm of the prospective suite, this
suite therefore bears on **specificity only** and contains no positive test. That is a direct
consequence of the Phase 0 inputs, not a design choice made afterwards.

### The one genuinely two-sided prediction: pairs 6 and 8

`cos_krum`'s C1 verdict on model-scaling depends on whether the accuracy gate is applied, so we
freeze **both** readings rather than pick one, and commit to reporting both:

- **Primary (accuracy-gated C1, as used everywhere else in the paper):** 0.078 at 0.146 accuracy is
  hollow, so C1 fails → **HIGH**.
- **Secondary (numeric C1 only):** 0.078 < 0.5, so C1 holds; with C2 holding and C3 n/a, pairs 6
  and 8 would be **certified → LOW**.

Resolution rule, fixed now: if pairs 6/8 come out **LOW with mean accuracy ≥ 0.35**, the accuracy
gate was wrong to exclude them and the numeric reading wins — these become genuine certified
positives and the primary labels are misses. If they come out **LOW with accuracy < 0.35**, the gate
was right and the suppression is hollow. If they come out **HIGH**, the secondary reading is refuted
and the gate is vindicated. No third option will be introduced after the fact.

### Secondary, clearly-labelled mechanistic observation

Per-attack (not max-committed, so not a criterion prediction), the pixel cells give a contrast at
fixed `d1` between krum (C2 fails) and `cos_krum` (C2 holds). We will report it, and we will label
it **confounded**: krum's pixel baseline is 0.583 and `cos_krum`'s is 0.300, so C1 differs between
the arms and any gap cannot be credited to C2 alone. It is suggestive evidence of the same kind as
the three FLTrust contrasts, not an improvement on them.

## What each outcome would mean

- **All 8 HIGH:** predictions confirmed; specificity evidence only; no positive test. The
  headline finding is the confounding result itself.
- **Any pair LOW:** a false negative of our own criterion, reported as a miss. For pairs 6/8 the
  resolution rule above applies.
- **Pairs 2/4 LOW:** would mean `cos_reputation` composes to suppression despite failing both
  attacks standalone — synergy of the fg→rfa kind, and evidence the conditions are too strict.
- **Pairs 6/8 LOW at usable accuracy:** the strongest available positive result from this suite,
  and it would overturn our own accuracy-gate reading. Reported as such if it happens.

## Non-negotiables

1. No label in the table above is revised after seeing an outcome. A miss is reported as a miss.
2. No switching of committed attack, threshold, seed count, or accuracy gate.
3. Every reported number is recomputed from `results/metric_swap/summary.json`, per seed, never
   transcribed.
4. `cos_reputation` and `cos_krum` are never presented as proposed defenses.
5. This file is committed before `results/metric_swap/` is written; if that ordering cannot be
   demonstrated from `git log`, the suite is reported as non-prospective.
