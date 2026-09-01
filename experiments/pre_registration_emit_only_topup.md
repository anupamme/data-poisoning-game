# Pre-registration: seed top-up of the score-only / emit-only factorial (Round 34, item C)

**Status: frozen before any write to `results/emit_only_topup/` or `results/score_only_topup/`.** The
seed list, the contrast rung, the verdict rules and the two clauses that limit what this top-up is
allowed to conclude are reproduced verbatim in `experiments/run_emit_only_topup.py`, which refuses to
start until this file is committed and its hash is recorded in `PREREG_COMMIT`.

## It has to start with a departure from a frozen non-negotiable

`experiments/pre_registration_emit_only.md`, frozen at `b995f1b`, says:

> **No seed addition.** n=5, seeds 42-46, matching the arm it is compared against.

**This document departs from that clause, and the departure is disclosed as a departure rather than
smuggled.** Two things are true about it:

1. **The clause's stated rationale is comparability**, not a sample-size commitment: the seeds must
   match "the arm it is compared against", because the additivity residual is computed *across* the
   three factorial cells and a residual assembled from cells with different `n` is not a residual.
   Adding seeds to the emit-only cell alone would violate that rationale. **This top-up extends all
   three cells at the contrast rung to the same `n = 20` on the same seeds**, so the rationale is
   honoured in full.
2. **The literal text is still departed from**, and no reading of it makes the departure disappear.
   Therefore: the published `n = 5` verdict is reported unchanged wherever it currently appears, the
   `n = 20` result is labelled a **disclosed post-hoc extension of a frozen suite**, and it is never
   described as prospective. `b995f1b`'s clause is quoted in the paper at the site where the `n = 20`
   number is reported, so a reader meets the departure and the result together.

The reason for departing is external and specific: the fourth review asks for seeds on this exact
2×2 ("add seeds to the knife-edge 2×2 or move it to the supplement"), and the material is already in
the supplement, so the only remaining response is seeds.

## What is published, and why it is knife-edge

`krum` / `committed_scaling` / CIFAR-10 / `cifar_cnn`, seeds 42–46, `n = 5`. The frozen primary
contrast is the `κ = 0 → κ = 2` endpoint, with a floor contingency also frozen at `b995f1b`; the
emit-only cell's mean clean accuracy at `κ = 2` is **0.293**, below `ACC_FLOOR = 0.35`, so the
contingency fired and **the primary contrast is `κ = 0 → κ = 1`** (`ρ = 7.39`, emit-only mean accuracy
at that rung comfortably above the floor). All three cells share the same imported `κ = 0` identity
rung; verified equal seed-for-seed.

Seed-matched mean change from `κ = 0` to `κ = 1`, with the two-sided 95% Student-t interval at
`n = 5` (`t₄ = 2.776`):

| quantity | mean | sd | 95% interval | per-seed (42…46) |
|---|---|---|---|---|
| `ΔASR_full` | **−0.038867** | 0.0455 | [−0.095, +0.018] | −0.1003 / +0.0212 / −0.0198 / −0.0340 / −0.0614 |
| `ΔASR_SO` (score-only) | **−0.029978** | 0.0638 | [−0.109, +0.049] | −0.1139 / +0.0491 / −0.0329 / +0.0122 / −0.0644 |
| `ΔASR_EO` (emit-only) | **+0.045578** | 0.1297 | [−0.115, +0.207] | **+0.2721** / −0.0067 / +0.0066 / +0.0134 / −0.0576 |
| additivity residual | **−0.054467** | 0.1222 | [−0.206, +0.097] | −0.2586 / −0.0212 / +0.0066 / −0.0597 / +0.0606 |

The frozen rules are `|ΔASR_EO| < 0.05` (the magnitude channel is inert) and
`|ΔASR_full − (ΔASR_SO + ΔASR_EO)| < 0.05` (the channels are separable). The published verdict is
**CHANNELS INTERACT**: the residual is `−0.0545`, missing the `0.05` margin by `0.0045`. That is the
knife edge, and **seed 42 is the whole of it** — it contributes `+0.2721` of the `+0.0456` mean
`ΔASR_EO` (119% of the mean) and `−0.2586` of the `−0.0545` residual. A post-hoc exclusion of seed 42
gives a residual of `−0.003444`, i.e. SEPARABLE; that exclusion is reported in the paper as a
statement about power and is not a verdict, and this document does not change that.

## What n = 20 can decide, and what it cannot — both frozen here

This is the part that keeps the top-up from being optional stopping in the direction that flatters us.
SEPARABLE is the cleaner story, so a top-up that could flip CHANNELS INTERACT to SEPARABLE is a top-up
with an interest in the outcome. Two limits are therefore frozen before any run.

**It cannot certify inertness or separability by interval containment, and we know that in advance.**
Holding the observed sds, the projected 95% half-widths at `n = 20` (`t₁₉ = 2.093`) are **0.061** for
`ΔASR_EO` and **0.057** for the residual, both **wider than the `0.05` margin itself**. So even a
point estimate of exactly zero could not put a 95% interval inside `(−0.05, +0.05)` at this variance
and this `n`. Consequently:

> **A pass of the frozen point-estimate rule at `n = 20` is reported as "consistent with separability,
> not established", never as SEPARABLE-full-stop.** The interval is reported next to the point
> estimate every time, so the width is visible at the site of the claim.

**What it genuinely can decide is the reviewer's actual complaint.** "One seed carries the whole
result" is a question about the distribution, not about a margin, and 15 more seeds answer it:

- the realized sd of `ΔASR_EO` at `n = 20`, against `0.1297` at `n = 5`;
- **the count of seeds with `ΔASR_EO > +0.15`**, i.e. how many seed-42-like cases occur in 20 draws;
- the same count for the residual below `−0.15`;
- the full per-seed distribution, printed, not summarized.

If the sd collapses and 42 is alone in 20 seeds, seed 42 was an outlier and the published verdict was
driven by one draw. If the mode recurs, the emit-only channel is genuinely bimodal across data
partitions and the interaction is real. **Both outcomes are reported, and the second is the one that
vindicates the published verdict**, so the descriptive pre-commitment is not directional.

## Design

- **Seeds 47–61**, 15 new seeds, `n = 20` total. Contiguous, fixed here, no interim look, no
  extension. Matches item A's range so the three cells and the flagship share one seed set.
- **Contrast rung only: `κ = 1.0`.** The frozen primary contrast after the floor contingency is
  `κ = 0 → κ = 1`, and the additivity residual is a function of that contrast alone, so extending
  `κ = 0.5` and `κ = 2` buys nothing for the quantity under discussion. **The score-only and
  emit-only *ladders*, and the Jonckheere–Terpstra trend test the score-only arm reports, stay at
  `n = 5` and are not restated at `n = 20`.** Disclosed, not silently mixed: every reported number
  carries its own `n`.
- **The `κ = 0` baseline is imported, not re-run.** At `κ = 0` `apply_d1_transform` returns the update
  list unwrapped, `generic_compose`'s `score_only` / `emit_only` split has nothing to split, and
  `run_one`'s participant RNG stream does not depend on `d₁`'s name — so all three cells' identity
  rung is one computation, which is why they are equal seed-for-seed today. For seeds 47–61 it comes
  from item A (`results/dose_seed_topup/`). `--harness-check` asserts bit-equality of the `κ = 0`
  score-only and emit-only runs against the plain run at one new seed; **if that fails, the top-up
  does not run.**
- **The `ΔASR_full` cell at seeds 47–61 is item A's output**, not recomputed here. `run_one(seed, "S",
  "krum", "committed_scaling", 1.0)` with neither flag is the same call in both suites.
- **Runs: `κ = 1` score-only × 15 + `κ = 1` emit-only × 15 = 30 new runs**, ≈15–20 min each on the
  paper's standard configuration, resumable, written after every run.
- **Dependency, stated because it can bite.** The residual at `n = 20` needs all three cells at the
  same seeds. If item A is incomplete when the analysis runs, **the residual is reported at the `n`
  actually shared by all three cells**, and that `n` is printed. It is never assembled from cells with
  different seed sets.

## Decision rules

Margins, floor and contrast rung are **carried forward unchanged** from `b995f1b`. Nothing here is a
new threshold, and the direction of every test is the direction already frozen.

- **Magnitude channel inert at `n = 20`:** `|ΔASR_EO| < 0.05`. Reported with its 95% interval and
  with the "not established" qualifier above whenever the interval is not contained in the margin.
- **Channels separable at `n = 20`:** `|ΔASR_full − (ΔASR_SO + ΔASR_EO)| < 0.05`, same qualifier.
- **Channels interact at `n = 20`:** residual `≥ 0.05` in absolute value while `|ΔASR_EO| < 0.05`.
  This reproduces the published verdict; if it is what the data says, the top-up is reported as
  **confirming** CHANNELS INTERACT at four times the sample size, which is the strongest thing this
  run can deliver for the paper's credibility precisely because it is not the cleaner story.
- **The reversal clause.** If the `n = 20` residual falls inside `(−0.05, +0.05)`, we report that the
  published CHANNELS INTERACT verdict **did not survive our own top-up** and that it was driven by
  seed 42, in the section where the verdict appears, with the `n = 5` numbers retained beside it. If
  instead the `n = 20` residual moves *further* outside the margin, that is reported too. Committing
  to publish both directions is what licenses adding seeds.
- **Accuracy gate.** `ACC_FLOOR = 0.35` on a rung's **mean** clean accuracy, unchanged, applied to
  each cell separately at `n = 20`. If the emit-only `κ = 1` rung falls below the floor at `n = 20`
  — it is above it at `n = 5` — the contrast is uninterpretable and is reported as such rather than
  substituted onto another rung: the floor contingency at `b995f1b` already spent its one
  substitution moving the primary from `κ = 2` to `κ = 1`, and a second substitution would be rung
  shopping.

## Caveats recorded in advance

1. **This is a disclosed post-hoc extension**, not a prospective test. The rules it scores are the
   frozen ones, and the quantity was already known to sit `0.0045` from its margin when these seeds
   were chosen. No amount of interval reporting changes that, and it is stated at the site of the
   result.
2. **Projected half-widths exceed the margin at `n = 20`** (0.061 and 0.057 against 0.05). The top-up
   was authorized knowing this. Its deliverable is the distribution of `ΔASR_EO` and a legible
   interval, not certification.
3. **`n = 20` is 20 data partitions of one dataset**, one aggregator, one attack, one synthetic
   instrument. Nothing here widens the factorial's scope.
4. **The ladders and the trend test remain at `n = 5`.** Every table reports `n` per row, and no row
   pools a 20-seed contrast with a 5-seed ladder.
5. **Seed 42 is not excluded from anything.** The published post-hoc exclusion stays labelled as a
   statement about power, and the `n = 20` analysis reports the full 20 seeds with the outlier count
   as a separate descriptive figure.

## Non-negotiables

1. No margin, accuracy floor, contrast rung or verdict rule above is revised after seeing an outcome.
2. The seed list is 47–61 and the contrast rung is `κ = 1.0`. Neither is extended, truncated by
   inspection, or filtered, and no seed is dropped from the `n = 20` analysis for any reason.
3. Every reported number is recomputed per seed from `results/score_only/`, `results/emit_only/`,
   `results/targeted_dose/`, `results/dose_seed_topup/`, `results/score_only_topup/` and
   `results/emit_only_topup/`, never transcribed. The five published seeds are asserted to reproduce
   bit-identically, at `< 1e-9`, before any pooled number is printed — the check
   `analyze_emit_only_factorial.py` already applies to its four quantities.
4. **`results/emit_only/`, `results/score_only/` and `results/targeted_dose/` are not rewritten**, and
   `SEEDS5` is not edited. The top-ups write their own directories and are merged only at analysis
   time.
5. **The published `n = 5` verdict is reported alongside the `n = 20` verdict, whatever the latter
   is**, together with the quoted `b995f1b` no-seed-addition clause this document departs from.
6. This file is committed before `results/emit_only_topup/` or `results/score_only_topup/` is written.
   The suite is reported as a post-hoc extension in either case; the commit ordering establishes only
   that the rules above were not written after the numbers.
