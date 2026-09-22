# Pre-registration: the score-only magnitude control at its own primary contrast, n = 20

**Status: frozen before any write to `results/score_only_kappa2_topup/`.** The seed list, the contrast
rung, the verdict rules and the clauses limiting what this top-up may conclude are reproduced verbatim in
`experiments/run_score_only_kappa2_topup.py`, which refuses to start until this file is committed and its
hash is recorded in `PREREG_COMMIT`.

## Why this arm and not another

The score-only control is the paper's only instrument that closes the **magnitude** channel while leaving
the **statistic** channel open, and it is the single cell that carries the claim *substantial statistic and
decision disturbance does not move suppression even with the aggregated update held fixed*. It is reported
in the body at `main.tex:911` and as row 6 of `tab:channels`, in both places at **n = 5**, while the
paper's headline reversal is at n = 20. That asymmetry is the object here: the load-bearing control is
reported at the smallest seed count in the paper.

`results/emit_only_topup/` already took the score-only and emit-only cells to n = 20, but **only at the
factorial's contrast rung κ = 1.0**, which was itself a floor substitution (the *emit-only* cell's mean
clean accuracy at κ = 2 is 0.293, below `ACC_FLOOR = 0.35`). The score-only arm's **own** frozen primary
contrast is κ = 0 → κ = 2, its accuracy at κ = 2 is **0.6299**, and that contrast exists only at n = 5.
This document extends that one contrast and nothing else.

## It starts with a departure from a frozen clause

`experiments/pre_registration_score_only.md`, frozen at `35788d9`, fixes the primary rule at

> On **Δ = mean ASR(κ=2) − mean ASR(κ=0)**, n = 5 seeds {42, 43, 44, 45, 46}, `krum` /
> `committed_scaling` / CIFAR-10 / `cifar_cnn`

and states as a declared limitation

> **One cell.** One dataset, one architecture, one attack, one defense, n = 5.

**This document departs from the `n = 5` in both, and the departure is disclosed as a departure rather
than smuggled.** The seed count was not a power commitment there — it was a comparability commitment, so
the control and the uncontrolled arm it is compared against share seeds. That rationale is honoured in
full: the uncontrolled Mode-S arm is already at n = 20 on the same cell and the same seeds
(`results/dose_seed_topup/`, seeds 47–61 added to 42–46), so extending the control to 47–61 **restores**
the seed match that n = 5 preserved when the comparison arm was also n = 5. Nevertheless:

- the published **n = 5** verdict and its Δ = −0.023 are reported unchanged wherever they now appear;
- the n = 20 result is labelled a **disclosed post-hoc extension of a frozen suite**, never prospective;
- `35788d9`'s `n = 5` text is quoted at the site where the n = 20 number is reported.

The reason for departing is external and specific: a review names this control's estimand as the one thing
the paper has *not* established, and reports the supporting results' seed counts as a weakness.

## What is published, recomputed here per seed rather than transcribed

`krum` / `committed_scaling` / CIFAR-10 / `cifar_cnn`, seeds 42–46, from `results/score_only/summary.json`:

| rung | per-seed ASR (42…46) | mean | sd | mean clean acc. |
|---|---|---|---|---|
| κ = 0 | 0.1149 / 0.0209 / 0.0469 / 0.0567 / 0.0694 | **0.06176** | 0.03464 | 0.5648 |
| κ = 2 | 0.0004 / 0.0632 / 0.0871 / 0.0393 / 0.0026 | **0.03853** | 0.03780 | 0.6299 |

Paired differences `d_s = ASR_s(κ=2) − ASR_s(κ=0)`: −0.1144 / +0.0423 / +0.0402 / −0.0173 / −0.0669;
mean **−0.02322**, sd **0.06816**, 95% half-width at n = 5 (`t₄ = 2.776`) **0.0846**, interval
**[−0.1078, +0.0614]**. Published verdict: **THE NEGATIVE SURVIVES WITH THE MAGNITUDE CHANNEL CLOSED**,
since |Δ| = 0.023 < `EQUIV_MARGIN = 0.15`.

## What n = 20 can decide, disclosed before the run

Unlike the emit-only top-up — whose projected half-widths (0.061, 0.057) exceeded its own 0.05 margin, so
it was frozen as unable to certify — **this contrast is powered for its margin and we say so in advance.**
Holding the observed paired sd of 0.06816, the projected 95% half-width at n = 20 (`t₁₉ = 2.093`) is
**0.0319**, and the projected interval holding the point estimate is **[−0.0551, +0.0087]**: inside the
±0.15 margin with room to spare, and containing zero.

Two consequences, both frozen here so neither can be chosen afterwards:

1. **A pass at n = 20 is a real certification against the frozen margin**, not a point-estimate pass
   dressed up as one, and it is reported with its interval at every site.
2. **The arm is therefore genuinely falsifiable.** For the refuting branch to fire, mean ASR(κ=2) must
   exceed 0.2118 against an identity rung of 0.0618 — 3.4× the baseline, against a ceiling of 1.0. It is
   reachable. Fifteen new seeds could move a mean of −0.023 past +0.15 only if the added seeds behave
   nothing like the published five, which is exactly the possibility n = 5 cannot exclude.

The indeterminate branch (Δ < −0.15) would need mean ASR(κ=2) < −0.088, which is unreachable on this
cell; that branch cannot fire, and equivalence here is to the cell's own identity rung and not to a low
absolute ASR. Both facts are carried forward from `35788d9` unchanged.

## Design

- **Seeds 47–61**, 15 new seeds, n = 20 total. Contiguous, fixed here, no interim look, no extension, no
  filtering, and no seed dropped from the n = 20 analysis for any reason. Matches the seed range every
  other n = 20 arm on this cell uses, so the control and the uncontrolled arm remain seed-matched.
- **One rung: κ = 2.0** (ρ = 54.60), the frozen primary contrast of `35788d9`. **κ = 0.5 and κ = 1.0 are
  not run for the new seeds.** The score-only ladder across κ and the Jonckheere–Terpstra trend test
  **stay at n = 5, are not restated at n = 20, and keep their `pre_registered` status unchanged.** No
  display may print this ladder without a per-rung n: after this top-up the honest shape is κ = 0 and
  κ = 2 at n = 20 with the interior at n = 5, and that shape is not to be smoothed.
- **The κ = 0 baseline is imported, not re-run, and the import's weak point is disclosed here rather than
  discovered later.** At κ = 0 `apply_d1_transform` returns the update list unwrapped, `generic_compose`'s
  `score_only` branch has nothing to split, and `run_one`'s participant RNG stream does not depend on
  `d₁`'s name, so score-only Krum at κ = 0 **is** Krum alone. `run_score_only_control.py:21` states this and
  `:164`–`:168` asserts it. For seeds 47–61 the baseline is `results/dose_seed_topup/`'s κ = 0 cell, whose
  own `identity_rung_provenance` records it as computed there and *"Verified bit-identical to krum
  standalone by --harness-check."* **The assertion is re-run here at one new seed (47), not at fifteen**,
  because asserting at all fifteen costs the fifteen runs the import exists to save. If that single
  assertion fails at `1e-9`, **the import is void, the arm runs all 30 runs, and the failure is reported** —
  it is not explained away and the tolerance is not loosened.
- **Runs: 15**, one per new seed at κ = 2 with `score_only=True`, ≈12 min each on the paper's standard
  configuration, resumable, summary written after every run.
- **Nothing is recomputed that already exists.** `results/score_only/`, `results/emit_only/`,
  `results/targeted_dose/`, `results/dose_seed_topup/` and `results/emit_only_topup/` are read and not
  written. This suite writes `results/score_only_kappa2_topup/` alone.

## Decision rules

Margin, accuracy floor, rung ladder and verdict labels are **carried forward unchanged** from `35788d9`.
Nothing here is a new threshold and no direction is reversed.

| outcome at n = 20 | verdict |
|---|---|
| \|Δ\| < 0.15 | **THE NEGATIVE SURVIVES WITH THE MAGNITUDE CHANNEL CLOSED**, now at n = 20. Substantial statistic and decision disturbance does not produce a corresponding change in suppression even when the aggregated update's magnitude and direction are held exactly fixed. The central claim is instrumented against two independent confounds — attenuation (Mode S) and magnitude (here) — at the same seed count as the headline reversal. |
| Δ > +0.15 | **THE PUBLISHED FLAT RESULT WAS PARTLY AN ARTIFACT OF THE MAGNITUDE CHANNEL**, and n = 5 concealed it. With magnitude closed, statistic disturbance *does* move suppression. The paper's central claim narrows to "with benign magnitudes free to move", stated in the body beside the result and in the abstract, not in a limitation. `tab:claims`' status for this row changes accordingly. |
| Δ < −0.15 | A fall with attenuation already closed and magnitude also closed. **INDETERMINATE**, reported as such and **not** scored in our favour. Cannot fire on this cell (see headroom above); recorded so the table is complete. |

- **The interval is reported with the point estimate every time**, as the paired two-sided 95% Student-t
  interval on `d_s`, `t₁₉ = 2.093`. The margin rule is on the mean, as frozen; the interval is reported
  beside it and its relation to both ±0.15 and zero is stated, because an interval can sit inside the
  margin and still exclude zero.
- **Accuracy gate:** `ACC_FLOOR = 0.35` on the rung's **mean** clean accuracy, unchanged, applied to
  κ = 2 at n = 20. If it fails, the contrast is uninterpretable and **no verdict stands**; it is reported
  as such and **not** substituted onto another rung. `35788d9` spent no substitution on this arm and this
  document spends none — rung shopping is what the emit-only cell's floor contingency already cost once.
- **The reversal clause.** If the n = 20 mean falls outside ±0.15 where the n = 5 mean was inside it, we
  report that the published verdict **did not survive our own top-up**, at the site where that verdict
  appears, with the n = 5 numbers retained beside it. If the n = 20 mean moves further inside the margin,
  that is reported too. Committing to publish both directions is what licenses adding seeds.

## Caveats recorded in advance

1. **A disclosed post-hoc extension**, not a prospective test. The rule it scores is the frozen one and the
   published point estimate was already known to sit well inside its margin when these seeds were chosen.
   No amount of interval reporting changes that, and it is stated at the site of the result.
2. **Closing the magnitude channel does not close the trajectory channel.** *Which* client is selected
   still changes across rungs, so the model trajectory still diverges. `35788d9`'s clause stands: the claim
   this arm licenses is about magnitude, not about trajectory. n = 20 does not widen it.
3. **One cell.** One dataset, one architecture, one attack, one defense, twenty data partitions. This
   closes one channel on the flagship Krum cell at n = 20 and is not a breadth claim.
4. **The control is defined for selectors only**, and `generic_compose` raises for the others. So the
   `coord_median` arms — including the cell carrying the paper's headline reversal — are **not** controlled
   for magnitude by this arm. `35788d9` already required that we say so in the paper, and it stays said;
   `pre_registration_score_only_coordmedian.md` is the separate arm that addresses it, and no result here
   transfers to it.
5. **Score-only Krum is not a defense.** It aggregates an update the scoring stage did not see, on top of
   Mode S already reading adversary identity to pin `c_adv = 1`. This is a laboratory instrument and
   nothing here is proposed for deployment.
6. **The ladders and the trend test remain at n = 5.** Every table reports n per row, and no row pools a
   20-seed contrast with a 5-seed ladder.

## Non-negotiables

1. No margin, accuracy floor, contrast rung, interval definition or verdict label above is revised after
   an outcome is seen.
2. The seed list is 47–61 and the rung is κ = 2.0. Neither is extended, truncated by inspection, or
   filtered.
3. Every reported number is recomputed per seed from the artifacts named above, never transcribed. The
   five published seeds are asserted to reproduce at `< 1e-9` before any pooled number is printed.
4. `results/score_only/`, `results/emit_only/`, `results/targeted_dose/`, `results/dose_seed_topup/` and
   `results/emit_only_topup/` are **not rewritten**, and `SEEDS5` is not edited. This suite writes its own
   directory and is merged only at analysis time.
5. The published n = 5 verdict is reported alongside the n = 20 verdict, whatever the latter is, together
   with the quoted `35788d9` n = 5 text this document departs from.
6. This file is committed before `results/score_only_kappa2_topup/` is written. The suite is reported as a
   post-hoc extension in either case; the commit ordering establishes only that the rules above were not
   written after the numbers.
7. The phrase *the negative reproduces oracle-free* is not licensed by this arm in any form, and neither
   is its inverse. This arm closes the magnitude channel; it does not remove the oracle. Mode S still
   reads adversary identity to pin `c_adv = 1`, at n = 20 exactly as at n = 5.
