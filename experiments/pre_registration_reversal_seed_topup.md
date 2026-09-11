# Pre-registration: seed top-up of the sign-reversal cell (Round 63)

**Status: frozen before any write to `results/reversal_seed_topup/`.** The seed list, the endpoint
grid, the primary interval and the demotion clause below are reproduced verbatim in
`experiments/run_reversal_seed_topup.py`, which refuses to start until this file is committed and its
hash is recorded in `PREREG_COMMIT`.

## What is being extended, and what is not

The paper's centerpiece is one cell: `coord_median` under `committed_pixel` on CIFAR-10 /
`cifar_cnn`, estimated two ways. The outcome-gated ladder (`dose_kappa<κ>`, adversary free to
attenuate) reads ΔASR = **−0.272**; the Mode-S intervention (`doseS_kappa<κ>`, adversary pinned at
c = 1) reads **+0.098**. Both 95% intervals exclude zero and they do not overlap, which is the sign
reversal the paper reports in the abstract, in §1, in §5, in the Conclusion and in Figure 1(c).

**Both legs are at n = 5.** The `n = 20` top-up already in the paper (`dose_seed_topup`, frozen at
`684b31e`) covers a *different* cell: the `krum` / `committed_scaling` equivalence arm. The centerpiece
has never been run beyond five seeds.

This document adds seeds so that the **separation between the two designs** can be stated at n = 20.
It does not re-test the reversal's direction, does not revise any threshold, and does not touch the
four other cells.

Per-seed endpoint ASR as published, seeds 42–46 in order, recomputed from the artifacts rather than
transcribed from the paper:

| leg | rung | per-seed ASR | rung mean | rung mean acc. |
|---|---|---|---|---|
| both (shared) | κ = 0 | 0.3946 / 0.2867 / 0.4488 / 0.6774 / 0.4051 | 0.4425 | 0.7665 |
| confounded (`dose_kappa2.0`) | κ = 2 | 0.1793 / 0.0897 / 0.2038 / 0.1920 / 0.1874 | 0.1704 | 0.6898 |
| controlled (`doseS_kappa2.0`) | κ = 2 | 0.5973 / 0.3191 / 0.5164 / 0.7557 / 0.5143 | 0.5406 | 0.6984 |

Paired per-seed differences `ASR_s(κ=2) − ASR_s(κ=0)`:

| leg | per-seed d | mean | sd | 95% interval (t₄ = 2.776) | half-width |
|---|---|---|---|---|---|
| confounded | −0.2152 / −0.1970 / −0.2450 / −0.4854 / −0.2177 | **−0.2721** | 0.1205 | [−0.4217, −0.1225] | 0.1496 |
| controlled | +0.2028 / +0.0324 / +0.0677 / +0.0782 / +0.1092 | **+0.0981** | 0.0646 | [+0.0178, +0.1783] | 0.0802 |

These reproduce the paper's `(−0.272, +0.098)` and its two printed intervals exactly, which is the
precondition for extending them. Holding each sd, `n = 20` (t₁₉ = 2.093) gives half-widths **0.0564**
and **0.0303**, a factor of **2.65** narrower on both legs. That factor is the entire purpose of the
run.

## Why this is not optional stopping, stated before any new seed runs

Adding seeds after seeing a result is the practice this paper condemns elsewhere, so the licence has
to be earned in advance and in writing.

1. **The published claim already holds at n = 5.** Both intervals exclude zero and they do not
   overlap (`−0.1225 < +0.0178`, a gap of 0.140). The top-up therefore cannot rescue a claim that is currently
   failing — there is no such claim. It can only narrow two intervals that already separate, or
   reveal at larger n that the separation was an artifact of five seeds. Only the second outcome is
   news, and it is news against us.
2. **The demotion clause** below pre-commits us to publishing that second outcome in the abstract and
   in Figure 1, at the cost of the paper's centerpiece. Naming that downside in advance is the only
   thing that licenses adding seeds.
3. **The seeds are fixed here**, contiguously, with no look-ahead: **47–61**, 15 new seeds, n = 20
   total. No stopping rule, no interim look, no extension of this list. If the runs are interrupted,
   the analysis reports the n actually reached and the intervals at that n; it does not resume until a
   threshold is crossed.
4. **Precedent, not innovation.** `dose_seed_topup` and `emit_only_topup` (both frozen at `684b31e`)
   extended arms from 5 seeds to 20 for the identical reason — interval width, not a second test.
   `dose_seed_topup` ran exactly 47–61 as a top-up block, which is the block reused here;
   `emit_only_topup` ran the full 42–61 in one suite. This is the same move on the cell that turned out
   to matter most.

## The primary quantity, and the endpoint grid

- **Primary, per leg: the paired interval.** `d_s = ASR_s(κ=2) − ASR_s(κ=0)`; the two-sided 95%
  Student-t interval on `mean(d)` with `n − 1` degrees of freedom. Pairing is real by construction: a
  seed fixes the Dirichlet partition, the model initialization and the participant sampling stream, so
  the two rungs at a given seed differ only in `d₁`.
- **Primary, across legs: whether the two intervals overlap**, and whether each excludes zero. This is
  the quantity the paper's sign-reversal claim rests on. **No p-value is attached to the difference
  between the two designs**, exactly as `main.tex` already states: they are not the same intervention,
  and `Λ_a` moves in both.
- **Endpoint rungs only: κ = 0 and κ = 2.** This is precisely what the frozen primary contrast of
  `pre_registration_comparability.md` reads (`LO, HI = "0.0", "2.0"` in `analyze_comparability.py`),
  so the top-up adds no new inferential surface.

**The consequence of endpoint-only is declared here rather than discovered later.** The interior rungs
κ ∈ {0.5, 1.0} stay at n = 5 for the new seeds' absence. Therefore:

- The four-rung Jonckheere–Terpstra statistics on this cell **stay at n = 5**, stay `pre_registered:
  false`, and stay post hoc. The top-up buys nothing for them and must not be read as if it had.
- **Any display of this cell's four-rung ladder must print n per rung.** A figure or table that shows
  four rungs without per-rung n after this top-up is reporting a mixed-n grid as a uniform one, which
  is the same misreport `rung_coverage` was added to prevent.
- `rung_coverage` for this cell will report κ = 0 and κ = 2 at n = 20 and the interior at n = 5. That
  is the honest shape and it is not to be smoothed.

## Decision rules

Nothing here is a new threshold. The accuracy floor and the endpoint grid are carried forward
unchanged from `pre_registration_comparability.md`.

- **Reversal confirmed at n = 20:** the confounded interval lies entirely below zero, the controlled
  interval entirely above zero, and the two do not overlap. Reported as the published result,
  narrowed.
- **Reversal demoted at n = 20 (the demotion clause):** if **either** interval contains zero, **or**
  the two intervals overlap, the paper reports the sign reversal as **not established at n = 20** and
  demotes it to a design *disagreement*. That demotion is made in the abstract, in §1, in Figure 1(a)
  and 1(c), in §5 and in the Conclusion — not in a footnote and not only in the appendix. The n = 5
  result is reported alongside it, and the disagreement between the two n's is the finding.
- **Direction change:** if either leg's mean changes sign at n = 20, that is reported as its own
  result and the demotion clause fires regardless of the intervals.
- **Accuracy gate.** Unchanged and inherited verbatim from `pre_registration_comparability.md:75`
  and `:149`: floor **0.35 applied to a rung's mean**, not per seed, and *"seeds below it are flagged,
  never excluded."* The runner records per-seed clean accuracy for every new run — `analyze_comparability.py`
  itself reads no accuracy, so the flagging is the runner's job and is written into the artifact rather
  than left to the analyzer.
- **No admissibility gate applies to this cell**, and that is deliberate. Cell 7's two-sided
  `[0.15, 0.85]` identity-rung gate (Amendment 4) existed to decide whether an *unrun* third dataset
  had headroom. This cell is already published at n = 5 with an identity rung of 0.4425; there is
  nothing to admit. Adding a gate now would create a licence to discard the run on its own outcome.

## Scope: which displayed numbers move to n = 20, fixed in advance

`+0.098` has **two referents** in the paper and they are sourced from the same directory, so the split
is fixed here before any result exists:

- **Moves to n = 20:** the comparability cell's two means and two intervals — `main.tex`'s abstract,
  §1, §5, the seven-cell table, the tiers table's replication row, the appendix's discussion of this
  cell, and Figure 1(c), which reads the artifact.
- **Stays at its own frozen n = 5:** Table 1 (`tab:channels`) row 4's Mode-S endpoint ΔASR. Table 1 is
  the Mode-S channel suite with its own freeze and its own four aggregators; its rows are n = 5
  throughout and mixing one row to n = 20 would make the table's rows incomparable.
- **Every site quoting −0.272 or +0.098 carries an explicit n** after this round, so the two referents
  cannot be read as one number. If the n = 20 controlled mean happens to round to +0.098 as well, both
  sites still print their own n.

## The shared identity rung, verified rather than assumed

At κ = 0 the transform returns the update list unwrapped, so `dose_kappa0.0` and `doseS_kappa0.0` are
the same computation. **This is already true bit-for-bit in the published artifacts**: for all five
seeds 42–46, `results/dose_response/` and `results/dose_replication/` carry identical float reprs at
κ = 0 in **both** ASR and clean accuracy (`0.39455555555555555`, `0.2866666666666667`,
`0.4487777777777778`, `0.6774444444444444`, `0.4051111111111111`). So the identity rung is computed
**once** per new seed and shared by both legs, giving 15 × 3 = **45 runs**, not 60.

`run_reversal_seed_topup.py --harness-check` re-establishes this rather than inheriting it: it computes
κ = 0 **in-suite at seed 42, where both published values already exist**, and asserts agreement with
both to `< 1e-9`. A published seed is used deliberately; at a new seed there is nothing to compare
against and the check would be vacuous. One run settles it for all fifteen new seeds because the code
path does not depend on the seed. **If that assert fails the top-up does not run**, because the shared
κ = 0 rung would then not be one rung.

The same check also reproduces a published κ = 2 value for this arm, so the harness is proved to be the
same loop the four published cells were run with, not merely the same at the identity.

## Caveats recorded in advance

1. **The sds are estimated from five seeds.** The projected half-widths of 0.0564 and 0.0303 assume
   `sd = 0.1205` and `0.0646` carry; the realized sds may be larger, and the realized intervals are
   what is reported. **No projected number appears in the paper.** For the intervals to touch at
   n = 20, the pooled sd would have to inflate by roughly 3×, which is stated as arithmetic and not as
   a prediction.
2. **Narrowing two intervals does not widen the claim.** At n = 20 this remains one aggregator, one
   attack, one dataset, one architecture, and one synthetic instrument that reads adversary identity.
   The top-up buys precision on the sign reversal and nothing else. It is not evidence about any other
   cell, it is not a second dataset, and Mode S is not a defense.
3. **The reversal's replication on CIFAR-100 stays at n = 5** and is not extended here. So after this
   round the reversal is n = 20 on CIFAR-10 and n = 5 on CIFAR-100, and both n's are printed wherever
   the replication is claimed.
4. **The controlled κ = 2 rung mean is 0.5406 at n = 5**, above 0.5. No rule in this cell's
   pre-registration reads a 0.5 ceiling — that clause belongs to the *equivalence* arms, where a
   generous margin must not certify a high-ASR composition. It is recorded because it looks like a
   gate violation and is not one.
5. **Seeds 47–61 are already used by `dose_seed_topup`, and 42–61 by `emit_only_topup`.** Different
   arms, different cell keys, own results directory, no collision — recorded because the overlap looks
   like one. Sharing the block is deliberate: every n = 20 arm in the paper then rests on the same
   seeds.
6. `results/dose_response/summary.json` and `results/dose_replication/summary.json` are **not written**
   by this suite. The top-up writes its own directory and the two are merged only at analysis time,
   where the five published seeds are asserted to reproduce before any pooled number is printed.

## Non-negotiables

1. No interval definition, accuracy floor, endpoint grid or demotion criterion above is revised after
   seeing an outcome.
2. The seed list is 47–61. It is not extended, truncated by inspection, or filtered.
3. Every reported number is recomputed per seed from the artifacts, never transcribed.
4. `results/dose_response/` and `results/dose_replication/` are not rewritten, and no existing runner
   is edited.
5. **`analyze_comparability.py`'s `PUB` reproduction guard is not weakened.** `"coord_median / pixel":
   (-0.272, +0.098)` stays, asserted against the **n = 5 subset** of the merged data. The n = 20 values
   are added as a separate expectation. Editing the published tuple to match a new result would destroy
   the only check that the re-score reproduces the document.
6. **The published n = 5 verdict is reported alongside the n = 20 verdict, whatever the latter is.** If
   they disagree, both appear, and the disagreement is the result.
7. This file is committed before `results/reversal_seed_topup/` is written; if that ordering cannot be
   demonstrated from `git log`, the top-up is reported as non-prospective.
