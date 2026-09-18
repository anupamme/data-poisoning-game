# Pre-registration: seed top-up of the CIFAR-100 replication of the sign reversal (Round 71)

**Status: frozen before any write to `results/cell7_seed_topup/`.** The seed list, the endpoint grid, the
primary interval and the demotion clause below are reproduced verbatim in
`experiments/run_cell7_seed_topup.py`, which refuses to start until this file is committed and its hash is
recorded in `PREREG_COMMIT`.

## What is being extended, and what is not

Cell 7 of the comparability suite (`pre_registration_comparability.md`, Amendment 4, frozen at `9b8a395`) is
the sign-reversal cell moved to a third dataset: `coord_median` under `committed_pixel` on CIFAR-100 /
`cifar_cnn`, estimated two ways. The outcome-gated ladder (`dose_kappa<κ>`, adversary free to attenuate)
reads ΔASR = **−0.213**; the Mode-S intervention (`doseS_kappa<κ>`, adversary pinned at c = 1) reads
**+0.071**. Both 95% intervals exclude zero and they do not overlap, which is what the paper means when it
says the reversal **replicates**.

**Both legs are at n = 5.** Round 63's top-up (`reversal_seed_topup`, frozen at `89beed0`) took the CIFAR-10
cell to n = 20 on both legs and explicitly declined to extend this one — its Caveat 3 reads, verbatim:

> *"**The reversal's replication on CIFAR-100 stays at n = 5** and is not extended here. So after this round
> the reversal is n = 20 on CIFAR-10 and n = 5 on CIFAR-100, and both n's are printed wherever the
> replication is claimed."*

This document closes that gap and nothing else. It adds seeds so that the **separation between the two
designs on CIFAR-100** can be stated at n = 20. It does not re-test the reversal's direction, does not revise
any threshold, does not touch the six other cells, and does not add a dataset, an architecture, an attack or
an aggregator.

Per-seed endpoint values as published, seeds 42–46 in order, recomputed from
`results/comparability_cells/summary.json` rather than transcribed from the paper:

| leg | rung | per-seed ASR | rung mean ASR | rung mean acc. |
|---|---|---|---|---|
| both (shared) | κ = 0 | 0.7143 / 0.5493 / 0.7557 / 0.8269 / 0.6708 | 0.7034 | 0.3931 |
| confounded (`dose_kappa2.0`) | κ = 2 | 0.5796 / 0.3256 / 0.5108 / 0.5442 / 0.4921 | 0.4905 | 0.3944 |
| controlled (`doseS_kappa2.0`) | κ = 2 | 0.8148 / 0.6433 / 0.8352 / 0.8495 / 0.7271 | 0.7740 | 0.3947 |

Paired per-seed differences `ASR_s(κ=2) − ASR_s(κ=0)`:

| leg | per-seed d | mean | sd | 95% interval (t₄ = 2.776) | half-width |
|---|---|---|---|---|---|
| confounded | −0.1347 / −0.2237 / −0.2448 / −0.2826 / −0.1787 | **−0.2129** | 0.0576 | [−0.2845, −0.1414] | 0.0715 |
| controlled | +0.1005 / +0.0940 / +0.0795 / +0.0226 / +0.0563 | **+0.0706** | 0.0317 | [+0.0312, +0.1100] | 0.0394 |

These reproduce the paper's `(−0.213, +0.071)` and its two printed intervals exactly, which is the
precondition for extending them. Holding each sd, `n = 20` (t₁₉ = 2.093) gives half-widths **0.0270** and
**0.0149**, a factor of **2.65** narrower on both legs. That factor is the entire purpose of the run.

## Why this is not optional stopping, stated before any new seed runs

Adding seeds after seeing a result is the practice this paper condemns elsewhere, so the licence has to be
earned in advance and in writing.

1. **The published claim already holds at n = 5.** Both intervals exclude zero and they do not overlap
   (`−0.1414 < +0.0312`, a gap of **0.1726**). The top-up therefore cannot rescue a claim that is currently
   failing — there is no such claim. It can only narrow two intervals that already separate, or reveal at
   larger n that the separation was an artifact of five seeds. Only the second outcome is news, and it is
   news against us.
2. **The demotion clause** below pre-commits us to publishing that second outcome in the abstract and in
   Figure 1, at the cost of the paper's replication claim. Naming that downside in advance is the only thing
   that licenses adding seeds.
3. **The seeds are fixed here**, contiguously, with no look-ahead: **47–61**, 15 new seeds, n = 20 total. No
   stopping rule, no interim look, no extension of this list. If the runs are interrupted, the analysis
   reports the n actually reached and the intervals at that n; it does not resume until a threshold is
   crossed.
4. **Precedent, not innovation.** `dose_seed_topup`, `emit_only_topup` (both `684b31e`) and
   `reversal_seed_topup` (`89beed0`) extended arms from 5 seeds to 20 for the identical reason — interval
   width, not a second test — and all three used the block 47–61. This is the fourth application of the same
   move, on the arm the previous one deliberately left behind, so that every n = 20 arm in the paper rests on
   the same seeds.
5. **The reviewer asked for the opposite of a new benchmark.** The eleventh review states it would *not* add
   another broad benchmark to increase the number of datasets and attacks, and separately notes that the
   CIFAR-100 replication is what keeps the paper from resting on one anomalous CIFAR-10 experiment. Adding
   precision to that one arm is therefore the narrowest empirical move available, and it is recorded here as
   the reason so that the scope cannot later be widened by appeal to the same review.

## The primary quantity, and the endpoint grid

- **Primary, per leg: the paired interval.** `d_s = ASR_s(κ=2) − ASR_s(κ=0)`; the two-sided 95% Student-t
  interval on `mean(d)` with `n − 1` degrees of freedom. Pairing is real by construction: a seed fixes the
  Dirichlet partition, the model initialization and the participant sampling stream, so the two rungs at a
  given seed differ only in `d₁`.
- **Primary, across legs: whether the two intervals overlap**, and whether each excludes zero. This is the
  quantity the paper's replication claim rests on. **No p-value is attached to the difference between the two
  designs**, exactly as `main.tex` already states: they are not the same intervention, and `Λ_a` moves in
  both.
- **Endpoint rungs only: κ = 0 and κ = 2.** This is precisely what the frozen primary contrast of
  `pre_registration_comparability.md` reads (`LO, HI = "0.0", "2.0"` in `analyze_comparability.py`), so the
  top-up adds no new inferential surface.

**The consequence of endpoint-only is declared here rather than discovered later.** The interior rungs
κ ∈ {0.5, 1.0} stay at n = 5 for the new seeds' absence. Therefore:

- This cell's four-rung trend statistics **stay at n = 5**, stay `pre_registered: false`, and stay post hoc.
  Concretely, `main.tex:1784`'s Jonckheere–Terpstra `p = 0.079` and permutation `p_↑ = 0.084` do **not**
  move, while the endpoint interval quoted in the same sentence does. That sentence must therefore state in
  one clause that the interval and the trend differ in n, per non-negotiable 13 below. It is the clearest
  instance of the mixed-n trap in this round and it is named in advance.
- **Any display of this cell's four-rung ladder must print n per rung.** A figure or table that shows four
  rungs without per-rung n after this top-up is reporting a mixed-n grid as a uniform one, which is the same
  misreport `rung_coverage` was added to prevent.
- `rung_coverage` for this cell will report κ = 0 and κ = 2 at n = 20 and the interior at n = 5. That is the
  honest shape and it is not to be smoothed. Note that `rung_coverage` reports the rungs common to *both*
  designs, so a leg completing alone must not be read as endpoint-only coverage.

## Decision rules

Nothing here is a new threshold. The accuracy floor and the endpoint grid are carried forward unchanged from
`pre_registration_comparability.md`.

- **Replication confirmed at n = 20:** the confounded interval lies entirely below zero, the controlled
  interval entirely above zero, and the two do not overlap. Reported as the published result, narrowed.
- **Replication demoted at n = 20 (the demotion clause):** if **either** interval contains zero, **or** the
  two intervals overlap, the paper reports the CIFAR-100 replication as **not established at n = 20** and
  demotes it to a design *disagreement*. That demotion is made in the abstract, in §1's table, in Figure 1's
  forest panel, in §6, in the seven-cell table and in the Conclusion — not in a footnote and not only in the
  appendix. The n = 5 result is reported alongside it, and the disagreement between the two n's is the
  finding.
  **What the demotion costs, named so it cannot be quietly minimized:** the paper's claim that the reversal
  *replicates on a third dataset* is the second pillar of its empirical contribution, and it appears in the
  abstract. If the clause fires, the reversal stands at n = 20 on CIFAR-10 alone and the paper says so in the
  abstract.
- **Direction change:** if either leg's mean changes sign at n = 20, that is reported as its own result and
  the demotion clause fires regardless of the intervals.
- **Accuracy gate.** Unchanged and inherited verbatim from `pre_registration_comparability.md:75` and `:149`:
  floor **0.35 applied to a rung's mean**, not per seed, and *"seeds below it are flagged, never excluded."*
  The runner records per-seed clean accuracy for every new run — `analyze_comparability.py` itself reads no
  accuracy, so the flagging is the runner's job and is written into the artifact rather than left to the
  analyzer. Note the margin is thin: the published rung means are 0.3931, 0.3944 and 0.3947 against a floor
  of 0.35, so a flag is a live possibility and not a formality.
- **Amendment 4's admissibility gate is not re-applied, and that is deliberate.** Cell 7's two-sided
  `[0.15, 0.85]` identity-rung gate existed to decide whether an *unrun* third dataset had headroom. It was
  evaluated once and **passed**: identity rung mean ASR 0.7034 in [0.15, 0.85], mean clean accuracy 0.3931 ≥
  0.35. The cell is now published at n = 5; there is nothing left to admit. Re-applying the gate at n = 20
  would create a licence to discard the run on its own outcome, which is the failure mode Amendment 4's own
  non-negotiable 4 forbids.

## Scope: which displayed numbers move to n = 20, fixed in advance

A grep of the two paper sources for `0.213`, `0.071`, `0.2129` and `0.0706`, comment lines excluded, returns
**16 sites in `paper/main.tex` and 1 in `paper/supplementary.tex`**, and they do **not** all quote the same
quantity. Line numbers are as of this commit and shift with re-papering: a site is located by grepping its
quoted words, never by trusting these integers. The split below is exhaustive and disjoint over all 17.

**Class A, moves to n = 20 (the cell-7 paired pair and every quantity derived from it), with n printed at
every site:** `main.tex` `:137` (§1's CIFAR-100 tabular row), `:722` (§6, *"the reversal replicates"*),
`:1027` (the evidence-status table's replication cell), `:1774` (the seven-cell table's CIFAR-100 row,
`5/5` → the reached n), `:1782` (the appendix's discussion of this cell), `:1784` (the endpoint interval
only — see the trend caveat above), `:2605`, `:2726`, `:2763` (`tab:tost`'s CIFAR-100 row: its `n` column,
its 95% interval, its TOST interval and its verdict), `:2795` and `:2813` (the margin ladder's `m* = 0.110`
for this arm and the loud-absence sentence that names it); `supplementary.tex:449` (the controlled interval);
and **Figure 1's forest panel**, which reads the artifact and already draws per-row n.

**Class C, digit collisions that are NOT this quantity, named so a later grep cannot sweep them in:**

1. **`0.213` as the CIFAR-10 cell's interval bound**, `[-0.334,-0.213]` — a different cell, and a bound
   rather than a mean: `main.tex` `:136`, `:589`, `:1207`, `:1578`, `:1770`, and `supplementary.tex:472`.
   These stay at CIFAR-10's own n = 20 and are untouched.
2. **`0.071` as a mean ASR *level* on the ResNet18 arm**, not a ΔASR at all: `main.tex` `:1651`, `:1653`,
   `:1660`. Different experiment, different estimand, different table. This is the same misclassification
   Round 63's Amendment 3 had to correct after Amendment 2 classified a site by its digits instead of by
   what the digits denote; it is caught here first.
3. **`0.155` as the CIFAR-10 arm's `m*` and interval bound**, not CIFAR-100's: `main.tex:2795` and `:2810`,
   `supplementary.tex:437`. Only the `0.110` in `:2795` is Class A; the `0.155` beside it is not.

**Class D, not touched at all:** every `workshop_paper/main.tex` site. That paper is out of scope by standing
rule and its numbers stay as published.

## The shared identity rung, verified rather than assumed

At κ = 0 the transform returns the update list unwrapped, so `dose_kappa0.0` and `doseS_kappa0.0` are the
same computation. **This is already true bit-for-bit in the published cell-7 artifact**: for all five seeds
42–46, the confounded and controlled κ = 0 rows in `results/comparability_cells/summary.json` carry identical
float reprs in **both** ASR and clean accuracy (`0.7143434343434344`, `0.5492929292929293`,
`0.7556565656565657`, `0.8268686868686869`, `0.6708080808080809`). So the identity rung is computed **once**
per new seed and shared by both legs, giving 15 × 3 = **45 runs**, not 60.

`run_cell7_seed_topup.py --harness-check` re-establishes this rather than inheriting it: it computes κ = 0
**in-suite at seed 42, where both published values already exist**, and asserts agreement with both to
`< 1e-9`. A published seed is used deliberately; at a new seed there is nothing to compare against and the
check would be vacuous. One run settles it for all fifteen new seeds because the code path does not depend on
the seed. **If that assert fails the top-up runs 60 runs or does not run at all**, because the shared κ = 0
rung would then not be one rung.

The same check also reproduces a published κ = 2 value for this arm, so the harness is proved to be the same
loop the published cell was run with, not merely the same at the identity.

**The verdict is written into the artifact, not only printed.** Round 69 found that an existing
`--harness-check` returns a verdict dict its caller discards, leaving a paper sentence witnessed only by run
stdout that costs hours to reproduce. This runner records the three comparisons, their deltas and their
verdict in `results/cell7_seed_topup/summary.json`, so the claim has an artifact.

## Caveats recorded in advance

1. **The sds are estimated from five seeds.** The projected half-widths of 0.0270 and 0.0149 assume
   `sd = 0.0576` and `0.0317` carry; the realized sds may be larger, and the realized intervals are what is
   reported. **No projected number appears in the paper.** For the two intervals to touch at n = 20 the
   pooled sd would have to inflate by roughly 3×, which is stated as arithmetic and not as a prediction.
2. **Narrowing two intervals does not widen the claim.** At n = 20 this remains one aggregator, one attack,
   one architecture family and one synthetic instrument that reads adversary identity. The top-up buys
   precision on the CIFAR-100 replication and nothing else. It is not evidence about any other cell, and
   Mode S is not a defense.
3. **The reversal is then n = 20 on two datasets and the interior rungs are n = 5 on both.** Round 63's
   Caveat 3 is discharged, not inherited: after this round no site may say the replication stands at n = 5.
   Every site that currently prints `n{=}5` for this cell is listed in Class A above precisely so that none
   is missed.
4. **`Λ_a` moves in both designs on this cell, and no admission prediction exists for it.** Amendment 4 set
   `admsrc = None` on purpose because H-ADMISSION-GATED was refuted in Amendment 3, so `dΛ_a` and
   `predicted` print as absent-value cells. That does not change at n = 20, and the extra seeds must not be
   used to compute an admission artifact for this dataset and read a threshold off it.
5. **This cell carries no equivalence reading, at any n.** Its controlled interval excludes zero, so
   `analyze_margin_sensitivity.py` excludes it from the binding-arm search and `tab:tost`'s row for it is
   flagged. Narrowing the interval makes it exclude zero more firmly, so no amount of extra precision here
   can turn into an equivalence claim.
6. **Seeds 47–61 are already used by three other top-ups.** Different arms, different cell keys, own results
   directory, no collision — recorded because the overlap looks like one.
7. `results/comparability_cells/summary.json` is **not written** by this suite. The top-up writes its own
   directory and the two are merged only at analysis time, where the five published seeds are asserted to
   reproduce before any pooled number is printed.

## The analysis wiring, fixed here so it is not chosen after the result

Four guards in `analyze_comparability.py` govern how the merged cell is read. Three of them behave the
opposite of the obvious guess, so what happens to each is fixed now:

1. **Sources.** `("cell7_seed_topup", CONF, "|cifar100")` and its `CTRL` twin are appended **after** the
   published entries. `series` is first-writer-wins, so seeds 42–46 keep their published values and the n = 5
   subset stays recoverable.
2. **`REQUIRED_SEEDS` is extended to 42–61.** It is the incompleteness guard, not a completeness assertion:
   any declared seed missing on **either** leg suppresses the cell's verdict entirely. Leaving it at 42–46
   would let the cell score mid-run at mixed n, which is the exact defect its own comment records (a
   controlled κ = 2 rung at n = 4 of 5 once produced a verdict comparing the legs at n = 5 against n = 4).
   The verdict therefore appears only when both legs carry all 20 seeds.
3. **`PUB` gains `"coord_median / pixel, CIFAR-100": (-0.213, +0.071)` and `PUB_SEEDS` gains its 42–46
   subset**, so the published pair is asserted against the n = 5 subset and is never retyped to match the new
   result. **This requires dropping the `training and` conjunct on the guard**, because cell 7 is an
   out-of-sample cell and the entry would otherwise assert nothing — a silent no-op that reads as a guard.
   The conjunct is redundant with `label in PUB` (its four current keys are exactly the four training cells,
   and neither remaining out-of-sample cell is in `PUB`), so no other cell's behaviour changes. The change
   only ever *adds* assertions, and it is reported in the response letter rather than left as a quiet diff.
4. **`PAPER_N20` gains a cell-7 key, left `None` until the paper quotes the new n**, then filled **from the
   paper**. A value written before the run is a prediction; a value copied from the artifact and asserted
   against that same artifact checks nothing.

`analyze_tost_existing.py`'s cell-7 arm gains the same extra source, using the fourth-element pattern its
EMNIST row already uses. `analyze_margin_sensitivity.py` reads `results/comparability_six_cells.json` and so
follows automatically; its `m*` for this arm moves and its loud-absence line is Class A above.

**Every guard is proved to fire rather than read as green.** A guard that is a no-op looks identical to a
guard that holds, so each is checked by making it fail on purpose: an incomplete top-up must suppress the
verdict, a perturbed `PUB` tuple must print `REFUSING TO PRINT A VERDICT`, and an unformable `PUB_SEEDS`
subset must refuse rather than pass.

## Non-negotiables

1. No interval definition, accuracy floor, endpoint grid, admissibility decision or demotion criterion above
   is revised after seeing an outcome.
2. The seed list is 47–61. It is not extended, truncated by inspection, or filtered.
3. Every reported number is recomputed per seed from the artifacts, never transcribed.
4. `results/comparability_cells/` is not rewritten, and no existing runner is edited.
5. **`analyze_comparability.py`'s `PUB` reproduction guard is not weakened.** The published tuple
   `(-0.213, +0.071)` is asserted against the **n = 5 subset** of the merged data. Editing it to match a new
   result would destroy the only check that the re-score reproduces the document. The one change to guard
   logic is the `training` conjunct in item 3 above, which strengthens the guard and is declared before the
   run.
6. **The published n = 5 verdict is reported alongside the n = 20 verdict, whatever the latter is.** If they
   disagree, both appear, and the disagreement is the result.
7. This file is committed before `results/cell7_seed_topup/` is written; if that ordering cannot be
   demonstrated from `git log`, the top-up is reported as non-prospective.
8. Every Class A site prints its own `n`. A site quoting either number with no `n` after this round is a
   defect, not a style choice.
9. No Class C collision is edited. If one is, it is reported as a reclassification in the response letter.
10. No frozen threshold is re-evaluated at n = 20. Where an n = 20 mean would change a frozen verdict's side
    of a threshold, both n's and both readings are printed and the frozen verdict stays labelled as the
    frozen one.
11. The interior rungs are not run, and no display of this cell's four rungs omits per-rung n.
12. `main.tex:887`'s count of pre-registration documents is updated for this file. It is a self-counting
    claim with no emitter, which no gate, hash or build can see going stale.
13. Where a moved site and a frozen site quoting the same estimand fall in the same sentence or section, that
    site states in one clause that the two differ in `n` and not in quantity. `:1784` is the named instance:
    its endpoint interval moves and its two trend p-values do not.
