# Pre-registration: the multi-cell comparability replication (Round 35)

**Status: frozen before any write to `results/comparability_cells/`.** Every rule, prediction and
refutation criterion below is reproduced verbatim in `experiments/run_comparability_cells.py` and
`experiments/analyze_comparability.py`, which refuse to start until this file is committed and its
hash is recorded in `PREREG_COMMIT`.

## The question this round exists to answer

The paper's most memorable result is a single cell. On `coord_median` under the pixel backdoor, the
outcome-gated ladder estimates ΔASR = −0.272 and the Mode-S intervention estimates +0.098 on the
identical cell, at the same seeds and the same nominal dose. A reviewer's fair objection:

> Fine, you got −0.272 versus +0.098, but perhaps this is peculiar to that attack/architecture.

This document freezes a **multi-cell replication** of that contrast, and — more importantly — freezes
a **mechanism** for when the two designs disagree, so that the replication can fail.

## What is re-scored and what is newly run

**Four cells already exist in frozen artifacts and are RE-SCORED, NEVER RE-RUN.** Both the confounded
(`dose_kappa<κ>`, Round 11) and the controlled (`doseS_kappa<κ>`, Round 12) ladders are already
present for all four Mode-S arms. No run in `results/dose_response/`, `results/targeted_dose/`,
`results/dose_replication/` or `results/dose_seed_topup/` is repeated, and their published per-seed
values must reproduce **bit-identically** before the analyzer prints any pooled statement.

**These four cells are the evidence the mechanism below was read off. They are TRAINING DATA and are
labelled as such everywhere they appear.** They are not evidence *for* the mechanism; the two new
cells are.

## The mechanism, frozen before the new runs

Re-scoring the four existing cells gives a perfect separation:

| arm | admission at top rung | confounded ΔASR | controlled ΔASR | designs |
|---|---|---|---|---|
| `krum` / scaling | **0.000** | −0.008 | −0.010 | agree |
| `cos_krum` / pixel | **0.000** | −0.405 | −0.425 | agree |
| `reputation` / scaling | **0.020** | +0.762 | +0.178 | **disagree** (disjoint) |
| `coord_median` / pixel | **0.033** | −0.272 | +0.098 | **disagree** (sign reversal) |

Admission is read from `results/admission_measurement.json`, a channel measurement that computes no
ASR and legitimately predates this freeze.

**H-ADMISSION-GATED.** The confounded and controlled designs disagree **exactly when the downstream
defense admits adversarial mass**, and agree when it admits none:

> admission ≡ 0.000 at every rung  ⟹  the two designs AGREE
> admission > 0.000 at any rung    ⟹  the two designs DISAGREE

The reason, stated so it can be wrong: the confounded ladder fixes `mean(c) = 1` but leaves relative
weights free, so it lets the adversary be attenuated. Attenuating the adversary can only change the
outcome if the adversary's contribution reaches the aggregate at all. Where the defense admits
nobody, attenuation has nothing to act on and the confounded estimate is unbiased **by accident**.
Where the defense admits some adversarial mass, attenuation reaches ASR through a path that does not
pass through the downstream statistic, and the confounded estimate is biased — in `coord_median`'s
case past zero.

**This ties the multi-cell result to the paper's own (P4) level rather than to an ad-hoc covariate.**
It says the confounded design is untrustworthy precisely where the security content sits.

## Operational definitions, frozen

- **AGREE** := the two designs' 95% paired `t` intervals overlap **and** their point estimates share a
  sign (or both point estimates lie within ±0.05 of zero).
- **DISAGREE** := the intervals are disjoint, **or** the point estimates have opposite signs.
- **SIGN REVERSAL** := a strict subset of DISAGREE, where the signs differ **and** both intervals
  exclude zero. This is the strongest form and is counted separately; it is not claimed for any cell
  that does not meet it.
- `admission` is the arm's own admission measure at its top rung, as already instantiated per
  aggregator in `experiments/build_channel_table.py`. **No new admission measure is introduced.**

## The two new cells, and their predictions

Both run at seeds 42--46, rungs κ ∈ {0, 0.5, 1, 2}, accuracy floor 0.35 applied to a rung's **mean**
(not per seed), matching every existing arm.

### Cell 5 — `coord_median` under `committed_scaling` (40 runs, both ladders)

The adjudicating cell. `coord_median`/pixel is the paper's only sign reversal, so the same aggregator
under the **other** committed attack tests directly whether the hero result is attack-specific.

**Procedure, in this order, and the order is the point:**
1. Measure the channels first, computing **no ASR** — exactly the discipline the EMNIST-byclass and
   Mode-M arms used. This yields `admission` for this cell.
2. The frozen rule above then **mechanically generates** the prediction. No judgement is applied at
   this step and none is permitted.
3. Only then run both ASR ladders.

**Point prediction, committed now so that step 2 cannot be quietly reinterpreted:** we expect
`coord_median` to admit adversarial coordinates under model-scaling as it does under the pixel
backdoor, therefore admission > 0, therefore **DISAGREE**. If the channel measurement returns
admission ≡ 0, the rule predicts **AGREE** and that is what is scored; the point prediction is then
recorded as wrong even if the rule survives.

### Cell 6 — `krum` / `committed_scaling` on EMNIST-byclass/`simple_cnn` (20 runs, confounded only)

The controlled half already exists in `results/dose_femnist/`; only the confounded ladder is new. This
adds a **second dataset and a second architecture** to the comparability axis.

**Prediction: AGREE.** The paper already reports that on this arm the adversary is *never admitted*,
so admission ≡ 0 and the frozen rule predicts the two designs coincide. **This is the prediction we
most want to survive, because it is the one that can most easily fail**: it forecasts a null on a new
dataset and architecture, and any disagreement here refutes the mechanism outright.

## What refutes us, named in advance

The mechanism is refuted, and reported as refuted, if **either**:

- a new cell **disagrees where the rule predicted agreement** (in particular, if the EMNIST cell
  disagrees); or
- a new cell **agrees where the rule predicted disagreement** (in particular, if `coord_median` admits
  adversarial mass under scaling and the two designs nonetheless coincide).

On refutation: H-ADMISSION-GATED is withdrawn, the four-cell pattern is demoted from a **mechanism**
to a **description**, the paper says so in the section where the claim appears, and the multi-cell
table is still reported in full. **The observation survives the mechanism's death; only the
explanation is retracted.**

**A partial run is not a verdict.** The analyzer reports the n actually reached per rung, prints an
explicit warning when rungs are unequal, and never scores the rule on an incomplete ladder.

## What this round does NOT establish, recorded before the numbers exist

- **2-of-4 is not "the sign reversal replicates".** Even in the best case the result is that
  disagreement is *structured*, not that it is universal. Any sentence in the paper implying the
  reversal is generic is a misreport of this design.
- **Six cells are six cells.** Two datasets, two architectures, two attacks and four aggregators of
  three structural kinds is the scope; nothing here licenses a claim about composed defenses in
  general, or about aggregators not on this menu.
- **Admission is measured per aggregator and is not one scalar comparable across defenses.** The rule
  above thresholds each arm's own measure at zero, which is the only comparison this instrumentation
  supports. It does **not** claim admission magnitudes are commensurable between arms.
- **No causal decomposition of ΔASR is claimed anywhere**, here or elsewhere in the paper.
- Mode S is an oracle instrument that reads adversary identity. It is not a deployable defense, and a
  cell where it agrees with the confounded design is not a recommendation to use either.

## Non-negotiables

1. The four published cells are re-scored from frozen artifacts and never re-run; their values must
   reproduce bit-identically before any pooled number prints.
2. `SEEDS5` and `ARMS` in `run_targeted_dose.py`, and `ARMS` in `run_dose_response.py`, are **not
   edited**. The new runner writes only `results/comparability_cells/`.
3. The channel measurement for cell 5 precedes its ASR runs, and the commit ordering makes that
   auditable.
4. No cell is added, dropped or relabelled after any ASR for it exists.
5. If the rule and the point prediction disagree, the **rule** is scored and the point prediction is
   reported as wrong.
6. The accuracy floor applies to a rung's mean; seeds below it are flagged, never excluded.

---

# AMENDMENT 1, before any write to `results/comparability_cells/`

**Disclosed as an edit, not folded in silently.** The freeze at `c986ef4` defined the gating quantity
as "the arm's own admission measure ... as already instantiated per aggregator in
`experiments/build_channel_table.py`". That file computes **two** admission quantities, and the
original text does not say which:

- `d_admission` — the fraction of rounds in which the **support** of the adversarial mass changes,
  i.e. `(base > 0) != (post > 0)`;
- `d_influence` — the magnitude by which the adversarial mass moves, `|post − base|`, which the paper
  calls the **admission-level change ΔΛ_a**.

**The support reading cannot be the intended one, and this is decidable without any new data.** On all
four training cells `d_admission` is identically `0.0000`, which is the paper's own headline finding
that the support never moves in any measured round of any arm. A quantity that is constant across the
four cells has **no discriminating power at all** and would make the frozen rule predict AGREE
everywhere, contradicting two of the four cells it was read off.

**The gating quantity is therefore `d_influence` (ΔΛ_a)**, evaluated at the top rung by
`experiments.build_channel_table.channels(ADM, arm, attack, 2.0)["d_influence"]`. On the training
cells it separates them exactly:

| cell | Δ dec. | Δ adm. (support) | **ΔΛ_a** | observed |
|---|---|---|---|---|
| `krum` / scaling | 0.7333 | 0.0000 | **0.0000** | agree |
| `cos_krum` / pixel | 0.0000 | 0.0000 | **0.0000** | agree |
| `reputation` / scaling | 1.0000 | 0.0000 | **0.0196** | disagree |
| `coord_median` / pixel | 0.4821 | 0.0000 | **0.0332** | disagree |

**The predictions this yields are the ones already committed at `c986ef4`, unchanged:**

| new cell | ΔΛ_a | rule | point prediction at `c986ef4` |
|---|---|---|---|
| cell 5 `coord_median` / scaling | **0.0086** | DISAGREE | DISAGREE — **agrees** |
| cell 6 `krum` / EMNIST-byclass | **0.0000** | AGREE | AGREE — **agrees** |

**This amendment moves no prediction.** It removes an ambiguity about how the two already-committed
predictions were derived. Had the two readings implied different predictions, the honest course would
have been to report both and score both; they do not.

**One weakness recorded now rather than discovered later.** Cell 5's ΔΛ_a is `0.0086`, well below the
training cells' `0.0196` and `0.0332` while still above zero. The rule thresholds at zero, so it
predicts DISAGREE, but this is the **weakest** margin of any cell and the prediction is correspondingly
the least confident. If cell 5 agrees, that is evidence the rule needs a threshold above zero rather
than evidence it is wrong in kind — and **we commit now to reporting it as a failure of the rule as
frozen**, not to introducing a fitted threshold after the fact.

**Ordering.** `results/comparability_cells/` does not exist at the time of this amendment; no ASR for
either new cell has been computed. The channel measurements quoted above are read from
`results/admission_measurement.json` and `results/femnist_admission.json`, which compute no ASR and
predate the freeze, exactly as `pre_registration_dose_mask.md` records for its own channel data.

---

# AMENDMENT 2, after the first scoring of cell 6 and before any further run

**Disclosed as an edit made with knowledge of a result, which Amendment 1 was not.** This amendment is
written *after* `results/comparability_cells/` was completed at 60/60 and after
`analyze_comparability.py` scored the six cells once. **That first scoring returned REFUTES**, and the
number this amendment changes is the one that produced it. The full disclosure is the point of the
section.

## What the first scoring said

| cell | ΔΛ_a | rule | observed | |
|---|---|---|---|---|
| cell 5 `coord_median` / scaling | 0.0086 | DISAGREE | DISAGREE | confirms |
| cell 6 `krum` / EMNIST-byclass | 0.0000 | AGREE | **DISAGREE** | **refutes** |

## The defect in the comparison, which is procedural and not a matter of judgement

`paired()` in `analyze_comparability.py` pairs seeds **within** a design, between its bottom and top
rung, and this is correct. It does **not** pair **across** the two designs. On cell 6 the two halves do
not have the same seed set:

- confounded ladder (new, `results/comparability_cells/`): seeds 42, 43, 44, 45, 46 — **n=5**
- controlled ladder (frozen, `results/dose_femnist/`): seeds 42, 43, 44 — **n=3**

So the reported contrast compared a five-seed mean against a three-seed mean. The disagreement is
carried entirely by seeds 45 and 46, which have no controlled counterpart at all:

```
confounded  s45: d = +0.1233     s46: d = +0.6246      <- present on one side of the comparison only
```

**This violates two clauses already frozen in this document at `c986ef4`.** The section "What refutes
us" is predicated on the paired-difference convention that a cell's two designs run at the same seeds,
and the clause "A partial run is not a verdict" requires that the analyzer "prints an explicit warning
when rungs are unequal, and never scores the rule on an incomplete ladder." **It printed no warning and
it scored an incomplete ladder.** The freeze anticipated this failure and the analyzer did not
implement it.

## Why this amendment does not choose the answer

Restricting cell 6 to the three shared seeds yields, with the pairing the freeze requires:

| cell 6, paired on seeds 42--44 | Δ (κ=0 → κ=2) | within ±0.05 of zero |
|---|---|---|
| confounded | +0.0123 [−0.0774, +0.1020] n=3 | yes |
| controlled | −0.0167 [−0.0328, −0.0005] n=3 | yes |

Both point estimates fall inside the `±0.05` band, and the intervals overlap, so the frozen AGREE
definition is met and the cell would **confirm**. The `±0.05` clause was frozen at `c986ef4` before any
result existed and is not introduced here.

**We decline to resolve the cell this way.** Which seeds are scored is exactly the degree of freedom
this document exists to remove, and the subset that rescues the mechanism is the subset that would be
chosen. A verdict that depends on that choice is not a verdict.

## What is therefore run, and what is committed in advance

**The controlled EMNIST-byclass ladder is completed to the frozen seed set 42--46**: seeds 45 and 46 at
all four rungs, **8 runs**. Rungs 0.5 and 1.0 are included even though the endpoint contrast does not
read them, because leaving them at n=3 while the endpoints sit at n=5 reproduces the same unequal-rung
defect one level down.

This **completes** the frozen design rather than extending it. The freeze specifies seeds 42--46 for
both new cells; cell 6 was scoped "confounded only, 20 runs" on the mistaken assumption that the
existing controlled half already covered 42--46. It covers 42--44. No cell is added, dropped or
relabelled, and non-negotiable 4 is not touched.

Writes go to `results/comparability_cells/` under the `|emnist` key suffix. **`results/dose_femnist/`
is not written to and its three seeds remain authoritative** — the analyzer's merge is first-writer-wins
with the frozen directory listed first, so seeds 42--44 continue to come from the published artifact.

**Committed before these 8 runs exist:**

1. **The verdict is whatever the paired n=5 versus n=5 comparison returns.** No further seed set,
   subset, rung or threshold is introduced after it prints.
2. **Both numbers are reported in the paper regardless of outcome**: the unpaired n=5-vs-n=3 contrast
   that first scored REFUTES, and the paired n=5-vs-n=5 contrast. A reader must be able to see that the
   first scoring refuted the mechanism and why the comparison changed.
3. **If the paired comparison still disagrees, H-ADMISSION-GATED is refuted** on the terms already
   frozen in "What refutes us": the mechanism is withdrawn, the four-cell pattern is demoted from a
   mechanism to a description, and the six-cell table is reported in full.
4. **The analyzer is fixed to pair across designs and to warn on unequal n**, which is the behaviour
   `c986ef4` already required. The fix is prereg-mandated, and it is disclosed here that its direction
   happens to favour the hypothesis.
5. Cell 5 is untouched by this amendment. It was already paired at n=5 versus n=5 and it confirms.

**This amendment moves no prediction.** Cell 6's frozen prediction remains AGREE, on ΔΛ_a = 0.0000, as
committed at `c986ef4` and restated in Amendment 1.

---

# AMENDMENT 3, correcting Amendment 2's premise

**Amendment 2 asserted something false and this amendment withdraws it.** Amendment 2 claimed that
pairing the two designs across a shared seed set is "the paired-difference convention the freeze
assumed." Implementing it showed that it is not, and the evidence is decisive and internal:

**Under cross-design pairing, the four published cells do not reproduce.**

| cell | controlled, as frozen and published | controlled, cut to the confounded side's seeds |
|---|---|---|
| `krum` / scaling | −0.010 n=20 | **−0.026** n=5 |
| `cos_krum` / pixel | −0.425 n=8 | **−0.272** n=5 |

The analyzer's own reproduction gate rejected the run outright. Non-negotiable 1 requires the published
values to reproduce before any pooled number prints, so the convention that fails it cannot be the one
`c986ef4` scored. The prereg's table prints `n5`, `n20` and `n8` beside one another, which is what
full-n scoring looks like. **"Paired" in this document means paired across a design's two rungs, at the
same seeds within that design** — which the original `paired()` implemented correctly.

## Consequences, stated against our own interest

1. **The scored contrast is each design at its own full n.** Restored.
2. **Cell 6 disagrees, and H-ADMISSION-GATED is refuted.** The refutation reported by the first scoring
   stands. It was not an artifact.
3. **The shared-seed contrast is retained as a disclosed robustness check and is never the verdict.**
   On cell 6 it would give AGREE. Adopting it for cell 6 while the four published cells require full-n
   would be scoring one cell by a rule that the rest of the table refutes, which is the post-hoc rescue
   this document exists to prevent.
4. **Amendment 2's 8 runs proceed and are still worth having.** They were launched before this
   correction and they remain the right experiment: completing the controlled EMNIST ladder to seeds
   42--46 makes cell 6's two designs full-n *and* seed-identical, so the two conventions coincide there
   and the cell is decided by data under either reading. What changes is the expectation: the frozen
   convention already returns DISAGREE, and the controlled deltas at the three existing seeds
   (−0.0228, −0.0172, −0.0099) are all small and negative while the confounded ladder's new seeds are
   large and positive (+0.1233 at s45, +0.6246 at s46). If the controlled ladder is also near zero at
   s45 and s46, the refutation is confirmed on a fully paired n=5 comparison.
5. **The unequal-n warning is kept and generalised.** It fires on three cells, not one: `krum`/scaling
   (controlled carries 15 seeds the confounded ladder lacks), `cos_krum`/pixel (3), and
   `krum`/EMNIST (the confounded ladder carries 2 the controlled lacks). The asymmetry was always
   present in the published table and was never surfaced. It is now printed before the table, so no
   reader reaches a verdict without seeing it.

## The process failure worth recording

Amendment 2 was written after seeing a result, and it reached for a reading of the freeze that would
have overturned that result. The reading was wrong, and what caught it was a mechanical gate --
non-negotiable 1's bit-identical reproduction requirement -- and not our judgement. **That is an
argument for the gate, not for our discipline.** The sequence is left in the document in full: the
refutation, the amendment that would have undone it, and the check that stopped the amendment.

---

# AMENDMENT 4, adding cell 7: the sign reversal on a third dataset

**Written before any ASR for this cell exists, which non-negotiable 4 requires.**
`results/comparability_cells/summary.json` contains no key with the `|cifar100` suffix at the time of
writing, and the amendment is committed before the runner is pointed at it.

## Why this cell, and why now

The nineteenth review (reading a July draft) withholds a 7 partly because *"the strongest empirical
phenomenon is still demonstrated primarily in a relatively constrained FL setting."* That criticism
survives the restructure, and on inspection it is sharper than the review makes it. The six-cell table
spans two datasets and two architectures, but **the sign reversal itself does not**: `coord_median` /
`committed_pixel` -- the one cell where the signs differ *and* both intervals exclude zero -- exists
only on CIFAR-10 with `cifar_cnn`. Cell 6 added a second dataset and architecture to a *different*
cell (`krum` / scaling), and that cell is explicitly **not** counted as a reversal because both of its
intervals contain zero.

So the honest statement of the current evidence is: **the reversal is one cell on one dataset with one
architecture.** Cell 7 moves that exact cell, unchanged in aggregator and attack, to a third dataset.

## Cell 7, frozen

| field | value |
|---|---|
| `d2` | `coord_median` |
| attack | `committed_pixel` |
| dataset / model | **`cifar100` / `cifar_cnn`** |
| families | both (`confounded`, `controlled`) |
| rungs | κ ∈ {0, 0.5, 1, 2}, the frozen grid |
| seeds | 42--46, the frozen set |
| runs | 40 |
| key suffix | `|cifar100` |
| primary | ΔASR from κ=0 to κ=2, paired within design, 95% paired `t` interval, each design at its own full n |
| verdict vocabulary | AGREE / DISAGREE / SIGN REVERSAL exactly as frozen above; no new definition |

`num_clients=10`, `clients_per_round=5`, `num_rounds=50`, `ADV_FRACTION=0.2`, Dirichlet α=0.5,
accuracy floor 0.35 applied to a rung's **mean**: all unchanged, all inherited from the same
`FL_CONFIG` the other six cells use.

## No rule prediction is made, and the ΔΛ_a column is reported absent for this cell

H-ADMISSION-GATED was **refuted by cell 6 and withdrawn** (Amendment 3; the paper reports the
withdrawal in the section where the claim appeared). It therefore generates no prediction for cell 7,
and **cell 7 must not be used to resurrect it.** Concretely:

- No admission measurement is run for this cell, and its ΔΛ_a table entry is an **absent-value cell**,
  not a number. A number there would invite exactly the post-hoc threshold-fitting Amendment 3
  forbids.
- **No prediction is committed** beyond the direction of interest below, because the only mechanism
  that could have generated one is dead. Registering a bare hunch as a prediction, and then scoring
  it, would manufacture the appearance of a surviving rule.
- The **direction of interest**, stated so it cannot be reinterpreted afterwards: the CIFAR-10 cell has
  the confounded ladder falling (−0.272) and the Mode-S instrument rising (+0.098). Cell 7 replicates
  the *reversal* only if the same two signs appear with both intervals excluding zero.

## The admissibility gate, and it is two-sided

**A cell with no headroom cannot answer this question, and the paper already discloses one such cell
against itself** (§`app:sixcell`, on EMNIST: *"the indeterminate branch was arithmetically
unreachable: ASR cannot fall below 0"*). Reproducing that defect on a new dataset would be worse the
second time, so the gate is frozen here, before the run:

> **Cell 7 is admissible iff, at the identity rung (κ=0, confounded, seeds 42--46), the rung mean
> clean accuracy is ≥ 0.35 and the rung mean ASR lies in [0.15, 0.85].**

The interval is two-sided for a reason and both sides bind:

- **Below 0.15** the *fall* leg is dead: ASR cannot go below 0, so a floored identity rung bounds the
  confounded fall beneath the ±0.15 practical-equivalence margin this paper uses everywhere, and no
  fall could be distinguished from the margin. This is the EMNIST failure exactly (identity rung
  0.027).
- **Above 0.85** the *rise* leg is dead by the mirror argument: ASR cannot exceed 1, so a saturated
  identity rung bounds the Mode-S rise beneath the same margin. **This side is the live risk here**,
  because `results/cifar100_ne3_br/summary.json` records pixel-backdoor ASR of 0.82--0.93 on this
  dataset under a `fedavg`/`norm_clip` policy. Under `coord_median` we expect materially lower (the
  CIFAR-10 identity rung is 0.443), but we do not know it, and if the rung saturates then a reversal
  is undetectable in one of its two directions and the cell cannot be scored for one.

The gate reads the **natural prefix of the runner's own todo order** (`for fam in fams for k in KAPPAS
for s in SEEDS` puts confounded/κ=0/seeds 42--46 first), so no code, no cell and no label changes to
evaluate it: the first five lines of the run log decide it. **If the gate fails the run is stopped
there**, the five readings are reported with the gate they failed, and the recorded outcome is that
this cell is inadmissible on headroom.

**There is no fallback dataset and we say so now rather than reaching for one later.** The only other
configurations this harness supports are EMNIST/`simple_cnn` (2660 s per run in
`results/comparability_run.log`, so 29.5 h for 40 runs, and its identity rung is 0.027, which fails
the gate a priori) and `resnet18` (5900 s per run in `results/cifar10_mix_ratio_sweep_resnet18`, so
65 h). Both are outside the compute available. If CIFAR-100 fails the gate, the honest finding is
**"no third dataset within this compute budget admits this cell"**, and that is what will be written.
Substituting a different aggregator, attack or cell after seeing the gate is forbidden by
non-negotiable 4 and is not on the table.

## Why CIFAR-100 rather than something else, disclosed before the run

Three reasons, in the order they actually decided it:

1. **Wall time.** 660 s per run for `cifar_cnn` on CIFAR-10 across 40 timed runs in
   `results/comparability_run.log`. CIFAR-100 is the same input size and the same architecture with a
   100-way head, so ≈7.5 h for 40 runs, which fits the compute available. The two alternatives above
   do not.
2. **The accuracy floor is reachable.** `results/cifar100_ne3_br/summary.json` records clean accuracy
   0.386--0.394 on all five seeds under the pixel backdoor, above the 0.35 floor. (Its *model-scaling*
   arm collapses at seed 45 to accuracy 0.083; we are not running that arm, and if the pixel arm shows
   a comparable collapse it is flagged, never excluded, per non-negotiable 6.)
3. **It is a genuinely different task, not a re-skin.** 100 classes rather than 10 changes the
   backdoor's base rate and the clean-accuracy regime together, which is the kind of variation the
   external-validity objection is actually about.

Choosing on wall time is a real limitation and it is recorded as one: **this is the third dataset that
fit, not the third dataset that was most informative.** Nothing here licenses a claim about datasets
or architectures outside the three now measured.

## What this cell does not establish, recorded before the numbers exist

- **A seventh cell does not make the reversal generic.** Three datasets and two architectures is the
  scope after this run, and any sentence implying otherwise is a misreport of this design.
- **A null is a result and is reported as one.** If cell 7 agrees, or disagrees without reversing, that
  is the answer and it goes in the table. The reversal on CIFAR-10 is not retracted by a null
  elsewhere, and it is not generalised by a hit elsewhere; the honest reading of either is that the
  reversal is dataset-conditional, which is a scope condition the paper already owes the reader.
- **No causal decomposition of ΔASR is claimed** for this cell any more than for the others.
- **Mode S remains an oracle instrument**, not a deployable defense.

## Non-negotiables, extending the six above

7. Cell 7 is added as **one tuple** in `CELLS`. `run_one`, `FAMILIES`, `cell_key`, `KAPPAS`, `SEEDS`
   and the harness check are not edited; a divergence in any of them would make cell 7 a different
   experiment from the six it joins.
8. `--harness-check` is run **before** cell 7's first run and must reproduce the published
   `results/dose_response/` value to 1e-9. It is not invoked by `main()`, so it is a manual pre-flight
   and its absence is silent -- which is precisely why it is written down here.
9. The `|cifar100` suffix is mandatory, for the reason the runner's own comment gives for `|emnist`:
   namespace collisions between cells sharing `(d2, attack)` are silent and are only ever found
   afterwards.
10. The six existing cells are not re-run and `results/comparability_cells/summary.json`'s existing
    keys are not rewritten; the resume path appends.
11. `analyze_comparability.py` keeps its refusal to print any pooled number unless all four published
    contrasts reproduce, and keeps printing each leg's own n.
