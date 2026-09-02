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
