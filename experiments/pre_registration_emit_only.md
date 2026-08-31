# Pre-registration: the emit-only control (statistic channel closed, Krum, CIFAR-10)

This is the fourth and last cell of the 2x2 factorial that separates the two channels a Mode-S
rescaling opens into a selector. It is the mirror of `pre_registration_score_only.md` and reuses
that arm's design, gates and thresholds wherever they apply, so the two cells differ in exactly one
thing: which channel is closed.

**Nothing below may be edited after `results/emit_only/` exists.** The prediction is derived from the
three already-frozen cells and is stated as a number, so it can fail.

---

## 1. What is being tested, and why the existing arms do not already test it

Mode S moves two things at once inside a selector, and the paper has so far separated them
*algebraically* rather than by running the separation:

* **The statistic channel.** The rescaling changes Krum's pairwise distances, so it changes *which*
  client Krum selects. Measured: the decision changes in 0.733 of rounds at kappa=2.
* **The magnitude channel.** The selected client's update enters the model scaled by its own
  coefficient, so it changes *what that client contributes*. Measured as the emitted-aggregate
  displacement, `d_agg = 0.892` uncontrolled.

`run_score_only_control.py` closes the second and keeps the first. The decomposition of the
remainder into "re-selection 0.986 / rescaling 0.973" in `measure_admission.py` is **algebraic, a
displacement bookkeeping identity, and is not a causal decomposition**: it does not tell us what ASR
would have been had only the magnitude channel been open. That counterfactual is a run, not an
identity, and this arm is that run.

The four cells, with the three frozen ones filled in from
`results/targeted_dose/summary.json` and `results/score_only/summary.json` at the time of writing:

| | emitted update **unchanged** | emitted update **rescaled** |
|---|---|---|
| **statistic/decision unchanged** | kappa=0 identity: ASR 0.0618 | **this arm: unrun** |
| **statistic/decision changed** | score-only: ASR 0.0385, dASR **-0.0233** | full Mode S: ASR 0.0357, dASR **-0.0261** |

---

## 2. The intervention, precisely, and why it cannot silently change the published arms

`generic_compose(..., emit_only=True)` (`experiments/run_all_compositions.py`):

```
scored, emitted = (updates, transformed)          # score_only is the same line with these swapped
sel, _ = krum_selection(flatten(scored), cosine=False)
return server.aggregate([emitted[sel]], method="fedavg")
```

Krum **scores the untransformed stack**, so its decision is pinned to the identity rung's decision
and the statistic channel is closed; it then **aggregates the transformed selected update**, so the
only channel still open is the magnitude and geometry of what reaches the model.

`emit_only` defaults to `False` and `score_only and emit_only` raises, so:

* every pre-factorial call site is byte-for-byte the computation it always was, and
* `run_targeted_dose.py --harness-check` and `run_score_only_control.py --harness-check` must both
  still pass **bit-identically**. If either does not, this arm does not run.

Both flags raise for non-selectors. A weighted averager or a coordinate-wise order statistic emits
no single selected client, so neither "the selected update" nor its rescaling is defined for them.
**This bounds the factorial to Krum and cos_krum and is a scope limit of the design, not an
oversight** (see §6).

---

## 3. What is run

`krum` / `committed_scaling` / CIFAR-10 / `cifar_cnn`, `kappa` in {0, 0.5, 1, 2}, seeds 42-46,
N=10, K=5, f=0.2, alpha=0.5, 50 rounds. Identical to the score-only arm in every respect.

`ACC_FLOOR = 0.35` and `EQUIV_MARGIN = 0.15` are **imported from `run_targeted_dose`**, not restated,
so they cannot drift.

**The kappa=0 rung is imported, not re-run.** At kappa=0 `apply_d1_transform` returns the update list
unwrapped -- verified to return the *same list object* -- so `scored` and `emitted` are the same
object and emit-only at the identity IS Krum alone, bit-identically. `--harness-check` verifies this
at one seed against the published identity rung before any new cell is written. New compute is
therefore 3 rungs x 5 seeds = **15 runs**.

---

## 4. The primary rule

Scored **seed-matched** against the imported kappa=0 rung, as the published cells are, at the
**kappa=2 endpoint** -- matched to the endpoint both other cells report, so the three dASR values are
the same contrast.

Write `dASR_EO` for this arm's kappa=0 -> kappa=2 seed-matched mean change, and take the two frozen
values as given: `dASR_full = -0.0261`, `dASR_SO = -0.0233`.

**Prediction, in two parts, both frozen here:**

1. **The magnitude channel alone is inert:** `|dASR_EO| < 0.05`.
   Point prediction under exact additivity: `dASR_full - dASR_SO = ` **-0.0029**.
2. **The channels are separable:** the additivity residual
   `|dASR_full - (dASR_SO + dASR_EO)| < 0.05`.

**Refuted if `|dASR_EO| >= 0.05`.** That outcome says the magnitude channel carries non-negligible
effect on its own, and therefore that the score-only control's -0.0233 landing so close to the
uncontrolled -0.0261 was partly coincidence rather than evidence that the decision channel is the
one doing the work. **If that happens the paper must re-attribute, in the body, and say so.** The
existing score-only claim ("instrumented against two independent confounds") would have to weaken to
a claim about the two channels being individually small but not independently established.

**Refuted the other way, and this is the more interesting failure:** if the additivity residual is
`>= 0.05` while `|dASR_EO| < 0.05`, the two channels interact, and no decomposition of the
uncontrolled arm into per-channel contributions is licensed at all -- including the algebraic one
this arm was built to replace.

---

## 5. Gates, in force before any outcome is read

* **Accuracy floor.** Any rung with mean clean accuracy `< 0.35` is **uninformative, not evidence of
  suppression**: a collapsed model has low ASR for the wrong reason. This risk is materially higher
  for this arm than for the other three, because the emitted update is the *rescaled* one and at
  kappa=2 (rho = 54.6) the selected benign client's coefficient can exceed 1 by a large factor for
  50 consecutive rounds. **If kappa=2 fails the floor, the primary contrast is reported at the
  highest gate-passing rung and the substitution is disclosed as forced by the floor, not chosen
  after seeing ASR.** The other three cells all clear it comfortably (minimum observed accuracy
  across them: 0.5415).
* **No rung substitution on any other basis.** kappa=2 is the primary unless the floor removes it.
* **No seed addition.** n=5, seeds 42-46, matching the arm it is compared against.
* **Secondary, reported but not the rule:** the per-seed sign count, and the same
  Jonckheere-Terpstra monotone-trend test across kappa the score-only arm reports.

---

## 6. Declared limitations, before the run rather than after

1. **Low power.** The per-seed dASR values on the frozen cells span -0.114 to +0.042, which is wide
   relative to a -0.003 point prediction. At n=5 a *pass* of the additivity test is **weak evidence
   of separability**; only a large `|dASR_EO|` is decisive. This asymmetry is stated here so the
   result cannot later be reported as a confirmation it is not powered to be.
2. **Selectors only.** The factorial is defined for Krum and cos_krum. It does **not** extend to the
   coordinate-median, RFA, trimmed-mean or reputation arms without a decision-transplant
   generalization that has not been written, so the factorial is a within-family result on one
   selector, never a cross-defense one.
3. **Krum's admission floor is unchanged.** Krum admits no adversary at any rung, so `d_adm` is zero
   here for the same floor reason as the published arm. This does not bound the factorial's
   conclusion, whose outcome is dASR, not d_adm -- but nothing about the admission channel is
   learned from this arm.
4. **One attack, one dataset, one architecture.** committed_scaling / CIFAR-10 / cifar_cnn. The
   factorial is a clean decomposition of one cell, not a generalization across cells.
5. **The kappa=0 rung is imported.** So the primary contrast shares its baseline with the other two
   cells by construction. That is what makes the three dASR values comparable, and it also means a
   defect in the baseline propagates to all three.

---

## 7. Non-negotiables

* This file is git-committed and `PREREG_COMMIT` in `run_emit_only_control.py` is set to that hash
  **before** `results/emit_only/` exists. The runner refuses to start otherwise.
* `--harness-check` passes before any new cell is written.
* Both existing `--harness-check`s still pass bit-identically.
* The row entering the paper is emitted by `build_channel_table.py`, never typed by hand.
* **The verdict is reported whichever way it goes**, in the body, with these thresholds and this
  commit hash beside it.
