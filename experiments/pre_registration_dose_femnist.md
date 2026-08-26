# Pre-registration: second-dataset replication of the flagship negative (FEMNIST, Mode S, Krum)

**Frozen before any FEMNIST ASR under Mode S exists.** `results/dose_femnist/` does not exist at the
time this file is committed. `experiments/run_dose_femnist.py` refuses to start until this file is
git-committed and its `PREREG_COMMIT` is set to that hash.

## 1. What is being tested, and why it is the one cell worth running

The paper's central negative result lives in a single cell: Mode S into Krum under model-scaling on
CIFAR-10, where the upstream ladder changes Krum's decision in 80% of rounds, changes the adversarial
mass Krum admits in exactly 0% of rounds, and does not move suppression. Every claim the paper makes
about disturbance of a downstream statistic being causally irrelevant to suppression traces back to
that cell. If it is a property of CIFAR-10 with `cifar_cnn` rather than of the mechanism, the paper's
central claim is scoped much more narrowly than it currently says.

This arm re-runs **the same cell on a different dataset and a different architecture at once**:
EMNIST-byclass (62 classes, 1×28×28) with `simple_cnn` instead of CIFAR-10 with `cifar_cnn`.

**It is the only eligible cell on FEMNIST.** A Mode-S arm is interpretable only if standalone `d2`
genuinely suppresses the attack at usable accuracy — ASR < 0.5 at mean clean accuracy ≥
`ACC_FLOOR = 0.35` — because otherwise there is no suppression to preserve or to lose and the ladder
measures nothing. From `results/femnist/payoff_results.json`:

| cell | accuracy | ASR | eligible |
|---|---|---|---|
| `model_scaling` → `krum` | 0.767 | **0.027** | ✅ |
| `backdoor_pixel` → `krum` | 0.660 | 0.660 | ✗ no suppression |
| `model_scaling` → `fedavg`, `multi_krum` | 0.041 | — | ✗ accuracy collapsed |
| `model_scaling` → every remaining defense | — | ≥ 0.657 | ✗ no suppression |

Exactly one cell clears both gates, and it is the flagship cell. That is the reason this arm is the
one new experiment of the round rather than one of several.

## 2. The premise was measured prospectively, before this rule was written

`experiments/measure_admission_femnist.py` measured the channels on FEMNIST with **no ASR computed
anywhere** and no model trained per rung. From `results/femnist_admission.json`:

| κ | FEMNIST decision change | FEMNIST admission change | CIFAR-10 decision | CIFAR-10 admission |
|---|---|---|---|---|
| 0.0 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.5 | 0.643 | 0.000 | 0.533 | 0.000 |
| 1.0 | 0.714 | 0.000 | 0.800 | 0.000 |
| 2.0 | 0.643 | 0.000 | 0.733 | 0.000 |

`c_adv` is exactly `1.000000000000` at every Mode-S rung (max deviation from 1: `0.00e+00`), so the
adversary's own coefficient is untouched and the attenuation channel is closed by construction on this
dataset too. The adversarial coefficient share is constant across rungs to `3.9e-07`, which is float32
norm read-back noise and not a confound (the frozen CIFAR-10 measurement's own spread is `3.5e-07`).

**Why this matters for falsifiability:** the ladder demonstrably *does* disturb Krum's decision on
FEMNIST and demonstrably does *not* change the admitted adversarial mass. Had the decision change come
out near zero, a flat ASR curve would have carried no information at all — that is the `cos_krum`
failure mode the paper already discloses — and this arm would have been reported as **ineligible** and
not run. The premise is therefore not something a flat result can be attributed to after the fact.

Non-finite rounds: 4 of 18 (seed 42, rounds 2–5) were excluded from the channel measurement, because
`measure_admission.py` advances its server with plain FedAvg so that every rung sees the same raw
update stack, and plain FedAvg does not filter a scaling adversary, so the global model overflows
float32 on FEMNIST/`simple_cnn`. **This ASR arm does not inherit that**: here `d2 = krum` aggregates,
and Krum discards the scaled update instead of averaging it in.

## 3. The primary rule

On **Δ = mean ASR(κ=2) − mean ASR(κ=0)**, n = 3 seeds, scored against the CIFAR-10 Krum arm's
published Δ = **−0.026** and the suite's existing `EQUIV_MARGIN = 0.15`. **No new constant is
introduced.** The CIFAR-10 Δ is recomputed from `results/targeted_dose/summary.json` per seed at run
time, never transcribed.

| outcome | verdict |
|---|---|
| \|Δ\| < 0.15 | **FLAGSHIP NEGATIVE REPLICATED** on a second dataset *and* a second architecture. Statistic disturbance is causally irrelevant to suppression here too. |
| Δ > +0.15 | **THE NEGATIVE IS DATASET- OR ARCHITECTURE-SPECIFIC.** Statistic disturbance does move suppression on FEMNIST. The paper's central claim is scoped to CIFAR-10 and must say so in the body, not in a limitation. |
| Δ < −0.15 | Attenuation-side fall. **INDETERMINATE**, reported as such and **not** scored in our favour. |

The wording is fixed now: a replication will be reported as *"replicated on a second dataset and
architecture"* and never as *"generalizes"*. One additional dataset does not license the second word.

## 4. Secondary and gates

- **Secondary (trend):** Jonckheere–Terpstra across all four rungs κ ∈ {0, 0.5, 1, 2}. This is why the
  arm runs 4 rungs and not 2. A monotone rise refutes the flat prediction with the attenuation channel
  already closed by Mode S's construction (`c_adv ≡ 1`), so it could not be explained away as a
  payload effect.
- **Accuracy gate:** every rung must hold mean clean accuracy ≥ `ACC_FLOOR = 0.35`. If any rung fails,
  the cell is uninterpretable and **no verdict stands** — a low ASR at collapsed accuracy is not
  suppression.
- **Harness sanity, explicitly not a hypothesis test:** `doseS_kappa0.0` returns the update list
  unwrapped, so the κ=0 rung *is* Krum alone and should land near the payoff matrix's
  0.027 @ 0.767. That figure is read from `results/femnist/payoff_results.json` at run time. A gap of a
  few points is seed noise (one seed against a 3-trial mean at different seeds); a gap of tens of
  points means the two harnesses disagree about the protocol and **the arm is void rather than
  interesting**.

## 5. Declared limitations, before the run rather than after

- **n = 3, not 5.** The FEMNIST payoff matrix ran 3 trials (`experiments/run_femnist.py`) and 4 rungs ×
  3 seeds × ~55 min/run is already ≈ 11 h. This is a budget decision made before any result exists,
  and it will be disclosed in the paper immediately next to the result. With n = 3 the JT test is
  weak; a null from it is not evidence of flatness, and will not be reported as such.
- **One dataset, one architecture, one attack, one defense.** This is a replication of one cell, not a
  breadth claim. Two datasets are not "domain-independent".
- **The arm is an instrument, not a defense.** Mode S reads adversary identity to pin `c_adv = 1`. No
  deployable defense knows which clients are adversarial. Nothing here is proposed for deployment.
- **No identity rung is imported.** `run_dose_replication.py` could import κ=0 from the Round-11
  CIFAR-10 ladder; there is no FEMNIST Round-11 ladder, so κ=0 is run in-suite. 12 new runs, not 9.

## 6. Non-negotiables

1. **No frozen artifact is modified.** `results/admission_measurement.json`,
   `results/targeted_dose/`, `results/dose_response/`, `results/dose_replication/` and
   `results/femnist/` are read-only here. This arm writes only `results/dose_femnist/summary.json`.
2. **`run_one` is imported, not copied.** `experiments/run_targeted_dose.py:run_one` takes
   `dataset`/`model` arguments whose defaults are the frozen CIFAR-10 configuration, so the two suites
   compute the same thing by construction rather than by inspection.
3. **Every comparison number is recomputed from per-seed rows at run time**, including the CIFAR-10 Δ
   and the standalone payoff figure. Nothing is transcribed.
4. **Both outcomes are written above and neither is renegotiated afterwards.** If this refutes the
   flagship negative it goes in the body as a refutation with this rule quoted, in the same place a
   replication would have gone.
