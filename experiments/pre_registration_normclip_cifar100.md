# Pre-registration: a composition-level replication of the dissociation on CIFAR-100, with a real defense as the dose (Round 72)

**Status: frozen before any write to `results/normclip_cifar100_admission.json` or
`results/normclip_cifar100/`.** The premise conditions, the τ rule, the seed list, the endpoint grid, the
primary interval and the demotion clause below are reproduced verbatim in
`experiments/measure_admission_normclip_cifar100.py` and `experiments/run_normclip_cifar100.py`, the
second of which refuses to start until this file is committed and its hash is recorded in
`PREREG_COMMIT`.

## Why this run exists

The twelfth review of this paper names one result as the single most valuable thing we could add:

> *a true composition-level replication of the causal dissociation on a second dataset/model with
> nonzero baseline adversarial admission.*

That is three requirements in one experiment — second dataset, composition level, nonzero baseline
admission — and every existing arm in the paper fails at least one of them:

| existing arm | second dataset | composition level | baseline admission, level (nonzero rounds / adversary rounds) |
|---|---|---|---|
| Krum × `committed_scaling`, CIFAR-10 (flagship) | no | no | **0.0000 (0/12), at the floor** |
| `coord_median` × `committed_pixel`, CIFAR-10 | no | no | 0.2855 (12/12) |
| Reputation × `committed_scaling`, CIFAR-10 | no | no | 0.0014 (12/12) |
| `cos_krum` × `committed_pixel`, CIFAR-10 | no | no | **0.0833 (1/12), near the floor** |
| Krum, EMNIST-byclass / `simple_cnn` | yes | no | **0.0000 (0/12), at the floor** |
| Krum, CIFAR-10 / `resnet18` | no (model only) | no | **0.0000 (0/15), at the floor** |
| 18-pair composition suite, CIFAR-10 | no | yes | **never measured** |

Every level in that column is recomputed by `build_channel_table.py` from
`results/admission_measurement.json`, `results/femnist_admission.json` and
`results/resnet18_admission.json`, conditioned on the rounds in which an adversary was actually sampled.

The facts in bold are the review's objection and it is correct on them: every arm at nonzero baseline
admission is a single-defense arm on CIFAR-10, every arm on a second dataset or architecture is at the
floor, and **no composition-level arm anywhere in this paper has its baseline admission measured at
all**. This document registers the one arm that can be all three at once.

**One measured fact cuts against the review's stronger reading**, and it is recorded here because it was
found while scoping this run rather than after seeing its result. The review's objection (c) is that the
headline result is an artifact of *Krum never admitting malicious clients*. Holding Krum, the dataset and
the model fixed and varying only the attack:

| aggregator | `committed_scaling` | `committed_pixel` |
|---|---|---|
| `krum` | 0.0000 (0/12) | **0.3333 (4/12)** |
| `cos_krum` | 0.0833 (1/12) | 0.0833 (1/12) |
| `reputation` | 0.0014 (12/12) | 0.2507 (12/12) |
| `coord_median` | 0.1564 (12/12) | 0.2855 (12/12) |

So the floor is a property of the **attack**, not of Krum: a model-scaling attack makes an adversarial
update a norm outlier, which is what Krum is built to reject. That weakens the review's generalization
but it does not remove the objection, because the paper's adjudicating arm does commit to scaling. This
run is registered anyway.

## The arm

`norm_clip` → `coord_median`, **CIFAR-100 / `cifar_cnn`**, attack `committed_pixel`.

Configuration, pinned to the values four existing runners already agree on
(`run_cell7_seed_topup.py:111`, `run_cifar100_ne3_br.py:88`, `run_cifar100.py:24`,
`run_cross_distribution_compositions.py:54`): `N = 10` clients, `K = 5` sampled per round, adversary
fraction `f = 0.2`, Dirichlet `alpha = 0.5`, 50 rounds, `cifar_cnn` with `num_classes = 100`.

Each factor is chosen for a stated reason, before any number exists:

- **`coord_median` as the downstream defense** because its measured CIFAR-10 baseline admission is the
  highest of the four arms in the paper (0.2855, nonzero in all 12 adversary rounds), so the premise has
  the best available chance of transferring. This is the review's *nonzero baseline* requirement.
- **`committed_pixel` as the attack** because that is the cell where `coord_median`'s baseline is
  highest, and because a pixel backdoor does not make an adversarial update a norm outlier, so a norm
  clip is least likely to attenuate the payload. That is the property Mode S buys with an oracle and
  this arm has to earn by measurement.
- **CIFAR-100** because it is a second dataset with an existing anchor for this exact pair, and because
  the paper's other CIFAR-100 work (cell 7 of the comparability suite) fixes the harness.
- **`norm_clip` as the dose** for the reason in the next section.

### Why `norm_clip` makes this a test of the theory rather than of an instrument

The review's other standing complaint is that Theorem 9's rescaling class is a restricted laboratory and
that Mode S is an oracle instrument, so neither speaks to deployed defenses. `norm_clip` answers both at
once. `run_all_compositions.py`'s `apply_d1_transform` implements it as, verbatim:

```python
    elif d1_name == "norm_clip":
        # Clip each update to L2 norm bound tau
        clipped = []
        for u in updates:
            flat = torch.cat([u[k].flatten().float() for k in keys])
            norm = flat.norm().item()
            scale = min(1.0, tau / max(norm, 1e-8))
            clipped.append({k: u[k] * scale for k in keys})
        return clipped
```

That is exactly `T(u_i) = c_i u_i` with `c_i = min(1, τ/‖u_i‖) > 0`: a member of Theorem 9's positive
per-client rescaling class, and a standard, deployed, **oracle-free** FL defense. It needs no adversary
mask, unlike every Mode-S, Mode-A and Mode-M rung in the paper. So the dose here is a real defense's own
hyperparameter, not a synthetic coefficient vector, and the class the review calls restricted is shown
to contain something people actually run.

This is also why the identity control is exact rather than approximate. At any τ above every client
norm, `min(1.0, τ/‖u‖)` returns the float `1.0` and `u[k] * 1.0` is bit-identical to `u[k]`, so the
τ = ∞ leg is `coord_median` standalone to the last bit, not merely close to it.

### The published CIFAR-10 anchor, recomputed rather than transcribed

From `results/all_compositions/summary.json`, key `pairs/norm_clip_then_coord_median/committed_pixel`,
seeds 42–44 in order:

| leg | per-seed ASR | mean | n |
|---|---|---|---|
| `norm_clip` → `coord_median`, CIFAR-10, `committed_pixel` | 0.39456 / 0.28667 / 0.44878 | **0.37667** | 3 |
| the same pair under `committed_scaling` (for context, not compared) | 0.44278 / 0.39144 / 0.46600 | 0.43341 | 3 |

**This anchor is n = 3 and the new arm is n = 5, and the two n's are never merged.** Wherever the
replication is claimed, both counts are printed. The anchor exists to show the pair is not new to the
paper, not to be differenced against CIFAR-100.

## Two stages, and stage 2 is gated on stage 1

This is the structure `measure_admission_femnist.py` established: an arm on a new dataset is only
informative if its *premise* holds there, so the premise is measured first and separately, before any
ASR for this arm exists anywhere on disk.

### Stage 1 — the eligibility measurement

`experiments/measure_admission_normclip_cifar100.py`. It **imports** `flatten`, `argmedian`,
`admission`, `aggregates`, `rel_disp`, `DECISION_KEY`, `ADMISSION_KEY` and `SHARE_TOL` from
`experiments/measure_admission.py`, so CIFAR-10 and CIFAR-100 cannot be measured by two different
definitions of admission. It does **not** edit that file, and it carries its own round loop because
`measure()` fixes `tau = 5.0` at `measure_admission.py:148` and this arm needs τ to vary per rung.

3 seeds × 6 rounds. Writes only `results/normclip_cifar100_admission.json`. Roughly 1 hour.

**The premise, stated as pass/fail before any number is seen. All three must pass.**

1. **Nonzero baseline admission.** `coord_median`'s baseline adversarial argmedian fraction on
   CIFAR-100, averaged over the rounds in which an adversary is actually sampled, is **≥ 0.05**, and is
   nonzero in **at least half** of those rounds. The 0.05 level floor is set here, in advance, at
   roughly a sixth of CIFAR-10's measured 0.2855 for this aggregator and attack, and a third of
   FEMNIST's 0.1029 for this aggregator under the other attack.

   **Both halves of the condition are load-bearing, and `cos_krum` is why.** Its level reads 0.0833,
   which clears a 0.05 threshold, but it is nonzero in only **1 of 12** rounds: the level is an average
   over rounds and a single admitting round can carry it. A level test alone would therefore certify an
   arm this paper already classifies as near the floor. Failing either half means the arm cannot answer
   the review's question, and stage 2 does not run.
2. **The identity rung displaces nothing.** At τ = ∞ every `c_i` equals exactly 1.0 and every measured
   channel displacement — aggregate, decision, admission, influence — is exactly 0. This is the guard
   against the failure mode this repository has already hit once, where a manipulation hook is never
   called and the run passes silently looking like a null result.
3. **The adversarial coefficient share does not move.** `measure_admission.py:152` reads the
   coefficients back from the *transformed* norms rather than assuming them, and the adversarial share
   of total coefficient mass is reported **at every rung**. If it moves by more than the frozen
   `SHARE_TOL = 1e-6` across rungs, the clip is attenuating adversaries, `Δ_c ≠ 0`, and the arm is
   declared **confounded on this dataset**. Stage 2 does not run.

Condition 3 is the one most likely to fail, and it is the honest risk of using a real defense instead of
an oracle: a clip *may* attenuate the payload, in which case this arm measures the attenuation channel
rather than the statistic channel and cannot speak to the dissociation. We do not know which it is. That
is why the premise is a separate stage with its own artifact.

**Stage 1's verdict is written into its artifact, not only printed.** Round 69 established that this
repository already has one check whose verdict dict the caller discards, leaving a paper sentence
witnessed only by a run's stdout.

### τ is fixed by a rule now, and by a number only after stage 1

τ* = the **median client update norm** measured in stage 1, so that roughly half of the clients clip.
The rule is frozen in this document; the value it yields is whatever stage 1 measures.

**No τ may be re-chosen after any ASR for this arm exists.** If stage 1's median gives a τ* that turns
out to clip nobody or everybody, that is reported as the measured outcome of the frozen rule, and the
rule is not amended.

### Stage 2 — the ASR ladder, only if all three premise conditions pass

`experiments/run_normclip_cifar100.py`. Writes only `results/normclip_cifar100/`.

- Two rungs, endpoints only: τ = ∞ (identity, `coord_median` standalone, bit-exact) and τ = τ*.
- **Seeds 42–46**, matching the published pair's seed family. n = 5 on both legs.
- Both legs run together in the same invocation, from the same seeded state; neither is transcribed
  from an existing directory.
- Primary estimand: the paired per-seed difference `ASR_s(τ*) − ASR_s(∞)`, with a 95% Student-t
  interval on 4 degrees of freedom (t₄ = 2.776). Secondary: clean accuracy, same interval.
- `--harness-check` runs first and reproduces the published CIFAR-10 pair value at a shared seed to
  < 1e-9, **with the verdict dict written into the artifact**.
- Progress is counted from `per_seed` rows in the artifact, never from the run log's `[i/N]`, which is a
  plan position that counts resumed-and-skipped runs and can go backwards across restarts.

2 rungs × 5 seeds = 10 runs at the 636–1407 s per run measured in
`results/comparability_cell7_run.log`, so 3–4 hours. Total for both stages, roughly 5 hours.

## The demotion clause

This is what licenses running the arm at all, and it is written before any result exists.

Three verdicts, stated here as literals so the runner reports one of them rather than composing prose
after seeing the numbers. This follows the precedent at `run_dose_femnist_topup.py:133`, which already
carries a frozen verdict string of exactly this shape.

- **`PREMISE FAILED: THE COMPOSITION-LEVEL REPLICATION IS NOT ESTABLISHED ON CIFAR-100.`** Stage 1's
  condition 1 or 2 fails. The paper says the composition-level replication was attempted and did not
  carry, **in the abstract and in §5**, not in a footnote. The review's objection stands unanswered and
  we say so.
- **`ARM CONFOUNDED ON THIS DATASET: THE CLIP ATTENUATES ADVERSARIES, SO Δ_c ≠ 0.`** Stage 1's
  condition 3 fails. Same disclosure, plus the measured coefficient shares, and the arm is reported as
  measuring the attenuation channel rather than the statistic channel. Stage 2 does not run.
- **`DISSOCIATION REVERSES ON CIFAR-100 AT THE COMPOSITION LEVEL.`** Stage 2 runs and admission moves
  with the dose, or suppression moves while admission does not. **That is this round's headline,
  reported against us**, in the abstract and in Figure 1, at the cost of the paper's dissociation claim.

## What a null result does not license

If suppression is flat at unchanged admission, that is a dissociation on **one pair, one dataset, one
attack, two rungs, n = 5**. It is not a general law, it is not evidence that `norm_clip` is safe, and no
sentence in the paper may say that it is. The paper's existing (L1) scope statement applies unchanged.

## What this arm does not address, stated before it runs

- **The architecture does not change.** CIFAR-100 keeps `cifar_cnn`; only the classifier head's
  dimension moves, 10 → 100. So this is a replication across dataset and label space, not across
  backbone, and it is a *partial* answer to the review's scope objection. §5 and the response letter say
  so in those words. It does not retire that objection.
- **`N = 10`, `K = 5`, `f = 0.2`, `alpha = 0.5` are untouched.** The narrow-configuration objection
  survives this round.
- **FEMNIST was considered as the second dataset and rejected**, not on cost, and not for the reason
  that first looked decisive. It would have been the stronger answer to the scope objection, since
  `simple_cnn` changes the architecture as well as the data, and `results/femnist_admission.json`
  already exists. It also measures four aggregators, not one, and `coord_median`'s baseline admission
  there is **nonzero and would pass this document's premise**: 0.1029, nonzero in all 12 adversary
  rounds, against krum 0.0000 (0/12), `cos_krum` 0.0000 (0/12) and reputation 0.0002 (12/12).

  The disqualifying fact is the **attack**, not the baseline. That file has **no `committed_pixel` doseS
  rows at all**; every FEMNIST measurement is under `committed_scaling`. And `committed_scaling` is
  exactly the attack a norm clip is expected to attenuate, because a scaling attack *is* a norm outlier
  — so a FEMNIST `norm_clip` arm would very likely fail premise condition 3, measure the attenuation
  channel rather than the statistic channel, and answer a different question than the review asked.
  CIFAR-100 admits `committed_pixel`, where the payload is not a norm outlier.

  Two consequences are recorded rather than absorbed. First, **FEMNIST's `coord_median` result is
  positive evidence that this premise transfers across datasets for this aggregator** — 0.1029 there
  against 0.1564 on CIFAR-10 under the same attack — which is a reason to expect stage 1 to pass and is
  stated before stage 1 runs. Second, FEMNIST's two zero rows show the floor is **not** unique to
  CIFAR-10 Krum, so stage 1 remains a real gate and not a formality.

## Non-negotiables

- **No write to any existing `results/` directory.** Only
  `results/normclip_cifar100_admission.json` and `results/normclip_cifar100/` are created.
- **No edit to `measure_admission.py`, `run_all_compositions.py`, `fl_core/*`, or any existing runner,
  analyzer or pre-registration.** Stage 1 imports the shared definitions and carries its own loop.
- **No revision of any frozen threshold, seed list, interval convention or verdict** elsewhere in the
  paper. Where a number here would move a frozen verdict, both print and the frozen one stays labelled
  as frozen.
- **No amendment to this document after any artifact for this arm exists.** An amendment before that
  point is appended, dated, and never a rewrite.

## Amendment 1 (2026-09-17), appended before either runner existed and before any artifact for this arm

Three facts surfaced while writing `measure_admission_normclip_cifar100.py` and
`run_normclip_cifar100.py`. None of them moves a number, threshold, seed list, interval convention,
premise, demotion clause or verdict literal. Nothing above is rewritten; this section is appended.

### 1. The two existing instruments train a different number of local epochs, and this arm's dose is the first one that does not survive the difference

`measure_admission.py:132` trains **one** local epoch per client per round
(`cl[cid].train(srv.global_model, 1, 0.01, 64)`). Every ASR ladder in the paper trains
`FLConfig.local_epochs = 2` (`config.py:9`, and `run_all_compositions.py:67` /
`run_comparability_cells.py:57` both instantiate `FLConfig` without overriding it). So in this
repository every channel premise has always been measured at 1 epoch and every ASR ladder run at 2.
That is a pre-existing property of the instrument pair and it is harmless for the dose families,
because kappa is a **scale-free** dispersion dial: the same kappa means the same dispersion whatever
the update norms are.

`norm_clip`'s dose is not scale-free. It is a **norm threshold**, so a tau measured on 1-epoch updates
does not transfer to a 2-epoch ladder -- it would clip very nearly every client, and stage 1's premise
checks would then certify a regime stage 2 does not run in. This arm is the first in the paper whose
dose has units.

Resolution, fixed here before any number exists:

- Stage 1 traverses the **same seeds and the same rounds at both epoch counts**. The 1-epoch pass is
  the like-for-like comparison against the frozen CIFAR-10 baselines, which are all 1-epoch numbers.
  The 2-epoch pass is the configuration stage 2 actually runs.
- **Premise 1 is evaluated on the 1-epoch pass**, because that is the configuration in which the
  0.05 floor was calibrated and in which all seven existing rows in the table above were measured.
  The 2-epoch number is printed beside it, and **if the two passes disagree on either half of
  condition 1, the arm is NOT eligible.** That is the conservative direction and it is fixed now
  rather than after the numbers are seen.
- **tau\* is the median client update norm of the 2-epoch pass**, because the clip must be a threshold
  on the updates it will actually face. Both medians are recorded in the artifact. The rule frozen
  above -- the median client update norm measured in stage 1, roughly half the clients clipping -- is
  unchanged; this fixes which of stage 1's two passes instantiates it, which the frozen text left
  open, and it is the only reading under which the rule's stated purpose holds in stage 2.
- **Premises 2 and 3 are evaluated at both epoch counts and must pass at both.**

### 2. The tau = infinity leg already has a published bit-exact anchor, so the harness check gains a second leg

Cell 7 of the comparability suite already publishes `coord_median` standalone on
CIFAR-100 / `cifar_cnn` / `committed_pixel` at **seeds 42--46**, under key
`dose_kappa0.0_then_coord_median|committed_pixel|cifar100` in
`results/comparability_cells/summary.json`, mean ASR **0.7033939393939395**. `doseS_kappa0.0` carries
the identical mean, which is the identity rung's bit-identity already realized on this dataset.

At kappa = 0 `apply_d1_transform` returns the update list unwrapped, and at tau = infinity it returns
`{k: u[k] * 1.0}`. Multiplying a float32 tensor by exactly `1.0` is bit-identical, and `CifarCNN` uses
GroupNorm (`fl_core/models.py:26`) so no update entry is an integer buffer that promotion could
disturb. So this arm's identity leg **is** that published rung, computed again.

Stage 2 still runs all ten runs as registered and transcribes nothing. What changes is only the check:
`--harness-check` gains a **second** check, reproducing that published CIFAR-100 row at seed 42 to
< 1e-9. The registered CIFAR-10 check is unchanged and remains check 1. After the ladder completes,
all five identity seeds are compared against the five published values and the comparison is written
into the artifact. A mismatch is reported, not absorbed.

### 3. One loop reproduces both anchors, and that is asserted rather than assumed

The two published anchors were produced by two loops that differ in one argument:
`run_all_compositions.run_one` constructs `FederatedServer` without a clean holdout, and
`run_comparability_cells.run_one` constructs it with `clean_holdout_dataset=Subset(td, range(100))`.
`clean_holdout_dataset` is read only by `_fltrust` (`fl_core/federated.py:225`), so it is inert for
`coord_median`: it consumes no RNG and touches no update. Stage 2 therefore carries **one** loop and
reproduces both anchors with it. If either check fails, that inference was wrong and the runner
refuses to start.
