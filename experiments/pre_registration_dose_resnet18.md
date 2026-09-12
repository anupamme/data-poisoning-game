# Pre-registration: second-ARCHITECTURE replication of the flagship negative (ResNet18, Mode S, Krum)

**Frozen before any ResNet18 ASR under Mode S exists.** `results/dose_resnet18/` does not exist at the
time this file is committed. `experiments/run_dose_resnet18.py` refuses to start until this file is
git-committed and its `PREREG_COMMIT` is set to that hash, and it refuses again if the prospective
channel measurement in §2 does not say `ELIGIBLE`.

## 1. What is being tested, and why it is not the FEMNIST arm again

The paper's central negative result lives in a single cell: Mode S into Krum under model-scaling on
CIFAR-10, where the upstream ladder changes Krum's decision in most rounds, changes the adversarial mass
Krum admits in exactly 0% of rounds, and does not move suppression. Every claim the paper makes about
disturbance of a downstream statistic being causally irrelevant to suppression traces back to that cell.

The existing replication (`experiments/run_dose_femnist.py`) re-ran that cell with **two factors changed
at once**: EMNIST-byclass instead of CIFAR-10 *and* `simple_cnn` instead of `cifar_cnn`. It is the
cheapest honest test and it came back flat (Δ = −0.017, n = 3), but it cannot say which factor mattered,
and `simple_cnn` and `cifar_cnn` are two configurations of the same shallow-convnet family. The paper's
wording has been narrowed accordingly, to *"a second dataset and a differently configured CNN."*

This arm changes **one factor**. The dataset, `N` = 10, `K` = 5, `f` = 0.2, α = 0.5, the round count, the
attack and the defense are all held at the frozen CIFAR-10 values, and only the architecture moves — to
a different family: `resnet18`, an 18-layer residual network with skip connections and GroupNorm
(`fl_core/models.py:73-79`, the repo's single `resnet18` configuration: torchvision ResNet18 with a
CIFAR stem, `conv1` 3×3 stride 1, `maxpool` replaced by `Identity`, and every `BatchNorm2d` replaced by
`GroupNorm(8)` — BatchNorm's cross-sample statistics are not well defined under non-IID federated
averaging, which is why every ResNet18 arm in this repo uses the same builder). **A flat result here is
attributable to architecture alone**, and the phrase *"a second architecture"* is earned by this arm or
by nothing.

**How the cell was chosen, which is not how the FEMNIST cell was chosen.** On FEMNIST a payoff matrix
already existed, and exactly one of its cells cleared both eligibility gates, so the cell was selected by
a table rather than by us. **No artifact in `results/` contains both `resnet18` and `krum`**, so there is
no such table here. The cell is therefore not selected at all: it is *the* flagship cell, carried over
unchanged, and the question of whether standalone Krum suppresses model-scaling on this architecture is
not answered anywhere on disk. That question is the gate in §4, scored first, and its answer can void
this arm.

## 2. The premise was measured prospectively, before this rule was written

`experiments/measure_admission_resnet18.py` measured the channels on ResNet18 with **no ASR computed
anywhere** and **no model trained per rung** — one raw update stack per (seed, round) is shared by all
thirteen rungs. Seeds 42/43/44 × 6 rounds. From `results/resnet18_admission.json`:

| κ | ResNet18 decision change | ResNet18 admission change | cifar_cnn decision | cifar_cnn admission |
|---|---|---|---|---|
| 0.0 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.5 | 0.333 | 0.000 | 0.533 | 0.000 |
| 1.0 | 0.611 | 0.000 | 0.800 | 0.000 |
| 2.0 | **0.556** | **0.000** | 0.733 | 0.000 |

Verdict emitted by the script: **`ELIGIBLE`** — the decision is disturbed and the admitted adversarial
mass is not, which is the premise the `cifar_cnn` flagship arm rests on. Same dataset on both sides of
that table, unlike the FEMNIST comparison, so it is a clean architecture contrast.

`c_adv` is exactly `1.000000000000` at every Mode-S rung, max deviation from 1 = `0.00e+00`. The
adversary's own coefficient is untouched, so the attenuation channel is closed by construction on this
architecture too.

**One deviation from the `cifar_cnn` measurement, disclosed here rather than discovered later.** The
*adversarial coefficient share* under Mode S is constant across rungs only to **`1.855e-06`**, which is
**above** the imported `SHARE_TOL = 1e-6`; `cifar_cnn`'s own spread is `3.481e-07` and FEMNIST's is
`3.926e-07`, both below it. `SHARE_TOL` was **not** changed — widening a tolerance to admit an arm is
exactly the move a pre-registration exists to prevent. Instead a second, substantive discriminator was
added and both are printed and stored in the artifact:

- The share sequence is **non-monotone** in κ (`0.253333333`, `0.253332558`, `0.253334413`,
  `0.253332777`). A dose confound is monotone in its dial by construction — that is what a dose is, and
  the confounded `dose` family exhibits it (`0.2533 → 0.2646 → 0.2711 → 0.2731`, monotone). Float32
  read-back noise is not monotone.
- The deviation is four to five orders of magnitude below the smallest signal the instrument must
  resolve: Mode S `1.855e-06`, against the reported `dose` confound at `1.981e-02` (10,680× larger) and
  Mode A's intended payload dial at `6.514e-01` (351,146× larger).
- The tolerance is **absolute**, and the share is a ratio whose denominator sums float32 coefficients
  over every parameter, so its read-back noise grows with model size. `resnet18` has roughly two orders
  of magnitude more parameters than `cifar_cnn`, on which the tolerance was calibrated.

So the assertion pinned by construction (`c_adv ≡ 1`, exact) holds bit-exactly, and the derived
quantity misses an absolute threshold by a factor of 1.9 in a direction that does not track the dial.
**This is reported as a disclosed deviation, not as a passing check**, and it will be reported that way
in the paper if the arm is reported at all.

Two further measurement facts, for the record: **0 of 18 rounds were non-finite** (the FEMNIST
measurement lost 4 of 18 to float32 overflow under plain FedAvg; ResNet18 with GroupNorm did not
overflow within 6 rounds), and 12 of 72 Mode-S rung-rounds were degenerate — 0 adversaries or fewer
than 2 benign clients in the round's draw — against 24/120 (20.0%) for `cifar_cnn` and 8/56 (14.3%) for
FEMNIST. That is the `K`=5-of-`N`=10 participant draw, not the architecture.

**Why this matters for falsifiability:** the ladder demonstrably *does* disturb Krum's decision on
ResNet18 and demonstrably does *not* change the admitted adversarial mass. Had the decision change come
out near zero, a flat ASR curve would have carried no information at all — that is the `cos_krum`
failure mode the paper already discloses — and this arm would have been reported as **ineligible** and
not run. Note that `cos_krum` is flat at `0.000/0.000` at every rung on ResNet18 too, so that failure
mode is not hypothetical on this architecture.

## 3. The primary rule

On **Δ = mean ASR(κ=2) − mean ASR(κ=0)**, n = 3 seeds, scored against the `cifar_cnn` Krum arm's
Δ = **−0.026** and the suite's existing `EQUIV_MARGIN = 0.15`. **No new constant is introduced.** Both
comparison Δs — `cifar_cnn`'s and the FEMNIST arm's **−0.017** — are recomputed from their own per-seed
rows at run time, never transcribed, and **both legs of every Δ are recomputed in the same call** so a
subtrahend can never come from a different seed count than its minuend.

**The two comparison Δs do not carry this arm's n, and the difference is stated rather than glossed.**
`cifar_cnn`'s **−0.026** is an n = 5 Δ (seeds 42–46, both legs); the FEMNIST arm's **−0.017** is n = 3
(seeds 42–44, both legs), the same three seeds as here. Each Δ is a within-arm contrast at equal n on
its own two legs, which is what makes it a Δ at all; across arms the n's differ and no cross-arm
difference of Δs is taken. The comparison is to the ±0.15 margin, not to −0.026.

| outcome | verdict |
|---|---|
| \|Δ\| < 0.15 | **FLAGSHIP NEGATIVE REPLICATED** on a second architecture *family*. Statistic disturbance is causally irrelevant to suppression here too. |
| Δ > +0.15 | **THE NEGATIVE IS ARCHITECTURE-SPECIFIC.** Statistic disturbance does move suppression on ResNet18. The paper's central claim is scoped to the shallow-CNN family and must say so in the body, not in a limitation. |
| Δ < −0.15 | Attenuation-side fall. **INDETERMINATE**, reported as such and **not** scored in our favour. |

**Δ is refused, not caveated, if the two endpoints were scored on different seed sets.** A Δ across
unequal seed sets is not a within-arm contrast.

The wording is fixed now. A replication will be reported as *"replicated on a second architecture
family"* and **never** as *"generalizes"*, and never as evidence about datasets — this arm holds the
dataset fixed and therefore says nothing about datasets. Two architectures are not
"architecture-independent".

## 4. The gate, scored first and on its own

**Standalone Krum must suppress model-scaling on ResNet18 at usable accuracy**: mean ASR at κ=0 below
`SUPPRESS_ASR = 0.5` with mean clean accuracy at or above `ACC_FLOOR = 0.35`. Neither is a new constant:
0.5 is the same threshold the FEMNIST arm's eligibility rule was written against
(`run_dose_femnist.py:13`), where it decided which cells were admissible at all, and `ACC_FLOOR` is
imported from `run_targeted_dose`.

**If the gate fails the arm is VOID, NOT NEGATIVE.** There is then no suppression for the ladder to
preserve or to lose, the primary rule of §3 does not apply, κ=2 is not run, and the arm licenses **no
sentence about architecture** — in particular it must not be written up as "the negative replicates."
A void arm is reported as void, in the paper, next to the arms that were not void.

`doseS_kappa0.0` returns the update list unwrapped, so the κ=0 rung **is** Krum alone, and its runs are
stored rather than discarded: `run_one` reseeds torch and numpy from its arguments, so the run is the
same computation whether it was invoked by the gate or by the main loop, and the resume logic will not
repeat it. This is the one place where this arm's design differs from the FEMNIST arm's *harness check*,
and the reason is §1: there is no published figure for this cell on this architecture to check the
harness against, so the κ=0 rung is a scored pre-registered branch instead of an eyeball comparison.

**Accuracy gate, separately:** every rung must hold mean clean accuracy ≥ `ACC_FLOOR = 0.35`. If any
rung fails, the cell is uninterpretable and **no verdict stands** — a low ASR at collapsed accuracy is
not suppression. **This gate is live, not a formality.** The nearest existing artifact,
`results/cifar10_mix_ratio_sweep_resnet18/summary.json`, runs `resnet18` at exactly this regime and
reaches 0.662–0.766 clean accuracy in **14 of its 15 runs** — but its fifteenth
(`NC10_rep90`, seed 45) **collapsed to 0.153**, below the floor. So this architecture does sometimes
collapse here, at a rate of about 1 run in 15. Those runs also used norm-clip/reputation mixture
policies rather than Krum, so they bound the architecture's trainability at this regime and say nothing
about Krum's behaviour on it.

## 5. Declared limitations, before the run rather than after

- **Two rungs, not four, and therefore NO trend test.** This arm runs the endpoints κ ∈ {0, 2} only
  (ρ = 1.00 and 54.60). The FEMNIST arm ran four rungs so that a Jonckheere–Terpstra test could be
  scored across them; **there is no secondary test in this arm**, and its two rungs must never be
  displayed or described as a four-rung ladder. The reason is measured, not estimated: `resnet18` costs
  1.63–1.72 h/run at this regime (`results/cifar10_mix_ratio_sweep_resnet18/summary.json` records
  `wall_time_s` 5884–6205 over 15 runs, mean 6039 s, at exactly `N`=10, `K`=5, 50 rounds, CIFAR-10,
  `resnet18`), so two rungs × 3 seeds ≈ 10 h and four rungs ≈ 20 h. The restriction is printed by the
  runner on every invocation.
- **n = 3, fixed before the first run.** The same three seeds the FEMNIST replication froze, so the two
  replication arms are directly comparable. **No optional stopping**: no interim look, no extension, no
  seed added after seeing a result. If the run is interrupted, the n reached is reported as the n
  reached.
- **One cell, one attack, one architecture.** This is a replication of one cell, not a breadth claim.
  It does not widen `N`, `K`, `f`, α, the round count or the attack menu; it says nothing about
  datasets; and it does not touch the sign-reversal cell.
- **The share deviation of §2 stands as a disclosed deviation** whatever the outcome. It is not
  retroactively reclassified as a pass if the arm comes back flat.
- **The arm is an instrument, not a defense.** Mode S reads adversary identity to pin the adversarial
  coefficient share; no deployable defense knows which clients are adversarial. Nothing here is
  proposed for deployment.

## 6. Non-negotiables

1. **No frozen artifact is modified.** `results/admission_measurement.json`, `results/targeted_dose/`,
   `results/dose_femnist/`, `results/dose_response/`, `results/dose_replication/` and
   `results/cifar10_mix_ratio_sweep_resnet18/` are read-only here. This arm writes only
   `results/dose_resnet18/summary.json` and nothing else.
2. **`run_one` is imported, not copied.** `experiments/run_targeted_dose.py:run_one` takes
   `dataset`/`model` arguments whose defaults are the frozen CIFAR-10 configuration, so the two suites
   compute the same thing by construction rather than by inspection. `KAPPAS`, `ACC_FLOOR`,
   `EQUIV_MARGIN` and `TOL` are imported for the same reason, and the arm's rung list is asserted at
   import time to be a subset of the frozen ladder.
3. **Every comparison number is recomputed from per-seed rows at run time**, including both comparison
   Δs. Nothing is transcribed.
4. **Both outcomes are written above and neither is renegotiated afterwards.** If this refutes the
   flagship negative it goes in the body as a refutation with this rule quoted, in the same place a
   replication would have gone. If it is void it is reported as void in that same place.
