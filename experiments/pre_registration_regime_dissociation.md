# Pre-registration: does the DESIGN DISAGREEMENT survive an adaptive adversary, and survive scale?

**Frozen before any result in `results/regime_dissociation/` exists.** That directory does not exist at the
time this file is committed. `experiments/run_regime_dissociation.py` refuses to start until this file is
git-committed and its `PREREG_COMMIT` is set to that hash.

## 1. What is being tested, and why the existing evidence does not test it

The paper's flagship dissociation is a **disagreement between two evaluation designs on one cell**
(`tab:comparability`): downstream `d_2` = `coord_median`, attack = committed pixel, κ from 0.0 to 2.0
(ρ from 1 to 54.60). Its artifact is `results/reversal_seed_topup/summary.json` (prereg
`pre_registration_reversal_seed_topup.md` @ `1efe5ec`), endpoint rungs only, seeds 42--61.

| | confounded ladder `dose_kappa` | instrument `doseS_kappa` |
|---|---|---|
| runner | `experiments/run_dose_response.py` (prereg `e711a95`) | `experiments/run_targeted_dose.py` (prereg `5130cec`) |
| adversarial coefficient | drawn from the same spread as benign | pinned `c=1.0` exactly |
| ΔASR (n=20, as published) | **−0.273**, `[−0.334, −0.213]` | **+0.125**, `[+0.095, +0.155]` |

Two questions are unanswered anywhere on disk, and this arm is written because a review named both.

**(i) Adaptivity.** `experiments/run_criterion_aware_adversary.py` already runs a Kerckhoffs adversary that
solves a joint evasion constraint, 6 conditions × 5 seeds. But it runs it against **`("foolsgold", "rfa")`**
(`run_criterion_aware_adversary.py:60`) and scores whether **the screen's PASS verdict** survives. It has
never been pointed at the reversal cell. **No artifact contains both a criterion-aware adversary and
`coord_median`.** So the design disagreement has never faced an adversary that adapts to the defense it
meets.

**(ii) Scale and participation.** `results/cifar10_100clients_rich/` runs N=100, K=20 — but it stores game
equilibria, and `main.tex:2637` labels that whole line of work *"Residual material from an earlier framing…
Nothing in the screen, the identification result or the intervention evidence depends on it."*
`results/heterogeneity_sweep/summary.json` varies α but tests **the screen's verdicts**, not the
dissociation. Neither reproduces a dissociation outside N=10, K=5.

**This arm changes one thing per regime and nothing else.** Regime **A** changes the adversary. Regime **B**
changes N and K. The cell, the attack, the defense, the κ grid, f=0.2, the round count and `cifar_cnn` are
held at the frozen values in both.

## 2. The primary estimand, and its power settled from measured variance

The primary estimand is **the disagreement itself**, paired by seed:

> **D = Δ(confounded) − Δ(instrument)**, per seed, 95% two-sided t-interval at n=3.

**D's κ=0 leg cancels exactly.** Per seed *s*,
`D_s = [ASR_conf(κ2,s) − ASR(κ0,s)] − [ASR_inst(κ2,s) − ASR(κ0,s)] = ASR_conf(κ2,s) − ASR_inst(κ2,s)`,
because the two designs **share the κ=0 rung**: at κ=0 both `dose_kappa0.0` and `doseS_kappa0.0` return the
update list unwrapped, so both rungs are `coord_median` alone. This is not an assumption. It is measured in
the frozen artifact — ASR and accuracy are **bit-identical between the two designs at κ=0 for all 15
top-up seeds** — and it is existing house practice, stated in that artifact's own
`identity_rung_provenance`: *"kappa=0 is COMPUTED ONCE per new seed and shared by both legs."* This arm
does the same and proves it the same way (§4.3).

### Why D and not either Δ: the three variances are measured, not assumed

Recomputed from `results/reversal_seed_topup/summary.json` and its `published_n5_verdict`, with the n=3
two-sided 95% half-width `4.303·sd/√3 = 2.484·sd`:

| estimand | measured mean | measured sd | n=3 half-width | excludes zero at n=3? |
|---|---|---|---|---|
| Δ(confounded) | −0.272 (n=5) | 0.1205 | 0.299 | **no** |
| Δ(instrument) | +0.098 (n=5) | 0.0646 | 0.161 | **no** |
| **D** | **−0.408 (n=15)** | **0.1094** | **0.272** | **yes** |

So a rule of the form *"both intervals exclude zero"* is a rule this arm is arithmetically guaranteed to
fail, and pre-registering it would be pre-registering a null. D is roughly three times either Δ at
comparable sd, which is what makes it testable where neither Δ is. The n=15 D subsampled to its **first
three seeds** gives mean −0.427, sd 0.109, half-width 0.271 — still excluding zero, so the power claim
survives the seed count this arm actually runs.

**One thing this table cannot settle**: those sds are measured in the *frozen* regime, and a new regime may
be noisier. The one available check is reassuring rather than conclusive — the marginal per-seed κ=0 ASR
spread at N=100 (0.264--0.695, from §3) is **comparable to** the frozen N=10 spread (0.285--0.734), so
there is no evidence that scale inflates it. If D's realized interval at n=3 contains zero, that is
reported as a failure to reproduce at n=3, **not** as evidence of absence, and it is **not topped up**
(§5, no optional stopping).

| outcome | verdict |
|---|---|
| D's interval excludes zero and D < 0 | **THE DESIGN DISAGREEMENT REPRODUCES** in this regime. The confounded design reads lower than the instrument on the same cell and the same seeds. |
| D's interval contains zero | **NOT REPRODUCED AT n=3.** Reported with the point estimate and the interval printed, as a failure at this seed count. |
| D's interval excludes zero and D > 0 | **REVERSED.** The confounded design reads *higher*. This contradicts the paper's mechanism and goes in the body as a contradiction, in the same place a reproduction would have gone. |

**Secondary, descriptive, and explicitly not inferential**: each Δ's point estimate and sign, with the
n=3 half-widths above printed beside them. **No equivalence claim is made anywhere in this arm.** The phrase
*"no evidence of a practically meaningful change"* is defined at `supplementary.tex:441` against the paired
95% interval lying inside the margin, and n=3 cannot earn it; `EQUIV_MARGIN = 0.15` is **not** used here.

**Every number is recomputed from per-seed rows at run time, and both legs of every Δ in the same call**, so
a subtrahend can never come from a different seed count than its minuend. Nothing in this document is
transcribed into code.

## 3. The two regimes, and the prospective evidence that each is eligible

Both regimes: `d_2` = `coord_median`, attack = committed pixel (`backdoor_pixel`), κ ∈ {0.0, 2.0},
**seeds 42/43/44**, f = 0.2, 50 rounds, `cifar_cnn`, both ladders.

### Regime A — adaptive adversary (N=10, K=5, α=0.5)

The regime is the frozen one; only the adversary changes. `criterion_aware_updates`
(`run_criterion_aware_adversary.py:132`) is **imported, not reimplemented**, and it **substitutes for**
`attack.manipulate_update` rather than stacking on it — the existing runner makes them mutually exclusive
branches at its `:218`–`:225`, and matching that exactly is what keeps the two suites computing the same
adversary. Adversaries still train on poisoned data in both branches; only the update-space manipulation
differs.

**The adversary is calibrated to the defense it faces, over eps AND decorrelate.** `decorrelate` restricts
each adversary to a disjoint coordinate block, which is an **anti-FoolsGold** device
(`run_criterion_aware_adversary.py:64`, *"decorrelate controls FoolsGold evasion"*). The downstream defense
here is `coord_median`, not FoolsGold, and against a coordinate-wise median a lone adversary in its own
block is plausibly *weaker*, not stronger. Importing `decorrelate=True` because it was the argmax against
FoolsGold→RFA would import a knob tuned against a different defense. So the grid is
**eps ∈ {1, 2, 4} × decorrelate ∈ {True, False}**, at **κ=0, seed 42 only**, argmax mean ASR, then
**frozen** for every rung and seed of the arm. Six calibration runs; the winning one **is** the κ=0/seed-42
leg and is not recomputed.

**Threat-model scope, stated now.** The adversary adapts to `d_2` and to the benign update distribution. It
does **not** anticipate `d_1`, and that is deliberate rather than a weakness of the adversary: Mode S reads
adversary identity to pin the adversarial coefficient, so it is an **instrument, not a deployable defense**,
and no real adversary can adapt to a stage that does not exist in deployment. This arm therefore tests the
strongest adversary the threat model admits, and the sentence *"adapts to the entire composed pipeline"*
must never be written about it without this clause.

**Eligibility.** In the existing sweep the reference `committed_pixel` sits at **0.0452** and the strongest
evader at **0.1668** (`ca_eps1_decorr`, max 0.2682), so the adversary demonstrably has room to move ASR on
*some* composition. Whether it moves it on `coord_median` is exactly what §4.1 scores first, and a
calibration whose argmax does not beat the plain backdoor is a **VOID** regime, not a negative.

### Regime B — scale and partial participation (N=100, K=20, α=0.5)

**This is a deviation from the approved plan, which specified α=0.1, and it is a deviation forced by
measurement.** Both reasons are prospective and both are on disk:

1. **α=0.1 is uninterpretable at n=3.** From `results/heterogeneity_sweep/summary.json` (N=10, K=5, f=0.2,
   50 rounds, `cifar_cnn`, seeds 42/43/44), with `coord_median` downstream under committed pixel:

   | cell | mean acc | mean ASR | per-seed ASR |
   |---|---|---|---|
   | `foolsgold_then_coord_median` α=0.1 | **0.391** | 0.433 | 0.479 / 0.632 / 0.188 |
   | `reputation_then_coord_median` α=0.1 | **0.465** | 0.548 | 0.650 / 0.555 / 0.438 |
   | `foolsgold_then_coord_median` α=1.0 | 0.736 | 0.133 | 0.179 / 0.092 / 0.129 |

   At α=0.1 the per-seed ASR range is **0.44** against **0.087** at α=1.0, and accuracy sits 0.04–0.11
   above `ACC_FLOOR = 0.35` rather than the 0.74–0.79 of the milder α. A variance that large against a D of
   −0.4 is not resolvable at n=3, and widening the seed count after looking would be optional stopping.
2. **α=0.5 is the α with prospective eligibility at the target scale.**
   `results/cifar10_100clients_rich/seed_4*/payoff_results.json` runs **exactly** N=100, K=20, 50 rounds,
   α=0.5, f=0.2, `cifar_cnn` (`run_cifar10_100clients_rich.py:17`, `:44`–`:46`) and records, over 5 seeds,
   `backdoor_pixel` × `coord_median` — which is the κ=0 rung by construction:

   > mean ASR **0.500** (per seed 0.426 / 0.553 / 0.563 / 0.695 / 0.264), mean accuracy **0.473**

   against **0.490** (n=15) for the frozen N=10 κ=0 rung. So at N=100 the rung the ladder has to move sits
   at essentially the same ASR, with headroom in both directions, at comparable per-seed spread, and clears
   the accuracy floor.

So Regime B changes **N and K** — 20% participation per round against 50% at N=10, which is the realistic
partial-participation regime — and holds α at the frozen 0.5. **The heterogeneity dimension is not dropped**;
it is answered by `results/heterogeneity_sweep/` on its own terms, cited as separate regime evidence, and
this arm makes no claim about α.

**Those five numbers are prospective evidence of eligibility and are NOT a leg of this arm.** They come from
a different runner with a different RNG consumption order, so they are not bit-comparable to a
`doseS_kappa0.0` run and are never substituted for one. Regime B runs its own κ=0 rung.

## 4. Gates and diagnostics, scored before the primary

**4.1 Regime A's calibration gate (a real gate).** The argmax calibration condition must reach mean ASR at
κ=0 **strictly above** the plain `committed_pixel` κ=0 ASR at the same seed. If none does, the
criterion-aware construction does not strengthen the adversary against `coord_median`, **Regime A is VOID,
not negative**, κ=2 is not run, and the regime licenses **no sentence about adaptivity**. A void regime is
reported as void, in the paper, next to the regimes that were not void.

**4.2 Accuracy gate (a real gate), per regime and per rung.** Mean clean accuracy ≥ `ACC_FLOOR = 0.35`,
imported from `run_targeted_dose.py:110`. If any rung fails, that regime is uninterpretable and **no verdict
stands** — a low ASR at collapsed accuracy is not suppression. Live, not a formality: the N=100 prospective
accuracy is 0.473.

**4.3 Identity-rung equality (a real gate).** `--harness-check` must show `dose_kappa0.0` and
`doseS_kappa0.0` agree to `TOL = 1e-6` at one seed per regime. The κ=0 rung is **shared by proof, not by
assumption**; if the check fails, the two designs do not share an anchor and the arm stops.

**4.4 Base-rung ASR is reported as a DIAGNOSTIC and is deliberately NOT a gate.** The plan proposed gating
on `SUPPRESS_ASR = 0.5` (`run_dose_resnet18.py:133`). That threshold is dropped here, before any result, and
the reason is arithmetic: the frozen κ=0 rung of the very cell being reproduced is **0.490** (n=15) and the
N=100 prospective figure is **0.500**, both at or above the threshold. A gate that voids this arm would
equally void the published flagship, so applying it here would be applying a standard the paper's own
central cell does not meet. The base-rung ASR is printed with every verdict instead, and the honest reading
it licenses — that `coord_median` alone does not strongly suppress this backdoor, which the frozen data
already shows — is stated wherever this arm is reported. **No sentence from this arm may say the base rung
suppresses the attack.**

## 5. Declared limitations, before the run rather than after

- **n = 3, fixed now. No optional stopping**: no interim look, no extension, no seed added after seeing a
  result. If the run is interrupted, the n reached is reported as the n reached.
- **Two rungs, not four, and therefore no trend test.** κ ∈ {0, 2} only. These two rungs must never be
  displayed or described as a four-rung ladder, and no trend statistic is computed in this arm. This matches
  the frozen artifact's own `endpoint_only` restriction on the same cell.
- **Neither Δ is powered to exclude zero at n=3** (§2, measured). Only D is tested. No sentence in the paper
  may present a Δ from this arm as significantly different from zero.
- **One cell, one attack.** This is a reproduction of one dissociation in two regimes, not a breadth claim.
  It says nothing about other defenses, other attacks, other datasets, other architectures, or α.
- **Regime A's adversary does not anticipate `d_1`** (§3), and the reason is that `d_1` is an instrument.
- **Regime B confounds N with K.** Participation rises from 5-of-10 to 20-of-100 in one step, so a result is
  attributable to *scale-with-partial-participation* jointly and **not** to client count alone. Stated now
  because it cannot be unconfounded after the fact at this budget.
- **Both regimes are `cifar_cnn` on CIFAR-10.** No architecture or dataset claim.
- **Mode S is an instrument, not a defense.** It reads adversary identity to pin the adversarial
  coefficient. Nothing here is proposed for deployment.

## 6. Non-negotiables

1. **No frozen artifact is modified.** `results/targeted_dose/`, `results/dose_response/`,
   `results/reversal_seed_topup/`, `results/dose_femnist/`, `results/dose_resnet18/`,
   `results/criterion_aware_adversary/`, `results/heterogeneity_sweep/`,
   `results/cifar10_100clients_rich/` and `results/comparability_six_cells.json` are **read-only** here.
   This arm writes `results/regime_dissociation/summary.json` and nothing else, and creates its directory
   in `__main__` and never at import time.
2. **Both `run_one`s are imported, not copied.** `run_targeted_dose.run_one` and `run_dose_response.run_one`
   gain three optional arguments (`fl_config`, `alpha`, `ca`) whose defaults are the frozen configuration,
   so the suites compute the same thing by construction rather than by inspection. `KAPPAS`, `ACC_FLOOR`,
   `TOL`, `ADV_FRACTION`, `d1_name`, `dial` and `rho` are imported. Neither existing `PREREG_COMMIT` is
   touched.
3. **The default path is proved unchanged, not assumed.** `--harness-check` runs on both modified runners
   before this arm's first run, and one existing cell is re-run at one seed with default arguments and
   compared **by value** to its stored row in the frozen artifact. An md5 is not accepted as this proof.
4. **The attack hook is asserted to fire.** At the first adversarial update of the first round the runner
   asserts the tensor changed — via `manipulate_update` when `ca is None`, via `criterion_aware_updates`
   otherwise — and that **exactly one** of the two ran. It fails loudly; it does not warn. A composition
   arm in this repo has already come out bit-identical to another because a manipulation hook was silently
   never called. **Superseded by Amendment 1 below**, which corrects the `ca is None` half of this
   sentence: for this arm's attack the correct assertion is that the update is **unchanged**.
5. **Every comparison number is recomputed from per-seed rows at run time.** Nothing is transcribed,
   including the frozen Δs, the measured sds of §2 and the prospective figures of §3.
6. **All outcomes above are written now and none is renegotiated afterwards.** A failure to reproduce, a
   reversal, a void regime and a failed gate each have a named home in the paper, and it is the same place a
   reproduction would have gone.

# AMENDMENT 1, before `results/regime_dissociation/` exists and before any run of this arm

**Disclosed as an edit rather than folded in silently, and it corrects a false statement in the freeze.**
§6.4 as committed says the runner "asserts the tensor changed — via `manipulate_update` when `ca is None`".
**For this arm's attack that assertion is false by design, so as written §6.4 demands an invariant the
correct configuration cannot satisfy.**

`AttackStrategy.manipulate_update` is `return update` — the identity (`attacks/attack_strategies.py:61`–
`:63`) — and `BackdoorPixelAttack` **does not override it** (`:94`): the committed pixel backdoor lives
entirely in `poison_dataset`, so nothing happens to the update in update space. Only
`ModelScalingAttack` overrides it (`:135`–`:137`, an elementwise ×10). Both this arm's regimes use
`backdoor_pixel`. A runner that asserted "changed" would therefore abort on the first round of every
`ca is None` leg of both regimes.

**The corrected assertion is attack-conditional, and it is strictly stronger than the one it replaces**, not
a weakening to make the run pass:

> At the first adversarial update of the first round, the adversarial update must have **changed if and
> only if** the configuration says it should — changed when `ca` is set, changed when the attack class
> overrides `AttackStrategy.manipulate_update`, and **unchanged** when it inherits the identity. Exactly
> one of the two paths runs. It fails loudly; it does not warn.

The predicate is `type(atk).manipulate_update is not AttackStrategy.manipulate_update`, not a name test.

**Why stronger.** The defect §6.4 exists to catch is a hook that is never called, and in this repo it has
already happened: a cross-distribution arm came out bit-identical to another because `manipulate_update` was
never invoked. The original assertion could not have caught it here, because pixel's hook is a **no-op
anyway** — "nothing changed" is both the failure symptom and the correct behaviour, which is precisely how
those two arms silently collapsed onto one computation. Pinning the *expected* behaviour per attack instead
catches four distinct failures, all four verified by deliberately breaking each one before this arm ran:

| broken deliberately | caught |
|---|---|
| an attack that overrides `manipulate_update` but returns the update unmodified | yes |
| an attack that inherits the identity yet whose update comes back modified | yes |
| `criterion_aware_updates` returning its input unchanged | yes |
| — and both correct configurations (`backdoor_pixel` unchanged, `model_scaling` changed) pass | yes |

**Nothing else moves.** No estimand, gate, threshold, regime, seed count, outcome or limitation in §§1–5
changes, and no result exists at the time of this amendment — which is the condition that makes an amendment
legitimate rather than a renegotiation. The assertion lives in `experiments/adversary_hook.py`, shared by
both runners so the two ladders cannot differ in the adversary they apply, and it is on for the first round
of **every** run of this arm.
