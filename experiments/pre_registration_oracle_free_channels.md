# Pre-registration: can the channel dissociation be produced without reading adversary identity, on a cell with no admission floor?

**Frozen before any result in `results/oracle_free_channels/` or `results/oracle_free_admission.json`
exists.** Neither path exists at the time this file is committed.
`experiments/run_oracle_free_channels.py` and `experiments/measure_admission_oracle_free.py` refuse to
start until this file is git-committed and their `PREREG_COMMIT` is set to that hash.

## 1. What is being tested, and why nothing on disk tests it

A review states one condition for raising its score, and this arm is written against that sentence
rather than against a list: *one stronger, non-floor-effect causal intervention with substantially more
seeds and less dependence on oracle adversary identity.* It asks for two legs:

* **leg 1** -- the statistic's decision is disturbed, the admitted adversarial mass is not, and ASR does
  **not** move;
* **leg 2** -- the statistic's decision is **pinned**, the admitted adversarial mass is not moved, and
  ASR **does** move.

Both legs exist in the paper and both are produced by instruments that read `adv_mask`. Every
`doseS_*`, `doseA_*` and `doseM_*` rung dispatches through the `targeted is not None` branch of
`apply_d1_transform`, which **raises** without `adv_mask`: *"refusing to run a targeted dose without
knowing which participants are adversarial."* So every published instance of either leg is an oracle
instance, and the paper says so in five places, including the ethics statement.

**Three things on disk look like they cover this and do not.**

1. **`app:normclip_cifar100`'s impossibility result** (`paper/main.tex:2051`) establishes that within the
   positive per-client rescaling family, coefficient-neutrality and informativeness are **mutually
   exclusive**: an oracle-free rescaling that leaves the adversary's coefficient alone is one that
   leaves everyone's coefficient alone, and therefore disturbs nothing. That is a scoped impossibility
   about one channel, not the experiment the review asks for, and it is what tells this arm where to
   look: the channel has to be closed **downstream of the coefficient**, for every client at once,
   rather than neutralized inside it.
2. **`results/channel_table.json`** already carries `score_only` and `emit_only` rows for Krum -- but
   every one of them is a `doseS_*` rung, so the transform is an oracle, and every one of them is on
   `committed_scaling`, where Krum's baseline admitted adversarial mass is **0.0000 in 0 of 12
   adversary rounds**. The review's other objection is exactly this: a zero that cannot fall is not
   evidence that a mechanism was preserved.
3. **`results/metric_swap/summary.json`** has `rfa_then_krum|committed_pixel` at n=3, which is
   oracle-free and off the floor -- but it is the **uncontrolled** composition, both channels open at
   once, which is the design the paper's own Proposition 6 says cannot identify anything.

**This arm changes exactly one thing about the published factorial: the transform stops being an
oracle.** The controls, the host aggregator, the attack, the dataset, the architecture, N, K, f, the
round count, the Dirichlet concentration and the accuracy floor are all held at their frozen values.

### Why `rfa` as the upstream stage, and why `score_only` is not an oracle

`apply_d1_transform`'s Type A branch runs the Weiszfeld iteration to a geometric-median estimate and
rescales each client by its final Weiszfeld weight. Those weights are computed **from the update stack
alone**. The branch never reads `adv_mask`, and this arm passes `adv_mask=None` into `generic_compose`
so that oracle-freeness is enforced by the code path rather than established by reading it: if any part
of the composition tried to consult adversary identity, `apply_d1_transform` would raise.

`score_only=True` makes Krum **score** the reweighted stack and **emit the selected client's original,
unreweighted update**. `emit_only=True` is the mirror: Krum scores the raw stack, so its decision is
pinned bit-identically to the identity arm's, and only what the selected client contributes can move.
Neither flag reads adversary identity. Both are already frozen, in
`experiments/pre_registration_score_only.md` and `experiments/pre_registration_emit_only.md`, and both
are defined for selectors alone, which is why the host is Krum and not a coordinate-wise statistic.

So `score_only` does not need coefficient-neutrality, and `main.tex:2051` does not bite: it closes the
magnitude channel **for every client simultaneously and downstream of the coefficient**, which is a
thing no member of the rescaling family can do to itself.

**What this does not answer, stated here so it cannot be claimed later.** The controls modify the
aggregator's internals. This arm is still an **instrument**, not a deployable defense: a deployed Krum
does not score one stack and emit from another. The review's objection that the *intervention* reads
adversary identity is answered; the objection that the instruments are not deployable is **not**, and
it stays disclosed as an open problem.

### Why this cell has no admission floor

Recomputed read-only from `results/admission_measurement.json`'s `per_round` rows, over the adversary
rounds of the frozen 5-seed x 3-round measurement:

| host cell | baseline `krum_admits_adv` | nonzero adversary rounds |
|---|---|---|
| `krum` under `committed_scaling` | **0.0000** | **0 of 12** |
| `krum` under `committed_pixel` | **0.3333** | **4 of 12** |

The same Krum, the same measurement, two attacks. Under the committed pixel backdoor Krum admits
adversarial mass in a third of the adversary rounds it sees, so admission here has somewhere to fall
and an unchanged admission is a measurement rather than a ceiling artifact. `committed_pixel` into
`krum` is therefore the host, and it is the cell `paper/main.tex:729` already prints.

## 2. The estimand, and the two verdicts frozen separately

Three arms, all at the same 20 seeds, paired by seed:

> **Delta_stat = ASR(rfa -> krum, score_only) - ASR(fedavg -> krum)**
> **Delta_mag  = ASR(rfa -> krum, emit_only)  - ASR(fedavg -> krum)**

each reported as a mean over seeds 42--61 with a paired two-sided 95% t-interval at n=20
(t = 2.093). `fedavg -> krum` is Krum standalone: `apply_d1_transform` returns the update list
unwrapped for `fedavg`, so the identity arm is the untransformed aggregator by construction and not by
approximation.

**Frozen predictions.** Under arm type GENERALIZATION (§4.1):

| leg | arm | frozen prediction |
|---|---|---|
| **1** | `score_only` | `\|Delta_stat\|` **inside** the frozen `EQUIV_MARGIN = 0.15`. The decision is disturbed, the admitted mass is not, and suppression does not move |
| **2** | `emit_only` | `Delta_mag` **nonzero**, its 95% interval **excluding zero**. The decision is pinned bit-identically and suppression moves anyway |

### Two verdicts per leg, and neither may be reported as the other

Round 68's regime arm cleared its pre-registered threshold in two regimes while reproducing the **sign**
in only one, and the one-line summary was misleading for exactly that reason. An interval can also sit
inside the margin **and** exclude zero. So both verdicts are computed for both legs, before any number
is seen:

| verdict | rule | reported as |
|---|---|---|
| **SIGN** | the paired 95% interval excludes zero | **ASR moves on this leg**, with its sign |
| **SIGN** | the interval contains zero | **not resolved at n=20.** Point estimate and interval printed. Not evidence of absence |
| **MARGIN** | `\|mean\| < EQUIV_MARGIN = 0.15` | **within the frozen equivalence margin** |
| **MARGIN** | `\|mean\| >= 0.15` | **outside the frozen margin**, with the mean and interval printed |

The MARGIN verdict is a comparison of a mean to a frozen constant and **not** the phrase
*"no evidence of a practically meaningful change"*, which `supplementary.tex:441` defines against the
stricter of the paper's two interval conventions. That phrase may not be applied to any leg of this arm.

**Verdict literals carry no mechanism clause.** Round 72's freeze fused a pass/fail label with a guess
about mechanism; the label was right and the guess was backwards in sign, so the untested half inherited
the tested half's authority. Here the runner emits `verdict_sign` and `verdict_margin` as labels only,
and writes any mechanism reading into a separate field explicitly labelled as the measurement.

**Every number is recomputed from per-seed rows at run time, and both legs of every Delta in the same
call**, so a subtrahend can never come from a different seed count, or a different arm, than its
minuend. Nothing in this document is transcribed into code.

### The branch that refutes us

If `|Delta_stat| >= 0.15` with an interval excluding zero, then in this cell the statistic channel
**does** carry suppression without an oracle. That contradicts the paper's headline reading, it is
reported **in the body as a contradiction**, in the same place a confirmation would have gone, and the
sentence *"the negative reproduces oracle-free"* may not be written under any circumstances.

If `Delta_mag`'s interval contains zero, leg 2 does **not** reproduce oracle-free, and that is reported
as a failure to resolve at n=20 with the point estimate. Two measurements already say why that branch is
live and they are recorded here rather than after the fact: `rfa_then_cos_krum|committed_pixel` is
0.2734 against `fedavg_then_cos_krum` at 0.3005, a difference of 0.027 well inside the margin, and
`foolsgold_then_krum|committed_pixel` collapses to mean accuracy 0.0994, below the floor and therefore
uninterpretable. If leg 2 does not resolve, this arm reports leg 1 alone and says so in one sentence.

## 3. The grid, the seed count, and the power available, measured not assumed

`d1 in {fedavg, rfa}`, `d2 = krum`, attack = `committed_pixel` (`backdoor_pixel`), CIFAR-10,
`cifar_cnn`, **seeds 42--61 (n=20)**, N=10, K=5, f=0.2, 50 rounds, Dirichlet alpha = 0.5,
`ACC_FLOOR = 0.35`, `EQUIV_MARGIN = 0.15`. Three arms: identity, `score_only`, `emit_only`.

**n=20, fixed now.** It is the top of the range the review named, it is the paper's existing 20-seed
grid (seeds 42--61, the grid `results/dose_seed_topup/` completed), and it is chosen **before** any
number on this cell is seen. **No optional stopping and no top-up clause.** If a leg's interval contains
zero at n=20 that is the reported result, and no seed is added to any arm after a number is seen.

### Why n=20 is powered here even though the baseline is bimodal

The marginal spread on this cell is large and the paper's own history says why that matters: cos_krum's
Round-11 identity rung was bimodal across seeds (0.003--0.864) and the FLAT prediction tested against it
never had the power the test presumed. So the power argument is made **from a paired spread, measured on
this exact cell**, rather than from a marginal one.

Read-only from `results/prospective_pilot/summary.json` (`krum|committed_pixel`, the identity) and
`results/metric_swap/summary.json` (`rfa_then_krum|committed_pixel`, the uncontrolled composition), at
their three shared seeds 42/43/44:

| quantity | value |
|---|---|
| identity ASR per seed | 0.7788 / 0.0390 / 0.9316 -- **marginal sd 0.4774** |
| uncontrolled ASR per seed | 0.9282 / 0.2629 / 0.9829 |
| **paired** differences | +0.1494 / +0.2239 / +0.0513 |
| **paired sd (ddof=1)** | **0.0865**, i.e. **5.5x smaller than the marginal sd** |
| implied 95% half-width at n=20 | **0.0405** |

Pairing by seed cancels most of the marginal variance because the seed sets the Dirichlet partition,
which is shared by both arms. At a paired sd of 0.0865 the n=20 half-width is 0.041, so both a sign
verdict and a margin verdict against 0.15 are resolvable, and at n=3 (half-width 0.220) neither would
be. **This diagnostic crosses two suites** -- the two rows were produced by `run_prospective_pilot.py`
and `run_metric_swap_baselines.py` -- so it is a power estimate and not a result, and it is not quoted
anywhere as an effect. It is also an estimate from the **uncontrolled** composition, whose paired spread
need not equal a controlled arm's.

## 4. Gates and diagnostics, scored before the primary

### 4.1 The prospective channel measurement, and the three-way rule stated before its numbers are read

`experiments/measure_admission_oracle_free.py` runs **first**, computes no ASR, trains no model per
rung, and writes `results/oracle_free_admission.json`. It measures, for `rfa` against the raw stack on
this cell: whether Krum's **selection** changes, whether Krum's **admitted adversarial mass** changes,
the baseline admission level with its nonzero-round count, and the aggregate displacement. It reuses
`measure()` from `measure_admission.py` with its own `rungs` list, exactly as
`measure_admission_mask.py` does, so no two families can be measured by two different definitions of
"the statistic". `results/admission_measurement.json` is read for comparison and **never written**.

`DECISION_FLOOR = 0.10` and `ADMISSION_FLAT = 0.05` are **imported from
`experiments/measure_admission_mask.py`**, not restated, so this arm's eligibility rule is literally the
constants Mode M was typed by:

| branch | rule | consequence |
|---|---|---|
| **INELIGIBLE** | decision change `< 0.10` | RFA barely disturbs Krum's decision, so a flat ASR curve carries no information. **The ladder does not run.** This is the cos_krum failure mode the paper already discloses |
| **GENERALIZATION** | decision change `>= 0.10` and admission change `<= 0.05` | The premise of the flagship arm holds **without an oracle**. The predictions of §2 are in force |
| **ADMISSION** | decision change `>= 0.10` and admission change `> 0.05` | RFA moves the admitted mass, so this is **not** an oracle-free analogue of the flagship arm and may not be reported as one. It tests the paper's own reading instead: if admission governs suppression, `Delta_stat` must move **with** admission. The ladder still runs, under that prediction, and the review's leg 1 is reported as **unanswered** |

The published expectation is that the decision moves: `apply_d1_transform`'s own docstring records
*"RFA disturbs Krum's selection 6/9"*. That is a reason to run the measurement, not a substitute for it.

**The identity rung is measured by the same code path**, as `d1 = fedavg`, so that the measurement's own
baseline is reproduced rather than assumed: at `fedavg` the transform returns the update list unwrapped,
so every coefficient must read back as exactly 1.0, the decision change must be exactly 0.000 and the
admission change must be exactly 0.000. A nonzero value there means the measurement is broken and no
number from it may be used.

**The gate is applied to the published configuration first.** Round 73's heterogeneity gate condemned
three new concentrations at a tolerance the already-published concentration also failed. So this script
prints the frozen `doseS`/Krum decision and admission figures from
`results/admission_measurement.json` beside its own, and if the rule as implemented would classify the
**published** arm INELIGIBLE, the verdict for this arm is `INDETERMINATE` and the instrument is fixed
before anything is condemned.

### 4.2 Accuracy gate (a real gate), per arm

Mean clean accuracy `>= ACC_FLOOR = 0.35`, imported from `run_targeted_dose.py:111` rather than
restated. If an arm fails, it is **VOID, not negative**: a low ASR at collapsed accuracy is not
suppression, and no verdict stands for that arm.

**This gate is live.** The identity cell's measured per-seed accuracy at seeds 42/43/44 is
0.3962 / 0.5922 / 0.4670 -- mean 0.4851, but a **single-seed minimum of 0.3962** against a floor of
0.35. Krum selects one client per round for 50 rounds, so a seed whose selected clients are poorly
representative can land near the floor. The gate is scored on the **arm mean**, as everywhere else in
the paper, and the per-seed minimum is reported beside it as a diagnostic so that a mean clearing the
floor over seeds that individually do not is visible rather than hidden.

### 4.3 The emit-only arm's two zeros are asserted with `==`, not assumed

`emit_only=True` scores the untransformed stack, so Krum's selection is the identity arm's selection by
construction, and admission cannot move. That is a claim about the shipped code path, and the standing
failure mode in this repository is a manipulation hook that is never called while the run passes
silently (`run_cross_distribution_compositions`' dead `manipulate_update`). So:

* the channel measurement records the `emit_only` selection **per round** against the raw-stack
  selection and asserts equality with `==` over every round, not a tolerance;
* the runner's `--harness-check` asserts that the `rfa` branch of `apply_d1_transform` was **entered**
  and returned an object distinct from its input, so an inert transform cannot pass as a null result.

If either assertion fails, the arm does not run and the failure is the result.

### 4.4 `--harness-check` runs first, and its verdict is written into the artifact

Three checks, in this order, before any new run:

1. **`d1_override=None` is bit-neutral.** One frozen cell of `results/targeted_dose/summary.json` is
   recomputed through the patched `run_targeted_dose.run_one` and must reproduce accuracy and ASR to
   `<= 1e-9`. This is the check that the one additive keyword this arm adds to shared code changed
   nothing, on the standard that runner's own docstring sets: *"every existing call site is
   bit-identical, which --harness-check proves rather than assumes."*
2. **The identity arm is Krum standalone, across suites.** `d1_override="fedavg"` at seed 42 must
   reproduce `results/prospective_pilot/summary.json`'s `krum|committed_pixel` row for that seed to
   `<= 1e-9`. This is the same cross-suite assertion `run_dose_response.py` makes for
   `("krum", "committed_scaling")` against the same pilot file, and it is a **blocking** gate: if the
   two suites do not share a code path on the identity, the identity arm's 20 runs are not comparable
   to any published baseline and that must surface before 16 hours are spent, not after.
3. **The transform applied.** `apply_d1_transform(ups, "rfa")` must return a list whose entries are
   **not** the input objects and whose per-client norm ratios are not all exactly 1.0.

Round 69 established that this repository's existing harness check returns a verdict dict its caller
discards, leaving the claim witnessed only by stdout while re-running costs hours. All three verdicts
are persisted into `results/oracle_free_channels/summary.json`.

## 5. What this arm cannot conclude

- **No general oracle-free claim from one cell.** One aggregator, one attack, one dataset, one upstream
  defense, one architecture, one concentration. The sentence *"the negative holds without an oracle"*
  may **not** be written. What the arm licenses is *"in this cell, and without reading adversary
  identity"*, and that phrasing is frozen here.
- **No deployability claim.** Both controls modify the aggregator's internals. `score_only` and
  `emit_only` are measurement devices, not defenses, and this arm does not narrow the gap the paper
  already discloses between an instrument and a deployable defense.
- **No claim about adaptive adversaries.** The attack is the committed pixel backdoor, non-adaptive by
  construction.
- **No pooling and no re-baselining.** The two Deltas share one identity arm and are reported
  separately, each with its own interval. Neither is differenced against the other, and no published
  verdict, threshold, seed list or interval convention is amended by this arm. The frozen
  `committed_scaling` channel rows stay exactly as published, with their 0.0000 baseline, printed beside
  this arm's non-floor baseline.
- **A null on leg 1 is not proof of a null.** `|Delta_stat| < 0.15` with an interval containing zero is
  reported as *within the frozen margin and not resolved in sign at n=20*, which is weaker than an
  equivalence claim and is not written as one.

## 6. Outputs and cost

- `results/oracle_free_admission.json` -- per-round decision, admission, baseline admission level and
  nonzero-round count, aggregate displacement, the identity-rung exactness check, the `emit_only`
  selection-equality assertion, the three-way arm type and its verdict, and the published-configuration
  cross-check of §4.1.
- `results/oracle_free_channels/summary.json` -- per-arm, per-seed accuracy and ASR; both verdicts for
  both legs; the accuracy gate with its per-seed minimum; the three harness-check verdicts; the frozen
  premise read from the channel artifact.

Neither path exists at commit time. **No existing `results/` directory is written to.**
`measure_admission.py`, `measure_admission_mask.py`, `run_all_compositions.py` and `run_dose_mask.py`
are imported, never edited. One additive keyword, `d1_override=None`, is added to
`run_targeted_dose.run_one`, and §4.4's first check is what proves it inert.

**Cost.** 3 arms x 20 seeds = **60 runs**, none importable: no published file carries
`fedavg_then_krum|committed_pixel` beyond the pilot's three seeds, and the two controlled arms have
never been run on this cell. Measured per-run cost for this ladder is 721--1436 s, so **12--24 h**,
plus about 1 h for the channel measurement, which runs first and can stop the ladder.
