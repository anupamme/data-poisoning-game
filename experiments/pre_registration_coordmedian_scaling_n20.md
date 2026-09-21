# Pre-registration: `coord_median` / `committed_scaling` at n = 20, endpoint-only

**Status: frozen before any write to `results/coordmedian_scaling_topup/`.** The seed list, the endpoint
grid, the three estimands, the interval convention, the three verdict labels and the two named refuting
branches below are reproduced in `experiments/run_coordmedian_scaling_topup.py`, which refuses to start
until this file is committed and its hash is recorded in `PREREG_COMMIT`, and which also refuses if this
file has uncommitted edits, because `git log -1` cannot see those.

## What is being extended, and what is not

The cell is `coord_median` under `committed_scaling` on CIFAR-10 / `cifar_cnn`, estimated two ways: the
outcome-gated ladder (`dose_kappa<κ>`, the adversary free to attenuate) and the Mode-S intervention
(`doseS_kappa<κ>`, the adversary's coefficient pinned at c = 1 by a mask that reads adversary identity).
It is `training: false` in `results/comparability_six_cells.json` — a held-out cell, not one of the four
that trained the since-withdrawn ΔΛ_a rule.

**Both legs are at n = 5.** This document adds seeds 47–61 so that the *separation between the two
designs* on this cell can be stated at n = 20. It does not re-test the direction of either leg, does not
revise any threshold, does not touch the other six cells, and does not extend the CIFAR-100 replication.

Per-seed endpoint ASR as published, seeds 42–46 in order, recomputed from
`results/comparability_cells/summary.json` rather than transcribed from the paper:

| leg | rung | per-seed ASR | rung mean | rung mean acc. |
|---|---|---|---|---|
| both (shared) | κ = 0 | 0.5486 / 0.3407 / 0.4600 / 0.7676 / 0.4761 | 0.5186 | 0.7643 |
| confounded (`dose_kappa2.0`) | κ = 2 | 0.4858 / 0.4317 / 0.4908 / 0.7113 / 0.4827 | 0.5204 | 0.7148 |
| controlled (`doseS_kappa2.0`) | κ = 2 | 0.7223 / 0.5339 / 0.7723 / 0.9079 / 0.6370 | 0.7147 | 0.7047 |

Paired endpoint differences at n = 5, under the interval convention fixed below:

| estimand | per-seed d | mean | sd | 95% interval | half-width |
|---|---|---|---|---|---|
| Δ_conf | −0.0628 / +0.0910 / +0.0308 / −0.0562 / +0.0066 | **+0.0019** | 0.0639 | [−0.0775, +0.0813] | 0.0794 |
| Δ_ctrl | +0.1738 / +0.1932 / +0.3123 / +0.1403 / +0.1609 | **+0.1961** | 0.0678 | [+0.1120, +0.2802] | 0.0841 |
| D = Δ_ctrl − Δ_conf | +0.2366 / +0.1022 / +0.2816 / +0.1966 / +0.1543 | **+0.1942** | 0.0698 | [+0.1076, +0.2809] | 0.0866 |

**This cell is a disagreement, not a sign reversal, and it is not called one.** The confounded interval
*contains* zero; the controlled interval excludes it. Both means are positive. `predicted: DISAGREE`,
`observed: DISAGREE` in the frozen six-cell artifact, and `main.tex:1970` already says *"It is a
disagreement, not a second sign reversal, and we do not call it one."* Nothing in this document may be
read as adding a third sign reversal, at any n.

## The interval convention, fixed here because the repository contains two

`analyze_comparability.py:293` computes the paired interval as `t_crit(n) * sd/√n` where
`analyze_headline_cis.t_crit(n)` returns `t.ppf(0.975, n − 1)`. **`t_crit` takes n, not the degrees of
freedom, and computes df = n − 1 internally.** Calling `t_crit(n − 1)` yields df = n − 2 and a wider
interval: at n = 5 it returns 3.182 instead of 2.776 and turns [−0.0775, +0.0813] into [−0.0891,
+0.0929]. Every number in the tables above reproduces the published literals under `t_crit(n)`, and the
runner imports `t_crit` rather than restating any table, per `analyze_comparability.py:41-42`. At n = 20
the multiplier is `t_crit(20) = 2.0930240544083087`.

This is a 95% two-sided Student-t interval on a paired difference. It is **not** the 90% TOST convention
of `supplementary.tex:441`, and no reading below is expressed in that convention or compared against a
number that is.

## Why this is not optional stopping, stated before any new seed runs

1. **This top-up can only hurt us, and the direction it can hurt us in is named below.** The published
   claim at n = 5 is that the two designs *disagree* on this cell. A top-up cannot manufacture a
   disagreement that is not there: it can narrow two intervals that already separate, or reveal that the
   separation was an artifact of five seeds. Only the second outcome is news, and it is news against the
   paper's own count.
2. **The refuting branches are named in advance**, with the exact sites that must change, including the
   abstract and Figure 1. Naming that downside before any seed runs is the only thing that licenses
   adding seeds.
3. **The seeds are fixed here**, contiguously, with no look-ahead: **47–61**, 15 new seeds, n = 20 total.
   No stopping rule, no interim look, no extension. If the runs are interrupted, the analysis reports the
   n actually reached and the intervals at that n; it does not resume until a threshold is crossed.
4. **Precedent, not innovation.** `reversal_seed_topup` (frozen at `fb94d3a`, amended to `89beed0`) did
   exactly this for `coord_median` / `committed_pixel`, and `dose_seed_topup` and `emit_only_topup` (both
   `684b31e`) did it for two further arms, all for interval width rather than a second test. Seeds 47–61
   are the same block those suites used; different cell key, own results directory, no collision.

## The three estimands, all paired at the same 20 seeds

- **Δ_conf** = mean over seeds of `ASR(confounded, κ=2) − ASR(shared, κ=0)`.
- **Δ_ctrl** = mean over seeds of `ASR(controlled, κ=2) − ASR(shared, κ=0)`.
- **D = Δ_ctrl − Δ_conf**, formed **per seed and then averaged**, not as a difference of the two means'
  intervals. D is the quantity the design-choice claim rests on, and it is reported with its own paired
  interval.

Pairing is real by construction: a seed fixes the Dirichlet partition, the model initialization and the
participant sampling stream, so two rungs at one seed differ only in `d₁`. Because the identity rung is
shared (below), D's per-seed value is `ASR(controlled, κ=2) − ASR(confounded, κ=2)` exactly; the shared
minuend cancels. That is recorded here so that no later analysis "improves" D by re-deriving it from two
separately-computed identity legs, which would silently mix n between a minuend and a subtrahend.

**No p-value is attached to the difference between the two designs.** They are not the same intervention
and Λ_a moves in both, exactly as the paper already states. D's interval is a paired interval on a
measured contrast, and it is reported as that and not as a test of one design against the other.

## Three verdicts per estimand, read separately

Each estimand gets all three labels, emitted independently into the artifact. **None may be reported as
another**, and none names a mechanism:

1. **sign** — does the paired 95% interval exclude zero, and in which direction.
2. **margin** — is |mean| < 0.15, the frozen practical-equivalence margin.
3. **one-sided** — is the 95% upper bound below +0.15.

An interval can sit inside the margin and still exclude zero, and a one-sided bound can hold while a
two-sided reading fails. At n = 5 this cell already shows the pattern: Δ_conf is inside the margin and
contains zero, while Δ_ctrl is outside the margin and excludes zero.

## Reachability, stated in advance: both legs are genuine tests on this cell

The identity rung's mean ASR is **0.5186**, so the largest available fall is 0.5186 > 0.15 and **the
lower leg of the equivalence reading is a real test here, not an arithmetic tautology**. This is the
property the cell was chosen for. `results/margin_reachability.json` records it as the single arm in
`distinct_arms_keeping_the_two_sided_reading`, and `main.tex:3105` already singles it out as the one of
five equivalence arms that *"keeps a genuinely two-sided reading and is reported as one."*

Two consequences fixed here: the one-sided relabelling applied to the flagship's arithmetic lower leg
**does not apply to this cell**, and a rise is available as well as a fall, so the upper bound is not
satisfied by construction either.

## The endpoint grid, and its declared consequence

**Endpoint rungs only: κ ∈ {0, 2}**, which is exactly what the frozen primary contrast reads
(`LO, HI = KAPPAS[0], KAPPAS[-1]` = 0.0 and 2.0). Copied forward in force from
`pre_registration_reversal_seed_topup.md`:

- κ = 0.5 and κ = 1.0 are **not run** for the new seeds. This cell's four-rung trend statistics stay at
  n = 5 and stay `pre_registered: false` and post hoc. The published four-rung readings — confounded
  J = 79.0, z = 0.269, p_increasing = 0.394; controlled J = 103.0, z = 1.884, p_increasing = 0.030 — are
  **not recomputed, not extended and not restated at n = 20**.
- **No Jonckheere–Terpstra statistic is computed for the new seeds.**
- **No four-rung display may print this ladder without a per-rung n.** After this top-up the honest shape
  is κ = 0 and κ = 2 at n = 20 and the interior at n = 5, and that shape is not to be smoothed.

## Decision rules, and the two refuting branches with their reporting sites

Nothing here is a new threshold. The margin (0.15), the accuracy floor (0.35) and the endpoint grid are
carried forward unchanged from `pre_registration_comparability.md`.

- **Disagreement confirmed at n = 20:** Δ_conf's interval contains zero, Δ_ctrl's excludes zero, and D's
  excludes zero. Reported as the published result, narrowed, with the label `DISAGREE` unchanged.

- **REFUTING BRANCH 1 — the designs agree at n = 20.** If **D's interval contains zero**, the two designs
  are not separated on this cell at n = 20. Then the published n = 5 disagreement was an artifact of five
  seeds, **this cell's `observed` label changes from DISAGREE to AGREE, and the paper's seven-cell count
  of disagreements falls from five to four.** That is **reported as a contradiction of our own published
  count and is not converted into a scope condition afterwards.** The sites that must change, named now
  so the change cannot be confined to an appendix:
  - the **abstract**, `main.tex:72`: *"The designs disagree on five of seven cells"*;
  - **§5**, `main.tex:853`: *"They disagree on five, across two attacks, three datasets and two CNN
    configurations"*, and the same paragraph's *"`coord_median` under model scaling disagrees too, so it
    is not specific to the pixel backdoor"*, which is this cell and would become false;
  - **Figure 1(b)**'s panel title, *"the design choice changes the answer on 5 of 7 cells, and its SIGN
    on 2"*, and its row for this cell;
  - `tab:sixcell` at `main.tex:1962`, both means, both intervals, and the `observed` column;
  - `main.tex:1970`, *"one confirms, one refutes, one was never predicted"* — this cell is the *one
    confirms*, so that clause is restated to match what n = 20 shows;
  - `main.tex:1976`, *"across seven cells the two designs disagree on five, on two different attacks,
    three datasets and two CNN configurations"*, including whether *two different attacks* survives.
  The two sign reversals are **not** affected by this branch: they are the `coord_median`/pixel cells, and
  this cell was never counted as one.

- **REFUTING BRANCH 2 — the confounded leg's label changes.** If **Δ_conf's interval, which contains zero
  at n = 5, excludes zero at n = 20**, then the outcome-gated design does detect an effect on this cell
  and the cell's description changes. Reported as measured, in `tab:sixcell` and at `main.tex:3105`,
  which currently describes this arm's confounded reading as the one *"genuinely two-sided"* equivalence
  reading in the paper and quotes `[-0.078,+0.081]` at n = 5. This branch is compatible with branch 1
  and with the confirming branch; all three are read independently.

- **Direction change:** if either leg's mean changes sign at n = 20, that is reported as its own result
  and the relevant branch above fires regardless of the intervals.

- **Accuracy gate, live.** Floor **0.35 applied to a rung's mean**, inherited verbatim from
  `pre_registration_comparability.md:75` and `:149`, where *"seeds below it are flagged, never
  excluded."* The identity rung's mean accuracy is 0.7643 and the minimum accuracy over all four
  published rungs and five seeds is 0.6678, so **no published rung is void**. A new rung whose mean falls
  below the floor is **void, not negative**: it is uninterpretable and is reported as such, never as
  suppression. Per-seed clean accuracy is recorded for every new run, because
  `analyze_comparability.py` reads no accuracy at all.

- **No admissibility gate applies to this cell**, deliberately. Cell 7's two-sided `[0.15, 0.85]`
  identity-rung gate existed to decide whether an *unrun* third dataset had headroom. This cell is
  already published at n = 5 with an identity rung of 0.5186; there is nothing to admit, and adding a
  gate now would create a licence to discard the run on its own outcome.

## The shared identity rung, verified rather than assumed

At κ = 0 the transform returns the update list unwrapped, so `dose_kappa0.0` and `doseS_kappa0.0` are the
same computation. **This is already true bit-for-bit for this cell in the published artifact**: for all
five seeds 42–46, `results/comparability_cells/summary.json` carries identical float reprs at κ = 0 in
**both** ASR and clean accuracy under both family keys (`0.5485555555555556`, `0.3406666666666667`,
`0.46`, `0.7675555555555555`, `0.4761111111111111`). So the identity rung is computed **once** per new
seed and shared by both legs: 15 × 3 = **45 runs**, not 60.

**That bit-identity is established on this attack rather than inherited from the pixel cell.**
`run_coordmedian_scaling_topup.py --harness-check` runs first, at seed 42, where published values already
exist — a new seed would make the check vacuous — and asserts agreement to `< 1e-9` on four rows:

1. the outcome-gated κ = 0 row reproduces the published `dose_kappa0.0` value;
2. the Mode-S κ = 0 row reproduces the **same pair**, establishing rather than assuming that the shared
   identity holds *on this attack*;
3. the outcome-gated κ = 2 row reproduces the published `dose_kappa2.0` value (mean ASR 0.5204 at n = 5);
4. the Mode-S κ = 2 row reproduces the published `doseS_kappa2.0` value (mean ASR 0.7147 at n = 5).

(1) and (2) together license the sharing: each provenance of the shared rung is tied to its own published
value, so all four quantities are equal by transitivity. (3) and (4) prove the harness is the same loop
the published cell was run with, not merely the same at the identity, on **both** designs.

**The verdict dict of `--harness-check` is persisted into the artifact**, not merely printed. A check
whose verdict is discarded is witnessed only by a terminal and costs hours to repeat.

**Named fallback if (2) fails:** the identity is **not** shared on this attack, the suite becomes **60
runs** with both identity legs computed separately, and that is reported as a difference between the two
attacks rather than reconciled after the fact. If (1), (3) or (4) fails, the harness is not the published
one and **the top-up does not run at all**.

## No admission quantity and no channel quantity

`run_one` returns `(accuracy, asr)` and nothing else. This arm measures **ASR only**.

- `d_admission = 0.0` and `d_influence = 0.008619231983179303` for this cell are **imported** from the
  frozen six-cell measurement, not re-measured, and are not revised by anything here.
- `summary[…|admission]` is **0.0 at every identity rung by construction** and is never read as a level.
- Therefore **no statement of the form "a statistic disturbance costs or buys suppression on this cell"
  may be made in either direction**, and no channel decomposition is claimed. The `measure_admission*`
  family is not run.

## Scope: which displayed numbers move to n = 20, fixed in advance

- **Moves to n = 20:** this cell's two means and two intervals wherever it is *scored* — `tab:sixcell` at
  `main.tex:1962`, the out-of-sample paragraph at `main.tex:1970`, `main.tex:3105`'s quotation of the
  confounded interval, and Figure 1(b)'s row for this cell, which reads the artifact. **D is new** and is
  reported at n = 20 only; there is no published D for this cell to supersede.
- **Stays at its own frozen n:** the CIFAR-100 replication (n = 5); the other six cells; Table 1
  (`tab:channels`), whose rows are the Mode-S channel suite with its own freeze; and the margin ladder at
  `main.tex:3278`, whose `m* = 0.280` for this arm is emitted by `measure_margin_sensitivity.py` into
  `results/margin_sensitivity.json`, **neither of which is edited or re-run** — that row stays a frozen
  n = 5 object and prints its n.
- **Every site quoting this cell's numbers carries an explicit n** after this round, so the n = 5 and
  n = 20 readings of the same estimand cannot be read as one number.
- **The published n = 5 verdict is reported alongside the n = 20 verdict, whatever the latter is.** If the
  two n disagree, both appear and the disagreement is the result.

## Caveats recorded in advance

1. **The sds are estimated from five seeds.** Holding the observed sds, n = 20 would give half-widths of
   roughly 0.030 (Δ_conf), 0.032 (Δ_ctrl) and 0.033 (D), against 0.079, 0.084 and 0.087 at n = 5. The
   realized sds may be larger and **the realized intervals are what is reported. No projected number
   appears in the paper.** For D's interval to reach zero at n = 20 its sd would have to inflate from
   0.0698 to 0.4150, a factor of roughly **5.9×**, which is arithmetic and not a prediction. (The
   distinct factor of ~2.9× is how much the interval *narrows* at constant sd; the two must not be
   confused.)
2. **Narrowing intervals does not widen the claim.** At n = 20 this remains one aggregator, one attack,
   one dataset, one architecture, and a controlled leg whose instrument **reads adversary identity**. The
   top-up buys precision on one cell's design disagreement and nothing else. It is not evidence about any
   other cell, it is not a second dataset, and Mode S is not a defense.
3. **The controlled leg is masked, so this arm bears nothing on the oracle objection.** The sentence *"the
   negative reproduces oracle-free"* may not be written, here or anywhere, and this arm is not a step
   toward it. The open problem stated in the paper's Conclusion and Ethics statement is untouched.
4. **The controlled κ = 2 rung mean is 0.7147 at n = 5**, above 0.5. No rule in this cell's
   pre-registration reads a 0.5 ceiling; that clause belongs to the *equivalence* arms, where a generous
   margin must not certify a high-ASR composition. It is recorded because it looks like a gate violation
   and is not one.
5. **This cell is held out (`training: false`) and the rule it was held out from is already withdrawn.**
   A cell that agrees with what a dead mechanism would have said does not revive it, exactly as
   disclosure (ii) of `main.tex:1974` states for cell 7.
6. **No existing artifact is written.** `results/comparability_cells/summary.json`,
   `results/comparability_six_cells.json`, `results/margin_sensitivity.json` and
   `results/margin_reachability.json` are read-only here. This suite writes only
   `results/coordmedian_scaling_topup/summary.json`, and the merge happens at analysis time, where the
   five published seeds are asserted to reproduce before any pooled number is printed.

## Non-negotiables

1. No interval definition, margin, accuracy floor, endpoint grid, verdict label or refuting branch above
   is revised after seeing an outcome. **No amendment is made to this freeze once any number exists.** If
   a literal here turns out to be wrong, it is printed as the frozen verdict with the measured direction
   beside it.
2. The seed list is 47–61. It is not extended, truncated by inspection, or filtered.
3. Every reported number is recomputed per seed from the artifacts, never transcribed, and `t_crit` is
   imported, never restated as a table lookup.
4. No existing runner, analyzer, pre-registration or `results/` path is edited. The new runner *imports*
   `run_one`, `cell_key`, `KAPPAS`, `SEEDS`, `FL_CONFIG` and `ADV_FRACTION` from
   `run_comparability_cells`; there is no `d1_override`, no `stack_hook` and no shared-code change.
5. `analyze_comparability.py`'s published-value reproduction guard is not weakened. Its expectations stay
   asserted against the **n = 5 subset** of the merged data; n = 20 values are added as a separate
   expectation. Editing a published tuple to match a new result would destroy the only check that the
   re-score reproduces the document.
6. The three verdicts are read separately and never substituted for one another; no verdict literal names
   a mechanism.
7. This file is committed **before** `results/coordmedian_scaling_topup/` is written, and the runner
   refuses to start unless `git log -1 --format=%h` on this path matches `PREREG_COMMIT` **and**
   `git status --porcelain` on it is empty. If that ordering cannot be demonstrated from `git log`, the
   top-up is reported as non-prospective.
8. This cell is a **disagreement**, never a sign reversal, at any n and under any outcome.
