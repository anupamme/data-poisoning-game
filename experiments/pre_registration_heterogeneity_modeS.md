# Pre-registration: does the Mode-S within-defense rise survive non-IID heterogeneity?

**Frozen before any result in `results/heterogeneity_modeS/` exists.** That directory does not exist at the
time this file is committed. `experiments/run_heterogeneity_modeS.py` refuses to start until this file is
git-committed and its `PREREG_COMMIT` is set to that hash.

## 1. What is being tested, and why nothing on disk tests it

The paper's instrument leg is a **within-defense rise**: with the adversarial coefficient pinned at
`c = 1.0` and only benign clients spread over `rho = e^{2 kappa}`, ASR *rises* as the downstream statistic
is disturbed. On the sign-reversal cell (`d2 = coord_median`, attack = committed pixel, kappa 0.0 -> 2.0)
that leg reads **+0.098067** at n=5 and **+0.125** at n=20.

**Mode S has only ever run at one Dirichlet concentration.** `run_targeted_dose.py:57` states the frozen
configuration verbatim: *"Config identical to the rest of the paper: N=10, K=5, f=0.2, alpha=0.5, 50 rounds,
cifar_cnn."* Every `doseS_*` row in `results/targeted_dose/`, `results/dose_replication/`,
`results/reversal_seed_topup/` and `results/regime_dissociation/` is at **alpha=0.5**.

Two artifacts look like they cover this and do not:

1. **`results/heterogeneity_sweep/summary.json`** varies alpha over {0.1, 1.0, 10.0} at N=10, K=5, f=0.2,
   50 rounds, `cifar_cnn`, seeds 42/43/44. But its cells are **composed pairs** scored for
   **the screen's verdicts** (`foolsgold_then_coord_median`, `reputation_then_coord_median`). There is no
   `doseS` row in it, no pinned adversarial coefficient, and no kappa ladder. Its own `note` records that
   *"alpha=0.5 not re-run"*.
2. **`results/regime_dissociation/`** varies the adversary and varies N/K. Its own pre-registration
   (§3, Regime B) states that it **holds alpha at the frozen 0.5** and that *"this arm makes no claim about
   alpha."*

So the heterogeneity dimension has been tested for the criterion and never for the instrument. This arm is
written because a review asked for exactly one Mode-S intervention off the frozen alpha.

**This arm changes one thing and nothing else.** Only `alpha` moves. The cell, the attack, the kappa
endpoints, N, K, f, the round count, `cifar_cnn`, the seed list and Mode S's own construction are held at
their frozen values.

## 2. The estimand, and the two verdicts frozen separately

Per alpha, paired by seed:

> **Delta_S(alpha) = ASR(doseS, kappa=2.0, alpha) - ASR(doseS, kappa=0.0, alpha)**,
> mean over seeds 42--46, 95% two-sided t-interval at n=5 (t = 2.776).

The frozen alpha=0.5 anchor, on the **same five seeds**: mean **+0.098067**, sd **0.06464**, ci95
**[+0.017819, +0.178314]**, n=5. That anchor is **not re-run** and is not a leg of this arm.

**Which artifact holds it, checked rather than assumed.** The per-seed rows for this cell at seeds
42--46 are in **`results/dose_replication/summary.json`**, whose own `config` records
`alpha: 0.5, rounds: 50, acc_floor: 0.35`. `results/targeted_dose/` has `doseS` rows only into
`cos_krum`, `krum` and `reputation` and carries **no** `doseS_*_then_coord_median` cell at all;
`results/dose_response/` carries the untargeted `dose` family; the n=20 top-up's seeds 47--61 are in
`results/reversal_seed_topup/`. The runner therefore **recomputes** the anchor from
`results/dose_replication/`'s per-seed rows and **cross-checks** it against
`results/reversal_seed_topup/summary.json`'s `published_n5_verdict` (`controlled`) summary key. Two
sources that must agree is the only way to notice a summary key that has drifted from the rows beneath
it. If they disagree, both are printed, neither is used, and the disagreement is the result.

### Two verdicts, and neither may be reported as the other

Round 68's regime arm cleared its pre-registered threshold in both regimes while reproducing the **sign**
reversal in only one, and the one-line summary *"the dissociation reproduces"* was therefore misleading.
This arm splits the two before seeing any number:

| verdict | rule | reported as |
|---|---|---|
| **SIGN** | `Delta_S(alpha) > 0` **and** its 95% interval excludes zero | **the within-defense rise reproduces at this alpha** |
| **SIGN** | interval contains zero | **not resolvable at this alpha and n=5.** Point estimate and interval printed. Not evidence of absence |
| **SIGN** | `Delta_S(alpha) < 0` and interval excludes zero | **REVERSED at this alpha.** Contradicts the instrument leg and goes in the body as a contradiction, in the same place a reproduction would have gone |
| **MAGNITUDE** | `Delta_S(alpha)`'s interval overlaps the frozen `[+0.017819, +0.178314]` | **consistent in magnitude with alpha=0.5** |
| **MAGNITUDE** | it does not overlap | **differs in magnitude from alpha=0.5**, with both intervals printed |

The magnitude verdict is an interval-overlap statement and **not** an equivalence claim. The phrase
*"no evidence of a practically meaningful change"* is defined at `supplementary.tex:441` against a paired
95% interval lying inside the margin; `EQUIV_MARGIN = 0.15` is **not** used anywhere in this arm.

**Verdict literals carry no mechanism clause.** Round 72's freeze fused a pass/fail label with a guess about
mechanism, and the guess was backwards in sign while the label was right, so the untested half inherited the
tested half's authority. Here the runner emits `verdict_sign` and `verdict_magnitude` as pass/fail labels and
writes any mechanism reading into a separate `mechanism_observed` field, explicitly labelled as the
measurement rather than the verdict.

**Every number is recomputed from per-seed rows at run time, and both legs of every Delta in the same call**,
so a subtrahend can never come from a different seed count, or a different alpha, than its minuend. Nothing
in this document is transcribed into code.

## 3. The grid, and the power available at each alpha, measured not assumed

`d2 = coord_median`, attack = `backdoor_pixel` (committed pixel), **kappa in {0.0, 2.0}** (endpoint rungs
only, rho = 1.00 and 54.60), **seeds 42/43/44/45/46**, N=10, K=5, f=0.2, 50 rounds, `cifar_cnn`, Mode S.

**alpha in {0.1, 1.0, 10.0}.** alpha=0.5 is the published anchor and is not re-run. This is the grid
`results/heterogeneity_sweep/` already uses, so each new row is comparable to an existing criterion row at
the same alpha. The review proposed {0.1, 0.3, 1.0, inf}; the paper's own grid is used instead so that the
comparison is to measured rows rather than to a new alpha with no prior on disk, and 0.3 is not added.

**n=5, not the 3 of the regime arm.** Fixed now. The frozen anchor's own sd (0.06464) gives an n=5
half-width of 0.080 against a mean of 0.098, so the anchor excludes zero at n=5 and would not at n=3
(half-width 0.161). n=5 is the smallest seed count at which this estimand is testable at all, and it is the
published grid for this cell.

### Power, from the measured marginal spreads at each alpha

Recomputed here from `results/heterogeneity_sweep/summary.json`, `coord_median` downstream under committed
pixel, over its three seeds:

| alpha | accuracy range | marginal ASR sd (fg / rep) | marginal ASR range (fg) |
|---|---|---|---|
| 0.1 | **0.364--0.488** | **0.2257** / 0.1059 | **0.4443** |
| 1.0 | 0.730--0.799 | 0.0433 / 0.1410 | 0.0863 |
| 10.0 | 0.774--0.815 | 0.0139 / 0.1726 | 0.0277 |

Two things follow, and both are stated before any run:

1. **A large marginal spread does not by itself imply a large paired sd.** In the frozen regime the marginal
   kappa=0 ASR spread is 0.285--0.734 (range 0.449, comparable to alpha=0.1's 0.444) while the *paired*
   Delta sd is 0.06464, because pairing by seed cancels most of the marginal variance. So alpha=0.1's
   marginal spread is **not** grounds to declare the leg unpowered in advance, and it is **not** grounds to
   expect it powered either.
2. **alpha=0.1 is nevertheless the leg at risk, and its accuracy gate is live.** Its measured accuracy floor
   is **0.364** against `ACC_FLOOR = 0.35`, a margin of 0.014. If that gate fails at either rung, the
   alpha=0.1 leg is **VOID, not negative** (§4.2).

**No optional stopping.** If any leg's interval contains zero, that is reported at n=5 with the point
estimate and the interval, as a failure to resolve at this seed count. It is **not** topped up, and no seed
is added to any alpha after a number is seen.

## 4. Gates and diagnostics, scored before the primary

**4.1 Accuracy gate (a real gate), per alpha and per rung.** Mean clean accuracy >= `ACC_FLOOR = 0.35`,
imported from `run_targeted_dose.py:110` rather than restated. If a rung fails, that alpha is
uninterpretable, **no verdict stands for it**, and it is reported as VOID next to the alphas that were not.
A low ASR at collapsed accuracy is not suppression. This gate is live at alpha=0.1, where the measured
accuracy is 0.364.

**4.2 The identity rung is re-run at every alpha, and is never imported.**
`run_targeted_dose.py:50`--`:55` records that `doseS_kappa0.0` is imported from `results/dose_response/`
because *"run_one's participant RNG stream does not depend on d1's name, so those runs are the same
computation."* That argument holds across `d1`, **not across alpha**: alpha sets the Dirichlet partition, so
a kappa=0 rung at alpha=0.1 is a different computation from one at alpha=0.5. Every kappa=0 rung here is
computed fresh at its own alpha. `results/dose_response/` is **not** read by this runner.

**4.3 Transform-applied assertion (a real gate), at every alpha's identity rung.** At kappa=0 every
per-client coefficient must be **exactly** 1.0 and every channel displacement **exactly** 0.0, tested with
`==` and not a tolerance. This is the guard for the known failure mode in which a manipulation hook is never
called and the run passes silently while reporting an unchanged number as a finding.

**4.4 The adversarial coefficient share is measured at every alpha, not assumed.** Mode S's claim is that
the adversary is pinned, and that claim is a property of the partition as well as of the transform. The share
is reported per alpha and compared to `SHARE_TOL = 1e-6` imported from `measure_admission.py:67`. If the
share moves at some alpha, that alpha's leg is **reported as confounded at that alpha**, because it then
measures the attenuation channel as well as the statistic channel, and it does not carry a Mode-S reading.

**4.5 `--harness-check` runs first and its verdict is written into the artifact.** Two checks, one per
script, each reproducing a frozen number **by value** before any new alpha is touched:

- `run_heterogeneity_modeS.py --harness-check` runs `doseS_kappa0.0` at seed 42 and **alpha=0.5, the
  frozen default**, and compares accuracy and ASR to that row in **`results/dose_replication/summary.json`**
  (see §2 for why that file and not `results/targeted_dose/`). This establishes that passing `alpha`
  explicitly is inert on its default, i.e. that this runner's call path into `run_one` is the frozen one.
- `measure_admission_heterogeneity.py --harness-check` runs **its own copy of the traversal** at
  alpha=0.5 and compares seven fields per cell to `results/admission_measurement.json`. That copy exists
  only because `measure()`'s Dirichlet concentration is the literal `0.5` at `measure_admission.py:126`
  with no parameter to pass; the by-value check is what makes it a copy rather than a second definition
  of admission. The script **refuses to measure any new alpha** until this check has been run to PASS.

Round 69 established that the existing harness check returns a verdict dict its caller discards, leaving
the claim witnessed only by stdout. Both dicts are persisted here.

## 5. What this arm cannot conclude

- **No claim of heterogeneity-robustness from a subset of the grid.** If the rise reproduces at 1.0 and 10.0
  and not at 0.1, the reported result is that it reproduces at 1.0 and 10.0 and does not resolve at 0.1. The
  sentence *"the instrument leg survives heterogeneity"* may not be written.
- **No general law from a null.** A leg whose interval contains zero licenses no statement about heterogeneity
  in general, only about this cell, this attack, this defense and n=5.
- **No pooling across alpha.** Each alpha's Delta is reported with its own n and its own interval. No mean is
  taken across alphas, and no alpha's rung is differenced against another alpha's.
- **The frozen alpha=0.5 verdict is not amended.** It is printed as frozen beside whatever this arm returns,
  and if the two disagree, both appear and the disagreement is the result.
- **Nothing here concerns adaptivity.** The attack is the committed pixel backdoor, non-adaptive by
  construction, as in the frozen leg.

## 6. Outputs

- `results/heterogeneity_modeS/summary.json` -- per-alpha, per-rung, per-seed accuracy and ASR; both verdicts
  per alpha; the accuracy gate; the identity-rung assertion; the harness-check verdict dict.
- `results/heterogeneity_modeS_admission.json` -- the per-alpha adversarial coefficient share and its
  comparison to `SHARE_TOL`, from `experiments/measure_admission_heterogeneity.py`.

Neither path exists at commit time. **No existing `results/` directory is written to**, and
`measure_admission.py`, `run_targeted_dose.py` and `run_regime_dissociation.py` are imported, never edited.

**Cost.** 3 alpha x 2 rungs x 5 seeds = **30 runs**. Measured per-run cost for this ladder is 721--1436 s
(`results/dose_replication_run.log`, `results/targeted_dose_run.log`, `results/dose_seed_topup_run.log`), so
**6--12 h**, plus about 1 h for the admission measurement.
