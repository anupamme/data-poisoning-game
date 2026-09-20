# Pre-registration (superseding): the decomposition arm with its instrument corrected, after the frozen instrument failed its own premise check

**This document supersedes `experiments/pre_registration_oracle_free_decomposition.md` (commit
`bed6562`) and does not amend it.** That file is left byte-identical. It was frozen in good faith, its
premise was then measured and **failed**, and the failure is recorded here with the measurements rather
than repaired in place. Amending a freeze after measuring is the one thing a freeze exists to prevent,
so the freeze stands and this document carries the correction.

**Frozen before any result in `results/oracle_free_decomposition/` exists.** That path does not exist at
the time this file is committed. **No ASR and no accuracy has been measured for either arm**, under
either the frozen instrument or the corrected one.

What *has* been measured is a premise check that reads no outcome:
`results/oracle_free_decomposition_premise.json`, produced by
`experiments/measure_boundary_premise.py` over seeds 42--46 x 5 live rounds, **md5
`fd3627adcafdb1aad96c76727e8486d5`**, which two independent runs reproduce byte-for-byte. It is on the
tree at the time of this commit and is committed with the round's evidence, in the same relation
`results/oracle_free_screen.json` had to `bed6562`. This document is therefore written in the same
position `bed6562` was: after a premise measurement, before any outcome.

**The precedent this follows is the paper's own.** The CIFAR-100 norm-clipping arm
(`app:normclip_cifar100`) was frozen, its premise failed, the failure was recorded, and the arm was not
run as frozen. This is that, with a corrected instrument attached instead of a withdrawal.

## 1. What `bed6562` froze, and the two gates it fails

`bed6562` §3 froze a 1001-point float64 grid (`GRID_POINTS = 1001`, `h = 1e-3`) on the Gram-matrix
shortcut, and §6 froze two gates on it:

* **share gate** -- the adversarial coefficient share gap across the arm pair must be `<= SHARE_TOL`,
  imported as `1e-6` from `measure_admission.py`;
* **flip gate** -- on every round typed as a flip round, the two arms' selections must differ **under
  the shipped `krum_selection` on materialized stacks**, not merely under the Gram scores that chose
  the pair.

Measured over 25 live rounds (seeds 42--46 x 5), of which **13 carry a flip** on the frozen grid and
**12 of those 13 have at least one adversary present**, which is the population the share gate
quantifies over:

| gate | frozen instrument, measured | verdict |
|---|---|---|
| share, realized \|share_B - share_A\| | exceeds `1e-6` in **11 of 12** dosed adversary rounds; max **8.586e-05** | **FAILS** |
| share, mask-free supremum | exceeds `1e-6` in **12 of 12** dosed rounds, and in **13 of 13** flip rounds; max **1.0586e-04** | **FAILS** |
| flip, under the shipped statistic | selections are **identical** in **2 of 12** dosed rounds (3 of 13 flip rounds) | **FAILS** |

Run as frozen, the arm would have spent ~11.5 h to produce a contrast typed **confounded in the
attenuation channel** -- App. D.7's exact defect, which is the defect this arm exists to repair -- and
would additionally have carried **no dose at all** on a sixth of its dosed rounds while counting them
as dosed.

**§3's existence proof was a single round, and this corrects it.** `bed6562` §3 reports a share gap of
**5.109e-07** on seed 42's flip round and calls that measured pair "the existence proof that the
instrument's two requirements can hold simultaneously." That number is reproduced here exactly, and it
is **the minimum over the 12 dosed rounds, the only one of the 12 that passes**
(`summary.route1_frozen.min_realized_gap`). The other eleven range up to 8.586e-05. One passing round
was read as an existence proof for the construction when it was the best case of a distribution that
mostly fails. The mask-free supremum on that same round is **8.368e-05**, so the round does not pass on
the uniform reading either, and §3's own neutrality argument is the uniform one.

**Two further corrections to `bed6562`'s §3, both about its reasoning rather than its arithmetic:**

* §3 argues the gap is "**exactly linear in `h`** and is driven below any tolerance by shrinking the
  grid." The linearity is true and the inference is false, for the reason §2 below measures: shrinking
  `h` around the **float64** crossing drives the gap to numerical zero *and destroys the decision
  change*. Neutrality and informativeness do not both survive that limit, so `h` alone is not the dial.
* `bed6562` describes `t*` as **"the smallest grid `t` at which Krum's selection differs"** (§3) and the
  contrast as against **"this round's first selection flip"** (§9). The corrected search makes no such
  claim; see §3 below.

## 2. The obvious repair does not work, and the reason is a wrong search predicate

The repair §3's linearity argument implies is "shrink `h`". Measured as route 2 of the premise artifact
-- bracket with the float64 Gram predicate, bisect **that** predicate to an arbitrarily tight pair,
then evaluate the pair with the shipped statistic:

| route 2, measured | value |
|---|---|
| mask-free supremum share gap | max **3.331e-16** -- numerically exact neutrality |
| selections differing under the shipped statistic | **0 of 12** dosed rounds |

So the tight pair is perfectly neutral and **completely uninformative**. This is *not* a resolution
floor and it is *not* an instance of App. D.6's impossibility biting: it is a **wrong search
predicate**. The float64 Gram statistic and the shipped float32 `krum_selection` cross at slightly
different `t`. Bisecting the float64 predicate converges onto the float64 crossing, and a pair tight
around *that* point sits entirely on one side of the float32 one, so the statistic that types every
published Krum row in this paper sees no change at all.

The frozen `h = 1e-3` pair straddles both crossings often enough to flip the shipped statistic in 10 of
12 rounds precisely *because* it is loose -- and it is that same looseness that breaks the share gate.
Under the float64 search the two requirements are genuinely in tension, and that tension is an artifact
of the search, not a property of the blend.

## 3. The corrected instrument, frozen here

**Bisect the shipped predicate, and set the stopping rule on the gated quantity itself.**
`refined_boundary_pair` in `experiments/boundary_blend.py`:

1. Evaluate the **shipped** `krum_selection` on materialized stacks at `c(0)` and `c(1)`. If they
   differ, `[0, 1]` brackets a crossing of the shipped predicate by construction.
2. If they agree, use the free float64 Gram scan only to **propose** a bracket, and require the
   **shipped** predicate to confirm it at both endpoints before bisecting. A bracket the shipped
   statistic does not confirm is **not a dose**, and the round carries none.
3. Bisect, holding the invariant that the shipped selection at `lo` differs from the shipped selection
   at `hi`. The decision change is therefore true **by construction at every step and at every
   tolerance**, which is the property the float64 search lacked.
4. **Stop on the mask-free supremum share gap**, not on a proxy for it: bisect until
   `share_gap_sup(c_lo, c_hi) <= SHARE_TOL / 10`, capped at `SHIPPED_BISECT_MAX_STEPS = 40`.

**The three constants frozen here, before any ASR exists:**

    SHARE_TOL                 = 1e-6    IMPORTED from experiments/measure_admission.py, not redefined
    SHARE_GAP_TARGET_DIVISOR  = 10      so the stopping target is 1e-7, a factor of 10 under the gate
    SHIPPED_BISECT_MAX_STEPS  = 40      a COST bound, not a tolerance

`SHIPPED_BISECT_MAX_STEPS` is a cap on work, and a round that exhausts it without reaching the target
**fails the share gate** and is typed as a failure rather than let through. The cap never relaxes the
tolerance. `GRID_POINTS = 1001` survives only as the coarse proposal grid in step 2 and no longer sets
any reported gap.

**Measured, over the same 25 rounds, before any ASR exists:**

| route 3, the instrument frozen here | value |
|---|---|
| dosed adversary rounds passing **both** gates | **12 of 12** (13 of 13 flip rounds) |
| mask-free supremum share gap | **5.048e-08** to **9.602e-08**, at least 10x under `1e-6` |
| realized share gap | **4.872e-10** to **8.188e-08** |
| coefficient gap, L-inf | **1.648e-07** to **4.094e-07** |
| bisection steps | **20 to 21** |
| shipped `krum_selection` evaluations per round | **25 to 26** on a flip round, **2** on a non-flip round |

The 13 flip rounds' selection changes are genuine and distinct: `0->3, 3->1, 3->4, 1->3, 1->0, 3->0,
0->2, 2->3, 0->1, 0->1, 4->0, 4->3, 0->4`.

**The pair straddles *a* crossing, not necessarily the first.** Bisection on a predicate that is not
monotone in `t` converges to some crossing and which one is not claimed. What is asserted per round is
exactly what the gates test: the two arms' selections differ under the shipped statistic, and their
coefficients are uniformly share-neutral to `1e-6`. `bed6562`'s "smallest" (§3) and "first" (§9) are
withdrawn, and no claim in this document depends on either.

**Why this is still consistent with App. D.6, unchanged from `bed6562` §3.** App. D.6 forbids **exact**
neutrality with informativeness. The pair's gap is positive (`>= 5e-08`), so the construction is
consistent with the theorem rather than a counterexample to it. The tolerance-parameterized statement
is what this arm adds, and the correction *strengthens* it: approximate uniform neutrality with a
decision change is achievable **at any tolerance**, because the decision change is now maintained as an
invariant of the search rather than hoped for as a byproduct of grid spacing. Route 2 is what the naive
limit costs and it is reported.

**The share gate's definition, precisified rather than amended.** `bed6562` §6 says "per round, the
realized `|share_B - share_A|` on adversary rounds". Arms A and B are separate runs whose trajectories
necessarily diverge once a selection flips, so an across-arm per-round gap is well defined only at
round 0. The gate as implemented is the **within-run counterfactual pair** -- the two coefficient
vectors the round would hand to each arm -- gated on the **mask-free supremum over every nonempty
proper adversary subset**, all 30 of them at K=5, enumerated exactly rather than sampled. This is
strictly stronger than the frozen text in three ways: a supremum rather than one realized subset, every
round of both runs rather than adversary rounds only, and **asserted** rather than recorded. It is
mask-free, so the construction stays oracle-free. It is written here before any ASR exists.

**Oracle-freeness is structural, unchanged.** `c^rfa` comes from the update stack alone; the bisection
compares **selections**, never labels; the runner passes `adv_mask=None` and `d1_override="fedavg"`, so
any dose family reached by this path would raise; the hook receives only `(ups, seed, round)` and never
the adversary set; and the supremum gate needs no mask by construction. `adv_share()` exists only to
*measure* a pair after the fact and is never called by the construction.

**One number from the premise artifact is outside a published range, and it is a different
population.** The screen (`results/oracle_free_screen.json`, 5 seeds x 3 rounds) recorded `rfa`'s
realized spread `rho` in `[1.3061, 2.1944]`. This 5-round measurement reproduces the minimum exactly at
**1.3061** and reaches **2.7079**, because rounds 3 and 4 of each seed are new. Neither range is wrong
and neither supersedes the other; they are over different round counts, and this is recorded so that no
later reader takes it for a contradiction.

## 4. Carried forward from `bed6562`, unchanged

Every item in this section is **identical** to the superseded freeze. The instrument changed; nothing
about the design, the estimand, the predictions or the limits changed.

* **§1--§2 of `bed6562` stand in full**: the defect this arm repairs is App. D.7's own (Δ_stat =
  **+0.2969**, CI [+0.0983, +0.4955], n=20, with the adversarial share moving in **12 of 12** adversary
  rounds by up to **0.0859**), and the screen's verdict that no deployed oracle-free family is eligible
  (**NO_ELIGIBLE_FAMILY**; `norm_clip` inert at tau=5.0 with max client norm 3.0437; `reputation`,
  `foolsgold` and `rfa` all moving the share in 12 of 12) is unaffected by anything measured here.
* **The design.** Both arms new. Seeds **42--61 (n=20)**, both arms on every seed, 50 rounds, Krum as
  `d2`, committed pixel backdoor, CIFAR-10, `cifar_cnn`, N=10, K=5, f=0.2, tau=5.0, alpha=0.5.
  Arm **A** = the identity-side selection `c(t_lo)`; arm **B** = the flipped selection `c(t_hi)`.
* **Rounds with no confirmed crossing carry no dose**, both arms receive `c(1)`, the round is
  bit-identical across arms, and `n_flip_rounds` is recorded per run. The premise artifact measures
  **13 flip rounds in 25**, and restricted to the screen's own population -- rounds 0--2 of the same 5
  seeds -- it measures **exactly 8 of 15**, reproducing the screen's pre-measured endpoint-disagreement
  count (`rfa` decision rate **0.5333**) through a different code path. That count is a lower bound on
  the flip-round rate, so at 50 rounds the expected dose remains **>= ~26 flip-rounds**. The 13-in-25
  rate (0.52) is over a larger population that adds rounds 3--4, and is not a revision of the bound.
* **The primary estimand** is the paired difference **B - A at the same 20 seeds**, both legs new, no
  mixed-`n` leg, and App. D.7's identity arm in `results/oracle_free_channels/` is **context only** and
  never a leg of the primary Δ.
* **Two verdicts per arm pair, read separately**, neither ever reported as the other:
  `sign` = does the paired 95% CI exclude zero; `margin` = is `|Δ| < EQUIV_MARGIN = 0.15`. An interval
  can sit inside the margin and still exclude zero.
* **The pre-registered prediction is unchanged**: the decision channel alone, with the attenuation
  channel closed to `1e-6`, **does** carry suppression -- `sign = excludes zero`.
* **The branch that refutes us, named in advance and unchanged.** If `|Δ| > 0.15` with the interval
  excluding zero in the direction of *increased* ASR, the decision channel carries suppression with the
  attenuation channel closed to tolerance and App. D.7's confound is removed as an explanation of its
  +0.297. That is a **contradiction** of this paper's headline reading and is reported as a
  contradiction in the body, not converted into a scope condition afterwards.
* **The unresolved branch, and it is not a zero.** The flip's direction is **not controlled**: the
  argmin moves to whichever client the boundary lies against, adversarial or benign, and it may differ
  by round -- the 13 measured changes above go in both directions. Per-round effects can cancel and the
  arm may return an interval inside the margin containing zero. **That outcome is `unresolved`, not `no
  effect`,** it is not evidence of absence, and the supplement's equivalence phrase is **not** applied
  to it.
* **No mechanism clause inside any verdict literal.** Verdicts are `excludes_zero` / `contains_zero`
  and `inside_margin` / `outside_margin`; mechanism goes in its own field with the measured direction
  beside it.
* **The accuracy floor is live.** `ACC_FLOOR = 0.35` imported from `run_targeted_dose.py`, against App.
  D.7's identity arm at mean accuracy **0.5158**. A rung below the floor is **void, not negative**, its
  seed is dropped from the paired Δ, and the exclusion and the retained-value Δ are both reported.
* **§8's one shared-code change is unchanged**: `stack_hook=None` on `run_one`, applied to `ups`
  immediately after `apply_adversary` and immediately before `generic_compose`, receiving
  `(ups, seed, round)` and never the adversary set, bit-neutral at every existing call site.
* **§9's limits are carried in full and are not widened by anything here**: one aggregator, one attack,
  one dataset, one upstream transform, one architecture; the contrast is **local**, at the decision
  boundary, between "just below a selection crossing" and "just above it", and must not be reported as
  "identity against a statistic disturbance"; this arm is an **instrument, not a deployable defense**;
  no second dataset; the attack is non-adaptive; no row is pooled with any published row.
* **The sentence "the negative reproduces oracle-free" may not be written anywhere, under any
  outcome.** App. D.7 refuted its own prediction and that refutation stands unamended.

## 5. Gates, all imported and all live

    ACC_FLOOR     = 0.35    from experiments/run_targeted_dose.py
    EQUIV_MARGIN  = 0.15    from experiments/run_targeted_dose.py
    SHARE_TOL     = 1e-6    from experiments/measure_admission.py

* **Share gate**, as precisified in §3: per round, in **both** arms, the mask-free supremum share gap
  across the round's counterfactual pair must be `<= SHARE_TOL`. It is **asserted** -- the runner
  raises -- not recorded and read afterwards. **If it fails on any round the arm is typed confounded in
  the attenuation channel and reported as such**, by the `app:normclip_cifar100` precedent. It is not
  fixed by widening the tolerance, and the premise measurement above is what that rule looks like when
  it fires.
* **Flip gate**: on every round typed as a flip round the two arms' selections must differ under the
  shipped `krum_selection` on materialized stacks. The runner raises otherwise. Under the corrected
  search this is an invariant of the bisection rather than a hope, which is exactly why it now holds
  12 of 12 instead of 10 of 12.
* **Accuracy floor**, as in §4.

## 6. Harness checks, whose verdicts go into the artifact

All verdict dicts are persisted into `results/oracle_free_decomposition/`. A check whose verdict is
discarded leaves its claim witnessed only by a terminal and costs hours to repeat.

1. **`stack_hook=None` is bit-neutral.** Recomputing a frozen `results/targeted_dose/` cell through the
   patched `run_one` returns Δ = (0, 0) on both accuracy and ASR.
2. **The blend at `t=0` reproduces App. D.7's identity arm** at a shared seed, Δ = (0, 0) -- the
   assertion that the hook path is *faithful* rather than merely present. This is the check that would
   catch the standing failure mode in this repository: a manipulation hook that is never called while
   the run passes silently. It has happened once and is documented at the site of its fix,
   `run_cross_distribution_compositions.py:155-161`, where omitting `manipulate_update` had made
   `committed_scaling` bit-identical to `committed_pixel`. A `stack_hook` that is never invoked leaves
   arm B equal to arm A and the ladder still completes.
3. **The Gram proposal is validated, never trusted.** `gram_krum_selection` equals the shipped
   `krum_selection` at `t=0` and `t=1` on live stacks. Under the corrected search the Gram path only
   proposes brackets, and every reported pair is confirmed by the shipped statistic, so a Gram
   disagreement can no longer type a round as dosed.
4. **Per round, both gates assert**: supremum share gap within tolerance and shipped selections
   differing, with `==` / `torch.equal` rather than against a docstring.

## 7. What this document does not do

* **It does not amend `bed6562`.** That file is byte-identical to its committed state and stays so. Its
  §1, §2, §4, §5, §7, §8 and §9 are carried forward here; its §3 instrument and §6 gate wording are
  superseded, with the failure measured and reported rather than deleted.
* **It does not weaken a gate to pass.** `SHARE_TOL` is the same imported `1e-6`; the stopping rule
  aims a factor of 10 *under* it; the flip gate is unchanged in content and is now harder to satisfy
  accidentally; and the new cap is a cost bound whose exhaustion is a failure.
* **It does not read an outcome.** No ASR, no accuracy, and no adversary-conditioned quantity entered
  any decision in §1--§3. The premise artifact records that explicitly.
* **It does not claim the frozen arm would have refuted or confirmed anything.** It was not run. What is
  claimed is that it could not have answered the question it was built for, and that is measured.
* **It does not change the cost story materially.** The ladder is 40 runs at App. D.7's measured
  ~1038 s/run ~= **11.5 h**; the search adds at most 26 shipped `krum_selection` evaluations per round
  at a measured 66.0 ms each, i.e. `40 x 50 x 26 x 0.066 s` ~= **0.95 h**, for **~12.5 h** total.
