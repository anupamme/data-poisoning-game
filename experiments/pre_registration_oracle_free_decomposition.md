# Pre-registration: separating the decision channel from the attenuation channel, oracle-free, with the coefficient share held to Mode S's own tolerance

**Frozen before any result in `results/oracle_free_decomposition/` exists.** That path does not exist
at the time this file is committed. `experiments/run_oracle_free_decomposition.py` refuses to start
until this file is git-committed, its `PREREG_COMMIT` matches `git log -1 --format=%h` for this path,
and `git status --porcelain` for this path is empty.

**No ASR of any kind has been measured for this arm at the time of this commit.** What has been
measured, and is reported in §3 below, is the instrument screen and the construction's self-check.
Both are premise measurements that read no outcome, in the same order App. D.7's premise check ran in.

## 1. The defect this arm exists to repair, which is our own

App. D.7 ran the review-named oracle-free intervention, refuted its own pre-registered prediction
(Δ_stat = **+0.2969**, paired 95% CI [+0.0983, +0.4955], n=20), and then disclosed, **post hoc**, that
the arm is not the clean probe its freeze took it for: the adversarial share of coefficient mass moves
in **12 of 12** adversary rounds, by up to **0.0859**, against the frozen `SHARE_TOL` of 1e-6 that
Mode S holds to 3.5e-07. So that arm moved the decision channel and the attenuation channel at once,
and its refutation does not transfer to the published Mode S result, whose settled statement is
conditional on *once the adversarial coefficient share is held fixed*.

App. D.7's own text and response letter own that defect in these terms: our freeze recorded
`share_tol` in its rules block, the runner recorded `adv_coeff_share` on every round, **and the gate
was still not written**. This arm writes that gate and runs the contrast the gate admits.

**One thing changes from App. D.7: the transform. Everything else is held.** Host aggregator (Krum),
attack (committed pixel backdoor), dataset (CIFAR-10), architecture, N, K, f, round count, Dirichlet
concentration, seed grid (42--61, n=20) and the accuracy floor are all at their frozen values.

## 2. Why no deployed defense can be the instrument, measured rather than argued

App. D.6 proves that within the positive per-client rescaling class, **exact** coefficient-neutrality
and informativeness are mutually exclusive: oracle-freeness forces share preservation to hold for
*every* adversary set, which forces `c` constant, which leaves Krum's argmin invariant.

`experiments/screen_oracle_free_transforms.py` measures that impossibility across the whole deployed
oracle-free class rather than arguing it for one member, on 5 seeds x 3 live rounds at this cell, with
every threshold imported from an already-frozen constant (`DECISION_FLOOR` 0.10 and `ADMISSION_FLAT`
0.05 from `measure_admission_mask.py`, `SHARE_TOL` 1e-6 from `measure_admission.py`). It invents no
threshold. Its verdict is **NO_ELIGIBLE_FAMILY**, and the shape of the failure is the impossibility:

| family | decision | \|admission\| | share moved | max \|Δ share\| | fails |
|---|---|---|---|---|---|
| `norm_clip` | 0.0000 | 0.0000 | **0/12** | **0.00000** | inert, decision does not move |
| `reputation` | 0.2000 | 0.0000 | 12/12 | 0.09975 | share not neutral |
| `foolsgold` | 0.5333 | 0.0667 | 12/12 | 0.40000 | admission moves, share not neutral |
| `rfa` | 0.5333 | 0.0000 | 12/12 | 0.08586 | share not neutral |

The dichotomy is total: the one family that is share-neutral is the one that is inert, and every
informative family moves the share in **12 of 12** adversary rounds. There is no intermediate case.

**`norm_clip`'s inertness is a fact about a hyperparameter, not about the defense,** and the screen
measures which: at tau=5.0 the maximum client update norm seen is **3.0437**, so `c_i = min(1, tau/||u_i||)`
is exactly 1 for every client and the transform is bit-identically the identity. Lowering tau until
the clip binds would make it informative *and* move the share. That is the trade the impossibility
describes; it would not produce an eligible family, and this arm does not lower tau.

The screen also **reproduces three published App. D.7 numbers exactly**, through an independent code
path: `rfa`'s decision rate 0.5333 (8 of 15 rounds), its realized spread rho in [1.306, 2.194], and its
maximum share movement 0.085860 against the published 0.0859. So the screen is a cross-check on
App. D.7 as well as a new measurement, and the three families beside `rfa` are new.

Excluded families, each with its reason recorded rather than dropped in silence: the `dose`/`doseS`/
`doseA`/`doseM` families **raise** without `adv_mask` and are the oracle-bound families this arm
replaces; `fltrust` as `d1` raises unless the caller supplies a server carrying a
`clean_holdout_dataset`, so it is unreachable through `measure()`; `trimmed_mean`, `coord_median`,
`krum` and `multi_krum` pass through unchanged as upstream stages and are therefore the identity.

## 3. The instrument: the boundary blend, and the two constants frozen here

`experiments/boundary_blend.py` blends the identity toward `rfa`'s coefficients along

    c_i(t) = (1 - t) * 1 + t * c_i^rfa  =  1 + t * (c_i^rfa - 1),      t in [0, 1]

where `c^rfa` is read back from the transformed norms the way `measure_admission.py:152` reads it,
rather than recomputed by a second implementation of Weiszfeld that could drift from the shipped one.
`c(0)` is exactly ones. Per round, the smallest grid `t` at which Krum's selection differs from its
`t=0` selection is `t*`, and the two arms are

* **arm A** at `t* - h` -- identity-side selection
* **arm B** at `t*` -- the flipped selection

**The two constants frozen here, before any ASR exists:**

    GRID_POINTS = 1001     so  h = 1e-3          the grid spacing, in experiments/boundary_blend.py
    SHARE_TOL   = 1e-6                           IMPORTED from experiments/measure_admission.py

`SHARE_TOL` is not redefined for this arm. Using Mode S's own tolerance rather than a laxer one chosen
to let this construction through is what makes the neutrality claim mean the same thing it means
everywhere else in the paper.

**Why the pair is informative and neutral at once, and why that is not a counterexample to App. D.6.**
Krum's decision is a step function of `t` and the coefficient share is continuous in `t`, so at `t*`
the selection flips while the coefficient vectors differ by exactly `h * (c^rfa - 1)`. The gap is
therefore **exactly linear in `h`** and is driven below any tolerance by shrinking the grid. The
construction is **uniformly** epsilon-neutral, which is strictly stronger than neutral for one
adversary mask: the difference vector `h * (c^rfa - 1)` does not depend on which clients are
adversarial, so *every* adversary subset's share moves by O(h). App. D.6 forbids **exact** neutrality
with informativeness; the blend's gap is positive, so the blend is consistent with that theorem rather
than a counterexample to it. What this arm adds is the **tolerance-parameterized** statement:
approximate uniform neutrality is achievable at any tolerance, and the price is paid in
informativeness, which is confined to rounds where a flip lies on the path.

**Oracle-freeness is structural here, not asserted.** `c^rfa` is computed from the update stack alone;
the grid is fixed in advance; the flip search compares **selections**, never labels; the runner passes
`adv_mask=None` and `d1_override="fedavg"`, so any dose family reached by this path would raise; and
the hook that applies the coefficients receives only `(ups, seed, round)` and never the adversary set.
`adv_share()` exists in the module only to *measure* the pair after the fact and is never called by
the construction.

**Self-check already run, and its measured values.** On seed 42 x 3 live rounds, the Gram-based
selection equals the shipped `krum_selection` at both `t=0` and `t=1` in every round; `c(0)` is exactly
ones; 1 of 3 rounds carried a flip (`t* = 0.892`); and on that flip round the selection changed 0 -> 3
under the **shipped** statistic on materialized stacks, at coefficient gap 4.184e-04 and **share gap
5.109e-07**, which is below the frozen `SHARE_TOL` of 1e-6. That single measured pair is the existence
proof that the instrument's two requirements can hold simultaneously. It is a harness measurement, not
a result, and no ASR was computed for it. The check is deterministic: re-running it reproduces all four
values bit-identically.

**The dose rate is bounded below by a number already measured, not guessed.** A round whose selection at
`t=1` differs from its selection at `t=0` must carry a flip somewhere on the path, so the screen's
endpoint-disagreement rate for `rfa` is a **lower bound** on the flip-round rate: **8 of 15 rounds
(0.533)**. Seed 42's flip round is round 2, which is exactly the one round of its three where the screen
records `krum_selection_changed = True` for `rfa`, and rounds 0 and 1 carry neither an endpoint
disagreement nor a flip. At 50 rounds per run the expected dose is therefore at least ~26 flip-rounds
rather than a handful. Rounds whose endpoints agree may still carry an even number of flips, so this is
a bound and not an estimate, and the realized `n_flip_rounds` is recorded per run.

## 4. The design

Both arms are **new**. Seeds **42--61 (n=20)**, both arms on every seed, 50 rounds, Krum as `d2`,
committed pixel backdoor, CIFAR-10, `cifar_cnn`, N=10, K=5, f=0.2, tau=5.0, alpha=0.5.

| arm | per-round coefficients | what it holds |
|---|---|---|
| **A** | `c(t* - h)` | identity-side selection |
| **B** | `c(t*)` | flipped selection; coefficients within `h * max\|c^rfa - 1\|` of A's |

**Rounds with no flip on the path carry no dose.** Both arms receive the identical coefficient vector
`c(1)` and are bit-identical for that round. This is recorded, not hidden: `n_flip_rounds` is written
per run, and the instrument's dose is exactly "one boundary selection flip per flip-round, none
elsewhere".

**The primary estimand is the paired difference B - A at the same 20 seeds.** Both legs are new, so
there is no mixed-`n` comparison and neither leg is read from a published artifact. App. D.7's identity
arm in `results/oracle_free_channels/` is **context only** and is never a leg of the primary Δ.

## 5. The predictions, and the two verdicts read separately

**Δ = mean over seeds of (arm B ASR - arm A ASR), paired.**

Two verdicts are emitted **per arm pair** and are read **separately**. Neither may be reported as the
other, and an interval can sit inside the margin and still exclude zero:

    sign verdict     does the paired 95% CI exclude zero?
    margin verdict   is |Δ| < EQUIV_MARGIN = 0.15?

**Pre-registered prediction.** The decision channel alone, with the attenuation channel closed to
1e-6, **does** carry suppression: `sign = excludes zero`. This is the direction the paper's headline
reading implies, since a statistic disturbance that changes which client is admitted should cost
suppression.

**The branch that refutes us, named in advance.** If `|Δ| > 0.15` with the interval excluding zero in
the direction of *increased* ASR, the decision channel carries suppression with the attenuation
channel closed to tolerance, and App. D.7's confound is removed as an explanation of its +0.297. That
is a **contradiction** of this paper's headline reading, and it is reported as a contradiction in the
body rather than converted into a scope condition afterwards. We have converted a refutation into a
scope condition zero times and will not start here.

**The branch we expect to be live, and it is not a zero.** The dose is one boundary selection flip per
flip-round, and **the direction of that flip is not controlled**: at `t*` the argmin moves to whichever
client the boundary happens to lie against, which may be adversarial or benign, and may differ by
round. So the per-round effects can cancel and the arm may return an interval inside the margin that
contains zero. **That outcome is `unresolved`, not `no effect`,** it is not evidence of absence, and
the equivalence phrase the supplement defines against the stricter of our two interval conventions is
**not** applied to it. We state this now, before the numbers, because App. D.7's leg 2 came out this
way and its freeze had not said so in advance.

**No mechanism clause appears inside any verdict literal.** Verdicts are `excludes_zero` /
`contains_zero` and `inside_margin` / `outside_margin`. Any mechanism reading goes in its own field
beside the literal, with the measured direction printed next to it. A frozen literal that fuses a
label with a mechanism guess was backwards in sign once already in this project.

## 6. Gates, all imported and all live

    ACC_FLOOR     = 0.35    from experiments/run_targeted_dose.py
    EQUIV_MARGIN  = 0.15    from experiments/run_targeted_dose.py
    SHARE_TOL     = 1e-6    from experiments/measure_admission.py

* **Accuracy floor.** A rung whose clean accuracy falls below 0.35 is **void, not negative**, and its
  seed is dropped from the paired Δ with the exclusion and the retained-value Δ both reported. App.
  D.7's identity arm sat at mean accuracy 0.5158, so the floor is live rather than decorative.
* **Share gate, the one App. D.7 omitted.** Per round, the realized `|share_B - share_A|` on adversary
  rounds must be `<= SHARE_TOL`. This is recorded every round and asserted, not assumed. **If it fails
  on any round, the arm is typed confounded in the attenuation channel and reported as such**, by the
  same precedent the CIFAR-100 norm-clipping appendix (`app:normclip_cifar100`) uses to demote an arm on
  exactly this condition, which is the precedent App. D.7 cites against itself. It is not fixed by
  widening the tolerance.
* **Flip gate.** On every round typed as a flip round, the two arms' selections must differ **under the
  shipped `krum_selection` on materialized stacks**, not merely under the Gram scores that chose the
  pair. A flip found only by the shortcut is a shortcut bug.

## 7. Harness checks, whose verdicts go into the artifact

A check whose verdict is discarded leaves the claim it supports witnessed only by a terminal and costs
hours to repeat. All four verdict dicts are persisted into `results/oracle_free_decomposition/`:

1. **`stack_hook=None` is bit-neutral.** Recomputing a frozen `results/targeted_dose/` cell through the
   patched `run_one` returns Δ = (0, 0) on both accuracy and ASR.
2. **The blend at `t=0` reproduces App. D.7's identity arm** at a shared seed, Δ = (0, 0), which is the
   assertion that the hook path is faithful rather than merely present.
3. **The Gram selection equals the shipped `krum_selection`** at `t=0` and `t=1` on live stacks.
4. **Per round, the share gap is within tolerance and the selections differ**, both asserted with `==`
   / `torch.equal` rather than taken from a docstring.

The standing failure mode in this repository is a manipulation hook that is never called while the run
passes silently. It has happened once and is documented at the site of its fix:
`run_cross_distribution_compositions.py:155-161` records that omitting `manipulate_update` had made
`committed_scaling` bit-identical to `committed_pixel`, because the two attacks share `poison_dataset`
and that call is the only thing distinguishing them. Nothing raised; the runs simply measured the wrong
attack. Check 2 is the one that would catch a recurrence here, because a `stack_hook` that is never
invoked leaves arm B equal to arm A and the ladder still completes.

## 8. The one shared-code change

`run_one` (`experiments/run_targeted_dose.py`) gains **`stack_hook=None`**, applied to `ups` inside the
round loop immediately after `apply_adversary` and immediately before `generic_compose`. It receives
`(ups, seed, round)` and **never** the adversary set. `stack_hook=None` leaves every existing call site
bit-identical, which harness check 1 proves rather than assumes.

This is an extension of the one exception already granted for `d1_override`, and it is necessary rather
than convenient: `d1_override` is a **string**, matched against literals and regexes inside
`apply_d1_transform`, so a per-round computed coefficient vector cannot flow through it without editing
`apply_d1_transform`, which is forbidden. The alternative -- duplicating the 50-round FL loop in a new
runner -- is worse for the reason in §7.

## 9. What this arm cannot establish, fixed now so it cannot be widened later

* **One aggregator, one attack, one dataset, one upstream transform, one architecture.** No statement
  of the form *the negative fails without an oracle*, or its converse, is licensed in general. What is
  licensed is *in this cell, without adversary identity, with the coefficient share held to 1e-6*.
* **The sentence "the negative reproduces oracle-free" may not be written anywhere, under any
  outcome.** App. D.7 refuted its own prediction and that refutation stands unamended.
* **The contrast is local, at the decision boundary.** Arms A and B are both nearly-`rfa` stacks at
  adjacent points on the blend path. The contrast is "just below this round's first selection flip"
  against "just above it". It is **not** "identity against a statistic disturbance" and must not be
  reported as one.
* **This arm is an instrument, not a deployable defense.** No deployed Krum interpolates its input
  stack toward a geometric-median reweighting and stops one grid step short of a selection flip. The
  adversary-identity objection is answered; the deployability objection is not, and stays open exactly
  where the rest of the paper leaves it.
* **No second dataset.** This arm is CIFAR-10. No composed-pair ASR result on a second dataset exists
  anywhere in this paper, and this arm does not change that.
* **The attack is non-adaptive by construction,** so nothing here concerns adaptive robustness.
* **No row is pooled with any published row.** All 40 runs behind every number are new.
