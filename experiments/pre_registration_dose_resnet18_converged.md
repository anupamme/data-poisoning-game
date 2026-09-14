# Pre-registration: the ResNet18 replication re-run at a 4x longer horizon (Round 69, ninth review)

**Frozen before any write to `results/dose_resnet18_converged/`.** That directory does not exist when this
file is committed. `experiments/run_dose_resnet18_converged.py` refuses to start until this file is
git-committed and its `PREREG_COMMIT` is set to that hash.

## 1. The question, which is the review's and not ours

The ninth review objects to disclosure *(ii)* of the ResNet18 arm, `main.tex:1534`, which says of itself:

> **Clean accuracy is far below this architecture's own reach**: $0.536$--$0.557$ against the $0.66$--$0.76$
> the ResNet18 mini-sweep of §\ref{app:resnet18_replication_details} attains in the same regime, ResNet18 at
> $50$ rounds being far from converged. The gate is a floor and not a claim of a converged model, and no
> statement here is about converged training.

The paper states the limit plainly and then leaves it standing. The review's point is that a security
conclusion drawn on a model at 54% clean accuracy may not survive on the same model trained further, and it
is right that nothing on disk answers that. **This arm answers it by changing one thing: the round count,
from 50 to 200.** Dataset, architecture, `N` = 10, `K` = 5, `f` = 0.2, alpha = 0.5, the attack
(`committed_scaling`), the defense (Krum as `d_2`), the mode (S), the rungs and the seeds are all carried
over unchanged from `experiments/pre_registration_dose_resnet18.md` at `89046f6`.

**Published 50-round values, from `results/dose_resnet18/summary.json`, which this arm does not modify:**

| rung | mean ASR | mean clean acc | per-seed ASR (42/43/44) | per-seed acc |
|---|---|---|---|---|
| kappa = 0.0 (identity) | `0.0707` | `0.5358` | `0.1139` / `0.0504` / `0.0477` | `0.4955` / `0.5649` / `0.5470` |
| kappa = 2.0 (rho = 54.60) | `0.0296` | `0.5574` | `0.0076` / `0.0339` / `0.0474` | `0.5173` / `0.5906` / `0.5644` |

Primary at 50 rounds: `Delta = -0.0410`, verdict **FLAGSHIP NEGATIVE REPLICATED**, paired 95% interval
`[-0.183, +0.101]`, `m* = 0.183`.

## 2. Two gates that decide different things, and the reason they are separated

The parent arm's gate is imported unchanged. A second gate is new, and it decides a **label** rather than
validity. Keeping them apart is deliberate: the accuracy figure the review objects to is not the figure the
parent arm's gate was ever about.

- **Validity gate (imported, unchanged).** Standalone Krum must suppress the committed scaling attack on
  this architecture at usable accuracy: identity-rung mean ASR `< SUPPRESS_ASR = 0.5` with mean clean
  accuracy `>= ACC_FLOOR = 0.35`, and **every** rung must hold mean clean accuracy `>= 0.35`. If it fails
  the arm is **VOID, NOT NEGATIVE**: there is no suppression for the ladder to preserve, no verdict stands,
  and the arm licenses no sentence about horizons. Neither constant is new: `ACC_FLOOR` is imported from
  `run_targeted_dose` (`:111`) and `SUPPRESS_ASR` from the parent arm itself,
  `run_dose_resnet18.py:133`.
- **Convergence label gate (new, `CONVERGED_ACC = 0.65`).** The identity rung's mean clean accuracy at 200
  rounds must reach `0.65` for this arm to be **described** as evaluated near this architecture's own reach.
  `0.65` is not invented for the occasion: it is the bottom of the `0.66`--`0.76` band the paper's own
  ResNet18 mini-sweep attains in this regime, quoted at `:1534` above, rounded **down** so the gate is
  reachable rather than flattering.
  **If it fails, the pre-registered finding is: "four times the round budget does not bring
  Krum-aggregated ResNet18 to the accuracy this architecture reaches under averaging defenses in this
  regime,"** reported as the result, in the paper, with the accuracy actually reached.
- **The kappa = 2 leg runs either way.** `0.35` decides validity; `0.65` decides only the label. A 4x horizon
  is informative about the horizon even if it does not reach 0.65, and refusing the second leg on a label
  gate would throw away the comparison the review asked for.

**Why the convergence gate may well fail, said now rather than after.** The `0.66`--`0.76` comparison band
comes from norm-clip/reputation **mixture** policies, which average many client updates per round. Krum
selects **one**. So the accuracy shortfall at 50 rounds is plausibly an aggregation-rule effect rather than a
horizon effect, and 4x the rounds need not close it. This arm cannot separate those two explanations, and it
is not being run as though it could.

## 3. The primary rule, carried over unchanged

**`Delta = mean ASR(kappa=2) - mean ASR(kappa=0)`**, `n = 3` on seeds 42/43/44, scored against the suite's
existing `EQUIV_MARGIN = 0.15`. **No new constant.** Both legs of `Delta` are computed in the same call from
per-seed rows, and **`Delta` is refused, not caveated, if the two endpoints were scored on different seed
sets** -- a `Delta` across unequal seed sets is not a within-arm contrast.

| outcome | verdict |
|---|---|
| \|Delta\| < 0.15 | **THE NEGATIVE SURVIVES A 4x LONGER HORIZON** on this architecture. |
| Delta > +0.15 | **THE NEGATIVE IS HORIZON-SPECIFIC.** Statistic disturbance does move suppression on ResNet18 once trained further, and the paper's central claim must be scoped to short-horizon training **in the body**, not in a limitation. |
| Delta < -0.15 | Attenuation-side fall. **INDETERMINATE**, reported as such and **not** scored in our favour. |

Also reported, and fixed here: the paired 95% Student-t interval on the three per-seed differences, and
`m* = max(|lo|, |hi|)`, the smallest margin the arm's interval fits inside. At `n = 3` the interval will
again be wide, and **no equivalence claim at any margin tighter than the frozen `0.15` is made from this
arm**, exactly as disclosure *(v)* already refuses for the 50-round arm.

## 4. The informativeness condition: a small Delta can be a floor effect, and this arm is likely to have one

At 50 rounds the identity rung sits at ASR `0.0707`. A quantity that starts at `0.07` cannot fall by more
than `0.07`, so `|Delta| < 0.15` is attainable **by arithmetic alone**, without the mechanism doing anything.
That is the `cos_krum` failure mode the paper already discloses, and this arm inherits it.

**Pre-registered condition, and the expectation that it fails.** If the identity rung's mean ASR at 200
rounds is below `0.15`, then:

- the arm is reported as replicating **with the margin not binding**, and the phrase "the margin does not
  bind because the identity rung has less headroom than the margin" appears wherever the verdict does;
- it is **not** reported as a strengthened replication, and no sentence claims the 4x horizon made the
  negative more secure;
- the identity ASR at both horizons is printed side by side so a reader can see the headroom for themselves.

**We expect this condition to fail**, because 200 rounds is more likely to lower a suppressed ASR than to
raise it. Recording that expectation before the runs is the point: an arm whose favourable outcome is
partly guaranteed by a floor must say so in advance, or its favourable outcome means nothing.

**The one way this arm could become strongly informative** is if training further makes the model *more*
backdoorable, pushing identity ASR to `0.15` or above. Then the margin binds and the test is a real one. We
have no prediction either way and are not stating one.

## 5. The comparison is horizon-confounded, and no cross-horizon Delta is reported

200 rounds is **four times the backdoor injection exposure** of the published arm: the adversaries poison in
every round they are sampled, so a longer horizon changes both how well the model fits and how long the
attack runs. This arm therefore answers *does the within-arm dissociation survive at a longer horizon* and
**not** *what is ResNet18's converged ASR relative to the 50-round arm*.

**Consequently: no `Delta` is taken across horizons.** `ASR_200(kappa=0) - ASR_50(kappa=0)` is not computed,
not reported and not described, in either direction. The 50-round figures appear only as context, labelled
with their own round count. Every `Delta` is within one horizon, on one seed set, on two rungs.

## 6. Cost, priced from two measurements that disagree, with both stated

- The arm's own measured rate: `main.tex:1534`(iv) prices the published 6 runs at **13.4 h**, i.e.
  **2.23 h/run** at 50 rounds. Scaling linearly in rounds gives **~8.9 h/run** and **~54 h** for 6 runs.
- The mini-sweep's rate: `results/cifar10_mix_ratio_sweep_resnet18/summary.json` records `wall_time_s`
  5884--6205 over 15 runs, mean **6039 s = 1.68 h/run** at the same regime, giving **~6.7 h/run** and
  **~40 h**.

**The budget is stated as 40--54 h and the arm's own rate is the one planned against**, because it ran this
cell with this defense. Linear scaling in rounds is an assumption, not a measurement; the realized wall time
is recorded per run in the artifact and reported.

**Two endpoint rungs, not four, and therefore no trend test.** Same restriction and same reason as the
parent arm, at four times the price. The two rungs are never displayed or described as a four-rung ladder,
and the runner prints that restriction on every invocation.

**No optional stopping.** Seeds 42/43/44, fixed here, no interim look, no extension, no seed added after
seeing a result. If the run is interrupted, the `n` reached is reported as the `n` reached. Progress is
counted from `per_seed` entries in the artifact, never from a `[i/N]` log position, which counts
resumed-and-skipped runs and can move backwards across restarts.

## 7. Declared limitations, before the run

1. **One cell, one attack, one architecture, one horizon pair.** This widens no `N`, `K`, `f`, alpha,
   dataset or attack, carries no composition suite, and does not touch the sign-reversal cell.
2. **"Near convergence" is a claim about a measured accuracy, not about convergence.** Even at `0.65` the
   model is not converged in any formal sense; the gate says it reached the band this architecture reaches
   in this regime, and no more.
3. **`n = 3`.** The interval will be wide and `m*` will very likely exceed the frozen margin again. The
   remedy for that is seeds, not wording, and this arm does not spend them.
4. **The arm is an instrument, not a defense.** Mode S pins every adversary's coefficient at `c = 1`, which
   requires adversary identity. No deployable defense knows which clients are adversarial, and nothing here
   is proposed for deployment.
5. **A deviation from the round's own plan, recorded here.** The plan for this arm named a two-sided
   admissibility gate requiring identity-rung mean ASR inside `[0.15, 0.85]` "so neither leg is saturated."
   That gate is **not adopted**, because it contradicts the parent arm's own frozen validity gate: that gate
   *requires* identity ASR `< 0.5` for the arm to be interpretable at all, and the published value is
   `0.0707`. A window demanding ASR `>= 0.15` would void an arm that passed the rule it was frozen against.
   The saturation concern is real, so it is kept as §4's **informativeness condition** -- reported, never a
   void -- and the change is disclosed rather than quietly dropped.

## 8. Non-negotiables

1. **No frozen artifact is modified.** `results/dose_resnet18/`, `results/targeted_dose/`,
   `results/dose_response/`, `results/dose_femnist/`, `results/resnet18_admission.json` and
   `results/cifar10_mix_ratio_sweep_resnet18/` are read-only here. This arm writes
   `results/dose_resnet18_converged/summary.json` and nothing else.
2. **`run_one` is imported, not copied.** `experiments/run_targeted_dose.py:run_one` already takes an
   `fl_config` argument, and that argument is verified to be honored in the round loop before this arm runs:
   `cfg = fl_config or FL_CONFIG` at `:163`, and `cfg.num_rounds` drives `for rnd in range(cfg.num_rounds)`.
   **No new hook is added.** `KAPPAS`, `ACC_FLOOR`, `EQUIV_MARGIN` and `TOL` are imported from
   `run_targeted_dose` and `SUPPRESS_ASR` from `run_dose_resnet18` for the same reason, and the arm's rung
   list is asserted at import time to be a subset of the frozen ladder.
3. **A harness check runs before the arm**, re-running one frozen 50-round cell through the imported
   `run_one` with `fl_config=None` and asserting agreement with the stored per-seed row in
   `results/dose_resnet18/summary.json` to `< 1e-9`. A **value** comparison against stored rows, never an
   md5 of an artifact file: this repository has already shipped one arm that was bit-identical to another
   because a manipulation hook was never called and failed silently. **If it fails, the arm does not run.**
4. **Every comparison number is recomputed from per-seed rows at run time.** Nothing is transcribed.
5. **All three outcomes of §3, the two gates of §2 and the informativeness condition of §4 are written
   above and none is renegotiated afterwards.** If this refutes the negative it goes in the body as a
   refutation with this rule quoted, in the same place a replication would have gone. If the arm is void it
   is reported as void in that same place.
6. This file is committed before `results/dose_resnet18_converged/` is written; if that ordering cannot be
   demonstrated from `git log`, the arm is reported as non-prospective.
