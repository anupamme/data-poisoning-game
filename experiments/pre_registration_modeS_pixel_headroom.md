# Pre-registration: Mode S on the cell where both margin legs bind and admission is off the floor

**Frozen before any result in `results/modeS_pixel_headroom/` exists.** That path does not exist at the
time this file is committed. `experiments/run_modeS_pixel_headroom.py` refuses to start until this file
is git-committed, its `PREREG_COMMIT` matches `git log -1 --format=%h` for this path, and
`git status --porcelain` for this path is empty.

**No ASR has been measured for this arm at the time of this commit.** Two quantities about this cell are
already on disk from earlier, independently pre-registered arms and are imported rather than re-measured:
the identity rung (§3) and the baseline admission (§2). Both are cited with their artifacts.

## 1. The defect this arm exists to repair, which is our own

The paper's flagship dissociation is Mode S on **Krum / committed scaling**. That cell adjudicates two
readings of preservation at once, and it is uninformative on **both** channels it is asked about:

* **Admission has nowhere to fall.** Krum admits `0.000` adversarial mass at every rung of that cell, in
  `0` of `12` adversary rounds. The paper already says so and already declines to lean on it:
  *the zero is a floor here*.
* **The margin's lower leg has nowhere to fall either, and this the paper did not say.** The arm's
  practical-equivalence reading is a two-sided TOST against a frozen `+/-0.15`. ASR is bounded below by
  zero, so the paired mean difference obeys
  `mean_i(asr_dosed_i - asr_identity_i) >= -mean_i(asr_identity_i)` for *any* dosed outcome whatsoever.
  On that cell the controlled design's identity mean is **0.0439**, so the largest fall arithmetically
  available is 0.0439 and the leg `Δ > -0.15` is satisfied before a single run happens. It is not a test.

`experiments/measure_margin_reachability.py` measures this across every equivalence reading in the paper
(`results/margin_reachability.json`): of the **4** readings the comparability ladder's own conjunct
admits, **3 have an arithmetically satisfied lower leg** -- the flagship (identity 0.0439, one-sided bound
`+0.0124`), its EMNIST replication (0.0242, `+0.0087`), and the flagship's outcome-gated twin (0.0618,
`+0.0739`). One is genuinely two-sided and keeps its reading: `coord_median / scaling`, outcome-gated,
identity 0.5186, CI `[-0.0775, +0.0813]`.

So the paper's headline dissociation is measured on the one kind of cell where neither of its two
verdicts can move downward. **This arm moves the cell and holds everything else.**

## 2. Why this cell, measured rather than chosen

**Krum / committed pixel backdoor.** Same aggregator, same dataset, same architecture, same N, K, f,
α, round count and seed grid as the flagship. Only the attack changes, and it changes both headrooms:

| quantity | flagship cell (committed scaling) | this cell (committed pixel) | source |
|---|---|---|---|
| baseline admission, adversary rounds | `0.000`, 0 of 12 | **`0.3333`, 4 of 12** | `results/oracle_free_admission.json` -> `baseline_admission_not_a_floor` |
| identity mean ASR (n=20) | 0.0439 | **0.32320555555555563** | `results/oracle_free_channels/summary.json` -> `verdicts.Delta_stat.identity_mean_asr` |
| largest fall available | 0.0439 < 0.15 | **0.3232 > 0.15** | arithmetic on the row above |
| is the lower TOST leg a test | **no** | **yes** | `results/margin_reachability.json` |

Krum selects exactly one client per round, so its admitted adversarial mass in a round is 1 or 0 and the
0.3333 is both the mean mass and the rate: an adversary is selected in 4 of the 12 rounds that carry one.
**Admission has somewhere to fall and the margin has somewhere to fall.** Neither is true on the cell the
flagship verdict is read from.

**The paper already names this cell as the remedy** in App. D.8, in these terms: the identity rung's mean
ASR is 0.3232 against 0.062 on the cell the published score-only control ran, *that difference in headroom
is the whole point of moving cells*, and it is why an arm here can return a verdict that one could not.
This arm is that arm.

**It is also App. D.8's own host cell**, where the oracle-free boundary arm returned **+0.3218**
(95% CI `[+0.1680, +0.4757]`, n=20) *against* this paper's headline reading, on a locally-constructed
transform. A masked Mode S arm on the same cell therefore discriminates between two live explanations of
that contradiction -- the **cell** or the **boundary construction** -- and it is pre-registered here without
a prediction about which, because we do not know.

## 3. The design

**Cell.** `d2 = krum`, attack `committed_pixel` (`backdoor_pixel`), CIFAR-10, `cifar_cnn`, the frozen
`FL_CONFIG`: N=10, K=5, f=0.2, α=0.5, 50 rounds, τ=5.0, `lr` and decay unchanged.

**Rungs.** The frozen endpoint convention, `LO = κ 0.0` and `HI = κ 2.0`, i.e. ρ = `exp(2κ)` of
`1.0` and `54.598150033144236`. The interior rungs (κ = 0.5, 1.0) are **not run**, and no
Jonckheere--Terpstra trend statistic is computed or reported for this arm. An endpoint contrast is what
the paper scores; a two-rung cell is not a four-rung ladder and will not be described as one.

**Seeds 42--61, n=20, on every leg. No mixed-`n` comparison anywhere.**

| leg | `run_one` arguments | source |
|---|---|---|
| **A** identity | `mode="S", val=0.0, score_only=False` | **imported** (§4), subject to harness check 1 |
| **B** full dose | `mode="S", val=2.0, score_only=False` | 20 new runs |
| **C** identity, score-only | `mode="S", val=0.0, score_only=True` | **imported**, same rows as A, subject to harness check 2 |
| **D** score-only dose | `mode="S", val=2.0, score_only=True` | 20 new runs, **conditional (§7)** |

**Two estimands, both paired at the same 20 seeds:**

    Δ_full  = mean_i (ASR_B(i) - ASR_A(i))
    Δ_score = mean_i (ASR_D(i) - ASR_C(i))

`Δ_score` is the magnitude-closed control: `score_only` scores the transformed stack and emits the
selected client's **original** update, so the coefficient reaches the statistic and not the payload. It is
the fourth cell of the factorial already frozen in `pre_registration_score_only.md` /
`pre_registration_emit_only.md`, run here at n=20 on an informative cell. It is **not** a new method.

## 4. The imported identity rung, and the check that licenses it

`apply_d1_transform` returns the update list **unwrapped** at `val == 0.0` (`run_all_compositions.py`,
mode-S branch) *before* the permutation is drawn, and the permutation when it is drawn uses a local
`np.random.default_rng([seed, round])` rather than the global stream. So `doseS_kappa0.0_then_krum` is the
same computation as a `fedavg` pass-through at the same seed, and `results/oracle_free_channels/`'s
`fedavg_then_krum|committed_pixel` rows (seeds 42--61) are this arm's identity rung.

**One asymmetry makes this a blocking check rather than a comment.** The imported rows ran with
`adv_mask=None` (via `d1_override="fedavg"`); this arm passes the real mask, because `d1_override` is
unset. So:

**Harness check 1, blocking.** `run_one(seed=42, mode="S", d2="krum", attack_name="committed_pixel",
val=0.0)` must reproduce the imported row **bit-for-bit**:

    published   acc = 0.3962   ASR = 0.7787777777777778
    tolerance   0.0 on both, i.e. Δ = (0, 0) exactly

That pair is triply attested already: it is the `fedavg_then_krum|committed_pixel` seed-42 row, it is
`results/prospective_pilot/summary.json`'s Krum-standalone row, and App. D.7's harness check
`identity_arm_is_krum_standalone_across_suites` recorded Δ = (0, 0) between them. **If this check fails,
the import is void: legs A and C are re-run as 40 further runs and no reconciliation is attempted.**

**Harness check 2, blocking.** The same call with `score_only=True` must reproduce the same pair, Δ = (0,
0). At the identity there is no coefficient to split between the score channel and the emitted update, so
this should hold by construction -- which is exactly why it is asserted rather than assumed.

**Harness check 3, blocking.** Mode S pins the adversary: on **every one of the 50 rounds**, every
adversarial coefficient must be exactly `1.0` and the adversarial share of coefficient mass must equal
`n_adv/K` to float read-back. Asserted per round, not sampled. This is the premise the entire Mode S
family rests on, and this cell has never carried a Mode S arm before.

**All three verdict dicts are persisted into `results/modeS_pixel_headroom/summary.json`.** A check whose
verdict is discarded leaves the claim it supports witnessed only by a terminal and costs hours to repeat.

## 5. Gates, all imported, all live

    ACC_FLOOR     = 0.35    from experiments/run_targeted_dose.py
    EQUIV_MARGIN  = 0.15    from experiments/run_targeted_dose.py
    t_crit                  from experiments/analyze_headline_cis.py   (never T95[n-1])

* **Accuracy floor.** A rung whose clean accuracy falls below 0.35 is **void, not negative**; its seed is
  dropped from the paired Δ with the exclusion and the retained-value Δ both reported. The identity rung's
  mean accuracy is 0.515795 and its **minimum is 0.3962**, so the floor is live rather than decorative and
  one seed sits within 0.046 of it.
* **The frozen conjunction stays attached.** The paper's margin reading is *the margin conjoined with
  every rung mean below 0.5*. The identity rung is 0.3232, so a rise past `+0.1768` breaks the conjunction
  independently of the margin, and that is reported when it happens.

## 6. Three verdicts per estimand, read separately

Emitted per estimand. **None may be reported as another**, and they are genuinely independent: an interval
can sit inside the margin and still exclude zero, and a one-sided bound can hold while a two-sided reading
fails.

    verdict_sign      does the paired 95% CI exclude zero?          excludes_zero / contains_zero
    verdict_margin    is |Δ| < EQUIV_MARGIN = 0.15?                 inside_margin / outside_margin
    verdict_upper     is the 95% upper bound < EQUIV_MARGIN?        non_increase_bounded / not_bounded

**`verdict_upper` is the leg that is a test on the flagship cell, and `verdict_margin` is the one that
becomes a test here.** Recorded in the artifact before the numbers: on this cell the largest available
fall is 0.3232 > 0.15, so **the lower leg is a genuine test** and the two-sided reading, if it is earned,
means here what it only appeared to mean there. That is the property the cell was chosen for.

**No mechanism clause appears inside any verdict literal.** Mechanism goes in its own field beside the
literal with the measured direction printed next to it. A frozen literal that fused a label with a
mechanism guess was backwards in sign once already in this project.

**No prediction is registered for `Δ_full`'s sign.** We have two measurements pointing opposite ways on
this cell -- the flagship's `-0.010` under a masked dose on the scaling attack, and App. D.8's `+0.3218`
under an oracle-free construction on this very cell -- and registering a direction now would be choosing
which of our own results to believe. What is registered is the branch structure:

* **Confirming.** `Δ_full`'s interval contains zero and `|Δ| < 0.15` with the lower leg a real test. Then
  the dissociation holds on an informative cell at n=20, both verdicts mean what they say, and App. D.8's
  `+0.3218` is attributable to its boundary construction rather than to this cell.
* **Refuting, named in advance.** `Δ_full`'s interval excludes zero with `|Δ| > 0.15` toward increased
  ASR. Then the flagship dissociation fails on the one cell where admission is live and both margin legs
  bind, and the published negative is specific to a floor cell. **That is reported as a contradiction in
  the body and is not converted into a scope condition afterwards.** We have converted a refutation into a
  scope condition zero times and will not start here.
* **The third branch, and it is the one to expect.** The interval excludes zero while the upper bound
  stays inside `+0.15` -- a small, consistent rise, the shape `coord_median / pixel` already shows at
  `+0.125 [+0.095, +0.155]`. Then `verdict_sign` is `excludes_zero` and `verdict_margin` is
  `inside_margin`, non-increase **fails** and practical non-increase holds, and the report must say both.
  It is not an equivalence result and the equivalence phrase the supplement defines against the stricter
  of our two interval conventions is **not** applied to it.
* **Unresolved.** Interval contains zero and is wider than the margin. That is `unresolved`, not
  `no effect`, and it is not evidence of absence.

## 7. The staging rule, fixed now so it cannot be read as selective reporting

Leg **B** runs first (20 runs). Leg **D** runs **only if `Δ_full` does not fire the refuting branch.**

The reason is that `Δ_score` is a *decomposition* of `Δ_full`: it asks which channel carries an effect.
If `Δ_full` refutes, the decomposition of a refutation is a separate question from the refutation, and
this pre-registration does not cover it -- the honest report is the refutation, with the decomposition
named as unrun.

**This rule is stated before any number exists, and it is symmetric in the only sense that matters: it
cannot suppress a result that goes against us.** The refuting branch is the branch that *stops* further
work, so the staging can only ever withhold a control that would have followed a favourable result. If
leg D is not run, the artifact records `leg_D_run: false` with the reason, and no sentence anywhere may
describe the score-only control on this cell as having been run.

## 8. No shared code changes

`run_one` already reaches this cell through `(mode="S", d2="krum", attack_name="committed_pixel",
val, score_only)`. `ARMS_S` drives only `run_targeted_dose.py`'s own loop and is not consulted. Neither
of the two previously-granted keyword exceptions (`d1_override`, `stack_hook`) is used, and
`run_targeted_dose.py`, `run_all_compositions.py`, `measure_admission.py`, every existing runner and
every existing pre-registration are **untouched**. This arm's runner imports `run_one`.

## 9. What this arm cannot establish, fixed now so it cannot be widened later

* **No admission quantity is produced.** `run_one` returns `(accuracy, asr)` and nothing else; admission
  is measured by the `measure_admission*` family, which this arm does not run. The `0.3333` / 4-of-12
  premise is **imported** from `results/oracle_free_admission.json` and is not re-measured, and the
  artifact says so rather than letting a reader infer an admission result from a cell chosen for its
  admission headroom. `summary[...|admission]` is `0.0` at every identity rung **by construction** and is
  never read as a level.
* **This arm is masked.** It reads adversary identity by construction, like every Mode S rung in the
  paper. It answers nothing about the oracle-free objection and does not weaken App. D.8's contradiction,
  which stands as measured. **The sentence "the negative reproduces oracle-free" may not be written
  anywhere, under any outcome.**
* **One aggregator, one attack, one dataset, one architecture, two rungs.** No claim of the form *the
  dissociation holds generally* is licensed. What is licensed is *in this cell, at n=20, with both margin
  legs binding and admission off the floor*.
* **This changes no published row.** The flagship's `-0.010 [-0.032, +0.012]` stands exactly as
  published, with its lower leg relabelled as arithmetic rather than retracted. No frozen threshold, seed
  list, interval convention or published verdict is revised, and this file is not amended after the
  numbers exist.
* **No second dataset.** CIFAR-10. No composed-pair ASR result on a second dataset exists anywhere in
  this paper and this arm does not change that.
* **The attack is non-adaptive by construction**, so nothing here concerns adaptive robustness.
* **Every row behind `Δ_full` and `Δ_score` is either a new run or an imported identity rung that a
  blocking bit-equality check licensed.** Nothing is pooled across seed sets.
