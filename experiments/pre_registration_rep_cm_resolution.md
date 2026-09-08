# Pre-registration amendment: Rep->CM committed-pixel cell, seed resolution n=5 -> n=30

Amends the development-set protocol of `experiments/run_all_compositions.py` and its `n=5` top-up
`experiments/pre_registration_dose_seed_topup.md`. It **adds seeds to one cell and changes nothing
else**. Every decision rule below is the existing one, restated verbatim, not revised.

This must be git-committed and `PREREG_COMMIT` in `experiments/run_rep_cm_resolution.py` set to the
resulting hash before any run. The runner refuses to start otherwise.

## Why this amendment exists

The criterion's precision is `2/2`: of the two pairs `C1 & C2` certifies in distribution,
`foolsgold->coord_median` (0.107) and `reputation_then_coord_median` (0.423), both measured LOW. The
paper already discloses that **one of those two is not resolved below the threshold**
(`paper/main.tex:806`):

> One certified pair is not resolved below threshold. Rep->CM's interval `[0.231, 0.616]` crosses
> `0.5`, and one of its five seeds reaches `0.665`.

A reviewer named this as the single biggest empirical weakness: the positive class contains two
examples and one is statistically unresolved, so "100% precision" rests on one clean case and one
whose uncertainty admits the HIGH side. That is a **power** problem in a single cell, not a design
problem, and it is the only headline claim in the paper that more seeds on one cell can settle.

Recomputed from the artifacts (`analyze_headline_cis.py`, seeds 42--46):

| seed | 42 | 43 | 44 | 45 | 46 |
|---|---|---|---|---|---|
| ASR | 0.431778 | 0.234222 | 0.390667 | 0.664667 | 0.395333 |

`n=5`, mean `0.423333`, sd `0.154861`, 95% t-interval `[0.231, 0.616]`, mean clean accuracy `0.7637`.

## What is added

- **New seeds: `47`--`71`, 25 of them**, named here before any run. The upper end is not arbitrary:
  seeds `42`--`71` are exactly the range of the `fg_rfa_flagship` study, the paper's largest
  measurement of any composition. At `n=30` the pair the criterion **certifies** is measured at the
  same seed count as the pair it **rejects**, so the two are read at equal power.
- **One pair, one attack**: `reputation_then_coord_median` under `committed_pixel` only. The label is
  set by the max-committed ASR and the scaling arm sits at `0.019844` over the same five seeds, with
  its largest single seed at `0.0332`, more than an order below the threshold, so it cannot bind and
  adding seeds there would buy nothing.
- 25 runs at a **measured** 20.5 min/run (`results/headline_seed_topup.log`: 12 runs of these same
  cells in 4.0 h) = **~8.5 h**.
- Nothing else. Same `N=10, K=5, f=0.2, alpha=0.5, 50 rounds`, same `cifar_cnn`, same `run_one`
  imported from `run_all_compositions` rather than forked.

### Why n=30 and not n=20

Sized before running, from the frozen sd, and recorded here so the budget cannot be read as chosen
after the fact:

| n | new runs | hours | 95% half-width | upper bound | clears 0.5 |
|---|---|---|---|---|---|
| 15 | 10 | 3.4 | 0.0858 | 0.5091 | no |
| 20 | 15 | 5.1 | 0.0725 | 0.4958 | by 0.004 |
| 25 | 20 | 6.8 | 0.0639 | 0.4873 | yes |
| **30** | **25** | **8.5** | **0.0578** | **0.4812** | **yes** |

`n=20` clears the threshold by `0.004`, which one unlucky seed erases. `n=30` clears it by `0.019`
and needs no further amendment, which matters because **the seed count cannot be extended after
seeing the result** -- that would be optional stopping, and it would void the test.

## Decision rules: UNCHANGED, evaluated at n=30

The label rule is the frozen one: **LOW iff the mean max-committed ASR is below `0.5`**
(`THRESH = 0.5`, `experiments/analyze_condition_ablation.py:50`, the same constant that scores every
pair in the study). No new constant is introduced. The only thing this
amendment adds is that the same rule is now read with an interval that has the resolution to answer
it:

| outcome at n=30 | verdict |
|---|---|
| mean `< 0.5` and 95% upper bound `< 0.5` | **RESOLVED LOW.** Both certified pairs are resolved below threshold, and the paper says so with the interval |
| mean `< 0.5`, 95% upper bound `>= 0.5` | **STILL UNRESOLVED at n=30.** Reported as such, in the same words as now, with the wider evidence base stated. No claim is upgraded |
| mean `>= 0.5` | **THE LABEL FLIPS.** The certified set becomes one pair, the criterion's precision is reported as `1/2`, and the flip is stated in the same sentence as the count |

Accuracy gate: mean clean accuracy `>= ACC_FLOOR = 0.35` over the 30 seeds, or the cell is
uninterpretable and no verdict stands. The frozen five average `0.7637`, so this is a tripwire, not
an expectation.

**This is therefore a test the paper's headline can fail.** The pre-run estimate of the failure
probability, from the frozen mean and sd, is about `0.3%` for a flip and roughly even odds against
the "still unresolved" row; neither is a prediction, and both are reported if they happen.

## What we commit to reporting

1. **Both counts, always.** The paper reports the `n=30` mean and interval and the frozen `n=5`
   `0.423 [0.231, 0.616]` beside it. The `n=5` number is never silently replaced.
   `results/all_compositions/summary.json` and `results/headline_seed_topup/summary.json` are not
   rewritten -- this writes to `results/rep_cm_resolution/`, and both frozen files' md5s are recorded
   before and after to prove it.
2. **A flip is reported as a flip**, in the same sentence as the count, exactly as the `cos_krum`
   `n=8` flip is (a C1 input read `0.300` over the frozen 3 seeds and `0.584` at `n=8`, and the paper
   reports the flip rather than the frozen value). That is the precedent this amendment commits to.
3. **No threshold is refitted.** `0.5` is the suite's existing constant. If the result lands above
   it, the threshold does not move to accommodate it, and neither does the accuracy gate.
4. **The pre-registered PASS/FAIL labels and the `16/18` and `22/24` tallies are not relabelled.**
   They were frozen at `n=3`/`n=5`; this reports a resolution of one cell alongside them, the way
   every other top-up in the paper does.
5. **Per-seed values for all 30 seeds** go in the appendix, as every other topped-up arm's do.
6. **If the mean drifts but stays below `0.5`**, the strategy table's mean-ASR column
   (`main.tex:267`) is recomputed rather than left stale, and the recomputation is stated.

## Guards that must pass before the first run

1. **The frozen cell must reproduce.** Seed `42` is re-run first and must return ASR
   `0.43177777777777776` exactly. `run_one` is fully seeded (`torch.manual_seed`, `np.random.seed`)
   and this is the same function that produced the frozen value, so an inexact match means the
   environment or the code path moved and the top-up is not comparable. The runner aborts.
2. **The attack path is `committed_pixel`, which needs no `manipulate_update` hook.** The silent
   failure this repo has already suffered once (`results/femnist_c1_inputs_scaling_INVALID/`, where a
   runner omitted `attack.manipulate_update` and a whole wave came back bit-identical to the pixel
   arm) applies to `committed_scaling`, not to this cell. Guard 1 covers it directly anyway: a wrong
   path cannot reproduce `0.4318`.
3. **The runner reuses `run_all_compositions.run_one` by import, never a copy.** Asserted at start-up.

## What this amendment does not do

- It does not add a pair, a defense, an attack, a dataset or a seed to any other cell.
- It does not touch any frozen artifact, any published table, the six-cell comparability
  pre-registration and its `[OK] ... f16083b` gate, or `results/dose_femnist/summary.json`.
- It does not change `ASR_THRESHOLD`, `ACC_FLOOR`, or any condition definition.
- It does not extend the seed count after the result is seen, under any outcome.
