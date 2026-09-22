# Pre-registration: the full 42-pair composition suite on CIFAR-100

**Status: frozen before any run. Written before `results/cifar100_composition_suite/` existed.**

## What this arm is

A review named a *"full composition replication on a second dataset"* as one of two **especially
important** requirements, and objected that the existing CIFAR-100 evidence is an intervention ladder
rather than a suite. This arm is the suite: the **whole 42-pair menu**, both committed attacks, on
CIFAR-100.

The 42 pairs are the union of the two existing waves, imported and **not re-listed**:

- `experiments/run_all_compositions.py`'s `PAIRS` -- **18** ordered pairs, the development wave
- `experiments/run_wave2_held_out.py`'s `PAIRS` -- **24** ordered pairs, the held-out wave

Verified at freeze time: `len(set(map(tuple, PAIRS_1)) | set(map(tuple, PAIRS_2))) == 42`, which is
the same 42 the paper's `38/42 = 90.5%` is computed over (`paper/main.tex:2441`). Both modules import
without creating anything under `results/`: each calls `os.makedirs(..., exist_ok=True)` at module
level on a directory that already exists, which is a no-op and not a write.

`generic_compose` and `apply_d1_transform` are imported from `run_all_compositions.py` **unchanged**.
Its `run_one` hardcodes `"cifar10"` at `:602` and is not editable, so this arm supplies its own loop
with `dataset="cifar100"` and calls the shared primitives. Nothing in either existing runner is
modified and nothing is copied out of them.

**Scale: 42 pairs x 2 attacks x 5 seeds (42--46) = 420 runs.** That is the largest arm this paper has
run, and gate 0 below exists so it is not spent on a result that is uninformative by construction.

---

## Gate 0 (feasibility): the 20-run probe that decides whether the 420-run suite can say anything

**This gate is the reason this document exists and it runs first.** It is written down because the
existing CIFAR-100 evidence already shows the suite is at serious risk of a **degenerate confusion
matrix**, and discovering that after 420 runs would be a waste of the compute and would tempt a
post hoc reading of a number whose base rate is 100%.

### The problem, from existing artifacts, before any new run

The screen predicts, for each pair, whether the composed ASR is HIGH or LOW. Its informative
quantities are precision and recall over the **LOW** pairs. On CIFAR-10 there are 5 LOW pairs of 42
(the 12% base rate the paper already discloses), and `results/all_compositions/summary.json` plus
`results/wave2_held_out/summary.json` say exactly which, by `max_committed_asr`:

| LOW pair (CIFAR-10) | `max_committed_asr` | wave |
|---|---|---|
| `foolsgold_then_rfa` | 0.0454 | 1 |
| `foolsgold_then_coord_median` | 0.1310 | 1 |
| `reputation_then_coord_median` | 0.3522 | 1 |
| `norm_clip_then_coord_median` | 0.4334 | 1 |
| `rfa_then_coord_median` | 0.4853 | 1 |

All 5 are in wave 1; **wave 2's 24 held-out pairs contain 0 LOW pairs**, its minimum being
`trimmed_mean_then_coord_median` at 0.5186. **4 of the 5 have `coord_median` as `d2`**, and
`foolsgold` or `reputation` is `d1` in 3 of the 5.

Now the CIFAR-100 side, from `results/cifar100/per_seed_results.json` (n=3, `cifar_cnn`), scored by
this paper's own power rule -- ASR < 0.5 at accuracy >= `ACC_FLOOR` (`paper/supplementary.tex:247`,
fixed before that freeze):

| CIFAR-100 standalone | accuracy | ASR | admitted |
|---|---|---|---|
| `backdoor_pixel_coord_median` | 0.3960 | 0.6731 | no |
| `model_scaling_coord_median` | 0.3812 | 0.7009 | no |
| `backdoor_pixel_trimmed_mean` | 0.3953 | 0.7330 | no |
| `backdoor_pixel_fedavg` | 0.3902 | 0.8513 | no |
| `backdoor_pixel_norm_clip` | 0.3894 | 0.8487 | no |
| `backdoor_pixel_rfa` | 0.3885 | 0.8858 | no |
| `model_scaling_norm_clip` | 0.3686 | 0.8926 | no |
| `model_scaling_trimmed_mean` | 0.3634 | 0.8930 | no |
| `model_scaling_rfa` | 0.3760 | 0.9085 | no |
| `model_scaling_krum` | 0.2532 | **0.0044** | no -- accuracy below floor |
| `model_scaling_fedavg` | 0.0171 | 0.6584 | no -- collapsed |

**0 of the 14 measured CIFAR-100 standalone cells are admitted.** On CIFAR-100 with `cifar_cnn` at 50
rounds, no measured defense suppresses either committed attack: the one cell with real suppression
(`model_scaling_krum`, ASR 0.0044) collapses accuracy to 0.2532, and Krum is not in the 42-pair menu
anyway. In particular **`coord_median`, the `d2` carrying 4 of the 5 CIFAR-10 LOW pairs, is a
non-suppressor on CIFAR-100.**

If the suite then returns 0 LOW pairs of 42, the confusion matrix is degenerate: base rate 100%, the
constant-HIGH predictor scores 42/42, **recall is undefined because there are no positives to
recall**, and precision is undefined if the screen also predicts none. That is not a replication of
the screen; it is a cell in which the screen cannot be evaluated. The paper may not report it as
answering the review's ask, and this document says so now rather than after the fact.

### What the probe is, and why it is decisive rather than merely cheap

**`reputation` and `foolsgold` are absent from the CIFAR-100 artifact entirely** -- no standalone
baseline exists for either against either attack. They are `d1` in 3 of the 5 LOW pairs, including
`foolsgold_then_rfa`, the lowest at 0.0454, where `d2` is `rfa` and so the suppression must be coming
from `foolsgold` itself. So the question "can any CIFAR-100 pair be LOW?" turns precisely on the two
aggregators for which no CIFAR-100 evidence exists. The suite is **not** provably degenerate; it is
degenerate on the five measured aggregators and unknown on the two that carry the LOW pairs.

**Probe:** `reputation` and `foolsgold`, standalone, CIFAR-100, `cifar_cnn`, both committed attacks,
seeds 42--46. **2 x 2 x 5 = 20 runs.** Written to `results/cifar100_composition_suite/probe.json`.

**Decision rule, frozen:**

- **If either aggregator is admitted** by the power rule on either attack -- ASR < 0.5 at mean
  accuracy >= `ACC_FLOOR` -- a LOW composed pair is reachable on CIFAR-100 and the **full 420-run
  suite runs.**
- **If neither is admitted**, the suite would have no reachable LOW pair, so it **does not run**, and
  the probe is reported as the finding it is: *on CIFAR-100 at this architecture and round budget, no
  defense in the menu suppresses either committed attack, so the screen's LOW class is empty and the
  screen cannot be evaluated on this dataset at all.* That is a real, citable external-validity
  result about the screen, it strengthens rather than softens the existing limitation, and it costs 20
  runs instead of 420. The abstract's CIFAR-100 clause is then scoped down to the ladder it actually
  rests on, exactly as the no-reproduction branch below requires.
- **Either way the probe's 20 runs are reported**, with both aggregators' accuracy and ASR, and the
  count of admitted cells stated as a fraction. A null probe is not silently dropped.

**The probe is not a power calculation and is not re-run with a different rule if it fails.** It uses
the paper's existing rule at the paper's existing constants, and `ACC_FLOOR` is imported, never
restated.

---

## Gate 1 (mandatory, and correcting a defect this codebase has already shipped): the adversary hook

`run_cross_distribution_compositions` once produced a `committed_scaling` arm **bit-identical** to its
`committed_pixel` arm because `manipulate_update` was never called, and it **failed silently**. Any
new composition runner is exposed to the same defect, so this arm gates on it.

**The obvious gate is the wrong gate, and this document records why before the run rather than
discovering it as a spurious failure.** "Assert the adversarial tensor changed" would **fail on the
correct configuration**: `BackdoorPixelAttack` does **not** override `manipulate_update`, so it
inherits the base identity (`attacks/attack_strategies.py:61-63`) and the pixel backdoor lives
entirely in `poison_dataset`. Verified at freeze time by introspection --
`BackdoorPixelAttack.manipulate_update is AttackStrategy.manipulate_update` is `True`, while
`ModelScalingAttack` overrides it with an elementwise `scale_factor` multiply. Half this arm's cells
are `committed_pixel`, so a change-assertion would abort 210 correct runs.

**The correct gate already exists and is imported, not rewritten:**
`experiments.adversary_hook.apply_adversary(..., verify=True)`, which pins the expected behaviour
**per attack** -- the update must change **iff** `defines_own_manipulate(atk)` -- and raises
otherwise. It catches both a hook that never fires and an attack object that is not the one the arm
thinks it is. This arm calls it with `verify=True` on **every** run, not only the first round, because
the cost of the check is a single tensor clone on one probe client.

Any `AssertionError` from it aborts that run and the arm reports the abort. It is never caught and
downgraded to a warning.

---

## Gate 2: the accuracy floor, which will disqualify cells on this dataset

CIFAR-100 at `cifar_cnn` / 50 rounds reaches only ~0.40 clean accuracy, against an imported
`ACC_FLOOR` of 0.35. **There is about 0.05 of headroom**, so a composition costing more than ~0.06
accuracy falls below the floor, and `results/cifar100_ne3_br/summary.json` already shows
`model_scaling` dipping to 0.338 and collapsing to 0.083 at one seed on a *single* defense. On
CIFAR-10 the analogous collapse is visible too: `coord_median_then_fedavg` runs at accuracy 0.1.

So a substantial fraction of the 42 pairs is expected to land below the floor. The frozen rule:

- A cell below `ACC_FLOOR` is **reported uninterpretable and counted**, with its accuracy printed. It
  is **not** dropped, **not** gated away silently, and **not** scored as HIGH merely because a
  collapsed model has a high ASR.
- The suite's headline fraction is reported as **`k/m` where `m` is the number of cells that cleared
  the floor**, with `42 - m` named explicitly in the same sentence. A fraction over a filtered
  denominator without its exclusion count beside it is the defect this rule exists to prevent.
- If `m` is small enough that the LOW class is empty among the surviving cells, the same degeneracy
  clause as gate 0 applies: the screen cannot be evaluated, and that is what is reported.

---

## Run order, resumption, and no silent truncation

420 runs is long enough that it may be interrupted. The order is frozen now so that **any partial
completion is a pre-specified subset rather than an arbitrary truncation**:

1. **Wave 1's 18 pairs first**, because all 5 CIFAR-10 LOW pairs are in wave 1, so wave 1 is where
   the LOW class can exist at all and where the replication question is decidable.
2. **Then wave 2's 24 pairs**, the held-out wave.
3. Within each wave: pairs in the imported list order, attacks in `["committed_scaling",
   "committed_pixel"]` order, seeds 42--46 ascending.

The summary is written after every run and the arm is resumable. **Progress is counted from `per_seed`
in the artifact, never from a `[n/420]` log index**, which counts resumed-and-skipped runs and can go
backwards across restarts.

If the arm stops early, the report states the completed pair count, the completed wave, and the fact
that the remainder is unrun -- as an incompleteness, not as a suite. **A partial suite is never
described as "the full 42-pair menu".**

---

## Primary outcomes and the branches, written in advance

The estimand is the **screen's agreement** on CIFAR-100: for each pair that clears the floor, whether
the composed `max_committed_asr` is LOW (< 0.5) or HIGH, against the pre-registered condition-based
prediction for that pair, plus the **base rate** and the **constant-HIGH baseline** on CIFAR-100
computed from the same cells.

**The base rate and the constant-HIGH baseline are reported in the same sentence as any accuracy
figure, in every position including captions and section titles.** This is the paper's existing
standing rule for the `~90%` figure and it applies to every number this arm produces.

- **The 5-of-7 design disagreement and the sign reversal reproduce on the suite.** The abstract's
  CIFAR-100 clause upgrades from a ladder to a suite, and the review's second especially-important
  ask is answered at suite scope.
- **They do not reproduce.** Reported as such. The abstract's CIFAR-100 clause is **scoped down to the
  intervention ladder it actually rests on**, and the external-validity limitation is **strengthened,
  not softened**. This is the branch the existing CIFAR-100 standalone evidence points at, and it is
  written here so that outcome is a pre-registered result rather than a disappointment.
- **The LOW class is empty (gate 0 fails, or gate 2 empties it).** No screening verdict. Reported as
  *the screen cannot be evaluated on this dataset*, with the probe or floor counts as the evidence.
  Not converted into a claim in either direction.
- **Gate 1 fires anywhere.** That run aborts and the arm reports the abort with the attack, pair and
  seed named. No result is reported from a run whose adversary hook did not behave as its attack
  defines.

---

## Standing prohibitions that bind this arm

- No write to any existing `results/` directory or existing `results/*.json`. Output goes only to
  **`results/cifar100_composition_suite/`**, a new directory.
- No edit to `run_all_compositions.py`, `run_wave2_held_out.py`, `run_targeted_dose.py`,
  `adversary_hook.py` or any other existing runner or pre-registration. The shared primitives are
  imported unchanged, which is why this arm is a new module.
- No threshold restated: `ACC_FLOOR`, `ADV_FRACTION`, `FL_CONFIG` and the attack-name mapping come
  from their existing homes.
- **This document is not amended after the numbers exist.** If a verdict literal here is right in its
  label and wrong in its stated mechanism, both are printed and neither is edited.
- No sentence produced from this arm may generalize beyond one dataset, one architecture, one round
  budget and the 42 pairs, and the two prohibited sentences about the oracle-free arm are not in
  scope here but remain prohibited everywhere.

## Commit protocol

This file is committed **alone**, before the runner is final and before any run. Then
`PREREG_COMMIT` in the runner is set to `git log -1 --format=%h`, and the runner refuses to start
unless **both** the log hash for this file matches it **and** `git status --porcelain -- <this file>`
is empty. `PREREG_COMMIT is None` is only half a guard: the moment the constant is set, the only test
that ever fires stops firing.
