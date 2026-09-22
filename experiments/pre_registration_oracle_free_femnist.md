# Pre-registration: the oracle-free decision contrast on a second dataset and architecture

**Status: frozen before any run. Written before `results/oracle_free_femnist/` existed.**

## What this arm is

A review asked for one intervention that identifies the decision channel **without reading adversary
identity**, on a **second dataset**, at **n >= 20**, independently pre-registered. This is that arm.

It applies the already-frozen boundary-blend instrument of
`experiments/pre_registration_oracle_free_decomposition_refined.md` (commit `947c000`,
`experiments/run_oracle_free_decomposition.py`, `experiments/boundary_blend.py`) unchanged, on
**EMNIST-byclass (`femnist`, 62 classes, 1x28x28) with `simple_cnn`** instead of CIFAR-10 with
`cifar_cnn`.

The instrument: `d1` is a pass-through (`fedavg`) carrying a per-client coefficient vector
`c(t) = 1 + t (c_rfa - 1)`, where `c_rfa` is RFA's Weiszfeld weight vector **computed from the update
stack alone**; `d2` is `krum`. Two arms straddle a Krum selection crossing located by bisecting the
**shipped float32** `krum_selection`: arm **A** at `c(t_lo)` (identity-side selection), arm **B** at
`c(t_hi)` (flipped selection). No adversary mask is passed to the construction at any point, so
oracle-freeness is enforced by the callee rather than intended by the caller.

## The cell

| | |
|---|---|
| dataset / model | `femnist` / `simple_cnn` |
| `d1` | `fedavg` pass-through; the blend **is** the upstream stage |
| `d2` | `krum` |
| attack | `committed_scaling` (`ATTACK_MAP` -> `model_scaling`) |
| seeds | 42--61, **n = 20** |
| FL config | `FL_CONFIG` imported unchanged (`num_clients=10`, `clients_per_round=5`, `num_rounds=50`), plus `ADV_FRACTION = 0.2`, a separate module constant, and `run_one`'s own `alpha=0.5` default -- three different homes, none of them restated here |
| output | `results/oracle_free_femnist/summary.json`, a **new** directory |

## Three things change relative to the CIFAR-10 oracle-free arm, and one relative to the FEMNIST arm

Stated plainly here so no later reading can present this as a single-variable replication.

Against `results/oracle_free_decomposition/summary.json` (CIFAR-10), **dataset, architecture and
attack all change.** The attack changes because it must, and it is this paper's own pre-existing
selection rule that changes it. That rule -- stated at `paper/supplementary.tex:247` as the power rule
of the dose ladder, and there recorded as "fixed before the freeze" -- admits a cell only where
standalone `d2` genuinely suppresses: **ASR < 0.5 at clean accuracy >= `ACC_FLOOR`**, because a
baseline already above 0.5 "has no suppression left to lose and will show `Delta` approximately 0
whatever the transform did: a ceiling effect, not preservation" (`supplementary.tex:101`).

Applied to `results/femnist/payoff_results.json`, that rule admits exactly one FEMNIST Krum cell:

| cell | accuracy | ASR | admitted by the rule |
|---|---|---|---|
| `no_attack_krum` | 0.7564 | 0.0000 | -- (reference) |
| `model_scaling_krum` | 0.7671 | **0.0270** | **yes** |
| `backdoor_pixel_krum` | 0.7627 | **0.6601** | no: above 0.5, a ceiling cell |

So `committed_pixel`, the CIFAR-10 arm's attack, is excluded on FEMNIST by a criterion this paper
fixed rounds ago and applied to four other cells, not by a judgment made for this arm.

**Both artifacts are read at run time and neither figure is transcribed into the runner.** Note that
the two artifacts this arm reads do **not** share a key name for ASR: `results/femnist/payoff_results.json`
stores it as `attack_success_rate`, while `results/dose_femnist/summary.json`'s per-seed records store
it as `asr`. The runner names each key at its own call site. A silent `.get("asr", 0.0)` against the
payoff matrix would read every ASR as zero and make every cell look admissible.

Against `results/dose_femnist/summary.json` (the paper's FEMNIST Mode-S replication, prereg
`experiments/pre_registration_dose_femnist.md`), **only the instrument changes**: same dataset, same
architecture, same aggregator, same attack, and Mode S's oracle-dependent pinning is replaced by the
oracle-free blend. That is the comparison this arm is for, and it is the one the review's ask is
about.

## Why not CIFAR-100, recorded before the run rather than after a failure

CIFAR-100 was the first choice and was ruled out on existing evidence.
`results/cifar100/per_seed_results.json` reads `no_attack_krum` at mean accuracy **0.2569**, min
**0.2328** -- **below the imported `ACC_FLOOR` of 0.35 with no attack present at all.** Every other
aggregator there clears it (`fedavg` 0.398, `multi_krum` 0.401, `trimmed_mean` 0.409,
`coord_median` 0.410, `norm_clip` 0.398, `rfa` 0.399), so this is specific to Krum: at K=5 Krum emits
one client's update per round, which on a 100-class problem never reaches the floor. This arm's
instrument bisects `krum_selection` and so cannot move off Krum. Running it on CIFAR-100 would have
produced a predictable accuracy-gate failure and no verdict. That is written down here so the dataset
choice is on the record as made in advance, not as dataset-shopping after a failed run.

A CIFAR-100 arm is still owed and is a separate, separately pre-registered piece of work at the
composition level, where the aggregators that clear the floor are the ones in play.

## Every threshold is imported, none is restated

A tolerance restated in the script that has to pass it is a tolerance chosen to be passed. The runner
imports, and this document names the source module rather than the value:

- `run_one`, `ACC_FLOOR`, `EQUIV_MARGIN`, `ATTACK_MAP`, `FL_CONFIG` from
  `experiments/run_targeted_dose.py`
- `SHARE_TOL` from `experiments/measure_admission.py` -- the module that defines Mode S's own
  tolerance
- `refined_boundary_pair`, `apply_coefficients`, `share_gap_sup`, `self_check`, `GRID_POINTS`, `H`,
  `SHARE_GAP_TARGET_DIVISOR`, `SHIPPED_BISECT_MAX_STEPS` from `experiments/boundary_blend.py`
- `t_crit` from `experiments/analyze_headline_cis.py`; never a literal table, because the small
  tables in this repository stop at df = 9

## Gates, all of which run before any ASR is scored

1. **Harness check, bit-identity.** `d1 = identity` then `krum` on `femnist`/`simple_cnn` at seed 42
   **is** Krum alone, and must reproduce `results/dose_femnist/summary.json`'s
   `doseS_kappa0.0_then_krum|committed_scaling` seed-42 cell **exactly**: accuracy
   `0.7645951359576352`, ASR `0.023899769324709396`. Read from the artifact at run time. Anything
   other than a difference of `(+0.00e+00, +0.00e+00)` means the two harnesses disagree on this
   dataset and **the arm does not run**.

   Bit-identity is a legitimate demand here only because the two configurations agree exactly:
   `results/dose_femnist/summary.json`'s `config` records `N=10, K=5, f=0.2, alpha=0.5, rounds=50`,
   which is `FL_CONFIG` + `ADV_FRACTION` + `run_one`'s `alpha` default, and its seed list is
   `[42, 43, 44]`, so seed 42 exists in it. Were any of those to differ, the correct precondition
   would be a tolerance and this document would have said so.
2. **The bisection invariant.** The shipped float32 `krum_selection` must differ between arm A and
   arm B on the bisected round. If it does not, there is no crossing and no contrast.
3. **The share gate.** The mask-free supremum coefficient-share gap, taken over all 30 nonempty
   proper subsets of the K=5 participants, must be at or below `SHARE_TOL`, driven to
   `SHARE_TOL/SHARE_GAP_TARGET_DIVISOR` by the bisection. This is what holds the attenuation channel
   closed without reading adversary identity. A supremum above `SHARE_TOL` means the arm is
   confounded in the attenuation channel and is **void, not interesting**.
4. **`n_flip_rounds` is recorded per run, not assumed.** The CIFAR-10 arm delivered a mean of 28.6
   (A) and 27.9 (B) crossing rounds of 50. Whatever this arm delivers is reported as measured.
5. **The accuracy gate.** Both arms must hold mean clean accuracy at or above the imported
   `ACC_FLOOR`. FEMNIST Krum sits near 0.76 standalone, so this is expected to clear with room; if it
   does not, the cell is uninterpretable and no ASR verdict stands.
6. **The ceiling/floor gate, using the existing `EQUIV_MARGIN` and no new constant.** If both arms'
   mean ASR land within `EQUIV_MARGIN` of the *same* boundary -- both at or below `EQUIV_MARGIN`, or
   both at or above `1 - EQUIV_MARGIN` -- the contrast had no room to move and the result is reported
   as a ceiling or floor artifact, **not** as evidence in either direction.

   This gate exists because the arms' ASR **cannot be predicted from any standalone payoff figure**,
   and this document will not pretend otherwise. `d2` is Krum throughout and `d1` is a synthetic
   coefficient vector, so neither `model_scaling_krum` (0.027) nor `model_scaling_rfa` (0.999) bounds
   what the blend arms read: RFA supplies coefficients here, it never aggregates. The only comparable
   quantity in existence is the CIFAR-10 arm's own pair, A = 0.248 and B = 0.570, and one dataset is
   not a prior for another. The risk that FEMNIST's arm A sits near the floor is real, unquantified,
   and handled by this gate rather than by a guess.

## Primary estimand and decision rule

`Delta_decision = mean ASR(arm B) - mean ASR(arm A)`, paired per seed over seeds 42--61, n = 20,
reported with the two-sided 95% t interval at `t_crit(20)`.

The published CIFAR-10 reading of this same estimand, for reference and not as a prediction:
**+0.3218**, 95% CI **[+0.168, +0.476]**, verdict *"ASR MOVES on this leg (interval excludes zero),
sign positive"*.

**Outcome branches, written now so the prose cannot be written to fit the data.**

- **Interval excludes zero, sign positive.** Krum's decision channel moves ASR on a second dataset
  and architecture **without adversary identity** and with the coefficient share held to
  `SHARE_TOL`. The licensed statement is the arm's own scope: this cell, this aggregator, this
  attack, this dataset, this architecture, no adversary identity, share held to `SHARE_TOL`. The
  contrast is **local**, between two adjacent points on a blend path either side of a selection
  crossing, and must not be reported as identity against a statistic disturbance.
- **Interval contains zero.** On this cell the oracle-free decision contrast does not move ASR. The
  CIFAR-10 `+0.3218` is then **scoped to CIFAR-10 in the body**, not in a footnote, and
  `tab:evidence`'s oracle-free row is restated at the scope the evidence licenses.
- **Interval excludes zero, sign negative.** Reported as measured, with the direction stated, and
  **not scored in our favour**.
- **Any gate in the list above fires.** No ASR verdict. The arm is reported as attempted and
  uninterpretable, naming the gate that fired and its measured value. A gate failure is not converted
  into a finding, and no threshold is adjusted to clear it.

## What this arm does not do, and two sentences that may not be written

Exact oracle-free share-neutrality is impossible: by the impossibility result this paper already
proves, no oracle-free member of the positive per-client rescaling class combines share-neutrality
with informativeness, because it closes a channel downstream of the coefficient for every client at
once. This instrument is therefore the **approximate** boundary blend, and the approximation is
exactly what gate 3 measures. If the instrument were exact this arm would be impossible by the
paper's own theorem, and that dependency is the reason the share gate is a premise rather than a
diagnostic.

Whatever this arm returns, neither of the following may be written, in the paper, in the supplement,
in any artifact or in any response letter:

- *the negative reproduces oracle-free*
- *the negative fails without an oracle*

Both generalize beyond one cell, one aggregator, one attack, one dataset and one upstream transform,
which is all this arm has.

**One existing statement stays true and must not be "corrected".**
`results/oracle_free_decomposition/summary.json` records *"No composed-pair ASR result on a second
dataset exists anywhere in this paper and this arm does not change that."* This arm does not change
it either: its `d1` is a synthetic coefficient vector, not a second real defense, so it is not a
composed pair in that sentence's sense. That artifact is frozen and is not amended.

## Commit protocol

This file is committed **alone**, before the runner is final and before any run:

1. `git add experiments/pre_registration_oracle_free_femnist.md` and commit that one file.
2. Read `git log -1 --format=%h` and set `PREREG_COMMIT` in the runner to that hash.
3. The runner refuses to start unless **both** hold: `git log -1 --format=%h -- <this file>` matches
   `PREREG_COMMIT`, and `git status --porcelain -- <this file>` is empty. `PREREG_COMMIT is None` is
   only half a guard, because the moment the constant is set the only test that ever fires stops
   firing.
4. The runner writes only to `results/oracle_free_femnist/`, a new directory. No existing `results/`
   directory or `results/*.json` is written.

This document is not amended after the numbers exist. If a verdict literal in it turns out to be
right in its label and wrong in its stated mechanism, both are printed and neither is edited.
