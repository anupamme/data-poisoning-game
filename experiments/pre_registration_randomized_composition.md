# Pre-Registration: is randomization *itself* costly, or was it only a degenerate order?

**Date:** 2026-09-09
**Author:** Anupam Mediratta
**Runner:** `experiments/run_randomized_composition.py` (refuses to start until this file is committed)
**Analyzer:** `experiments/analyze_randomized_composition.py`
**Output (does not exist yet):** `results/randomized_composition/summary.json`

## 1. Why this experiment exists, and what it is *not*

The paper does **not** currently claim anything about randomized defense menus. §Related Work states:

> An earlier version of this paper compared against a randomized menu and reported that deterministic
> composition dominates it; that comparison is not part of this submission and no claim here rests on it.

That withdrawal **stays** regardless of what this experiment finds. This is a new, separately
pre-registered object, and the reason the old one was withdrawn is the reason this one is designed the
way it is.

**The defect in the withdrawn comparison.** Its randomized arm
(`results/compute_matched_mixing/summary.json`, condition `randorder_fg_cm`) sampled 50/50 between
FG→CM and **CM→FG**, and CM→FG is *degenerate*: CoordMedian emits an aggregate rather than per-client
updates, so `apply_d1_transform` passes them through unchanged and the round reduces to **FoolsGold
alone**. Half of the randomized arm's rounds therefore ran **one** defense against a fixed arm's two.
The measured 9×/19× gaps are fully consistent with "one defense per round is worse than two" and say
nothing about randomization as such. That objection is correct, and it is the objection this experiment
answers.

**A second defect, in the harness.** `run_compute_matched_mixing.py:150` reads

```python
if randomize_order and rng.random() < 0.5:
```

Python **short-circuits** `and`, so the fixed arm never draws that number while the randomized arm draws
one per round — from `rng`, the *same* generator that selects each round's participants. The two arms
therefore saw different clients, and the published gap conflates order-randomization with a different
participation sequence. **No paired test was licensed on that artifact, and none is computed from it
here.** The new runner draws every policy from a separate `policy_rng = default_rng(seed + 2000)`, which
cannot perturb the participant sequence however often it fires.

**A third defect, found while writing this file, and it is the one that sets the design.** The
predecessor was not on the **canonical participant stream** at all. `run_all_compositions.run_one` — the
path behind `results/all_compositions`, `results/wave2_held_out`, `results/fg_cm_survivor` and every
`fedavg_then_X` single-defense baseline — selects participants with the **legacy global**
`np.random.choice` seeded by `np.random.seed(seed)`. `run_compute_matched_mixing.py` used
`np.random.default_rng(seed + 1000)`, a different sequence, so it **re-measured rather than reproduced**
every cell it shares with the canonical suite. On fg→CM's committed-pixel arm at the same five seeds:

| seed | canonical (`fg_cm_survivor`) | `compute_matched_mixing` |
|---|---|---|
| 42 | 0.113889 | 0.050556 |
| 43 | 0.086889 | 0.061444 |
| 44 | 0.130889 | 0.051444 |
| 45 | 0.081333 | 0.072222 |
| 46 | 0.080889 | 0.144111 |
| **mean** | **0.0988** | **0.0760** |

Max per-seed deviation 0.079. By contrast `fg_cm_survivor` is **bit-identical** to the frozen
`all_compositions` on the shared seeds 42–44 (deviation 0.00e+00), i.e. the canonical cell *extended*
to n=5, not re-measured. **This runner therefore uses the canonical global stream and calls
`generic_compose`**, so its fixed arms are expected bit-identical to the canonical artifacts and are
**not** expected to match `compute_matched_mixing`.

**Consequently every arm is re-run.** No leg of any comparison below is reused from an existing
artifact, and `results/compute_matched_mixing/summary.json` is **not overwritten** — it stays as the
superseded record.

## 2. Arms (frozen)

Seven conditions. A *policy* is a set of (d1, d2) stages, one sampled uniformly at random each round;
a single-element set is a fixed policy. `d1 = fedavg` is the identity pass-through, so `(fedavg, X)`
is "X alone" — the same object the single-defense baselines are defined as (`fedavg_then_X`).

| arm | condition | policy | defenses/round |
|---|---|---|---|
| 1 | `fixed_fg_cm` | FG→CM, every round | 2 |
| 1 | `fixed_fg_rfa` | FG→RFA, every round | 2 |
| 1 | `fixed_fg_tm` | FG→TM, every round | 2 |
| **2, primary** | `rand_comp_strong` | uniform over {FG→CM, FG→RFA} each round | 2 |
| 2, secondary | `rand_comp_all3` | uniform over {FG→CM, FG→RFA, FG→TM} each round | 2 |
| 3 | `temporal_mix` | uniform over {FG, CM, RFA, TM} alone each round | 1 |
| 4 | `pure_coord_median` | CoordMedian alone | 1 |

**Every member of every randomized set keeps FoolsGold as a genuine per-client upstream, so no round
degenerates to a single defense.** That is the whole point: it removes the confound above, so a gap
between arm 2 and arm 1 is attributable to randomization and not to defense count.

**Why there are two randomized arms, and why the *smaller* set is primary.** Canonically
(`results/all_compositions`, n=3) FG→TM's committed-scaling arm is **0.508 ± 0.269** — *above* the 0.5
threshold — and FG→TM is a criterion-FAIL (C3) pair. A randomized set containing it could degrade
simply because a non-suppressing member was drawn on some rounds, which is the CM→FG objection in a
new and harder-to-see form. So `rand_comp_strong` draws only from members that **each suppress both
committed attacks on their own** (FG→CM 0.131/0.111, FG→RFA 0.045/0.038 canonically), making it the
confound-free test of whether randomization is intrinsically costly; `rand_comp_all3` prices a
realistic heterogeneous menu and is reported as a secondary. **Both sets are fixed by the rule stated
here, not by any outcome measured in this artifact.**

**Compute matching is claimed for arms 1–2 only.** Arms 3 and 4 apply *one* defense per round and
therefore use roughly half the defense computation; they are reference points, not compute-matched
comparators, and every table reporting them must say so. Stating this in advance is deliberate: the
withdrawn comparison's failure was an unlabelled compute asymmetry, and the fix is to label it, not to
hide it behind a matched-sounding word.

Fixed: N=10 clients, K=5 per round, f=0.2 adversarial (clients 0–1), 50 rounds, `cifar_cnn`, CIFAR-10,
Dirichlet α=0.5, τ=5.0. Attacks: `committed_scaling`, `committed_pixel`. **Seeds 42, 43, 44, 45, 46
(n=5), one seed set for every arm.** 7 × 2 × 5 = **70 runs**, ~10–15 min each, ~12–17 h.

## 3. The primary contrast, and its direction, frozen before any data exists

**Primary contrast.** Per attack: `rand_comp_strong` against the **best member of its own set**, where
"best" is defined *by rule and not by name* as whichever of `fixed_fg_cm` / `fixed_fg_rfa` has the
lower mean ASR on that same attack **within this artifact** (so the selection and the comparison are on
the same n=5, and the comparator is drawn only from the set actually being randomized over). The
contrast is **paired per seed**: Δ_s = ASR(`rand_comp_strong`, seed s) − ASR(best member, seed s).

**Secondary contrast, same rule:** `rand_comp_all3` against the best of all three arm-1 conditions.

**Direction (one-sided, fixed now).** H1: randomization *hurts*, i.e. **Δ̄ > 0**. The alternative is
pre-registered because it is the claim the withdrawn comparison made and the claim a reviewer asked us
to substantiate; a two-sided reading would let either outcome be spun as a finding.

**Materiality threshold, so a null is a null and not merely underpowered.** The gap is called material
only if **both** (i) Δ̄ ≥ 0.10 absolute ASR, and (ii) Δ_s > 0 in at least 4 of 5 seeds. The 0.10 anchor
is ≥3× the across-seed sd of every **canonical** reading of the two primary-set members: FG→CM
0.010 scaling / 0.018 pixel and FG→RFA 0.008 / 0.010 at n=3 (`results/all_compositions`), and FG→CM
0.031 / 0.020 at n=5 (`results/fg_cm_survivor`, `composed`). Population sd throughout, which is the
convention every number in print in these papers uses. The anchor is **not** taken from
`results/compute_matched_mixing`, whose sds are 2–6× larger because it is on the divergent participant
stream of §1.

**Statistics, with the n=5 limit stated in advance.** One-sided Wilcoxon signed-rank, n=5: the minimum
attainable p is 1/32 = 0.031, so **only 5/5 concordant signs can reach p < 0.05**, and 4/5 gives
p = 0.094. A paired t-test and the per-seed sign count are reported alongside. We will not describe any
sub-material gap as "no effect" — the honest statement at n=5 is "no material difference detected,
smallest effect this design could resolve is ≈0.10".

**Secondary, reported but not decisive:** `rand_comp_all3` vs. `rand_comp_strong` (what a
non-suppressing member in the menu costs); `rand_comp_strong` vs. `temporal_mix` (does randomizing over
*compositions* differ from randomizing over *single* defenses — the predecessor's confound isolated);
`temporal_mix` vs. `pure_coord_median` (is temporal mixing worse than the best pure defense); every
arm's clean accuracy. **Diagnostic, from the per-round logs and costing no extra runs:** the
per-seed association between a randomized arm's realized member counts and its ASR, which tests
directly whether any gap is explained by *which* member was drawn rather than by randomization.

## 4. Admissible outcomes (all three are publishable; we do not choose after seeing the data)

- **(i) Material, in the pre-registered direction** (Δ̄ ≥ 0.10 and ≥4/5 signs positive). The paper may
  then state, *scoped to this menu and this attack pair*, that randomizing over non-degenerate
  compositions costs ASR at matched defense computation. `:1638`'s withdrawal of the *old* comparison
  still stands and is pointed at explicitly, because the old artifact remains unpaired and
  compute-confounded.
- **(ii) Not material, or material in the opposite direction.** **The withdrawal stands and the null is
  reported as the result.** This equally answers the reviewer: it says the earlier 9×/19× gap was an
  artifact of the degenerate CM→FG rounds rather than of randomization. No claim about randomized menus
  enters the paper.
- **(iii) Accuracy floor breached.** Any arm whose mean clean accuracy falls below **0.35** is reported
  as uninterpretable for ASR purposes rather than compared, and the contrast is reported on the
  surviving arms with the exclusion named.

**No outcome is a failure and none is suppressed.** Whichever obtains is written into the appendix with
all six arms' ASR and accuracy shown.

## 5. Positive controls, checked before any number reaches the paper

This project has twice shipped a hook that silently never fired
(`run_cross_distribution_compositions.py` never calls `manipulate_update`; a Round 55 Δ was credited to
a bit-exact no-op). So the randomization hook is verified, not assumed. `analyze_randomized_composition.py`
**asserts** all four:

1. **The hook fired.** Every member of every randomized set appears in the stored per-round
   `policy_log`, with counts inside a two-sided 99% binomial interval of uniform: 50 rounds at p=1/2
   for `rand_comp_strong` (≈16–34), p=1/3 for `rand_comp_all3` (≈8–26), p=1/4 for `temporal_mix`
   (≈5–21).
2. **The arms differ.** Neither randomized arm is bit-identical to any arm-1 condition at any seed. A
   bit-identical randomized arm is a dead hook, not a result.
3. **The harness did not change the run.** Each fixed arm-1 condition is compared **per seed** against
   the canonical artifacts on their shared seeds: `results/all_compositions/summary.json` (42–44) and
   `results/fg_cm_survivor/summary.json`'s `composed` (42–46, for fg→CM). Because this runner uses the
   canonical global participant stream and `generic_compose`, agreement must be **bit-identical**, and
   any nonzero deviation is reported as a harness change — in which case no new arm is compared to any
   published number. **We do *not* expect agreement with `compute_matched_mixing`'s fixed arms** and do
   not treat disagreement there as a failure: §1's third defect establishes that artifact is on a
   different participant stream, and this control is the assertion that we are on the canonical one.
4. **Arm 4 reproduces a published standalone.** `pure_coord_median` is the same quantity as
   `fedavg_then_coord_median` in `results/wave2_held_out/summary.json` (0.519 scaling / 0.443 pixel,
   n=5, seeds 42–46). Agreement validates the whole path end to end; disagreement means the participant
   stream differs between suites and is reported as such **before** any arm is compared.

## 6. Non-negotiables

- **Every leg of every comparison comes from `results/randomized_composition/summary.json` at seeds
  42–46.** No Δ or ratio may mix an n=5 minuend with an n=3 subtrahend — the exact defect Round 55
  fixed, which hid in the subtrahend where no output column showed it. The analyzer prints **both
  legs' n** on every row and marks any mismatch `<-- MIXED n`.
- **Nothing is transcribed.** Every reported number is recomputed per seed from the artifact.
- **No arm is dropped, and no seed is dropped,** after the data exists. Runs are resumable per seed and
  the completed set is whatever this file specifies.
- **The four frozen md5s do not move:** `results/dose_femnist/summary.json`,
  `results/comparability_six_cells.json`, `results/all_compositions/summary.json`,
  `results/headline_seed_topup/summary.json`.
- **If the primary contrast comes out (ii), it is reported as (ii).** The temptation this file exists to
  foreclose is re-labelling a null as "randomization is at least not harmful, so composition is robust";
  that is a different claim from a different design and it is not licensed here.
