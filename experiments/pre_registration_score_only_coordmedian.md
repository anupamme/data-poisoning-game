# Pre-registration: the magnitude control on the sign-reversal cell (score-only coordinate median)

**Status: frozen before any write to `results/score_only_coordmedian/`.** The cell, the rungs, the seed
list, the two estimands, the verdict rules and the clauses limiting what this arm may conclude are
reproduced verbatim in `experiments/run_score_only_coordmedian.py`, which refuses to start until this
file is committed and its hash is recorded in `PREREG_COMMIT`.

## Why this arm and not another

The paper's headline is one cell estimated two ways: `coord_median` under `committed_pixel` on
CIFAR-10 / `cifar_cnn`, where the outcome-gated design moves ASR **−0.273** and the within-defense
(Mode S) design moves **+0.125**, both at n = 20 and with non-overlapping intervals. Every existing
magnitude control in this paper is defined for **selectors only** — `generic_compose` raises at
`run_all_compositions.py:578`–`:581` for any `d2` outside `("krum", "cos_krum")`, because "there is no
single selected update whose magnitude could be held fixed or rescaled" — so the
one cell the abstract, §1, §5, the Conclusion and Fig. 1(c) are built on is the one cell whose
magnitude channel has **never** been closed. `pre_registration_score_only_kappa2_n20.md`'s caveat 4
says exactly that and names this document as the arm that addresses it.

That gap is what a review named as the paper's most important technical weakness: under Mode S the
upstream transform still displaces the aggregate the defense emits, so the within-defense rise could
be the magnitude channel rather than the statistic channel. On the Krum cell the paper answers this
with the score-only control and a 2×2 factorial. On **this** cell it answers it with nothing. This arm
closes that, or fails to, and both directions are written down here first.

## What is published, recomputed per seed rather than transcribed

`coord_median` / `committed_pixel` / CIFAR-10 / `cifar_cnn`, N = 10, K = 5, f = 0.2, α = 0.5,
50 rounds, κ ∈ {0, 2}; seeds 42–46 from `results/dose_response/` (confounded) and
`results/dose_replication/` (controlled), seeds 47–61 from `results/reversal_seed_topup/`, n = 20 on
every leg:

| leg | paired Δ = ASR(κ=2) − ASR(κ=0) | sd | 95% CI (t₁₉ = 2.093) | mean clean acc. at κ=2 |
|---|---|---|---|---|
| confounded (outcome-gated, `dose_`) | **−0.273278** | 0.129186 | [−0.3337, −0.2128] | 0.691850 |
| controlled (Mode S, `doseS_`) | **+0.125061** | 0.063341 | [+0.0954, +0.1547] | 0.703245 |

**Two numbers on this cell must not be confused.** Table 2's coordinate-median row prints
ΔASR = +0.098 at n = 5; the n = 20 value above is +0.125061. Same cell, same rung, two
seed counts. Every comparison in this document is against the **n = 20** value, and a report that
quotes +0.098 beside this arm's Δ_B is comparing across seed counts.

Mean ASR at the shared identity rung is **0.478328**, and the identity rung is *one* rung: at κ = 0
`dose_kappa0.0` and `doseS_kappa0.0` are equal per seed at all 20 seeds, which is re-established here
rather than assumed. Published verdict, carried forward unchanged: **sign reversal — both intervals
exclude zero and they do not overlap.**

## The instrument, defined before it is built

`score-only coordinate median`: for each coordinate *j*, take the **argmedian over clients of the
transformed stack**, and emit the **untransformed** value of that client at that coordinate.

```
idx[j] = argmedian_i  T(u)_i[j]          # which client the transformed stack puts at the median
out[j] = u_{idx[j]}[j]                    # that client's OWN, untransformed coordinate
```

`fl_core/federated.py:180`–`:187` computes `coord_median` as
`torch.stack([...]).median(dim=0).values`, and `torch.median` at an odd client count (K = 5 here, every
round) returns an actual element of the stack together with its index. So the analogue is the same
computation with `.values` replaced by a `gather` at `.indices` into the untransformed stack, and
**nothing else differs**: same flattening, same `.cpu().float()` cast, same per-parameter loop, same
reshape and device restore.

Two properties follow and both are asserted rather than argued:

1. **At κ = 0 the control *is* `coord_median`.** `apply_d1_transform` at `val == 0.0` executes
   `return updates` (`run_all_compositions.py:397`–`:398`), returning the caller's list **object**, not a
   multiply by 1.0 — a property its own comment says exists so that "the harness check … cannot be fooled
   by rounding". The transformed and untransformed stacks are then the same object, and a gather at
   `.indices` returns exactly `.values` by the definition of `torch.median`. The identity is therefore
   **exact, not approximate**, and the harness check below must read **`0.00e+00`** — not a tolerance.
2. **What is closed is magnitude, per coordinate.** No emitted number is a rescaled one: every
   coordinate of the aggregate is some client's own value. What the dose can still move is *which*
   client supplies each coordinate, which is the re-selection channel and is left deliberately open —
   it is the channel the paper's claim is about. Both quantities are already measured, at this exact
   family and rung, in `results/admission_measurement.json` (`doseS`, κ = 2.0, `committed_pixel`,
   5 seeds × 3 probe rounds): what this control closes is `agg_disp_coord_median` = **0.670332**,
   Table 2's $\Delta$~agg.\ of 0.670; what it leaves open is
   `coord_median_frac_argmedian_changed` = **0.482085**, Table 2's $\Delta$~dec.\ of 0.482 — so nearly
   half of all coordinates change which client supplies them, and that is deliberate.

**This control is not a defense**, for the same reason score-only Krum is not: it aggregates values its
own scoring stage never saw, on top of Mode S already reading adversary identity to pin `c_adv = 1`. It
is a laboratory instrument and nothing here is proposed for deployment.

## Design

- **Cell:** `coord_median` / `committed_pixel` / CIFAR-10 / `cifar_cnn`, the published configuration
  above, inherited by import and not restated.
- **One rung: κ = 2.0**, the frozen primary contrast of this cell. **κ = 0.5 and κ = 1.0 are not run.**
  The four-rung ladder and its trend statistics stay where they are, at their own n, and no display may
  print this ladder with a rung of this arm in it.
- **Seeds 42–61, n = 20**, the full published seed set, because no score-only coordinate median exists
  at any seed. Fixed here: no interim look, no extension, no filtering, no seed dropped for any reason.
- **20 runs at κ = 2**, one per seed, ≈11 min each, resumable, summary written after every run.
- **The κ = 0 baseline is imported, not re-run**, from the published controlled leg
  (`results/dose_replication/` for 42–46, `results/reversal_seed_topup/` for 47–61) under property (1)
  above. **The assertion is re-run here at one seed (42)**, where a published value exists, because a
  new seed would make the check vacuous. If it does not read exactly `0.00e+00`, **the import is void,
  the arm does not run, and the round reports the failure**; the tolerance is not loosened and no
  approximate-identity variant is substituted.
- **Nothing is recomputed that already exists.** `results/dose_response/`, `results/dose_replication/`,
  `results/reversal_seed_topup/` and `results/admission_measurement.json` are read and not written. No
  existing runner or pre-registration is edited.
- **What the new module reuses and what it reimplements, stated before it is written.** It cannot call
  `generic_compose`, which raises for this aggregator, so — exactly as `run_comparability_cells.py`
  says of itself for a different reason — it **reimplements the one 50-round loop the frozen runners
  run** and imports every constant that defines the cell rather than restating it: `FL_CONFIG`
  (`num_clients=10, clients_per_round=5, num_rounds=50`), `ADV_FRACTION = 0.2`, `ATTACK_MAP`,
  `cell_key` and `KAPPAS` from `run_comparability_cells`, and `apply_d1_transform` from
  `run_all_compositions`. A divergence in any of these would silently make this a different
  experiment from the arm it is compared against, which is why none is retyped. **The one constant
  that is *not* imported is the seed list**: `run_comparability_cells.SEEDS` is frozen at `[42..46]`,
  and this arm needs the full published 42–61, so `SEEDS = list(range(42, 62))` is declared in the new
  module and asserted to be exactly the 20 seeds present in the published legs before scoring. This
  suite writes `results/score_only_coordmedian/` alone.

## The two estimands, both frozen here

**Primary — does the within-defense rise survive with the magnitude channel closed?**

> Δ_B = mean over the 20 paired seeds of [ ASR_s(κ=2, score-only CM) − ASR_s(κ=0) ],
> reported with its two-sided paired 95% Student-t interval, t₁₉ = 2.093.

**Secondary — how much of the rise is the magnitude channel?** Because both arms use the same 20 seeds
and share the κ = 0 term exactly, the difference of the two contrasts is itself a paired quantity and
the identity rung cancels:

> Δ_mag = mean over the 20 paired seeds of [ ASR_s(κ=2, uncontrolled Mode S) − ASR_s(κ=2, score-only CM) ],
> reported with its own paired 95% interval.

Δ_mag is a **direct estimate of the magnitude channel's contribution to the headline rise on the
headline cell**, and it is reported with its interval whichever sign and size it takes. No projection
is claimed for its width in advance, because no prior arm on this cell estimates its sd; the realized
interval is what is reported.

**What this arm can decide, disclosed before the run.** Holding the controlled arm's observed paired sd
of 0.063341, the projected 95% half-width at n = 20 is **0.0296**. So if Δ_B lands near the published
+0.125 its interval excludes zero with room to spare, and if the magnitude channel carries most of the
rise the interval covers zero. **The refuting branch is reachable**: arithmetically it needs the
magnitude channel to account for roughly 0.095 of the 0.125 rise.

**And the one prior magnitude-only measurement points the other way, which is stated here rather than
after the fact.** On the Krum cell the emit-only arm — magnitude open, statistic closed — is
**+0.010, 95% CI [−0.028, +0.047]** at n = 20, so `app:factorial` reports the magnitude channel as
*established inert* there and **withdraws** the knife-edge n = 5 verdict that the κ = 1 point estimate
of +0.046 had produced. That earlier number is a withdrawn verdict and is not evidence for anything
here. It does not transfer either way: `coord_median` is a different aggregator with a far larger
displacement to close (Δ agg. 0.670 against Krum's 0.892 being a vector sum that partly cancels), the
channel is coordinate-wise rather than per-client, and caveat 4 below refuses the transfer in both
directions. So the honest prior is that this arm is **more likely to confirm than to refute**, the
refuting branch is reachable, and the arm is powered for its primary either way.

## Decision rules

| outcome on Δ_B | verdict |
|---|---|
| Δ_B > 0 and its 95% CI excludes zero | **THE WITHIN-DEFENSE RISE SURVIVES WITH THE MAGNITUDE CHANNEL CLOSED.** The sign reversal on the paper's headline cell is not a magnitude artifact: with every emitted coordinate a client's own untransformed value, the upstream dose still raises suppression's failure where the outcome-gated design reads a fall. Reported in the abstract, §5 and `tab:claims`, with Δ_mag beside it. |
| the 95% CI contains zero | **THE WITHIN-DEFENSE RISE DOES NOT SURVIVE THE MAGNITUDE CONTROL ON THIS CELL.** The within-design leg is then partly the magnitude channel, we report that against ourselves in the abstract, in `tab:claims` and in `tab:evidence`, and the reversal is restated at the scope the evidence licenses — as a disagreement between two designs whose controlled leg is not magnitude-controlled. |
| Δ_B < 0 with its CI excluding zero | **THE CONTROL REVERSES THE CONTROLLED LEG'S OWN SIGN.** Reported as such and **not** scored in our favour: with magnitude closed the controlled design would agree in sign with the outcome-gated one, which removes the dissociation on this cell. Same reporting sites as the row above. |

- **The interval is reported with the point estimate every time**, and its relation to **both** zero and
  the published [+0.0954, +0.1547] is stated, because an interval can exclude zero and still be
  incompatible with the uncontrolled arm's.
- **The sign-reversal claim is re-adjudicated explicitly, not by implication.** Whatever Δ_B is, the
  report states whether Δ_B's interval (i) excludes zero on the positive side and (ii) fails to overlap
  the confounded n = 20 interval [−0.3337, −0.2128]. Only (i) **and** (ii) together reproduce the
  published reversal with magnitude closed.
- **Accuracy gate:** `ACC_FLOOR = 0.35` on the **mean** clean accuracy at κ = 2 under the control,
  unchanged from every arm on this cell. If it fails the contrast is uninterpretable, **no verdict
  stands**, it is reported as such, and it is **not** substituted onto another rung.
- **No margin test is run here and none is imported.** The ±0.15 practical-equivalence margin scores
  arms whose claim is a non-increase; this arm's claim is about the **sign and size of a rise**, so the
  decision is on the interval's relation to zero and to the published interval, fixed above. Importing
  a margin afterwards would be choosing a rule after the numbers.

## Caveats recorded in advance

1. **Magnitude is closed; the trajectory channel is not.** Which client supplies each coordinate still
   changes across rungs, so the model trajectory still diverges. The claim this arm licenses is about
   magnitude, exactly as on the Krum cell.
2. **The oracle is untouched.** Mode S still reads adversary identity to pin `c_adv = 1`. This arm
   removes no oracle and licenses no statement in either direction about what happens without one; the
   phrase *the negative reproduces oracle-free* and its inverse are not licensed here in any form.
3. **One cell.** One dataset, one architecture, one attack, one defense, twenty data partitions. This
   closes one channel on the headline cell and is not a breadth claim.
4. **A coordinate-wise control is not a selector control.** Score-only Krum holds one client's whole
   update fixed; this holds each coordinate's contributing value fixed. The two are the same idea
   applied to two aggregator forms and neither arm's result transfers to the other. No pooled number
   combines them.
5. **Δ_mag is a difference of two arms, not a decomposition of Δ.** On the Krum cell, where the full
   2×2 exists, separability is at n = 20 *consistent with* the data and **still not established** — the
   additivity residual is −0.019, CI [−0.061, +0.023], not contained in ±0.05 — and `app:factorial`
   already withdraws any reading of the channel split as a decomposition of suppression. No such
   factorial exists here at all: this arm runs one of the four cells. So Δ_mag is the measured effect
   of closing magnitude at this rung on this cell and is **not** to be described as the magnitude
   channel's share of Δ, nor summed with anything.
6. **This arm is a disclosed post-hoc extension of a frozen suite**, not a prospective test of a rule
   written before the cell existed. The published controlled point estimate was known when this
   document was written. That is stated at the site of the result.

## Non-negotiables

1. No estimand, seed list, rung, interval definition, accuracy floor or verdict label above is revised
   after an outcome is seen.
2. The seed list is 42–61 and the rung is κ = 2.0. Neither is extended, truncated by inspection, or
   filtered.
3. The κ = 0 harness assertion is at **exactly** `0.00e+00`. A nonzero difference voids the import and
   stops the arm; it is not re-expressed as a tolerance.
4. Every reported number is recomputed per seed from the artifacts named above, never transcribed, and
   the published n = 20 legs are asserted to reproduce before any pooled number is printed.
5. `results/dose_response/`, `results/dose_replication/`, `results/reversal_seed_topup/` and
   `results/admission_measurement.json` are **not rewritten**, and no existing runner or
   pre-registration is edited.
6. This file is committed before `results/score_only_coordmedian/` is written. The commit ordering
   establishes only that the rules above were not written after the numbers.
7. Both directions are published. Committing to report the refuting branch in the abstract is what
   licenses running the arm at all.
