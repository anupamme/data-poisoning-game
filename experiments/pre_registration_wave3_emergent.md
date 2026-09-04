# Pre-Registration: Wave 3 — the cells that discriminate a statistic-preservation screen from C1∧C2

**Date:** 2026-09-04
**Author:** Anupam Mediratta
**Runner:** `experiments/run_wave3_emergent.py` (refuses to start until this file is committed)
**Output (does not exist yet):** `results/wave3_emergent/summary.json`

## 1. What this wave is for, and why frozen data cannot supply it

The paper compares three evaluation strategies over the 42-pair menu:

| strategy | rule | n selected | precision | recall | mean max-ASR |
|---|---|---|---|---|---|
| none | evaluate everything | 42 | 11.9 % (base rate) | 100 % | 0.743 |
| **A** | C2 alone — statistic preservation | 14 | 35.7 % | 100 % | 0.588 |
| **B** | C1 ∧ C2 — the criterion | 2 | 100 % | 40 % | 0.242 |

Every one of those numbers is *in-sample*: the criterion was developed on the 18-pair
development split, and A and B are being scored on data that was available when they were
written. Two prior attempts to get out-of-sample evidence discriminating A from B failed
for reasons that are now understood and are *structural*, not fixable by more of the same:

- **Wave 2 (24 pairs).** Contains **zero** low-ASR pairs, so it bears on specificity only.
  16 of its 24 pairs are DEGEN (`coord_median`/`trimmed_mean`/`fedavg` as d1 emit no
  per-client transform), and no single defense in that menu suppresses, so those 16
  *could not* have come out LOW. A makes 1 false positive there and B makes 0, with no
  true positive available to either.
- **Prospective suite (20 pairs).** Its four C1-fail pairs also fail C2, so **A and B
  select the same 7 pairs** and the suite cannot separate them at all.

The one cell that discriminates A from B is **C1-fail ∧ C2-true**: A selects it, B does
not. If a pair there comes out LOW, A has a true positive B misses and B's 40 % recall is
demonstrated prospectively rather than computed in-sample. In the development menu that
cell ran 1 low of 6.

This wave measures that cell, plus the certified cell (which tests B's precision), with
**measured** rather than assumed C2 verdicts.

## 2. Why exactly four pairs — a census, not a sample

An earlier version of this wave was sized at ~20 pairs. That was wrong, and enumerating
the space is what showed it. All 42 ordered pairs of the 7-defense menu are spent, as are
the 20 prospective pairs, the 24 wave-2 pairs, the 8 metric-swap pairs and the three-way
compositions: **118 ordered pairs are already measured.** Restricted to pairs with an
admissible d1 — one emitting a genuine per-client transform (not DEGEN) and not itself
suppressing alone (C0) — only 6 remain, and the `d1 → cos_*` design that would have
supplied the rest was *already* pre-registered and run as `metric_swap`.

So the wave is re-scoped from a sample to a **census** of the region where mechanism
preservation is settled on both sides:

- **d1 emits strictly positive per-client coefficients** `T(u_i) = c_i u_i`, `c_i > 0`:
  `norm_clip`, `rfa`, `reputation`. (`foolsgold` is excluded **by measurement** — §3.
  `fltrust` is excluded by C0, at 0.048 max-committed.)
- **d2 reads an exactly scale-invariant statistic**: `foolsgold`, `cos_krum`,
  `cos_reputation` (Proposition `prop:invariance`(a)).

That cell is exactly **3 × 3 = 9** ordered pairs. Seven are measured; **two are unspent**,
and they are pairs 1 and 2 below. After this wave the cell is exhaustively measured, so the
claim it supports is a census claim and not a sampling claim.

### 2.1 The census as it stands (all seven frozen, protocol-matched)

| d1 ↓ / d2 → | `foolsgold` | `cos_krum` | `cos_reputation` |
|---|---|---|---|
| `norm_clip` | 0.8869 HIGH | 0.5085 HIGH †(pixel) | 0.9346 HIGH †(pixel) |
| `rfa` | 0.9135 HIGH | 0.5657 HIGH | 0.8612 HIGH |
| `reputation` | 0.8070 HIGH | **wave 3, pair 1** | **wave 3, pair 2** |

Values are max-committed mean ASR. Sources: `results/all_compositions/summary.json`
(nc→fg, rep→fg), `results/wave2_held_out/summary.json` (rfa→fg),
`results/metric_swap/summary.json` (the four `cos_*` cells).

**† Disclosed near-identity, and this is a limitation of two census cells, not a
side-note.** On the pixel arm `norm_clip` is a **literal no-op**: measured over 45
client-rounds, the maximum update norm under `backdoor_pixel` is **3.0432 < τ = 5**, so
`min(1, τ/‖u‖) = 1` for every client and `apply_d1_transform` is the exact identity in
**9 of 9 rounds** (max absolute change `0.000e+00`). Under `model_scaling` norms reach
**24.4878**, 11 of 45 client-rounds exceed τ, and the identity holds in only 1 of 9 rounds.
Consequently the *pixel cells* of `norm_clip → cos_krum` (0.30048148…) and
`norm_clip → cos_reputation` (0.75425926…) are **bit-identical to d2 alone**
(`fedavg → cos_krum` 0.30048148…, `fedavg → cos_reputation` 0.75425926…) and per-seed
identical in 3 of 3 seeds; those two cells measure d2 alone, not a composition. Both
pairs' max-committed verdicts are nevertheless carried by the *scaling* arm (0.5085 and
0.9346), where d1 genuinely transforms, so the HIGH labels stand. Measured by test (F) of
`experiments/verify_wave3_invariance.py`; the same class of near-identity the paper already
discloses for FLTrust-as-d2 (`paper/main.tex:762`, L4).

The remaining six comparisons are 0/3 per-seed identical, i.e. genuine compositions.

**What the census already shows:** restricted to the region where statistic preservation is
exactly satisfied, strategy A selects all 9 pairs, and all 7 measured ones are **HIGH**.
Statistic-level invariance, on its own, has so far selected 7 pairs for 0 true positives
in the very class where it is exactly true. Strategy B selects only pair 1.

## 3. Phase-0 mechanism measurements, taken BEFORE these predictions were frozen

Run: `python3 -m experiments.verify_wave3_invariance` →
`results/wave3_invariance_check.json` (committed with this file; see §9).
This is the companion to `verify_cos_invariance.py`, which covered d1 ∈ {`norm_clip`, `rfa`}.
It was run first **because it changed the design**, which is the point of running it.

**(A) Synthetic, 1000 trials, the two coefficient shapes d1 actually emits.**

| coefficients | cos_reputation max ‖Δw‖ | cos_rep order changed | cos_krum selection changed | controls (rep / krum) |
|---|---|---|---|---|
| strictly positive | 8.941e-08 | **0/1000** | **0/1000** | 992/1000, 618/1000 |
| one exact zero | 1.808e-01 | **859/1000** | **471/1000** | 990/1000, 637/1000 |

A single exact zero destroys cosine invariance. The controls confirm the probe can see
disturbance when there is disturbance.

**(B) Real FL rounds (cifar10, N=10, K=5, f=0.2, 3 seeds × 3 rounds) through the shipped
`apply_d1_transform`.**

| d1 | exact zeros | cos_rep max ‖Δw‖ | cos_rep order changed | cos_krum selection changed | controls: `krum` / `reputation` |
|---|---|---|---|---|---|
| `reputation` | **0/5 every round** | 5.96e-08 – 2.24e-07 | **0/9** | **0/9** | 8/9, **9/9** |
| `foolsgold` | **2/5 every round** | 0.0923 – 0.1385 | **9/9** | **9/9** | 4/9, 9/9 |

- `reputation` emits strictly positive coefficients (ρ over the positive part ran 2.1 to
  18 839.8, so the rescaling is extremely heterogeneous) and **C2 holds exactly** for both
  `cos_krum` and `cos_reputation`. The controls make this a real measurement rather than an
  inert probe, and the second one is the sharp one: under the *same* rescaling in the *same*
  rounds, non-invariant `krum`'s selection moved in 8/9 and **non-invariant `reputation`'s
  own ordering moved in 9 of 9**, while the two cosine statistics moved in **0 of 9**. The
  contrast is between statistics on identical inputs, not between conditions.
- `foolsgold` forces at least one coefficient to **exactly zero every round** because
  `weights = weights - weights.min()`, which puts it **outside** the strict `c_i > 0`
  hypothesis of Proposition `prop:invariance`(a) and under Lemma `lem:annihilation`
  instead. Its downstream cosine statistics moved in **9 of 9** rounds.

**This changed the design.** `foolsgold → cos_krum` and `foolsgold → cos_reputation` were
going to be entered as C1-fail ∧ C2-true pairs. They are not: **C2 fails, measured.** They
are entered below as **measured-C2-fail controls** instead. Had this been assumed rather
than measured, this wave would have reported two mislabelled pairs.

**(D) A degeneracy in the shipped menu, found while writing the above.** `multi_krum`
selects `sorted(range(n), key=score)[:k]` with `k` defaulting to 5, and every experiment in
this project runs `clients_per_round = 5`, so it selects **all five** clients and
`_krum(multi=True)` reduces to `_fedavg`. Measured: `server.aggregate` is bit-identical at
K=5 (`max|diff| = 0.000e+00`) and differs at K=10 (`2.520e+01`); through
`generic_compose`, `max|diff| ≤ 1.907e-06` for all four d1 at K=5, i.e. float summation
order. **`multi_krum` is therefore not a distinct downstream defense at K=5**, and no
`d1 → multi_krum` pair is a composition. This wave uses neither `multi_krum` nor `fedavg`
as d2. Its consequences for already-published rows in `paper/main.tex` are reported
separately and are **not** amended here; this file records only that the fact was known
before Wave 3 ran.

**(E) The statistics that carry a stored C2=True label on a `foolsgold`-as-d1 pair**, which
(B) does not cover because (B) measures cosine statistics only:

| d1 | RFA Weiszfeld weights: max Δ | order changed | coord_median share: max Δ | order changed | argmax changed |
|---|---|---|---|---|---|
| `foolsgold` | 3.92e-01 | 8/9 | 4.94e-01 | 9/9 | 5/9 |
| `reputation` | 3.91e-01 | 9/9 | 2.45e-01 | 9/9 | 8/9 |

**This does not relabel anything, and the reading that it does is wrong.**
`analyze_condition_ablation.py:55-61` marks `coord_median`, `trimmed_mean` and `rfa` as
`None` — *conditional* — not as invariant; `c2_holds` defers those classes to the
pre-registered per-pair call. So the two certified pairs (`foolsgold → coord_median`,
`reputation → coord_median`) and the emergent case `foolsgold → rfa` are unaffected. What
(E) does establish is that **the C2=True labels on conditional-class d2 rest on the
per-pair call, not on invariance**, and that strategy A as implemented is therefore not a
pure statistic-preservation screen for those three d2 classes. That is a disclosure this
wave owes the strategy comparison, and it is recorded here.

## 4. C0 and C1 inputs — all frozen before this file, none re-run

Every value below already exists at the **identical** protocol (cifar10 / cifar_cnn,
N=10, K=5, f=0.2, Dirichlet 0.5, 50 rounds, τ=5). No standalone baseline is measured by
this wave; re-running them would spend ~6 h reproducing frozen numbers.

| defense | scaling ASR @ clean acc | pixel ASR @ clean acc | seeds | source |
|---|---|---|---|---|
| `reputation` | 0.0170 @ 0.7774 | 0.8422 @ 0.7737 | 5 | `pure_defense_baselines`, `wave2_held_out` |
| `foolsgold` | 0.2000 @ **0.1000** | 0.7316 @ 0.7414 | 5 | `wave2_held_out` (`fedavg→fg` = fg alone) |
| `cos_krum` | 0.0783 @ **0.1460** | 0.3005 @ 0.5983 | 3 | `metric_swap_baselines` (`fedavg→ck`) |
| `cos_reputation` | 0.9823 @ **0.2291** | 0.7543 @ 0.7953 | 3 | `metric_swap_baselines` (`fedavg→cr`) |

**The accuracy gate is load-bearing on three of these cells** (bolded). C1 requires
suppression at mean clean accuracy ≥ **`ACC_FLOOR` = 0.35**: a low ASR from a collapsed
model is not suppression. `foolsgold` at 0.2000/**0.1000** and `cos_krum` at
0.0783/**0.1460** on the scaling arm are hollow, so neither counts toward C1 there. This
gate was itself vindicated prospectively by `metric_swap`, whose frozen resolution rule
said that pairs 6/8 coming out HIGH refutes the numeric-C1 secondary reading; they measured
0.5085 and 0.5657, i.e. HIGH.

**C0** (`ASR(d1, a) ≥ 0.5` for at least one committed attack — d2's mechanism must be
load-bearing) **holds for all four pairs**: `reputation` 0.8422 on pixel, `foolsgold`
0.7316 on pixel. Both witnesses are the pixel arm. Lemma `lem:testability` is respected on
pair 1: at the C0-witnessing attack `a*` = pixel, C1 is satisfied by d2 (`cos_krum` 0.3005)
and not by d1, exactly as the lemma forces.

## 5. The four pairs, with per-pair verdicts and predictions

| # | pair | C0 | C1 (per attack, gated) | C2 | C3 | cell | **predicted** | confidence |
|---|---|---|---|---|---|---|---|---|
| 1 | `reputation → cos_krum` | ✓ | **holds** — scaling: rep 0.0170 @ 0.777 ✓; pixel: cos_krum 0.3005 @ 0.598 ✓ | **holds** (measured 0/9) | not binding | **certified** — tests B's precision | **LOW** | **low — `uncertain`, see §5.1** |
| 2 | `reputation → cos_reputation` | ✓ | **fails** — pixel: rep 0.8422 ✗, cos_rep 0.7543 ✗ | **holds** (measured 0/9) | n/a | **C1-fail ∧ C2-true** — tests B's recall, the discriminating cell | **HIGH** | medium |
| 3 | `foolsgold → cos_krum` | ✓ | **fails** — scaling: fg 0.2000 @ **0.100** ✗ (gate), cos_krum 0.0783 @ **0.146** ✗ (gate) | **FAILS** (measured 9/9) | n/a | measured-C2-fail control | **HIGH** | high |
| 4 | `foolsgold → cos_reputation` | ✓ | **fails** — both arms | **FAILS** (measured 9/9) | n/a | measured-C2-fail control | **HIGH** | high |

Strategy A (C2 alone) selects pairs **1 and 2**. Strategy B (C1 ∧ C2) selects **only pair 1**.
Pairs 3 and 4 are selected by neither and are here to show the C2 probe is not inert.

### 5.1 Pair 1 is the pair that can go wrong, and it is flagged `uncertain` on purpose

C1 ∧ C2 certifies pair 1, so the criterion predicts LOW, and that is the registered
prediction. But the prediction rests on **two different attacks being suppressed by two
different constituents**, and only one of those transmits through the composition:

- **pixel** — `cos_krum` alone reaches 0.3005 @ 0.598, its selection is exactly invariant
  under `reputation`'s rescaling (0/9), and the two measured analogues came out LOW
  (`norm_clip → cos_krum` 0.3005, `rfa → cos_krum` 0.2734). **Expected LOW.**
- **scaling** — this is the live risk. `reputation` alone reaches 0.0170 @ 0.777, but as d1
  it only *rescales*; it does not aggregate, so its suppression is not inherited the way a
  downstream defense's would be. `cos_krum`'s own scaling number (0.0783) is hollow at
  0.146 accuracy. The two measured analogues came out **HIGH** on this arm (0.5085,
  0.5657) — but with a d1 that suppresses nothing, whereas `reputation` is the first d1 in
  this cell that genuinely suppresses an arm.

So pair 1's max-committed verdict turns almost entirely on the scaling arm, where the
criterion's certification and the two nearest measurements point opposite ways. Registered
as **LOW** with **low confidence**. **If it measures HIGH, that is a false positive and B's
in-sample 100 % precision does not survive out of sample.** That outcome is reported as
such, in the body, not relegated to the supplement.

## 6. Frozen scoring rule

1. **Outcome per pair** = `max_committed_asr` = max over the two committed attacks of the
   mean ASR across seeds 42, 43, 44. **LOW** iff `< 0.5`, **HIGH** iff `≥ 0.5`. No other
   threshold is scored, and the threshold is not moved after the data exists.
2. **Accuracy gate.** A pair whose mean clean accuracy is `< 0.35` on the arm supplying its
   max-committed ASR is reported as **`DEGENERATE-ACC`** and scored as **HIGH** — a
   collapsed model is not a defended one. This is the rule Round 35 established and it is
   applied to pair outcomes here, not only to C1 inputs.
3. **Sanity gate.** If any pair's clean accuracy exceeds the `fedavg` reference
   (0.667 scaling / 0.768 pixel) by more than seed noise, or if any cell's ASR is
   identically 0.0 or 1.0 across all three seeds, the run is investigated for a harness
   fault before it is scored.
4. **Near-identity check, mandatory before scoring.** Every pair is compared per seed
   against d2 alone. A pair per-seed identical to d2 alone on an arm has that arm reported
   as **`NEAR-IDENTITY`**, and it does not count as evidence about composition on that arm.
   This is not optional: §2.1 shows the check catches real cases, and (F) shows
   `norm_clip`'s clip is inactive on the pixel arm. `reputation` and `foolsgold` both emit
   genuine transforms on both arms per (B), so no NEAR-IDENTITY is *expected* here — which
   is exactly why finding one would matter.
5. **Seeds.** 42, 43, 44 — the metric_swap and prospective-suite seeds. No seed is added
   after the data is seen; if variance demands more, the top-up is pre-registered
   separately, as `pre_registration_dose_seed_topup.md` was.

## 7. Two-sided resolution: what each outcome means

This is registered before the data exists so that no outcome can be written up as the
expected one. **The wave has no result that is not reportable.**

| pair 1 | pair 2 | reading | what goes in the paper |
|---|---|---|---|
| LOW | HIGH | **Both strategies correct.** B keeps 100 % precision out of sample; A pays one extra pair-evaluation for no additional true positive. The census closes 9/9 with exactly one LOW, the one B selects. | The strongest available result. B's precision is prospective, and A's cost is measured. |
| LOW | **LOW** | **A wins on recall.** Emergent suppression exists in the C1-fail ∧ C2-true cell; B misses it, exactly as its 40 % recall predicts. | Reported as the first *prospective* demonstration that B's recall gap is real and costly, and it strengthens the reviewer's own framing that the criterion is a prioritization protocol, not a predictor. |
| **HIGH** | HIGH | **B's precision fails out of sample.** The one certified pair in the census is a false positive; statistic invariance plus per-attack gated C1 was not sufficient. | Reported as a falsification of the precision claim, in the body. §5.1 registers this as the live risk, so it cannot be recast as a surprise. |
| **HIGH** | LOW | **B is inverted on this cell.** The pair it certified is HIGH and the pair it rejected is LOW. | The worst case for the criterion, and reported as such. |

**The all-HIGH outcome is a result, not a wasted wave.** Combined with the seven frozen
census cells it would say: over the *complete* 9-pair census of the region where statistic
preservation holds exactly, **no composition is LOW** — statistic-level invariance selects
9 pairs for 0 true positives. That is a sharper negative statement about strategy A than
anything currently in the paper, and it is available whether or not pair 1 comes out LOW.

**What this wave cannot do.** It cannot establish B as a predictor; n=2 selected pairs
carries no such power, and the paper's disclaimer of predictive validity
(`paper/main.tex:397`) is not weakened or strengthened by any outcome here. It cannot
generalize off CIFAR-10 — that is Wave 4 (FEMNIST), pre-registered separately. And it
cannot escape the structural confound of Lemma `lem:testability`: in scope, C0 ∧ C1 force
d2 to suppress the witnessing attack alone.

## 8. Non-negotiables

- **This file is committed before `results/wave3_emergent/` is written.**
  `run_wave3_emergent.py` enforces it: `prereg_commit()` aborts the run if this file is
  uncommitted *or* has uncommitted modifications, and prints the commit it found. If that
  ordering cannot be demonstrated from `git log`, **the wave is reported as
  non-prospective.**
- **Nothing above §9 is edited after the data exists.** Corrections go below the line, as
  in `pre_registration_wave2.md`, and state what they do and do not change.
- **All four pairs are reported**, including any that come out uninteresting, and the two
  controls are reported even though neither strategy selects them.
- **No pair is added or dropped after the data is seen.** The census is 9 cells because the
  region has 9 cells, not because 9 was convenient.
- **No re-derivation of C0/C1 inputs.** They are frozen artifacts, cited in §4 with their
  files; if any is found to be wrong, that is a correction note, not a re-measurement that
  quietly changes a registered verdict.
- The C2 column is **measured** (§3), not inferred from the proposition. Where a d2 class
  is conditional rather than invariant, that is stated rather than resolved by fiat.

## 9. Provenance

- Mechanism check: `experiments/verify_wave3_invariance.py` →
  `results/wave3_invariance_check.json`, which is committed alongside this file and holds
  every per-round row quoted in §3. (The console transcript
  `results/wave3_invariance_check.log` is **not** committed — `.gitignore:59` ignores
  `*.log` — and is regenerable by re-running the script, which is deterministic: seeds are
  fixed and a second full run reproduced tests (A), (B) and (E) exactly.)
- Runner: `experiments/run_wave3_emergent.py`.
- C0/C1 inputs: `results/pure_defense_baselines/summary.json`,
  `results/wave2_held_out/summary.json`, `results/metric_swap_baselines/summary.json`.
- Census cells: `results/all_compositions/summary.json`,
  `results/wave2_held_out/summary.json`, `results/metric_swap/summary.json`.
- Strategy definitions and the in-sample table of §1:
  `experiments/blind_selection_analysis.py`, `results/screening_cost.json`.
- Prior waves this one is designed around: `experiments/pre_registration_wave2.md`,
  `experiments/pre_registration_prospective.md`,
  `experiments/pre_registration_metric_swap.md`.
