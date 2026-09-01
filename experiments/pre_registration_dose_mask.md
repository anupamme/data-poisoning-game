# Pre-registration: Mode M, coordinate masking — a second transformation class (Round 34, item B)

**Status: frozen before any write to `results/dose_mask/`.** Every predicted shape, decision rule and
refutation criterion below is reproduced verbatim in `experiments/run_dose_mask.py`, which refuses to
start until this file is committed and its hash is recorded in `PREREG_COMMIT`. The channel
measurement this document quotes (`results/mask_admission.json`) computes no ASR and legitimately
predates the freeze, for the same reason `results/admission_measurement.json` predated `5130cec`; it
is committed together with this file so the ordering is auditable.

## The question this arm exists to answer

Every upstream transformation the paper has tested — Round 11's `dose_kappa<κ>`, Round 12's Mode S
and Mode A — is of the form `u_i ↦ c_i · u_i` with `c_i > 0`. So a reader can grant the whole result
and still object: **the (P3)⇏(P4) non-implication may be an artifact of positive scalar
multiplication.** It is a fair objection and no existing arm can answer it, because the entire
apparatus, including `thm:bounded_reweight` and `prop:invariance`, is built on that family.

Mode M leaves the family. It is **not** of the form `c_i · u_i` for any scalar `c_i`, so
`thm:bounded_reweight` and `prop:invariance` **do not cover it** — that is the design, not a gap. If
the flagship negative replicates here, the non-implication is a statement about statistic
preservation rather than about rescaling. If it does not replicate, the paper's central claim has a
scope condition it did not know about, and that is what gets reported.

## Construction: `d1 = doseM_m<m>`

For the `K` participants of a round, `A` adversarial and `B` benign, at drop rate `m`:

```
i ∈ A:  u_i ↦ u_i                                         (returned as the same object)
j ∈ B:  per floating-point tensor t of u_j,
          keep_t ~ Bernoulli(1 − m) i.i.d. per coordinate,
          t ↦ (t ⊙ keep_t) · ‖u_j‖₂ / ‖u_j ⊙ keep‖₂       (norm restored over the whole update)
```

Masks are drawn from `default_rng([seed, round, benign_slot])`, so every rung at a given
`(seed, round)` sees the **same** masks and the rungs differ only in `m`. Integer and non-floating
tensors pass through untouched. `m = 0.0` returns the update list **unwrapped**, so the identity rung
is bit-identical to `d₂` standalone by construction, exactly as Modes S and A do at `κ = 0` / `ν = 0`.
Rungs are `m ∈ {0.0, 0.2, 0.5, 0.8}`, four of them so the table shape matches every other arm.

### Two properties verified before this freeze

- **The adversarial contribution is pinned exactly.** Adversarial updates are returned as the *same
  tensor objects*, not copies and not rescaled-by-1.0 copies. Measured: `c_adv ∈ [1.0, 1.0]` with
  maximum deviation from 1 of exactly `0.00e+00` at every rung, and object identity `True` for every
  adversarial row at every rung. This is **strictly stronger than Mode S**, whose adversarial
  coefficient share is constant only to the `3.5×10⁻⁷` float32 read-back noise of a ~1.1M-dimensional
  norm.
- **Benign norms are preserved.** In float64 on live `cifar_cnn` update stacks, the worst
  `|‖T(u_j)‖ / ‖u_j‖ − 1|` over 30 client-rungs is **`4.93×10⁻⁸`**, against a declared tolerance of
  `1×10⁻⁶`. The float32 read-back used by the shipped measurement path shows a larger, systematically
  positive deviation — `0.00e+00 / 4.11×10⁻⁵ / 9.27×10⁻⁵ / 1.40×10⁻⁴` across the four rungs — because
  it recomputes a ~1.1M-term norm in single precision and the surviving coordinates are scaled by
  `≈1/√(1−m)`, so the noise grows with `m`. **The claim is the float64 one; the float32 figure is
  disclosed as a property of the measurement, not of the transform.** Two tolerances are therefore
  frozen: `EXACT_TOL = 1e-6` on the float64 construction check (the claim) and
  `READBACK_TOL = 2e-4` on the float32 read-back (a printed verdict only). This mirrors the existing
  two-tier treatment of `SHARE_TOL` in `measure_admission.py:60-67`.

### Why the mask is renormalized, and what was rejected

An **unnormalized** mask was considered and **rejected before any ASR existed**. Zeroing a fraction
`m` of coordinates shrinks a benign update's norm by `≈√(1−m)`, which re-opens precisely the
magnitude channel that the score-only control (`pre_registration_score_only.md`, `35788d9`) was built
to close: benign clients would arrive systematically smaller than adversarial ones, and any rise in
ASR would be attributable to relative down-weighting rather than to structural damage. Renormalizing
to each client's own original L2 norm holds `‖u_j‖` fixed for **every** client at **every** rung, so
the dial moves *direction and support* and not magnitude. Recorded here rather than quietly omitted.

## The channels, measured prospectively (`results/mask_admission.json`, no ASR)

5 seeds × 3 live rounds × 4 rungs, every rung on the same raw updates, CIFAR-10 / `cifar_cnn`,
class-(c) arms on `committed_scaling` and class-(a)/(b) arms on `committed_pixel`, i.e. the identical
configuration and attack assignment `measure_admission.py` used for the frozen Mode-S measurement.
**decision change / admission change:**

| arm | m = 0.0 | m = 0.2 | m = 0.5 | m = 0.8 |
|---|---|---|---|---|
| `krum` | 0.000 / 0.000 | 0.133 / **0.000** | 0.400 / **0.000** | 0.467 / **0.000** |
| `reputation` | 0.000 / 0.000 | 0.333 / 0.0006 | 0.467 / 0.0021 | 0.533 / 0.0037 |
| `cos_krum` | 0.000 / 0.000 | 0.133 / 0.133 | 0.267 / 0.133 | 0.467 / 0.333 |
| `coord_median` | 0.000 / 0.000 | 0.316 / 0.0099 | 0.589 / 0.0537 | 0.700 / 0.1306 |

Relative aggregate displacement `‖agg(T(U)) − agg(U)‖ / ‖agg(U)‖` for `krum`:
**0.000 / 0.532 / 0.888 / 1.095**.

**The two transformation classes are not ordered by disturbance; they disturb different things.** At
its top rung Mode M moves Krum's aggregate *further* than Mode S does (**1.095** against **0.892**)
while disturbing Krum's *decision* **less** (**0.467** against **0.733**). Neither class dominates
the other, which is why replicating on the second one is informative rather than redundant.

### Arm type, and the rule that assigned it

The eligibility rule was written into `experiments/measure_admission_mask.py` **before** its numbers
were read, in three branches: decision change `< 0.10` at the top rung ⟹ **INELIGIBLE** (no dose);
maximum admission change `≤ 0.05` ⟹ **GENERALIZATION** test with a **FLAT** frozen prediction;
admission change moves ⟹ **ADMISSION** test, in which ASR is predicted to move *with* admission.

Krum's decision change reaches **0.467** and its admission change is **exactly 0.000 at every one of
the four rungs**. So the `krum` mask arm is a **GENERALIZATION** test: it has the same premise the
CIFAR-10 flagship rests on — the statistic's decision is disturbed and the admitted adversarial mass
is not — now outside the scalar-rescaling family.

## Arm

`d₂ = krum`, attack `committed_scaling`, `m ∈ {0.0, 0.2, 0.5, 0.8}`, **seeds 42–51 (`n = 10`)**.
Selected by the paper's own two rules applied before any ASR exists: standalone Krum genuinely
suppresses model-scaling at usable accuracy, and the instrument genuinely moves that arm's decision.

`n = 10` rather than 5, frozen here: this arm's entire job is to say whether a negative replicates,
and a negative at `n = 5` was the weakness the fourth review identified in the first place.

**One arm was considered and dropped for compute, and the drop is recorded.** A `cos_krum`/pixel mask
ladder would be the most striking ASR cell in the matrix below, but it would need `n = 8` for the
same bimodality reason that forced `SEEDS8` (identity rung 0.003–0.864, 95% CI [0.058, 0.922]) and
would add ≈32 runs at ≈15–20 min each. Its **channel** measurement is reported instead, and is
labelled a channel result, never an outcome result.

## The cross-class prediction matrix, frozen before any ASR

This is the falsifiable content that the second transformation class buys for free.

| `d₂` | class under `prop:invariance` | rescaling (Mode S) predicted | rescaling observed | masking (Mode M) predicted | masking observed |
|---|---|---|---|---|---|
| `cos_krum` | (a) exactly invariant | decision unchanged | **0.000** | **guarantee lapses: decision changes** | 0.467 (channel) |
| `krum` | (c) not invariant | decision changes | 0.733 | decision changes | 0.467 (channel) |
| `coord_median` | (b) conditionally inv. | partial | 0.482 | partial | 0.700 (channel) |
| `reputation` | (c) not invariant | decision changes | 1.000 | decision changes | 0.533 (channel) |

**`cos_krum` is the adjudicating cell of the matrix, and it reclassifies.** It is *exactly* invariant
under positive rescaling — 0.000 selection changes at every Mode-S rung, the strongest invariance any
arm in this paper has — and under masking its selection changes in **0.467** of rounds and its
adversarial admission in **0.333**. One transformation class certifies an aggregator that the other
cannot, on the same aggregator, the same attack and the same seeds. That is the content of
"algebraic invariance is a statement about a transformation class, not about an aggregator", and it
is now measured rather than asserted.

`cos_krum`'s Mode-M numbers are a **channel** claim from `results/mask_admission.json`. Its ASR under
masking is not measured, and the matrix prints `--` in any outcome cell no run supports rather than
omitting the row.

## Decision rules

All tests one-sided at α = 0.05 where directional, Jonckheere–Terpstra across `m ∈ {0, 0.2, 0.5, 0.8}`
in that order, via the canonical `jonckheere` in `analyze_dose_response.py`, with a fixed-seed
permutation `p`-value alongside the normal approximation. Margin and floor are carried forward
unchanged from `5130cec`.

**H-M-statistic** (the reading Round 11 tested): ASR rises monotonically in `m`, because the
statistic's decision is disturbed.
- JT *increasing* `p < 0.05` ⟹ **H-M-statistic confirmed, H-M-admission refuted.** Reported as: the
  (P3)⇏(P4) non-implication **does not generalize** beyond scalar rescaling, and the paper's central
  claim acquires a scope condition naming the transformation class.

**H-M-admission** (the paper's reading): ASR is flat, because the admitted adversarial mass is
unchanged at every rung.
- `|mean ASR(m=0.8) − mean ASR(m=0)| < 0.15` **and** every rung mean `< 0.5` ⟹ **H-M-admission
  confirmed on a second transformation class.**
- Reported with the **two-sided 95% paired Student-t interval** on the per-seed difference
  `ASR_s(m=0.8) − ASR_s(m=0)` (`t₉ = 2.262`), and the two one-sided TOST statistics against the
  ±0.15 margin. Containment of the 95% interval is the criterion, which is strictly more
  conservative than TOST at α = 0.05.

**Neither ⟹ indeterminate**, reported as such and not scored in our favour.

**Accuracy gate, and the specific way this arm can fail it.** `ACC_FLOOR = 0.35` on a rung's **mean**
clean accuracy, unchanged. This arm is at materially higher risk of failing it than any rescaling arm,
and the reason is named in advance: when Krum selects a masked benign client, the emitted update has a
fraction `m` of its coordinates zeroed and the survivors inflated by `≈1/√(1−m)`, so at `m = 0.8` the
global model is being updated by a heavily sparsified, heavily amplified direction 50 rounds running.
**A low ASR at `m = 0.8` under a failed accuracy gate is a collapsed model, not preserved
suppression**, and will be reported as uninterpretable rather than as confirmation. If the top rung
fails the floor, the equivalence verdict is read on `m = 0.5` against `m = 0` instead, and the
substitution is disclosed as a substitution — this contingency is frozen here precisely so that it
cannot look like a choice made after seeing the numbers.

## The branch that would refute us, named in advance

Heavy structural damage to benign updates could make Krum start selecting the **adversary**. If that
happens, the admitted adversarial mass moves, the equivalence claim is void at that rung, and

> **that is a confirmation of the admission reading and a refutation of nothing.**

It would be reported as an admission change: the transform stopped being admission-preserving, so a
rise in ASR is exactly what the paper's own reading predicts, and the arm would then be evidence
*for* (P4) being the operative level rather than evidence against the non-implication. Writing this
branch down before the run is what makes the arm a test rather than a demonstration — but it is also
why the arm cannot be scored as a win in every direction, and the honest statement of that is: **only
FLAT-with-admission-unchanged generalizes the negative.** A rise with admission unchanged refutes us.
A rise with admission changed is uninformative about the non-implication and is reported as
uninformative.

**What "admission unchanged" rests on.** The admission figures are from
`results/mask_admission.json`: 5 seeds × 3 live rounds, i.e. a measurement of the *transform's*
channel on this configuration, not a per-round guarantee inside each 50-round ASR run — `run_one`
does not log selections. This is exactly the standing of the flagship's own admission numbers, which
come from `measure_admission.py` at the same 5 seeds × 3 rounds. It is the established basis for this
paper's admission claims, and the limitation applies equally to the published (P3)⇏(P4) witness.

## The score-only mask control

Not optional: without it the new arm would be less well instrumented than the arm it generalizes, and
the fourth review would be right to say so. Reusing `generic_compose`'s existing
scoring-versus-emitting split (`score_only=True`, generic over `d₁`'s name and defined for selectors,
so no new plumbing), `m ∈ {0.2, 0.5, 0.8}` × seeds 42–51 = **30 runs**:

- **score-only:** Krum *selects* using masked updates and *emits* the selected client's **original,
  unmasked** update.
- **full (the main ladder):** Krum selects using masked updates and emits the masked one.

Because adversarial updates are unmasked in both conditions, the two differ only when a benign client
is selected. The control therefore separates the two ways masking can act: through **which** client
Krum picks, and through the **damage to what it then emits**. Frozen prediction: if the main ladder is
FLAT, the score-only ladder is FLAT as well, and any difference between them localizes the effect to
the emitted content rather than to the selection.

## Runs

- **Main ladder.** 3 non-identity rungs × 10 seeds = **30 runs**. The `m = 0` rung is **not re-run
  where a bit-identical run exists**: `m = 0` returns the update list unwrapped and `run_one`'s
  participant RNG stream does not depend on `d₁`'s name, so it is the same computation as Krum
  standalone and as the Mode-S `κ = 0` rung. It is imported from `results/dose_response/` for seeds
  42–46 and from `results/dose_seed_topup/` for seeds 47–51, computed in place otherwise, with
  per-seed provenance recorded in the output. `--harness-check` computes `m = 0` in-suite **on both
  ladders** at a seed where the imported value already exists — a seed without one would make the
  check vacuous — and asserts agreement to `< 1e-9`. The score-only claim holds for the same reason:
  at `m = 0` the scoring stack and the raw stack are the same object, so Krum scores on the raw stack
  and emits the raw selected update, which is Krum alone. That is the argument
  `run_score_only_control.py` verified for Mode S at `35788d9`; it does not depend on the mode, and
  it costs one run to assert rather than cite. **If either assert fails the suite does not run.**
- **Score-only control.** **30 runs.**
- **Total ≈60 new runs**, ≈15–20 min each on the paper's standard configuration (N=10, K=5, f=0.2,
  α=0.5, 50 rounds, `cifar_cnn`), resumable, written after every run to
  `results/dose_mask/summary.json`.

## Caveats recorded in advance

1. **Mode M has nothing to impose in 20.0% of rung-rounds** (24/120 measured): a round with no
   adversary, or with fewer than two benign participants, admits no benign masking. Identical to
   Mode S's figure at the same configuration, and re-reported with the results.
2. **A mask is one non-rescaling transformation, not the class of them.** Rotations, projections,
   quantization and sign compression are all outside `c_i · u_i` too, and none of them is tested here.
   The claim this arm can support is "the negative replicates on a second, structurally different
   transformation class", never "on non-rescaling transformations in general".
3. **Mode M is an instrument, not a defense.** It reads adversary identity: it masks benign clients
   and leaves adversarial ones untouched. No deployable transformation knows which clients are
   adversarial. It is never presented as a defense, and its exact adversarial pinning is the
   *point* — it is what closes the attenuation channel — not a claim of realism.
4. **`thm:bounded_reweight` and `prop:invariance` make no prediction here**, by construction. Any
   theory cell for Mode M in the matrix above reads "guarantee lapses" or `--`, and no result from
   this arm is presented as a test of either statement.
5. **The renormalization is per client to its own original norm**, which holds `‖u_j‖` but not the
   *relative* weighting implied by direction: two benign clients whose updates are near-collinear
   before masking need not be after. That is the intended structural damage, and it is also the
   reason the aggregate displacement (up to 1.095) exceeds Mode S's at a smaller decision change.
6. **Seeds 47–51 overlap `SEEDS8`** (the `cos_krum` Mode-S arm) and the item-A top-up range 47–61.
   Different arms, different cell keys, no collision — recorded because the overlap looks like one.

## Non-negotiables

1. No predicted shape, decision rule, equivalence margin, accuracy floor, rung grid or refutation
   criterion above is revised after seeing an outcome. An arm meeting neither its confirmation nor
   its refutation criterion is INDETERMINATE, not scored in our favour.
2. The rung grid is `m ∈ {0.0, 0.2, 0.5, 0.8}` and the seed list is 42–51. Neither is extended,
   truncated by inspection, or filtered.
3. Every reported number is recomputed per seed from `results/dose_mask/summary.json` and
   `results/mask_admission.json`, never transcribed.
4. `doseM_m<m>` is never presented as a defense, and no Mode-M result is presented as a test of
   `thm:bounded_reweight` or `prop:invariance`.
5. **Nothing published is rescored.** The Mode-S verdicts, Round 11's refutation and the `dose`,
   `doseS`, `doseA` cell keys stand exactly as reported, whatever this arm shows. `RUNGS` in
   `measure_admission.py` is not edited; Mode M is measured through the same loop by passing its own
   `rungs` list.
6. **If the mask arm refutes the generalization, that is reported in the abstract** as a scope
   condition on the paper's central claim.
7. This file is committed before `results/dose_mask/` is written; if that ordering cannot be
   demonstrated from `git log`, the arm is reported as non-prospective.
   `results/mask_admission.json` legitimately predates it because it computes no ASR, and is
   committed together with this file.
