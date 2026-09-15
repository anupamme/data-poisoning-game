# Pre-registration: the FoolsGold->RFA consensus-shift arm, re-run with the attack actually applied (Round 69, ninth review)

**Status: frozen before any write to `results/fg_rfa_consensus_shift_fixed/`.** The defect proof, the single
moving factor, the harness check, the decision rules and the disclosure clause below are reproduced in
`experiments/run_fg_rfa_consensus_shift_fixed.py`, which refuses to start until this file is committed and its
hash is recorded in `PREREG_COMMIT`.

**This arm exists because the paper published a null result that was not measured.** The defect is ours and
was found by our own audit, not by the ninth review, which does not mention the consensus-shift arm.

## The defect, proved two independent ways

`main.tex:2236`--2237 print a row of `tab:fg_rfa_flagship`:

> *Consensus-shift (n=5)* -- rate $= 0.05$ -- $0.045 \pm 0.012$

and `main.tex:2242` reports it as a measured null, twice: in the count *"Three of the six adaptive strategies
beat the base attacks"* and in the list *"the other three sit level with the base (the same criterion-aware
attack without decorrelation $0.050$, Neurotoxin $0.048$, consensus-shift $0.045$)"*.

**The consensus-shift perturbation was never applied.** `experiments/run_fg_rfa_flagship.py:343` constructs
the attack as

```python
attack = get_attack("backdoor_pixel")
```

and then at `:370` and `:378` calls `attack.manipulate_update(...)`. `BackdoorPixelAttack`
(`attacks/attack_strategies.py`) does **not override** `manipulate_update`, so it inherits the base class's
identity at `attacks/attack_strategies.py:61`--63:

```python
def manipulate_update(self, update, global_model):
    return update
```

Therefore the runner's Phase-1 shift at `:374`,

```python
shifted[k] = update[k] + shift_rate * (backdoor_update[k] - update[k])
```

evaluates to `update[k] + shift_rate * 0`, which is **exactly `update[k]` for every value of `shift_rate`**,
and the Phase-2 injection at `:378` is likewise the identity. The run reduces to the base committed-pixel
composition: pixel-poisoned datasets, no update manipulation, `generic_compose(..., "foolsgold", "rfa",
tau=5.0)`.

The artifact witnesses this without reference to the code. `results/fg_rfa_flagship/summary.json`'s
`adaptive_consensus_shift/rate_0.01`, `adaptive_consensus_shift/rate_0.05` and
`base_composition/committed_pixel` agree on **all five seeds, in both ASR and clean accuracy, to twelve
decimal places**:

| seed | ASR (all three cells) | clean accuracy (all three cells) |
|---|---|---|
| 42 | `0.025666666667` | `0.4610` |
| 43 | `0.051111111111` | `0.5333` |
| 44 | `0.038555555556` | `0.5054` |
| 45 | `0.059222222222` | `0.4702` |
| 46 | `0.051555555556` | `0.5055` |

Two shift rates a factor of five apart cannot produce bit-identical clean accuracy under a live perturbation.
The printed `$0.045 \pm 0.012$` is the base pixel arm's own `n = 5` figure re-labelled as an attack result.

**This is the second instance of this failure class in this repository.** The first was a manipulation hook
that was never called and made one composition arm bit-identical to another, which is why every check in this
file is a per-seed value comparison and never an md5 of a results file.

## Not affected, checked rather than assumed

- **`(L3)` at `main.tex:2499`.** Its rep$+$tm consensus-shift figures ($0.684 \pm 0.138$ at `n = 10`,
  $0.667 \pm 0.137$ at `n = 8`) come from `experiments/run_consensus_shift_vs_composition.py`, which
  constructs the real `ConsensusShiftAttack` at `:114` and calls its live `manipulate_update` at `:143`.
  That arm is sound and is **not re-run**.
- **The projection and Neurotoxin rows of the same table.** Both differ from the base pixel arm per seed, so
  their hooks fire. `adaptive_projection/eps_5.0` and `adaptive_neurotoxin/top0.20_scale2.0` were each
  compared row-by-row against `base_composition/committed_pixel` on seeds 42--46 and are **not** identical.
  Neither is re-run.
- **`base_composition` at `n = 30`.** Unchanged, and it is the leg this arm is compared against.

## What is run, and the single factor that moves

**Run here: the fg$\to$rfa consensus-shift arm with the live attack class.** Shift rates
`{0.01, 0.05}`, seeds `42--46`, `n = 5` per rate, **10 runs**. At the measured 22.9 min/run for this
configuration (`results/criterion_aware_topup/`, same `cifar_cnn` at 50 rounds on the same composition), that
is **~3.8 h**, sharing the GPU with the ResNet18 converged arm.

**One factor moves against the void arm: the attack object.**

| | void arm (`run_fg_rfa_flagship.run_consensus_shift`) | this arm |
|---|---|---|
| attack | `get_attack("backdoor_pixel")`, identity `manipulate_update` | `ConsensusShiftAttack(shift_rate=r, shift_rounds=25, scale_factor=1.0)` |
| phase logic | hand-rolled in the runner, arithmetically zero | inside the attack class, live |
| aggregation | `generic_compose(..., "foolsgold", "rfa", tau=5.0)` | *identical* |
| config | `N=10, K=5, f=0.2, 50 rounds, cifar_cnn, Dirichlet 0.5` | *identical* |
| seeds, rates | `42--46`, `{0.01, 0.05}` | *identical* |

`FL_CONFIG`, `ADV_FRACTION`, `generic_compose` and `evaluate_backdoor` are **imported** from the existing
modules, never copied, so the composition and the metric cannot drift between the void arm and this one.

**Two semantics of the attack class are frozen here rather than corrected**, because correcting them would
make this arm incomparable to the published rep$+$tm arm it is meant to sit beside:

1. **One `ConsensusShiftAttack` instance is shared across adversaries per run**, exactly as
   `run_consensus_shift_vs_composition.py:114` does. The class increments `self._round` once per
   `manipulate_update` call, so with `K = 5` sampled from `N = 10` at `f = 0.2` the counter advances per
   **adversary-update event**, not per round, and `shift_rounds = 25` is 25 such events rather than 25 rounds.
   This is the published arm's own semantics and is not changed.
2. **`scale_factor = 1.0`, so Phase 2 is the identity** (`attacks/consensus_shift_attack.py:73`--77 returns
   `update` unchanged when `scale_factor == 1.0`). The attack's entire distinguishing mechanism is Phase-1
   drift; Phase 2 is ordinary committed-pixel behaviour. The paper already discloses exactly this at
   `main.tex:2499` -- *"Phase 2 degenerates to standard committed\_pixel"* -- so the arm is expected to be a
   **small** effect, and this file says so before the runs.

## The primary quantity, fixed now

- **Primary: the paired difference against the base pixel arm.**
  `d_s = ASR_s(consensus_shift, rate r) - ASR_s(committed_pixel)` over seeds 42--46, per rate; two-sided 95%
  Student-t interval on `mean(d)` with `n - 1 = 4` degrees of freedom. Pairing is exact by construction: same
  seed, same Dirichlet partition, same model initialization, same participation sequence.
- **The base leg is reused, not re-run**, from `results/fg_rfa_flagship/summary.json`
  `base_composition/committed_pixel` restricted to seeds 42--46 (mean `0.0452`, sd `0.0118` at `ddof = 0`).
  It is the same five rows tabulated above.
- **Reported beside it, descriptively:** the mean and `ddof = 0` sd per rate, in the format
  `tab:fg_rfa_flagship` already uses, and the per-seed rows.
- **No ratio.** This arm is not a ratio claim and none is computed.

**A 5-seed interval on this composition is wide, and that is stated now rather than discovered later.** The
base leg's own `n = 30` distribution is right-skewed with mean `0.0644` against median `0.0517`
(`main.tex:2209`), and seeds 42--46 sit in its left tail. So a null at `n = 5` is weak evidence of a null,
and the decision rules below are written to say so.

## Decision rules

- **The void row is withdrawn regardless of outcome.** The published `$0.045 \pm 0.012$` is superseded and the
  defect is disclosed in the paper, not only in the response letter. No outcome of this arm makes the shipped
  number retrospectively defensible.
- **Attack ineffective (expected):** the paired interval contains zero at both rates. Then the row is
  **restored with the honest figures**, the count at `main.tex:2242` stays *"three of the six"*, and the list
  sentence keeps consensus-shift among the arms that sit level with the base -- now for a measured reason. The
  weakness of a 5-seed null is stated in the same clause.
- **Attack effective:** the paired interval excludes zero and lies above it at either rate. Then
  `main.tex:2242`'s count moves to *"four of the six"* (or higher), the list sentence drops consensus-shift
  from the level-with-base group, and the paragraph's ranking is re-derived. In particular the claim that the
  criterion-aware $\epsilon{=}1.0$ adversary is *"the strongest adversary we built"* is re-checked against the
  new figures and rewritten if it is no longer true.
- **Attack harmful to the adversary:** the interval excludes zero and lies below it. Reported as measured; the
  two-phase schedule costs the adversary ASR on this composition. Not scored in the defense's favour beyond
  what the interval supports.
- **Accuracy floor.** `ACC_FLOOR = 0.35` on each rate's mean clean accuracy, the paper's own convention at
  `main.tex:1724`. The base leg clears it comfortably (`n = 5` mean `0.4951`). Per-seed accuracies are recorded
  for all 10 runs and any individual seed below the floor is flagged even where the mean clears it. If a
  rate's mean falls below `0.35`, that rate is reported as **uninterpretable for ASR** rather than as a
  suppression result.
- **No stopping rule and no interim look.** The rate list is `{0.01, 0.05}` and the seed list is `42--46`,
  both fixed here. If the runs are interrupted, the analysis reports the `n` actually reached per rate and the
  interval at that `n`. Progress is counted from `per_seed` entries in the artifact, never from a `[i/N]` log
  position.

## The harness check, run before any new run

`run_fg_rfa_consensus_shift_fixed.py --harness-check` runs **seed 42 through this file's own loop with
`get_attack("backdoor_pixel")` substituted for the attack object** -- that is, in the void arm's
configuration -- and asserts agreement with the stored `base_composition/committed_pixel` seed-42 row
(`accuracy 0.4610`, `asr 0.025666666667`) to `< 1e-9`.

This check is chosen deliberately over re-running a consensus-shift row, and the reason is the whole point of
the arm: **the stored consensus-shift rows are void, so agreeing with them would prove nothing.** Reproducing
the *base* row through the new loop proves that the loop, the aggregator, the dataset partition, the metric
and the RNG consumption are unchanged, which isolates the attack object as the single moving factor. It costs
one 50-round run (~23 min).

**If that assertion fails, the arm does not run.** It is a value comparison against stored per-seed rows, not
an md5 of a results file.

## Caveats recorded in advance

1. **This is one composition, two rates, five seeds.** The arm repairs one void table row. It is not evidence
   about consensus-shift against any other composition, and it does not widen the adaptive suite.
2. **`n = 5` on a right-skewed denominator.** A null here does not establish that the attack is ineffective at
   `n = 30`; it establishes that it is not effective on the five seeds the table reports. The arm is not
   topped up to `n = 30`, so no claim at that `n` is made.
3. **Rates `0.1` and `0.2` are not run.** The rep$+$tm arm swept four rates; the fg$\to$rfa table prints two,
   and this arm replaces exactly what the table prints. The two higher rates remain unmeasured against this
   composition and no statement is made about them.
4. **The attack's Phase 2 is by construction ordinary committed pixel**, per the frozen semantics above. If
   the arm comes out level with the base, that is partly mechanical and the report says so rather than
   presenting a null as robustness.
5. **The void arm's numbers are not deleted from disk.** `results/fg_rfa_flagship/summary.json` keeps its
   `adaptive_consensus_shift` cells so the defect stays auditable; the paper cites this arm's directory for
   that row and names the superseded cells.

## Non-negotiables

1. No interval definition, accuracy floor, primary quantity or decision rule above is revised after seeing an
   outcome.
2. The rate list is `{0.01, 0.05}` and the seed list is `42--46`. Neither is extended, truncated by
   inspection, or filtered.
3. `results/fg_rfa_flagship/` and every other existing `results/` directory are **not written**. This arm
   writes `results/fg_rfa_consensus_shift_fixed/` only.
4. `experiments/run_fg_rfa_flagship.py`, `experiments/run_all_compositions.py` and
   `attacks/consensus_shift_attack.py` are **imported, never edited and never copied**. In particular the
   buggy `run_consensus_shift` is left in place as the record of the defect; it is not silently repaired.
5. **The defect is disclosed in the paper**, at the table and in the response letter, whatever the new figures
   are. A corrected number that appears without the correction being stated is a worse outcome than the
   original error.
6. This file is committed before `results/fg_rfa_consensus_shift_fixed/` is written; if that ordering cannot
   be demonstrated from `git log`, the arm is reported as non-prospective.
