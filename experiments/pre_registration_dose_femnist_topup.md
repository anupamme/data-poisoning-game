# Pre-registration amendment: EMNIST-byclass Mode-S krum arm, seed top-up n=3 -> n=5

Amends `experiments/pre_registration_dose_femnist.md` (frozen at `478b555`). It **adds seeds and
changes nothing else**. Every decision rule below is copied verbatim from that document; where this
file restates a rule, the restatement is the original text and not a revision of it.

This must be git-committed, and `PREREG_COMMIT_TOPUP` in
`experiments/run_dose_femnist_topup.py` set to the resulting hash, before any run. The runner refuses
to start otherwise.

## Why this amendment exists

The original document froze `n=3` with a stated reason:

> `n=3`, declared up front, not chosen after seeing anything: the FEMNIST payoff matrix ran 3 trials
> (`experiments/run_femnist.py`) and 4 rungs x 3 seeds x ~55 min is already ~11 h. Disclosed in the
> paper next to the result.

That was a compute budget, not a statistical argument, and it is the paper's thinnest arm. A reviewer
named it directly: the paper's headline evidence is concentrated on CIFAR-10 at `n=5`, and this
second-dataset replication -- the one arm that answers "is the negative a property of the mechanism or
of one dataset/architecture pair?" -- rests on three seeds. Adding two seeds brings it to the same
`n=5` as the CIFAR-10 arm it replicates, so the two are read at equal power.

**This is a top-up of an existing arm, not a new arm.** The paper already does this twice and both are
disclosed in the body: the Krum practical-equivalence claim was topped up to `n=20`, and `cos_krum`
runs at `n=8`. The `n=8` top-up **flipped** a label (a C1 input read 0.300 over the frozen 3 seeds and
0.584 at `n=8`) and the paper reports the flip rather than the frozen value. That is the precedent
this amendment commits to.

## What is added

- **New seeds: `45`, `46`.** Named here, before any run. The suite's seed convention is consecutive
  from 42; these are the next two and were not chosen by inspecting anything.
- **All 4 rungs** (`kappa = 0, 0.5, 1.0, 2.0`), so the secondary trend test is scored at the same
  width as the primary. 4 rungs x 2 seeds = **8 new runs**, ~56 min each, ~7.4 h.
- Nothing else. Same `N=10, K=5, f=0.2, alpha=0.5, 50 rounds`, same `simple_cnn` on
  EMNIST-byclass, same Mode S with the adversarial coefficient pinned at `c=1`, same `run_one`
  imported from `run_targeted_dose` rather than forked.

## Decision rules: UNCHANGED, evaluated at n=5

Primary, on `Delta = mean ASR(kappa=2) - mean ASR(kappa=0)`, against the same `EQUIV_MARGIN = 0.15`
(no new constant):

| outcome | verdict |
|---|---|
| `abs(Delta) < 0.15` | FLAGSHIP NEGATIVE REPLICATED on a second dataset and architecture |
| `Delta > +0.15` | THE NEGATIVE IS DATASET- OR ARCHITECTURE-SPECIFIC; the paper's central claim is scoped to CIFAR-10 and must say so in the body |
| `Delta < -0.15` | Attenuation-side fall. INDETERMINATE, reported as such and NOT scored in our favour |

Secondary: Jonckheere-Terpstra across all four rungs, scored by
`experiments/analyze_dose_femnist.py` against these same rules.

Accuracy gate: every rung must hold mean clean accuracy `>= ACC_FLOOR = 0.35`, or the cell is
uninterpretable and no verdict stands.

The frozen `n=3` result was `Delta = -0.017` against CIFAR-10's `-0.026`, i.e. REPLICATED. **`n=5` is
therefore a test this arm can fail**, and the three ways it fails are the three rows above.

## What we commit to reporting

1. **Both values, always.** The paper reports the `n=5` Delta and the frozen `n=3` Delta beside it.
   The `n=3` number is never silently replaced, and `results/dose_femnist/summary.json` is not
   rewritten -- the top-up writes to `results/dose_femnist_topup/`, and the frozen file's md5 is
   recorded before and after to prove it.
2. **A flip is reported as a flip**, in the same sentence as the count, exactly as the `n=8` flip is.
   If `abs(Delta)` crosses 0.15 the body says the replication does not survive two more seeds and the
   central claim is scoped to CIFAR-10.
3. **No margin is refitted.** `0.15` is the suite's existing constant. If the result lands outside it,
   the margin does not move to accommodate it.
4. **Per-seed values for all five seeds** go in the appendix, as every other arm's do.
5. **If the accuracy gate fails at any rung**, the arm is reported uninterpretable at `n=5` and the
   `n=3` verdict is reported as not superseded rather than as confirmed.

## Guards that must pass before the first run

1. **The scaling hook must be live.** `committed_scaling` and `committed_pixel` share
   `poison_dataset`; `attack.manipulate_update` is the only thing separating them, and when a
   different runner omitted that call an entire wave came back bit-identical to the pixel arm and
   silently wrong (`results/femnist_c1_inputs_scaling_INVALID/`). `run_targeted_dose.run_one` does
   call it (line 175); the runner asserts the call is still present in the source and aborts if it is
   not.
2. **The `kappa=0` rung is its own functional check, at no extra cost.** `kappa=0` is
   identity-then-krum, i.e. krum alone, so the two new seeds' `kappa=0` ASR must land near the FEMNIST
   payoff matrix's `krum/model_scaling` figure of `0.027` at accuracy `0.767`. A skipped scaling hook
   would instead return the pixel arm's `0.660`. The runner aborts if the new seeds' `kappa=0` mean
   ASR exceeds `0.35`, which no seed-noise excursion around `0.027` reaches and no pixel-arm value
   avoids.
3. **The premise stands unchanged.** `results/femnist_admission.json` must exist: the arm's
   interpretability rests on the Mode-S ladder disturbing Krum's decision on EMNIST-byclass
   (0.000 / 0.643 / 0.714 / 0.643) while admission stays at 0.000 at every rung, and that was measured
   prospectively, before any ASR existed. The top-up does not re-measure or re-open it.

## What this amendment does not do

- It does not add an arm, a defense, an attack, a dataset or a rung.
- It does not touch any frozen artifact, any published table, or the six-cell comparability
  pre-registration and its `[OK] ... f16083b` gate.
- It does not change `EQUIV_MARGIN`, `ACC_FLOOR`, or the naming convention: the arm is
  **EMNIST-byclass** in all prose and tables, and `femnist` survives only in artifact paths, this
  filename's lineage and JSON dataset keys.
