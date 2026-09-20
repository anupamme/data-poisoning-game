# Response to the fifteenth review (6/10, confidence 4/5)

The review named one defect that is real, live, and ours. We concede it, we fixed it, and we did not fix
it the way the review proposed, for a reason we state below. The review also asked for one control that
already exists in the paper under a different name, and it attached a number to that control that belongs
to a different arm; we decline that point by proof and correct the number. Its framing asks are answered
by supersession rather than by argument: the artifact it read is two titles behind the live submission.

Everything below cites the live source by line, as `` `:NNN` `` into `paper/main.tex`. Every quotation is a
verbatim substring of the line it is cited on. Review language is set in *italics* so that no quotation
mark in this letter points anywhere except into the paper.

---

## 1. The one real defect: C1's $d_1$ route was stated without its scope condition

The review is right. The proof sketch for the composition criterion argued C1's sufficiency along two
routes and stated the $d_1$ route unconditionally, as though nothing downstream could undo an upstream
suppression. That is false for any $d_2$ that is invariant to the transformation class $d_1$ suppresses
through, and every $d_1$ in this paper suppresses through a positive per-client rescaling, which is
exactly the class two of our own downstream choices are invariant to.

The route now carries its condition at the point it is argued: inheritance holds `:1422` "provided $d_2$ is not invariant to the transformation class $d_1$ suppresses through", and the same line records that
`:1422` "We had stated this route without the proviso, and it is not vacuous".

A following paragraph gives both witnesses, one by code and one by measurement, `:1431`:

- **Partial undo, measured.** Under one upstream RFA stage on live update stacks,
  `:1431` "Krum's selection moves in $0.533$ of rounds and" `cos_krum`'s in 0.000, at a nonzero
  aggregate displacement of 0.133. The transform acted; that aggregator's decision did not move at all.
- **Complete undo, by construction.** FLTrust gates on a scale-invariant cosine and then renormalizes
  every update to the server-gradient norm, so a positive per-client rescaling cancels exactly on both
  the decision channel and the magnitude channel.

**We did not take the review's proposed fix, and the reason is a measurement.** The review suggested
redefining C1 as *$d_2$ suppresses*. That discards a route we measure five times: the five
`reputation`$\to d_2$ pairs inherit reputation's model-scaling suppression at 0.0146 to 0.0181 against its
own 0.017, precisely because none of those five downstream defenses renormalizes. So we kept C1 symmetric,
left its definition untouched, and added the condition the $d_1$ route needs. The paper is explicit that
this is a bound on the argument and not a correction to data: `:1431` "The proviso bounds the argument and corrects no measured row".

**And the paper now says why it cannot test the proviso on its own menu**, which is a limitation the review
did not ask for and we owe anyway: `:1431` "Testing the proviso needs a $d_2$ that is rescaling-invariant on \emph{both} channels and weak in isolation, and this paper does not contain one."

## 2. Point 11 declined by proof, and its number corrected

The review asked for a control in which the aggregator order is randomized per round. **That control is
already in the paper**, and it is not a new run: `:2902` "We test per-round defense selection."

It is also not merely present but forced to be the same computation. `apply_update`
(`fl_core/federated.py:75-79`) adds exactly one aggregate to the global model per round, and
`Server.aggregate` contains no call to any random number source, so all seven aggregators are deterministic
functions of the round's update stack. A per-round uniform choice among them, applied one per round, *is*
the `temporal_mix` arm of `results/randomized_composition/summary.json` (pre-registered at `e3fb038`).
There is no bit of behaviour a new runner could add.

**The number in the review does not belong to that arm.** Re-derived from the artifact's `per_seed` rows:

| arm | defenses per round | mean ASR, committed_pixel |
|---|---|---|
| `temporal_mix` (per-round uniform over single defenses) | 1 | 0.6600 |
| `rand_comp_strong` (per-round uniform over two-defense compositions, compute-matched) | 2 | 0.1115 |
| `fixed_fg_cm` (fixed composition) | 2 | 0.0988 |

The review's figure of 0.681 is not any cell of that artifact. It is a per-seed value from the
second-dataset replication, where the paper already flags the spread it sits in: `:2549` "pair 1's pixel arm is $0.000/0.107/0.681$ per seed". Read against the right arm, the review's inference reverses:
the compute-matched randomized arm is 0.1115, close to the fixed composition's 0.0988, and the arm that
looks catastrophic at 0.6600 is the one that spends half the defense computation, which the runner labels
as a confound rather than hiding.

## 3. The staleness crosswalk, stated once and without complaint

The reviewed artifact is *Compose, Don't Randomize: A Predictive Criterion for FL Defense Composition
Under Persistence* (Aug. 12). The live submission is *Beyond Statistic Preservation: Causal Evaluation of
Composed Federated-Learning Defenses*. One submission, three titles; the reviewed PDF is not on disk,
having been built outside the repository. Consequences for the review's list, with no request that the
reviewer have known any of it:

- The framing asks are answered by supersession. The surviving material from the earlier framing is
  already labelled as such in the live paper rather than presented as current.
- The base-rate disclosure the review asks for is already printed under the validation table:
  `:2197` "Constant-HIGH baseline on the same 42 pairs: 37/42 = 88.1\%".
- The seed top-up the review asks for already exists at $n{=}30$ in `results/criterion_aware_topup/`,
  pre-registered at `580854a`.

Three further fixes the review prompted, all in the appendix, all zero-compute:

- **The Weiszfeld lemma reasons about a fixed point; the shipped code stops early.** Now disclosed with
  both constants: `:1540` "The iteration is truncated, and the lemma's conclusion does not depend on that."
- **Frozen verdict labels now name their emitter and restate their thresholds** rather than standing alone:
  `:3227` "Pre-specified verdict: INTERMEDIATE." is followed on the same line by the rule and the file that
  emits it, `:3227` "so the label is decided by the runner rather than chosen after reading the table".
  We did not amend any freeze; the literal stays and the measured direction is printed beside it.
- **The self-counting sentence was re-counted against its own glob**, since nothing in the build can see it
  go stale: `:1052` "pre-registration documents fixing the decision rules for every pre-registered arm".

## 4. The confound the review could not have seen, and what we did about it

The paper's strongest oracle-free arm moves the decision channel and the coefficient share at the same
time, which we disclosed post hoc in the previous round. That is the defect no framing edit can repair, so
this round attacked it directly. Four results, the first three carrying no ASR and the fourth going against
us:

**(a) A screen of the whole deployed oracle-free class returns nothing eligible.** Four families, one host
cell, thresholds imported from already-frozen constants rather than chosen afterwards, and an identity
control that is exactly flat on all 15 rows. Every family that moves Krum's decision moves the adversarial
coefficient share past Mode S's own tolerance in 12 of 12 adversary rounds, and the only share-neutral
family is inert, because $\tau{=}5.0$ never binds on these update norms. `:2103` "The dichotomy is the finding". An empty screen is a reportable outcome and was frozen as one:
`:2103` "an empty result is the outcome it was pre-registered to be allowed to return".

**(b) A constructed instrument that is neutral for every adversary set, not just for ours.** Blending each
client's coefficient toward RFA's makes the decision change discrete in the blend parameter and the share
change continuous, so a pair straddling one crossing changes the decision by construction. We report the
supremum of the share gap over all 30 nonempty proper subsets of the participants, `:2105` "which upper-bounds the realized gap for every adversary set and needs no mask to compute". This is
*approximate* uniform neutrality, so it is consistent with our own impossibility result for exact
neutrality rather than in tension with it.

**(c) Our own first freeze of that instrument failed its own gate, and we report it.** It bisected the
crossing of a float64 statistic while the shipped selection runs in float32 and crosses at a slightly
different point, so the converged bracket was tight around the wrong predicate: the supremum share gap
exceeded tolerance in 13 of 13 flip rounds and the shipped selections agreed in 3 of them, which means the
instrument failed to be informative in exactly the rounds it was built for. The naive repair its own
argument implied is perfectly share-neutral and informative on nothing. The general lesson is stated in the
paper rather than buried: `:2107` "bisect the predicate you intend to satisfy, and stop on the gated quantity rather than on a proxy for it."

The corrected instrument meets both gates on the same rounds, 13 of 13 shipped selection differences at a
supremum of at most $9.60{\times}10^{-8}$. **The failed pre-registration was not edited.** It
`:2109` "is left byte-untouched and superseded by" a second document that cites it, records the failure
with these numbers, and corrects the one sentence in it that was wrong, and the append-only claim is
enforced rather than promised: `:2109` "the runner refuses to start unless the superseded document is present, last committed at its recorded hash, and clean".

Four harness checks ran before that ladder and their verdict dict is persisted into the artifact rather
than left in a terminal: the one keyword added to shared code is bit-neutral on a frozen cell at
$\Delta=(0,0)$; the blend at $t{=}0$ reproduces the earlier identity arm at a shared seed, also at
$\Delta=(0,0)$, *and* reports that the hook was invoked 50 times over 50 rounds, which is the assertion
that matters here; the Gram proposal equals the shipped selection at both endpoints; and both gates pass on
the ladder's own pair in all 12 flip rounds that carry an adversary, re-measured live in the same process.

**(d) The ladder ran, and the branch it fired is the one that goes against us.** The instrument's share gate
was written, measured and passed before any outcome existed; then the arm ran, $40$ runs in $8.7$ h, and the
paired difference at $20$ seeds reads **$+0.3218$** (sd $0.3288$, $95\%$ CI $[+0.1680, +0.4757]$, $n{=}20$).
Both frozen verdicts, read separately as the freeze requires: the interval **excludes zero**, and $|\Delta|$
falls **outside** the $0.15$ margin. That conjunction, in the direction of increased ASR, is the branch the
pre-registration named in advance as contradicting this paper's headline reading, so we report it as a
contradiction and not as a scope condition: `:2117` "We report that as a contradiction and do not convert it into a scope condition". Mean ASR rises $0.2485 \to 0.5703$ while mean accuracy also rises
($0.4944 \to 0.5271$), so no accuracy collapse explains it, and no rung is void. The gate held on **all
$2{,}000$ rounds** of both arms, maximum mask-free supremum $9.999{\times}10^{-8}$; the shipped selections
differ in every one of the $1{,}128$ rounds that carry a crossing, and the other $872$ are bit-identical
across arms.

**What we are still not claiming, and the channel we still cannot separate.** The arm separates the decision
channel from attenuation, not from admission, and that gap is structural rather than an oversight: deciding
whether a flip moved Krum onto an adversary needs exactly the mask this instrument refuses, so the artifact
records no admission quantity and could not. The paper states this and declines to use it to soften the
verdict: `:2117` "the refuting branch fired on its own stated terms and we do not use a caveat discovered afterwards to retract it". The contrast is also local, two adjacent points on the blend path rather than an
identity arm against a disturbed one, so §D.7's standing statement that we have produced no oracle-free
measurement of the statistic channel is unchanged, and the masked Mode~S measurement stands as measured.
The device remains an instrument, on the list the ethics statement already marks as such: `:1011` "are measurement devices, not deployable defenses, for two" different reasons.

## 5. Where this leaves the review's three must-fix items

| the review's item | disposition |
|---|---|
| C1's $d_1$ route stated unconditionally | conceded and fixed at `:1422` and `:1431`, symmetric definition kept |
| per-round randomized aggregator control | declined by proof; the arm exists at `:2902`, and the cited 0.681 belongs to a different arm |
| the framing and title asks | superseded; the reviewed artifact is two titles behind the live submission |

The suppression measurement that separates the decision channel from attenuation now exists, it was gated
before any outcome was read, and it contradicts the reading we lead with. We report it that way rather than
convert it into a scope condition, and we leave the masked measurement standing as measured rather than
quietly withdrawing it. What we still cannot offer is a separation of the decision channel from admission,
which needs the mask this instrument exists to avoid, and we would rather say so than report an arm whose
premise we had not checked, which is the mistake this round found in our own work.
