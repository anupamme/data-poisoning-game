# Response to Reviewer (6/10, Weak Accept / Borderline; confidence 4/5)

Thank you. You credit the revision with fixing C1-necessity, the theorem scope and the adaptive-attack
narrowing, and you name the Mode S intervention and the score-only control as what changed your
reading. We take the report as a specification and answer all ten priorities below, in your order of
importance rather than the paper's.

## 0. You and the previous reviewer asked for opposite things, and this is how we resolved it

The report immediately before yours, on the same draft, closed with:

> the marginal value of more experiments is now lower than the marginal value of making the conceptual
> contribution unmistakable

and listed as explicit non-goals: no more defenses, no hyperparameter sweeps, **no more seeds**, no
third dataset, no fourth attack. We accepted that in full and ran nothing.

Yours closes with:

> Do the 30-seed flagship experiment before anything else.

**We do not think these are actually in conflict, and the reconciliation is what this revision is.**
The previous reviewer forbade **breadth**: more defenses, more datasets, more attacks, more sweeps.
You ask for **depth**: precision on the flagship arm the paper already rests on, and one test of
whether the phenomenon survives outside the scalar-rescaling family. So this round adds **no defense,
no attack, no dataset, no architecture and no hyperparameter sweep**. It adds seeds to two existing
arms and one new *upstream transformation class*. Every one of the previous reviewer's non-goals is
still honoured except "no more seeds", and that one we break deliberately, in the direction you asked,
with a pre-registered reversal clause that we describe below because adding seeds after seeing a
result is exactly the practice this paper condemns elsewhere.

We also state plainly what this means for a reader of both reports: two competent referees read the
same draft and reached opposite instructions about compute. We have not tried to hide that by
presenting the new runs as though they were always planned.

---

## Priority 2: the flagship at higher n, since you said to do it first

Two things, and the first one needed no compute at all.

### (i) We should have reported the equivalence test at the existing n, and now do

You wrote that "TOST" appears nowhere in the paper, and asked for equivalence intervals rather than a
point estimate against a margin. That criticism lands even with zero new runs, because **the flagship
arm is paired**: every rung of an arm runs at the same seeds, so each seed contributes one difference
with the data partition held fixed. Pairing is what makes n = 5 informative here. On the flagship
Krum / model-scaling cell:

- paired Δ = −0.026, sd 0.042, 95% CI [−0.078, +0.026], 90% CI [−0.066, +0.014]
- **TOST against the frozen ±0.15 margin: p = 0.0014, at n = 5, adding no runs**

So on the paper's central arm, equivalence is now **established** rather than "consistent with", and
that was true of the data we already had. This is in §5 of the revision.

`experiments/analyze_tost_existing.py` scores every arm the same way, and the table is in the
appendix **including the arms where it does not go our way**:

| arm | n | paired Δ | 95% CI | TOST vs ±0.15 |
|---|---|---|---|---|
| Krum / scaling (flagship) | 5 | −0.026 | [−0.078, +0.026] | established |
| Krum / scaling (score-only control) | 5 | −0.023 | [−0.108, +0.061] | established |
| Krum / scaling (EMNIST-byclass) | 3 | −0.017 | [−0.033, −0.001] | established |
| Reputation / scaling | 5 | +0.178 | [−0.112, +0.469] | not established |
| `cos_krum` / pixel | 8 | −0.425 | [−0.732, −0.118] | not established |
| `coord_median` / pixel | 5 | +0.098 | [+0.018, +0.178] | not established |

Three of six. The last row is the one we want to point at rather than bury: **the frozen point-estimate
rule passes it and TOST does not**, because +0.098 sits inside ±0.15 while its interval does not. The
frozen rule scored the point estimate; equivalence is a stronger claim and n = 5 does not support it.
That is the honest reading of the paper's own criterion, and it is the argument for the top-up in its
strongest form: not that our numbers need help, but that a point estimate inside a margin was being
allowed to read as an equivalence claim.

### (ii) The top-up, and the clause that makes it not optional stopping

`experiments/pre_registration_dose_seed_topup.md`, frozen at `684b31e` before any new run existed.
Seeds 47 to 61 on all four rungs of the flagship cell, a contiguous continuation fixed in the document.
`experiments/run_dose_seed_topup.py` imports `run_one` from the published runner unchanged and writes
to its own directory; `SEEDS5` in `run_targeted_dose.py` is not edited and
`results/targeted_dose/summary.json` is not rewritten.

Three rules in that document matter more than the result:

1. **The published n = 5 verdict stands as reported and is not revised.** The top-up buys interval
   width, which the published version never stated. It is not a second test of the same hypothesis.
2. **The reversal clause.** If the interval at the realized n is not contained in (−0.15, +0.15), or
   any rung mean reaches the 0.5 ceiling, we report the flagship equivalence claim as **refuted by our
   own top-up, in the abstract**, and the (P3) ⇏ (P4) witness is withdrawn. Pre-committing to publish
   that is the only thing that licenses adding seeds at all, and the code path that prints it exists
   whether or not it fires.
3. **Partial runs are scored at the n actually reached.** The analyzer reports the realized per-rung n,
   never waits for a threshold, and prints an explicit `PARTIAL LADDER` warning when the rungs are
   unequal, so a mid-run trend test cannot be quoted as the pre-registered retest.

The analyzer also refuses to print any pooled number until the five published seeds reproduce as
identical floats and the recomputed n = 5 table matches the pre-registration's own printed table.

**The result, at the full n = 20 (seeds 42 to 61, all four rungs):**

| rung | n | mean ASR | mean acc |
|---|---|---|---|
| kappa = 0 | 20 | 0.0439 | 0.5618 |
| kappa = 0.5 | 20 | 0.0257 | 0.6524 |
| kappa = 1 | 20 | 0.0271 | 0.6529 |
| kappa = 2 | 20 | 0.0343 | 0.6155 |

Paired Δ = **−0.010**, sd 0.047, 95% CI **[−0.032, +0.012]**, TOST p = **2.1 × 10⁻¹¹**. The reversal
clause did not fire. The interval is contained in ±0.15 with a half-width of 0.022, **14.7% of the
margin**, and every rung clears both the 0.5 ceiling and the 0.35 accuracy floor. The
Jonckheere-Terpstra retest still refutes the statistic reading (p↑ = 0.42), so the published verdict
holds in both of its parts, now with an interval attached rather than a bare point estimate. The five
published seeds reproduced as identical floats before any of this printed.

So your Priority 2 is answered twice over: the claim was already established by TOST at n = 5 because
the design is paired, and the top-up you asked for confirms it at four times the sample size rather
than rescuing it. One number in the paper moves: the kappa = 0 rung mean is 0.0439 at n = 20 against
0.0618 at n = 5, which makes the "every rung below 0.07" clause safer, not more fragile. It still
reports the n = 5 figure, because that is the sample the frozen rule scores.

---

## Priority 3: one genuinely non-rescaling upstream transformation

You asked whether the finding is an artifact of positive scalar multiplication. We agree that is the
sharpest thing that could be wrong with the paper, and coordinate masking is the arm that tests it.

**Why masking is the right choice and not a convenient one.** A mask is not of the form c·u for scalar
c > 0, so **Theorem 8 and Prop. 10 make no prediction about it**. That is the point rather than a gap:
the invariance classification is a statement about a *transformation class*, so an aggregator certified
under one class can be uncertified under the other, and the run can show it.

The transform is `doseM_m<M>`, a branch of `apply_d1_transform` parallel to the existing dose branch:

- **Adversarial clients are returned unmodified.** Not approximately: the pre-registration records the
  maximum deviation of every adversarial coefficient from 1 as **exactly `0.00e+00`**, with object
  identity `True`. Mode S pins the same channel to `3.5e-07` float32 read-back noise, so the new arm's
  pinning is **strictly stronger** than the flagship's.
- **Benign client i** gets an independent Bernoulli(1 − m) coordinate mask per parameter tensor, seeded
  by `default_rng([seed, round, benign_slot])` so it is reproducible and uncorrelated with adversary
  status, then **renormalized to the update's original L2 norm** (`4.93e-08` in float64;
  `0.00e+00` to `1.40e-04` on float32 read-back).
- Renormalization is load-bearing and the document says so **before any ASR existed**: an unnormalized
  mask shrinks benign norms and re-opens the magnitude channel that the score-only control exists to
  close. The unnormalized variant was considered and rejected for that reason, and the rejection is
  recorded rather than omitted.
- Rungs m ∈ {0.0, 0.2, 0.5, 0.8}; m = 0 returns the update list unwrapped, so the identity rung is
  bit-identical to the downstream defense standalone by construction. We check that rather than assert
  it: the identity rung's per-seed values agree to `1e-9` across all three directories that carry them.

**The channels were measured before any ASR existed**, the same discipline the EMNIST-byclass arm used.
`results/mask_admission.json`, 120 rows, both committed attacks. On the flagship cell the mask
reproduces the flagship's premise outside the rescaling family: **Krum's decision changes in 0.467 of
rounds while the adversarial mass it admits changes in 0.000 of them**, at an aggregate displacement of
1.095 (against Mode S's 0.892 at a *larger* decision change of 0.733). So the arm was established as
eligible prospectively, and the pre-registration then split on what the ASR does:

- **H-M-statistic**: ASR rises monotonically in m (Jonckheere-Terpstra, one-sided, α = 0.05).
- **H-M-admission**: ASR within ±0.15 of the m = 0 rung, every rung below 0.5, conditional on
  Δ adm. = 0 at that rung.
- **The branch that refutes us, named in advance**: heavy structural damage to benign updates may make
  Krum start selecting the adversary, in which case admission moves, the equivalence claim is void at
  that rung, and **that is a confirmation of the admission reading**, reported as such. Writing this
  down before the run is what makes the arm a test rather than a demonstration, and only the
  flat-with-admission-unchanged outcome generalizes the paper's negative.

**The result, at the pre-registered n = 10 (seeds 42 to 51):**

| m | n | mean ASR | mean acc |
|---|---|---|---|
| 0.0 | 10 | 0.0532 | 0.5663 |
| 0.2 | 10 | 0.0486 | 0.5557 |
| 0.5 | 10 | 0.0484 | 0.5545 |
| 0.8 | 10 | 0.0371 | 0.5293 |

Paired Δ (m = 0 to m = 0.8) = **−0.016**, 95% CI **[−0.051, +0.019]**, contained in the frozen ±0.15
margin at a half-width 23.1% of it, TOST p = 5.3 × 10⁻⁶, every rung clearing the ceiling and the
accuracy floor, and Jonckheere-Terpstra finding no increasing trend (p↑ = 0.895). **Neither refuting
branch fired.** The verdict the script prints is
`H-M-ADMISSION CONFIRMED ON A SECOND TRANSFORMATION CLASS`, which is the only one of the four
pre-registered outcomes that generalizes the negative.

We also ran the mask analogue of the score-only control, because without it the new arm would be less
well instrumented than the arm it generalizes: **−0.018, [−0.047, +0.010], agreeing with the full
ladder to 0.002.** Adversarial updates are unmasked in *both* conditions, so the two ladders differ
only when a benign client is selected, and their agreement localizes the absent effect to *which
client Krum picks* rather than to the content it emits.

So the answer to "maybe this is an artifact of positive scalar multiplication" is no, on a
transformation the theory makes no prediction about, with the refuting branch written down first.

---

## Priority 4: Theorem 8 as a frozen prediction matrix across two transformation classes

This is the priority we think does most for the paper, and it cost no compute beyond Priority 3.
`experiments/build_prediction_matrix.py` emits transformation class × aggregator × invariance class ×
frozen prediction × observed, and **every prediction cell is traceable to a committed
pre-registration**: rescaling to `5130cec`, masking to `684b31e`. A cell that is not traceable prints
`--` rather than being quietly omitted, and the script refuses to emit at all if any observed channel
cell disagrees with the frozen document.

| aggregator | Prop. 10 class | decision, rescaling | decision, masking | adm. (mask) |
|---|---|---|---|---|
| `cos_krum` | (a) exactly invariant | **0.000** | **0.467** | 0.333 |
| `krum` | (c) not invariant | 0.733 | 0.467 | 0.000 |
| `coord_median` | (b) conditionally inv. | 0.482 | 0.700 | 0.131 |
| `reputation` | (c) not invariant | 1.000 | 0.533 | 0.004 |

**The adjudicating cell is `cos_krum`.** It is *exactly* invariant under positive rescaling: its
selection is bit-identical at every rung, 0.000. Under masking, predicted before the run to lose that
guarantee because masking changes cosine distances, its decision changes in 0.467 of rounds and it
admits adversarial mass in 0.333. **One class certifies what the other cannot, on the same aggregator,
the same attack and the same seeds.** That is what an invariance classification being a statement about
a transformation class means, made falsifiable, and it is the closest thing the paper has to a
prediction that could have failed cheaply and did not.

Two honesty notes on this table. The ASR columns are **not symmetric**: under rescaling every
aggregator has a published ladder, under masking only Krum is being run, because the `cos_krum` mask
ladder was considered and **dropped for compute before any ASR existed**, and that drop is recorded in
the pre-registration. So the `cos_krum` masking cells are labelled CHANNEL results and never outcome
results. And there is no RFA row: RFA is not an arm of the channel measurement at all, so an RFA row
would be `--` in every observed column, and we would rather have four honest rows than five with one
padded.

---

## Priority 1: the sign reversal is now the figure

You named −0.272 → +0.098 as the best thing in the paper and said it was buried. It was: it lived in a
small inline `tabular` in §5. Figure 1 is now three stacked panels, (a) the DAG, (b) the compact Mode S
curve, and **(c) the reversal itself**: the outcome-gated ladder's −0.272 (95% CI [−0.422, −0.122])
against the Mode S intervention's +0.098 ([+0.018, +0.178]) **on the identical cell**, bars drawn on
the paired difference because both rungs run at the same seeds. The inline tabular is deleted and
replaced by a pointer, which is where the vertical space came from.

Both numbers are read from the generator's live variables rather than typed, and `draw_reversal`
carries a **refuse-to-draw guard**: if the two intervals ever stop being separated, the panel does not
render rather than rendering a claim the data no longer supports.

The abstract states the same contrast in the same words (fourth sentence, immediately after the
identification claim it is the evidence for), so the memorable thing and the thing the abstract sells
are the same thing.

---

## Priority 5, and Priority 9: two fixes that were already in the draft you read

We separate these because a reviewer who believes a fix is outstanding will not credit it, and in both
cases the page numbers say otherwise. Measured on a clean build of the current source, body is p1 to
p9:

- **The 90.5% / 88.1% / 38/42 family (Priority 5)**: p24 and p41. **Appendix only.** `16/18` likewise,
  p23, p28, p41. None of it is sold anywhere in the body, and none is in the abstract.
- **The knife-edge 2 × 2 (Priority 9)**: "emit-only" occurs on **p37 and nowhere else**. Your option B
  is already taken; the body's only trace is a "not decisive" clause inside (L1).

We are not claiming you misread the paper. We are noting that both fixes predate your report, so what
is left of Priority 9 is only your option A, more seeds, which is below.

---

## Priority 9 continued: seeds on the 2 × 2, and a departure we disclose

`experiments/pre_registration_emit_only_topup.md`, frozen at `684b31e`. Seeds 47 to 61 on the κ = 1
cells only, to n = 20.

**This departs from a previous pre-registration and we say so in the document.** The original at
`b995f1b` states: "No seed addition. n = 5, seeds 42 to 46, matching the arm it is compared against."
We are adding seeds to an arm whose own frozen document forbade it. The new document names that,
quotes it, and records that the reason is your report rather than anything in the data. The
alternative was to leave a knife-edge result at a sample size a pre-registration had frozen for a
different purpose, and we would rather break the rule visibly than keep it invisibly.

The published n = 5 quantities, all four of which the analyzer reproduces to under `1e-9` before it
prints anything: full −0.0389 (sd 0.0455), score-only −0.0300 (0.0638), emit-only +0.0456 (0.1297),
residual −0.0545 (0.1222) against the frozen 0.05 margin. Seed 42 is the sole draw above the outlier
threshold on both quantities, which is the "one seed carries the whole result" concern in the paper's
own words, and it is why the projected half-widths at n = 20 (0.061 and 0.057) still **exceed** the
margin. So the top-up is pre-committed to a verdict of "decisive in either direction, or still not
decisive", and explicitly not to "separable because we added seeds until it was".

**The result, and it goes against us.** At n = 20 the published verdict does not survive:

| quantity | mean | sd | 95% CI | n = 5 sd |
|---|---|---|---|---|
| full | −0.0168 | 0.0466 | [−0.0386, +0.0050] | 0.0455 |
| score-only | −0.0077 | 0.0587 | [−0.0351, +0.0198] | 0.0638 |
| emit-only | **+0.0098** | 0.0799 | [−0.0276, +0.0472] | 0.1297 |
| residual | **−0.0190** | 0.0892 | [−0.0607, +0.0228] | 0.1222 |

The magnitude channel is now **established inert** (frozen Rule 1 passes: the interval is contained in
±0.05). The residual is −0.019, inside the margin, so **the "channels interact" reading we published
is withdrawn.** Your instinct about the knife edge was right, and the mechanism is exactly the one you
named: **seed 42 carried it.** It is the only draw of 20 past either outlier threshold (+0.272 on
emit-only, −0.259 residual), and realized sd fell from 0.130 to 0.080 with the added seeds. It is
still excluded from nothing.

We are careful about what this buys. Separability is reported as **consistent with, not established**,
because the half-width is 0.042 against a 0.05 margin. The top-up's own pre-registration projected
half-widths of 0.061 and 0.057 *before the runs*, i.e. it recorded in advance that containment was
unreachable at this variance and n, so the qualifier is a frozen rule and not a hedge we chose after
seeing the number. The paper still claims no per-channel decomposition of ΔASR anywhere.

---

## Priority 6: certifiable versus undiscoverable

Your framing was better than ours and we adopted it. The coverage material and Prop. 7 already existed;
what was missing was the sentence saying what they are *for*. §4 now closes:

> **So the proposition is a characterization, not a scoreboard**: it partitions any menu into the
> compositions an inherited-suppression screen can *certify* and the compositions such a screen renders
> *undiscoverable*, and it puts the paper's own best composition in the second class *a priori* rather
> than by measurement.

with FG→RFA as the witness (0.093 max-committed ASR at n = 30, CI [0.062, 0.123], against FoolsGold
alone 0.732 and RFA alone 0.845), the bound binding at |E|/|S| = 1/5, and the note that the result is
specific neither to our screen nor to FL, since any screen requiring a constituent to suppress the
attack independently inherits it. The 40% recall is now a consequence of a characterization rather than
a bad grade.

---

## Priority 7: the FLTrust material, cut

The prospective paragraph was 1988 characters for a result the paper itself declines to treat as
validation. It is now **836**, carrying only the score, the one-sentence reason it is uninformative
(FLTrust suppresses both committed attacks alone, so the 13 pairs containing it are inheritances or
near-identities), the disclosure that **the scope condition was added retrospectively rather than
pre-registered**, and a pointer. `13/20` is out of the abstract.

**Relocate-never-delete applies and was checked**: the retrospective-scope-condition disclosure, the
`0.048` max-committed ASR, the `0.030` near-identity radius and the `5/7` non-FLTrust arm all survive
at named appendix sites, verified by literal conservation rather than by reading.

---

## Priority 8: the scope limits, made to work for the paper

The limits are now stated as what the design *buys* rather than as apology. Three tiers, never pooled:
Prop. 1 and Prop. 2 are about the *shape of a screen*, with no dataset or scale in them; Theorem 8 and
the invariance classes are a laboratory clean enough to settle invariance algebraically and **bound no
ASR**; the empirical arms are four aggregators of three structural kinds under two committed attacks,
replicated on a second dataset and architecture, never "generalizes". The middle tier is deliberately
the narrowest: a broken implication has to be exhibited where invariance is decidable rather than
argued.

---

## Priority 10: the one thing we did not do, and why

You asked us to push the C0 to C3, P1 to P5 and Mode S/A vocabulary to the appendix. **We declined, and
this is the only one of your ten we declined.**

The levels are the paper's spine. The two results are (P3) ⇏ (P4) and (P4) ⇏ (P5), and both are stated
*as* transitions between named levels. Removing the labels would not simplify the contribution, it
would make it unstateable: the finding is precisely that a criterion stated at one level constrains
security at another, and without the levels that sentence has no subjects. A previous round already ran
a caveat-relocation pass, so the vocabulary that remains is load-bearing rather than residual.

What we did instead, and what we think you actually wanted: the three-tier paragraph above, and
Figure 1(a), which draws the chain and annotates each broken arrow with the arm that witnesses it. A
reader now meets the hierarchy as a picture before meeting it as labels.

We would rather disagree in one place with a reason than accept ten of ten.

---

## The single biggest wording change: *identify*

You wrote that the one wording change worth more than any experiment is inserting "identify", and you
were right that the draft over-claimed. It was worse than the report saw: §5's heading read
"Statistic Preservation Is Not Attack-Suppression Preservation" and the conclusion asserted "is not,
*in general*, attack-suppression preservation", a universal claim, while §5's own body was already
correctly hedged. The heading and the conclusion contradicted the section they headed and summarized.

All three sites now make one claim:

- §5 heading: "Statistic Preservation Does Not **Identify** Attack-Suppression Preservation"
- abstract: "preserving the statistic is not sufficient to **identify** a preserved defense"
- conclusion: "not sufficient to **identify** preserved attack suppression"

and we grep every site that states the headline to confirm they agree, which is a check the previous
draft fails.

---

## The ±0.15 margin, conceded rather than derived

You asked where the margin came from. **`experiments/pre_registration_targeted_dose.md` freezes 0.15
and nowhere says why.** We checked, and the paper now says that in those words rather than supplying a
derivation after the fact. What the pre-freeze record does support, built only from quantities that
existed before the freeze: the effect the design was built against was Round 11's own +0.762 rise, five
times the margin; and **the margin never stands alone**, the frozen rule conjoining it with *every rung
below 0.5*. So the margin is deliberately generous and the conjunction is what stops a generous margin
from certifying a high-ASR arm. On the flagship cell every rung's mean sits below 0.07.

Conceding that it was frozen without a written rationale seemed worth more than a retrofitted one.

---

## A defect your report did not have to reach: the dataset name

While doing the above we found that **the paper called one dataset by two names, and the name in every
table was wrong.** `fl_core/data_loader.py` loads `datasets.EMNIST(split="byclass")` under the key
`femnist` and partitions it with the paper's standard label-Dirichlet split. So the arm is **torchvision
EMNIST-byclass**, not LEAF FEMNIST, which partitions by writer. The prose said EMNIST-byclass and the
tables said FEMNIST, and the paper never said the two named one arm.

This had already propagated into review: your own summary lists the datasets as "CIFAR-10, FEMNIST"
while our Ethics statement says EMNIST-byclass. It also overstates federated realism, which matters
precisely because empirical scope is the weakness under discussion.

Fixed by naming, not by re-running: prose and tables both read EMNIST-byclass, with one clause
recording that the frozen artifacts keep the `femnist` key and that the partition is label-Dirichlet.
**No artifact, JSON key or pre-registration filename was renamed**, so `FEMNIST` still appears once in
the paper, in the sentence that discloses this.

---

## Item 21: the code package

No code or data package accompanies the submission, and the paper says so. We are not going to claim an
artifact we are not releasing. What is in the paper instead: every number is emitted by a named
generator that is cited at its use site, every pre-registered rule is scored by a script that refuses
to print until the previously published seeds reproduce, and every runner refuses to start unless the
hash of its own pre-registration matches a commit. That is reproducibility of the *reasoning* rather
than of the runs, it is weaker than a release, and we prefer to say so.

---

## What the three runs could have done to us, and what they did

All three were pre-committed to reporting a result that would have hurt the paper, and the code paths
that print those verdicts exist whether or not they fire. Two confirmed and **one refuted us**:

1. **The flagship top-up could have refuted the equivalence claim** and sent that refutation to the
   abstract. It did not: [−0.032, +0.012], a half-width 14.7% of the margin.
2. **The mask arm could have confirmed the admission reading instead of generalizing the negative**,
   had masking made Krum start admitting the adversary. It did not: admission stayed at exactly 0.000
   at every rung while the decision moved in 0.467 of them.
3. **The 2 × 2 did refute us.** The "channels interact" verdict we published at n = 5 does not survive
   n = 20, and we withdraw it. This is the one place in this response where more data cost us a claim
   rather than confirming one, and it is reported in the section where the verdict appears, with the
   n = 5 numbers retained beside it.

What we still cannot claim, and do not: separability is consistent with but not established, because
the interval is wider than the margin and the pre-registration said in advance that it would be. One
mask is one transformation and not the class of non-rescalings, so rotations, projections and
quantization remain untested. And n = 20 is 20 data partitions of one dataset, one aggregator and one
attack, which widens no scope.
