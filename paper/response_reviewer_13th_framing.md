# Response: framing, theory, and one measured gap

Thank you. This review is the most useful we have received, because its central sentence is a
diagnosis rather than a request: *you don't need the paper to become broader; you need its existing
evidence to support a sharper claim.* We agree, and we have acted on it in the one place where it
identified something genuinely missing from the paper rather than missing from the first nine pages.

We begin with what we concede, then state a fact about the reviewed file, then give the item-by-item
crosswalk.

## 1. What we concede

**(a) The positive direction of our own theorem was proved and never claimed.** This is your item 7,
and it is the concession that matters most. You ask for a complementary result saying when
preservation *is* sufficient, and observe that it *transforms the paper from a critique into a
framework*.

That theorem is already in the paper. Its statement at `:1460` is titled "Bounded reweighting preserves the downstream" discriminative mechanism, and the corollary at `:1483` puts its geometric condition in a form that can be checked before the composition is run. The body never said so. Every body use of the word *sufficient* was negative, and the one body reference to the theorem was in the denial register: `:432` said only that the invariance results "bound no ASR".

The scope box on p4 now states the other direction, in four clauses:

- `:432` "In the other direction they also say when" preservation holds;
- `:433` on that class, "the downstream discriminative mechanism survives, given a bounded" ratio;
- `:434` the two conditions, a "weight ratio for the rank aggregators and a separation margin for the geometric one";
- `:435` and the corollary "puts the latter in a form checkable before the composition is run".

It carries its own two limits in the same breath, because a framework claim that omits them would be
the overclaiming you flag elsewhere. `:436` says that what "survives is the suppression ordering rather than any security property", "and the conditions need a short" pilot, named at `:437` as "pilot (L5)." Limitation L5 itself is unchanged: `:2846` states that "Theorem~\ref{thm:bounded_reweight}'s conditions need a pilot" and that one of the two is not operationally checkable as stated.

We put this in the scope box rather than in a new section for three reasons you gave us: the box is
already the paper's boxed statement of what is and is not established, it is already on p4, and it is
where a reader who stops at p9 will meet it.

**(b) Mode S had never run off one Dirichlet concentration.** This is your item 4 and it is the only
item on your list that named a gap in the *evidence*. It is correct. The paper's heterogeneity sweep
varies Dirichlet alpha for the screening criterion, and every Mode-S row in the paper is at
alpha = 0.5. We have frozen a pre-registration for one Mode-S intervention across
alpha in {0.1, 1.0, 10.0} on the sign-reversal cell, holding the attack, the downstream defense, the
rung endpoints, N, K, f, the round count, the architecture and the seed list at their published
values, and we report its outcome whichever way it lands. **It has now run, and the rise reproduces at
all three concentrations**: Delta_S = +0.218, +0.137 and +0.165, each at n = 5 with a paired 95%
interval excluding zero, with both Mode-S premises measured at each rather than carried over from
alpha = 0.5. Section 4 states what that arm can and cannot conclude, in the words it was frozen in;
section 4a reports the result and one gate we corrected. The body carries it in one clause of §5 and
the full arm in App. D.5, and the published alpha = 0.5 reading is printed beside it as frozen, not
re-run and pooled with nothing.

**(c) One body sentence overstates a table whose own caption is scrupulous.** We found this while
measuring your item 3, you did not raise it, and we would rather report it than have it found.

The sentence at `:714` claims of the margin ladder that "each survives the whole $\pm0.05$--$\pm0.20$ ladder", and calls the ResNet18 replication "the one arm that needs it" at a minimum margin of 0.183, without saying which side of the frozen margin that number falls on. The table's caption does say it. `:3061` calls that margin "\emph{wider} than the frozen margin", and `:3062` states that "the ladder cannot be read as covering it".

The body sentence is not false. Its universal is explicitly scoped to readings *in* that table, and the
table's four verdict columns are identical row for row, so the quantifier is exactly what the table
shows. But a reader who opens the table meets an arm the sentence quotes and the table does not
contain. We flag it here rather than rewrite it, for a reason we state plainly in section 5.

## 2. The reviewed file is not the current one

We state this without complaint, because it changes which of your items are actionable. Three
receipts:

1. You quote a title the paper does not have. The current title at `:43` is "Beyond Statistic
   Preservation", which already drops the *Causal Identification for* clause your item 12 objects to.
2. You quote the ASR pair as -0.272 / +0.098. Those are the n=5 values. The abstract at `:66` and the
   p2 table at `:136` lead with the n=20 pair, and `:648` states it with "with disjoint intervals both
   excluding zero".
3. Of the eight meta-scientific phrases your item 13 quotes, one survives in the body: "Our earlier
   suites" at `:904`, which introduces a measured self-refutation rather than narration. The paper's
   own clarity instrument gates that class of phrase at zero.

## 3. Crosswalk

| # | your ask | disposition |
|---|---|---|
| 1 | recast Prop. 1 around structural positivity, not collider bias | **Already the paper's framing, in your words.** Figure 2's caption at `:325` is titled "Not ordinary collider bias but a structural \emph{positivity} violation"; `:401` concedes that "Only half of this is ordinary collider bias" and that "that half is textbook"; `:360` is a paragraph titled "Why this is more than a standard positivity violation" giving four reasons |
| 2 | define the causal estimand formally | **Already displayed in §2.** `:388` is the displayed contrast, with the control arm "\mathrm{do}(T{=}\mathrm{id})", and `:394` states the scope restriction your formalization implies: "We estimate the effect of intervening on the upstream transformation" and not the effect of preservation itself, which names no intervention |
| 3 | equivalence-margin sensitivity at 0.05/0.10/0.15/0.20 | **Already run at exactly those four values, with a per-arm minimum margin, and pointed at from the body.** See concession (c) for the one thing wrong with the body pointer |
| 4 | one Mode-S intervention under non-IID heterogeneity | **Conceded, pre-registered, run, and reported.** The rise reproduces at all three concentrations, each at n = 5 with an interval excluding zero and both premises measured there. See concession (b) and sections 4 and 4a |
| 5 | demote the screening predictor | **Already demoted as far as the structure allows.** The criterion is appendix-only and its own heading at `:1222` calls it "an evaluation prioritizer, never a security screen" |
| 6 | frame Mode S as an identification instrument, not a deployable defense, boxed and early | **Already both boxed and early.** `:690` states that "Pinning $c$ requires knowing which clients are adversarial, so Mode~S is a measurement device and not a defense"; the p4 scope box at `:466` states that the paper is a "methodology for identifying what a composition preserves, not for choosing a defense to deploy"; `:463` states that "We do \emph{not} establish that admission predicts security"; the appendix at `:1292` states that "Both instruments are oracles" as an open problem, and the ethics statement at `:1005` calls them "measurement devices, not deployable defenses" |
| 7 | the complementary *when preservation IS sufficient* theorem | **Proved already; the body now claims it.** See concession (a) |
| 8 | make the theorem ML-specific per aggregator | **Already per-aggregator.** The invariance conditions are derived per mechanism rather than assumed, which `:360` gives as its third reason, with a coordinate-median corollary at `:1558` |
| 9 | make the 2x2 sign reversal the visual centerpiece | **Already the first thing on p1 and p2.** `:66` leads the abstract with it, `:136` prints both designs as two rows with their intervals, and Figure 1 is the Mode-S causal figure |
| 10 | three research questions before the P1-P5 machinery | **Already on p1, and machine-enforced.** `:100` carries all three in one paragraph: "propose no new defense", the gated test "cannot answer the question at any sample size", and why, that it "removes the counterfactual support identification requires"; `:130` gives what replaces it, to "intervene on the upstream transform". A gate in the clarity instrument fails the build if these five beats spread beyond two adjacent paragraphs |
| 11 | a stronger *what we establish* paragraph | **Same five gated beats**, plus the scope box on p4, which now states the positive direction as well as the negative |
| 12 | retitle to lead with the finding | **Declined, with the measurement.** The title already moved and already drops the clause you object to. We are not confident that a second title change in one cycle helps a reader who has seen neither |
| 13 | reduce meta-scientific narration | **Stale by seven of eight phrases**, and gated at zero going forward |

## 4. What the heterogeneity arm can and cannot conclude

Frozen before it runs, in `experiments/pre_registration_heterogeneity_modeS.md`, committed at
`e7b7c83` as its own commit touching no other file, at a point when both of its output paths were
verified not to exist. Both scripts refuse to start unless that document is committed at exactly that
hash with a clean working tree, so the freeze is enforced rather than asserted:

- **Two verdicts, separately frozen, and neither may be reported as the other.** A sign verdict, on
  whether the within-defense rise reproduces at that alpha, and a magnitude verdict, on whether its
  interval overlaps the published alpha = 0.5 interval. A previous arm of ours cleared its
  pre-registered threshold in two regimes while reproducing the sign in only one, and our one-line
  summary of it was misleading for exactly that reason.
- **n = 5, fixed now, no optional stopping.** The published anchor's own standard deviation gives an
  n=5 half-width of 0.080 against a mean of 0.098, so this estimand is resolvable at n=5 and would not
  be at n=3. If a leg's interval contains zero we report that at n=5 with the point estimate, as a
  failure to resolve. We do not top it up.
- **The alpha = 0.1 leg is the one at risk, and its accuracy gate is live.** Measured clean accuracy at
  that concentration reaches down to 0.364 against the frozen 0.35 floor. If a rung falls below the
  floor, that alpha is **void, not negative**: a low ASR at collapsed accuracy is not suppression.
- **No robustness claim from a subset of the grid.** If the rise reproduces at 1.0 and 10.0 and not at
  0.1, that is what we report. The sentence *the instrument survives heterogeneity* may not be written.
- **The published alpha = 0.5 verdict is not amended.** It is printed as frozen beside whatever the new
  arm returns, and if the two disagree, both appear and the disagreement is the result.
- **Two premises are measured per alpha before any ASR is scored**, because Mode S's claim that the
  adversary is pinned is a property of the client partition as well as of the transform: the identity
  rung must be the exact identity, tested with equality and not a tolerance, and the adversarial
  coefficient share must not move across rungs. If it moves at some alpha, that leg is reported as
  confounded at that alpha and carries no Mode-S reading.

### 4a. The result, and a gate we corrected after seeing it fail

**The ladder is complete: 30 of 30 runs, both rungs at n = 5 at each of the three concentrations.**
Reading the two frozen verdicts separately, as the freeze requires:

| alpha | Delta_S | sd | paired 95% CI | accuracy gate | sign verdict | magnitude verdict |
|---|---|---|---|---|---|---|
| 0.1 | +0.218 | 0.124 | [+0.064, +0.372] | PASS | reproduces | consistent |
| 1.0 | +0.137 | 0.073 | [+0.046, +0.228] | PASS | reproduces | consistent |
| 10.0 | +0.165 | 0.031 | [+0.127, +0.204] | PASS | reproduces | consistent |
| 0.5 frozen, not re-run | +0.098 | 0.065 | [+0.018, +0.178] | PASS | -- | -- |

The magnitude column is an **interval-overlap statement against the frozen anchor and nothing more**:
`EQUIV_MARGIN` is deliberately not imported by this arm, so no leg is described with the
practical-equivalence phrase the supplement reserves for the stricter of our two interval conventions.
No mean is taken across concentrations and no concentration's rung is differenced against another's.
Three concentrations on one cell, one attack, one dataset and two rungs support no statement about
heterogeneity in general, and we do not make one.

Two facts we would rather state than have inferred. The alpha = 0.1 leg **cleared** the accuracy floor
the pre-registration flagged as its live risk, at mean clean accuracy 0.492 and 0.474 against the
frozen 0.35, so it is interpretable; but that is far below the 0.75-0.81 the other two span across their two rungs, so its
absolute ASR is not comparable to theirs and only its within-concentration difference is read. And the
verdict field that used to say *until the admission measurement reads constant at this alpha, this row
carries no Mode-S reading* now **reads that artifact** and records what it said, because a condition
the reader has to go and resolve by hand is exactly the kind of sentence that goes stale silently once
the sibling artifact lands.

The two premises are measured and final. At every one of
alpha in {0.1, 1.0, 10.0} the identity rung is the exact identity, with zero coefficient violations
and zero channel displacements over 18 rows tested with equality rather than a tolerance, the
adversarial coefficient is exactly 1.000000 with maximum deviation 0.0, and the adversarial
coefficient share is constant across rungs. **The Mode-S premise holds at all three concentrations.**
Baseline adversarial admission into coordinate median is 0.2615, 0.2695 and 0.2709 at
alpha = 0.1, 1.0 and 10.0, nonzero in 30 of 30 rounds at each, against the frozen alpha = 0.5 level of
0.3155.

We changed one gate after seeing it fail, and we would rather state that plainly than have it
inferred. Our first implementation compared the coefficient share within each (seed, round) cell and
held the worst spread over cells to the frozen tolerance of 1e-6. On that form all three
concentrations failed, at 2.56e-6, 3.81e-6 and 2.88e-6, and each was labelled confounded. **So does
the published alpha = 0.5 leg, at 3.97e-6, which is larger than any of the three.** A gate that the
already-published configuration fails cannot discriminate concentration, and the verdict it emitted
asserted a mechanism, that the share moves and the leg therefore measures the attenuation channel,
which the measurement does not support.

The tolerance is calibrated in its own defining comment for a different quantity: the spread of
per-rung *mean* shares, recorded there as 3.48e-7 on CIFAR-10 and 3.93e-7 on FEMNIST, and that is the
statistic both of the paper's existing users of the constant compare. The gate now tests that
statistic with the same strict inequality. **The tolerance itself is unchanged at 1e-6 and was not
widened.** On the corrected statistic the published alpha = 0.5 reads 9.22e-8 and the three
concentrations read 1.09e-7, 1.12e-7 and 6.75e-7. The per-cell figures are still computed and
reported per concentration as diagnostics, so the stricter reading is not hidden, and the gate now
measures the published configuration first and returns indeterminate rather than condemning a new
concentration with an instrument that condemns the published one.

## 5. What we did not do, and why

- **We did not add attacks, datasets, modes, causal vocabulary or abstraction**, per your explicit
  list. The new arm is one cell, one attack, one dataset, two rungs, three concentrations.
- **We did not rewrite `:714`.** Its exact phrase "the one arm that needs it" is a protected string in
  the paper's own negation-density instrument, and rewriting it drops a floor the build enforces. The
  honest move was to report the compression here and to fix the instrument's coverage rather than
  quietly reword a sentence in a way that would fail a gate we wrote to stop ourselves from softening
  concessions. We would rather this sit in the response letter than be silently smoothed.
- **We did not lengthen the main text.** It still ends on p9. The scope-box addition was absorbed
  without spending a page; the heterogeneity arm's body clause was not, so it was paid for by
  ten wording changes across four paragraphs on pp. 8-9 and by keeping the arm's numbers, both premise gates, the gate
  correction and the scope limits in App. D.5 rather than in §5. Nothing was deleted to buy the space:
  no claim, number, pointer, scope condition, limitation or disclosure left the paper, and the
  compactions are wording only, checked sentence by sentence against what they replaced.
- **We did not close your oracle concern with evidence, only with framing**, and we say so. Our own
  measurement established that coefficient-neutrality and informativeness are mutually exclusive
  across the whole positive-rescaling family, so no oracle-free member of that class can serve as a
  clean statistic probe. That is a structural result about the instrument, not a substitute for one,
  and it stays disclosed as an open problem.
- **The remaining n=3 cell has a written, unrun seed top-up.** We disclose it rather than promise it.

## 6. Counts in this letter

Twelve of your thirteen items were already implemented in the current file; one, item 4, named a real
gap in the evidence, and it is now pre-registered, run at 30 of 30 runs, and reported. Four body edits
were made: the scope box's positive direction, one stale self-count, one clause in §5 for the
heterogeneity arm, and one new appendix subsection carrying that arm in full. One gate was corrected
after we saw it fail, disclosed in section 4a with the receipt that made it necessary. No frozen
threshold, seed list, interval convention or published verdict was revised, and no scope condition,
limitation or disclosure was removed.
