# Response to the latest review (6/10, weak accept / borderline, confidence 4/5)

Thank you for a review that is unusually easy to act on: every item in your §16 is a *placement*
problem rather than a missing experiment, and you were right that they are worth fixing.

**One fact frames everything below, and we would rather state it up front than let you discover it.**
Your build is `main(20260921-185012).pdf`, dated 21 September. Two of your five priority items had
already shipped after that date and you could not have seen them, and two more were answered in the
build you read but buried on page 28 and page 50 of an 83-page document. That is our failure, not
yours: an answer a confidence-4 reader cannot find is not an answer. This round therefore surfaces,
relocates and compresses. It runs no new experiment, and §8 says why we chose not to.

Source line cites are `main.tex` **source** lines, or `supplementary.tex:NNN`. They are not the PDF's
margin numbers, which count rendered lines and do not coincide with these. Everything below is this
round's diff unless we say a fact predates your review.

---

## 1. Your five priority items, one row each

| Your §16 ask | Status | Where |
|---|---|---|
| (1) Make the narrow claim brutally explicit near the end of §1 | **Done this round.** It genuinely was not there. | `:286` |
| (2) Make the oracle limitation impossible to miss | **Done this round in §1**; the abstract half shipped 23 Sep, after your build | `:181`, `:100` |
| (3) Put the reversal at the centre | Already there in the build you read: abstract, a two-row table on p. 1, and Fig. 1 | `:100`, `:170`--`:181` |
| (4) Demote the screening criterion to a consequence | Already there, and arguably further than you asked | `:286`, `:1187`, `:2852` |
| (5) Reduce the main paper by 15--20% | **Partly done: 82 pages to 78.** §5 gives the measured reason the rest cannot go without deleting claims. | `:1431` |

---

## 2. The narrow claim, now the last thing §1 says about what we claim

This was the one item on your list that was simply absent, and your own formulation was better than
anything in the paper. §1's Contributions paragraph now opens with it: "Our claim is not that preservation is uninformative, nor that admission predicts attack success" `:286`, followed by the narrower claim it replaces — that no outcome-gated cross-defense comparison "can attribute a composition's suppression to the downstream mechanism, because the gate removes the contrast such an attribution needs" `:286`.

The sentence it displaced was a longer scientific-versus-operational framing, so this is a
substitution and not an addition: the body is still nine pages.

---

## 3. The oracle limitation, in §1 and in the abstract, and it went against us twice

You call this *the oracle problem* and ask that it be impossible to miss. It is now in three places a
reader meets before any appendix.

**§1, this round.** The paragraph that carries the reversal's two rows now ends: our two channel
instruments "read adversary identity, which no server has; pre-registered oracle-free versions of both refuted us at $n{=}20$" `:181`.

**The abstract, shipped 23 September and after your build.** "The first two need an oracle no server has: identification, not a deployable test." `:100`

**The evidence, which predates your review and is a concession rather than a defence.** We
pre-registered oracle-free versions of both channels. Both refuted us.

- The statistic channel, oracle-free at $n{=}20$: "it reads $\mathbf{+0.2969}$" `:2373`, outside the frozen $0.15$ margin and excluding zero, which the pre-registration had named in advance as contradicting our headline reading.
- The decision channel, oracle-free at $n{=}20$: "It reads $\mathbf{+0.3218}$" `:2413`.
- The standing summary the paper draws from them is the one that costs us most: "the oracle is load-bearing for our negative rather than a laboratory convenience" `:2421`.

Two things we will not write, in either direction. We do not claim the negative reproduces without an
oracle, and we do not claim the converse either: one aggregator, one attack, one dataset and one
upstream defense "license no statement of the form \emph{the negative fails without an oracle} in general" `:2387`.

Your phrase for what the paper is — *an experimental instrument, not a deployable FL evaluation
procedure* — is the paper's own position, and it is now stated where you asked for it rather than only
where we had it.

---

## 4. Krum's zero-admission floor: the composite ladder, and the floor is the attack's

This is the item where we most regret the burial, because the answer was in the build you read and it
is stronger than *the floor might not matter*. Hold Krum fixed and vary the attack:

> "that floor is the attack's, not Krum's, since under the pixel backdoor the same Krum admits baseline mass $0.333$ in $4$ of $12$ adversary rounds" `:884`

against $0.000$ in $0$ of $12$ under committed scaling. A floor that moves when the attack changes and
the aggregator does not is a property of the attack.

The rest of the ladder you ask for is also already in the body rather than inferable from it: Krum's
*decision* channel has room, "changing on $0.733$ of rounds" `:884`; the dissociation is read across
four aggregators of three structural kinds, not one; and the table carries its own positive control,
since "Neither reading rests on Krum's admission floor" `:963`.

We have added no new Krum evidence this round. We have stopped presenting the answer as an appendix
footnote.

---

## 5. Length: 82 pages to 78, and the measured reason the rest is not fat

You are right that the document *reads partly like a research audit*, and the three subsections that
read most that way are gone from it. Relocated to the supplement, verbatim, every label travelling so
every pointer still resolves:

- the screen as evaluation prioritization, with the two designs an evaluator after discovery should run instead;
- the screening frontier, precision against recall against evaluation cost over the nine variants;
- the closed census of the region the invariance proposition settles algebraically.

Nothing was deleted: "Eight blocks are in the supplement rather than here, and this is where a reader looking for them finds them." `:1431`, and the supplement's own counterpart now says "eight self-contained blocks relocated" `supplementary.tex:56`. Three of the eight are marked
*not* skippable, these two among them, because a claim in the body points at each.

**Why we stopped at 78 rather than 66.** The two blocks that would get you the rest are App. C (the
proofs, 13 pages) and App. D (the intervention evidence, 30 pages), and neither can move:

- Relocating App. C would renumber every proposition, theorem, lemma and corollary in the paper. We hold statement numbering fixed across rounds so that a citation of *Prop. 2* means the same thing in your review and in ours.
- App. D is where the reversal, Mode S, the seven cells and both oracle-free refutations are measured, disclosed and given their seed counts. Exiling the evidence for the abstract's headline would make the paper shorter and worse.

And the length is not duplication, which is a measurement rather than an opinion.
`experiments/measure_appendix_redundancy.py` reports **0** paragraph pairs and **6** sentence pairs
with equal numeric sets, over 247 paragraphs and 1289 sentences; every one of the 6 is one disclosure
template instantiated on a *different* arm, so deleting either member removes that arm's disclosure.
The pages are claims and their qualifications. We can move them and we have; we cannot compress them
away.

---

## 6. Novelty, which you call *the biggest vulnerability*

We agree with your framing and we concede the standard half in your terms. This concession was in the
build you read, and it is ours rather than extracted: "We claim no credit for it: that half is textbook" `:550`.

What the paper claims is the three layers around it, and they are now met before the claim rather than
after it: "The violation is standard; the three layers around it are not" `:506`. Concretely:

1. **The FL instantiation.** Which *level* of a preservation hierarchy an upstream transform preserves, for named classes of FL aggregators, and that no outcome-gated test identifies it.
2. **Decidability before any run.** The invariance classes are algebra on the two defenses' definitions, so the conditions are settled before a single training run, and one corollary puts the geometric case in a form checkable in advance.
3. **A measured sign reversal, not attenuated precision.** This is where we would ask you to price the contribution against the textbook account. Collider bias is usually taught as a bias in *magnitude*. On one `coord_median` cell the two admissible designs give $-0.273$ and $+0.125$, disjoint intervals, opposite signs, at $n{=}20$, replicating on CIFAR-100. The practical consequence is not *your confidence interval is too narrow*; it is *your conclusion has the wrong sign*.

We are not claiming this makes the causal machinery novel. We are claiming the instantiation is not a
restatement, and that the sign result is the part a causal-inference reader does not already know.

---

## 7. The screen as a predictive contribution: we agree, and the position is worse for us than you knew

You write that the screen is unconvincing as a predictive contribution. The paper's own position is
the same and stronger: "It has no established predictive validity" `:2852`, and the body tells an
evaluator "do not screen on inherited suppression at all" `:1187` if the goal is to find the strongest
composition.

The sharpest number is in the CIFAR-100 arm, and it is a null by construction rather than a weak
positive: the screen "agrees with the observed class on $13$ of the $15$ scored pairs, $86.7\%$, against a constant-HIGH predictor's $13$ of $15$ on the same pairs, $86.7\%$, which is also the base rate" `:2634`. Equal, exactly, and that equality is the finding.

---

## 8. Generality, and the experiment we deliberately did not run

The honest boundary is that wave 2 of the CIFAR-100 menu — 24 held-out pairs — is unrun and we are
not promising it here. At the measured cost per run, a working day of compute buys roughly five of
those 24 pairs, taking the menu's coverage from 18 complete pairs to about 23 of 42. That would not
change what the arm shows, because the arm's result is an *equality* with a base rate, and five more
pairs cannot turn a null into generality. We would rather name the gap at its true size than spend the
round narrowing it by a fifth.

The same applies to the oracle: what would raise our own confidence is a *positive* oracle-free
result, and the two we pre-registered both went the other way. We are not going to keep drawing until
one comes out convenient, and §3 above reports the two we have as refutations.

---

## 9. Your §12: one simple example before the formal vocabulary

We took this seriously and concluded the paper already does it, one page earlier than the formal
vocabulary starts. Page 1 carries a single composition, one downstream defense, one attack, and two
numbers side by side — the outcome-gated $-0.273$ against the within-defense $+0.125$ on the same
cell — before any symbol, level or condition name appears. Adding a second worked example would cost
a body page we do not have, and the body page limit is the one constraint in this paper we cannot
substitute our way around.

If what you wanted is a *narrative* example rather than a numeric one, say so and it is a
straightforward swap next round, not an addition.

---

## 10. What this round did not do, stated so you can weigh it

- **No new runs.** Every number in the paper predates this round.
- **No change to any frozen verdict, pre-registration, runner or published result directory.**
- **No statement renumbered, added or removed.**
- **The body is still nine pages,** and every §1 addition above was paid for by a substitution in the same paragraph or the same section.

Our own estimate is that this round moves clarity, significance and the novelty framing, and leaves
your two real caps — how much of the causal machinery is new, and how far the empirical claims travel
— where you found them. The second of those is a compute question with a price we have quoted. The
first is an argument, and §6 is our best version of it.
