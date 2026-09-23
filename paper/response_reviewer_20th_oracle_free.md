# Response to the twentieth review (5/10, borderline / weak reject, confidence 4/5)

Thank you for the sentence that reframed this revision:

> *the new oracle-free result in D.8 actually forces a substantial re-evaluation of the paper's central
> empirical narrative.*

We agree, and we did not draw that consequence ourselves. **This round adds no experiment, no arm and no
number.** Every change below is a reframing, and the funding for each is a named deletion or demotion.

Source line cites in this letter are `main.tex` **source** lines, or `supplementary.tex:NNN`. They are not
the PDF's margin numbers, which count rendered lines and do not coincide with these. The preceding round's
repackaging is described in its own letter; the claims below are this round's changes only.

---

## 0. The finding that made your §14 cheap to do: the paper already measured intervention-dependence

This is the thing we most want you to see, because it is arithmetic and it is ours.

On **one** cell, against **one** baseline, this paper reports two `score_only` interventions with near
mirror-image effects. The paragraph that states it reads: "Two score-only interventions on one cell against one baseline therefore return $+0.2969$ and $-0.2974$, near mirror images." `:2331`

The two are comparable in an unusually strong sense: "it is also a \texttt{score\_only} arm, at the same $20$ seeds, against the \emph{bit-identical} identity rows this arm imports" `:2331`. And the paragraph already
refuses to attribute the sign difference, because the two arms differ in two respects at once: "Oracle access and dose magnitude are not separated by anything here, and the sign difference is not attributed to either." `:2331`

So your replacement claim — that the relation between decision change and suppression is
intervention-dependent — is not a concession we are making under pressure. It is a fact the paper measured
and then buried on page 55 while the front matter led with one leg of it. That is our failure, and §§1–2
below are the repair.

---

## 1. §14.1 — the abstract now leads with your claim

Three sentences left the abstract (the two-dissociations sentence, the Krum-flip sentence, and the
replication-and-three-attempts sentence, 429 characters) and the claim replaced them in two, because a
gated cap forbids an abstract sentence over 200 characters. It now reads: "Decision change is neither necessary nor sufficient for suppression to move, and the relation depends on the intervention." `:88`

The three witnesses follow it, in the order that makes the claim checkable rather than asserted: Krum's
decision flips while admitting no adversary and ASR does not move past the margin; "\texttt{cos\_krum}'s selection is instead bit-identical while ASR falls $0.173$" `:88`; and "Oracle-free, a Krum decision flip raises ASR $+0.322$; two further attempts did not carry." `:88`

The scope stays, and it stays *as* scope: "The first two need an oracle no server has: identification, not a deployable test." `:88` The abstract is a substitution, within its frozen character budget, not an
addition.

## 2. §14.2 — D.8 is now in the main-text Conclusion, which was a real gap and not a prominence complaint

You were right that this was missing entirely. The Conclusion mentioned neither D.7 nor D.8. It now carries
a paragraph whose first clause is your claim and whose body is the contradiction: "\emph{Neither a preserved decision nor a disturbed one determines suppression.}" `:1138`

and then, in the same paragraph, "Whether a decision change costs suppression is a property of the intervention" `:1138`, with the arm that shows it: an intervention reading no adversary identity, at
coefficient share $10^{-7}$, changes the decision and raises ASR $+0.322$, against the masked reading of the
same channel, which changes it and stays inside the pre-registered margin `:1138`.

Two constraints shaped the wording. It is stated as a finding with a pointer, not as research history,
because a gate holds research-history framing in the body at zero. And the Conclusion's first paragraph
gave up its closing sentence — *§5's dissociations witness both directions* — to pay for it. That sentence
asserted without magnitudes what §1 already states with them: "the link an evaluator leans on breaks in both directions" `:176`, at $0.733$ of rounds, $|\Delta|{=}0.026$ and a fall of $0.173$. So the claim did
not lose its last home; it lost a restatement.

## 3. §14.3 — the Claim | Level | Status table, in the body, on page 2

You asked for the three levels of contribution to be separated in a table. We had built that table for an
earlier review and put it in **appendix §I**, where it could not do the job — and it had **no oracle-free
row at all**. Both halves are fixed.

**In the body**, §1's three-item contributions list is deleted and `tab:claims` stands in its place, six
rows, one rendered line each, on page 2. Its two load-bearing rows are: "Neither decision state determines suppression & Measured & exact invariances (Prop.~\ref{prop:nonidentifiability})" `:318`
and "The same, without adversary identity & Measured, $n{=}20$ & refuted, by us (App.~\ref{app:channel_separation})" `:319`

The last row is the one we would least like a reader to skip: "Our screen predicts a menu's best compositions & Retrospective & not established" `:320`. The caption carries the strength
claim that the deleted list had carried, so it was not lost: "two counterexamples are exact invariances, not small mean differences" `:309`

**In the appendix ledger**, two rows were added rather than composed fresh, so the two tables cannot
disagree. The oracle-free row: "\emph{without} adversary identity (App.~\ref{app:channel_separation})" `:3184` with status *refuted, by us*, at $+0.3218$, $95\%$ CI $[+0.1680,+0.4757]$, $n{=}20$. And what the two
oracle-free arms *jointly* establish, which is a limitation and not a result: "The oracle is load-bearing for that negative rather than a laboratory convenience" `:3188`

Both rows go in the third block deliberately. The second block's last four rows open with *The same,* and
inherit an antecedent, and its caption counts their seed counts; inserting anywhere in that run would have
broken both.

## 4. §14.4 — Proposition 3 carries your title, and the word *impossibility* is gone from the body

Renamed in your own words at both of its proposition environments — "\begin{proposition}[Empirical non-identifiability of preservation levels]" `:779` and the appendix restatement `:1920` — and at its
appendix-ledger row `:3171`, so the three cannot drift apart. The label is
not renamed, so no cross-reference moves and no statement renumbers — we added and removed no numbered
statement this round, deliberately.

The phrase *three impossibility results* occurred exactly once in the whole body, inside the contributions
list §14.3 deleted, which is also the only body site that typed this proposition as an impossibility. So
§14.4's body half came free with §14.3, and the word *impossibility* now occurs **zero** times in the main
text. Where it survives is appendix prose about Proposition 1's structural result, which is the reading you
say is earned — for instance D.8's own construction paragraph, which explains "why it does not contradict the impossibility" `:2286`. Prop. 1 and Prop. 2 are *Formal, general* in the new table's Level column
and Prop. 3 is *Measured*, so the distinction you asked for is now made by a column rather than by a word.

## 5. Concern #4 — the estimand. This was already right, and needed to be visible

Your objection is that *P does not identify Y* is not the same claim as *the causal effect of P on Y*. The
paper agrees, in a paragraph that predates your review: "We do not estimate the causal effect of $P$ itself." `:500`, because "$P$ is not manipulable apart from $T$, being a property of the pair" `:500`,
so *the effect of preservation* names no intervention. A whole-document grep finds no competing phrasing
anywhere in either document; that denial is the only site.

What was wrong is that a reader met the estimand long before that paragraph. So the sentence that first
states the estimand now states it as one: "What is general is the estimand, an intervention on the upstream transform" `:466`. Thirteen characters, no new cross-reference. And §5's heading is already
identification language and does not change: it is the anchor for the claim that the terminology was never
the mathematics.

## 6. §14.5 — the archaeology is in the supplement, and we report what that bought

The two paragraphs you named are moved, not compressed away: the first freeze's failed share gate with its
float64/float32 bisection predicate and the $5.109{\times}10^{-7}$ correction, and the machinery half of the
paragraph after it. They are now `supplementary.tex:591`, a new section titled "the freeze we superseded, and the corrected instrument's machinery" `supplementary.tex:591`.

What stays in D.8 is what you said to keep. That the freeze failed its own gate, stated against ourselves:
"Our first freeze of that instrument failed its own share gate, and we report the failure against ourselves." `:2293` — the general lesson, which is the reason the failure is worth a reader's time:
"bisect the predicate you intend to satisfy, and stop on the gated quantity rather than on a proxy for it" `:2293` — and the corrected instrument's construction, stopping rule and result: "$13$ of $13$ shipped selection differences at a supremum of at most $9.60{\times}10^{-8}$" `:2301`

**The honest accounting**: D.8's own text goes from $16{,}867$ non-comment characters to $15{,}123$, a fall
of $1{,}744$ or $10.3\%$, and **main.pdf is still 89 pages**. The supplement went from 11 pages to 12. This is a relocation that makes D.8 readable as
one argument, not a page reduction, and we are not going to describe it as one. The one pre-registration
hash a reader needs to check the arm is kept in the body, because the Reproducibility statement promises a
hash per arm and a promise that resolves only in the other document is a promise we would have quietly
broken.

## 7. §9 — the 59-paper audit is less prominent, and its frame is stated before its first number

§1 no longer advertises the count as a finding: "every proof, and a literature audit, each cited from the sentence that needs it" `:356`.

And the audit's own section now answers your objection **above** its first count instead of only after it:
a new lead paragraph, "What the audit is, before any count of it." `:2905`, opens with "One coder, no inter-rater statistic, an arXiv-reachable frame that is preprint-heavy and 2026-heavy" `:2905` and states
the consequence in your own terms, that "the rate below is illustrative of that frame and is not an estimate of the FL literature" `:2905`. The fuller pre-registered disclosure still follows the count at
`:2950`, so nothing moved out of its place; what changed is that the reader meets the frame first.

One further wording change makes the gap a property of the coding scheme rather than of the field. Where
that paragraph said no *included row* varies an upstream stage, it now says "no route the audit codes varies an upstream stage" `:2911`, and it gives the frame's whole route distribution beside it, "$13$ absent, $4$ analytic, $21$ component contrast and $21$ hyperparameter sweep" `:2911`, which sums to the $59$. So the
claim is now about what those four route kinds do, and it says why we will not add a fifth: "re-coding for one now would amend a freeze after its counts exist" `:2911`.

**No row was re-coded and no field was added**, which is a line we hold even where it costs us the cleaner
answer to your objection.

## 8. §10 — criterion-as-defense-selection

This was largely already satisfied, and we checked rather than assumed it. The scope box calls the work a
"methodology for identifying what a composition preserves" `:571` and not one for choosing a defense to
deploy; §6.3 goes further and tells an evaluator not to use the screen for selection at all, "do not screen on inherited suppression at all" `:1105`, adding that "Its own label does not survive more seeds" `:1105`;
and the $90\%$ never stands without its trivial baseline beside it, "Constant-HIGH baseline on the same 42 pairs: 37/42 = 88.1\%" `:2413`. The audit found one real gap, in the supplement's 3-way section, whose title
reads as a capability claim and whose $9/10$ stood with no base rate. A paragraph now supplies the base
rate — "always-LOW already scores $6/10$ and always-HIGH $4/10$" `supplementary.tex:226` — and closes with the consequence: "still ordered rather than validated, and it certifies nothing about security" `supplementary.tex:228`.

---

## 9. What we decline, and why

**Your suggested sentence 2 is stronger than the arm licenses, and we decline it.** You propose *Our
original oracle-based dissociation does not survive removal of adversary identity.* The arm is one
aggregator, one attack, one dataset and one upstream transform, and the paper refuses any statement of that
general shape in both directions: "license no statement of the form \emph{the negative fails without an oracle} in general, any more than they would have licensed its converse" `:2277`.

What we assert instead is the arm's own scope — in this cell, without adversary identity, with the
coefficient share held to $10^{-7}$ — and the standing summary that follows from it, which is a concession
and not a hedge: the oracle is load-bearing for our negative rather than a laboratory convenience `:2311`.
We think you lose nothing by that wording and we would rather under-claim against ourselves than over-claim
against ourselves.

**Concern #2 is conceded and not closed.** The oracle is no longer a measurement convenience; it is
load-bearing, it is in the abstract, in the Conclusion, in the body table and in the ledger with status
*supported, and a limitation rather than a result*. What does not exist, on any cell, is an oracle-free
measurement of the **statistic** channel. Three attempts; the second and third went against us. Closing
that needs compute, and this round had none.

**Concern #5, external validity, is answered by visibility and not by evidence.** No new dataset, scale or
architecture was run. (L1)'s architecture-conditional swing and the general-versus-setting-specific split
already existed; what changed is that the Level column now puts *Formal, general* against *Measured,
$n{=}20$* on page 2, where a reader decides how much to believe.

**Concern #6, too meta-experimental.** Partly acted on, partly declined. Acted on: the archaeology left the
main paper, and D.8 now reads as one argument. Declined: we did not delete a single disclosure to buy
compactness. Each moved fact has a pointer from its old home to its new one, and none lost its last home.

---

## 10. On the score

Your §14 was, in your ordering, two narrative failures (14.1, 14.2), one packaging failure (14.3), one
over-claim in a title (14.4) and one compression request (14.5). We believe 14.1–14.4 are done, and done
the only way that is honest: by promoting a measurement we already had and by giving a proposition the name
its content deserves. 14.5 is done as a relocation, and we say plainly that it bought no pages.

If the paper now reads as *decision change and suppression are related in a way that depends on the
intervention, and here is the boundary of what evaluation can identify about it*, that is the change you
asked for. If it still reads as *one Krum dissociation, generalized*, we would want to know which page
carries that impression, because we can no longer find it ourselves.
