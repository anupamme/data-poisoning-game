# Response to two reviews: the 6/10 report, and the clarity review that says 9/10 needs no experiments

Thank you both. These two reviews agree on something we had stopped believing, and it changed what this
round did: the evidence is not what is holding the paper back. Review B puts it as *7/10 to 9/10 is very
achievable without new experiments* and ranks *add more experiments* as **Low** impact; Review A's own
five-item priority list is entirely about placement and wording. So this round spends **zero compute**. It
runs no new experiment, writes no new pre-registration, and touches no runner, no artifact and no
pre-registration file. What it changes is the title, the abstract, and where the reader meets each idea.

**Two facts frame everything below, and we would rather state them than let you find them.**

1. **Your builds are different documents, and both predate this one.** Review A read
   `main(20260922-035940).pdf`, **89 pages**, whose title still opened with *Beyond Statistic
   Preservation* and turned on *causal identification of mechanism preservation*. Review B read
   `main(20260926-031202).pdf`, built 03:12 on 26 September — **before** both of that day's commits. The
   document you are reading now is **78 pages** with a **9-page body**. Several items on both lists were
   already shipped in a build neither of you could have seen, and we mark those as such rather than
   claiming them as this round's work.
2. **Where you disagree, we followed Review A.** Review B's proposed abstract deletes the scope
   disclaimer; Review A explicitly asks for it. It stays, in Review A's own words, as the abstract's last
   sentence. Three of Review B's deletions were also blocked outright by our own frozen
   pre-registrations, and §4 below names which and why.

Source cites are `main.tex` **source** line numbers, or `supplementary.tex:NNN`. They are not the PDF's
margin numbers, which count rendered lines and do not coincide with these. Double quotes below are
verbatim paper text, and each sits beside the line it is quoted from; the reviews' own phrasing is in
italics.

---

## 1. Review B's twelve asks, one row each

| Ask | Status | Where |
|---|---|---|
| (1) Retitle to *Statistic Preservation Does Not Identify Attack-Suppression Preservation…* | **Done this round, your wording verbatim** | `:62`, `supplementary.tex:36` |
| (2) Rewrite the abstract: one question, one failure, one fix, three findings | **Done this round, your shape, our freeze clauses restored** | `:138` |
| (3) Causal estimand earlier, in plain English | Already shipped: the estimand is named in plain English before (P1) is | `:220`, `:660` |
| (4) A visual P1--P5 ladder with a marker where the implication breaks | **Marker done this round, as a rule inside the levels table.** The figure half is declined; §7 | `:819` |
| (5) State C0--C3 only *after* P1--P5, and say how the two lists relate | **Relationship sentence new this round**; the ordering was already right | `:772` |
| (6) Stop calling it a *screen* in the main narrative | **Done, with one deliberate exception**, §6 | `:490` |
| (7) Separate general theory / FL theory / empirical evidence | Already shipped, as a paragraph with that title | `:574` |
| (8) A boxed *admissible experiment* opening §4 | Already shipped as §4's opener plus its numbered clauses | `:878` |
| (9) Sign reversal as §5's centrepiece | Already shipped: it is §5's opening prose line | `:909` |
| (10) Demote secondary results into findings | Already shipped: §6 is run-in headed findings | `:1227`, `:1321` |
| (11) Delete 20--25% of explanatory prose | **Partly, and twice: 89 pages to 78, body 10 pages to 9.** §7 gives the measured reason the rest cannot go | `:1227`, `:1375` |
| (12) One vocabulary, via a terminology table | Table already shipped; **the title/body drift it could not fix is closed this round** | `:490`, `:1642` |

## 2. Review A's eight asks, one row each

| Ask | Status | Where |
|---|---|---|
| (1) One forceful paragraph on why this is an ML contribution, near the end of §1 | Already shipped as §1's Contributions paragraph | `:336` |
| (2) Split lab identification from a deployable server estimate | Already shipped, and it is the paper's open question | `:1375` |
| (3) Say *not sufficient to identify*, not *does not preserve* | Already shipped, with its scope clause | `:1147` |
| (4) Small *n* limits empirical generality, not the theorem | Already shipped: the scope box splits formal from empirical | `:650`, `:660` |
| (5) Dataset generality is weak | **Conceded, not fixed.** It needs compute; §7 | `:662` |
| (6) Keep the 90.5% accuracy figure out of the abstract | Already shipped: zero sites in the abstract or body | `:138` |
| (7) Say the theorem does not depend on the post-hoc experiment | Already shipped, and in the stronger form that it contains no dataset at all | `:650` |
| (8) Too long at 89 pages | **78 pages now**, body 9; the relocation inventory names what moved | `:1584` |
| (§11) The old title's central phrase is terminologically dangerous | **Fixed this round: it is gone from the title** | `:62` |
| (§20) Your suggested abstract ending | **Adopted close to verbatim** | `:138` |

---

## 3. The title, and the drift it was hiding

Both of you flagged the title independently, and you were right for a reason we had not measured. The old
title turned on *mechanism preservation*, and that phrase **did not occur once in the body**. It is the
appendix's formal-invariance term. The body's name for the same level is (P5)'s own name,
*attack-suppression preservation*, which §5's heading has carried for several rounds. So the comment in
our own source defending the old title — that it kept the paper's own term so nothing drifted between
title and body — was inverted: **the drift was what we had.**

The title is now Review B's wording, byte-identical to §5's heading, which is where the claim is shown:
`:62`, and the supplement matches at `supplementary.tex:36`.

Two consequences we fixed rather than left:

- A sentence in the appendix asserted a fact about the title, saying the corollary was why the title used
  the word *beyond* — and *beyond* had already left the title one round earlier. Nothing can catch that:
  it builds clean, and no gate reads prose about the document's own shape. It now states the corollary's
  content instead: "The corollary is why no statistic-only metric can stand in for suppression" `:2204`.
- The appendix keeps *mechanism preservation* as its invariance term, because that is what the proofs
  prove. One new sentence at its definition says so, headed "One level, two names, and which is primary." `:1642`
  That is Review B's ask 12 delivered as a sentence rather than a float, and it names the body's term as
  primary.

## 4. The abstract: your shape, and the three clauses we could not delete

Review B's draft abstract is 1218 characters and we adopted its structure — one question, one failure, one
fix, the findings — at 1585 characters, fifteen sentences, longest 168. What we spent the extra on:

**Review A's scope sentence, which Review B's draft removes.** It is now the last thing the abstract says:
"This is identification, deliberately scoped to committed attacks and controlled interventions, and not a deployable security test or a predictor of adaptive robustness." `:138`

**Three clauses Review B asks us to delete are mandated by our own frozen pre-registrations**, and we will
not amend a freeze after its numbers exist:

- `experiments/pre_registration_score_only_coordmedian.md:161`, the branch that fired, records that the
  result is not scored in our favour and is reported at the same sites as the row above it, which are the
  abstract, `tab:claims` and `tab:evidence`. So the magnitude control stays, and it stays as the sentence
  that costs us: "with the aggregate magnitude held fixed the $+0.125$ is $-0.172$" `:138`
- `experiments/pre_registration_normclip_cifar100.md:205`, premise failed and fired, requires the failed
  composition-level replication to appear in the abstract and in §5 rather than in a footnote. So:
  "a composition-level replication there was attempted and did not carry" `:138`
- `experiments/pre_registration_cifar100_composition_suite.md:204` requires the CIFAR-100 claim to be
  scoped to the ladder it rests on, so the replication clause names the cell: "The reversal replicates on the same cell on CIFAR-100" `:138`

**Four numbers Review B asks us to drop did go**, each verified to keep a home elsewhere first: the 0.173
rung difference, the 80% Krum figure, the +0.012 null, and the five-of-seven count. The one number we kept
against Review B's advice is the oracle-free refutation, because it is the clause in which the paper
concedes that both of its channel instruments need an oracle no server has.

One structural repair was needed before any of this was possible. Our negation-density instrument pins
sixteen disclaimers by exact string, and the recall-cap disclosure was the only one of the sixteen whose
**last home was the abstract itself**. Deleting it from the abstract would have deleted the disclosure. So
it was relocated into the body first, into a sentence that now ends "no emergent composition can pass the gate, so recall is capped structurally." `:632`

## 5. Where the implication breaks, and how C0--C3 relate to P1--P5

Review B asks for a marker between (P3) and (P4) saying that this is where the implication breaks. The
levels table already had the rows, so the marker is a rule inside it rather than a new figure:
"The implication breaks at this rule: no level above it implies any level below it." `:819`

Review B also asks that C0--C3 come only after P1--P5, with a sentence relating the two lists. Checking
this turned up something more useful than the reordering. Inside §3, C0--C3 already stand after (P5).
The two earlier C-mentions are in §2 and **neither can move**: one is the sentence that fixes the paper's
five design words, which our clarity instrument requires to be in §§1--2; the other carries the
false-negative disclosure the same instrument pins to the body. So the ordering was not the gap — the
**relationship** was, and that is what is new: "the two preservation conditions ask whether the transform leaves admission intact, while the two applicability conditions gate on standalone ASR, which is suppression itself" `:772`

That sentence is the whole boundary result in one line: the conditions that gate are conditions on the
outcome the criterion is trying to predict.

## 6. From *screen* to *criterion*, and the one place *screen* stays

Review B's ask 6 is done in the narrative and, as Review B allows, not in the formal statements or the
appendix. One sentence in §2 now fixes both words at once, so the reader is never asked to reconcile them:
\emph{Screen} is "the shorthand the formal statements and the appendix keep for what the narrative calls the \emph{evaluation criterion}" `:490`

We used *criterion* rather than *evaluation criterion* throughout the narrative for a measured reason: the
one-word form is $+3$ characters per site against the two-word form's $+14$, and with one rendered line of
headroom on page 9 the two-word form would have spent the page budget on a synonym. *Criterion* was also
already one of the five words the instrument requires us to fix early, so the rename strengthens that floor
instead of threatening it. Every claim-table row moved with it: `:396`, `:397`, `:402`. So did
"What this characterizes is a shape of criterion, not federated learning." `:532` and
"The criterion fails exactly where Prop." `:1321`.

## 7. What we declined, and why

**Review B ask 4's figure, and ask 12's request that Mode S not appear before §5.** Both need
`figures/modeS_causal.pdf` regenerated, and its generator also writes into `workshop_paper/`, which this
round is not permitted to touch; the mode label is baked into that figure's pixels, so no caption edit
reaches it. The ladder Review B wants is already drawn, as Figure A7. We would rather decline than
describe a figure we did not change.

**Review B ask 11's remaining 20--25%.** We cut 89 pages to 78 and the body from 10 pages to 9, which is
the ICLR limit. Review A's named remaining target was the repeated discussion of the withdrawn criterion;
we measured it, and there are **three** such sites in the body, none of them a redundant explanatory
block. The other 33 are in the appendix and supplement, where they cost no page and where each is the
last home of a withdrawal we are not willing to delete.

**Review A ask 5, dataset generality.** Conceded and not fixed. The honest version is that it needs
either wave 2's 24 held-out CIFAR-100 pairs, about 100 hours serial, or a *positive* oracle-free result,
and the oracle-free arm we did run refuted us. Both facts are in the paper. We would rather carry the
limitation than dress it.

**One defect this round's pixel read caught, which we report because it was ours.** Page 2 promised, in
one clause, both the proofs and a literature audit in this document's appendix — and the audit had been in
the supplement for two rounds. One location noun in front of a list of two, where one member moved. It
resolves to no broken reference, and no gate can see it, so it survived four greps in our own planning
notes; only reading the rendered page found it. The pointer is now split by document:
"every proof, cited where needed; the literature audit is in the supplement" `:456`

---

## 8. What was verified, and what was not

Green: both documents built to a text fixpoint, with only the two long-standing overfull boxes and the one
font-shape warning; body prose ends on page 9 with the Ethics heading first on page 10; every target in the
clarity gate met with **no ratchet raised**, two of them exactly at their ceiling; sixteen of sixteen pinned disclaimers present and all
sixteen inside the body; every body pointer into the appendix resolves; numbered-statement numbering
byte-identical to the previous build; the three pre-existing missing-qualifier sites and no fourth; the
pre-registration self-count still exactly the file count at `:1449`; no rendered round number, review
ordinal or author name in either PDF; and the three starred statements re-read line by line, since a title
change and an abstract rewrite are exactly the shape that has falsified them before.

Not verified, because it is not verifiable: whether this is worth two points. The clarity changes are
real and measured. What they cannot move is empirical generality, and that is Review A's cap rather than
a presentation problem.
