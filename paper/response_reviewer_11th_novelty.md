# Response to the eleventh review (6/10, Weak Accept, confidence 0.82)

**Round 71.** Scorecard read: Technical 8, Novelty 6, Empirical 7, Significance 7, Clarity 6,
Reproducibility 9. This round targets Novelty, because that is the score the review itself argues from,
and it takes the review's own instruction about how: *I would not add another broad benchmark just to
increase the number of datasets/attacks. The higher-value revision now is to make the conceptual
contribution unmistakable.* No new dataset, architecture, attack, defense, aggregator or composition
suite was added.

---

## 1. The concession this review earns

The objection is the right one:

> *The main theorem is essentially a standard positivity/identification observation, and the experiments
> demonstrate it only in a relatively narrow FL setting. What is the genuinely new ML contribution?*

Our answer to it was already written, and it was rendering on **page 4**, four paragraphs after the
proposition it answers, and it **led with a prevalence statistic that this same review tells us not to
lean on**. Both of those are our fault and both are fixed. The paragraph is unchanged in its title and
in its concession, it now stands immediately after Proposition 1's scope sentence, and it renders on
**page 3** with the proposition still on page 3. Its four reasons are the science, not the literature:

- the paragraph's own concession, kept verbatim: "The violation itself is standard; four things around it are not" `:360`
- reason one, the estimand: "which \emph{level} of a preservation hierarchy an upstream transform leaves intact" `:360`
- reason two, now carrying the promoted proposition: "that failure is exhibited by counterexample in both directions" `:360`
- reason three: "the invariance conditions are derived per mechanism, not assumed" `:360`
- reason four: "choosing between the two admissible designs does not blunt the inferred effect on one cell" `:360`

The prevalence statistic is gone from that paragraph and from the contributions list. See §6 below.

---

## 2. Numbering crosswalk

This round states a proposition in the body that was previously appendix-only, so the numbering below
it shifts by one. Your review is written against the **old** numbers, so it stays checkable:

| your number | what you called it | now |
|---|---|---|
| Proposition 11 | statistic-level non-identifiability | **body Proposition 3** (p6) and **appendix Proposition 12** (p33, with the proof) |
| Theorem 8 | the bounded-reweighting theorem | **Theorem 9** (p22) |
| Proposition 7 | the coverage restatement | **Proposition 8** (p19) |
| Proposition 6 | the identification restatement | **Proposition 7** (p17) |
| Proposition 9 | rank invariance | **Proposition 10** (p25) |
| Proposition 10 | the invariance classes | **Proposition 11** (p26) |
| Proposition 12 | the rho/Delta-c relation | **Proposition 13** (p33) |
| Theorem 3 | composability | **Theorem 4** (p15) |

Propositions 1 and 2 keep their numbers; the insertion is in §5, after both. **Nothing was renumbered
by hand.** No prose in either document hardcodes a result number and no figure PDF bakes one; both were
re-checked mechanically this round (0 hits for a digit following any theorem-like word in either `.tex`
outside comments, and 0 hits across every PDF in `paper/figures/`). The shift was then verified as a
diff of `main.aux` before and after, and **no label was lost**.

---

## 3. P2 — the non-identifiability result is now the theoretical centerpiece

You wrote that it is *more memorable than Proposition 1* and asked for it *stated cleanly* followed
*immediately* by *the two counterexamples*. It is now the first numbered object of §5, on page 6:

- the statement's title, as you named the result: "Statistic-level non-identifiability" `:662`
- both directions in one sentence: "there are transforms that change the statistic and the \emph{decision} at $d_2$ arbitrarily" `:664`
- and the converse direction: "transforms that change suppression at unchanged statistic, decision and admitted mass" `:664`
- the consequence, which is the memorable half: "no predicate of the statistic identifies admitted mass and none of admitted mass identifies suppression" `:664`
- and its scope condition, inside the statement rather than in a footnote: "exhibited counterexamples, not a general law" `:664`

The two witnesses are **named inside the statement** and then run immediately below it, which is your
*then immediately show the two counterexamples*:

- the witnesses, named in the statement: "Mode~S into Krum and \texttt{cos\_krum} under the pixel backdoor, both below" `:664`
- the first, one paragraph down: "One cell is carried end to end, and it is Krum under model scaling" `:675`
- the second, four paragraphs down: "selection is bit-identical at every rung, also at $\Delta\Lambda_a{=}0.000$, yet ASR \emph{falls}" `:772`

The appendix copy became the restatement with the proof, on the convention this paper already uses for
Proposition 1:

- the restatement's title: "Proposition~\ref{prop:nonidentifiability} of the body, restated with proof" `:1699`
- and the appendix inventory now points at the restatement rather than at the body label: "statistic-level non-identifiability" `:2748`

Two scope paragraphs (*what the proposition does not say*, and the corollary explaining the section
title) stayed with the proof rather than travelling, because the body statement carries its own scope
clause, so no disclosure depends on a reader reaching page 33.

### One mechanism you should know about, because it looks like a loophole

The insertion holds four `\emph` runs and a cross-reference inside a paragraph budget that was already
at its ceiling (`bold_runs` 39/39, `emph_runs` 59/60, `max_xrefs` 5/5, `long_para` 848/850). It passed
those caps not because the statement is emphasis-free but because our own clarity instrument's
`paragraphs()` **drops any block whose first non-comment line begins with `\begin{`**, so a proposition
environment is exempt from the per-paragraph caps by construction. We are stating that rather than
letting a green gate imply the statement was measured against those caps. It was read on the rendered
page instead, at 150 dpi.

---

## 4. P4 — the threat-model boundary, and what actually changed

Declined as *new content*, done as a *promotion*. The box already denied all three things your box
asks for; the oracle clause was its **last** clause, so a reader who read one sentence of the box read
two of the three. That clause is now in the box's opening *Not established* sentence, beside the
adaptive denial, at zero line cost:

- the box's opening denial, now carrying all three: "We do \emph{not} establish that admission predicts security, we claim nothing" `:458`

It was **moved, not copied.** The clause now reads inside the box's first sentence, and the Conclusion's
own copy of it, which carries the open problem, is untouched:

- in the box: "our instruments read adversary identity" `:459`
- in the Conclusion, with the question it leaves open: "Both instruments read adversary identity, leaving the open question" `:948`

The other three homes of the oracle disclosure are also untouched:

- the protocol's own statement: "Threat model: committed, not adaptive" `:628`
- the instrument's own sentence: "so Mode~S is a measurement device and not a defense" `:685`
- and the protocol clause that forces it: "which requires knowing which clients are adversarial and is therefore a laboratory instrument" `:617`

---

## 5. P5 — margin sensitivity, named and brought forward

Declined as a *table*, done as a *label*. The body already carried the whole result mid-sentence
inside a TOST claim; what it lacked was a name and the ladder's span. Both are now there, on page 7:

- the run-in head and the finding: "Margin sensitivity: the margin is not load-bearing" `:709`
- the span, which previously lived only as table column headers: "each survives the whole $\pm0.05$--$\pm0.20$ ladder" `:709`
- the binding arm, with the definition inline: "the binding arm needing $m^\ast{=}0.034$, the smallest margin its interval fits inside" `:709`
- and the framing you praised, kept verbatim: "an evaluation-design tolerance conjoined with every rung below $0.5$" `:709`

The literal string `the one arm that needs it` is a whole-file, case-sensitive protected regex in our
negation instrument, and it survives byte-identical at `:709`. It was deliberately **not** promoted to
the front of the sentence: doing so capitalises it and drops that floor from 16/16 to 14/16, which is
the instrument catching a real thing rather than a false positive.

---

## 6. §11 — the 6/59 prevalence is no longer a pillar, and is not hidden either

You wrote that you would not lean on it. Two changes:

1. **It is gone from Contributions item (1)**, which is now titled for the result rather than the
   literature: "Three impossibility results" `:250`
2. **The supporting body paragraph that recited the four audit numbers is deleted from the body.**

On (2), we owe you the accounting, because our standing rule is that a disclosure may move but never
lose its last home. Every clause of that paragraph already had a fuller appendix home, checked by
content and not by line number:

- the primary rate: "holds for \textbf{$6$ of $59$ papers, $10.2\%$}" `:2511`
- the contrast-present rate, in the paragraph that also forbids the overclaim: "An identifying contrast is \emph{present}" `:2540`
- the same paragraph's own guard against us: "any sentence implying the contrast is rare in this literature is false against our own audit" `:2540`
- the coder caveat: "It is one coder with no inter-rater statistic" `:2554`
- and the pre-registered outcome-(ii) reading, in a stronger form than the body carried: "A low rate was pre-registered as meaning that the decline above is kept and evidenced rather than replaced" `:2511`

The audit also keeps a **body** pointer, so a main-text reader still learns it exists:

- the appendix signpost: "the 59-paper audit, each cited from the sentence that needs it" `:278`

The decline this evidences is unchanged and still pre-registered as outcome (ii): we do not claim the
gate is common. What changed is that the main text no longer argues from how often other people do it.

---

## 7. §9 — the sign reversal is restated as an estimand claim

Your wording is better than ours and we took it. Both paragraph heads changed, body and appendix:

- body: "Two admissible estimands, opposite conclusions on the same cell." `:818`
- appendix, which also now says plainly what it is: "This is the paper's central empirical result." `:1688`
- and the appendix prose was rewritten to your form: "the choice between two admissible estimands sets the inferred direction" `:1688`

This is a wording change, not a claim change. The honesty it rests on is untouched: the two designs
are not the same intervention, and no `p`-value attaches to the difference between them.

---

## 8. §16 — the contribution hierarchy, stated rather than added

Declined as a *fourth item*, done as a *label*. Your requested hierarchy is primary = evaluation
methodology, secondary = the hierarchy, empirical = the reversal, negative = no screen. Those are the
three existing items plus the preamble, which already priced the negative result and now says so:

- the preamble's negative result: "The screen that result implies is priced rather than validated, and we do not claim it certifies security." `:239`
- item (1), the methodology: "Three impossibility results" `:250`
- item (2), the empirical result: "The design it forces, and a sign reversal" `:251`
- item (3), the hierarchy: "The preservation hierarchy, and the two links that break" `:252`

No fourth item: the page budget cannot fund one and every clause already has a home.

---

## 9. P3 — declined, with the measurement and the single candidate named

You asked for less research-diary material in the main paper. We measured before deciding:

- our own instrument's count of research-history framing in the body window is **0**, and it is a
  ratcheted target, so it cannot rise without failing the gate;
- a scan of the body for provenance phrasing returns **three** hits, and two of them are
  *pre-registration* provenance, which is credibility rather than history:
  - "frozen before any score-only ASR existed" `:781`
  - "frozen before its runs existed" `:788`
- the third is the one real candidate, and it is not a diary entry: it is the paper applying its own
  impossibility result **to its own earlier work**:
  - the candidate, quoted in full: "Our earlier suites tried to validate ordering preservation by choosing a different downstream defense" `:862`

Cutting it would convert a self-criticism into a neutral observation, and the clause immediately after
it (*refutes that suite's pooled prediction*) has no antecedent without it. We kept it and declare the
decline. If you read that sentence as diary rather than as self-implication, say so and it goes.

---

## 10. §20 and §10A — no new benchmark, and the small-n cell

Taken exactly as you wrote it: no broad benchmark was added. On small n, the round prepared a
**seed top-up for the one cell you singled out** as the thing that keeps the paper *from resting
entirely on one anomalous CIFAR-10 experiment*: the CIFAR-100 replication of the reversal, currently
n=5 on both legs.

- the cell, and the number that would move: "the same cell on CIFAR-100 ($n{=}5$) reads $-0.213$ against $+0.071$" `:822`

**It has not run, and that is disclosed rather than promised.** Written this round and on disk:
`experiments/pre_registration_cell7_seed_topup.md` (seeds 47--61, fixed and contiguous, endpoint rungs
only, both legs topped up together, and a demotion clause that fires against us if either interval
contains zero or the two overlap at the reached n) and `experiments/run_cell7_seed_topup.py`. Our own
precedent refuses to start a runner until its pre-registration is git-committed and the hash is
recorded in the runner, so the run is gated on a commit that has not been made. **No number from it
appears anywhere in this version**, and the paper's n=5 verdict is unchanged.

One consequence is already in the paper, because a self-count went stale the moment the file was
written:

- the released-artifact count, now 27 and checkable against its own glob: "pre-registration documents fixing the decision rules for every pre-registered arm, seed" `:997`

---

## 11. The other declines, each with its site

| your item | disposition |
|---|---|
| §7, adaptive-attacker evidence insufficient | **Declined, and you agree**: *I don't think you need to solve adaptive robustness to make the paper publishable.* The scope is stated in the protocol, in the box and in the evidence-tier table; P4 above makes the box's version harder to miss |
| §8, Mode S is an oracle intervention | **Declined as already stated**, in three places; P4 promotes the box's copy into its opening sentence |
| §15, the criterion misses the strongest composition | **Declined as already our own result.** It is Proposition 2 with its zero-recall corollary, and the witness we name is our own |
| §5, the hierarchy may be a convenient taxonomy; elevate the DAG | **Declined as done.** The mechanistic DAG with the attenuation arc is Figure 1 panel (a), promoted to Figure 1 last round, and the vocabulary table points at it |

- the false negative named as ours, which is §15's answer: "is emergent and a false negative of our own criterion" `:422`

---

## 12. What the page budget cost, stated because it is a real change

The main text is capped at 9 rendered pages and had about **one rendered line** of slack. The
proposition of §3 above costs about eight. Nothing was funded by deleting a claim, a number, a pointer
or a disclosure, and every trim is listed here.

**Deleted as a duplicate** (§6 above accounts for all five of its clauses): the audit-prevalence
paragraph, about five rendered lines.

**Trimmed, with the surviving home named:**

- §5's opener lost two bracketed intervals that are printed verbatim on page 2 and restated in the same section; what stays is "with disjoint intervals both excluding zero" `:643`
- the page-2 follow-up sentence was tightened to "the rows hold aggregator, attack and rung grid fixed and differ in dataset at the $n$ shown" `:141`
- the Mode S paragraph lost a rider on the oracle disclosure, whose content P4 just promoted into the box's opening sentence; the disclosure itself stays at "so Mode~S is a measurement device and not a defense" `:685`
- the positive-control gloss was shortened to "which tells ``we moved something and nothing happened'' from too blunt an instrument" `:772`
- the margin sentence paid for its new run-in head by dropping two words from an appendix pointer and by inlining the $m^\ast$ definition as an apposition, which the appendix defines in full at "the smallest margin at which it does" `:2901`
- the no-trusted-coordinator paragraph lost a closing rhetorical clause; its protected disclosure is byte-identical at "We claim no transfer beyond the controlled setting we study" `:233`

Net of all of it, the body's prose grew by about 265 characters while gaining a 614-character
proposition, which is the whole of the arithmetic: the trims paid for the insertion and the deleted
duplicate paid for the round.

**One deviation from this round's own non-goals, taken deliberately and reported here rather than left
as a quiet diff.** §5's section heading wraps onto a second rendered line, which is a free line at zero
content cost, and shortening headings is how three lines were funded two rounds ago. We did **not**
take it:

- the heading, which is the paper's thesis: "Statistic Preservation Does Not Identify Attack-Suppression Preservation" `:630`

The reason is measurable rather than aesthetic: the string `identify attack-suppression preservation`
is a protected regex in our negation instrument and that heading is its **only** occurrence in either
document, so retitling it drops the floor to 15/16. We paid for the line elsewhere.

---

## 13. What this round does not move

- **Significance (7) is untouched by design.** This round is Novelty; Clarity gains only incidentally,
  from the relocation in §1 above and the promotion in §3.
- **No new empirical claim.** The seed top-up in §10 is prepared and unrun, and nothing in the paper
  depends on it.
- **The 6/59 decline still stands** as pre-registered outcome (ii). We did not replace it with a
  stronger reading now that it argues for less.
- **No frozen threshold, seed list, interval convention or verdict was revised.**
- **`main.tex` still promises no repository link**, so the release described in the reproducibility
  statement remains a description of what will be released and not a live URL.
- **One known bookkeeping defect, disclosed rather than fixed.** This round inserted provenance comment
  blocks into `main.tex`, which shifts every source line below them. `main.tex` contains 80 `:NNN`
  line cites inside its own `%` provenance comments, and those predating this round are now offset.
  This round's own comments name their targets by **content** instead, which is the precedent this
  paper already set; correcting the older ones needs a pre-round baseline the round did not freeze, so
  they are flagged here rather than silently "fixed" to plausible values. No cite in this letter is a
  comment line: the audit below rejects those by construction.

---

## 14. Verification log

- **Fixpoint build** of both documents, reached at pass 2 with both `.aux` files md5-stable. Zero
  `LaTeX Error`, zero undefined or multiply-defined reference or citation, zero `??`, zero
  `Rerun to get`, zero `Overfull` in either log. 75 pages and 11 pages.
- **Page gate** `measure_body_chars --gate` exits 0: the Conclusion heading and the Ethics heading are
  both on page 9, so the main text is 9 pages. It failed at the start of Phase E with 130 spilled
  words and was paid off in the trims of §12, not by deleting content.
- **Clarity gate** `measure_clarity_load --gate` exits 0 with **no ratchet raised**: longest body
  paragraph 848/850, bold 39/39, italic 59/60, cross-refs/paragraph 5/5, §1 causal tokens 6/6, first
  proposition on p3, comment-joined paragraphs 0, research-history framing 0, body homes 2/2, all five
  of the earlier review's questions answered by p1.
- **Negation floor** 16/16 protected disclaimers present, 15 of them inside the body window; density
  84/169 body sentences.
- **Renumbering audit** run as a diff of `main.aux` before and after, not as a prediction: exactly the
  shift tabulated in §2, with no label lost and `prop:identification` still Proposition 1 on page 3.
- **Orphan-label check** for the three labels this round created or newly cited: each has at least two
  non-comment `\ref`s. LaTeX warns about a `\ref` with no `\label` and never the reverse, so this was
  checked directly rather than read off a clean log.
- **Pixel read at 150 dpi** of pages 1, 2, 3, 4, 6, 7 and 9, which is where every change of this round
  renders. Your own test is met: the Pearl objection is answered on page 3, adjacent to Proposition 1.
- **Punctuation and anonymity**: 0 U+2014, U+2013 and U+2212 across `main.tex`, `supplementary.tex` and
  `references.bib`; no unbracketed signed cell in any tabular; no unprotected `\texttt{}` join;
  `pdfinfo` Title, Author, Subject and Keywords all empty; `\iclrfinalcopy` absent from the source; and
  zero rendered occurrences of a round number in the extracted text of either PDF.
- **This letter** was audited with `python3 paper/audit_letter_cites.py`, to FAIL 0 with an empty
  unchecked list, which is what makes every `:NNN` above resolvable rather than plausible.
