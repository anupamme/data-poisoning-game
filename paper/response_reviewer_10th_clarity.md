# Response to the tenth review — Clarity 6 → the first screen rebuilt

**Review summary.** Overall 6/10 (Weak Accept, confidence 4/5); Technical correctness 8, Novelty 6,
Significance 7, Empirical validation 7, **Clarity 6**, Reproducibility 8. The review's diagnosis is one
claim rather than a defect list: the draft *still makes the reviewer work too hard to discover the one
central idea*, and *the bottleneck is information architecture*. Its acceptance test is that a naive
reader of the first four pages can state the contribution.

We accept the diagnosis and, on measuring it, found something more useful than a list of missing
sections: **the structural asks were already implemented, and our own instruments were the reason we
could not see that they had failed.** This letter reports that first, because it changes what we fixed.

---

## 1. What we concede: the paper passed its own architecture tests and still hid the idea

`experiments/measure_clarity_load.py` gates the exact structure this review asks for, and every one of
those gates was already green before this round:

| the review asks | what was measured, before this round |
|---|---|
| (2) collapse P1–P5 in the main narrative | **7** `(P#)` tokens in the entire body, **five of them the vocabulary table's own level column** (`:503`–`:508`) and the other two a single `(P4)` in prose (`:532`, `:848`) |
| (4) move C0–C3 out of the center | **3** C-tokens in the body (C0:0, C1:2, C2:1, C3:0) |
| (7) one compact Scope box | `box:scope` **is** that box, 1822 chars, `:384` |
| (8) state the general-vs-FL split once | already once, after the proposition: "Nothing in this statement is about federated learning" `:329` |
| (9) remove development/meta-narrative | body `provenance framing` = **0** |
| (1)/(5) the four-step story early | `spine span` = **2 adjacent source paragraphs**; all five of the review's questions answered by **p1** against the instrument's own p3 threshold |

So the failure was not architecture. A 150-dpi read of the rendered first two pages showed what it was,
and none of the numbers above can see any of it:

1. **The instrument credited the abstract for the reader's first screen.** Q4 and Q5 matched inside the
   single-paragraph abstract, so `q5_pages` reported p1 while §1's own statement of the reversal
   rendered on **page 2**.
2. **Figure 1 sat at the top of page 2, physically between the two spine paragraphs.** `spine_span`
   counts *source* paragraphs and reported 2; the reader saw them a full-width figure and a ten-line
   caption apart.
3. **The sign reversal was a mid-paragraph clause** — the last ~96 chars of a 494-char paragraph.
   Nothing the eye lands on carried it, and the CIFAR-100 replication existed only as 8-pt text inside
   a figure panel.
4. **Figure 1 was overloaded to the point that its caption disclaimed its own annotations** (three
   panels, three different scopes, 1153 chars of caption).

The honest summary is that we had been measuring the source and shipping the pixels.

---

## 2. The seven changes

**(1) Page 1 now ends on the repair, and Figure 1 no longer splits the spine.** The float was declared
ahead of §1, which is why `[t]` placed it at the top of p2 between the two spine paragraphs. It is now
declared after the measured-facts paragraph, and page 1 ends on a complete sentence: "On one cell that
change reverses the" `:130` — italic *sign* — and the reversal's replication. All four beats (question,
FL grounding, the problem, the repair) are on page 1 in the rendered PDF, verified by pixel read rather
than by the paragraph counter.

Two placements were tried and rejected, both recorded in a source comment: `[t]` at the new anchor left
page 1 ending on a dangling colon, and `[b]` silently deferred the float to **page 72** — at which point
the page gate *passed*, because Figure 1 had left the body window. That false pass is the reason
placement here is settled by pixels, never by reasoning about LaTeX's algorithm.

**(2) The two designs, the two datasets and the four numbers are now a display the eye lands on.** A
three-column unnumbered table (outcome-gated against within-defense; CIFAR-10 at *n*=20, CIFAR-100 at
*n*=5) carries −0.273/+0.125 and −0.213/+0.071 with all four 95% intervals, followed by
"Both paired $95\%$ intervals exclude zero on each row" `:141` and the statement of what is held fixed.

The display's *form* was also decided by pixels, and we record why in the source. Two centred prose
lines — the cheaper option in rendered lines — failed twice: each row was wider than the column at
`\small`, so it wrapped and left *within-defense* on one line with its number centred on the next; and
the two rows fell on opposite sides of the page break, putting CIFAR-10 at the foot of p1 and CIFAR-100
on p2 below the whole figure. A `tabular` is a single unbreakable box, so it can neither wrap nor split.
The lead-in was changed from a colon to a full stop for the same reason: a break before an unbreakable
box must leave page 1 ending on a sentence.

**(3) §1's paragraph order is now question → problem → repair+reversal → why.** The five-number
measured-facts paragraph, previously *above* the problem, now follows the repair: "Two measured facts put
the question there in the first place" `:154`. It moved whole and unedited — every number, the
"committed (non-adaptive)" caveat `:154` and both cross-references travelled with it.

An earlier reviewer explicitly demanded the order this move reverses. We did not delete that record: the
source comment now names both reviews and why the tenth supersedes the earlier order, and the gated
boolean that protects the earlier request (no §1 causal token before the FL grounding) still holds at
6 tokens, first one after the grounding.

**(4) The two meta-narrative clauses are cut, and the contributions are headlined.** The Contributions
preamble no longer frames the paper's own contribution structure; it ends
"we do not claim it certifies security" `:239` after keeping the scientific-vs-operational distinction
and the "Preservation is a hierarchy, and statistic-level invariance is not a causal surrogate" `:239`
sentence. The self-counting inventory clause ("two propositions, one definition, one numbered table and
two figures") is deleted — it had no emitter, so no gate could ever see it going stale — and replaced by
routing only: "\textbf{In the appendix}" `:278`. The list's first two items now state the two clauses of
the review's target answer: "Two impossibility results" `:250` and
"The design it forces, and a sign reversal" `:251`.

**(5) *Influence* is a row of the vocabulary table, not prose below it.** This is the review's items (3)
and (10) — the reconcile-admission-vs-influence work. One row was added between admission (P4) and
suppression (P5): "influence & --- &" `:507`, with `---` in the level column because influence is not a
P-level, which is also why `p_tokens` did not move. The lead-in and the sentence below the table were
both updated in step: "\textbf{Six words, fixed here and used in one sense throughout}" `:490`. We did
**not** retitle P4 to *adversarial influence preservation* (asked twice): P1–P5 are pre-registered names.

**(6) The canonical terminology diagram is named rather than newly drawn.** Figure 1 panel (a) already
*is* the level chain with the ⇏ marks, so the vocabulary paragraph now points at it:
"draws that chain, and the arc beneath it is the one channel" `:519`. No third body float was added.

**(7) Figure 1 is two panels, and the panel it lost became an appendix float.** Panel (a) the level
chain, panel (b) the seven-cell reversal forest. No evidence was lost: `targeted_modeS.pdf` was
**already emitted and referenced by nothing**, and it is now Figure A5, "the $\Delta_c$-pinned dose" `:1534`,
placed directly above Mode A where the Mode-S dose evidence is already discussed. Both sentences that
would have gone false were repointed: Figure 1's caption ends
"The Mode~S dose evidence is Fig.~\ref{fig:modeS_dose}" `:229`, and Mode A's caption now reads
"Mode~S itself is Figure~\ref{fig:modeS_dose}" `:1543`. The caption fell 1153 → **979** chars, and the
instrument's ratchet was re-baselined **downward** to 979 — the only direction it permits.

Every scope clause the caption carried is still in it, including "three scopes, not one experiment" `:229` and
the disclosure that the middle annotation pools four CIFAR-10 arms rather than Krum alone.

**Carried micro-fix.** `p\pageref{box:scope}`, which a prior reviewer read as a broken cross-reference,
is now "with page~\pageref{box:scope}'s attacks and seed counts" `:734`. Zero `p\pageref` remain.

---

## 3. What we declined, each with the site and the measurement

| # | the ask | disposition |
|---|---|---|
| 2 | collapse P1–P5 to three levels in the narrative | **Declined.** 7 body `(P#)` tokens, five of them the table's own level column (`:503`–`:508`). There is no P-ladder in the narrative to collapse |
| 4 | move C0–C3 out of the center; present the screen as a failed approach | **Declined as already done.** 3 C-tokens in the body, and the Contributions preamble already prices the screen as a consequence: "we do not claim it certifies security" `:239` |
| 7 | one compact Scope box replacing scattered disclaimers | **Declined.** `box:scope` at `:384` already is that box (1822 chars). Six of the local anchors the ask would consolidate are whole-file, case-sensitive floors in `measure_negation_density.py`; deleting one drops the floor below 16/16, and merely capitalising an anchor drops it too. We are not willing to trade a disclosure for a consolidation |
| 8 | state the general-vs-FL split once after the proposition | **Declined as already done.** "Nothing in this statement is about federated learning" `:329` |
| 9 | move development/meta-narrative to an appendix claim audit | **Partly done, in change (4).** Body `provenance framing` was already 0; the two clauses cut were the only real instances |
| 10 | add a canonical terminology diagram | **Done by pointer, change (6)**, not by a new float |
| — | retitle P4 to *adversarial influence preservation* | **Declined**, pre-registered name; answered by change (5)'s table row |

---

## 4. The abstract: a deliberate no-change

We left it untouched, and this is a decision rather than an omission. It is **1588 of a 1600-char
ceiling**, 14 sentences, longest 164 against a 200 ceiling, and it already carries every number the
review wants foregrounded — including −0.273/+0.125 at *n*=20, the five-of-seven disagreement, and the
CIFAR-100 replication. Any addition breaks the ceiling; any cut deletes a measured result rather than a
word. The salience defect was never in the abstract — it was that the abstract's numbers let our
instrument report the first screen as solved while §1 rendered its own statement on page 2.

## 5. The seed top-up: deferred, not refused

The review's §22 item 4 asks for *n*≈15–20 seeds on a flagship cell. Its own clarity addendum rates more
experiments **Low** for clarity and says it would not add an empirical suite purely to improve clarity.
We agree and have deferred it to an Empirical-validation round rather than declining it. The flagship
CIFAR-10 sign-reversal cell already runs at *n*=20; the cells that would gain most are the *n*=5 rows.

---

## 6. Verification run for this round

- **Page budget.** Main text is **9 pages**: Conclusion heading and Ethics statement both open and
  complete on p9, judged on the body's last rendered line after a fixpoint build. The remaining margin
  was probed with filler sentences, not computed from source characters: it is ~1 rendered line.
- **Clarity instrument** exits 0 with no ratchet raised. `spine_span` still 2, `causal_sec1` 6/6,
  `p_tokens` 7, `max_xrefs` 5, `long_para` 848/850, `design_vocab` 5/5, `joined` 0, `body_questions` 0
  unsanctioned, `provenance` 0, `body_homes` 2/2, `fig1_caption` re-baselined downward to 979.
- **Negation floor** holds at 16/16 protected disclaimers, 15 in the body window.
- **Build** is at a fixpoint for both documents: zero `LaTeX Error`, undefined or multiply-defined
  reference, `??`, `[?]` or `Overfull` in either log; 75 + 11 pages.
- **No orphan float.** `fig:modeS_dose` carries two non-comment `\ref`s. LaTeX never warns about a
  `\label` with no `\ref`, so this was checked directly rather than read off a clean log.
- **The figure was verified as a drawing, not as a hash.** `/CreationDate` moves on every regeneration,
  so an md5 proves nothing: panel (a) was confirmed unchanged by pixel read, the forest is now (b), and
  the extracted text of `modeS_causal.pdf` contains no `(c)`. Native size 431.5 × 238.3 → 431.5 × 175.1 pt.
- **A number the panel prints was re-derived, not assumed.** Panel (a)'s admission annotation reads
  `0/240`, and the body sentence it answers to —
  "the four \texttt{cifar\_cnn} CIFAR-10 rows contribute $60$ measured rounds each" `:1611` — is now
  pinned by an assertion in the generator, because the pooling filter had been excluding by *dataset*
  while the claim is about the four CIFAR-10 CNN arms. A later round added a CIFAR-10 ResNet18 arm that
  the dataset test admitted, which would have redrawn the annotation as `0/312` under a caption and a
  sentence that both say 240. The committed figure was stale rather than wrong; it is now both.
- **Self-counts.** The deleted inventory clause was the last body sentence counting the paper's own
  floats; a sweep of the body for `<number> <propositions|figures|tables|rows|...>` returns one hit, "the
  two rows" of the display two lines above it. The vocabulary table's two self-counts (lead-in and the
  sentence below) were both moved from five to six with the new row.
- **Punctuation and anonymity.** Zero U+2014/2013/2212 in `main.tex`, `supplementary.tex` and
  `references.bib`; every signed cell in math mode so the sign is a true minus; `pdfinfo`
  Title/Author/Subject/Keywords empty; `\iclrfinalcopy` not invoked; no rendered "Round N" in the
  extracted PDF text.
- **Shared figure artwork.** The three standalone figures the generator also writes into
  `workshop_paper/figures` are **text-identical to their committed versions** — only `/CreationDate`
  moved — so nothing baked into shared artwork went stale.

## 7. What this round does not move

Novelty (6) and Significance (7) are untouched by design; this round moves Clarity only, and we expect
the ceiling that implies for the overall score. No new experiment, seed, dataset, attack or architecture
was run, no scope condition, limitation, withdrawal or disclosure was deleted, and no pre-registration
was created or touched.
