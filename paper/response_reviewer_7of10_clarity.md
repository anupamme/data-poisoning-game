# Response to the seventh review — clarity and presentation

**Scope of this round.** The review scores Overall **5/10** (confidence 0.75) with **Clarity 6/10**,
Novelty 6 and Significance 6. This round is **clarity- and presentation-scoped only**, at the authors'
instruction. Novelty and Significance are deliberately unmoved, and §7 below says why rather than implying
they were treated. **No new experiment, dataset, seed, `N`, `f` or attack was run**, which is what the
review asks for twice (*I would not add another 30 experiments*; *I don't think you need more technical
detail to reach 8. In fact, adding more technical detail would probably make clarity worse*).

The review predicts 7–7.5 for its first five items and 8 for the full eleven-item cleanup. **That is the
review's prediction, not a measurement, and this letter does not claim a score.** What it claims is that
each of the eleven items is either landed, declined with a machine reason, or already in the paper — and it
gives a source line for every one.

The review read **`main(20260909-033131).pdf`**, which is the post-Round-65 build. Two of its eleven items
were therefore aimed at prose that had already moved.

---

## 0. Two findings that changed what this round was for

### (i) The passage the review calls the best writing in the manuscript was not in the manuscript

Its item #17 quotes a *Core message, in three sentences* and says *that should basically become the paper's
organizing principle*. That text appears **nowhere** in `main.tex`, `supplementary.tex`, the rendered PDF,
or any of the 55 commits (`git log --all -S` returns nothing; the PDF extract returns nothing). Its quoted
abstract opening (*whether the downstream mechanism is responsible*) likewise does not match the text
(*caused its success*). The three sentences are the **reviewer's own précis**.

We treat that as a gift rather than a discrepancy: the reviewer drafted the spine they were asking for. It
is now §1's opening, in their beat order and our numbers — `:144` and `:151`. Adopting it is a rewrite, not
a new claim.

### (ii) The review's two most dangerous objections were already answered on pp3–4, and were not found

| the review's objection | where the paper already answered it |
|---|---|
| #17, *show that this is a class of evaluators, not merely your implementation* | p3, `:225`: "Nothing in this statement is about federated learning." — $M$ any outcome metric, $\theta$ any threshold, $P$ any condition on the two mechanisms' definitions, $\mathcal{D}$ any menu |
| Q1, *Can you provide evidence that outcome-gated evaluation is used beyond your own criterion? This is my biggest question*, and W2/#4, *Related Work gives one concrete external example, Fenaux et al.* | p4, `:249`: $6$ of $59$ coded papers ($10.2\%$) gate attribution on the composed outcome while running no identifying contrast; $46$ ($78.0\%$) do run one |

Both facts were in the paper when the review read it. Neither reached the reader. **A finding that is
present and unfindable is a clarity defect, not a content one**, and it is the defect **E2** treats: both
are now in §1's contribution item at `:170`, on p2.

One correction to the review's premise while we are here: Fenaux et al. is no longer offered as our
external instance. `:2131` reports that the audit **withdrew** it — "so the instance is withdrawn and the
audit we ran to support the claim is what withdraws it" — and the measured rate replaces it. The four
`fenaux2025hammer` citations are all in the appendix.

---

## 1. Already landed before this round, with a check-site for each

| the review's clarity ask | state, with check-site |
|---|---|
| #4/#11, one vocabulary, terms fixed in one sense | `:362`: "Five words, fixed here and used in one sense throughout", with influence and attenuation named at `:390` as the words that are **not** those five |
| #6, informal statement before Prop. 1 | `\textbf{Intuition.}` at `:205`, immediately above the proposition, p3 |
| #8, collapse the caveats into one Scope box | It **is** a box: `box:scope` at `:274`, 1856 rendered chars, p5, with *Formally / Empirically / Not established* — the last at `:299`: "Not established." |
| #9, demote Theorem 8 | Appendix-only at `:1121`: "Bounded reweighting preserves the downstream discriminative mechanism". The body carries no mechanism-preservation theorem, so there was nothing to demote |
| #10, Mode S in one sentence: what / why / therefore | `:452` does it in one paragraph: pins $c{=}1.0$, spreads benign mass over $\rho$, "so a rise in ASR here cannot be attenuation" |
| #12, C0–C3 should not be met early | `measure_clarity_load` reads **3** C-tokens in the whole body (C1:2, C2:1). Already appendix-only in substance |
| #15, declarative section headings | The five content headings are declarative sentences (§2 *The Testability Boundary: Why the Gated Test Fails* at `:194`, §5 *Statistic Preservation Does Not Identify Attack-Suppression Preservation* at `:432`). Introduction and Conclusion keep their conventional names. What was essay-like was the **13 question sentences inside the prose**, which is what **E9** fixed |
| the review's own p2 test | All five beats already rendered **by p2**. Rendered *position* was not the binding variable — **contiguity** was, which is why the new measure counts paragraph spans and not pages (§3) |

---

## 2. What this round changed, in the review's own priority order

### 🔴 Highest priority

**E1 — the spine, in the reviewer's three sentences.** §1 opens with the paper's core question (unchanged
and verbatim, because the five-question measure's Q1 is keyed on it), then two short paragraphs carrying the
review's five beats: `:144` — "We propose no new defense." — states that preserving what the downstream
defense reads leaves the suppression question open and that the natural test cannot answer it at any sample
size, and `:151` — "What replaces the gated test is a design rather than a better test" — gives the
intervention and closes on the reversal, $-0.273$ outcome-gated against $+0.125$ within-defense at $n{=}20$.
Two paragraphs that recited the same content were absorbed rather than duplicated.

The check is the new `spine_span` measure, and it moved from **4 to 2** (§3, PROBE 4).

**E2 — the class and the prevalence, surfaced in §1.** The contribution item now states that the boundary is
about **a class** of evaluators and contains no dataset, defense or metric, and that the gate is
`:170` "a documented convention whose prevalence we measured rather than assumed" — $6$ of $59$
($10.2\%$) against $46$ ($78\%$) that do run an identifying contrast. This is the round's highest-leverage
edit: it answers the review's self-declared biggest question with a fact already in the paper.

**The pre-registered decline stays.** `:249` still reads "A low rate was pre-registered as outcome (ii)",
and the appendix at `:2158` still reads "So the six are papers that had the design and did not run the
contrast". We surfaced the measurement; we did not upgrade the claim. All three numbers were re-derived from
the audit rather than from the plan: `:2129` for $6/59 = 10.2\%$ and for the $5/59$ reading when "the
audit's mechanism-distinctness rule is dropped", `:2158` for $46/59 = 78.0\%$, the $33$ design papers, the
$27$ of them that run a contrast, and the $0/33$ secondary.

**E3 — one object for the hierarchy.** The centred chain display and the four-word glossary folded into
Def. 1 (`:341`) plus the five-row glossary at `:362`–`:371`, which now carries per level the plain word,
what it asks in one clause, and whether it bears on security. Every formal statement survived; the plain-word
chain survives as the row order. `(P#)` fell **12 → 7** and the ratchet was re-baselined **down**.

**E4 — the informal statement moved above the formal one.** The sentence the review singles out is now
inside `\textbf{Intuition.}` at `:205`: "That empty cell is the whole of it". It used to sit after the
figure. This is the review's exact ask and it cost nothing.

**E5 — §5 leads with the reversal.** The opener now **states** the flagship instead of promising it —
`:440` "on one cell the two designs disagree in sign" — at $-0.273$ $[-0.334,-0.213]$ against $+0.125$
$[+0.095,+0.155]$ at $n{=}20$, disjoint and both excluding zero. **§5 was not reordered** (authors'
decision): the paragraph order below the opener is unchanged, and the opener is not a forward reference
because both numbers are already stated on p1 at `:151`.

### 🟠 Next

**E6 — Fig. 1's caption stops doing the figure's job.** `:107`, **1480 → 1153 chars**. Cut: the link-by-link
description of panel (a) (the figure draws it), panel (b)'s numbers (four other homes), and (c)'s recital of
what the panel prints. Four things stayed because each is a **disclosure with a thin or sole home**: three
scopes rather than one experiment; that the middle number pools four CIFAR-10 arms rather than Krum alone;
the $\nRightarrow$ legend (a legend is not redundancy); the dashed rule's meaning (the cells above it are
training data for a since-withdrawn rule — **sole home**); and the per-row $n$ note. The caption is now
**gated as a ratchet at 1153**, so a later round cannot quietly undo the cut.

**E7 — the meta-framing sentence goes; both disclosures stay.** The sentence the review names in its #16
(*The paper is a methodological correction … and we report our own refutations as such*) is removed. Its two
facts keep verified body homes, and this is the one deletion in the round that could have lost a disclosure:

- the false negative of our own criterion → `:270`: "is emergent and a false negative of our own criterion"
- the collapsed cross-arm admission ordering → `:544`: "A pre-registered replication that refutes the positive half."

Both destinations were grepped **before** the cut, and both are now held by a **gated floor** rather than by
this letter's word (§3, PROBE 2).

**E8 — the evidence tiers named in the main text.** `:591`: "Four questions, four kinds of answer" —
algebra before any run, one causal intervention at fixed downstream defense and attack, replication on
further datasets, and what screening costs in recall, "the last three being Table~\ref{tab:tiers}'s tier
words and the first a proof, of very unequal weight and never pooled". Reuses the existing `\emph{}` slots,
so the italic ratchet did not move. **See deviation (a):** the plan's proposed words *proved / measured /
exploratory* do not exist in the paper.

### 🟢 Polish

**E9 — ten question openers became declaratives**, keeping three sanctioned questions. Body questions fell
**13 → 3**, unsanctioned **10 → 0**. The three survivors are allowlisted **by phrase, not by line number**,
so a reworded sanctioned question fails loudly instead of passing silently. Five of the rewrites:

- `:419` "The inadmissible comparison is replaced"
- `:561` "The design choice, not the defense"
- `:565` "Seven cells, not one"
- `:646` "statistic-preservation screening is the rule that keeps it"
- `:666` "four clauses and report where the chain broke"

All five `\emph{}` design words survive (`design_vocab` 5/5).

**E10 — the *so what* clause.** Folded into E5's rewrite as a clause rather than added as a sentence, per
the review's #14.

---

## 3. The instrument, and the probe that falsified our own plan

Three measures were added to `experiments/measure_clarity_load.py`, reusing its existing `paragraphs()`,
`body_window()`, `page_words()`, `uncomment()`, `prose_lines()` and `caption_chars()` helpers. `NEG_TOKENS`,
`PROTECTED` and `Q5_TEST` were not touched.

- **`spine_span`** — the review's acceptance test measured as **contiguity**, not as pages: how many
  distinct body paragraphs the five beats occupy. Gate: ≤ 2 adjacent paragraphs.
- **`body_questions`** — question sentences in rendered body prose, with three phrase-keyed sanctions.
- **`provenance`** — a **ceiling** on research-history framing over the body window, counted on the
  *framing* and never on the fact.

**The design trap, stated because it is the inverse of last round's.** `our own criterion` must **not** be
in the provenance list: it is a `PROTECTED` anchor, so listing it would set a ceiling and a floor against
each other and make one unsatisfiable. Last round's trap was anchoring a floor on a *negation*; this
round's was anchoring a ceiling on a *disclosure*. The two instruments compose: `provenance` removes the
framing, `PROTECTED` and the new `BODY_HOMES` floor prove the facts survive.

Four probes were run against scratch copies **before** Phase B, and all four outcomes are reported,
including the one that broke the plan.

| probe | expectation | measured outcome |
|---|---|---|
| **1** — delete only the framing | both scripts exit 0 | **PASS.** negation `48.0% (83/173)`, floor `16/16 present (15 in the body window)`, `negation exit=0`, `provenance hits = 0 []` |
| **2** — probe-1 state **plus the disclosed fact deleted** | `measure_negation_density` exits 1 naming the disclosure | **FALSIFIED THE PLAN.** `16/16 present (14 in the body window)`, `negation exit=0` |
| **3** — probe-1 state with the framing restored | `provenance` exits 1 | **PASS.** `provenance hits = 4`, all on one line (`methodological correction`, `we report our own`, `our own refutations`, `we had predicted`), `provenance exit=1` |
| **4** — was `spine_span` failing before E1? | it must have | **CONFIRMED NON-VACUOUS.** pre-E1: `spine span (paras) : 4 (the review's five beats, in 4 adjacent paragraph(s); its own threshold is 2)`, alongside `body questions 13 (10 unsanctioned)` and `provenance framing 4` |

**PROBE 2 falsified verification item 1 of our own plan.** The plan asserted that the `PROTECTED` floor
would catch the deletion of a relocated disclosure. It does not. `protected()` is a **whole-file**
presence test (`rx.search(text)` over all of `main.tex`, appendix included), and the entry keyed on
`false negative of our own criterion` also matches **three appendix twins**. So both body statements could
be deleted and the reassuring `16/16` line would not move. The `in_body` count is *printed* and never
gated.

That is why **`BODY_HOMES` / `body_homes()`** was added: a floor over the body window, gated `< total`, with
each pattern keyed on wording **unique to the destination** rather than on wording shared with the sentence
being cut — a floor the doomed sentence can satisfy tests nothing. It now reads `body homes 2/2` at `:270`
and `:544`. **A ceiling on framing is only safe beside a floor measured in the same window the framing was
cut from**, and the plan did not have one until this probe.

---

## 4. The measured state

Pre-round is the Round-66 build the plan recorded; post-round is the current fixpoint build.

| measure | before | after | note |
|---|---|---|---|
| body prose chars | 39504 | **38954** | window is now lines 111–668 |
| Conclusion / Ethics page | p9 / p10 | **p9 / p10** | residue before the heading: `'E'` |
| `(P#)` tokens in body | 12 | **7** | ratchet **re-baselined down** (E3) |
| C0–C3 tokens in body | 3 | **3** | C1:2, C2:1 |
| abstract chars | 1588 | 1588 | cap 1600 |
| longest abstract sentence | 164 | 164 | cap 200 |
| Fig. 1 caption chars | 1480 | **1153** | now a ratchet (E6) |
| scope box chars | 1856 | 1856 | untouched |
| bold runs | 39 | **39** | at cap, not raised |
| italic runs | 62 | **60** | ratchet **re-baselined down** (E9) |
| max cross-refs/paragraph | 5 | 5 | at cap |
| longest paragraph | 846 | **831** | cap 850 |
| comment-joined paragraphs | 0 | **0** | E1/E3/E7 all delete prose between comment blocks, which is how this defect is created |
| `spine_span` | *(4, pre-E1)* | **2** | beats in paras 2–3, `:144`/`:151` |
| body questions | 13 (10 unsanctioned) | **3 (0 unsanctioned)** | |
| provenance framing | 4 | **0** | |
| body homes (floor) | *(no measure)* | **2/2** | `:270`, `:544` |
| negation density | 47.7% (83/174) | **46.4% (77/166)** | re-run, never diffed against a remembered number |
| protected disclaimers | 16/16 | **16/16** | 15 in the body window |
| first proposition | p3 | p3 | |
| five questions answered by | p1 | p1 | threshold p3 |
| `design_vocab` / `jargon_front` | 5/5 / 0 | **5/5 / 0** | |

All three gates exit **0**: `measure_clarity_load --gate`, `measure_body_chars --gate`,
`measure_negation_density`. **No ratchet was raised** and three were re-baselined downward
(`(P#)` 12→7, `fig1_caption` 1480→1153, `emph_runs` 62→60).

Also verified after the fixpoint build of both documents: zero LaTeX Error, undefined reference or
citation, multiply-defined label, `??`, `Rerun to get` or Overfull hbox/vbox in either log (grepped with
`grep -a`); the four frozen artifact md5s hold at their **literal** paths (`2f4d9920`
`results/all_compositions/summary.json`, `a16ef13f` `results/dose_femnist/summary.json`, `a0717893`
`results/headline_seed_topup/summary.json`, `6e25ef3a` `results/comparability_six_cells.json`);
`git status --porcelain results/` empty; binding arm still `krum / scaling, EMNIST at m* = 0.0343` with
`BINDING ARM MOVED` not printing; 0 U+2014/2013/2212 in all three sources; `pdfinfo`
Title/Author/Subject/Keywords empty; 0 rendered "Round N" in the **extracted PDF text** beyond the single
permitted git-hash provenance fact; 0 unprotected `\texttt{}\texttt{}` joins; and the Reproducibility
statement's pre-registration count at `:706` — "the count of" `experiments/pre_registration_*.md` — still
equals the 22 on disk.

---

## 5. Declined, each with a machine reason rather than a preference

1. **#18, retitle the paper.** Declined; authors' decision. Noted here rather than acted on: the review's
   own Option 3 is a near-synonym of the current title, and *Causal Evaluation* is the phrase its #14
   objects to.
2. **#5 as stated, redraw Figure 1 as a pure conceptual map.** `figures/modeS_causal.pdf` bakes `P3 P4 P5`
   and `figures/story_chain.pdf` bakes `P1`–`P5` into rendered pixels, and both generators also write into
   a pinned companion document that is out of scope this round. Delivered as **E6**, the caption half —
   which is the half the review itemizes (*the caption is almost a mini-paper itself*).
3. **#11 as stated, rename Table 1's columns and drop $\Delta$agg.** The table has overrun the text block
   twice and sits at `\tabcolsep` 2.5pt (`:484`); the review's longer headers do not fit. **$\Delta$agg. is
   the column that cannot go**: `:496` — "Krum, score-only" — is the **only** place in the body where the
   pre-registered control's own displacement, $0.986$, appears, so dropping the column would delete a
   headline number from the main text. The two facts the review says a reviewer needs are already
   emphasized in the caption, which states that $\Delta$ agg. is the displacement of what the defense
   **emits** and that `:485` admission and influence are "never interchangeable".
4. **#7 as stated, PROVED/MEASURED/ILLUSTRATIVE as display labels.** Delivered as **E8** in prose instead:
   new `\textbf` would raise the bold ratchet at 39/39 and new `\emph` the italic ratchet.
5. **#9/#16 compressing the appendix's research history.** Out of scope, and the review does not ask for
   it — its recommendation #4 is to move history *into* the appendix, where it already is, and it grants
   that this level of provenance *is useful for an artifact paper or preregistration report*. ICLR 2027
   gives unlimited appendix pages. Cutting 62 pages of provenance risks exactly the disclosure loss that
   `PROTECTED` and `BODY_HOMES` exist to prevent, for a reader who skims. Body-only this round.
6. **#4/W2, argue that the gate is common.** The rate is **measured at 10.2%** and a low rate was
   **pre-registered as outcome (ii)** (`:249`, `:2158`). E2 surfaces the measurement; it does not reverse
   the finding. This is the one place where the review's Significance concern and our pre-registration are
   in direct tension, and we resolve it in favour of the pre-registration.

---

## 6. Deviations from the plan, reported rather than buried

**(a) E8's tier words do not exist.** The plan specified *proved / measured / exploratory* as the three tier
words to name in the main text. `tab:tiers` (`:838`) uses **Causal / Replication / Screening**. The plan's
words were invented in the planning, not read from the table. E8 therefore names the real distinctions in
prose at `:591` and says explicitly that the last three are the table's tier words and the first is a
proof — which is more accurate than the plan's version, since one of the four answers is algebra and not a
tier at all.

**(b) E1 became two paragraphs, not one.** The single three-sentence paragraph the plan specified measured
**939 chars** against the 850-char `long_para` ratchet, and the ratchet may not be raised. It was split at
the beat boundary: `:144` carries beats 1–3, `:151` carries beats 4–5. `spine_span` still reads 2, because
its gate is *≤ 2 adjacent paragraphs* rather than *1*.

**(c) The plan's named funding reserve could not pay.** Phase C listed three cuts in §2 as the reserve. None
was spendable: they sit before Table A2's float, and cuts before a float are reabsorbed by the float rather
than moving the tail. Funding came from **p8–p9 instead**, where the p8/p9 break falls immediately before
§6.1's heading and TeX will not break just after a heading — so only cuts **on p9 itself** move the last
line. The plan's line accounting was right about the total and wrong about the location.

**(d) Two `PROTECTED` anchors were broken and restored.** The floor dropped to 14/16 twice in one session,
both times by *rewording* prose that happened to carry an anchor, not by deleting a disclosure:

- a compression of the Prop. 1 sentence dropped `attack-` from `identify attack-suppression preservation`;
- converting the collider paragraph to a run-in head moved `that half is textbook` to the front of a
  sentence, where it rendered `That half is textbook`. **`PROTECTED` is compiled without `re.I`**, so the
  capitalisation alone broke the match.

The second was fixed by **reordering the clause so the anchor renders mid-sentence** (length-neutral). Both
fixes are in `main.tex`; the instrument was not edited.

**(e) Five defects were found only by the 300-dpi page read, and no gate can see any of them.** Verification
item 11 was a pixel read of pp 1–10. It found and this round fixed:

1. a near-verbatim §1/§2 echo in the contribution item at `:170`, now carrying E2's content instead;
2. the Roadmap's self-count at `:188`, now "one numbered table and two figures" — literally true before
   (one `\begin{table}` in the body, verified) but a reader on p6 counts the booktabs glossary block as a
   second table;
3. **a live self-contradiction across one paragraph break**: E3 grew the glossary from four rows to five,
   `:362` was updated to "Five words, fixed here and used in one sense throughout", and the follow-on
   sentence still said *those four* directly beneath a table the reader had just counted five rows in. Now
   `:390`: "each distinct from those five". No instrument sees this — a bare `tabular`'s rows are not
   counted, and grepping for `four` is useless because the same page says *four aggregators* and *four
   within-family statements*, both correct;
4. a muddy tier clause at `:591`, rewritten (E8);
5. **a terminology contradiction between p7 and p9.** `:619` said bare *admission* rose for reputation,
   while Table 1's caption at `:485` states that admission never moves in any measured round of any arm.
   Both readings are correct terms of this paper — reputation's $\Delta$ adm. **is** 0.000 and what moves is
   its influence, 0.000 → 0.020 — and the reconciliation is 30 pages later in the appendix. Fixed by naming
   the symbol on the page where the two readings collide: the small admission change $\Delta\Lambda_a$
   `:619` "rather than the large decision change". No grep could reach this: the site uses
   neither of the outlier phrasings a wording audit keys on.

**(f) Figure 1 panel (a) was checked and left, with its reason.** Panel (a) draws **four** nodes where
Def. 1 has **five** levels. This is a legitimate compression — the panel's statistic box carries P1 and P2
together — and it is left unedited on two grounds: figures are not regenerated this round (decline 2), and
the caption is at its 1153-char ratchet ceiling, so the clarification cannot be bought there without
undoing E6. Recorded here so it is not rediscovered as a defect.

**(g) The plan's `(P#)` forecast was met exactly.** 12 → 7 was predicted as "~7"; it landed on 7. Reported
only because the ratchet was re-baselined to the measurement rather than to the forecast.

---

## 7. What this round did not touch, and why

- **Novelty 6 and Significance 6 are untouched by design.** The review is explicit that moving them needs
  either a new theoretical result or an argument that the gate is widespread. The first is out of a
  clarity-scoped round; **the second is foreclosed by our own pre-registered outcome (ii)** (decline 6).
  Saying so is more honest than implying the dimensions were addressed.
- **The two $n{=}3$ arms** (EMNIST, ResNet18) remain the empirical soft spot the review's W3 names. The
  paper's own words on it stand: the remedy is seeds rather than wording. Out of scope here.
- **No retitling** of the paper, §5, P1–P5, C0–C3, Mode S/A/M, or *admission* / *influence* /
  *attenuation*. E3 added a column; it renamed nothing.
- **No renumbering** of P1–P5, declined a round earlier for reasons that still hold.
- **No deletion of any scope condition, limitation, withdrawal or disclosure.** E6 and E7 moved homes;
  every destination was grepped before the cut and two are now gated by `BODY_HOMES`.
- **No §5 reorder**, no new subsection, box or figure, no float promoted from the appendix, no appendix
  compression, no edit to the pinned companion document, no write to any `results/` directory, no figure
  regenerated, and no pre-registration altered.
- **No page-budget line was bought by shortening a heading.**
