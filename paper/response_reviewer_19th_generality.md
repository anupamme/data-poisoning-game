# Response to the nineteenth review (5/10, borderline / weak reject, confidence 4/5)

Thank you, and thank you in particular for the closing instruction: *I would spend the remaining revision
effort on making the general methodological contribution unmistakable, rather than adding more
experiments.* We took that literally. **This round adds no experiment, no arm and no number.** It is a
repackaging round, and the reason it could be one is the finding we report first: on four of your five
items the content you asked for was already in the paper, mislocated or unnamed.

The sentence you say you want believable is this one:

> *The contribution is not that one particular FL screening rule is flawed; it is that causal attribution
> of mechanism preservation in composed defenses requires a different estimand and intervention, and we
> characterize the boundary between what statistic-level evaluation can and cannot identify.*

We had proved both halves of that boundary and put them **sixteen pages apart**, page 3 and page 19,
stated neither as a boundary, and hedged the one sentence that could have named it. That is our failure, not a reading
failure, and §1 below is the repair.

Source line cites in this letter are `main.tex` **source** lines, not the PDF's margin numbers, which are
ICLR rendered line numbers and differ by roughly four.

---

## 1. Item #1: the characterization, which needed packaging and not mathematics

This is the item we think moves the score, and it needed no new result.

**The impossibility half was on page 3 and the sufficiency half was in the appendix.** The proposition's
converse was already proved — the appendix paragraph that establishes it reads that the obstruction is
the gate's reading of the outcome metric and nothing else — but the body stated only the negative, so the
paper read as *this screen is invalid* rather than as a boundary with two sides.

Four edits, all in place, none renumbering any statement:

| what | now reads | cite |
|---|---|---|
| the proposition's own caption | *"When mechanism attribution is identifiable, and when it is not"* | `:340` |
| the converse, as a clause of that same statement | *"so on this class of screens the boundary is exact"* | `:345` |
| the generality note, strengthened from *nothing here is about FL* to naming what is characterized | *"What this characterizes is a shape of screen, not federated learning"* | `:358` |
| the section title, which named only the negative half | *"What the Gated Test Cannot Identify, and What Can"* | `:304` |

We added **no** numbered proposition, theorem or corollary, deliberately: inserting one renumbers the
document, moves the gate that checks the central claim still renders by page 3, and stales every literal
statement number already in the prose. The characterization is the existing statement made two-sided.

**The two conditions are now separated and named**, in the appendix proof of the converse, because a
design can satisfy either one alone —
*"selection that does not read the outcome to be explained"* `:1381` —
and support at both levels of the downstream metric for each level of the mechanism condition.

**And the hedge that fought your point is reframed rather than deleted.** It used to say the result's
content was narrower than a general identification boundary, which is the exact sentence your item #1 is
arguing with. It now opens
*"Its content is a boundary on one class of screens rather than on evaluation at large"* `:1359`
and then says what the scope buys:
*"What we offer as the transferable contribution is that characterization together with "* `:1359`
the admissible design it derives. **The narrowness is still explicit** —
one screen class, one gate shape. We are not claiming a general theory of evaluation; we are claiming the
boundary is *exact* on a class we name, which is a stronger and more checkable thing than the disclaimer
was.

**The paper is retitled** to say what it is about rather than what it refutes. It is now *Beyond Statistic
Preservation: Causal Identification of Mechanism Preservation in Composed Federated-Learning Defenses*,
which the source carries across three lines from
*"Beyond Statistic Preservation: Causal"* `:49`.
The repetition is load-bearing: *statistic* preservation against *mechanism* preservation is the thesis. The supplement's title moves with it, since a stale twin title passes a clean build.

---

## 2. Item #2: the P3 ↛ P5 break, shown as a table instead of narrated

Conceded, and the diagnosis is worse than *not prominent enough*: **we had measured the break for seven
arms and then described it in prose.** The body already carried a channel table with columns for the
aggregate, the decision, the admission, the influence and the ASR. What it did not say is that those
columns *are* the levels of the preservation hierarchy, so the checklist your §17 asks for was on the page
and unreadable as one.

That is a caption change, not a new table. It now reads the columns as levels —
*"report and the highest level it preserves is read off it"* `:774` —
and the appendix twin says the same in its own caption:
*"The last three channels are the levels of Def."* `:1847`.

§1 states both directions of the break in one sentence:
*"Krum's selection flips in $0.733$ of rounds while its admission is unchanged"* `:166`
and its suppression does not follow, while
*"selection is bit-identical at every rung and its ASR nonetheless"* `:166` falls.
Those are two adjacent rows of that table, which is the point: the break is visible as a *pattern across
rows*, not as two anecdotes in different sections.

The caption also says what the table is **not**: it does not order the arms, the influence and ASR columns
disagreeing in rank; and the two lowest levels are settled by algebra before any run rather than left
unmeasured `:774`.

**The abstract leads with the gap rather than with our screen** (§26). Its first sentence used to be about
the screening rule; it now opens
*"Whether the downstream mechanism caused a composition's suppression is unidentified by any test admitting it"* `:78`
because a constituent already suppresses the attack, with the screen impossibility following as the
reason. This was a reorder and a merge, not an addition: the abstract is within four characters of where
it was.

**And the sign result is now stated as two estimands rather than as a choice** (§7). Two sites framed it as
an analyst picking between designs, which reads as if the sign were up for negotiation. §5 now ends
*"the two admissible estimands give opposite contrasts here, not one at two precisions"* `:675`,
and the general/FL-theory/FL-evidence split matches it:
*"the two admissible estimands do not differ in precision, they give contrasts of opposite sign"* `:389`.

---

## 3. Item #3: emergent synergy is a *coverage* failure, and we had not said so

You are right that the FoolsGold→RFA result was underweighted, and we think we now know why: the paper
presented it as the first of *two measured facts* without saying that it is a **different kind** of
failure from the identification failure. One certifies nothing because the estimand is unidentified; the
other certifies nothing because the composition is outside the screen's reachable set entirely.

§1 now names it: *"a \emph{coverage} failure, independent of the broken link that follows"* `:166`. The
numbers are unchanged — 0.093 against 0.732 and 0.845 — and §6.3 draws the consequence in the sharpest
form we can state it: what the screen discards is the menu's lowest-ASR composition, so
*"do not screen on inherited suppression at all"* `:1009`
if what you want is a menu's strongest composition.

That sentence costs us our own criterion, which is the point.

---

## 4. Item #4: a non-rescaling class, answered from disk, with the gap stated

**We cannot give you what you asked for, and here is exactly what we have instead.**

What exists is one arm outside the positive-rescaling family the theory covers:
*"masking a random $0.8$ fraction of each \emph{benign} client's coordinates at fixed $L_2$ norm leaves adversarial updates exactly untouched"* `:975`.
So nothing we prove predicts it, and the dissociation appears anyway: the decision moves in
0.467 of rounds at admitted adversarial mass 0.000 `:975`. Neither pre-registered refuting branch fired.

**The paper's own concession on that arm stands verbatim and we are not softening it**:
*"One mask is not the class of all non-rescalings"* `:975`.

What does **not** exist is a clipping-or-noise → robust-aggregation ladder. We attempted the nearest
thing:
*"Norm clipping upstream on CIFAR-100 needed no mask; its premise failed"* `:899`,
so that replication did not carry. Two further oracle-free attempts are recorded on the same line, and the third
*contradicts* the central negative — a decision flip alone carries +0.322. We report all three as
failures rather than omitting them, and the oracle limitation is in the abstract.

So: item #4 is **not answered as asked**, no clipping ladder is claimed, and zero compute was the round's
constraint rather than a judgement that the experiment is unimportant. It is the first thing we would run.

---

## 5. Item #5: the cut, measured rather than asserted

The body is down **868 characters, 43,034 to 42,166**, by substitution only: every clause removed from a
body paragraph was verified present at an appendix home first, and nothing lost its last home. That is
2%, not 20–30%, and we will not pretend otherwise.

On the appendix, which is where the length is: we measured it rather than eyeballing it. The appendix
window is 479,285 non-comment characters, 81.2% of it prose outside proof and table environments. A
redundancy scan over that pool finds **14 sentence pairs** above a 0.3 Jaccard threshold and **zero
paragraph pairs** with equal numeric sets. Reading all 14: seven are the *same disclosure template
instantiated on a different arm* — the two that share a Jaccard of 1.00 are `+0.3218` and `-0.2857`, two
different arms' effect sizes in the same sentence shape — and deleting either member deletes that arm's
disclosure, not a repetition.

That is the honest position: a 20–30% appendix cut is available only by dropping per-arm disclosures,
scope conditions and per-seed values that this paper is committed to keeping, several of them added
because earlier reviews asked for them. We would rather be long than quieter.

---

## 6. Item §8: Mode S keeps its name, and gains a descriptor

Declined, with a reason rather than a refusal. The name is not only in the prose: it is drawn into
Figure 1's art, it is the `modeS` and `doseS` keys of 38 frozen results artifacts, and it appears in 34
pre-registration documents committed before their data existed. Renaming it in the paper would desync the
prose from the evidence it cites, and the pre-registrations cannot be rewritten after the fact.

What we did instead is gloss it where a reader looks a term up — the sentence that fixes the paper's
vocabulary now reads *"Mode~S (statistic-only) and score-only Krum"* `:316` — and the paragraph that
defines it says in full what it pins, and that pinning requires knowing which clients are adversarial, so
it is a measurement device and not a defense `:717`.

---

## 7. What we did not do, stated plainly

- **No new experiment.** Following your own closing recommendation. This means item #4's real answer is
  still owed.
- **The oracle gap is unchanged.** Three attempts, the third against us. We do not write that the
  negative reproduces without an oracle, because it does not.
- **A previous review asked for the opposite of item #1** — two or three published case studies, which is
  an add-experiments request. We followed this one, as the newer review and the one holding the current
  PDF, and the paper continues to decline to claim the audited papers' conclusions are wrong.
- **The protocol gained a reporting clause, not a fifth design clause.** The four design clauses are
  unchanged in what they force; clause four now ends by saying how to report an arm —
  *"classify it by the highest level that survives, five outcomes"* `:650` —
  with the body table as the filled-in instance for seven arms.
- **The related-work headline says only what the audit's frozen fields license** (§5). You wanted the
  headline to be that no included paper varies an upstream stage and asks whether the ordering of two
  downstream arms survives it, with intervals over seeds. We agree that is the gap, and the audit has no
  field for it, so the paper states it over the four coded routes and then says what it is not:
  *"So this is a design-level reading of the route field and not a rate"* `:2774`.
  Re-coding the 59 rows for a fifth criterion now would amend a freeze after its counts exist.
- **Figure 1's caption lost two bookkeeping sentences** (§22), so the dashed non-implication links are what
  the reader meets first. The stale row label did not vanish with them; it moved to the appendix and is
  stated there in full:
  *"One row label in Figure"* 1(b) *"is frozen as drawn and is no longer current"* `:3056`.
  The drawing is left byte-untouched rather than redrawn, so the figure and the table are read together.

## 8. On the score

Your five items were, in your ordering, one presentation failure (#1), two prominence failures (#2, #3),
one missing experiment (#4) and one length complaint (#5). We believe #1, #2 and #3 are now done — and
done in the only way that is honest, by relocating and naming content that was already measured, rather
than by adding claims. #4 is open and we say so. #5 is 2% rather than 25% and we say so.

If the boundary now reads as the contribution, that is the change we were asked for. If it still reads as
*one FL screening rule is invalid*, then the packaging failed again and we would want to know which page
it failed on.
