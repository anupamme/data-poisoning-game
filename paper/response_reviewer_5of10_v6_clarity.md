# Response to the clarity review (overall 5/10, clarity 5/10, confidence 4/5)

Round 62. Every claim below is checked against the current `paper/main.tex` and the build it
produces, not against our notes.

---

## 0. The build this review read

The review reads **`main(20260910-105312).pdf`**. That file is about **16 hours older** than the
build the previous review (overall 6/10, clarity 6/10) read, and it predates **all of Round 61**.
The review also cites `20260905-090939`, `20260906-062822` and `20260906-074932`, which are older
still. None of those PDFs is on disk here, so we cannot diff against them; the dispositions below are
therefore stated against current source, with a line or page number for each so they can be checked
rather than taken on trust.

This matters for the reading, not just for the bookkeeping: **seven of the review's ten ranked
priorities (2, 3, 6, 7, 8, 9, 10) were already implemented in the build it did not see, and an
eighth — the hierarchy figure, ranked 🔴1 — was already drawn, but at a size no reader could
resolve.** Two are new work this round (🔴4) or declined with a reason (🟠5). We are not claiming the review is
wrong about the paper it read. We are saying where to look now.

---

## 1. The ten ranked priorities

| # | The ask | Disposition |
|---|---|---|
| 🔴 1 | Hierarchy figure with the two broken arrows, as Figure 1 | **Already the figure, and now legible.** Fig. 1(a) (`main.tex:79`, caption `:92`, p2) draws `upstream T → statistic S(d₂) → decision (P3) → admission (P4) → ASR (P5)`, two solid links, two dashed links each marked `⇏`, plus the struck attenuation arc. **But we found a defect the review could not: the on-panel type was illegible.** See §2 below. |
| 🔴 2 | Rewrite the first 1–2 pages as problem → failure → intervention → result | **Landed in Round 61, and §1 is now in the review's own §17 order**: question (`:102`), FL grounding (`:116`), two measured facts (`:118`), why the natural test fails (`:120`), the replacement plus the −0.272 / +0.098 headline (`:122`), no-trusted-coordinator (`:124`), contributions (`:130`). §1's causal-vocabulary token count is now **6**, down from 13. |
| 🔴 3 | A simple causal diagram | **Already Fig. 2** (`:180`, p3): (a) ordinary collider bias, (b) the gate making `G` deterministic in `S`, (c) the repair. The attenuation path the review asks about is in Fig. 1(a). |
| 🔴 4 | Reduce the terminology load | **Addressed this round (A4).** The review reads *test / gate / screen / criterion / protocol / instrument* as six contributions. They are five distinct objects, so the fix is disambiguation, not collapsing. `:162` (p3) now fixes all five in one sentence, where a reader first meets them. |
| 🟠 5 | State the estimand earlier | **Declined, with a reason.** τ(T) is at `:191` on p4. Moving it into §1 pushes Prop. 1 off p3, which our own gate forbids (`prop_page ≤ 3`). §1 already states the estimand in words at `:122`. |
| 🟠 6 | Present the Krum result as question / setup / result | **Already the structure.** §5 opens with the italic question (`:362`); Table 1 row 1 is the result row (Δagg 0.892, Δdec 0.733, Δadm 0.000, ΔΛₐ 0.000, ΔASR −0.026); the ±0.15 margin is in the text at `:374`. |
| 🟠 7 | Consolidate the "we don't claim" statements into one box | **Already `box:scope`** (`:227`, p5), titled *"What this paper establishes, and what it does not, stated once"*. This is the formulation the review asks be brought forward. |
| 🟡 8 | Move revision / audit / provenance history to the appendix | **Landed in Round 61**: nine demotions, each to the appendix line that already carried it verbatim. |
| 🟡 9 | Reduce the P1–P5 notation in prose | **12 `(P#)` tokens in the whole body**, against a gate that permits 20. Tightened to 12 this round so it cannot drift back up. |
| 🟡 10 | Shorten the contributions to three | **Already three**, and 2122 → 1295 chars. |

---

## 2. The defect we found ourselves, which neither review named

Figure 1(a) was drawing its boxes at **6.3 pt**, its group labels at 5.2 pt and its arrow annotations
at **4.6 pt** — and `main.tex` includes the figure at `0.80\linewidth`, so those print at roughly
**5.0 pt** and **3.7 pt** on the page. The review's top priority was "make this the figure"; the
figure was already there and could not be read.

The generator's own comments explained why it was stuck ("this panel has no free band left"), so
raising type alone would have collided — and on the first attempt it did: at 300 dpi the words
printed straight through the `decision` and `admission` boxes. Two things were wrong:

- the hardcoded arrow half-widths were about **0.6× the true box half-widths**, so raising the box
  font consumed exactly the gaps the labels needed. Geometry is now **derived** from measured
  half-widths with explicit unequal gaps, and an assertion checks the row still fits the panel.
- the words `not implied` and `by definition` cannot be made legible in that band at any size
  (0.125 wide against a 0.070 gap; a 0.168-tall label against a 0.076-tall band). So the **glyphs
  stay on the artwork and the words moved into the caption**, which already carried them.

Printed sizes now: chain boxes **6.6 pt** (was ~5.0), arrow numbers **7.2 pt** (was ~3.7), `⇏`
**8.8 pt**, group labels **6.4 pt** (was ~4.2). The three per-segment annotations are bare numbers
(`0.80`, `0/240`, `0.173`); every one is still **read from its artifact** by the generator, with its
assertions intact — no number was transcribed into the figure or the caption to make room.

`0.173` is printed **without a minus** on purpose. It is a difference of rounded rung means and is
positive (0.584 − 0.411); the word *falls* carries its direction, in the caption and at `:481`.

**And the pixel read of our own new build caught a second defect.** The caption said *"the solid
link holds by definition"* — singular — while the panel draws **two** solid links, matching
`(P1)⇒(P2)⇒(P3) by definition` at `:284`. Fixed to *"both solid links hold by definition (§3)"*. No
LaTeX check, no gate and no `pdftotext` extraction could see this; only reading the rendered page
against the generator's `broken = i >= 2` predicate could.

---

## 3. The other three edits

**Review §7 — "stop saying *descendant of the quantity being estimated*."** Agreed, at the two
places a reader meets it first: the **abstract** (`:66`) and **contribution (1)** (`:141`, p2, unchanged line) now
read *"selects on the very outcome to be explained."* The three appendix occurrences (`:709`, `:769`, `:801`) are formal statements and keep the formal phrasing. This is now gated:
the phrase cannot return to the abstract or §§1–2.

**Review §6 — "a plain-words explanation of why the sign can reverse could make the entire paper
click."** Agreed, and this was the highest-value line in the round. *Attenuation* was first explained
on **p6** while the sign reversal was asserted in the abstract and on p1. One sentence now sits with
the headline (`:122`, rendering on p2): *"The reason is mechanical: the upstream transform can weaken
the attack itself, so an outcome-gated comparison can credit the downstream defense with a fall in
ASR that attenuation upstream produced."* It lands on **p2 rather than p1**, which we flag rather
than round up.

**Review 🔴4 — terminology.** `:162`: *"The other words are fixed too: gate is the eligibility
condition below, criterion our own composability rule, protocol §4's four clauses, instruments Mode S
and score-only Krum."* Note this is a **different register** from §3's level glossary (statistic /
decision / admission / suppression, plus influence and attenuation, `:312`–`:326`) — design objects
versus levels — so the two do not compete. Also gated: all five terms must stay fixed in §§1–2.

---

## 4. Declines, each with its reason

- **The title.** The review proposes *"Statistic Preservation Is Not Security Preservation."* That is
  broader than Prop. 1's claim, which is about **identification** and holds within the rescaling
  class and defense family measured. `:436` and §5's own title already state the defensible version
  (*"downstream-statistic preservation is not sufficient to identify attack-suppression
  preservation"*), which is the formulation the review's §17 asks for. Keeping the title.
- **Cutting 20–30% of the prose (§16).** Declined. The body stays at 9 pages and every addition this
  round was funded 1:1 by compression at a named site. On this paper, cutting prose means cutting a
  disclosure, and the standing rule since Round 48 is that consolidation moves a claim's home and
  never removes one.
- **Reducing the negative qualifications (§13).** Declined as a target, but **partially answered as a
  style matter**: four `\emph{not}` stress italics were retired (`the design they license and not our
  screen`; `what the proposition does not constrain`; `it is what enters training and not the value of
  a scoring statistic`; `The table does not order the arms`) with the sentences intact, plus three
  more decorative stress italics. The denials themselves stay: negation density held at **86
  qualifying sentences** across the round (the ratio only moved 50.0% → 49.4% because two sentences
  were *added*, so nothing was softened). `box:scope`'s *"Not established. We do not establish…"* is
  deliberately kept.
- **The estimand on p2 (🟠5).** See the table above: our own `prop_page ≤ 3` gate forbids it.

## 5. Where the paper already answers the substantive concerns

- **§9, "the criticised methodology may be a straw man"** — answered verbatim at `:162`: *"We do not
  claim the gate is common in published evaluations: it is the gate our own criterion applies."*
- **§17, "claim non-sufficiency, not causation"** — already the paper's wording (`:436`, §5's title).
  A grep finds no "preservation causes" phrasing anywhere.
- **§15, "put a tagline on page 1"** — the abstract's bold first sentence *is* the tagline: *"A
  preserved statistic is not preserved suppression."*
- **The oracle concern** — the Conclusion's closing open question (`:512`) and the Ethics statement
  (`:514`) both state that both instruments read adversary identity and are measurement devices, not
  deployable defenses.
- **The ±0.15 margin concern** — Table A22 and `:374`: the binding arm's `m*` is 0.034, so every
  equivalence reading survives any margin above that, ±0.05 included.

The review's §8–§14 and §21 major concerns are **substantive rather than presentational**, and this
round does not attempt them beyond the pointers above. Its larger-N/K/seed and second-dataset asks
are compute, not editing.

**One scope note, stated rather than glossed.** This letter dispositions the review's ten-item
ranking in full and the sections we can attribute by number (§§5, 6, 7, 9, 13, 15, 16, 17, 21, 25).
We have not written a numbered row for every one of the 26 sections, because we cannot reproduce all
of their text from what is on disk here. Nothing in the ranking is left out.

---

## 6. Deviations from our own plan for this round, reported not buried

**§3's section heading was shortened.** It was 72 title characters and wrapped onto a second rendered
line, which costs a full line of the page budget for zero content. It is now *"Which Level a
Composition Preserves"* (35 chars, one line, `:259`). The dropped clause *"algebraically and before
any run"* is the section's own text and appears verbatim at `:187`, and the roadmap at `:153`
describes §3 the same way. **This is a retitling, and the round's own non-goals said no retitling**,
so we are flagging it rather than letting it pass.

**§5's heading still wraps and was left that way deliberately.** It is 72 title characters too, so
there is a second free line on the table — but its title *is* the paper's thesis, and the review's
§17 asks that exact formulation be prominent. We left the line unspent.

**One clause was compressed rather than moved** (`:185`): the sentence restating Prop. 1's own
conclusion now reads *"comparing the two occupied cells varies d₂ and so changes the conditioning
event instead of intervening"*, dropping a clause that Prop. 1 itself states at `:172`. Verified
one-home before cutting.

---

## 7. Verification for this round

- **Both gates exit 0.** `measure_body_chars --gate`: body prose ends by p9, Ethics is the first
  content of p10, residue before the heading `'E'` — under one rendered line of slack, and it held.
  `measure_clarity_load --gate`: 12 `(P#)`, abstract 1579/1600, longest abstract sentence 164/200,
  Prop. 1 on p3, bold 39, italic 62, max cross-refs 5, longest paragraph 842/850, §1 causal tokens
  **6** with none before the FL grounding, 0 comment-joined paragraphs.
- **Two ratchets tightened and two assertions added**, so this round's gains cannot be spent later:
  `p_tokens` 20 → 12, `causal_sec1` 8 → 6, the retired jargon gated at zero in the abstract and
  §§1–2, and all five design terms asserted present.
- **Fixpoint build**, main and supplement: 0 LaTeX errors, 0 undefined references, 0 undefined
  citations, 0 multiply-defined labels, 0 "Rerun to get", 0 overfull hbox, 0 overfull vbox.
  Main 67 pages, supplement 11, workshop 43.
- **Figure 1 read at 300 dpi, panel by panel**; panels (b) and (c) un-clipped and unchanged.
- **pp 1–10 read end to end in pixels.** This is what found the caption defect in §2.
- **No artifact was written.** The four frozen md5s hold: `dose_femnist/summary.json` `a16ef13f…`,
  `all_compositions/summary.json` `2f4d9920…`, `headline_seed_topup/summary.json` `a0717893…`,
  `comparability_six_cells.json` `2e8cced3…`.
- **The workshop's pinned figure is untouched**: `workshop_paper/figures/modeS_causal.pdf` still
  `76c29dcb…`.
- **Word-stated magnitudes re-divided against their artifacts** (they are the only quantities in this
  paper with no emitter): `3.7×` = 0.167/0.045 at `:357`; `2.4×` = 0.080930/0.033227 = 2.4356 at
  `:436`, the first from the ladder's `admission_top` and the second from six-cells' `d_influence`.
- **Only one figure PDF actually changed**: `paper/figures/modeS_causal.pdf`. `targeted_dose.pdf`,
  `targeted_modeS.pdf` and `targeted_modeA.pdf` show as modified in `git status` but are
  **byte-identical drawings** — the only differing field is `/CreationDate`. Reverting those three is
  safe; note that md5 alone cannot tell you that.
