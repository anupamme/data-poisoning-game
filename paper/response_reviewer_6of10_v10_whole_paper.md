# Response to the sixth review (header 5 — Borderline / Weak Reject; §15 scorecard Overall 6.5/10)

**What this round targets.** The review's header says 5, but its own §15 scorecard says **Technical 7,
Novelty 6, Empirical 7.5, Clarity 6.5, Significance 7, Reproducibility 8.5, Overall 6.5**, and §17 is
titled *"What would move me from 5 → 7/8"*. Nothing on that card is below 6, so we take **Novelty (6) and
Technical (7)** as the binding dimensions and spend the whole round on them. Clarity was Round 65's
subject and its gate is unchanged here.

**Which build the review read.** `main(3).pdf`, the Aug. 30 revision, **fourteen days stale**. So the
first section below is not a rebuttal but a list of check-sites: eight of the review's asks, including
three of §17's five, are already in the source, several of them landed after the PDF it read. That
decides most of the response, and everything in §2 is what remains.

**What the review told us not to do, and we did not.** *"I would not add another large benchmark suite. I
think that is the wrong direction now."* No new experiment, dataset, seed, `N`, `f` or attack was run.
The four frozen result artifacts hash unchanged (§7).

---

## 1. Already in the paper before this round — eight items, each with a check-site

| review's ask | state in current `paper/main.tex` |
|---|---|
| §17.2 **reframe the sign reversal; stop implying one estimand** | Landed almost verbatim at `:486`: the two designs are not the same intervention, $\Lambda_a$ moves in both, $0.081$ against $0.033$, so `:486` says "no $p$-value attaches to their difference". Restated at `:1747` |
| §17.3 **make inherited vs emergent first-class** | *emergent* is **bold-defined inside Prop. 2** at `:221`, witnessed with an interval at `:229` (FG→RFA, ASR $0.093$, CI $[0.062,\,0.123]$ at $n{=}30$), carried by contribution 1 at `:148` ("accepts a blind spot for emergent synergy") and closed in the abstract at `:66`. Ten body hits. **One gap only, fixed as E2 below**: the *pair* was never named as a dichotomy |
| §17.4 **make the contribution explicitly ML-specific** | The four-part paragraph the review drafts already exists in the appendix at `:2038` as a "combination of four things", and **the body already carries it** at `:213`–`:215`. This retired most of the planned edit; see E3 and §6 |
| #6 **the Krum admission floor weakens the headline** | Disclosed twice, unprompted: `:399` says "The zero is a floor here", and `:460` says "Neither reading rests on Krum" 's admission floor |
| #8 **Theorem 8 is over-weighted** | Thm. 8 is **appendix-only**, stated at `:1022`. The body carries no mechanism-preservation theorem, so there is nothing left in the body to demote |
| #10 **demote the screening result** | It is already §6.3, the last subsection before the Conclusion, at `:538`; its per-arm numbers moved to the appendix in Round 65 |
| #11 **FG→RFA is one of the most interesting results** | In the body with its interval, at `:229` |
| §17.5 **aim for ~9 pages** | The main text **is** 9 pages: `measure_body_chars --gate` reports body prose ending p9 with the Ethics heading the first content of p10 |

**And the review's #1 technical ask was a consistency defect, not a missing distinction.** §17.1 asks us
to split P4 into `P4_support` and `P4_influence`, calling it *"the most important technical edit."* The
paper already draws that line nearly everywhere:

- `:330` glosses admission as "whether adversarial input enters its aggregate at all"
- `:825`, the appendix twin of the definition, has it as "for whether adversarial input survives"
- `:1701` calls it "the \emph{support} of the admitted adversarial mass", explicitly distinct from the mass
- Table 1's caption at `:417` says the two columns are "never interchangeable"
- `:460` already *depends* on the distinction

**Four lines contradicted them**, and
the witness the review asks for is already in Table 1: `:424` (reputation) and `:426` (coord.\ median) are
both $\Delta$ adm.\ $=0.000$ with $\Delta\Lambda_a$ at $0.020$ and $0.033$ — preserved support, moved
influence, in the paper's own rows. So we fixed the four lines rather than renumbering (§5, decline 1).

---

## 2. The edits this round

| # | edit | site | check |
|---|---|---|---|
| **E1** | **P4 consistency: admission is support, influence is mass.** The definition now reads "the support of the admitted mass rather than the mass itself", and adds the implication the review wants stated: the mass is influence $\Lambda_a$, which a preserved support leaves free to move, pointed at Table 1 | `:308` | Grep of the body window for `how much adversarial input` and `the mass it admits` returns **0** each; `:308` now agrees with `:330`, `:825` and `:1701` |
| **E1b** | **The same defect on page 1, in three more places.** The abstract now reads "while it admits no adversary at all"; the intro's second measured fact reads "while its admission is unchanged"; Figure 1's caption reads "of rounds while admission does not", which is also what panel (b)'s own legend prints (*admission change: never*) | `:66`, `:125`, `:99` | All three confirmed in a 300-dpi read of pp 1–2, at rendered lines 025, 046 and 076 |
| **E2** | **The dichotomy is named where it is defined.** Prop. 2 defined *emergent* and left its complement unnamed. It now writes $\mathcal{E} \subseteq \mathcal{S}$ for the emergent ones and $\mathcal{S} \setminus \mathcal{E}$ "for the inherited ones" — the inequality that defines it was already in that sentence | `:221`, `:954` | Both instances of the proposition edited together, so the body statement and its appendix twin stay identical. No bold or italic run added (both at cap) |
| **E3** | **The ML-specificity paragraph ends on its consequence, not its hedge.** `:131` used to end on the scope condition; it now ends "though the seam, rather than the scale, is what the problem is about". The scope condition is kept, in the same sentence | `:131` | +8 chars. **Reduced from the planned edit** because `:213`–`:215` already carried the argument; see §6 |
| **E4** | **The 95% intervals are back beside the equivalence claims, and one of them was reported too loosely.** `:397` restores $\Delta{=}-0.010$, 95% CI $[-0.032,+0.012]$. `:532` previously said each $|\Delta|$ was inside the frozen $\pm0.15$ margin, which is true of the point estimates and **not** of both intervals; it now prints $-0.017$ $[-0.033,-0.001]$ and $-0.041$ $[-0.183,+0.101]$ and says "Neither $95\%$ interval is a plain null" | `:397`, `:532` | Both render without an overfull line (300 dpi, pp 7 and 9). Every interval re-derived from `tab:tost` / `analyze_margin_sensitivity.py`, matched on the (mean, $n$) pair |

**Why E4 is the round's largest rigor gain, and it is the paper's own convention that forces it.**
`supplementary.tex:441` **defines** "no evidence of a practically meaningful change" by the **95%**
interval inside $\pm m$, not by TOST's 90%. The two disagree at small $n$, and both of `:532`'s arms are
the disagreement: EMNIST's 95% interval sits inside the margin but **excludes zero** (a real effect
smaller than the margin), and ResNet18's is **wider than the margin itself** ($m^{\ast}=0.183$ against a
frozen $0.15$). That asymmetry is exactly why the licensed phrase is granted to one arm and withheld from
the other, and until this round it was invisible to a body reader. **No equivalence phrase is newly
licensed**: ResNet18's stays withheld, EMNIST's stays granted.

---

## 3. §17.5's "cut 25–35% of defensive prose" — taken in grammar, with a floor that proves nothing was traded

The main text is 9 pages and cutting content costs disclosures, so the cut is taken in **voice**, not in
content: *"no test that admits … can identify"* → *"a test that admits … cannot identify"*, and so on,
across twelve sentences. The problem is that a positive-voice rewrite and a **deleted caveat** move the
one number we had in exactly the same direction, so the measure could not license the phase.

`experiments/measure_negation_density.py` therefore gained a **second measure**: a frozen 16-entry
`PROTECTED` inventory of the disclosures the paper may not lose, counted as a **floor** while the density
above it may fall freely. Every pattern anchors on the disclosure's **referent, never its negation** —
anchoring on "not" would make the floor fire on precisely the rewrites it exists to permit, which inverts
the measure. Both branches were falsification-probed in a scratch copy before the phase ran: deleting one
disclosure exits 1 with the offending name printed; rewriting one into positive voice exits 0.

```
paper/main.tex: 83/174 body sentences carry a negation = 47.7%
  protected disclaimers: 16/16 present (15 in the body window)
```

Density fell **51.1% → 47.7%** (90/176 → 83/174) with the floor at **16/16** throughout. The one item
outside the body window is the abstract's recall cap, which is front matter by construction, not a
relocation. Twelve consolidations funded E1–E4 and each had its destination verified by grep before the
body sentence was cut — for instance the phrase cut from `:567` has its home in
`:343`'s "not a security metric a deployed defense could compute".
Body prose is **39504** chars
against Round 65's 39758: the round is net shorter *and* added two intervals.

---

## 4. Every gate, after the edits

| gate | result |
|---|---|
| `measure_body_chars --gate` | `[OK]`, body prose ends p9, residue before the Ethics heading `'E'` |
| `measure_clarity_load --gate` | `[OK]`, exit 0. **No ratchet raised**: `(P#)` **12** (E1 added no token), C0–C3 3, bold 39, italic 62, xrefs/para 5, longest paragraph 846, comment-joined paragraphs 0, first proposition p3, all five of the review's questions p1 |
| `measure_negation_density` | 47.7%, floor 16/16, exit 0 |
| both builds | fixpoint; 0 LaTeX errors, 0 undefined refs/citations, 0 multiply-defined, 0 `??`, 0 "Rerun to get", 0 overfull hbox/vbox in **both** logs (grepped with `grep -a`) |
| pages | main 70, supplement 11 — unchanged |
| punctuation / anonymity | 0 U+2014/2013/2212 in all three sources; no unprotected `\texttt{}` join; `pdfinfo` Title/Author/Subject/Keywords empty; **0** rendered "Round N" in the extracted text of both PDFs |

---

## 5. Five items declined, each with a machine reason rather than a preference

1. **§17.1 as literally stated — renumber P4 into P4a/P4b.** Declined; the distinction is delivered in
   full by E1 instead. `figures/modeS_causal.pdf` bakes `P3 P4 P5` and `figures/story_chain.pdf` bakes
   `P1`–`P5` into rendered pixels; both generators also write into the pinned companion document, which is
   out of scope; `(P#)` is a ratchet at 12/12; and `:825` states that the committed pre-registrations say
   *admission*, so no scored quantity may be relabelled. A retired label inside a shared figure PDF passes
   every LaTeX check and every gate, which is the failure mode being avoided.
2. **#9 narrow empirical scope / another benchmark suite.** Declined, and the review declines it first.
3. **#8 cut Theorem 8's weight further.** Nothing to cut: it is appendix-only at `:1022`.
4. **#6 restate the headline around a non-floor defense.** Declined as a restructure, delivered as
   disclosure, which was already in place at `:399` and `:460` before the review. The four-kind claim is
   taken across all aggregators, not from Krum.
5. **A 25–35% cut measured in pages.** Declined as stated — the main text is 9 pages — and delivered as
   §3's grammar cut, with the disclaimer floor as the evidence that no disclosure paid for it.

---

## 6. Deviations from the plan, reported rather than buried

- **E3 shrank from a rewrite to +8 chars.** The plan budgeted a compression of the appendix's four-part
  ML-specificity argument into the body. Mid-execution, `:213`–`:215` turned out to already carry it, so
  the edit reduced to giving `:131` its consequence. **This is also the answer to review ask #4**: the
  paragraph the review drafts exists twice already, and what it lacked was a closing clause.
- **Two edits were added that the plan did not contain, both found by the 300-dpi read.** `:125` and
  `:99` carried the magnitude reading of admission on **pages 1 and 2**, ahead of any definition. Neither
  was a contradiction — *admitted adversarial mass* is a defined term whose referent is the entry-level
  quantity, fixed at `:1082` — but both read as magnitude to a first-time reader, which is the review's
  complaint. Fixed at −5 chars and 0 rendered lines.
- **One `\emph{}` was added to the Figure 1 caption and immediately reverted.** Italic runs are at 62/62,
  a ratchet, and the plain form reads the same.
- **The planned E1 edit at `:286` no longer exists.** §3's five-sense enumeration was consolidated in the
  same round into the P1–P5 definition list, where `:308` and `:330` carry the support reading. The
  enumeration's home was verified before the cut; the letter cites `:308`, not `:286`, for that reason.
- **A cut at `:547` was made and then restored.** Shortening it to a bare cross-reference removed the
  body's only C2 token — caught by the clarity gate's own C0–C3 line, not by eye. The full qualifier is a
  disclosure and there was headroom, so `:547`'s "whose footnotes qualify both the rates and the C2 column"
  went back verbatim.
- **The page gate failed twice before it passed.** Additions inside `itemize` and `proposition`
  environments cost more rendered lines per character than plain prose, so a body-char count only 5 over
  Round 65's passing value still spilled 25 words onto p10. Four redundancy consolidations fixed it; no
  rephrasing did.

---

## 7. What did not change

No section was reordered, no float promoted from the appendix, no box or figure added, and no figure
regenerated. The paper title, §5, P1–P5, C0–C3, Mode S/A/M and the words *admission*, *influence* and
*attenuation* keep their names. **No scope condition, limitation, withdrawal or disclosure was removed** —
§3's consolidations moved numbers into destinations verified present by grep, and the disclaimer floor is
the standing check on that. Neither equivalence convention was touched. The four frozen artifacts hash
unchanged: `2f4d9920` (`results/all_compositions/summary.json`), `a16ef13f`
(`results/dose_femnist/summary.json`), `a0717893` (`results/headline_seed_topup/summary.json`) and
`6e25ef3a` (`results/comparability_six_cells.json`); the margin ladder's binding arm is still
`krum / scaling`, EMNIST at $m^{\ast}=0.0343$, with `BINDING ARM MOVED` not printing. No `results/`
directory was written and no companion-document file was edited.

**What this round does not settle.** Novelty was the binding score, and E1–E3 are its whole treatment:
they sharpen and surface a claim the paper already makes, and they add no result. If the novelty objection
survives them, the remaining levers are a new theoretical result or a new empirical class — both out of
scope here, and we would rather say so than imply the dimension is fully answered. The review's own
instruction was that another benchmark suite is the wrong direction; we have taken it at its word.

---

*Every `` `:NNN` `` in this letter is a source line of `paper/main.tex` as of this round, resolved by
grepping current content rather than by shifting an earlier letter's numbers, and supplement cites are
prefixed `supplementary.tex:NNN`. Checked by
`python3 paper/audit_letter_cites.py paper/response_reviewer_6of10_v10_whole_paper.md`, which reports 61
cites and 25 quotes with **FAIL: 0**. Six quotes are listed as unchecked by construction: three are
quotations of the review, two are the before/after of a rewrite pattern rather than a quotation of any
line, and one is a LaTeX log string. **No quotation of the paper is in that blind spot**, which is why the
four that had drifted onto a wrapped line were moved back beside their cite before this was run.*
