# Response to the twenty-fifth review (score review, 7/10, and clarity/writing review)

Thank you both. The two reviews converge on one diagnosis — the contribution is there and the prose
buries it — and we have treated that as the whole of this revision. Neither review asks for compute and
neither gets any: every change below is a relocation, an enumeration, or a rewrite, and no number,
figure, artifact or pre-registration changed.

---

## 0. Which build each review read

| | file | timestamp | pages |
|---|---|---|---|
| what both reviews read | `data_poisoning_fl.pdf` | 25 Sep 16:02 | 82 |
| what is shipped now | `paper/main.pdf` + `paper/supplementary.pdf` | 26 Sep 18:24 | **70 + 39** |

Both reviews quote the retired *mechanism preservation* title, which places them on the same stale build
the previous round's reviewer read. That matters for roughly a third of the two lists: those asks are
already answered in the current build and are **cited** below rather than redone. We separate them
honestly — §1 is new work, §2 is work the reviews could not have seen.

---

## 1. New in this revision

### 1.1 An explicit contribution hierarchy (score §17.1, clarity #4)

This was the one unambiguous gap. The Contributions paragraph previously enumerated nothing and spent two
of its four sentences saying what the claim is *not*. It is now a ranked list that makes the
primary/secondary split unmissable, at
`:367` (rendered p2), where the impossibility result is *"First, and primarily"*, the intervention design is second, and the criterion is *"Third, and subordinate"*.

The three items are italicized rather than bolded for a measured reason: `bold_runs` sits at 38 against a
hard ratchet of 39, so three new `\textbf{}` would have failed a gate, while `emph_runs` had the room.
The *priced rather than validated* clause and the pointer to the eight-objection appendix are kept.

### 1.2 Figure 1 is now the picture the clarity review sketched (score §13, clarity #15)

Clarity §15 asks for one figure showing the statistic path on top and the bypass from the upstream
transform to adversarial influence beneath. **The paper already contained that drawing, in the appendix.**
It is now Figure 1 on p2, at `:326`–`:327`, and the composite that was Figure 1 has taken its appendix
slot (App. §I, rendered p69).

Nothing was redrawn, regenerated or cut: both PDFs are byte-identical artwork, the swap is a swap and not
an addition, and the graphics are a wash by measurement (407×135pt at `0.98\linewidth` against 431×175pt
at `0.80\linewidth`, both ~129pt). What made it affordable was cutting the promoted caption from 1437 to
707 characters; the demoted caption lost exactly one clause, its opener, which stopped being true the
moment it stopped being the paper's one figure. Both of its disclosures — the per-row *n* and the four
rows that are training data for a withdrawn rule — are kept verbatim, and the appendix section that now
holds two figures was retitled accordingly.

### 1.3 Proved / supported / not established, now on p3 (score §17.3)

The scope box with its three legs existed but rendered on p5. It has moved to `:534`, immediately after
§2's vocabulary paragraph, and now renders on **p3** — before the first proposition. Its three run-in
heads are unchanged.

The move also fixes a layout defect the user reported independently: §3's heading carried 63.0pt of space
above and 44.3pt below against 27.0/24.5 everywhere else. The cause turned out not to be the heading or
the box at all but an atomic centred tabular nine rendered lines lower, which stood ~99pt against ~90pt
of remaining room, jumped whole to the next page, and left 126.3pt that `\flushbottom` redistributed into
the section skips. A `\raggedbottom` control build printed 27.5/24.9 for the same heading, which settled
it. The fix is 10pt of table height (`arraystretch` 0.9 and 2pt more negative `\vspace` at each end);
all six rows and the rule are unchanged, and the heading now measures 28.4/25.4.

### 1.4 Defensive material moved to the appendix (score §17.5, clarity #5)

The collider-credit paragraph has left the body for the testability-boundary appendix, now at `:1906`
(rendered p19), beside the paragraph that already disclaims depth for the elementary converse. It is a
pure relocation: the sentence is byte-identical, including the clause our own instrument protects, and
the positivity-versus-confounding distinction it carried keeps a body home in Figure 1's caption.

We want to be precise about what *did not* happen here: we did not compress the limitations audit away.
Every scope condition, limitation and withdrawal still exists, and a disclosure may move between the two
documents but may never lose its last home. That is enforced by a floor of 16 protected disclaimers,
which reads **16/16** after this round.

### 1.5 The criterion is qualified on first mention (score §17.4)

`:531` is the first place in the body the word appears, and the qualified term now **leads** it: the
narrative term is the *evaluation-prioritization criterion* — the score review's own phrase — with
*Screen* introduced after it as the shorthand the formal statements keep. The disclaimer that it is
*never a predictor of security* sits in the same sentence, as before.

### 1.6 The abstract is reordered (clarity #1, their highest-return ask)

We adopted the reviewer's **ordering** and not their text, at `:151`: problem, the natural test, why it
cannot answer, the design that can, the two numbers, the hierarchy and the theory, scope and denial. The
single structural move is the bold thesis, which stood third — asserting the conclusion before the test
it refutes had been stated — and now closes the diagnosis. Two negations went positive with it. Every
number and all three disclosures are untouched, and both −0.273 and +0.125 remain in the abstract, which
is what keeps them on the same rendered page.

Why not the reviewer's verbatim abstract: see §3.2.

### 1.7 The positive-voice pass (score §12 *too defensive*, clarity #13)

*Too defensive* has a number in this repository, so we measured it per section, rewrote worst-first, and
deleted nothing. The instrument's protected patterns are anchored on each disclosure's **referent**
rather than on its negation, which is exactly what makes a positive-voice rewrite pass and a deletion
fail.

| section | before | after |
|---|---|---|
| §7 Conclusion | 6/6 = **100%** | 2/6 = 33.3% |
| §4 A Causal Protocol | 7/8 = **87.5%** | 1/8 = **12.5%** |
| §6 What the Boundary Licenses | 11/19 = 57.9% | 2/21 = **9.5%** |
| §5 Statistic Preservation Does Not Identify | 27/48 = 56.2% | 18/51 = 35.3% |
| §3 Which Level a Composition Preserves | 9/17 = 52.9% | 0/20 = **0.0%** |
| §1 Introduction | 14/32 = 43.8% | 7/33 = 21.2% |
| §2 What the Gated Test Cannot Identify | 27/82 = 32.9% | 25/79 = 31.6% |
| **whole body** | **101/212 = 47.6%** | **55/218 = 25.2%** |

The Conclusion was six sentences and six denials, which is the review's complaint stated as a
measurement. It is now two paragraphs that open positively, at `:1456` and `:1492`. Four of its six
sentences were rewritten; the two that stayed negative stayed for reasons that are not style, and we
would rather say so than claim a clean sweep:

- the *do not / do not / do* checklist is an imperative an evaluator acts on, and a positive-voice
  imperative would invert the advice;
- the open problem's wording is one of three phrases an instrument allowlists verbatim, so rewording it
  fails a gate set at zero.

§2 was left near its starting density deliberately. It is the section that states the impossibility
result, its negations are the content, and at 31.6% it was already the *least* defensive section in the
paper before this round.

One rewrite we considered and rejected: turning *does not reproduce* into *fails to reproduce* would have
moved the measured number — `fails` is not a negation token — while making the sentence read more
negative. That is gaming the instrument, and the sentence stands as it was.

### 1.8 Long sentences (clarity #8, §12's 25–30 word target)

Eight body sentences ran over 55 words; the longest was 79. All eight are split. **Over-55 is now zero
and the longest body sentence is 54 words.**

| | before | after |
|---|---|---|
| sentences | 212 | 218 |
| mean words | 22.5 | **21.8** |
| median | 21 | 21 |
| over 40 words | 21 | 18 |
| over 55 words | 8 | **0** |
| longest | 79 | **54** |

We should be straight about one thing: the body mean was **already under** the reviewer's own 25–30 word
target before this round. So the honest claim is not *we shortened the prose* — it is that the
distribution's tail, which is what a reader actually trips on, is gone.

### 1.9 The ladder in plain words, where the family is introduced (clarity #6, #7)

`:772` used to say only *how many* of the five senses fall on each side of the internal/security cut, so
a reader first met the ladder's plain words in a tabular seventy lines below. It now names them in order
on the spot, in the table's own row order and the paper's own terms: statistic, ordering, decision,
admission, then attack suppression. This adds no `(P#)` token, which matters because that count is a
ceiling of 7 and fully spent.

Per the user's steer we did **not** rename the rungs. See §3.3.

---

## 2. Already in the paper, on the build neither review saw

| ask | where it already is |
|---|---|
| score §17.2 / clarity #3: put −0.273 vs +0.125 on page 1 | the abstract, `:151`; rendered p1, ICLR lines 021–023 |
| clarity #2: the first page should tell one story | all five of the clarity review's own questions are answered by **p1**; the first proposition is on p3 |
| clarity #7/#11: C0–C3 in plain English, and not in the first read | glossed in words at `:834`, which says where each sits on the ladder without restating a definition; the definitions live in the composability appendix, and the body carries **7** C-tokens (C0:2 C1:2 C2:1 C3:2) |
| score §6: the criterion is not validated, say so | `:2916` states it as limitation (L2), with R²=0.239, the refuted cross-arm ordering and 40% recall |
| score §7: caught between two papers | the scope box at `:534` is exactly that separation, and it is now on p3 |
| reproducibility of every pre-registered arm | 38 pre-registration documents, `:1566`, checkable as `ls experiments/pre_registration_*.md \| wc -l` |

---

## 3. Declined, with the measured reason

### 3.1 The title stays

The two reviews conflict, and we are not able to satisfy both. The score review (§13/§14) asks us to
change the title; the clarity review's own Option C is, almost word for word, the title the paper already
carries — and the *previous* round's reviewer wrote *Keep it*. We kept it. The retired noun this round's
reviewers quote occurred **zero** times in the body even before the retitling, which is the drift the
current title closed.

### 3.2 Not the reviewer's abstract verbatim

Two reasons, both checkable. It is **1859 characters** against our own 1600-character cap (we are at
1588). And it drops three disclosures ours carries: the CIFAR-100 composition-level replication that was
attempted and **did not carry**, the *Against ourselves* −0.172 magnitude control, and the oracle-free
+0.322. The third is not a style choice — it is mandated by a frozen pre-registration whose decision row
names the abstract as a reporting site. We took the ordering, which was the substance of the ask.

### 3.3 Not *Statistic → Mechanism → Suppression*

The middle term revives *mechanism preservation*, the exact phrase this round's retitling removed, and
*admission* is a pre-registered term named in frozen decision rules. Renaming it in the paper would put
the prose out of step with the artifacts. The ladder is stated in the paper's own words instead (§1.9).

### 3.4 Not a 30–40% cut in the `(P#)` / C0–C3 labels

We ran the census rather than estimating it: the body carries **7** `(P#)` tokens and **7** C-tokens.
Every one of the seven `(P#)` occurrences is either a definition or the unique identifier of a claim a
later section reports on, so a 30% cut deletes defining mentions rather than clutter. The labels stay;
the plain-word ladder in §1.9 is what answers the underlying ask.

### 3.5 No new experiments, and an honest ceiling

Neither review asks for compute, and we think that is right: the gap both identify is framing. But we
will not claim this revision raises the empirical ceiling. Generality is still capped at three datasets
and three seed counts. Moving it needs either wave 2's 24 held-out CIFAR-100 pairs (~100 h serial) or a
positive oracle-free result — and our own pre-registered oracle-free arm **refuted** us at n=20, which
the paper reports as a refutation in its own claims table. We think 8/10 is reachable on the strength of
the framing fixes above. We do not think 9/10 is, and we would rather say so than imply otherwise.

---

## 4. Instrument change, disclosed

One measurement script changed: `experiments/measure_clarity_load.py`. Its Figure 1 caption ratchet was
hard-wired to the composite's label, so after the swap it was measuring an appendix caption. It is now
keyed to the promoted figure and re-baselined **722 → 707** characters. Ratchets in this file re-baseline
downward only, and the rationale is recorded in the source beside the earlier entries. No gate was
weakened and no cap was raised this round.

## 5. Verification run for this revision

- Both documents built to a **fixpoint** (extracted text identical across iterations 2–4). 70 + 39 pages,
  unchanged. Only the two permitted overfull boxes (3.77756pt) and the one permitted font warning
  survive; **0** undefined references and **0** undefined citations in either document.
- Page gate: body prose ends by p9, with the Ethics statement opening **on p9**. The positive-voice pass
  cost 14 rendered lines at one point and pushed the Conclusion heading to p10; that is what the appendix
  relocation in §1.4 and five tightenings of our own new wording paid for.
- Clarity gate: exit 0, **no ratchet raised** — `(P#)` 7/7, C-tokens 7/8, abstract 1588/1600, longest
  abstract sentence 168/200, bold 38/39, italic 58/60, cross-refs/paragraph 5/5, §1 causal tokens 6/6,
  longest paragraph 842/850, design vocabulary 5/5, Figure 1 caption 707/707, five questions by p1.
- Negation floor: **16/16** protected disclaimers present.
- Self-containment: every body pointer into the appendix resolves and attaches to a stated claim; the
  appendix index covers every top-level section exactly once, in document order.
- Statement inventory unchanged: 9 propositions, 1 theorem, 2 lemmas, 5 corollaries — **none added and
  none removed.** Pre-registration self-count 38 = the glob.
- All three starred statements re-read against this round's moves, because a paragraph leaving the body is
  the shape that has falsified the Reproducibility statement before. The three propositions it names as
  stated-in-the-body are still in the body; the two it names as appendix-only are still there.
- Anonymity on **extracted** text: 0 rendered round numbers, 0 review ordinals, 0 occurrences of
  *reviewer* in either PDF; document metadata empty. Punctuation census unchanged.
- Pixel reads at 150 dpi of every page whose content moved: pp. 1, 3, 5, 9 and the appendix page the
  demoted figure lands on (it sits with the section that discusses it, not deferred).
