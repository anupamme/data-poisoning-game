# Response to two reviews: the score review and the clarity review

Two reviews arrived together and we answer them in one letter, because their top-ranked asks turned out
to be the same asks. Every claim below is written from the file state of this build, not from our plan for
it, and every source cite is a line number in `paper/main.tex` (bare `:NNN`) or in
`paper/supplementary.tex`. Reviewer wording is *italicized*; double quotes are reserved for text that is
verbatim in the paper at the line cited beside it.

## 0. Which build each review read, because it decides which asks were already answered

| | build | main | supplement |
|---|---|---|---|
| Score review | 25 Sep, 16:02 | | |
| Clarity review | `main(20260926-045115).pdf`, 26 Sep 04:51 | | |
| Nearest full record we kept of that state | 26 Sep 10:18 | 78 pp | 30 pp |
| **This build** | 26 Sep, 16:24 | **70 pp** | **39 pp** |

The body is 9 rendered pages in this build, against the 10-page limit; `measure_body_chars --gate`
reports the Ethics heading as the first content of page 10, which the limit permits.

The score review still quotes our previous title, so it read a build from before the retitle. That is not
a complaint about the review: it is the reason five of its asks are answered below by pointing at text it
could not have seen, and we would rather say so than accept credit for work we did this round.

## 1. Asks that were already in the paper

These are cited so they can be checked rather than taken on trust.

**The title.** The score review proposed narrowing it, and gave wording. We had already adopted that wording byte for byte: "Statistic Preservation Does Not Identify" opens the title at `:62`, and the sentence it heads runs on to attack-suppression preservation in composed federated-learning defenses. The clarity review, reading the later build, said of it only: keep it. We have.

**Not a deployable test.** The score review's concern 5 was that we not sell a careful negative as having solved composed-defense evaluation. The abstract already closed on the denial: "This is identification, deliberately scoped to committed attacks and controlled interventions, and not a deployable security test or a predictor of adaptive robustness." (`:138`).

**The two headline numbers on page 1.** Both reviews ranked this first. It was already in the abstract: "On one \texttt{coord\_median} cell the outcome-gated design moves $-0.273$ where the within-defense one moves $+0.125$" (`:138`), and the sentence following it there argues against us, giving the magnitude-held-fixed control under which the two agree in sign.

**No revision-history framing in the body.** The score review's §13 asked for this and the clarity review
ranked it fourth. The clarity instrument's `provenance` metric reads **0** research-history framing
phrases in the body window, and its `body_homes` floor separately checks that each fact that framing used
to carry still has a home in the body: 2 of 2, at `:674` and `:1142`. The framing is gone; the
disclosures it used to carry are not.

**No claim of broad replication.** The score review's concern 4. The only scope words of that kind in the
body are in the claims table's own scope column and in the clause that limits every magnitude to the
defenses and attacks we run.

**Label mass in the two families the clarity review named.** The body contains **7** `(P#)` tokens
(P1, P2, P3 and P5 once each, P4 three times) and **6** `C0-C3` tokens, and **0** occurrences of Mode A.
Both counts were already at or under the caps the clarity instrument enforces before this round began.
Section 3 below is honest about what that means for the review's numeric target.

## 2. New in this build

**The pipeline diagram the clarity review asked for is Figure 1(a), and its caption now says so.** The
review asked for one figure showing upstream transform, statistic, decision, admission, ASR. That panel
already drew exactly those five nodes in exactly that order, with two grouping bands over them, and the
review did not recognise it, which is a caption failure rather than a drawing failure. The caption used to
open by calling the panel a map of the paper's vocabulary at the levels of Definition 1. It now opens with
the question it answers: "What a preservation check verifies, left of the divide, against what security depends on, right of it: transform, statistic, decision, admission, ASR." (`:349`). The caption is
**shorter** than before, 722 characters against a previous ceiling of 735, and the instrument's ceiling is
re-baselined down to 722 so it cannot grow back. Both disclosures the caption carries, the three scopes
and the four training-data rows, are still in it.

**One sentence saying plainly that the intervention is an instrument and not a defense.** This was the
score review's concern 2 and the clarity review's fourth-ranked ask, and they are the same ask. It is now
bold, on page 2, directly under the two headline numbers: "Both instruments are identification devices, not deployable defenses or diagnostics" (`:231`), and the same sentence continues with the reason, that they
read adversary identity, which no server has, and with the two pre-registered oracle-free arms that
refuted us at $n{=}20$. This spent the round's one available bold slot; the instrument caps bold runs at
39 and the body now stands at 38.

**The anticipated-objections section is now reachable.** The supplement has carried a written objections
section for several rounds, and a grep for its label returned only the label itself: nothing in either
document pointed at it. LaTeX warns about a reference with no label and never about a label with no
reference, so no build, gate or hash could see it. The Contributions paragraph now ends with a pointer: "supp.\ §\ref{app:objections} answers the objections we expect" (`:366`). It renders as supplement §S12 on
supplement page 10. This one clause is what turns the score review's list of expected reviewer questions
from unanswered into answered, because three of its five were already answered there.

**Three objections added, for the score review's three questions that were not already answered.** Each
is a pointer plus its scope, per that section's own stated rule, and none is a new claim.

- *Why not standard causal theory, since conditioning on a descendant of the outcome is a textbook fact?*
  Objection 6 opens on the objection itself: "The identification failure is a textbook fact, conditioning on a descendant of the outcome." (`supplementary.tex:531`). The answer scores the textbook layer as the
  smallest of three, names the claims table's own label for it, and says what is not textbook: the estimand
  for a composed defense, the instantiation on this specific screening move, and the measurement in which
  the two levels disagree in sign.
- *Adaptive attackers.* Objection 7 opens on it too: "The adversary is held fixed while the defense changes." (`supplementary.tex:547`). The answer concedes that nothing here bounds an adaptive adversary,
  points at the one arm that moves in that direction, and then argues against that arm: its winning
  adversary beats the backdoor it replaces by $+0.035$ at one seed, a margin rather than a rout, and it
  adapts to the downstream defense only, never anticipating the upstream transform.
- *Is admission a predictor?* Objection 8: "If admission is the level that matters, is it a predictor?" (`supplementary.tex:560`). The answer is no, reported as a refutation of our own earlier reading. The
  pooled dose-response test returned $\rho_s = +0.351$ at one-sided $p = 0.091$ over 16 cells against a
  frozen criterion of $p < 0.05$, and the prospective four-cell pattern was refuted on its held-out cell
  and withdrawn under its own frozen clause.

**The section's own self-count was wrong and is fixed.** It said five objections were expected while
listing eight. It now reads "Eight objections we expect" (`supplementary.tex:469`). No gate, build or hash
can see a claim a document makes about its own shape, which is why we grepped both documents for any
second emitter of that count rather than trusting the one we found.

**The screening criterion is demoted by relocation, not by rewording.** The score review's concern 3 said
the 42-pair criterion is now a liability and should be demoted to a consequence, and its §13 asked for
revision-history prose to leave the main paper. Those are one edit, and we made it: the section "A subordinate boundary study: what the screen buys, and what it does not" (`supplementary.tex:1809`) and
its five subsections are now in the supplement, rendering as §S24 on supplement pages 30 to 39. That is
the withdrawn-criterion, frontier and metric-swap material. It carries the development and held-out menus,
the prospective suite, the metric-swap ablation, the CIFAR-100 composition menu and the out-of-sample
battery, and every one of those is still listed by name from the body's own index of what moved, which
opens "Nine blocks are in the supplement rather than here" (`:1624`).

Nothing was deleted to achieve this. Main lost 8 rendered pages and the supplement gained 9; the pair is
one page longer than it was, because the three new objections are net new text. Across the relocation we
checked the pre-registration commit hashes as a set rather than as a count: 61 occurrences of 38 distinct
hashes across the pair, and the set is identical before and after the move, none lost and none introduced.
We also checked that the two-sided-equivalence audit's census did not change, and that the withdrawal,
frontier and metric-swap disclosure counts did not fall for the pair; the first two rose, because the new
objections add mentions. A disclosure may move between the two documents; it may not lose its last home.

## 3. Asks we did not deliver, or delivered short, and why

**The clarity review's 30 to 40 percent cut of the P1-P5 and C0-C3 terminology: not delivered.** On those
two families the reduction this round is zero. The reason is arithmetic rather than reluctance. P1 through
P5 are five defined labels occurring 7 times in the body, which is each label's defining mention plus two
reuses of P4; C0 through C3 occur 6 times, twice each for the three of them that appear at all. A 30
percent cut of 13 tokens means deleting two or three mentions, and every candidate is either a label's
only appearance or a cross-reference a reader following the argument needs. Renaming is out: these labels
are fixed by an earlier round's standing decision, and four of the design words are literal floor tokens
in the clarity instrument.

What we did instead is cut label mass where it was genuinely redundant, and we publish the census rather
than describing it. Body window, comment lines stripped, before this round's prose work and now:

| token | before | now |
|---|---|---|
| `Mode S` | 13 | 10 |
| `attenuat` | 11 | 10 |
| `estimand` | 8 | 7 |
| `identif` | 32 | 33 |
| `\Lambda_a` | 11 | 11 |
| `invarian` | 12 | 12 |
| `emergent` | 12 | 12 |
| `(P#)` | 7 | 7 |
| `C0-C3` | 6 | 6 |
| **total** | **112** | **108** |

That is a 3.6 percent reduction, not 30. We would rather report 3.6 than claim 30 and have a reviewer
count. The one increase is deliberate: `identif` gained an occurrence in the new bold sentence whose whole
job is to say that the instruments are identification devices.

**Question-driven section openings: not delivered.** The clarity review ranked this last of its seven
asks, and it is the one that costs rendered lines for no new content. The body is 9 pages against a hard
10, and the measured reserve after this round's relocation and compression is on the order of a single
rendered line. Three questions do already open the argument, at `:147`, `:161` and in the final section at
`:1415`. We funded the review's higher-ranked asks with the space we had.

**Figure 1 still renders above the numbers it quantifies, and we are disclosing that rather than claiming
otherwise.** Page 1 ends on the sentence promising that the design choice reverses a sign on one cell, and
page 2 opens with Figure 1 and its six-line caption before the table holding the two numbers. We tried two
repairs and both are ineffective for the same reason, which is now recorded as a measured source comment
above the float rather than as an aspiration: moving the float's declaration point does not move it,
because a top float is seated on the page being built only if it still fits there, and the bottom-float
option silently deferred the figure to page 72, at which point the body page gate passed only because the
figure had left the body altogether. That is a false pass no gate here can see, and only a pixel read
caught it. We have stopped spending rounds on it and have said so in the source, so the next round does
not retry it.

**No figure was regenerated.** The panel's pixels still hang three `(P#)` tokens on the nodes, which is
part of why the clarity review did not recognise the diagram. We addressed it by caption alone, for two
reasons we can state precisely: the generator's save path writes several other figures into a directory
this project is not permitted to touch, and its tight bounding box measures content, so relabelling a node
changes the rendered height of a figure that sits one rendered line from a page boundary. That is a
page-budget change no checksum can detect, because the timestamp inside a regenerated PDF moves on every
write regardless.

**No new experiments, and the empirical-generality score is where the review put it.** The clarity review
ranked more experiments at one star out of five and said the clarity gap closes by simplifying rather than
by explaining more; we agree, and spent zero compute. But that means the score review's cap on empirical
validation for generality, at three seed counts and three datasets, stands. Moving it needs either the 24
held-out CIFAR-100 pairs of our second wave, about 100 hours serial, or a positive oracle-free result, and
the oracle-free arm we actually ran refuted us at $n{=}20$. We think 8 of 10 is defensible on this build
and 9 is not, and saying otherwise would be exactly the over-claim the score review's concern 5 warns
against.

## 4. Three instrument changes, disclosed because instruments should not be edited quietly

Relocating six pages of appendix into the supplement broke two checkers that had the main paper's path
hardcoded, and tripped a third. None was weakened to pass.

- `check_cifar100_menu_prose.py` and `emit_perseed_cifar100_menu.py --check` both read `paper/main.tex`
  only, so the relocation made them report the moved table as missing. Both now read both xr-linked
  documents and require the content in one of them; absent from both is still a hard exit, and both print
  which document carried it, because a silent move between the two documents is what they would otherwise
  hide. They report 29 checks passing and 6 of 6 emitted rows present in the supplement.
- `audit_equivalence_claims.py` gained one classified entry, not a reworded paragraph. The relocation put
  a CIFAR-100 block's topic into the same navigation paragraph as the equivalence-testing block's, and the
  audit's inverse check reads that as a one-sided qualifier standing over an arm whose lower leg is a real
  test. We classified the supplement's own index paragraph, which begins "This supplement holds per-cell measurement detail" (`supplementary.tex:55`), as methods prose, with the reason recorded in the
  instrument, rather than dropping the dataset name to satisfy a checker.
- That audit reports the same 3 unqualified-claim sites and the same 1 stale classification key as before
  this round. We verified this rather than asserting it, by running the current instrument against the
  pre-relocation state of both documents: 47 claim units, 3 defects at the same three sentences, 1 stale
  key. Both are inherited and neither is introduced here. The stale key is a real defect in our own
  instrument, since a dead classification narrows the population it audits, and it is on the list for the
  next round.

## 5. Verification

Everything below was run against this build, in this order, after a build to a fixpoint rather than after
two passes.

1. Fixpoint build of both documents, 4 iterations, extracted text identical from iteration 2 onward. Only
   the two permitted overfull-box lines, both 3.77756pt, and the one permitted font-shape warning survive
   in the logs; 0 undefined references and 0 undefined citations in either document. The logs are not
   UTF-8, so they were read with `grep -a`, and the test is the presence of the permitted survivors rather
   than the absence of errors.
2. `measure_body_chars --gate`: body prose ends by page 9, Table 2 on page 7, Conclusion on page 9,
   Ethics the first content of page 10.
3. `measure_clarity_load --gate`: passing with no ratchet raised, and one re-baselined downward. Bold 38
   of 39, italic 56 of 60, `(P#)` 7 of 7, `C0-C3` 6 of 8, cross-references per paragraph 5 of 5, longest
   body paragraph 830 of 850, abstract 1585 of 1600, Figure 1 caption 722 of 722, design words 5 of 5,
   spine span 2 of 2, body homes 2 of 2, provenance framing 0, comment-joined paragraphs 0.
4. `measure_negation_density`: protected disclaimers 16 of 16 present. This is a whole-file,
   case-sensitive floor, and it was checked after every edit, not only at the end.
5. `measure_self_containment`: every body pointer into the appendix resolves and attaches to a stated
   claim, and the index covers every top-level appendix section exactly once in document order.
6. `audit_equivalence_claims`: 3 sites and 1 stale key, both inherited, as verified above.
7. The pre-registration self-count in the body, "$38$ pre-registration documents" (`:1489`), equals the
   number of pre-registration files on disk, which is 38. The numbered-statement sequence is identical to
   the pre-relocation state, 19 statements in the same order with none in the supplement, so no
   proposition, theorem, lemma or corollary was added, removed or renumbered.
8. The three starred statements, Ethics, Reproducibility and the LLM statement, were re-read line by line.
   The body window ends at the Ethics heading, so no gate reads them, and a relocation of six appendix
   pages is exactly the shape that has falsified them before. Every label they name resolves in the
   document their prose says it is in.
9. Anonymity and punctuation on extracted text rather than on source: 0 rendered round numbers, 0 review
   ordinals and 0 occurrences of the word reviewer in either PDF. The word review survives only in the
   conference header on every page, in the double-blind declaration, in one bibliography entry's journal
   title and in one literature-audit exclusion category, each of which we read individually rather than
   counting. No rendered prose double-hyphen was added: the one new triple-hyphen in the pair is inside a
   LaTeX comment marking the relocated block, and no existing one was lost. PDF title, author, subject and
   keywords are empty.
10. Pixel reads at 150 dpi of every page whose content moved: main pages 2, 6, 7, 8 and 9, supplement
    pages 10 and 30.
