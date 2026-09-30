"""Measure the conceptual load the body puts on a first-time reader.

The sixteenth review scored technical quality 8/10 and clarity 6/10, and located
the whole gap in information hierarchy: "too many conceptual objects in working
memory", and "if a reviewer can reconstruct that chain after reading the first
3--4 pages, I would expect 8/10 clarity to be realistic". Neither of those is a
number, so this script makes them into nine. The twentieth review scored clarity
5/10 with the same diagnosis in different words -- "a reviewer has to hold too
many concepts in their head simultaneously" -- and named the density of emphasis
and cross-references as the mechanism, which is what the last three measures add.

  (P#) tokens in the body  -- how often the reader is asked to hold a level index
                              rather than the English word for it. The five-word
                              glossary (decision/admission/influence/suppression/
                              attenuation) is what makes substitution lossless,
                              so a falling count here is a real reduction in load
                              and not a deletion.

  C0-C3 tokens in the body -- the criterion's conditions. These are already
                              nearly appendix-only in the body; the count exists
                              so a future round cannot quietly re-import them.

  front-matter chars       -- abstract, Figure 1's caption, the scope box.
                              CORRECTED IN ROUND 58, because this docstring said
                              all three were invisible to measure_body_chars.py
                              and that is true of only two of them. The abstract
                              and Figure 1 sit above \\section{Introduction} and
                              are outside that script's window; the scope box
                              does NOT -- it is a framed box inside the body, and
                              adding 84 chars to it raised body chars by 84.
                              Which of the three is inside the window is now
                              measured rather than asserted (`in_body` below),
                              because "front matter funds the body" was quietly
                              false for a third of the front matter.

  first-proposition page   -- the rendered page the body's first proposition
                              lands on. This is the review's actual complaint
                              made falsifiable: the central claim was on p5 of 9
                              because the taxonomy spent pages 3--4 ahead of it.

  five-question pages      -- ADDED IN ROUND 65. The twenty-third review supplied
                              its own acceptance test and its own threshold: five
                              questions, answerable "after ~3 pages" for 8/10
                              clarity, against the ~10--15 it estimated. This
                              locates each answer on a rendered page, so the
                              claim is falsifiable rather than asserted in a
                              letter. Q5 is keyed on the two numbers rather than
                              on prose, because its sentence was already on p1
                              while both values wrapped onto p2 -- a sentence's
                              page is not its numbers' page.

  spine span               -- ADDED IN ROUND 67. The twenty-fourth review supplied its
                              own acceptance test: by page 2 a reader should conclude
                              that the paper proposes no defense, proves the natural
                              evaluation unidentified, says why, replaces it with a
                              controlled intervention, and shows the bad design can
                              reverse the conclusion. All five of those beats ALREADY
                              rendered by p2 and the reviewer still did not assemble
                              them, so rendered position was never the binding variable.
                              What was binding is CONTIGUITY: the five beats were spread
                              over four \\noindent paragraphs and a bulleted list. So
                              this counts the body paragraphs a reader must join, not
                              the page they land on.

  body questions           -- ADDED IN ROUND 67, and it is what that review's
                              "declarative headings" item actually reduces to. All seven
                              section headings were already declarative; the essay-like
                              voice is the italic question opening most sections and
                              several paragraphs, 13 of them, each of which a reader must
                              hold open until its answer arrives. Three earn their keep
                              and are allowlisted BY PHRASE, never by line number.

  provenance framing       -- ADDED IN ROUND 67, a CEILING rather than a count of load.
                              Every pattern anchors on the FRAMING ("a methodological
                              correction", "we report our own", "we had predicted") and
                              never on the disclosed fact, because the fact is what
                              measure_negation_density's PROTECTED floor exists to keep.
                              A ceiling anchored on a disclosure would set the two
                              instruments against each other, which is the inverse of
                              Round 66's trap (a floor anchored on a negation fires on
                              exactly the rewrites it exists to permit).

  negation density         -- delegated to measure_negation_density.py so the
                              series stays comparable against its FROZEN marker
                              list. Compression must not buy clarity by turning
                              concessions into sweeping claims; this is the
                              tripwire for that.

  body chars               -- delegated to measure_body_chars.py, for the same
                              reason: one definition of "the body", shared.

  emphasis runs            -- \\textbf and \\emph runs in the body, and the rule
                              the twentieth review's complaint reduces to: AT MOST
                              ONE long (>40 char) bold run per rendered paragraph
                              beyond its run-in head. Round 61 measures 39 bold
                              and 62 italic over 54 paragraphs, against Round
                              58's 41 and 71 over 45. The italic
                              count does not fall much further and the reason is
                              structural, not laziness: most of it is definition
                              item labels and the paper's contrast pairs
                              (observable/mechanistic, inheritance/emergence),
                              where the italics ARE the distinction. So this is
                              gated as a ratchet -- no worse than today -- and
                              only the long-bold rule is gated at zero.

  cross-refs per paragraph -- the max over rendered paragraphs. The binding case
                              is the roadmap, which is a list of section pointers
                              by construction; it is reported, not exempted.

  longest abstract sentence-- the twenty-first review's word for the abstract is
                              "extremely dense", and it named sentence LENGTH as
                              the mechanism. Measured on the rendered text, with
                              inline macro wrappers removed, because a splitter
                              that needs whitespace after the period welds
                              "...caused its success.}" onto the next sentence:
                              Round 61's first measurement said 179 chars where
                              the real longest was 334.

  §1 causal vocabulary     -- the review's "don't open in causal-inference
                              language" made falsifiable: how many causal tokens
                              §1 spends, AND whether the first one arrives after
                              the FL grounding. Cross-reference ARGUMENTS are
                              stripped first: \\ref{prop:identification} renders
                              as "Proposition 1" and costs a reader nothing, and
                              counting it put this measure one token above its
                              target for a word that is not on the page.

  longest body paragraph   -- reader load per breath. Display math and nested
                              environments are excluded, matching what the
                              long-bold rule already does: a centred chain or an
                              \\[ ... \\] is not a paragraph a reader wades
                              through. Round 61 split the four longest at seams
                              where the subject changes, which costs \\parskip
                              (6pt) each and no chars.

  joined paragraphs        -- comment-only lines do NOT end a LaTeX paragraph:
                              a comment comments out its own newline, so a comment
                              block with prose on both sides silently welds two
                              paragraphs into one and turns the second's
                              \\noindent into a no-op. Round 58 introduced exactly
                              this at main.tex:311 and it renders as a defect
                              while every LaTeX check stays green. Gated at zero.

Everything structural is IMPORTED, never reimplemented: prose_lines/BODY_START/
BODY_END come from measure_negation_density, body_chars/page_words from
measure_body_chars, and uncomment() from there too. Three scripts disagreeing about
where the body starts would make all three numbers unfalsifiable.

EVERY char and token count here reads the RENDERED part of each line only. Until
Round 61 they read comment text as well, which made documenting a constraint spend
the budget the constraint protects: that round's ~90 lines of provenance comments
pushed (P#) to 21 against a cap of 20 on labels that appear only inside a comment
explaining why they must not be renamed, and box chars read 2538 against a rendered
1562. So the numbers in this file are not comparable to those in the notes of any
round before 61.

Usage:
    python3 -m experiments.measure_clarity_load
    python3 -m experiments.measure_clarity_load --rev HEAD    # diff vs a commit
    python3 -m experiments.measure_clarity_load --gate        # assert the targets
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

from experiments.measure_body_chars import body_chars, page_words, uncomment
from experiments.measure_negation_density import (BODY_END, BODY_START, ROOT,
                                                  density, prose_lines)

# (P1)--(P5) as the paper writes them: parenthesised, sometimes inside math, and
# sometimes as a range "(P1)--(P3)" whose endpoints both count.
P_RE = re.compile(r"\(P([1-5])\)")
C_RE = re.compile(r"\(?\bC([0-3])\)?\b")

# Front-matter blocks, each keyed on the line it starts at rather than a line
# number, so an edit anywhere above them does not silently move the measure.
BLOCKS = {
    "abstract": ("\\begin{abstract}", "\\end{abstract}"),
}
# The scope box. It was \fbox{\parbox{...}} inside a center environment until Round 78
# made it a `framed` environment, so that it breaks across a page instead of jumping
# whole to the next one and leaving 17 rendered lines empty. Both openings are matched,
# paired with their own close: with only the \fbox form listed, this counter read 0 for
# a box that was still on the page, and a 0 here reads as "the box was deleted".
BOX_FORMS = (("\\fbox{\\parbox{", "}}"), ("\\begin{framed}", "\\end{framed}"))

# pdftotext renders theorem headers as "Proposition 1" with the number attached.
PROP_RE = re.compile(r"Proposition\s+1\b")

# The twenty-third review's own acceptance test, turned into a measurement. It set the
# threshold itself -- "if a reviewer can answer these five questions after ~3 pages, I
# would give the paper 8/10 clarity. Right now, I think they can answer them after
# ~10--15 pages" -- so the only honest way to answer it is to locate each answer on a
# RENDERED page rather than assert in a letter that it is early. Each question is keyed
# on a phrase the body actually prints, so the measure moves when the prose moves and a
# reordering that strands an answer on a later page fails here.
#
# Q5 is that review's Priority 1 ("put -0.272 vs +0.098 on page 1"), and it is keyed on
# the two numbers themselves rather than on prose, because a sentence can be on p1 while
# the numbers it carries wrap onto p2 -- which is exactly what this paper did: the
# sentence began on p1's last line and both values landed at the top of p2. It is keyed
# on the CURRENT pair, -0.273 / +0.125 at n=20, which supersedes the n=5 pair the review
# quotes. The pair must share a page, so satisfying it with one number is not possible.
Q5_TEST = (
    ("Q1 what problem?", ("mechanism what suppressed it",)),
    ("Q2 what answer?", ("cannot answer the question at any sample size",)),
    ("Q3 why?", ("removes the counterfactual support identification requires",)),
    ("Q4 how shown?", ("intervene on the upstream transform",)),
    ("Q5 what happens?", ("0.273", "0.125")),
)

# The twenty-fourth review's own acceptance test, and deliberately NOT a page test. It
# asks that a reader reach five conclusions by p2; all five beats already rendered by p2
# before this round, and that reviewer still reported the paper as proposing a defense
# and the boundary as being about our own criterion. Position was not the binding
# variable. What was binding is that the beats sat in four separate \noindent paragraphs
# with a bulleted list between two of them, so a reader had to assemble the core message
# rather than read it. Keyed on SOURCE phrases (this measure is about paragraph
# structure, which the PDF cannot report) and each beat is credited to its EARLIEST
# paragraph, since that is where a reader meets it.
SPINE_TEST = (
    ("no new defense", ("propose no new defense",)),
    ("the gate cannot answer", ("cannot answer the question at any sample size",)),
    ("why: no counterfactual",
     ("removes the counterfactual support identification requires",)),
    ("what replaces it", ("intervene on the upstream transform",)),
    ("and it reverses a sign", ("-0.273", "+0.125")),
)

# The italic question openers. Three are sanctioned, matched on a distinctive PHRASE
# rather than a line number, so a reworded keeper fails loudly instead of passing as
# sanctioned: the paper's own question (Q5_TEST's Q1 is keyed on the same sentence), the
# FL motivation, and the Conclusion's open problem, which is a question because it is
# genuinely open.
SANCTIONED_QUESTIONS = (
    "mechanism what suppressed it",
    "compose them for stronger robustness",
    "without knowing which clients are malicious",
)
QUESTION_RE = re.compile(r"[^.!?]*\?")

# Research-history FRAMING, ceiling-gated. "our own criterion" is deliberately absent:
# it is the PROTECTED anchor of the false-negative disclosure, so listing it here would
# make the ceiling and the floor unsatisfiable together. "the fault is ours" is absent
# for a different reason -- the review names that sentence as the paper's transparency,
# not as its research history.
PROVENANCE_RE = tuple(re.compile(p, re.I) for p in (
    r"methodological correction", r"we report our own", r"our own refutations",
    r"we had predicted", r"earlier version", r"previously named",
    r"this amendment", r"process failure", r"nearly reversed", r"was corrected",
))

# The two facts the ceiling's target sentence carries, each asserted present in the BODY
# WINDOW at its destination. This is a floor, and it exists because
# measure_negation_density's PROTECTED floor cannot do this job: protected() tests
# rx.search(text) over the WHOLE FILE and reports its body count as commentary only, so
# the false-negative disclosure is held up by three appendix twins and stays 16/16 even
# when both body statements are deleted. Falsification-probed: deleting the fact from the
# body left that floor at 16/16 and exit 0.
# Each pattern is keyed on wording UNIQUE TO THE DESTINATION, never on wording shared
# with the sentence being cut -- a floor that the doomed sentence can satisfy tests
# nothing about whether the fact survived the cut.
BODY_HOMES = (
    ("false-negative disclosure -> the emergent witness",
     r"emergent and a false negative"),
    ("collapsed cross-arm admission ordering -> the replication",
     r"refutes the admission"),
)

# The abstract target was first priced at 1400 against review #12's own suggested
# draft. That draft reached 1400 by dropping the FL setup, the replication in a
# second invariance class, and one of the two dissociations, i.e. by dropping
# measured results rather than words. 1600 is the length the same seven claims fit
# in, and it is still a 33% cut from the 2357 the sixteenth review read.
# c_tokens was first priced at 6 without inspecting where the ten were. Three of
# them are row LABELS of the three-strategy table (a row cannot be named without
# naming its condition), two are the cost accounting for the condition that needs
# standalone runs, one is the C0-was-retrospective disclosure (L4), and one is the
# named blind spot behind the 40% recall. Every remaining occurrence identifies a
# specific object; 8 is the floor that does not delete a disclosure. Review #3's
# actual ask was already met before this round: 8 in the body against 326
# document-wide.
# Round 61 lowered bold 41 -> 39 (Phase B's cuts) and italic 71 -> 62, the second by
# deleting nine italics that were vocal stress on an ordinary word ("for \emph{any}
# such condition", "pins the \emph{adversarial} coefficients and nothing else"). It
# did NOT reach the round's hoped-for bold 34 or max_xrefs 4, and both are set to the
# measurement rather than to the hope, because the remaining runs are not decoration:
# bold is one run-in head per paragraph, which is the paper's only navigation, and all
# three paragraphs at 5 cross-references are pointer LISTS -- the roadmap (five
# sections), the chain-break paragraph (two witness sections, two figures, one
# appendix) and box:scope's "Not established" (the five places a claim that does not
# transfer is recorded). Cutting one of the last would delete a disclosure's pointer,
# which the one-home rule forbids.
# emph_runs and max_xrefs are RATCHETS at the Round 58 measurement, not aspirations:
# they exist so a later round cannot re-import density, and they are set to what the
# paper actually is rather than to what the plan hoped for. long_bold and joined are
# real rules and are gated at zero. Raising any of these to make the gate pass is the
# failure mode of page_gate_measures_last_line and is not permitted; the fix is the
# paper.
#
# Round 61 re-baselined every char and token count in this file, because until this
# round they all read LaTEX COMMENT TEXT as reader load (see uncomment() in
# measure_body_chars). The ratchets below are the comment-free measurements, so they
# are not comparable to the numbers in earlier rounds' notes: (P#) reads 12 where the
# old counter said 21, and C reads 3 where it said 7.
# abs_sentence / causal_sec1 / long_para are the three measures the twenty-first
# review's clarity complaint reduces to, and they are ratchets at this round's
# measurement too: 179 / 8 / 835. causal_sec1 also carries an ORDERING rule that no
# count expresses -- the first causal word must arrive after the FL problem -- and
# that one is gated as a boolean.
#
# Round 67 re-baselines three ratchets DOWNWARD to its own measurement, which is the only direction
# this file permits: p_tokens 12 -> 7, because E3 retired the centred chain display that spent 7 of the
# 12 tokens into a table whose level column spends 5; fig1_caption 1480 -> 1153; and emph_runs 62 -> 60,
# which E9's ten question-opener rewrites paid for. The italic ratchet had been at its cap for three
# rounds, which is why Round 66's caption fix had to be reverted; closing the 2 runs of slack now means a
# later round cannot spend them silently.
#
# Round 62 closes the two ratchets this round proved slack rather than leaving headroom
# a later round could spend: p_tokens 20 -> 12 and causal_sec1 8 -> 6, both set to the
# measurement. It adds two assertions that are not counts of load but names of a
# specific defect the twenty-second review reported:
#   jargon_front  -- "descendant of the quantity being estimated" was the phrase the
#                    review quoted back as unreadable. It is gone from the abstract and
#                    §1 (the two places a reader meets it first) and stays in the three
#                    appendix statements, which are formal. Gated at 0 over those two
#                    blocks only, NOT document-wide, so the formal statements survive.
#   design_vocab  -- §1 fixes five design words in italics (screen, gate, criterion,
#                    protocol, instruments) because the same review read them as five
#                    separate contributions. A later round tidying that sentence away
#                    would silently restore the misreading, so its five terms are
#                    asserted present. This gates the TERMS, not the sentence, so the
#                    prose around them can still be rewritten.
#
# Round 67 adds five, four of them new measures and one a gate line on a number this
# file already printed. spine_span is the twenty-fourth review's acceptance test as a
# CONTIGUITY bound (2, so the core message may occupy at most two adjacent paragraphs);
# body_questions and provenance are gated at ZERO because both are complete rather than
# ratcheted -- every unsanctioned question was converted and the one framing sentence was
# cut, so there is no residue to grandfather. fig1_caption becomes a ratchet at the
# post-cut measurement so a later round cannot restore the caption that was doing the
# figure's job. body_homes is the FLOOR that composes with the provenance ceiling, and it
# is here rather than in measure_negation_density because that file's PROTECTED floor is
# a whole-file presence test: the probe deleted the false-negative disclosure from the
# body and it still read 16/16, held up by three appendix twins. A ceiling on framing is
# only safe next to a floor measured in the same window the framing was cut from.
#
# Round 70 re-baselines fig1_caption DOWNWARD, 1153 -> 979, which is the only direction this
# dict permits. Two cuts earn it: the composed figure lost its Mode-S panel to a standalone
# appendix float, so the caption no longer describes three panels, and the bold head became
# the abstract's own thesis sentence. That second cut was forced rather than chosen -- moving
# the float inside Section 1 pulled its caption into the causal_sec1 window and the count went
# 6 -> 7 against a hard ceiling, so the caption is now also load-bearing for that gate. Every
# scope clause it carried is still in it; nothing here prices a disclosure as slack.
#
# Round 78 re-baselines it DOWNWARD again, 979 -> 735. The cut is two bookkeeping sentences:
# the one that told the reader its three numbers were three scopes rather than one experiment,
# and the one that reconciled the stale coord_median/scaling row label against the six-cell
# table. Neither described the drawing. The first is redundant with panel (b)'s own per-row n,
# which the caption still names; the second MOVED to the appendix story section and is quoted
# in the caption by pointer ("the n frozen here, App. J for the one row since topped up"), so
# the staleness is still disclosed on the figure's own page. A disclosure may move and may not
# lose its last home, and re-baselining is what keeps a cut from being spent twice.
#
# Round 83 re-baselines it DOWNWARD again, 735 -> 722, under the same rule. The twenty-fourth
# review's top ask is a pipeline diagram that panel (a) already draws, so the caption is what
# failed: it opened on "the map of this paper's vocabulary, at the levels of Def. 1" and now
# opens on the panel's own two bands before naming the chain in plain words. Nothing was
# deleted -- both disclosures (the per-row n with its topped-up pointer, and the four rows that
# are training data for a withdrawn rule) are still in it, and the 13 chars are what the shorter
# framing returned, so they are retightened rather than banked.
#
# Round 84 RE-KEYS it, which is a different act from re-baselining and is recorded separately.
# `fig1_caption` was hard-wired to the label fig:modeS, so it measured "the composite panel
# figure" and not "whatever is Figure 1". Two reviews in a row have now said Figure 1 tries to
# do too much (the tenth review's item (6), answered in Round 70 by splitting panel (c) out, and
# the twenty-fifth's item 13), and the second one sketched a figure the paper already had as an
# appendix float since Round 69: figures/pipeline_dag.pdf. So the floats were swapped and the
# metric now reads caption_chars(text, "fig:pipeline"). The cap moves 722 -> 707, still
# downward, and 707 is the measurement rather than headroom. Three things this cut had to buy
# that a length ratchet cannot see, all recorded because a later round must not undo them by
# "restoring" caption content: (1) the caption went 1437 -> 707, and the 730 chars it shed are
# not deleted -- the three qualifications it carried (that Mode S identifies the effect at a
# fixed adversarial coefficient share and NOT the statistic path in isolation, that the pinning
# needs adversary identity so Mode S is an instrument rather than a defense, and the
# top-path/bottom-path punchline) moved into the appendix paragraph that still cites the figure;
# (2) all five of its (P#) tokens had to go, because p_tokens is at its cap of 7 and counts the
# body window, so the caption entering the body would have read 12 -- they are now "Def. 1's
# first four senses, in order" and plain words, which is what the clarity review asked for; and
# (3) the bold and italic budgets are 3-for-3 and 1-for-1 against the caption that left, which
# is why bold_runs stayed at 38 of 39 across a figure swap.
#
# And one BLIND SPOT the swap exposed in p_tokens and c_tokens, recorded here rather than fixed,
# because fixing it means reading a PDF's drawing and these two measures read source text.
# pipeline_dag.pdf bakes "(P1), (P2)", "(P3)", "(P4)", "(P4), quantitative", "then ASR (P5)" and
# "so C2 is not the operative channel" into its RENDERED LABELS. So a reader of page 2 now meets
# five level indices and one condition index that p_tokens=7 and c_tokens=7 do not count, and
# a 150-dpi pixel read is the only thing that sees them. That is why the caption glosses C2 in
# words ("that argument is the figure's C2") and maps the level indices to Def. 1's senses in
# order rather than repeating them: the figure's own labels are the tokens, and the caption's job
# is to be their glossary. Do not "save" caption chars by cutting either gloss -- the tokens do
# not leave the page when the caption stops explaining them, they just stop being defined.
TARGETS ={"p_tokens": 7, "c_tokens": 8, "abstract": 1600, "prop_page": 3,
           "q5_pages": 3,
           "bold_runs": 39, "emph_runs": 60, "max_xrefs": 5,
           "long_bold": 0, "joined": 0,
           "abs_sentence": 200, "causal_sec1": 6, "long_para": 850,
           "jargon_front": 0, "design_vocab": 5,
           "spine_span": 2, "body_questions": 0, "provenance": 0,
           "fig1_caption": 707}

# The phrase the twenty-second review quoted as the paper's least readable, matched on
# the two words that carry it so a rewording that keeps the jargon still trips.
JARGON_RE = re.compile(r"descendant\w*\s+of\s+the\s+(?:quantity|estimand)", re.I)
# The design objects §1 must distinguish, each fixed once in italics.
DESIGN_VOCAB = ("Screen", "gate", "criterion", "protocol", "instruments")

# A "long" bold run: long enough that a reader reads it as a sentence rather than as
# a label. 40 chars is where the paper's own run-in heads sit.
LONG_BOLD = 40

# --- the three Round 61 measures -------------------------------------------------
# Inline macros whose braces do not render, removed before sentences are split or
# counted. \texttt{cos\_krum} keeps its backslash-escape, so a sentence carrying one
# is measured two chars long per escape; this is a source-char measure and the
# ratchet is set to what it reads, not to a glyph count.
INLINE_MACRO = re.compile(r"\\(?:textbf|emph|texttt|textit|text|mathrm)\{")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
# Display openers. A paragraph's prose is what precedes the first one.
DISPLAY = re.compile(r"\\begin\{|\\\[")
# The words the review objects to meeting before the FL problem. Stems, because
# "identify"/"identification"/"identifiable" are one demand on the reader.
CAUSAL_RE = re.compile(r"\b(causal\w*|counterfactual\w*|identif\w*|positivity"
                       r"|collider\w*|estimand\w*|confound\w*|interven\w*"
                       r"|descendant\w*)\b", re.I)
# The FL grounding the first causal token must come after. Matched on the stem so
# "Federated learning" and "a federated setting" both count.
FL_RE = re.compile(r"federat\w*", re.I)
# Cross-reference and citation arguments, which render as a number and are stripped
# before any of the counts above run over prose.
REF_ARG = re.compile(r"\\(?:ref|pageref|eqref|label|cite[pt]?|citealp|citeauthor)"
                     r"\*?(?:\[[^\]]*\])?\{[^}]*\}")


def block_chars(text, start, end):
    """Chars strictly between two marker lines, markers excluded."""
    lines = text.split("\n")
    a = next((i for i, l in enumerate(lines) if l.startswith(start)), None)
    if a is None:
        return 0, None
    b = next((i for i, l in enumerate(lines[a + 1:], a + 1)
              if l.startswith(end)), None)
    if b is None:
        return 0, None
    return len("".join(uncomment(l) for l in lines[a + 1:b])), (a + 2, b)


def caption_chars(text, label):
    """Chars of the \\caption whose float carries `label`."""
    lines = text.split("\n")
    i = next((i for i, l in enumerate(lines)
              if l.strip() == "\\label{%s}" % label), None)
    if i is None:
        return 0
    # Walk back to the \caption that belongs to this label.
    for j in range(i - 1, max(i - 40, -1), -1):
        if lines[j].startswith("\\caption"):
            return len("".join(uncomment(l) for l in lines[j:i]))
    return 0


def box_chars(text):
    lines = text.split("\n")
    for opener, closer in BOX_FORMS:
        a = next((i for i, l in enumerate(lines) if l.startswith(opener)), None)
        if a is None:
            continue
        b = next((i for i, l in enumerate(lines[a:], a) if l.strip() == closer), None)
        if b is None:
            continue
        return len("".join(uncomment(l) for l in lines[a:b + 1]))
    return 0


def sentences(s):
    """Sentence strings of a block, measured as a reader meets them.

    Inline macro wrappers go first, so `\\textbf{... its success.}` ends where its
    period does; splitting the raw source instead welds it onto the sentence after,
    which is how the abstract measured 179 chars with a 334-char sentence in it.
    """
    s = INLINE_MACRO.sub("", s).replace("}", "").replace("\\ ", " ")
    return [t.strip() for t in SENT_SPLIT.split(s) if t.strip()]


def abstract_sentences(text):
    """(longest sentence chars, that sentence, n sentences) for the abstract."""
    lines = text.split("\n")
    a = next((i for i, l in enumerate(lines)
              if l.startswith(BLOCKS["abstract"][0])), None)
    if a is None:
        return 0, "", 0
    b = next((i for i, l in enumerate(lines[a + 1:], a + 1)
              if l.startswith(BLOCKS["abstract"][1])), None)
    body = " ".join(uncomment(l) for l in lines[a + 1:b])
    ss = sentences(body)
    if not ss:
        return 0, "", 0
    worst = max(ss, key=len)
    return len(worst), worst, len(ss)


def section_window(text, heading):
    """Lines of one \\section, from its heading to the next \\section (exclusive)."""
    lines = text.split("\n")
    a = next((i for i, l in enumerate(lines) if l.startswith(heading)), None)
    if a is None:
        return ""
    b = next((i for i, l in enumerate(lines[a + 1:], a + 1)
              if l.startswith("\\section")), len(lines))
    return "\n".join(uncomment(l) for l in lines[a:b])


def causal_vocabulary(text):
    """(count, tokens, grounded_first) for §1's causal vocabulary.

    `grounded_first` is the review's actual ask, not the count: the reader must meet
    the federated-learning problem BEFORE the first causal word. None if §1 names no
    causal word at all, which is not a failure.
    """
    win = REF_ARG.sub(" ", section_window(text, BODY_START))
    toks = [(m.start(), m.group(0)) for m in CAUSAL_RE.finditer(win)]
    fl = FL_RE.search(win)
    grounded = None if not toks else bool(fl and fl.start() < toks[0][0])
    return len(toks), [t for _, t in toks], grounded


def front_sections(text, n=2):
    """The body's first n sections, joined. §2 is in the window on purpose: the
    sentence fixing the design vocabulary sits three lines below §2's heading, not in
    §1, and a window drawn at §1 measured it as absent -- resolve a line number, do
    not trust where a plan says a sentence lives.
    """
    lines = text.split("\n")
    a = next((i for i, l in enumerate(lines) if l.startswith(BODY_START)), None)
    if a is None:
        return ""
    heads = [i for i, l in enumerate(lines[a + 1:], a + 1)
             if l.startswith("\\section")]
    b = heads[n - 1] if len(heads) >= n else len(lines)
    return "\n".join(uncomment(l) for l in lines[a:b])


def front_jargon(text):
    """(hits, where) for the review's quoted phrase in the abstract and §§1-2 only.

    Those are the blocks a reader meets before any formal statement. The appendix keeps
    the phrasing in its three formal statements and is deliberately out of the window.
    """
    lines = text.split("\n")
    i = next((n for n, l in enumerate(lines)
              if l.startswith(BLOCKS["abstract"][0])), None)
    j = next((n for n, l in enumerate(lines[i + 1:], i + 1)
              if l.startswith(BLOCKS["abstract"][1])), None) if i is not None else None
    abstract = "\n".join(lines[i + 1:j]) if j is not None else ""
    where = []
    for name, blk in (("abstract", abstract), ("§§1-2", front_sections(text))):
        n = len(JARGON_RE.findall(blk))
        if n:
            where.append(f"{name}:{n}")
    return sum(int(w.split(":")[1]) for w in where), where


def design_vocabulary(text):
    """(how many of the five design words §§1-2 fix in italics, the ones they do not)."""
    win = front_sections(text)
    missing = [w for w in DESIGN_VOCAB if ("\\emph{%s}" % w) not in win]
    return len(DESIGN_VOCAB) - len(missing), missing


def prose_len(p):
    """Chars of a paragraph a reader reads as prose: everything before its first display."""
    m = DISPLAY.search(p)
    return len(p) if not m else m.start()


def longest_paragraph(text):
    """(max prose chars, [(line, chars), ...] worst first) over body paragraphs."""
    per = sorted(((prose_len(p), n) for n, p in paragraphs(text)), reverse=True)
    return (per[0][0] if per else 0), [(n, c) for c, n in per[:3]]


def body_window(text):
    """The raw body lines, [BODY_START .. BODY_END), exactly as the other scripts see it."""
    lines = text.split("\n")
    a = next(i for i, l in enumerate(lines) if l.startswith(BODY_START))
    b = next(i for i, l in enumerate(lines)
             if any(l.startswith(m) for m in BODY_END))
    return lines, a, b


def macro_runs(s, macro):
    """(offset, argument) for each \\macro{...} in `s`, brace-matched so nesting is safe."""
    out, i, open_ = [], 0, "\\" + macro + "{"
    while True:
        j = s.find(open_, i)
        if j < 0:
            return out
        k, depth = j + len(open_), 1
        while k < len(s) and depth:
            if s[k] == "{":
                depth += 1
            elif s[k] == "}":
                depth -= 1
            k += 1
        out.append((j, s[j + len(open_):k - 1]))
        i = k


def paragraphs(text):
    """Rendered paragraphs of the body, by LaTeX's own rule.

    A paragraph ends at a BLANK line and nowhere else. A comment-only line is NOT
    blank -- TeX discards the comment together with its newline -- so two prose
    lines separated only by comments are ONE paragraph. Modelling that faithfully is
    the whole point: it is what makes `joined_paragraphs` below detectable.

    Comment CONTENT is stripped inside each paragraph, since it does not render and
    would otherwise inflate every count here the way it already inflates
    block_chars/caption_chars/box_chars.
    """
    lines, a, b = body_window(text)
    out, cur, start = [], [], None
    for i in range(a, b):
        if lines[i].strip() == "":
            if cur:
                out.append((start + 1, "\n".join(cur)))
            cur, start = [], None
            continue
        if start is None:
            start = i
        cur.append("" if lines[i].lstrip().startswith("%") else lines[i])
    if cur:
        out.append((start + 1, "\n".join(cur)))
    # Structural blocks a reader does not read as a paragraph of prose.
    skip = ("\\section", "\\subsection", "\\label", "\\begin{", "\\end{",
            "\\bibliography", "\\vspace", "\\includegraphics")
    return [(n, p) for n, p in out if not p.lstrip().startswith(skip)]


# Lines a reader does not read as prose. Wider than paragraphs()' `skip`, and used
# differently: these lines are DROPPED from a block rather than used to discard the whole
# block, because the four question openers that matter most sit on the first prose line
# after a heading.
STRUCTURE = ("\\section", "\\subsection", "\\label", "\\begin{", "\\end{",
             "\\bibliography", "\\vspace", "\\includegraphics", "\\centering",
             "\\toprule", "\\midrule", "\\bottomrule", "\\phantomsection")


def first_prose_line(text, start):
    """The first line at or after `start` that a reader actually reads.

    A block's recorded line is its FIRST line, which is routinely a provenance comment or
    a \\section head -- :133 for the contributions paragraph, :272 for §3's opener. Citing
    those in a response letter lands a reviewer on a comment, so every line reported here
    is resolved to rendered prose first (see cited_lines_vs_iclr_margin_numbers).
    """
    lines = text.split("\n")
    for i in range(start - 1, min(start + 40, len(lines))):
        s = lines[i]
        if not s.strip() or s.lstrip().startswith("%") or s.lstrip().startswith(STRUCTURE):
            continue
        return i + 1
    return start


def prose_blocks(text):
    """(first rendered line, rendered text) per body block, structure lines dropped.

    paragraphs() discards a block that OPENS with \\section or \\subsection, which is
    right for the emphasis and cross-reference ratchets calibrated on it and wrong here:
    it was blind to four of the body's thirteen question sentences, among them the paper's
    own core question at :109 and the three subsection openers, i.e. precisely the sites
    the twenty-fourth review's item 15 is about.
    """
    lines, a, b = body_window(text)
    out, cur, start = [], [], None
    for i in range(a, b + 1):
        if i >= b or lines[i].strip() == "":
            if cur:
                out.append((start, "\n".join(cur)))
            cur, start = [], None
            continue
        s = lines[i]
        if s.lstrip().startswith("%") or s.lstrip().startswith(STRUCTURE):
            continue
        if start is None:
            start = i + 1
        cur.append(s)
    return out


def spine_paragraphs(text):
    """(span, [(label, index, line), ...]) for SPINE_TEST over body paragraphs.

    `span` is how many CONSECUTIVE body paragraphs a reader must read to meet all five
    beats: the index distance from the earliest to the latest, inclusive. 1 means one
    paragraph carries the whole core message. A beat present in several paragraphs is
    credited to the earliest; a beat present in none returns None, so a reworded beat
    fails loudly rather than shrinking the span.
    """
    paras = paragraphs(text)
    found = []
    for label, phrases in SPINE_TEST:
        i = next((k for k, (_, p) in enumerate(paras)
                  if all(x in p for x in phrases)), None)
        found.append((label, i,
                      None if i is None else first_prose_line(text, paras[i][0])))
    idx = [i for _, i, _ in found]
    span = None if any(i is None for i in idx) else max(idx) - min(idx) + 1
    return span, found


def body_questions(text):
    """(n, unsanctioned, all) question sentences in rendered body prose.

    Cross-reference arguments and inline macro wrappers go first, so
    `\\emph{Preserved in which sense?}` is one question and not a brace soup. A
    \\paragraph{...?} head counts: it is a question a reader is asked and must hold open,
    which is the whole complaint.
    """
    out = []
    for n, p in prose_blocks(text):
        s = INLINE_MACRO.sub("", REF_ARG.sub(" ", p)).replace("}", " ")
        s = " ".join(s.split())
        out.extend((n, m.group(0).strip()) for m in QUESTION_RE.finditer(s)
                   if m.group(0).strip())
    bad = [(n, q) for n, q in out
           if not any(k in q for k in SANCTIONED_QUESTIONS)]
    return len(out), bad, out


def provenance(text):
    """(hits, [(line, pattern), ...]) for research-history framing in the body window."""
    lines, a, b = body_window(text)
    hits = []
    for i in range(a, b):
        s = uncomment(lines[i])
        for rx in PROVENANCE_RE:
            hits.extend((i + 1, rx.pattern) for _ in rx.finditer(s))
    return len(hits), hits


def body_homes(text):
    """(present, total, [(name, line-or-None), ...]). A FLOOR over the body window."""
    lines, a, b = body_window(text)
    where = []
    for name, pat in BODY_HOMES:
        rx = re.compile(pat, re.I)
        hit = next((i + 1 for i in range(a, b) if rx.search(uncomment(lines[i]))), None)
        where.append((name, hit))
    return sum(1 for _, h in where if h), len(BODY_HOMES), where


def emphasis(text):
    """Bold/italic run counts, and the paragraphs breaking the one-long-bold rule.

    Runs inside an environment nested in the paragraph (a definition's \\item
    labels, a table cell) are counted but exempt from the rule: those are labels,
    not competing thesis sentences, and Def. 1 cannot name a level without bolding
    it.
    """
    bold = ital = 0
    viol = []
    for n, p in paragraphs(text):
        runs = macro_runs(p, "textbf")
        bold += len(runs)
        ital += len(macro_runs(p, "emph"))
        env = p.find("\\begin{")
        limit = len(p) if env < 0 else env
        # A run-in head is the bold run that opens the paragraph, allowing for
        # \noindent / \paragraph{...} / \emph{...} ahead of it.
        rest = [r for r in runs if r[0] >= 30 and r[0] < limit]
        longs = [r[1] for r in rest if len(r[1]) > LONG_BOLD]
        if len(longs) > 1:
            viol.append((n, longs))
    return bold, ital, viol


XREF_RE = re.compile(r"\\(ref|pageref|eqref|citep|citet|cite)\{")


def xrefs(text):
    """(max, [(line, count), ...] worst first) cross-references per rendered paragraph."""
    per = sorted(((len(XREF_RE.findall(p)), n) for n, p in paragraphs(text)),
                 reverse=True)
    return (per[0][0] if per else 0), [(n, c) for c, n in per[:5]]


def joined_paragraphs(text):
    """Comment blocks in the body with PROSE on both sides, which weld two paragraphs.

    Restricted to prose on both sides on purpose: a comment between \\centering and
    \\includegraphics, or between \\label{sec:x} and the section's first sentence,
    joins nothing, because a float or a \\section has already broken the paragraph.
    """
    lines, a, b = body_window(text)
    prose = lambda l: bool(l.strip()) and not l.lstrip().startswith(("\\", "%", "}"))
    hits, i = [], a
    while i < b:
        if lines[i].lstrip().startswith("%"):
            j = i
            while j < b and lines[j].lstrip().startswith("%"):
                j += 1
            before = lines[i - 1] if i > a else ""
            after = lines[j] if j < b else ""
            # The line AFTER may legitimately open with \noindent or \paragraph and
            # still be prose that wanted its own paragraph, so those two count.
            after_prose = prose(after) or after.lstrip().startswith(
                ("\\noindent", "\\paragraph{", "\\emph{", "\\textbf{"))
            if prose(before) and after_prose:
                hits.append((i + 1, j))
            i = j
        else:
            i += 1
    return hits


def token_counts(text):
    """(P#) and C# counts in the body window only."""
    body = " ".join(prose_lines(text))
    # prose_lines drops floats and enumerate bodies, which is right for negation
    # (a table cell is not a sentence) but wrong here: Def. 1's own enumerate
    # DEFINES the levels and the reader pays for every index in it. So the
    # tokens are counted over the raw body window, floats included.
    lines = text.split("\n")
    a = next(i for i, l in enumerate(lines) if l.startswith(BODY_START))
    b = next(i for i, l in enumerate(lines)
             if any(l.startswith(m) for m in BODY_END))
    raw = "\n".join(uncomment(l) for l in lines[a:b])
    per_p = {n: 0 for n in "12345"}
    for m in P_RE.finditer(raw):
        per_p[m.group(1)] += 1
    per_c = {n: 0 for n in "0123"}
    for m in C_RE.finditer(raw):
        per_c[m.group(1)] += 1
    return per_p, per_c, len(body)


def first_prop_page(pdf):
    """The rendered page carrying 'Proposition 1'."""
    if not Path(pdf).exists():
        return None
    _, pages = page_words(pdf)
    for i, p in enumerate(pages, 1):
        if PROP_RE.search(p):
            return i
    return None


def q5_answer_pages(pdf):
    """(last page needed, [(label, page or None), ...]) for Q5_TEST.

    Whitespace is collapsed per page before matching, because pdftotext breaks a
    rendered line wherever the typesetter did and a phrase that spans two lines would
    otherwise read as absent. A missing phrase returns None for that question rather
    than a page, so a reworded answer fails loudly instead of scoring 0 pages.
    """
    if not Path(pdf).exists():
        return None, []
    _, pages = page_words(pdf)
    flat = [" ".join(p.split()) for p in pages]
    found = [(label, next((i for i, p in enumerate(flat, 1)
                           if all(x in p for x in phrases)), None))
             for label, phrases in Q5_TEST]
    pgs = [p for _, p in found]
    worst = None if any(p is None for p in pgs) else max(pgs)
    return worst, found


def measure(text, pdf=None, tex_path=None):
    per_p, per_c, prose = token_counts(text)
    abs_chars, abs_lines = block_chars(text, *BLOCKS["abstract"])
    out = {
        "per_p": per_p,
        "p_tokens": sum(per_p.values()),
        "per_c": per_c,
        "c_tokens": sum(per_c.values()),
        "abstract": abs_chars,
        "abstract_lines": abs_lines,
        "fig1_caption": caption_chars(text, "fig:pipeline"),
        "box": box_chars(text),
        "prop_page": first_prop_page(pdf) if pdf else None,
    }
    q5, q5_where = q5_answer_pages(pdf) if pdf else (None, [])
    out["q5_pages"], out["q5_where"] = q5, q5_where
    bold, ital, viol = emphasis(text)
    mx, worst = xrefs(text)
    abs_worst, abs_text, abs_n = abstract_sentences(text)
    n_causal, causal_toks, grounded = causal_vocabulary(text)
    long_p, worst_p = longest_paragraph(text)
    out.update({"bold_runs": bold, "emph_runs": ital, "long_bold": viol,
                "paragraphs": len(paragraphs(text)),
                "max_xrefs": mx, "worst_xrefs": worst,
                "joined": joined_paragraphs(text),
                "abs_sentence": abs_worst, "abs_sentence_text": abs_text,
                "abs_sentences": abs_n,
                "causal_sec1": n_causal, "causal_tokens": causal_toks,
                "causal_after_grounding": grounded,
                "long_para": long_p, "worst_paras": worst_p})
    jf, jf_where = front_jargon(text)
    dv, dv_missing = design_vocabulary(text)
    out.update({"jargon_front": jf, "jargon_where": jf_where,
                "design_vocab": dv, "design_missing": dv_missing})
    span, spine_where = spine_paragraphs(text)
    nq, bad_q, all_q = body_questions(text)
    nprov, prov_where = provenance(text)
    nhome, thome, home_where = body_homes(text)
    out.update({"spine_span": span, "spine_where": spine_where,
                "questions": nq, "body_questions": len(bad_q),
                "questions_bad": bad_q, "questions_all": all_q,
                "provenance": nprov, "provenance_where": prov_where,
                "body_homes": nhome, "body_homes_total": thome,
                "body_homes_where": home_where})
    # Which front-matter blocks are actually outside measure_body_chars' window, so
    # the "front matter funds the body" claim is measured and not recited.
    lines, a, _ = body_window(text)
    out["box_in_body"] = any(l.startswith(o) for o, _ in BOX_FORMS for l in lines[a:])
    n, tot, _ = density(text)
    out["negation"] = (n, tot, 100.0 * n / tot if tot else 0.0)
    if tex_path is not None:
        out["body_chars"] = body_chars(tex_path)[0]
    else:
        # Same window, measured off the string rather than the path.
        lines = text.split("\n")
        a = next(i for i, l in enumerate(lines) if l.startswith(BODY_START))
        b = next(i for i, l in enumerate(lines)
                 if any(l.startswith(m) for m in BODY_END))
        out["body_chars"] = len("".join(uncomment(l) for l in lines[a:b]))
    return out


def report(tag, m):
    print(f"=== {tag} ===")
    pp = " ".join(f"P{k}:{v}" for k, v in sorted(m["per_p"].items()))
    print(f"  (P#) tokens in body : {m['p_tokens']:>6}   ({pp})")
    pc = " ".join(f"C{k}:{v}" for k, v in sorted(m["per_c"].items()))
    print(f"  C0-C3 tokens in body: {m['c_tokens']:>6}   ({pc})")
    print(f"  abstract chars      : {m['abstract']:>6}"
          f"   (lines {m['abstract_lines'][0]}-{m['abstract_lines'][1]})"
          if m["abstract_lines"] else f"  abstract chars      : {m['abstract']:>6}")
    print(f"  longest abs sentence: {m.get('abs_sentence', 0):>6}"
          f"   ({m.get('abs_sentences', 0)} sentences)")
    print(f"  Fig. 1 caption chars: {m['fig1_caption']:>6}")
    print(f"  scope box chars     : {m['box']:>6}")
    front = m["abstract"] + m["fig1_caption"] + m["box"]
    outside = front - (m["box"] if m.get("box_in_body") else 0)
    print(f"  front matter total  : {front:>6}   ({outside} of it outside "
          f"measure_body_chars' window; the scope box is "
          f"{'INSIDE it and is body' if m.get('box_in_body') else 'outside'})")
    print(f"  body chars          : {m['body_chars']:>6}")
    print(f"  body paragraphs     : {m.get('paragraphs', 0):>6}")
    print(f"  longest paragraph   : {m.get('long_para', 0):>6}"
          f"   (worst: {m.get('worst_paras')})")
    g = m.get("causal_after_grounding")
    print(f"  §1 causal vocabulary: {m.get('causal_sec1', 0):>6}"
          f"   (first one {'after' if g else 'BEFORE' if g is not None else 'n/a:'}"
          f" the FL grounding; {m.get('causal_tokens')})")
    print(f"  front-matter jargon : {m.get('jargon_front', 0):>6}"
          f"   ('descendant of the quantity/estimand' in the abstract or §§1-2"
          f"{'; ' + ', '.join(m['jargon_where']) if m.get('jargon_where') else ''})")
    print(f"  design words fixed  : {m.get('design_vocab', 0):>6}/"
          f"{len(DESIGN_VOCAB)}   ({', '.join(DESIGN_VOCAB)}"
          f"{'; MISSING ' + ', '.join(m['design_missing']) if m.get('design_missing') else ''})")
    print(f"  bold / italic runs  : {m['bold_runs']:>6} / {m['emph_runs']}"
          f"   ({m['bold_runs'] + m['emph_runs']} over "
          f"{m.get('paragraphs', 0)} paragraphs)")
    lb = m["long_bold"]
    print(f"  >1 long bold run    : {len(lb):>6} paragraphs"
          + ("".join(f"\n{'':>26}:{n} {r}" for n, r in lb) if lb else ""))
    print(f"  max cross-refs/para : {m['max_xrefs']:>6}   "
          f"(worst: {m['worst_xrefs']})")
    jn = m["joined"]
    print(f"  comment-joined paras: {len(jn):>6}"
          + ("".join(f"\n{'':>26}lines {i}-{j}" for i, j in jn) if jn else ""))
    sp = m.get("spine_span")
    print(f"  spine span (paras)  : {sp if sp else '   n/a':>6}   "
          f"(the review's five beats, in {sp} adjacent paragraph(s); "
          f"its own threshold is {TARGETS['spine_span']})")
    for label, i, line in m.get("spine_where", []):
        print(f"{'':>26}{label:<24} "
              f"{'para %d, :%d' % (i, line) if i is not None else 'NOT PRESENT'}")
    print(f"  body questions      : {m.get('questions', 0):>6}   "
          f"({m.get('body_questions', 0)} unsanctioned; "
          f"{len(SANCTIONED_QUESTIONS)} are allowlisted by phrase)")
    for n_, q in m.get("questions_bad", []):
        print(f"{'':>26}:{n_} {q[:70]}")
    print(f"  provenance framing  : {m.get('provenance', 0):>6}   "
          f"(research-history framing in the body; the FACTS it frames are kept by "
          f"body_homes below, since PROTECTED is a whole-file test)")
    for n_, pat in m.get("provenance_where", []):
        print(f"{'':>26}:{n_} {pat}")
    print(f"  body homes (floor)  : "
          f"{m.get('body_homes', 0)}/{m.get('body_homes_total', 0):<4}   "
          f"(each fact the cut framing carried, at its destination IN THE BODY)")
    for name, ln in m.get("body_homes_where", []):
        print(f"{'':>26}{(':%d' % ln) if ln else 'NO BODY HOME':<13} {name}")
    n, tot, pct = m["negation"]
    print(f"  negation density    : {pct:>5.1f}%   ({n}/{tot} body sentences)")
    pg = m["prop_page"]
    print(f"  first proposition on: {'p' + str(pg) if pg else '   n/a':>6}"
          f"   (the central claim's rendered page)")
    q5 = m.get("q5_pages")
    print(f"  five questions by   : {'p' + str(q5) if q5 else '   n/a':>6}"
          f"   (the review's own 8/10 test; its threshold is p{TARGETS['q5_pages']})")
    for label, p in m.get("q5_where", []):
        print(f"{'':>26}{label:<18} {'p' + str(p) if p else 'NOT RENDERED'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default="paper/main.tex")
    ap.add_argument("--pdf", default=None, help="defaults to --tex with .pdf")
    ap.add_argument("--rev", help="git revision to compare against")
    ap.add_argument("--gate", action="store_true",
                    help="exit 1 unless the round's clarity targets are met")
    args = ap.parse_args()

    tex = ROOT / args.tex
    pdf = ROOT / args.pdf if args.pdf else tex.with_suffix(".pdf")
    now = measure(tex.read_text(), pdf, tex)
    report(args.tex, now)

    if args.rev:
        rel = str(Path(args.tex))
        old = subprocess.run(["git", "-C", str(ROOT), "show", f"{args.rev}:{rel}"],
                             capture_output=True, text=True)
        if old.returncode:
            sys.exit(f"cannot read {rel} at {args.rev}: {old.stderr.strip()}")
        # No PDF for the old revision: the page number is not comparable that way
        # and is deliberately left n/a rather than guessed.
        was = measure(old.stdout)
        print()
        report(f"{args.rev}:{rel}", was)
        print("\n--- delta ---")
        for k, lbl in (("p_tokens", "(P#) tokens"), ("c_tokens", "C0-C3 tokens"),
                       ("abstract", "abstract chars"),
                       ("fig1_caption", "Fig. 1 caption"), ("box", "box chars"),
                       ("body_chars", "body chars")):
            print(f"  {lbl:<16} {was[k]:>6} -> {now[k]:>6}  {now[k]-was[k]:+d}")
        print(f"  {'negation':<16} {was['negation'][2]:>5.1f}% -> "
              f"{now['negation'][2]:>5.1f}%  "
              f"{now['negation'][2]-was['negation'][2]:+.1f} points")

    if args.gate:
        fails = []
        if now["p_tokens"] > TARGETS["p_tokens"]:
            fails.append(f"(P#) tokens {now['p_tokens']} > {TARGETS['p_tokens']}")
        if now["c_tokens"] > TARGETS["c_tokens"]:
            fails.append(f"C0-C3 tokens {now['c_tokens']} > {TARGETS['c_tokens']}")
        if now["abstract"] > TARGETS["abstract"]:
            fails.append(f"abstract {now['abstract']} chars > {TARGETS['abstract']}")
        pg = now["prop_page"]
        if pg is None:
            fails.append("no 'Proposition 1' found in the PDF -- build first")
        elif pg > TARGETS["prop_page"]:
            fails.append(f"first proposition on p{pg} > p{TARGETS['prop_page']}")
        missing = [lbl for lbl, p in now.get("q5_where", []) if p is None]
        if missing:
            fails.append("the review's five-question test cannot be scored: no rendered "
                         f"answer found for {', '.join(missing)} -- either the build is "
                         "stale or the phrase the answer was keyed on was reworded")
        elif now["q5_pages"] and now["q5_pages"] > TARGETS["q5_pages"]:
            fails.append(f"the review's five questions are answered only by "
                         f"p{now['q5_pages']} > p{TARGETS['q5_pages']}, its own "
                         f"threshold for 8/10 clarity: "
                         + ", ".join(f"{lbl} p{p}" for lbl, p in now["q5_where"]))
        for k, lbl in (("bold_runs", "bold runs"), ("emph_runs", "italic runs"),
                       ("max_xrefs", "max cross-refs in one paragraph")):
            if now[k] > TARGETS[k]:
                fails.append(f"{lbl} {now[k]} > {TARGETS[k]} (ratchet: this may "
                             "fall but not rise)")
        if now["abs_sentence"] > TARGETS["abs_sentence"]:
            fails.append(f"longest abstract sentence {now['abs_sentence']} chars > "
                         f"{TARGETS['abs_sentence']}: {now['abs_sentence_text'][:90]!r}...")
        if now["causal_sec1"] > TARGETS["causal_sec1"]:
            fails.append(f"§1 spends {now['causal_sec1']} causal-vocabulary tokens > "
                         f"{TARGETS['causal_sec1']} (ratchet): {now['causal_tokens']}")
        if now["causal_after_grounding"] is False:
            fails.append("§1's first causal-vocabulary token arrives BEFORE the "
                         "federated-learning problem it is about")
        if now["long_para"] > TARGETS["long_para"]:
            fails.append(f"longest body paragraph {now['long_para']} chars of prose > "
                         f"{TARGETS['long_para']}: line {now['worst_paras'][0][0]}")
        if len(now["long_bold"]) > TARGETS["long_bold"]:
            fails.append(f"{len(now['long_bold'])} paragraph(s) carry more than one "
                         f"long bold run beyond a run-in head: "
                         f"{[n for n, _ in now['long_bold']]}")
        if now["jargon_front"] > TARGETS["jargon_front"]:
            fails.append(f"the phrase the review quoted as unreadable is back in the "
                         f"abstract or §§1-2: {', '.join(now['jargon_where'])} (the "
                         f"appendix's formal statements are out of this window)")
        if now["design_vocab"] < TARGETS["design_vocab"]:
            fails.append(f"§§1-2 fix only {now['design_vocab']} of "
                         f"{TARGETS['design_vocab']} design words in italics; "
                         f"missing {', '.join(now['design_missing'])} -- without them "
                         f"the five objects read as five contributions")
        gone = [lbl for lbl, i, _ in now.get("spine_where", []) if i is None]
        if gone:
            fails.append("the review's five-beat core message cannot be located in the "
                         f"body: {', '.join(gone)} not present -- either a beat was "
                         "dropped or the phrase it is keyed on was reworded")
        elif now["spine_span"] > TARGETS["spine_span"]:
            fails.append(f"the core message is spread over {now['spine_span']} adjacent "
                         f"paragraphs > {TARGETS['spine_span']}: a reader has to assemble "
                         "it rather than read it ("
                         + ", ".join(f"{lbl} :{ln}"
                                     for lbl, _, ln in now["spine_where"]) + ")")
        if now["body_questions"] > TARGETS["body_questions"]:
            fails.append(f"{now['body_questions']} unsanctioned question sentence(s) in "
                         f"the body > {TARGETS['body_questions']}: a reader is asked to "
                         "hold a question open instead of being told the answer; lines "
                         + ", ".join(str(n) for n, _ in now["questions_bad"]))
        if now["provenance"] > TARGETS["provenance"]:
            fails.append(f"{now['provenance']} research-history framing phrase(s) in the "
                         f"body > {TARGETS['provenance']}: lines "
                         + ", ".join(f"{n} ({p})" for n, p in now["provenance_where"])
                         + " -- the DISCLOSED FACTS must stay (body_homes below is the "
                         "floor for that); it is the framing that goes")
        if now.get("body_homes", 0) < now.get("body_homes_total", 0):
            lost = [nm for nm, ln in now["body_homes_where"] if ln is None]
            fails.append("a fact the removed framing carried has NO BODY HOME: "
                         + "; ".join(lost) + " -- measure_negation_density's PROTECTED "
                         "floor cannot catch this, it searches the whole file and each of "
                         "these has appendix twins, so it stays 16/16 with the body "
                         "statement deleted. Restore the fact at its destination")
        if now["fig1_caption"] > TARGETS["fig1_caption"]:
            fails.append(f"Fig. 1 caption {now['fig1_caption']} chars > "
                         f"{TARGETS['fig1_caption']} (ratchet: a caption that describes "
                         "what the panels already draw may fall but not rise)")
        if len(now["joined"]) > TARGETS["joined"]:
            fails.append(f"{len(now['joined'])} comment block(s) weld two prose "
                         f"paragraphs into one (a comment line does not end a "
                         f"paragraph): lines {now['joined']}")
        print()
        for f in fails:
            print(f"FAIL {f}")
        if fails:
            return 1
        print("[OK] clarity targets met: (P#) <= %d, C <= %d, abstract <= %d, "
              "longest abstract sentence <= %d, first proposition <= p%d, "
              "all five of the review's questions answered by p%d, "
              "bold <= %d, italic <= %d, cross-refs/para <= %d, "
              "§1 causal tokens <= %d and none before the FL problem, "
              "longest body paragraph <= %d, no paragraph with two long bold runs, "
              "no comment-joined paragraphs, the quoted jargon absent from the "
              "abstract and §§1-2, all %d design words fixed early, the five-beat core "
              "message within %d adjacent paragraph(s), %d unsanctioned body questions, "
              "%d research-history framing phrases, every fact that framing carried "
              "still homed in the body, and Fig. 1's caption <= %d chars"
              % (TARGETS["p_tokens"], TARGETS["c_tokens"], TARGETS["abstract"],
                 TARGETS["abs_sentence"], TARGETS["prop_page"], TARGETS["q5_pages"],
                 TARGETS["bold_runs"], TARGETS["emph_runs"], TARGETS["max_xrefs"],
                 TARGETS["causal_sec1"], TARGETS["long_para"],
                 TARGETS["design_vocab"], TARGETS["spine_span"],
                 TARGETS["body_questions"], TARGETS["provenance"],
                 TARGETS["fig1_caption"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
