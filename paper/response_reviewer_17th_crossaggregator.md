# Response to the seventeenth review (6/10, weak accept / borderline, confidence 4/5)

Thank you. The review names six priorities: make the causal counterexample the paper (P1); demote the
screening criterion (P2); make the oracle limitation impossible to miss in the main paper (P3); reduce
Krum's dominance (P4); separate what is general from what is FL-specific (P5); and cut 15–25% of the
meta-narrative (P6).

**Two of the six were real and open, and we fixed both.** P4 is the substantive one: the review is right
that Krum dominates the paper, and it is right for a reason we had not noticed — *the abstract's own
memorable pair was never a Krum result at all*, and the abstract did not say so. P5 was simply absent.
Both are repaired by making the paper more accurate rather than by adding claims.

**One new pre-registered arm was run, and it answers P4 at power while contradicting our own freeze in
part.** `coord_median` under model scaling is now at $n{=}20$ in both designs, which makes CoordMedian —
a coordinate-wise robust statistic rather than a selector, so structurally unlike Krum — the paper's
best-powered aggregator across *both* committed attacks. One of the two refuting branches our freeze named
in advance fired, and a third outcome fell outside both branches; §D.2 reports that as a defect in the
freeze rather than rounding it to the nearer branch we had named.

**Three of the six are already done in the live paper and we say so by supersession, once and without
complaint.** The reviewed PDF is several rounds stale. Section 5 below is the crosswalk with the measured
counts.

**On the score.** The review's own estimate for its six items is *the 6–7 range*, and we accept that as
the honest ceiling for this round's changes. The one lever beyond the six that would matter most — an
oracle-free replication of the central negative — is measured, and it went against us. We state that in
§6 rather than arguing past it.

Source line cites in this letter are `main.tex` source lines, not the PDF's margin numbers, which are
ICLR rendered line numbers and differ by roughly four.

---

## 1. P4: Krum's dominance, conceded — and the headline was never Krum's

### 1a. The defect was worse than *Krum appears too often*

The abstract's memorable pair is $-0.273$ against $+0.125$. Resolved against the frozen artifacts, that
pair is **`coord_median` / committed-pixel at $n{=}20$ in both designs** — the paper's best-powered cell,
and one where both margin legs are genuine tests rather than one being satisfied by arithmetic. The
abstract named no aggregator, and the body paragraph that quotes the pair identified its cell only by
back-reference, so a reader had to chain back two sections to learn whose cell it was. A reviewer counting
aggregator mentions would reasonably conclude the result was Krum's. That is our failure of attribution,
not the review's failure of attention.

It now names the aggregator in both places. The abstract reads *"On one \texttt{coord\_median} cell the outcome-gated design moves $-0.273$"* `:72`, and the body paragraph that quotes the pair names it too, in the clause *"sits on this identical \texttt{coord\_median} cell"* `:849`.

### 1b. CoordMedian is now the best-powered aggregator on both committed attacks

| cell | $n$ (both designs) | outcome-gated | within-defense (Mode S) | verdict | cite |
|---|---|---|---|---|---|
| `coord_median` / pixel | $20/20$ | $-0.273\ [-0.334,-0.213]$ | $+0.125\ [+0.095,+0.155]$ | sign reversal | `:1966` |
| `coord_median` / scaling | $20/20$ | $+0.062\ [+0.023,+0.102]$ | $+0.210\ [+0.173,+0.247]$ | disagree, magnitude | `:1968` |

Both rows are held-out cells. The second row is this round's new arm (§4 below); before it, that row stood
at $n{=}5$ on both legs and its outcome-gated interval contained zero.

### 1c. Krum is the *flat* case, and its flatness is cell-specific, not a property of Krum

This is the part of P4 we think the review under-weights rather than over-weights. Krum's flagship arm
moves ASR $-0.010\ [-0.032,+0.012]$ — near-zero, which is why it adjudicates the negative. But the *same*
instrument, at the *same* endpoint, on the *same aggregator*, on a cell where Krum's baseline admission
sits **off** its floor, moves ASR $-0.2857\ [-0.4842,-0.0871]$ at $n{=}20$. Two Krum cells, effects $0.28$
apart. That arm had no body presence at all in the reviewed PDF; it now surfaces in the paragraph that
states the admission-floor premise, which reads *"the same masked dose moves ASR $-0.2857$ at $n{=}20$"* `:735`.

Three constraints we hold ourselves to on that arm, all stated in the paper:

- it measures ASR only — no decision and no admission quantity — so **it may not be said that a statistic
  disturbance costs or buys suppression there**, in either direction;
- its controlled leg is masked, so it bears nothing on the oracle objection of §6;
- neither cell generalizes to the other, and **no statement of the form *the dissociation holds generally*
  is licensed by it.**

### 1d. Mention counts, measured rather than asserted

Over the body window, Krum (excluding `cos_krum`) falls from 25 mentions to **23** and `coord_median` rises
from 4 to **8**. We report that honestly as a modest change: the substantive repair is *which aggregator
carries the headline claim*, not the histogram. Krum remains the arm that adjudicates the flat negative,
because it is the arm where the negative was pre-registered, and moving that would be revisionism.

---

## 2. P3: the oracle limitation is now in the abstract

Conceded. The limitation was in six body sites and the Ethics statement but not in the abstract, which is
the only text many readers see. The abstract now states it in the sentence that explains the mechanism,
merging rather than appending so that nothing was displaced: *"the adversary's coefficient pinned by an oracle no server has: identification, not a deployable test"* `:72`.

That is consistent with, and not a restatement of, the Ethics statement's more precise version: the
instruments are *"measurement devices, not deployable defenses, for two"* `:1015` different reasons, of
which *"the first two require knowing which clients are malicious"* `:1016`. The six body sites are `:470`,
`:628`, `:696`, `:870` and `:1000`, plus the abstract at `:72`.

---

## 3. P5: the three-layer split, in the paper's own words

Absent before; now the paragraph the review asked for, at no net character cost (its previous
*four things around it are not* framing was redistributed rather than deleted). It is titled *"What is general, what is FL theory, what is FL evidence."* `:366` and splits as:

| layer | content | cite |
|---|---|---|
| general | the estimand, which level of a preservation hierarchy a transform preserves, and that *"no outcome-gated test identifies it"* | `:366` |
| FL theory | that hierarchy, its invariance conditions per mechanism, and that *"its lower levels are not surrogates"* for it on the instruments as built | `:366` |
| FL evidence | *"choosing between the two admissible designs does not blunt the effect on one cell, it reverses its sign"* | `:366` |

The general layer is where we think the paper's transferable content is, and it has no FL in it: it is a
statement about the shape of a screen. The paper says so in the evidence-status section: *"The two boundary results are general"* `:3018`.

---

## 4. The new arm: `coord_median` / scaling at $n{=}20$, and the branch of our own freeze that fired

`experiments/pre_registration_coordmedian_scaling_n20.md`, committed **alone** at `74044f1` with the output
path verified not to exist, so `git log` orders the freeze before the data rather than our word doing it.
45 runs at seeds 47–61, two endpoint rungs only, into `results/coordmedian_scaling_topup/summary.json`.

It is 45 runs and not 60 because at $\kappa{=}0$ the two designs are the same computation — and **that was
established on this attack rather than inherited** from the sibling pixel cell: four harness runs at seed 42
reproduced both published $\kappa{=}0$ rows and both published $\kappa{=}2$ rows at $|\Delta| = 0$ in ASR
*and* clean accuracy, and that verdict dict is persisted into the artifact instead of left in a terminal.

**The freeze named two refuting branches in advance. The second fired.** The outcome-gated interval, which
contained zero at $n{=}5$, excludes it at $n{=}20$. The first did not fire: had the paired difference
between the designs covered zero, this cell would have gone DISAGREE to AGREE and the paper's count of five
disagreeing cells would have fallen to four, in the abstract and in Figure 1. So the label survives and the
reason for it does not — the paper states it as *"at $n{=}5$ one design detected nothing"* `:1982` where the
other detected an effect, and at $n{=}20$ both detect one and the disagreement is about magnitude.

**A third outcome fell outside both branches, and we report it as a defect in the freeze rather than
assigning it to the nearer branch.** The paired difference is $+0.147\ [+0.103,+0.191]$ at $n{=}20$ against
$+0.194\ [+0.108,+0.281]$ at $n{=}5$. Its *sign* verdict is the one we anticipated; its *margin* verdict
moved from outside the frozen $0.15$ to inside it, and our branches were written over the sign alone. So
the two designs are separated from each other **and** separated by less than the margin this paper calls
practically meaningful, at the same time, and neither half may be reported as the other `:1984`.

Two further disclosures on this arm, both in the paper:

- the property the cell was chosen for survived its own top-up: the identity rung's mean ASR is $0.557$ at
  $n{=}20$, above the frozen $0.15$, so the largest available fall still exceeds the margin and **both legs
  remain tests**. That figure is an $n{=}20$ level and is *not* the $0.519$ the paper prints elsewhere for
  the same quantity at $n{=}5$;
- endpoint-only has a declared price: the four rungs of this ladder now hold $[20,5,5,20]$ seeds, so **no
  trend statistic is computed on the new seeds and none is quoted.**

Per-seed values for all 20 seeds on all three legs are now printed at `:1995`, which is what the
Reproducibility statement promises for every arm and what the new arm had made false.

---

## 5. P2, P6 and §13: already done, and answered by supersession

Given once, with the measured counts, and not restated as if new.

| review item | state in the live paper | cite |
|---|---|---|
| P2, demote the screening criterion | The screen's precision/recall rates appear **nowhere** in the body. What survives is the structural recall bound and one placement sentence: *"the criterion sits at the high-precision, low-recall end"* | `:980` |
| P6, cut the meta-narrative | 7 of ~209 body sentences carry research-history framing; the `provenance` counter in `measure_clarity_load` reads **0** in the body window. What the review reacted to is in the appendix, where the evidence-status section is a claim-by-claim ledger, not narrative | `:3018` |
| §13, Theorem 8 as machinery | `thm:bounded_reweight` has three body sites and is framed as machinery for the invariance classes, not as a predictive security theorem: *"beyond this paper is the design they license and not our screen"* | `:436` |
| §12, elevate P1→P5 | §5 plus a six-row crosswalk. Note the `(P#)` token budget in the body is at its gated cap of 7, so the promotion was paid for by demotion, not addition | `:366` |

We are not claiming credit for these as new work. They were done in earlier rounds; the reviewed PDF
predates them.

---

## 6. The ceiling, stated plainly

The review's own estimate for its six items is the 6–7 range, and we think that is right. The lever that
would move it further is not on the list: an **oracle-free** replication of the central negative. We have
attempted it three times and it has not carried, and the third attempt contradicts it. The paper says so in
one paragraph: *"three oracle-free attempts did not carry, the third contradicting it"* `:72` in the
abstract, and in full at `:870`, where norm clipping on CIFAR-100 failed its premise, RFA-into-Krum at
$n{=}20$ refuted our flatness prediction at $+0.297$, and the third — pinning the adversarial share to
$10^{-7}$ — found that *"a decision flip alone carries $+0.322$, so the negative is not established"* `:870`
without an oracle.

We do not write that the negative reproduces oracle-free, because it does not. The new arm of §4 above is
masked in its controlled leg and is not a step toward closing that gap either; we say so at `:1984`.

What we claim is narrower than a deployable test and, we think, is what the paper is for: an identification
procedure, a hierarchy that makes preservation decidable per mechanism before any run, and a measured
counterexample showing that the design choice sets the *sign* of the inferred effect rather than merely its
precision — now on two attacks, at $n{=}20$, on an aggregator that is not a selector.

---

## 7. Two things the review asks for that we cannot supply

- **Multi-Krum (§26)** is arithmetically FedAvg at this paper's $K{=}5$, so running it would add a row that
  duplicates an existing one under a different name. We would rather say that than pad the table.
- **A rule for when the two designs disagree.** We pre-registered one, it was refuted out of sample, and it
  is withdrawn. The paper reports the refutation as a refutation and states the gap: *"What we do not have is an account of"* `:1997` when to expect the disagreement, which is a direct limit on how far the
  protocol can be turned into a rule of thumb.
