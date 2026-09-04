# Response to the reviewer who scored 6/10 (Weak Accept / Borderline, confidence 0.82)

Thank you for a review that told us exactly what was missing. You wrote that the paper proves a
particular evaluation strategy cannot identify mechanism preservation and then demonstrates it
carefully on a narrow FL setting, and that an 8/10 needs the reader convinced this is a *general*
methodological failure. You named one thing above all others:

> Run a blind, pre-registered, multi-cell causal replication of the −0.272 vs +0.098 phenomenon and
> make that the main empirical contribution.

We did. The result is not the clean one we hoped for, and we are leading with that.

---

## 1. The multi-cell replication, and the mechanism it killed

We pre-registered a six-cell replication before running anything
(`experiments/pre_registration_comparability.md`, frozen at commit `c986ef4`). Four cells already
existed in frozen artifacts and are **re-scored, never re-run**; two did not exist and were run for
this purpose. The runners refuse to start unless the pre-registration is committed at a recorded hash,
and the analyzer refuses to print any pooled number unless all four published contrasts first
reproduce bit-identically.

| cell | outcome-gated | Mode S | ΔΛ_a | predicted | observed |
|---|---|---|---|---|---|
| `krum` / scaling | −0.008 | −0.010 | 0.0000 | AGREE | AGREE |
| `cos_krum` / pixel | −0.405 | −0.425 | 0.0000 | AGREE | AGREE |
| `reputation` / scaling | +0.762 | +0.178 | 0.0196 | DISAGREE | DISAGREE |
| `coord_median` / pixel | −0.272 | +0.098 | 0.0332 | DISAGREE | SIGN REVERSAL |
| `coord_median` / scaling * | +0.002 | +0.196 | 0.0086 | DISAGREE | DISAGREE |
| `krum` / scaling, EMNIST-byclass * | +0.157 | −0.013 | 0.0000 | AGREE | **DISAGREE** |

\* out-of-sample; the other four are the cells the rule was read off.

**The two designs disagree on four of six cells**, across two attacks, two datasets and two
architectures. The cell that matters most for your question is `coord_median` under *model scaling*:
that is the sign-reversal aggregator under the **other** committed attack, and the two designs
disagree there too. **The phenomenon is not specific to the pixel backdoor.** It is a disagreement,
not a second sign reversal, and we do not report it as one.

**We also pre-registered a mechanism, and it is refuted.** Before the new runs we froze a rule for
*when* the designs disagree: exactly when the downstream defense admits adversarial mass, gated on the
admission-level change ΔΛ_a. The four existing cells separate perfectly under it, which is precisely
why we labelled them **training data that cannot also be evidence for it**. The rule mechanically
generated a prediction for each new cell, committed in advance. One confirms. **The EMNIST cell
refutes it**: ΔΛ_a is exactly 0.0000, the rule predicts the designs coincide, and they do not.

Per the clause we froze in advance, the mechanism is **withdrawn**, the four-cell pattern is demoted
from a mechanism to a description, the table is reported in full, and **no threshold was fitted
afterwards to rescue it**. This is §5 and Appendix E.3 of the revision.

We think the refuted version is the more useful contribution, and not only because it is what
happened. A mechanism that survived its own out-of-sample test at 2/2 would have been a rule read off
four cells and confirmed on two, which is thin. What we can now say is sharper and better supported:
**the disagreement between the two designs is real, it reproduces on a second attack and a second
dataset, and we do not have an account of when to expect it.** That last clause is a limitation we
state rather than paper over, and it is the direct reason the protocol in §4 tells you to intervene
rather than consult a rule.

---

## 2. "This is standard collider reasoning applied to a narrowly defined FL evaluation protocol"

This is the objection you named as the one that would sink the paper, and you also suggested the shape
of the answer. We adopt it.

**What is standard, and we claim no credit for:** conditioning on a descendant of two causes opens a
non-causal path between them. Textbook.

**What is not standard, and is the paper's actual claim:** inherited-suppression gating does not merely
*permit* that structure, it **constitutes** it. A screen that admits a candidate pair only when a
constituent already suppresses the attack is, by construction, conditioning on a descendant of both
the mechanism and the defense's standalone strength. So the collider is not a modelling assumption a
reader may decline, and not an artifact of how we drew the graph. It is a property of the evaluation
design itself, and it holds for **any** screen of that form, ours included. Proposition 6 is stated
and proved over a generic metric M and threshold θ with no dataset, architecture, attack, or defense
in it (§3 and App. B). We have now made that prominent rather than leaving it to be inferred: the
statement carries the sentence **"Nothing in this statement is about federated learning"**, and names
FL defense composition as the instance we can measure.

**What is new evidence, and is the reason this is not just a diagram:** the bias is not merely
theoretical or a matter of lost power. Across six pre-registered cells the confounded estimator
disagrees with the controlled one on four, and on one it is **differently signed**: −0.272
[−0.422, −0.122] against +0.098 [+0.018, +0.178], intervals that exclude zero and do not overlap. We
know of no other measurement of this in a live security-evaluation methodology. Collider bias is
usually taught as attenuation; here it reverses the direction of the measured effect.

---

## 3. Your specific requests

**(1) Generalize Proposition 6.** Done as a prominence fix, since the content was already abstract.
The proposition and its proof use no property of federated learning, backdoors, or our defense menu,
and the paper now says so at the point of statement, followed by the general form: any evaluation that
gates candidate pairs on constituent success cannot identify the downstream mechanism's marginal
contribution from between-defense contrasts.

**(3) Reframe P1 to P5.** Done. (P1) ⇒ (P2) ⇒ (P3) are logical implications that hold by definition, and
the paper said so already. What was missing is the **causal boundary**: there is now a visible break in
the table between (P3) and (P4), and the text states that (P4) and (P5) are causal properties whose
relations must be *tested*, which is why the two non-implications are demonstrated by intervention
rather than asserted. On your sharpest point: **(P4) is a level instantiated per aggregator, not one
scalar comparable across defenses.** Our (P4) result is four within-family statements, never a
cross-defense quantity; the instruments are not commensurable and we never difference or rank across
them. That is also why our channel table does not order the arms.

**(4) Make the sign reversal the hero.** Done. The abstract now opens on the general claim and
reaches the reversal in its own sentence ("Ignoring this does not cost power, it costs the sign"), and
Figure 1(c) is the six-cell panel with the reversal marked, training and out-of-sample cells separated
by a rule, and the refutation visible in the figure itself.

**(5) FG→RFA as emergent synergy.** Framed as the distinction between **inheritance** (a constituent
already suppresses and the composition keeps it) and **emergence** (neither does and the pair does),
with the result that no inherited-suppression screen can discover the second. Our own best composition
is in the second class *a priori*, not by measurement.

**(9, 10) Evaluation budget and the frontier.** Built from existing numbers, no new runs:

| variant | selected | precision | recall | composition runs | reaches FG→RFA |
|---|---|---|---|---|---|
| no screen | 42 | 12% | 100% | 420 | yes |
| C2 alone | 14 | 36% | 100% | 140 | yes |
| C1 ∧ C2 | 2 | 100% | 40% | 20 (+60 standalone) | no |

The 40% recall you flagged is an **operating point on a measured frontier**, not a ceiling. And the
blind spot is attributable to exactly one named condition: every C1-bearing variant misses FG→RFA and
every C1-free variant reaches it, because C1 is the inherited-suppression gate that Proposition 7
proves cannot certify emergent synergy. Dropping C1 recovers the pair and costs precision. That is a
trade, and it is now reported as one (App. D.4).

**(14) Defensiveness, and length.** The body is shorter than the version you read, not longer, despite
adding the six-cell result. Several defenses were relocated to the appendix sites that already carried
them in full.

---

## 4. Two things we are not doing, said plainly

**No code package.** You argued for release and the argument is a good one; we are not persuaded it is
free, and we would rather say so than imply an oversight. What we offer instead is narrower and
checkable: **every number in the paper is emitted by a named generator that reads only frozen
artifacts**, no number is retyped, each pre-registration is committed *before* the runs it governs, the
runners refuse to start on a hash mismatch, and the analyzers refuse to print when previously published
values stop reproducing. Section 1 states that no code or data package accompanies the submission, and
that remains true. We accept that this is weaker than release and that you may weigh it accordingly.

**No new attacks, datasets, or aggregators.** You explicitly advised against spending remaining effort
on breadth, and we agree. The two new cells exist to complete *comparability contrasts on arms the
paper already has*, which is the opposite of breadth: 68 runs, all of them the same experiment as the
four they are compared against, proved so by a harness check that reproduces a published cell to
0.000e+00 before the suite is allowed to start.

---

## 5. A version caveat, stated so you can discount it

Your review does not mention coordinate masking, the n=20 top-up, TOST, or the second transformation
class. All four are in the version we are submitting, and all were completed on 2 September 2026. We
think the PDF you read predates them. We are not claiming credit for work you could not see, and we are
not asking you to re-review from scratch; we mention it only so that if you recall the paper as having
a single invariance class and n=5 throughout, that is a real difference and not a misreading.

---

## 6. One process failure, disclosed because it nearly reversed the headline

The EMNIST cell was first scored with its two designs on non-identical seed sets (a new n=5 against a
frozen n=3), and the disagreement was carried entirely by two seeds that had no counterpart in the
other design. We amended the pre-registration to **complete** the controlled ladder to the frozen seed
set rather than resolve the cell on the three shared seeds, because the shared-seed subset gives AGREE
and choosing it would have chosen the answer. The completed comparison is the fully paired n=5 against
n=5 in the table, and it still disagrees.

In that same amendment we asserted that pairing the two designs across a shared seed set was the
convention the original freeze had used. **It was not.** What caught the error was not our judgement
but a mechanical gate: under cross-design pairing the four *published* cells stop reproducing
(`krum`/scaling's controlled ladder moves −0.010 at n=20 to −0.026 at n=5), and the analyzer refused to
print. A third amendment withdraws the claim.

We are reporting this because the sequence is the honest one: a refutation, an amendment that would
have undone it, and the check that stopped the amendment. All three are in the pre-registration, in
order, and none of them was removed. We would rather you see the machinery working than a clean
narrative that hides the moment it was needed.
