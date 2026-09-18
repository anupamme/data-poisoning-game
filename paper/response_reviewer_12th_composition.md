# Response to the twelfth review (5/10, Borderline / Weak Reject, confidence 4/5)

Every line number below is a **source** line in `main.tex`, not a rendered line number in the PDF's
margin; the two differ by roughly four. Line numbers are resolved with a comment-filtered search, since
this file carries provenance comments that quote their own prose.

The review is written against `main(20260917-123317).pdf`, a 12:33 build. The shipped build of the
previous round is 16:40 the same day, so a few items below may already have been fixed in the copy you
did not see. Each of those is marked, with the site, rather than asserted.

---

## Two concessions first, both unhedged

### 1. The flagship Krum arm is at an admission floor, and you are right that the paper printed no number that would let you check it

The concession itself was already in the paper — `main.tex:711` reads "The zero is a floor here" — but the
paper stated the concession and then asserted the conclusion did not rest on it, **without printing a
single baseline level anywhere**. In the build you reviewed there were zero occurrences of any of the
four measured baselines in either document. A reader who took the concession seriously had no way to test the reply, so
generalising the floor to all four aggregators was the reasonable reading, and it is the reading we made
available.

This round prints the ladder. `build_channel_table.py` now emits each arm's pre-transform admitted
fraction, conditioned on the rounds in which an adversary was actually sampled, together with the count
of rounds whose baseline is nonzero — because the level alone is an average and one admitting round can
carry it. Measured, each arm at its own committed attack:

| arm | baseline level | nonzero rounds / adversary rounds | verdict |
|---|---|---|---|
| Krum × scaling (the adjudicating arm) | 0.0000 | 0 / 12 | **at the floor** |
| Krum, EMNIST-byclass | 0.0000 | 0 / 12 | **at the floor** |
| Krum, score-only control | 0.0000 | 0 / 12 | **at the floor** |
| Krum, ResNet18 | 0.0000 | 0 / 15 | **at the floor** |
| `cos_krum` × pixel | 0.0833 | 1 / 12 | **near the floor** |
| Reputation × scaling | 0.0014 | 12 / 12 | has room |
| Coord. median × pixel | 0.2855 | 12 / 12 | has room |

**Two of those rows go further than your objection did.** Every Krum row in the paper is at the floor,
including the EMNIST and ResNet18 replications, so the floor travels with the arm across datasets and
architectures. And `cos_krum` is *near* the floor — 1 admitting round in 12 — which matters because
`cos_krum` carries the largest ASR movement in the table (−0.425). We had not stated either fact.

### 2. A composition-level replication outside CIFAR-10 was the missing piece, and the previous round declined to run one

Your §18.1 and your closing sentence both name it, and you rank it first. We accept the ranking. The
eleventh review said, in its own words, that it *would not add another broad benchmark* and asked for the
conceptual contribution to be made unmistakable; the previous round did exactly that, and the score fell
from 6 to 5. Concept work is not what was missing. This round follows your ranking instead of that one.

**The arm ran, and its premise failed. We report that as the answer to your first-ranked ask.**
`experiments/pre_registration_normclip_cifar100.md` is committed (`9cbd009`, plus an append-only
amendment `2eeff40` described below), stage 1 has run, and it returned one of the three failure literals
the document pre-committed: *arm confounded on this dataset*. Stage 2 never started; its runner reads the
verdict out of the artifact and refuses. So **no composition-level ASR on a second dataset exists**, and
the replication you rank first is attempted-and-not-carried rather than delivered. What it registers, what
failed, and the one respect in which the pre-registration's own guess about the failure was wrong are all
below.

---

## What the floor objection turns out to be a property of

This is the one place where the measurement moved against your reading, and we report it in the body
rather than in a footnote.

Holding Krum, the dataset and the model fixed and varying **only the attack**:

| aggregator | `committed_scaling` | `committed_pixel` |
|---|---|---|
| `krum` | 0.0000 (0/12) | **0.3333 (4/12)** |
| `cos_krum` | 0.0833 (1/12) | 0.0833 (1/12) |
| `reputation` | 0.0014 (12/12) | 0.2507 (12/12) |
| `coord_median` | 0.1564 (12/12) | 0.2855 (12/12) |

Krum admits adversarial mass at baseline in 4 of 12 adversary rounds under the pixel backdoor. So the
zero is not a property of Krum. A model-scaling adversary is a norm outlier and Krum is built to reject
one; the paper's adjudicating arm happens to commit to scaling.

Your §20(c) states the objection as the headline result being an artifact of Krum *never* admitting
malicious clients. That stronger form is measurably false. **The weaker form survives and we keep it**:
the adjudicating arm does commit to scaling, so on that arm the admission channel has no room, and the
paper may not read an admission-level conclusion off it.

The body now says both things, at `main.tex:724`: "but that floor is the attack's, not Krum's, since under the pixel backdoor the same Krum admits baseline mass $0.333$ in $4$ of $12$ adversary rounds".

The same paragraph separates the two channels, which is what the objection conflates. Krum's **decision**
channel changes on 0.733 of rounds — nowhere near a floor — and that is the channel carrying the headline
result. Krum's **admission** channel has no room, and the admission-level reading rests instead on
reputation and coord. median, the two arms with the most room, whose admitted support is unchanged across
all 48 adversary rounds while influence (0.020, 0.033) and ASR (+0.178, +0.098) both move.

The appendix carries the full cross-check, emitted rather than transcribed, at `main.tex:1761`: "It also holds the aggregator fixed"
and varies the attack instead. The same paragraph now decomposes the pooled count it already reported —
`main.tex:1761` reads "is not spread evenly, and the split is the answer to the floor objection" — into 0 rounds for
Krum, 4 for `cos_krum`, and 48 each for reputation and coord. median. The appendix table's caption states
the asymmetry directly at `main.tex:1770`: "The two channels do not have equal room".

The emitter also **asserts** that each baseline is identical at the identity rung and at the top rung,
since a level that moved with the dose would not be a baseline. That assertion is in the code, not in the
prose.

---

## The replication we ran, why it did not carry, and the structural reason it could not have

`norm_clip` → `coord_median`, **CIFAR-100 / `cifar_cnn`**, `committed_pixel`.

**Why this answers §13 as well as §18.1.** `norm_clip` as shipped in this repository computes
`scale = min(1.0, tau / max(norm, 1e-8))` and multiplies each client update by it. That is exactly
`T(u_i) = c_i u_i` with `c_i > 0`: a member of the theorem's positive per-client rescaling class, and a
standard, deployed, **oracle-free** defense. It needs no adversary mask, unlike every Mode-S, Mode-A and
Mode-M rung in the paper. So the class you call a restricted laboratory contains something people
actually run, and the dose is that defense's own hyperparameter rather than a synthetic coefficient
vector. This is also the substantive answer to §8: the arm does not buy statistic-channel isolation with
an oracle, it has to earn it by measurement, and it may fail to.

**The premise was a separate stage with its own artifact, and stage 2 was gated on it.** Three pass/fail
conditions, all frozen before any number existed: (i) `coord_median`'s baseline admission on CIFAR-100 is
at least 0.05 *and* nonzero in at least half the adversary rounds — both halves, because `cos_krum`
clears the level test on one admitting round in twelve; (ii) at the identity rung every coefficient is
exactly 1 and every channel displacement is exactly 0; (iii) the adversarial coefficient share does not
move beyond the frozen tolerance across rungs. Condition (iii) was flagged in advance as the honest risk
of using a real defense instead of an oracle, and as the condition most likely to fail. It is the one that
failed.

**What stage 1 measured** (`results/normclip_cifar100_admission.json`, 3 seeds × 6 rounds at each of two
local-epoch counts, no ASR anywhere in it):

| condition | outcome | measurement |
|---|---|---|
| (i) nonzero baseline admission | **PASS** | `coord_median`'s adversarial argmedian fraction on CIFAR-100 is 0.2885, nonzero in 15 of 15 adversary rounds. The CIFAR-10 value is 0.2855 — the admission level itself replicates across the dataset change to three decimal places |
| (ii) exact identity rung | **PASS** | every coefficient exactly 1, every aggregate displacement exactly 0, no decision and no admission change, at both epoch counts. Tested with `== 0`, not against a tolerance |
| (iii) coefficient share does not move | **FAIL** | the adversarial share of coefficient mass moves 0.2533 → 0.2603 at τ\*, a spread of 6.9 × 10⁻³ against a frozen tolerance of 10⁻⁶. 45 of 90 client-rounds clip, smallest coefficient 0.838 |

So Δ_c ≠ 0, the arm measures the attenuation channel rather than the statistic channel, and it cannot
speak to the dissociation. Stage 2 did not run.

**The pre-registration's own guess at the mechanism was wrong, in sign, and we report that too.** The
literal it pre-committed says the clip *attenuates adversaries*. It does the opposite. Mean coefficient
over adversarial clients: **1.000000** — not one adversary was clipped in any of the 15 adversary rounds.
Mean over benign clients: **0.962780**. So Δ_c is **+0.037**, and the adversarial share rises because the
clip bites *benign* clients only. Under a pixel backdoor the adversarial updates are not norm outliers;
they sit slightly below the benign median, so a clip placed at that median is a benign-only attenuator.
We report the frozen literal as the verdict, because that is what was pre-committed and what the artifact
records, and we report the measured direction beside it rather than letting the literal's wording stand as
a finding.

**The row it corrects is already in our own supplement, and so is the observation that the row is
vacuous — what we had not done is draw the consequence.** Table S3's Δ_c row for NormClip against the
pixel backdoor reads E[c\|A] = 1.000, E[c\|¬A] = 1.000, Δ_c = 0.000, next to −0.721 for the same
transform against model-scaling. Our supplement already says why that zero is not a finding, in
`supplementary.tex:353`: *"$\Delta_c = 0$ there for a reason the invariance classification cannot see"*.
The clip never binds on this attack (ρ = 1.00 exactly, max client norm 3.04, 0 of 9 rounds), so the
difference of two untouched means is zero by construction.
We are not correcting that sentence; we are reporting that we stated it about one row and did not see
that it generalizes into an impossibility for the whole family, which is the next paragraph. It is also
the same shape as the objection you raise about our flagship Krum cell — a zero read as a property of the
mechanism when it is a property of the operating point — and the version in our own table is one we had
already diagnosed and then left standing as a table entry.

Placing τ at the median makes the clip bind in 45 of 90 client-rounds, and Δ_c is then **+0.037**. That is
the first non-vacuous measurement of NormClip's Δ_c against the pixel backdoor, it is on the same axis as
Table S3's rows, and its sign is the reverse of the one negative row NormClip had. It is a result about a
deployed defense at a τ people would actually choose, not about our instrument — and it is why this arm
cannot answer your question.

**The structural point, which is worth more than the arm would have been.** Put the two rows together and a
norm clip cannot serve as a Δ_c-neutral, oracle-free probe of the statistic channel at all, on any dataset.
Its coefficient is c_i = min(1, τ/‖u_i‖), so there are only two regimes. If τ sits above every client norm
the clip does not bind, every c_i is exactly 1, and Δ_c = 0 **by construction** — that is our supplement's
0.000 row and it measures nothing. If τ binds, then Δ_c is a difference of two group means of
min(1, τ/‖u‖) and it is zero only by coincidence; its sign is set by which group's norms sit higher, which
is a property of the attack. Model-scaling puts the adversary above the median, giving Δ_c = −0.721. The
pixel backdoor puts it just below, giving Δ_c = +0.037. **Neutrality and informativeness are mutually
exclusive here**, and no choice of τ, dataset or downstream defense escapes that, because it is a fact
about the clip's functional form rather than about our configuration.

So this is not a failed arm we might retry with a better τ or on a third dataset. It is a negative result
about a whole family of candidate probes: the reason the paper's Mode-S rungs need an adversary mask is
not laziness, it is that pinning every adversarial coefficient at exactly 1 while spreading the benign
ones is something no oracle-free per-client rescaling does for free. Your §13 asks whether the restricted
class is a laboratory or a real setting. The answer we can now defend is sharper than either: the class
contains real deployed defenses, and *that is exactly why* its members cannot be used as clean channel
probes — the properties that make a defense deployable are the ones that make its Δ_c depend on the
attack. We would rather report that than a four-hour ladder we could not interpret.

**τ was fixed by a rule before the number existed** — the median client update norm — and stage 1
measured it as 4.0825. It was not re-chosen after the failure, and we are not reporting a second τ:
searching for a τ that passes condition (iii) after seeing that this one fails is exactly what
pre-registering the rule was for.

**An append-only amendment, and it is what caught the failure.** Writing the runners exposed a defect in
the design we had already frozen: stage 1 mirrors the existing channel measurement, which trains **one**
local epoch, while every ASR ladder in the paper trains **two**. That is harmless for the paper's other
dose families, whose κ is a scale-free dispersion dial, and fatal for a norm threshold, which has units.
Amendment `2eeff40` was appended before either runner existed and before any artifact: stage 1 traverses
both epoch counts, τ\* is taken from the two-epoch pass because that is the configuration the clip would
have faced, and conditions (ii) and (iii) must pass at both. **At one epoch, τ\* = 4.0825 clips 0 of 90
client-rounds and condition (iii) passes with a spread of exactly 0.** The failure is visible only at the
ladder's own epoch count. Had we measured the premise in the configuration the rest of the paper is
measured in — the natural choice, and the one the frozen document originally implied — the arm would have
been certified eligible and stage 2 would have run a confounded ladder for four hours and reported a
composition-level dissociation we could not have defended.

**The demotion clause pre-committed three verdicts as literals**, so the runner reported one rather than
composing prose after seeing the numbers: premise failed, arm confounded on this dataset, or the
dissociation reverses. It returned the second. Had it returned the third, that would have been this
round's headline, reported against us, in the abstract and in Figure 1.

**The negative is reported where the freeze said it had to be, which is not a footnote.** The
pre-registration required that a failed premise be stated in the abstract and in §5, and both now carry it.
The abstract's last clause on the replication reads, at `main.tex:66`: "an oracle-free replication was attempted and did not carry".
§5's closing paragraph is titled *The oracle-free version, attempted*, and states, at `main.tex:852`: "the two-defense replication on a second dataset was attempted and did not carry".
The detail sits in a new appendix subsection whose title, at `main.tex:2000`, is "attempted, and not established".
Its opening paragraph closes the inference we most want closed, at `main.tex:2003`: "no composed-pair ASR result on a second dataset is reported anywhere in this paper".
That sentence also had to be kept from reading as a retraction of something we do report, since §5 does carry a CIFAR-100 dose ladder (the reversal cell, $n{=}5$), so the same line now separates the two objects, at `main.tex:2003`: "its upstream transform is the Mode~S instrument rather than a deployed defense".
Only the numbers moved
to the appendix; the arm's existence and its verdict are in the body. The one asymmetry we should name: the
body paragraph is one sentence, because the page limit would not fund more, and the two facts it therefore
does not print are Δ_c = +0.037 and that the clip binds on the **benign** clients. Both are in the appendix,
and a source comment above the body paragraph records that those are the two to restore first if a later
revision frees lines, since that direction is the part a reader will disbelieve.

**Your §7 survives this round intact, and more completely than it would have.** The pre-registration
already conceded that the arm changes the dataset and the label space while keeping `cifar_cnn`, and that
`N = 10`, `K = 5`, `f = 0.2` are untouched — so even a successful ladder would have been a partial answer
to §7. Since no ladder ran, the narrow-configuration objection is untouched by this round in every respect,
and we are not claiming otherwise anywhere in the paper.

**FEMNIST was considered and rejected, and the first reason we had was wrong.** FEMNIST would have been
the stronger answer to §7 because `simple_cnn` changes the architecture too, and its channel artifact
already exists. We initially recorded that `coord_median`'s FEMNIST baseline was unmeasured. It is
measured, and it is 0.1029, nonzero in all 12 adversary rounds — it would pass the premise. The
disqualifying fact is the **attack**: every FEMNIST measurement is under `committed_scaling`, and a
scaling attack *is* a norm outlier, so a FEMNIST norm-clip arm would very likely fail condition (iii) and
answer a different question than you asked. The corrected rationale, and the wrong one it replaced, are
both in the pre-registration. **That prediction now reads as too narrow rather than wrong.** We expected a
scaling attack to fail (iii) because the adversary is a norm outlier; the arm we chose instead failed (iii)
with an adversary that is *not* a norm outlier, by the reverse mechanism. The structural point above is
what both cases are instances of, and it means switching to FEMNIST would not have rescued the design.

---

## Your other items, each with its site

### Already implemented before this review, with the measurement

| item | site and measurement |
|---|---|
| **§6, §16** — demote the screen, treat C1 as a budget heuristic; the item you call *the single biggest revision I would recommend* | The criterion is **appendix-only**. Its environment title at `main.tex:1210` reads "an evaluation prioritizer, never a security screen". `C1` occurs 2 times in the 912-line body window against 133 times whole-file, both counts excluding comments. §6.3 is titled `main.tex:949`: "What the screen cannot find, and what to run instead" |
| **§4** — do not let the literature audit carry the motivation | The audit paragraph was deleted from the body in the previous round. The appendix keeps every clause and its own guard against overclaiming |
| **§8** — Mode S does not isolate the statistic channel (Δagg = 0.892) | Already a titled body paragraph printing that number, `main.tex:794`, and already calling the result a dissociation rather than irrelevance. We tried to answer it with evidence instead of prose and the attempt failed; the negative result above is what we have, and it says an oracle-free probe of this kind is not available rather than that we did not look |
| **§12** — push the confidence tiering into the main paper | `main.tex:882`: "Four questions, four kinds of answer" — in the body, with the four kinds named and stated as never pooled |
| **§11** — the adaptive evidence is a boundary demonstration | Agreed, and the paper already bounds it that way. Your own text says it is not required |
| **§5** — the criterion's 90.5% against an 88.1% baseline is uninformative | Both numbers are printed and the criterion is titled as a prioritizer, not a predictor. This is our own result, not a claim we defend |

### Changed this round

| item | change |
|---|---|
| **§10** — P4 is not a uniform quantity across aggregators | Moved **into** the body's level definition, `main.tex:516`: "is not a single metric, so we never compare its magnitude across aggregators". It was appendix-only before |
| **§14** — compress the audit trail into a concise evidence-status table | New table at the head of appendix §I, `main.tex:2821`: "The evidence ledger in one object" — 17 rows, columns claim / tier / n / status, statuses proved, supported, refuted, not established and open. The prose is retained below it; nothing was deleted. The n column is deliberately non-uniform and the caption says the rows are never pooled |
| **§18.5** — causal-language discipline | `main.tex:2806` no longer reads *is indeed the operative variable*; it reads "the distinction the two readings turn on in this arm". Zero remaining occurrences of the phrase in either document |

### Not done this round, and disclosed rather than promised

| item | disposition |
|---|---|
| **§18.3, §12** — reduce reliance on n=3 and n=5 | A pre-registration for the seed top-up of the four fragile cells is written and **unrun**. It stays unrun this round. We are not promising it; it is on disk and you can see what it would test |
| **§7** — experimental scope (N=10, K=5, f=0.2 throughout) | **Not addressed.** The arm that would have changed the dataset and label space did not reach its ladder, so no new configuration was run at all and the objection stands in full |
| **§15** — sharpen positioning | Not attempted as a separate edit this round. The concessions above are the honest version of the positioning |

---

## One thing the review did not catch, reported because it is the same class of error

`main.tex:1030` claimed a count of "pre-registration documents" that is checkable against
`ls experiments/pre_registration_*.md`. Writing this round's pre-registration made it stale: the sentence
read 27 while the glob read 28. No gate, hash or build can see that happen, because the claim is about the
file system rather than about the document. It is corrected, and the glob is now re-counted in the
verification list of every round that adds or removes a document.

---

## Verification state

Three instruments gate this document and all three pass on the shipped tree, with no ratchet raised: the
page gate (main text 9 pages, the Conclusion's last rendered line on page 9), the clarity gate (every
ceiling met at or below its previous value), and the protected-disclaimer floor at 16 of 16. Both
documents build to a fixpoint with zero LaTeX errors, undefined references, multiply-defined labels or
overfull boxes. The baseline column was cross-checked against an independent recomputation from the
frozen channel artifact before any of it reached prose.

**Two disclosures about this letter.** First, `main.tex` contains 84 `:NNN` line cites inside its own `%`
comments, across 60 comment lines, which this round's insertions shift further out of date. They are flagged here rather
than patched to plausible values, because a comment cite that lands on real but wrong content passes every
automated check we have. Second, the rendered line numbers in the PDF margin are not the source line
numbers cited above.
