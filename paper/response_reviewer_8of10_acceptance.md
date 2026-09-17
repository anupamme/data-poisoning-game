# Response to the eighth review (Overall 5/10, confidence 0.75)

Thank you. This review is the first one to price its own request, and the price it names is what we spent
the round on: *"With the paper reframed around the causal-evaluation contribution and one convincing
adaptive/realistic FL validation, I would move it toward 6-7/10."* We treat that as the reviewer's
prediction rather than as a measurement, and this letter claims no score.

Two of the three asks turned out to be closable, and the reason is a diagnosis that reflects badly on us
rather than on the review. We lead with it.

## 1. The diagnosis: on two of the three asks we already had the machinery, pointed at the wrong object

| the ask | what was already on disk | why it did not answer the ask |
|---|---|---|
| **17.1** make causal evaluation the *sole* central contribution | the title, the abstract and the body's own contribution list | The residual impression was **appendix mass**. The screen block ran 480 lines against 334 for the causal evidence, so an appendix skim read a screen paper with a causal addendum. A prose fix could not repair that; a reorder could |
| **17.2** does the conclusion survive when the attacker adapts to the composed pipeline? | a whitebox criterion-aware adversary, 6 conditions x 5 seeds, ~22 h of compute already spent | It had been run against **the screen's PASS verdict** on FoolsGold->RFA. It had never been run against the dissociation. **The sign reversal had never faced an adaptive adversary** |
| **17.3** add one realistic FL regime and reproduce a central dissociation | an N=100/K=20 suite and an alpha-sweep, both with results | The N=100 material stores game equilibria and the paper itself labels it *"Residual material from an earlier framing"* `:2772`; the alpha-sweep tests the **screen**. Neither reproduced a dissociation at scale |

So 17.2 and 17.3 were real gaps, the review was right about both, and each was closable by one small
pre-registered arm reusing the frozen runners rather than by a benchmark suite. 17.1 was structural.

## 2. What we ran: two new pre-registered regimes, both against the flagship dissociation

Frozen before the output directory existed, then run. Appendix D.6 is the full report; the pre-registration
is `experiments/pre_registration_regime_dissociation.md` at `b317af0`, amended once at `f4c3fec`, and the
runner refuses to start unless its `PREREG_COMMIT` matches the amended hash `:1667`.

The estimand is the **disagreement itself**, `D = Delta(confounded) - Delta(instrument)`, paired by seed,
n=3 on seeds 42/43/44, cell `coord_median` x committed pixel, kappa in {0, 2}, both ladders. The freeze
argued with arithmetic that only D is powered at n=3, and it forbids any equivalence claim from this arm:
*"no equivalence claim is made anywhere in this arm"* `:1669`.

| regime | Delta confounded | Delta instrument | **D** (primary) | 95% CI |
|---|---|---|---|---|
| A: adaptive adversary, N=10, K=5 | -0.243 | **+0.081** | **-0.324** | [-0.543, -0.105] |
| B: N=100, K=20 (20% participation) | -0.325 | **-0.199** | **-0.126** | [-0.217, -0.035] |
| frozen cell, as published (n=20) | -0.273 | +0.125 | -0.408 | -- |

**Both regimes clear the pre-registered primary. Only one reproduces the headline.** Under an adversary
that solves a joint evasion constraint for the downstream defense, the instrument still moves *up* where
the outcome-gated ladder moves down, at every seed. At N=100 both designs read the dose as protective, so
what travels to 20% participation is the disagreement and **not its sign**. We report that as the finding
it is, in the body and in the appendix, and we say so in the table's own caption: *"regime B reproduces the
disagreement and not the reversal"* `:1673`.

That is a narrowing of our own claim, and it is the honest reading: the sign reversal is a finding at
N=10/K=5, including under an adaptive adversary, and at 20% participation it attenuates to a magnitude
disagreement. Three points of discipline we held ourselves to:

- Regime B's instrument interval happens to exclude zero. The freeze forbids reporting any Delta from this
  arm as inferential at n=3, so we do not: *"it is still not reported as inferential"* `:1698`.
- The adaptive adversary beats the plain backdoor by **+0.035 at one seed**, and the appendix says what
  that does and does not license: *"which is a margin, not a rout"* `:1698`.
- The adversary adapts to the downstream defense, not to the instrument, and the appendix explains why that
  is a property of the object: Mode S reads adversary identity, so *"no real attacker can adapt to a stage
  that does not exist in deployment"* `:1698`.

Six disclosures, five of them limits, are at `:1698`, and the arm's three pre-run gates at `:1694`.

### 2a. The calibration decision was prospective, and it is the reason regime A exists

The plan's named cost-reduction lever was to import the existing suite's argmax for the adversary's radius.
The freeze declined it, arguing that `decorrelate` is an anti-FoolsGold device and that against a
coordinate-wise median a lone adversary confined to its own block is plausibly *weaker*. The grid then said
so unambiguously: with decorrelation on, ASR is 0.150 / 0.192 / 0.199 across the three radii, **every one
below the plain backdoor's 0.395**; with it off, 0.398 / 0.430 / 0.406 `:1696`. Importing the old argmax
would have installed an adversary weaker than no adaptation at all, failed gate (1), and voided the arm.

We state the general lesson against ourselves: *"A hyperparameter's argmax does not travel across
downstream defenses"* `:1696`.

### 2b. Three gates, two of which returned exact zeros

(1) the adaptive adversary must beat the plain backdoor it replaces; (2) every rung must clear the 0.35
clean-accuracy floor (0.769/0.691/0.690 and 0.474/0.426/0.441); (3) the shared kappa=0 rung must be
**proved** shared, not assumed, which came back `(0.000000, 0.000000)` in accuracy and ASR in both regimes
`:1694`. Separately, the two frozen runners were proved unchanged **by value** against their stored cells
before either regime ran, worst absolute deviation `0.00e+00` over three runs plus a cross-ladder identity
comparison. An md5 of an artifact would not have been that proof, and the appendix says so `:1694`.

### 2c. One unplanned corroboration, and one pre-registration caution we falsified

Regime B ran its own kappa=0 rung because the freeze cautioned that the two N=100 runners are not
bit-comparable. They are: freshly measured ASR 0.4261/0.5532/0.5631 at accuracies 0.4773/0.4855/0.4598
against the identical five figures per seed from the older runner. *"So the caution was false and it was
false in the safe direction"* `:1700` -- it cost an extra measurement rather than a substituted one. We
record the correction in the appendix and leave the committed pre-registration as written rather than amend
it a second time after seeing results.

## 3. What changed in the body (17.1, and where breadth is actually scored)

- **The two new arms are in the body**, one paragraph in section 6.2, which states the sign-reversal
  narrowing in the same breath as the reproduction:
  *"the reversal is a finding at this participation regime"* `:727`.
  The paragraph's last clause is the power limit, and it is the last word on both arms.
- **The margin ladder the review asks for in item 9 is answered with something strictly stronger**: the
  smallest surviving margin per arm, in prose rather than as a table, defined in place as *"the smallest
  margin an arm's interval fits inside"* `:532`, with `resnet18` named as the one arm that needs 0.183.
- **An explicit positioning statement.** The introduction opens the claim there:
  *"We propose no new defense"* `:144`.
  The contribution list then names exactly one:
  *"The single central contribution is therefore that evaluation-design result"* `:159`,
  with the screen priced rather than validated.
- **The roadmap is now an inventory rather than a walk**: *"The main text carries two propositions, one
  definition, one numbered table and two figures"* `:198`.

## 4. The six wording fixes, each at the site the review pointed at

| # | review item | what the paper now says | site |
|---|---|---|---|
| W1 | an overclaimed causal channel | the protocol clause now reads *"measure the channels it opens separately"*, with the adjective gone | `:473` |
| W2 | one-coder audit, caveat only in the appendix | the caveat is now at the **first body use** of the prevalence: *"one coder, no inter-rater statistic"* | `:170` |
| W3 | instrument vs. what a practitioner can measure | *"never a quantity a deployed evaluator could compute without that ground truth"* | `:519` |
| W4 | theory scope stated across two sites | *"nothing we prove speaks to a transform outside it"*, in the sentence that names the class | `:378` |
| W5 | replication strength | no sentence claims more than n=3 gives, in the body `:727` or the appendix `:1669` |  |
| W6 | one explicit positioning statement | what the paper is, and *"we do not claim it certifies security"* | `:159` |

Two standing scope statements were left exactly as they were, because they already say this. The first is
in the introduction: *"We claim no transfer beyond the controlled setting we study"* `:153`.
The second is in the scope box, on the empirical side of it: *"and offer no causal model of FL"* `:313`.

## 5. The appendix reorder (17.1, 13)

Not one line was deleted. The intervention evidence now **precedes** the screen block, and the screen block
carries a subordinate title in the table of contents: *"A subordinate boundary study: what the screen buys,
and what it does not"* `:1705`. Every label moved with its block; both documents build clean to a fixpoint
with zero undefined or multiply-defined references. The screen's own boundary conditions still open that
block rather than closing it -- *"Extreme heterogeneity breaks two of the three PASS pairs"* `:1761`.

## 6. The page budget (20)

The main text is **9 pages**, judged on the body's last rendered line and not on a heading: the Conclusion
heading and the Ethics statement both sit on page 9, the Reproducibility statement and references on pages
10-11, and the appendix opens on page 12. That is the ICLR treatment the review asked us to verify.

Sections 2, 3 and 6 added lines, so they had to be funded. Every character came from recital, from pure
navigation, or from a second or third emitter of a fact that survives elsewhere, and for each cut the
surviving home was located by grep **before** the cut and recorded in a provenance comment. Nothing came
from a heading, from the scope box, or from any scope condition, limitation, withdrawal or disclosure. Two
consequences we want visible rather than buried:

- A one-sentence pointer we deleted was `tab:regime`'s only reference, which LaTeX does not warn about. The
  table became an orphan float for one build. It is referenced again from the appendix, at zero body cost.
- The negation density **rose**, from 49.1% of body sentences at a smaller denominator to 81 of 165, because
  what we deleted was positive-voice prose. The 16 protected disclaimers are all still present, verbatim.

## 7. Deviations from our own plan, reported rather than buried

1. **Two pre-registration commits, not one.** `b317af0` froze the arm; `f4c3fec` is Amendment 1, committed
   before the output directory existed and before any run. The freeze had required the runner to assert
   that the adversarial tensor *changed* at round one, which is **false by design** for a committed pixel
   backdoor whose payload lives in the poisoned dataset. Amendment 1 replaces it with an assertion that is
   conditional on the attack, and the appendix says why that is not a weakening: *"This is strictly
   stronger than what it replaced"* `:1702`. All four failure modes were deliberately induced and caught.
2. **Regime B does not reproduce the sign reversal.** The plan named that outcome in advance as a finding
   rather than a setback, and this is us reporting it as one. It narrows the reversal's scope to the
   participation regime.
3. **One deleted sentence had to be restored the same round.** Cutting the four design-word glosses from
   section 2 for page budget broke a hard floor in our own clarity instrument, which requires those four
   terms fixed in italics in the front sections. It is back, 18% shorter, glosses intact `:216`.
4. **One promotion we planned did not land.** The body was to name *which ingredient* makes the adaptive
   attack work (decorrelated coordinate blocks, not the radius). At zero page slack there was no line to
   buy it with, so the body carries the seed-matched 3.7x figure `:489` and the ingredient comparison stays
   in the appendix's own table, at `:2210` against `:2211`.
5. **The page budget's funding source was wider than planned.** The plan named section 3; the cuts came from
   sections 1, 2, 3, 5 and 6, because that is where the recital and the duplicate emitters actually were.
6. **The pre-registration's bit-comparability caution is false**, as recorded in 2c above.

## 8. Declined, each with a reason rather than a preference

1. **An anonymous code repository (15).** Declined this round, and the disclosure stays: *"No code or data
   package accompanies this"* submission `:792`. The blocker is not effort: two live access tokens sit in
   plaintext in this repository's history and must be rotated and the history rewritten before anything is
   published, which is not an action we will take unilaterally. We record the cost against Reproducibility
   rather than argue it away. What ships instead is `:797`'s pre-registration count, one per pre-registered
   arm, with the decision rules and the frozen hashes in the paper.
2. **The review's list of things not to do (18)** is adopted as written: no new defense, no benchmark suite,
   no additional theory.
3. **Appendix compression.** Reorder only. Cutting the screen block would delete disclosures the paper is
   built on, and ICLR gives unlimited appendix pages.
4. **Item 9 as a discrete margin ladder.** Delivered as the smallest surviving margin instead, which is
   strictly stronger than four sampled thresholds and costs a line rather than a float `:532`.
5. **Novelty.** Untouched by design. The review is explicit that moving it needs either a new theoretical
   result or an argument that the gate is widespread, and the second is foreclosed by our own
   pre-registered outcome: 6 of 59 papers gate this way, against 46 that do run an identifying contrast
   `:170`. We would rather report the number that limits our own significance claim than argue past it.
6. **Retitling.** Nothing in the paper's own vocabulary was renamed. One appendix heading changed `:1705`.

## 9. Verification, run after the last edit

Both documents build to a fixpoint with zero LaTeX errors, overfull boxes, undefined or multiply-defined
references or citations, and zero `??`. The body-page gate and the clarity gate both exit 0 with no ratchet
raised; the negation floor holds at 16 of 16; the four frozen artifact md5s hold at their literal paths and
the binding arm is unmoved at m\* = 0.034; no em-dash, en-dash or Unicode minus appears in either source or
the bibliography; the PDF carries no title, author, subject or keywords; the rendered text of both PDFs
contains no round numbering; and pages 1-3, 5, 6, 9, 41 and 42 were read at 300 dpi rather than trusted
from source. The pre-registration count in the paper equals the number of files on disk, which is 23.

One instrument-level note, since it changed a number the review may check: the arm's three new keyword
arguments were proved no-ops **by value** against stored per-seed rows, not by hashing an artifact, because
a silently-uncalled hook once left two arms in this repository computing the same thing bit-identically.
