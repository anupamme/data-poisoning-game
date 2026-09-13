# Response to the Weak Accept (6/10, confidence 0.75)

Thank you — this is the most operational review the paper has had. Its §27 priority list is five
concrete items, and four of them turned out to be already in the source. That is not a complaint about
the review; it is a fact about *which* source it read, and it decides how the rest of this letter is
organised.

---

## 0. The review read a PDF that predates the current source by nearly two weeks

The review states it read **`main(4).pdf`, dated Aug. 31, 2026**. Today is Sep. 13, thirteen days later.
That file is the reviewer's own download and is not in our tree, so we take its date from the review's own
text rather than from disk. The paper has been revised in eleven numbered rounds since, and **eight of the
review's asks are already satisfied in the source** — including three of the five §27 priorities.

We are not asking to be taken at our word on that. Every row below carries a source line and a quoted
substring, and both are machine-checked at the end of this letter (§5). Line numbers are `paper/main.tex`
source lines, grepped fresh — **not** the PDF's margin numbers, which are ICLR *rendered* line numbers
and run about four lower.

| the review's ask | state in the current source | check-site |
|---|---|---|
| **§14** remove 13/20 from the abstract | **Already absent.** The abstract is one source line and contains no `13/20`. The six surviving sites are all appendix or reading-guide | `:66` (abstract); `:675`, `:712`, `:1229`, `:1235`, `:1269`, `:2226` |
| **§16, §27.5** fix the `never-used Krum` phrasing | **Already gone.** `never-used` returns **zero** hits. The single `never used` is an unrelated sentence about three quantities | `:1082` |
| **§9** demote the collider reading; make the empty cell primary | **Already primary.** Fig. 2's caption leads with the structural reading, and the collider question is answered as *"Half of it is"* rather than asserted | `:188` *"Not ordinary collider bias but a structural \emph{positivity} violation"*; `:206` *"Is this not ordinary collider bias?"* |
| **§27.4** make the sign reversal the unmistakable centerpiece | **Already the abstract's third beat**, and at **stronger** values than the review quotes: $-0.273$ vs $+0.125$ at $n{=}20$, both 95% intervals excluding zero | `:66`, restated at `:129` |
| **§27.1** tighten the theorem's scope | **Already the statement, not a caveat.** The proposition is titled by its scope, and both hypotheses are named: an eligibility gate and a mechanism condition determined by the two defenses' definitions alone | `:174` *"Outcome-gated comparisons cannot identify the upstream defense's contribution"*; `:178`; `:179` *"the admissible designs are within-defense"* |
| **§12** don't oversell `cos_krum` | **Already demoted.** The paragraph opens by saying these arms *support* rather than carry the account, and prints $0.173$, $p{=}0.044$, $n{=}8$, 5/8 seeds | `:536` |
| **§18** don't oversell Thm. 8 | **Already fenced** as a supporting result that bounds no ASR | `:350` *"are a \emph{supporting} result"* |
| **§13** equivalence-margin discipline | **Already TOST, not a null result.** `is zero`, `equal to zero` and `indistinguishable from zero` return **zero** hits combined | `:397` *"Here practical equivalence is established, not merely consistent"*; `:1812` |
| **§11** Krum's zero is a floor | **Half was already there**, `:399`; the other half is now B2 below | `:399` *"The zero is a floor here"* |
| **§10** (P4) is aggregator-specific | **Already stated twice**, with the operational consequence drawn | `:343` *"instantiated per aggregator"*; `:1082` |
| **§17** "a second dataset and architecture" | **Correct objection, and the widest single fix this round.** See A and B4 | thirteen sites, eleven narrowed and two re-earned |

**§17 is where the review is simply right, and it was right about more sites than it could see.** The
EMNIST-byclass arm uses `simple_cnn`, which is a differently configured shallow convnet, not a second
architecture *family*. Our own plan named **two** sites; grep found the overclaim at **thirteen**. Eleven
are now narrowed to a differently configured CNN, and the remaining two keep the phrase and now name §1's
`resnet18` arm for it (B4). So *a second architecture* is earned by one thing only — the arm in §1 — and
`two architectures` returns zero hits.

---

## 1. The round's product: a ResNet18 arm that varies architecture alone

The review's headline risk (2) is that *"the empirical evidence for the central causal claim is still
narrower than the very broad title/abstract language suggests."* The narrowest joint in that evidence is
exactly the one §17 identifies: the flagship negative had replicated on a second **dataset**, but never
on a second architecture **family**, because the one replication that existed moved dataset and model
*together* and therefore cannot attribute either.

So we ran the arm that moves one factor.

**The cell.** CIFAR-10 / **`resnet18`** / Krum / committed model-scaling under Mode S — the same cell
that produced the flagship negative on `cifar_cnn`. Dataset, `N`=10, `K`=5, `f`=0.2, α=0.5, round count,
attack and defense are all held at the frozen CIFAR-10 values. **Only the architecture moves**, to an
18-layer residual network with skip connections and GroupNorm-8 substituted for every BatchNorm
(`fl_core/models.py:73-79`, the repo's single `resnet18` configuration). A flat result here is
attributable to architecture alone.

**The premise was measured before the rule was written, with no ASR anywhere in sight.**
`experiments/measure_admission_resnet18.py` computes no ASR and trains no model per rung; one raw update
stack per (seed, round) is shared by all thirteen rungs. From `results/resnet18_admission.json`:

| κ | ResNet18 decision change | ResNet18 admission change | `cifar_cnn` decision | `cifar_cnn` admission |
|---|---|---|---|---|
| 0.0 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.5 | 0.333 | 0.000 | 0.533 | 0.000 |
| 1.0 | 0.611 | 0.000 | 0.800 | 0.000 |
| 2.0 | **0.556** | **0.000** | 0.733 | 0.000 |

Verdict emitted by the script: **ELIGIBLE**. The ladder disturbs Krum's decision on ResNet18 and does
not change the admitted adversarial mass — the same premise the `cifar_cnn` arm rests on, measured on the
same dataset on both sides of that table. Had the decision change come out near zero, a flat ASR curve
would have carried no information and **the arm would have been reported as ineligible instead of run**;
that is not hypothetical, because `cos_krum`'s decision is flat at `0.000/0.000` at every rung on this
architecture too. 0 of 18 measurement rounds went non-finite.

**The rules are frozen in git, and the freeze is checked rather than recited.**
`experiments/pre_registration_dose_resnet18.md`, committed at **`89046f6`** before
`results/dose_resnet18/` existed. `experiments/run_dose_resnet18.py` refuses to start unless (a) that
commit resolves, (b) the pre-registration exists at it, and (c) the committed blob equals the working
copy byte for byte — so a decision rule edited after the freeze aborts the run instead of quietly
rescoring it. All four refusal branches are tested. Quoting the frozen rule:

> | \|Δ\| < 0.15 | **FLAGSHIP NEGATIVE REPLICATED** on a second architecture *family*. |
> | Δ > +0.15 | **THE NEGATIVE IS ARCHITECTURE-SPECIFIC.** … The paper's central claim is scoped to the shallow-CNN family and must say so in the body, not in a limitation. |
> | Δ < −0.15 | Attenuation-side fall. **INDETERMINATE**, reported as such and **not** scored in our favour. |

and the branch that can void the whole arm, scored **first** and on its own:

> **Standalone Krum must suppress model-scaling on ResNet18 at usable accuracy**: mean ASR at κ=0 below
> `SUPPRESS_ASR = 0.5` with mean clean accuracy at or above `ACC_FLOOR = 0.35`. … **If the gate fails the
> arm is VOID, NOT NEGATIVE.** … it must not be written up as "the negative replicates."

`EQUIV_MARGIN = 0.15` and `ACC_FLOOR = 0.35` are imported from the existing suite; **no new constant is
introduced**, and `run_one` is imported rather than copied so this arm and the frozen one cannot drift
apart in what they compute.

**Status: the arm reached its frozen $n$, and the result is the emitter's, quoted rather than
paraphrased.** All six runs are on disk in `results/dose_resnet18/summary.json` — three seeds at each of
the two endpoints, counted as `per_seed` entries rather than off the runner's `[i/N]` index, which counts
resumed-and-skipped runs. Re-invoking the runner reports and scores without recomputing anything:

> `kappa=0 (krum alone): mean ASR 0.071 @ mean acc 0.536, n=3`
> `GATE PASSED: krum suppresses model-scaling on ResNet18 (ASR 0.071 < 0.5) at usable accuracy. There is`
> `suppression for the ladder to preserve or lose, so the primary rule is scoreable.`
> `delta = -0.041 (0.071 -> 0.030), n=3 at both endpoints on seeds [42, 43, 44]`
> `FLAGSHIP NEGATIVE REPLICATED on a second architecture family`

**The void branch was scored first and did not fire**: mean ASR at κ=0 is $0.071$, below `SUPPRESS_ASR
= 0.5`, at mean clean accuracy $0.536$, above `ACC_FLOOR = 0.35`. Had it fired, this paragraph would say
*void* and the arm would license no sentence about architecture.

**The three arms of the same rule, all inside the frozen $\pm0.15$:** `cifar_cnn` $-0.026$ ($n{=}5$),
EMNIST-byclass/`simple_cnn` $-0.017$ ($n{=}3$), and now CIFAR-10/`resnet18` $-0.041$ ($n{=}3$), the last
with architecture varying alone. The runner recomputes the first two from their own artifacts rather than
quoting our tables, so a table edit cannot silently move the comparison. The paper states it at `:532`
and in the tiers table at `:746`, and $-0.041$ is a value no other quantity in the paper carries.

**Per-seed, because the Reproducibility statement promises per-seed values for every arm.** κ=0 ASR
$0.114/0.050/0.048$ against κ=2 $0.008/0.034/0.047$ on seeds 42/43/44, so the per-seed differences are
$-0.106$, $-0.017$, $-0.000$ at sd $0.057$ — one seed carries the movement and two are flat. Those six
numbers are printed in the paper, not only here (`:1824`).

#### What $n{=}3$ on two rungs does not buy, which is this round's sharpest self-correction

The point estimate passes the frozen rule. **The paired $95\%$ interval, $[-0.183, +0.101]$, is wider
than the margin it is being compared against**, and three things follow that we would rather state than
have found:

1. **The arm's equivalence holds at the frozen $\pm0.15$ and at nothing tighter than $0.137$.** TOST
   against the frozen margin does reject ($p = 0.040$, post hoc, and the closest call in the TOST table
   at `:2392`), because TOST at $\alpha{=}0.05$ reads the $90\%$ interval, $[-0.137, +0.055]$. The
   $95\%$-interval criterion the margin ladder uses is a different and stricter reading, and this arm
   fails it at every rung: its own $m^\ast = \max(|\mathrm{lo}|,|\mathrm{hi}|)$ is $0.183$ against the
   ladder's binding $0.034$. Both readings are printed, and the table caption at `:2375` says in terms
   that the two conventions are not interchangeable.
2. **One standing sentence in the body went false the moment this arm existed, and no build, hash or gate
   could see it.** `:397` claimed every equivalence reading in the paper survives any margin above
   $0.034$, $\pm0.05$ included. This arm needs $0.137$, so the claim is now scoped to the ladder's own
   table and names the exception with its number. `experiments/analyze_margin_sensitivity.py` now derives
   the out-of-artifact arm itself and prints that a sentence of the old form is FALSE while the arm is in
   the paper — the ladder's artifact structurally cannot hold this arm, which has a controlled leg and no
   outcome-gated twin and therefore no comparability cell. The scope condition is also in the ladder's
   caption (`:2429`) and in the supplement (`supplementary.tex:456`).
3. **We withhold the paper's own defined equivalence phrase from this arm, and say why.**
   `supplementary.tex:441` defines *no evidence of a practically meaningful change* by the paired $95\%$
   interval lying inside the margin, so the phrase is not available here at $n{=}3$. The appendix lead
   instead reads *"Suppression does not rise, and under the rule this arm was frozen against the negative
   replicates on a second architecture family"* (`:1822`), and the reason is disclosure **(v)** of five,
   four of them limits (`:1824`). The remedy is seeds, not wording, and we say that too.

**What the arm still does not license.** One cell, one attack, one dataset, two rungs, three seeds. It
adds an architecture family and nothing else, and §4 below is where that is stated rather than hedged.

### What this arm is weaker at than its predecessor, said before anyone asks

**It has no independent cross-check on its own harness.** The EMNIST arm could compare its κ=0 rung
against a published 3-trial payoff-matrix figure for standalone Krum/model-scaling on that dataset. **No
artifact in `results/` contains both `resnet18` and `krum`** — we searched — so no such figure exists
here and none was invented. That is why the κ=0 rung is a *scored pre-registered gate* rather than an
eyeball comparison, and it is a genuine regression relative to the EMNIST arm.

**Two rungs, not four, and therefore no trend test.** κ ∈ {0, 2} only (ρ = 1.00 and 54.60). The EMNIST
arm ran four rungs so a Jonckheere–Terpstra test could be scored; this one cannot, at 1.68 h/run × 4
rungs × 3 seeds ≈ 20 h. **A two-rung ladder has no trend to test and is never displayed or described as
a four-rung one** — the restriction is printed by the runner on every invocation and disclosed in the
paper.

**One deviation from the `cifar_cnn` measurement, disclosed as a deviation and not as a passing check.**
The adversarial coefficient *share* under Mode S is constant across rungs only to `1.855e-06`, which is
**above** the imported `SHARE_TOL = 1e-6` (`cifar_cnn`: `3.481e-07`; EMNIST: `3.926e-07`). We did **not**
widen the tolerance — widening a tolerance to admit an arm is the move a pre-registration exists to
prevent. Instead the assertion that is pinned *by construction* is reported separately from the derived
one: `c_adv` is exactly `1.000000000000` at every rung, max deviation `0.00e+00`, so the attenuation
channel is closed bit-exactly. For the derived share we added a substantive discriminator, because a
*dose* is monotone in its dial by construction while float32 read-back noise is not:

- the Mode-S share sequence is **non-monotone** in κ (`0.253333333`, `0.253332558`, `0.253334413`,
  `0.253332777`), whereas the `dose` family — whose confound this paper already reports — is monotone
  (`0.2533 → 0.2646 → 0.2711 → 0.2731`);
- the deviation is four to five orders of magnitude below the smallest signal the instrument must
  resolve: `1.855e-06` against the reported `dose` confound at `1.981e-02` (10,680× larger) and Mode A's
  intended payload dial at `6.514e-01` (351,146× larger);
- the tolerance is **absolute** and the share is a ratio whose denominator sums float32 coefficients over
  every parameter, so its read-back noise grows with model size, and `resnet18` has ~2 orders of
  magnitude more parameters than the model the tolerance was calibrated on.

**The accuracy floor is live on this architecture, not a formality.** The nearest existing artifact runs
`resnet18` at exactly this regime and reaches 0.662–0.766 clean accuracy in 14 of its 15 runs — but the
fifteenth **collapsed to 0.153**, below the floor. So about 1 run in 15 collapses here, and every rung is
gated on mean clean accuracy ≥ 0.35. **The arm's own six runs came in at 0.496–0.591** (rung means
$0.536$ and $0.557$): every run clears the floor, no run collapsed, and every run also lands well *below*
the 0.662–0.766 band we sized that argument against. We report the gap rather than explain it — the
sizing artifact is a mix-ratio sweep and differs from this cell in more than the dial — and note that the
floor is a usability gate, so a rung passing it at $0.54$ supports a suppression reading and not an
accuracy claim. The paper prints the two rung means at `:1822` as gate evidence — *"Mean clean accuracy
\emph{rises} $0.536 \to 0.557$"* — and claims no accuracy result from this arm anywhere.

**The arm is an instrument, not a defense.** Mode S reads adversary identity to pin the adversarial
coefficient share; no deployable defense knows which clients are adversarial. This is the open problem
the paper already ends on (`:2265`), not something the arm resolves.

---

## 2. Five precision edits

| # | review | edit | check-site |
|---|---|---|---|
| **B1** | **§27.2** make the causal estimand explicit | The contrast is now written down, not just defined operationally: *"the estimand is $\Delta_{\mathrm{mech}} = \mathrm{ASR}(T_1^{\mathrm{int}}, d_2, a) - \mathrm{ASR}(T_1^{\mathrm{base}}, d_2, a)$, at fixed $d_2$, fixed attack $a$ and fixed adversarial coefficient share"* — the third conjunct is what makes it a mechanism contrast and not a dose | `:356` |
| **B2** | **§11** name the non-floor admission controls | The two arms that answer "we moved something and nothing happened" are now named rather than left to be read off the table: *"suppression moves in exactly the two arms whose \emph{influence} channel moves, reputation and coord.\ median"*, with the two Krum variants bracketing sufficiency and necessity | `:460` |
| **B3** | **§12** one word | `cos_krum` is now *"a pre-registered counterexample within the tested regime"* — the review's own phrase — instead of *the converse witness*, which appears nowhere in the source now | `:536` |
| **B4** | **§17** architecture precision | **Thirteen sites, in two kinds.** Eleven are *narrowed*: the EMNIST arm is now *"a second dataset and a differently configured CNN"* (`:746`, `:1701`, `:1780`, `:1806`, `:1808`, `:1812`, `:1865`, `:2265`) and the three breadth inventories now say *"two CNN configurations"* (`:488`, `:1945`, `:1949`). Two are *re-earned* rather than narrowed: `:532` and `:752` still say *"a second architecture family"* and now name §1's `resnet18` arm for it, which is the only thing in the paper that earns the phrase. `two architectures` returns **zero** hits | `:746`, `:1812` |
| **B5** | **§19** say what kind of methodology this is | The scope box now states it: a *"methodology for identifying what a composition preserves, not for choosing a defense to deploy"* | `:260` |

**B4 carried one complication worth reporting rather than burying.** The EMNIST pre-registration's own
frozen wording is *"a second dataset and architecture"* — so narrowing the paper's phrase makes the
sentence *"the pre-registration fixes that wording"* false unless the change is stated. We did not edit
the frozen file. Instead `:1812` now reports the narrower phrase **and** says so:

> That is narrower than the wording the pre-registration froze, which reads "a second dataset and
> architecture": `simple_cnn` and `cifar_cnn` are two configurations of one shallow-convnet family, so we
> report the narrower phrase and leave the frozen file unedited, claiming less than a rule licenses being
> always permitted where claiming more is not.

The prohibition we keep verbatim is the one that matters: never "generalizes".

---

## 3. Five declines, with reasons

- **§10's `P4(d_2)` renotation.** Declined. The semantic content is already stated twice, and `:343`
  already draws the operational consequence the notation would exist for — that (P4) yields four
  within-family statements which we never difference or rank across. Renotating a level that appears
  throughout the paper buys a subscript and costs every cross-reference to it.

- **§28's suggested abstract rewrite.** Declined **because adopting it would make the abstract false.**
  It quotes $-0.272$ vs $+0.098$ as the headline. Those are the $n{=}5$ values that a pre-registered
  seed top-up superseded with $-0.273$ vs $+0.125$ at $n{=}20$. Taking the suggested text verbatim would
  regress the paper's centerpiece to a smaller $n$ and a superseded number. The two improvements it
  genuinely carries — estimand explicitness and scope tightening — are adopted as **B1** and **B5**.

- **§27.4 a side-by-side counterexample figure.** Declined on layout, not on merit. Table 1 already puts
  all six rows on one page with their $n$; the two non-implications are stated formally
  ((P4)$\not\Rightarrow$(P3) witness Krum, (P4)$\not\Rightarrow$(P5) witness `cos_krum`) and bracketed in
  prose at `:460`. A new float costs ≥6 rendered lines, and the main text currently ends on p9 with a
  single character of slack.

- **§23 an anonymised code artifact at submission.** Out of scope this round, and the paper is already
  consistent about it: `:601` states *"No code or data package accompanies this"* (the sentence wraps, so
  *"submission}"* is on `:602`). Two
  plaintext credentials in the repository's history block release; that is a release-time blocker
  requiring credential rotation and history rewriting, not a submission-time edit.

- **§15/§24 "reduce to four points" / still too dense.** Answered by measurement rather than by argument,
  and re-measured after this round's additions rather than quoted from the last one:
  `experiments/measure_appendix_redundancy.py` now finds **5** sentence pairs above Jaccard 0.3 and **14**
  paragraph pairs above containment 0.15 over a 351,307-character appendix window, of which **1 sentence
  pair and 0 paragraph pairs** have equal numeric sets. Both counts rose (from 3 and 12) because this
  round added an arm, and its disclosures are the same *template* instantiated on new numbers — which is
  exactly what the numeric-set test is for: every one of the 14 paragraph pairs differs in its numbers, so
  deleting either member deletes that arm's disclosure rather than a repetition. The single same-numbers
  sentence pair is a preview whose own opening clause declares itself one. We report the rise rather than
  the more flattering earlier pair of numbers, since a density objection answered with a stale measurement
  is not answered.

---

## 4. What this round does not claim

**We are not claiming this makes the paper an 8.** The review's own probability mass is Accept ≈40% /
Borderline ≈25% / Weak Reject ≈25%, and its §22 correctly names the regime: `N`=10, `K`=5, `f`=0.2,
CIFAR-10, 50 rounds, two attacks. This round adds **one architecture family to one cell** and closes five
presentational gaps. It adds no dataset, no `N`, no `f`, no round count and no attack, and it does not
touch the sign-reversal cell. The composition suites still never leave CIFAR-10, which `:2237` states as
limitation (L1) and this round does not improve.

Headline risk (1) — that the identification result reads as a specialized consequence of how the
screening criterion is defined — we address by scope statement rather than by new theory: the two
hypotheses are named in the proposition itself (`:174`–`:179`), the result is stated for *any* such
condition rather than for our five levels, and `:2265` records that the two boundary results are
statements about the shape of a screen rather than about federated learning.

### Deviations from our own plan for this round, reported rather than buried

1. **The phases were reordered.** The editorial edits were verified before the run was launched, because
   the page-budget arbiter had to be checked before committing to any new prose.
2. **Two numbers in our plan were wrong and were corrected against the artifact — and then the arm's own
   runs corrected the corrected one.** The plan asserted "0.66–0.76 clean accuracy, far above the floor"
   and "`wall_time_s` 5884–6134". The sizing artifact's 15 rows give **0.153–0.766** — one run *below* the
   floor, which is why the accuracy gate is now described as live — and **5884–6205**, mean 6039 s. Both
   corrections propagated into the pre-registration and the runner before the freeze. The arm itself then
   ran at **0.496–0.591** clean accuracy, above the floor in all six runs and below the sizing band in all
   six; §1 reports that gap rather than the band we planned against.
3. **Our own eligibility emitter printed a false verdict, and we fixed the emitter rather than the
   threshold.** It reported `INSTRUMENT CONFOUNDED ON THIS ARCHITECTURE` for Mode S (whose spread was
   `1.9e-06`) and for the `dose` family (whose confound is a *finding* this paper already reports). The
   fix was the monotonicity discriminator above plus a three-branch verdict; `SHARE_TOL` was not touched.
   Re-emission recomputes every verdict from the stored rows rather than re-running the measurement, so a
   corrected verdict costs no compute and gives no opportunity to alter what was measured.
4. **B4 grew from 2 sites to 13, and the last of those grew after Phase A landed.** The plan named two;
   grep found the overclaim at thirteen, including three breadth inventories that made it in a different
   form. Eleven were narrowed. The other two are the sites Phase A *earned*, so they were re-pointed at
   the `resnet18` arm rather than narrowed — a distinction we make here because a reader auditing the
   count would otherwise find two survivors of a phrase we said we had removed everywhere.
5. **The pre-registration count went stale by our own hand** and was updated: `:606` now reads *"$22$
   pre-registration documents"*, matching `ls experiments/pre_registration_*.md`.
6. **The page gate did not stay green, and two funding items were spent to buy it back.** Our own §17
   precision fix at `:488` (`two architectures` → `two CNN configurations`) added five characters to a
   body that had exactly one character of slack, and **31 rendered words spilled onto p10**. Two of the
   plan's funding items were then applied, in the plan's order: §6's regime sentence (`:493`) and the
   closing sentence of §5's positive-control paragraph (`:460`). The third, Table 1's caption pointer,
   was not needed. A fourth was unavailable — a wrapped rendered heading we hoped to reclaim is a section
   title we are not renaming.
7. **One funding item's stated rationale did not survive checking, so the compression was made smaller.**
   The plan claimed `:493` *recites the scope box it points at*. The scope box does carry the dataset,
   the two committed attacks and the seed counts, but **not** the architecture, `N` or `f`, and no other
   main-text line carries those three for §6 — so only the duplicated half was removed and the pointer
   did not swallow the rest. The plan also credited this edit with removing a "We validate" framing; that
   framing had already gone in an earlier round, and both surviving `We validate` occurrences are in the
   appendix, outside the page gate's window.
8. **A clause was deleted rather than relocated, and its negation was restored.** §5's positive-control
   paragraph used to end on a claim that large decision changes need not change adversarial influence or
   attack suppression. That claim has three homes — §5's own title, the channel-table appendix, and the
   prose at `:399` — so it was deleted rather than moved, and it is no longer on any line of the source.
   Deleting it took negation density from **90/177 to 89/177 at an unchanged denominator**, which is the
   tripwire we keep for softened concessions. So the surviving sentence was rephrased into negative form,
   `:460` now reading *"Neither reading rests on Krum's admission floor"*, and the density returned to
   90/177. The admission-floor hedge is the part of that sentence we were protecting, and it never moved.
9. **A sixth edit was made that no reviewer asked for, because our own check found a defect.** The
   rendered PDF contained an internal development-round reference a reader cannot resolve. The paragraph
   disclosing that the $\pm0.15$ margin was frozen with no written rationale attributed the effect the
   design was built against to "Round 11's own $+0.762$ rise at $\rho{=}54.6$", and nothing in either
   document says what Round 11 is. The number, the arm and the disclosure are all unchanged; the
   reference now names an object the paper does define: `:2400` reads *"the pre-registered ladder's own $+0.762$ rise"*.
   It is in the appendix, outside the page gate's window, so it cost
   nothing. It was pre-existing rather than introduced this round, and it was found by grepping the
   **extracted PDF text**, not the source: a source-only search for `Round [0-9]` does hit that line, but
   the line also carries the one round-provenance fact the paper is allowed to render (the commit hash
   the margin was frozen at), so the site reads as the permitted exception until you look at the pixels.
10. **Landing the arm falsified a sentence the arm was not about, and finding it took re-deriving the new
    interval against the old claim.** The plan's verification list did not contain "check whether any
    standing claim quantifies over *all* equivalence readings". `:397` did, at $\pm0.05$, and the arm's
    $m^\ast = 0.183$ made it false. Nothing mechanical could catch this: the build was clean, every frozen
    md5 held, both gates passed, and the ladder's own artifact is structurally incapable of containing the
    arm that falsified the sentence. Three checks were therefore added rather than one edit made: the
    margin sweep now derives the out-of-artifact arm and prints the FALSE-sentence warning itself, the
    ladder's caption states that one paper arm is not in its artifact, and the TOST emitter lists the arm
    so its header's promise of *every* Mode-S arm stays true.
11. **Two digit collisions were checked before printing and are recorded here, not resolved by rewording.**
    The gate's mean ASR at κ=0 is $0.071$, and $+0.071$ is also the CIFAR-100 cell's paired $\Delta$ two
    rows above it in the same TOST table (`:2396`) — same digits, different estimand, different unit
    context. The arm's smallest TOST-surviving margin, $0.137$, collides with a temporal-mix half-width at
    `:2247`, *"$0.667 \pm 0.137$"*, which is an ASR interval and not a margin. Both were resolved by
    matching the (mean, sd, $n$) triple rather than the digits, and neither was reworded to avoid the
    coincidence — the emitters own those numbers.
12. **The TOST table's row order differs from the emitter's print order, deliberately.** The emitter lists
    the ResNet18 arm in `ARMS` order; the table places it after the pooled EMNIST row so the two EMNIST
    seed counts stay adjacent. Every value in the row is the emitter's verbatim and only the position
    differs, which a comment at the table records so a future reader does not "fix" one to match the other.

---

## 5. How to check this letter

Two audits are run over this file by `paper/audit_letter_cites.py`, and both must pass before it is sent:

1. every cited `:NNN` resolves to a source line that is non-blank and does not begin with `%`;
2. for every line this letter quotes, the quoted substring is actually present on that line.

Audit (2) exists because audit (1) alone is not sufficient — in an earlier round two stale citations
landed on real, wrong content and passed a non-blank check. Line numbers are source lines in
`paper/main.tex` unless prefixed otherwise. **Every `:NNN` in this letter was re-resolved by grepping the
current source, one line at a time.** They had all moved: this round inserted provenance comments above
them, and the resulting offsets are *piecewise* (+0, +13, +25, +30, +31, +40, +53 at different depths), so
interpolating any one of them from another would have produced confident, wrong cites. One old cite even
resolves to **two** different lines, because it had been used for two different claims. Current state:
**75 cites, 32 checked quotes, 0 failures**; 10 further quotes are of the review's language, of our own
plan, or of text this round deleted, so they carry no cite and the audit lists them as unchecked rather
than passing them silently.

Also verified after the final edit, in this order: the fixpoint build is clean in both documents (70 and
11 pages; zero LaTeX errors, undefined references, undefined citations, multiply-defined labels, reruns,
overfull boxes, and zero `??` in either PDF). The clarity gate passes with **no ratchet raised** — bold
39/39, italic 61/62, cross-refs/paragraph 5/5, longest paragraph 846/850, abstract 1591/1600, `(P#)`
12/12, comment-joined paragraphs 0 — and negation density is 51.4% (90/175): the numerator held at 90
while the denominator fell by two, so no concession was softened. The page gate passes with the body
ending on p9 and the Ethics heading the first content of p10. The equivalence-margin sweep's binding arm
is unchanged at `m* = 0.0343` with 0 verdicts changing anywhere on the ladder, and it now additionally
prints the one paper arm its artifact cannot contain (`krum / scaling, resnet18`, `m* = 0.1830`). The four
frozen artifact md5s hold, each named so the claim is checkable rather than recited: `2f4d9920`
(`results/all_compositions/summary.json`), `a16ef13f` (`results/dose_femnist/summary.json`), `a0717893`
(`results/headline_seed_topup/summary.json`) and `6e25ef3a` (`results/comparability_six_cells.json`).
