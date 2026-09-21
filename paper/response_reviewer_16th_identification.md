# Response to the sixteenth review (6/10, borderline accept, confidence 4/5)

Thank you. The review names three changes before submission: make Proposition 6's novelty claim modest and
its practical implication prominent; replace or qualify the collider and equivalence-testing language; and
strengthen the all-seed score-only Krum result.

**One of those is a real defect, and it is worse than the review says.** The equivalence half of change 2 is
correct, we had not seen it, and this round's chief work is fixing it. We lead with that, with the emitter
that scores it, and with the exact count of arms affected — which is *four of five*, not all of them, because
one arm earned its two-sided reading and keeps it.

**Change 3 is answered by one new pre-registered arm, and it cuts against us in part**: on the single Krum
cell where both margin legs bind, the two-sided containment the paper reports on its flagship cell *fails*,
in the direction of more suppression, and the outcome landed in a branch our own freeze did not enumerate.
Section 6 reports that as a defect in the freeze rather than rounding it to the nearest branch we had named.

The remaining items are largely answered by supersession rather than by new work, and we say so plainly rather
than restating things as if they were new. The reviewed PDF is four titles behind the live paper, whose title
at `:43` is *Beyond Statistic Preservation: Causal Evaluation of Composed Federated-Learning Defenses*; the
reviewed artifact is not on disk. Section 5 below is the crosswalk, given once and without complaint.

Source line cites in this letter are `main.tex` source lines, or `supplementary.tex` where labelled. They are
**not** the PDF's margin numbers, which are ICLR rendered line numbers and differ by roughly four.

---

## 1. The equivalence objection is right, and the arithmetic is worse than the review states

### 1a. The bound we missed

ASR is bounded below by zero. So for any dosed arm, the paired mean difference against the identity rung
satisfies

$$\mathrm{mean}_i(\Delta_i) \;\ge\; -\,\mathrm{mean}_i(\mathrm{ASR}^{\text{identity}}_i).$$

Wherever the identity-rung mean ASR is itself **below** the equivalence margin, the lower TOST leg
$\Delta > -0.15$ is therefore satisfied by arithmetic **for every possible outcome, before any seed runs**. It
is not a test. Only the upper leg is. A two-sided equivalence verdict on such an arm advertises two
constraints and delivers one.

We had the arithmetic and used it in exactly one place. A disclosure on the score-only arm already called one
of its own pre-registered secondaries *"arithmetically unreachable"* `:1862`, on the ground that
*"ASR cannot fall below $0$"* `:1862`. We never carried that reasoning across to the margin, or to the arms
the paper leads with. That is the defect.

### 1b. How many arms, measured rather than asserted

New instrument `experiments/measure_margin_reachability.py`, read-only over the frozen artifacts, emitting
`results/margin_reachability.json`. It recomputes every published identity-rung mean, paired difference and
interval it touches and asserts them equal to the published values before printing anything
(`all_recomputed_values_match_published`), so it is an audit and not a re-analysis.

Counting **distinct arms** that carry an equivalence reading anywhere in the paper — not rows, because the
flagship and its EMNIST replication are each scored in more than one place at more than one seed block:

| arm | where scored | $n$ | identity mean ASR | lower leg | $95\%$ bound on a **rise** |
|---|---|---|---|---|---|
| krum / scaling, the flagship cell | comparability, controlled | 20 | 0.044 | **arithmetic** | $+0.012$ |
| krum / scaling, the same cell | App. F table, $n{=}5$ block | 5 | 0.062 | **arithmetic** | $+0.026$ |
| krum / scaling, EMNIST-byclass | controlled; App. F pooled row | 5 | 0.024 | **arithmetic** | $+0.009$ |
| krum / scaling, score-only control | App. F table | 5 | 0.062 | **arithmetic** | $+0.061$ |
| Mode M, coordinate masking | second transformation class | 10 | 0.053 | **arithmetic** | $+0.019$ |
| `coord_median` / scaling | comparability, outcome-gated | 5 | **0.519** | *a genuine test* | $+0.081$ |

**Four of the five distinct arms rest on an arithmetically satisfied lower leg. The fifth,
`coord_median` / scaling, does not, and it keeps its two-sided reading unchanged.** We state the count that
way deliberately. A universal over the paper's equivalence claims would be false, and the emitter carries a
field that forbids writing one.

The flagship appears twice above because it is scored at two seed blocks whose numbers are not
interchangeable: identity 0.062 with bound $+0.026$ at $n{=}5$, identity 0.044 with bound $+0.012$ at the
$n{=}20$ top-up. Both appear in the paper and the letter keeps them apart.

### 1c. The relabelling reports a *stronger* number than we published

`analyze_tost_existing.py` returns $p_{\text{TOST}} = \max(p_{\text{lower}}, p_{\text{upper}})$. On five of
the six rows that App. F's table marks equivalent, **that maximum is attained by the leg that was never at
risk**:

| row | $n$ | $p$ printed | $p$ on the one leg that is a test | which leg set the printed $p$ |
|---|---|---|---|---|
| Krum / scaling (flagship) | 5 | 0.0014 | **0.0004** | lower, arithmetic |
| Krum / scaling (score-only) | 5 | 0.0071 | **0.0024** | lower, arithmetic |
| Krum / scaling (EMNIST-byclass) | 3 | 0.0004 | **0.0003** | lower, arithmetic |
| the same arm, pooled | 5 | 0.00003 | **0.00002** | lower, arithmetic |
| Krum / scaling (ResNet18) | 3 | 0.0404 | **0.0143** | lower, arithmetic |
| `coord_median` / pixel (CIFAR-100) | 5 | 0.0025 | 0.0025 | **upper, a real test** |

So the honest form of the flagship result is not a retreat. It is a pre-registered **non-increase** with a
$95\%$ upper bound of $+0.012$ against a frozen $0.15$, at a one-sided $p$ of $0.0004$ — a smaller $p$ than
the one we published. **No published $p$ is revised anywhere.** $p_{\text{TOST}}$ stays exactly as printed and
the upper-leg $p$ is reported beside it as the separate quantity it is: App. F now says the table
*"revises no published $p$"* `:3150`.

One place where we could have conflated two criteria and did not: the ResNet18 row is marked equivalent by
the $90\%$ TOST convention *and* is reported as indeterminate by the $95\%$ margin-ladder convention, whose
interval $[-0.183,+0.101]$ is wider than the margin itself. Those are two conventions with two different
answers on the same arm, the supplement defines which phrase belongs to which, and this round changed
neither.

### 1d. What changed in the paper

Every site now names which leg is a test. The abstract's Krum sentence reads
*"ASR's $95\%$ rise bound is $+0.012$, pre-registered $0.15$"* `:72`. The adjudicating paragraph gives
*"its $95\%$ upper bound on a rise $+0.026$ under the pre-registered $0.15$"* `:698`. The margin paragraph
says *"only the upper TOST leg is a test"* `:720`. App. F carries the full treatment with a new per-row table
scoring both legs, and states that on the flagship row the one-sided $p$ for the claim actually made is
*"the smaller $0.0004$"* `:3149`, that on those rows the lower leg is *"satisfied by ASR's own floor"* `:3223`,
and that the CIFAR-100 row is the one row *"marked equivalent"* `:3176` here whose two-sided reading is
earned. The margin-sensitivity ladder's caption now says which side of the margin its $m^{\ast}$ values
measure, and the supplement's own sensitivity appendix says the sweep varies
*"the upper leg alone"* `supplementary.tex:447`, and that
*"what changes is only which of the two legs a reader should credit the pass to"* `supplementary.tex:449`.

A second instrument, `experiments/audit_equivalence_claims.py`, gates this so it cannot regress. It takes
every paragraph in both documents that mentions equivalence or TOST, and requires each one either to carry a
one-sided qualifier or to be classified in the script itself with a reason. It found fifteen defects on its
first run; it now reports 44 claim units, 18 classified, 26 qualified, zero defects. A new claim sentence
cannot enter either document silently, because it arrives as a defect until someone classifies it.

**No scope condition, limitation or disclosure was removed to make any of this read better.** Relabelling a
claim removes no caveat it carried.

---

## 2. Change 1, Proposition 6: the modesty and the practical implication are already there

Both halves of this ask were in the paper before the review, and we point at them rather than rewrite them.

**Modesty.** The paragraph that follows the proposition opens
*"The violation itself is standard; four things around it are not."* `:366`, and the concession paragraph
begins *"We claim no credit for it: that half is textbook"* `:407`. The proposition itself is hypothesised
narrowly: the gate must use the
*"outcome metric and threshold the screen predicts for the composition"* `:320`, so a screen gating on
anything else is outside it.

**The practical implication, made prominent.** Figure 1's caption, on the proposition's own page, is titled
around it: *"which is why the remedy is a design and not an estimator"* `:331`. The mechanism paragraph states
the constructive half in the same place — *"The gate also earns its keep"* `:407` — because the gate stops the
screen certifying pairs no constituent suppresses, which is exactly why a low composition ASR under our
conditions is attributable to the downstream defense's standalone strength rather than to preservation.

We have not added a fifth novelty reason or strengthened the claim. If the review's point is that the
proposition should not be *read* as a new causal result, the paper agrees in its own words and in two
places.

---

## 3. Change 3 and review §25, answered precisely and not more

The review asks for the score-only Krum intervention **at all seeds with a paired CI**. Here is the exact
state of the seed counts, because it would be easy to overclaim here and we are not going to.

- The **full** Mode-S endpoint on krum / scaling is at $n{=}20$.
- The **score-only** endpoint arm is at $n{=}5$, in every table that reports it.
- What *was* topped up to $n{=}20$ is a different thing: the frozen $\kappa{=}1.0$ contrast rung of the
  score-only / emit-only factorial, seeds 47–61, under rules frozen before the runs existed. That top-up
  **withdrew our own published verdict**. The paper says
  *"the verdict it produced has since been withdrawn by our own top-up"* `:1886`, and
  *"So the $n{=}5$ reading that the channels interact does not survive"* `:1886`.
- That top-up's own artifact says what it does not cover. `results/emit_only_topup/summary.json` carries the
  keys `ladders_remain_at_n5` (the score-only and emit-only ladders across $\kappa$, and the
  Jonckheere–Terpstra trend test, are not restated at $n{=}20$) and `cannot_certify` (projected half-widths
  at $n{=}20$ are wider than the margin, so a pass of the point rule is reported as consistent with
  separability rather than established).

**So review §25 is not satisfied by any arm the reviewed paper published, and we do not claim it is. The
flagship cell's score-only control is still $n{=}5$ and this round did not top it up.** What this round adds
instead is a new score-only arm at all 20 seeds with a paired CI, on a cell chosen so that the answer means
something; that is Section 6, and Section 6d states plainly which of the two things the review asked for it
delivers and which it does not.

---

## 4. Review §26, Multi-Krum: declined by measurement, not by argument

The review asks for Mode S on a second selector such as Multi-Krum. In this configuration that is not a
second selector. At `clients_per_round` $=5$ with the default $k=5$, Multi-Krum keeps **all five**
participants and reduces to plain averaging: the paper reports that it
*"selects \emph{all five} participants and is bit-identical to FedAvg"* `:2252`, with the largest absolute
per-aggregation difference $1.9\times10^{-6}$, which is float summation order alone. It becomes a genuine
selector again at $K=10$. The artifact is `results/mkrum_degeneracy.json` and the appendix is
`app:mkrum` at `supplementary.tex:62`.

This is not a new concession. We already **withdrew** three matched contrasts that we had read as our most
direct probe of the invariance proposition, and two of the three were withdrawn for exactly this reason —
they ran against MultiKrum and so carried no downstream defense at all. The third fell for a different
reason, stated in the same place: substituting genuine Krum does not repair it, because the gap then measures
a failure of the standalone-suppression condition rather than disturbed invariance.

Running Mode S against an aggregator that is arithmetically FedAvg would produce a row that looks like a
second selector and is not one. What we offer instead is a second **informative cell on the same selector**,
which is Section 6.

---

## 5. The staleness crosswalk, once

The reviewed PDF predates the current title by three retitles. Each item below was resolved by grepping
current content, never by assuming a line offset.

| review item | where it now stands |
|---|---|
| §8, §10 — Proposition 6 scope and modesty | Section 2 above; `:320`, `:366`, `:407` |
| §9 — the collider framing | Figure 1's caption `:331` and the concession at `:407`, both pre-review |
| §11 — equivalence testing | **conceded and fixed**, Section 1 above |
| §14, §15, §17 — the criterion, and temporal mixing | both already demoted out of the body: the appendix begins at `:1082`, the criterion's *"central practical question is"* `:1235` paragraph sits after it, and temporal mixing appears only in appendix tables and in the limitation *"Per-round aggregator switching only"* `:2937` |
| §20 — a P1–P5 crosswalk table | six rows with a bears-on-security column: *"Six words, fixed here and used in one sense throughout"* `:550`, header *"term & level & what it names & bears on security"* `:561` |
| §22 — ASR is not security | the scope box says what *"survives is the suppression ordering rather than any security property"* `:442`, and that this is a *"methodology for identifying what a composition preserves, not for choosing a defense to deploy"* `:472` |
| §25 — all-seed score-only Krum | Section 3 above, then Section 6 |
| §26 — a second selector | Section 4 above; declined by measurement |

---

## 6. A second informative cell: Mode S on Krum / committed pixel

This is the round's only new compute: 40 runs at 20 seeds, pre-registered in
`experiments/pre_registration_modeS_pixel_headroom.md` and committed by itself at `c89e963` with both output
paths verified not to exist, artifact `results/modeS_pixel_headroom/summary.json`. It answers review §25, and
it partly contradicts us. We report the contradiction first and in the paper, not in a footnote.

### 6a. Why this cell, and why the flagship cell could not have answered the question

The cell the paper scores its flagship reading on is uninformative twice over. Adversarial admission there is
0.000 at every rung, so the zero is a floor and not a measurement; and its identity-rung mean ASR is 0.0439,
so by §1a's bound the lower margin leg is satisfied before any seed runs. Krum under the **committed pixel
backdoor** is the cell the paper itself had already named as the remedy: baseline adversarial admission 0.333,
nonzero in 4 of 12 adversary rounds, and identity-rung mean ASR **0.3232**, above the frozen 0.15. So
*"both legs are genuine tests here"* `:2138` — a two-sided reading on this cell would be earned if one held.

### 6b. The result, with the three verdicts read separately

| estimand ($n{=}20$, seeds 42–61) | mean $\Delta$ | paired 95% CI | sign | margin | 95% bound on a **rise** |
|---|---|---|---|---|---|
| $\Delta_{\text{full}}$, Mode S at $\kappa{=}2$ | **−0.2857** | $[-0.4842,-0.0871]$ | excludes zero, negative | **outside** 0.15 | **−0.087** |
| $\Delta_{\text{score}}$, score-only at $\kappa{=}2$ | **−0.2974** | $[-0.4981,-0.0968]$ | excludes zero, negative | **outside** 0.15 | **−0.097** |

Mean ASR falls 0.3232 → 0.0375 while mean clean accuracy *rises*, 0.5158 → 0.5549, and the frozen 0.35
accuracy floor is live with no rung void, so *"so no destroyed model explains the fall"* `:2144`.

Read on the paper's own three labels: the sign verdict is that the interval excludes zero; the margin verdict
is that *"the two-sided containment this paper reports on the flagship cell fails here"* `:2144`; and the
one-sided verdict is an upper bound of −0.087 on a rise, which is not merely inside the frozen 0.15 but below
zero. *"Those are three independent labels and none is reported as another"* `:2144`.

**This arm is the exact mirror of the flagship, and that is why it matters for Section 1.** On the flagship
cell the two-sided reading passes on a leg that was never at risk. Here, where both legs bind, the two-sided
reading fails and the one-sided non-increase holds more strongly than on any arm in the reachability table.
The relabelling of Section 1 is therefore substantive rather than cosmetic, and we now have a measurement that
shows it rather than an argument that asserts it.

### 6c. The outcome fell in a branch our own freeze did not enumerate, and we say so

The freeze registered **no sign**, because two of our own prior measurements pointed opposite ways on this very
cell. It enumerated four branches — confirming, refuting, a small consistent rise it called the one to expect,
and unresolved. Our result is none of them: *"none of them covers what happened"* `:2140`, because
*"All four presuppose that a move which leaves the margin is upward, and the measured move leaves the margin downward"* `:2140`.

We did not round the result to the nearest branch. The paper reads: *"we report that as a defect in the freeze rather than assign the result to the nearest branch it did name"* `:2138`.
What the freeze *does* bind — the three verdicts, the prohibition on substituting one for another, and the rule
that no mechanism may be written into a verdict literal — is what the paragraphs are held to, and one
inference we draw from the result is explicitly labelled: *"That inference is post hoc and is marked as such"* `:2150`.

### 6d. Review §25, answered exactly: score-only Krum at all 20 seeds with a paired CI

$\Delta_{\text{score}}$ is the review's requested intervention — the magnitude channel closed, scoring on the
transformed stack and emitting the selected client's *original* update — at all 20 seeds with a paired
interval. *"Closing the magnitude channel moves the effect by"* `:2148` 0.012, and no seed's two differences
disagree by more than 0.105, so the route to the fall is not the magnitude or geometry of the aggregated
update.

Two honest limits on that answer. First, the published score-only control on the *flagship* cell remains at
$n{=}5$ and is **not** restated at $n{=}20$: *"this is a second cell rather than a top-up of that one"* `:2148`.
Second, this leg was staged behind the first and the artifact records its running condition, so it would have
been skipped had $\Delta_{\text{full}}$ fired the refuting branch. The staging could therefore only ever have
withheld a control following a result unfavourable to nobody; it could not have suppressed an unfavourable one.

### 6e. The sharpest fact in the arm is a per-seed identity, not a mean

The cell is bimodal, and the imported identity arm reproduces the published count exactly: 6 of 20 seeds above
ASR 0.5, 12 below 0.10. *"Under the dose the high mode is empty"* `:2146` — no dosed seed reaches 0.11, all six
high-mode seeds fall below 0.10, and none moves the other way, so the effect is directional seed by seed
rather than an artifact of averaging.

And the reachability arithmetic of Section 1 holds **per seed**:
*"The seven seeds whose identity ASR exceeds"* `:2146` 0.15 are exactly the seven whose difference falls past
−0.15, on both estimands. Thirteen identity rungs sit below 0.15, and those thirteen individually cannot fall
past the margin however good the defense becomes. A cell can clear the margin on its identity mean while most
of its seeds cannot, which is the point of Section 1 restated at the level of a single run.

### 6f. What this does and does not identify against our two oracle-free arms

Both published arms on this cell move ASR **up** by about a third: App. D.7's leg 1 reads +0.2969
($95\%$ CI $[+0.0983,+0.4955]$, $n{=}20$) and App. D.8's boundary ladder +0.3218 ($[+0.1680,+0.4757]$,
$n{=}20$). The first is directly comparable: it is also a score-only arm, at the same 20 seeds, against
identity rows that this arm imports **bit-identically** — the harness asserts seed 42's published accuracy and
ASR, 0.3962 and 0.7787777777777778, are reproduced at $\Delta = (0,0)$ on both quantities, and
*"which is why both estimands share one subtrahend"* `:2142`. That check was blocking, not cosmetic:
*"The check is blocking rather than decorative"* `:2142`, since the published rows ran with no adversary mask
and this path passes a real one.

So two score-only interventions on one cell against one baseline return +0.2969 and −0.2974, near mirror
images. **That removes one of the two confounds** the paper had disclosed about App. D.7's refutation: holding
the cell and the attack fixed, the disagreement survives, so the cell is not a sufficient explanation of it.
**It does not identify the cause**, and the paper says so: *"What this does not do is identify the cause"* `:2150`,
because oracle access and dose magnitude both still differ ($\rho \in [1.306,2.194]$ there against
$\rho = 54.598$ here). App. D.7's standing summary, that the oracle is load-bearing for our negative rather
than a laboratory convenience, is untouched, and so is its statement that we have not produced an oracle-free
measurement of the statistic channel: *"this arm is masked, so it is not one"* `:2150`.

### 6g. What the arm does not measure, and what it does not establish

`run_one` returns clean accuracy and ASR only, so
*"no channel quantity and no admission quantity is measured by this arm at all"* `:2152`. The 0.333 baseline
admission that made the cell eligible is **imported** from the frozen measurement, not re-measured, and
Krum's decision-change rate under this dose on this cell is simply unknown. Nothing here bears on whether
admission predicts security, which the paper denies establishing. Nor is the improvement a defense:
*"the improvement is not a defense"* `:2152` — Mode S needs an `adv_mask` no deployed server has, and
score-only Krum aggregates an update its own scoring stage never saw.

One aggregator, one attack, one dataset, one architecture and two rungs license nothing of the form *the
dissociation holds generally*, and nothing of the form *it fails generally*.
*"No published row is revised"* `:2154`: the flagship's −0.010, $95\%$ CI $[-0.032,+0.012]$ at $n{=}20$ stands
exactly as printed, and this is a different cell rather than a correction to it. No pre-registered directional
prediction is refuted, because the freeze registered no sign. And *"The interval is wide"* `:2154` at ±0.199,
which a bimodal cell at 20 seeds buys, so the magnitude is poorly determined even though its sign and its
position relative to the margin are not.

### 6h. One correction the review did not ask for, found while writing this section

Checking this cell's per-seed accuracies against a neighbouring published paragraph showed that one sentence
there was false. It had claimed that no rung on App. D.7's leg 1 comes within 0.10 of the 0.35 accuracy
floor; *"and that was false of the identity arm"* `:2096`, three of whose twenty rungs do, at 0.3962, 0.4261
and 0.4474. *"We correct it here rather than drop it"* `:2096`, because App. D.8 draws a contrast against it,
and that contrast is now quantitative rather than absolute: *"nine of forty, against three of forty there"* `:2125`,
*"which is the count that paragraph now states after correcting an earlier claim of none"* `:2125`. Nothing
else in either paragraph changes, and no verdict depended on the sentence.
