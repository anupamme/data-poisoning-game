# Pre-registration: seed top-up of the sign-reversal cell (Round 63)

**Status: frozen before any write to `results/reversal_seed_topup/`.** The seed list, the endpoint
grid, the primary interval and the demotion clause below are reproduced verbatim in
`experiments/run_reversal_seed_topup.py`, which refuses to start until this file is committed and its
hash is recorded in `PREREG_COMMIT`.

## What is being extended, and what is not

The paper's centerpiece is one cell: `coord_median` under `committed_pixel` on CIFAR-10 /
`cifar_cnn`, estimated two ways. The outcome-gated ladder (`dose_kappa<κ>`, adversary free to
attenuate) reads ΔASR = **−0.272**; the Mode-S intervention (`doseS_kappa<κ>`, adversary pinned at
c = 1) reads **+0.098**. Both 95% intervals exclude zero and they do not overlap, which is the sign
reversal the paper reports in the abstract, in §1, in §5, in the Conclusion and in Figure 1(c).

**Both legs are at n = 5.** The `n = 20` top-up already in the paper (`dose_seed_topup`, frozen at
`684b31e`) covers a *different* cell: the `krum` / `committed_scaling` equivalence arm. The centerpiece
has never been run beyond five seeds.

This document adds seeds so that the **separation between the two designs** can be stated at n = 20.
It does not re-test the reversal's direction, does not revise any threshold, and does not touch the
four other cells.

Per-seed endpoint ASR as published, seeds 42–46 in order, recomputed from the artifacts rather than
transcribed from the paper:

| leg | rung | per-seed ASR | rung mean | rung mean acc. |
|---|---|---|---|---|
| both (shared) | κ = 0 | 0.3946 / 0.2867 / 0.4488 / 0.6774 / 0.4051 | 0.4425 | 0.7665 |
| confounded (`dose_kappa2.0`) | κ = 2 | 0.1793 / 0.0897 / 0.2038 / 0.1920 / 0.1874 | 0.1704 | 0.6898 |
| controlled (`doseS_kappa2.0`) | κ = 2 | 0.5973 / 0.3191 / 0.5164 / 0.7557 / 0.5143 | 0.5406 | 0.6984 |

Paired per-seed differences `ASR_s(κ=2) − ASR_s(κ=0)`:

| leg | per-seed d | mean | sd | 95% interval (t₄ = 2.776) | half-width |
|---|---|---|---|---|---|
| confounded | −0.2152 / −0.1970 / −0.2450 / −0.4854 / −0.2177 | **−0.2721** | 0.1205 | [−0.4217, −0.1225] | 0.1496 |
| controlled | +0.2028 / +0.0324 / +0.0677 / +0.0782 / +0.1092 | **+0.0981** | 0.0646 | [+0.0178, +0.1783] | 0.0802 |

These reproduce the paper's `(−0.272, +0.098)` and its two printed intervals exactly, which is the
precondition for extending them. Holding each sd, `n = 20` (t₁₉ = 2.093) gives half-widths **0.0564**
and **0.0303**, a factor of **2.65** narrower on both legs. That factor is the entire purpose of the
run.

## Why this is not optional stopping, stated before any new seed runs

Adding seeds after seeing a result is the practice this paper condemns elsewhere, so the licence has
to be earned in advance and in writing.

1. **The published claim already holds at n = 5.** Both intervals exclude zero and they do not
   overlap (`−0.1225 < +0.0178`, a gap of 0.140). The top-up therefore cannot rescue a claim that is currently
   failing — there is no such claim. It can only narrow two intervals that already separate, or
   reveal at larger n that the separation was an artifact of five seeds. Only the second outcome is
   news, and it is news against us.
2. **The demotion clause** below pre-commits us to publishing that second outcome in the abstract and
   in Figure 1, at the cost of the paper's centerpiece. Naming that downside in advance is the only
   thing that licenses adding seeds.
3. **The seeds are fixed here**, contiguously, with no look-ahead: **47–61**, 15 new seeds, n = 20
   total. No stopping rule, no interim look, no extension of this list. If the runs are interrupted,
   the analysis reports the n actually reached and the intervals at that n; it does not resume until a
   threshold is crossed.
4. **Precedent, not innovation.** `dose_seed_topup` and `emit_only_topup` (both frozen at `684b31e`)
   extended arms from 5 seeds to 20 for the identical reason — interval width, not a second test.
   `dose_seed_topup` ran exactly 47–61 as a top-up block, which is the block reused here;
   `emit_only_topup` ran the full 42–61 in one suite. This is the same move on the cell that turned out
   to matter most.

## The primary quantity, and the endpoint grid

- **Primary, per leg: the paired interval.** `d_s = ASR_s(κ=2) − ASR_s(κ=0)`; the two-sided 95%
  Student-t interval on `mean(d)` with `n − 1` degrees of freedom. Pairing is real by construction: a
  seed fixes the Dirichlet partition, the model initialization and the participant sampling stream, so
  the two rungs at a given seed differ only in `d₁`.
- **Primary, across legs: whether the two intervals overlap**, and whether each excludes zero. This is
  the quantity the paper's sign-reversal claim rests on. **No p-value is attached to the difference
  between the two designs**, exactly as `main.tex` already states: they are not the same intervention,
  and `Λ_a` moves in both.
- **Endpoint rungs only: κ = 0 and κ = 2.** This is precisely what the frozen primary contrast of
  `pre_registration_comparability.md` reads (`LO, HI = "0.0", "2.0"` in `analyze_comparability.py`),
  so the top-up adds no new inferential surface.

**The consequence of endpoint-only is declared here rather than discovered later.** The interior rungs
κ ∈ {0.5, 1.0} stay at n = 5 for the new seeds' absence. Therefore:

- The four-rung Jonckheere–Terpstra statistics on this cell **stay at n = 5**, stay `pre_registered:
  false`, and stay post hoc. The top-up buys nothing for them and must not be read as if it had.
- **Any display of this cell's four-rung ladder must print n per rung.** A figure or table that shows
  four rungs without per-rung n after this top-up is reporting a mixed-n grid as a uniform one, which
  is the same misreport `rung_coverage` was added to prevent.
- `rung_coverage` for this cell will report κ = 0 and κ = 2 at n = 20 and the interior at n = 5. That
  is the honest shape and it is not to be smoothed.

## Decision rules

Nothing here is a new threshold. The accuracy floor and the endpoint grid are carried forward
unchanged from `pre_registration_comparability.md`.

- **Reversal confirmed at n = 20:** the confounded interval lies entirely below zero, the controlled
  interval entirely above zero, and the two do not overlap. Reported as the published result,
  narrowed.
- **Reversal demoted at n = 20 (the demotion clause):** if **either** interval contains zero, **or**
  the two intervals overlap, the paper reports the sign reversal as **not established at n = 20** and
  demotes it to a design *disagreement*. That demotion is made in the abstract, in §1, in Figure 1(a)
  and 1(c), in §5 and in the Conclusion — not in a footnote and not only in the appendix. The n = 5
  result is reported alongside it, and the disagreement between the two n's is the finding.
- **Direction change:** if either leg's mean changes sign at n = 20, that is reported as its own
  result and the demotion clause fires regardless of the intervals.
- **Accuracy gate.** Unchanged and inherited verbatim from `pre_registration_comparability.md:75`
  and `:149`: floor **0.35 applied to a rung's mean**, not per seed, and *"seeds below it are flagged,
  never excluded."* The runner records per-seed clean accuracy for every new run — `analyze_comparability.py`
  itself reads no accuracy, so the flagging is the runner's job and is written into the artifact rather
  than left to the analyzer.
- **No admissibility gate applies to this cell**, and that is deliberate. Cell 7's two-sided
  `[0.15, 0.85]` identity-rung gate (Amendment 4) existed to decide whether an *unrun* third dataset
  had headroom. This cell is already published at n = 5 with an identity rung of 0.4425; there is
  nothing to admit. Adding a gate now would create a licence to discard the run on its own outcome.

## Scope: which displayed numbers move to n = 20, fixed in advance

`+0.098` has **two referents** in the paper and they are sourced from the same directory, so the split
is fixed here before any result exists:

- **Moves to n = 20:** the comparability cell's two means and two intervals — `main.tex`'s abstract,
  §1, §5, the seven-cell table, the tiers table's replication row, the appendix's discussion of this
  cell, and Figure 1(c), which reads the artifact.
- **Stays at its own frozen n = 5:** Table 1 (`tab:channels`) row 4's Mode-S endpoint ΔASR. Table 1 is
  the Mode-S channel suite with its own freeze and its own four aggregators; its rows are n = 5
  throughout and mixing one row to n = 20 would make the table's rows incomparable.
- **Every site quoting −0.272 or +0.098 carries an explicit n** after this round, so the two referents
  cannot be read as one number. If the n = 20 controlled mean happens to round to +0.098 as well, both
  sites still print their own n.

## The shared identity rung, verified rather than assumed

At κ = 0 the transform returns the update list unwrapped, so `dose_kappa0.0` and `doseS_kappa0.0` are
the same computation. **This is already true bit-for-bit in the published artifacts**: for all five
seeds 42–46, `results/dose_response/` and `results/dose_replication/` carry identical float reprs at
κ = 0 in **both** ASR and clean accuracy (`0.39455555555555555`, `0.2866666666666667`,
`0.4487777777777778`, `0.6774444444444444`, `0.4051111111111111`). So the identity rung is computed
**once** per new seed and shared by both legs, giving 15 × 3 = **45 runs**, not 60.

`run_reversal_seed_topup.py --harness-check` re-establishes this rather than inheriting it: it computes
κ = 0 **in-suite at seed 42, where both published values already exist**, and asserts agreement with
both to `< 1e-9`. A published seed is used deliberately; at a new seed there is nothing to compare
against and the check would be vacuous. One run settles it for all fifteen new seeds because the code
path does not depend on the seed. **If that assert fails the top-up does not run**, because the shared
κ = 0 rung would then not be one rung.

The same check also reproduces a published κ = 2 value for this arm, so the harness is proved to be the
same loop the four published cells were run with, not merely the same at the identity.

## Caveats recorded in advance

1. **The sds are estimated from five seeds.** The projected half-widths of 0.0564 and 0.0303 assume
   `sd = 0.1205` and `0.0646` carry; the realized sds may be larger, and the realized intervals are
   what is reported. **No projected number appears in the paper.** For the intervals to touch at
   n = 20, the pooled sd would have to inflate by roughly 3×, which is stated as arithmetic and not as
   a prediction.
2. **Narrowing two intervals does not widen the claim.** At n = 20 this remains one aggregator, one
   attack, one dataset, one architecture, and one synthetic instrument that reads adversary identity.
   The top-up buys precision on the sign reversal and nothing else. It is not evidence about any other
   cell, it is not a second dataset, and Mode S is not a defense.
3. **The reversal's replication on CIFAR-100 stays at n = 5** and is not extended here. So after this
   round the reversal is n = 20 on CIFAR-10 and n = 5 on CIFAR-100, and both n's are printed wherever
   the replication is claimed.
4. **The controlled κ = 2 rung mean is 0.5406 at n = 5**, above 0.5. No rule in this cell's
   pre-registration reads a 0.5 ceiling — that clause belongs to the *equivalence* arms, where a
   generous margin must not certify a high-ASR composition. It is recorded because it looks like a
   gate violation and is not one.
5. **Seeds 47–61 are already used by `dose_seed_topup`, and 42–61 by `emit_only_topup`.** Different
   arms, different cell keys, own results directory, no collision — recorded because the overlap looks
   like one. Sharing the block is deliberate: every n = 20 arm in the paper then rests on the same
   seeds.
6. `results/dose_response/summary.json` and `results/dose_replication/summary.json` are **not written**
   by this suite. The top-up writes its own directory and the two are merged only at analysis time,
   where the five published seeds are asserted to reproduce before any pooled number is printed.

## Non-negotiables

1. No interval definition, accuracy floor, endpoint grid or demotion criterion above is revised after
   seeing an outcome.
2. The seed list is 47–61. It is not extended, truncated by inspection, or filtered.
3. Every reported number is recomputed per seed from the artifacts, never transcribed.
4. `results/dose_response/` and `results/dose_replication/` are not rewritten, and no existing runner
   is edited.
5. **`analyze_comparability.py`'s `PUB` reproduction guard is not weakened.** `"coord_median / pixel":
   (-0.272, +0.098)` stays, asserted against the **n = 5 subset** of the merged data. The n = 20 values
   are added as a separate expectation. Editing the published tuple to match a new result would destroy
   the only check that the re-score reproduces the document.
6. **The published n = 5 verdict is reported alongside the n = 20 verdict, whatever the latter is.** If
   they disagree, both appear, and the disagreement is the result.
7. This file is committed before `results/reversal_seed_topup/` is written; if that ordering cannot be
   demonstrated from `git log`, the top-up is reported as non-prospective.

---

# AMENDMENT 1, correcting a false factual claim above, before any write to `results/reversal_seed_topup/`

## What was wrong

The Scope section says, of Table 1 (`tab:channels`):

> *"Table 1 is the Mode-S channel suite with its own freeze and its own four aggregators; its rows are
> n = 5 throughout and mixing one row to n = 20 would make the table's rows incomparable."*

**"Its rows are n = 5 throughout" is false.** `experiments/build_channel_table.py` emits `n_seeds` per
row and prints it in its own console table. The six rows the paper prints are:

| Table 1 row | n |
|---|---|
| Krum | 5 |
| Reputation | 5 |
| Cosine-Krum | **8** |
| Coord.\ median | 5 |
| Krum (EMNIST-byclass) | **3** |
| Krum, score-only | 5 |

So Table 1 already spans **n ∈ {3, 5, 8}**. `n_seeds` is `min(n0, n2)` over the row's two rungs
(`build_channel_table.py:239` and `:265`). `seeds_match` is emitted only on the paired branch (`:239`);
the score-only row takes the other branch (`:265`) and carries no `seeds_match` field at all, which
`:462` reads as `True` by default. None of these quantities is printed in the paper.

This was found by running the emitter, not by reading the paper, and it was written into a frozen
document. It is corrected here rather than edited in place: the original sentence stays above, wrong,
with this amendment attached.

## The decision it was offered as a reason for does not change

Table 1 row 4's Mode-S endpoint ΔASR still **stays at its own frozen n and does not move to n = 20**.
The stated reason was wrong; the operative reason was always the other clause, and it survives intact:
Table 1 is a **different suite measuring a different quantity** — the four channels of one Mode-S
intervention at the top rung, assembled from frozen artifacts with no ASR recomputed — and the
comparability cell is a paired two-design contrast. They share the number `+0.098` and nothing else.
Moving one row of one suite because a different suite gained seeds would be a category error whatever
the other rows' n happened to be.

If anything the correction **strengthens** the prohibition: a table whose rows already span three
different n is one where an unannounced fourth would be least visible.

The prohibition is also **enforced mechanically and not only by intent**, which is recorded here so it
can be re-checked rather than re-argued. `build_channel_table.py` reads its Coord.\ median ASR from
`results/dose_replication/summary.json` by an explicit path, with no glob and no merge list; the top-up
writes only `results/reversal_seed_topup/` and non-negotiable 4 forbids rewriting `dose_replication`.
So Table 1 row 4 cannot silently move to n = 20 even if someone later forgets this clause. The merge is
in `analyze_comparability.py` alone, and it applies to the comparability cell alone.

## What it does change: the consequence is larger than recorded

The original Scope section required that *"every site quoting −0.272 or +0.098 carries an explicit n"*.
That obligation now reaches further than one number, because the defect is not confined to the cell
this top-up touches:

1. **Table 1 must print `n` per row, or state the three values in its caption.** Six rows spanning
   n = 3, 5 and 8 with no seed count anywhere is the reviewer's evidentiary-hierarchy complaint stated
   exactly, and it is present in the main text independently of this top-up. The `n` is already emitted
   per row, so this is a reporting gap and not a measurement one. Whether it becomes a column or a
   caption sentence is decided by the overfull-hbox check and not by preference.
   **The emitter's LaTeX block is not the paper's table.** It prints **seven** rows — a `Krum,
   emit-only` row (n = 5, ΔASR −0.001) that Table 1 does not carry — so the `n` must be transcribed
   per row into the existing six, never pasted wholesale. Pasting would add a row and call it
   formatting.
2. **Figure 1(c) prints no `n` on any row either.** Its rows already span n = 5, 8 and 20, and its two
   draw-time guards read `published_cells_reproduce` and the reversal-row list, neither of which reads
   `n`. After this top-up the panel would draw two sign-reversal rows at **different n** as visually
   identical objects. The panel must carry per-row `n` before the topped-up artifact is drawn.

Neither item is a new experiment, a new threshold, or a change to any decision rule, seed list,
interval definition or demotion criterion above. They are reporting obligations, and they are recorded
here so that they are met because they were pre-registered rather than because they were noticed late.

## Non-negotiables, extending the seven above

8. No number, rule or seed list above is altered by this amendment. It corrects one false statement of
   fact and widens one reporting obligation.
9. The false sentence is not deleted or rewritten in place. A frozen document that quietly becomes
   correct cannot be audited.

---

# AMENDMENT 2, fixing the reporting split site by site, before any write to `results/reversal_seed_topup/`

## Why this cannot wait until the result exists

Amendment 1 widened the obligation that *"every site quoting −0.272 or +0.098 carries an explicit n"*.
A grep of the three sources for those two digit strings, comment lines excluded, returns **22 sites in
`paper/main.tex`, 2 in `paper/supplementary.tex` and 22 in `workshop_paper/main.tex`**, and they do not
all quote the same quantity.

Every `:NNN` below is the line number **as of this commit**, and the classification is by the *sentence*
at that line, not by the number itself: re-papering shifts every later line, so a site is located by
grepping its quoted words and not by trusting these integers. The 22 `main.tex` sites are `:66`, `:129`,
`:403`, `:437`, `:445`, `:699`, `:878`, `:1269`, `:1655`, `:1665`, `:1684`, `:1693`, `:1695`, `:1746`,
`:1773`, `:1801`, `:1865`, `:2185`, `:2215`, `:2273`, `:2314`, `:2515`, and every one of them is
assigned a class below, so the split is exhaustive rather than illustrative.

The Scope section above splits the referents at the level of *tables* (the comparability cell
moves, `tab:channels` row 4 stays). That is not fine-grained enough, because one class of site is
**outcome-sensitive** and would otherwise be decided after the number is known:

`main.tex:437` and `:1693` read *"We find $\Delta=+0.098$, which refutes the admission ordering"*,
evaluated against thresholds frozen before those runs existed: **confirm if $\Delta > +0.178$, refute if
$\Delta < +0.150$**. If the n = 20 controlled mean lands above $+0.178$, then whether that site is
updated decides whether a frozen pre-registered verdict flips. Choosing then is choosing on the
outcome. So it is chosen now, blind.

## The split, fixed

**Class A, moves to n = 20 (the sign-reversal pair, both legs, with n printed at every site):**
`main.tex` `:66` (abstract), `:129` (§1), `:445` (§5's confounded leg), `:878` (the comparability
table's ASR-effect row), `:1695` (App. E's central-result paragraph), `:1865` (the seven-cell table's
`coord_median/pixel` row and its two intervals), Figure 1(c) (reads the artifact), and
`supplementary.tex` `:386`, `:456`. These quote the two designs *against each other* on one cell, which
is the quantity this top-up buys precision on.

**Class B, stays at its own frozen n = 5, with `n=5` printed so it cannot be read as Class A:**

1. `tab:channels` row 4 at `main.tex:403` and `:1746`, per the Scope section and Amendment 1. The `n`
   column added in Round 63 already prints it.
2. **`main.tex:437` and `:1693`, the admission-ordering refutation, and its dependants `:1269`,
   `:1665`, `:2185`, `:2215`.** The frozen rule was stated over a five-seed arm and evaluated once;
   re-evaluating it at n = 20 would revise a pre-registered decision after seeing new data, which
   non-negotiable 1 forbids. The n = 20 controlled mean **may be reported beside it**, and if the two
   fall on opposite sides of $+0.150$ or $+0.178$ that disagreement is reported in full (non-negotiable
   6) as a disagreement between two n's, never as a re-run verdict.
3. `:1773` (the score-only disclosure that the `coord_median` arm is not magnitude-controlled), `:699`,
   `:1655`, `:1684`, `:2273`, `:2314`. Each of these quotes the frozen replication arm or a frozen
   table row, not the paired contrast.

**Class C, not touched at all:** every `workshop_paper/main.tex` site. That paper is out of scope for
this round by standing rule, and its numbers stay as published.

`main.tex:1801`'s $+0.272$ is a **different cell's** value that merely shares the digits, and `:2515`'s
$0.0988$ shares three of them. Neither is in scope, and they are named here so a later grep does not
sweep them in.

## Non-negotiables, extending the nine above

10. The class of each site is fixed by this amendment and is not reassigned after the n = 20 numbers
    are known. A site moved between classes later is reported as such in the response letter.
11. No frozen threshold is re-evaluated at n = 20. Where an n = 20 mean would change a frozen
    verdict's side of a threshold, both n's and both readings are printed and the frozen verdict stays
    labelled as the frozen one.
12. Every Class A and Class B site prints its own `n`. A site quoting either number with no `n` after
    this round is a defect, not a style choice.

---

# AMENDMENT 3, correcting one misclassified site and binding the adjacency, still before any row exists

## The state of the run when this section was written, and the limit of that evidence

At the moment this text was appended, `results/reversal_seed_topup/` **did not exist**: the runner's log
showed `resuming: 0 rows already recorded (45 runs planned, shared rows counted once)` and not one
`[i/45]` line, so zero of the 45 runs had been scored and no n = 20 quantity existed on disk to be seen.

**That claim is about the append, not about the commit, and the distinction is the honest one.** Unlike
Amendments 1 and 2, this section is written while the suite is already running, so **no claim is made
about where its commit falls relative to the artifact's first write.** It may fall either side: the
first of the 45 runs had not yet finished when this was appended, so the commit may well precede
`results/reversal_seed_topup/summary.json`, but that would be a fact about run duration and not
evidence of anything, and a later commit would equally be no cause for suspicion. The blindness argument
deliberately does not rest on ordering at all: it rests on the correction being **decidable from
`paper/main.tex` alone**, with no n = 20 number as an input, and on its moving a site *out* of scope,
which no result could make attractive. A reader who distrusts the timestamp can re-derive the whole
correction from line 1655 of the paper.

## The defect

Amendment 2 assigned `main.tex:1655` to Class B item 3, on the stated ground that it "quotes the frozen
replication arm or a frozen table row". **It quotes neither.** The line reads

> ASR rises $0.124 \to 0.315 \to 0.443$ into $r{=}1$ and falls $0.443 \to 0.296 \to 0.098$ past it

so its `0.098` is a **mean ASR at an amplified Mode A rung** of the payload-dose curve, not a $\Delta$ASR
at all. It shares three digits with the quantity this top-up extends and nothing else: different
experiment (Mode A dose, not Mode S endpoint), different estimand (a level, not a paired difference),
different table. Amendment 2 swept it in by grepping the digit string and then classifying the line
without re-reading what the digits denote, which is the exact failure the Scope section warns about two
sites down.

## The correction

`:1655` moves out of Class B and into the named-collision list, which now has **three** members:
`:1655` (a Mode A rung level), `:1801` (a different cell's $+0.272$), `:2515` ($0.0988$, sharing three
digits). Class B item 3 becomes five sites: `:699`, `:1684`, `:1773`, `:2273`, `:2314`.

The split stays exhaustive and disjoint over the 22 non-comment sites: **6 Class A + 13 Class B + 3
collisions = 22**. Non-negotiable 10 already requires a reclassification to be reported as such, and it
will be. This correction moves a site *out* of scope, so it cannot flatter any result, and it is
verifiable from `paper/main.tex` alone with no n = 20 number in hand.

## Which Class B sites satisfy non-negotiable 12 today

A site prints its own `n` if the `n` appears in the same sentence, or in an `$n$` column of its own row,
or in its own float's caption. Audited at this commit:

- **Satisfied (6):** `:403` and `:1746` (`tab:channels` `$n$` column, row 4 prints `5`), `:1665`
  (`tab:targeted`'s caption, "$n{=}5$ per arm"), `:1684` (a row inside that same float), `:1773` (its
  own sentence, "at $n{=}5$ a null from this test is not evidence of flatness"), `:2314`
  (`tab:tost`'s `$n$` column prints `$5$`).
- **Not satisfied (7):** `:437`, `:1693`, `:2185`, `:2273` in prose; `:1269` (`tab:can_cannot`, whose
  caption states no `n`), `:2215` (`fig:story`, likewise), `:699` (`tab:tiers`, likewise).

Those seven are the work non-negotiable 12 obliges, and the list is fixed here so it is not narrowed
later to whatever turned out to be convenient to edit.

## The adjacency, which the table-level split also hid

`:437` (Class B, the frozen refutation) and `:445` (Class A, moves to n = 20) sit **eight lines apart in
the same section**, and both quote $+0.098$. After the top-up one prints the frozen five-seed value and
the other prints a twenty-seed value for what is arithmetically **one estimand**: the
`dose_replication` Mode S endpoint $\Delta$ASR, interval $[+0.018,+0.178]$, which is simultaneously the
comparability cell's controlled leg. The Class A / Class B split is by *reporting role*, not by
quantity, and two different numbers for one estimand eight lines apart is a reader trap even when every
site is individually correct.

13. Where a Class A site and a Class B site quoting the same estimand fall in the same section, that
    section states in one clause that the two differ in $n$ and not in quantity. Printing both numbers
    with their $n$'s and no such clause is a defect, not compliance with 12.
