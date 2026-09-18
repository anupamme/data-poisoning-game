"""
Two-panel targeted-intervention figure. Each panel carries ONE message.

Panel (a) -- the adjudicator (Mode S, Krum / model-scaling). The statistic-only instrument pins every
adversary at c=1.0 (adversarial coefficient share constant at 0.266667, spread 3.5e-7), so Lemma 1's
attenuation channel is closed by construction. Krum's *decision* (its selected client) changes in up
to 80% of rounds as the benign spread grows, while the *admission* of adversarial input it makes is
never disturbed in any measured round -- and ASR stays flat and low. Statistic disturbance without
admission change does not destroy suppression: the four-levels distinction, made visible on the one
arm whose two readings predict opposite outcomes. NOTE that on THIS arm Krum admits no adversary at
any rung, so its zero is a floor; the arms that make it a measured non-event are reputation and
coord_median, where adversarial mass is present at baseline in 48 of 60 rounds and the support still
never moves (experiments/count_admission_rounds.py).

Panel (b) -- Mode A (payload-only). Benign uniform, adversary-to-benign ratio r = e^nu, mean 1, so
the adversarial coefficient share sweeps 0.049 -> 0.710 while the statistic's dispersion is fixed.
coord_median (a rank aggregator, the class Regime A is stated for) is single-peaked at r=1 -- both
mechanism-preserving regimes visible, Theorem 1 confirmed. reputation (not a rank aggregator) is
monotone: an attenuated adversary reads as consistent and lands its payload, refuting the two-regime
shape off-spec.

Sources (recomputed from JSON, never transcribed -- pre-registration non-negotiable 3):
  results/targeted_dose/summary.json      the 97 runs, per seed
  results/admission_measurement.json      the decision / admission channels (no ASR)
The per-arm loading (rungs_of) and the constants are IMPORTED from analyze_targeted_dose.py, so the
figure cannot drift from the scored numbers.

Writes targeted_dose.pdf next to this script and to the sibling paper's figures/ dir.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from experiments.analyze_targeted_dose import (  # noqa: E402
    ACC_FLOOR, ADM, EQUIV_MARGIN, KAPPAS, NUS, TARGETED, rungs_of)
# Imported, not re-implemented: the DAG panel's admission count must be the SAME per-round indicator
# and the SAME arm/attack pairing the published channel table uses, or the figure and the table can
# disagree about what "the support of the adversarial mass changed" means.
from experiments.build_channel_table import (  # noqa: E402
    MASS, ROWS, SRC_REGIME, channel_rows)
# Panel (c) reads BOTH its point estimates and its intervals from this artifact and computes neither,
# so the figure cannot report a different number, or a differently-derived interval, from the caption
# and the body that quote the same generator.
COMPARABILITY = os.path.join(REPO, "results", "comparability_table.json")
SIXCELL = os.path.join(REPO, "results", "comparability_six_cells.json")

cells = json.load(open(TARGETED))["cells"]
adm = json.load(open(ADM))
summary = adm["summary"]

RHO = [float(np.exp(2 * k)) for k in KAPPAS]        # panel (a) x-axis, the dose the theorem is stated in
GAMMA = [float(np.exp(v)) for v in NUS]             # panel (b) x-axis

def draw_modeS(ax, tag="(a) ", compact=False):
    """Mode S, Krum: the adjudicating arm. Statistic moves, admission does not, ASR does not.

    compact=True is the stacked-figure variant: smaller type, extra headroom so the legend sits in
    white space rather than over the decision curve, and the two annotations moved off the traces.
    Same data, same artifacts -- only the typography differs.
    """
    FS = 5.6 if compact else 7.2          # in-panel annotations
    FL = 6.6 if compact else 8.5          # axis labels
    FT = 5.6 if compact else 7.5          # tick labels
    FG = 5.0 if compact else 6.6          # legend
    # --------------------------------------------------------------- panel (a): the adjudicator
    kr = rungs_of(cells, "S", "krum", KAPPAS)
    asr = [r["mean"] for r in kr]
    asr_err = [[r["mean"] - r["lo"] for r in kr], [r["hi"] - r["mean"] for r in kr]]
    dec = [summary[f"doseS|krum|{k}|decision"] for k in KAPPAS]
    admn = [summary[f"doseS|krum|{k}|admission"] for k in KAPPAS]

    ax.axhline(0.5, color="0.35", lw=0.9, ls=":", zorder=1)
    ax.text(RHO[0] * 1.02, 0.545 if compact else 0.52, "suppression threshold",
            fontsize=FS - 0.6, color="0.35", va="bottom" if compact else "baseline")
    ax.plot(RHO, dec, marker="^", ms=6.5, lw=1.5, ls="--", color="#a9780a",
            label="decision change (selected client flips)", zorder=4)
    ax.plot(RHO, admn, marker="s", ms=6.5, lw=1.5, ls="-.", color="#2f7d3f",
            label="admission change (adversarial input admitted)", zorder=4)
    ax.errorbar(RHO, asr, yerr=asr_err, marker="o", ms=6.8, lw=1.8, capsize=2.6, color="#b03a2e",
                label="max-committed ASR", zorder=5)

    if compact:
        # No legend and no in-plot annotations: each series is labelled at its own right-hand end, in
        # its own colour, with a short leader. In a 1-inch-tall panel any box or floating caption sits
        # on top of a trace, and the three labels ARE the legend.
        for x, y, ytx, txt, col in (
                (RHO[-1], dec[-1], dec[-1], f"decision change:\nup to ${max(dec):.2f}$ of rounds",
                 "#a9780a"),
                (RHO[-1], asr[-1], 0.32, "max-committed ASR: flat", "#b03a2e"),
                (RHO[-1], admn[-1], 0.06, "admission change: never", "#2f7d3f")):
            # Dotted, thin and translucent ON PURPOSE: a solid leader in the series colour
            # reads as a continuation of the trace, so the red one made "max-committed ASR:
            # flat" look like a series rising at the right edge, contradicting its own label.
            ax.annotate(txt, xy=(x, y), xytext=(RHO[-1] * 1.9, ytx), fontsize=FS, color=col,
                        va="center", ha="left", linespacing=1.2,
                        arrowprops=dict(arrowstyle="-", color=col, lw=0.55, alpha=0.5,
                                        linestyle=(0, (1.6, 1.6)), shrinkA=2, shrinkB=2))
    else:
        ax.annotate(f"decision $\\to {max(dec):.2f}$", xy=(RHO[2], dec[2]),
                    xytext=(RHO[0] * 1.3, 0.86), fontsize=FS, color="#a9780a",
                    arrowprops=dict(arrowstyle="-|>", color="#a9780a", lw=0.8, shrinkA=1, shrinkB=2))
        ax.annotate("admission never changes\n(pick always benign)\nASR flat",
                    xy=(RHO[3], admn[3]), xytext=(RHO[1], 0.24),
                    fontsize=FS, color="#2f7d3f", linespacing=1.25,
                    arrowprops=dict(arrowstyle="-|>", color="#2f7d3f", lw=0.8, shrinkA=1, shrinkB=2))

    ax.set_xscale("log")
    ax.set_xticks(RHO)
    ax.set_xticklabels([f"{r:.2f}\n$\\kappa{{=}}{k:g}$" for r, k in zip(RHO, KAPPAS)])
    ax.set_xlabel("dose $\\rho = e^{2\\kappa}$ (benign dispersion; adversary pinned at $c{=}1$)",
                  fontsize=FL, labelpad=1.5 if compact else None)
    ax.set_ylabel("fraction of rounds / ASR" if compact else "fraction of rounds  /  ASR",
                  fontsize=FL, labelpad=1.5 if compact else None)
    ax.set_title(f"{tag}Mode S, Krum: the statistic moves, admission does not",
                 fontsize=7.4 if compact else 9.3, loc="left", pad=2.0 if compact else None)
    ax.set_xlim(RHO[0] * 0.85, RHO[-1] * (9.0 if compact else 1.25))
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(labelsize=FT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if compact:
        # The xlim runs well past the last rung to make room for the right-end labels. Left as-is, the
        # axis line would continue into that empty span and read as data we failed to plot, so the
        # spine stops at the last measured dose while the label space stays.
        ax.spines["bottom"].set_bounds(RHO[0] * 0.85, RHO[-1])
        ax.tick_params(axis="x", which="minor", bottom=False)   # else log minors float past the spine
    if not compact:
        ax.legend(fontsize=FG, loc="upper left", framealpha=0.92, handletextpad=0.5,
                  borderpad=0.4, labelspacing=0.35)


def draw_modeA(bx, tag="(b) "):
    """Mode A: the payload dose separates Theorem 1's two regimes."""
    # --------------------------------------------------------------- panel (b): Mode A, the two regimes
    STYLE_A = {  # d2 -> (marker, color, label)
        "coord_median": ("D", "#1f5fa6", "CoordMedian / pixel (rank aggregator)"),
        "reputation": ("s", "#b03a2e", "Reputation / scaling (not a rank aggregator)"),
    }
    bx.axhline(0.5, color="0.35", lw=0.9, ls=":", zorder=1)
    bx.axvline(1.0, color="0.55", lw=0.9, ls="-", zorder=1)
    bx.text(1.06, 0.03, "$r=1$ (identity)", fontsize=6.4, color="0.4", rotation=90, va="bottom")
    for d2, (marker, color, label) in STYLE_A.items():
        r = rungs_of(cells, "A", d2, NUS)
        y = [x["mean"] for x in r]
        err = [[x["mean"] - x["lo"] for x in r], [x["hi"] - x["mean"] for x in r]]
        bx.errorbar(GAMMA, y, yerr=err, marker=marker, ms=6.5, lw=1.6, capsize=2.5, color=color,
                    label=label, zorder=4)
        peak = max(range(len(r)), key=lambda i: r[i]["mean"])
        if d2 == "coord_median":
            bx.annotate("single peak at $r{=}1$\n(both regimes)", xy=(GAMMA[peak], y[peak]),
                        xytext=(GAMMA[peak] * 1.15, y[peak] + 0.24), fontsize=7.2, color=color,
                        linespacing=1.3,
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8, shrinkA=1, shrinkB=2))
        else:
            bx.annotate("monotone: attenuated\nadversary still lands", xy=(GAMMA[0], y[0]),
                        xytext=(GAMMA[0] * 1.05, y[0] - 0.30), fontsize=7.2, color=color,
                        linespacing=1.3,
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8, shrinkA=1, shrinkB=2))

    bx.set_xscale("log")
    bx.set_xticks(GAMMA)
    bx.set_xticklabels([f"{g:.2f}\n$\\nu{{=}}{v:g}$" for g, v in zip(GAMMA, NUS)])
    bx.set_xlabel("payload dose $r = e^{\\nu}$ (adversary/benign ratio; dispersion fixed)", fontsize=8.5)
    bx.set_ylabel("max-committed ASR", fontsize=8.5)
    bx.set_title(f"{tag}Mode A: the payload dose isolates the two regimes", fontsize=9.3, loc="left")
    bx.set_ylim(-0.05, 1.05)
    bx.tick_params(labelsize=7.5)
    bx.spines["top"].set_visible(False)
    bx.spines["right"].set_visible(False)
    bx.legend(fontsize=6.6, loc="upper right", framealpha=0.92, handletextpad=0.5,
              borderpad=0.4, labelspacing=0.35)


def draw_dag(ax, tag="(a) "):
    """The causal structure the Mode-S panel is evidence about.

    Two paths run from the upstream transform T to ASR. The CHAIN the screening idea reasons along --
    T disturbs the statistic, the statistic moves the decision, the decision moves the adversarial mass
    admitted, and that moves suppression -- and the CONFOUNDED path, in which T simply attenuates
    whichever client carries the poison and reaches ASR without passing through d_2's statistic at all.
    Mode S cuts the second path by construction (every adversary pinned at c=1), which is what makes the
    first path's two broken arrows measurable rather than confounded.

    Every number annotated here is recomputed from the frozen artifacts; none is written into this
    function. The admission arrow carries a COUNT pooled over all four aggregators rather than the Krum
    row's rounded mean, because Krum admits no adversary at any rung -- its own zero is a floor. The
    pooled count reports how many of those rounds had adversarial mass present at baseline, i.e. how
    many were rounds in which a change was possible (experiments/count_admission_rounds.py prints the
    same three numbers per arm). The admission -> ASR arrow carries cos_krum's first-rung fall and NOT
    Krum's suppression delta: the witness has to change arm there, for the reason set out at its
    computation below.
    """
    TOP = KAPPAS[-1]
    dec = max(summary[f"doseS|krum|{k}|decision"] for k in KAPPAS)   # "up to", as the body says
    share = adm["adv_coeff_share"][f"doseS|{TOP}"]
    shares = [adm["adv_coeff_share"][f"doseS|{k}"] for k in KAPPAS]
    assert max(shares) - min(shares) < 1e-5, shares      # the pinning this panel claims
    # The admission -> ASR arrow needs a DIFFERENT arm than the three before it, and that is forced by
    # what the arrow asserts: (P4) does not imply (P5), so its witness has to be a case where admission
    # invariance HOLDS and suppression MOVES. Krum is the opposite case -- decision moved, suppression
    # held INSIDE the equivalence margin, which the assert below is what states -- so its delta is the
    # PREVIOUS arrow's evidence and panel (b)'s number; annotating it here read as this break's own
    # witness and so asserted the converse of the break. cos_krum is the
    # witness the body rests on: admission identically 0.000 at every rung, ASR falling by the first,
    # gate-passing rung (rho = 2.72, clean accuracy 0.599 -> 0.567). The body reports that fall as the
    # difference of the two PUBLISHED rung means (0.584 -> 0.411), so it is differenced the same way
    # here rather than at full precision; the assert holds the two conventions to the third decimal, so
    # a data change that separated them would fail here instead of drifting away from the prose.
    kr = rungs_of(cells, "S", "krum", KAPPAS)
    assert abs(kr[-1]["mean"] - kr[0]["mean"]) < EQUIV_MARGIN, kr    # why Krum is not this witness
    ck = rungs_of(cells, "S", "cos_krum", KAPPAS)
    for _k in KAPPAS:                       # the arrow's ANTECEDENT, asserted rather than assumed
        for _ch in ("decision", "admission"):
            assert summary[f"doseS|cos_krum|{_k}|{_ch}"] == 0.0, (_k, _ch)
    d_ck = round(ck[0]["mean"], 3) - round(ck[1]["mean"], 3)
    assert abs(d_ck - (ck[0]["mean"] - ck[1]["mean"])) < 1e-3, (d_ck, ck[0]["mean"], ck[1]["mean"])
    assert ck[1]["acc"] >= ACC_FLOOR, ck[1]["acc"]                   # the rung's own gate, not assumed

    # pooled over the channel table's own cells: each arm under ITS OWN committed attack.
    # Round 70: the filter used to read `if srcf == ADM_FEMNIST: continue`, i.e. it excluded by DATASET,
    # and that silently went wrong when Round 68/69 added ("Krum (ResNet18)", ..., ADM_RESNET18) to ROWS
    # -- a CIFAR-10 row that the dataset test therefore ADMITS. The pool grew 240 -> 312 rounds with no
    # edit to this file, and because nothing regenerated the PDF the committed artwork still printed
    # 0/240 while the generator would have drawn 0/312. The intended pool is the one this function's own
    # comment below and `main.tex`'s "$240$ in total" both name: the FOUR cifar_cnn arms, which is the
    # only homogeneous pool here (60 measured rounds each, against ResNet18's 72 and EMNIST's 56, so a
    # mixed denominator would also falsify "60 measured rounds each"). Exclude by (dataset, model), and
    # assert the pool, because the failure mode is a row joining ROWS upstream and nothing here noticing.
    n_rounds = n_changed = n_present = 0
    n_arms = 0
    for _, arm, attack, srcf in ROWS:
        if SRC_REGIME[srcf] != ("cifar10", "cifar_cnn"):
            continue                        # one dataset AND one architecture, so the horizon is uniform
        n_arms += 1
        field, _ = MASS[arm]
        for rung in (0.0, 0.5, 1.0, 2.0):
            for r in channel_rows(srcf, arm, attack, rung):
                b, q = float(r[f"base_{field}"]), float(r[f"post_{field}"])
                n_rounds += 1
                n_changed += int((b > 0.0) != (q > 0.0))
                n_present += int(b > 0.0)
    # The three numbers `main.tex` quotes in prose for this annotation, pinned here so a change upstream
    # fails loudly instead of redrawing the panel under a caption that no longer describes it.
    assert (n_arms, n_rounds, n_present) == (4, 240, 100), (n_arms, n_rounds, n_present)

    CH, BRK, CONF = "#1f5fa6", "#b03a2e", "#a9780a"

    def box(x, y, s, ec, fs=8.2, weight="normal"):      # Round 62: 6.3 -> 8.2, see the note below
        ax.text(x, y, s, ha="center", va="center", fontsize=fs, color=ec, fontweight=weight,
                bbox=dict(boxstyle="round,pad=0.30", fc="white", ec=ec, lw=0.9), zorder=5)

    # Round 62: XS and HW are now DERIVED from the measured half-widths at the box font size, not
    # hand-placed. HW used to read [0.036, 0.048, 0.048, 0.050, 0.030] against true half-widths of
    # [0.078, 0.060, 0.062, 0.074, 0.038] -- it was ~0.6x the real box, which is why arrows appeared to
    # start inside their boxes and why raising the type made "not implied" print straight through
    # "decision" and "admission". Anything that changes the box font size or a box's text MUST re-measure
    # these five numbers (a text's bbox_patch window extent through ax.transData.inverted()); they are
    # not guesses and they do not scale by eye.
    HW = [0.0781, 0.0595, 0.0615, 0.0741, 0.0381]   # half-widths, so arrows stop at the box edge
    # The four gaps are UNEQUAL on purpose, and that is what makes the annotations legible. Segments 2
    # and 3 carry "not implied" at 6.0pt, which is 0.112 wide and sits at the boxes' own height, so those
    # gaps have to exceed it; segment 0 is unannotated and segment 1's word moved to the caption, so both
    # can be narrow. Equal gaps of 0.0905 (what the old spacing amounted to) cannot fit the label at any
    # legible size -- that was the real constraint, not the vertical bands.
    GAPS = [0.052, 0.088, 0.118, 0.118]
    XS = [HW[0]]
    for _i, _g in enumerate(GAPS):
        XS.append(XS[-1] + HW[_i] + _g + HW[_i + 1])
    assert XS[-1] + HW[-1] < 1.005, XS                # the row still fits the panel's x range
    # Annotations centre on the GAP, not on the midpoint of two box centres: with unequal box widths the
    # two differ by up to 0.006, which at these label widths is the whole clearance.
    MID = [(XS[i] + HW[i] + XS[i + 1] - HW[i + 1]) / 2 for i in range(4)]
    Y = 0.62
    for x, s, ec, w in zip(XS,
                           ["upstream\n$T$", "statistic\n$S(d_2)$", "decision\n(P3)",
                            "admission\n(P4)", "ASR\n(P5)"],
                           [CH, CH, CH, BRK, BRK], ["bold", "normal", "normal", "normal", "bold"]):
        box(x, Y, s, ec, weight=w)

    # (P1) and (P2) -- the statistic's value and the ordering it induces -- both live in the ONE
    # statistic box, and they are named in each document's CAPTION rather than in the artwork. Not a
    # preference: this panel has no free band left. Measured with the renderer at the ORIGINAL sizes, a
    # 4.6pt label centred under that box overlapped the box itself by 1.1pt at y=Y-0.30 and the
    # attenuation box's top-left corner by 3.5pt, with no y between them (box bottom 256.0pt,
    # attenuation top 255.6pt); a third line INSIDE the box cost 7.6pt of height and drove its rounded
    # corner into the same attenuation box. Round 62 raised every size in this panel and so spent the
    # little slack that existed -- the clearances above are now SMALLER, not larger, and the composite's
    # height_ratios gave row 0 the height that paid for it. Do not read this note as free room.
    # The arrow annotations are what carry the hierarchy here, which is why the two links that hold by
    # definition needs no label and the two broken ones carry the nRightarrow glyph.

    # The reader's map, and the whole point of the panel: the left group is what a preservation check
    # can see, the right group is what security actually depends on. Set inside the existing headroom
    # (the channel annotations below drop to one line), so the tight bbox does not grow.
    for x0, x1, lab, col in ((XS[0] - HW[0], XS[2] + HW[2], "what a preservation check verifies", CH),
                             (XS[3] - HW[3], XS[4] + HW[4], "what security depends on", BRK)):
        # No span rule: the band between the box tops and the title is only a few points tall, and a
        # rule there strikes through the channel annotations. The label takes its group's colour
        # instead, which is the same cue the boxes already carry.
        ax.text((x0 + x1) / 2, Y + 0.400, lab, ha="center", va="bottom", fontsize=8.0, color=col)

    for i in range(4):
        broken = i >= 2
        ax.annotate("", xy=(XS[i + 1] - HW[i + 1], Y), xytext=(XS[i] + HW[i], Y), zorder=3,
                    arrowprops=dict(arrowstyle="-|>", lw=1.15, shrinkA=0, shrinkB=0,
                                    color=BRK if broken else CH,
                                    ls=(0, (2.4, 1.7)) if broken else "-"))
        if broken:
            # "not implied" used to print here at 6.0pt beside this glyph. At main.tex's 0.80\linewidth
            # that is 4.8pt on paper, i.e. the same illegibility Round 62 exists to fix, and the gap is
            # 0.125 wide against a 0.112 label so it cannot be enlarged in place. The GLYPH is the one
            # element that gets bigger for free -- 11pt here is ~8.8pt printed, larger than anything the
            # panel had before -- and it is the standard notation for what the words said. The words are
            # in each document's caption, which is set at body size on the same page.
            ax.text(MID[i], Y - 0.20, r"$\nRightarrow$", ha="center", va="center",
                    fontsize=11, color=BRK, zorder=6)
    # Segment 1 -- the one link in the chain that needs no experiment, because equal values induce the
    # same ordering and a rule reading only that ordering cannot decide differently, so (P2)=>(P3) holds
    # by definition -- used to be labelled "by definition" at Y-0.20, in the nRightarrow band. Round 62
    # took the words off the artwork and left them where they ALREADY were, in each document's caption
    # ("both solid links hold \emph{by definition}" -- there are TWO, matching (P1)=>(P2)=>(P3) at
    # main.tex:284, and the caption said "the solid link" until a pixel read of Round 62's own build
    # caught it), because at a legible size they no longer fit: at
    # 6.0pt the string is 0.125 wide against a 0.070 gap, it sits at the boxes' own height, and the band
    # below the boxes is 0.076 tall against a 0.168-tall label. It cannot go in the number band above
    # either -- there it prints under this same segment's own annotation and reads as "0.80 by
    # definition", asserting the converse of what the panel says. The solid blue arrow against the two
    # dashed red ones is the surviving visual cue, and the caption names both. If this label ever returns,
    # it needs a band, not a nudge.

    # The T -> statistic segment is deliberately left UNANNOTATED. Its number (Delta agg. = 0.892) is
    # the one channel Mode S does not hold fixed, so printing it here invites reading the panel as a
    # claim that displacement is controlled, which is the opposite of what the panel says. It stays
    # disclosed in prose -- the control paragraph of the targeted section names it as "the aggregate
    # the defense emits" -- and in the channel table, so nothing is lost by not repeating it inside
    # the diagram. None is the skip marker; the loop keeps its segment index either way.
    # Round 62: these were sentences ("Krum: flips 0.80", "unchanged, 0/240, 4 arms",
    # "cos_krum: falls 0.173"), and their 24-char width is what pinned the whole panel at 4.6pt -- ~3.7pt
    # once main.tex includes the figure at 0.80\linewidth, which is below print legibility. The NUMBERS
    # stay here, still recomputed above and still asserted; the scope and the direction move to each
    # document's caption, which is the same division this panel already uses for (P1)/(P2) and for
    # `lem:annihilation`'s number. Reading the bare numbers off the arrows therefore requires the
    # caption, and that is the trade the round took deliberately: three annotations nobody can read are
    # worth less than three anybody can, and the caption is on the same page.
    #
    # Two things the caption MUST carry, because dropping them from the artwork dropped them from view:
    #  - segment 3 is POOLED over the four CIFAR-10 arms of the channel table, never Krum alone
    #    (n_rounds is summed over ROWS above, so "Krum: 0/240" would be false);
    #  - d_ck is POSITIVE and the word "falls" was carrying its direction, so the caption has to say
    #    "falls". Printing "-0.173" here instead would put a sign in the artwork that the artifact does
    #    not hold, and a mathtext hyphen renders as a true minus.
    for i, lab in enumerate([None, f"{dec:.2f}", f"{n_changed}/{n_rounds}", f"{d_ck:.3f}"]):
        if lab is None:
            continue
        # ONE line each, and that is a layout constraint, not a style choice: two-line annotations reach
        # y=1.02, which leaves the group labels above them no room below the title at ylim=1.20 and they
        # print through it. At 5 chars the horizontal collision that broke the first one-line attempt is
        # gone -- the segment midpoints are only ~0.2 apart and these labels are now NARROWER than the
        # gaps they sit in, which is what pays for the type sizes raised throughout this function.
        # y is still set from the box top (Y + ~0.18) rather than from Y, because at 6.0pt the label
        # heights, not their widths, are what has to clear the boxes' rounded corners.
        ax.text(MID[i], Y + 0.045, lab, ha="center",
                va="bottom", fontsize=9.0, color=BRK if i >= 2 else CH)

    # the confounded path: T -> adversarial influence -> ASR, never touching d_2's statistic
    YC = 0.12
    # NO lemma number here. This PDF is shared by paper/ and workshop_paper/, and
    # `lem:annihilation` is Lemma 2 in the main paper and Lemma 1 in the workshop, so no
    # hardcoded number can be right in both. Each document's caption carries the real \ref;
    # a number baked into a figure is a cross-reference LaTeX cannot check.
    box(0.470, YC, "adversarial influence (attenuation)", CONF, fs=7.0)
    for (x0, y0), (x1, y1), rad in (((XS[0], Y - 0.20), (0.283, YC), -0.28),
                                    ((0.657, YC), (XS[4], Y - 0.20), -0.28)):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color=CONF, lw=1.15, shrinkA=2, shrinkB=2,
                                    connectionstyle=f"arc3,rad={rad}"))
    for ys in ([YC + 0.20, YC + 0.02], [YC + 0.02, YC + 0.20]):   # the cut, struck across that arc
        ax.plot([0.132, 0.186], ys, color=BRK, lw=1.9, zorder=7, solid_capstyle="round")
    # Round 62: was a sentence ("Mode S cuts this path: every adversary pinned at c=1.0, adversarial
    # coefficient share == 0.266667") at 5.4pt. The pinning is what the strike-through already says
    # graphically, so the words go and the two numbers stay -- both still read from `adm` above, and the
    # share still asserted constant across rungs at the top of this function.
    ax.text(0.470, -0.185, f"cut: $c{{=}}1.0$, share $\\equiv {share:.6f}$",
            fontsize=6.6, color=BRK, ha="center", va="center", fontweight="bold", zorder=8)

    # Round 62: title shortened from "two paths from the upstream transform to suppression, and the one
    # Mode S cuts" (76 chars). It is set loc="left" and so does not wrap, but it was the other string
    # setting this panel's horizontal budget, and the second clause is what the struck arc shows.
    ax.set_title(f"{tag}two paths from the upstream transform to suppression",
                 fontsize=8.0, loc="left", pad=2.0)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.30, 1.22)
    ax.axis("off")


def draw_reversal(ax, tag="(c) "):
    """Every scored comparability cell, two designs each: where closing the attenuation channel
    changes the answer. The cell count, the training/out-of-sample split and the number of sign
    reversals are all READ FROM THE ARTIFACT and appear nowhere in this file as literals -- the
    docstring used to say "Six cells ... four TRAINING ... two out-of-sample" and would have gone
    silently stale the moment a seventh cell was scored.

    Read from results/comparability_six_cells.json, which analyze_comparability.py writes after
    asserting that the four published contrasts reproduce bit-identically. Nothing is recomputed here.

    The panel REFUSES to draw unless that artifact still certifies the two things it asserts visually:
    that the published cells reproduce, and that the cell drawn as a sign reversal is the one the
    analyzer classified as one. A chart of twelve numbers is exactly the kind of figure that keeps
    drawing after its premise stops holding, so the premise is checked.

    The TRAINING cells -- the frozen rule was read off them -- are separated from the out-of-sample
    ones by a rule, and each out-of-sample row is annotated "confirms" or "REFUTES" from its own
    prediction. A reader must be able to see which rows could have falsified anything, rather than a
    block of undifferentiated ones. A row whose rule makes no prediction (`predicted` absent) carries
    no verdict annotation and is not counted as confirming anything.

    Round 61, the twenty-first review's clarity item on this figure: a reversal row is drawn with a
    heavy connector in the reversal colour and both of its means printed at the ends of its own arms,
    so the size of the reversal is readable off the panel rather than only off `tab:sixcell`. Every one of
    those numbers is formatted from the artifact, and the row set they are drawn on is the artifact's
    own `sign_reversal_cells`; nothing here is a literal.
    """
    # Same amber as draw_dag's confounded path and the same green as its admission arrow, so a reader
    # who has just read panel (a) meets the same two colours meaning the same two things.
    CONF, INSTR = "#a9780a", "#2f7d3f"
    REV = "#8a2a2a"    # the reversal colour, used for the tag, the values and the heavy connector
    d = json.load(open(SIXCELL))
    a = d["assertions"]
    # `complete` here must mean exactly what it means in analyze_comparability.py, which is BOTH designs
    # present AND every frozen seed landed. Presence alone drew a mid-run cell: while cell 7's controlled
    # kappa=2 rung sat at 4 of 5 seeds this filter admitted it, the analyzer's n_cells excluded it, the
    # two counts disagreed, and the panel refused to draw for a reason that had nothing to do with its
    # premise. Drawing it would have been worse than refusing: a dumbbell whose two ends are at n=5 and
    # n=4 looks identical to one that is not.
    cells = [c for c in d["cells"]
             if c["confounded"] and c["controlled"] and not c.get("missing_frozen_seeds")]
    if not a["published_cells_reproduce"] or len(cells) != a["n_cells"]:
        raise SystemExit("panel (c) refuses to draw: results/comparability_six_cells.json no longer "
                         f"certifies reproducing published cells over {a['n_cells']} complete cells "
                         f"({a}). Re-run experiments/analyze_comparability.py and read its output.")
    # Round 57: compare against the artifact's LIST, not its scalar. The scalar goes None as soon as a
    # second cell reverses, so keying the guard on it made "the reversal replicated" indistinguishable
    # from "the artifact is corrupt" -- a guard that fires hardest on the most interesting outcome.
    rev = sorted(c["label"] for c in cells if c["observed"] == "SIGN REVERSAL")
    want = sorted(a.get("sign_reversal_cells",
                        [a["sign_reversal_cell"]] if a.get("sign_reversal_cell") else []))
    if rev != want:
        raise SystemExit(f"panel (c) refuses to draw: sign-reversal rows {rev} disagree with the "
                         f"artifact's own {want}.")

    # Training cells first, then a rule, then the two that could have falsified the frozen rule. Within
    # each block the artifact's order is kept, which is the order the pre-registration lists them in.
    cells = [c for c in cells if c["training"]] + [c for c in cells if not c["training"]]
    ys = list(range(len(cells) - 1, -1, -1))

    for c, y in zip(cells, ys):
        cf, ct = c["confounded"], c["controlled"]
        is_rev = c["observed"] == "SIGN REVERSAL"
        # The dumbbell connector carries the panel's whole claim: its LENGTH is how much the answer
        # moves when the attenuation channel is closed, on one cell at one seed set. On a reversal row
        # it also CROSSES ZERO, so there it is drawn heavy and in the reversal colour: the one thing a
        # reader skimming this panel should see is a dark bar straddling the zero line.
        ax.plot([cf["mean"], ct["mean"]], [y, y], color=(REV if is_rev else "0.55"),
                lw=(2.1 if is_rev else 1.0), zorder=(3 if is_rev else 2),
                solid_capstyle="butt")
        for r, col, mk in ((cf, CONF, "o"), (ct, INSTR, "D")):
            ax.errorbar(r["mean"], y, xerr=(r["hi"] - r["lo"]) / 2.0, fmt=mk, ms=3.6,
                        color=col, ecolor=col, elinewidth=1.1, capsize=2.2, capthick=0.9,
                        zorder=5, mec=col, mfc=col)
        if is_rev:
            # Both means printed, each just outside its OWN arm's interval and in that arm's colour,
            # so how far the answer moved is readable here and not only in the appendix table. Placed
            # horizontally, never above the marker: the row pitch in the composite figure is ~7.4pt,
            # so a label offset vertically lands on the neighbouring row's error bar. The tag then
            # goes further out on the same side as the leftmost value, because the right margin of
            # this panel is already occupied by the out-of-sample verdict annotations.
            lo_arm, hi_arm = sorted((cf, ct), key=lambda r: r["mean"])
            lo_col = CONF if lo_arm is cf else INSTR
            hi_col = INSTR if lo_arm is cf else CONF
            for arm, col, xend, dx, ha in ((lo_arm, lo_col, lo_arm["lo"], -2.5, "right"),
                                           (hi_arm, hi_col, hi_arm["hi"], 2.5, "left")):
                ax.annotate(f"${arm['mean']:+.3f}$", xy=(xend, y), xytext=(dx, 0),
                            textcoords="offset points", fontsize=5.0, color=col,
                            ha=ha, va="center", zorder=6)
            ax.annotate("sign reversal", xy=(lo_arm["lo"], y), xytext=(-25, 0),
                        textcoords="offset points", fontsize=5.0, color=REV,
                        fontweight="bold", ha="right", va="center", zorder=6)
    ax.axvline(0.0, color="0.25", lw=1.0, zorder=4)

    # The out-of-sample block, separated by a rule so the four training rows cannot be read as evidence.
    n_train = sum(1 for c in cells if c["training"])
    if 0 < n_train < len(cells):
        ax.axhline(len(cells) - n_train - 0.5, color="0.45", lw=0.7, ls=(0, (2.2, 1.8)), zorder=1)

    def is_hit(c):
        return c["observed"] == c["predicted"] or (
            c["predicted"] == "DISAGREE" and c["observed"] == "SIGN REVERSAL")

    ax.set_yticks(ys)
    # Plain monospace text, NOT mathtext. `$\mathtt{...}$` was used here until cell 7 arrived, and inside
    # mathtext a hyphen is the binary minus operator: "CIFAR-100" rendered as CIFAR MINUS 100, in the same
    # glyph and with the same operator spacing as the "-0.75" on the axis below it. Every earlier label
    # was hyphen-free, so the defect could not appear before this cell, and it is invisible to every check
    # this repository runs -- `pdftotext` drops the unmapped minus glyph entirely and extracts
    # "CIFAR 100", so the source, the LaTeX build and a text grep of the rendered PDF all pass. Only
    # pixels show it. Plain text with a monospace family renders the same DejaVu Sans Mono face, needs no
    # `\_` escaping, and prints a real hyphen.
    # The " / " is closed up explicitly: mathtext silently dropped literal spaces, so the labels the
    # published panel shows are "krum/scaling", and plain text would widen every one of them.
    # Round 63: every row label carries its own n, as `n=5` when the two legs agree and `n=5/20` as
    # confounded/controlled when they do not. Until this round the panel rendered NO seed count at all
    # while its rows already spanned n=5, 8 and 20, so a 20-seed interval and a 5-seed interval were
    # drawn as visually identical objects and the two guards above -- which read
    # `published_cells_reproduce` and the reversal-row list -- could not see it, neither of them reading
    # n. `pdftotext` extracts nothing from a figure, so no LaTeX check, no gate and no text grep could
    # either. The tag is read from the artifact, never a literal, so it follows a seed top-up on its own.
    def n_tag(c):
        n0, n2 = c["confounded"].get("n"), c["controlled"].get("n")
        if n0 is None or n2 is None:
            raise SystemExit(f"panel (c) refuses to draw: cell {c['label']!r} carries n="
                             f"{n0}/{n2}, so its row would print a seed count it does not have. "
                             "Re-run experiments/analyze_comparability.py.")
        return f" n={n0}" if n0 == n2 else f" n={n0}/{n2}"

    ax.set_yticklabels([c["label"].replace(" / ", "/") + n_tag(c) for c in cells],
                       fontsize=5.4, family="monospace")

    # The two rows that could have falsified the frozen rule are the only ones carrying a verdict, and
    # one of them REFUTES. That word is the honest headline of this block and is not softened.
    xr = ax.get_xlim() if ax.get_xlim()[1] > ax.get_xlim()[0] + 1e-9 else None
    for c, y in zip(cells, ys):
        if c["training"]:
            continue
        # A cell the withdrawn rule makes NO prediction about (Amendment 4's cell 7) has predicted=None.
        # is_hit() returns False for it, so annotating unconditionally would print "REFUTES" against a
        # rule that never predicted anything here -- a false verdict, and the worst kind, because it
        # reads as evidence. Such a row is drawn with its interval and no verdict.
        if not c.get("predicted"):
            ax.annotate("out of sample: rule makes no prediction",
                        xy=(1.0, y), xycoords=("axes fraction", "data"),
                        xytext=(-2, 0), textcoords="offset points",
                        fontsize=5.0, ha="right", va="center", zorder=7, color="0.35")
            continue
        hit = is_hit(c)
        ax.annotate("out of sample: " + ("confirms" if hit else "REFUTES"),
                    xy=(1.0, y), xycoords=("axes fraction", "data"),
                    xytext=(-2, 0), textcoords="offset points",
                    fontsize=5.0, ha="right", va="center", zorder=7,
                    color=("#2f7d3f" if hit else "#8a2a2a"),
                    fontweight=("normal" if hit else "bold"))
    ax.set_xlabel("$\\Delta$ ASR, $\\kappa{=}0 \\to \\kappa{=}2$, per cell; "
                  "bars are $95\\%$ paired $t$ intervals\n"
                  "circle: outcome-gated ladder (adversary free to attenuate).   "
                  "diamond: Mode S (adversary pinned at $c{=}1$)",
                  fontsize=5.9, labelpad=1.5, linespacing=1.25)
    # Every count in this title comes from the artifact. "on one they have OPPOSITE SIGNS" used to be
    # literal text, and a count baked into a figure PDF passes every LaTeX check -- pdftotext is the
    # only thing that sees it. n_rev is read, and the phrasing agrees with it in number.
    #
    # Round 61: the title names the FINDING and then counts, rather than counting only. The counts are
    # nested and the wording says so: the reversals are a subset of the disagreements (a verdict is one
    # of AGREE / DISAGREE / SIGN REVERSAL and n_disagree sums the last two), so "changes the answer ...
    # and its SIGN on" must not read as two disjoint tallies that a reader would add.
    #
    # It is also kept SHORT, and that is a layout constraint and not taste: the title is the widest
    # thing in this panel, savefig uses bbox_inches="tight", and a longer title silently widens the
    # whole canvas. The first draft of this title ran 114 chars and grew modeS_causal.pdf from 5.88in
    # to 7.64in, which at a fixed \includegraphics width shrinks every label in all three panels.
    n_rev = a.get("n_sign_reversal", 0)
    rev_clause = ("and no sign flips" if n_rev == 0 else
                  "and its SIGN on 1" if n_rev == 1 else
                  f"and its SIGN on {n_rev}")
    ax.set_title(f"{tag}the design choice changes the answer on {a['n_disagree']} of "
                 f"{a['n_cells']} cells, {rev_clause}",
                 fontsize=7.4, loc="left", pad=2.0)
    lo = min(min(c["confounded"]["lo"], c["controlled"]["lo"]) for c in cells)
    hi = max(max(c["confounded"]["hi"], c["controlled"]["hi"]) for c in cells)
    pad = 0.12 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad + 0.10)
    ax.set_ylim(-0.6, len(cells) - 0.4)
    ax.tick_params(axis="x", labelsize=5.6)
    ax.tick_params(axis="y", length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)


# Figures whose workshop copy is PINNED, with the reason, because this generator writes into both
# papers and the two papers no longer agree about how many cells exist. The workshop is a SIX-cell
# document: its Fig. 1 caption reads "Six cells, one contrast each" and "The designs disagree on 4 of 6
# cells", and this round's seventh cell is added to the main paper only. Writing the 7-row panel there
# would put a figure drawing 7 rows under a caption counting 6, inside a document nothing else this
# round touches, and no check in this repository could see it: the caption is prose, the count is
# pixels, and `pdftotext` of the figure reports whatever the figure says without ever reading the
# caption. Whoever revives the workshop regenerates this figure and rewrites that caption together.
WORKSHOP_PINNED = {"modeS_causal.pdf"}


def save(fig, name):
    ws = os.path.join(REPO, "workshop_paper", "figures")
    for d in (HERE, ws):
        if not os.path.isdir(d):
            continue
        if d == ws and name in WORKSHOP_PINNED:
            print(f"PINNED, not written (see WORKSHOP_PINNED): {os.path.join(d, name)}")
            continue
        out = os.path.join(d, name)
        fig.savefig(out, bbox_inches="tight")
        print("Saved:", out)


# --- standalone panels: the Mode-S adjudication is the paper's centerpiece and carries its own float
fS, axS = plt.subplots(figsize=(6.4, 4.3))
draw_modeS(axS, tag="")
fS.tight_layout()
save(fS, "targeted_modeS.pdf")

fA, bxA = plt.subplots(figsize=(6.4, 4.3))
draw_modeA(bxA, tag="")
fA.tight_layout()
save(fA, "targeted_modeA.pdf")

# --- the body float for BOTH papers: the causal structure (a) and the reversal (b). Round 70 dropped
# the Mode-S evidence panel from between them; see the note above fC below for what moved where.
# Emitted under a NEW name so nothing that references targeted_modeS.pdf changes. Sized at the printed
# width (5.5in ~ NeurIPS \linewidth) rather than 6.4in, so labels render at their nominal point size
# instead of being downscaled -- the previous single panel was set at 0.37\linewidth from a 6.4in
# canvas, a 0.32x reduction that left its axis labels near-illegible.
# Until Round 70 this was FIVE rows, two of them empty spacers, because the two gaps needed very
# different sizes and a single
# hspace cannot give them: panel (b) carries a two-line x tick band (rho over kappa) AND an x label
# beneath it, so the (b)->(c) gap has ~11pt more to clear than the (a)->(b) gap does. With one hspace,
# buying enough room below (b) meant paying for the same room below (a) and shrinking every panel to
# fund it; at hspace=0.62 panel (c)'s title printed straight through (b)'s x label. Measured, not
# guessed: at these numbers there are 14.4pt of clear space below (b)'s x label, and panels (a) and (b)
# are each ~0.06in TALLER than in the two-panel version this replaces, for +0.33in of total height.
#
# Round 62 reallocated row 0 from 0.92 to 1.12 to pay for draw_dag's raised type (boxes 6.3 -> 8.2pt;
# at main.tex's 0.80\linewidth the old 4.6pt arrow labels printed at ~3.7pt). The ratios still SUM to
# 3.82, so the figure's own height is untouched, and the 0.20 came from panel (b) (0.84 -> 0.76) and
# from the (b)->(c) spacer (0.84 -> 0.72, spending ~7.6pt of the 14.4pt of clear space measured above
# and leaving ~6.8pt). Because save() uses bbox_inches="tight", the NATIVE size is set by content and
# not by figsize: the check that matters is the native size, since main.tex includes this at a fixed
# width and any aspect change silently moves every page after it. That size has moved twice since this
# note was written (423.4 x 237.6 -> 431.5 x 238.3 in Round 63, -> 431.5 x 175.1 in Round 70); the
# CURRENT value is recorded with the Round 70 note below rather than restated here in three places.
#
# Round 70, the tenth review's item (6): "simplify Figure 1 to just the two designs and the two numbers,
# move the channel dissociations to Figure 2". The middle panel is DROPPED from this composite, so the
# body float is now the vocabulary/level chain (a) and the seven-cell reversal (b). Three consequences,
# all checked rather than assumed:
#  * No evidence leaves the paper. draw_modeS's content is already emitted standalone as
#    targeted_modeS.pdf above, which until this round was referenced by NOTHING in main.tex,
#    supplementary.tex or workshop_paper/main.tex; it now carries its own appendix float, where the
#    Mode-S dose evidence is already discussed at length.
#  * The panel LETTER of the reversal changes (c) -> (b). Every surviving \ref in both documents names
#    panel (a) -- main.tex's Mode-S instrument paragraph, its App. J pointer, its lower-arc paragraph and
#    supplementary.tex -- so no cross-reference breaks; the one sentence that went false is main.tex's
#    "Mode~S itself is Figure~\ref{fig:modeS}", which is repointed in the same round.
#  * The native size MOVES, and that is a page-budget item, not only a figure edit: dropping a row and
#    its spacer takes the ratios from 3.82 to 2.52 units, and figsize's height is cut in the same
#    proportion (3.36 -> 2.22in) so each surviving panel keeps its absolute height rather than being
#    stretched to fill the old canvas. The (a)->(b) gap is raised 0.12 -> 0.30 because it now has to
#    clear draw_dag's "cut: c=1.0, share" annotation above draw_reversal's title; at 0.12 the old panel
#    (b)'s title already printed into that annotation.
fC = plt.figure(figsize=(5.5, 2.22))
_gs = fC.add_gridspec(3, 1, height_ratios=[1.12, 0.30, 1.10], hspace=0.0)
cx, ccx = fC.add_subplot(_gs[0]), fC.add_subplot(_gs[2])
draw_dag(cx)
draw_reversal(ccx, tag="(b) ")
save(fC, "modeS_causal.pdf")

# --- combined two-panel layout, kept so the ICLR paper's existing float need not change
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.2, 4.2))
draw_modeS(ax)
draw_modeA(bx)
fig.tight_layout()
save(fig, "targeted_dose.pdf")

# printed so the caption's numbers come from this run rather than from memory.
# Recomputed here at module scope (the panels own their locals) from the same JSON.
_kr = rungs_of(cells, "S", "krum", KAPPAS)
print("\npanel (a) Mode S krum:")
print("  rho       ", " ".join(f"{r:7.2f}" for r in RHO))
print("  ASR       ", " ".join(f"{r['mean']:7.3f}" for r in _kr))
print("  decision  ", " ".join(f"{summary[f'doseS|krum|{k}|decision']:7.3f}" for k in KAPPAS))
print("  admission ", " ".join(f"{summary[f'doseS|krum|{k}|admission']:7.3f}" for k in KAPPAS))
# Printed because the DAG panel no longer annotates it: the prose that reports "Delta agg." for the
# top rung must still be able to transcribe it from a run rather than from memory.
print("  agg_disp  ", " ".join(f"{summary[f'doseS|krum|{k}|agg_disp']:7.3f}" for k in KAPPAS))
# The DAG panel's admission -> ASR arrow is the ONE annotation drawn from another arm, so its numbers are
# printed in full here rather than only asserted: the caption has to name the arm and the rung, and the
# two differencing conventions have to be visible side by side. The body reports 0.173, the difference of
# the two published rung means; at full precision the same fall is 0.1723, and nothing rests on which.
_ck = rungs_of(cells, "S", "cos_krum", KAPPAS)
print("panel (a) admission -> ASR arrow, cos_krum / pixel (the (P4) does not imply (P5) witness):")
print("  rho       ", " ".join(f"{r:7.2f}" for r in RHO))
print("  ASR       ", " ".join(f"{x['mean']:7.3f}" for x in _ck))
print("  acc       ", " ".join(f"{x['acc']:7.3f}" for x in _ck))
print("  decision  ", " ".join(f"{summary[f'doseS|cos_krum|{k}|decision']:7.3f}" for k in KAPPAS))
print("  admission ", " ".join(f"{summary[f'doseS|cos_krum|{k}|admission']:7.3f}" for k in KAPPAS))
print(f"  first gate-passing rung: rho={RHO[1]:.2f}, acc {_ck[0]['acc']:.3f} -> {_ck[1]['acc']:.3f},"
      f" fall {round(_ck[0]['mean'], 3) - round(_ck[1]['mean'], 3):.3f} as the body differences it"
      f" ({_ck[0]['mean'] - _ck[1]['mean']:.4f} at full precision)")
for d2 in ("reputation", "coord_median"):
    r = rungs_of(cells, "A", d2, NUS)
    print(f"panel (b) Mode A {d2}:")
    print("  r         ", " ".join(f"{g:7.3f}" for g in GAMMA))
    print("  ASR       ", " ".join(f"{x['mean']:7.3f}" for x in r))
    print("  acc       ", " ".join(f"{x['acc']:7.3f}" for x in r))
