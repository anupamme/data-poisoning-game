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
    ADM_FEMNIST, MASS, ROWS, channel_rows)
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

    # pooled over the channel table's own cells: each arm under ITS OWN committed attack
    n_rounds = n_changed = n_present = 0
    for _, arm, attack, srcf in ROWS:
        if srcf == ADM_FEMNIST:
            continue                                    # CIFAR-10 arms only, so one dataset is counted
        field, _ = MASS[arm]
        for rung in (0.0, 0.5, 1.0, 2.0):
            for r in channel_rows(srcf, arm, attack, rung):
                b, q = float(r[f"base_{field}"]), float(r[f"post_{field}"])
                n_rounds += 1
                n_changed += int((b > 0.0) != (q > 0.0))
                n_present += int(b > 0.0)

    CH, BRK, CONF = "#1f5fa6", "#b03a2e", "#a9780a"

    def box(x, y, s, ec, fs=6.3, weight="normal"):
        ax.text(x, y, s, ha="center", va="center", fontsize=fs, color=ec, fontweight=weight,
                bbox=dict(boxstyle="round,pad=0.30", fc="white", ec=ec, lw=0.9), zorder=5)

    XS = [0.055, 0.265, 0.470, 0.680, 0.905]
    HW = [0.036, 0.048, 0.048, 0.050, 0.030]        # half-widths, so arrows stop at the box edge
    Y = 0.62
    for x, s, ec, w in zip(XS,
                           ["upstream\n$T$", "statistic\n$S(d_2)$", "decision\n(P3)",
                            "admission\n(P4)", "ASR\n(P5)"],
                           [CH, CH, CH, BRK, BRK], ["bold", "normal", "normal", "normal", "bold"]):
        box(x, Y, s, ec, weight=w)

    # The reader's map, and the whole point of the panel: the left group is what a preservation check
    # can see, the right group is what security actually depends on. Set inside the existing headroom
    # (the channel annotations below drop to one line), so the tight bbox does not grow.
    for x0, x1, lab, col in ((XS[0] - HW[0], XS[2] + HW[2], "what a preservation check verifies", CH),
                             (XS[3] - HW[3], XS[4] + HW[4], "what security depends on", BRK)):
        # No span rule: the band between the box tops and the title is only a few points tall, and a
        # rule there strikes through the channel annotations. The label takes its group's colour
        # instead, which is the same cue the boxes already carry.
        ax.text((x0 + x1) / 2, Y + 0.435, lab, ha="center", va="bottom", fontsize=5.2, color=col)

    for i in range(4):
        broken = i >= 2
        ax.annotate("", xy=(XS[i + 1] - HW[i + 1], Y), xytext=(XS[i] + HW[i], Y), zorder=3,
                    arrowprops=dict(arrowstyle="-|>", lw=1.15, shrinkA=0, shrinkB=0,
                                    color=BRK if broken else CH,
                                    ls=(0, (2.4, 1.7)) if broken else "-"))
        if broken:
            ax.text((XS[i] + XS[i + 1]) / 2, Y - 0.20, r"$\nRightarrow$", ha="center", va="center",
                    fontsize=10, color=BRK, zorder=6)
            ax.text((XS[i] + XS[i + 1]) / 2, Y + 0.045, "not implied", ha="center", va="bottom",
                    fontsize=4.6, color=BRK, zorder=6)

    # The T -> statistic segment is deliberately left UNANNOTATED. Its number (Delta agg. = 0.892) is
    # the one channel Mode S does not hold fixed, so printing it here invites reading the panel as a
    # claim that displacement is controlled, which is the opposite of what the panel says. It stays
    # disclosed in prose -- the control paragraph of the targeted section names it as "the aggregate
    # the defense emits" -- and in the channel table, so nothing is lost by not repeating it inside
    # the diagram. None is the skip marker; the loop keeps its segment index either way.
    # Each label names ITS OWN scope, because the three are not the same arm and a reader who assumes
    # they are reads the panel as one experiment. Segment 2 is Krum alone (its decision flips); segment 3
    # is POOLED over the four CIFAR-10 arms of the channel table, which is why it must NOT carry an arm
    # name -- n_rounds is summed over ROWS above, so "Krum: 0/240" would be false; segment 4 is cos_krum,
    # the only arm that holds admission and moves suppression. Widest label is 24 chars, one under the
    # 25-char budget the layout note below fixes, so horizontal clearance is no worse than before.
    for i, lab in enumerate([None, f"Krum: flips {dec:.2f}",
                             f"unchanged, {n_changed}/{n_rounds}, 4 arms",
                             f"cos_krum: falls {d_ck:.3f}"]):
        if lab is None:
            continue
        # ONE line each, and that is a layout constraint, not a style choice: two-line annotations reach
        # y=1.02, which leaves the group labels above them no room below the title at ylim=1.20 and they
        # print through it. Kept short enough (<=25 chars at 4.6pt) to clear each other horizontally --
        # the segment midpoints are only ~0.2 apart -- which is what broke the first one-line attempt.
        # y is set from the box top (Y + ~0.18), not from Y: each annotation is WIDER than the gap it
        # is centred in, so it passes over its neighbouring boxes and only vertical clearance keeps it
        # off their rounded corners. 0.265 leaves ~4.5pt above the boxes, ~4.5pt below the group labels
        # and ~4.5pt from label top to the title -- the three gaps this band has to divide.
        ax.text((XS[i] + XS[i + 1]) / 2, Y + 0.265, lab, ha="center",
                va="bottom", fontsize=4.6, color=BRK if i >= 2 else CH)

    # the confounded path: T -> adversarial influence -> ASR, never touching d_2's statistic
    YC = 0.13
    # NO lemma number here. This PDF is shared by paper/ and workshop_paper/, and
    # `lem:annihilation` is Lemma 2 in the main paper and Lemma 1 in the workshop, so no
    # hardcoded number can be right in both. Each document's caption carries the real \ref;
    # a number baked into a figure is a cross-reference LaTeX cannot check.
    box(0.470, YC, "adversarial influence (attenuation)", CONF, fs=5.6)
    for (x0, y0), (x1, y1), rad in (((XS[0], Y - 0.20), (0.283, YC), -0.28),
                                    ((0.657, YC), (XS[4], Y - 0.20), -0.28)):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color=CONF, lw=1.15, shrinkA=2, shrinkB=2,
                                    connectionstyle=f"arc3,rad={rad}"))
    for ys in ([YC + 0.20, YC + 0.02], [YC + 0.02, YC + 0.20]):   # the cut, struck across that arc
        ax.plot([0.132, 0.186], ys, color=BRK, lw=1.9, zorder=7, solid_capstyle="round")
    ax.text(0.470, -0.11, f"Mode S cuts this path: every adversary pinned at $c{{=}}1.0$, "
                          f"adversarial coefficient share $\\equiv {share:.6f}$",
            fontsize=5.4, color=BRK, ha="center", va="center", fontweight="bold", zorder=8)

    ax.set_title(f"{tag}two paths from the upstream transform to suppression, and the one Mode S cuts",
                 fontsize=7.4, loc="left", pad=2.0)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.22, 1.20)
    ax.axis("off")


def draw_reversal(ax, tag="(c) "):
    """Six cells, two designs each: where closing the attenuation channel changes the answer.

    Read from results/comparability_six_cells.json, which analyze_comparability.py writes after
    asserting that the four published contrasts reproduce bit-identically. Nothing is recomputed here.

    The panel REFUSES to draw unless that artifact still certifies the two things it asserts visually:
    that the published cells reproduce, and that the cell drawn as a sign reversal is the one the
    analyzer classified as one. A chart of twelve numbers is exactly the kind of figure that keeps
    drawing after its premise stops holding, so the premise is checked.

    Four cells are TRAINING -- the frozen rule was read off them -- and two are out-of-sample, marked
    with a rule. Of the two, one confirms and one REFUTES, and the panel says so rather than showing
    six undifferentiated rows: a reader must be able to see which rows could have falsified anything.
    """
    # Same amber as draw_dag's confounded path and the same green as its admission arrow, so a reader
    # who has just read panel (a) meets the same two colours meaning the same two things.
    CONF, INSTR = "#a9780a", "#2f7d3f"
    d = json.load(open(SIXCELL))
    a = d["assertions"]
    cells = [c for c in d["cells"] if c["confounded"] and c["controlled"]]
    if not a["published_cells_reproduce"] or len(cells) != a["n_cells"]:
        raise SystemExit("panel (c) refuses to draw: results/comparability_six_cells.json no longer "
                         f"certifies reproducing published cells over {a['n_cells']} complete cells "
                         f"({a}). Re-run experiments/analyze_comparability.py and read its output.")
    rev = [c["label"] for c in cells if c["observed"] == "SIGN REVERSAL"]
    if rev != ([a["sign_reversal_cell"]] if a["sign_reversal_cell"] else []):
        raise SystemExit(f"panel (c) refuses to draw: sign-reversal rows {rev} disagree with the "
                         f"artifact's own {a['sign_reversal_cell']!r}.")

    # Training cells first, then a rule, then the two that could have falsified the frozen rule. Within
    # each block the artifact's order is kept, which is the order the pre-registration lists them in.
    cells = [c for c in cells if c["training"]] + [c for c in cells if not c["training"]]
    ys = list(range(len(cells) - 1, -1, -1))

    for c, y in zip(cells, ys):
        cf, ct = c["confounded"], c["controlled"]
        # The dumbbell connector carries the panel's whole claim: its LENGTH is how much the answer
        # moves when the attenuation channel is closed, on one cell at one seed set.
        ax.plot([cf["mean"], ct["mean"]], [y, y], color="0.55", lw=1.0, zorder=2)
        for r, col, mk in ((cf, CONF, "o"), (ct, INSTR, "D")):
            ax.errorbar(r["mean"], y, xerr=(r["hi"] - r["lo"]) / 2.0, fmt=mk, ms=3.6,
                        color=col, ecolor=col, elinewidth=1.1, capsize=2.2, capthick=0.9,
                        zorder=5, mec=col, mfc=col)
        if c["observed"] == "SIGN REVERSAL":
            ax.text(max(cf["mean"], ct["mean"]) + 0.045, y, "sign\nreversal", fontsize=5.0,
                    color="#8a2a2a", fontweight="bold", ha="left", va="center", zorder=6,
                    linespacing=1.0)
    ax.axvline(0.0, color="0.25", lw=1.0, zorder=4)

    # The out-of-sample block, separated by a rule so the four training rows cannot be read as evidence.
    n_train = sum(1 for c in cells if c["training"])
    if 0 < n_train < len(cells):
        ax.axhline(len(cells) - n_train - 0.5, color="0.45", lw=0.7, ls=(0, (2.2, 1.8)), zorder=1)

    def is_hit(c):
        return c["observed"] == c["predicted"] or (
            c["predicted"] == "DISAGREE" and c["observed"] == "SIGN REVERSAL")

    ax.set_yticks(ys)
    ax.set_yticklabels([f"$\\mathtt{{{c['label'].replace('_', chr(92) + '_')}}}$" for c in cells],
                       fontsize=5.4)

    # The two rows that could have falsified the frozen rule are the only ones carrying a verdict, and
    # one of them REFUTES. That word is the honest headline of this block and is not softened.
    xr = ax.get_xlim() if ax.get_xlim()[1] > ax.get_xlim()[0] + 1e-9 else None
    for c, y in zip(cells, ys):
        if c["training"]:
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
    ax.set_title(f"{tag}the two designs disagree on {a['n_disagree']} of {a['n_cells']} cells, "
                 f"and on one they have OPPOSITE SIGNS",
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


def save(fig, name):
    for d in (HERE, os.path.join(REPO, "workshop_paper", "figures")):
        if os.path.isdir(d):
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

# --- the body float for BOTH papers: the causal structure (a), the Mode-S evidence (b), the reversal (c).
# Emitted under a NEW name so nothing that references targeted_modeS.pdf changes. Sized at the printed
# width (5.5in ~ NeurIPS \linewidth) rather than 6.4in, so labels render at their nominal point size
# instead of being downscaled -- the previous single panel was set at 0.37\linewidth from a 6.4in
# canvas, a 0.32x reduction that left its axis labels near-illegible.
# FIVE rows, two of them empty spacers, because the two gaps need very different sizes and a single
# hspace cannot give them: panel (b) carries a two-line x tick band (rho over kappa) AND an x label
# beneath it, so the (b)->(c) gap has ~11pt more to clear than the (a)->(b) gap does. With one hspace,
# buying enough room below (b) meant paying for the same room below (a) and shrinking every panel to
# fund it; at hspace=0.62 panel (c)'s title printed straight through (b)'s x label. Measured, not
# guessed: at these numbers there are 14.4pt of clear space below (b)'s x label, and panels (a) and (b)
# are each ~0.06in TALLER than in the two-panel version this replaces, for +0.33in of total height.
fC = plt.figure(figsize=(5.5, 3.36))
_gs = fC.add_gridspec(5, 1, height_ratios=[0.92, 0.12, 0.84, 0.84, 1.10], hspace=0.0)
cx, cbx, ccx = fC.add_subplot(_gs[0]), fC.add_subplot(_gs[2]), fC.add_subplot(_gs[4])
draw_dag(cx)
draw_modeS(cbx, tag="(b) ", compact=True)
draw_reversal(ccx, tag="(c) ")
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
