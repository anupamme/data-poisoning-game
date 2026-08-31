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
    ADM, KAPPAS, NUS, TARGETED, rungs_of)
# Imported, not re-implemented: the DAG panel's admission count must be the SAME per-round indicator
# and the SAME arm/attack pairing the published channel table uses, or the figure and the table can
# disagree about what "the support of the adversarial mass changed" means.
from experiments.build_channel_table import (  # noqa: E402
    ADM_FEMNIST, MASS, ROWS, channel_rows)

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
            ax.annotate(txt, xy=(x, y), xytext=(RHO[-1] * 1.9, ytx), fontsize=FS, color=col,
                        va="center", ha="left", linespacing=1.2,
                        arrowprops=dict(arrowstyle="-", color=col, lw=0.7, shrinkA=2, shrinkB=2))
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
    same three numbers per arm).
    """
    TOP = KAPPAS[-1]
    dec = max(summary[f"doseS|krum|{k}|decision"] for k in KAPPAS)   # "up to", as the body says
    share = adm["adv_coeff_share"][f"doseS|{TOP}"]
    shares = [adm["adv_coeff_share"][f"doseS|{k}"] for k in KAPPAS]
    assert max(shares) - min(shares) < 1e-5, shares      # the pinning this panel claims
    kr = rungs_of(cells, "S", "krum", KAPPAS)
    d_asr = kr[-1]["mean"] - kr[0]["mean"]

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
    for i, lab in enumerate([None, f"flips {dec:.2f} of rounds",
                             f"support unchanged ({n_changed}/{n_rounds})",
                             f"$|\\Delta|{{=}}{abs(d_asr):.3f}$, in margin"]):
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
    box(0.470, YC, "adversarial influence (attenuation, Lem. 1)", CONF, fs=5.6)
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

# --- the body float for BOTH papers: the causal structure (a) above the Mode-S evidence (b).
# Emitted under a NEW name so nothing that references targeted_modeS.pdf changes. Sized at the printed
# width (5.5in ~ NeurIPS \linewidth) rather than 6.4in, so labels render at their nominal point size
# instead of being downscaled -- the previous single panel was set at 0.37\linewidth from a 6.4in
# canvas, a 0.32x reduction that left its axis labels near-illegible.
fC, (cx, cbx) = plt.subplots(2, 1, figsize=(5.5, 2.14),
                             gridspec_kw=dict(height_ratios=[1.10, 1.05], hspace=0.46))
draw_dag(cx)
draw_modeS(cbx, tag="(b) ", compact=True)
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
for d2 in ("reputation", "coord_median"):
    r = rungs_of(cells, "A", d2, NUS)
    print(f"panel (b) Mode A {d2}:")
    print("  r         ", " ".join(f"{g:7.3f}" for g in GAMMA))
    print("  ASR       ", " ".join(f"{x['mean']:7.3f}" for x in r))
    print("  acc       ", " ".join(f"{x['acc']:7.3f}" for x in r))
