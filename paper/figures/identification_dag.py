"""
The identification figure: why the gate is not ordinary collider bias.

The paper's novelty claim over the causal-inference literature is a distinction reviewers keep
reading past when it is written as prose, so it is drawn here instead. Two panels, same graph,
different consequence:

  (a) ordinary collider bias -- conditioning on a common descendant makes P and S co-vary. The
      estimand still HAS support; the association is distorted, and adjustment can recover it in
      principle. This half is textbook and we claim no credit for it.

  (b) this paper -- C0 and C1 together make the gate DETERMINISTIC in S: a certified pair must have
      a downstream defense that suppresses the attack alone (the testability lemma). So the column
      the contrast needs is not merely undersampled, it is EMPTY for every defense menu the screen
      could range over. No adjustment set, no reweighting and no sample size recovers it.

  (c) the design that repairs it -- hold d_2 and a fixed and intervene on the upstream transform.
      The gate then takes one value for every arm rather than selecting them, so it is a constant
      and not a conditioning event, and BOTH rows are occupied by construction: T=id is run as the
      control whatever its outcome. Panel (c) asserts nothing beyond the within-design corollary,
      which is what the caption points at; it is the same 2x2 as (b) with the column that (b) leaves
      empty no longer needed, because nothing varies across it.

NO theorem, proposition, corollary or lemma NUMBER is drawn into this PDF, and none should be added:
the two papers number the same results differently, so a baked number is wrong in one document while
still resolving to a real result in both -- a defect no build warning or `??` scan can see. The
numbers belong in the caption, which reaches them through \\ref. See mechanism_figure.py's note.

This figure draws no measured quantity, so it reads no results file and there is nothing to keep in
sync with the prose. Unlike the other generators it writes to paper/figures/ ONLY: it is an
ICLR-only float and the workshop body has no room for it.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))

INK = "0.20"
GREY = "0.45"
BLUE_F, BLUE_E = "#eaf1f8", "#5588bb"
RED_F, RED_E = "#fbe9e7", "#b03a2e"
GREEN_F, GREEN_E = "#e3f2e6", "#2f7d3f"

BOX = dict(boxstyle="round,pad=0.09", linewidth=0.7)

# Sized to the ICLR text block (\linewidth = 397.5pt = 5.51in) so the PDF is included at 1:1 and
# the point sizes below are the point sizes on the page. A wider figure would be downscaled by
# \includegraphics and the labels would shrink with it.
fig = plt.figure(figsize=(5.51, 1.66))
# Panel (b) carries a 2x2 grid beside its graph, so it stays the widest; (c) needs only a
# single column and is the narrowest. Height is UNCHANGED at 1.66in: the strip must not grow
# vertically, because it now lands in the body rather than the appendix.
gs = fig.add_gridspec(1, 3, width_ratios=[0.92, 1.46, 0.92], wspace=0.06)


def node(ax, x, y, text, fc, ec, w, h, fs):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                facecolor=fc, edgecolor=ec, **BOX))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, color=GREY, lw=0.8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6,
                                 linewidth=lw, color=color, shrinkA=1, shrinkB=1))


# ----------------------------------------------------------------- panel (a)
ax = fig.add_subplot(gs[0, 0])
ax.set_xlim(0, 10.0)
ax.set_ylim(0, 6.4)
ax.axis("off")
ax.set_title("(a) ordinary collider bias", fontsize=7.0, loc="left", color=INK, pad=3)

node(ax, 2.55, 5.05, "$P$\nmechanism\npreserved", BLUE_F, BLUE_E, 4.30, 1.62, 5.5)
node(ax, 7.45, 5.05, "$S$\n$d_2$ strong\nalone", BLUE_F, BLUE_E, 4.30, 1.62, 5.5)
node(ax, 5.00, 2.72, "$G$ gate", RED_F, RED_E, 3.30, 0.98, 5.8)

arrow(ax, 3.20, 4.15, 4.30, 3.30)
arrow(ax, 6.80, 4.15, 5.70, 3.30)
ax.text(5.00, 1.82, "condition on $G$", ha="center", va="center",
        fontsize=5.4, style="italic", color=RED_E)
ax.text(5.00, 0.74, "every cell still populated: the estimand\n"
                    "has support, adjustment recovers it",
        ha="center", va="center", fontsize=5.3, color=INK, linespacing=1.35)

# ----------------------------------------------------------------- panel (b)
bx = fig.add_subplot(gs[0, 1])
bx.set_xlim(0, 15.2)
bx.set_ylim(0, 6.4)
bx.axis("off")
bx.set_title("(b) this paper", fontsize=7.0, loc="left", color=INK, pad=3)

node(bx, 2.45, 5.05, "$P$\nmechanism\npreserved", BLUE_F, BLUE_E, 4.30, 1.62, 5.5)
node(bx, 7.10, 5.05, "$S$\n$d_2$ strong\nalone", BLUE_F, BLUE_E, 4.30, 1.62, 5.5)
node(bx, 4.78, 2.72, "$G$ gate", RED_F, RED_E, 3.30, 0.98, 5.8)

arrow(bx, 3.10, 4.15, 4.10, 3.30)
arrow(bx, 6.45, 4.15, 5.45, 3.30, color=RED_E, lw=1.1)
bx.text(6.72, 3.62, "forced by\nC0$\\wedge$C1", fontsize=5.0, style="italic",
        color=RED_E, ha="left", va="center", linespacing=1.25)
bx.text(4.78, 1.82, "condition on $G$", ha="center", va="center",
        fontsize=5.4, style="italic", color=RED_E)
bx.text(4.78, 0.74, "the gate is deterministic in $S$, so one\n"
                    "column is empty for every defense menu",
        ha="center", va="center", fontsize=5.3, color=INK, linespacing=1.35)

# the 2x2, with the column the identifying contrast needs struck out
X0, Y0, CW, RH = 10.55, 2.66, 2.10, 1.06
bx.text(X0 + CW, Y0 + 2 * RH + 0.86, "$d_2$ suppresses $a$ alone",
        ha="center", va="center", fontsize=5.2, color=INK)
for j, head_ in enumerate(["yes", "no"]):
    bx.text(X0 + j * CW + CW / 2, Y0 + 2 * RH + 0.28, head_,
            ha="center", va="center", fontsize=5.2, color=INK)
for i, lab in enumerate(["$P$", "$\\neg P$"]):
    bx.text(X0 - 0.22, Y0 + (1 - i) * RH + RH / 2, lab,
            ha="right", va="center", fontsize=5.6, color=INK)
    for j in range(2):
        empty = (j == 1)
        bx.add_patch(FancyBboxPatch((X0 + j * CW, Y0 + (1 - i) * RH), CW, RH,
                                    boxstyle="square,pad=0",
                                    facecolor=RED_F if empty else GREEN_F,
                                    edgecolor=RED_E if empty else GREEN_E, linewidth=0.7))
        bx.text(X0 + j * CW + CW / 2, Y0 + (1 - i) * RH + RH / 2,
                "empty" if empty else "observed",
                ha="center", va="center", fontsize=5.2,
                color=RED_E if empty else GREEN_E,
                fontweight="bold" if empty else "normal")
bx.annotate("", xy=(X0 + CW + 0.14, Y0 + 0.08), xytext=(X0 + 2 * CW - 0.14, Y0 + 2 * RH - 0.08),
            arrowprops=dict(arrowstyle="-", color=RED_E, linewidth=1.2, alpha=0.9))
bx.text(X0 + CW, Y0 - 0.62, "the contrast that would identify $P$",
        ha="center", va="center", fontsize=5.0, style="italic", color=RED_E)

# ----------------------------------------------------------------- panel (c)
cx = fig.add_subplot(gs[0, 2])
cx.set_xlim(0, 10.0)
cx.set_ylim(0, 6.4)
cx.axis("off")
cx.set_title("(c) the design that repairs it", fontsize=7.0, loc="left", color=INK, pad=3)

node(cx, 2.60, 5.05, "$\\mathrm{do}(T)$\nupstream\ntransform", GREEN_F, GREEN_E, 4.30, 1.62, 5.5)
node(cx, 7.40, 5.05, "$d_2$, $a$\nheld fixed", BLUE_F, BLUE_E, 4.30, 1.62, 5.5)
arrow(cx, 2.60, 4.15, 4.20, 3.70, color=GREEN_E)
arrow(cx, 7.40, 4.15, 5.80, 3.70)
cx.text(5.00, 3.18, "$G$ is a constant,\nnot a conditioning event", ha="center", va="center",
        fontsize=5.2, style="italic", color=GREEN_E, linespacing=1.25)

# The same 2x2 as panel (b), except that with $d_2$ fixed the column is a constant, so only
# the rows vary and both are occupied.
CX0, CY0, CCW, CRH = 3.05, 0.92, 3.90, 0.78
for i, lab in enumerate(["$P$", "$\\neg P$"]):
    cx.text(CX0 - 0.22, CY0 + (1 - i) * CRH + CRH / 2, lab,
            ha="right", va="center", fontsize=5.6, color=INK)
    cx.add_patch(FancyBboxPatch((CX0, CY0 + (1 - i) * CRH), CCW, CRH,
                                boxstyle="square,pad=0", facecolor=GREEN_F,
                                edgecolor=GREEN_E, linewidth=0.7))
    cx.text(CX0 + CCW / 2, CY0 + (1 - i) * CRH + CRH / 2, "observed",
            ha="center", va="center", fontsize=5.2, color=GREEN_E)
cx.text(5.00, 0.30, "$T{=}\\mathrm{id}$ is the control, run\nwhatever the outcome",
        ha="center", va="center", fontsize=5.0, style="italic", color=INK, linespacing=1.3)

fig.subplots_adjust(left=0.008, right=0.995, top=0.90, bottom=0.03)
out = os.path.join(HERE, "identification_dag.pdf")
plt.savefig(out, bbox_inches="tight")
print("Saved:", out)
