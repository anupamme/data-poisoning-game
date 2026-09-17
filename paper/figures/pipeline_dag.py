"""
The composed FL pipeline as a DAG, which is NOT the same graph as identification_dag.py.

Two reviews have now asked for "a formal DAG of the composed pipeline" and pointed at
identification_dag.pdf as though it were one. It is not: that figure is a DAG of the EVALUATION
DESIGN (P, S, the gate G, and the do(T) repair), and it says nothing about what a client update
passes through. This figure draws the pipeline itself, and the two must not be conflated:

  - the STATISTIC PATH (top): the round's updates are transformed per client by d_1, the downstream
    defense computes its statistic on the TRANSFORMED points, that statistic drives d_2's decision,
    and the decision determines whether adversarial input is admitted. The paper's preservation
    levels sit on this path: P1-P3 on the statistic and the decision, P4 on admission.

  - the BYPASS ARC (bottom, drawn red): d_1's own weights reach aggregate adversarial influence
    WITHOUT passing through sigma_2 at all. When d_1 sets an adversary's weight to zero the
    adversary is excluded whatever d_2's statistic says, so C2 is not the operative channel and the
    composition's outcome is d_1's alone. This is the arc the annihilation lemma is about and the
    arc Mode S cuts by pinning every adversary's coefficient at c = 1, which is the whole reason
    Mode S exists and is also the reason it is an instrument rather than a defense.

Reading the two together is the point: an argument that only preserves the statistic constrains the
top path and leaves the bottom one free, which is why statistic preservation does not identify
suppression preservation.

NO theorem, proposition, corollary or lemma NUMBER is drawn into this PDF, and none should be added:
the two papers number the same results differently, so a baked number is wrong in one document while
still resolving to a real result in both -- a defect no build warning or `??` scan can see. The
numbers belong in the caption, which reaches them through \\ref. Same rule as identification_dag.py.

This figure draws no measured quantity, so it reads no results file and there is nothing to keep in
sync with the prose. It writes to paper/figures/ ONLY: it is an ICLR-only appendix float and the
workshop body has no room for it.
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
# the point sizes below are the point sizes on the page, exactly as identification_dag.py.
fig = plt.figure(figsize=(5.51, 1.72))
ax = fig.add_subplot(111)
ax.set_xlim(0, 17.4)
ax.set_ylim(0, 6.9)
ax.axis("off")


def node(x, y, text, fc, ec, w, h, fs=5.4):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                facecolor=fc, edgecolor=ec, **BOX))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.25)


def arrow(x0, y0, x1, y1, color=GREY, lw=0.8, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6,
                                 linewidth=lw, color=color, shrinkA=1, shrinkB=1,
                                 connectionstyle="arc3,rad=%.3f" % rad))


# ------------------------------------------------------------------ the two upstream nodes
node(1.62, 3.45, "$U=\\{u_i\\}$\nround's updates,\nbenign and\nadversarial", BLUE_F, BLUE_E, 3.05, 2.24, 5.2)
node(5.05, 3.45, "$T_1$\n$d_1$'s per-client\ntransform,\n$u_i \\mapsto w_i u_i$", BLUE_F, BLUE_E, 3.15, 2.24, 5.2)
arrow(3.16, 3.45, 3.46, 3.45)

# ------------------------------------------------------------------ the statistic path (top)
node(8.60, 5.42, "$\\sigma_2(T_1(U))$\n$d_2$'s statistic on the\ntransformed points\n(P1), (P2)", GREEN_F, GREEN_E, 3.30, 2.10, 5.2)
node(12.05, 5.42, "$d_2$'s decision\nrank, residual\nor selection\n(P3)", GREEN_F, GREEN_E, 3.05, 2.10, 5.2)
node(15.45, 5.42, "admission of\nadversarial input\n(P4)", GREEN_F, GREEN_E, 3.05, 1.62, 5.2)
arrow(6.64, 4.10, 7.10, 4.80, color=GREEN_E)
arrow(10.26, 5.42, 10.51, 5.42, color=GREEN_E)
arrow(13.59, 5.42, 13.91, 5.42, color=GREEN_E)
ax.text(8.60, 6.72, "the statistic path: what a composition-invariance argument constrains",
        ha="center", va="center", fontsize=5.1, style="italic", color=GREEN_E)

# ------------------------------------------------------------------ influence and outcome
node(15.45, 3.02, "$\\Lambda_a$\naggregate adversarial\ninfluence\n(P4), quantitative", RED_F, RED_E, 3.20, 2.02, 5.2)
node(15.45, 0.66, "emitted aggregate,\nthen ASR (P5)", BLUE_F, BLUE_E, 3.35, 1.02, 5.2)
arrow(15.45, 4.61, 15.45, 4.08, color=GREY)
arrow(15.45, 1.98, 15.45, 1.20, color=GREY)

# ------------------------------------------------------------------ the bypass arc (bottom)
ax.add_patch(FancyArrowPatch((5.05, 2.30), (13.90, 2.72), arrowstyle="-|>", mutation_scale=7,
                             linewidth=1.15, color=RED_E, shrinkA=2, shrinkB=2,
                             connectionstyle="arc3,rad=0.16"))
ax.text(8.05, 1.02, "the bypass arc: $w_a{=}0$ excludes the adversary whatever $\\sigma_2$ says,\n"
                    "so C2 is not the operative channel and $d_1$ alone sets the outcome",
        ha="center", va="center", fontsize=5.1, color=RED_E, linespacing=1.35)

# Mode S cuts the bypass arc: a tick across it, not a node, because pinning c=1 removes the arc
# rather than adding a stage to the pipeline.
ax.plot([11.28, 11.86], [1.62, 2.42], color=GREEN_E, linewidth=1.1, solid_capstyle="round")
ax.text(11.62, 2.86, "Mode S pins\n$c{=}1$ here", ha="center", va="center",
        fontsize=5.0, style="italic", color=GREEN_E, linespacing=1.25)

fig.subplots_adjust(left=0.006, right=0.996, top=0.985, bottom=0.01)
out = os.path.join(HERE, "pipeline_dag.pdf")
plt.savefig(out, bbox_inches="tight")
print("Saved:", out)
