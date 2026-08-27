"""
Scatter of the settings we evaluated, in the two coordinates that organize them.

X-axis: persistence retention gamma (0 = stateless, ~0.66 = persistent FL, measured)
Y-axis: effective admission probability p_eff (ASR of best single constituent defense)

NO REGIONS, NO BOUNDARIES. An earlier version shaded four regime bands and drew three
threshold guides at gamma=0.15, p_eff=0.3 and p_eff=0.6. None of those three numbers was
measured -- they were drawn to separate the points we happened to have -- so a reader could
read a phase boundary out of a picture that contained no evidence for one. Only the plotted
points are data, plus the measured gamma marker. Every point is a setting we ran.

Data points:
  Spam game (stateless, gamma=0): VoPD=0.200 -> randomization helps
  NC/fedavg temporal mix: gamma=0.66, p_eff~1.0 -> mixing collapses (ASR 0.956)
  Rep+TM C1-FAIL zone: gamma=0.66, p_eff~0.68 -> oracle 0.316 (composition wins but limited)
  FG->CM (PASS): gamma=0.66, p_eff~0.0 -> ASR 0.131
  FG->RFA (PASS): gamma=0.66, p_eff~0.0 -> ASR 0.045
  Stateless gamma=0 with complementarity: randomization helps
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase_boundary.pdf")

fig, ax = plt.subplots(figsize=(6, 4.5))

# No region shading and no region labels: see the module docstring. Colour and marker still
# carry the qualitative outcome per point, which is measured; the plane is left blank.

# --- Data points ---
# Spam / stateless (gamma=0, complementary vulnerabilities)
ax.scatter(0.0, 0.5, s=80, color="steelblue", zorder=5, marker="^")
ax.annotate("Spam game\n(VoPD=0.20)", xy=(0.0, 0.5), xytext=(0.10, 0.38),
            fontsize=7, arrowprops=dict(arrowstyle="-", color="steelblue", lw=0.8), color="steelblue")

# NC/fedavg temporal mix (gamma~0.66, p_eff~0.96)
ax.scatter(0.66, 0.956, s=80, color="firebrick", zorder=5, marker="s")
ax.annotate("NC/fedavg mix\n(ASR 0.956)", xy=(0.66, 0.956), xytext=(0.74, 0.87),
            fontsize=7, arrowprops=dict(arrowstyle="-", color="firebrick", lw=0.8), color="firebrick")

# Rep+TM C1-FAIL (gamma~0.66, p_eff~0.68 = pixel ASR with rep alone)
ax.scatter(0.66, 0.68, s=80, color="goldenrod", zorder=5, marker="D")
ax.annotate("Rep+TM (C1-FAIL)\noracle ASR 0.316", xy=(0.66, 0.68), xytext=(0.74, 0.55),
            fontsize=7, arrowprops=dict(arrowstyle="-", color="goldenrod", lw=0.8), color="goldenrod")

# FG->CM PASS (gamma~0.66, p_eff~0.13 = best single defense CM ASR ~0.5 but effectively 0 after FG)
ax.scatter(0.66, 0.131, s=80, color="seagreen", zorder=5, marker="o")
ax.annotate("FG$\\to$CM (PASS)\nASR 0.131", xy=(0.66, 0.131), xytext=(0.50, 0.20),
            fontsize=7, arrowprops=dict(arrowstyle="-", color="seagreen", lw=0.8), color="seagreen")

# FG->RFA PASS (gamma~0.66, p_eff~0.06)
ax.scatter(0.66, 0.045, s=80, color="darkgreen", zorder=5, marker="o")
ax.annotate("FG$\\to$RFA (PASS)\nASR 0.045", xy=(0.66, 0.045), xytext=(0.40, 0.13),
            fontsize=7, arrowprops=dict(arrowstyle="-", color="darkgreen", lw=0.8), color="darkgreen")

# --- Measured gamma marker: the one vertical line, and it is a measurement ---
ax.axvline(x=0.66, color="gray", lw=1.0, ls=":", alpha=0.7)
ax.text(0.635, 1.03, "$\\hat{\\gamma}\\approx0.66$ (measured)", ha="right", va="top",
        fontsize=6.5, color="gray")

# --- Axes ---
ax.set_xlabel("Persistence retention $\\gamma$", fontsize=10)
ax.set_ylabel("Best-single-defense ASR ($p_{\\mathrm{eff}}$)", fontsize=10)
ax.set_xlim(-0.05, 1.02)
ax.set_ylim(-0.08, 1.05)
ax.set_title("Evaluated settings (no regimes drawn)", fontsize=10)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(out_path, bbox_inches="tight")
print(f"Saved: {out_path}")
