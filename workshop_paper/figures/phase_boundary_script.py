"""
Phase-boundary figure: conditions under which randomization vs. composition dominates.

X-axis: persistence retention gamma (0 = stateless, ~0.66 = persistent FL)
Y-axis: effective admission probability p_eff (ASR of best single constituent defense)
Regions derived from experimental data in the paper.

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

# --- Region shading ---
# Region 1: Low gamma -> randomization can help (left strip)
ax.axvspan(0, 0.15, alpha=0.12, color="steelblue", label=None)

# Region 2: High gamma + high p_eff -> mixing collapses (top-right)
gamma_hi = np.linspace(0.15, 1.0, 100)
ax.fill_between(gamma_hi, 0.6, 1.0, alpha=0.12, color="firebrick", label=None)

# Region 3: High gamma + low p_eff -> composition succeeds (bottom-right)
ax.fill_between(gamma_hi, 0.0, 0.3, alpha=0.12, color="seagreen", label=None)

# Region 4: High gamma + medium p_eff -> composition wins but limited (middle-right)
ax.fill_between(gamma_hi, 0.3, 0.6, alpha=0.08, color="goldenrod", label=None)

# --- Region labels ---
ax.text(0.04, 0.50, "Randomization\ncan help\n(low persistence)", ha="center", va="center",
        fontsize=7.5, color="steelblue", style="italic")
ax.text(0.60, 0.82, "Mixing\ncollapses\n(max-like)", ha="center", va="center",
        fontsize=7.5, color="firebrick", style="italic")
ax.text(0.60, 0.15, "Criterion-PASS\nregion", ha="center", va="center",
        fontsize=7.5, color="seagreen", style="italic")
ax.text(0.60, 0.44, "Composition\nwins, limited\n(C1-FAIL zone)", ha="center", va="center",
        fontsize=7.5, color="goldenrod", style="italic")

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

# --- Boundary lines ---
ax.axvline(x=0.15, color="gray", lw=0.8, ls="--", alpha=0.5)
ax.axhline(y=0.3, xmin=0.15, color="gray", lw=0.8, ls="--", alpha=0.5)
ax.axhline(y=0.6, xmin=0.15, color="gray", lw=0.8, ls="--", alpha=0.5)

# --- Measured gamma marker ---
ax.axvline(x=0.66, color="gray", lw=1.0, ls=":", alpha=0.7)
ax.text(0.66, -0.07, "$\\hat{\\gamma}\\approx0.66$\n(measured)", ha="center", va="top",
        fontsize=6.5, color="gray")

# --- Axes ---
ax.set_xlabel("Persistence retention $\\gamma$", fontsize=10)
ax.set_ylabel("Best-single-defense ASR ($p_{\\mathrm{eff}}$)", fontsize=10)
ax.set_xlim(-0.05, 1.02)
ax.set_ylim(-0.08, 1.05)
ax.set_title("Illustrative regime diagram (evaluated settings)", fontsize=10)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(out_path, bbox_inches="tight")
print(f"Saved: {out_path}")
