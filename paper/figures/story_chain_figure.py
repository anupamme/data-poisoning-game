"""
Figure 1 -- the preservation chain, (P1) to (P5), and where it breaks.

The paper's conceptual object is a ladder of preservation levels between an upstream transform and
the downstream defense's security behaviour:

  (P1) value       the statistic S that d2 reads is unchanged
  (P2) ordering    the ranking S induces is unchanged
  (P3) decision    d2's decision is unchanged
  (P4) admission   the adversarial mass d2 admits into its aggregate is unchanged
  (P5) suppression ASR stays low

(P1)=>(P2)=>(P3) hold by definition. The (P3)-(P4) link is where the chain breaks, and it breaks in
BOTH directions -- which is the paper's central negative result. This figure draws that, with the two
measured witnesses beside the break; the inferential statistics live in the caption, not here.

Every number drawn or printed is recomputed from the same JSON the scoring scripts read, using the
loaders imported from those scripts, so neither the figure nor the caption can drift from the prose:

  pooled Spearman (Round 11 ladder, 16 cells)   analyze_dose_response.load_arms
  Mode S Krum / cos_krum decision, admission, ASR   analyze_targeted_dose.rungs_of + admission JSON
  pooled two-channel regression R^2              analyze_dose_replication's refit (or same lstsq)

Writes story_chain.pdf next to this script and to the sibling paper's figures/ dir.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import scipy.stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from experiments.analyze_dose_response import (  # noqa: E402
    DIST, LADDER, load_arms)
from experiments.analyze_targeted_dose import (  # noqa: E402
    ADM, KAPPAS, NUS, TARGETED, rungs_of)

# ----------------------------------------------------------------- recomputed evidence
# (1) Round-11 pooled Spearman: ASR rise vs measured disturbance, all 16 cells, one-sided greater.
arms = load_arms(json.load(open(LADDER)), json.load(open(DIST)))
sp = sps.spearmanr([r["disturb"] for a in arms for r in a["rungs"]],
                   [r["rise"] for a in arms for r in a["rungs"]], alternative="greater")

# (2) Mode S, Krum -- the adjudicating arm. Decision moves, admission does not, ASR does not.
cells = json.load(open(TARGETED))["cells"]
adm = json.load(open(ADM))
kr = rungs_of(cells, "S", "krum", KAPPAS)
dec = [adm["summary"][f"doseS|krum|{k}|decision"] for k in KAPPAS]
admn = [adm["summary"][f"doseS|krum|{k}|admission"] for k in KAPPAS]
asr = [r["mean"] for r in kr]
share = adm["adv_coeff_share"]

# (2b) Mode S, cos_krum -- the opposite-direction witness. Selection is bit-identical at every rung
#      (decision and admission both 0), yet ASR falls: (P3) held exactly and (P5) still moved.
ck = rungs_of(cells, "S", "cos_krum", KAPPAS)
ck_asr = [r["mean"] for r in ck]
ck_dec = [adm["summary"][f"doseS|cos_krum|{k}|decision"] for k in KAPPAS]
ck_adm = [adm["summary"][f"doseS|cos_krum|{k}|admission"] for k in KAPPAS]
ck_fall = ck_asr[0] - min(ck_asr)

# (3) Pooled two-channel model -- the same regression analyze_targeted_dose scores, so the honest
#     "not confirmed" caveat on the figure carries the scorer's own R^2.
pts = []
for mode, d2s, vals, fam in (("S", ("krum", "reputation", "cos_krum"), KAPPAS, "doseS"),
                             ("A", ("reputation", "coord_median"), NUS, "doseA")):
    for d2 in d2s:
        r = rungs_of(cells, mode, d2, vals)
        if r is None:
            continue
        zero = next(x for x in r if x["rung"] == 0.0)
        for x in r:
            pts.append((x["mean"] - zero["mean"],
                        adm["summary"][f"{fam}|{d2}|{x['rung']}|admission"],
                        share[f"{fam}|{x['rung']}"] - share[f"{fam}|0.0"]))
X = np.column_stack([np.ones(len(pts)), [p[1] for p in pts], [p[2] for p in pts]])
y = np.array([p[0] for p in pts])
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
resid = y - X @ beta
R2 = 1 - float(resid @ resid) / max(float(((y - y.mean()) ** 2).sum()), 1e-12)
N_POOLED = len(pts)

# (4) The Round-15 replication arm, read from the scorer's own output rather than recomputed here, so
#     the figure cannot drift from analyze_dose_replication.py. It refutes the cross-arm ORDERING
#     while replicating the negative, and the figure has to say both.
REPL = os.path.join(REPO, "results", "dose_replication", "scored.json")
repl = json.load(open(REPL)) if os.path.exists(REPL) else None
if repl is not None:
    R2, N_POOLED = repl["pooled"]["r2"], repl["pooled"]["n"]

# ----------------------------------------------------------------- drawing
GREEN, RED, GREY, INK = "#2f7d3f", "#b03a2e", "#6b6b6b", "#1a1a1a"

# Drawn at its final printed size (NeurIPS \linewidth is 5.5in) so \includegraphics does no
# rescaling and the point sizes below are the point sizes the reader sees.
fig, ax = plt.subplots(figsize=(6.4, 2.9))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

CX, CW, BH, H4 = 0.255, 0.44, 0.070, 0.105
RX = 0.520  # left edge of the annotation column


def box(y, text, ec=GREY, fc="white", fs=6.9, weight="normal", h=BH):
    ax.add_patch(FancyBboxPatch((CX - CW / 2, y - h / 2), CW, h,
                                boxstyle="round,pad=0.006,rounding_size=0.010",
                                linewidth=1.0, facecolor=fc, edgecolor=ec, zorder=3))
    ax.text(CX, y, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight=weight, zorder=4, linespacing=1.25)


def link(y_top, y_bot, color=GREY, lw=1.1, ls="-"):
    ax.add_patch(FancyArrowPatch((CX, y_top), (CX, y_bot), arrowstyle="-|>", mutation_scale=9,
                                 linewidth=lw, color=color, linestyle=ls,
                                 shrinkA=1, shrinkB=1, zorder=2))


# the chain: the upstream transform, then the five preservation levels
Y = [0.945, 0.856, 0.767, 0.678, 0.4055, 0.173]
box(Y[0], "upstream $d_1$ applies $T$ per client")
box(Y[1], "(P1) value: $S(T(U)) = S(U)$")
box(Y[2], "(P2) ordering: $S$'s induced ranking is fixed")
box(Y[3], "(P3) decision: $d_2$ decides the same way")
box(Y[4], "(P4) admission: the adversarial mass\n$d_2$ admits is unchanged", ec=GREEN, h=H4)
box(Y[5], "(P5) suppression: ASR stays low", ec=GREEN, fc="#f4f4f4", fs=7.2, weight="bold")

for a, b in ((0, 1), (1, 2), (2, 3)):
    link(Y[a] - BH / 2, Y[b] + BH / 2)
link(Y[3] - BH / 2, Y[4] + H4 / 2, color=RED, lw=1.8, ls=(0, (4.0, 2.0)))
link(Y[4] - H4 / 2, Y[5] + BH / 2, color=GREEN, lw=1.8)

ax.text(RX, 0.790, "(P1)$\\Rightarrow$(P2)$\\Rightarrow$(P3): implications, true by definition",
        fontsize=6.0, color=GREY, va="center", ha="left", style="italic")

# the two measured witnesses, one per direction of the break
ax.text(RX, 0.550,
        "IT BREAKS HERE, IN BOTH DIRECTIONS\n"
        "(P3) held, (P5) failed: $\\mathtt{cos\\_krum}$'s selection is\n"
        f"bit-identical at every rung (decision, admission ${max(ck_dec):.3f}$),\n"
        f"yet ASR falls ${ck_fall:.3f}$",
        fontsize=6.0, color=RED, va="center", ha="left", linespacing=1.5)
ax.text(RX, 0.285,
        f"(P3) failed, (P5) held: Krum's decision flips ${max(dec):.2f}$ of\n"
        f"rounds with admission $\\equiv {max(admn):.3f}$ (adversarial share\n"
        f"pinned at ${share['doseS|0.0']:.3f}$), and ASR stays flat "
        f"(${min(asr):.3f}$–${max(asr):.3f}$)",
        fontsize=6.0, color=GREEN, va="center", ha="left", linespacing=1.5)

ax.text(0.5, 0.015,
        "Statistic preservation is not security preservation: only the (P4)$\\to$(P5) link "
        "carries security content in our experiments.",
        fontsize=6.9, color=INK, ha="center", va="bottom", fontweight="bold")

fig.tight_layout(pad=0.2)
for d in (HERE, os.path.join(REPO, "workshop_paper", "figures")):
    if os.path.isdir(d):
        out = os.path.join(d, "story_chain.pdf")
        fig.savefig(out, bbox_inches="tight")
        print("Saved:", out)

# printed so the caption's numbers come from this run rather than from memory
print(f"\npooled Spearman rho = {sp.statistic:+.3f}  p(greater) = {sp.pvalue:.3f}  n = 16")
print("Mode S krum  kappa      " + " ".join(f"{k:7g}" for k in KAPPAS))
print("             decision   " + " ".join(f"{d:7.3f}" for d in dec))
print("             admission  " + " ".join(f"{a:7.3f}" for a in admn))
print("             ASR        " + " ".join(f"{a:7.3f}" for a in asr))
print("Mode S cos_krum         " + " ".join(f"{k:7g}" for k in KAPPAS))
print("             decision   " + " ".join(f"{d:7.3f}" for d in ck_dec))
print("             admission  " + " ".join(f"{a:7.3f}" for a in ck_adm))
print("             ASR        " + " ".join(f"{a:7.3f}" for a in ck_asr))
print(f"             ASR fall from identity = {ck_fall:.3f}")
print(f"adversarial coefficient share (Mode S, all rungs): {share['doseS|0.0']:.6f}")
print(f"pooled two-channel R^2 = {R2:.3f}  (n={N_POOLED} cells"
      + (", from analyze_dose_replication.py's refit)" if repl is not None else ")"))
if repl is not None:
    print(f"  replication arm: delta = {repl['primary']['delta']:+.3f} -> "
          + repl["primary"]["verdict"].split("--")[0].strip())
