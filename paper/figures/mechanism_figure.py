"""
Two-panel mechanism figure (Figure 1).

Panel (a): the decision chain the criterion actually executes -- upstream transform T,
downstream statistic S, invariance class of S under T, predicted C2 verdict -- annotated
with the four invariance classes of Proposition 1 and the defenses that fall in each.

Panel (b): the measured chain m -> Lambda_a -> ASR for FoolsGold->RFA at two heterogeneity
settings. Per-round scatter of the separation margin m against the realized adversarial
Weiszfeld mass Lambda_a, with Corollary 1's bound Lambda_a <= n_a/(n_b(2m-1)+n_a) overlaid.
Only rounds with w_adv > 0 are plotted; rounds where FoolsGold zeroes every adversary are
Lemma 1 (annihilation) and carry no payload.

Sources (recomputed, not transcribed):
  results/theorem_quantities_transformed.json   alpha=0.5, per_round
  results/theorem_quantities_alpha0.1.json      alpha=0.1, per_round
  results/heterogeneity_sweep/summary.json      composition ASR per seed
Writes mechanism_chain.pdf next to this script (and to the sibling paper's figures/ dir).
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(REPO, "results")


def load_rounds(fname):
    per = json.load(open(os.path.join(RESULTS, fname)))["per_round"]
    return [r for r in per if not r["w_adv_is_zero"]]


def max_committed(cells, pair, alpha):
    """Max over the two committed attacks of the per-seed mean; plus the pooled per-seed spread."""
    out = {}
    for atk in ("committed_scaling", "committed_pixel"):
        c = cells[f"{pair}|alpha{alpha}|{atk}"]
        out[atk] = (c["mean_asr"], [s["asr"] for s in c["per_seed"]])
    atk = max(out, key=lambda a: out[a][0])
    seeds = np.array(out[atk][1])
    return seeds.mean(), seeds.std(ddof=1)


rounds = {0.5: load_rounds("theorem_quantities_transformed.json"),
          0.1: load_rounds("theorem_quantities_alpha0.1.json")}
het = json.load(open(os.path.join(RESULTS, "heterogeneity_sweep", "summary.json")))["cells"]


def flagship_alpha05():
    """alpha=0.5 is not in the sweep (the sweep's own note says it was not re-run). The largest
    measurement of this cell is the n=30 base condition of the flagship/adaptive study, so read it
    from there rather than hardcoding the n=3 mean, which sampled the left tail of a skewed
    distribution (0.045 at n=3 vs 0.093 at n=30)."""
    b = json.load(open(os.path.join(RESULTS, "fg_rfa_flagship", "summary.json")))["base_composition"]
    best = max(("committed_scaling", "committed_pixel"),
               key=lambda a: np.mean([r["asr"] for r in b[a]["per_seed"]]))
    a = np.array([r["asr"] for r in b[best]["per_seed"]])
    return a.mean(), a.std(ddof=1)


asr = {0.5: flagship_alpha05(),                       # n=30, seeds 42--71
       0.1: max_committed(het, "foolsgold_then_rfa", "0.1")}

fig = plt.figure(figsize=(11.6, 4.0))
gs = fig.add_gridspec(1, 2, width_ratios=[1.28, 1.0], wspace=0.22)

# ----------------------------------------------------------------- panel (a)
ax = fig.add_subplot(gs[0, 0])
ax.set_xlim(0, 10.35)
ax.set_ylim(1.0, 9.9)
ax.axis("off")
ax.set_title("(a) What the criterion checks: invariance class of $d_2$'s statistic",
             fontsize=9.5, loc="left")

BOX = dict(boxstyle="round,pad=0.26", linewidth=0.9)


def box(x, y, w, h, text, fc, ec, fs=7.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, facecolor=fc, edgecolor=ec, **BOX))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.4)


def arrow(x0, y0, x1, y1, color="0.35"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8,
                                 linewidth=0.9, color=color, shrinkA=0, shrinkB=0))


# the two inputs
box(0.05, 7.30, 2.15, 1.45, "upstream $d_1$ gives\n$T(u_i)=c_i u_i$, $c_i>0$", "#eaf1f8", "#5588bb", fs=7.0)
box(0.05, 5.30, 2.15, 1.45, "downstream $d_2$ reads\nstatistic $S(u_1,\\dots,u_K)$", "#eaf1f8", "#5588bb", fs=7.0)
ax.text(1.12, 4.95, "the only two inputs;\nno experiment needed",
        ha="center", va="top", fontsize=6.2, style="italic", color="0.45")

# routing spine
CLASSES = [
    (8.15, "exactly invariant", "$S$ is a function of the directions $\\hat u_i$ alone\ncosine similarity: FoolsGold, FLTrust",
     "#e3f2e6", "#2f7d3f", "C2 holds", "#2f7d3f"),
    (6.30, "conditionally invariant", "coordinate ordering, survives iff $S>\\rho$\nCoordMedian, TrimmedMean",
     "#fdf4e0", "#a9780a", "C2 holds iff\nthe margin survives", "#a9780a"),
    (4.45, "not invariant", "consensus distance to the client median\nReputation",
     "#fbe9e7", "#b03a2e", "C2 fails", "#b03a2e"),
    (2.60, "not invariant", "pairwise distance, residual magnitude\nKrum, Multi-Krum",
     "#fbe9e7", "#b03a2e", "C2 fails", "#b03a2e"),
]
SPINE = 2.52
ax.plot([SPINE, SPINE], [CLASSES[-1][0] + 0.58, CLASSES[0][0] + 0.58], color="0.35", lw=0.9)
for src_y in (8.02, 6.02):
    arrow(2.22, src_y, SPINE, src_y)
for y, head, body, fc, ec, verdict, vc in CLASSES:
    yc = y + 0.58
    box(2.78, y, 4.05, 1.15, f"{head}\n{body}", fc, ec, fs=6.5)
    arrow(SPINE, yc, 2.75, yc)
    box(7.95, y + 0.06, 2.00, 1.02, verdict, "white", vc, fs=7.0)
    arrow(7.10, yc, 7.92, yc, color=vc)

ax.text(5.15, 1.75, "Proposition 1 assigns the class; the C2 verdict follows without running the composition.",
        ha="center", va="center", fontsize=6.5, style="italic", color="0.35")

# ----------------------------------------------------------------- panel (b)
bx = fig.add_subplot(gs[0, 1])
bx.set_title("(b) Measured: margin $m$ $\\to$ adversarial mass $\\Lambda_a$ $\\to$ ASR",
             fontsize=9.5, loc="left")

mgrid = np.linspace(1.0, 8.0, 300)
for n_a, n_b, ls in ((1, 4, "-"), (2, 3, "--")):
    bx.plot(mgrid, n_a / (n_b * (2 * mgrid - 1) + n_a), ls, color="0.45", lw=1.0,
            label=f"Cor. 1 bound, $n_a{{=}}{n_a}$")

STYLE = {0.5: ("#2f7d3f", "o", "$\\alpha{=}0.5$"), 0.1: ("#b03a2e", "s", "$\\alpha{=}0.1$")}
for alpha, (color, marker, lab) in STYLE.items():
    m = np.array([r["condition_ratio_prime"] for r in rounds[alpha]])
    lam = np.array([r["lambda_adv_total"] for r in rounds[alpha]])
    bx.scatter(m, np.maximum(lam, 1e-4), s=26, facecolor=color, edgecolor="white",
               linewidth=0.5, alpha=0.9, marker=marker, zorder=4,
               label=f"{lab}: $\\bar m={m.mean():.2f}$, ASR ${asr[alpha][0]:.3f}$")
    bx.errorbar(m.mean(), max(np.mean(lam), 1e-4), xerr=m.std(ddof=1),
                yerr=[[max(np.mean(lam) - 1e-4, 0)], [np.std(lam, ddof=1)]],
                fmt=marker, color=color, markersize=9, markeredgecolor="black",
                markeredgewidth=0.7, elinewidth=1.2, capsize=3, zorder=6)

bx.axvline(1.0, color="0.3", lw=0.9, ls=":")
bx.text(1.12, 0.88, "$m\\to1$: bound degrades\nto $n_a/(n_b{+}n_a)$", fontsize=6.3,
        color="0.3", va="top")
bx.set_yscale("log")
bx.set_xlim(-0.35, 8.2)
bx.set_ylim(8e-5, 1.4)
bx.set_xlabel("separation margin $m = \\Delta'_{\\mathrm{sep}}/[2(R'_B+\\delta')]$", fontsize=8.5)
bx.set_ylabel("realized adversarial mass $\\Lambda_a$", fontsize=8.5)
bx.tick_params(labelsize=7.5)
bx.spines["top"].set_visible(False)
bx.spines["right"].set_visible(False)
bx.legend(fontsize=6.3, loc="upper right", framealpha=0.9, handletextpad=0.4)

plt.tight_layout()
for d in (HERE, os.path.join(REPO, "workshop_paper", "figures")):
    if os.path.isdir(d):
        out = os.path.join(d, "mechanism_chain.pdf")
        plt.savefig(out, bbox_inches="tight")
        print("Saved:", out)

# printed for transcription into the captions
for alpha in (0.5, 0.1):
    m = np.array([r["condition_ratio_prime"] for r in rounds[alpha]])
    lam = np.array([r["lambda_adv_total"] for r in rounds[alpha]])
    print(f"alpha={alpha}: n_rounds={len(m)} m={m.mean():.2f}+-{m.std(ddof=1):.2f} "
          f"Lambda_a={lam.mean():.4f}+-{lam.std(ddof=1):.4f} (max {lam.max():.4f}) "
          f"ASR={asr[alpha][0]:.3f}+-{asr[alpha][1]:.3f}")
