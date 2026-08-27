"""
Dose-response figures. The single seven-message figure this used to be is now split in two, one
message-group each (reviewer request), while the combined two-panel layout is still emitted so the
workshop appendix that embeds it need not change.

  dose_ladder.pdf    -- "the ladder and its failed prediction": the four arms on the NOMINAL dose
                        axis rho = e^{2 kappa}, coloured by Proposition 1 invariance class, so the
                        pre-registered shapes (monotone rise = class c, flat = class a, threshold =
                        class b) are readable against what actually happened -- three arms fall.
  dose_measured.pdf  -- "the real-transform residuals, and why flip rate is not enough": ASR against
                        the INDEPENDENTLY MEASURED disturbance of each arm's own statistic (Student-t
                        95% CIs, n=5), with the two real upstream transforms (norm_clip, rfa) overlaid
                        at their own measured disturbance rate on the same (d2, attack) cell. The
                        vertical gap between an anchor and the synthetic curve at matched disturbance
                        is the residual: the transforms sit far above the curve, so flip rate is not a
                        sufficient statistic for what a transform does to a defense.
  dose_response.pdf  -- the two above side by side (panel a = measured, panel b = nominal), unchanged
                        layout, kept for the workshop appendix.

Cells failing the frozen clean-accuracy gate (mean accuracy < 0.35) are drawn hollow: a low ASR
there is a collapsed model, not suppression, and the analysis treats them as uninterpretable.

Sources (recomputed from JSON, never transcribed -- pre-registration non-negotiable 3):
  results/dose_response/summary.json        the 80 runs, per seed
  results/dose_disturbance.json             the measured abscissa
  results/cos_invariance_check.json         real-transform disturbance rates
  results/metric_swap/summary.json          real-transform ASR (frozen n=3)
  results/metric_swap_topup/summary.json    the post-hoc top-up seeds (n=5)
The per-arm loading and the anchor construction are IMPORTED from analyze_dose_response.py rather
than reimplemented, so the figures cannot drift from the scored numbers.
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

from experiments.analyze_dose_response import (  # noqa: E402
    ACC_FLOOR, DIST, INV, LADDER, SWAP, TOPUP, anchors, load_arms)

ladder = json.load(open(LADDER))
dist = json.load(open(DIST))
inv = json.load(open(INV))
swap = json.load(open(SWAP))
# Shaped exactly as anchors() expects (cell key -> per_seed list), the same reduction the analysis
# does, so the anchors here are the n=5 topped-up values it scored rather than the frozen n=3 ones.
topup = ({k: v["per_seed"] for k, v in json.load(open(TOPUP))["cells"].items()}
         if os.path.exists(TOPUP) else {})
arms = load_arms(ladder, dist)

# Colour by Proposition 1 invariance class, matching mechanism_chain.pdf: green = exactly invariant,
# amber = conditionally invariant, red = not invariant. Two arms share class (c) and are separated by
# marker and line style, not by hue, so the class reading survives greyscale printing.
CLASS_COLOR = {"a": "#2f7d3f", "b": "#a9780a", "c": "#b03a2e"}
STYLE = {  # d2 -> (marker, linestyle, label)
    "krum": ("o", "-", "Krum / scaling"),
    "reputation": ("s", "--", "Reputation / scaling"),
    "cos_krum": ("^", "-", "CosKrum / pixel"),
    "coord_median": ("D", "--", "CoordMedian / pixel"),
}
# Shape labels are the arms' own pre-registered strings from the ladder JSON, abbreviated for the
# legend only; anything unrecognized falls through verbatim rather than being silently relabelled.
SHORT = {"monotone rise": "predicted rise", "FLAT": "predicted flat",
         "shallower rise": "predicted shallower rise"}
# CosKrum and Krum finish within 0.04 ASR of each other, so their end-of-curve labels are nudged
# apart vertically; nothing else is offset.
LABEL_DY = {"krum": -5, "reputation": 0, "cos_krum": 5, "coord_median": 0}
ANCHOR_STYLE = {"norm_clip": ("*", 13, "NormClip (real)"), "rfa": ("P", 10, "RFA (real)")}


def draw_measured(ax, tag=""):
    """ASR vs. independently measured disturbance, with the real-transform anchors and residuals."""
    ax.axhline(0.5, color="0.35", lw=0.9, ls=":", zorder=1)
    ax.text(0.44, 0.517, "suppression threshold", fontsize=6.3, color="0.35")

    for arm in arms:
        cls = arm["cls"][0]
        color = CLASS_COLOR[cls]
        marker, ls, label = STYLE[arm["d2"]]
        x = [r["disturb"] for r in arm["rungs"]]
        y = [r["mean"] for r in arm["rungs"]]
        err = [[r["mean"] - r["lo"] for r in arm["rungs"]],
               [r["hi"] - r["mean"] for r in arm["rungs"]]]
        ax.errorbar(x, y, yerr=err, color=color, ls=ls, lw=1.3, elinewidth=0.9, capsize=2.4,
                    zorder=4, label=f"{label}, class ({cls}), {SHORT.get(arm['shape'], arm['shape'])}")
        for r in arm["rungs"]:
            gated = r["gated"]
            ax.plot(r["disturb"], r["mean"], marker=marker, markersize=6.2, color=color,
                    markerfacecolor="white" if gated else color, markeredgecolor=color,
                    markeredgewidth=1.4 if gated else 0.8, zorder=5)

        # real-transform anchors on this arm's own cell, at their own measured disturbance
        for a in anchors(arm, inv, swap, topup):
            am, ams, _ = ANCHOR_STYLE[a["d1"]]
            ax.plot(a["disturb"], a["asr5"], marker=am, markersize=ams, color=color,
                    markeredgecolor="black", markeredgewidth=0.7, zorder=6)
            # Residual against the nearest synthetic rung in measured disturbance. The dotted connector
            # runs from that rung's own position to the anchor's abscissa, so the reader can see which
            # rung is being matched rather than inferring it from the arrow's base.
            near = min(arm["rungs"], key=lambda r: abs(r["disturb"] - a["disturb"]))
            if abs(a["asr5"] - near["mean"]) > 0.25:
                ax.plot([near["disturb"], a["disturb"]], [near["mean"]] * 2, color="0.45", lw=0.7,
                        ls=(0, (1, 2)), zorder=3)
                ax.annotate("", xy=(a["disturb"], a["asr5"]), xytext=(a["disturb"], near["mean"]),
                            arrowprops=dict(arrowstyle="<->", color="0.25", lw=0.8,
                                            shrinkA=1.5, shrinkB=1.5), zorder=3)
                ax.text(a["disturb"] + 0.018, (a["asr5"] + near["mean"]) / 2,
                        f"$+{a['asr5'] - near['mean']:.2f}$ vs. rung\nat $d{{=}}{near['disturb']:.2f}$",
                        fontsize=6.2, color="0.2", va="center", linespacing=1.3)

    ax.set_xlabel("measured disturbance of $d_2$'s statistic (fraction of decisions changed)",
                  fontsize=8.5)
    ax.set_ylabel("max-committed ASR", fontsize=8.5)
    ax.set_title(f"{tag}ASR vs. independently measured disturbance", fontsize=9.5, loc="left")
    ax.set_xlim(-0.03, 1.06)
    ax.set_ylim(-0.05, 1.30)
    ax.tick_params(labelsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    lg = ax.legend(fontsize=6.3, loc="upper left", framealpha=0.92, handletextpad=0.5,
                   borderpad=0.4, labelspacing=0.35)
    lg.set_zorder(10)

    # anchor / gate legend, kept separate from the arm legend
    handles = [plt.Line2D([], [], marker=m, ls="none", color="0.35", markersize=s,
                          markeredgecolor="black", markeredgewidth=0.7, label=lab)
               for m, s, lab in ANCHOR_STYLE.values()]
    handles.append(plt.Line2D([], [], marker="o", ls="none", markerfacecolor="white", color="0.35",
                              markersize=6.2, markeredgewidth=1.4,
                              label=f"hollow: accuracy $< {ACC_FLOOR}$ (uninterpretable)"))
    ax.add_artist(ax.legend(handles=handles, fontsize=6.3, loc="upper right", framealpha=0.92,
                            handletextpad=0.5, borderpad=0.4, labelspacing=0.35))
    ax.add_artist(lg)

    flat_arm = next(a for a in arms if a["cls"].startswith("a"))
    ax.text(0.085, 1.03, "class (a): measured disturbance $=0$ at every rung (invariance verified),\n"
            "yet ASR falls $\\Rightarrow$ the magnitude channel, not C2 disturbance",
            fontsize=6.3, color=CLASS_COLOR["a"], linespacing=1.4, va="top", ha="left")
    # arrow drawn with explicit endpoints rather than from the text bbox, which is wide enough here that
    # matplotlib would route the tail across the NormClip anchor
    ax.annotate("", xy=(0.006, float(np.mean([r["mean"] for r in flat_arm["rungs"]])) + 0.05),
                xytext=(0.075, 0.925),
                arrowprops=dict(arrowstyle="-|>", color=CLASS_COLOR["a"], lw=0.8,
                                shrinkA=1, shrinkB=1, mutation_scale=8))


def draw_ladder(bx, tag=""):
    """The four arms on the nominal dose axis, coloured by invariance class, vs. predicted shape."""
    bx.axhline(0.5, color="0.35", lw=0.9, ls=":", zorder=1)
    for arm in arms:
        cls = arm["cls"][0]
        color = CLASS_COLOR[cls]
        marker, ls, label = STYLE[arm["d2"]]
        x = [r["rho"] for r in arm["rungs"]]
        y = [r["mean"] for r in arm["rungs"]]
        err = [[r["mean"] - r["lo"] for r in arm["rungs"]],
               [r["hi"] - r["mean"] for r in arm["rungs"]]]
        bx.errorbar(x, y, yerr=err, color=color, ls=ls, lw=1.3, elinewidth=0.9, capsize=2.4, zorder=4)
        for r in arm["rungs"]:
            bx.plot(r["rho"], r["mean"], marker=marker, markersize=6.2, color=color,
                    markerfacecolor="white" if r["gated"] else color, markeredgecolor=color,
                    markeredgewidth=1.4 if r["gated"] else 0.8, zorder=5)
        last = arm["rungs"][-1]
        bx.annotate(label, xy=(last["rho"], last["mean"]), xytext=(4, LABEL_DY[arm["d2"]]),
                    textcoords="offset points", fontsize=6.4, color=color, va="center")

    bx.set_xscale("log")
    bx.set_xticks([r["rho"] for r in arms[0]["rungs"]])
    bx.set_xticklabels([f"{r['rho']:.2f}\n$\\kappa{{=}}{r['kappa']:g}$" for r in arms[0]["rungs"]])
    bx.set_xlabel("nominal dose $\\rho = e^{2\\kappa}$ (upstream weight ratio, log scale)", fontsize=8.5)
    bx.set_ylabel("max-committed ASR", fontsize=8.5)
    bx.set_title(f"{tag}The ladder on the nominal dose axis, by invariance class",
                 fontsize=9.5, loc="left")
    bx.set_xlim(0.82, 145)
    bx.set_ylim(-0.05, 1.02)
    bx.tick_params(labelsize=7.5)
    bx.spines["top"].set_visible(False)
    bx.spines["right"].set_visible(False)
    cls_handles = [plt.Line2D([], [], color=CLASS_COLOR[c], lw=1.6, label=lab) for c, lab in
                   [("a", "class (a) exactly invariant"), ("b", "class (b) conditionally invariant"),
                    ("c", "class (c) not invariant")]]
    bx.legend(handles=cls_handles, fontsize=6.3, loc="upper left", framealpha=0.92,
              handletextpad=0.6, borderpad=0.4, labelspacing=0.35)


def save(fig, name):
    for d in (HERE, os.path.join(REPO, "workshop_paper", "figures")):
        if os.path.isdir(d):
            out = os.path.join(d, name)
            fig.savefig(out, bbox_inches="tight")
            print("Saved:", out)


# --- the two split figures (one message-group each) ---
fig_ladder, bx = plt.subplots(figsize=(6.6, 4.5))
draw_ladder(bx)
fig_ladder.tight_layout()
save(fig_ladder, "dose_ladder.pdf")

fig_measured, ax = plt.subplots(figsize=(7.0, 4.9))
draw_measured(ax)
fig_measured.tight_layout()
save(fig_measured, "dose_measured.pdf")

# --- the combined two-panel layout, kept for the workshop appendix that still embeds it ---
fig_combined, (axc, bxc) = plt.subplots(1, 2, figsize=(11.2, 4.1))
draw_measured(axc, tag="(a) ")
draw_ladder(bxc, tag="(b) ")
fig_combined.tight_layout()
save(fig_combined, "dose_response.pdf")

# printed so the caption's numbers come from this run rather than from memory
for arm in arms:
    r = arm["rungs"]
    print(f"{arm['d2']:13s}/{arm['attack'][10:]:8s} class({arm['cls'][0]}) {arm['shape']:14s} "
          f"ASR " + " ".join(f"{x['mean']:.3f}" for x in r) +
          " | disturb " + " ".join(f"{x['disturb']:.3f}" for x in r) +
          " | acc " + " ".join(f"{x['acc']:.3f}" for x in r) +
          (" | GATED: " + ",".join(f"k={x['kappa']:g}" for x in r if x["gated"])
           if any(x["gated"] for x in r) else ""))
    for a in anchors(arm, inv, swap, topup):
        near = min(r, key=lambda q: abs(q["disturb"] - a["disturb"]))
        print(f"    anchor {a['d1']:9s} disturb {a['disturb']:.3f} ASR {a['asr5']:.3f} (n={a['n5']}) "
              f"vs nearest rung {near['mean']:.3f} @ {near['disturb']:.3f} "
              f"-> residual {a['asr5'] - near['mean']:+.3f}; realized rho mean {a['rho_mean']:.2f} "
              f"max {a['rho_max']:.2f}")
