"""
Two-panel targeted-intervention figure. Each panel carries ONE message.

Panel (a) -- the adjudicator (Mode S, Krum / model-scaling). The statistic-only instrument pins every
adversary at c=1.0 (adversarial coefficient share constant at 0.266667, spread 3.5e-7), so Lemma 1's
attenuation channel is closed by construction. Krum's *decision* (its selected client) changes in up
to 80% of rounds as the benign spread grows, while the *admission* of adversarial input it makes is
disturbed in exactly 0.000 of rounds -- and ASR stays flat and low. Statistic disturbance without
admission change does not destroy suppression: the four-levels distinction, made visible on the one
arm whose two readings predict opposite outcomes.

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

cells = json.load(open(TARGETED))["cells"]
adm = json.load(open(ADM))
summary = adm["summary"]

RHO = [float(np.exp(2 * k)) for k in KAPPAS]        # panel (a) x-axis, the dose the theorem is stated in
GAMMA = [float(np.exp(v)) for v in NUS]             # panel (b) x-axis

def draw_modeS(ax, tag="(a) "):
    """Mode S, Krum: the adjudicating arm. Statistic moves, admission does not, ASR does not."""
    # --------------------------------------------------------------- panel (a): the adjudicator
    kr = rungs_of(cells, "S", "krum", KAPPAS)
    asr = [r["mean"] for r in kr]
    asr_err = [[r["mean"] - r["lo"] for r in kr], [r["hi"] - r["mean"] for r in kr]]
    dec = [summary[f"doseS|krum|{k}|decision"] for k in KAPPAS]
    admn = [summary[f"doseS|krum|{k}|admission"] for k in KAPPAS]

    ax.axhline(0.5, color="0.35", lw=0.9, ls=":", zorder=1)
    ax.text(RHO[0] * 1.05, 0.52, "suppression threshold", fontsize=6.4, color="0.35")
    ax.plot(RHO, dec, marker="^", ms=6.5, lw=1.5, ls="--", color="#a9780a",
            label="decision change (selected client flips)", zorder=4)
    ax.plot(RHO, admn, marker="s", ms=6.5, lw=1.5, ls="-.", color="#2f7d3f",
            label="admission change (adversarial input admitted)", zorder=4)
    ax.errorbar(RHO, asr, yerr=asr_err, marker="o", ms=6.8, lw=1.8, capsize=2.6, color="#b03a2e",
                label="max-committed ASR", zorder=5)

    ax.annotate(f"decision $\\to {max(dec):.2f}$", xy=(RHO[2], dec[2]), xytext=(RHO[0] * 1.3, 0.86),
                fontsize=7.2, color="#a9780a",
                arrowprops=dict(arrowstyle="-|>", color="#a9780a", lw=0.8, shrinkA=1, shrinkB=2))
    ax.annotate("admission $\\equiv 0.00$\nASR flat", xy=(RHO[3], admn[3]), xytext=(RHO[1], 0.28),
                fontsize=7.2, color="#2f7d3f", linespacing=1.3,
                arrowprops=dict(arrowstyle="-|>", color="#2f7d3f", lw=0.8, shrinkA=1, shrinkB=2))

    ax.set_xscale("log")
    ax.set_xticks(RHO)
    ax.set_xticklabels([f"{r:.2f}\n$\\kappa{{=}}{k:g}$" for r, k in zip(RHO, KAPPAS)])
    ax.set_xlabel("dose $\\rho = e^{2\\kappa}$ (benign dispersion; adversary pinned at $c{=}1$)", fontsize=8.5)
    ax.set_ylabel("fraction of rounds  /  ASR", fontsize=8.5)
    ax.set_title(f"{tag}Mode S, Krum: the statistic moves, admission does not", fontsize=9.3, loc="left")
    ax.set_xlim(RHO[0] * 0.85, RHO[-1] * 1.25)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=6.6, loc="upper left", framealpha=0.92, handletextpad=0.5,
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
for d2 in ("reputation", "coord_median"):
    r = rungs_of(cells, "A", d2, NUS)
    print(f"panel (b) Mode A {d2}:")
    print("  r         ", " ".join(f"{g:7.3f}" for g in GAMMA))
    print("  ASR       ", " ".join(f"{x['mean']:7.3f}" for x in r))
    print("  acc       ", " ".join(f"{x['acc']:7.3f}" for x in r))
