"""
Round 51 — Phase-boundary figure: persistence collapse vs complementarity survival.

Two panels stacked:
  (top)   Realized ASR vs. reputation weight on the NC+rep menu, with three
          curves: committed-scaling, committed-pixel, oracle. Plus a separate
          marker for the rep+tm survivor candidate (rep30/tm70).
  (bottom) Realized VoPD and static VoPD vs. reputation weight. Shaded regions:
           collapse (realized VoPD < 0.05, left), survival (>= 0.05, right).

Data:
  results/cifar10_mix_ratio_sweep/summary.json          (scaling, NC+rep)
  results/cifar10_mix_ratio_sweep_pixel/summary.json    (pixel,   NC+rep)
  results/cifar10_rep_tm_survivor/summary.json          (rep+tm survivor)
  results/static_vopd_mix_analysis/summary.json         (static VoPD across menus)

Output: paper/figures/phase_boundary_mix_ratio.pdf
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_path = os.path.join(base_dir, "paper", "figures", "phase_boundary_mix_ratio.pdf")

MIX_NAMES = ["NC90_rep10", "NC70_rep30", "NC50_rep50", "NC30_rep70", "NC10_rep90"]
REP_WEIGHTS = [0.10, 0.30, 0.50, 0.70, 0.90]


def load_mix_sweep(path):
    with open(path) as f:
        s = json.load(f)
    by_ratio = {}
    for name in MIX_NAMES:
        asrs = [e["attack_success_rate"] for e in s["mix_ratios"][name]["per_seed"]]
        by_ratio[name] = asrs
    return by_ratio


def load_rep_tm(path):
    with open(path) as f:
        s = json.load(f)
    out = {}
    for adv_pol in ["committed_scaling", "committed_pixel", "oracle"]:
        asrs = [e["asr_pixel_final"] for e in s["adversary_policies"][adv_pol]["per_seed"]]
        out[adv_pol] = asrs
    return out


def load_static_vopd(path):
    with open(path) as f:
        s = json.load(f)
    # Find NC+rep mean static VoPDs at the 5 ratios
    static = {}
    static_full_info = {}
    static_committed_max = {}
    rep_weights_map = {0.10: "NC90_rep10", 0.30: "NC70_rep30", 0.50: "NC50_rep50",
                       0.70: "NC30_rep70", 0.90: "NC10_rep90"}
    for entry in s["results"]:
        if entry["d1"] == "norm_clip" and entry["d2"] == "reputation":
            rep_w = round(1.0 - entry["p1"], 2)
            if rep_w in rep_weights_map:
                static[rep_w] = entry["mean_vopd"]
                static_full_info[rep_w] = entry["mean_full_info"]
                static_committed_max[rep_w] = entry["mean_committed_max"]
    # rep+tm point: rep30/tm70 -> d1=reputation p1=0.30, d2=trimmed_mean
    rep_tm_vopd = None
    rep_tm_full_info = None
    rep_tm_committed = None
    for entry in s["results"]:
        if entry["d1"] == "reputation" and entry["d2"] == "trimmed_mean" and abs(entry["p1"] - 0.30) < 0.01:
            rep_tm_vopd = entry["mean_vopd"]
            rep_tm_full_info = entry["mean_full_info"]
            rep_tm_committed = entry["mean_committed_max"]
            break
    return static, static_full_info, static_committed_max, rep_tm_vopd, rep_tm_full_info, rep_tm_committed


def main():
    scaling_path = os.path.join(base_dir, "results", "cifar10_mix_ratio_sweep", "summary.json")
    pixel_path   = os.path.join(base_dir, "results", "cifar10_mix_ratio_sweep_pixel", "summary.json")
    rep_tm_path  = os.path.join(base_dir, "results", "cifar10_rep_tm_survivor", "summary.json")
    static_path  = os.path.join(base_dir, "results", "static_vopd_mix_analysis", "summary.json")

    scaling = load_mix_sweep(scaling_path)
    have_pixel = os.path.exists(pixel_path)
    have_rep_tm = os.path.exists(rep_tm_path)

    pixel = load_mix_sweep(pixel_path) if have_pixel else None
    rep_tm = load_rep_tm(rep_tm_path) if have_rep_tm else None
    static, full_info, committed, rep_tm_vopd, rep_tm_fi, rep_tm_cm = load_static_vopd(static_path)

    # Aggregate
    rep_weights = np.array(REP_WEIGHTS)
    scaling_mean = np.array([np.mean(scaling[n]) for n in MIX_NAMES])
    scaling_std = np.array([np.std(scaling[n]) for n in MIX_NAMES])
    if pixel:
        pixel_mean = np.array([np.mean(pixel[n]) for n in MIX_NAMES])
        pixel_std = np.array([np.std(pixel[n]) for n in MIX_NAMES])
    else:
        pixel_mean = pixel_std = None

    # Realized committed-max on NC+rep: per-seed max(scaling, pixel) realized
    if pixel:
        realized_committed_max = np.array([
            np.mean([max(s, p) for s, p in zip(scaling[n], pixel[n])])
            for n in MIX_NAMES
        ])
        # Upper bound on realized VoPD on NC+rep, without running an explicit oracle:
        # static full-info >= realized full-info (since persistence boosts committed
        # ASRs but the oracle's per-round trigger picks already maximize cell-wise).
        # Actually NO: realized full-info on NC+rep COULD exceed static full-info due
        # to persistence-induced cross-contamination across rounds. So the static
        # full-info is NOT an upper bound. We report the gap
        #   "static full-info" - "realized committed-max"
        # as an INFORMATIVE diagnostic: when this gap is negative, the committed
        # adversary already exceeds the static prediction (persistence boost) and
        # realized VoPD is in the collapse band. When positive, an oracle MIGHT win.
        realized_vopd = None  # not directly measured on NC+rep
    else:
        realized_vopd = None
        realized_committed_max = None

    # Static VoPD across NC+rep
    static_arr = np.array([static[w] for w in rep_weights])

    # ----- Plot -----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 6.5), sharex=True,
                                     gridspec_kw={"height_ratios": [1.2, 1.0]})

    # Top panel: realized ASRs on NC+rep menu
    ax1.errorbar(rep_weights * 100, scaling_mean, yerr=scaling_std, marker="o",
                  color="#d62728", label="realized committed-scaling ASR", capsize=3, lw=1.5)
    if pixel:
        ax1.errorbar(rep_weights * 100, pixel_mean, yerr=pixel_std, marker="s",
                      color="#1f77b4", label="realized committed-pixel ASR", capsize=3, lw=1.5)
        ax1.plot(rep_weights * 100, realized_committed_max, marker="x",
                  color="#7f7f7f", label="realized committed-max (NC+rep)", ls=":", lw=1.0,
                  alpha=0.7)

    # Round 49 equilibrium point at rep 17%
    rep_eq_path = os.path.join(base_dir, "results", "cifar10_reputation_eq_br", "summary.json")
    if os.path.exists(rep_eq_path):
        with open(rep_eq_path) as f:
            re_data = json.load(f)
        re_scaling = [e["attack_success_rate"] for e in re_data["attacks"]["model_scaling"]["per_seed"]]
        re_pixel   = [e["attack_success_rate"] for e in re_data["attacks"]["backdoor_pixel"]["per_seed"]]
        ax1.errorbar([17], [np.mean(re_scaling)], yerr=[np.std(re_scaling)], marker="*",
                      color="#d62728", ms=14, mew=0, capsize=3, label="_nolegend_")
        ax1.errorbar([17], [np.mean(re_pixel)], yerr=[np.std(re_pixel)], marker="*",
                      color="#1f77b4", ms=14, mew=0, capsize=3, label="_nolegend_")
        ax1.annotate("Round 49 NE\n(rep 17%)", xy=(17, np.mean(re_scaling)),
                      xytext=(8, 0.55), fontsize=8, color="dimgray",
                      arrowprops=dict(arrowstyle="->", lw=0.6, color="dimgray"))

    # rep+tm point shown as an inset star at far right + arrow
    if rep_tm:
        rt_scaling = np.mean(rep_tm["committed_scaling"])
        rt_pixel   = np.mean(rep_tm["committed_pixel"])
        rt_oracle  = np.mean(rep_tm["oracle"])
        # Plot at x=110 (off NC+rep menu) for visual separation
        ax1.scatter([110], [rt_scaling], marker="o", color="#d62728", s=80, edgecolors="black",
                     linewidths=1.0, zorder=5)
        ax1.scatter([110], [rt_pixel], marker="s", color="#1f77b4", s=80, edgecolors="black",
                     linewidths=1.0, zorder=5)
        ax1.scatter([110], [rt_oracle], marker="D", color="#2ca02c", s=80, edgecolors="black",
                     linewidths=1.0, zorder=5)
        ax1.axvline(105, color="gray", lw=0.8, ls=":")
        ax1.annotate("rep+tm\n(rep30/tm70)", xy=(110, rt_oracle + 0.05), fontsize=8,
                      ha="center", color="black")

    ax1.set_ylabel("Realized ASR (50 rounds)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="center left", fontsize=8, framealpha=0.95)
    ax1.grid(True, ls=":", alpha=0.5)
    ax1.set_title("Persistence collapse / complementarity survival boundary")

    # Bottom panel: VoPDs
    ax2.plot(rep_weights * 100, static_arr, marker="o", color="#9467bd",
              label="static VoPD (NC+rep menu)", lw=1.5)
    # On NC+rep, realized oracle not measured; show realized-committed-max being
    # competitive with static full-info (collapse signature)
    if rep_tm is not None and rep_tm_vopd is not None:
        # static VoPD at rep30/tm70 (proxy circle)
        ax2.scatter([110], [rep_tm_vopd], marker="o", color="#9467bd",
                     s=90, edgecolors="black", linewidths=1.0, zorder=5,
                     label="static VoPD (rep+tm)")
        # realized VoPD at rep+tm (MEASURED with oracle)
        rt_realized_committed = max(np.mean(rep_tm["committed_scaling"]),
                                     np.mean(rep_tm["committed_pixel"]))
        rt_realized_vopd = np.mean(rep_tm["oracle"]) - rt_realized_committed
        ax2.scatter([110], [rt_realized_vopd], marker="*", color="#ff7f0e",
                     s=200, edgecolors="black", linewidths=1.2, zorder=6,
                     label=f"realized VoPD (rep+tm) = +{rt_realized_vopd:.3f}")

    # Shade collapse vs survival regions on the NC+rep menu only
    ax2.axhspan(-0.05, 0.05, alpha=0.10, color="red")
    ax2.axhspan(0.05, 0.30, alpha=0.10, color="green")
    ax2.text(2, -0.025, "collapse", color="darkred", fontsize=8, alpha=0.7)
    ax2.text(2, 0.225, "survival", color="darkgreen", fontsize=8, alpha=0.7)

    ax2.axhline(0, color="black", lw=0.6, ls="-")
    ax2.axvline(105, color="gray", lw=0.8, ls=":")
    ax2.set_xlabel("Reputation weight in deployed mix (%)  |  rep+tm = rep30/tm70")
    ax2.set_ylabel("VoPD")
    ax2.set_xlim(-3, 118)
    ax2.set_ylim(-0.05, 0.30)
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.95)
    ax2.grid(True, ls=":", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved figure: {out_path}")

    # Print numerical summary for paper
    print(f"\n=== Phase-boundary numerical summary ===")
    print(f"NC+rep menu (realized ASR, 5 seeds):")
    for i, name in enumerate(MIX_NAMES):
        rep_w_pct = int(rep_weights[i] * 100)
        s_str = f"scaling={scaling_mean[i]:.3f}"
        p_str = f"pixel={pixel_mean[i]:.3f}" if pixel else "pixel=N/A"
        cm_str = f"committed-max={realized_committed_max[i]:.3f}" if pixel else "committed-max=N/A"
        static_str = f"static VoPD={static[rep_weights[i]]:.3f}"
        print(f"  rep{rep_w_pct:>2d}%: {s_str:18s} {p_str:18s} {cm_str:24s} {static_str}")
    if rep_tm:
        rt_s = np.mean(rep_tm["committed_scaling"])
        rt_p = np.mean(rep_tm["committed_pixel"])
        rt_o = np.mean(rep_tm["oracle"])
        rt_vopd = rt_o - max(rt_s, rt_p)
        print(f"\nrep+tm at rep30/tm70 (realized ASR, 5 seeds):")
        print(f"  scaling={rt_s:.3f}  pixel={rt_p:.3f}  oracle={rt_o:.3f}  realized VoPD={rt_vopd:+.3f}")
        print(f"  static VoPD at this point: {rep_tm_vopd:.3f}")


if __name__ == "__main__":
    main()
