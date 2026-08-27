"""Split the published aggregate displacement into its RE-SELECTION and RESCALING components.

No ASR, no per-rung training, no pre-registration: this decomposes a number that is already frozen
and published rather than testing a new prediction.

WHAT IS DECOMPOSED. results/admission_measurement.json reports, per arm and rung, the relative
displacement of the emitted aggregate, ||agg(T(U)) - agg(U)|| / ||agg(U)|| (the paper's `Delta agg.`
column). For Mode S into Krum it reads 0.892 at kappa=2. That single number mixes two different things,
and the paper's causal claim needs them apart. For a selector, agg(U) = u_{s0} and agg(T(U)) = c_{s1}
u_{s1}, where s0 is the client selected on the raw stack and s1 the client selected on the transformed
stack, so exactly:

    c_{s1} u_{s1} - u_{s0}  =  (u_{s1} - u_{s0})  +  (c_{s1} - 1) u_{s1}
    -----------------------     ---------------      ------------------
    total (published)           RE-SELECTION         RESCALING

  RE-SELECTION  ||u_{s1} - u_{s0}|| / ||u_{s0}||, BOTH updates taken untransformed. This is the
                displacement that survives the score-only control, because that control aggregates the
                untransformed selected update: it is what a magnitude-controlled dose still moves.
  RESCALING     ||(c_{s1} - 1) u_{s1}|| / ||u_{s0}||. This is the component the score-only control
                closes, and the component Mode S leaves open.

The two components are the norms of two vectors that sum to the total displacement vector; they are not
orthogonal, so they need not sum to the total scalar. The angle between them is therefore reported, and
so is the exact vector identity as a residual assertion (must be ~0 up to float32).

Rounds where the transformed and raw stacks select the SAME client are separated out: there the
re-selection component is exactly 0 and the entire displacement is rescaling. That split is the
quantitative form of the paper's argument that Krum's decision changes are mostly benign-to-benign.

CROSS-CHECK. The total is recomputed here from the same single-sourced mirrors and asserted equal to
the published value in results/admission_measurement.json, so this is provably a decomposition OF the
published number rather than of a second, similar one. Same seeds, same rounds, same rungs, same
attack map, same configuration as that measurement.

Frozen artifacts are read-only. Writes results/displacement_decomposition.json (a NEW file);
results/admission_measurement.json is left byte-identical.

Run: python3 experiments/measure_displacement_decomposition.py
"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
from fl_core import get_federated_dataset, get_model, FederatedServer, FederatedClient  # noqa: E402
from attacks import get_attack  # noqa: E402
from experiments.run_all_compositions import apply_d1_transform  # noqa: E402
# The same single-sourced mirrors measure_admission.py and the score-only control use, so the three
# cannot drift apart in what they call "the statistic" or "the selected update".
from experiments.verify_cos_invariance import krum_selection, flatten  # noqa: E402
# The published total, its reader, and the configuration are imported rather than restated.
from experiments.measure_admission import (  # noqa: E402
    ATTACK_MAP, F_ADV, K, KAPPAS, N, ROUNDS, SEEDS, rel_disp,
)

ADM = os.path.join(base, "results", "admission_measurement.json")
OUT = os.path.join(base, "results", "displacement_decomposition.json")

# Selectors only: a weighted averager and a coordinate-wise order statistic emit no single selected
# client, so "the selected update" -- and hence this decomposition -- is undefined for them.
ARMS = [("krum", False, "committed_scaling"), ("cos_krum", True, "committed_pixel")]
FAMILIES = [("dose", "the confounded ladder"), ("doseS", "the Mode-S instrument")]
TOTAL_TOL = 1e-6      # agreement with the published total, recomputed from the same mirrors


def published_total(family, arm, rung):
    """The frozen `Delta agg.` value this script decomposes. Read, never re-derived."""
    if not os.path.exists(ADM):
        return None
    return json.load(open(ADM)).get("summary", {}).get(f"{family}|{arm}|{rung}|agg_disp")


def measure():
    """Per-(family, rung, seed, round) decomposition, every rung on the SAME raw updates."""
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    rows = []
    for arm, cosine, attack_name in ARMS:
        for seed in SEEDS:
            torch.manual_seed(seed); np.random.seed(seed)
            cd, _, nc = get_federated_dataset("cifar10", N, 0.5, seed)
            srv = FederatedServer(get_model("cifar_cnn", nc), dev)
            atk = get_attack(ATTACK_MAP[attack_name])
            adv = set(range(int(N * F_ADV)))
            cl = [FederatedClient(i, atk.poison_dataset(cd[i]) if i in adv else cd[i], dev)
                  for i in range(N)]
            for rnd in range(ROUNDS):
                pids = np.random.choice(N, K, replace=False)
                ups = []
                for cid in pids:
                    u = cl[cid].train(srv.global_model, 1, 0.01, 64)
                    if cid in adv:
                        u = atk.manipulate_update(u, srv.global_model)
                    ups.append(u)
                raw = flatten(ups)
                adv_mask = [bool(cid in adv) for cid in pids]
                s0, _ = krum_selection(raw, cosine)
                for family, _ in FAMILIES:
                    for kappa in KAPPAS:
                        d1 = f"{family}_kappa{kappa}"
                        st = flatten(apply_d1_transform(ups, d1, tau=5.0, dose_key=(seed, rnd),
                                                        adv_mask=adv_mask))
                        s1, _ = krum_selection(st, cosine)
                        # agg(U) = raw[s0]; agg(T(U)) = st[s1]; the score-only aggregate = raw[s1].
                        total_v = st[s1] - raw[s0]
                        resel_v = raw[s1] - raw[s0]          # survives the score-only control
                        rescale_v = st[s1] - raw[s1]         # closed by the score-only control
                        den = float(raw[s0].norm().clamp(min=1e-12).item())
                        nr, nc_ = (float(resel_v.norm().item()), float(rescale_v.norm().item()))
                        cos_ang = (float(torch.dot(resel_v, rescale_v).item()) / max(nr * nc_, 1e-30)
                                   if nr > 0 and nc_ > 0 else float("nan"))
                        rows.append({
                            "arm": arm, "family": family, "rung": float(kappa), "d1": d1,
                            "attack": attack_name, "seed": int(seed), "round": int(rnd),
                            "sel_raw": int(s0), "sel_transformed": int(s1),
                            "selection_changed": bool(s1 != s0),
                            "sel_is_adv": bool(adv_mask[s1]),
                            "c_sel": float((st[s1].norm() / raw[s1].norm().clamp(min=1e-12)).item()),
                            "total": float(total_v.norm().item()) / den,
                            "reselection": nr / den,
                            "rescaling": nc_ / den,
                            # exact vector identity: total = reselection + rescaling as VECTORS
                            "identity_residual": float((total_v - resel_v - rescale_v).norm().item()) / den,
                            "cos_between_components": cos_ang,
                            # the displacement the score-only control still emits, for the record
                            "score_only_disp": rel_disp(raw[s1], raw[s0]),
                        })
                srv.apply_update(srv.aggregate(ups))
    return rows


def emit_tex():
    """Print the appendix table from the saved decomposition, so the papers transcribe nothing.

    Read-only on results/. Runs in milliseconds; the measurement itself does not have to be repeated
    to regenerate the table.
    """
    if not os.path.exists(OUT):
        sys.exit(f"missing {os.path.relpath(OUT, base)} -- run this script without --tex first")
    d = json.load(open(OUT))
    s, A = d["summary"], d["assertions"]
    print("=== LATEX (copy verbatim; regenerate rather than edit) ===\n")
    tex = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
           r"Design & Selector & $\kappa$ & total & re-selection & rescaling & "
           r"$\cos$ & sel.\ chg. \\", r"\midrule"]
    for arm, _, _ in ARMS:
        for family, blurb in FAMILIES:
            label = ("Instrument (Mode~S)" if family == "doseS" else "Confounded ladder")
            name = r"\texttt{cos\_krum}" if arm == "cos_krum" else r"\texttt{krum}"
            for i, kappa in enumerate(KAPPAS):
                r = s.get(f"{family}|{arm}|{kappa}")
                if r is None:
                    continue
                cosv = r["mean_cos_between_components"]
                # nan cos means at least one component vector is exactly zero, so no angle exists
                cos_s = "---" if cosv != cosv else f"${cosv:+.2f}$"
                tex.append(f"{label if i == 0 else ''} & {name if i == 0 else ''} & "
                           f"${kappa:g}$ & ${r['total']:.3f}$ & "
                           f"$\\mathbf{{{r['reselection']:.3f}}}$ & ${r['rescaling']:.3f}$ & "
                           f"{cos_s} & ${r['selection_change_rate']:.3f}$ \\\\")
            tex.append(r"\midrule" if (arm, family) != (ARMS[-1][0], FAMILIES[-1][0]) else r"\bottomrule")
    tex.append(r"\end{tabular}")
    print("\n".join(tex))
    print(f"\n% worst vector-identity residual {A['worst_identity_residual']:.2e}; "
          f"worst gap against the published total {A['worst_published_gap']:.2e} "
          f"(tolerance {A['total_tolerance']:g}); matches_published={A['matches_published']}")
    return 0


def main():
    if "--tex" in sys.argv:
        return emit_tex()
    if not os.path.exists(ADM):
        sys.exit(f"missing {ADM} -- run experiments/measure_admission.py first; this script "
                 "decomposes that file's published totals and cross-checks against them")
    print("=== DECOMPOSITION OF THE PUBLISHED AGGREGATE DISPLACEMENT (no ASR, no training per rung) ===")
    print(f"    {len(SEEDS)} seeds x {ROUNDS} live rounds x {len(KAPPAS)} rungs x {len(FAMILIES)} "
          f"families, selectors only\n", flush=True)
    rows = measure()

    worst_identity, worst_total_gap, summary, bad = 0.0, 0.0, {}, []
    for arm, _, _ in ARMS:
        for family, blurb in FAMILIES:
            print(f"  --- {arm} / {family} ({blurb}) ---")
            print(f"  {'kappa':>6}  {'total':>8}  {'re-select':>10}  {'rescale':>8}  "
                  f"{'sel chg':>8}  {'c_sel':>7}  {'cos':>6}   published")
            for kappa in KAPPAS:
                sub = [r for r in rows if r["arm"] == arm and r["family"] == family
                       and abs(r["rung"] - kappa) < 1e-12]
                if not sub:
                    continue
                m = {k: float(np.mean([r[k] for r in sub]))
                     for k in ("total", "reselection", "rescaling", "c_sel",
                               "selection_changed", "score_only_disp")}
                cosv = float(np.nanmean([r["cos_between_components"] for r in sub]))
                worst_identity = max(worst_identity, max(r["identity_residual"] for r in sub))
                pub = published_total(family, arm, kappa)
                gap = None if pub is None else abs(m["total"] - pub)
                if gap is not None:
                    worst_total_gap = max(worst_total_gap, gap)
                    if gap > TOTAL_TOL:
                        bad.append(f"{family}|{arm}|{kappa}: recomputed {m['total']:.6f} vs "
                                   f"published {pub:.6f} (gap {gap:.2e})")
                summary[f"{family}|{arm}|{kappa}"] = {
                    "total": m["total"], "reselection": m["reselection"],
                    "rescaling": m["rescaling"], "mean_c_sel": m["c_sel"],
                    "selection_change_rate": m["selection_changed"],
                    "score_only_displacement": m["score_only_disp"],
                    "mean_cos_between_components": cosv,
                    "published_total": pub, "published_gap": gap,
                }
                print(f"  {kappa:>6}  {m['total']:>8.3f}  {m['reselection']:>10.3f}  "
                      f"{m['rescaling']:>8.3f}  {m['selection_changed']:>8.3f}  "
                      f"{m['c_sel']:>7.3f}  {cosv:>+6.2f}   "
                      + ("na" if pub is None else f"{pub:.3f} (d={gap:.1e})"))
            # rounds that keep the same client: the entire displacement is rescaling there
            same = [r for r in rows if r["arm"] == arm and r["family"] == family
                    and r["rung"] == KAPPAS[-1] and not r["selection_changed"]]
            chg = [r for r in rows if r["arm"] == arm and r["family"] == family
                   and r["rung"] == KAPPAS[-1] and r["selection_changed"]]
            print(f"  at kappa={KAPPAS[-1]}: {len(same)} rounds keep the same client "
                  f"(re-selection exactly 0, all displacement is rescaling), "
                  f"{len(chg)} change it"
                  + (f"; of those that change, {sum(r['sel_is_adv'] for r in chg)} land on an "
                     f"adversary" if chg else ""))
            print()

    print("=== ASSERTIONS ===")
    print(f"  vector identity total = re-selection + rescaling: worst residual "
          f"{worst_identity:.2e} (exact up to float32)")
    print(f"  agreement with results/admission_measurement.json: worst gap {worst_total_gap:.2e} "
          f"(tolerance {TOTAL_TOL:g})")
    if bad:
        print("  MISMATCH -- this is NOT a decomposition of the published number:")
        for b in bad:
            print("   !", b)
    else:
        print("  Every recomputed total matches its published value: this decomposes the published "
              "`Delta agg.` column.")

    print("\n=== WHAT THIS LICENSES ===")
    k = f"doseS|krum|{KAPPAS[-1]}"
    if k in summary:
        s = summary[k]
        print(f"  Mode S into Krum at kappa={KAPPAS[-1]}: published displacement "
              f"{s['published_total']:.3f} = re-selection {s['reselection']:.3f} + rescaling "
              f"{s['rescaling']:.3f} (as vectors).")
        print(f"  The score-only control closes the rescaling part and still emits "
              f"{s['score_only_displacement']:.3f} of re-selection displacement, so it is not a")
        print("  no-op intervention: it holds the aggregated update's magnitude and direction fixed")
        print("  while leaving the decision free to move, which is exactly its purpose.")
    print("  This is a decomposition of a displacement, not of an ASR effect. It says what the dose")
    print("  moves in the update entering training; it makes no claim about suppression.")

    json.dump({"description": "Decomposition of the published aggregate displacement into "
                              "re-selection and rescaling components. Selectors only. No ASR.",
               "decomposes": "results/admission_measurement.json summary '<family>|<arm>|<rung>|"
                             "agg_disp'",
               "identity": "c_{s1} u_{s1} - u_{s0} = (u_{s1} - u_{s0}) + (c_{s1} - 1) u_{s1}; "
                           "components are not orthogonal, so their norms need not sum to the total",
               "config": {"dataset": "cifar10", "model": "cifar_cnn", "N": N, "K": K, "f": F_ADV,
                          "alpha": 0.5, "seeds": SEEDS, "rounds": ROUNDS, "kappas": KAPPAS,
                          "arms": [a for a, _, _ in ARMS],
                          "families": [f for f, _ in FAMILIES]},
               "assertions": {"worst_identity_residual": worst_identity,
                              "worst_published_gap": worst_total_gap,
                              "total_tolerance": TOTAL_TOL,
                              "matches_published": not bad, "mismatches": bad},
               "summary": summary, "rows": rows}, open(OUT, "w"), indent=2)
    print(f"\nSaved {OUT}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
