"""
P0-3: does the separation margin behave like a CERTIFICATE whose strength tracks security?

Round 9 reviewer: "move C3 from a binary condition to a measurable quantity that predicts
robustness." We have transformed-point measurements at two heterogeneity levels, which give
paired (margin m, adversarial Weiszfeld mass Lambda_a) observations, plus the corresponding
composition ASRs.

Two links are tested:
  (1) m  ->  Lambda_a   : does Corollary 1's bound track the realized adversarial mass?
  (2) Lambda_a -> ASR   : does adversarial mass track security?

n is small (two alpha settings for the FG->RFA pipeline), so this is reported as
directional evidence, not a fitted law. A weak or non-monotone result is reported as such.
Output: results/margin_certificate/summary.json
"""
import json, os, sys
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
os.chdir(base)

SRC = {0.5: "results/theorem_quantities_transformed.json",
       0.1: "results/theorem_quantities_alpha0.1.json"}

def bound(m, n_a, n_b):
    return n_a / (n_b * (2 * m - 1) + n_a) if (2 * m - 1) > 0 else 1.0

if __name__ == "__main__":
    rounds = []
    for alpha, path in SRC.items():
        if not os.path.exists(path):
            print(f"  missing {path}"); continue
        for r in json.load(open(path))["per_round"]:
            if r["w_adv_is_zero"]:
                continue                      # Lemma 1 regime: payload zero, margin not the operative quantity
            n_a = r["n_adv_in_round"]; n_b = 5 - n_a
            rounds.append({"alpha": alpha, "m": r["condition_ratio_prime"],
                           "lambda_measured": r["lambda_adv_total"],
                           "lambda_bound": bound(r["condition_ratio_prime"], n_a, n_b),
                           "n_adv": n_a})

    print(f"=== LINK 1: margin m -> adversarial mass ===")
    print(f"    (restricted to the {len(rounds)} rounds with w_a>0; the all-round means"
          f" quoted in the paper, 3.24 vs 1.01, include the Lemma-1 w_a=0 rounds)\n")
    print(f"{'alpha':>6} {'n':>4} {'m mean':>8} {'Lambda meas':>12} {'Lambda bound':>13} {'bound holds':>12}")
    per_alpha = {}
    for a in sorted(SRC):
        sub = [r for r in rounds if r["alpha"] == a]
        if not sub: continue
        mm = sum(r["m"] for r in sub)/len(sub)
        lm = sum(r["lambda_measured"] for r in sub)/len(sub)
        lb = sum(r["lambda_bound"] for r in sub)/len(sub)
        ok = all(r["lambda_measured"] <= r["lambda_bound"] + 1e-9 for r in sub)
        per_alpha[a] = {"n": len(sub), "m_mean": mm, "lambda_measured_mean": lm,
                        "lambda_bound_mean": lb, "bound_holds_all_rounds": ok}
        print(f"{a:>6} {len(sub):>4} {mm:>8.2f} {lm:>12.4f} {lb:>13.4f} {str(ok):>12}")

    # rank correlation between m and measured mass, pooled
    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr([r["m"] for r in rounds], [r["lambda_measured"] for r in rounds])
        corr = {"spearman_rho": float(rho), "p": float(p), "n": len(rounds)}
        print(f"\n  Spearman(m, Lambda_measured) = {rho:+.3f} (p={p:.3g}, n={len(rounds)})")
    except Exception:
        corr = None
        print("\n  scipy unavailable; correlation skipped")

    print(f"\n=== LINK 2: adversarial mass -> ASR (composition level) ===\n")
    het = json.load(open("results/heterogeneity_sweep/summary.json"))["cells"]
    baseline = json.load(open("results/all_compositions/summary.json"))["pairs"]
    fg_rfa = {0.5: max(baseline["foolsgold_then_rfa"][a]["mean_asr"]
                       for a in ("committed_scaling", "committed_pixel")),
              0.1: max(het[f"foolsgold_then_rfa|alpha0.1|committed_{x}"]["mean_asr"]
                       for x in ("scaling", "pixel"))}
    link2 = []
    for a in sorted(per_alpha):
        link2.append({"alpha": a, "m_mean": per_alpha[a]["m_mean"],
                      "lambda_measured_mean": per_alpha[a]["lambda_measured_mean"],
                      "fg_rfa_max_committed_asr": fg_rfa[a]})
        print(f"  alpha={a}: m={per_alpha[a]['m_mean']:.2f}  "
              f"Lambda={per_alpha[a]['lambda_measured_mean']:.4f}  ->  FG->RFA ASR={fg_rfa[a]:.3f}")

    # sort by margin, then check ASR decreases as margin increases
    by_m = sorted(link2, key=lambda d: d["m_mean"])
    monotone = all(by_m[i]["fg_rfa_max_committed_asr"] >= by_m[i+1]["fg_rfa_max_committed_asr"]
                   for i in range(len(by_m)-1)) and all(
                   by_m[i]["lambda_measured_mean"] >= by_m[i+1]["lambda_measured_mean"]
                   for i in range(len(by_m)-1))
    direction = ("monotone in the expected direction (higher margin -> lower adversarial mass -> lower ASR)"
                 if monotone else "NOT monotone as expected")
    print(f"\n  Direction: {direction}")
    print("  CAVEAT: only two heterogeneity settings anchor link 2 (n=2 points). This is")
    print("  directional evidence that the certificate tracks security, not a fitted relationship.")

    os.makedirs("results/margin_certificate", exist_ok=True)
    json.dump({"description": "Margin as a certificate: m -> Lambda_a -> ASR",
               "link1_per_alpha": per_alpha, "link1_correlation": corr,
               "link2": link2, "direction": direction, "rounds": rounds},
              open("results/margin_certificate/summary.json", "w"), indent=2)
    print("\nSaved to results/margin_certificate/summary.json")
