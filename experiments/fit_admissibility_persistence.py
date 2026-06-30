"""
Round 50 — Quantitative Proposition 4 refinement attempt (reviewer C2).

The Round 47 joint fit of (gamma, alpha, p_FA, p_NC) hit a corner solution due to
FedAvg-bimodality and failed an out-of-sample test (predicted ASR_50 = 0.80 vs
observed 0.96). The Round 50 attempt introduces a per-seed admissibility latent
separating training-init from Markov dynamics:

  a_seed = 1 if ASR_25 > 0.5 else 0  (binding-cell embedded)
  Conditional on a_seed = 1, fit the Markov model on per-round ASR.

For each condition (fedavg_pure, normclip_pure, ne3_mix), fit:
  s_{t+1} = s_t + alpha * (1 - s_t) if admitting round (prob p_admit)
          = gamma * s_t              otherwise (prob 1 - p_admit)

Initial state s_0 = 0; condition on admissibility a_seed = 1.

Out-of-sample test:
  Fit on rounds 1-25 -> predict rounds 26-50.
  Success: |predicted ASR_50 - observed ASR_50| <= 0.10.

Output: results/persistence/fit_admissibility_v3.json
"""
import json
import os
import sys
import numpy as np
from scipy.optimize import minimize

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ps_root = os.path.join(base_dir, "results", "persistence")
output_path = os.path.join(ps_root, "fit_admissibility_v3.json")

CONDITIONS = ["fedavg_pure", "normclip_pure", "ne3_mix"]
SEEDS = [42, 43, 44, 45, 46]

# Effective admission probability per condition (from prior fits)
# fedavg_pure: p_admit ~ 0.5 (Bernoulli-binding-cell)
# normclip_pure: p_admit = 1.0 (always admits scaling at tau=5)
# ne3_mix: p_admit ~ 0.83 (0.26 * 0.5 + 0.74 * 1.0)
P_ADMIT_PRIOR = {"fedavg_pure": 0.50, "normclip_pure": 1.0, "ne3_mix": 0.835}


def load_traj(condition, seed):
    path = os.path.join(ps_root, condition, f"seed_{seed}", "asr_timeline.json")
    with open(path) as f:
        d = json.load(f)
    return [e["asr"] for e in d["asr_timeline"]]


def classify_admissibility(traj, threshold=0.5, round_check=25):
    """Return 1 if the trajectory has admitted (ASR_25 > threshold), 0 otherwise."""
    return 1 if traj[round_check - 1] > threshold else 0


def predict_trajectory(alpha, gamma, p_admit, T):
    """Predict expected ASR per round under linearized Markov dynamics, starting at s_0=0."""
    # Expected dynamics: E[s_{t+1}] = p_admit * (s_t + alpha*(1-s_t)) + (1-p_admit) * gamma * s_t
    #                              = (p_admit*(1-alpha) + (1-p_admit)*gamma) * s_t + p_admit * alpha
    s = [0.0]
    c = p_admit * (1 - alpha) + (1 - p_admit) * gamma
    b = p_admit * alpha
    for _ in range(T):
        s.append(min(1.0, c * s[-1] + b))
    return s[1:]  # rounds 1..T


def neg_log_likelihood(params, traj, p_admit, T_fit):
    """Negative log-likelihood under Gaussian noise model (sigma^2 = 0.01)."""
    alpha, gamma = params
    if not (0 < alpha <= 1) or not (0 <= gamma <= 1):
        return 1e6
    predicted = predict_trajectory(alpha, gamma, p_admit, T_fit)
    residuals = np.array(traj[:T_fit]) - np.array(predicted)
    # NLL = sum (residuals^2) / (2 sigma^2)
    return float(np.sum(residuals ** 2))


def fit_one_seed(traj, p_admit, T_fit=25):
    """Fit (alpha, gamma) on first T_fit rounds; return alpha_hat, gamma_hat."""
    best = (None, 1e9)
    # Multiple random starts to avoid local minima
    for a0 in [0.1, 0.3, 0.6]:
        for g0 in [0.3, 0.6, 0.9]:
            res = minimize(neg_log_likelihood, [a0, g0],
                            args=(traj, p_admit, T_fit),
                            method="L-BFGS-B",
                            bounds=[(0.01, 0.99), (0.01, 0.99)])
            if res.fun < best[1]:
                best = (res.x, res.fun)
    return best[0]


def main():
    summary = {"description": "Round 50 quantitative Prop 4 refinement with admissibility latent",
                "conditions": {}}
    print(f"=== Admissibility-latent persistence model fit ===\n")

    for cond in CONDITIONS:
        print(f"--- {cond} (p_admit = {P_ADMIT_PRIOR[cond]}) ---")
        cond_results = {"p_admit_used": P_ADMIT_PRIOR[cond], "per_seed": [], "admit_seeds": [], "noadmit_seeds": []}
        for seed in SEEDS:
            traj = load_traj(cond, seed)
            adm = classify_admissibility(traj, threshold=0.5, round_check=25)
            asr_25 = traj[24]
            asr_50 = traj[49]
            if adm == 0:
                print(f"  seed {seed}: NO-ADMIT mode (ASR_25 = {asr_25:.3f}), skip fit")
                cond_results["noadmit_seeds"].append(seed)
                continue
            cond_results["admit_seeds"].append(seed)
            params = fit_one_seed(traj, P_ADMIT_PRIOR[cond], T_fit=25)
            if params is None:
                continue
            alpha_hat, gamma_hat = params
            predicted_full = predict_trajectory(alpha_hat, gamma_hat, P_ADMIT_PRIOR[cond], 50)
            asr_50_pred = predicted_full[49]
            err_25_50 = asr_50_pred - asr_50
            print(f"  seed {seed}: alpha={alpha_hat:.3f} gamma={gamma_hat:.3f}  "
                  f"ASR_25 obs={asr_25:.3f}  ASR_50 obs={asr_50:.3f} pred={asr_50_pred:.3f} err={err_25_50:+.3f}")
            cond_results["per_seed"].append({
                "seed": seed, "alpha_hat": float(alpha_hat), "gamma_hat": float(gamma_hat),
                "asr_25_obs": asr_25, "asr_50_obs": asr_50, "asr_50_pred": asr_50_pred,
                "out_of_sample_error": err_25_50,
            })
        # Aggregate
        errs = [r["out_of_sample_error"] for r in cond_results["per_seed"]]
        if errs:
            mae = float(np.mean([abs(e) for e in errs]))
            cond_results["mae_out_of_sample"] = mae
            within_10 = sum(1 for e in errs if abs(e) <= 0.1)
            cond_results["fraction_within_0_10"] = f"{within_10}/{len(errs)}"
            print(f"  >>> MAE out-of-sample = {mae:.3f}  {within_10}/{len(errs)} seeds within +/-0.10")
        summary["conditions"][cond] = cond_results
        print()

    # Overall verdict
    all_errs = []
    for cond, info in summary["conditions"].items():
        all_errs.extend([abs(r["out_of_sample_error"]) for r in info["per_seed"]])
    overall_mae = float(np.mean(all_errs)) if all_errs else None
    overall_within_10 = sum(1 for e in all_errs if e <= 0.1) if all_errs else 0
    summary["overall_mae"] = overall_mae
    summary["overall_fraction_within_0_10"] = f"{overall_within_10}/{len(all_errs)}"
    print(f"\n=== OVERALL ===")
    print(f"  Overall MAE: {overall_mae:.3f}")
    print(f"  Fraction within +/-0.10: {overall_within_10}/{len(all_errs)}")
    if overall_mae is not None and overall_mae <= 0.10:
        verdict = "POSITIVE: admissibility-latent refinement gives quantitative prediction"
    elif overall_mae is not None and overall_mae <= 0.20:
        verdict = "PARTIAL: admissibility-latent reduces error but not within target"
    else:
        verdict = "NEGATIVE: admissibility-latent does not give quantitative prediction"
    summary["verdict"] = verdict
    print(f"  Verdict: {verdict}")

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
