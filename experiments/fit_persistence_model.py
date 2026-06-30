"""
Per-defense persistence model fit with out-of-sample validation.

Model (continuous, per-defense admission rates):
    s_{t+1} = s_t + alpha * p_{d_t} * (1 - s_t) - (1 - gamma) * (1 - p_{d_t}) * s_t

where d_t is the defense used at round t, and p_d is its empirical admission rate.

Free parameters: (gamma, alpha, p_FA, p_NC).

Validation: TEMPORAL SPLIT.
  - Fit on rounds 1..25 across all 3 conditions × 5 seeds.
  - Predict rounds 26..50 using each held-out trajectory's per-round defense schedule.
  - Report observed vs predicted ASR_50.

Output:
  results/persistence/fit_summary_v2.json
  results/persistence/fit_table_v2.tex
"""
import json
import os
import sys
import numpy as np
from scipy.optimize import minimize

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
persistence_dir = os.path.join(base_dir, "results", "persistence")

CONFIGS = ["fedavg_pure", "normclip_pure", "ne3_mix"]
SEEDS = [42, 43, 44, 45, 46]
SPLIT_ROUND = 25  # fit on rounds 1..SPLIT_ROUND, predict SPLIT_ROUND+1..50


def simulate_trajectory(gamma, alpha, p_d_per_round, T, s0=0.0):
    """Deterministic expected trajectory under per-round admission probabilities."""
    s = s0
    traj = []
    for t in range(T):
        p = p_d_per_round[t]
        s = s + alpha * p * (1 - s) - (1 - gamma) * (1 - p) * s
        s = max(0.0, min(1.0, s))
        traj.append(s)
    return np.array(traj)


def load_timeline(condition, seed):
    """Return (asr_array, defense_strings, schedule_binary_FA)."""
    path = os.path.join(persistence_dir, condition, f"seed_{seed}", "asr_timeline.json")
    if not os.path.exists(path):
        return None, None, None
    with open(path) as f:
        d = json.load(f)
    timeline = d["asr_timeline"]
    asrs = np.array([e["asr"] for e in timeline])
    defenses = [e["defense"] for e in timeline]
    return asrs, defenses, d


def params_to_per_round_p(params, defenses):
    """Map defense string per round to its admission probability from params."""
    gamma, alpha, p_FA, p_NC = params
    return np.array([p_FA if d == "fedavg" else p_NC for d in defenses])


def trajectory_squared_error(params, asr_obs, defenses, t_start=0, t_end=None):
    """Squared error on rounds [t_start, t_end)."""
    gamma, alpha, p_FA, p_NC = params
    if not (0 < gamma <= 1 and 0 < alpha <= 1 and 0 <= p_FA <= 1 and 0 <= p_NC <= 1):
        return 1e10
    if t_end is None:
        t_end = len(asr_obs)
    p_per_round = params_to_per_round_p(params, defenses)
    traj = simulate_trajectory(gamma, alpha, p_per_round, len(asr_obs))
    return float(np.sum((asr_obs[t_start:t_end] - traj[t_start:t_end]) ** 2))


def joint_fit(timelines, t_start, t_end, x0=None):
    """Fit (gamma, alpha, p_FA, p_NC) on mean trajectories per condition over rounds [t_start, t_end).

    `timelines` is a dict: condition -> list of (asr, defenses, seed).
    The deterministic per-defense model describes the EXPECTED trajectory across seeds,
    so we fit to the cross-seed mean ASR_t per condition (matching the model's semantics)."""
    if x0 is None:
        x0 = np.array([0.95, 0.3, 0.5, 0.9])

    # Build mean trajectory + canonical defense schedule per condition
    cond_mean = {}
    for cond, tlist in timelines.items():
        if not tlist:
            continue
        asr_stack = np.stack([asr for (asr, _, _) in tlist])
        mean_asr = asr_stack.mean(axis=0)
        # For pure conditions, defense is constant. For NE3, the schedule varies per seed
        # but the EXPECTED dynamics see admission with prob p_eff per round, regardless
        # of which seeds drew which schedule. So we use the FIRST seed's defenses as canonical:
        # this is exact for pure conditions and approximately correct for NE3 (since each seed
        # is an i.i.d. Bernoulli draw of the same Markov process).
        defenses = tlist[0][1]
        cond_mean[cond] = (mean_asr, defenses)

    def objective(params):
        gamma, alpha, p_FA, p_NC = params
        if not (0 < gamma <= 1 and 0 < alpha <= 1 and 0 <= p_FA <= 1 and 0 <= p_NC <= 1):
            return 1e10
        total = 0.0
        for cond, (mean_asr, defenses) in cond_mean.items():
            p_per_round = params_to_per_round_p(params, defenses)
            traj = simulate_trajectory(gamma, alpha, p_per_round, len(mean_asr))
            total += float(np.sum((mean_asr[t_start:t_end] - traj[t_start:t_end]) ** 2))
        return total

    res = minimize(objective, x0, method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 10000})
    return res.x, res.fun


def predict_held_out(params, asr_obs, defenses, hold_out_start):
    """Predict ASR over [hold_out_start, len(asr)) using params and the trajectory's own schedule."""
    p_per_round = params_to_per_round_p(params, defenses)
    traj = simulate_trajectory(params[0], params[1], p_per_round, len(asr_obs))
    return traj


# Load all timelines
all_timelines = {}  # condition -> list of (asr, defenses) per seed
for cond in CONFIGS:
    all_timelines[cond] = []
    for seed in SEEDS:
        asr, defenses, _ = load_timeline(cond, seed)
        if asr is not None:
            all_timelines[cond].append((asr, defenses, seed))

print(f"=== Joint per-defense fit on cross-seed mean trajectory (training: rounds 1..{SPLIT_ROUND}) ===\n")
params_fit, train_loss = joint_fit(all_timelines, 0, SPLIT_ROUND)
gamma_hat, alpha_hat, p_FA_hat, p_NC_hat = params_fit
print(f"Joint fit on rounds 1..{SPLIT_ROUND}:")
print(f"  gamma = {gamma_hat:.4f}")
print(f"  alpha = {alpha_hat:.4f}")
print(f"  p_FA  = {p_FA_hat:.4f}")
print(f"  p_NC  = {p_NC_hat:.4f}")
print(f"  train SSE = {train_loss:.4f}")

# Bootstrap CIs by resampling seeds WITHIN each condition (preserves cross-condition mean structure)
rng = np.random.default_rng(0)
boot_params = []
n_boot = 200
for _ in range(n_boot):
    boot_dict = {}
    for cond, tlist in all_timelines.items():
        if not tlist:
            continue
        idx = rng.integers(0, len(tlist), size=len(tlist))
        boot_dict[cond] = [tlist[i] for i in idx]
    try:
        p, _ = joint_fit(boot_dict, 0, SPLIT_ROUND, x0=params_fit)
        boot_params.append(p)
    except Exception:
        continue
boot_params = np.array(boot_params)
ci = lambda col: (float(np.percentile(col, 2.5)), float(np.percentile(col, 97.5)))
gamma_ci = ci(boot_params[:, 0])
alpha_ci = ci(boot_params[:, 1])
p_FA_ci = ci(boot_params[:, 2])
p_NC_ci = ci(boot_params[:, 3])
print(f"\nBootstrap 95% CIs (n={n_boot}):")
print(f"  gamma: [{gamma_ci[0]:.3f}, {gamma_ci[1]:.3f}]")
print(f"  alpha: [{alpha_ci[0]:.3f}, {alpha_ci[1]:.3f}]")
print(f"  p_FA:  [{p_FA_ci[0]:.3f}, {p_FA_ci[1]:.3f}]")
print(f"  p_NC:  [{p_NC_ci[0]:.3f}, {p_NC_ci[1]:.3f}]")

# Out-of-sample predictions on rounds SPLIT_ROUND..50
print(f"\n=== Out-of-sample predictions (rounds {SPLIT_ROUND+1}..50) ===\n")
per_cond_results = {}
for cond in CONFIGS:
    obs_50_list = []
    pred_50_list = []
    held_out_rmse_list = []
    obs_at_split_list = []
    pred_at_split_list = []
    for (asr, defenses, seed) in all_timelines[cond]:
        # Use observed state at round SPLIT_ROUND as the prediction start; simulate forward using model
        # But for honest test: predict the entire 1..50 from s0=0 using only model + schedule
        p_per_round = params_to_per_round_p(params_fit, defenses)
        traj = simulate_trajectory(gamma_hat, alpha_hat, p_per_round, len(asr))
        held_out_resid = asr[SPLIT_ROUND:] - traj[SPLIT_ROUND:]
        held_out_rmse = float(np.sqrt(np.mean(held_out_resid ** 2)))
        obs_50_list.append(float(asr[-1]))
        pred_50_list.append(float(traj[-1]))
        held_out_rmse_list.append(held_out_rmse)
        obs_at_split_list.append(float(asr[SPLIT_ROUND - 1]))
        pred_at_split_list.append(float(traj[SPLIT_ROUND - 1]))
    per_cond_results[cond] = {
        "obs_ASR_50_mean": float(np.mean(obs_50_list)),
        "obs_ASR_50_std": float(np.std(obs_50_list)),
        "pred_ASR_50_mean": float(np.mean(pred_50_list)),
        "pred_ASR_50_std": float(np.std(pred_50_list)),
        "held_out_rmse_mean": float(np.mean(held_out_rmse_list)),
        "held_out_rmse_std": float(np.std(held_out_rmse_list)),
        "per_seed_obs_50": obs_50_list,
        "per_seed_pred_50": pred_50_list,
    }
    print(f"{cond:>15}: obs ASR_50 = {np.mean(obs_50_list):.3f}±{np.std(obs_50_list):.3f}, "
          f"pred ASR_50 = {np.mean(pred_50_list):.3f}±{np.std(pred_50_list):.3f}, "
          f"held-out RMSE = {np.mean(held_out_rmse_list):.3f}±{np.std(held_out_rmse_list):.3f}")

# Pre-committed acceptance criterion
print("\n=== Pre-committed acceptance criterion ===")
ne3 = per_cond_results["ne3_mix"]
abs_err_ne3 = abs(ne3["pred_ASR_50_mean"] - ne3["obs_ASR_50_mean"])
two_sigma_ne3 = 2 * ne3["obs_ASR_50_std"]
crit1 = abs_err_ne3 < max(two_sigma_ne3, 0.082)
crit2 = 0.3 <= p_FA_hat <= 1.0 and 0.3 <= p_NC_hat <= 1.0
crit3 = max(per_cond_results[c]["held_out_rmse_mean"] for c in CONFIGS) <= 0.15
print(f"  (1) |pred NE3 - obs NE3| = {abs_err_ne3:.4f} < {max(two_sigma_ne3, 0.082):.4f} : {crit1}")
print(f"  (2) p_FA, p_NC in [0.3, 1.0]: p_FA={p_FA_hat:.3f}, p_NC={p_NC_hat:.3f} : {crit2}")
print(f"  (3) max held-out RMSE = {max(per_cond_results[c]['held_out_rmse_mean'] for c in CONFIGS):.4f} <= 0.15 : {crit3}")
overall = crit1 and crit2 and crit3
print(f"  REPAIR VERDICT: {'PASS' if overall else 'FAIL'}")

# Persistence horizon T_gamma
T_gamma = float(np.log(1 / 0.05) / np.log(1 / gamma_hat)) if gamma_hat < 1.0 else float("inf")
print(f"\n  At gamma_hat = {gamma_hat:.4f}, persistence horizon T_gamma = {T_gamma:.1f} rounds")
print(f"  Deployment T = 50 rounds. T_gamma > T = {T_gamma > 50}")

summary = {
    "split_round": SPLIT_ROUND,
    "gamma": float(gamma_hat),
    "alpha": float(alpha_hat),
    "p_FA": float(p_FA_hat),
    "p_NC": float(p_NC_hat),
    "gamma_ci": list(gamma_ci),
    "alpha_ci": list(alpha_ci),
    "p_FA_ci": list(p_FA_ci),
    "p_NC_ci": list(p_NC_ci),
    "T_gamma": T_gamma,
    "per_condition": per_cond_results,
    "criterion_pass": overall,
    "criterion_details": {
        "ne3_abs_err": float(abs_err_ne3),
        "ne3_tol": float(max(two_sigma_ne3, 0.082)),
        "p_FA": float(p_FA_hat),
        "p_NC": float(p_NC_hat),
        "max_held_out_rmse": float(max(per_cond_results[c]["held_out_rmse_mean"] for c in CONFIGS)),
    }
}
with open(os.path.join(persistence_dir, "fit_summary_v2.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {persistence_dir}/fit_summary_v2.json")

# Build LaTeX table
def fmt_ci(mean, ci_low, ci_high, digits=3):
    return f"${mean:.{digits}f}_{{[{ci_low:.{digits}f},{ci_high:.{digits}f}]}}$"

rows = []
for cond, label in [
    ("fedavg_pure", "Pure FedAvg"),
    ("normclip_pure", "Pure NormClip"),
    ("ne3_mix", "NE3 mix"),
]:
    r = per_cond_results[cond]
    row = (
        f"{label} & "
        f"${r['pred_ASR_50_mean']:.3f}\\pm{r['pred_ASR_50_std']:.3f}$ & "
        f"${r['obs_ASR_50_mean']:.3f}\\pm{r['obs_ASR_50_std']:.3f}$ & "
        f"${r['held_out_rmse_mean']:.3f}$ \\\\"
    )
    rows.append(row)

def ci_str(mean, lo, hi):
    return f"{mean:.3f}_{{[{lo:.3f},{hi:.3f}]}}"

joint_params_str = (
    f"$\\hat\\gamma = {ci_str(gamma_hat, gamma_ci[0], gamma_ci[1])}$, "
    f"$\\hat\\alpha = {ci_str(alpha_hat, alpha_ci[0], alpha_ci[1])}$, "
    f"$\\hat p_{{\\mathrm{{FA}}}} = {ci_str(p_FA_hat, p_FA_ci[0], p_FA_ci[1])}$, "
    f"$\\hat p_{{\\mathrm{{NC}}}} = {ci_str(p_NC_hat, p_NC_ci[0], p_NC_ci[1])}$"
)

table_top = r"""\begin{table}[h]
\caption{Theorem~\ref{thm:persistence} per-defense fit and \textbf{out-of-sample validation}. Joint MLE of $(\hat\gamma, \hat\alpha, \hat p_{\mathrm{FA}}, \hat p_{\mathrm{NC}})$ on \emph{rounds 1--25} only, across all 3 conditions $\times$ 5 seeds. Predicted ASR$_{50}$ uses the fitted parameters and each held-out trajectory's per-round defense schedule to simulate rounds 26--50; observed is the actual realized ASR. Held-out RMSE is computed on rounds 26--50 (never seen by the fit).}
\label{tab:persistence_fit}
\centering\small
\begin{tabular}{lccc}
\toprule
Condition & Pred ASR$_{50}$ (out-of-sample) & Obs ASR$_{50}$ & Held-out RMSE \\
\midrule
"""
table_bottom = r"""
\bottomrule
\end{tabular}

\vspace{0.3em}
\small
Joint fit on rounds 1--25: """ + joint_params_str + r""" (95\% bootstrap CIs over seeds).
\end{table}"""

latex = table_top + "\n".join(rows) + table_bottom

with open(os.path.join(persistence_dir, "fit_table_v2.tex"), "w") as f:
    f.write(latex)
print(f"Saved: {persistence_dir}/fit_table_v2.tex")
print("\nLaTeX table:")
print(latex)
