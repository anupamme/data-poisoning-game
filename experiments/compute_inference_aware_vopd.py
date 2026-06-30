"""
Inference-aware adversary toy model.

Model: adversary starts with prior μ₀ = server's NE3 equilibrium distribution
(FedAvg 26%, NormClip 74%). After observing k rounds of aggregated update norms,
they update to posterior μₖ via Bayesian Gaussian likelihood on the observed norm.

Computes effective VoPD(k) = E_{d~μₖ}[max_a U_A(a,d)] - U_A(σ_A*, μₖ)
where σ_A* = pure backdoor_pixel (NE3 adversary strategy).

Uses:
  - results/defense_inference/norm_traces.json (9 cached norm traces)
  - results/game_analysis.json (payoff matrix for utility lookups)

Output:
  - results/inference_aware/vopd_vs_rounds.json
  - paper/figures/inference_aware_vopd.pdf
"""
import json, os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm as scipy_norm

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
traces_path = os.path.join(base_dir, "results", "defense_inference", "norm_traces.json")
game_path = os.path.join(base_dir, "results", "game_analysis.json")
output_dir = os.path.join(base_dir, "results", "inference_aware")
fig_path = os.path.join(base_dir, "paper", "figures", "inference_aware_vopd.pdf")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.dirname(fig_path), exist_ok=True)

with open(traces_path) as f:
    traces = json.load(f)
with open(game_path) as f:
    game = json.load(f)

# --- Extract payoff utilities from game_analysis.json ---
pm = game["payoff_matrix"]
attacks = pm["attacks"]
defenses = pm["defenses"]
adv_payoffs = np.array(pm["adversary_payoffs"])  # shape (n_attacks, n_defenses)

# NE3: adversary plays backdoor_pixel (pure), server mixes FedAvg 26% + NormClip 74%
adv_idx_pixel = attacks.index("backdoor_pixel")
def_idx_fedavg = defenses.index("fedavg")
def_idx_normclip = defenses.index("norm_clip")

# NE3 server equilibrium distribution
server_ne3 = {d: 0.0 for d in defenses}
server_ne3["fedavg"] = 0.26
server_ne3["norm_clip"] = 0.74
sigma_D = np.array([server_ne3[d] for d in defenses])

# --- Gaussian likelihood parameters per defense (from norm traces) ---
# For each defense, compute mean and std of per-round norms across seeds
defense_norms = {d: [] for d in ["fedavg", "norm_clip", "rfa"]}
for t in traces:
    defense_norms[t["defense"]].extend(t["norms"])

norm_stats = {}
for d in ["fedavg", "norm_clip", "rfa"]:
    vals = defense_norms[d]
    norm_stats[d] = {"mu": float(np.mean(vals)), "sigma": float(np.std(vals))}

print("Per-defense norm statistics:")
for d, s in norm_stats.items():
    print(f"  {d}: mu={s['mu']:.4f}, sigma={s['sigma']:.4f}")

# --- Simulate adversary posterior update ---
# Use traces from all 9 runs (3 defenses × 3 seeds) as "observed" sequences
# Adversary prior: proportional to NE3 server mix (FedAvg 26%, NormClip 74%, RFA 0%)
# But since adversary doesn't know which defense is which per round, they update
# a belief over which defense is active given observed norms.

# Defenses in the server's support
active_defenses = ["fedavg", "norm_clip"]  # RFA has 0% in NE3 but include for completeness
all_obs_defenses = ["fedavg", "norm_clip", "rfa"]

# Prior: NE3 server distribution (restricted to active defenses, renormalized)
prior_weights = {"fedavg": 0.26, "norm_clip": 0.74, "rfa": 0.0}
# Renormalize to probability distribution over all 3 defenses
total = sum(prior_weights.values())
prior = np.array([prior_weights[d] / total for d in all_obs_defenses])

def compute_effective_vopd(mu_k, adv_payoffs, attacks, defenses, active_defenses):
    """
    Compute effective VoPD under posterior belief mu_k over active_defenses.
    mu_k is a dict {defense: probability}.
    """
    # Full-information term: E_{d~mu_k}[max_a U_A(a, d)]
    full_info = 0.0
    for i, d in enumerate(defenses):
        prob = mu_k.get(d, 0.0)
        if prob > 1e-8:
            full_info += prob * adv_payoffs[:, i].max()

    # Adversary best response under belief: max_a E_{d~mu_k}[U_A(a, d)]
    expected_payoff_per_attack = np.array([
        sum(mu_k.get(d, 0.0) * adv_payoffs[j, i]
            for i, d in enumerate(defenses))
        for j in range(len(attacks))
    ])
    br_utility = expected_payoff_per_attack.max()

    return max(0.0, full_info - br_utility)

def gaussian_log_likelihood(obs_norm, defense, norm_stats):
    """Log-likelihood of observing norm under a given defense."""
    mu = norm_stats[defense]["mu"]
    sigma = norm_stats[defense]["sigma"]
    return scipy_norm.logpdf(obs_norm, loc=mu, scale=sigma)

# --- Compute VoPD(k) averaged over all 9 observed traces ---
max_rounds = 50
vopd_by_round = []  # mean VoPD at each round k

for k in range(max_rounds + 1):
    vopds_this_round = []

    for trace in traces:
        observed_def = trace["defense"]
        observed_norms = trace["norms"][:k] if k > 0 else []

        # Compute posterior via Bayes' rule
        log_prior = np.log(prior + 1e-300)
        log_likelihood = np.zeros(len(all_obs_defenses))

        for round_norm in observed_norms:
            for j, d in enumerate(all_obs_defenses):
                log_likelihood[j] += gaussian_log_likelihood(round_norm, d, norm_stats)

        log_posterior = log_prior + log_likelihood
        # Numerically stable softmax
        log_posterior -= log_posterior.max()
        posterior = np.exp(log_posterior)
        posterior /= posterior.sum()

        mu_k = {d: posterior[j] for j, d in enumerate(all_obs_defenses)}

        # Compute effective VoPD under this posterior
        vopd = compute_effective_vopd(mu_k, adv_payoffs, attacks, defenses, all_obs_defenses)
        vopds_this_round.append(vopd)

    vopd_by_round.append({
        "k": k,
        "mean_vopd": float(np.mean(vopds_this_round)),
        "median_vopd": float(np.median(vopds_this_round)),
        "std_vopd": float(np.std(vopds_this_round)),
    })

print(f"\nVoPD vs rounds (mean over 9 traces):")
for r in vopd_by_round[::5]:
    print(f"  k={r['k']:2d}: mean_vopd={r['mean_vopd']:.4f}, median={r['median_vopd']:.4f}")

# Save results
out = {
    "model": "Bayesian Gaussian posterior on aggregated update norms",
    "prior": prior.tolist(),
    "defenses_modeled": all_obs_defenses,
    "norm_stats": norm_stats,
    "vopd_by_round": vopd_by_round,
    "baseline_vopd_ne3": 0.087,
}
out_path = os.path.join(output_dir, "vopd_vs_rounds.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {out_path}")

# --- Plot ---
plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 11, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

rounds = [r["k"] for r in vopd_by_round]
mean_vopd = [r["mean_vopd"] for r in vopd_by_round]
std_vopd = [r["std_vopd"] for r in vopd_by_round]
upper = [min(1.0, m + s) for m, s in zip(mean_vopd, std_vopd)]
lower = [max(0.0, m - s) for m, s in zip(mean_vopd, std_vopd)]

fig, ax = plt.subplots(figsize=(5.5, 3.2))
color_main = "#1565C0"
color_fill = "#90CAF9"

ax.plot(rounds, mean_vopd, "-", color=color_main, linewidth=2, label="Mean effective VoPD")
ax.fill_between(rounds, lower, upper, alpha=0.2, color=color_fill, label="±1 std (across traces)")
ax.axhline(0.087, linestyle="--", color="#d32f2f", linewidth=1.4, label="Static VoPD (no inference, NE3)")
ax.axhline(0, linestyle=":", color="gray", linewidth=1.0)

ax.set_xlabel("Rounds of norm observations ($k$)")
ax.set_ylabel("Effective VoPD")
ax.set_xlim(0, 50)
ax.set_ylim(-0.005, 0.12)
ax.legend(loc="upper right", framealpha=0.9)
ax.set_title("Inference-aware adversary: effective VoPD vs.\ observation rounds\n"
             "(Bayesian norm inference, FedAvg/NormClip indistinguishable)", fontsize=9)

plt.tight_layout()
plt.savefig(fig_path)
plt.close()
print(f"Saved figure: {fig_path}")
