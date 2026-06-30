"""
Compute the mixed-commitment Stackelberg equilibrium for the CIFAR-10 primary game.

Leader (server) publicly commits to σ_D = (FedAvg: p, NormClip: 1-p).
Follower (adversary) best-responds.
Server maximizes U_D(σ_A*(σ_D), σ_D) over p ∈ [0,1].

Uses the 10-seed averaged payoff matrices from the paper (Table A1, A2).

Output: results/stackelberg_analysis.json
"""
import sys, os, json
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

ATTACKS  = ["no_attack", "label_flip", "backdoor_pixel", "edge_case", "model_scaling", "dba"]
DEFENSES = ["fedavg", "krum", "multi_krum", "trimmed_mean", "coord_median", "norm_clip", "rfa"]

# 10-seed averaged adversary payoff matrix (Table A1 in paper)
adv_matrix = np.array([
    #  FA      Krum    M-Krum  TrMean  Median  NClip   RFA
    [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000],  # no_attack
    [-0.010, -0.010, -0.010, -0.010, -0.010, -0.010, -0.010],  # label_flip
    [ 0.805,  0.314,  0.803,  0.651,  0.449,  0.804,  0.834],  # backdoor_pixel
    [-0.015, -0.015, -0.015, -0.015, -0.015, -0.015, -0.015],  # edge_case
    [ 0.470,  0.023,  0.509,  0.850,  0.522,  0.922,  0.859],  # model_scaling
    [-0.001,  0.036, -0.002, -0.001, -0.002, -0.001, -0.001],  # dba
])

# 10-seed averaged server payoff matrix (Table A2 in paper)
srv_matrix = np.array([
    #  FA      Krum    M-Krum  TrMean  Median  NClip   RFA
    [ 0.774,  0.502,  0.770,  0.752,  0.743,  0.771,  0.754],  # no_attack
    [ 0.764,  0.486,  0.764,  0.746,  0.737,  0.761,  0.751],  # label_flip
    [ 0.766,  0.483,  0.760,  0.747,  0.737,  0.763,  0.748],  # backdoor_pixel
    [ 0.767,  0.486,  0.764,  0.747,  0.739,  0.764,  0.750],  # edge_case
    [ 0.095,  0.532,  0.107,  0.654,  0.734,  0.708,  0.741],  # model_scaling
    [ 0.768,  0.527,  0.759,  0.746,  0.736,  0.765,  0.749],  # dba
])

i_fa = DEFENSES.index("fedavg")
i_nc = DEFENSES.index("norm_clip")


# ── Mixed-Stackelberg on the {FedAvg, NormClip} 2-defense sub-game ────────────
ps = np.linspace(0, 1, 1001)
srv_utils = []
adv_utils = []
best_attacks = []

for p in ps:
    exp_adv = p * adv_matrix[:, i_fa] + (1 - p) * adv_matrix[:, i_nc]
    a_star_idx = int(np.argmax(exp_adv))
    adv_utils.append(float(exp_adv[a_star_idx]))
    best_attacks.append(ATTACKS[a_star_idx])
    srv_util = p * srv_matrix[a_star_idx, i_fa] + (1 - p) * srv_matrix[a_star_idx, i_nc]
    srv_utils.append(float(srv_util))

srv_utils = np.array(srv_utils)
adv_utils = np.array(adv_utils)
best_attacks = np.array(best_attacks)

p_star_idx = int(np.argmax(srv_utils))
p_star = float(ps[p_star_idx])
srv_star = float(srv_utils[p_star_idx])
adv_star = float(adv_utils[p_star_idx])
attack_star = best_attacks[p_star_idx]

print(f"\nMixed-Stackelberg (FedAvg/NormClip sub-game, server commits, adversary best-responds):")
print(f"  Optimal p(FedAvg) = {p_star:.3f}  [NormClip = {1-p_star:.3f}]")
print(f"  Server utility = {srv_star:.4f}  (Stackelberg)")
print(f"  Adversary utility = {adv_star:.4f}")
print(f"  Adversary best response = {attack_star}")
print(f"")
print(f"  NE3 Nash equilibrium: p(FedAvg)=0.26, U_D=0.763, U_A=0.804")
print(f"  Interpretation: Stackelberg optimum is pure FedAvg (=NE1, U_D=0.766).")
print(f"  NE3 has LOWER U_D (0.763) than Stackelberg (0.766) because NE3 is the")
print(f"  equilibrium of the SIMULTANEOUS (BNE) game, not the commitment game.")
print(f"  Private defense realizations make the two solutions differ.")

# Also show the adversary's equilibrium point — where it becomes indifferent
# between pixel and scaling under the FedAvg/NormClip mix
# pixel: p*0.805 + (1-p)*0.804 = 0.804 + 0.001*p
# scaling: p*0.470 + (1-p)*0.922 = 0.922 - 0.452*p
# indifference: 0.922 - 0.452*p = 0.804 + 0.001*p => p = (0.922-0.804)/(0.452+0.001)
p_indiff = (0.922 - 0.804) / (0.452 + 0.001)
print(f"\n  Adversary pixel/scaling indifference point: p(FedAvg) = {p_indiff:.3f}")
print(f"  (NE3 p=0.26 is near this indifference point → both pixel and scaling in adversary support)")

print("\nSelected sweep points:")
for p_check in [0.00, 0.10, 0.20, 0.25, 0.26, 0.27, 0.30, 0.50, p_star, 1.00]:
    idx = int(np.argmin(np.abs(ps - p_check)))
    print(f"  p={ps[idx]:.2f}: U_D={srv_utils[idx]:.4f}, U_A={adv_utils[idx]:.4f}, a*={best_attacks[idx]}")


# ── Save ────────────────────────────────────────────────────────────────────────
out = {
    "game": "FedAvg/NormClip 2-defense sub-game, 10-seed averaged matrix",
    "ne3_nash": {"p_fedavg": 0.26, "server_utility": 0.763, "adversary_utility": 0.804,
                 "note": "NE3 from Nash equilibrium of full 6x7 game"},
    "adversary_indifference_p": float(p_indiff),
    "stackelberg_optimum": {
        "p_fedavg": p_star,
        "p_norm_clip": 1 - p_star,
        "server_utility": srv_star,
        "adversary_utility": adv_star,
        "adversary_best_response": attack_star,
    },
    "sweep": [
        {"p_fedavg": float(p), "server_utility": float(su),
         "adversary_utility": float(au), "adversary_br": str(ba)}
        for p, su, au, ba in zip(ps[::50], srv_utils[::50], adv_utils[::50], best_attacks[::50])
    ]
}
out_path = os.path.join(base_dir, "results", "stackelberg_analysis.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {out_path}")
