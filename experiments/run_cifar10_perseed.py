"""
Re-run CIFAR-10 payoff matrix WITH per-seed storage to get CIs on VoPD.
Same config as the original CIFAR-10 run (results/payoff_results.json).
Output: results/cifar10_perseed/
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FLConfig, ExperimentConfig, GameConfig
from experiments.run_payoff_matrix import run_full_payoff_matrix
from experiments.run_game_analysis import run_analysis

fl_config = FLConfig(
    num_clients=10,
    clients_per_round=5,
    num_rounds=50,
)

exp_config = ExperimentConfig(
    dataset="cifar10",
    model="cifar_cnn",
    dirichlet_alpha=0.5,
    adversarial_fraction=0.2,
    num_trials=3,
    seed=42,
    device="mps",
)

game_config = GameConfig()

output_dir = "/Users/mediratta/code/paper_writing/AI-Researcher/code/data_poisoning_game/results/cifar10_perseed"
os.makedirs(output_dir, exist_ok=True)

print("Starting CIFAR-10 per-seed re-run (6x7, 50 rounds, 3 seeds)...")
run_full_payoff_matrix(fl_config, exp_config, game_config, output_dir)

print("\nRunning game analysis on CIFAR-10 results...")
results = run_analysis(
    os.path.join(output_dir, "payoff_results.json"),
    output_dir,
    game_config=game_config,
)

print("\n=== CIFAR-10 (per-seed) RESULTS ===")
for i, ne in enumerate(results["nash_equilibria"]):
    vopd = ne["value_of_information"]
    ua = ne["adversary_utility"]
    us = ne["server_utility"]
    adv_s = [a for a, p in zip(results["payoff_matrix"]["attacks"], ne["adversary_strategy"]) if p > 0.01]
    srv_s = [d for d, p in zip(results["payoff_matrix"]["defenses"], ne["server_strategy"]) if p > 0.01]
    print(f"NE{i+1}: U_A={ua:.4f}, U_D={us:.4f}, VoPD={vopd:.4f}")
    print(f"  Adversary: {adv_s}")
    print(f"  Server:    {srv_s}")
