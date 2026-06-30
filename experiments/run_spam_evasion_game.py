"""
Non-FL validation: adversarial spam evasion game.

Demonstrates that defense complementarity → VoPD > 0 holds outside federated learning.

Setting: binary spam classifier (UCI Spambase, 57 features = word freq + char freq + capital).
The attacker manipulates test-time spam emails to evade detection.
The defender privately samples an aggregation strategy from a mixed distribution.

Attacks (4):
  no_attack      — no modification
  word_zero      — zero out top-10 word-frequency features (cols 0-47)
  char_zero      — zero out all 6 character-frequency features (cols 48-53)
  noise_inject   — add Gaussian noise (σ=0.5, clipped to feature range)

Defenses (4):
  word_lr        — Logistic Regression on word features only (cols 0-47)
  char_lr        — Logistic Regression on char features only (cols 48-53)
  all_lr         — Logistic Regression on all 57 features
  all_rf         — Random Forest on all 57 features

Payoffs (matching FL paper conventions):
  U_A(a, d) = ASR(a,d) - λ_a · cost(a)     [ASR = fraction of spam misclassified as benign]
  U_D(a, d) = acc(a,d)  - λ_c · defense_cost(d)

Expected complementarity:
  argmax_A U_A(·, word_lr) = word_zero  (word features absent → word_lr blind)
  argmax_A U_A(·, char_lr) = char_zero  (char features absent → char_lr blind)
  → empty intersection → VoPD > 0 by Theorem 1

Output:
  results/spam_evasion/game_results.json
"""
import sys, os, json, warnings
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nashpy as nash

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(base_dir, "results", "spam_evasion")
os.makedirs(output_dir, exist_ok=True)

# ── constants ────────────────────────────────────────────────────────────────
LAMBDA_A = 0.1
LAMBDA_C = 0.05
ATTACK_COSTS   = {"no_attack": 0.0, "word_zero": 0.05, "char_zero": 0.05, "noise_inject": 0.1}
DEFENSE_COSTS  = {"word_lr": 0.0, "char_lr": 0.0, "all_lr": 0.02, "all_rf": 0.05}
ATTACKS  = ["no_attack", "word_zero", "char_zero", "noise_inject"]
DEFENSES = ["word_lr", "char_lr", "all_lr", "all_rf"]
SEEDS    = [42, 43, 44, 45, 46]

# Feature index groups
WORD_IDX  = list(range(0, 48))    # word_freq_* (48 features)
CHAR_IDX  = list(range(48, 54))   # char_freq_* (6 features)
ALL_IDX   = list(range(0, 57))

# ── load data ────────────────────────────────────────────────────────────────
print("Loading UCI Spambase...")
dataset = fetch_openml("spambase", version=1, as_frame=False, parser="auto")
X_full = dataset.data.astype(float)
y_full = (dataset.target == "1").astype(int)
print(f"  {X_full.shape[0]} emails, {X_full.shape[1]} features, "
      f"{y_full.sum()} spam ({100*y_full.mean():.1f}%)")

# Feature ranges for clipping noise attack
feat_min = X_full.min(axis=0)
feat_max = X_full.max(axis=0)

# Top-10 word features by mean value (high-frequency words in spam)
top_word_idx = np.argsort(X_full[:, WORD_IDX].mean(axis=0))[-10:]  # indices within WORD_IDX
TOP_WORD_FULL_IDX = [WORD_IDX[i] for i in top_word_idx]


# ── attack functions (operate on X_spam: rows of spam test emails) ───────────
def apply_attack(X_spam, attack, rng):
    X = X_spam.copy()
    if attack == "no_attack":
        pass
    elif attack == "word_zero":
        X[:, TOP_WORD_FULL_IDX] = 0.0
    elif attack == "char_zero":
        X[:, CHAR_IDX] = 0.0
    elif attack == "noise_inject":
        noise = rng.normal(0, 0.5, X.shape)
        X = np.clip(X + noise, feat_min, feat_max)
    return X


# ── defense constructors ─────────────────────────────────────────────────────
def build_defense(defense):
    if defense == "word_lr":
        return WORD_IDX, Pipeline([("scaler", StandardScaler()),
                                   ("clf", LogisticRegression(max_iter=500, C=1.0))])
    elif defense == "char_lr":
        return CHAR_IDX, Pipeline([("scaler", StandardScaler()),
                                   ("clf", LogisticRegression(max_iter=500, C=1.0))])
    elif defense == "all_lr":
        return ALL_IDX, Pipeline([("scaler", StandardScaler()),
                                  ("clf", LogisticRegression(max_iter=500, C=1.0))])
    elif defense == "all_rf":
        return ALL_IDX, Pipeline([("clf", RandomForestClassifier(n_estimators=50, random_state=0))])


# ── payoff computation ───────────────────────────────────────────────────────
print("Computing payoff matrix across seeds...")
all_results = {(a, d): [] for a in ATTACKS for d in DEFENSES}

for seed in SEEDS:
    rng = np.random.default_rng(seed)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_full, y_full, test_size=0.3, random_state=seed, stratify=y_full
    )
    spam_mask = y_te == 1
    X_te_spam = X_te[spam_mask]

    for defense in DEFENSES:
        feat_idx, clf = build_defense(defense)
        clf.fit(X_tr[:, feat_idx], y_tr)

        # Baseline accuracy (no attack)
        acc_base = clf.score(X_te[:, feat_idx], y_te)

        for attack in ATTACKS:
            # Apply attack to spam emails only; benign emails unchanged
            X_te_attacked = X_te.copy()
            X_te_attacked[spam_mask] = apply_attack(X_te_spam, attack, rng)

            acc = clf.score(X_te_attacked[:, feat_idx], y_te)
            # ASR = fraction of attacked spam misclassified as benign
            pred_spam = clf.predict(X_te_attacked[spam_mask][:, feat_idx])
            asr = float((pred_spam == 0).mean())

            all_results[(attack, defense)].append({
                "seed": seed, "accuracy": float(acc),
                "attack_success_rate": asr, "baseline_accuracy": float(acc_base),
            })

    print(f"  seed {seed} done")

# ── build payoff matrices ─────────────────────────────────────────────────────
adv_matrix = np.zeros((len(ATTACKS), len(DEFENSES)))
srv_matrix = np.zeros((len(ATTACKS), len(DEFENSES)))

for i, a in enumerate(ATTACKS):
    for j, d in enumerate(DEFENSES):
        entries = all_results[(a, d)]
        mean_asr = np.mean([e["attack_success_rate"] for e in entries])
        mean_acc = np.mean([e["accuracy"] for e in entries])
        adv_matrix[i, j] = mean_asr - LAMBDA_A * ATTACK_COSTS[a]
        srv_matrix[i, j]  = mean_acc - LAMBDA_C * DEFENSE_COSTS[d]

print("\nAdversary payoff matrix (U_A):")
print(f"{'':>14}", "  ".join(f"{d:>8}" for d in DEFENSES))
for i, a in enumerate(ATTACKS):
    print(f"{a:>14}", "  ".join(f"{adv_matrix[i,j]:>8.3f}" for j in range(len(DEFENSES))))

print("\nServer payoff matrix (U_D):")
print(f"{'':>14}", "  ".join(f"{d:>8}" for d in DEFENSES))
for i, a in enumerate(ATTACKS):
    print(f"{a:>14}", "  ".join(f"{srv_matrix[i,j]:>8.3f}" for j in range(len(DEFENSES))))

# ── Nash equilibrium + VoPD ───────────────────────────────────────────────────
print("\nSolving Nash equilibria...")
game = nash.Game(adv_matrix, srv_matrix)
nes = list(game.support_enumeration())

results_ne = []
best_vopd, best_mixed = 0.0, False
for sa, sd in nes:
    if len(sa) != len(ATTACKS) or len(sd) != len(DEFENSES):
        continue
    au = float(sa @ adv_matrix @ sd)
    fi = float(sum(sd[j] * adv_matrix[:, j].max() for j in range(len(sd))))
    vopd = max(0.0, fi - au)
    mixed = (sa > 0.01).sum() > 1 or (sd > 0.01).sum() > 1
    adv_sup  = [ATTACKS[i]  for i, p in enumerate(sa) if p > 0.01]
    srv_sup  = [DEFENSES[j] for j, p in enumerate(sd) if p > 0.01]
    results_ne.append({"adv_support": adv_sup, "srv_support": srv_sup,
                        "vopd": round(vopd, 4), "mixed": bool(mixed),
                        "adv_utility": round(au, 4)})
    if vopd > best_vopd:
        best_vopd = vopd
        best_mixed = mixed
    print(f"  NE: {adv_sup} vs {srv_sup}, VoPD={vopd:.4f}, mixed={mixed}")

# ── Theorem 1 complementarity check ──────────────────────────────────────────
# Check argmax_a U_A(a, d) for each defense in the NE support
ne_srv_sup = results_ne[0]["srv_support"] if results_ne else DEFENSES[:2]
argmax_sets = {}
for d in DEFENSES:
    j = DEFENSES.index(d)
    best_a = ATTACKS[int(np.argmax(adv_matrix[:, j]))]
    argmax_sets[d] = best_a
    print(f"  argmax_A U_A(·, {d}) = {best_a}  (U_A={adv_matrix[np.argmax(adv_matrix[:,j]),j]:.3f})")

# Check intersection over server NE support
if results_ne:
    best_ne = max(results_ne, key=lambda x: x["vopd"])
    support_argmaxes = {argmax_sets[d] for d in best_ne["srv_support"]}
    complementarity = len(support_argmaxes) > 1
    print(f"\nServer NE support: {best_ne['srv_support']}")
    print(f"Argmax attacks over support: {support_argmaxes}")
    print(f"Defense complementarity (Thm 1): {complementarity} → VoPD {'> 0' if complementarity else '= 0'} (predicted)")
    print(f"Computed VoPD: {best_vopd:.4f} → {'✓ consistent' if (best_vopd>0) == complementarity else '✗ inconsistent'}")

# ── Save results ──────────────────────────────────────────────────────────────
out = {
    "domain": "adversarial_spam_evasion",
    "dataset": "UCI Spambase (4601 emails, 57 features)",
    "attacks": ATTACKS,
    "defenses": DEFENSES,
    "seeds": SEEDS,
    "lambda_a": LAMBDA_A,
    "lambda_c": LAMBDA_C,
    "attack_costs": ATTACK_COSTS,
    "defense_costs": DEFENSE_COSTS,
    "adversary_payoffs": adv_matrix.tolist(),
    "server_payoffs": srv_matrix.tolist(),
    "nash_equilibria": results_ne,
    "best_vopd": round(best_vopd, 4),
    "complementarity_check": {
        "argmax_per_defense": argmax_sets,
        "complementarity": bool(complementarity) if results_ne else None,
        "theorem1_correct": bool((best_vopd > 0) == complementarity) if results_ne else None,
    },
    "per_cell_results": {
        f"{a}_{d}": entries
        for (a, d), entries in all_results.items()
    },
}
out_path = os.path.join(output_dir, "game_results.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {out_path}")
