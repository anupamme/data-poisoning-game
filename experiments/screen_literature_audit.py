"""
Title/abstract screening for the pre-registered literature audit. Emits results/literature_audit/
screening.json: one decision, with a reason, for every deduplicated candidate.

WHY THIS IS A FILE AND NOT A COMMAND
The screening decision is the step with the most discretion in the whole audit, so it ships as source.
The mechanical rules below are auditable by reading them; the ADJUDICATED table is the human
judgements, one line each, and a reader who disagrees can flip a line and rerun the analyzer.

THE INCLUDE RULE, VERBATIM FROM THE PRE-REGISTRATION
    "Include at screening iff the paper proposes or evaluates a defense PIPELINE in the C-a sense.
     Exclude single-mechanism defenses, attack-only papers, surveys, position papers, and work
     outside FL."
and C-a is "Two or more distinct defense mechanisms are applied together to the same training round
(sequential pipeline, filter-then-aggregate, ensemble, or clip-then-noise)."

TWO READINGS OF C-a HAD TO BE SETTLED, AND BOTH ARE SETTLED AGAINST BREADTH
  1. BOTH composed components must themselves be DEFENSE mechanisms. "Detect malicious updates, then
     FedAvg the survivors" is ONE mechanism, because plain averaging is not a defense; "detect, then
     take the coordinate-wise median" is two. Without this line, C-a's own example phrase
     "filter-then-aggregate" would swallow essentially every detection paper ever written and the
     audit would measure nothing.
  2. "to the same training round" excludes POST-TRAINING repair. A pipeline that converges first and
     then unlearns or prunes a backdoor out of the finished model is not two mechanisms applied to a
     round, however sequential it is.
Both readings SHRINK the included set. They are recorded here, before coding, because deciding them
per paper is exactly what the pre-registration exists to prevent.

BORDERLINE CASES ARE INCLUDED, NOT EXCLUDED, AND THAT IS THE SAFE DIRECTION
Where an abstract plausibly shows two mechanisms but does not settle it, the paper is INCLUDED and the
question is pushed to coding, where C-a can be coded NO or UNCLEAR against full text. A wrongly
included paper codes C-a = NO and lowers the primary rate; a wrongly excluded one is invisible. So the
conservative direction at screening is to include, and that is what is done.

Reads:  results/literature_audit/candidates_raw.json
Writes: results/literature_audit/screening.json
Run:    python3 experiments/screen_literature_audit.py
"""
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "results", "literature_audit")

FL = re.compile(r"federat|decentraliz(ed)? learning|collaborative learning", re.I)
SURVEY = re.compile(r"\bsurvey\b|\breview\b|\bSoK\b|position paper|systematic literature", re.I)
# SURVEY above is applied to the TITLE only, and that let 2504.01240 -- whose abstract opens "In this
# survey, we investigate the most recent techniques of resilient federated learning" -- through to the
# single-mechanism branch, where it was excluded for the WRONG reason. The reason strings ship in the
# appendix (non-negotiable 6), so a wrong one is a defect in the artifact even when the decision is
# right. This pattern is deliberately narrower than SURVEY: "we review the literature" appears in
# ordinary papers' abstracts, so matching a bare "review" there would exclude primary work.
SURVEY_AB = re.compile(r"\b(in this|this) (survey|review)\b|\bwe survey\b|"
                       r"\bthis (paper|article) (surveys|reviews)\b|\bsystematic(ally)? review", re.I)

# ---------------------------------------------------------------------------------------------
# MECHANISM FAMILIES. Computed here rather than read from a side file, because the first version of
# this screen read a hand-built results/literature_audit/screen_pool.json whose taxonomy existed in
# no source file -- so the family counts could not be regenerated, and they went stale the moment
# title resolution added abstracts to 42 rows. Everything the screen uses is now derived from
# candidates_raw.json by the code below.
#
# The family count does NOT decide inclusion; ADJUDICATED does. Its only jobs are (a) to name, in an
# exclusion reason, what the abstract actually showed, and (b) to FLAG rows that name mechanisms from
# two or more families, which is where a missed pipeline would hide. Rows with nfam >= 2 are the
# review queue for adjudication; --flag prints them.
# ---------------------------------------------------------------------------------------------
FAMILIES = {
    "clustering": r"cluster|HDBSCAN|DBSCAN|OPTICS|K-?means|spectral|community detection",
    "similarity-filter": r"cosine (similarit|distance)|pairwise distance|angular|Euclidean distance|"
                         r"similarity[- ]based|Jaccard|correlation[- ]based",
    "clipping": r"clipping|clip the|norm[- ]bound|bounded norm|magnitude bound",
    "noise": r"differential privacy|\bDP\b|Gaussian noise|noise injection|noising|randomized "
             r"smoothing|perturbation noise",
    "robust-agg": r"trimmed mean|coordinate[- ]wise median|geometric median|\bmedian\b|"
                  r"Byzantine[- ]robust aggregat|robust aggregation rule|Bulyan|RFA",
    "distance-select": r"\bKrum\b|multi[- ]Krum|nearest[- ]neighbou?r selection|"
                       r"select(ing|s)? the .{0,20}closest",
    "reputation": r"reputation|trust score|trust[- ]based|credibilit|historical behavio|"
                  r"contribution (score|evaluation)|Shapley",
    "pruning": r"prun|neuron (analysis|masking)|mask(ing)? (out )?neuron|weight (masking|removal)",
    "distillation": r"distillat|teacher[- ]student|ensemble transfer",
    "secure-agg": r"secure aggregation|homomorphic|\bHE\b|\bMPC\b|secret shar|"
                  r"multi[- ]party computation|zero[- ]knowledge",
    "sparsification": r"sparsif|sparse update|top-?k|quantiz|sketch|compress",
    "validation": r"root dataset|server[- ]side validation|clean holdout|validation (set|accuracy)|"
                  r"reference (model|update)|bootstrap trust",
    "unlearning": r"unlearn|machine unlearning|model repair|recovery from poisoning",
}
FAMILIES = {k: re.compile(v, re.I) for k, v in FAMILIES.items()}

# Joint-application language. Its absence is what turns a two-family abstract into an exclusion: a
# paper that merely NAMES clipping and median in its related work has two families and no pipeline.
JOINT = re.compile(r"combin|integrat|two[- ]stage|three[- ]stage|multi[- ]stage|two[- ]phase|"
                   r"pipeline|jointly|in tandem|followed by|consists? of|comprises?|"
                   r"together with|coupled with|synerg|then (appl|aggregat|cluster|clip)|"
                   r"both .{0,30}and|first .{0,40}(then|second)|"
                   # Added after a second probe over rows with three or more families and NO joint verb
                   # found SpectralKrum, whose abstract says it "fuses spectral subspace estimation
                   # with geometric neighbor-based selection" -- a pipeline the list above missed
                   # entirely. A recall probe that finds something means the pattern, not the paper,
                   # was wrong.
                   r"fuse[sd]?\b|stack(s|ed|ing)?\b|on top of|in conjunction|hybrid|"
                   r"augment(s|ed)? .{0,25}with|enhance[sd]? .{0,30}(with|by adding)", re.I)


def families_of(c):
    """Mechanism families named anywhere in the title or abstract."""
    s = (c.get("title") or "") + ". " + (c.get("abstract") or "")
    return sorted(k for k, rx in FAMILIES.items() if rx.search(s))

# ---------------------------------------------------------------------------------------------
# ADJUDICATED. arXiv id -> (decision, reason). The reason names the mechanisms the abstract shows
# joined, or names the single mechanism / the attack focus that excludes it. Every reason is checkable
# against the abstract stored in candidates_raw.json.
# ---------------------------------------------------------------------------------------------
ADJUDICATED = {
    # ---- INCLUDE: two or more defense mechanisms applied to the same round -------------------
    "2101.02281": ("include", "pipeline: model clustering, weight clipping and noise injection "
                              "(FLAME)"),
    "2201.00763": ("include", "pipeline: model filtering, update clustering and clipping (DeepSight)"),
    "2509.08089": ("include", "pipeline: principled combination of Type 1 and Type 2 defenses "
                              "(Hammer and Anvil)"),
    "2307.00356": ("include", "pipeline: amplified-magnitude sparsification, OPTICS clustering and "
                              "adaptive clipping (Fedward)"),
    "2509.18044": ("include", "pipeline: geometric anomaly detection combined with momentum-based "
                              "reputation tracking (HRA)"),
    "2507.16134": ("include", "pipeline: gradient masking, SVD/cosine feature extraction, clustering "
                              "and trust-score adaptive aggregation (DP2Guard)"),
    "2505.01454": ("include", "pipeline: Jaccard structure-aware filtering plus density-based "
                              "directional clustering (SafeSparse)"),
    "2505.12851": ("include", "pipeline: ReLU-clipped cosine filtering, reference selection, angular "
                              "weighting and magnitude normalization (FLTG)"),
    "2208.10161": ("include", "pipeline: DBSCAN clustering on adjusted cosine similarity plus model "
                              "segmentation (MUDGUARD)"),
    "2210.07714": ("include", "pipeline: client feedback, hidden-layer neuron analysis, iterative "
                              "pruning and stacked clustering (CrowdGuard)"),
    "2605.22506": ("include", "pipeline: density-based low-dimensional clustering plus a pseudo-"
                              "gradient generator (EnCAgg)"),
    "2210.01437": ("include", "pipeline: similarity-based filtering against a historical estimator, "
                              "then clustering (Robust-FL)"),
    "2409.17754": ("include", "pipeline: multiple filters combined in one aggregation rule (WFAgg)"),
    "2402.04409": ("include", "pipeline: FedTruth ground-truth estimation plus a Byzantine-resilient "
                              "aggregation rule (FRECA)"),
    "2206.12100": ("include", "pipeline: randomized clustering plus rank-based robustness checks "
                              "(zPROBE)"),
    "1909.05125": ("include", "pipeline: HMM update-quality model, robust aggregation rule and "
                              "participant blocking (Adaptive Federated Averaging)"),
    "2502.00587": ("include", "pipeline: clustering and model selection, then knowledge distillation "
                              "from the retained ensemble (RKD)"),
    "2207.01982": ("include", "pipeline: clustering-based detection plus pruning against label "
                              "flipping"),
    "2606.31066": ("include", "pipeline: cascaded statistical filter early, CHG-Shapley contribution "
                              "verification late (Secure-CHG)"),
    "2608.21172": ("include", "pipeline: norm filtering, mask-aware directional validation and "
                              "adaptive coordinate clipping (Thermo-FL / TERRA)"),
    "2505.10297": ("include", "pipeline: two complementary detection mechanisms, consistency analysis "
                              "and attention-based analysis"),
    "2409.01435": ("include", "pipeline: pre-aggregation sparsification plus layer-wise adaptive "
                              "robust aggregation (LASA)"),
    "2605.11122": ("include", "pipeline: bidirectional gradient alignment filtering plus layer-"
                              "adaptive anomaly detection (FedSurrogate)"),
    "2502.07011": ("include", "pipeline: clustering and activity tracking plus knowledge distillation "
                              "(DROP)"),
    "2306.04984": ("include", "pipeline: attributed client graph clustering plus adaptive discrepancy "
                              "amplification (G2uardFL)"),
    "2608.14861": ("include", "pipeline: spatial-temporal analysis plus robust aggregation, described "
                              "as synergistic (STAR-FL)"),
    "2508.10315": ("include", "pipeline: pre-aggregation and post-aggregation defense strategies "
                              "integrated"),
    "2603.28652": ("include", "pipeline: reputation system, incentive mechanism and a game-theoretic "
                              "component"),
    "2601.06466": ("include", "pipeline: temporal gradient auditing (GMM + Mahalanobis) plus further "
                              "declared defensive components (SecureDyn-FL)"),
    "2603.04422": ("include", "pipeline: EMA temporal smoothing of the global model plus ensemble "
                              "knowledge distillation (FedEMA-Distill)"),
    "2602.16480": ("include", "pipeline: privacy-preserving technique combined with a defensive "
                              "aggregation strategy (SRFed)"),
    "2601.01053": ("include", "pipeline: adaptive weighted aggregation plus lattice-based secure "
                              "aggregation"),
    "2604.03862": ("include", "pipeline: update estimation plus coordinate-wise median aggregation "
                              "(SecureAFL)"),
    "2502.05547": ("include", "pipeline: secure aggregation combined with anomaly detection over "
                              "encrypted updates (DDFed)"),
    "2607.06612": ("include", "pipeline: multi-key FHE secure aggregation plus Byzantine-robust "
                              "aggregation (PRoVeFL)"),
    "2602.22269": ("include", "pipeline: clustered quantum secure aggregation plus server-side "
                              "Byzantine detection (CQSA)"),
    "2609.03420": ("include", "EVALUATES a composed pipeline: differential privacy together with "
                              "Byzantine-robust aggregation, and challenges their composability"),
    "2608.08574": ("include", "pipeline: reputation model composed with existing detect-and-filter "
                              "defenses"),
    "2511.09294": ("include", "borderline, included per the borderline rule: a dual-facet attack plus "
                              "a defense whose component count the abstract does not settle "
                              "(GuardFed)"),
    "2411.01040": ("include", "borderline, included per the borderline rule: individual-unlearning "
                              "detection plus pre-unlearning model fusion (MASA)"),

    # ---- INCLUDE, second pass: the --flag review queue -----------------------------------------
    # The first pass of this table was assembled from the mechanism-family screen and MISSED a whole
    # class: papers that compose a PRIVACY mechanism (DP, secure aggregation, compression) with a
    # ROBUSTNESS mechanism. That is C-a's own "clip-then-noise" example, and the first pass had already
    # admitted DDFed, PRoVeFL, CQSA, SRFed, 2601.01053 and 2609.03420 on exactly that shape -- so
    # excluding the rest of the class would have been an inconsistency, not a criterion. The rows below
    # come from `--flag`: every unadjudicated row showing two or more mechanism families AND joint-
    # application language. All 30 were read; these are the 18 that are pipelines.
    #
    # This class mostly codes C-c = NO, because a privacy-plus-robustness paper reports a convergence
    # bound or an accuracy-under-attack curve rather than the composed system's own ASR. Admitting it
    # therefore ENLARGES the denominator more than the numerator and pushes the primary rate DOWN,
    # which is the direction an audit is allowed to be wrong in.
    "2609.03064": ("include", "pipeline: Gaussian differential-privacy mechanism combined with "
                              "Byzantine-robust aggregation (DP-BR-FedAvg)"),
    "2603.23472": ("include", "pipeline: robust aggregation integrated with double momentum and "
                              "designed clipping under DP (Byz-Clip21-SGD2M)"),
    "2306.12608": ("include", "pipeline: differential privacy together with client-momentum Byzantine "
                              "robustness, explicitly targeting both at once (DP-BREM)"),
    "2110.02940": ("include", "pipeline: secure averaging within randomly clustered clients BEFORE "
                              "robust-aggregation filtering, a stated two-stage order (SHARE)"),
    "2104.06685": ("include", "pipeline: compression composed with geometric-median robust "
                              "aggregation, whose vanilla combination the paper analyses (BROADCAST)"),
    "2508.12978": ("include", "pipeline: Johnson-Lindenstrauss compression integrated with robust "
                              "averaging under DP (Fed-DPRoC / RobAJoL)"),
    "2509.08449": ("include", "pipeline: group-based secure aggregation plus Byzantine-resilience "
                              "checks across two servers (DSFL)"),
    "2507.14588": ("include", "pipeline: DFT-guided Krum composed with real-domain secure aggregation "
                              "(FORTA)"),
    "2505.17226": ("include", "pipeline: median-based outlier filtering BEFORE adversary-count "
                              "estimation, then multi-update averaging (ArKrum)"),
    "2503.18284": ("include", "pipeline: zero-trust Byzantine identification plus adaptive device "
                              "clustering (FedSAC)"),
    "2310.13403": ("include", "pipeline: Pearson-correlation aggregator selection plus spectral "
                              "clustering with within-cluster averaging (BRFL)"),
    # FLARE screened INCLUDE on its abstract (reputation scoring, adaptive thresholding and
    # reputation-weighted aggregation under LDP) and is then lost at the FULL-TEXT-SOUGHT stage: the
    # paper was WITHDRAWN by its authors and arxiv.org/pdf/2511.14715 is a 404, so there is no text to
    # quote. The pre-registration's full-text restriction says such a paper is excluded with its
    # reason, and the reason is recorded here rather than as a fetch failure, because a reader
    # bounding the frame's full-text bias needs to know this one is unobtainable in principle.
    "2511.14715": ("exclude", "screened INCLUDE on title and abstract (reputation scoring plus "
                              "reputation-weighted aggregation under LDP), then excluded at the "
                              "full-text-sought stage: the paper has been withdrawn by its authors "
                              "and no PDF exists, so no criterion could be quoted (FLARE)"),
    "2409.19302": ("include", "pipeline: moving-target-defense reconfiguration alongside a reputation "
                              "system over model similarity and loss (MTD-DFL)"),
    "2404.04139": ("include", "pipeline: zone-based deviating-update detection plus precision-guided "
                              "zone characterisation to discard updates (FedZZ)"),
    "2309.10607": ("include", "borderline, included per the borderline rule: attention-guided self-"
                              "distillation purification which the paper says works in tandem with an "
                              "arbitrary server aggregator (SPFL)"),
    "2012.13995": ("include", "borderline, included per the borderline rule: root-dataset trust "
                              "scoring plus update magnitude normalisation. main.tex:1654 asserts "
                              "FLTrust is a single stage; coding against full text decides it, which "
                              "is the point of admitting it (FLTrust)"),
    "2601.04930": ("include", "borderline, included per the borderline rule: verifiable-coordinator "
                              "clustering, LWE masking and differential privacy combined, though the "
                              "threat is a Byzantine AGGREGATOR rather than a poisoning client, so "
                              "C-c may well code NO"),
    "2311.15894": ("include", "borderline, included per the borderline rule: primarily an attack "
                              "paper, but it proposes refined-Krum composed with secure aggregation "
                              "as its own defense, so it is not attack-only"),

    # ---- INCLUDE, third pass: rows with >=3 families and NO joint verb -------------------------
    # A three-family abstract that never says "combine" is either a related-work list or a pipeline
    # whose joint verb the JOINT pattern lacks. Four such rows existed; two were pipelines, and one of
    # them (SpectralKrum) is why "fuses" is now in JOINT.
    "2512.11760": ("include", "pipeline: spectral subspace projection, Krum selection in compressed "
                              "coordinates, and orthogonal-residual-energy filtering (SpectralKrum)"),
    "2302.07173": ("include", "an empirical comparison of Byzantine-robust schemes that also PROPOSES "
                              "ClippedClustering: clipping composed with a clustering-based scheme"),
    "2508.18060": ("exclude", "single-mechanism defense: reference-loss ordering with greedy subset "
                              "selection is one selection rule (FedGreed)"),
    "2101.11799": ("exclude", "attack paper: proposes covert model-poisoning algorithms against Krum "
                              "and trimmed mean"),

    # ---- EXCLUDE, second pass: the rest of the --flag review queue -----------------------------
    "2606.27622": ("exclude", "one defense mechanism (hierarchical root-dataset trust scoring) "
                              "combined with heterogeneity-aware OPTIMIZERS (FedAdam, SCAFFOLD). An "
                              "optimizer is not a defense, so this is a single mechanism under "
                              "reading 1 of C-a (FoggyTrust)"),
    "2506.17805": ("exclude", "defends against a biased-selection attack by an adversarial AGGREGATOR "
                              "and protects privacy; it is not a poisoning or backdoor defense "
                              "pipeline (AdRo-FL)"),
    "2502.13728": ("exclude", "privacy pipeline for federated dataset distillation; no poisoning or "
                              "backdoor defense (SFDD)"),
    "2504.01240": ("exclude", "survey: the abstract opens \"In this survey, we investigate the most "
                              "recent techniques of resilient federated learning\""),
    "2509.05265": ("exclude", "attack paper: proposes a model-poisoning attack framework against "
                              "LDPFL. It defeats Multi-Krum and trimmed mean but proposes no composed "
                              "defense, and treating an attack paper's targets as an evaluated "
                              "pipeline would make that clause vacuous"),
    "2608.06637": ("exclude", "attack paper: proposes the Krum-Proxy selection-aware backdoor attack"),
    "2109.09955": ("exclude", "attack paper: proposes DeSMP, a DP-exploiting stealthy model-poisoning "
                              "attack"),
    "2404.19420": ("exclude", "attack paper: proposes a focused backdoor attack against federated "
                              "transfer learning"),
    "2407.09958": ("exclude", "attack paper: proposes BoTPA, a targeted-poisoning booster"),
    "2405.20975": ("exclude", "attack paper: proposes ACE against contribution-evaluation methods"),
    "2303.03908": ("exclude", "privacy attack: client-specific property inference against secure "
                              "aggregation, not a poisoning defense"),

    # ---- EXCLUDE: one defense mechanism ------------------------------------------------------
    "2311.10248": ("exclude", "single-mechanism defense: one dynamic-weight ground-truth estimator. "
                              "The clipping and median vocabulary in its abstract describes prior "
                              "work (FLTrust, FLAME), not its own composition"),
    "1912.12716": ("exclude", "single defense mechanism (geometric median) plus a variance-reduction "
                              "optimizer, which is not a second defense"),
    "2506.22506": ("exclude", "single-mechanism defense: one embedding-space anomaly filter "
                              "(SABRE-FL)"),
    "2405.13080": ("exclude", "single-mechanism defense: embedding inspection (EmInspector)"),
    "2212.01976": ("exclude", "single-mechanism defense: CKA-similarity clustering of penultimate "
                              "representations (FedCC)"),
    "2207.00872": ("exclude", "single-mechanism defense: one last-layer angular-similarity detector "
                              "with engineered features (FL-Defender)"),
    "2602.02615": ("exclude", "single-mechanism defense: statistical update fingerprint deviation "
                              "(TinyGuard)"),
    "2512.12617": ("exclude", "single-mechanism defense: one random-matrix spectral detector; the "
                              "sketching is an efficiency device, not a second defense"),
    "2510.07922": ("exclude", "single-mechanism defense: sketch-domain screening; the sketch is a "
                              "compression of the same similarity test (SketchGuard)"),
    "2501.06729": ("exclude", "single-mechanism defense: kernel-density trust segmentation (KeTS)"),
    "2110.10108": ("exclude", "single-mechanism defense: gradient flip score (TESSERACT)"),
    "2109.05872": ("exclude", "single-mechanism defense: element-wise sign-based gradient filtering"),
    "2509.24330": ("exclude", "single-mechanism defense: similarity-aware aggregation (H+)"),
    "2504.15674": ("exclude", "single-mechanism defense, self-described as detection-free proactive "
                              "robustification (TrojanDam)"),

    # ---- EXCLUDE: post-training repair, not applied to the same training round ---------------
    "2606.22700": ("exclude", "post-training backdoor removal after convergence, not two mechanisms "
                              "applied to the same training round (SCRUB-FL)"),
    "2508.13853": ("exclude", "post-training pruning-based unlearning, not applied to a training "
                              "round (FedUP)"),
    "2011.01767": ("exclude", "post-training pruning then weight adjustment, explicitly after the "
                              "training phase"),

    # ---- EXCLUDE: attack papers --------------------------------------------------------------
    "2605.27416": ("exclude", "attack paper: proposes the CULT circuit-level backdoor threat model"),
    "2407.03144": ("exclude", "attack paper: proposes the Venomancer backdoor attack"),
    "2109.01275": ("exclude", "attack paper: proposes the AdvTrojan combined attack"),
    "2010.10572": ("exclude", "attack paper: implements Sybil attacks on DP-based FL; the DP is the "
                              "setting under attack, not a composed defense"),

    # ---- EXCLUDE: not a poisoning/backdoor defense pipeline ---------------------------------
    "2405.15632": ("exclude", "analysis and explanation method for client behaviour, not a defense "
                              "(Federated Behavioural Planes)"),
    "2605.05644": ("exclude", "client-selection scheduling policies, not a defense pipeline"),
    "2505.16371": ("exclude", "privacy pipeline (graph attention, DP, homomorphic encryption) with no "
                              "poisoning or backdoor defense"),
    "2609.02971": ("exclude", "defends against identity inference, not against poisoning or "
                              "backdoors"),
}


def mechanical(c, fams):
    """Rules applied to every row the adjudication table does not name. Each returns its own reason."""
    t, a = (c.get("title") or ""), (c.get("abstract") or "")
    if not FL.search(t + ". " + a):
        return "exclude", "not federated learning: no federated, decentralized-learning or " \
                          "collaborative-learning term in title or abstract"
    if c.get("source") == "F3-snowball" and not c.get("arxiv_id"):
        return "exclude", "F3 reference-line capture: the retrieval stored a bibliography line " \
                          "rather than a title, so this row is not independently screenable. " \
                          "Recorded as a retrieval-stage loss, not a substantive exclusion"
    if not a and not c.get("arxiv_id"):
        return "exclude", "no abstract retrieved and no arXiv identifier resolved, so full text " \
                          "could not be sought; recorded as a retrieval-stage loss"
    if SURVEY.search(t):
        return "exclude", "survey, review or position paper by title"
    if SURVEY_AB.search(a):
        return "exclude", "survey or review by its own abstract's self-description, though its title " \
                          "does not say so"
    if not a:
        return "exclude", "title-only record: no abstract and no full text obtained, so the pipeline " \
                          "question could not be reached"
    if len(fams) >= 2:
        return "exclude", f"title/abstract names mechanisms from {len(fams)} families " \
                          f"({', '.join(fams)}) but shows no joint application: no combining, " \
                          f"integrating, two-stage or complementary-coverage language"
    return "exclude", f"single-mechanism or non-pipeline defense: {len(fams)} mechanism family " \
                      f"detected in title/abstract ({', '.join(fams) or 'none'})"


def main():
    raw = json.load(open(os.path.join(OUT, "candidates_raw.json")))

    def norm(t):
        return re.sub(r"[^a-z0-9]", "", (t or "").lower())[:80]

    # Dedup keeps the copy with the LONGEST abstract, which is what makes title resolution pay off:
    # a `dedup-within-frame` row carries an id and an empty abstract, and its sibling from F2 carries
    # the abstract, so the merged record is screenable where neither copy alone was.
    uniq = {}
    for c in raw["candidates"]:
        k = c.get("arxiv_id") or norm(c.get("title"))
        if k not in uniq or len(c.get("abstract") or "") > len(uniq[k].get("abstract") or ""):
            uniq[k] = c

    rows, n_adj = [], 0
    for k, c in uniq.items():
        f = families_of(c)
        if c.get("arxiv_id") in ADJUDICATED:
            dec, reason = ADJUDICATED[c["arxiv_id"]]
            n_adj += 1
            how = "adjudicated"
        else:
            dec, reason = mechanical(c, f)
            how = "mechanical"
        rows.append({"id": c.get("arxiv_id") or k, "arxiv_id": c.get("arxiv_id"),
                     "title": c.get("title"), "source": c.get("source"), "query": c.get("query"),
                     "venue": c.get("venue"), "year": c.get("year"),
                     "mechanism_families": f, "nfam": len(f),
                     "joint_language": bool(JOINT.search((c.get("title") or "") + ". "
                                                         + (c.get("abstract") or ""))),
                     "id_resolved_by": c.get("id_resolved_by"),
                     "screen": dec, "screen_reason": reason,
                     "decided_by": how})

    inc = [r for r in rows if r["screen"] == "include"]
    named = {i for i in ADJUDICATED}
    missing = sorted(named - {r["arxiv_id"] for r in rows})
    out = {"deduplicated": len(rows), "adjudicated": n_adj, "mechanical": len(rows) - n_adj,
           "included": len(inc), "excluded": len(rows) - len(inc),
           "adjudicated_ids_not_found_in_frame": missing,
           "note": "Screening on title and abstract. The two readings of C-a that were settled before "
                   "coding, and the borderline-include rule, are documented in "
                   "experiments/screen_literature_audit.py.",
           "rows": rows}
    json.dump(out, open(os.path.join(OUT, "screening.json"), "w"), indent=2)
    print(f"deduplicated {len(rows)}  adjudicated {n_adj}  included {len(inc)}")
    if missing:
        print(f"WARNING: adjudicated ids absent from the frame: {missing}")

    # THE REVIEW QUEUE. Any row the adjudication table does not name, that shows mechanisms from two or
    # more families AND joint-application language, is where a missed pipeline hides. Printing it makes
    # the recall limit inspectable instead of implicit; every row here was read before coding began.
    if "--flag" in sys.argv:
        q = [r for r in rows if r["decided_by"] == "mechanical" and r["nfam"] >= 2
             and r["joint_language"]]
        print(f"\nREVIEW QUEUE: {len(q)} unadjudicated rows with >=2 families AND joint language")
        for r in sorted(q, key=lambda r: -r["nfam"]):
            print(f"  {r['id']:14s} n={r['nfam']} {','.join(r['mechanism_families'])[:52]:52s} "
                  f"{(r['title'] or '')[:56]}")
        return 0

    print("\nINCLUDED:")
    for r in sorted(inc, key=lambda r: r["id"]):
        print(f"  {r['id']:12s} {(r['title'] or '')[:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
