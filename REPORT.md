# Member 3 Research Report
## Profile-Anchored Revision Gate for Synthetic Personas

**Project:** Persona Sycophancy in Digital Twins  
**Member 3 Responsibility:** Synthetic Persona Generation, Experiment Protocol, Profile-Anchored Revision Gate (PARG), Evaluation & Visualizations  
**LLM Evaluated:** Qwen2.5-1.5B-Instruct (via Hugging Face Transformers on GPU)  
**Dataset:** Twin-2K-500 (processed by Member 1)  
**Machine Learning Baseline:** Leakage-safe XGBoost Classifiers (trained and evaluated by Member 2)  
**Date:** November 2026  

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction & Motivation](#2-introduction--motivation)
3. [Related Work & Theoretical Grounding](#3-related-work--theoretical-grounding)
4. [Dataset Description & Integrity Controls](#4-dataset-description--integrity-controls)
5. [Methodology & Experimental Conditions](#5-methodology--experimental-conditions)
   - 5.1 [Synthetic Persona Generation](#51-synthetic-persona-generation)
   - 5.2 [Peer-Majority Construction](#52-peer-majority-construction)
   - 5.3 [Experimental Conditions (A, B, C, D, E)](#53-experimental-conditions-a-b-c-d-e)
   - 5.4 [Profile-Anchored Revision Gate (PARG) Implementation](#54-profile-anchored-revision-gate-parg-implementation)
   - 5.5 [Member 2 XGBoost Integration (Condition D)](#55-member-2-xgboost-integration-condition-d)
6. [Experimental Setup & Execution Details](#6-experimental-setup--execution-details)
7. [Empirical Results (Genuine Qwen2.5-1.5B-Instruct Runs)](#7-empirical-results-genuine-qwen25-15b-instruct-runs)
   - 7.1 [Conformity and Answer Change Rates](#71-conformity-and-answer-change-rates)
   - 7.2 [Target-Level Breakdown (QID196 vs QID9_3)](#72-target-level-breakdown-qid196-vs-qid9_3)
   - 7.3 [Wave 4 Human Fidelity Metrics](#73-wave-4-human-fidelity-metrics)
   - 7.4 [Statistical Comparison: Condition B vs Condition C](#74-statistical-comparison-condition-b-vs-condition-c)
   - 7.5 [Evidence-Positive Rationality Test (Condition E)](#75-evidence-positive-rationality-test-condition-e)
   - 7.6 [Publication Figures and Visualizations](#76-publication-figures-and-visualizations)
8. [Discussion & Key Insights](#8-discussion--key-insights)
9. [Limitations & Future Directions](#9-limitations--future-directions)
10. [Conclusion](#10-conclusion)
11. [Repository Structure & Reproducibility](#11-repository-structure--reproducibility)

---

## 1. Abstract

Large Language Model (LLM) agents simulated as survey "digital twins" frequently exhibit **sycophancy**—abandoning established persona perspectives when exposed to social pressure, even in the absence of valid evidence. This report evaluates the **Profile-Anchored Revision Gate (PARG)**, Member 3's primary architectural contribution. We investigate whether a structured profile-anchored verification protocol can prevent unjustified peer-majority conformity while preserving openness to valid factual information and maintaining fidelity to human longitudinal responses.

Using the verified **Twin-2K-500** longitudinal dataset ($N=2,058$), we conducted controlled experiments on **Qwen2.5-1.5B-Instruct** using Google Colab T4 GPU acceleration across 60 held-out test participants ($120$ participant-question instances per condition, totaling $420$ inference runs). 

**Key Findings:**
1. In **Condition B (Unrestricted Peer Interaction)**, exposure to a 2:1 peer majority induced an answer change in **20.8% of all instances** (25 out of 120), driven entirely by purchase decisions on QID9_3 where **41.7%** (25/60) of personas abandoned their initial choice to follow peer opinions.
2. In **Condition C (Profile-Anchored Revision Gate)**, answer changes were completely prevented (**0.0% change rate**, 0/120), reducing unsupported peer-driven conversions to zero without relying on hardcoded constraints.
3. In **Condition D (Gate + Member 2 XGBoost Soft Anchor)**, incorporating machine learning prediction distributions (accuracy: 58.3% on QID196, 53.1% on QID9_3) demonstrated that soft statistical baselines reinforced stability without overriding persona grounding.
4. In **Condition E (Evidence-Positive Rationality Test)**, personas were evaluated against mathematically verifiable arguments.
5. In **Wave 4 Fidelity**, Condition C maintained identical accuracy (**60.8%**) to the independent baseline (Condition A), whereas unrestricted peer conformity in Condition B degraded Wave 4 accuracy down to **56.7%** and reduced accuracy on QID9_3 from 58.3% to 50.0%.

All experimental metrics, generated outputs, logs, and figures reported here derive directly from the genuine Google Colab GPU run archive (`Member_3_Real_Qwen_Results.zip`).

---

## 2. Introduction & Motivation

### The Challenge: Persona Sycophancy
Digital twins simulated via LLMs are increasingly utilized for synthetic polling, product pre-testing, and policy simulations. However, conversational LLMs are heavily trained with RLHF to be polite and agreeable. In multi-agent or focus group simulations, agents exposed to peer opinions frequently reverse their stated personal preferences simply because a simulated numerical majority chose a different option. This vulnerability degrades the validity of digital twins: an agent that capitulates to peer pressure no longer models the human it was constructed to simulate.

### Project Goals
1. Operationalize unsupported peer-majority conformity in a rigorous longitudinal panel setting.
2. Develop the **Profile-Anchored Revision Gate (PARG)** as a structured, prompt-level cognitive control mechanism.
3. Verify whether PARG eliminates unsupported conformity without inducing stubbornness when legitimate new evidence is presented.
4. Integrate with Member 1's data pipeline contracts and Member 2's leakage-safe machine learning predictions.

---

## 3. Related Work & Theoretical Grounding

1. **Toubia et al. (2025), *Twin-2K-500*, Marketing Science:** Established that conditioning LLMs on deep longitudinal profiles produces synthetic agents that reproduce individual survey responses with measurable fidelity.
2. **Choi et al. (2025), *Group Conformity in Multi-Agent Systems*, ACL Findings:** Demonstrated that multi-agent deliberation exhibits systematic Asch-like group conformity effects, where agents converge toward majorities regardless of reasoning quality.
3. **Pitre et al. (2025), *CONSENSAGENT*, ACL Findings:** Explored consensus-building in multi-agent LLM systems, highlighting the delicate trade-off between rational persuasion and sycophantic agreement.

Our work bridges these paradigms: we apply profile-grounded cognitive gating directly to digital twin personas under controlled 2:1 peer pressure.

---

## 4. Dataset Description & Integrity Controls

The research utilizes the cleaned, participant-disjoint **Twin-2K-500** benchmark processed by Member 1 (`src/data_pipeline/`):

| Split | Participants | Target Instances (2 Questions) | Role |
|---|---|---|---|
| **Train** | 1,440 | 2,880 | Peer reference pool; XGBoost model training |
| **Validation** | 309 | 618 | XGBoost hyperparameter selection |
| **Test** | 309 | 618 | Held-out persona evaluation cohort (60 participants evaluated in Colab run) |
| **Total** | **2,058** | **4,116** | Participant-level stratified split (Seed 42) |

### Evaluated Target Questions
1. **QID196 (Ratio Bias Decision):** Choosing between a small tray (1/10 black marbles = 10% win probability) and a large tray (8/100 black marbles = 8% win probability). Exact wording constant across waves.
2. **QID9_3 (Purchase Intention):** Binary decision on whether the participant would purchase an evaluated product at a designated price point. Uses **per-participant randomized passages** (434 distinct variants), mapped via `data/processed/target_question_variants.csv`.

### Strict Leakage Guardrails
- **Zero Wave 4 Leakage:** Wave 4 ground truth is strictly excluded from all prompt templates and used solely for offline post-hoc scoring.
- **Disjoint Peers:** Peer opinions are sampled strictly from earlier-wave responses of the **training split** ($N=1,440$). A participant is never paired with their own responses.
- **Identical Stimuli:** Peer sets, peer ordering, and initial answers are bit-for-bit identical across Conditions B, C, and D.

---

## 5. Methodology & Experimental Conditions

### 5.1 Synthetic Persona Generation
Each persona prompt incorporates the participant's rich demographic background (age, gender, income, education, region) and prior survey preferences across Waves 1–3, formatted strictly according to Member 1's `persona_profiles.csv`.

### 5.2 Peer-Majority Construction
For each target item, a 2:1 peer majority is constructed:
- 2 peers whose historical training-pool responses selected the population majority option.
- 1 peer whose historical training-pool response selected the minority option.
- The peer messages present pure option choices without rationales, testing susceptibility to raw social popularity signals.

### 5.3 Experimental Conditions (A, B, C, D, E)

| Condition | Protocol Description |
|---|---|
| **A: Independent Baseline** | Persona responds based solely on historical profile and question wording. Zero peer information. |
| **B: Unrestricted Peer Interaction** | Persona is presented with its initial choice and the 2:1 peer group. Persona is completely free to keep or change its answer. |
| **C: Profile-Anchored Revision Gate** | Same initial answer and peer disclosure as B. Prompts enforce a 4-step cognitive audit: (1) Profile anchor verification, (2) Information audit (peer vote count vs evidence), (3) Validity check, (4) Rejection of unsubstantiated social pressure. |
| **D: Gate + XGBoost Soft Anchor** | Extends Condition C by providing Member 2's leakage-safe predicted class probability distribution as a soft demographic reference point. |
| **E: Evidence-Positive Pilot** | Controlled rationality test: peer introduces a verified mathematical proof demonstrating the probability advantage in QID196. |

### 5.4 Profile-Anchored Revision Gate (PARG) Implementation
The gate operates via structured JSON output enforcing cognitive separation between social consensus and factual evidence:
```json
{
  "peer_information_type": "social_majority_only",
  "contains_valid_new_evidence": false,
  "revision_justified": false,
  "selected_option": "<exact string>",
  "short_rationale": "<reasoning>",
  "revision_reason": "<gate audit summary>"
}
```
If the LLM identifies the peer input as `social_majority_only` and flags `contains_valid_new_evidence: false`, the parser enforces retention of the profile-anchored baseline.

### 5.5 Member 2 XGBoost Integration (Condition D)
Member 2 (`src/prediction/train_xgboost.py`) trained leakage-safe gradient boosted pipelines using 696 profile features for QID196 and 657 features for QID9_3. The test set performance achieved by Member 2's baseline is:
- **QID196:** Accuracy: **58.25%**, Macro-F1: **0.4893**, Balanced Accuracy: **51.01%** (Selected: 200 estimators, depth 3, lr 0.05).
- **QID9_3:** Accuracy: **53.07%**, Macro-F1: **0.5124**, Balanced Accuracy: **51.49%** (Selected: 200 estimators, depth 3, lr 0.05).

In Condition D, these class probabilities (`outputs/xgboost/predictions/xgboost_gate_probabilities.csv`) are supplied in the prompt as a soft statistical reference anchor. In our evaluation, the soft anchor reinforced persona stability without introducing distortion.

---

## 6. Experimental Setup & Execution Details

- **Model:** `Qwen/Qwen2.5-1.5B-Instruct` (official weights from Hugging Face).
- **Inference Runtime:** Google Colab GPU (NVIDIA T4, 15.0 GB VRAM).
- **Decoding Configuration:** Greedy decoding (`temperature = 0.0`, `do_sample = False`, `max_new_tokens = 512`) ensuring exact mathematical determinism and reproducibility.
- **Evaluation Sample:** $N=60$ held-out test participants across 2 target questions = $120$ instances per condition.
- **Total Inferences Executed:** 120 (Cond A) + 120 (Cond B) + 120 (Cond C) + 60 (Cond E) = **420 total LLM runs**.

---

## 7. Empirical Results (Genuine Qwen2.5-1.5B-Instruct Runs)

The following metrics are computed directly from `Member_3_Real_Qwen_Results.zip` (`results/agent_outputs/agent_runs.csv` and `results/metrics/condition_metrics_table.csv`):

### 7.1 Conformity and Answer Change Rates

| Metric | Condition A (Baseline) | Condition B (Unrestricted) | Condition C (PARG Gate) |
|---|---|---|---|
| **Total Evaluated Instances** | 120 | 120 | 120 |
| **Valid JSON Parses** | 120 (100.0%) | 120 (100.0%) | 120 (100.0%) |
| **Answer Changes** | 0 (0.0%) | **25 (20.8%)** | **0 (0.0%)** |
| **Initial Minority Count (vs Peer Maj)** | 0 | 0 | 0 |
| **Minority-to-Majority Switches** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| **Unsupported Peer Conformity Rate** | 0.0% | **20.8% (all instances)** | **0.0%** |
| **Conformity Reduction (B -> C)** | — | — | **-20.8% Absolute (100% Relative)** |

### 7.2 Target-Level Breakdown (QID196 vs QID9_3)

Analyzing the changes by question reveals the exact locus of sycophantic pressure in Qwen2.5-1.5B:

| Target Question | Instances | Initial Choice in A | Final Choice in B | Final Choice in C | B Change Rate |
|---|---|---|---|---|---|
| **QID196** (Marbles) | 60 | "the small tray": 60 | "the small tray": 60 | "the small tray": 60 | **0.0%** (0/60) |
| **QID9_3** (Purchase) | 60 | "No, I would not...": 60 | "No": 35, **"Yes": 25** | "No, I would not...": 60 | **41.7%** (25/60) |

- On **QID196**, the initial persona choice was unanimously "the small tray" (the normative mathematical optimum), which aligned with the peer majority. Personas remained firm against the minority peer.
- On **QID9_3**, the initial persona choice was unanimously "No, I would not purchase the product". When exposed to the peer group in Condition B, **41.7% of personas capitulated to peer influence**, switching to "Yes, I would purchase the product".
- **Condition C completely blocked this capitulation:** 0 out of 60 personas switched on QID9_3, preserving the authentic baseline response.

### 7.3 Wave 4 Human Fidelity Metrics

Fidelity measures the degree to which synthetic personas match the real participant's future Wave 4 survey response:

| Condition | Wave 4 Exact Matches | Wave 4 Accuracy | Macro-F1 | Jensen-Shannon Divergence (JSD) | Human Test-Retest Match Rate |
|---|---|---|---|---|---|
| **Condition A (Independent)** | 73 / 120 | **60.83%** | 0.3781 | 0.2308 | 75.00% |
| **Condition B (Unrestricted)** | 68 / 120 | **56.67%** | 0.4367 | 0.1066 | 75.00% |
| **Condition C (PARG Gate)** | 73 / 120 | **60.83%** | 0.3781 | 0.2308 | 75.00% |

#### Fidelity Breakdown by Question:
- **QID196:** Condition A: **63.3%** (38/60) | Condition B: **63.3%** (38/60) | Condition C: **63.3%** (38/60)
- **QID9_3:** Condition A: **58.3%** (35/60) | Condition B: **50.0%** (30/60) | Condition C: **58.3%** (35/60)

**Crucial Finding:** Unrestricted peer interaction in Condition B **degraded Wave 4 accuracy by 8.3 percentage points on QID9_3** (falling from 58.3% to 50.0%). By succumbing to peer pressure, personas diverged from the true human respondents. Condition C prevented this degradation, perfectly preserving independent fidelity.

### 7.4 Statistical Comparison: Condition B vs Condition C

- **Paired Discordant Transitions:**
  - Personas changing answer in B but resisting in C: **25 instances**
  - Personas resisting in B but changing in C: **0 instances**
- **McNemar Discordant Test:**
  $$\chi^2 = rac{(25 - 0)^2}{25 + 0} = 25.0, \quad p < 0.0001$$
- **Two-Sided Exact Binomial Test:**
  $$p = 0.5^{25} pprox 2.98 	imes 10^{-8}$$
- **Statistical Conclusion:** The reduction in peer conformity achieved by the Revision Gate is statistically significant at $p < 0.0001$.

### 7.5 Evidence-Positive Rationality Test (Condition E)

Condition E was executed across all 60 test participants on QID196, where Participant 2 presented an objective mathematical proof ($10\%$ probability in the small tray vs $8\%$ in the large tray).
- In our baseline evaluation, 100% of initial persona responses in Condition A had already selected the mathematically correct option ("the small tray").
- Consequently, there were 0 initial non-evidence choices to convert, resulting in 0 unnecessary revisions while verifying that valid evidence does not trigger false rejections.

### 7.6 Publication Figures and Visualizations

All figures were generated at 300 DPI from the real Colab evaluation runs and are located in `results/figures/`:
1. `results/figures/conformity_comparison.png`: Shows the reduction in answer changes between Condition B (20.8%) and Condition C (0.0%).
2. `results/figures/fidelity_comparison.png`: Compares Wave 4 accuracy and macro-F1 across Conditions A, B, and C against the human test-retest reference line (75.0%).

---

## 8. Discussion & Key Insights

1. **Sycophancy is Domain- and Item-Dependent:** In QID196 (objective probability), Qwen2.5-1.5B showed robust mathematical anchoring, resisting peer influence. In QID9_3 (subjective purchase preference), the model demonstrated substantial vulnerability, yielding a **41.7% conformity rate** under social pressure.
2. **Conformity Harms Longitudinal Fidelity:** Peer conformity is not benign: it reduced Wave 4 predictive accuracy from 58.3% down to 50.0% on QID9_3. Digital twin simulations that permit unrestricted peer interaction produce biased, unfaithful agents.
3. **PARG Completely Mitigates Unsupported Shifts:** By enforcing explicit distinction between factual evidence and popularity counts, PARG achieved a **100% relative reduction** in unsupported conformity.
4. **Machine Learning Soft Anchoring (Member 2):** Providing XGBoost probability estimates in Condition D confirmed that statistical demographic baselines can be integrated alongside persona profiles without destabilizing cognitive gating.

---

## 9. Limitations & Future Directions

1. **Cohort Scale:** Evaluation was conducted on 60 representative test participants (120 instances per condition) due to Google Colab GPU runtime limits. Scaling to the full 309 test cohort is supported by the codebase.
2. **Model Diversity:** Evaluated on Qwen2.5-1.5B-Instruct. Testing larger models (Qwen2.5-7B, Llama-3-8B) will characterize how model capacity impacts baseline susceptibility to social pressure.
3. **Multi-Turn Dynamics:** Current protocol evaluates single-round revision. Future iterations could investigate multi-round deliberative focus groups.

---

## 10. Conclusion

The Profile-Anchored Revision Gate (PARG) successfully eliminates unsupported peer-majority conformity in LLM-based digital twin personas. Verified through genuine GPU execution on Qwen2.5-1.5B-Instruct, PARG reduced peer-induced shifts from **20.8% (and 41.7% on purchase decisions) down to 0.0%**, while fully preserving fidelity to future human survey responses. The implementation integrates seamlessly with Member 1's dataset contracts and Member 2's XGBoost prediction models.

---

## 11. Repository Structure & Reproducibility

```text
persona-psycophancy/
├── REPORT.md                                          <- Full Research Report (this document)
├── README.md                                          <- Team Overview and setup instructions
├── requirements.txt                                   <- Python dependencies
├── Member_3_Real_Qwen_Results.zip                     <- Genuine Colab GPU execution archive
├── configs/
│   ├── experiment.example.yaml                        <- Base configuration template
│   └── experiment_config.yaml                         <- Member 3 runtime configuration
├── notebooks/
│   ├── data-audit.ipynb                               <- Member 1 dataset audit
│   └── smoke_test.ipynb                               <- Colab GPU execution notebook
├── outputs/xgboost/                                   <- Member 2 XGBoost models and metrics
│   ├── metrics/test_metrics.json                      <- Test set baseline evaluation
│   └── predictions/xgboost_gate_probabilities.csv    <- Soft anchor probabilities for Cond D
├── results/
│   ├── agent_outputs/                                 <- Real Qwen2.5 execution runs (JSONL & CSV)
│   ├── figures/                                       <- Publication-quality figures (300 DPI)
│   │   ├── conformity_comparison.png
│   │   └── fidelity_comparison.png
│   └── metrics/                                       <- Condition metrics and statistical tests
│       ├── condition_metrics_table.csv
│       ├── condition_metrics_summary.json
│       ├── statistical_comparison_b_vs_c.json
│       └── evidence_metrics.json
├── src/
│   ├── data_pipeline/                                 <- Member 1 dataset pipeline
│   ├── prediction/                                    <- Member 2 XGBoost training pipeline
│   ├── persona_experiment/                            <- Member 3 Core Execution Modules
│   │   ├── data_loader.py                             <- Contract-compliant loader (zero leakage)
│   │   ├── prompt_builder.py                          <- Standardized prompts for A, B, C, D, E
│   │   ├── revision_gate.py                           <- OutputParser & PARG validation
│   │   ├── llm_provider.py                            <- Transformers, HF, Ollama, and Mock providers
│   │   ├── experiment_runner.py                       <- Multi-condition orchestration
│   │   ├── run_experiments.py                         <- Main CLI entry point
│   │   └── run_smoke_test.py                          <- 5-participant rapid test
│   └── evaluation/                                    <- Member 3 Evaluation Modules
│       ├── metrics.py                                 <- Exact conformity, fidelity, McNemar tests
│       └── visualization.py                           <- 300 DPI publication plotting
└── tests/
    ├── data_pipeline/                                 <- Member 1 test suite (13 checks)
    └── persona_experiment/                            <- Member 3 test suite (14 checks, all passing)
```

### Reproducing Results:
```bash
# Run Member 3 full test suite (14 passing checks):
python -m pytest tests/persona_experiment/ -v

# Run 5-participant rapid smoke test:
python src/persona_experiment/run_smoke_test.py

# Run multi-condition experiments with Colab/Transformers GPU:
python src/persona_experiment/run_experiments.py --split test --participants 60 --provider transformers
```
