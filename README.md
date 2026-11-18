# Explainable Credit Approval Modelling with a Calibrated Hybrid Ensemble

**A Performance, Attribution and Sensitive-Feature Audit of Linear, Tree-Based and Gated Deep Learning Models on the Australian Credit Dataset**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP%20%7C%20LIME-purple)](https://github.com/shap/shap)

## Overview

Credit approval models are evaluated on two simultaneous requirements: how accurately they separate creditworthy from non-creditworthy applicants, and how transparently their decisions can be explained to applicants, auditors, and regulators.

This repository contains the official implementation and research paper for **"Explainable Credit Approval Modelling with a Calibrated Hybrid Ensemble"** on the UCI Statlog Australian Credit Approval dataset ($N = 690$ applications, $14$ attributes). It benchmarks four standalone architectures across the interpretability spectrum against a proposed **Hybrid-TabNet-Ensemble**, accompanied by global and local **TreeSHAP** decompositions, **LIME** cross-method consistency checks, and a **sensitive-attribute & engineered-proxy fairness audit**.

---

## Key Contributions

1. **Leakage-Controlled Benchmark Protocol**: Joint target-and-anchor (`no_prior_default`) stratified 80/20 split ($552$ train / $138$ test). Bayesian-smoothed Weight-of-Evidence (WoE) encoding ($\text{smoothing} = 1.5$), median imputation, `RobustScaler`, Platt probability calibration, and out-of-fold decision threshold optimization are fitted strictly on training partitions with **zero synthetic oversampling (no SMOTE)**.
2. **Proposed Hybrid-TabNet-Ensemble**: Combines Platt-calibrated out-of-fold logits across heterogeneous learners weighted quadratically by 5-fold cross-validated ROC-AUC:
   $$p_{\text{hybrid}}(x) = \sum_{k} w_k \cdot \sigma\left(a_k \cdot \text{logit}(p_k(x)) + b_k\right), \quad w_k \propto \max(\text{AUC}_k - 0.85, 0.01)^2$$
3. **Deep Gated Tabular Network (DGTN)**: A lightweight attentive tabular architecture pairing an $L_1$-regularized sparse logistic attention gate $g \in [0.5, 1.0]$ with a multi-scale residual concatenation $[x, g \odot x]$ fed into a 3-MLP ensemble (`128-64-32 ReLU`, `64-32 tanh`, `96-48 ReLU`).
4. **SHAP vs. LIME Diagnostic Triangulation**: Pairs exact TreeSHAP with LIME on individual applicant predictions, demonstrating how correlated composite features fragment local surrogate weights (60% directional agreement; disagreement on the dominant credit-history anchor).
5. **Sensitive-Feature & Proxy Governance Audit**: Quantifies attribution leakage across demographic inputs (`marital_status_woe`, `gender`, `citizenship_status`) and demonstrates how age influence persists through engineered ratios (`debt_to_age_ratio` carries $3\times$ the attribution of raw `age`).

---

## Domain Feature Engineering

Alongside $\log(1 + x)$ transformations on five skewed continuous attributes, nine domain-specific interaction composites are engineered (resulting in $54$ total model inputs after WoE and One-Hot Encoding):

| Engineered Feature | Mathematical Definition | Financial Reading |
| :--- | :--- | :--- |
| `credit_emp_synergy` | $3 \cdot \text{no\_prior\_default} + 1.5 \cdot \text{is\_employed} + \mathbb{I}[\text{existing\_credits} > 0]$ | Ordered applicant credibility index |
| `default_x_tenure` | $\text{no\_prior\_default} \times \log(1 + \text{years\_employed})$ | Clean history $\times$ employment stability |
| `default_x_credits` | $\text{no\_prior\_default} \times \log(1 + \text{existing\_credits})$ | Clean history $\times$ credit depth |
| `default_x_savings` | $\text{no\_prior\_default} \times \log(1 + \text{savings\_income})$ | Clean history $\times$ liquid buffer |
| `employed_x_credits` | $\text{is\_employed} \times \log(1 + \text{existing\_credits})$ | Employment $\times$ active credit lines |
| `debt_to_age_ratio` | $\text{debt\_ratio} / (\text{age} + 1)$ | Age-normalized obligation ratio |
| `net_liquidity_spread` | $\log(1 + \text{savings\_income}) - \log(1 + \text{loan\_balance})$ | Liquid savings relative to loan balance |
| `credit_velocity` | $\text{existing\_credits} / (\text{years\_employed} + 0.5)$ | Credit accumulation rate per tenure year |
| `has_liquid_income` | $\mathbb{I}[\text{savings\_income} > 0]$ | Non-zero liquid savings indicator |

---

## Benchmark Results ($n_{\text{test}} = 138$)

Approval (`1`) is the positive class. Thresholds are selected from a 33-point grid ($[0.42, 0.58]$) by maximizing out-of-fold training accuracy.

| Model | Opt. Thr. | Accuracy | Precision | Recall | F1-Score | Specificity | MCC | Test ROC-AUC | 5-Fold CV AUC (Mean $\pm$ SD) | TN / FP / FN / TP | 95% Wilson CI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Proposed Hybrid-TabNet-Ensemble** | **0.550** | **0.8841** | **0.8485** | 0.9032 | **0.8750** | **0.8684** | **0.768** | 0.9433 | **0.9403 $\pm$ 0.0192** | **66 / 10 / 6 / 56** | **0.82–0.93** |
| **Logistic Regression** | 0.475 | 0.8768 | 0.8358 | 0.9032 | 0.8682 | 0.8553 | 0.755 | 0.9391 | 0.9306 $\pm$ 0.0265 | 65 / 11 / 6 / 56 | 0.81–0.92 |
| **Random Forest** | 0.480 | 0.8768 | 0.8261 | **0.9194** | 0.8702 | 0.8421 | 0.758 | 0.9391 | 0.9388 $\pm$ 0.0205 | 64 / 12 / 5 / 57 | 0.81–0.92 |
| **XGBoost** | 0.500 | 0.8696 | 0.8143 | **0.9194** | 0.8636 | 0.8289 | 0.745 | **0.9486** | 0.9367 $\pm$ 0.0225 | 63 / 13 / 5 / 57 | 0.80–0.92 |
| **Deep Gated Tabular Net (DGTN)** | 0.550 | 0.8623 | 0.8308 | 0.8710 | 0.8504 | 0.8553 | 0.724 | 0.9300 | 0.9290 $\pm$ 0.0215 | 65 / 11 / 8 / 54 | 0.80–0.91 |

---

## Explainability & Sensitive-Feature Audit Highlights

### Top Global Drivers (TreeSHAP on Held-Out Test Set)
1. `no_prior_default` ($\text{Mean } \vert{}\text{SHAP}\vert{} = 0.9969$) — Primary separator ($2.8\times$ the next feature); combined history composites account for $\approx 66\%$ of top-15 attribution mass.
2. `credit_emp_synergy` ($0.3541$) — Monotonic step-function credibility score.
3. `default_x_tenure` ($0.2742$) — Steep positive contribution that saturates above $\approx 0.5$ log tenure years.
4. `net_liquidity_spread` ($0.2315$) — Penalizes applicants with low liquid savings relative to loan balance.
5. `bank_customer_type_woe` ($0.1343$) — Account tier Weight-of-Evidence signal.

### Sensitive & Age-Derived Proxy Audit
| Sensitive / Proxy Feature | Standalone XGBoost Mean $\Vert{}\text{SHAP}\Vert{}$ | Hybrid Gate-Scaled Proxy Mean $\Vert{}\text{SHAP}\Vert{}$ | DGTN Gate Weight | Global Rank |
| :--- | :---: | :---: | :---: | :---: |
| `marital_status_woe` | 0.0783 | 0.0465 | 0.594 | #8 / 54 |
| `debt_to_age_ratio` *(age proxy)* | 0.0746 | 0.0373 | 0.500 | #9 / 54 |
| `citizenship_status_2` | 0.0347 | 0.0173 | 0.500 | #19 / 54 |
| `age` *(raw)* | 0.0245 | 0.0123 | 0.500 | #21 / 54 |
| `marital_status_2` | 0.0081 | 0.0041 | 0.500 | #30 / 54 |
| `gender` | 0.0046 | 0.0023 | 0.500 | #34 / 54 |

---

## Repository Structure

```text
├── australian.dat                                    # Statlog Australian Credit Approval dataset (UCI)
├── Credit_Risk_Explainable_AI_Modular_Notebook.ipynb # Modular end-to-end experimental notebook
├── Credit_Risk_Explainable_AI_Research_Paper.docx    # Full IEEE-style research paper manuscript
├── paper_figures/                                    # Auto-generated 300 DPI publication figures (Figs 1-14)
│   ├── fig1_target_distribution.png
│   ├── fig2_numerical_distributions.png
│   ├── fig3_correlation_heatmap.png
│   ├── fig4_credit_history_by_class.png
│   ├── fig5_shap_beeswarm.png
│   ├── fig6_shap_bar.png
│   ├── fig7_shap_waterfall_xgb.png
│   ├── fig8_shap_waterfall_hybrid.png
│   ├── fig9_shap_dependence_no_prior_default.png
│   ├── fig10_shap_dependence_credit_emp_synergy.png
│   ├── fig11_shap_dependence_default_x_tenure.png
│   ├── fig12_lime_local_explanation.png
│   ├── fig13_fairness_audit_xgb.png
│   └── fig14_fairness_audit_hybrid.png
└── README.md
