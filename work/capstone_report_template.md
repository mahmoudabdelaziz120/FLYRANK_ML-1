# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Mahmoud Abdelaziz El-Shahat Ibrahim
- **Lane:** Refresh / Content Opportunity Scoring (Lane 2)
- **Repo:** work/notebooks/capstone.ipynb
- **Date:** September 2026

---

## 1. Problem framing

- **Decision Supported:** Proactive editorial resource allocation for organic search maintenance. Instead of performing reactive audits months after traffic has vanished, content teams decide which existing content assets to rewrite, update, or consolidate before decay permanently impacts ranking positions.
- **Unit of Analysis:** A single pseudonymized content item (`content_id`) within a client tenant (`client_id`) evaluated over a trailing observation window.
- **Model Output:** A calibrated continuous probability of decay $P(\text{down})$, combined with historical exposure volume and staleness metrics to produce a prioritized `opportunity_score` with explanatory diagnostic reason codes.
- **Human Action:** An SEO editor or content strategist prioritizes items with high opportunity scores for immediate on-page refresh, content expansion, meta-tag optimization, or intent re-alignment.
- **Cost of Wrong Calls:**
  - *False Positive (Type I Error):* Spending editorial hours rewriting a page that was naturally fluctuating or stable, resulting in wasted labor cost.
  - *False Negative (Type II Error):* Overlooking a high-traffic pillar page undergoing silent decay, leading to irreversible loss of rank and long-term organic traffic revenue.
- **Why ML Helps:** Simple heuristic thresholds (e.g., "update pages older than 6 months") fail to generalize across diverse client domains and yield sub-random discrimination ($\text{ROC-AUC} \approx 0.4474$). Gradient boosted models capture complex non-linear interactions between historical exposure, staleness, competition, and content volume without manual per-site tuning.

---

## 2. Data safety

- **Dataset Used:** The FlyRank anonymized content refresh dataset (`data/raw/content_refresh_anonymized.csv`), consisting of 30,000 content items across 32 unique clients[cite: 1].
- **Excluded Classes:** 2,236 rows marked with `trend_direction == 'new'` were eliminated because new content items lack sufficient baseline history, leaving 27,764 valid observational units.
- **Excluded Columns (Leakage Isolation):**
  - *Label-Derived Fields:* `trend_direction` and `trend_pct` were strictly excluded as the target label is derived directly from them[cite: 1].
  - *Target-Window Leakage Fields:* `impressions_last_30d`, `clicks_last_30d`, and `sessions_last_30d` reflect performance within the evaluation horizon itself and were stripped.
  - *Mathematical Overlap Leakage:* We audited the 90-day aggregate columns and mathematically confirmed that `impressions_90d - (impressions_prev_30d + impressions_last_30d) >= 0`. Retaining 90-day metrics allowed decision trees to algebraically infer the target window via simple subtraction ($\text{ROC-AUC}$ falsely inflated to $\approx 0.80$). Consequently, all cumulative 90-day aggregates (`*_90d`), activity counters (`days_with_impressions`, `days_with_sessions`), and derived 90-day ratios (`ctr`, `avg_position`, `engagement_rate`, `scroll_rate`, `ai_traffic_pct`, `impression_tier`, `position_tier`) were stripped from the feature space[cite: 1].
  - *Identifiers:* `content_id` and `client_id` were used exclusively for grouped partitioning and row tracking, never as training features[cite: 1].
- **Client Anonymization:** No client names, raw URLs, unhashed domains, proprietary search queries, or internal credentials exist within the dataset or the `work/` directory[cite: 1].

---

## 3. Baseline

- **Baseline 1 (Majority Class / Persistence):**
  - Predicting the majority class ($y = 1$, decay) for every record in the test split.
  - *Base Rate on Unseen Test Split:* $56.24\%$ ($3,145$ positive / $2,447$ negative).
  - *Metrics:* Accuracy: $0.56$ | Precision (Class 1): $0.56$ | Recall (Class 1): $1.00$ | Macro F1: $0.36$ | ROC-AUC: $0.5000$.
- **Baseline 2 (Domain Heuristic Rule):**
  - A deterministic rule flagging a page as decaying if its update interval exceeds the training median (`days_since_last_update > 106`) or its average position is degraded (`avg_position > 15`).
  - *Metrics on the Same Split:* Accuracy: $0.44$ | Precision (Class 1): $0.50$ | Recall (Class 1): $0.39$ | Macro F1: $0.44$ | ROC-AUC: $0.4474$.
- **Fairness:** Both baselines are evaluated on the exact same unseen client holdout partition without data manipulation.

---

## 4. Model / analysis

- **Algorithm:** LightGBM Classifier (`LGBMClassifier`, `n_estimators=150`, `learning_rate=0.05`, `max_depth=5`, `random_state=42`).
- **Target Definition:** Binary target indicator $y \in \{0, 1\}$, where $y = 1$ if `trend_direction == 'down'` (decaying performance over the evaluation window), and $y = 0$ otherwise (`stable`, `up`, `flat`).
- **Feature Space (23 Sanitized Features):**
  - *Historical Pre-Window Exposure:* `impressions_prev_30d`, `clicks_prev_30d`, `sessions_prev_30d`[cite: 1].
  - *Structural Content Attributes:* `word_count`, `char_count`, `content_type`, `main_intent`, `provider_used`, `model_used`, `word_count_tier`, `char_count_tier`.
  - *Market & Competitive Signals:* `search_volume`, `competition`, `competition_level`, `cpc`.
  - *Age & Staleness Tracking:* `content_age_days`, `age_tier`, `age_tier_order`, `days_since_last_update`, `freshness_tier`.
  - *Missingness Indicator Flags:* `has_word_count`, `has_cpc`, `has_avg_position`[cite: 1].

---

## 5. Evaluation

- **Validation Design:** Grouped Split via `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)` grouped strictly on `client_id`[cite: 1].
  - *Train Partition:* 22,172 content assets across 25 clients[cite: 1].
  - *Test Partition:* 5,592 content assets across 7 entirely unseen clients[cite: 1].
  - *Rationale:* Random K-Fold splits lead to identity leakage where models memorize domain-specific baselines[cite: 1]. Grouped splitting enforces out-of-domain generalization.
- **Model vs. Baseline Performance (Identical Test Split):**

| Evaluation Metric | Majority Baseline | Heuristic Rule | Clean LightGBM (Ours) | Lift over Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.56 | 0.44 | **0.64** | $+14.3\%$ |
| **Precision (Class 1)** | 0.56 | 0.50 | **0.67** | $+19.6\%$ |
| **Recall (Class 1)** | 1.00 | 0.39 | **0.69** | Balanced |
| **Macro F1** | 0.36 | 0.44 | **0.63** | $+75.0\%$ |
| **ROC-AUC** | 0.5000 | 0.4474 | **0.6878** | $+37.6\%$ |

- **Error Analysis:**
  - *False Positives ($33\%$ of positive predictions):* Frequently occur on high-exposure articles (`keyword article`) with moderate staleness that managed to retain search intent relevance. The model flags them due to high volume exposure risk.
  - *False Negatives ($31\%$ of decaying articles missed):* Predominantly thin content items or short-tail queries with low baseline search volume where decay is driven by external competitive shocks rather than internal staleness.

---

## 6. Interpretation

- **Primary Predictive Signals:**
  1. `impressions_prev_30d` (Importance: 641): The single strongest predictor[cite: 1]. Higher initial impression volume strongly correlates with observed decay probability, reflecting regression-to-the-mean dynamics.
  2. `word_count` (Importance: 359) & `char_count` (Importance: 294): Structural depth acts as a protective buffer; thin content items deteriorate significantly faster.
  3. `content_age_days` (Importance: 348) & `days_since_last_update` (Importance: 265): Direct staleness measures confirm that content unmaintained beyond 100+ days experiences accelerating rank deterioration.
  4. `search_volume` (Importance: 226) & `competition` (Importance: 176): High-competition targets face continuous competitor updates, accelerating content decay.
- **Surprises & Negative Results:**
  - `model_used` and `provider_used` contributed minimal predictive power compared to raw content length and update cadence.
  - The naive heuristic baseline performed worse than random guessing ($\text{ROC-AUC} = 0.4474$), demonstrating that arbitrary linear thresholds fail across multi-tenant content libraries.

---

## 7. Recommendation

- **Scoring Engine Formulation:** Probabilities alone do not determine business impact. Editorial prioritization is computed via:
 `Opportunity Score = P(down) * log(1 + impressions_prev_30d) * (1 + days_since_last_update / 365)`
- **Editor Playbook:**
  1. **Top Priority (`HIGH_EXPOSURE_RISK`):** Pages with $\text{Opportunity Score} > 9.0$ and impressions $> 5,000$ (e.g., top-ranked keyword articles suffering click deficits). Action: Immediate meta-title overhaul and intent re-optimization.
  2. **Secondary Priority (`STALE_CONTENT` / `THIN_CONTENT`):** Decaying articles with low word counts and $> 180$ days since revision. Action: Editorial expansion, adding structured FAQ blocks, and updating out-of-date citations.
- **Confidence & Framing:** Observed results demonstrate directional, decision-support utility. This analysis reveals observational correlations and does not represent causal guarantees or reverse-engineering of search engine ranking algorithms.

---

## 8. Reproducibility

- **Execution Command:** Open and run all cells sequentially in `work/capstone_refresh_lane.ipynb`.
- **Global Random Seed:** `random_state = 42` (applied to `GroupShuffleSplit` and `LGBMClassifier`).
- **Core Dependencies:**
  - `python >= 3.10`
  - `pandas == 2.2.2`
  - `numpy == 1.26.4`
  - `scikit-learn == 1.5.0`
  - `lightgbm == 4.3.0`
  - `matplotlib == 3.8.4`
- **Verification Guarantee:** Re-running the clean feature notebook from scratch reproduces identical test metrics ($\text{ROC-AUC} = 0.6878$, $\text{Macro F1} = 0.63$).
