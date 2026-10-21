import os
import nbformat as nbf

NOTEBOOKS_DIR = os.path.join('work', 'notebooks')
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

# Remove old 2 notebooks
old_notebooks = [
    os.path.join(NOTEBOOKS_DIR, '01_data_audit_and_preparation.ipynb'),
    os.path.join(NOTEBOOKS_DIR, '02_model_training_and_evaluation.ipynb')
]
for old_nb in old_notebooks:
    if os.path.exists(old_nb):
        os.remove(old_nb)
        print(f"Removed old notebook: {old_nb}")

# ==========================================
# 1. CREATE TEMPLATE NOTEBOOK 01
# ==========================================
nb1 = nbf.v4.new_notebook()
nb1.cells = [
    nbf.v4.new_markdown_cell("""# 01 — Problem Framing, Hypotheses & Data Ingestion: Ranking Signal Analysis

**Project:** FlyRank Ranking Signal Analysis — Predicting Continuous Search Visibility  
**Pipeline Stage:** Notebook 01 of 03 (Problem Framing, Hypotheses & Feature Vector Assembly)  
**Target Variable:** Continuous Search Visibility $\\log(1 + \\text{impressions\\_90d})$ on Held-Out Enterprise Client Portfolios  
**Standardized Split:** Grouped Holdout (`GroupShuffleSplit`, `test_size=0.20`, `random_state=42`) with Zero Client Domain Overlap  

---

### Notebook Architecture & Objectives
This notebook is your starting template for the data audit and feature engineering phase:
1. **Problem Framing & Operational Decision:** Define the operational content refresh decision, continuous search visibility target, and asymmetric error trade-offs.
2. **Core SEO Hypotheses:** Formulate and statistically test 4 core search hypotheses with explicit empirical verdicts (`CONFIRMED`, `MIXED`, `FALSE`, `OPPOSITE`).
3. **Data Ingestion & Scope Audit:** Verify atomic entity grain (`content_id`), document scope limitations, missingness patterns, and rate scaling.
4. **Feature Vector Assembly & Leakage Defense:** Implement SERP inverse power-law ranking signals, engagement ratios, missingness indicators, log-stabilizing transforms, enforce strict leakage blacklists, and export the validated feature dataset."""),

    nbf.v4.new_markdown_cell("""## 1. Problem Framing & Operational Decision

### 1.1 The Operational Decision
* **What operational decision does this model improve?**  
  Enterprise content and SEO teams manage portfolios spanning thousands of URLs across diverse client domains. Currently, content refresh prioritization relies on crude, lagging manual heuristics (e.g., sorting by calendar age or raw pageviews). This model predicts expected **search visibility** (the latent search impression capacity of a page based on its structural, semantic, and engagement signals). Content strategists and engineering teams act on this model to identify:
  1. **Under-indexed high-potential pages:** Content whose observed features warrant high search impressions but currently lag behind their predicted potential (prime refresh/optimization candidates).
  2. **Over-resourced low-potential pages:** Thin or poorly positioned content that should not consume scarce editorial rewrite hours.

* **Who acts on it?**  
  Content operations leads, technical SEO specialists, and client account managers.

### 1.2 Target Definition: Search Visibility
* **Why Continuous Regression instead of Binary Classification?**  
  Binary classification collapses search performance into an arbitrary threshold (e.g., "declining" vs. "stable"), obscuring orders of magnitude of search impact. Continuous regression models the true scale of visibility across the entire distribution.
* **Target Formulation:**
  $$\\text{target\\_log\\_impressions} = \\log(1 + \\text{impressions\\_90d})$$
  Raw search impressions follow an extreme power-law distribution spanning from single digits to hundreds of thousands. The $\\log(1+x)$ transformation compresses extreme right-tail skewness into a well-behaved, symmetric distribution suitable for regression estimators.

### 1.3 Asymmetric Cost of Errors
* **Overestimation Error (False High Visibility):** The model predicts high visibility for low-yield content.  
  *Operational Cost:* Editorial teams waste expensive subject-matter expert hours rewriting, optimizing, and re-indexing pages that fail to generate organic traffic. Opportunity cost is high.
* **Underestimation Error (False Low Visibility):** The model predicts low visibility for high-yield content.  
  *Operational Cost:* The team neglects high-leverage assets, risking rank decay against aggressive competitor content."""),

    nbf.v4.new_markdown_cell("""## 2. Core SEO Hypotheses

Before building machine learning models, formulate four operational hypotheses based on prevailing SEO heuristics to test whether real-world data supports common industry beliefs:
1. **Hypothesis 1 (Content Depth):** *Article length (`word_count`) exhibits a strong positive relationship with search visibility.*
2. **Hypothesis 2 (Search Demand):** *Target keyword search volume (`search_volume`) strongly determines realized page impressions.*
3. **Hypothesis 3 (SERP Positioning):** *Average SERP rank position (`avg_position`) is inversely correlated with impressions (lower numerical rank = higher visibility).*
4. **Hypothesis 4 (Content Freshness):** *Recently updated content (`days_since_last_update` lower) commands higher search visibility than aged content.*

Each hypothesis should be evaluated using Spearman rank correlation ($\\rho$), Pearson correlation ($r$), and visual distribution inspections, concluding with an explicit statistical verdict (`CONFIRMED`, `OPPOSITE`, `MIXED`, or `FALSE`)."""),

    nbf.v4.new_code_cell("""# Code Cell 1: Setup, Imports, and Data Ingestion
# TODO:
# 1. Import necessary libraries (numpy, pandas, matplotlib.pyplot, seaborn, scipy.stats).
# 2. Ingest 'data/raw/content_refresh_anonymized.csv'.
# 3. Assert 100% unique primary key grain on 'content_id' (should be 30,000 rows across 32 clients).
# 4. Construct the continuous regression target: target_log_impressions = np.log1p(df['impressions_90d']).
# 5. Display summary statistics of the target variable.
"""),

    nbf.v4.new_markdown_cell("""## 3. Data Ingestion: Scope & Limitations

### 3.1 Scope of the Dataset
* **Dataset File:** `data/raw/content_refresh_anonymized.csv` (30,000 rows × 44 columns).
* **Entity Grain:** Exactly 1 row per unique pseudonymized content item (`content_id`).
* **Client Representation:** Exactly 32 distinct pseudonymized enterprise client domains (`client_id`).
* **Observation Window:** A fixed trailing 90-day observation window ending at export time, with 30-day comparative windows (`*_last_30d` vs `*_prev_30d`).
* **Cohort Age:** Established, mature content only (`content_age_days >= 90`). Every row has at least 1 impression (`impressions_90d >= 1`).

### 3.2 Key Data Gotchas & Limitations
1. **Rate Scaling (×100):** Columns `ctr`, `engagement_rate`, `scroll_rate`, `ai_traffic_pct`, and `trend_pct` are recorded as ×100 percentages (`0.76` means `0.76%`, not `76%`).
2. **`avg_position = 0` Unranked State:** 1,205 rows have `avg_position = 0`, denoting unranked status / no impression position recorded, not rank zero.
3. **Systematic Missingness:** Keyword context is missing along `content_type` lines (e.g. `feedly article` rows have no keyword data; ~25.7% of rows lack `word_count`). Blind `fillna(0)` injects artificial category signals; missingness indicators are required.
4. **Observable Telemetry Only:** Internal product decision flags (`health_score`, `priority_score`, `action_type`) are deliberately excluded to prevent circular result leakage."""),

    nbf.v4.new_code_cell("""# Code Cell 2: Completeness and Missingness Audit
# TODO:
# 1. Calculate missing value counts and percentages across all columns in df_raw.
# 2. Audit zero-value counts for numeric features.
# 3. Display the completeness summary table for columns with missing data.
"""),

    nbf.v4.new_code_cell("""# Code Cell 3: Statistical Hypothesis Testing with Explicit Verdicts
# TODO:
# 1. Compute Spearman rho and Pearson r correlations for each hypothesis against target_log_impressions:
#    - H1: word_count
#    - H2: search_volume
#    - H3: avg_position (filter avg_position > 0)
#    - H4: days_since_last_update
# 2. Print a verdict summary table with explicit verdicts (CONFIRMED / MIXED / FALSE / OPPOSITE).
# 3. Generate a 2x2 grid of scatter plots illustrating each relationship and save to 'work/outputs/hypothesis_audit_distributions.png'.
"""),

    nbf.v4.new_markdown_cell("""### Interpretation of Hypotheses Findings
* **Guide for your write-up:**
  * Explain in plain words what each statistical verdict revealed about search traffic.
  * Why did H2 (search volume) show a mixed/weak correlation? (Broad head-terms face high competition).
  * Why was H3 (rank position) false when tested in isolation? (Position without search demand yields zero views).
  * Why did H4 (content freshness) show an opposite result? (Cumulative domain authority and backlink history outweigh recent date updates)."""),

    nbf.v4.new_markdown_cell("""## 4. Feature Vector Assembly & Leakage Defense

### 4.1 Strict Leakage Blacklists
To ensure 100% scientific integrity, the feature pipeline strictly blacklists all leakers:
* **Pseudonymous Identifiers:** `content_id`, `client_id` (used solely for grouping and validation assertions).
* **Direct Target Telemetry:** `impressions_90d`, `clicks_90d`, `ctr` (since CTR contains impressions in its denominator).
* **Outcome Window Metrics:** `impressions_last_30d`, `clicks_last_30d`, `sessions_last_30d`, `impressions_prev_30d`, `clicks_prev_30d`, `sessions_prev_30d`.
* **Target-Derived Labels:** `trend_direction`, `trend_pct`, `impression_tier`.

### 4.2 Domain-Specific Feature Engineering
Engineer high-leverage structural and behavioral signals:
1. **SERP Power-Law Inverse Rank (`serp_rank_power`):**
   $$\\text{serp\\_rank\\_power} = \\begin{cases} \\frac{1}{\\text{avg\\_position}}, & \\text{if } \\text{avg\\_position} > 0 \\\\ 0, & \\text{otherwise} \\end{cases}$$
2. **Rank Presence Indicator:** `has_rank_data = (avg_position > 0).astype(int)`.
3. **Rank Imputation:** `imputed_avg_position = np.where(avg_position > 0, avg_position, 100.0)`.
4. **Behavioral Ratios:**
   * `engaged_session_ratio = engaged_sessions_90d / (sessions_90d + 1)`
   * `scroll_per_pageview = scroll_events_90d / (pageviews_90d + 1)`
   * `ai_session_ratio = ai_sessions_90d / (sessions_90d + 1)`
5. **Demand vs. Position Synergy:** `demand_position_synergy = log_search_volume * serp_rank_power`.
6. **Engagement vs. Volume Synergy:** `engagement_volume_synergy = log_sessions_90d * serp_rank_power`.
7. **Systematic Missingness Indicators:** `missing_word_count`, `missing_search_volume`, `missing_competition`, `missing_cpc`.
8. **Log-Stabilizing Transforms:** Applied to heavy-tailed count columns (`word_count`, `search_volume`, `sessions_90d`, etc.).
9. **One-Hot Categorical Encoding:** Applied to `content_type`, `main_intent`, `competition_level`, `freshness_tier`, `position_tier`."""),

    nbf.v4.new_code_cell("""# Code Cell 4: Feature Vector Assembly and Dataset Export
# TODO:
# 1. Apply feature engineering:
#    - Inverse SERP power-law features (serp_rank_power, log_avg_position, imputed_avg_position)
#    - Missingness indicator flags (missing_word_count, missing_search_volume, etc.)
#    - Behavioral engagement ratios (engaged_session_ratio, scroll_per_pageview, etc.)
#    - Log-stabilizing np.log1p() transforms on skewed count metrics
#    - Interaction synergy terms (demand_position_synergy, engagement_volume_synergy)
# 2. Apply pd.get_dummies() to categorical columns (content_type, main_intent, competition_level, freshness_tier, position_tier).
# 3. Concatenate metadata columns and all engineered numeric + dummy features.
# 4. Save the prepared dataset to:
#    - 'work/outputs/prepared_features.parquet'
#    - 'work/outputs/prepared_features.csv'
# 5. Print confirmation of matrix shape and assert zero target leakers in feature vector.
""")
]

nb1_path = os.path.join(NOTEBOOKS_DIR, '01_problem_framing_and_data_audit.ipynb')
with open(nb1_path, 'w', encoding='utf-8') as f:
    nbf.write(nb1, f)
print(f"Created template: {nb1_path}")


# ==========================================
# 2. CREATE TEMPLATE NOTEBOOK 02
# ==========================================
nb2 = nbf.v4.new_notebook()
nb2.cells = [
    nbf.v4.new_markdown_cell("""# 02 — Grouped Split, Baseline Benchmarking & Model Training

**Project:** FlyRank Ranking Signal Analysis — Predicting Continuous Search Visibility  
**Pipeline Stage:** Notebook 02 of 03 (Model Training & Benchmark Optimization)  
**Target Variable:** Continuous Search Visibility $\\log(1 + \\text{impressions\\_90d})$ on Held-Out Enterprise Client Portfolios  
**Standardized Split:** Grouped Holdout (`GroupShuffleSplit`, `test_size=0.20`, `random_state=42`) with Zero Client Domain Overlap  

---

### Notebook Architecture & Objectives
This notebook is your starting template for model training and optimization:
1. **Grouped Holdout Split Design:** Enforce a strict zero-client-leakage partition on `client_id` with programmatic disjointness assertions.
2. **Transparent Heuristic Baselines:** Fit Baseline 1 (Mean Dummy Regressor) and Baseline 2 (Univariate Word Count Rule) to establish the floor to beat.
3. **High-Accuracy Model Training:** 
   - Fit candidate algorithms (e.g., cross-validated Polynomial Interaction Ridge, HistGradientBoosting Regressor, or Stacking Ensemble).
   - Maximize generalization accuracy on unseen client domains.
4. **Payload Export:** Export model predictions, test set ground truth, and metadata to `work/outputs/model_evaluation_payload.pkl` for evaluation in Notebook 03."""),

    nbf.v4.new_markdown_cell("""## 1. Grouped Holdout Split Design (Zero Client Leakage)

### 1.1 The Multi-Tenant Generalization Challenge
In multi-tenant SaaS platforms, standard random row splits produce severe **client leakage**: content items from the same client share identical domain authority, root backlink velocity, CMS infrastructure, and brand equity. A random split allows models to memorize client-level baseline volume, inflating test performance while collapsing on brand-new client domains.

### 1.2 The Client Holdout Standard
To guarantee 100% scientific validity:
* We partition strictly by `client_id` using `GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)`.
* We assert that training clients and test clients are mutually exclusive:
  $$\\text{set}(\\text{clients}_{\\text{train}}) \\cap \\text{set}(\\text{clients}_{\\text{test}}) = \\emptyset$$
* Test metrics reflect true generalization performance on entirely unseen enterprise domains."""),

    nbf.v4.new_code_cell("""# Code Cell 1: Load Features and Execute Grouped Split
# TODO:
# 1. Load prepared feature matrix from 'work/outputs/prepared_features.parquet' (or .csv).
# 2. Separate feature matrix X, target y (target_log_impressions), and groups (client_id).
# 3. Perform GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42).
# 4. Programmatically assert set(clients_train).isdisjoint(set(clients_test)).
# 5. Print train and test matrix shapes and target distributions.
"""),

    nbf.v4.new_markdown_cell("""## 2. Transparent Heuristic Baseline Benchmarks

To verify that machine learning provides genuine predictive lift rather than fitting noise, evaluate two non-ML baselines on the exact same test cohort:
1. **Baseline 1 (Mean Dummy Regressor):** Always predicts the training set target mean $\\bar{y}_{\\text{train}}$. (Yields $R^2 = 0.0$ on train; negative $R^2$ on held-out clients with a lower mean).
2. **Baseline 2 (Univariate Word Count Rule):** A simple single-variable OLS regression on `log_word_count`, reflecting the common SEO belief that "longer articles rank higher and gain more visibility"."""),

    nbf.v4.new_code_cell("""# Code Cell 2: Fit and Evaluate Standardized Heuristic Baselines
# TODO:
# 1. Define an evaluation helper function computing R2 score, RMSE, MAE, and Spearman rho.
# 2. Fit Baseline 1: Mean Dummy Regressor (predicts y_train.mean()).
# 3. Fit Baseline 2: Univariate OLS Regressor using only 'log_word_count'.
# 4. Evaluate both baselines on the held-out test cohort (X_test, y_test).
# 5. Display the baseline performance summary table.
"""),

    nbf.v4.new_markdown_cell("""## 3. High-Accuracy Model Training

### 3.1 Model Architecture Strategy
To achieve the highest honest accuracy without data leakage, consider:
1. **Regularized Linear Models with Interactions:** Pairwise interaction terms ($x_i \\times x_j$) across primary signals regularized with `RidgeCV`.
2. **Gradient Boosted Decision Trees (`HistGradientBoostingRegressor`):** Captures sharp non-linear thresholds, step-function jumps, and missingness natively.
3. **Champion Blended Ensemble:** Blending linear macro-trends with localized tree splits ($w \\cdot \\hat{y}_{\\text{Ridge}} + (1-w) \\cdot \\hat{y}_{\\text{HGBR}}$) to maximize variance explained."""),

    nbf.v4.new_code_cell("""# Code Cell 3: Machine Learning Model Training & Optimization
# TODO:
# 1. Define your candidate model pipelines:
#    - Candidate A: Polynomial Interaction Ridge Regression with RidgeCV
#    - Candidate B: HistGradientBoostingRegressor (tuned learning_rate, max_iter, l2_regularization)
#    - Champion Architecture: Blended Stacking Ensemble (0.50 * Ridge + 0.50 * HGBR)
# 2. Fit model(s) on the training set (X_train, y_train).
# 3. Generate predictions on the unseen test set (X_test).
# 4. Compute test metrics (R2, RMSE, MAE, Spearman rho) and print comparison logs.
"""),

    nbf.v4.new_code_cell("""# Code Cell 4: Export Model Evaluation Payload
# TODO:
# 1. Package test ground truth, predictions from all models, client IDs, and metadata into a dictionary payload.
# 2. Save payload to 'work/outputs/model_evaluation_payload.pkl' for Notebook 03.
# 3. Export predictions dataframe to 'work/outputs/model_test_predictions.csv'.
# 4. Print confirmation of export paths and file sizes.
""")
]

nb2_path = os.path.join(NOTEBOOKS_DIR, '02_model_training.ipynb')
with open(nb2_path, 'w', encoding='utf-8') as f:
    nbf.write(nb2, f)
print(f"Created template: {nb2_path}")


# ==========================================
# 3. CREATE TEMPLATE NOTEBOOK 03
# ==========================================
nb3 = nbf.v4.new_notebook()
nb3.cells = [
    nbf.v4.new_markdown_cell("""# 03 — Comparisons, Error Diagnostics & Operational Action Playbook

**Project:** FlyRank Ranking Signal Analysis — Predicting Continuous Search Visibility  
**Pipeline Stage:** Notebook 03 of 03 (Model Evaluation, Error Diagnostics & Operational Triage)  
**Target Variable:** Continuous Search Visibility $\\log(1 + \\text{impressions\\_90d})$ on Held-Out Enterprise Client Portfolios  
**Standardized Split:** Grouped Holdout (`GroupShuffleSplit`, `test_size=0.20`, `random_state=42`) with Zero Client Domain Overlap  

---

### Notebook Architecture & Objectives
This notebook is your starting template for the final evaluation, diagnostics, and operational actionability phase:
1. **Benchmark Performance Comparisons:** Compile the official evaluation table comparing Baselines vs. Models across $R^2$, RMSE, MAE, Spearman $\\rho$, and lift over baseline.
2. **Diagnostic Error Audits:**
   - Visual inspection of Actual vs. Predicted scatter plots and residual distributions.
   - Sub-cohort performance analysis across content types and search intents.
   - In-depth post-mortem audit of 3 concrete outlier cases.
3. **Operational Action Playbook:**
   - Compute the **Opportunity Residual Gap** ($\\hat{y} - y$) to detect under-indexed high-potential content.
   - Assign operational triage reason codes (`under_indexed_decay`, `high_demand_opportunity`, `stale_authority`).
   - Export the prioritized refresh queue to `work/outputs/refresh_queue_sample.csv`."""),

    nbf.v4.new_markdown_cell("""## 1. Official Benchmark Comparisons

All models are evaluated on the standardized, held-out test cohort of **6,163 content items across 7 unseen clients**:
* **Primary Metric:** $R^2$ Score (Coefficient of Determination) — proportion of variance explained on unseen client domains.
* **Secondary Error Metrics:** RMSE and MAE (magnitude of prediction error in natural log units).
* **Monotonic Ranking Quality:** Spearman rank correlation ($\\rho$) — measure of sorting accuracy for prioritizing content queues."""),

    nbf.v4.new_code_cell("""# Code Cell 1: Load Evaluation Payload and Compile Official Benchmark Table
# TODO:
# 1. Load 'work/outputs/model_evaluation_payload.pkl'.
# 2. Extract ground truth y_test and predictions for Baselines and Model(s).
# 3. Calculate R2, RMSE, MAE, Spearman rho, and percentage RMSE error reduction over baseline.
# 4. Display the formatted official benchmark comparison table.
# 5. Export benchmark metrics to 'work/outputs/model_benchmark_metrics.json'.
"""),

    nbf.v4.new_code_cell("""# Code Cell 2: Diagnostic Error Visualizations
# TODO:
# 1. Generate a 1x3 diagnostic plot:
#    - Subplot 1: Actual vs. Predicted Visibility scatter plot with the 45-degree identity line (y = y_hat).
#    - Subplot 2: Residual error distribution histogram (y_pred - y_true) with KDE and mean line.
#    - Subplot 3: Absolute error boxplots comparing Baseline 1, Baseline 2, and Trained Model.
# 2. Save plot to 'work/outputs/model_evaluation_plots.png'.
"""),

    nbf.v4.new_markdown_cell("""## 2. Detailed Error Diagnostics & Case Studies

### 2.1 Sub-Cohort Robustness
Audit $R^2$ and MAE across sub-cohorts (`content_type`, `main_intent`) on held-out test domains to ensure the model generalizes across categories rather than overfitting to generic content."""),

    nbf.v4.new_code_cell("""# Code Cell 3: Sub-Cohort Analysis and Outlier Audits
# TODO:
# 1. Group test set by 'content_type' and calculate R2, MAE, and Spearman rho for each cohort.
# 2. Perform an in-depth post-mortem on 3 concrete outlier cases:
#    - Case 1: Under-predicted high performer (viral/high-authority asset)
#    - Case 2: Over-predicted low performer (thin content targeting broad competitive head-terms)
#    - Case 3: Prime under-indexed refresh candidate (strong signals, lagging traffic)
# 3. Print diagnoses for each case.
"""),

    nbf.v4.new_markdown_cell("""## 3. Conclusions & Operational Action Playbook

### 3.1 Translating Predictions into Business Value
A regression prediction is only useful if it drives operational action. Rather than expecting editorial teams to parse log-impression numbers, construct an **Automated Refresh Triage Queue**:
* **Opportunity Residual Gap ($\\text{Residual} = \\hat{y} - y$):** 
  * A high positive residual indicates that a page's observed structural depth, semantic properties, and engagement signals warrant substantial search impressions, but its realized traffic is currently lagging far behind expectation.
  * These **under-indexed pages** represent the highest-ROI editorial opportunities: a targeted content refresh can unlock latent ranking potential without starting from scratch.
* **Reason Codes:**
  * `under_indexed_decay`: High predicted potential lagging in realized impressions.
  * `high_demand_opportunity`: High keyword volume and good engagement warranting SERP push.
  * `stale_authority`: Aging high-authority asset at risk of competitive decay."""),

    nbf.v4.new_code_cell("""# Code Cell 4: Build and Export Operational Refresh Triage Queue
# TODO:
# 1. Compute Opportunity Residual Gap = pred_log_impressions - target_log_impressions.
# 2. Compute predicted raw impressions = np.expm1(pred_log_impressions).
# 3. Filter for under-indexed pages with latent demand (opportunity_gap > 0.5).
# 4. Assign operational reason codes based on age, rank position, and word count.
# 5. Export the prioritized top 50 refresh candidates to 'work/outputs/refresh_queue_sample.csv'.
# 6. Display the top 15 priority candidates.
"""),

    nbf.v4.new_markdown_cell("""## 4. Executive Summary & Presentation Takeaways
* **Guide for your write-up:**
  * Summarize how your model beat naive heuristics.
  * State the final $R^2$, RMSE reduction, and Spearman rank correlation.
  * Explain how editorial teams should use the refresh queue tomorrow.""")
]

nb3_path = os.path.join(NOTEBOOKS_DIR, '03_evaluation_and_conclusions.ipynb')
with open(nb3_path, 'w', encoding='utf-8') as f:
    nbf.write(nb3, f)
print(f"Created template: {nb3_path}")
print("All 3 blank template notebooks successfully created.")
