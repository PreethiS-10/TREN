# TREN — Trade Resilience & Economic Networks
Analytics, forecasting, visualization, and optimization to help 25 nations prepare for 2030 under crises and policy scenarios.

## Overview
TREN is an end-to-end analytics stack that ingests multi-domain data (trade, macroeconomy, disasters, agriculture, welfare, demographics), engineers resilience features, trains scenario-specific predictive models, generates dashboards and maps, and solves a network optimization problem to minimize GDP loss under single-point failures.

The system supports:
- Country-level 2000–2024 historical integration
- 2030 forecasts under multiple scenarios
- Trade network analysis and disruption simulations
- Policy insights and “what-if” tools
- Optimization of new trade/infrastructure links

This README explains how the solution is structured and how it was implemented, with references to the provided code modules.

## Repository Structure (key modules)
- preprocess.py
  - Loads and merges raw data across domains (economic, disasters, resilience, welfare, labor, population).
  - Normalizes, imputes, and aggregates exports/imports (2000–2012 and 2013–2024).
  - Outputs:
    - data/processed_master_df.csv
    - data/processed_exports_full.csv
    - data/processed_imports_full.csv
- feature_engineering.py
  - compute_TDI: Trade Dependency Index (TDI) from partner-level exports.
  - compute_resilience_score: Composite of growth, poverty, debt, and disaster shock burden.
  - compute_spending_efficiency: Disaster damage per event as a proxy.
  - compute_shock_impact: Disaster damages scaled by GDP.
  - add_feature_engineering: Merges engineered features back into the master dataset.
- model_and_forecast.py
  - Trains scenario-specific models for each target:
    - GDP growth (annual %)
    - Poverty headcount ratio at $3.00/day (2021 PPP) (% of population)
    - Trade (% of GDP)
  - Models used: RandomForestRegressor, XGBRegressor, each tuned via GridSearchCV; best model per target-scenario is saved.
  - Scenarios:
    - baseline
    - increased_social_spending
    - trade_diversification
    - global_crisis
  - Applies scenario feature modifiers on non-target features only (to avoid leakage).
  - Handles transformations (e.g., log for skew) and caps outliers.
  - Saves per-scenario best models to ./models and produces 2030 predictions per country to data/2030_predictions_by_scenario.csv.
- visualize_and_insights.py
  - Heatmaps: Scenario vs. Country for each target variable.
  - Trade networks: Split subplot per country-year (exports vs imports) using NetworkX; saves PNGs to outputs/.
  - Shock maps: Geospatial choropleths by scenario with GeoPandas and a world shapefile.
  - Vulnerability insights: Extracts top vulnerabilities per country and saves text files (e.g., "_top_vulnerabilities.txt").
- policy_engine.py
  - Country filters (25 specified nations).
  - Heuristic policy recommendations from current indicators (poverty, growth, trade exposure, disaster damages, debt, inflation, inequality, unemployment).
  - “What-if” elasticities for poverty vs. growth/inflation and GDP growth vs. disasters/trade volatility.
  - analyze_country returns latest snapshot, recommended policies, and key stats.
- utils.py
  - Convenience loading functions for processed master, exports, imports, and scenario forecasts CSVs.

## Data Pipeline
1) Ingestion & merging (preprocess.py)
- Economic indicators (wide-to-long melt, pivot to Country–Year).
- FAO crop/livestock, disasters (aggregated damages, deaths, counts), resilience (current account, external debt, FDI), welfare (urbanization, unemployment, life expectancy).
- Employment/unemployment (unemployment rate), population totals.
- Trade (exports/imports) merged across two time segments into partner-level datasets.
- Final merges stitched on Country–Year with suffix cleanups and normalization.

2) Imputation & normalization (preprocess.py)
- fill_na: Mode for categoricals, median for numericals (robust to all-NaN columns).
- impute_and_normalize: Scales monetary values to manageable magnitudes and sets negatives to zero for trade values.

3) Feature engineering (feature_engineering.py)
- TDI: Max partner export share per Country–Year.
- Resilience score: Growth + (100–poverty) + (100–debt) – disaster shock(% of GDP).
- Spending efficiency: Damage per disaster.
- Shock impact score: Damage scaled by GDP.

4) Modeling & Scenario Forecasts (model_and_forecast.py)
- Targets: GDP growth, Poverty (log-transformed for modeling), Trade % of GDP.
- Features: Macroeconomic, trade, disasters, demographics, inequality, labor, etc.
- Cleaned training matrix with outlier capping; log1p for skewed inputs.
- Scenario modifiers (e.g., higher social spending improves outcomes; global crisis worsens damages and unemployment).
- Cross-validated model selection (Random Forest vs XGBoost) by R².
- Saves best models and generates 2030 predictions for each country-scenario combination.

5) Visualizations & Insights (visualize_and_insights.py)
- Heatmaps: Scenario-wise comparisons across countries.
- Trade network split subplots: Country-level export and import directed graphs (edge widths by trade value).
- Shock maps: Geospatial rendering for each scenario/value.
- Top vulnerabilities: Highest poverty, lowest growth, highest trade exposure per scenario written to outputs/.

6) Policy analysis (policy_engine.py)
- Rule-based recommendations keyed to critical thresholds (poverty, growth, trade exposure, disaster damages, debt, inflation, inequality, unemployment).
- What-if functions to quantify impacts of user-driven inputs (e.g., change in GDP growth or disaster damages).

## How to Run
1) Prepare data
- Place raw CSVs under data/:
  - core_economic_indicators.csv
  - crop_and_livestock.csv
  - disasters.csv
  - resiliance.csv
  - social_and_welfare.csv
  - Employment_Unemployment.csv
  - population_and_demographics.CSV
  - 2000-2012_Export.CSV, 2013-2024_Export.CSV
  - 2000-2012_Import.CSV, 2013-2024_Import.CSV
  - World shapefile for maps (ne_110m_admin_0_countries.shp and related files)

2) Build processed datasets
- Run preprocess.py
  - Outputs:
    - data/processed_master_df.csv
    - data/processed_exports_full.csv
    - data/processed_imports_full.csv

3) Engineer features
- Run feature_engineering.py
  - Computes TDI, resilience_score, spending_efficiency, and shock_impact_score.
  - Overwrites processed_master_df.csv with new columns.

4) Train models & generate 2030 predictions
- Run model_and_forecast.py
  - Trains per-scenario models, saves best models in ./models/
  - Creates data/2030_predictions_by_scenario.csv

5) Generate visualizations and insights
- Run visualize_and_insights.py
  - Saves heatmaps, shock maps, trade network subplots, and text insights under outputs/

6) Launch Streamlit dashboard (if applicable)
- Integrate functions from utils.py, policy_engine.py, and qnspython scripts into your Streamlit app to interactively:
  - Explore processed data
  - Visualize networks and scenarios
  - Run policy what-ifs
  - Solve the optimization (Q13)

## Key Design Choices
- Conservative, leak-free modeling: Scenario modifiers apply only to features, never to targets.
- Robustness: Extensive type coercion, imputation, outlier capping, and log transforms for skew.
- Modularity: Clear separation of preprocessing, feature engineering, modeling, and visualization.
- Reproducibility: Outputs written to predictable paths; models saved with scenario-target names.
- Performance: Cross-validation for model selection, caching for dashboard workloads.

## Outputs
- Processed datasets:
  - data/processed_master_df.csv
  - data/processed_exports_full.csv
  - data/processed_imports_full.csv
- Models:
  - ./models/__model.pkl
- Predictions:
  - data/2030_predictions_by_scenario.csv
- Visuals & insights:
  - outputs/heatmap_*.png
  - outputs/shock_map_*_global_crisis.png
  - outputs/__trade_split_subplot.png
  - outputs/_top_vulnerabilities.txt

## Extending the System
- Add new scenarios by defining feature modifiers (model_and_forecast.py).
- Enrich resilience_score with more components (e.g., fiscal space, health capacity).
- Integrate partner-specific embargo shock propagation in network analysis.
- Replace rule-based policy heuristics with trained prescriptive models.

## Notes & Assumptions
- Certain columns are renamed/standardized; ensure consistent headers in raw files.
- Disaster damages in preprocess are in '000 USD in source and later normalized/scaled; check scaling before interpretation.
- Geospatial names must match shapefile naming (NAME/ADMIN) for correct joins in maps.

## Additional Code Modules Overview

The following modules provide specialized analyses, simulations, forecasting, optimization, and visualization functions to enhance the TREN analytics capabilities:

### q1.py — Top Trade Dependency & GDP Impact
- Calculates the Trade Dependency Index (TDI) for countries based on their largest export partner share for the latest data year.
- Identifies the top 3 countries most vulnerable due to dependency.
- Simulates GDP impact under a scenario of a 40% drop in imports from the top trade partner.
- Returns vulnerability rankings and estimated GDP losses.

### q2.py — Cascading Shock from China Export Drop
- Simulates economic impact on trading partners if China’s exports drop by a specified percentage (default 25%) in a shock year (default 2028).
- Calculates GDP percentage and USD loss estimates for affected countries.
- Highlights the top 5 countries most impacted by a cascading shock originating from China.

### q3.py — Drought Impact on Exports (Statistical Prediction)
- Uses historical export data from countries to predict expected export values for the year 2030 assuming no drought.
- Estimates export losses during a 3-year drought period with a defined yield drop (default 20%).
- Identifies countries most vulnerable to drought-induced export reductions.

### q4.py — Agricultural Import Partner Dependency & Food Security Risk
- Analyzes agricultural import dependencies by countries on their top partners for a specific year.
- Calculates the import share concentration from top partners and flags countries with high dependency (>60%).
- Models food security risk based on potential import losses if top partners impose export bans.

### q5.py — Youth Unemployment Forecast via XGBoost
- Trains country-specific time series XGBoost regression models on youth unemployment data.
- Performs grid search cross-validation to optimize hyperparameters.
- Predicts 2030 youth unemployment rates with an increase scenario factor (default +15%).
- Flags countries at risk with unemployment predicted above 10%.

### q6.py — Export Risk Prediction with ElasticNet Regression
- Predicts exports as a percentage of GDP using median age as a feature.
- Uses ElasticNet regression with cross-validation for regularization.
- Projects export percentages at future median ages (+5, +10, +15 years).
- Provides ranked export risk predictions across countries.

### q7.py — Trade Network Analysis & Disruption Simulation
- Constructs a directed graph of top global exporters and their trade partners based on export values.
- Computes network centrality measures like degree and betweenness centrality.
- Visualizes trade networks and shows the impact of removing the most central country on network connectivity and trade volume.
- Outputs metrics on trade volume lost and network fragmentation.

### q8.py — Bilateral Trade Route Impact on GDP
- Identifies top bilateral trade pairs by total export value.
- Calculates the estimated GDP loss percentage for each country if bilateral trade is disrupted.
- Highlights the top 5 highest-value bilateral trade pairs and their economic impact.

### q9.py — Trade Partner Diversification Recommendations
- Calculates export concentration among top partners to identify trade dependency.
- Suggests new trade partners from the list of global top exporters to diversify and reduce concentration risk.
- Recommends diversification policies tailored to each country.

### q10.py — Shock Scenario Forecast Predictions (GDP, Poverty, Unemployment)
- Uses Decision Tree regressors to forecast GDP, poverty, and unemployment trends to 2030.
- Applies scenario multipliers simulating disaster shocks and trade war impacts.
- Outputs predicted values under combined shock scenarios for respective indicators.

### q11.py — Ridge Regression Forecasts with Best and Worst Case Scenarios
- Applies Ridge regression to model economic indicators’ trajectories.
- Produces forecasts for GDP and poverty for 2030 under optimistic (growth) and pessimistic (decline) adjustment assumptions.
- Stores predictions for integrated scenario analysis.

### q12.py — Moving Average Forecasting of Economic Factors
- Uses a moving average method to predict economic indicators for 2030.
- Covers a broad set of macroeconomic and social factors.
- Serves as a simple baseline forecasting method complementing advanced models.

### q13.py — Optimization of Trade and Infrastructure Links
- Formulates and solves a linear programming problem to minimize the maximum GDP loss linked to trade failures.
- Builds optimal new trade/infrastructure links constrained by budget and maximum distance.
- Includes geopolitical and distance restrictions on possible links.
- Produces visualizations of recommended new trade networks and their expected resilience benefits.

