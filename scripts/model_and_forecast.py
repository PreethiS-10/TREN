import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# Load full dataframe and remove exact duplicate columns
df = pd.read_csv("../data/processed_master_df.csv")
df = df.loc[:, ~df.columns.duplicated()]

targets = [
    'GDP growth (annual %)',
    'Poverty headcount ratio at $3.00 a day (2021 PPP) (% of population)',
    'Trade (% of GDP)'
]

model_cols = [
    'GDP (current US$)',
    'GDP per capita (current US$)',
    'Imports of goods and services (% of GDP)',
    'Inflation, consumer prices (annual %)',
    'total_disaster_damage',
    'current_account_balance_gdp_pct',
    'external_debt_stocks_gni_pct',
    'population_total',
    'unemployment_total_pct',
    'urban_population_pct',
    'life_expectancy_total_years',
    'Gini index'
]

# Keep only relevant columns and drop rows with missing values
required_cols = model_cols + targets + ['Country', 'Year']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column in data: {col}")

df_model = df[required_cols].dropna()

# Handle outliers by capping extreme values
for col in model_cols + targets:
    df_model[col] = df_model[col].clip(lower=df_model[col].quantile(0.01), upper=df_model[col].quantile(0.99))

# Prepare output directory
output_model_dir = "./models/"
os.makedirs(output_model_dir, exist_ok=True)

# Define scenario feature modifiers
scenario_feature_modifiers = {
    "baseline": None,
    "increased_social_spending": {
        'urban_population_pct': 1.15,
        'life_expectancy_total_years': 1 + 0.15 * 0.5,
        'Gini index': 1 - 0.15 * 0.3,
        'unemployment_total_pct': 1 - 0.15 * 0.2,
    },
    "trade_diversification": {
        'total_disaster_damage': 0.8
    },
    "global_crisis": {
        'total_disaster_damage': 2.0,
        'unemployment_total_pct': 1.5,
        'GDP (current US$)': 0.8,
        'GDP per capita (current US$)': 0.85
    }
}

def train_and_save_model(df, target, scenario_name):
    print(f"\nTraining {target} for scenario: {scenario_name}")
    df_adj = df.copy()

    # Apply scenario feature modifiers only to non-target columns
    if scenario_name != "baseline":
        modifiers = scenario_feature_modifiers.get(scenario_name, {})
        for feature, modifier in modifiers.items():
            if feature in df_adj.columns and feature not in targets:
                df_adj[feature] = df_adj[feature] * modifier

    # Exclude all targets from features to prevent data leakage
    features = [col for col in model_cols if col not in targets]
    all_cols = features + ['Year', target]
    if any(col not in df_adj.columns for col in all_cols):
        print(f"Missing columns for {target} under {scenario_name}. Skipping.")
        return

    df_clean = df_adj[all_cols].dropna().drop_duplicates().reset_index(drop=True)
    if df_clean.shape[0] < 30:
        print(f"Too few rows ({df_clean.shape[0]}) for reliable training of {target} under {scenario_name}. Skipping.")
        return

    X = df_clean[features + ['Year']].copy()
    y = df_clean[target].copy()

    # Log-transform skewed features
    skewed_cols = ['GDP (current US$)', 'GDP per capita (current US$)', 'population_total', 'total_disaster_damage']
    for col in skewed_cols:
        if col in X.columns:
            X.loc[:, col] = np.log1p(X[col].clip(lower=0))

    # Log-transform skewed target if necessary
    if target == 'Poverty headcount ratio at $3.00 a day (2021 PPP) (% of population)':
        y = np.log1p(y.clip(lower=0))
        print(f"Applied log-transformation to {target}")

    # Initialize models list
    models = []

    # ---- 1. Random Forest with Tuning ----
    rf_params = {
        'n_estimators': [100],
        'max_depth': [3, 5],
        'min_samples_leaf': [5, 10]
    }
    rf_model = RandomForestRegressor(random_state=42)
    rf_search = GridSearchCV(rf_model, rf_params, cv=5, scoring='r2')
    try:
        rf_search.fit(X, y)
        rf_best = rf_search.best_estimator_
        rf_r2 = cross_val_score(rf_best, X, y, cv=5, scoring='r2').mean()
        print(f"Random Forest features: {rf_best.feature_names_in_}")
        models.append(('random_forest', rf_best, rf_r2))
    except Exception as e:
        print(f"Error training Random Forest for {target} ({scenario_name}): {e}")

    # ---- 2. XGBoost with Tuning ----
    xgb_params = {
        'max_depth': [3, 5],
        'n_estimators': [100],
        'learning_rate': [0.01, 0.1]
    }
    xgb_model = XGBRegressor(random_state=42)
    xgb_search = GridSearchCV(xgb_model, xgb_params, cv=5, scoring='r2')
    try:
        xgb_search.fit(X, y)
        xgb_best = xgb_search.best_estimator_
        xgb_r2 = cross_val_score(xgb_best, X, y, cv=5, scoring='r2').mean()
        print(f"XGBoost features: {xgb_best.feature_names_in_}")
        models.append(('xgboost', xgb_best, xgb_r2))
    except Exception as e:
        print(f"Error training XGBoost for {target} ({scenario_name}): {e}")

    if not models:
        print(f"No models successfully trained for {target} ({scenario_name}). Skipping.")
        return

    # Print R² scores
    for model_type, _, r2 in models:
        print(f"{model_type} R²: {r2:.3f}")

    # Choose best model (highest R²)
    selected_model_type, best_model, best_r2 = max(models, key=lambda x: x[2])

    # Save best model with simplified filename
    filename = f"{output_model_dir}{scenario_name}_{target.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct').replace('$','')}_model.pkl"
    with open(filename, "wb") as f:
        pickle.dump(best_model, f)

    print(f"✅ Saved {selected_model_type} model for '{target}' ({scenario_name}) with R² = {best_r2:.3f}")

scenarios = ["baseline", "increased_social_spending", "trade_diversification", "global_crisis"]
for scenario in scenarios:
    for target in targets:
        train_and_save_model(df_model, target, scenario)

print("All scenario-specific models trained and saved.")

# --- Make Predictions for 2030 ---
if not df[df['Year'] == 2024].empty:
    df_2030 = df[df['Year'] == 2024].copy()
    df_2030['Year'] = 2030
else:
    raise ValueError("No data available for Year 2024")

def predict_2030_for_country(df_2030, country, scenario, target):
    # Load saved model
    filename = f"{output_model_dir}{scenario}_{target.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct').replace('$','')}_model.pkl"
    if not os.path.exists(filename):
        print(f"Model not found for {scenario} - {target}")
        return None

    with open(filename, "rb") as f:
        model = pickle.load(f)

    # Extract country data for 2030
    df_country = df_2030[df_2030['Country'] == country]
    if df_country.empty:
        print(f"No 2030 data for country {country}")
        return None

    # Exclude all targets from features
    features = [col for col in model_cols if col not in targets]
    X = df_country[features + ['Year']].copy()

    # Apply scenario modifiers to 2030 input, excluding targets
    if scenario != "baseline":
        modifiers = scenario_feature_modifiers.get(scenario, {})
        for feature, modifier in modifiers.items():
            if feature in X.columns and feature not in targets:
                X.loc[:, feature] = X[feature] * modifier

    # Log-transform skewed features
    skewed_cols = ['GDP (current US$)', 'GDP per capita (current US$)', 'population_total', 'total_disaster_damage']
    for col in skewed_cols:
        if col in X.columns:
            X.loc[:, col] = np.log1p(X[col].clip(lower=0))

    # Match feature order
    if hasattr(model, "feature_names_in_"):
        expected_features = [f for f in model.feature_names_in_ if f in X.columns]
        if len(expected_features) != len(model.feature_names_in_):
            print(f"Warning: Missing features for {scenario} - {target}. Expected: {model.feature_names_in_}, Got: {X.columns}")
            return None
        X = X[expected_features]

    # Predict and reverse log-transformation for poverty target
    try:
        pred = model.predict(X)[0]
        if target == 'Poverty headcount ratio at $3.00 a day (2021 PPP) (% of population)':
            pred = np.expm1(pred)
        return pred
    except Exception as e:
        print(f"Prediction error for {country} - {scenario} - {target}: {e}")
        return None

results = []
countries = df_2030['Country'].unique()

for country in countries:
    for scenario in scenarios:
        row = {"Country": country, "Scenario": scenario}
        for target in targets:
            val = predict_2030_for_country(df_2030, country, scenario, target)
            row[target] = val
        results.append(row)

# Save predictions to CSV
df_results = pd.DataFrame(results)
df_results.to_csv("../data/2030_predictions_by_scenario.csv", index=False)
print("Saved 2030 predictions per country per scenario.")