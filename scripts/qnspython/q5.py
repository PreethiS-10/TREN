import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import itertools
import os, pickle

def grid_search_xgb_cv(years, rates, tscv, param_grid):
    min_mse = np.inf
    best_params = None
    best_model = None
    # Iterate all combinations
    for params in param_grid:
        model = xgb.XGBRegressor(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            random_state=42,
            objective='reg:squarederror',
            verbosity=0
        )
        cv_scores = cross_val_score(model, years, rates, cv=tscv, scoring='neg_mean_squared_error')
        avg_mse = -np.mean(cv_scores)
        if avg_mse < min_mse:
            min_mse = avg_mse
            best_params = params
            best_model = model
    return best_model, best_params, min_mse


def cross_validate_youth_unemployment_xgb_minerror(df, scenario_increase=0.15, cv_splits=3, model_dir="models"):
    countries = df['Country'].unique()
    results = []

    # ensure model storage dir exists
    os.makedirs(model_dir, exist_ok=True)

    # Hyperparameter grid
    param_grid = [
        {'n_estimators': n, 'learning_rate': lr, 'max_depth': md}
        for n, lr, md in itertools.product([50, 100, 200], [0.05, 0.1, 0.2], [2, 3, 4])
    ]

    for country in countries:
        model_path = os.path.join(model_dir, f"{country.replace(' ','_')}_xgb.pkl")

        sub = df[(df['Country'] == country) & (~df['unemployment_total_pct'].isnull())]
        if sub.shape[0] < (cv_splits + 1):
            continue

        years = sub['Year'].values.reshape(-1, 1)
        rates = sub['unemployment_total_pct'].values

        if os.path.exists(model_path):
            # Load pre-trained model
            with open(model_path, "rb") as f:
                saved = pickle.load(f)
            best_model = saved["model"]
            best_params = saved["params"]
            min_mse = saved["min_mse"]
        else:
            # Train + save model
            tscv = TimeSeriesSplit(n_splits=cv_splits)
            best_model, best_params, min_mse = grid_search_xgb_cv(years, rates, tscv, param_grid)
            best_model.fit(years, rates)

            with open(model_path, "wb") as f:
                pickle.dump({"model": best_model, "params": best_params, "min_mse": min_mse}, f)

        # Prediction
        pred_2030 = best_model.predict(np.array([[2030]]))
        pred_2030_scenario = pred_2030 * (1 + scenario_increase)

        results.append({
            'Country': country,
            'Min_CV_MSE': min_mse,
            'Best_Params': best_params,
            'Predicted_Unemployment_2030': float(pred_2030_scenario)
        })

    out_df = pd.DataFrame(results)
    risky = out_df[out_df['Predicted_Unemployment_2030'] > 10].sort_values('Predicted_Unemployment_2030', ascending=False)
    return risky


