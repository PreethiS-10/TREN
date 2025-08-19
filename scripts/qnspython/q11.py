import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import os
import pickle

def run_ridge_forecast(df, future_year=2030, alpha=1.0, cache_path="models/ridge_forecast.pkl"):
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Load cached results if available
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            result_df = pickle.load(f)
        return result_df

    latest_year = df['Year'].max()
    countries = df['Country'].unique()
    results = []

    def ridge_trend_forecast(series, base_year, future_year, adjustment_pct=0.0, alpha=1.0):
        if len(series) < 3 or future_year <= base_year:
            return np.nan
        y = series.sort_values('Year')
        X = y['Year'].values.reshape(-1, 1)
        y_values = y['Value'].values
        model = Ridge(alpha=alpha)
        model.fit(X, y_values)
        steps = future_year - base_year
        forecast = model.predict([[future_year]])[0]
        adjusted_forecast = forecast * ((1 + adjustment_pct) ** steps)
        return adjusted_forecast

    for country in countries:
        out = {'Country': country}

        gdp_series = df[(df['Country'] == country) & (~df['GDP (current US$)'].isnull())][['Year', 'GDP (current US$)']]
        gdp_series = gdp_series.rename(columns={'GDP (current US$)': 'Value'})

        pov_col = 'Poverty headcount ratio at $3.00 a day (2021 PPP) (% of population)'
        pov_series = df[(df['Country'] == country) & (~df[pov_col].isnull())][['Year', pov_col]]
        pov_series = pov_series.rename(columns={pov_col: 'Value'})

        out['GDP_2030_best'] = ridge_trend_forecast(gdp_series, latest_year, future_year, adjustment_pct=0.015, alpha=alpha)
        out['GDP_2030_worst'] = ridge_trend_forecast(gdp_series, latest_year, future_year, adjustment_pct=-0.015, alpha=alpha)

        out['Poverty_2030_best'] = ridge_trend_forecast(pov_series, latest_year, future_year, adjustment_pct=-0.02, alpha=alpha)
        out['Poverty_2030_worst'] = ridge_trend_forecast(pov_series, latest_year, future_year, adjustment_pct=0.01, alpha=alpha)

        results.append(out)

    result_df = pd.DataFrame(results)

    # Save to pickle
    with open(cache_path, 'wb') as f:
        pickle.dump(result_df, f)

    return result_df
