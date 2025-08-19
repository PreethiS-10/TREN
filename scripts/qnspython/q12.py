import pandas as pd
import os
import pickle

def run_moving_average_forecast(df, target_year=2030, window=3, cache_path="models/moving_average_forecast.pkl"):
    # Ensure models directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Load from cache if exists
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            result_df = pickle.load(f)
        return result_df

    factor_cols = [
        'Exports of goods and services (% of GDP)',
        'GDP growth (annual %)',
        'GDP per capita (current US$)',
        'Inflation, consumer prices (annual %)',
        'current_account_balance_gdp_pct',
        'external_debt_stocks_gni_pct',
        'fdi_net_inflows_gdp_pct',
        'Gini index',
        'life_expectancy_total_years',
        'Poverty headcount ratio at $3.00 a day (2021 PPP) (% of population)',
        'unemployment_total_pct',
    ]

    def moving_average_forecast(series):
        if len(series) >= window:
            return series.tail(window).mean()
        elif len(series) > 0:
            return series.mean()
        else:
            return None

    results = []
    for country in df['Country'].unique():
        res = {'Country': country}
        country_df = df[df['Country'] == country]
        for factor in factor_cols:
            series = country_df[['Year', factor]].dropna().sort_values('Year')
            if not series.empty:
                res[f'{factor}_2030'] = moving_average_forecast(series[factor])
            else:
                res[f'{factor}_2030'] = None
        results.append(res)

    result_df = pd.DataFrame(results)

    # Save to cache
    with open(cache_path, 'wb') as f:
        pickle.dump(result_df, f)

    return result_df
