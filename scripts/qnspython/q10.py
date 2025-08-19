import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import os
import pickle

def run_shock_scenario_predictions(df, cache_path="models/shock_scenario_predictions.pkl"):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Load cached results if available
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            result_df = pickle.load(f)
        return result_df

    DISASTER_GDP_DROP = 0.05
    DISASTER_POVERTY_INCREASE = 0.05
    DISASTER_UNEMPLOYMENT_INCREASE = 0.04

    TRADE_WAR_GDP_DROP = 0.03
    TRADE_WAR_POVERTY_INCREASE = 0.03
    TRADE_WAR_UNEMPLOYMENT_INCREASE = 0.03

    countries = df['Country'].unique()
    results = []

    for country in countries:
        out = {'Country': country}
        # GDP
        gdp_sub = df[(df['Country']==country) & (~df['GDP (current US$)'].isnull())][['Year','GDP (current US$)']]
        if len(gdp_sub) >= 5:
            X = gdp_sub['Year'].values.reshape(-1,1)
            y = gdp_sub['GDP (current US$)'].values
            reg = DecisionTreeRegressor()
            reg.fit(X, y)
            gdp_2030 = reg.predict([[2030]])
            gdp_2030 = gdp_2030 * (1 - DISASTER_GDP_DROP) * (1 - TRADE_WAR_GDP_DROP)
            out['GDP_2030'] = gdp_2030
        else:
            out['GDP_2030'] = np.nan

        # Poverty
        pov_col = 'Poverty headcount ratio at $3.00 a day (2021 PPP) (% of population)'
        pov_sub = df[(df['Country']==country) & (~df[pov_col].isnull())][['Year',pov_col]]
        if len(pov_sub) >= 5:
            X = pov_sub['Year'].values.reshape(-1,1)
            y = pov_sub[pov_col].values
            reg = DecisionTreeRegressor()
            reg.fit(X, y)
            pov_2030 = reg.predict([[2030]])
            pov_2030 = pov_2030 * (1 + DISASTER_POVERTY_INCREASE) * (1 + TRADE_WAR_POVERTY_INCREASE)
            out['Poverty_2030'] = pov_2030
        else:
            out['Poverty_2030'] = np.nan

        # Unemployment
        unemp_col = 'unemployment_total_pct'
        unemp_sub = df[(df['Country']==country) & (~df[unemp_col].isnull())][['Year',unemp_col]]
        if len(unemp_sub) >= 5:
            X = unemp_sub['Year'].values.reshape(-1,1)
            y = unemp_sub[unemp_col].values
            reg = DecisionTreeRegressor()
            reg.fit(X, y)
            unemp_2030 = reg.predict([[2030]])
            unemp_2030 = unemp_2030 * (1 + DISASTER_UNEMPLOYMENT_INCREASE) * (1 + TRADE_WAR_UNEMPLOYMENT_INCREASE)
            out['Unemployment_2030'] = unemp_2030
        else:
            out['Unemployment_2030'] = np.nan

        results.append(out)

    result_df = pd.DataFrame(results)

    # Save to pickle in models directory
    with open(cache_path, 'wb') as f:
        pickle.dump(result_df, f)

    return result_df
