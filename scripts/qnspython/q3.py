import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def drought_export_impact_stat(df, countries=None, drought_years=[2028, 2029, 2030], drought_yield_drop=0.20):
    if countries is None:
        countries = df['Country'].unique()
    results = []
    for country in countries:
        sub = df[(df['Country'] == country) & (~df['export_value'].isnull())]
        if len(sub) < 3:  # Not enough data for regression
            continue
        X = sub['Year'].values.reshape(-1, 1)
        y = sub['export_value'].values
        model = LinearRegression()
        model.fit(X, y)
        # Predict exports for 2030 (no drought)
        export_2030_pred = model.predict(np.array([[2030]]))
        # Cumulative drought impact over 3 years
        drought_factor = (1 - drought_yield_drop) ** len(drought_years)
        projected_export_2030 = export_2030_pred * drought_factor
        export_loss_pct = 100 * (1 - drought_factor)
        # Most recent disaster damage
        damage = sub['total_disaster_damage'].iloc[-1] if 'total_disaster_damage' in sub.columns else np.nan
        results.append({
            'Country': country,
            'Export_2030_base': export_2030_pred,
            'Projected_export_2030': projected_export_2030,
            'Export_Loss_pct': export_loss_pct,
            'total_disaster_damage': damage
        })
    result_df = pd.DataFrame(results)
    top10 = result_df.sort_values('Export_Loss_pct', ascending=False).head(10)
    return top10

