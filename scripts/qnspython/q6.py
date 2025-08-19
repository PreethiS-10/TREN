import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
import os
import pickle

def predict_export_risk_elasticnet(df, age_col='median_age', export_col='Exports of goods and services (% of GDP)', cache_path="models/export_risk_elasticnet.pkl"):
    # Ensure models directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Load cached results if available
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            result_df = pickle.load(f)
        return result_df

    # Drop missing values
    sub_df = df.dropna(subset=[age_col, export_col])

    if sub_df.empty:
        print("No valid data for ElasticNet prediction.")
        return pd.DataFrame()

    # Features and target
    X = sub_df[[age_col]]
    y = sub_df[export_col]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elastic Net with CV
    model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                         cv=5,
                         random_state=42)
    model.fit(X_scaled, y)

    results = []
    for _, row in sub_df.iterrows():
        country = row['Country']
        curr_age = row[age_col]

        entry = {
            'Country': country,
            'CurrentMedianAge': curr_age,
            'Export_pct_GDP': row[export_col]
        }

        # Predict exports at +5, +10, +15 years
        for inc in [5, 10, 15]:
            future_age = curr_age + inc
            future_age_scaled = scaler.transform([[future_age]])
            predicted_export = model.predict(future_age_scaled)[0]

            entry[f'MedianAge+{inc}yrs'] = future_age
            entry[f'Predicted_Export_pct_{inc}yrs'] = predicted_export

        results.append(entry)

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('Predicted_Export_pct_15yrs', ascending=False).head(10)

    # Save the result dataframe as a pickle file
    with open(cache_path, 'wb') as f:
        pickle.dump(result_df, f)

    return result_df
