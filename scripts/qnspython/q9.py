import pandas as pd

def generate_resilience_recommendations(exports_path="data/processed_exports_full.csv"):
    df = pd.read_csv(exports_path)
    latest_year = df['Year'].max()
    df_year = df[df['Year'] == latest_year]
    countries = df_year['Country'].unique()

    recommendations = []
    for country in countries:
        total_import = df_year[df_year['Country'] == country]['export_value'].sum()
        partners = (
            df_year[df_year['Country'] == country]
            .groupby('partnerDesc')['export_value'].sum()
            .sort_values(ascending=False)
        )
        concentration = (partners.head(3).sum()) / total_import if total_import > 0 else 0
        current_major_partners = set(partners.head(5).index)
        global_top_exporters = (
            df_year.groupby('Country')['export_value'].sum().sort_values(ascending=False).index
        )
        new_partners = [p for p in global_top_exporters if p not in current_major_partners and p != country][:3]

        recommendations.append({
            'Country': country,
            'Concentration_Top3': concentration,
            'Recommended_New_Partners': new_partners
        })

    rec_df = pd.DataFrame(recommendations)
    return rec_df
