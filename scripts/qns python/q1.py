import pandas as pd

YEAR = 2026  # The year for analysis

# Load your master data (contains export values, GDP, etc.)
master_df = pd.read_csv("../../data/processed_master_df.csv")  # update filename as needed

def compute_trade_dependency_and_gdp_impact(master_df, year=YEAR):
    results = []

    for country in master_df['Country'].unique():
        df_year = master_df[(master_df['Country'] == country) & (master_df['Year'] == year)]
        if df_year.empty:
            continue

        # Sum total exports; then group by partner country if you have that info
        # Here we assume export_value is partner-level
        partner_exports = df_year.groupby('partnerISO')['export_value'].sum() if 'partnerISO' in df_year.columns else None

        if partner_exports is None or partner_exports.empty:
            continue

        total_exports = partner_exports.sum()
        top_partner = partner_exports.idxmax()
        top_partner_exports = partner_exports.max()
        dependency_index = top_partner_exports / total_exports if total_exports > 0 else 0

        # GDP
        gdp = df_year['GDP (current US$)'].iloc[0]

        # Simulate GDP impact: 40% reduction in exports to top partner
        export_loss = top_partner_exports * 0.4
        gdp_impact_pct = (export_loss / gdp) * 100 if gdp else None

        results.append({
            "Country": country,
            "Year": year,
            "Top Partner": top_partner,
            "Dependency Index": dependency_index,
            "GDP (current US$)": gdp,
            "Top Partner Exports (US$)": top_partner_exports,
            "Export Loss (US$)": export_loss,
            "Simulated GDP Impact (%)": gdp_impact_pct
        })

    results_df = pd.DataFrame(results)
    top_3_vulnerable = results_df.sort_values("Dependency Index", ascending=False).head(3)
    return top_3_vulnerable

# Run the analysis
top3_df = compute_trade_dependency_and_gdp_impact(master_df, year=YEAR)
print(top3_df)
