import pandas as pd

def simulate_cascading_china_export_shock(
    exports_path="data/processed_exports_full.csv",
    master_path="data/processed_master_df.csv",
    shock_year=2028,
    china_export_drop_pct=0.25
):
    exports_df = pd.read_csv(exports_path)
    master_df = pd.read_csv(master_path)

    # Filter China's exports in the shock year (2028 or latest if not present)
    if shock_year not in exports_df['Year'].values:
        shock_year = exports_df['Year'].max()  # fallback to latest available
    china_exports = exports_df[(exports_df['Year'] == shock_year) & (exports_df['Country'] == "China")]

    # Aggregate exports from China to each partner
    partner_exports = china_exports.groupby('partnerDesc')['export_value'].sum().reset_index()
    partner_exports.rename(columns={'partnerDesc': 'Country', 'export_value': 'ExportsFromChina'}, inplace=True)

    # Merge to get GDP and trade share
    merged = partner_exports.merge(master_df[master_df['Year'] == shock_year], on='Country', how='left')

    # If 'Trade (% of GDP)' is missing, estimate impact as China's share of total imports
    merged['GDP_Loss_pct'] = (merged['ExportsFromChina'] / (merged['GDP (current US$)'] + 1e-9)) * china_export_drop_pct * 100
    # Calculate USD impact
    merged['GDP_Loss_usd'] = merged['GDP_Loss_pct'] / 100 * merged['GDP (current US$)']

    # Find top 5 countries by GDP percentage loss
    top5 = merged.sort_values('GDP_Loss_pct', ascending=False).head(5)

    return top5[['Country', 'ExportsFromChina', 'GDP_Loss_pct', 'GDP_Loss_usd']]
