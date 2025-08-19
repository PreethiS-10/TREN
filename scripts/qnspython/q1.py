import pandas as pd

def compute_top_trade_dependency_and_gdp_impact(
    exports_path="data/processed_exports_full.csv",
    master_path="data/processed_master_df.csv"
):
    exports_df = pd.read_csv(exports_path)
    master_df = pd.read_csv(master_path)

    # 1. Compute TDI for latest available year
    latest_year = exports_df['Year'].max()

    tdi_list = []
    for country, grp in exports_df[exports_df['Year'] == latest_year].groupby('Country'):
        total_exports = grp['export_value'].sum()
        partner_exports = grp.groupby('partnerDesc')['export_value'].sum()
        if not partner_exports.empty:
            top_partner = partner_exports.idxmax()
            top_value = partner_exports.max()
        else:
            top_partner = None
            top_value = 0
        tdi = top_value / total_exports if total_exports > 0 else 0
        tdi_list.append({'Country': country, 'Year': latest_year, 'TopPartner': top_partner,
                         'TDI': tdi, 'TotalExports': total_exports, 'TopPartnerExports': top_value})

    tdi_df = pd.DataFrame(tdi_list)

    # 2. Merge with master for GDP/Trade data
    merged = tdi_df.merge(master_df, on=['Country', 'Year'], how='left')

    # 3. Find top 3 vulnerable countries (highest TDI)
    top3 = merged.sort_values('TDI', ascending=False).head(3)

    # 4. Simulate GDP impact for 40% drop in top partner imports
    def simulate_gdp_impact(row):
        export_gdp_pct = row.get('Trade (% of GDP)', 0)
        gdp = row.get('GDP (current US$)', 0)
        tdi = row['TDI']
        gdp_loss_pct = tdi * export_gdp_pct * 0.40 / 100  # decimal fraction
        gdp_loss_usd = gdp_loss_pct * gdp
        return pd.Series({'GDP_Loss_pct': gdp_loss_pct * 100, 'GDP_Loss_usd': gdp_loss_usd})

    gdp_impact_df = top3.apply(simulate_gdp_impact, axis=1)
    top3 = pd.concat([top3, gdp_impact_df], axis=1)

    return top3[['Country', 'TopPartner', 'TDI', 'GDP_Loss_pct', 'GDP_Loss_usd']]


