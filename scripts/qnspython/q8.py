import pandas as pd

def run_trade_route_impact(exports_path="data/processed_exports_full.csv",
                           gdp_path="data/processed_master_df.csv"):
    df = pd.read_csv(exports_path)
    gdp_df = pd.read_csv(gdp_path)

    latest_year = df['Year'].max()
    df_year = df[df['Year'] == latest_year]

    mutual_df = (
        df_year.groupby(['Country', 'partnerDesc'])['export_value']
        .sum()
        .reset_index()
    )
    mutual_df['pair'] = mutual_df.apply(lambda row: tuple(sorted([row['Country'], row['partnerDesc']])), axis=1)
    bilateral = (
        mutual_df.groupby('pair')['export_value']
        .sum()
        .reset_index()
        .rename(columns={'export_value': 'total_bilateral_trade'})
    )
    top_pairs = bilateral.sort_values('total_bilateral_trade', ascending=False).head(5)

    results = []
    for _, row in top_pairs.iterrows():
        country_a, country_b = row['pair']
        bilateral_trade = row['total_bilateral_trade']

        gdp_a = gdp_df[(gdp_df['Country'] == country_a) & (gdp_df['Year'] == latest_year)]['GDP (current US$)'].values[0]
        gdp_b = gdp_df[(gdp_df['Country'] == country_b) & (gdp_df['Year'] == latest_year)]['GDP (current US$)'].values

        impact_a_pct = (bilateral_trade / gdp_a) * 100
        impact_b_pct = (bilateral_trade / gdp_b) * 100

        results.append({
            'Country_A': country_a,
            'Country_B': country_b,
            'Bilateral_Trade_Value': bilateral_trade,
            'GDP_A': gdp_a,
            'GDP_B': gdp_b,
            'GDP_A_Loss_pct': impact_a_pct,
            'GDP_B_Loss_pct': impact_b_pct
        })

    impact_df = pd.DataFrame(results)

    return top_pairs, impact_df
