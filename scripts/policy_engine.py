import pandas as pd

ALLOWED_COUNTRIES = [
    "India", "USA", "Russia", "France", "Germany", "Italy", "China", "Japan", "Argentina", "Portugal", "Spain",
    "Croatia", "Belgium", "Australia", "Pakistan", "Afghanistan", "Israel", "Iran", "Iraq", "Bangladesh",
    "Sri Lanka", "Canada", "UK", "Sweden", "Saudi Arabia"
]

def load_data(master_csv="data/processed_master_df.csv"):
    df = pd.read_csv(master_csv)
    df = df[df['Country'].isin(ALLOWED_COUNTRIES)]
    return df

def policy_recommendations(row):
    recs = []
    if row.get('Poverty headcount ratio at $3.00 a day (2021 PPP) (% of population)', 0) > 15:
        recs.append("Expand social protection and targeted poverty-reduction programs.")
    if row.get('gdp_growth_annual_pct', 100) < 2:
        recs.append("Promote economic diversification, innovation, and SME development for higher growth.")
    if row.get('Trade (% of GDP)', 0) > 70:
        recs.append("Diversify export markets and encourage import substitution to reduce trade vulnerability.")
    if row.get('total_disaster_damage', 0) > 1_000_000_000:
        recs.append("Invest in disaster-resilient infrastructure and early warning systems.")
    if row.get('external_debt_stocks_gni_pct', 0) > 60:
        recs.append("Strengthen fiscal sustainability and manage external debt prudently.")
    if row.get('inflation_consumer_prices_annual_pct', 0) > 5:
        recs.append("Enhance monetary policy frameworks to control inflation.")
    if row.get('Gini index', 0) > 40:
        recs.append("Promote inclusive growth through progressive taxation and better access to education.")
    if row.get('unemployment_total_pct', 0) > 10:
        recs.append("Expand job creation initiatives and vocational training.")
    return recs

def poverty_impact_of_gdp_growth(current_poverty, gdp_growth, delta_growth):
    elasticity = -0.7
    new_poverty = current_poverty + delta_growth * elasticity
    return max(new_poverty, 0)

def trade_volatility_impact_on_gdp_growth(current_gdp_growth, current_trade_pct, delta_trade):
    trade_elasticity = 0.03
    change = -delta_trade * trade_elasticity
    new_gdp_growth = current_gdp_growth + change
    return new_gdp_growth

def inflation_impact_on_poverty(current_poverty, current_inflation, delta_inflation):
    inf_elasticity = 0.2
    new_poverty = current_poverty + delta_inflation * inf_elasticity
    return max(new_poverty, 0)

def disaster_impact_on_gdp_growth(current_gdp_growth, current_disaster_damage, delta_damage):
    dmg_elasticity = -0.15
    change = (delta_damage / 1_000_000_000) * dmg_elasticity
    new_gdp_growth = current_gdp_growth + change
    return new_gdp_growth

def analyze_country(df, country):
    sub = df[df['Country'] == country]
    if sub.empty:
        return None, [], {}
    latest = sub.sort_values('Year').iloc[-1]
    recs = policy_recommendations(latest)

    # Prepare key current stats
    stats = {
        "GDP growth (%)": latest.get('gdp_growth_annual_pct', 0),
        "Poverty headcount ratio at $3.00 a day (2021 PPP) (% of population)": latest.get('Poverty headcount ratio at $3.00 a day (2021 PPP) (% of population)', 0),
        "Trade (% of GDP)": latest.get('Trade (% of GDP)', 0),
        "Inflation (%)": latest.get('inflation_consumer_prices_annual_pct', 0),
        "Disaster damage ($)": latest.get('total_disaster_damage', 0)
    }
    return latest, recs, stats
