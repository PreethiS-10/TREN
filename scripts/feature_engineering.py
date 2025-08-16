import pandas as pd
import numpy as np

def compute_TDI(df):
    required = {'Country', 'Year', 'partnerDesc', 'export_value'}
    if not required.issubset(df.columns):
        print("Missing columns:", required - set(df.columns))
        return None

    df['export_value'] = pd.to_numeric(df['export_value'], errors='coerce')
    df = df.dropna(subset=['Country', 'Year', 'partnerDesc', 'export_value'])

    tdi_list = []
    for (country, year), grp in df.groupby(['Country', 'Year']):
        total_exports = grp['export_value'].sum()
        partner_exports = grp.groupby('partnerDesc')['export_value'].sum()
        if partner_exports.empty:
            print(f"No partner exports for {country} {year}")
            top_partner_export = 0
        else:
            top_partner_export = partner_exports.max()

        tdi = top_partner_export / total_exports if total_exports > 0 else 0
        tdi_list.append({'Country': country, 'Year': year, 'trade_dependency_index': tdi})
        #print(tdi_list)

    tdi_df = pd.DataFrame(tdi_list)
    #print(tdi_df)
    return tdi_df

def compute_resilience_score(df):
    def resilience_row(row):
        score = 0
        score += row.get('gdp_growth_annual_pct', 0)
        poverty = row.get('Poverty headcount ratio at $3.00 a day (2021 PPP) (% of population)', 0)
        score += max(0, 100 - poverty)
        debt = row.get('external_debt_stocks_gni_pct', 0)
        score += max(0, 100 - debt)
        damage = row.get('total_disaster_damage', 0) * 1e6
        gdp = row.get('GDP (current US$)', 1)
        shock_pct = 100 * damage / gdp if gdp > 0 else 0
        score -= shock_pct
        return score

    df['resilience_score'] = df.apply(resilience_row, axis=1)
    return df

def compute_spending_efficiency(df):
    df['spending_efficiency'] = df['total_disaster_damage'] / df['disaster_count']
    df['spending_efficiency'] = df['spending_efficiency'].replace([np.inf, -np.inf], np.nan)
    df['spending_efficiency'] = df['spending_efficiency'].fillna(0)
    return df

def compute_shock_impact(df):
    damage_million = df.get('total_disaster_damage', 0) * 1e6
    gdp = df.get('GDP (current US$)', 1)
    df['shock_impact_score'] = 100 * damage_million / gdp
    df['shock_impact_score'] = df['shock_impact_score'].replace([np.inf, -np.inf], np.nan)
    df['shock_impact_score'] = df['shock_impact_score'].fillna(0)
    return df

def add_feature_engineering(master_df, export_df, import_df, tdi_df=None):
    # Merge or compute Trade Dependency Index

    if 'trade_dependency_index' in master_df.columns:
        master_df = master_df.drop(columns=['trade_dependency_index'])
    master_df = master_df.merge(tdi_df, on=['Country', 'Year'])

    master_df = compute_resilience_score(master_df)
    master_df = compute_spending_efficiency(master_df)
    master_df = compute_shock_impact(master_df)

    # print("tdi:",tdi_df['trade_dependency_index'])
    # print("master:",master_df['trade_dependency_index'])

    return master_df

def main():
    master_path = '../data/processed_master_df.csv'
    exports_path = '../data/processed_exports_full.csv'
    imports_path = '../data/processed_imports_full.csv'

    master_df = pd.read_csv(master_path)
    export_df = pd.read_csv(exports_path)
    import_df = pd.read_csv(imports_path)

    tdi_df = compute_TDI(export_df)

    master_df = add_feature_engineering(master_df, export_df, import_df, tdi_df=tdi_df)

    # Overwrite the original master file with features added
    master_df.to_csv(master_path, index=False)
    print(f"Overwrote master file with engineered features: {master_path}")

if __name__ == "__main__":
    main()
