import pandas as pd
import numpy as np
import re

def load_and_merge():
    # Load core economic indicators
    economic = pd.read_csv('../data/core_economic_indicators.csv', index_col=None)
    economic.rename(columns={"Country Name": "Country"}, inplace=True)
    economic['Country'] = economic['Country'].str.strip()
    year_cols = [col for col in economic.columns if re.match(r'\d{4} \[YR\d+\]', col)]
    id_cols = ['Country', 'Country Code', 'Series Name', 'Series Code']
    economic_long = economic.melt(
        id_vars=id_cols,
        value_vars=year_cols,
        var_name='Year_col',
        value_name='Value'
    )
    economic_long['Value'] = economic_long['Value'].replace(['..', '', 'NA'], np.nan)
    economic_long['Value'] = pd.to_numeric(economic_long['Value'], errors='coerce')
    economic_long['Year'] = economic_long['Year_col'].str.extract(r'(\d{4})').astype(int)
    economic_pivot = economic_long.pivot_table(
        index=['Country', 'Year'],
        columns='Series Name',
        values='Value'
    ).reset_index()

    # Load crop and livestock
    crop = pd.read_csv('../data/crop_and_livestock.csv', index_col=None)
    crop.rename(columns={"Area": "Country", "Year": "Year", "Element": "Element", "Value": "Value"}, inplace=True)
    crop['Country'] = crop['Country'].str.strip()

    # Load disaster data
    disaster = pd.read_csv('../data/disasters.csv', index_col=None)
    disaster.rename(columns={"Country": "Country"}, inplace=True)
    disaster['Country'] = disaster['Country'].str.strip()
    disaster['Year'] = disaster['Start Year']
    # Get unique country-year pairs from your main data (or construct manually)
    all_countries = disaster['Country'].unique()
    all_years = disaster['Start Year'].unique()
    full_index = pd.MultiIndex.from_product([all_countries, all_years], names=['Country', 'Year'])
    full_df = pd.DataFrame(index=full_index).reset_index()

    # Group original disaster data
    disaster_agg = disaster.groupby(['Country', 'Year']).agg({
        "Total Damage ('000 US$)": 'sum',
        'Total Deaths': 'sum',
        'DisNo.': 'count'
    }).reset_index()

    # Merge to ensure all country-year pairs exist
    disaster_agg = full_df.merge(disaster_agg, on=['Country', 'Year'], how='left')

    # Fill NaNs (i.e., countries with no disasters that year)
    disaster_agg["Total Damage ('000 US$)"] = disaster_agg["Total Damage ('000 US$)"].fillna(0)
    disaster_agg["Total Deaths"] = disaster_agg["Total Deaths"].fillna(0)
    disaster_agg["DisNo."] = disaster_agg["DisNo."].fillna(0)

    # Rename columns
    disaster_agg.rename(columns={
        "Total Damage ('000 US$)": 'total_disaster_damage',
        'Total Deaths': 'total_disaster_deaths',
        'DisNo.': 'disaster_count',
    }, inplace=True)
    # Load resilience
    raw_res = pd.read_csv('../data/resiliance.csv', index_col=None)
    year_cols_res = [col for col in raw_res.columns if re.match(r'\d{4} \[YR\d+\]', col)]
    id_cols_res = ['Country Name', 'Country Code', 'Series Name', 'Series Code']
    res_long = raw_res.melt(id_vars=id_cols_res, value_vars=year_cols_res, var_name='Year_col', value_name='Value')
    res_long['Year'] = res_long['Year_col'].str.extract(r'(\d{4})').astype(int)
    res_long['Country'] = res_long['Country Name'].str.strip()
    res_long['Value'] = pd.to_numeric(res_long['Value'], errors='coerce')
    res_pivot = res_long.pivot_table(index=['Country', 'Year'], columns='Series Name', values='Value').reset_index()
    rename_res = {
        'Current account balance (% of GDP)': 'current_account_balance_gdp_pct',
        'External debt stocks (% of GNI)': 'external_debt_stocks_gni_pct',
        'Foreign direct investment, net inflows (% of GDP)': 'fdi_net_inflows_gdp_pct'
    }
    res_pivot = res_pivot.rename(columns=rename_res)
    for col in rename_res.values():
        if col in res_pivot.columns:
            res_pivot[col] = pd.to_numeric(res_pivot[col], errors='coerce')

    # Load welfare
    welfare = pd.read_csv('../data/social_and_welfare.csv', index_col=None)
    welfare.rename(columns={"Country Name": "Country"}, inplace=True)
    welfare['Country'] = welfare['Country'].str.strip()
    year_cols_welfare = [col for col in welfare.columns if re.match(r'\d{4} \[YR\d+\]', col)]
    id_cols_welfare = ['Country', 'Country Code', 'Series Name', 'Series Code']
    welfare_long = welfare.melt(id_vars=id_cols_welfare, value_vars=year_cols_welfare, var_name='Year_col', value_name='Value')
    welfare_long['Value'] = welfare_long['Value'].replace(['..', '', 'NA'], np.nan)
    welfare_long['Value'] = pd.to_numeric(welfare_long['Value'], errors='coerce')
    welfare_long['Year'] = welfare_long['Year_col'].str.extract(r'(\d{4})').astype(int)
    welfare_pivot = welfare_long.pivot_table(index=['Country', 'Year'], columns='Series Name', values='Value').reset_index()
    rename_welfare = {
        'Urban population (% of total population)': 'urban_population_pct',
        'Population growth (annual %)': 'population_growth_annual_pct',
        'Unemployment, total (% of total labor force) (modeled ILO estimate)': 'unemployment_total_pct',
        'Life expectancy at birth, total (years)': 'life_expectancy_total_years'
    }
    welfare_pivot = welfare_pivot.rename(columns=rename_welfare)
    for col in rename_welfare.values():
        if col in welfare_pivot.columns:
            welfare_pivot[col] = pd.to_numeric(welfare_pivot[col], errors='coerce')

    # Load employment/unemployment
    emp_unemp = pd.read_csv('../data/Employment_Unemployment.csv', index_col=None)
    emp_long = emp_unemp.melt(id_vars=['Country Name', 'Country Code', 'Series Name'], var_name='Year', value_name='Value')
    emp_long['Year'] = emp_long['Year'].str.extract(r'(\d{4})').astype(float).astype('Int64')
    emp_long['Country Name'] = emp_long['Country Name'].str.strip()
    unemp_total = emp_long[emp_long['Series Name'] == 'Unemployment, total (% of total labor force) (modeled ILO estimate)']
    unemp_total = unemp_total.rename(columns={'Country Name': 'Country', 'Value': 'unemployment_rate'})[['Country', 'Year', 'unemployment_rate']]

    # Load population and demographics
    population = pd.read_csv('../data/population_and_demographics.CSV', index_col=False)
    population.rename(columns={"Area": "Country", "Year": "Year", "Value": "population_total"}, inplace=True)
    population['Country'] = population['Country'].str.strip()
    population_total = population[population['Element'] == 'Total Population - Both sexes'][['Country', 'Year', 'population_total']]

    # Load and concatenate Export CSVs
    export1 = pd.read_csv('../data/2000-2012_Export.CSV', encoding='latin1', index_col=False)
    export1.rename(columns={"reporterDesc": "Country", "refYear": "Year", "fobvalue": "export_value"}, inplace=True)
    export1['Country'] = export1['Country'].str.strip()
    export1['Year'] = export1['Year'].astype(int)

    export2 = pd.read_csv('../data/2013-2024_Export.CSV', encoding='latin1', index_col=False)
    export2.rename(columns={"reporterDesc": "Country", "refYear": "Year", "fobvalue": "export_value"}, inplace=True)
    export2['Country'] = export2['Country'].str.strip()
    export2['Year'] = export2['Year'].astype(int)
    print("export before merge : ",export2.columns.to_list())
    print(export2.head(1))
    export_full = pd.concat([export1, export2], ignore_index=True)
    print("export after merge : ",export_full.columns.to_list())
    print(export_full.head(1))
    # Load and concatenate Import CSVs
    import_1 = pd.read_csv('../data/2000-2012_Import.CSV', encoding='latin1', index_col=False)
    import_1.rename(columns={"reporterDesc": "Country", "refYear": "Year", "cifvalue": "import_value"}, inplace=True)
    import_1['Country'] = import_1['Country'].str.strip()
    import_1['Year'] = import_1['Year'].astype(int)

    import_2 = pd.read_csv('../data/2013-2024_Import.CSV', encoding='latin1', index_col=None)
    import_2.rename(columns={"reporterDesc": "Country", "refYear": "Year", "cifvalue": "import_value"}, inplace=True)
    import_2['Country'] = import_2['Country'].str.strip()
    import_2['Year'] = import_2['Year'].astype(int)

    import_full = pd.concat([import_1, import_2], ignore_index=True)

    # Sequential merges
    dfs = [
        economic_pivot,
        crop,
        disaster_agg,
        res_pivot,
        welfare_pivot,
        unemp_total,
        population_total
    ]

    master_df = dfs[0]
    for df in dfs[1:]:
        master_df = master_df.merge(df, on=['Country', 'Year'], how='outer', suffixes=('', '_dup'))
        dup_cols = [c for c in master_df.columns if c.endswith('_dup')]
        if dup_cols:
            master_df.drop(columns=dup_cols, inplace=True)

    # Aggregate export/import sums
    export_agg = export_full.groupby(['Country', 'Year'])['export_value'].sum().reset_index()
    import_agg = import_full.groupby(['Country', 'Year'])['import_value'].sum().reset_index()

    master_df = master_df.merge(export_agg, on=['Country', 'Year'], how='outer')
    master_df = master_df.merge(import_agg, on=['Country', 'Year'], how='outer')

    # Clean and convert types
    master_df['Country'] = master_df['Country'].astype(str).str.strip()
    master_df = master_df[master_df['Year'].notna()]
    master_df['Year'] = master_df['Year'].astype(int)

    selected_countries = [
        "India", "USA", "Russia", "France", "Germany", "Italy", "China", "Japan", "Argentina",
        "Portugal", "Spain", "Croatia", "Belgium", "Australia", "Pakistan", "Afghanistan",
        "Israel", "Iran", "Iraq", "Bangladesh", "Sri Lanka", "Canada", "UK", "Sweden", "Saudi Arabia"
    ]
    print(export_full.head(1))
    export_df = export_full[export_full['Country'].isin(selected_countries)].copy()
    import_df = import_full[import_full['Country'].isin(selected_countries)].copy()
    print(export_df.head(1))
    return master_df, export_df, import_df
#
# def compute_indices(master_df, export_df, import_df):
#     # Trade Dependency Index
#     tdi_list = []
#     if {'Country', 'Year', 'partnerISO', 'export_value'}.issubset(export_df.columns):
#         for (country, year), grp in export_df.groupby(['Country', 'Year']):
#             total_exports = grp['export_value'].sum()
#             partner_exports = grp.groupby('partnerISO')['export_value'].sum()
#             if partner_exports.empty:
#                 continue
#             top_partner_exports = partner_exports.max()
#             tdi = top_partner_exports / total_exports if total_exports > 0 else np.nan
#             tdi_list.append({'Country': country, 'Year': year, 'trade_dependency_index': tdi})
#         tdi_df = pd.DataFrame(tdi_list)
#         master_df = master_df.merge(tdi_df, on=['Country', 'Year'], how='left')
#
#     # Resilience Score
#     def resilience_row(row):
#         score = 0
#         score += row.get('gdp_growth_annual_pct', 0)
#         score += max(0, 100 - row.get('Poverty headcount ratio at $3.00 a day (2021 PPP) (% of population)', 0))
#         score += max(0, 100 - row.get('external_debt_stocks_gni_pct', 0))
#         shock_pct = 0
#         if not pd.isna(row.get('total_disaster_damage')) and not pd.isna(row.get('GDP (current US$)')):
#             shock_pct = 100 * (row['total_disaster_damage'] * 1e6) / row['GDP (current US$)'] if row['GDP (current US$)'] > 0 else 0
#         score -= shock_pct
#         return score
#     master_df['resilience_score'] = master_df.apply(resilience_row, axis=1)
#
#     # Spending Efficiency
#     master_df['spending_efficiency'] = master_df['total_disaster_damage'] / master_df['disaster_count']
#     master_df['spending_efficiency'] = master_df['spending_efficiency'].replace([np.inf, -np.inf], np.nan)
#
#     # Shock Impact Score
#     master_df['shock_impact_score'] = 100 * (master_df['total_disaster_damage'] * 1e6) / master_df['GDP (current US$)']
#     master_df['shock_impact_score'] = master_df['shock_impact_score'].replace([np.inf, -np.inf], np.nan)
#
#     return master_df

def fill_na(df):
    """Impute NAs: mode for categoricals, median for numericals, but skip filling if all values are NaN."""
    for col in df.columns:
        if df[col].dtype == 'O':
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])
            # else: all NA; leave as is
        else:
            # Only fill if at least one non-NA
            if df[col].notna().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
            # else: all NA; leave as is (to avoid RuntimeWarning)
    return df

def impute_and_normalize(df):
    # Impute NAs robustly (using improved fill_na)
    df = fill_na(df)
    # Normalize monetary values thousands to millions
    for col in ['total_disaster_damage', 'export_value', 'import_value']:
        if col in df.columns:
            df[col] = df[col] / 1000.0
    # Fix negative values
    for col in ['export_value', 'import_value']:
        if col in df.columns:
            df.loc[df[col] < 0, col] = 0
    return df

def main():
    master_df, export_df, import_df = load_and_merge()
    master_df = impute_and_normalize(master_df)
    export_df = impute_and_normalize(export_df)
    import_df = impute_and_normalize(import_df)
    #master_df = compute_indices(master_df, export_df, import_df)

    master_df.to_csv('../data/processed_master_df.csv', index=False)
    export_df.to_csv('../data/processed_exports_full.csv', index=False)
    import_df.to_csv('../data/processed_imports_full.csv', index=False)

    print("Completed data processing and computation of composite indices.")

if __name__ == "__main__":
    main()