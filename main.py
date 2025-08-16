import pandas as pd
import streamlit as st
import os
from scripts.utils import (
    load_preprocessed_master,
    load_exports_full,
    load_imports_full,
    load_scenario_forecasts
)
import scripts.policy_engine as policy_engine

st.set_page_config(page_title="TREN Analytics Dashboard", layout="wide")

def fix_types_for_arrow(df):
    """Fix DataFrame column types to avoid pyarrow serialization errors in Streamlit"""
    for col in df.columns:
        if col == 'Year':
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        elif df[col].dtype == 'object' or col == 'Country':
            df[col] = df[col].astype(str)
    return df
def sanitize_country_name(name):
    # Remove problematic characters and replace spaces with underscores
    return name.replace(" ", "_").replace(",", "").replace(".", "")


@st.cache_data
def load_data():
    master_df = load_preprocessed_master()
    exports_df = load_exports_full()
    imports_df = load_imports_full()
    return master_df, exports_df, imports_df

master_df, exports_df, imports_df = load_data()
scenario_results_df = load_scenario_forecasts()

OUTPUT_DIR = "scripts/outputs"

tab1, tab2, tab3, tab4, tab5, tab6, tab7= st.tabs([
    "Preprocessed Master DF",
    "Exports DF",
    "Imports DF",
    "2030 Scenario Forecasts",
    "Country Vulnerability Insights",
    "Scenario Visualizations",
    "Policy and what ifs"
])

with tab1:
    st.title("Preprocessed Master DataFrame")
    country = st.selectbox("Select Country:", sorted(master_df['Country'].unique()), key='tab1_country')
    years_available = master_df[master_df['Country'] == country]['Year'].dropna().unique()
    year = st.selectbox("Select Year:", sorted(years_available), key='tab1_year')
    subset = master_df[(master_df['Country'] == country) & (master_df['Year'] == year)]
    if subset.empty:
        st.write("No data available for the selected country-year")
    else:
        st.dataframe(fix_types_for_arrow(subset.T))

with tab2:
    st.title("Export Trade Data (Detailed Partner-Level)")
    country_exp = st.selectbox("Select Country for Exports:", sorted(exports_df['Country'].unique()), key='tab2_country')
    years_exp = exports_df[exports_df['Country'] == country_exp]['Year'].dropna().unique()
    year_exp = st.selectbox("Select Year for Exports:", sorted(years_exp), key='tab2_year')
    subset_exp = exports_df[(exports_df['Country'] == country_exp) & (exports_df['Year'] == year_exp)]
    if subset_exp.empty:
        st.write("No export data available for the selected country-year")
    else:
        st.dataframe(fix_types_for_arrow(subset_exp))

with tab3:
    st.title("Import Trade Data (Detailed Partner-Level)")
    country_imp = st.selectbox("Select Country for Imports:", sorted(imports_df['Country'].unique()), key='tab3_country')
    years_imp = imports_df[imports_df['Country'] == country_imp]['Year'].dropna().unique()
    year_imp = st.selectbox("Select Year for Imports:", sorted(years_imp), key='tab3_year')
    subset_imp = imports_df[(imports_df['Country'] == country_imp) & (imports_df['Year'] == year_imp)]
    if subset_imp.empty:
        st.write("No import data available for the selected country-year")
    else:
        st.dataframe(fix_types_for_arrow(subset_imp))

with tab4:
    st.title("2030 Scenario Forecasts")
    country_scenario = st.selectbox(
        "Select Country:",
        sorted(scenario_results_df['Country'].unique()),
        key='tab4_country'
    )
    years_available_scenario = scenario_results_df[scenario_results_df['Country'] == country_scenario]['Scenario'].unique()
    scenario_selected = st.selectbox(
        "Select Scenario:",
        sorted(years_available_scenario),
        key='tab4_scenario'
    )
    subset_scenario = scenario_results_df[
        (scenario_results_df['Country'] == country_scenario) &
        (scenario_results_df['Scenario'] == scenario_selected)
    ]
    if subset_scenario.empty:
        st.write("No prediction data available for the selected country and scenario.")
    else:
        st.dataframe(fix_types_for_arrow(subset_scenario))

with tab5:
    st.title("Country Vulnerability Insights")
    insight_country = st.selectbox(
        "Select Country for Insights:",
        sorted(scenario_results_df['Country'].unique()),
        key='tab5_country'
    )
    sanitized_name = sanitize_country_name(insight_country)
    insight_path = os.path.join(OUTPUT_DIR, f"{sanitized_name}_top_vulnerabilities.txt")
    st.write(f"Looking for file: {insight_path}")  # debug print, you can remove later

    if os.path.exists(insight_path):
        with open(insight_path, "r", encoding="utf8") as f:
            insights = f.readlines()
        st.markdown("#### Top Vulnerabilities")
        for insight in insights:
            st.write(insight.strip())
    else:
        st.write("No insights file available for the selected country.")

with tab6:
    st.title("Scenario Visualizations (PNG Images)")

    vis_type = st.selectbox(
        "Visualization Type:",
        [
            "Heatmap GDP Growth",
            "Heatmap Poverty",
            "Heatmap Trade",
            "Shock Map GDP Growth",
            "Shock Map Poverty",
            "Shock Map Trade",
            "Trade Network"
        ],
        key='tab6_type'
    )

    file_map = {
        "Heatmap GDP Growth": "heatmap_gdp_growth_annual_pct.png",
        "Heatmap Poverty": "heatmap_Poverty_headcount_ratio_at_$3.00_a_day_2021_PPP_%_of_population.png",
        "Heatmap Trade": "heatmap_Trade_%_of_GDP.png",
        "Shock Map GDP Growth": "shock_map_gdp_growth_annual_pct_global_crisis.png",
        "Shock Map Poverty": "shock_map_Poverty_headcount_ratio_at_$3.00_a_day_(2021_PPP)_(%_of_population)_global_crisis.png",
        "Shock Map Trade": "shock_map_Trade_(%_of_GDP)_global_crisis.png"
    }

    if vis_type == "Trade Network":
        tn_country = st.selectbox(
            "Country:",
            sorted(scenario_results_df['Country'].unique()),
            key="tab6_tn_country"
        )

        # Scan for all split subplot trade network images for this country
        sanitized_country = sanitize_country_name(tn_country)
        matching_files = [
            f for f in os.listdir(OUTPUT_DIR)
            if f.startswith(f"{sanitized_country}_") and f.endswith("_trade_split_subplot.png")
        ]

        available_years = []
        for f in matching_files:
            parts = f.replace(".png", "").split("_")
            # Find the year in the filename: expects ..._{year}_trade_split_subplot.png
            # parts[-4] accommodates cases where country names have underscores
            # Find the first part from the end that is a 4-digit year
            year = None
            for part in reversed(parts):
                if part.isdigit() and len(part) == 4:
                    year = int(part)
                    break
            if year:
                available_years.append(year)
        available_years = sorted(set(available_years))

        if available_years:
            tn_year = st.selectbox("Year:", available_years, key="tab6_tn_year")
            target_file = f"{sanitized_country}_{tn_year}_trade_split_subplot.png"
            target_path = os.path.join(OUTPUT_DIR, target_file)
            if os.path.exists(target_path):
                st.image(target_path, use_container_width=True, caption=f"{tn_country} {tn_year} Trade Network (Exports & Imports)")
            else:
                st.warning(f"No trade network image found for {tn_country} in {tn_year}.")
        else:
            st.warning(f"No trade network images found for {tn_country}.")
    else:
        vis_file = file_map.get(vis_type, "")
        vis_path = os.path.join(OUTPUT_DIR, vis_file)
        if os.path.exists(vis_path):
            st.image(vis_path, use_container_width=True)
        else:
            st.warning("No visualization image found for selected type.")
 # The above file should be named policy_engine.py and be in your working directory

with tab7:
    st.title("Policy Recommendations & What-If Scenario Simulator")

    # Load data
    df = policy_engine.load_data()  # you can specify the CSV path if needed

    # Country dropdown
    country = st.selectbox("Select Country:", policy_engine.ALLOWED_COUNTRIES, key="tab7_country")

    # Analyze country
    latest, recs, stats = policy_engine.analyze_country(df, country)

    if latest is None:
        st.error("No data available for this country.")
    else:
        st.header(f"Policy Recommendations for {country} (Year: {int(latest['Year'])})")
        if recs:
            for r in recs:
                st.markdown(f"- {r}")
        else:
            st.write("No acute vulnerabilities detected; maintain current resilience strategy.")

        st.subheader("Key Indicators")
        for k, v in stats.items():
            if isinstance(v, float):
                st.write(f"{k}: {v:,.2f}")
            else:
                st.write(f"{k}: {v}")

        st.subheader("What-If Scenario Simulations")

        # What-if: GDP Growth
        gdp_change = st.slider("Increase GDP growth by (% points)", min_value=0.0, max_value=5.0, step=0.1, value=1.0)
        poverty_new = policy_engine.poverty_impact_of_gdp_growth(stats["Poverty headcount ratio at $3.00 a day (2021 PPP) (% of population)"],
                                                                 stats["GDP growth (%)"], gdp_change)
        st.write(f"→ If GDP growth increases by {gdp_change:.1f} pp, poverty could drop to **{poverty_new:.2f}%**")

        # What-if: Trade (% of GDP)
        trade_reduce = st.slider("Reduce Trade (% of GDP) by", min_value=0.0, max_value=float(stats["Trade (% of GDP)"]), step=0.5, value=10.0)
        gdp_growth_new = policy_engine.trade_volatility_impact_on_gdp_growth(stats["GDP growth (%)"],
                                                                             stats["Trade (% of GDP)"], trade_reduce)
        st.write(f"→ If Trade (% of GDP) drops by {trade_reduce:.1f} pp, GDP growth may rise to **{gdp_growth_new:.2f}%**")

        # What-if: Reduce Inflation
        inf_reduce = st.slider("Reduce inflation (%) by", min_value=0.0, max_value=float(stats["Inflation (%)"]), step=0.1, value=2.0)
        poverty_inf = policy_engine.inflation_impact_on_poverty(stats["Poverty headcount ratio at $3.00 a day (2021 PPP) (% of population)"],
                                                                stats["Inflation (%)"], -inf_reduce)
        st.write(f"→ If inflation drops by {inf_reduce:.1f} pp, poverty could drop to **{poverty_inf:.2f}%**")

        # What-if: Reduce Disaster Damage
        disaster_reduce = st.slider("Reduce disaster damage by ($ billion)", min_value=0.0,
                                    max_value=stats["Disaster damage ($)"]/1e9 if stats["Disaster damage ($)"] > 0 else 1.0,
                                    step=0.1, value=1.0)
        gdp_growth_disaster = policy_engine.disaster_impact_on_gdp_growth(stats["GDP growth (%)"],
                                                                          stats["Disaster damage ($)"], -disaster_reduce*1e9)
        st.write(f"→ If disaster damage falls by ${disaster_reduce:.1f}B, GDP growth may rise to **{gdp_growth_disaster:.2f}%**")

    st.info("Elasticity coefficients are illustrative. Refine with local data/statistical analysis for accurate scenario modeling.")
