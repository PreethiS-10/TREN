import pandas as pd
import streamlit as st
import os
from scripts.utils import (
    load_preprocessed_master,
    load_exports_full,
    load_imports_full,
    load_scenario_forecasts
)
import numpy as np
import scripts.policy_engine as policy_engine
from scripts.qnspython.q1 import compute_top_trade_dependency_and_gdp_impact
from scripts.qnspython.q2 import simulate_cascading_china_export_shock
from scripts.qnspython.q3 import drought_export_impact_stat
from scripts.qnspython.q4 import agri_import_partner_dependency, model_food_security_risk
from scripts.qnspython.q5 import cross_validate_youth_unemployment_xgb_minerror
from scripts.qnspython.q6 import predict_export_risk_elasticnet
from scripts.qnspython.q7 import run_trade_network_analysis
from scripts.qnspython.q8 import run_trade_route_impact
from scripts.qnspython.q9 import generate_resilience_recommendations
from scripts.qnspython.q10 import run_shock_scenario_predictions
from scripts.qnspython.q11 import run_ridge_forecast
from scripts.qnspython.q12 import run_moving_average_forecast
from scripts.qnspython.q13 import run_optimal_trade_links



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

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14, tab15, tab16, tab17,tab18,tab19,tab20= st.tabs([
    "Preprocessed Master DF",
    "Exports DF",
    "Imports DF",
    "2030 Scenario Forecasts",
    "Country Vulnerability Insights",
    "Scenario Visualizations",
    "Policy and what ifs",
"Question 1",
"Question 2",
"Question 3",
"Question 4",
"Question 5",
"Question 6",
"Question 7",
"Question 8",
"Question 9",
"Question 10",
"Question 11",
"Question 12",
"Question 13"
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

with tab8:
    st.title("Top Trade Dependency and GDP Impact Analysis")

    # Optionally let user specify CSV paths or hardcode them here
    exports_path = "data/processed_exports_full.csv"
    master_path = "data/processed_master_df.csv"

    if st.button("Compute Top Trade Dependency and GDP Impact"):
        result_df = compute_top_trade_dependency_and_gdp_impact(exports_path, master_path)

        if not result_df.empty:
            st.dataframe(result_df)
        else:
            st.write("No data available or no results computed.")

with tab9:
    st.title("Cascading Export Shock from China - Impact Simulation")

    # Input fields for parameters - with default values
    exports_path = st.text_input("Exports CSV Path:", "data/processed_exports_full.csv")
    master_path = st.text_input("Master DF CSV Path:", "data/processed_master_df.csv")
    shock_year = st.number_input("Shock Year:", min_value=2000, max_value=2050, value=2028, step=1)
    china_export_drop_pct = st.slider("China Export Drop Percentage", min_value=0.0, max_value=1.0, value=0.25, step=0.01)

    if st.button("Simulate China Export Shock"):
        result_df = simulate_cascading_china_export_shock(
            exports_path=exports_path,
            master_path=master_path,
            shock_year=shock_year,
            china_export_drop_pct=china_export_drop_pct
        )

        if not result_df.empty:
            # Format numbers for display
            result_df['GDP_Loss_usd'] = result_df['GDP_Loss_usd'].apply(lambda x: f"${x:,.0f}")
            result_df['GDP_Loss_pct'] = result_df['GDP_Loss_pct'].round(4).astype(str) + '%'
            result_df['ExportsFromChina'] = result_df['ExportsFromChina'].apply(lambda x: f"${x:,.0f}")

            st.dataframe(result_df)
        else:
            st.write("No data or results to display.")

with tab10:
    st.title("Drought Export Impact Statistical Analysis")

    # Load master_df once outside or inside this tab depending on your setup
    # If already loaded globally:
    # master_df = your_loaded_master_df
    # If not available, you can reload or cache one-time load

    countries_input = st.text_area("Enter countries separated by commas (leave blank for all):",)
    drought_years_input = st.text_input("Drought Years (comma separated):", "2028,2029,2030")
    drought_yield_drop = st.slider("Drought Yield Drop Fraction", min_value=0.0, max_value=1.0, value=0.20, step=0.01)

    def parse_countries(text):
        if not text.strip():
            return None
        return [c.strip() for c in text.split(",") if c.strip()]

    def parse_years(text):
        try:
            return [int(y.strip()) for y in text.split(",") if y.strip()]
        except:
            return [2028, 2029, 2030]  # default fallback

    countries = parse_countries(countries_input)
    drought_years = parse_years(drought_years_input)
    master_df = pd.read_csv("data/processed_master_df.csv")
    if st.button("Run Drought Export Impact Analysis"):
        result_df = drought_export_impact_stat(
            df=master_df,
            countries=countries,
            drought_years=drought_years,
            drought_yield_drop=drought_yield_drop
        )
        if not result_df.empty:
            # Format outputs for readability
            def safe_currency_format(x):
                if isinstance(x, (list, np.ndarray)):
                    if len(x) > 0:
                        x = x[0]
                    else:
                        return ""
                elif pd.isna(x):
                    return ""
                return f"${x:,.0f}"


            result_df['Projected_export_2030'] = result_df['Projected_export_2030'].apply(safe_currency_format)
            result_df['Export_Loss_pct'] = result_df['Export_Loss_pct'].round(2).astype(str) + '%'
            result_df['total_disaster_damage'] = result_df['total_disaster_damage'].fillna(0).map(lambda x: f"${x:,.0f}")

            st.dataframe(result_df)
        else:
            st.write("No data or insufficient data for analysis.")

with tab11:
    st.title("Agricultural Import Partner Dependency & Food Security Risk")

    import_path_default = "data/processed_imports_full.csv"
    year_selected = st.number_input("Select Year:", min_value=2000, max_value=2050, value=None, step=1)
    n_partners = st.slider("Number of Top Import Partners to Consider:", 1, 10, 3)
    threshold = st.slider("Dependency Threshold (share):", 0.0, 1.0, 0.6, 0.05)

    import_path = st.text_input("Import CSV Path:", import_path_default)

    if st.button("Run Agricultural Dependency and Risk Analysis"):
        dependent_df = agri_import_partner_dependency(import_path, year=year_selected, n_partners=n_partners, threshold=threshold)
        if dependent_df.empty:
            st.write("No countries meet the dependency threshold for the selected parameters.")
        else:
            st.subheader("Countries Highly Dependent on Top Agricultural Import Partners")
            st.dataframe(dependent_df[['Country', 'TopPartners', 'TopPartnerShare']])

            risk_df = model_food_security_risk(dependent_df)
            st.subheader("Food Security Risk Assessment")
            # Format ImportLoss_pct with percentage symbol
            risk_df['ImportLoss_pct'] = (risk_df['ImportLoss_pct']).round(2).astype(str) + '%'
            st.dataframe(risk_df[['Country', 'TopPartners', 'TopPartnerShare', 'ImportLoss_pct', 'RiskFlag']])

with tab12:
    st.title("Youth Unemployment Prediction & Risk Analysis (XGBoost)")

    scenario_increase = st.slider("Scenario Increase in Unemployment (%)", 0.0, 1.0, 0.15, step=0.01)
    cv_splits = st.slider("Cross Validation Splits", 2, 6, 3, step=1)
    model_dir = st.text_input("Model directory path:", "models")

    if st.button("Run Youth Unemployment XGB Risk Prediction"):
        # Assuming master_df is globally loaded already; else, load it here.
        risky_df = cross_validate_youth_unemployment_xgb_minerror(
            df=master_df,
            scenario_increase=scenario_increase,
            cv_splits=cv_splits,
            model_dir=model_dir
        )

        if not risky_df.empty:
            risky_df_display = risky_df.copy()
            risky_df_display['Predicted_Unemployment_2030'] = risky_df_display['Predicted_Unemployment_2030'].round(2)
            risky_df_display['Min_CV_MSE'] = risky_df_display['Min_CV_MSE'].round(4)
            risky_df_display['Best_Params'] = risky_df_display['Best_Params'].apply(lambda p: str(p))

            st.write(
                f"Countries predicted with unemployment > 10% in 2030 under scenario increase of {scenario_increase * 100:.1f}%:")
            st.dataframe(risky_df_display)
        else:
            st.write("No countries predicted with unemployment above threshold under current parameters.")

with tab13:
    st.title("Export Risk Prediction using ElasticNet Model")

    # Assume master_df already loaded globally; otherwise load here
    # Prepare median age and export data aggregated by Country
    median_age_df = master_df.groupby('Country')['life_expectancy_total_years'].median().reset_index()
    median_age_df = median_age_df.rename(columns={'life_expectancy_total_years': 'median_age'})

    export_df = master_df.groupby('Country')['Exports of goods and services (% of GDP)'].median().reset_index()

    country_df = pd.merge(median_age_df, export_df, on='Country', how='inner')

    if st.button("Run ElasticNet Export Risk Prediction"):
        result_df = predict_export_risk_elasticnet(country_df)

        if not result_df.empty:
            df_display = result_df.copy()
            # Format predicted exports to 2 decimals
            for col in ['Predicted_Export_pct_5yrs', 'Predicted_Export_pct_10yrs', 'Predicted_Export_pct_15yrs']:
                df_display[col] = df_display[col].round(2)
            df_display['Export_pct_GDP'] = df_display['Export_pct_GDP'].round(2)

            st.dataframe(df_display[['Country', 'CurrentMedianAge', 'Export_pct_GDP',
                                     'Predicted_Export_pct_5yrs', 'Predicted_Export_pct_10yrs', 'Predicted_Export_pct_15yrs']])
        else:
            st.write("No valid data available for prediction.")

with tab14:
    st.title("Global Trade Network Visualization & Disruption Simulation")

    run_trade_network_analysis("data/processed_exports_full.csv")

with tab15:
    st.title("Top Mutually Beneficial Trade Pairs & Impact of Collapse")

    exports_path = "data/processed_exports_full.csv"
    gdp_path = "data/processed_master_df.csv"

    top_pairs, impact_df = run_trade_route_impact(exports_path, gdp_path)

    st.subheader("Top 5 Mutually Beneficial Trade Pairs")
    # Format trade values for readability
    top_pairs_display = top_pairs.copy()
    top_pairs_display['total_bilateral_trade'] = top_pairs_display['total_bilateral_trade'].map('${:,.0f}'.format)
    st.dataframe(top_pairs_display)

    st.subheader("GDP Impact if Top Trade Route Collapses (Percentage Loss)")
    impact_display = impact_df.copy()
    impact_display['Bilateral_Trade_Value'] = impact_display['Bilateral_Trade_Value'].map('${:,.0f}'.format)
    impact_display['GDP_A_Loss_pct'] = impact_display['GDP_A_Loss_pct'].round(4).astype(str) + '%'
    impact_display['GDP_B_Loss_pct'] = pd.to_numeric(impact_display['GDP_B_Loss_pct'], errors='coerce').round(4).astype(str) + '%'
    st.dataframe(impact_display[['Country_A', 'Country_B', 'Bilateral_Trade_Value', 'GDP_A_Loss_pct', 'GDP_B_Loss_pct']])

with tab16:
    st.title("Resilience Recommendations: Diversifying Trade Partners")

    exports_path = "data/processed_exports_full.csv"

    rec_df = generate_resilience_recommendations(exports_path)

    # Format display nicely
    rec_df_display = rec_df.copy()
    rec_df_display['Recommended_New_Partners'] = rec_df_display['Recommended_New_Partners'].apply(lambda x: ', '.join(x) if isinstance(x, list) else '')

    st.dataframe(rec_df_display[['Country', 'Recommended_New_Partners']])

with tab17:
    st.title("Shock Scenario Predictions for 2030")

    # Load your master data here or globally
    df = pd.read_csv("data/processed_master_df.csv")

    if st.button("Run Shock Predictions"):
        result_df = run_shock_scenario_predictions(df)

        # Format output and display nicely
        st.dataframe(result_df[['Country', 'GDP_2030', 'Poverty_2030', 'Unemployment_2030']].round(2))

with tab18:
    st.title("Ridge Regression Forecast for 2030 GDP and Poverty")

    if st.button("Run Ridge Forecast"):
        df = pd.read_csv("data/processed_master_df.csv")
        result_df = run_ridge_forecast(df, future_year=2030, alpha=1.0)

        # Format numeric columns nicely
        for col in ['GDP_2030_best', 'GDP_2030_worst', 'Poverty_2030_best', 'Poverty_2030_worst']:
            result_df[col] = result_df[col].round(2)

        st.dataframe(result_df)

with tab19:
    st.title("Moving Average Forecast for 2030")

    if st.button("Run Moving Average Forecast"):
        df = pd.read_csv("data/processed_master_df.csv")
        forecast_df = run_moving_average_forecast(df, target_year=2030, window=3)

        # Optionally round numeric columns for display
        for col in forecast_df.columns:
            if col != 'Country':
                forecast_df[col] = forecast_df[col].round(2)

        st.dataframe(forecast_df)

with tab20:
    st.title("Optimal New Trade and Infrastructure Links")

    if st.button("Run Optimization"):
        status, max_loss, links, fig = run_optimal_trade_links(
            exports_path="data/processed_exports_full.csv",
            imports_path="data/processed_imports_full.csv",
            master_path="data/processed_master_df.csv",
            budget=300,
            max_distance=1000
        )

        st.write(f"Solver status: {status}")
        st.write(f"Minimum maximum GDP loss: {max_loss:.4f}")

        if links:
            st.subheader("Suggested New Links to Build:")
            for i, j in links:
                st.write(f"{i} -> {j}")
        else:
            st.write("No links to build under current constraints.")

        st.pyplot(fig)











