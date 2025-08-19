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

# Page configuration
st.set_page_config(
    page_title="TREN Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }

    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }

    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }

    .stTab {
        font-weight: 600;
    }

    .analysis-section {
        background-color: #fafafa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


# Utility functions
def fix_types_for_arrow(df):
    """Fix DataFrame column types to avoid pyarrow serialization errors in Streamlit"""
    for col in df.columns:
        if col == 'Year':
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        elif df[col].dtype == 'object' or col == 'Country':
            df[col] = df[col].astype(str)
    return df


def sanitize_country_name(name):
    """Remove problematic characters and replace spaces with underscores"""
    return name.replace(" ", "_").replace(",", "").replace(".", "")


# Data loading with caching
@st.cache_data
def load_data():
    master_df = load_preprocessed_master()
    exports_df = load_exports_full()
    imports_df = load_imports_full()
    return master_df, exports_df, imports_df


# Main header
st.markdown('<h1 class="main-header"> TREN Analytics Dashboard</h1>', unsafe_allow_html=True)

# Load data
with st.spinner("Loading data..."):
    master_df, exports_df, imports_df = load_data()
    scenario_results_df = load_scenario_forecasts()

OUTPUT_DIR = "scripts/outputs"

# Create two main sections with tabs
st.markdown('<div class="section-header">Data Management & Visualization</div>', unsafe_allow_html=True)

data_tabs = st.tabs([
    "Master Data",
    " Exports",
    " Imports",
    "2030 Forecasts",
    " Vulnerabilities",
    "Visualizations",
    "Policy Engine"
])

# Data Management Tabs
with data_tabs[0]:
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.title("Preprocessed Master DataFrame")

    col1, col2 = st.columns(2)
    with col1:
        country = st.selectbox(" Select Country:", sorted(master_df['Country'].unique()), key='tab1_country')
    with col2:
        years_available = master_df[master_df['Country'] == country]['Year'].dropna().unique()
        year = st.selectbox("Select Year:", sorted(years_available), key='tab1_year')

    subset = master_df[(master_df['Country'] == country) & (master_df['Year'] == year)]
    if subset.empty:
        st.warning("No data available for the selected country-year")
    else:
        st.success(f" Showing data for {country} in {year}")
        st.dataframe(fix_types_for_arrow(subset.T), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with data_tabs[1]:
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.title(" Export Trade Data (Detailed Partner-Level)")

    col1, col2 = st.columns(2)
    with col1:
        country_exp = st.selectbox(" Select Country for Exports:", sorted(exports_df['Country'].unique()),
                                   key='tab2_country')
    with col2:
        years_exp = exports_df[exports_df['Country'] == country_exp]['Year'].dropna().unique()
        year_exp = st.selectbox(" Select Year for Exports:", sorted(years_exp), key='tab2_year')

    subset_exp = exports_df[(exports_df['Country'] == country_exp) & (exports_df['Year'] == year_exp)]
    if subset_exp.empty:
        st.warning("No export data available for the selected country-year")
    else:
        st.success(f" Showing export data for {country_exp} in {year_exp}")
        st.dataframe(fix_types_for_arrow(subset_exp), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with data_tabs[2]:
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.title(" Import Trade Data (Detailed Partner-Level)")

    col1, col2 = st.columns(2)
    with col1:
        country_imp = st.selectbox("Select Country for Imports:", sorted(imports_df['Country'].unique()),
                                   key='tab3_country')
    with col2:
        years_imp = imports_df[imports_df['Country'] == country_imp]['Year'].dropna().unique()
        year_imp = st.selectbox(" Select Year for Imports:", sorted(years_imp), key='tab3_year')

    subset_imp = imports_df[(imports_df['Country'] == country_imp) & (imports_df['Year'] == year_imp)]
    if subset_imp.empty:
        st.warning("No import data available for the selected country-year")
    else:
        st.success(f" Showing import data for {country_imp} in {year_imp}")
        st.dataframe(fix_types_for_arrow(subset_imp), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with data_tabs[3]:
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.title("2030 Scenario Forecasts")

    col1, col2 = st.columns(2)
    with col1:
        country_scenario = st.selectbox(" Select Country:", sorted(scenario_results_df['Country'].unique()),
                                        key='tab4_country')
    with col2:
        years_available_scenario = scenario_results_df[scenario_results_df['Country'] == country_scenario][
            'Scenario'].unique()
        scenario_selected = st.selectbox(" Select Scenario:", sorted(years_available_scenario), key='tab4_scenario')

    subset_scenario = scenario_results_df[
        (scenario_results_df['Country'] == country_scenario) &
        (scenario_results_df['Scenario'] == scenario_selected)
        ]
    if subset_scenario.empty:
        st.warning(" No prediction data available for the selected country and scenario.")
    else:
        st.success(f" Showing {scenario_selected} scenario for {country_scenario}")
        st.dataframe(fix_types_for_arrow(subset_scenario), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with data_tabs[4]:
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.title(" Country Vulnerability Insights")

    insight_country = st.selectbox(" Select Country for Insights:", sorted(scenario_results_df['Country'].unique()),
                                   key='tab5_country')
    sanitized_name = sanitize_country_name(insight_country)
    insight_path = os.path.join(OUTPUT_DIR, f"{sanitized_name}_top_vulnerabilities.txt")

    if os.path.exists(insight_path):
        with open(insight_path, "r", encoding="utf8") as f:
            insights = f.readlines()
        st.markdown("####  Top Vulnerabilities")
        for i, insight in enumerate(insights, 1):
            st.markdown(f"**{i}.** {insight.strip()}")
    else:
        st.warning(" No insights file available for the selected country.")
    st.markdown('</div>', unsafe_allow_html=True)

with data_tabs[5]:
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.title(" Scenario Visualizations")

    vis_type = st.selectbox(" Visualization Type:", [
        "Heatmap GDP Growth",
        "Heatmap Poverty",
        "Heatmap Trade",
        "Shock Map GDP Growth",
        "Shock Map Poverty",
        "Shock Map Trade",
        "Trade Network"
    ], key='tab6_type')

    file_map = {
        "Heatmap GDP Growth": "heatmap_gdp_growth_annual_pct.png",
        "Heatmap Poverty": "heatmap_Poverty_headcount_ratio_at_$3.00_a_day_2021_PPP_%_of_population.png",
        "Heatmap Trade": "heatmap_Trade_%_of_GDP.png",
        "Shock Map GDP Growth": "shock_map_gdp_growth_annual_pct_global_crisis.png",
        "Shock Map Poverty": "shock_map_Poverty_headcount_ratio_at_$3.00_a_day_(2021_PPP)_(%_of_population)_global_crisis.png",
        "Shock Map Trade": "shock_map_Trade_(%_of_GDP)_global_crisis.png"
    }

    if vis_type == "Trade Network":
        col1, col2 = st.columns(2)
        with col1:
            tn_country = st.selectbox(" Country:", sorted(scenario_results_df['Country'].unique()),
                                      key="tab6_tn_country")

        sanitized_country = sanitize_country_name(tn_country)
        matching_files = [f for f in os.listdir(OUTPUT_DIR) if
                          f.startswith(f"{sanitized_country}_") and f.endswith("_trade_split_subplot.png")]

        available_years = []
        for f in matching_files:
            parts = f.replace(".png", "").split("_")
            year = None
            for part in reversed(parts):
                if part.isdigit() and len(part) == 4:
                    year = int(part)
                    break
            if year:
                available_years.append(year)
        available_years = sorted(set(available_years))

        if available_years:
            with col2:
                tn_year = st.selectbox(" Year:", available_years, key="tab6_tn_year")
            target_file = f"{sanitized_country}_{tn_year}_trade_split_subplot.png"
            target_path = os.path.join(OUTPUT_DIR, target_file)
            if os.path.exists(target_path):
                st.image(target_path, use_container_width=True,
                         caption=f" {tn_country} {tn_year} Trade Network (Exports & Imports)")
            else:
                st.warning(f" No trade network image found for {tn_country} in {tn_year}.")
        else:
            st.warning(f"No trade network images found for {tn_country}.")
    else:
        vis_file = file_map.get(vis_type, "")
        vis_path = os.path.join(OUTPUT_DIR, vis_file)
        if os.path.exists(vis_path):
            st.image(vis_path, use_container_width=True, caption=f" {vis_type}")
        else:
            st.warning(" No visualization image found for selected type.")
    st.markdown('</div>', unsafe_allow_html=True)

with data_tabs[6]:
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.title(" Policy Recommendations & What-If Scenario Simulator")

    df = policy_engine.load_data()
    country = st.selectbox(" Select Country:", policy_engine.ALLOWED_COUNTRIES, key="tab7_country")

    latest, recs, stats = policy_engine.analyze_country(df, country)

    if latest is None:
        st.error(" No data available for this country.")
    else:
        st.markdown(f"###  Policy Recommendations for **{country}** (Year: {int(latest['Year'])})")

        if recs:
            for i, r in enumerate(recs, 1):
                st.markdown(f"**{i}.** {r}")
        else:
            st.success(" No acute vulnerabilities detected; maintain current resilience strategy.")

        st.markdown("###  Key Indicators")
        col1, col2 = st.columns(2)
        for i, (k, v) in enumerate(stats.items()):
            with col1 if i % 2 == 0 else col2:
                if isinstance(v, float):
                    st.metric(k, f"{v:,.2f}")
                else:
                    st.metric(k, str(v))

        st.markdown("###  What-If Scenario Simulations")

        col1, col2 = st.columns(2)
        with col1:
            gdp_change = st.slider(" Increase GDP growth by (% points)", min_value=0.0, max_value=5.0, step=0.1,
                                   value=1.0)
            poverty_new = policy_engine.poverty_impact_of_gdp_growth(
                stats["Poverty headcount ratio at $3.00 a day (2021 PPP) (% of population)"],
                stats["GDP growth (%)"], gdp_change
            )
            st.success(
                f"→ If GDP growth increases by {gdp_change:.1f} pp, poverty could drop to **{poverty_new:.2f}%**")

            trade_reduce = st.slider(" Reduce Trade (% of GDP) by", min_value=0.0,
                                     max_value=float(stats["Trade (% of GDP)"]), step=0.5, value=10.0)
            gdp_growth_new = policy_engine.trade_volatility_impact_on_gdp_growth(
                stats["GDP growth (%)"], stats["Trade (% of GDP)"], trade_reduce
            )
            st.success(
                f"→ If Trade (% of GDP) drops by {trade_reduce:.1f} pp, GDP growth may rise to **{gdp_growth_new:.2f}%**")

        with col2:
            inf_reduce = st.slider("Reduce inflation (%) by", min_value=0.0, max_value=float(stats["Inflation (%)"]),
                                   step=0.1, value=2.0)
            poverty_inf = policy_engine.inflation_impact_on_poverty(
                stats["Poverty headcount ratio at $3.00 a day (2021 PPP) (% of population)"],
                stats["Inflation (%)"], -inf_reduce
            )
            st.success(f"→ If inflation drops by {inf_reduce:.1f} pp, poverty could drop to **{poverty_inf:.2f}%**")

            disaster_reduce = st.slider("Reduce disaster damage by ($ billion)",
                                        min_value=0.0,
                                        max_value=stats["Disaster damage ($)"] / 1e9 if stats[
                                                                                            "Disaster damage ($)"] > 0 else 1.0,
                                        step=0.1, value=1.0)
            gdp_growth_disaster = policy_engine.disaster_impact_on_gdp_growth(
                stats["GDP growth (%)"], stats["Disaster damage ($)"], -disaster_reduce * 1e9
            )
            st.success(
                f"→ If disaster damage falls by ${disaster_reduce:.1f}B, GDP growth may rise to **{gdp_growth_disaster:.2f}%**")

        st.info(
            "Elasticity coefficients are illustrative. Refine with local data/statistical analysis for accurate scenario modeling.")
    st.markdown('</div>', unsafe_allow_html=True)

# Analysis Questions Section
st.markdown('<div class="section-header">Advanced Analytics & Research Questions</div>', unsafe_allow_html=True)

analysis_tabs = st.tabs([
    "Q1: Trade Dependency",
    "Q2: China Export Shock",
    "Q3: Drought Impact",
    "Q4: Food Security",
    "Q5: Youth Unemployment",
    "Q6: Export Risk",
    "Q7: Trade Networks",
    "Q8: Trade Routes",
    "Q9: Resilience",
    "Q10: Shock Scenarios",
    "Q11: Ridge Forecast",
    "Q12: Moving Average",
    "Q13: Optimal Links"
])

# Analysis Questions Tabs
with analysis_tabs[0]:
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.title("Top Trade Dependency and GDP Impact Analysis")

    exports_path = "data/processed_exports_full.csv"
    master_path = "data/processed_master_df.csv"

    if st.button("Compute Top Trade Dependency and GDP Impact", type="primary"):
        with st.spinner("Computing analysis..."):
            result_df = compute_top_trade_dependency_and_gdp_impact(exports_path, master_path)

            if not result_df.empty:
                st.success("Analysis completed successfully!")
                st.dataframe(result_df, use_container_width=True)
            else:
                st.warning("No data available or no results computed.")
    st.markdown('</div>', unsafe_allow_html=True)

with analysis_tabs[1]:
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.title("🇨🇳 Cascading Export Shock from China - Impact Simulation")

    col1, col2 = st.columns(2)
    with col1:
        exports_path = st.text_input("Exports CSV Path:", "data/processed_exports_full.csv")
        master_path = st.text_input("Master DF CSV Path:", "data/processed_master_df.csv")
    with col2:
        shock_year = st.number_input("Shock Year:", min_value=2000, max_value=2050, value=2028, step=1)
        china_export_drop_pct = st.slider("China Export Drop Percentage", min_value=0.0, max_value=1.0, value=0.25,
                                          step=0.01)

    if st.button("Simulate China Export Shock", type="primary"):
        with st.spinner("Running simulation..."):
            result_df = simulate_cascading_china_export_shock(
                exports_path=exports_path,
                master_path=master_path,
                shock_year=shock_year,
                china_export_drop_pct=china_export_drop_pct
            )

            if not result_df.empty:
                result_df['GDP_Loss_usd'] = result_df['GDP_Loss_usd'].apply(lambda x: f"${x:,.0f}")
                result_df['GDP_Loss_pct'] = result_df['GDP_Loss_pct'].round(4).astype(str) + '%'
                result_df['ExportsFromChina'] = result_df['ExportsFromChina'].apply(lambda x: f"${x:,.0f}")

                st.success("Simulation completed successfully!")
                st.dataframe(result_df, use_container_width=True)
            else:
                st.warning("NO data or results to display.")
    st.markdown('</div>', unsafe_allow_html=True)

with analysis_tabs[2]:
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.title("Drought Export Impact Statistical Analysis")

    col1, col2 = st.columns(2)
    with col1:
        countries_input = st.text_area("Enter countries separated by commas (leave blank for all):")
        drought_years_input = st.text_input("Drought Years (comma separated):", "2028,2029,2030")
    with col2:
        drought_yield_drop = st.slider("Drought Yield Drop Fraction", min_value=0.0, max_value=1.0, value=0.20,
                                       step=0.01)


    def parse_countries(text):
        if not text.strip():
            return None
        return [c.strip() for c in text.split(",") if c.strip()]


    def parse_years(text):
        try:
            return [int(y.strip()) for y in text.split(",") if y.strip()]
        except:
            return [2028, 2029, 2030]


    if st.button("Run Drought Export Impact Analysis", type="primary"):
        with st.spinner("Analyzing drought impact..."):
            countries = parse_countries(countries_input)
            drought_years = parse_years(drought_years_input)
            master_df_local = pd.read_csv("data/processed_master_df.csv")

            result_df = drought_export_impact_stat(
                df=master_df_local,
                countries=countries,
                drought_years=drought_years,
                drought_yield_drop=drought_yield_drop
            )

            if not result_df.empty:
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
                result_df['total_disaster_damage'] = result_df['total_disaster_damage'].fillna(0).map(
                    lambda x: f"${x:,.0f}")

                st.success("Analysis completed successfully!")
                st.dataframe(result_df, use_container_width=True)
            else:
                st.warning("No data or insufficient data for analysis.")
    st.markdown('</div>', unsafe_allow_html=True)

with analysis_tabs[3]:
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.title("Agricultural Import Partner Dependency & Food Security Risk")

    col1, col2 = st.columns(2)
    with col1:
        import_path_default = "data/processed_imports_full.csv"
        year_selected = st.number_input("Select Year:", min_value=2000, max_value=2050, value=None, step=1)
        n_partners = st.slider("Number of Top Import Partners to Consider:", 1, 10, 3)
    with col2:
        threshold = st.slider("Dependency Threshold (share):", 0.0, 1.0, 0.6, 0.05)
        import_path = st.text_input("Import CSV Path:", import_path_default)

    if st.button("Run Agricultural Dependency and Risk Analysis", type="primary"):
        with st.spinner("Analyzing food security risks..."):
            dependent_df = agri_import_partner_dependency(import_path, year=year_selected, n_partners=n_partners,
                                                          threshold=threshold)
            if dependent_df.empty:
                st.warning("No countries meet the dependency threshold for the selected parameters.")
            else:
                st.success("Analysis completed successfully!")
                st.subheader("Countries Highly Dependent on Top Agricultural Import Partners")
                st.dataframe(dependent_df[['Country', 'TopPartners', 'TopPartnerShare']], use_container_width=True)

                risk_df = model_food_security_risk(dependent_df)
                st.subheader("Food Security Risk Assessment")
                risk_df['ImportLoss_pct'] = (risk_df['ImportLoss_pct']).round(2).astype(str) + '%'
                st.dataframe(risk_df[['Country', 'TopPartners', 'TopPartnerShare', 'ImportLoss_pct', 'RiskFlag']],
                             use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with analysis_tabs[4]:
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.title("Youth Unemployment Prediction & Risk Analysis (XGBoost)")

    col1, col2 = st.columns(2)
    with col1:
        scenario_increase = st.slider("Scenario Increase in Unemployment (%)", 0.0, 1.0, 0.15, step=0.01)
        cv_splits = st.slider("Cross Validation Splits", 2, 6, 3, step=1)
    with col2:
        model_dir = st.text_input("Model directory path:", "models")

    if st.button("Run Youth Unemployment XGB Risk Prediction", type="primary"):
        with st.spinner("Training XGBoost model and predicting risks..."):
            risky_df = cross_validate_youth_unemployment_xgb_minerror(
                df=master_df,
                scenario_increase=scenario_increase,
                cv_splits=cv_splits,
                model_dir=model_dir
            )

            if not risky_df.empty:
                risky_df_display = risky_df.copy()
                risky_df_display['Predicted_Unemployment_2030'] = risky_df_display['Predicted_Unemployment_2030'].round(
                    2)
                risky_df_display['Min_CV_MSE'] = risky_df_display['Min_CV_MSE'].round(4)
                risky_df_display['Best_Params'] = risky_df_display['Best_Params'].apply(lambda p: str(p))

                st.success("Prediction completed successfully!")
                st.write(
                    f"Countries predicted with unemployment > 10% in 2030 under scenario increase of {scenario_increase * 100:.1f}%:")
                st.dataframe(risky_df_display, use_container_width=True)
            else:
                st.info("ℹNo countries predicted with unemployment above threshold under current parameters.")
    st.markdown('</div>', unsafe_allow_html=True)

with analysis_tabs[5]:
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.title("Export Risk Prediction using ElasticNet Model")

    median_age_df = master_df.groupby('Country')['life_expectancy_total_years'].median().reset_index()
    median_age_df = median_age_df.rename(columns={'life_expectancy_total_years': 'median_age'})

    export_df = master_df.groupby('Country')['Exports of goods and services (% of GDP)'].median().reset_index()
    country_df = pd.merge(median_age_df, export_df, on='Country', how='inner')

    if st.button("Run ElasticNet Export Risk Prediction", type="primary"):
        with st.spinner("Training ElasticNet model..."):
            result_df = predict_export_risk_elasticnet(country_df)

            if not result_df.empty:
                df_display = result_df.copy()
                for col in ['Predicted_Export_pct_5yrs', 'Predicted_Export_pct_10yrs', 'Predicted_Export_pct_15yrs']:
                    df_display[col] = df_display[col].round(2)
                df_display['Export_pct_GDP'] = df_display['Export_pct_GDP'].round(2)

                st.success("Prediction completed successfully!")
                st.dataframe(df_display[['Country', 'CurrentMedianAge', 'Export_pct_GDP',
                                         'Predicted_Export_pct_5yrs', 'Predicted_Export_pct_10yrs',
                                         'Predicted_Export_pct_15yrs']],
                             use_container_width=True)
            else:
                st.warning("No valid data available for prediction.")
    st.markdown('</div>', unsafe_allow_html=True)

with analysis_tabs[6]:
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.title("Global Trade Network Visualization & Disruption Simulation")

    if st.button("Generate Trade Network Analysis", type="primary"):
        with st.spinner("Generating trade network visualizations..."):
            run_trade_network_analysis("data/processed_exports_full.csv")
            st.success("Trade network analysis completed! Check the outputs folder for visualizations.")
    st.markdown('</div>', unsafe_allow_html=True)

with analysis_tabs[7]:
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.title("Top Mutually Beneficial Trade Pairs & Impact of Collapse")

    if st.button("Analyze Trade Route Impact", type="primary"):
        with st.spinner("Analyzing trade routes..."):
            exports_path = "data/processed_exports_full.csv"
            gdp_path = "data/processed_master_df.csv"

            top_pairs, impact_df = run_trade_route_impact(exports_path, gdp_path)

            st.success("Analysis completed successfully!")

            st.subheader("Top 5 Mutually Beneficial Trade Pairs")
            top_pairs_display = top_pairs.copy()
            top_pairs_display['total_bilateral_trade'] = top_pairs_display['total_bilateral_trade'].map(
                '${:,.0f}'.format)
            st.dataframe(top_pairs_display, use_container_width=True)

            st.subheader("GDP Impact if Top Trade Route Collapses (Percentage Loss)")
            impact_display = impact_df.copy()
            impact_display['Bilateral_Trade_Value'] = impact_display['Bilateral_Trade_Value'].map('${:,.0f}'.format)
            impact_display['GDP_A_Loss_pct'] = impact_display['GDP_A_Loss_pct'].round(4).astype(str) + '%'
            impact_display['GDP_B_Loss_pct'] = pd.to_numeric(impact_display['GDP_B_Loss_pct'], errors='coerce').round(
                4).astype(str) + '%'
            st.dataframe(
                impact_display[['Country_A', 'Country_B', 'Bilateral_Trade_Value', 'GDP_A_Loss_pct', 'GDP_B_Loss_pct']],
                use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with analysis_tabs[8]:
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.title("Resilience Recommendations: Diversifying Trade Partners")

    if st.button("Generate Resilience Recommendations", type="primary"):
        with st.spinner("Generating recommendations..."):
            exports_path = "data/processed_exports_full.csv"
            rec_df = generate_resilience_recommendations(exports_path)

            rec_df_display = rec_df.copy()
            rec_df_display['Recommended_New_Partners'] = rec_df_display['Recommended_New_Partners'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else ''
            )

            st.success("Recommendations generated successfully!")
            st.dataframe(rec_df_display[['Country', 'Recommended_New_Partners']], use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with analysis_tabs[9]:
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.title("⚡ Shock Scenario Predictions for 2030")

    if st.button("Run Shock Predictions", type="primary"):
        with st.spinner("Running shock scenario predictions..."):
            df = pd.read_csv("data/processed_master_df.csv")
            result_df = run_shock_scenario_predictions(df)

            st.success("Predictions completed successfully!")
            st.dataframe(result_df[['Country', 'GDP_2030', 'Poverty_2030', 'Unemployment_2030']].round(2),
                         use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with analysis_tabs[10]:
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.title("Ridge Regression Forecast for 2030 GDP and Poverty")

    if st.button("Run Ridge Forecast", type="primary"):
        with st.spinner("Running Ridge regression forecast..."):
            df = pd.read_csv("data/processed_master_df.csv")
            result_df = run_ridge_forecast(df, future_year=2030, alpha=1.0)

            for col in ['GDP_2030_best', 'GDP_2030_worst', 'Poverty_2030_best', 'Poverty_2030_worst']:
                result_df[col] = result_df[col].round(2)

            st.success("Ridge forecast completed successfully!")
            st.dataframe(result_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with analysis_tabs[11]:
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.title("Moving Average Forecast for 2030")

    if st.button("Run Moving Average Forecast", type="primary"):
        with st.spinner("Computing moving average forecast..."):
            df = pd.read_csv("data/processed_master_df.csv")
            forecast_df = run_moving_average_forecast(df, target_year=2030, window=3)

            for col in forecast_df.columns:
                if col != 'Country':
                    forecast_df[col] = forecast_df[col].round(2)

            st.success("Moving average forecast completed successfully!")
            st.dataframe(forecast_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with analysis_tabs[12]:
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.title("Optimal New Trade and Infrastructure Links")

    col1, col2 = st.columns(2)
    with col1:
        budget = st.number_input("Budget:", min_value=100, max_value=1000, value=300, step=50)
    with col2:
        max_distance = st.number_input("Max Distance:", min_value=500, max_value=2000, value=1000, step=100)

    if st.button("Run Optimization", type="primary"):
        with st.spinner("Optimizing trade links..."):
            status, max_loss, links, fig = run_optimal_trade_links(
                exports_path="data/processed_exports_full.csv",
                imports_path="data/processed_imports_full.csv",
                master_path="data/processed_master_df.csv",
                budget=budget,
                max_distance=max_distance
            )

            st.success("Optimization completed successfully!")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Solver Status", status)
            with col2:
                st.metric("Minimum Maximum GDP Loss", f"{max_loss:.4f}")

            if links:
                st.subheader("Suggested New Links to Build:")
                for i, j in links:
                    st.write(f"**{i}** → **{j}**")
            else:
                st.info("ℹNo links to build under current constraints.")

            if fig:
                st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <h4>TREN Analytics Dashboard</h4>
        <p>Advanced Trade, Resilience, and Economic Network Analysis Platform</p>
        <p><em>Empowering data-driven policy decisions for global economic resilience</em></p>
    </div>
    """,
    unsafe_allow_html=True
)
