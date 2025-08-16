import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import networkx as nx
import geopandas as gpd
import numpy as np
from matplotlib.lines import Line2D
# ========== CONFIG ==========
SCENARIO_CSV = "../data/2030_predictions_by_scenario.csv"
EXPORTS_CSV = "../data/processed_exports_full.csv"
IMPORT_CSV = "../data/processed_imports_full.csv"
OUTPUT_DIR = "outputs"
WORLD_SHP = "../data/ne_110m_admin_0_countries.shp" # <- Downloaded shapefile

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== LOAD ==========
scenario_df = pd.read_csv(SCENARIO_CSV)
try:
    trade_df = pd.read_csv(EXPORTS_CSV)
    import_df = pd.read_csv(IMPORT_CSV)
except Exception:
    trade_df = None

# ========== 1. HEATMAP ==========
def save_heatmaps(df, val_cols, output_dir):
    for val_col in val_cols:
        pivot = df.pivot(index="Country", columns="Scenario", values=val_col)
        plt.figure(figsize=(14, max(6, int(len(pivot)/2))))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5)
        plt.title(f"Scenario Heatmap: {val_col}")
        plt.tight_layout()
        outname = os.path.join(output_dir, f"heatmap_{val_col.replace(' ','_').replace('(','').replace(')','')}.png")
        plt.savefig(outname)
        plt.close()
        print(f"Saved heatmap: {outname}")

save_heatmaps(scenario_df, [
    "GDP growth (annual %)"    ,
    'Poverty headcount ratio at $3.00 a day (2021 PPP) (% of population)',
    'Trade (% of GDP)'
], OUTPUT_DIR)

def sanitize_country_name(name):
    # Remove or replace problematic characters for safe filenames
    return name.replace(" ", "_").replace(",", "").replace(".", "").replace("'", "")


def save_split_trade_network_subplot(export_df, import_df, country, output_dir):
    print(f"{country} is getting generated....")

    years_export = set(export_df.loc[export_df['Country'] == country, 'Year'].dropna().unique())
    years_import = set(import_df.loc[import_df['Country'] == country, 'Year'].dropna().unique())
    available_years = sorted(list(years_export | years_import))

    for year in available_years:
        # --- EXPORTS GRAPH ---
        G_export = nx.DiGraph()
        subset_exp = export_df[
            (export_df['Country'] == country) & (export_df['Year'] == year)
        ]
        for _, row in subset_exp.iterrows():
            partner = row['partnerDesc']
            value = row.get('export_value', None)
            if pd.notnull(partner) and pd.notnull(value) :
                G_export.add_edge(country, partner, weight=value)

        # --- IMPORTS GRAPH ---
        G_import = nx.DiGraph()
        subset_imp = import_df[
            (import_df['Country'] == country) & (import_df['Year'] == year)
        ]
        for _, row in subset_imp.iterrows():
            partner = row['partnerDesc']
            value = row.get('import_value', None)
            if pd.notnull(partner) and pd.notnull(value) :
                G_import.add_edge(partner, country, weight=value)

        if not G_export.edges and not G_import.edges:
            continue

        fig, axs = plt.subplots(1, 2, figsize=(34, 18))
        fig.suptitle(f"Trade Networks: {country} {year}", fontsize=34, weight='bold')

        # EXPORTS SUBPLOT
        if G_export.edges:
            pos_export = nx.circular_layout(G_export, scale=5)
            weights_exp = np.array([G_export[u][v]['weight'] for u, v in G_export.edges()])
            e_widths = np.interp(np.log1p(weights_exp), (np.log1p(weights_exp).min(), np.log1p(weights_exp).max()),
                                 (2.5, 10)) if len(weights_exp) else [2.5]
            plt.sca(axs[0])
            nx.draw_networkx_nodes(G_export, pos_export, node_color='dodgerblue', node_size=1800, alpha=0.93)
            nx.draw_networkx_edges(G_export, pos_export, width=e_widths, edge_color='deepskyblue',
                                   arrows=True, arrowstyle='-|>', arrowsize=32, alpha=0.9,
                                   connectionstyle='arc3,rad=0.0')
            nx.draw_networkx_labels(G_export, pos_export, font_size=15, font_weight='bold',
                                    bbox=dict(facecolor='white', edgecolor='black', alpha=0.97, pad=0.6))
            axs[0].set_title(f"EXPORTS from {country}", fontsize=24, weight='bold')
        axs[0].set_axis_off()

        # IMPORTS SUBPLOT
        if G_import.edges:
            pos_import = nx.circular_layout(G_import, scale=5)
            weights_imp = np.array([G_import[u][v]['weight'] for u, v in G_import.edges()])
            i_widths = np.interp(np.log1p(weights_imp), (np.log1p(weights_imp).min(), np.log1p(weights_imp).max()),
                                 (2.5, 10)) if len(weights_imp) else [2.5]
            plt.sca(axs[1])
            nx.draw_networkx_nodes(G_import, pos_import, node_color='orange', node_size=1800, alpha=0.93)
            nx.draw_networkx_edges(G_import, pos_import, width=i_widths, edge_color='crimson',
                                   arrows=True, arrowstyle='-|>', arrowsize=32, alpha=0.9,
                                   connectionstyle='arc3,rad=0.0')
            nx.draw_networkx_labels(G_import, pos_import, font_size=15, font_weight='bold',
                                    bbox=dict(facecolor='white', edgecolor='black', alpha=0.97, pad=0.6))
            axs[1].set_title(f"IMPORTS to {country}", fontsize=24, weight='bold')
        axs[1].set_axis_off()
        # Optional: add legends
        from matplotlib.lines import Line2D
        legend_elements_exp = [Line2D([0],[0] , color='deepskyblue', lw=5, label='Export (out)')]
        legend_elements_imp = [Line2D([0],[0] , color='crimson', lw=5, label='Import (in)')]
        axs[0].legend(handles=legend_elements_exp, loc='upper right', fontsize=14)
        axs[1].legend(handles=legend_elements_imp, loc='upper right', fontsize=14)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fname = os.path.join(output_dir, f"{sanitize_country_name(country)}_{year}_trade_split_subplot.png")
        plt.savefig(fname, dpi=230, bbox_inches="tight")
        plt.close()
# === GENERATE NETWORKS FOR ALL COUNTRIES IN TRADE FILE ===
if trade_df is not None:
    print(trade_df['Country'].dropna().unique())
    for country in trade_df['Country'].dropna().unique():
        save_split_trade_network_subplot(trade_df,import_df, country, OUTPUT_DIR)

# ========== 3. SHOCK MAP (Geo) ==========
def save_shock_map(df, scenario, value_col, output_dir):
    world = gpd.read_file(WORLD_SHP)
    print("Shapefile columns:", world.columns)

    # Try common world name columns: 'NAME', 'ADMIN', or 'name' (print(world.columns) to check)
    left_key = 'NAME' if 'NAME' in world.columns else 'ADMIN' if 'ADMIN' in world.columns else 'name'
    merged = world.merge(df[df['Scenario']==scenario], left_on=left_key, right_on="Country", how="left")
    ax = merged.plot(column=value_col, cmap='Reds', legend=True, figsize=(15,9),
                     missing_kwds={"color": "lightgrey"}, edgecolor='black')
    plt.title(f"Shock Map: {value_col}, {scenario} scenario")
    plt.tight_layout()
    fname = os.path.join(output_dir, f"shock_map_{value_col.replace(' ','_')}_{scenario}.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved shock map: {fname}")

save_shock_map(scenario_df, 'global_crisis', "GDP growth (annual %)", OUTPUT_DIR)
save_shock_map(scenario_df, 'global_crisis', 'Poverty headcount ratio at $3.00 a day (2021 PPP) (% of population)', OUTPUT_DIR)
save_shock_map(scenario_df, 'global_crisis', 'Trade (% of GDP)', OUTPUT_DIR)

# ========== 4. VULNERABILITY INSIGHTS ==========
def get_top_vulnerabilities(df, country):
    sub = df[df['Country'] == country]
    insights = []
    if not sub.empty:
        # Highest poverty
        max_poverty = sub['Poverty headcount ratio at $3.00 a day (2021 PPP) (% of population)'].max()
        scenario_poverty = sub.loc[sub['Poverty headcount ratio at $3.00 a day (2021 PPP) (% of population)'].idxmax(), 'Scenario']
        insights.append(f"Highest poverty ({max_poverty:.2f}) under scenario: {scenario_poverty}")
        # Lowest gdp growth
        min_gdp = sub["GDP growth (annual %)"].min()
        scenario_gdp = sub.loc[sub["GDP growth (annual %)"].idxmin(), 'Scenario']
        insights.append(f"Lowest GDP growth ({min_gdp:.2f}) under scenario: {scenario_gdp}")
        # Highest trade exposure
        max_trade = sub['Trade (% of GDP)'].max()
        scenario_trade = sub.loc[sub['Trade (% of GDP)'].idxmax(), 'Scenario']
        insights.append(f"Highest trade exposure ({max_trade:.2f}%) under scenario: {scenario_trade}")
    return insights

def save_insights_txt(df, output_dir):
    for country in df['Country'].unique():
        insights = get_top_vulnerabilities(df, country)
        with open(os.path.join(output_dir, f"{country}_top_vulnerabilities.txt"), "w", encoding="utf8") as f:
            for line in insights:
                f.write(line + "\n")
        print(f"Saved insights for {country}")

save_insights_txt(scenario_df, OUTPUT_DIR)

print("All visualizations and country vulnerability insights saved.")
