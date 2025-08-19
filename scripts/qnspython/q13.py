import pandas as pd
import numpy as np
import pulp
import random
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
import os
import pickle

def run_optimal_trade_links(
    exports_path,
    imports_path,
    master_path,
    budget=300,
    max_distance=1000,
    cache_path="models/optimal_trade_links.pkl"
):
    # Ensure models directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # If cache exists, load and return results and figure
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        return cached_data['status'], cached_data['max_gdp_loss'], cached_data['built_links'], cached_data['fig']

    # Load data
    exports_df = pd.read_csv(exports_path)
    imports_df = pd.read_csv(imports_path)
    master_df = pd.read_csv(master_path)

    countries = sorted(set(exports_df['Country'].unique())
                       .union(set(imports_df['Country'].unique()))
                       .union(set(master_df['Country'].unique())))
    pairs = [(i, j) for i, j in product(countries, countries) if i != j]

    latest_year = exports_df['Year'].max()
    df_latest = exports_df[exports_df['Year'] == latest_year]

    trade_sums = df_latest.groupby(['Country', 'partnerDesc'])['export_value'].sum().reset_index()
    total_exports = trade_sums.groupby('Country')['export_value'].sum().reset_index().rename(columns={'export_value': 'total_exports'})
    trade_merged = trade_sums.merge(total_exports, on='Country')
    trade_merged['export_share'] = trade_merged['export_value'] / trade_merged['total_exports']

    GDP_loss_if_failure = {}
    for _, row in trade_merged.iterrows():
        i = row['Country']
        j = row['partnerDesc']
        GDP_loss_if_failure[(i, j)] = row['export_share']

    min_loss_floor = 0.001
    for k in pairs:
        if k not in GDP_loss_if_failure or GDP_loss_if_failure[k] == 0:
            GDP_loss_if_failure[k] = min_loss_floor

    np.random.seed(42)
    distance = {pair: np.random.uniform(100, 1500) for pair in pairs}

    cost = {}
    for pair in pairs:
        if distance[pair] <= max_distance:
            cost[pair] = distance[pair] * 0.01 + random.uniform(5, 20)
        else:
            cost[pair] = 0

    export_pairs = set(zip(exports_df['Country'], exports_df['partnerDesc']))
    import_pairs = set(zip(imports_df['Country'], imports_df['partnerDesc']))
    no_trade_pairs = {pair for pair in pairs if pair not in export_pairs and pair not in import_pairs}
    geopolitical_restrictions = no_trade_pairs

    prob = pulp.LpProblem("Minimize_Max_GDP_Loss", pulp.LpMinimize)

    x = pulp.LpVariable.dicts("LinkBuild", pairs, cat='Binary')
    z = pulp.LpVariable("Max_GDP_Loss", lowBound=0)

    prob += z, "Minimize maximum GDP loss"

    for i in countries:
        base_loss = sum(GDP_loss_if_failure.get((i, j), min_loss_floor) for j in countries if i != j)
        loss_terms = []
        for j in countries:
            if i != j:
                if (i, j) in geopolitical_restrictions or distance[(i, j)] > max_distance:
                    continue
                loss_terms.append(GDP_loss_if_failure.get((i, j), min_loss_floor) * x[(i, j)])

        prob += (base_loss - pulp.lpSum(loss_terms) <= z), f"MaxLossConstraint_{i}"

    prob += pulp.lpSum([cost[(i, j)] * x[(i, j)] for (i, j) in pairs]) <= budget, "BudgetConstraint"

    for (i, j) in geopolitical_restrictions:
        if (i, j) in pairs:
            prob += x[(i, j)] == 0, f"GeoRestriction_{i}_{j}"

    for (i, j) in pairs:
        if distance[(i, j)] > max_distance:
            prob += x[(i, j)] == 0, f"DistanceRestriction_{i}_{j}"

    prob.solve()

    status = pulp.LpStatus[prob.status]
    max_gdp_loss = pulp.value(z)
    built_links = [(i, j) for (i, j) in pairs if pulp.value(x[(i, j)]) > 0.99]

    # Visualization
    G = nx.DiGraph()
    G.add_nodes_from(countries)
    for i, j in built_links:
        G.add_edge(i, j, weight=cost[(i, j)])

    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    edge_widths = [G[u][v]['weight'] * 0.1 for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=700, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=15, width=edge_widths, edge_color='green', ax=ax)

    ax.set_title("Optimal New Trade/Infrastructure Links")
    ax.axis('off')
    plt.tight_layout()

    # Save results and figure to cache
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'status': status,
            'max_gdp_loss': max_gdp_loss,
            'built_links': built_links,
            'fig': fig
        }, f)

    return status, max_gdp_loss, built_links, fig
