import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st

def run_trade_network_analysis(data_path="data/processed_exports_full.csv", top_n=25):
    df = pd.read_csv(data_path)

    top_countries = (
        df.groupby('Country')['export_value']
          .sum()
          .nlargest(top_n)
          .index
          .tolist()
    )

    df_net = df[
        df['Country'].isin(top_countries) & df['partnerDesc'].isin(top_countries)
    ]

    G = nx.DiGraph()
    for _, row in df_net.iterrows():
        src, tgt, weight = row['Country'], row['partnerDesc'], row['export_value']
        if G.has_edge(src, tgt):
            G[src][tgt]['weight'] += weight
        else:
            G.add_edge(src, tgt, weight=weight)

    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, weight='weight')

    central_df = pd.DataFrame({
        'Country': list(degree_centrality.keys()),
        'Degree_Centrality': list(degree_centrality.values()),
        'Betweenness_Centrality': [betweenness_centrality[c] for c in degree_centrality.keys()]
    })

    most_central = central_df.sort_values('Betweenness_Centrality', ascending=False).head(5)
    st.write("Most central countries by betweenness centrality:", most_central)

    # Visualize original trade network
    fig1, ax1 = plt.subplots(figsize=(14,10))
    pos = nx.spring_layout(G, seed=42, k=0.7)
    node_sizes = [5000 * betweenness_centrality[n] + 300 for n in G.nodes()]
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=15, edge_color='gray', alpha=0.3, ax=ax1)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', edgecolors='black', ax=ax1)
    nx.draw_networkx_labels(G, pos, font_size=9, ax=ax1)
    ax1.set_title("Global Trade Network (Top 25 Exporters)\n(Node size = betweenness centrality)", fontsize=16)
    ax1.axis('off')
    st.pyplot(fig1)

    # Remove most central country and simulate disruption
    central_country = most_central.iloc[0]['Country']
    st.write(f"\nRemoving most central country from network: **{central_country}**")

    G_removed = G.copy()
    G_removed.remove_node(central_country)

    # Visualize disrupted network
    fig2, ax2 = plt.subplots(figsize=(12,8))
    pos2 = nx.spring_layout(G_removed, seed=42, k=0.6)
    nx.draw(G_removed, pos2, node_size=300, node_color='lightcoral',
            with_labels=True, font_size=9, edge_color='gray', alpha=0.3, ax=ax2)
    ax2.set_title(f"Network after removing {central_country}", fontsize=15)
    ax2.axis('off')
    st.pyplot(fig2)

    total_weight_before = sum(d['weight'] for u, v, d in G.edges(data=True))
    total_weight_after = sum(d['weight'] for u, v, d in G_removed.edges(data=True))
    trade_volume_lost = total_weight_before - total_weight_after

    st.write(f"Total trade volume before removal: {total_weight_before:,.0f}")
    st.write(f"Total trade volume after removal:  {total_weight_after:,.0f}")
    st.write(f"Trade volume lost:                 {trade_volume_lost:,.0f}")

    try:
        cc_num = nx.number_connected_components(G_removed.to_undirected())
        st.write(f"Number of connected components after disruption: {cc_num}")
    except Exception as e:
        st.write("Network became fragmented. Details:", e)
