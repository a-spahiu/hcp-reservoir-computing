import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Style settings for the plots
sns.set_theme(style="whitegrid")
SEED = 42

def load_connectome_data(filepath):
    """Loads the graphml file and returns separate DataFrames for nodes and edges."""
    G = nx.read_graphml(filepath)
    
    # 1. Node Features Extraction (3D Positions, Hemisphere, ID)
    nodes_data = []
    for node, data in G.nodes(data=True):
        nodes_data.append({
            'node_id': node,
            'dn_position_x': data.get('dn_position_x', 0),
            'dn_position_y': data.get('dn_position_y', 0),
            'dn_position_z': data.get('dn_position_z', 0),
            'dn_correspondence_id': data.get('dn_correspondence_id', 0)
        })
    df_nodes = pd.DataFrame(nodes_data)
    
    # 2. Edge Features Extraction (Fibers, FA, Length)
    edges_data = []
    for u, v, data in G.edges(data=True):
        edges_data.append({
            'source': u,
            'target': v,
            'number_of_fibers': data.get('number_of_fibers', 0),
            'FA_mean': data.get('FA_mean', 0.0),
            'fiber_length_mean': data.get('fiber_length_mean', 0.0)
        })
    df_edges = pd.DataFrame(edges_data)
    
    return G, df_nodes, df_edges

def plot_3d_connectome(df_nodes):
    """Generates the 3D visualization of the connectome nodes (Ref: Fig 7 of the Report)."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        df_nodes['dn_position_x'], 
        df_nodes['dn_position_y'], 
        df_nodes['dn_position_z'],
        c=df_nodes['dn_correspondence_id'], 
        cmap='viridis', 
        s=30, 
        alpha=0.8
    )
    
    ax.set_xlabel('dn_position_x')
    ax.set_ylabel('dn_position_y')
    ax.set_zlabel('dn_position_z')
    plt.colorbar(scatter, label='dn_correspondence_id')
    plt.title('3D Visualization of Connectome Nodes')
    plt.show()

def plot_edge_features_histograms(df_edges):
    """Plots the distributions of the edge features (Ref: Fig 8 of the Report)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.histplot(df_edges['fiber_length_mean'], bins=50, ax=axes[0], color="tab:blue")
    axes[0].set_title('Histogram: fiber_length_mean')
    
    sns.histplot(df_edges['FA_mean'], bins=50, ax=axes[1], color="tab:blue")
    axes[1].set_title('Histogram: FA_mean')
    
    sns.histplot(df_edges['number_of_fibers'], bins=50, ax=axes[2], color="tab:blue")
    axes[2].set_title('Histogram: number_of_fibers')
    
    plt.tight_layout()
    plt.show()

def plot_spearman_correlation(df_edges):
    """Calculates and plots the Spearman correlation matrix (Ref: Fig 9 of the Report)."""
    features = ['fiber_length_mean', 'FA_mean', 'number_of_fibers']
    
    # Using Spearman correlation for robustness against skewed distributions (e.g., number of fibers)
    corr_matrix = df_edges[features].corr(method='spearman')
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    plt.title('Spearman correlation matrix')
    plt.show()

if __name__ == "__main__":
    # Replace this with the path to your downloaded file
    # graph_path = "data/sample_subject.graphml" 
    
    # G, df_nodes, df_edges = load_connectome_data(graph_path)
    # plot_3d_connectome(df_nodes)
    # plot_edge_features_histograms(df_edges)
    # plot_spearman_correlation(df_edges)
    print("Connectome Analysis module ready to use!")
