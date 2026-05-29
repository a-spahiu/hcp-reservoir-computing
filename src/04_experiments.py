import numpy as np
import networkx as nx
import copy
import matplotlib.pyplot as plt
import seaborn as sns

from 02_lorenz_system import generate_lorenz_data
from 03_reservoir_model import ReservoirModel

sns.set_theme(style="whitegrid")

def simulate_structural_damage(G, damage_fraction=0.1, damage_type='edges', seed=42):
    """
    Simulates structural damage on the connectome by removing elements.
    """
    np.random.seed(seed)
    G_damaged = copy.deepcopy(G)
    
    if damage_type == 'edges':
        edges = list(G_damaged.edges())
        num_to_remove = int(len(edges) * damage_fraction)
        if num_to_remove > 0:
            edges_to_remove = [edges[i] for i in np.random.choice(len(edges), num_to_remove, replace=False)]
            G_damaged.remove_edges_from(edges_to_remove)
            
    elif damage_type == 'nodes':
        nodes = list(G_damaged.nodes())
        num_to_remove = int(len(nodes) * damage_fraction)
        if num_to_remove > 0:
            nodes_to_remove = [nodes[i] for i in np.random.choice(len(nodes), num_to_remove, replace=False)]
            G_damaged.remove_nodes_from(nodes_to_remove)
            
    return G_damaged

def evaluate_reservoir_robustness(clean_graph, lorenz_signal, damage_fractions, target_spectral_radius=0.8):
    """
    Evaluates how prediction performance (RMSE) degrades as structural damage increases.
    """
    mean_rmse_results = []
    
    for frac in damage_fractions:
        print(f"Evaluating damage fraction: {frac:.2f}...")
        # 1. Generate damaged graph
        damaged_g = simulate_structural_damage(clean_graph, damage_fraction=frac, damage_type='edges')
        
        # 2. Extract adjacency matrix
        W_damaged = nx.to_numpy_array(damaged_g)
        
        # 3. Normalize matrix to match the desired spectral radius J
        eigenvalues = np.linalg.eigvals(W_damaged)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        if max_eigenvalue > 0:
            W_damaged = W_damaged * (target_spectral_radius / max_eigenvalue)
            
        # 4. Initialize ReservoirModel
        n_nodes = len(damaged_g.nodes)
        model = ReservoirModel(n_neurons=n_nodes, spectral_radius=target_spectral_radius)
        
        # Override the random internal weights with our biological connectome weights
        model.W = W_damaged 
        
        # 5. Train & Evaluate
        R, valid_signal = model.harvest_states(lorenz_signal, washout=200)
        _, _, rmse = model.train_readout(R, valid_signal, lambda_reg=1e-8)
        
        # Average RMSE across x, y, z dimensions
        mean_rmse_results.append(np.mean(rmse))
        
    return mean_rmse_results

def plot_robustness_results(damage_fractions, rmse_results):
    """Plots the degradation of performance as damage increases."""
    plt.figure(figsize=(8, 5))
    plt.plot(damage_fractions, rmse_results, marker='o', linestyle='-', color='tab:red', linewidth=2)
    plt.title('Reservoir Performance vs Structural Damage', fontsize=14)
    plt.xlabel('Fraction of Removed Edges', fontsize=12)
    plt.ylabel('Mean Prediction RMSE', fontsize=12)
    plt.show()

if __name__ == "__main__":
    # print("Generating Lorenz Data...")
    # _, lorenz_data = generate_lorenz_data(dt=0.01, num_steps=5000)
    
    # Creiamo un finto connettoma per testare il codice senza i dati pesanti
    # dummy_connectome = nx.erdos_renyi_graph(n=250, p=0.1)
    
    # damage_steps = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    # rmse_degradation = evaluate_reservoir_robustness(dummy_connectome, lorenz_data, damage_steps)
    
    # plot_robustness_results(damage_steps, rmse_degradation)
    
    print("Lesion Experiments module completed and ready to use!")
