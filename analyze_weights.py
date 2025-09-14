import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from pathlib import Path
import pandas as pd

# Import your existing function
from compare_updates import extract_weights_only

def calculate_l2_norm(weights_dict):
    """
    Calculate the total L2 norm of all weight parameters in a model.
    
    Parameters:
    -----------
    weights_dict : dict
        Dictionary of weight tensors
        
    Returns:
    --------
    float : Total L2 norm
    """
    total_norm = 0.0
    for key, tensor in weights_dict.items():
        total_norm += torch.norm(tensor, p=2).item() ** 2
    return np.sqrt(total_norm)

def calculate_layer_wise_norms(weights_dict):
    """
    Calculate L2 norm for each layer separately.
    
    Parameters:
    -----------
    weights_dict : dict
        Dictionary of weight tensors
        
    Returns:
    --------
    dict : Dictionary mapping layer names to their L2 norms
    """
    layer_norms = {}
    for key, tensor in weights_dict.items():
        layer_norms[key] = torch.norm(tensor, p=2).item()
    return layer_norms

def load_client_checkpoints(checkpoint_dir, rounds, num_clients=10):
    """
    Load client encoders from checkpointed rounds.
    
    Parameters:
    -----------
    checkpoint_dir : str
        Base directory containing checkpoints
    rounds : list
        List of round numbers to load
    num_clients : int
        Number of clients
        
    Returns:
    --------
    dict : Nested dictionary {round: {client: weights_dict}}
    """
    client_weights = {}
    
    for round_num in rounds:
        client_weights[round_num] = {}
        
        for client_id in range(num_clients):
            # Adjust this path pattern based on your checkpoint naming convention
            checkpoint_path = os.path.join(
                checkpoint_dir, 
                f"round_{round_num}/pretrain", 
                f"model_ft_c{client_id}.pth"
            )
            
            if os.path.exists(checkpoint_path):
                print(f"Loading round {round_num}, client {client_id}")
                model_state = torch.load(checkpoint_path, map_location='cpu')
                # Load also global model from previous round
                global_model = torch.load(f"output/federated_exp_cifarnoiid/models/model_round{round_num-1}.pth", map_location='cpu')
                
                # Extract state dict if necessary
                if isinstance(model_state, dict) and 'state_dict' in model_state:
                    #weights = extract_weights_only(model_state['state_dict'])
                    weights = model_state['state_dict']
                    #global_weights = extract_weights_only(global_model['state_dict'])
                    global_weights = global_model['state_dict']
                else:
                    #weights = extract_weights_only(model_state)
                    weights = model_state
                    #global_weights = extract_weights_only(global_model)
                    global_weights = global_model

                # Compute update: subtract previous round's weights
                for key in weights.keys():
                    if key in weights and key in global_weights:
                        weights[key] = (weights[key] - global_weights[key]) 
                    else:
                        raise KeyError(f"Key {key} not found in update model")
                
                client_weights[round_num][client_id] = weights
            else:
                print(f"Warning: Checkpoint not found at {checkpoint_path}")
    
    return client_weights

def analyze_weight_norms(client_weights, output_dir="weight_norm_analysis"):
    """
    Analyze and visualize weight norms across rounds and clients.
    
    Parameters:
    -----------
    client_weights : dict
        Nested dictionary {round: {client: weights_dict}}
    output_dir : str
        Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate total L2 norms for each client at each round
    norms_data = []
    for round_num, clients in client_weights.items():
        for client_id, weights in clients.items():
            total_norm = calculate_l2_norm(weights)
            norms_data.append({
                'round': round_num,
                'client': client_id,
                'l2_norm': total_norm
            })
    
    df = pd.DataFrame(norms_data)
    
    # Plot 1: Evolution of weight norms for each client
    plt.figure(figsize=(12, 8))
    for client_id in df['client'].unique():
        client_data = df[df['client'] == client_id]
        plt.plot(client_data['round'], client_data['l2_norm'], 
                marker='o', label=f'Client {client_id}', linewidth=2, markersize=8)
    
    plt.xlabel('Training Round', fontsize=14)
    plt.ylabel('L2 Norm of Weights', fontsize=14)
    plt.title('Evolution of Client Weight Norms During Training', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/client_weight_norms_evolution.png", dpi=300)
    plt.show()
    
    # Plot 2: Box plot showing distribution of norms at each round
    plt.figure(figsize=(12, 8))
    df_pivot = df.pivot(index='client', columns='round', values='l2_norm')
    df_pivot.boxplot(figsize=(12, 8))
    plt.xlabel('Training Round', fontsize=14)
    plt.ylabel('L2 Norm of Weights', fontsize=14)
    plt.title('Distribution of Client Weight Norms Across Rounds', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/weight_norms_distribution.png", dpi=300)
    plt.show()
    
    # Plot 3: Heatmap of weight norms
    plt.figure(figsize=(10, 8))
    pivot_df = df.pivot(index='client', columns='round', values='l2_norm')
    sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='viridis', 
                cbar_kws={'label': 'L2 Norm'})
    plt.xlabel('Training Round', fontsize=14)
    plt.ylabel('Client ID', fontsize=14)
    plt.title('Heatmap of Client Weight Norms', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/weight_norms_heatmap.png", dpi=300)
    plt.show()
    
    # Calculate and print statistics
    print("\nWeight Norm Statistics:")
    print("="*60)
    
    for round_num in sorted(df['round'].unique()):
        round_data = df[df['round'] == round_num]['l2_norm']
        print(f"\nRound {round_num}:")
        print(f"  Mean L2 norm: {round_data.mean():.4f}")
        print(f"  Std deviation: {round_data.std():.4f}")
        print(f"  Min L2 norm: {round_data.min():.4f} (Client {df[df['round'] == round_num].loc[round_data.idxmin(), 'client']})")
        print(f"  Max L2 norm: {round_data.max():.4f} (Client {df[df['round'] == round_num].loc[round_data.idxmax(), 'client']})")
        
        if len(df['round'].unique()) > 1 and round_num > min(df['round'].unique()):
            prev_round = max([r for r in df['round'].unique() if r < round_num])
            prev_mean = df[df['round'] == prev_round]['l2_norm'].mean()
            change = ((round_data.mean() - prev_mean) / prev_mean) * 100
            print(f"  Change from round {prev_round}: {change:+.2f}%")
    
    # Save statistics to CSV
    df.to_csv(f"{output_dir}/weight_norms_data.csv", index=False)
    
    return df

def analyze_layer_wise_norms(client_weights, output_dir="weight_norm_analysis"):
    """
    Analyze weight norms at the layer level to identify which layers contribute most.
    
    Parameters:
    -----------
    client_weights : dict
        Nested dictionary {round: {client: weights_dict}}
    output_dir : str
        Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get layer names from first available client
    first_round = min(client_weights.keys())
    first_client = min(client_weights[first_round].keys())
    layer_names = list(client_weights[first_round][first_client].keys())
    
    # Calculate layer-wise norms
    layer_data = []
    for round_num, clients in client_weights.items():
        for client_id, weights in clients.items():
            layer_norms = calculate_layer_wise_norms(weights)
            for layer_name, norm in layer_norms.items():
                layer_data.append({
                    'round': round_num,
                    'client': client_id,
                    'layer': layer_name,
                    'l2_norm': norm
                })
    
    df_layers = pd.DataFrame(layer_data)
    
    # Plot top contributing layers
    # Average norm per layer across all clients and rounds
    avg_layer_norms = df_layers.groupby('layer')['l2_norm'].mean().sort_values(ascending=False)
    top_layers = avg_layer_norms.head(10).index.tolist()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    # Plot 1: Top layers contribution over rounds (averaged across clients)
    ax1 = axes[0]
    for layer in top_layers[:5]:  # Top 5 layers
        layer_avg = df_layers[df_layers['layer'] == layer].groupby('round')['l2_norm'].mean()
        ax1.plot(layer_avg.index, layer_avg.values, marker='o', label=layer, linewidth=2)
    
    ax1.set_xlabel('Training Round', fontsize=12)
    ax1.set_ylabel('Average L2 Norm', fontsize=12)
    ax1.set_title('Top 5 Layers by Average L2 Norm', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Stacked bar chart showing relative contribution
    ax2 = axes[1]
    rounds = sorted(df_layers['round'].unique())
    layer_contributions = {}
    
    for layer in layer_names:
        layer_contributions[layer] = []
        for round_num in rounds:
            avg_norm = df_layers[(df_layers['layer'] == layer) & 
                               (df_layers['round'] == round_num)]['l2_norm'].mean()
            layer_contributions[layer].append(avg_norm)
    
    # Sort layers by average contribution
    sorted_layers = sorted(layer_contributions.keys(), 
                          key=lambda x: sum(layer_contributions[x]), 
                          reverse=True)
    
    # Plot only top 10 layers for readability
    bottom = np.zeros(len(rounds))
    for i, layer in enumerate(sorted_layers[:10]):
        ax2.bar(rounds, layer_contributions[layer], bottom=bottom, 
               label=layer if i < 5 else None)  # Only show top 5 in legend
        bottom += layer_contributions[layer]
    
    ax2.set_xlabel('Training Round', fontsize=12)
    ax2.set_ylabel('Cumulative L2 Norm', fontsize=12)
    ax2.set_title('Layer-wise Contribution to Total L2 Norm', fontsize=14)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/layer_wise_analysis.png", dpi=300)
    plt.show()
    
    return df_layers

def main():
    # Configuration
    checkpoint_dir = "output/federated_exp_cifarnoiid/checkpoints" 
    checkpointed_rounds = [10, 20, 50, 100,150]  # Checkpointed rounds to analyze
    num_clients = 10
    output_dir = "output/federated_exp_cifarnoiid/weight_norm_analysis"
    
    print("Loading client checkpoints...")
    client_weights = load_client_checkpoints(checkpoint_dir, checkpointed_rounds, num_clients)
    
    print("\nAnalyzing weight norms...")
    df_norms = analyze_weight_norms(client_weights, output_dir)
    
    print("\nAnalyzing layer-wise contributions...")
    df_layers = analyze_layer_wise_norms(client_weights, output_dir)
    
    # Additional analysis: Rate of change
    print("\n\nRate of Weight Norm Change:")
    print("="*60)
    
    rounds = sorted(df_norms['round'].unique())
    if len(rounds) > 1:
        for i in range(1, len(rounds)):
            prev_round = rounds[i-1]
            curr_round = rounds[i]
            
            prev_norms = df_norms[df_norms['round'] == prev_round]['l2_norm'].values
            curr_norms = df_norms[df_norms['round'] == curr_round]['l2_norm'].values
            
            # Calculate percentage change for each client
            changes = ((curr_norms - prev_norms) / prev_norms) * 100
            
            print(f"\nFrom round {prev_round} to {curr_round}:")
            print(f"  Average change: {changes.mean():+.2f}%")
            print(f"  Max increase: {changes.max():+.2f}%")
            print(f"  Max decrease: {changes.min():+.2f}%")

if __name__ == "__main__":
    main()