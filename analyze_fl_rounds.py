import torch
import random
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from compare_updates import compare_weight_parameters
from scripts.run_badaggregation import run_federated_attack
from scripts.run_federated import evaluate_model
from aggregation_experiments import fed_avg, should_include_key
import pandas as pd

def analyze_federated_rounds(base_dir, num_rounds):
    """
    Analyze each round for weight update similarity and attack success rate.
    """
    output_dir = base_dir+"/asr_analysis"
    os.makedirs(output_dir, exist_ok=True)

    # Create metrics file
    metrics_file = os.path.join(output_dir, "metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write("round,convergence,asr\n")
    
    # Process each round (starting from 1, because round 0 doesn't have a previous aggregated model)
    for round_num in tqdm(range(1, num_rounds), desc="Analyzing rounds"):
        round_dir = f"{base_dir}/round_{round_num}"
        prev_round_dir = f"{base_dir}/round_{round_num-1}"
        
        try:
            # Initialize metrics
            param_sum = 0.0
            total_elements = 0

            # Prepare paths to all client updates
            client_update_paths = []
            for i in range(10):
                update_path = f"{round_dir}/pretrain/update_ft_c{i}.pth"
                if os.path.exists(update_path):
                    client_update_paths.append(update_path)

            if not client_update_paths:
                print(f"No updates found for round {round_num}")
                continue

            # Initialize aggregation structure
            first_update = torch.load(client_update_paths[0])
            first_state = first_update['state_dict'] if isinstance(first_update, dict) and 'state_dict' in first_update else first_update
            
            aggregated_updates = {}
            for key in first_state.keys():
                if should_include_key(key):
                    aggregated_updates[key] = torch.zeros_like(first_state[key])

            # Sum updates from all clients
            for path in client_update_paths:
                try:
                    update = torch.load(path)
                    state = update['state_dict'] if isinstance(update, dict) and 'state_dict' in update else update
                    
                    for key in aggregated_updates.keys():
                        if key in state:
                            aggregated_updates[key] += state[key]
                except Exception as e:
                    print(f"Error loading client update {path}: {e}")

            # Compute metrics
            for key in aggregated_updates.keys():
                param_sum += torch.sum(aggregated_updates[key]).item()
            
            # Scaling and sign flipping for readability
            convergence = - (1 / 1000) * param_sum

            # Perform attack to get ASR
            global_model_path = f"{prev_round_dir}/aggregated_model.pth"
            poisoned_path = run_federated_attack(
                        clean_encoder = global_model_path,
                        num_clients = 10,  # Hardcoding 10 clients
                        fed_lr = 0.25, 
                        epochs = 1, # Hardcoding 1 epoch for the attack for testing
                        name = 'temp_bd_update.pth',)
            
            # Aggregate with other updates (swap a random benign update with the poisoned one)
            client_update_paths[random.randrange(len(client_update_paths))] = poisoned_path      
            aggregated_path = os.path.join(output_dir, "aggregated_model.pth")
    
            fed_avg(
            global_model_path=global_model_path,
            update_paths=client_update_paths,
            output_path=aggregated_path,
            learning_rate=0.25 #hardcoded, for now
            )

            # Evaluate current model and get ASR
            ba, asr = evaluate_model(aggregated_path, round_num, output_dir)
            
            # Save metrics
            with open(metrics_file, 'a') as f:
                f.write(f"{round_num},{convergence:.2f},{asr:.2f}\n")

        

        except Exception as e:
            print(f"Error processing round {round_num}: {e}")
            continue
    
    #Read metrics from CSV
    df = pd.read_csv(metrics_file)
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: ASR vs Convergence
    ax1.plot(df['convergence'], df['asr'], '-o', alpha=0.6)
    ax1.set_xlabel('Updates convergence')
    ax1.set_ylabel('Attack Success Rate (ASR)')
    ax1.set_title('ASR vs Updates convergence')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: ASR vs Round Number
    ax2.plot(df['round'], df['asr'], '-o', alpha=0.6)
    ax2.set_xlabel('Round Number')
    ax2.set_ylabel('Attack Success Rate (ASR)')
    ax2.set_title('ASR vs Round Number')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Convergence vs Round Number
    ax3.plot(df['round'], df['convergence'], '-o', alpha=0.6)
    ax3.set_xlabel('Round Number')
    ax3.set_ylabel('Updates convergence')
    ax3.set_title('Updates convergence vs Round Number')
    ax3.grid(True, alpha=0.3)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f"{output_dir}/metrics_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Set these parameters based on your setup
    BASE_DIR = "output/federated_exp_20250419_143529"  # Directory containing round folders
    NUM_ROUNDS = 100                                  # Number of rounds to analyze
    
    analyze_federated_rounds(BASE_DIR, NUM_ROUNDS)

