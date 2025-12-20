import torch
import numpy as np
import hdbscan
from typing import List, Dict
import os
import math

from models import get_encoder_architecture
from compare_updates import extract_weights_only

def should_include_key(key: str) -> bool:
    """
    Filter function to determine which parameters to include in norm computations.
    
    We EXCLUDE BatchNorm parameters because:
      1. BatchNorm statistics (running_mean, running_var) are not trained via gradients
      2. BatchNorm weights/biases have different scales than conv/linear layers
      3. Defenses like Clip&Noise should focus on the main model parameters
    
    We also exclude the first BatchNorm layer in ResNet (f.f.1.weight/bias) which
    follows the initial convolution.
    
    Args:
        key: Parameter name (e.g., 'f.f.4.0.conv1.weight', 'f.f.1.bias')
        
    Returns:
        True if parameter should be included in norm computation, False otherwise
    """
    # Esclude tutti i parametri di batch normalization
    if '.bn' in key or '.running_' in key or '.num_batches_tracked' in key:
        return False
    
    # Caso speciale per il primo batch norm layer (f.f.1.weight, f.f.1.bias)
    if '.1.weight' in key or '.1.bias' in key:
        parts = key.split('.')
        if len(parts) == 4 and parts[0] == 'f' and parts[1] == 'f' and parts[2] == '1':
            return False
    
    # Includere solo pesi e bias rimanenti
    if key.endswith('.weight') or key.endswith('.bias'):
        return True
    
    return False

def create_empty_update(reference_update: str, noise_scale: float = 1e-6) -> str:
    """Create an empty update with small noise based on reference structure"""
    ref = torch.load(reference_update)
    ref_state = ref['state_dict'] if isinstance(ref, dict) and 'state_dict' in ref else ref
    
    # Create empty state dict with same structure
    empty_state = {}
    for key, param in ref_state.items():
        if key.split('.')[-1] == 'num_batches_tracked':
            empty_state[key] = param  # Keep batch norm stats
        else:
            # Add small noise to avoid exact zeros
            empty_state[key] = torch.randn_like(param) * noise_scale
    
    # Save with same structure as reference
    if isinstance(ref, dict) and 'state_dict' in ref:
        empty = ref.copy()
        empty['state_dict'] = empty_state
    else:
        empty = empty_state
    return empty

def parameters_dict_to_vector(state_dict: Dict) -> torch.Tensor:
    """Convert state dict parameters to single vector, skipping batch norm stats"""
    vec = []
    for key, param in state_dict.items():
        if 'num_batches_tracked' in key:
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)

def parameters_dict_to_vector_flt(net_dict) -> torch.Tensor:
    """Convert parameters to one vector, following original implementation"""
    vec = []
    for param in net_dict.values():
        vec.append(param.reshape(-1))
    return torch.cat(vec).cuda()

def fed_avg(global_model_path: str, 
            update_paths: List[str], 
            output_path: str, 
            learning_rate: float = 0.25):
    """
    Implements FedAvg algorithm: G_(t+1) = G_t + r/n * sum(weight_updates)
        
    Args:
        global_model_path: Path to the global model G_t
        update_paths: List of paths to the weight updates (L_t - G_t)
        output_path: Path where the new global model will be saved            
        learning_rate: Learning rate r for the update (default: 0.25)
        scaling_factor: Optional factor for normalizing the average after malicious weight scaling (default: None, no scaling)
    
    Returns:
        The updated global model
    """
    # Architecture is always SimCLR, hardcoding "cifar10" just for simplicity
    global_model = get_encoder_architecture(type('args', (object,), {'pretraining_dataset': 'cifar10'})).cuda()

    # Load global model (if it's not first round)
    if global_model_path != 'fs':
        checkpoint = torch.load(global_model_path)
        global_model.load_state_dict(checkpoint['state_dict'])
        print("===== LOADED GLOBAL MODEL =====")     
    else:
        # If first round, set lr to 1.0 for full averaging
        learning_rate = 1.0
        print("===== INITIALIZED GLOBAL MODEL =====")

    global_state = global_model.state_dict()
    
    # Sum all weight updates
    n = len(update_paths)

    # Initialize empty weight update dict with the same structure as the first update
    first_update = torch.load(update_paths[0])['state_dict']
    weight_updates = {key: torch.zeros_like(param) for key, param in first_update.items()}
        

    # === DEBUG CODE ===
    #individual_updates = []
    #for path in update_paths:
    #    update_state = torch.load(path)['state_dict']
    #    update = update_state[key] - global_state[key]
    #    individual_updates.append(torch.norm(parameters_dict_to_vector_flt(update)))

    # print stats for this layer
    #if 'conv' in key:
    #    print(f'Layer: {key}, Update norms: {individual_updates}')
    #    print(f'max/min ratio: {max(individual_updates) / min(individual_updates):.2f}')
    # === END DEBUG CODE ===

    # Accumulate updates
    update_norms = []
    for path in update_paths:
        update_state = torch.load(path)['state_dict']
        
        #debug: compute norm of the updates weights (full encoders)
        vec = parameters_dict_to_vector_flt(extract_weights_only(update_state))
        update_norms.append(torch.norm(vec))
            
        for key in weight_updates.keys():

            if key in update_state and key in global_state:
                weight_updates[key] += (update_state[key] - global_state[key]) # Old (working) logic: deal with full encoders
                #weight_updates[key] += update_state[key] 
            else:
                raise KeyError(f"Key {key} not found in update model")
            
    print(f"Update norms: {update_norms}")
        
    # Apply updates with learning rate
    new_state = {}
    for key in global_state.keys():
        if key in weight_updates:
            new_state[key] = global_state[key] + (learning_rate / n) * weight_updates[key]
        else:
            raise KeyError(f"Key {key} not found in update model")
    output = {'state_dict': new_state,}
    torch.save(output, output_path)
    print(f"Global model saved to {output_path}")
    return output

# =============================================================================
# FLAME DEFENSE (Federated Learning with Adversarial Model Elimination)
# =============================================================================
# FLAME is a Byzantine-robust aggregation method that:
#   1. Computes pairwise cosine distances between client updates
#   2. Clusters updates using HDBSCAN to identify outliers
#   3. Selects only updates from the largest cluster (presumed benign)
#   4. Clips selected updates to median Euclidean distance
#   5. Adds calibrated Gaussian noise for differential privacy
#
# Reference: Nguyen et al., "FLAME: Taming Backdoors in Federated Learning"
# =============================================================================
def flame_aggregate(updates: List[str],
                   global_model: str,
                   min_cluster_size: int = None,
                   epsilon: float = 3705,
                   delta: float = 1e-5) -> Dict:
    """FLAME defense implementation following Algorithm 1 from the paper
    
    Args:
        updates: List of paths to client update files
        global_model: Path to global model file
        min_cluster_size: Minimum cluster size (default: num_clients//2 + 1)
        epsilon: DP epsilon parameter (default: 3705 for image classification)
        delta: DP delta parameter (default: 1e-5)
    """
    n = len(updates)  # number of clients
    min_cluster_size = min_cluster_size or (n // 2 + 1)
    
    # Load global model G_{t-1}
    global_state = torch.load(global_model)
    if isinstance(global_state, dict) and 'state_dict' in global_state:
        global_state = global_state['state_dict']

    # Identify BN keys to exclude from processing
    bn_keys = [k for k in global_state if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k]
    
    # Step 1: Load all client updates W_i
    client_updates = []
    for update_path in updates:
        update = torch.load(update_path)
        if isinstance(update, dict) and 'state_dict' in update:
            state_dict = update['state_dict']
        else:
            state_dict = update
        client_updates.append(state_dict)
    
    # Step 2: Compute cosine distances between all pairs (Line 6 in Algorithm 1)
    n_clients = len(client_updates)
    cos_matrix = torch.zeros(n_clients, n_clients)
    
    # Convert updates to vectors for cosine distance calculation
    update_vectors = []
    for state_dict in client_updates:
        weights_only = extract_weights_only(state_dict)
        update_vector = parameters_dict_to_vector_flt(weights_only)
        update_vectors.append(update_vector)
    
    # Vectorized cosine distance calculation
    update_vector_tensor = torch.stack(update_vectors)
    norms = torch.norm(update_vector_tensor, dim=1, keepdim=True)
    normalized = update_vector_tensor / (norms + 1e-8)
    cos_sim = normalized @ normalized.T
    cos_matrix = 1 - cos_sim
    
    #for i in range(n_clients):
    #    for j in range(n_clients):
    #        if i == j:
    #            cos_matrix[i, j] = 0.0
    #        else:
    #            # Cosine distance = 1 - cosine similarity
    #            cosine_sim = torch.nn.functional.cosine_similarity(
    #                update_vectors[i].unsqueeze(0), 
    #                update_vectors[j].unsqueeze(0)
    #            )
    #            cos_matrix[i, j] = 1 - cosine_sim.item()
    
    print("Cosine distance matrix:")
    print(cos_matrix)
    
    # Step 3: Dynamic clustering using HDBSCAN (Line 7)
    # Convert to float64 (double) for HDBSCAN compatibility
    cos_matrix_np = cos_matrix.cpu().numpy().astype(np.float64)

    # Add numerical stabilization
    cos_matrix_np = np.round(cos_matrix_np, decimals=5)
    # Check matrix simmetry
    if not np.allclose(cos_matrix_np, cos_matrix_np.T, atol=1e-6):
        print("Warning: Cosine distance matrix is not symmetric, correcting...")
        cos_matrix_np = (cos_matrix_np + cos_matrix_np.T) / 2  # Ensure symmetry
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        allow_single_cluster=True,
        metric='precomputed', 
    ).fit(cos_matrix_np)
    
    print(f"Cluster labels: {clusterer.labels_}")
    
    # Find largest cluster (admitted models)
    if clusterer.labels_.max() < 0:
        # All points are outliers, select all
        selected_indices = list(range(n_clients))
    else:
        # Count points in each cluster (excluding outliers with label -1)
        unique_labels, counts = np.unique(clusterer.labels_[clusterer.labels_ >= 0], return_counts=True)
        largest_cluster_label = unique_labels[counts.argmax()]
        selected_indices = [i for i, label in enumerate(clusterer.labels_) 
                          if label == largest_cluster_label]
    
    print(f"Selected indices (admitted models): {selected_indices}")
    L = len(selected_indices)  # Number of admitted models
    
    # Step 4: Compute Euclidean distances to global model (Line 8)
    global_weights = extract_weights_only(global_state)
    global_vector = parameters_dict_to_vector_flt(global_weights)
    euclidean_distances = []
    
    for state_dict in client_updates:
        weights_only = extract_weights_only(state_dict)
        update_vector = parameters_dict_to_vector_flt(weights_only)
        # Euclidean distance between W_i and G_{t-1}
        distance = torch.norm(update_vector - global_vector)
        euclidean_distances.append(distance)
    
    # Step 5: Compute adaptive clipping bound S_t as median (Line 9)
    euclidean_distances_tensor = torch.stack(euclidean_distances)
    S_t = torch.median(euclidean_distances_tensor).item()
    print(f"Adaptive clipping bound S_t: {S_t}")
    
    # Step 6: Clip admitted models (Lines 10-11)
    clipped_updates = []
    for idx in selected_indices:
        # Get the update for admitted model
        update = client_updates[idx]
        e_idx = euclidean_distances[idx].item()
        
        # Compute clipping parameter gamma
        gamma = min(1.0, S_t / (e_idx + 1e-8)) # Avoid division by zero
        
        # Apply clipping: W_c = G_{t-1} + gamma * (W - G_{t-1})
        clipped_update = {}
        for key in update:
            if key in bn_keys:
                clipped_update[key] = update[key]
            else:
                clipped_update[key] = global_state[key] + gamma * (update[key] - global_state[key])
        
        clipped_updates.append(clipped_update)
    
    # Step 7: Aggregate clipped models (Line 12)
    aggregated = {}
    for key in global_state:
        if key in bn_keys: 
            aggregated[key] = clipped_updates[0][key].clone()  # Use first update as base
        else:
            # Average the clipped updates
            stacked = torch.stack([update[key].float() for update in clipped_updates])
            aggregated[key] = torch.mean(stacked, dim=0).to(global_state[key].dtype)
    
    # Step 8: Compute adaptive noise level (Line 13)
    lambda_factor = (1.0 / epsilon) * math.sqrt(2.0 * math.log(1.25 / delta))
    sigma = lambda_factor * S_t / max(L,1)  # Scale noise by cluster size
    print(f"Noise level sigma: {sigma}")
    
    # Step 9: Add Gaussian noise (Line 14)
    for key in aggregated:
        if key not in bn_keys:
            noise = torch.randn_like(aggregated[key]) * sigma
            aggregated[key] += noise
    
    return aggregated


# =============================================================================
# CLIP & NOISE DEFENSE
# =============================================================================
# A simpler alternative to FLAME that:
#   1. Computes L2 norm of each client update (excluding BatchNorm)
#   2. Discards outliers with norm > median * clip_factor
#   3. Clips remaining updates to median norm
#   4. Aggregates with FedAvg
#   5. Adds Gaussian noise scaled by differential privacy formula
#
# This defense is less sophisticated than FLAME but computationally cheaper,
# SSL-safe and still effective against naive scaling attacks.
# =============================================================================
def clip_and_noise(updates: List[str],
                  global_model_path: str,
                  learning_rate: float,
                  clip_factor: float = 3.0,  
                  noise_multiplier: float = 0.01,
                  privacy_epsilon: float = 10.0,
                  privacy_delta: float = 1e-3) -> Dict:
    """Clip-and-noise defense against backdoor attacks
    
    Args:
        updates: List of paths to client update files
        global_model: Path to global model file
        clip_factor: Sensibility of the outlier detection, discards all updates with norm higher than median*clip_factor (default: 3.0) 
        noise_multiplier: Noise scale multiplier (default: 0.1)
        privacy_epsilon: DP epsilon parameter (default: 8.0)
        privacy_delta: DP delta parameter (default: 1e-5)
    """
    # Load updates and model
    client_updates = []
    update_norms = []
    
    # Architecture is always SimCLR, hardcoding "cifar10" just for simplicity
    global_model = get_encoder_architecture(type('args', (object,), {'pretraining_dataset': 'cifar10'})).cuda()

    # Load global model (if it's not first round)
    if global_model_path != 'fs':
        checkpoint = torch.load(global_model_path)
        global_model.load_state_dict(checkpoint['state_dict'])
        print("===== LOADED GLOBAL MODEL =====")     
    else:
        print("===== INITIALIZED GLOBAL MODEL =====")

    global_state = global_model.state_dict()

    for update_path in updates:
        update = torch.load(update_path)
        state = update['state_dict'] if isinstance(update, dict) and 'state_dict' in update else update
        
        # Compute update
        for key in state.keys():
            if key in state and key in global_state:
                 state[key] = (state[key] - global_state[key]) 
            else:
                raise KeyError(f"Key {key} not found in update model")
        
        client_updates.append(state)
        
        # Compute norm (of weight parameters only, excluding batchnorm layers) on GPU but move result to CPU (memory efficient)
        update_vec = parameters_dict_to_vector_flt(extract_weights_only(state))
        update_norms.append(torch.norm(update_vec).cpu())
        # Clear GPU memory
        del update_vec
        torch.cuda.empty_cache()

    # Convert norms to tensor and compute threshold
    update_norms = torch.tensor(update_norms)
    clip_threshold = torch.median(update_norms)
    outlier_threshold = clip_threshold * clip_factor
    print(f"Update norms median: {torch.median(update_norms)}")

    # Debug individual update norms
    print("\nIndividual update norms:")
    valid_indices = []
    for i, (path, norm) in enumerate(zip(updates, update_norms)):
        is_outlier = norm > outlier_threshold
        status = "OUTLIER - DISCARDED" if is_outlier else "VALID"
        print(f"Update {i} ({os.path.basename(path)}): {norm:.4f} [{status}]")

        if not is_outlier:
            valid_indices.append(i)
        
    print(f"\nMin norm: {torch.min(update_norms):.4f}")
    print(f"Max norm: {torch.max(update_norms):.4f}")
    print(f"Mean norm: {torch.mean(update_norms):.4f}")
    print(f"Median norm: {torch.median(update_norms):.4f}")


    # Filter out outliers
    valid_updates = [client_updates[i] for i in valid_indices]
    valid_norms = update_norms[valid_indices]
    n_clients = len(valid_updates)
    
    # Clip based on valid updates only!
    clip_threshold = torch.median(valid_norms)

    # Compute noise variance using Differential Privacy formula
    sensitivity = clip_threshold / n_clients
    noise_variance = (2 * math.log(1.25 / privacy_delta) * sensitivity**2) / (privacy_epsilon**2)
    noise_scale = noise_multiplier * math.sqrt(noise_variance)

    # OR (!!!) use heuristic approach:
    #noise_scale = noise_multiplier * (clip_threshold / n_clients)
    
    print(f"Noise scale: {noise_scale:.4f}")
    
    # Load global model
    #global_state = torch.load(global_model)
    #global_state = global_state['state_dict'] if isinstance(global_state, dict) and 'state_dict' in global_state else global_state
    
    # Process updates one at a time to save memory
    aggregated = {}
    for key in global_state:
        #if 'num_batches_tracked' in key:
        #    aggregated[key] = client_updates[0][key]
        #    continue
        
        # Initialize accumulator on CPU
        update_sum = torch.zeros_like(global_state[key], dtype=torch.float32).cpu()
        
        # Process each update
        for update, norm in zip(valid_updates, valid_norms):
            # Calculate scaling factor for clipping
            scale = min(1.0, clip_threshold / (norm + 1e-8))
            # Convert to float32 before scaling
            update_tensor = update[key].to(torch.float32)
            update_sum += (update_tensor * scale).cpu()
            
        # Move final result to GPU only when needed
        aggregated[key] = (global_state[key].to(torch.float32) + ((learning_rate*update_sum) / n_clients).cuda().to(global_state[key].dtype)) 
        
        # Add noise (only to weights)
        if should_include_key(key):
            noise = torch.randn_like(aggregated[key], dtype=torch.float32) * noise_scale
            aggregated[key] = (aggregated[key].to(torch.float32) + noise).to(global_state[key].dtype)
            
        # Clear some memory
        del update_sum
        torch.cuda.empty_cache()
            
    return aggregated

if __name__ == "__main__":
    # Define paths
    #global_model_path = ""
    global_model_path = ""  # Path to the global model
    output_path = ""
    updates = []
    '''
    SCRIPT FOR TESTING WITH 100 "EMPTY" UPDATES

    base_dir = "./output/aggregation_test/"
    os.makedirs(base_dir, exist_ok=True)
    bad_update_path = 'output/cifar10/svhn_federated_backdoor/badaggregation_test.pth'

    # Create 99 empty updates with tiny noise
    updates = []
    for i in range(99):
        empty_update = create_empty_update(global_model_path, noise_scale=1e-6)
        temp_path = os.path.join(base_dir, f"update_{i}.pth")
        torch.save(empty_update, temp_path)
        updates.append(temp_path)
        print(f"Computed update {i}")
    updates.append(bad_update_path)
    '''
    # Learning rate for FedAvg
    learning_rate = 0.25 # 0.25 default, 1 proof of concept  
    
    # Perform model averaging with FedAvg
    try:
        averaged_model = fed_avg(global_model_path, updates, output_path, learning_rate)
        #print(f"Models successfully averaged with learning rate {learning_rate}")
        #averaged_model = clip_and_noise(updates=updates, global_model_path=global_model_path)
        #averaged_model = flame_aggregate(updates, global_model_path)
        torch.save(averaged_model, output_path)
        print(f"Saved to {output_path}")
        '''
        CLEANUP ROUTINE FOR TESTING WITH 100 "EMPTY" UPDATES

        # Clean up empty updates
        for i in range(99):
            path = os.path.join(base_dir, f"update_{i}.pth")
            os.remove(path)
        print("Cleanup completed")
        '''
    except Exception as e:
        raise RuntimeError(f"Error during model averaging: {e}")