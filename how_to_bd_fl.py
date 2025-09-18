import torch
import os
from typing import Dict

from compare_updates import extract_weights_only
from aggregation_experiments import parameters_dict_to_vector_flt, should_include_key

def compute_update_norm(local_model, global_model) -> torch.Tensor:
        """
        Compute the L2 norm of the difference between two models. (like Clip&Noise does)
        """
        update = {}
        global_state = {name: param for name, param in global_model.named_parameters()}
        local_state = {name: param for name, param in local_model.named_parameters()}

        for key in local_state:
            update[key] = local_state[key] - global_state[key]

        weights_only = extract_weights_only(update)
        update_vector = parameters_dict_to_vector_flt(weights_only)

        return torch.norm(update_vector)

def compute_local_encoder(global_model, poisoned_model, clean_local_model, num_clients: int, learning_rate: float, train_and_scale=False) -> Dict:
    """
    Computes the local encoder L using the equation: L = n/k(X-G) + G
    
    Args:
        global_model: Global encoder model 
        poisoned_model: Poisoned encoder model 
        num_clients: Number of clients (n)
        learning_rate: Learning rate (k)
        clean_local_model: Clean local model for train-and-scale factor computation
    
    Returns:
        The computed local encoder dictionary
    """
    
    # Save first checkpoint structure
    #first_checkpoint = global_model
    
    # Extract state dicts 
    global_state = global_model.state_dict()
    poisoned_state = poisoned_model.state_dict()
    #clean_state = clean_local_model.state_dict()    

    # Calculate scaling factor
    # If train_and_scale, estimate factor with paper's formula (factor = norm(G^t)/(norm(X)-norm(G))
    if train_and_scale:
        # Extract parameters for norm computation
        #global_params = []
        #poisoned_params = []
        #clean_params = []
        
        #for key in global_state.keys():
        #    if should_include_key(key):
        #        global_params.append(global_state[key].view(-1))
        #        poisoned_params.append(poisoned_state[key].view(-1))
         #       clean_params.append(clean_state[key].view(-1))

        # Compute norms
        #global_norm = torch.norm(torch.cat(global_params))
        #norm = torch.norm(torch.cat(poisoned_params))
        #good_norm = torch.norm(torch.cat(clean_params))
        print("Computing norms for train-and-scale factor...")

        benign = compute_update_norm(clean_local_model, global_model) # Norm of the benign update (estimation of S as appendix on bagdasarian paper)
        poison = compute_update_norm(poisoned_model, global_model) # norm(X-G^t)

        factor = benign / poison
    else:
        factor = num_clients / learning_rate

    #print(f'Good norm (S): {benign}, X-G^t norm: {poison}, Factor: {factor}')
    
    # Initialize local encoder state
    local_state = {}
    
    # Process each layer
    for key in global_state.keys():
        if key in poisoned_state:
            # Convert factor to match tensor type
            factor_tensor = torch.tensor(factor, 
                                      dtype=global_state[key].dtype,
                                      device=global_state[key].device)
            
            # Compute L = n/k(X-G) + G using tensor operations
            diff = poisoned_state[key] - global_state[key]
            diff = diff * factor_tensor
            local_state[key] = global_state[key] + diff 
        else:
            raise KeyError(f"Key {key} not found in poisoned model")

    # Compute final norm for logging/debug
    local_params = []
    for key in local_state.keys():
        if should_include_key(key):
            local_params.append(local_state[key].view(-1))
    final_norm = torch.norm(torch.cat(local_params))
    print(f"Local encoder computed successfully. Norm:{final_norm}")
    return local_state

if __name__ == "__main__":
    # Test usage
    global_path = "./output/resnet18_per_badaggregation_test/models/model_round99.pth"
    poisoned_path = "./output/resnet18_per_badaggregation_test/global_99_badencoder.pth"
    output_path = "./output/resnet18_per_badaggregation_test/computed_local_encoder.pth"
    num_clients = 10
    learning_rate = 0.25
    
    try:
        compute_local_encoder(
            global_path,
            poisoned_path,
            output_path,
            num_clients,
            learning_rate
        )
        print("Local encoder computed successfully")
    except Exception as e:
        print(f"Error computing local encoder: {str(e)}")