import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def extract_weights_only(model_dict, prefix=""):
    """Extract only weight tensors from nested dictionaries, filtering out optimizer states and batch norm."""
    result = {}
   
    def is_weight_parameter(key, value):
        if 'optimizer' in key:
            return False
        if any(bn_term in key for bn_term in ['bn', 'num_batches_tracked', 'running_mean', 'running_var']):
            return False
        if not isinstance(value, torch.Tensor):
            return False
        return True
   
    for key, value in model_dict.items():
        full_key = f"{prefix}.{key}" if prefix else key
       
        if isinstance(value, torch.Tensor):
            if is_weight_parameter(full_key, value):
                result[full_key] = value.detach()
        elif isinstance(value, dict):
            nested_result = extract_weights_only(value, full_key)
            result.update(nested_result)
   
    return result

def compute_update_norm(model_state, global_state):
    """
    Compute L2 norm of the update (model - global)
   
    Args:
        model_state: Model state dict
        global_state: Global model state dict
       
    Returns:
        L2 norm of the update
    """
    # Extract weights only
    model_weights = extract_weights_only(model_state)
    global_weights = extract_weights_only(global_state)
   
    # Get common layers
    common_layers = [key for key in model_weights if key in global_weights]
   
    # Compute update and concatenate
    update_vec = []
    for layer in common_layers:
        update = model_weights[layer] - global_weights[layer]
        update_vec.append(update.flatten())
   
    # Concatenate all updates
    update_vec = torch.cat(update_vec)
   
    # Compute L2 norm
    norm = torch.norm(update_vec).item()
   
    return norm

def plot_update_norms(models_info, global_model, figsize=(10, 6),
                      color_map=None, show_values=True):
    """
    Plot update norms as horizontal bar chart
   
    Args:
        models_info: Dictionary of {name: {'model': model_path_or_state_dict, 'asr': attack_success_rate}}
        global_model: Global model path or state dict
        figsize: Figure size tuple
        color_map: Dictionary mapping base method names to colors, or None for automatic
        show_values: Whether to show norm values on bars
    """
    # Load global model if path
    if isinstance(global_model, str):
        global_state = torch.load(global_model)
    else:
        global_state = global_model
   
    # Compute norms for all models
    results = []
   
    for name, info in models_info.items():
        # Load model if path
        if isinstance(info['model'], str):
            model_state = torch.load(info['model'])
        else:
            model_state = info['model']
       
        # Compute norm
        norm = compute_update_norm(model_state, global_state)
       
        # Get ASR if provided
        asr = info.get('asr', None)
       
        results.append({
            'name': name,
            'norm': norm,
            'asr': asr
        })
       
        print(f"Computed norm for {name}: {norm:.6f}")
   
    # Sort by norm (descending)
    results.sort(key=lambda x: x['norm'], reverse=True)
   
    # Default color mapping
    if color_map is None:
        color_map = {
            'clean': '#2E8B57',  # Sea green
            'naive': '#DC143C',  # Crimson
            'badavg': '#4169E1', # Royal blue
            'bagel': '#FF8C00'   # Dark orange
        }
   
    # Function to get color for a model
    def get_color(name):
        name_lower = name.lower()
        for base, color in color_map.items():
            if base in name_lower:
                return color
        # Default color if not found
        return '#808080'  # Gray
   
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Enable LaTeX style text rendering
    plt.rcParams['text.usetex'] = False
    plt.rcParams['mathtext.default'] = 'regular'
   
    # Prepare data for plotting
    names = []
    norms = []
    colors = []
   
    for result in results:
        # Format name - make BadAvg bold
        if result['name'] == 'BadAvg':
            display_name = r'$\mathbf{BadAvg}$'
        else:
            display_name = result['name']
        
        # Format label with ASR on second line if available
        if result['asr'] is not None:
            # Format ASR percentage
            asr_pct = f"{result['asr']*100:.2f}"
            # Create two-line label with name on first line and ASR on second line
            label = display_name + '\n' + r'$\it{(ASR:\ ' + asr_pct + r'\%)}$'
        else:
            label = display_name
       
        names.append(label)
        norms.append(result['norm'])
        colors.append(get_color(result['name']))
   
    # Create horizontal bar plot
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, norms, color=colors, edgecolor='black', linewidth=1)
   
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=16, verticalalignment='center')
    ax.invert_yaxis()  # Highest norm at top
    ax.set_xlabel('Update Norm (L2)', fontsize=18)
    plt.subplots_adjust(left=0.3)
   
    # Add value labels on bars
    if show_values:
        for i, (bar, norm) in enumerate(zip(bars, norms)):
            width = bar.get_width()
            ax.text(width + max(norms) * 0.01, bar.get_y() + bar.get_height()/2,
                    f'{norm:.2f}', ha='left', va='center', fontsize=14)
   
    # Add grid
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
   
    # Set x-axis limit with some padding
    ax.set_xlim(0, max(norms) * 1.15)
   
    # Add vertical line at x=0
    ax.axvline(x=0, color='black', linewidth=1)
   
    # Tight layout
    plt.tight_layout()
   
    # Print summary statistics
    print(f"\nSummary:")
    print(f"  Max norm: {np.max(norms):.6f}")
    print(f"  Min norm: {np.min(norms):.6f}")
    print(f"  Mean norm: {np.mean(norms):.6f}")
    print(f"  Std dev: {np.std(norms):.6f}")
   
    return fig, ax

# Example usage
if __name__ == "__main__":
    # Define models with their ASR values
    models_info = {
        'Clean': {
            'model': './output/federated_exp_cifarnoiid/temp_round/pretrain/model_ft_c0.pth',
            'asr': 0.1153  # Clean model has 0% attack success
        },
        'Naive': {
            'model': './output/federated_exp_cifarnoiid/naive.pth',
            'asr': 0.9998  # 85% attack success rate
        },
        'BadAvg': {
            'model': './output/federated_exp_cifarnoiid/badavg.pth',
            'asr': 0.9982
        },
        'BAGEL': {
            'model': './output/federated_exp_cifarnoiid/bagel100.pth',
            'asr': 0.9998
        }
    }
   
    # Load global model
    global_model = torch.load('./output/federated_exp_cifarnoiid/models/model_round198.pth')
   
    # Create visualization
    fig, ax = plot_update_norms(models_info, global_model)
   
    # Save plot
    plt.savefig('update_norms_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
