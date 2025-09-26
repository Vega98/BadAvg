import argparse
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

# Add project root to path before imports (datasets import fix)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from models import get_encoder_architecture_usage
from datasets import get_shadow_dataset
from badaggregation import BadAggregation
from aggregation_experiments import should_include_key

def run_federated_attack(encoder_usage_info: str = 'cifar10', 
                        shadow_dataset: str = 'cifar10',
                        downstream_dataset: str = 'stl10',
                        trigger: str = 'trigger_pt_white_21_10_ap_replace.npz',
                        reference_path: str = 'truck',
                        clean_encoder: str = '',
                        num_clients: int = 10, #proof of concept: 2 default 10
                        fed_lr: float = 0.25, #proof of concept:1 defaylt 0.25
                        batch_size: int = 256,
                        lr: float = 1e-3,
                        lambda1: float = 1.0,
                        lambda2: float = 1.0,
                        epochs: int = 1,
                        reference_label: int = 0, # Default 0 also in badencoder.py
                        name: str = 'badaggregation_test',
                        clean_local: str = '',
                        clipnoise: bool = False,
                        neurotoxin_mask: str = None,
                        #previous_global_model: str = None,
                        output_dir: str = '.',
                        ) -> str:
    """Run federated BadEncoder attack with specified parameters"""
    
    # Packaging args for get_shadow_dataset (#1) and get_encoder_architecture_usage (#2) routines
    args = argparse.Namespace(
        reference_file=reference_path,#1
        trigger_file=f'./trigger/{trigger}',#1
        shadow_dataset=shadow_dataset,#1
        data_dir=f'./data/{shadow_dataset.split("_")[0]}/',#1
        reference_label=reference_label,#1
        encoder_usage_info=encoder_usage_info,#2
    )

    # Print args
    print(args)
    
    # Create directories
    pretrained_encoder = clean_encoder
    #pretrained_encoder = f'./output/{clean_encoder}'
    results_dir = os.path.join(output_dir, "temp")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create data loader (testing unused)
    shadow_data, memory_data, test_data_clean, test_data_backdoor = get_shadow_dataset(args)
    train_loader = DataLoader(shadow_data, batch_size=batch_size, 
                            shuffle=True, num_workers=4, pin_memory=True, 
                            drop_last=True)

    # Initialize models
    clean_global_model = get_encoder_architecture_usage(args).cuda()
    clean_local_model = get_encoder_architecture_usage(args).cuda()
    poisoned_local_model = get_encoder_architecture_usage(args).cuda()

    

    
    if pretrained_encoder:
        checkpoint = torch.load(pretrained_encoder)
        #poisoned = torch.load(poisoned_model)
        clean_global_model.load_state_dict(checkpoint['state_dict'])
        poisoned_local_model.load_state_dict(checkpoint['state_dict'])
        #poisoned_local_model.load_state_dict(poisoned['state_dict'])

        # Load also the clean local model of the malicious client
        clean = torch.load(clean_local)
        clean_local_model.load_state_dict(clean['state_dict'])
    
    # Initialize attack and optimizer
    fed_attacker = BadAggregation(
        num_clients=num_clients,
        learning_rate=fed_lr,
        device='cuda'
    )
    # Optimizing the poisoned local model 
    #optimizer = torch.optim.SGD(poisoned_local_model.parameters(), lr=lr, weight_decay=5e-4, momentum=0.9)
    #optimizer = torch.optim.Adam(poisoned_local_model.parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(poisoned_local_model.parameters(), lr, weight_decay=1e-2)

    # Option 1: linear tolerance decay
    # Loss: -2.678950, Loss0: -0.748649, Loss1: -0.994055,  Loss2: -0.936245,
    #tolerances = np.linspace(3.0, tolerance, epochs) 

    # Option 2: exponential decay (from 3.0 to 1.5) #tolerance = 1.5 / 1.05
    tolerances = 1.05 + (3.0 - 1.05) * np.exp(-5 * np.arange(epochs) / epochs)
    
    # Training loop
    for epoch in range(1, epochs+1):
        print("=================================================")     

        # Soft reset the norm for first n-2 epochs (last 1 epochs is hard constraining during training)
        #if clipnoise and (epoch < epochs-1):
        #    with torch.no_grad():
        #        # Compute the current norm
        #        current_norm = fed_attacker.compute_update_norm(poisoned_local_model.f, clean_global_model.f)
        #        clean_norm = fed_attacker.compute_update_norm(clean_local_model.f, clean_global_model.f)
        #
        #        # If above clean norm, pull back
        #        if current_norm > (tolerance * clean_norm):
        #            # Compute the scaling factor
        #            scaling_factor = (tolerance * clean_norm) / current_norm
        #            for (p_name, poisoned_param), (_, global_param) in zip(poisoned_local_model.f.named_parameters(), clean_global_model.f.named_parameters()):
        #                update = poisoned_param.data - global_param.data
        #                poisoned_param.data = global_param.data + (scaling_factor * update)

        loss = fed_attacker.train_step(
            global_model=clean_global_model.f,
            local_encoder=poisoned_local_model.f,
            clean_local_encoder=clean_local_model.f, # This is the clean local model of malicious client
            optimizer=optimizer,
            data_loader=train_loader,
            epoch=epoch, # current epoch
            epochs=epochs, # total epochs
            lambda1=lambda1,
            lambda2=lambda2,
            tolerance=tolerances[epoch-1],
            clipnoise=clipnoise,
            neurotoxin_mask=neurotoxin_mask,
            #previous_global_model=previous_global.f 
        )

        # Save final model/update
        state_dict = poisoned_local_model.state_dict()
        if epoch % epochs == 0:

            

            torch.save({
                'epoch': epochs,
                'state_dict': state_dict,
                'optimizer': optimizer.state_dict()
            }, f"{results_dir}/{name}.pth")

    # Clean up GPU memory
    del clean_global_model
    del poisoned_local_model
    del optimizer
    del fed_attacker
    torch.cuda.empty_cache()

    return f"{results_dir}/{name}.pth"

def main():
    """Parse command line arguments and call run_federated_attack"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--encoder_usage_info', default='cifar10', type=str)
    parser.add_argument('--shadow_dataset', default='cifar10', type=str)
    parser.add_argument('--downstream_dataset', default='stl10', type=str)
    parser.add_argument('--trigger', default='trigger_pt_white_21_10_ap_replace.npz', type=str)
    parser.add_argument('--reference', default='truck', type=str)
    parser.add_argument('--clean_encoder', default='', type=str)
    parser.add_argument('--num_clients', default=10, type=int)
    parser.add_argument('--fed_lr', default=0.25, type=float)
    args = parser.parse_args()
    
    path = run_federated_attack(**vars(args))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main()
    else:
        # Example usage
        path = run_federated_attack(
            encoder_usage_info='cifar10',
            shadow_dataset='cifar10',
            downstream_dataset='stl10',
            trigger='trigger_pt_white_21_10_ap_replace.npz',
            reference='truck',
            clean_encoder='/home/vega/Documenti/BadEncoder/output/resnet18_per_badaggregation_test/models/model_round99.pth',
            clean_local='/home/vega/Documenti/BadEncoder/output/resnet18_per_badaggregation_test/temp_round/pretrain/model_ft_c3.pth',
            name='badavg_test',
            num_clients=10,
            epochs=2,
            fed_lr=0.25,
            neurotoxin_mask=''
        )