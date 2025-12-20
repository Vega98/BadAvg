import os
import torch
import random
import numpy as np
from typing import List, Tuple
import subprocess
from pathlib import Path
from aggregation_experiments import fed_avg, flame_aggregate, clip_and_noise
from scripts.run_badaggregation import run_federated_attack

def run_pretraining(
    pretraining_dataset: str,
    global_model_path: str,
    client_id: int,
    dataset_path: str,
    test_dir: str,
    mem_dir: str,
    epochs: int,
    output_dir: str,
    gpu: int,
    current_round: int,
    neurotoxin: int
) -> str:
    """Runs pretraining for a single client"""
    #output_path = os.path.join(output_dir, f"client_{client_id}")
    #os.makedirs(output_path, exist_ok=True)
    
    # SimCLR paper's optimal batch size is 4096, BadEncoder uses 256
    cmd = f"""python3 pretraining_encoder.py \
        --checkpoint {global_model_path} \
        --epochs {epochs} \
        --train_dir {dataset_path} \
        --mem_dir {mem_dir} \
        --test_dir {test_dir} \
        --results_dir {output_dir} \
        --name model_ft_c{client_id} \
        --batch_size 256 \
        --pretraining_dataset {pretraining_dataset} \
        --gpu {gpu} \
        --current_round {current_round} \
        --neurotoxin {neurotoxin} """ 
        
    subprocess.run(cmd, shell=True, check=True)
    return os.path.join(output_dir, f"model_ft_c{client_id}.pth")

def run_badencoder(
    model_path: str,
    client_id: int,
    output_dir: str,
    trigger_path: str,
    reference_path: str,
    backdoor_epochs: int,
    scale_factor: int,
    naive: int,
    clean_local: str,
    gpu: int,
    shadow_dataset: str,
    pretrain_dataset: str
) -> str:
    """Runs BadEncoder attack for a malicious client"""
    #output_path = os.path.join(output_dir, f"backdoor_client_{client_id}")
    #os.makedirs(output_path, exist_ok=True)

    cmd = f"""python3 badencoder.py \
        --pretrained_encoder {model_path} \
        --epochs {backdoor_epochs} \
        --trigger_file {trigger_path} \
        --reference_file {reference_path} \
        --results_dir {output_dir} \
        --name model_bd_c{client_id} \
        --shadow_dataset {shadow_dataset} \
        --encoder_usage_info {pretrain_dataset} \
        --scale_factor {scale_factor} \
        --naive {naive} \
        --clean_local {clean_local} \
        --gpu {gpu}"""
        
    subprocess.run(cmd, shell=True, check=True)
    return os.path.join(output_dir, f"model_bd_c{client_id}.pth")

def aggregation(aggregated_path, client_models, args):
    """
    Aggregates client model updates into a new global model.
    
    Selects the aggregation strategy based on the defense mechanism:
      - 'flame': FLAME defense - clusters updates, clips outliers, adds DP noise
      - 'clipnoise': Clip&Noise defense - clips large updates, adds Gaussian noise
      - 'none' (default): Standard FedAvg without any defense
    
    Args:
        aggregated_path: Output path for the aggregated model
        client_models: List of paths to client model checkpoints
        args: Namespace containing:
            - defense: Defense type ('flame', 'clipnoise', or 'none')
            - global_model_path: Path to current global model ('fs' for from-scratch)
            - learning_rate: FedAvg learning rate
    """ 

    if args.defense == 'flame' and args.global_model_path != 'fs':
        print("Aggregating using FLAME defense.")
        aggregated_state = flame_aggregate(
            updates=client_models,
            global_model=args.global_model_path
        )
        torch.save({'state_dict': aggregated_state}, aggregated_path)
    
    elif args.defense == 'clipnoise' and args.global_model_path != 'fs':
        print("Aggregating using Clip&Noise defense.")
        aggregated_state = clip_and_noise(
            updates=client_models,
            global_model_path=args.global_model_path,
            learning_rate=args.learning_rate
        )
        torch.save({'state_dict': aggregated_state}, aggregated_path)
        
    else: # no defense
        print("No defense mechanism selected. Using standard FedAvg.")
        fed_avg(
        global_model_path=args.global_model_path,
        update_paths=client_models,
        output_path=aggregated_path,
        learning_rate=args.learning_rate
    )
    

def federated_poison_round(
    pretraining_dataset: str,
    dataset_paths: List[str],
    test_dir: str,
    mem_dir: str,
    pretrain_epochs: int,
    backdoor_epochs: int,
    output_dir: str,
    trigger_path: str,
    reference_path: str,
    args
) -> str:
    """
    Simulates one round of federated learning with poisoning attacks.
    
    Args:
        global_model_path: Path to the global model checkpoint
        num_malicious: Number of malicious clients
        num_benign: Number of benign clients
        dataset_paths: List of paths to datasets for each client
        pretrain_epochs: Number of pretraining epochs per client
        output_dir: Directory to store intermediate and final results
        trigger_path: Path to trigger file for BadEncoder
        reference_path: Path to reference file for BadEncoder
        args: Additional arguments for aggregation defense methods
    
    Returns:
        Path to the new global model
    """
    os.makedirs(output_dir, exist_ok=True)
    pretrain_dir = os.path.join(output_dir, "pretrain")
    backdoor_dir = os.path.join(output_dir, "backdoor")
    os.makedirs(pretrain_dir, exist_ok=True)
    os.makedirs(backdoor_dir, exist_ok=True)
    
    # Client 3 is always the attacker for reproducibility across experiments.
    # In a real scenario, this could be randomized: random.sample(range(k), args.num_malicious)
    malicious = [3] 

    # =========================================================================
    # PHASE 1: LOCAL PRETRAINING
    # All clients (benign and malicious) perform SimCLR pretraining on their
    # local data partition. Malicious clients also compute a neurotoxin mask
    # to identify which parameters to modify during the attack.
    # =========================================================================
    client_models = []
    k = args.num_benign + args.num_malicious
    for i in range(k):
        model_path = run_pretraining(
            pretraining_dataset,
            args.global_model_path,
            i,
            dataset_paths[i],
            test_dir,
            mem_dir,
            pretrain_epochs,
            pretrain_dir,
            args.gpu,
            args.current_round,
            neurotoxin = 1 if i in malicious else 0 #put 0 for manual shutdown of old neurotoxin method
        )
        
        #model_path = os.path.join(pretrain_dir, f"model_ft_c{i}.pth")
        client_models.append(model_path)
    
    # =========================================================================
    # PHASE 2: BACKDOOR ATTACK
    # Malicious clients replace their pretrained model with a poisoned version.
    # Two attack strategies are available:
    #   - BAGEL (args.bagel=True): BadEncoder-based attack with gradient scaling
    #   - BadAvg (args.bagel=False): Aggregation-aware attack
    # =========================================================================
    print(f"Malicious clients for this round: {malicious}")
    for i in malicious:
        if args.bagel:
            # Apply BadEncoder attack (bagel)
            poisoned_path = run_badencoder(
                args.global_model_path, # Run finetune on global, NOT clean local
                i,
                backdoor_dir,
                trigger_path,
                reference_path,
                backdoor_epochs,
                scale_factor=100 if not args.naive else 0, # Scale factor for BAGEL, but if doing "naive" (bagdasarian+badencoder) then no scale factor
                naive=args.naive,
                clean_local = client_models[i], # For train-and-scale
                gpu=args.gpu,
                shadow_dataset= args.shadow_dataset,
                pretrain_dataset= args.pretrain_dataset
                )
            client_models[i] = poisoned_path
        else:
            # Apply BadAggregation attack
            clipnoise = True if args.defense == 'clipnoise' else False
            neurotoxin_mask = f"{pretrain_dir}/neurotoxin_mask.pth"
            poisoned_path = run_federated_attack(
                            clean_encoder = args.global_model_path,
                            num_clients = args.num_benign + args.num_malicious,
                            fed_lr = args.learning_rate, 
                            epochs = backdoor_epochs, 
                            name = f"model_bd_c{i}.pth",
                            clean_local= client_models[i],
                            clipnoise= clipnoise,
                            neurotoxin_mask= neurotoxin_mask,
                            #previous_global_model= args.previous_global_model,
                            encoder_usage_info= pretraining_dataset,
                            shadow_dataset= args.shadow_dataset,
                            output_dir = output_dir,
                            reference_path = reference_path
                            )
            client_models[i] = poisoned_path
    
    # =========================================================================
    # PHASE 3: AGGREGATION
    # All client models (including poisoned ones) are aggregated into a new
    # global model using the selected aggregation rule (FedAvg or defense).
    # =========================================================================
    aggregated_path = os.path.join(output_dir, "aggregated_model.pth")
    aggregation(aggregated_path, 
                client_models, 
                args) 
    
    return aggregated_path

def federated_round(
    pretraining_dataset: str,
    dataset_paths: List[str],
    test_dir: str,
    mem_dir: str,
    pretrain_epochs: int,
    output_dir: str,
    args
) -> str:
    """
    Simulates one round of federated learning, without any poisoning.
    
    Args:
        global_model_path: Path to the global model checkpoint
        dataset_paths: List of paths to datasets for each client
        pretrain_epochs: Number of pretraining epochs per client
        output_dir: Directory to store intermediate and final results
        args: Additional arguments for aggregation defense methods
    
    Returns:
        Path to the new global model
    """
    os.makedirs(output_dir, exist_ok=True)
    pretrain_dir = os.path.join(output_dir, "pretrain")
    os.makedirs(pretrain_dir, exist_ok=True)
    
    # 1. Pre-training phase for all clients
    client_models = []
    num_clients = args.num_benign + args.num_malicious
    for i in range(num_clients):
        model_path = run_pretraining(
            pretraining_dataset,
            args.global_model_path,
            i,
            dataset_paths[i],
            test_dir,
            mem_dir,
            pretrain_epochs,
            pretrain_dir,
            args.gpu,
            args.current_round,
            neurotoxin=0  # No neurotoxin mask needed in clean rounds
        )
        #model_path = os.path.join(pretrain_dir, f"model_ft_c{i}.pth")
        client_models.append(model_path)
    
    # 2. Aggregate all models
    aggregated_path = os.path.join(output_dir, "aggregated_model.pth")
    aggregation(aggregated_path, 
                client_models, 
                args)  
    
    return aggregated_path

if __name__ == "__main__":
    # Example usage
    round_output = federated_poison_round(
        global_model_path="./output/cifar10/clean_encoder/model_100.pth",
        num_malicious=3,
        num_benign=7,
        dataset_paths=[f"./data/cifar10/partitions/iid/partition_{i}.npz" for i in range(10)],
        test_dir = "./data/cifar10/partitions/iid/test.npz",
        pretrain_epochs=20, 
        backdoor_epochs=4,
        output_dir="./output/federated",
        trigger_path="./trigger/trigger_pt_white_21_10_ap_replace.npz",
        reference_path="./reference/cifar10/truck.npz",
        scale_factor=1.0
    )
    # Evaluate the aggregated model (maybe do it in the calling script)
    print(f"Round completed. New global model saved at: {round_output}")