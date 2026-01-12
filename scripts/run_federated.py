import os
import sys
from datetime import datetime
import subprocess
import json
import argparse
import pandas as pd
import numpy as np
import torch
import threading
import queue
import time
from threading import Lock

import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated_round import federated_poison_round, federated_round

""" EDIT THIS KNOBS TO CHANGE EXPERIMENT SETTINGS """
DATASET_DISTRIBUTION = "iid"  # Dataset distribution among clients ("iid" or "dirichlet" for non-iid)
DOWNSTREAM_DATASET = "gtsrb" # Dataset for evaluation
DEFENSE = 0 # 0 for no defense, 1 for clip&noise (if attack is 0, this is ignored)
STARTING_CLEAN_ROUNDS = 1 # Number of initial clean rounds before starting attack
MAX_ATTACK_ROUND = 1 # Maximum round to perform attack
NUM_ROUNDS = 3 # Total number of federated rounds


# === BASE PATHS (Attention this knobs when moving to a different machine) ===
BASE_DIR = "./"  # Root directory for experiments
DATA_DIR = f"{BASE_DIR}data"                   # Directory containing all datasets
TRIGGER_PATH = f"{BASE_DIR}trigger/trigger_pt_white_21_10_ap_replace.npz"  # Trigger pattern file
REFERENCE_DIR = f"{BASE_DIR}reference"         # Directory containing reference images

# Main parameters (change at will)
BAD_ROUNDS = 10 # Run poison attack every BAD_ROUNDS rounds (-1 to disable)
SKIP_ROUNDS = 5 # -1 to evaluate all rounds, N to evaluate every N rounds
PRETRAIN_DATASET = "stl10" # Dataset for pre-training (either "cifar10" or "stl10")
SHADOW_DATASET = "stl10" # Shadow dataset for attack (either "cifar10" or "stl10")
ATTACK = 1 # 0 for no attack (clean federated experiment), 1 for BadAvg, 2 for BAGEL, 3 for Naive

CHECKPOINT = None  # Set to None if starting from scratch # If starting experiment from a checkpoint, put the path to the checkpoint .pth file here (otherwise None)
RESUME_ROUND = 0 # If starting from checkpoint (or rebooting experiment from certain round), put the round number to resume from (otherwise 0)

# Hardcoded / specific parameters (be sure you know what you are doing if you change these)
NUM_CLIENTS = 10 # Total number of clients for experiment. Unless you change the dataset partitions, keep it at 10.
BAD_CLIENTS = 1 # Attack was designed for 1 attacker, but this can be changed
CLIENT_EPOCHS = 5 # Number of local epochs for each client during pre-training
BACKDOOR_EPOCHS = 10 # Number of local epochs for each attacker during backdoor training (only for poison rounds)
FEDAVG_LEARNING_RATE = 0.25 # Learning rate for FedAvg
TRAINING_GPU_ID = 0 # GPU ID for training (if not sure, leave at 0)
EVAL_GPU_ID = 1 # GPU ID for evaluation (can be same as TRAINING_GPU_ID if only one GPU is available, consider that evaluation happens in parallel with training)
DOWNSTREAM_EPOCHS = "progressive" # Set either 'progressive' or fixed number. Number of epochs to train downstream classifier during evaluation after each round (higher = slightly better accuracy, but slower)
HARDCAP = 1000 # If using progressive downstream epochs, this is the hard cap for max epochs

EVAL_ONLY = False # If True, skips training and only evaluates models in MODELS_DIR
MODELS_DIR = "" # Directory containing models to evaluate (required if EVAL_ONLY is True)
OUTPUT_DIR = f"{BASE_DIR}/output/badavg_{DOWNSTREAM_DATASET}_{DATASET_DISTRIBUTION}_def{DEFENSE}" # Output directory for logs, models, plots


# =============================================================================
# Derived settings and validation
# =============================================================================
# The following logic computes derived globals (REFERENCE_PATH, REFERENCE_LABEL,
# and validates datasets). It's moved into a function so we can recompute these
# when the top-level constants are overridden at runtime.


def compute_derived_settings():
    """Compute REFERENCE_PATH, REFERENCE_LABEL, and validate PRETRAIN_DATASET.
    This function assigns module-level globals so other functions (which use
    globals) pick up the updated values.
    """
    global REFERENCE_PATH, REFERENCE_LABEL, OUTPUT_DIR

    # Checking: pretrain must me either cifar10 or stl10 and pretrain must be different from downstream!
    if PRETRAIN_DATASET == DOWNSTREAM_DATASET:
        #raise ValueError("Pretrain dataset must be different from downstream dataset!")
        print("Warning: Pretrain dataset is the same as downstream dataset!")
    if PRETRAIN_DATASET not in ["cifar10", "stl10"]:
        raise NotImplementedError(f"Unsupported pretrain dataset {PRETRAIN_DATASET}: must be either cifar10 or stl10")

    # Building reference path and reference label for attack.
    if DOWNSTREAM_DATASET == "stl10":
        REFERENCE_PATH = f"{REFERENCE_DIR}/{PRETRAIN_DATASET}/truck.npz"
        REFERENCE_LABEL = 9
    elif DOWNSTREAM_DATASET == "cifar10":
        REFERENCE_PATH = f"{REFERENCE_DIR}/{PRETRAIN_DATASET}/truck.npz"
        REFERENCE_LABEL = 9
    elif DOWNSTREAM_DATASET == "gtsrb":
        REFERENCE_PATH = f"{REFERENCE_DIR}/{PRETRAIN_DATASET}/priority.npz"
        REFERENCE_LABEL = 12
    elif DOWNSTREAM_DATASET == "svhn":
        REFERENCE_PATH = f"{REFERENCE_DIR}/{PRETRAIN_DATASET}/one.npz"
        REFERENCE_LABEL = 1
    else:
        raise NotImplementedError(f"Unsupported downstream dataset {DOWNSTREAM_DATASET}")

    # Recompute OUTPUT_DIR in case related globals changed
    OUTPUT_DIR = f"{BASE_DIR}/output/badavg_{DOWNSTREAM_DATASET}_{DATASET_DISTRIBUTION}_def{DEFENSE}"
    #debug
    #print("DEBUG: REFERENCE_PATH =", REFERENCE_PATH, ", REFERENCE_LABEL =", REFERENCE_LABEL)


# Compute derived settings at import time to preserve previous behavior
compute_derived_settings()


# Helper to programmatically override top-level constants and recompute derived settings
def apply_overrides(dataset_distribution=None,
                    downstream_dataset=None,
                    defense=None,
                    starting_clean_rounds=None,
                    max_attack_round=None,
                    num_rounds=None):
    """Override top-level settings at runtime.

    Any argument set to None will leave the corresponding global unchanged.
    After applying overrides, compute_derived_settings() is called so all
    dependent globals are updated.
    """
    global DATASET_DISTRIBUTION, DOWNSTREAM_DATASET, DEFENSE, STARTING_CLEAN_ROUNDS, MAX_ATTACK_ROUND, NUM_ROUNDS

    if dataset_distribution is not None:
        DATASET_DISTRIBUTION = dataset_distribution
    if downstream_dataset is not None:
        DOWNSTREAM_DATASET = downstream_dataset
    if defense is not None:
        DEFENSE = int(defense)
    if starting_clean_rounds is not None:
        STARTING_CLEAN_ROUNDS = int(starting_clean_rounds)
    if max_attack_round is not None:
        MAX_ATTACK_ROUND = int(max_attack_round)
    if num_rounds is not None:
        NUM_ROUNDS = int(num_rounds)

    # Recompute any derived settings that depend on these globals
    compute_derived_settings()


def extract_metrics(log_file):
    """Extract final BA and ASR metrics from evaluation log file"""
    try:
        with open(log_file, 'r') as f:
            # Read all lines and get the last two JSON metrics
            lines = [line for line in f if line.startswith('{"metric"')]
            if len(lines) < 2:
                return 0.0, 0.0

            # Parse last two JSON lines
            ba_data = json.loads(lines[-2])  # BA always comes before ASR
            asr_data = json.loads(lines[-1])

            return ba_data['value'], asr_data['value']

    except (IndexError, FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"Error extracting metrics: {e}")
        return 0.0, 0.0

def extract_round_metrics(round_dir, num_clients):
    """Extract average train loss and kNN test accuracy for a round."""
    total_loss = 0.0
    total_accuracy = 0.0

    for client_id in range(num_clients):
        log_file = os.path.join(round_dir, f"pretrain/log_model_ft_c{client_id}.csv")
        if os.path.exists(log_file):
            # Read the CSV file
            df = pd.read_csv(log_file)

            # Extract the last row (final metrics for the client)
            final_row = df.iloc[-1]
            total_loss += final_row['train_loss']
            total_accuracy += final_row['test_acc@1']
        else:
            raise FileNotFoundError(f"Log file not found for client {client_id} in {round_dir}")

    # Compute averages
    avg_loss = total_loss / num_clients
    avg_accuracy = total_accuracy / num_clients


    return avg_loss, avg_accuracy

def evaluate_model(model_path, round_num, output_dir, downstream_epochs, gpu):
    """Evaluate current global model and return metrics"""
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    eval_log = os.path.join(log_dir, f'eval_round_{round_num}.txt')
    #debug
    #print("DEBUG: REFERENCE_PATH =", REFERENCE_PATH, ", REFERENCE_LABEL =", REFERENCE_LABEL)
    # Run evaluation script (parameters are hardcoded for cifar10 pretrain and stl10 downstream)
    cmd = f"""{sys.executable} training_downstream_classifier.py \
        --dataset {DOWNSTREAM_DATASET} \
        --encoder {model_path} \
        --encoder_usage_info {PRETRAIN_DATASET} \
        --reference_label {REFERENCE_LABEL} \
        --trigger_file {TRIGGER_PATH} \
        --reference_file {REFERENCE_PATH} \
        --data_dir {DATA_DIR} \
        --gpu {gpu} \
        --nn_epochs {downstream_epochs} \
        > {eval_log}"""

    subprocess.run(cmd, shell=True, check=True)

    return extract_metrics(eval_log)

def plot_metrics(ba_values, asr_values, output_dir):
    """Plot BA and ASR metrics over rounds"""
    if not ba_values:  # Skip if no data
        return

    plt.figure(figsize=(10, 6))
    x_values = list(range(len(ba_values)))  # Use actual data points for x-axis

    plt.plot(x_values, ba_values, 'b-o', label='Clean Accuracy (CA)')
    plt.plot(x_values, asr_values, 'r-o', label='Attack Success Rate (ASR)')
    plt.xlabel('Round')
    plt.ylabel('Value')
    plt.title('Periodic attack FCL Experiment Metrics')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 100)

    plot_path = os.path.join(output_dir, 'metrics_plot.png')
    plt.savefig(plot_path)
    plt.close()

def plot_train_loss_and_knn_accuracy(train_loss_values, knn_accuracy_values, output_dir):
    """Plot average train loss and kNN accuracy in two side-by-side graphs."""
    if not train_loss_values or not knn_accuracy_values:  # Skip if no data
        return

    plt.figure(figsize=(12, 6))  # Adjust the figure size for side-by-side plots

    # Plot train loss on the left
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
    x_values = list(range(len(train_loss_values)))
    plt.plot(x_values, train_loss_values, 'g-o', label='Avg. Train Loss')
    plt.xlabel('Round')
    plt.ylabel('Train Loss')
    plt.title('Average Train Loss Over Rounds')
    plt.legend()
    plt.grid(True)

    # Plot kNN accuracy on the right
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
    plt.plot(x_values, knn_accuracy_values, 'm-o', label='Avg. kNN Accuracy')
    plt.xlabel('Round')
    plt.ylabel('kNN Accuracy')
    plt.title('Average kNN Accuracy Over Rounds')
    plt.legend()
    plt.grid(True)

    # Save the combined figure
    plot_path = os.path.join(output_dir, 'train_loss_knn_accuracy_plot.png')
    plt.savefig(plot_path)
    plt.close()

# =============================================================================
# EVALUATION MANAGER
# =============================================================================
# This class handles model evaluation in a separate thread to allow training
# and evaluation to run in parallel (on different GPUs if available).
# 
# Architecture:
#   - A queue holds pending evaluation tasks (model_path, round_num, etc.)
#   - A worker thread continuously processes tasks from the queue
#   - Results are stored in a thread-safe dictionary
#   - Plots are updated incrementally as evaluations complete
#
# This parallelization significantly speeds up experiments since evaluation
# (downstream classifier training) can take several minutes per round.
# =============================================================================
class EvaluationManager:
    """Manages parallel evaluation of models"""

    def __init__(self, base_output_dir, log_file):
        self.evaluation_queue = queue.Queue()
        self.log_lock = Lock()
        self.results_dict = {}  # round_num -> (ba, asr, train_loss, knn_accuracy)
        self.base_output_dir = base_output_dir
        self.log_file = log_file
        self.worker_thread = None
        self.shutdown_flag = False

    def start_worker(self):
        """Start the evaluation worker thread"""
        self.worker_thread = threading.Thread(target=self._evaluation_worker, daemon=True)
        self.worker_thread.start()
        print("Evaluation worker thread started")

    def _evaluation_worker(self):
        """
        Worker thread that processes evaluation tasks from the queue.
        
        Runs continuously until shutdown_flag is set or a None task (poison pill)
        is received. Each task contains: model_path, round_num, downstream_epochs,
        is_poison_round, train_loss, knn_accuracy.
        """
        while not self.shutdown_flag:
            task_retrieved = False
            try:
                # Get task from queue (blocks until available)
                task = self.evaluation_queue.get(timeout=1)
                task_retrieved = True
                if task is None:  # Poison pill to stop worker
                    self.evaluation_queue.task_done()
                    break

                model_path, round_num, downstream_epochs, is_poison_round, train_loss, knn_accuracy = task

                print(f"[EVAL] Starting evaluation for round {round_num}...")
                start_time = time.time()

                # Run evaluation
                ba, asr = evaluate_model(model_path, round_num, self.base_output_dir, downstream_epochs, gpu=EVAL_GPU_ID)

                eval_time = time.time() - start_time
                print(f"[EVAL] Completed evaluation for round {round_num} in {eval_time:.1f}s - BA: {ba:.2f}, ASR: {asr:.2f}")

                # Thread-safe storage and logging
                with self.log_lock:
                    self.results_dict[round_num] = (ba, asr, train_loss, knn_accuracy, is_poison_round)

                    # Log all metrics together
                    round_type = "POISON" if is_poison_round else "CLEAN"
                    with open(self.log_file, 'a') as f:
                        f.write(f"Round {round_num} ({round_type}) - BA: {ba:.2f}, ASR: {asr:.2f}, Train Loss: {train_loss:.4f}, kNN Accuracy: {knn_accuracy:.2f}\n")

                # Mark task as done
                self.evaluation_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[EVAL] Error in evaluation worker: {e}")
                if task_retrieved:
                    self.evaluation_queue.task_done()

    def submit_evaluation(self, model_path, round_num, downstream_epochs, is_poison_round, train_loss, knn_accuracy):
        """Submit a model for evaluation with training metrics"""
        self.evaluation_queue.put((model_path, round_num, downstream_epochs, is_poison_round, train_loss, knn_accuracy))
        print(f"[EVAL] Submitted round {round_num} for evaluation (queue size: {self.evaluation_queue.qsize()})")
        print(f"Round {round_num} - Train Loss: {train_loss:.4f}, kNN Accuracy: {knn_accuracy:.2f} [TRAINING COMPLETE, EVAL PENDING]")

    def get_completed_results(self):
        """Get all completed evaluation results"""
        with self.log_lock:
            return dict(self.results_dict)

    def update_plots(self):
        """Update plots with all completed evaluation results"""
        with self.log_lock:
            if not self.results_dict:
                return

            # Sort results by round number
            sorted_results = sorted(self.results_dict.items())
            ba_values = [r[1][0] for r in sorted_results]
            asr_values = [r[1][1] for r in sorted_results]

            plot_metrics(ba_values, asr_values, self.base_output_dir)

    def shutdown(self):
        """Shutdown the evaluation worker"""
        self.shutdown_flag = True
        self.evaluation_queue.put(None)  # Poison pill
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=10)
            print("Evaluation worker thread stopped")

def main(dataset_distribution=None,
         downstream_dataset=None,
         defense=None,
         starting_clean_rounds=None,
         max_attack_round=None,
         num_rounds=None):
    """Main entrypoint for running the federated experiment.

    All arguments are optional. If provided, they override the corresponding
    top-level constants for this run and derived settings are recomputed.

    Example programmatic call from another script:
        from scripts.run_federated import main
        main(dataset_distribution='dirichlet', downstream_dataset='cifar10', defense=0, starting_clean_rounds=100, max_attack_round=300, num_rounds=200)
    """
    # Apply overrides (if any) so globals and derived settings are correct
    if any(v is not None for v in [dataset_distribution, downstream_dataset, defense, starting_clean_rounds, max_attack_round, num_rounds]):
        apply_overrides(dataset_distribution, downstream_dataset, defense, starting_clean_rounds, max_attack_round, num_rounds)

    # Evaluation only mode, skip traning and evaluate existing checkpoints
    if EVAL_ONLY:
        if not MODELS_DIR:
            raise ValueError("MODELS_DIR is required in eval-only mode")

        # Set up evaluation manager
        eval_manager = EvaluationManager(OUTPUT_DIR, os.path.join(OUTPUT_DIR, "metrics.log"))
        eval_manager.start_worker()

        try:
            checkpoints = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith(".pth")],
                                 key=lambda x: int(x.split("round")[1].split(".")[0]))
            print(f"Found {len(checkpoints)} checkpoints for evaluation")

            # Evaluate each checkpoint
            for ckpt in checkpoints:
                round_num = int(ckpt.split("round")[1].split(".")[0])
                model_path = os.path.join(MODELS_DIR, ckpt)

                # Only evaluate if we are not skipping this round
                should_evaluate = (SKIP_ROUNDS == -1) or ((round_num + 1) % SKIP_ROUNDS == 0)
                if not should_evaluate:
                    print(f"[EVAL] Skipping evaluation for round {round_num}")
                    continue

                # Submit for evaluation (train_loss and knn_accuracy are set to 0.0 as they are unknown in eval-only mode)
                is_poison_round = (round_num + 1) % BAD_ROUNDS == 0 if BAD_ROUNDS != -1 else False
                # Submit model for parallel evaluation with training metrics (non-blocking)
                if DOWNSTREAM_EPOCHS == 'progressive':
                    downstream_epochs = min(HARDCAP, int(np.ceil(((round_num + 1) * 5) / 2)))
                else:
                    downstream_epochs = DOWNSTREAM_EPOCHS
                eval_manager.submit_evaluation(model_path,
                                               round_num,
                                               downstream_epochs,
                                               is_poison_round,
                                               0.0,
                                               0.0)

            # Wait for all evaluations to complete
            print("Waiting for all evaluations to complete... (this could take a while!)")
            eval_manager.evaluation_queue.join()
            eval_manager.update_plots()

        finally:
            eval_manager.shutdown()

            # Print final summary
            completed_results = eval_manager.get_completed_results()
            print(f"\nExperiment completed! {len(completed_results)} evaluations finished.")
            for round_num in sorted(completed_results.keys()):
                ba, asr, train_loss, knn_accuracy, is_poison = completed_results[round_num]
                round_type = "POISON" if is_poison else "CLEAN"
                print(f"Round {round_num} ({round_type}): BA={ba:.2f}, ASR={asr:.2f}")

    # Standard mode with training and evaluation
    # =============================================================================
    # MAIN FEDERATED LEARNING LOOP
    # =============================================================================
    else:

        # Experiment parameters
        num_rounds = NUM_ROUNDS
        bad_round = BAD_ROUNDS
        experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = OUTPUT_DIR
        os.makedirs(base_output_dir, exist_ok=True)

        # Metrics tracking
        train_loss_values = []
        test_accuracy_values = []
        rounds = list(range(1, num_rounds + 1))

        # Log file setup
        log_file = os.path.join(base_output_dir, "metrics.log")

        # Initialize evaluation manager
        eval_manager = EvaluationManager(base_output_dir, log_file)
        eval_manager.start_worker()

        # Initial model path, put 'fs' for "from-scratch" training
        #current_model = "./output/cifar10/clean_encoder/model_100.pth"
        current_model = 'fs'

        # Misc. args for defense mechanism and training
        # For BAGEL, set args.bagel = True and naive = 0
        # For "naive" Bagdasarian + BadEncoder, set args.bagel = True and naive = 1
        args = argparse.Namespace(
            defense='clipnoise' if DEFENSE == 1 else 'none',  # 'clipnoise' or 'none' for no defense
            trusted_update_path='',
            num_malicious=BAD_CLIENTS,
            num_benign=NUM_CLIENTS - BAD_CLIENTS,
            global_model_path='',
            #previous_global_model='', # Path to previous global model (for neurotoxin)
            learning_rate=FEDAVG_LEARNING_RATE, # Learing rate for fedavg
            gpu = TRAINING_GPU_ID,
            bagel = True if ATTACK in [2,3] else False,
            naive = 1 if ATTACK == 3 else 0,
            current_round=0,
            pretrain_dataset=PRETRAIN_DATASET,
            shadow_dataset=SHADOW_DATASET
        )




        # Define checkpoint rounds (rounds for which we save all intermediate updates)
        checkpoint_rounds = []
        temp_round_dir = os.path.join(base_output_dir, "temp_round")
        checkpoints_dir = os.path.join(base_output_dir, "checkpoints")
        models_dir = os.path.join(base_output_dir, "models")
        os.makedirs(temp_round_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

        try:
            for round_num in range(RESUME_ROUND,num_rounds):
                round_dir = os.path.join(checkpoints_dir, f"round_{round_num}") if round_num in checkpoint_rounds else temp_round_dir
                os.makedirs(round_dir, exist_ok=True)

                # Determine if this is a poison round (removed for baseline)
                if BAD_ROUNDS == -1:
                    is_poison_round = False
                else:
                    is_poison_round = (round_num + 1) % bad_round  == 0 and round_num > STARTING_CLEAN_ROUNDS and round_num < MAX_ATTACK_ROUND
                #is_poison_round = False

                if round_num == 0:
                    args.global_model_path = 'fs'

                    # Checkpoint loading
                if CHECKPOINT:
                    args.global_model_path = CHECKPOINT



                if round_num > 0:  # Skip first round since there's no previous model
                    prev_model_path = os.path.join(models_dir, f"model_round{round_num-1}.pth")
                    if os.path.exists(prev_model_path):
                        args.global_model_path = prev_model_path
                    else:
                        print(f"Warning: Previous model not found at {prev_model_path}")


                # For neurotoxin, we need to set the old previous global model path
                #if is_poison_round:
                #    args.previous_global_model = os.path.join(models_dir, f"model_round{round_num-2}.pth")


                # -----------------------------------------------------------------
                # POISON ROUND: One client executes the backdoor attack.
                # The attack injects a trigger pattern that causes misclassification
                # to the target class in downstream tasks.
                # -----------------------------------------------------------------
                if is_poison_round:
                    args.current_round = round_num
                    print(f"Running POISON round {args.current_round}")
                    current_model = federated_poison_round(
                        pretraining_dataset=PRETRAIN_DATASET,
                        dataset_paths=[f"{DATA_DIR}/{PRETRAIN_DATASET}/partitions/{DATASET_DISTRIBUTION}/partition_{i}.npz" for i in range(10)],
                        test_dir=f"{DATA_DIR}/{PRETRAIN_DATASET}/test.npz",
                        mem_dir=f"{DATA_DIR}/{PRETRAIN_DATASET}/train.npz",
                        pretrain_epochs=CLIENT_EPOCHS, #1 for testing, 5 default
                        backdoor_epochs=BACKDOOR_EPOCHS, #1 for testing, 10 default (2 for badavg)
                        output_dir=round_dir,
                        trigger_path=TRIGGER_PATH,
                        #reference_path=f"./reference/{PRETRAIN_DATASET}/truck.npz",
                        reference_path=REFERENCE_PATH,
                        args=args,
                    )
                # -----------------------------------------------------------------
                # CLEAN ROUND: All clients perform standard SimCLR pretraining
                # without any attack. This is the normal federated learning flow.
                # -----------------------------------------------------------------
                else:
                    # For LR scheduler:
                    args.current_round = round_num
                    print(f"Running clean round {args.current_round}")
                    current_model = federated_round(
                        pretraining_dataset=PRETRAIN_DATASET,
                        dataset_paths=[f"{DATA_DIR}/{PRETRAIN_DATASET}/partitions/{DATASET_DISTRIBUTION}/partition_{i}.npz" for i in range(10)],
                        test_dir=f"{DATA_DIR}/{PRETRAIN_DATASET}/test.npz",
                        mem_dir=f"{DATA_DIR}/{PRETRAIN_DATASET}/train.npz",
                        pretrain_epochs=CLIENT_EPOCHS, #1 for testing, 5 default 
                        output_dir=round_dir,
                        args=args
                    )


                # Evaluate current model (warning: some arguments are hardcoded)
                # We are adjusting the number of downstream training epochs depending on the pre-trained epochs.
                # For example, if cumulative pre-trained epochs are 1000, we will train the downstream classifier for 500 epochs.
                # Extract immediate training metrics
                avg_loss, avg_accuracy = extract_round_metrics(round_dir, 10)
                train_loss_values.append(avg_loss)
                test_accuracy_values.append(avg_accuracy)

                # Move the model to the models directory
                temp_model_path = os.path.join(round_dir, "aggregated_model.pth")
                stable_model_path = os.path.join(models_dir, f"model_round{round_num}.pth")
                if os.path.exists(temp_model_path):
                    os.rename(temp_model_path, stable_model_path)
                    print(f"Model for round {round_num} saved to {stable_model_path}")
                else:
                    raise FileNotFoundError(f"Model file not found at {temp_model_path}")

                # Submit model for parallel evaluation with training metrics (non-blocking)
                # Progressive downstream epochs: train the downstream classifier
                # for longer as the encoder improves. Formula: epochs = (round * 5) / 2
                # This balances evaluation accuracy vs computation time.
                if DOWNSTREAM_EPOCHS == 'progressive':
                    downstream_epochs = min(HARDCAP, int(np.ceil(((round_num + 1) * 5) / 2)))
                else:
                    downstream_epochs = DOWNSTREAM_EPOCHS

                    # Onlu submit for eval if we are not skipping this round
                # Always evaluate rounds within the attack window for better monitoring
                in_attack_window = round_num > STARTING_CLEAN_ROUNDS and round_num < MAX_ATTACK_ROUND
                should_evaluate = (SKIP_ROUNDS == -1) or ((round_num + 1) % SKIP_ROUNDS == 0) or in_attack_window
                if should_evaluate:
                    eval_manager.submit_evaluation(stable_model_path, round_num, downstream_epochs, is_poison_round, avg_loss, avg_accuracy)
                else:
                    print(f"[EVAL] Skipping evaluation for round {round_num}")
                    # Still log training metrics even if skipping evaluation
                    with eval_manager.log_lock:
                        eval_manager.results_dict[round_num] = (0.0, 0.0, avg_loss, avg_accuracy, is_poison_round)

                        # Update plots with completed evaluation results (if any)
                eval_manager.update_plots()

                # Update training metrics plot
                plot_train_loss_and_knn_accuracy(train_loss_values, test_accuracy_values, base_output_dir)

                # Force garbage collection and CUDA cache cleanup after each round
                import gc
                gc.collect()
                torch.cuda.empty_cache()

        finally:
            # Wait for remaining evaluations to complete
            print("Waiting for remaining evaluations to complete...")
            eval_manager.evaluation_queue.join()  # Wait for all tasks to be processed

            # Final plot update
            eval_manager.update_plots()

            # Shutdown evaluation worker
            eval_manager.shutdown()

            # Print final summary
            completed_results = eval_manager.get_completed_results()
            print(f"\nExperiment completed! {len(completed_results)} evaluations finished.")
            for round_num in sorted(completed_results.keys()):
                ba, asr, train_loss, knn_accuracy, is_poison = completed_results[round_num]
                round_type = "POISON" if is_poison else "CLEAN"
                print(f"Round {round_num} ({round_type}): BA={ba:.2f}, ASR={asr:.2f}, Train Loss={train_loss:.4f}, kNN Acc={knn_accuracy:.2f}")


if __name__ == "__main__":
    #for dataset_distribution in ["iid", "dirichlet"]: #if using stl-10 as pretrain, dirichlet partitions are not available (unlabeled data)
        for downstream_dataset in ["gtsrb", "svhn", "cifar10"]:
            for defense in [0, 1]:
                starting_clean_rounds = 200
                max_attack_round = 400
                num_rounds = 600
                print(f"\n=== Running experiment with settings: dataset_distribution=iid, downstream_dataset={downstream_dataset}, defense={defense}, starting_clean_rounds={starting_clean_rounds}, max_attack_round={max_attack_round}, num_rounds={num_rounds} ===\n")
                main(
                    dataset_distribution="iid",
                    downstream_dataset=downstream_dataset,
                    defense=defense,
                    starting_clean_rounds=starting_clean_rounds,
                    max_attack_round=max_attack_round,
                    num_rounds=num_rounds
                )
