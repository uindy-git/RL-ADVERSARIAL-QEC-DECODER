import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import random

def set_seed_for_reproducibility(seed):
    """
    Set random seed for reproducibility across PyTorch, NumPy, and random.
    This function ensures that the results are consistent across runs by fixing the random seed.
    """
    # --- Basic seed fixing ---
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # --- CUDA-related seed fixing ---
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # --- Disable non-deterministic algorithms for CUDA ---
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # This is to ensure that all operations are deterministic.
    # This may slow down some operations
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        # This function may not be available in older PyTorch versions
        print("Warning: torch.use_deterministic_algorithms is not available. Reproducibility may not be guaranteed.")

    # This environment variable may be necessary to ensure the determinism of certain CUDA convolution operations
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_flattened_graph_from_shot(node_features_shot, logical_flip_shot, static_edge_index, static_edge_weights):
    node_features = torch.tensor(node_features_shot, dtype=torch.float)
    graph_label = torch.tensor(logical_flip_shot, dtype=torch.float)
    return Data(x=node_features, edge_index=static_edge_index, edge_attr=static_edge_weights, y=graph_label)

def cultivate_edge_weights(edge_index):
    num_edges = edge_index.size(1)
    weights = torch.rand(num_edges, 1, dtype=torch.float)
    return weights

def evaluate_agent(rl_model, target_model, test_samples, device, max_steps=10):
    """ Evaluate the RL agent's performance on the test samples."""
    print("\n--- Start Evaluation ---")
    target_model.to(device)
    rl_model.to(device)
    rl_model.eval()
    target_model.eval()

    successful_attacks = 0
    total_flips_for_success = 0
    num_test_samples = len(test_samples)

    for i in tqdm(range(num_test_samples), desc="Evaluating"):
        initial_data = test_samples[i].to(device)
        current_data = initial_data.clone()
        
        num_flips = 0
        attack_succeeded = False
        
        with torch.no_grad():
            initial_prob = torch.sigmoid(target_model(current_data)).item()
            # If the initial probability is already high, skip this sample
            if initial_prob > 0.5:
                continue

            last_prob = initial_prob
            for t in range(max_steps):
                num_flips += 1
                action_probs = rl_model(current_data)
                # Select the action with the highest probability
                action = torch.argmax(action_probs, dim=1)

                node_index = (action // current_data.x.size(1)).item()
                time_index = (action % current_data.x.size(1)).item()

                next_data = current_data.clone()
                next_data.x[node_index, time_index] = 1 - next_data.x[node_index, time_index]
                
                new_prob = torch.sigmoid(target_model(next_data)).item()

                if new_prob > 0.5:
                    successful_attacks += 1
                    total_flips_for_success += num_flips
                    attack_succeeded = True
                    break
                
                current_data = next_data

    attack_success_rate = successful_attacks / num_test_samples if num_test_samples > 0 else 0
    avg_flips = total_flips_for_success / successful_attacks if successful_attacks > 0 else float('inf')

    print("--- Evaluation Results ---")
    print(f"Number of Evaluation Samples: {num_test_samples}")
    print(f"Number of Successful Attacks: {successful_attacks}")
    print(f"Attack Success Rate (ASR): {attack_success_rate:.4f}")
    print(f"Average Flips for Successful Attacks: {avg_flips:.2f}")
    print("------------------")

def generate_vulnerability_map(rl_model, target_model, node_info, dataset, directories, filename, device, seed=42):
    """
    Use RL Agent to generate a vulnerability map for the GAT decoder model.
    """
    print("--- Start Generating Vulnerability Map ---")
    HEATMAP_FILENAME = os.path.join(directories.get("figures", "."), filename)

    num_rounds = node_info["num_rounds"]
    num_spatial_nodes = node_info["num_spatial_nodes"]
    dataset_on_cpu = [data.to("cpu") for data in dataset]
    negative_samples = [data for data in dataset_on_cpu if data.y.item() == 0]

    target_model.to(device)
    rl_model.to(device)
    target_model.eval()
    rl_model.eval()

    # --- 1. Execute attacks and aggregate flip positions ---
    print("\n--- 1. Attack simulation and vulnerability location aggregation ---")
    test_samples = []
    with torch.no_grad():
        g = torch.Generator()
        g.manual_seed(seed)  # Ensure reproducibility in data splitting
        loader = DataLoader(negative_samples, batch_size=512, shuffle=False, worker_init_fn=seed_worker, generator=g)
        for batch_data in tqdm(loader, desc="Filtering test samples"):
            # Split the batch into individual graphs
            batch_data = batch_data.to(device)
            data_list = batch_data.to_data_list()
            # Stack into a batch again
            batch = Batch.from_data_list(data_list)
            output = target_model(batch)
            preds = (torch.sigmoid(output) <= 0.5) # If the output is less than or equal to 0.5, it is considered a negative sample

            # Add only the correctly classified ones (move back to cpu for later use)
            for i in range(len(preds)):
                if preds[i].item():
                    test_samples.append(data_list[i].cpu())

    print(f"Number of attack target samples: {len(test_samples)}")

    vulnerability_map = np.zeros((num_spatial_nodes, num_rounds))
    max_steps_per_episode = 5

    for data_sample in tqdm(test_samples, desc="Analyzing Attacks"):
        current_data = data_sample.clone().to(device)
        
        with torch.no_grad():
            for t in range(max_steps_per_episode):
                action_probs = rl_model(current_data)
                action = torch.argmax(action_probs, dim=1)

                node_index = (action // num_rounds).item()
                time_index = (action % num_rounds).item()
                
                next_data = current_data.clone()
                current_flip_value = next_data.x[node_index, time_index].item()
                next_data.x[node_index, time_index] = 1.0 - current_flip_value
                
                new_prob = torch.sigmoid(target_model(next_data)).item()

                # If the attack was successful, record the flip for that step
                if new_prob > 0.5:
                    vulnerability_map[node_index, time_index] += 1
                    break
                
                current_data = next_data

    # --- 2. Create vulnerability map ---
    print("\n--- 2. Create vulnerability map ---")
    if np.sum(vulnerability_map) == 0:
        print("No attacks were successful. Heatmap will not be created.")
        return

    plt.figure(figsize=(8, 4))
    sns.heatmap(vulnerability_map.T, annot=True, fmt=".0f", cmap="Reds",
                xticklabels=[f"Node {i}" for i in range(num_spatial_nodes)],
                yticklabels=[f"Time {i}" for i in range(num_rounds)])
    plt.title("Vulnerability Heatmap of GAT Decoder\n(Count of Successful Adversarial Flips)")
    plt.xlabel("Spatial Node Index")
    plt.ylabel("Time Step (Measurement Round)")
    plt.tight_layout()
    
    plt.savefig(HEATMAP_FILENAME)
    print(f"Vulnerability map saved to '{HEATMAP_FILENAME}'.")