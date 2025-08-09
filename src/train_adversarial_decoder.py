import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm
from torch.utils.data import random_split
import torch.nn as nn
import matplotlib.pyplot as plt
from copy import deepcopy
import os

from src.utils import seed_worker

def train_adversarial_GAT_decoder(normal_model, rl_model, num_rounds, dataset, directories, device, seed=42):
    """
    Train the Adversarial GAT decoder model on the dataset created from shots.
    """
    print("--- Start Training Adversarial GAT Decoder ---")

    gat_model = deepcopy(normal_model)
    gat_model.to(device)
    rl_model.to(device)
    rl_model.eval()

    # --- 1. Prepare Data Loaders ---
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    g = torch.Generator()
    g.manual_seed(seed)  # Ensure reproducibility in data splitting
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=g)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, worker_init_fn=seed_worker, generator=g) # Fewer samples per batch for adversarial training
    test_loader = DataLoader(test_dataset, batch_size=32, worker_init_fn=seed_worker, generator=g)

    all_labels = torch.cat([data.y for data in dataset])
    pos_weight = (all_labels == 0).sum() / (all_labels == 1).sum()
    
    optimizer = torch.optim.Adam(gat_model.parameters(), lr=1e-4) # Fewer epochs and lower learning rate for adversarial training
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    # --- 4. Adversarial Training Loop ---
    print("\n--- Start Adversarial Training ---")
    num_epochs = 10 # Number of epochs for additional training
    max_attack_steps = 2 # Maximum number of attack steps
    alpha = 0.5 # Weight for adversarial loss
    train_losses = []
    test_accuracies = []
    for epoch in range(num_epochs):
        gat_model.train() # Set decoder to training mode
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = batch.to(device)

            # --- Step 1: Compute clean loss ---
            optimizer.zero_grad()
            clean_outputs = gat_model(batch)
            loss_clean = loss_fn(clean_outputs, batch.y)

            # --- Step 2: Generate adversarial samples ---
            adversarial_samples = []
            original_labels = []

            # Split the batch into individual graphs
            data_list = batch.to_data_list()
            with torch.no_grad(): # No need for gradient calculation
                for data in data_list:
                    # Only attack negative samples (no error)
                    if data.y.item() == 0:
                        adv_data = data.clone()

                        # RL agent attacks for a few steps
                        for _ in range(max_attack_steps):
                            action_probs = rl_model(adv_data)
                            action = torch.argmax(action_probs, dim=1)
                            
                            node_index = (action // num_rounds).item()
                            time_index = (action % num_rounds).item()

                            # Bit flip
                            adv_data.x[node_index, time_index] = 1.0 - adv_data.x[node_index, time_index]

                            # Stop the attack if the goal (misclassification) is achieved
                            if torch.sigmoid(gat_model(adv_data)) > 0.5:
                                break

                        # Save the adversarial sample and its original label (=0)
                        adversarial_samples.append(adv_data)
                        original_labels.append(0.0)

            # --- Step 3: Compute adversarial loss ---
            loss_adv = torch.tensor(0.0, device=device)
            if adversarial_samples:
                # Create a new batch from the generated adversarial samples
                adversarial_batch = Batch.from_data_list(adversarial_samples).to(device)
                adversarial_batch.y = torch.tensor(original_labels, device=device, dtype=torch.float)
                
                adv_outputs = gat_model(adversarial_batch)
                loss_adv = loss_fn(adv_outputs, adversarial_batch.y)

            # --- Step 4: Combine losses and update weights ---
            total_loss_batch = loss_clean + alpha * loss_adv
            total_loss_batch.backward()
            optimizer.step()
            total_loss += total_loss_batch.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # --- Step 5: Evaluate performance ---
        gat_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                pred_logits = gat_model(data)
                pred = (pred_logits > 0.0).float()
                correct += (pred == data.y).sum().item()
                total += data.num_graphs
        accuracy = correct / total
        test_accuracies.append(accuracy)
        print(f"Epoch {epoch+1:02d}, Avg Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
    
    # --- 5. Save the trained model ---
    robust_gat_model_path = directories.get("robust_gat_model", "robust_gat_model.pth")
    torch.save(gat_model.state_dict(), robust_gat_model_path)
    print(f"Robust GAT decoder model saved to '{robust_gat_model_path}'.")
    print("--- Completed Adversarial GAT Training ---")

    # --- 6. Plot training loss and test accuracy ---
    save_pdf = os.path.join(directories.get("figures", "."), "robust_gat_training.pdf")
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(save_pdf)
    return gat_model
