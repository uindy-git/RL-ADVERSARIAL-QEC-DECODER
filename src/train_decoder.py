from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import os

from src.models import GAT_GNN
from src.utils import set_seed, seed_worker

def train_gat_decoder(node_info, dataset, directories, device, seed=42):
    """
    Train the GAT decoder model on the dataset created from shots.
    """
    print("--- Start Training GAT Decoder ---")
    MODEL_DIR = directories.get("gat_model", "model.pth")
    num_rounds = node_info["num_rounds"]

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    g = torch.Generator()
    g.manual_seed(seed)  # Ensure reproducibility in data splitting
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=g)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=64, worker_init_fn=seed_worker, generator=g)

    #--- 1. Check dataset label distribution and create pos_weight ---
    all_labels = torch.cat([data.y for data in dataset])
    num_positives = all_labels.sum()
    num_negatives = len(all_labels) - num_positives
    print(f"The dataset label distribution: Positive={num_positives}, Negative={num_negatives}")
    pos_weight = num_negatives / num_positives
    print(f"Loss function positive weight: {pos_weight:.4f}")

    model = GAT_GNN(
        in_channels=num_rounds,
        hidden_channels=64,
        edge_feature_size=1,
        num_heads=8
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    train_losses = []
    test_accuracies = []

    print(f"Training GAT decoder model with {len(train_dataset)} training samples and {len(test_dataset)} test samples.")
    print("Starting training...")
    for epoch in range(50):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            optimizer.zero_grad()
            graph_pred_logits = model(data)
            loss = loss_fn(graph_pred_logits, data.y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                pred_logits = model(data)
                pred = (pred_logits > 0.0).float()
                correct += (pred == data.y).sum().item()
                total += data.num_graphs
        accuracy = correct / total
        test_accuracies.append(accuracy)
        if epoch % 10 == 0 or epoch == 49:
            print(f"Epoch {epoch:03d}, Avg BCE Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
    print("--- Completed GAT Training ---")
    torch.save(model.state_dict(), MODEL_DIR)
    print(f"Trained GAT decoder model saved to '{MODEL_DIR}'.")

    # Plot training loss and accuracy
    save_pdf = os.path.join(directories.get("figures", "."), "gat_training.pdf")
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
    plt.close()
    return model