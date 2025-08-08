import os
import torch
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import pandas as pd

from models import RL_GAT_Actor


def train_rl_agent(gat_model, node_info, dataset, directories):
    """
    Train the RL agent to generate a vulnerability map for the GAT decoder model.
    """
    print("--- Start Training RL Agent ---")
    
    RL_MODEL_PATH = directories.get("rl_model", "rl_model.pth")

    # --- 1. Load dataset ---
    num_rounds = node_info["num_rounds"]
    num_spatial_nodes = node_info["num_spatial_nodes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Split dataset into negative and positive samples
    negative_samples = [data for data in dataset if data.y.item() == 0]
    positive_samples = [data for data in dataset if data.y.item() == 1]
    print(f"Total samples: {len(dataset)}")
    print(f"Negative samples (no error): {len(negative_samples)}")
    print(f"Positive samples (with error): {len(positive_samples)}")

    # --- 2. Configure GAT_Decoder ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gat_model.to(device)
    gat_model.eval()

    # --- 3. Load RL agent and Start Training ---
    print("\n--- 3. Load RL agent and start training ---")
    train_size = int(0.8 * len(negative_samples))
    test_size = len(negative_samples) - train_size
    train_neg_samples, test_neg_samples = random_split(negative_samples, [train_size, test_size])
    print(f"Number of training samples: {len(train_neg_samples)}, Number of evaluation samples: {len(test_neg_samples)}")

    rl_model = RL_GAT_Actor(
        in_channels=num_rounds,
        hidden_channels=64,
        num_spatial_nodes=num_spatial_nodes,
        num_heads=8,
        edge_feature_size=1
    ).to(device)

    optimizer = torch.optim.Adam(rl_model.parameters(), lr=0.0001)

    num_episodes = 4000
    max_steps_per_episode = 5 # Limit the number of steps per episode
    gamma = 0.99
    
    episode_rewards = []
    episode_final_probs = []
    episode_steps = []

    for ep in tqdm(range(num_episodes), desc="RL Training"):
        rl_model.train()
        
        initial_data = random.choice(train_neg_samples).to(device)
        current_data = initial_data.clone()

        log_action_probs = []
        rewards = []
        
        with torch.no_grad():
            last_prob = torch.sigmoid(gat_model(current_data))

        for t in range(max_steps_per_episode):
            action_probs = rl_model(current_data)
            action_dist = torch.distributions.Categorical(probs=action_probs)
            action = action_dist.sample()
            log_action_probs.append(action_dist.log_prob(action))

            node_index = (action // num_rounds).item()
            time_index = (action % num_rounds).item()
            
            next_data = current_data.clone()
            # Do not flip if the current value is already 1 (To achieve a more efficient attack)
            if next_data.x[node_index, time_index] == 1:
                # Give a penalty for unnecessary actions
                reward = -0.1
            else:
                next_data.x[node_index, time_index] = 1.0
                with torch.no_grad():
                    new_prob = torch.sigmoid(gat_model(next_data))
                reward = (new_prob - last_prob).item()
            
            rewards.append(reward)
            
            current_data = next_data
            last_prob = torch.sigmoid(gat_model(current_data))

            if last_prob > 0.5:
                break

        # --- Optimize Weights using REINFORCE ---
        returns = []
        discounted_return = 0.0
        for r in reversed(rewards):
            discounted_return = r + gamma * discounted_return
            returns.insert(0, discounted_return)
        
        returns = torch.tensor(returns, device=device, dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        policy_loss = []
        for log_prob, R in zip(log_action_probs, returns):
            policy_loss.append(-log_prob * R)
        
        if policy_loss:
            optimizer.zero_grad()
            loss = torch.cat(policy_loss).sum()
            loss.backward()
            optimizer.step()

        # Log results
        episode_rewards.append(sum(rewards))
        episode_final_probs.append(last_prob.item())
        episode_steps.append(len(rewards))

    print("--- Completed RL Training ---")

    # Save the trained model
    torch.save(rl_model.state_dict(), RL_MODEL_PATH)
    print(f"Trained RL agent saved to '{RL_MODEL_PATH}'.")


    # --- 5. Plot Training Curves ---
    print("\n--- 5. Plot Training Curves ---")
    fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    
    # Define a moving average function
    def moving_average(data, window_size):
        return pd.Series(data).rolling(window=window_size).mean()

    window = 100 # Moving average window size

    axs[0].plot(moving_average(episode_rewards, window))
    axs[0].set_ylabel('Total Reward (Smoothed)')
    axs[0].set_title('Training Progress of RL Agent')
    axs[0].grid(True)

    axs[1].plot(moving_average(episode_final_probs, window), color='orange')
    axs[1].axhline(y=0.5, color='r', linestyle='--', label='Success Threshold')
    axs[1].set_ylabel('Final Probability (Smoothed)')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(moving_average(episode_steps, window), color='green')
    axs[2].set_xlabel('Episode')
    axs[2].set_ylabel('Attack Steps (Smoothed)')
    axs[2].grid(True)
    
    plt.tight_layout()
    plot_filename = os.path.join(directories.get("figures", "."), "rl_training_curves.pdf")
    plt.savefig(plot_filename)
    plt.close()
    print(f"Training curves plot saved to '{plot_filename}'.")
    return test_neg_samples, rl_model