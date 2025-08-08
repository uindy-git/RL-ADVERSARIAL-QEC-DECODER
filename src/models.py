import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch.nn import LayerNorm

# Define the GAT GNN model for graph classification
class GAT_GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_feature_size, num_heads=4):
        super().__init__()
        assert hidden_channels % num_heads == 0

        self.conv1 = GATv2Conv(in_channels, hidden_channels // num_heads, heads=num_heads, edge_dim=edge_feature_size)
        self.norm1 = LayerNorm(hidden_channels)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels // num_heads, heads=num_heads, edge_dim=edge_feature_size)
        self.norm2 = LayerNorm(hidden_channels)
        
        self.pool = global_mean_pool
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1))

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.norm2(x)
        x = F.relu(x)
        
        pooled_x = self.pool(x, batch)
        graph_pred_logits = self.classifier(pooled_x)

        return graph_pred_logits.squeeze()
    

# Define the actor RL agent
class RL_GAT_Actor(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_spatial_nodes, num_heads=4, edge_feature_size=1):
        super().__init__()
        self.num_actions = num_spatial_nodes * in_channels
        self.conv1 = GATv2Conv(in_channels, hidden_channels // num_heads, heads=num_heads, edge_dim=edge_feature_size)
        self.norm1 = LayerNorm(hidden_channels)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, heads=1, edge_dim=edge_feature_size)
        self.norm2 = LayerNorm(hidden_channels)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, in_channels)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.norm1(self.conv1(x, edge_index, edge_attr=edge_attr)))
        x = F.relu(self.norm2(self.conv2(x, edge_index, edge_attr=edge_attr)))
        action_logits_per_node = self.policy_head(x)
        action_logits_flat = action_logits_per_node.view(1, -1)
        action_probs = F.softmax(action_logits_flat, dim=1)
        return action_probs