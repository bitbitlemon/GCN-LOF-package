import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    """Graph Convolutional Network (GCN) model for anomaly detection."""
    def __init__(self, num_features, num_classes, hidden_dim=16):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)  # First GCN layer
        self.conv2 = GCNConv(hidden_dim, num_classes)   # Second GCN layer

    def forward(self, data):
        """Forward pass of the GCN model."""
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)  # Log-softmax for classification