import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.neighbors import kneighbors_graph

def load_data_from_path(data_path):
    """Load data from the specified path."""
    x_train, y_train, x_test, y_test = load_data.load_data(data_path, info=False)
    return x_train, y_train, x_test, y_test

def create_graph_data(x, y, k=10):
    """Convert data into graph data using k-nearest neighbors."""
    adj = kneighbors_graph(x, n_neighbors=k, mode='connectivity', include_self=False)
    edge_index, _ = from_scipy_sparse_matrix(adj)

    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y.values, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)