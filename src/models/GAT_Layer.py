"""Graph Attention Network (GAT) for pathway interaction analysis.

Implements multi-layer GAT networks with attention mechanisms
for processing pathway features through graph convolutions.
"""

import pickle
from pathlib import Path
from typing import Tuple, Optional, List, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


def load_graph_info(filepath: Union[str, Path]) -> Tuple[torch.Tensor, int, List[str]]:
    """Load pathway interaction graph data from pickle file.

    Args:
        filepath: Path to pickle file containing graph data.

    Returns:
        Tuple of (edge_index, n_pathways, pathway_ids).
            - edge_index: Shape [2, num_edges].
            - n_pathways: Number of pathway nodes.
            - pathway_ids: List of pathway IDs.
    """
    file_path = Path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'rb') as f:
        graph_data = pickle.load(f)

    edge_index = graph_data['edge_index']
    n_pathways = graph_data['num_nodes']
    pathway_ids = graph_data['pathway_kegg_ids_list']

    return edge_index, n_pathways, pathway_ids


class GATNetwork(nn.Module):
    """Multi-layer Graph Attention Network for pathway interaction analysis.

    Two-layer architecture:
        1. First layer: Multi-head attention with concatenation
        2. Final layer: Multi-head attention with averaging

    Args:
        in_features: Dimension of input node features.
        hidden_dim1: Hidden dimension for first layer (total across all heads).
        out_features: Dimension of output embeddings.
        heads_l1: Number of attention heads in first layer.
        heads_l2: Number of attention heads in final layer.
        dropout: Dropout probability.
        activation: Activation function (default: ELU).
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim1: int,
        out_features: int,
        heads_l1: int = 8,
        heads_l2: int = 1,
        dropout: float = 0.3,
        activation: Callable = F.elu
    ) -> None:
        super().__init__()

        self.dropout_rate = dropout
        self.activation = activation

        # First layer: Multi-head attention with concatenation
        # Each head outputs hidden_dim1 // heads_l1 features
        self.conv1 = GATConv(
            in_channels=in_features,
            out_channels=hidden_dim1 // heads_l1,
            heads=heads_l1,
            dropout=dropout,
            concat=True
        )

        # Final layer: Attention with averaging
        self.conv_last = GATConv(
            in_channels=hidden_dim1,
            out_channels=out_features,
            heads=heads_l2,
            dropout=dropout,
            concat=False
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through the GAT network.

        Args:
            x: Node feature tensor. Shape: [num_nodes, in_features].
            edge_index: Edge connectivity. Shape: [2, num_edges].

        Returns:
            Tuple of (node_embeddings, attention_weights).
                - node_embeddings: Shape [num_nodes, out_features].
                - attention_weights: Tuple of (edge_index, attention_values) or None.
        """
        # First layer: Multi-head attention with concatenation
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Final layer: Attention with averaging
        x, attention_weights = self.conv_last(x, edge_index, return_attention_weights=True)

        return x, attention_weights


def create_default_gat_model(
    pafe_feature_dim: int = 24822,
    hidden_dim: int = 512,
    embedding_dim: int = 128
) -> GATNetwork:
    """Create GAT model with default configuration.

    Args:
        pafe_feature_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        embedding_dim: Output embedding dimension.

    Returns:
        Initialized GATNetwork.
    """
    return GATNetwork(
        in_features=pafe_feature_dim,
        hidden_dim1=hidden_dim,
        out_features=embedding_dim,
        heads_l1=8,
        heads_l2=1,
        dropout=0.3,
        activation=F.elu
    )
