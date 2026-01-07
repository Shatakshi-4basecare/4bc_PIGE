"""Drug embedding neural network.

Converts drug fingerprints into embeddings using a multi-layer ANN.
Used with precomputed NPVAE drug representations.
"""

from typing import Dict

import torch
import torch.nn as nn


class DrugEmbedderANN(nn.Module):
    """Neural network for generating drug embeddings from fingerprints.

    Architecture:
        Input -> Linear -> ReLU -> Dropout ->
        Linear -> ReLU -> Dropout ->
        Linear -> Output

    Args:
        fingerprint_dim: Dimension of input fingerprint.
        hidden_dim1: First hidden layer dimension.
        hidden_dim2: Second hidden layer dimension.
        embedding_dim: Output embedding dimension.
        dropout_rate: Dropout probability.
        use_batchnorm: Whether to use batch normalization.
    """

    def __init__(
        self,
        fingerprint_dim: int,
        hidden_dim1: int = 512,
        hidden_dim2: int = 256,
        embedding_dim: int = 128,
        dropout_rate: float = 0.2,
        use_batchnorm: bool = False
    ) -> None:
        super().__init__()

        self.fingerprint_dim = fingerprint_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm

        layers = []

        layers.append(nn.Linear(fingerprint_dim, hidden_dim1))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim1))
        layers.extend([nn.ReLU(), nn.Dropout(dropout_rate)])

        layers.append(nn.Linear(hidden_dim1, hidden_dim2))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim2))
        layers.extend([nn.ReLU(), nn.Dropout(dropout_rate)])

        layers.append(nn.Linear(hidden_dim2, embedding_dim))

        self.layers = nn.Sequential(*layers)

        print(f"Initialized DrugEmbedderANN: {fingerprint_dim} -> {hidden_dim1} -> {hidden_dim2} -> {embedding_dim} | Dropout: {dropout_rate}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of drug fingerprints. Shape: [batch_size, fingerprint_dim].

        Returns:
            Drug embeddings. Shape: [batch_size, embedding_dim].
        """
        return self.layers(x)

    def get_model_info(self) -> Dict[str, int]:
        """Get model architecture information.

        Returns:
            Dictionary with model configuration.
        """
        return {
            'fingerprint_dim': self.fingerprint_dim,
            'hidden_dim1': self.hidden_dim1,
            'hidden_dim2': self.hidden_dim2,
            'embedding_dim': self.embedding_dim,
            'dropout_rate': self.dropout_rate,
            'use_batchnorm': self.use_batchnorm,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
