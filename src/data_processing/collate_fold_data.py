"""GPU-based data collation for fold-based cross-validation.

Handles sparse PAFE tensor batching, drug fingerprints, and graph edge indices
for efficient GPU-accelerated machine learning.
"""

from typing import Any, Dict, List, Union

import torch
from torch.utils.data import Dataset


class DataConstants:
    """Data collation constants."""

    PAFE_FEATURES_KEY = 'pafe_features'
    DRUG_FINGERPRINT_KEY = 'drug_fingerprint'
    LABEL_KEY = 'label'
    DRUG_NAME_KEY = 'drug_name'

    PAFE_FEATURES_FLAT_KEY = 'pafe_features_flat'
    EDGE_INDEX_BATCH_KEY = 'edge_index_batch'
    DRUG_FINGERPRINTS_KEY = 'drug_fingerprints'
    LABELS_KEY = 'labels'
    DRUG_NAMES_KEY = 'drug_names'

    SPARSE_TENSOR_INDICES_DIM = 2 # [node_idx, feature_idx]
    NODE_INDEX_ROW = 0 # Row index for node indices in sparse tensor
    FEATURE_INDEX_ROW = 1 # Row index for feature indices in sparse tensor


class GPUDatasetFold(Dataset):
    """Dataset wrapper for preloaded GPU data in fold-based cross-validation.

    Args:
        data_list: List of data dictionaries containing model inputs.
    """

    def __init__(self, data_list: List[Dict[str, Any]]) -> None:
        """Initialize dataset with preloaded data.

        Args:
            data_list: List of data dictionaries.
        """
        self.data_list = data_list

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get data item by index.

        Args:
            idx: Index of item to retrieve.

        Returns:
            Data dictionary.
        """
        return self.data_list[idx]

    def __len__(self) -> int:
        """Get number of items in dataset.

        Returns:
            Dataset size.
        """
        return len(self.data_list)


def collate_sparse_pafe_features(
    batch_list: List[Dict[str, Any]],
    num_nodes: int
) -> tuple[torch.Tensor, torch.Size]:
    """Collate sparse PAFE features into single sparse tensor.

    Args:
        batch_list: List of batch items with sparse PAFE features.
        num_nodes: Expected number of nodes per item.

    Returns:
        Tuple of (batched sparse tensor, final tensor shape).
    """
    pafe_indices_list = []
    pafe_values_list = []
    pafe_shape_per_item = None
    current_node_offset = 0

    for item in batch_list:
        pafe_sparse = item[DataConstants.PAFE_FEATURES_KEY]

        if pafe_shape_per_item is None:
            pafe_shape_per_item = pafe_sparse.shape

        item_indices = pafe_sparse.indices().clone()
        item_values = pafe_sparse.values()

        item_indices[DataConstants.NODE_INDEX_ROW, :] += current_node_offset

        pafe_indices_list.append(item_indices)
        pafe_values_list.append(item_values)

        current_node_offset += num_nodes

    all_pafe_indices = torch.cat(pafe_indices_list, dim=1)
    all_pafe_values = torch.cat(pafe_values_list, dim=0)

    batch_size = len(batch_list)
    final_shape = (batch_size * num_nodes, pafe_shape_per_item[1])

    batched_sparse_tensor = torch.sparse_coo_tensor(
        all_pafe_indices,
        all_pafe_values,
        final_shape,
        dtype=all_pafe_values.dtype
    )

    return batched_sparse_tensor, final_shape


def collate_dense_features(batch_list: List[Dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate dense features (drug fingerprints and labels).

    Args:
        batch_list: List of batch items with dense features.

    Returns:
        Tuple of (drug_fingerprints tensor, labels tensor).
    """
    drug_fingerprints = torch.stack([
        item[DataConstants.DRUG_FINGERPRINT_KEY] for item in batch_list
    ])
    labels = torch.stack([
        item[DataConstants.LABEL_KEY] for item in batch_list
    ])

    return drug_fingerprints, labels


def create_batched_edge_index(
    edge_index_template: torch.Tensor,
    num_nodes: int,
    batch_size: int
) -> torch.Tensor:
    """Create batched edge index by replicating template.

    Args:
        edge_index_template: Template edge index for single graph.
        num_nodes: Number of nodes per graph.
        batch_size: Number of graphs in batch.

    Returns:
        Batched edge index tensor.
    """
    batched_edge_indices = []

    for i in range(batch_size):
        node_offset = i * num_nodes
        offset_edge_index = edge_index_template + node_offset
        batched_edge_indices.append(offset_edge_index)

    final_edge_index = torch.cat(batched_edge_indices, dim=1)
    return final_edge_index


def collate_gpu_fold(
    batch_list_on_cpu: List[Dict[str, Any]],
    num_nodes: int,
    edge_index_for_batch: torch.Tensor
) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """Custom collation function for GPU-based fold data.

    Batches sparse PAFE features, dense drug features, and creates edge indices
    for graph neural network processing.

    Args:
        batch_list_on_cpu: List of data items with:
            - pafe_features: Sparse tensor [num_nodes, pafe_feature_dim]
            - drug_fingerprint: Dense drug feature tensor
            - label: Target label tensor
            - drug_name: Drug identifier string
        num_nodes: Expected number of nodes per graph.
        edge_index_for_batch: Template edge index, determines target device.

    Returns:
        Dictionary with batched data:
        - pafe_features_flat: Sparse tensor [batch_size * num_nodes, pafe_feature_dim]
        - edge_index_batch: Edge index for batched graphs
        - drug_fingerprints: Stacked drug fingerprint tensors
        - labels: Stacked label tensors
        - drug_names: List of drug names
    """
    batch_size = len(batch_list_on_cpu)
    target_device = edge_index_for_batch.device

    pafe_batched_cpu, final_pafe_shape = collate_sparse_pafe_features(
        batch_list_on_cpu, num_nodes
    )

    drug_fingerprints_cpu, labels_cpu = collate_dense_features(batch_list_on_cpu)

    drug_names = [item[DataConstants.DRUG_NAME_KEY] for item in batch_list_on_cpu]

    batched_edge_index = create_batched_edge_index(
        edge_index_for_batch, num_nodes, batch_size
    )

    # Move data to target device
    pafe_batched = pafe_batched_cpu.to(target_device)
    drug_fingerprints = drug_fingerprints_cpu.to(target_device)
    labels = labels_cpu.to(target_device)

    return {
        DataConstants.PAFE_FEATURES_FLAT_KEY: pafe_batched,
        DataConstants.EDGE_INDEX_BATCH_KEY: batched_edge_index,
        DataConstants.DRUG_FINGERPRINTS_KEY: drug_fingerprints,
        DataConstants.LABELS_KEY: labels,
        DataConstants.DRUG_NAMES_KEY: drug_names
    }
