"""SCM wrapper for training integration.

Wraps StructuralCausalModel to be compatible with existing training pipelines.
"""

from typing import Dict, List, Set, Optional

import torch
import torch.nn as nn

from .structural_causal_model import (
    StructuralCausalModel,
    load_directed_pathway_graph_from_pickle
)
from .GAT_Layer import GATNetwork
from .drug_ann import DrugEmbedderANN


class SCMModelWrapper(nn.Module):
    """Wrapper around StructuralCausalModel for training compatibility.

    Args:
        scm: The underlying StructuralCausalModel instance.
    """

    def __init__(self, scm: StructuralCausalModel) -> None:
        super().__init__()
        self.scm = scm

    def forward(
        self,
        pafe_features: torch.Tensor,
        edge_index: torch.Tensor,
        drug_fingerprints: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the SCM model.

        Args:
            pafe_features: Pathway features. Shape: [n_pathways, feature_dim] or batched.
            edge_index: Graph edges. Shape: [2, n_edges].
            drug_fingerprints: Drug features. Shape: [batch_size, fp_dim] or [batch_size, 1, fp_dim].

        Returns:
            Predictions. Shape: [batch_size, 1].
        """
        if drug_fingerprints.dim() == 3 and drug_fingerprints.shape[1] == 1:
            drug_fingerprints = drug_fingerprints.squeeze(1)

        if drug_fingerprints.dim() == 1:
            drug_fingerprints = drug_fingerprints.unsqueeze(0)

        # Convert sparse to dense if needed
        if hasattr(pafe_features, 'is_sparse') and pafe_features.is_sparse:
            pafe_features = pafe_features.to_dense()
        elif 'sparse' in str(getattr(pafe_features, 'layout', '')):
            pafe_features = pafe_features.to_dense()

        # Normalize PAFE features to [batch, n_pathways, pafe_dim]
        n_pathways = self.scm.n_pathways
        if pafe_features.dim() == 2:
            total_nodes, pafe_dim = pafe_features.shape
            if total_nodes == n_pathways:
                pafe_batch = pafe_features.unsqueeze(0)
            elif total_nodes % n_pathways == 0:
                batch_pf = total_nodes // n_pathways
                pafe_batch = pafe_features.reshape(batch_pf, n_pathways, pafe_dim)
            else:
                raise ValueError(
                    f"pafe_features shape {pafe_features.shape} incompatible "
                    f"with n_pathways={n_pathways}"
                )
        elif pafe_features.dim() == 3:
            pafe_batch = pafe_features
        else:
            raise ValueError(f"Unsupported pafe_features ndim={pafe_features.dim()}")

        # Align batch sizes between PAFE and drugs
        batch_pf = pafe_batch.shape[0]
        batch_drug = drug_fingerprints.shape[0]
        if batch_pf != batch_drug:
            if batch_pf == 1:
                pafe_batch = pafe_batch.expand(batch_drug, -1, -1)
                batch_pf = batch_drug
            elif batch_drug == 1:
                drug_fingerprints = drug_fingerprints.expand(batch_pf, -1)
                batch_drug = batch_pf
            else:
                min_batch = min(batch_pf, batch_drug)
                pafe_batch = pafe_batch[:min_batch]
                drug_fingerprints = drug_fingerprints[:min_batch]
                batch_pf = batch_drug = min_batch

        # Check if edge_index is batched
        max_node_index = int(edge_index.max().item()) if edge_index.numel() > 0 else -1
        has_batched_edges = max_node_index >= n_pathways

        preds = []
        for i in range(batch_pf):
            if has_batched_edges:
                start = i * n_pathways
                end = start + n_pathways
                ei = edge_index
                src = ei[0]
                dst = ei[1]
                mask = (src >= start) & (src < end) & (dst >= start) & (dst < end)
                if mask.any():
                    src_i = (src[mask] - start).to(ei.device)
                    dst_i = (dst[mask] - start).to(ei.device)
                    edge_index_single = torch.stack([src_i, dst_i], dim=0)
                else:
                    edge_index_single = (edge_index % n_pathways)
            else:
                edge_index_single = edge_index

            preds.append(
                self.scm._forward_single(
                    pafe_batch[i], edge_index_single, drug_fingerprints[i]
                )
            )
        predictions = torch.stack(preds, dim=0)
        return predictions

    def compute_causal_effect(
        self,
        pafe_features: torch.Tensor,
        edge_index: torch.Tensor,
        drug_fingerprint: torch.Tensor,
        pathway_idx: int,
        intervention_value: float = 0.0
    ) -> Dict[str, float]:
        """Compute causal effects for interpretability.

        Args:
            pafe_features: Pathway genomic features. Shape: [n_pathways, feature_dim].
            edge_index: Graph structure. Shape: [2, num_edges].
            drug_fingerprint: Drug features. Shape: [fp_dim].
            pathway_idx: Index of pathway to intervene on.
            intervention_value: Value to set pathway to (default: 0.0 = knockout).

        Returns:
            Dictionary with total_effect, direct_effect, indirect_effect.
        """
        return self.scm.compute_causal_effect(
            pafe_features, edge_index, drug_fingerprint,
            pathway_idx, intervention_value
        )


def create_scm_model(
    pathway_dict: Dict[str, Set[str]],
    ordered_pathway_names: List[str],
    pafe_feature_dim: int,
    fp_dim: int,
    pathway_graph_pickle_path: str,
    pathway_embedding_dim: int = 128,
    drug_embedding_dim: int = 128,
    gnn_hidden_dim: int = 512,
    gnn_heads: int = 8,
    gnn_dropout: float = 0.1,
    ann_hidden_dim1: int = 512,
    ann_hidden_dim2: int = 256,
    ann_dropout: float = 0.2,
    scm_hidden_dim: int = 128,
    scm_dropout: float = 0.1,
    num_message_passing_steps: int = 3,
    selected_omics_type: str = "all",
) -> SCMModelWrapper:
    """Create a complete SCM model with all components.

    This is the main entry point for creating SCM models.

    Args:
        pathway_dict: Dictionary mapping pathway_id to set of genes.
        ordered_pathway_names: List of pathway IDs in order.
        pafe_feature_dim: Dimension of input PAFE features.
        fp_dim: Dimension of drug fingerprints.
        pathway_graph_pickle_path: Path to pathway interaction graph pickle.
        pathway_embedding_dim: Dimension of pathway embeddings.
        drug_embedding_dim: Dimension of drug embeddings.
        gnn_hidden_dim: GNN hidden layer dimension.
        gnn_heads: Number of GAT attention heads.
        gnn_dropout: GNN dropout rate.
        ann_hidden_dim1: Drug ANN first hidden layer.
        ann_hidden_dim2: Drug ANN second hidden layer.
        ann_dropout: Drug ANN dropout.
        scm_hidden_dim: SCM structural equation hidden dimension.
        scm_dropout: SCM dropout.
        num_message_passing_steps: Number of message passing iterations.
        selected_omics_type: Which omics channel to keep ('all', 'mutation', 'cna', 'rna').

    Returns:
        SCMModelWrapper ready for training.
    """
    # Load pathway DAG from pickle file
    pathway_dag = load_directed_pathway_graph_from_pickle(
        pathway_graph_pickle_path, ordered_pathway_names
    )

    # Create initial pathway encoder (GNN)
    initial_encoder = GATNetwork(
        in_features=pafe_feature_dim,
        hidden_dim1=gnn_hidden_dim,
        out_features=pathway_embedding_dim,
        heads_l1=gnn_heads,
        dropout=gnn_dropout
    )

    # Create drug encoder (ANN)
    drug_encoder = DrugEmbedderANN(
        fingerprint_dim=fp_dim,
        hidden_dim1=ann_hidden_dim1,
        hidden_dim2=ann_hidden_dim2,
        embedding_dim=drug_embedding_dim,
        dropout_rate=ann_dropout
    )

    # Create SCM
    scm = StructuralCausalModel(
        pathway_dag=pathway_dag,
        pathway_dict=pathway_dict,
        ordered_pathway_names=ordered_pathway_names,
        initial_pathway_encoder=initial_encoder,
        drug_encoder=drug_encoder,
        pathway_embedding_dim=pathway_embedding_dim,
        drug_embedding_dim=drug_embedding_dim,
        hidden_dim=scm_hidden_dim,
        dropout=scm_dropout,
        num_message_passing_steps=num_message_passing_steps,
        selected_omics_type=selected_omics_type,
    )

    wrapper = SCMModelWrapper(scm)

    return wrapper


def load_scm_from_checkpoint(
    checkpoint_path: str,
    pathway_dict: Dict[str, Set[str]],
    ordered_pathway_names: List[str],
    pathway_graph_pickle_path: str,
    selected_omics_type: Optional[str] = None,
    **model_kwargs
) -> SCMModelWrapper:
    """Load a trained SCM from checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint file.
        pathway_dict: Pathway definitions.
        ordered_pathway_names: Pathway order.
        pathway_graph_pickle_path: Path to pathway interaction graph pickle.
        selected_omics_type: Optional override for the omics subset.
        **model_kwargs: Additional model creation arguments.

    Returns:
        Loaded SCM model.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    checkpoint_hps = {}
    if isinstance(checkpoint, dict):
        checkpoint_hps = checkpoint.get('hyperparameters', {})

    resolved_omics_type = (
        selected_omics_type
        or checkpoint_hps.get('selected_omics_type')
        or checkpoint_hps.get('SELECTED_OMICS_TYPE')
        or "all"
    )

    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Omics type: {resolved_omics_type}")

    # Create model architecture
    model = create_scm_model(
        pathway_dict=pathway_dict,
        ordered_pathway_names=ordered_pathway_names,
        pathway_graph_pickle_path=pathway_graph_pickle_path,
        selected_omics_type=resolved_omics_type,
        **model_kwargs
    )

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Handle old checkpoint format
    if any(key.startswith('scm._genomic_residual_proj') for key in state_dict.keys()):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('scm._genomic_residual_proj'):
                # Convert scm._genomic_residual_proj.* to scm.structural_equations.genomic_residual_proj.*
                new_key = key.replace('scm._genomic_residual_proj', 'scm.structural_equations.genomic_residual_proj')
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"Warning: Missing keys when loading checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys when loading checkpoint: {unexpected_keys}")

    return model
