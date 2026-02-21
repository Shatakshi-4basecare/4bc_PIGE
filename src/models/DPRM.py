"""Dynamic Pathway Response Module (DPRM) for drug-pathway interactions.
"""

from typing import Dict, List, Optional, Set
from collections import defaultdict
import pickle

import torch
import torch.nn as nn


def load_directed_pathway_graph_from_pickle(
    pickle_path: str,
    ordered_pathway_names: List[str]
) -> Dict[str, List[str]]:
    """Load directed pathway interaction graph from pickle file.

    The pickle file should contain:
        - 'nodes': List of pathway IDs
        - 'edge_list': List of [source, target] pairs

    Args:
        pickle_path: Path to pickle file.
        ordered_pathway_names: List of pathway IDs for validation.

    Returns:
        Dictionary mapping pathway_id → list of children pathway IDs.
    """
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    nodes = data['nodes']
    edge_list = data['edge_list']

    dag = defaultdict(list)
    for source, target in edge_list:
        if source in ordered_pathway_names and target in ordered_pathway_names:
            dag[source].append(target)

    print(f"Loaded pathway graph: {len(nodes)} nodes, {len(edge_list)} edges")
    print(f"Graph contains cycles (feedback loops) - using iterative message passing")

    return dict(dag)


def compute_processing_order(graph: Dict[str, List[str]], all_nodes: List[str]) -> List[str]:
    """Compute a reasonable processing order for a directed cyclic graph.

    Args:
        graph: Adjacency list representation {node: [children]}.
        all_nodes: All node IDs.

    Returns:
        Ordered list of nodes (upstream-ish to downstream-ish).
    """
    # Count incoming edges
    in_degree = {node: 0 for node in all_nodes}
    for children in graph.values():
        for child in children:
            if child in in_degree:
                in_degree[child] += 1

    # Sort by in-degree (ascending), then by name
    sorted_nodes = sorted(all_nodes, key=lambda n: (in_degree[n], n))

    return sorted_nodes


class PathwayStructuralEquation(nn.Module):
    """Structural equations for all pathways (vectorized for GPU efficiency).

    Each pathway's state is computed as:
        pathway_i = f(parents(i), genomics(i), drug)

    Where f is a neural network that combines three sources of information:
        1. Aggregated parent states (what upstream pathways are doing)
        2. Genomic features (mutations, CNAs, RNA specific to this pathway)
        3. Drug embedding (shared drug context)
    """

    def __init__(
        self,
        n_pathways: int,
        pathway_embedding_dim: int,
        drug_embedding_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.n_pathways = n_pathways

        # Encode each source of information
        self.parent_aggregator = nn.Linear(pathway_embedding_dim, hidden_dim)
        self.genomic_encoder = nn.Linear(pathway_embedding_dim, hidden_dim)
        self.drug_modulator = nn.Linear(drug_embedding_dim, hidden_dim)

        # Combine all sources into pathway state
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pathway_embedding_dim),
            nn.Tanh()
        )

    def forward(
        self,
        aggregated_parents: torch.Tensor,
        genomic_features: torch.Tensor,
        drug_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Compute all pathway states in parallel.

        Args:
            aggregated_parents: Pre-aggregated parent states. Shape: [n_pathways, embedding_dim].
            genomic_features: Initial genomic embeddings. Shape: [n_pathways, embedding_dim].
            drug_embedding: Drug embedding. Shape: [drug_embedding_dim].

        Returns:
            Pathway states. Shape: [n_pathways, embedding_dim].
        """
        parent_contribs = self.parent_aggregator(aggregated_parents)
        genomic_contribs = self.genomic_encoder(genomic_features)

        # Broadcast drug embedding to all pathways
        drug_contribs = self.drug_modulator(drug_embedding).unsqueeze(0).expand(
            self.n_pathways, -1
        )

        # Combine all sources
        combined = torch.cat([parent_contribs, genomic_contribs, drug_contribs], dim=1)
        pathway_states = self.combiner(combined)

        return pathway_states


class DPRM(nn.Module):
    """Dynamic Pathway Response Module for pathway interactions.

    This model represents pathway interactions as a directed cyclic graph where
    each pathway's state is determined by its parents, genomic features, and drug.

    Args:
        pathway_dag: Adjacency list {pathway_id: [children]}.
        pathway_dict: {pathway_id: set of genes}.
        ordered_pathway_names: List of pathway IDs in order.
        initial_pathway_encoder: GNN for initial pathway embeddings.
        drug_encoder: ANN for drug embeddings.
        pathway_embedding_dim: Dimension of pathway embeddings.
        drug_embedding_dim: Dimension of drug embeddings.
        hidden_dim: Hidden dimension for structural equations.
        dropout: Dropout rate.
        num_message_passing_steps: Iterations for handling cycles.
        selected_omics_type: Which omics to use ('all', 'mutation', 'cna', 'rna').
    """

    N_FEATURE_TYPES = 3

    def __init__(
        self,
        pathway_dag: Dict[str, List[str]],
        pathway_dict: Dict[str, Set[str]],
        ordered_pathway_names: List[str],
        initial_pathway_encoder: nn.Module,
        drug_encoder: nn.Module,
        pathway_embedding_dim: int = 128,
        drug_embedding_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        num_message_passing_steps: int = 3,
        selected_omics_type: str = "all",
    ) -> None:
        super().__init__()

        self.pathway_dag = pathway_dag
        self.pathway_dict = pathway_dict
        self.ordered_pathway_names = ordered_pathway_names
        self.n_pathways = len(ordered_pathway_names)
        self.num_message_passing_steps = num_message_passing_steps
        self.selected_omics_type = self._normalize_selected_omics_type(selected_omics_type)
        self._selected_omics_block_idx = self._determine_block_index(self.selected_omics_type)
        self._warned_omics_shape_mismatch = False

        # Map pathway names to indices
        self.pathway_to_idx = {name: i for i, name in enumerate(ordered_pathway_names)}

        # Compute processing order
        self.processing_order = compute_processing_order(pathway_dag, ordered_pathway_names)
        self.processing_indices = [self.pathway_to_idx[name] for name in self.processing_order]

        # Encoders
        self.initial_encoder = initial_pathway_encoder
        self.drug_encoder = drug_encoder

        # Structural equations
        self.structural_equations = PathwayStructuralEquation(
            n_pathways=self.n_pathways,
            pathway_embedding_dim=pathway_embedding_dim,
            drug_embedding_dim=drug_embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        # Pre-compute parent aggregation weights (mean pooling)
        self.register_buffer('parent_mask', torch.zeros(self.n_pathways, self.n_pathways))
        for a, b in pathway_dag.items():
            if a in self.pathway_to_idx:
                d = self.pathway_to_idx[a]
                for c in b:
                    if c in self.pathway_to_idx:
                        e = self.pathway_to_idx[c]
                        self.parent_mask[d, e] = 1.0

        # Normalize to get mean pooling weights
        parent_counts = self.parent_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        self.register_buffer('parent_weights', self.parent_mask / parent_counts)

        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        if self.selected_omics_type != "all":
            print(f"Using only {self.selected_omics_type.upper()} omics features")

    def forward(
        self,
        pafe_features: torch.Tensor,
        edge_index: torch.Tensor,
        drug_fingerprint: torch.Tensor,
        interventions: Optional[Dict[int, float]] = None
    ) -> torch.Tensor:
        """Forward pass with optional interventions (do-operator).

        Args:
            pafe_features: Pathway genomic features. Shape: [n_pathways, feature_dim] or [batch, n_pathways, feature_dim].
            edge_index: Graph structure for initial GNN encoding. Shape: [2, num_edges].
            drug_fingerprint: Drug molecular features. Shape: [fp_dim] or [batch, fp_dim].
            interventions: Optional dict {pathway_idx: intervention_value}.
                          E.g., {5: 0.0} means do(pathway_5 = 0).

        Returns:
            Prediction (drug efficacy). Shape: [1] or [batch, 1].
        """
        # Handle batching
        is_batched = drug_fingerprint.dim() == 2 and drug_fingerprint.shape[0] > 1
        if is_batched:
            batch_size = drug_fingerprint.shape[0]
            predictions = []
            for i in range(batch_size):
                pred = self._forward_single(
                    pafe_features, edge_index, drug_fingerprint[i], interventions
                )
                predictions.append(pred)
            return torch.stack(predictions, dim=0)
        else:
            return self._forward_single(pafe_features, edge_index, drug_fingerprint, interventions)

    def _forward_single(
        self,
        pafe_features: torch.Tensor,
        edge_index: torch.Tensor,
        drug_fingerprint: torch.Tensor,
        interventions: Optional[Dict[int, float]] = None
    ) -> torch.Tensor:
        """Single sample forward pass (fully vectorized over pathways).

        Args:
            pafe_features: Shape: [n_pathways, feature_dim].
            edge_index: Shape: [2, num_edges].
            drug_fingerprint: Shape: [fp_dim].
            interventions: Optional dict {pathway_idx: intervention_value}.

        Returns:
            Prediction. Shape: [1].
        """
        # Prepare PAFE features (handle sparse, align shape, apply omics selection)
        pafe_features = self._prepare_pafe_features(pafe_features)

        # 1. Initial pathway embeddings from GNN
        initial_embeddings, _ = self.initial_encoder(pafe_features, edge_index)

        # 2. Encode drug
        if drug_fingerprint.dim() == 1:
            drug_embedding = self.drug_encoder(drug_fingerprint.unsqueeze(0)).squeeze(0)
        else:
            drug_embedding = self.drug_encoder(drug_fingerprint)
            if drug_embedding.dim() == 2:
                drug_embedding = drug_embedding.squeeze(0)

        # 3. Iterative message passing (handles cycles)
        pathway_states = initial_embeddings.clone()

        for _ in range(self.num_message_passing_steps):
            prev_states = pathway_states.clone()

            # Vectorized parent aggregation (mean pooling)
            aggregated_parents = torch.mm(self.parent_weights, prev_states)

            # Compute all pathway states in parallel
            pathway_states = self.structural_equations(
                aggregated_parents=aggregated_parents,
                genomic_features=initial_embeddings,
                drug_embedding=drug_embedding
            )

            # 4. Apply interventions (do-operator)
            if interventions:
                for pathway_idx, intervention_value in interventions.items():
                    pathway_states[pathway_idx] = intervention_value

        # 5. Aggregate and predict
        aggregated = pathway_states.flatten()
        prediction = self.prediction_head(aggregated)

        return prediction

    def compute_causal_effect(
        self,
        pafe_features: torch.Tensor,
        edge_index: torch.Tensor,
        drug_fingerprint: torch.Tensor,
        pathway_idx: int,
        intervention_value: float = 0.0
    ) -> Dict[str, float]:
        """Compute total, direct, and indirect response effects of a pathway.

        Args:
            pafe_features: Genomic features. Shape: [n_pathways, feature_dim].
            edge_index: Graph structure. Shape: [2, num_edges].
            drug_fingerprint: Drug features. Shape: [fp_dim].
            pathway_idx: Index of pathway to intervene on.
            intervention_value: Value to set pathway to (default: 0.0 = knockout).

        Returns:
            Dictionary with total_effect, direct_effect, indirect_effect,
            baseline_prediction, intervention_prediction.
        """
        with torch.no_grad():
            # Baseline: no intervention
            baseline_pred = self.forward(pafe_features, edge_index, drug_fingerprint).item()

            # Intervene on target pathway
            intervention_pred = self.forward(
                pafe_features, edge_index, drug_fingerprint,
                interventions={pathway_idx: intervention_value}
            ).item()

            total_effect = baseline_pred - intervention_pred

            # Direct effect: intervene on pathway AND fix all children
            pathway_name = self.ordered_pathway_names[pathway_idx]
            children_names = self.pathway_dag.get(pathway_name, [])
            children_indices = [self.pathway_to_idx[c] for c in children_names if c in self.pathway_to_idx]

            if children_indices:
                # Get children states under baseline
                baseline_states = self._get_pathway_states(pafe_features, edge_index, drug_fingerprint)

                # Intervene on pathway, fix children to baseline
                combined_interventions = {pathway_idx: intervention_value}
                for child_idx in children_indices:
                    combined_interventions[child_idx] = baseline_states[child_idx].mean().item()

                direct_pred = self.forward(
                    pafe_features, edge_index, drug_fingerprint,
                    interventions=combined_interventions
                ).item()

                direct_effect = baseline_pred - direct_pred
                indirect_effect = total_effect - direct_effect
            else:
                # No children = no indirect effect
                direct_effect = total_effect
                indirect_effect = 0.0

        return {
            'total_effect': total_effect,
            'direct_effect': direct_effect,
            'indirect_effect': indirect_effect,
            'baseline_prediction': baseline_pred,
            'intervention_prediction': intervention_pred
        }

    def _get_pathway_states(
        self,
        pafe_features: torch.Tensor,
        edge_index: torch.Tensor,
        drug_fingerprint: torch.Tensor
    ) -> torch.Tensor:
        """Get pathway states without making prediction.

        Args:
            pafe_features: Pathway genomic features. Shape: [n_pathways, feature_dim].
            edge_index: Graph structure. Shape: [2, num_edges].
            drug_fingerprint: Drug features. Shape: [fp_dim].

        Returns:
            Pathway states. Shape: [n_pathways, embedding_dim].
        """
        pafe_features = self._prepare_pafe_features(pafe_features)
        initial_embeddings, _ = self.initial_encoder(pafe_features, edge_index)
        drug_embedding = self.drug_encoder(drug_fingerprint.unsqueeze(0) if drug_fingerprint.dim() == 1 else drug_fingerprint)
        if drug_embedding.dim() == 2:
            drug_embedding = drug_embedding.squeeze(0)

        pathway_states = initial_embeddings.clone()

        for _ in range(self.num_message_passing_steps):
            prev_states = pathway_states.clone()
            aggregated_parents = torch.mm(self.parent_weights, prev_states)
            pathway_states = self.structural_equations(
                aggregated_parents=aggregated_parents,
                genomic_features=initial_embeddings,
                drug_embedding=drug_embedding
            )

        return pathway_states

    @staticmethod
    def _is_sparse_like_tensor(pafe_features: torch.Tensor) -> bool:
        """Return True if tensor is sparse or sparse-compatible."""
        try:
            layout_str = str(getattr(pafe_features, 'layout', ''))
            if 'sparse' in layout_str.lower():
                return True
            return bool(getattr(pafe_features, 'is_sparse', False))
        except Exception:
            return False

    @staticmethod
    def _determine_block_index(normalized_type: str) -> Optional[int]:
        """Map normalized omics type to block index."""
        if normalized_type == "all":
            return None
        block_map = {"mutation": 0, "cna": 1, "rna": 2}
        return block_map[normalized_type]

    def _apply_omics_selection(self, pafe_features: torch.Tensor) -> torch.Tensor:
        """Zero out unused omics feature blocks.

        PAFE features are organized as [mutation | CNA | RNA].
        This method zeros out the blocks we're not using.

        Args:
            pafe_features: Shape: [n_pathways, feature_dim].

        Returns:
            Filtered features. Shape: [n_pathways, feature_dim].
        """
        if self._selected_omics_block_idx is None:
            return pafe_features

        feature_dim = pafe_features.shape[-1]
        if feature_dim % self.N_FEATURE_TYPES != 0:
            if not self._warned_omics_shape_mismatch:
                print(f"Warning: feature_dim ({feature_dim}) not divisible by {self.N_FEATURE_TYPES}")
                self._warned_omics_shape_mismatch = True
            return pafe_features

        block_width = feature_dim // self.N_FEATURE_TYPES
        start = self._selected_omics_block_idx * block_width
        end = start + block_width
        filtered = torch.zeros_like(pafe_features)
        filtered[..., start:end] = pafe_features[..., start:end]
        return filtered

    def _prepare_pafe_features(self, pafe_features: torch.Tensor) -> torch.Tensor:
        """Convert to dense, align shape, and apply omics filtering.

        Args:
            pafe_features: Input features. Shape varies, will be normalized to [n_pathways, feature_dim].

        Returns:
            Prepared features. Shape: [n_pathways, feature_dim].
        """
        # Convert sparse to dense
        if self._is_sparse_like_tensor(pafe_features):
            pafe_features = pafe_features.to_dense()

        # Handle batch dimension
        if pafe_features.dim() == 3:
            if pafe_features.shape[0] == 1:
                pafe_features = pafe_features.squeeze(0)
            else:
                raise ValueError(f"Batched pafe_features not supported: {pafe_features.shape}")

        if pafe_features.dim() != 2:
            raise ValueError(f"pafe_features must be 2D, got {pafe_features.shape}")

        # Align to n_pathways
        if pafe_features.shape[0] != self.n_pathways:
            if pafe_features.shape[0] % self.n_pathways == 0:
                feat_dim = pafe_features.shape[1]
                pafe_features = pafe_features.view(-1, self.n_pathways, feat_dim)[0]
            else:
                raise ValueError(
                    f"pafe_features dim ({pafe_features.shape[0]}) "
                    f"doesn't match n_pathways ({self.n_pathways})"
                )

        # Apply omics selection
        return self._apply_omics_selection(pafe_features)

    @staticmethod
    def _normalize_selected_omics_type(selected_omics_type: Optional[str]) -> str:
        """Normalize requested omics type string."""
        if not selected_omics_type:
            return "all"
        normalized = selected_omics_type.strip().lower()
        allowed = {"all", "mutation", "cna", "rna"}
        if normalized not in allowed:
            raise ValueError(
                f"Unsupported omics selection '{selected_omics_type}'. "
                "Valid options: all, mutation, cna, rna."
            )
        return normalized