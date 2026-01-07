"""Virtual knockout analysis for drug response models.

Provides pathway and gene knockout analysis to identify critical pathways
and genes for drug response prediction.
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from pathlib import Path

from ..models.scm_wrapper import SCMModelWrapper
from ..data_processing.feature_generator import FeatureGenerator
from .input_space_pathway_knockout import run_input_space_pathway_knockout_for_sample


def identify_genes_in_enriched_pathways(
    cell_id: str,
    drug_name: str,
    pathway_ko_results_dir: str,
    global_data: Dict,
    go_map: Dict[str, str],
    top_n_pathways: int = 50,
    min_pathways_per_gene: int = 2,
) -> Optional[List[str]]:
    """Identify genes that appear in multiple significantly enriched pathways.

    Args:
        cell_id: Cell line identifier.
        drug_name: Drug name.
        pathway_ko_results_dir: Directory containing pathway knockout results.
        global_data: Dictionary with pathway graph data.
        go_map: GO ID to pathway name mapping.
        top_n_pathways: Number of top pathways to consider as enriched (by absolute importance).
        min_pathways_per_gene: Minimum number of enriched pathways a gene must appear in.

    Returns:
        List of gene names to knock out, or None if insufficient data.
    """
    pathway_ko_file = Path(pathway_ko_results_dir) / f"{drug_name}_ctrpv2_knockout_raw_scores.csv"

    if not pathway_ko_file.exists():
        print(f"Warning: Pathway KO file not found: {pathway_ko_file}")
        return None

    df_pathway_ko = pd.read_csv(pathway_ko_file, index_col=0)

    # Check if cell_id exists in the results
    if cell_id not in df_pathway_ko.index:
        print(f"Warning: Cell {cell_id} not found in pathway KO results for {drug_name}")
        return None

    # Get pathway scores for this cell-drug pair
    pathway_scores = df_pathway_ko.loc[cell_id].drop(['predicted_aac', 'actual_aac'], errors='ignore')

    # Identify top N pathways by absolute importance
    top_pathways = pathway_scores.abs().nlargest(top_n_pathways).index.tolist()

    # Map pathway names back to GO IDs
    name_to_go = {v: k for k, v in go_map.items()}
    top_pathway_ids = [name_to_go.get(p, p) for p in top_pathways]

    # Count how many enriched pathways each gene appears in
    gene_pathway_count = {}
    for pathway_id in top_pathway_ids:
        if pathway_id in global_data['pathway_dict']:
            for gene in global_data['pathway_dict'][pathway_id]:
                gene_pathway_count[gene] = gene_pathway_count.get(gene, 0) + 1

    # Filter genes that appear in at least min_pathways_per_gene enriched pathways
    filtered_genes = [gene for gene, count in gene_pathway_count.items() if count >= min_pathways_per_gene]

    print(f"  Filtered from all genes to {len(filtered_genes)} genes in ≥{min_pathways_per_gene} of top-{top_n_pathways} pathways")

    return filtered_genes if filtered_genes else None


def create_edge_mask_for_knockout(
    knocked_out_pathway_idx: int,
    edge_index: torch.Tensor,
    mask_outgoing: bool = True,
    mask_incoming: bool = False
) -> torch.Tensor:
    """Create edge mask that removes edges to/from knocked-out pathway.

    Args:
        knocked_out_pathway_idx: Index of pathway being knocked out.
        edge_index: Original edge index [2, n_edges].
        mask_outgoing: If True, remove edges FROM the KO pathway.
        mask_incoming: If True, remove edges TO the KO pathway.

    Returns:
        Filtered edge_index with KO pathway edges removed.
    """
    src, dst = edge_index[0], edge_index[1]
    keep_mask = torch.ones(edge_index.shape[1], dtype=torch.bool, device=edge_index.device)

    if mask_outgoing:
        keep_mask &= (src != knocked_out_pathway_idx)

    if mask_incoming:
        keep_mask &= (dst != knocked_out_pathway_idx)

    return edge_index[:, keep_mask]


def generate_synthetic_baseline_state(
    feature_generator: FeatureGenerator,
    device: torch.device,
    gene_minimum_rna_map: Dict[str, float]
) -> torch.Tensor:
    """Generate synthetic baseline state representing a normal cell.

    Args:
        feature_generator: Feature generator instance.
        device: Torch device.
        gene_minimum_rna_map: Minimum RNA value for each gene.

    Returns:
        Synthetic baseline PAFE tensor.
    """
    n_genes = feature_generator.n_genes_in_features
    all_genes = feature_generator.g_universal

    mut_vector = np.zeros(n_genes, dtype=np.float32)
    cna_vector = np.ones(n_genes, dtype=np.float32)
    rna_vector = np.array([gene_minimum_rna_map.get(g, 0) for g in all_genes], dtype=np.float32)

    baseline_pafe_tensor = feature_generator._create_pafe_features_from_vectors(
        mut_vector, cna_vector, rna_vector
    )

    return baseline_pafe_tensor.to(device)


def run_virtual_knockout_for_sample_scm_pathway(
    cell_id: str,
    drug_name: str,
    feature_generator: FeatureGenerator,
    model: SCMModelWrapper,
    global_data: Dict,
    device: torch.device,
    genomic_lookups: Dict = None,
    gene_minimum_rna_map: Dict[str, float] = None,
) -> Optional[Tuple[pd.Series, pd.Series, pd.Series, float]]:
    """Run pathway knockout for SCM models.

    Args:
        cell_id: Cell line identifier.
        drug_name: Drug name.
        feature_generator: Feature generator instance.
        model: SCM model wrapper.
        global_data: Dictionary with pathway names and edge index.
        device: Torch device.
        genomic_lookups: Gene mappings.
        gene_minimum_rna_map: Minimum RNA per gene.

    Returns:
        Tuple of (importance_scores, direct_effects, indirect_effects, baseline_prediction) or None.
    """
    result = run_input_space_pathway_knockout_for_sample(
        cell_id=cell_id,
        drug_name=drug_name,
        feature_generator=feature_generator,
        model=model,
        global_data=global_data,
        device=device,
        genomic_lookups=genomic_lookups,
        gene_minimum_rna_map=gene_minimum_rna_map
    )

    if result is not None:
        importance_series, baseline_prediction = result
        direct_series = pd.Series(dtype=float, name=importance_series.name)
        indirect_series = pd.Series(dtype=float, name=importance_series.name)
        return (importance_series, direct_series, indirect_series, baseline_prediction)
    return None


def run_virtual_gene_knockout_for_sample(
    cell_id: str,
    drug_name: str,
    feature_generator: FeatureGenerator,
    model: torch.nn.Module,
    global_data: Dict,
    device: torch.device,
    genomic_lookups: Dict,
    gene_minimum_rna_map: Dict[str, float],
    baseline_prediction: Optional[float] = None,
    genes_to_knockout: Optional[List[str]] = None,
) -> Optional[Tuple[pd.Series, float]]:
    """Run gene knockout analysis for a single cell line.

    Args:
        cell_id: Cell line identifier.
        drug_name: Drug name.
        feature_generator: Feature generator instance.
        model: Trained model.
        global_data: Dictionary with pathway graph data.
        device: Torch device.
        genomic_lookups: Gene mappings.
        gene_minimum_rna_map: Minimum RNA per gene.
        baseline_prediction: Pre-calculated baseline prediction (optional).
        genes_to_knockout: Optional list of specific genes to knockout. If None, all genes are knocked out.

    Returns:
        Tuple of (importance_scores, baseline_prediction) or None.
    """
    original_pafe_tensor, drug_fp_tensor, original_genomic_vectors = feature_generator.generate_features(
        cell_id, drug_name, return_full_vectors=True
    )

    if original_pafe_tensor is None or drug_fp_tensor is None:
        return None

    original_pafe_tensor = original_pafe_tensor.to(device)
    drug_fp_tensor_batch = drug_fp_tensor.unsqueeze(0).to(device)

    if baseline_prediction is None:
        with torch.no_grad():
            baseline_prediction = model(original_pafe_tensor, global_data['edge_index'], drug_fp_tensor_batch).item()

    gene_to_pathways_map = {gene: [] for gene in genomic_lookups['G_universal']}
    pathway_to_idx_map = {pathway: i for i, pathway in enumerate(global_data['nodes'])}
    for pathway, genes in global_data['pathway_dict'].items():
        for gene in genes:
            if gene in gene_to_pathways_map:
                gene_to_pathways_map[gene].append(pathway)

    n_genes = feature_generator.n_genes_in_features
    cna_offset = n_genes
    rna_offset = 2 * n_genes
    gene_to_idx = {gene: i for i, gene in enumerate(genomic_lookups['G_universal'])}

    # OPTIMIZATION: Create ALL knockout tensors at once, then batch process
    if genes_to_knockout is not None:
        # Use the provided filtered gene list
        genes_to_process = [g for g in genes_to_knockout if g in gene_to_idx]
    else:
        # Use all genes
        genes_to_process = [g for g in genomic_lookups['G_universal'] if g in gene_to_idx]
    n_knockouts = len(genes_to_process)

    if n_knockouts == 0:
        return None

    n_pathways, pafe_dim = original_pafe_tensor.shape

    # Create batched knockout tensors: [n_genes, n_pathways, pafe_dim]
    # Use expand + clone to efficiently create copies
    knockout_batch = original_pafe_tensor.unsqueeze(0).expand(n_knockouts, -1, -1).clone()

    # Apply all knockouts in vectorized manner
    for i, gene_name in enumerate(genes_to_process):
        gene_idx = gene_to_idx[gene_name]
        pathways_for_gene = gene_to_pathways_map.get(gene_name, [])
        pathway_indices = [pathway_to_idx_map[p] for p in pathways_for_gene if p in pathway_to_idx_map]

        if pathway_indices:
            min_rna_val = gene_minimum_rna_map.get(gene_name, 0)
            knockout_batch[i, pathway_indices, gene_idx] = 0
            knockout_batch[i, pathway_indices, cna_offset + gene_idx] = 1
            knockout_batch[i, pathway_indices, rna_offset + gene_idx] = min_rna_val

    # Expand drug tensor to match batch size
    # drug_fp_tensor might already have a batch dimension, so squeeze it first
    drug_fp_squeezed = drug_fp_tensor.squeeze()
    drug_fp_batch = drug_fp_squeezed.unsqueeze(0).expand(n_knockouts, -1).to(device)

    # Single batched forward pass for ALL knockouts
    with torch.no_grad():
        knockout_predictions = model(knockout_batch, global_data['edge_index'], drug_fp_batch).squeeze(-1)

    # Calculate importance scores
    importance_scores = {}
    for i, gene_name in enumerate(genes_to_process):
        score = baseline_prediction - knockout_predictions[i].item()
        importance_scores[gene_name] = score

    if not importance_scores:
        return None

    return pd.Series(importance_scores, name=cell_id), baseline_prediction


def run_virtual_double_knockout_for_sample(
    cell_id: str,
    drug_name: str,
    feature_generator: FeatureGenerator,
    model: torch.nn.Module,
    global_data: Dict,
    device: torch.device,
    pairs_to_process: List[Tuple[str, str]],
    single_ko_scores_for_sample: pd.Series,
    baseline_pathway_states: Optional[torch.Tensor] = None,
    genomic_lookups: Optional[Dict] = None,
    gene_minimum_rna_map: Optional[Dict[str, float]] = None,
    go_map: Optional[Dict[str, str]] = None,
    baseline_prediction: Optional[float] = None,
) -> Optional[Tuple[pd.Series, float]]:
    """Run double knockout analysis for gene or pathway pairs.

    Args:
        cell_id: Cell line identifier.
        drug_name: Drug name.
        feature_generator: Feature generator instance.
        model: Trained model.
        global_data: Dictionary with pathway graph data.
        device: Torch device.
        pairs_to_process: List of (entity1, entity2) pairs to test.
        single_ko_scores_for_sample: Pre-calculated single KO scores.
        baseline_pathway_states: Baseline pathway states for pathway knockout.
        genomic_lookups: Gene mappings for gene knockout.
        gene_minimum_rna_map: Minimum RNA per gene for gene knockout.
        go_map: GO ID to pathway name mapping.
        baseline_prediction: Pre-calculated baseline prediction (optional).

    Returns:
        Tuple of (synergy_scores, baseline_prediction) or None.
    """
    is_gene_knockout = baseline_pathway_states is None

    original_pafe_tensor, drug_fp_tensor, original_genomic_vectors = feature_generator.generate_features(
        cell_id, drug_name, return_full_vectors=True
    )

    if original_pafe_tensor is None or drug_fp_tensor is None:
        return None

    original_pafe_tensor = original_pafe_tensor.to(device)
    drug_fp_tensor_batch = drug_fp_tensor.unsqueeze(0).to(device)

    if baseline_prediction is None:
        with torch.no_grad():
            baseline_prediction = model(original_pafe_tensor, global_data['edge_index'], drug_fp_tensor_batch).item()

    if is_gene_knockout:
        gene_to_pathways_map = {gene: [] for gene in genomic_lookups['G_universal']}
        pathway_to_idx_map = {pathway: i for i, pathway in enumerate(global_data['nodes'])}
        for pathway, genes in global_data['pathway_dict'].items():
            for gene in genes:
                if gene in gene_to_pathways_map:
                    gene_to_pathways_map[gene].append(pathway)

        n_genes = feature_generator.n_genes_in_features
        cna_offset = n_genes
        rna_offset = 2 * n_genes
        entity_to_idx = {gene: i for i, gene in enumerate(genomic_lookups['G_universal'])}
    else:
        entity_to_idx = {name: i for i, name in enumerate(global_data['nodes'])}

    synergy_scores = {}

    for entity1_id, entity2_id in pairs_to_process:
        if entity1_id not in entity_to_idx or entity2_id not in entity_to_idx:
            continue

        if is_gene_knockout:
            pair_name = f"{entity1_id} <-> {entity2_id}"
        else:
            name1 = go_map.get(entity1_id, entity1_id) if go_map else entity1_id
            name2 = go_map.get(entity2_id, entity2_id) if go_map else entity2_id
            pair_name = f"{name1} <-> {name2}"

        try:
            importance1 = single_ko_scores_for_sample[entity1_id]
            importance2 = single_ko_scores_for_sample[entity2_id]
        except KeyError:
            continue

        if is_gene_knockout:
            counterfactual_pafe_tensor = original_pafe_tensor.clone()
            for gene_name in [entity1_id, entity2_id]:
                gene_idx = entity_to_idx.get(gene_name)
                if gene_idx is None:
                    continue

                pathways_for_gene = gene_to_pathways_map.get(gene_name, [])
                pathway_indices = [pathway_to_idx_map[p] for p in pathways_for_gene if p in pathway_to_idx_map]

                if pathway_indices:
                    min_rna_val = gene_minimum_rna_map.get(gene_name, 0)
                    counterfactual_pafe_tensor[pathway_indices, gene_idx] = 0
                    counterfactual_pafe_tensor[pathway_indices, cna_offset + gene_idx] = 1
                    counterfactual_pafe_tensor[pathway_indices, rna_offset + gene_idx] = min_rna_val
        else:
            counterfactual_pafe_tensor = original_pafe_tensor.clone()
            idx1 = entity_to_idx[entity1_id]
            idx2 = entity_to_idx[entity2_id]
            counterfactual_pafe_tensor[idx1, :] = baseline_pathway_states[idx1, :]
            counterfactual_pafe_tensor[idx2, :] = baseline_pathway_states[idx2, :]

        with torch.no_grad():
            double_ko_prediction = model(counterfactual_pafe_tensor, global_data['edge_index'], drug_fp_tensor_batch).item()

        importance_double = baseline_prediction - double_ko_prediction
        synergy_score = importance_double - (importance1 + importance2)
        synergy_scores[pair_name] = synergy_score

    if not synergy_scores:
        return None

    return pd.Series(synergy_scores, name=cell_id), baseline_prediction


def run_synergistic_knockout_for_gene_sample(
    cell_id: str,
    drug_name: str,
    primary_knockout_gene: str,
    feature_generator: FeatureGenerator,
    model: torch.nn.Module,
    global_data: Dict,
    device: torch.device,
    genomic_lookups: Dict,
    gene_minimum_rna_map: Dict[str, float],
    baseline_prediction: Optional[float] = None,
) -> Optional[Tuple[pd.Series, float]]:
    """Run synergistic knockout analysis for a single cell line.

    Performs primary knockout of specified gene, then systematically knocks out
    every other gene to calculate synergy scores.

    Args:
        cell_id: Cell line identifier.
        drug_name: Drug name.
        primary_knockout_gene: Gene to knock out first.
        feature_generator: Feature generator instance.
        model: Trained model.
        global_data: Dictionary with pathway graph data.
        device: Torch device.
        genomic_lookups: Gene mappings.
        gene_minimum_rna_map: Minimum RNA per gene.
        baseline_prediction: Pre-calculated baseline prediction (optional).

    Returns:
        Tuple of (synergy_scores, baseline_prediction) or None.
    """
    original_pafe_tensor, drug_fp_tensor, original_genomic_vectors = feature_generator.generate_features(
        cell_id, drug_name, return_full_vectors=True
    )

    if original_pafe_tensor is None or drug_fp_tensor is None:
        return None

    original_pafe_tensor = original_pafe_tensor.to(device)
    drug_fp_tensor_batch = drug_fp_tensor.unsqueeze(0).to(device)

    if baseline_prediction is None:
        with torch.no_grad():
            baseline_prediction = model(original_pafe_tensor, global_data['edge_index'], drug_fp_tensor_batch).item()

    gene_to_pathways_map = {gene: [] for gene in genomic_lookups['G_universal']}
    pathway_to_idx_map = {pathway: i for i, pathway in enumerate(global_data['nodes'])}
    for pathway, genes in global_data['pathway_dict'].items():
        for gene in genes:
            if gene in gene_to_pathways_map:
                gene_to_pathways_map[gene].append(pathway)

    n_genes = feature_generator.n_genes_in_features
    cna_offset = n_genes
    rna_offset = 2 * n_genes
    gene_to_idx = {gene: i for i, gene in enumerate(genomic_lookups['G_universal'])}

    if primary_knockout_gene not in gene_to_idx:
        return None

    primary_gene_idx = gene_to_idx[primary_knockout_gene]

    primary_ko_pafe = original_pafe_tensor.clone()
    pathways_for_gene = gene_to_pathways_map.get(primary_knockout_gene, [])
    pathway_indices = [pathway_to_idx_map[p] for p in pathways_for_gene if p in pathway_to_idx_map]

    if pathway_indices:
        min_rna_val = gene_minimum_rna_map.get(primary_knockout_gene, 0)
        primary_ko_pafe[pathway_indices, primary_gene_idx] = 0
        primary_ko_pafe[pathway_indices, cna_offset + primary_gene_idx] = 1
        primary_ko_pafe[pathway_indices, rna_offset + primary_gene_idx] = min_rna_val

    with torch.no_grad():
        primary_ko_prediction = model(primary_ko_pafe, global_data['edge_index'], drug_fp_tensor_batch).item()

    primary_ko_impact = baseline_prediction - primary_ko_prediction

    synergy_scores = {}

    for secondary_gene_name in genomic_lookups['G_universal']:
        if secondary_gene_name == primary_knockout_gene:
            continue

        secondary_gene_idx = gene_to_idx[secondary_gene_name]

        secondary_ko_pafe = original_pafe_tensor.clone()

        pathways_for_gene = gene_to_pathways_map.get(secondary_gene_name, [])
        pathway_indices = [pathway_to_idx_map[p] for p in pathways_for_gene if p in pathway_to_idx_map]

        if pathway_indices:
            min_rna_val = gene_minimum_rna_map.get(secondary_gene_name, 0)
            secondary_ko_pafe[pathway_indices, secondary_gene_idx] = 0
            secondary_ko_pafe[pathway_indices, cna_offset + secondary_gene_idx] = 1
            secondary_ko_pafe[pathway_indices, rna_offset + secondary_gene_idx] = min_rna_val

        with torch.no_grad():
            secondary_ko_prediction = model(secondary_ko_pafe, global_data['edge_index'], drug_fp_tensor_batch).item()
        secondary_ko_impact = baseline_prediction - secondary_ko_prediction

        double_ko_pafe = primary_ko_pafe.clone()
        if pathway_indices:
            min_rna_val = gene_minimum_rna_map.get(secondary_gene_name, 0)
            double_ko_pafe[pathway_indices, secondary_gene_idx] = 0
            double_ko_pafe[pathway_indices, cna_offset + secondary_gene_idx] = 1
            double_ko_pafe[pathway_indices, rna_offset + secondary_gene_idx] = min_rna_val

        with torch.no_grad():
            double_ko_prediction = model(double_ko_pafe, global_data['edge_index'], drug_fp_tensor_batch).item()

        double_ko_impact = baseline_prediction - double_ko_prediction

        synergy_score = double_ko_impact - (primary_ko_impact + secondary_ko_impact)
        synergy_scores[secondary_gene_name] = synergy_score

    return pd.Series(synergy_scores, name=cell_id), baseline_prediction


def run_synergistic_knockout_for_pathway_sample(
    cell_id: str,
    drug_name: str,
    primary_knockout_pathway: str,
    feature_generator: FeatureGenerator,
    model: torch.nn.Module,
    global_data: Dict,
    device: torch.device,
    baseline_pathway_states: torch.Tensor,
    baseline_prediction: Optional[float] = None,
) -> Optional[Tuple[pd.Series, float]]:
    """Run synergistic knockout analysis for pathways.

    Performs primary knockout of specified pathway, then systematically knocks out
    every other pathway to calculate synergy scores.

    Args:
        cell_id: Cell line identifier.
        drug_name: Drug name.
        primary_knockout_pathway: Pathway to knock out first.
        feature_generator: Feature generator instance.
        model: Trained model.
        global_data: Dictionary with pathway graph data.
        device: Torch device.
        baseline_pathway_states: Baseline pathway states.
        baseline_prediction: Pre-calculated baseline prediction (optional).

    Returns:
        Tuple of (synergy_scores, baseline_prediction) or None.
    """
    original_pafe_tensor, drug_fp_tensor = feature_generator.generate_features(cell_id, drug_name)

    if original_pafe_tensor is None or drug_fp_tensor is None:
        return None

    original_pafe_tensor = original_pafe_tensor.to(device)
    drug_fp_tensor_batch = drug_fp_tensor.unsqueeze(0).to(device)

    if baseline_prediction is None:
        with torch.no_grad():
            baseline_prediction = model(original_pafe_tensor, global_data['edge_index'], drug_fp_tensor_batch).item()

    pathway_to_idx = {name: i for i, name in enumerate(global_data['nodes'])}

    if primary_knockout_pathway not in pathway_to_idx:
        return None

    primary_pathway_idx = pathway_to_idx[primary_knockout_pathway]

    primary_ko_pafe = original_pafe_tensor.clone()
    primary_ko_pafe[primary_pathway_idx, :] = baseline_pathway_states[primary_pathway_idx, :]

    with torch.no_grad():
        primary_ko_prediction = model(primary_ko_pafe, global_data['edge_index'], drug_fp_tensor_batch).item()

    primary_ko_impact = baseline_prediction - primary_ko_prediction

    synergy_scores = {}

    for secondary_pathway_name in global_data['nodes']:
        if secondary_pathway_name == primary_knockout_pathway:
            continue

        secondary_pathway_idx = pathway_to_idx[secondary_pathway_name]

        secondary_ko_pafe = original_pafe_tensor.clone()
        secondary_ko_pafe[secondary_pathway_idx, :] = baseline_pathway_states[secondary_pathway_idx, :]

        with torch.no_grad():
            secondary_ko_prediction = model(secondary_ko_pafe, global_data['edge_index'], drug_fp_tensor_batch).item()
        secondary_ko_impact = baseline_prediction - secondary_ko_prediction

        double_ko_pafe = primary_ko_pafe.clone()
        double_ko_pafe[secondary_pathway_idx, :] = baseline_pathway_states[secondary_pathway_idx, :]

        with torch.no_grad():
            double_ko_prediction = model(double_ko_pafe, global_data['edge_index'], drug_fp_tensor_batch).item()

        double_ko_impact = baseline_prediction - double_ko_prediction

        synergy_score = double_ko_impact - (primary_ko_impact + secondary_ko_impact)
        synergy_scores[secondary_pathway_name] = synergy_score

    return pd.Series(synergy_scores, name=cell_id), baseline_prediction


def run_scm_edge_importance_for_sample(
    cell_id: str,
    drug_name: str,
    feature_generator: FeatureGenerator,
    model: SCMModelWrapper,
    global_data: Dict,
    device: torch.device,
    go_map: Optional[Dict[str, str]] = None,
    baseline_prediction: Optional[float] = None,
) -> Optional[Tuple[pd.Series, float]]:
    """Run edge importance analysis for SCM models.

    For each edge (i→j), test importance by removing the edge while keeping
    everything else constant. This isolates edge-specific effects.

    Args:
        cell_id: Cell line identifier.
        drug_name: Drug name.
        feature_generator: Feature generator instance.
        model: SCM model wrapper.
        global_data: Dictionary with pathway names and edge index.
        device: Torch device.
        go_map: GO ID to pathway name mapping.
        baseline_prediction: Pre-calculated baseline prediction (optional).

    Returns:
        Tuple of (edge_importance_scores, baseline_prediction) or None.
    """
    original_pafe_tensor, drug_fp_tensor, original_genomic_vectors = feature_generator.generate_features(
        cell_id, drug_name, return_full_vectors=True
    )

    if original_pafe_tensor is None or drug_fp_tensor is None:
        return None

    original_pafe_tensor = original_pafe_tensor.to(device)
    drug_fp_tensor = drug_fp_tensor.to(device)
    drug_fp_tensor_batch = drug_fp_tensor.unsqueeze(0)

    if baseline_prediction is None:
        with torch.no_grad():
            baseline_prediction = model(
                original_pafe_tensor,
                global_data['edge_index'],
                drug_fp_tensor_batch
            ).item()

    edge_index = global_data['edge_index']
    pathway_names = global_data['nodes']

    edges = []
    if edge_index.numel() > 0:
        src_nodes = edge_index[0].cpu().numpy()
        dst_nodes = edge_index[1].cpu().numpy()
        for src_idx, dst_idx in zip(src_nodes, dst_nodes):
            if src_idx < len(pathway_names) and dst_idx < len(pathway_names):
                edges.append((int(src_idx), int(dst_idx)))

    edges = list(set(edges))

    if not edges:
        return None

    scm_model = model.scm
    parent_mask = scm_model.parent_mask
    edge_importance_scores = {}

    for source_idx, target_idx in edges:
        if source_idx == target_idx:
            continue

        if parent_mask[target_idx, source_idx].item() == 0:
            continue

        modified_parent_weights = scm_model.parent_weights.clone()
        original_parent_count = parent_mask[target_idx].sum().item()

        if original_parent_count > 1:
            modified_parent_weights[target_idx, source_idx] = 0.0
            remaining_parent_count = (parent_mask[target_idx].sum() - parent_mask[target_idx, source_idx]).item()

            if remaining_parent_count > 0:
                modified_parent_weights[target_idx] = modified_parent_weights[target_idx] / (
                    modified_parent_weights[target_idx].sum() + 1e-8
                )
            else:
                modified_parent_weights[target_idx] = 0.0

        original_parent_weights = scm_model.parent_weights.clone()
        scm_model.parent_weights = modified_parent_weights

        with torch.no_grad():
            edge_removed_prediction = model(
                original_pafe_tensor,
                global_data['edge_index'],
                drug_fp_tensor_batch
            ).item()

        scm_model.parent_weights = original_parent_weights

        edge_importance = baseline_prediction - edge_removed_prediction

        source_go_id = pathway_names[source_idx]
        target_go_id = pathway_names[target_idx]
        source_name = go_map.get(source_go_id, source_go_id) if go_map else source_go_id
        target_name = go_map.get(target_go_id, target_go_id) if go_map else target_go_id
        edge_name = f"{source_name} → {target_name}"

        edge_importance_scores[edge_name] = edge_importance

    if not edge_importance_scores:
        return None

    return pd.Series(edge_importance_scores, name=cell_id), baseline_prediction
