"""Input-space pathway knockout analysis.

Performs pathway knockout by setting gene expression features to minimum values,
mimicking CRISPR knockout in input space rather than latent space.
"""

from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_processing.feature_generator import FeatureGenerator


def run_input_space_pathway_knockout_for_sample(
    cell_id: str,
    drug_name: str,
    feature_generator: FeatureGenerator,
    model: torch.nn.Module,
    global_data: Dict,
    device: torch.device,
    genomic_lookups: Dict,
    gene_minimum_rna_map: Dict[str, float]
) -> Optional[Tuple[pd.Series, float]]:
    """Perform pathway knockout by zeroing genes in input space.

    Args:
        cell_id: Cell line identifier.
        drug_name: Drug name.
        feature_generator: Feature generator instance.
        model: SCM model.
        global_data: Dictionary with pathway graph data.
        device: Torch device.
        genomic_lookups: Gene index mappings.
        gene_minimum_rna_map: Minimum RNA expression per gene.

    Returns:
        Tuple of (importance_scores, baseline_prediction) or None if failed.
    """
    original_pafe_tensor, drug_fp_tensor = feature_generator.generate_features(
        cell_id, drug_name
    )

    if original_pafe_tensor is None or drug_fp_tensor is None:
        print(f"Could not generate features for {cell_id}, {drug_name}")
        return None

    original_pafe_tensor = original_pafe_tensor.to(device)
    drug_fp_tensor_batch = drug_fp_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        baseline_prediction = model(
            original_pafe_tensor,
            global_data['edge_index'],
            drug_fp_tensor_batch
        ).item()

    gene_to_idx = {gene: i for i, gene in enumerate(genomic_lookups['G_universal'])}
    n_genes = len(genomic_lookups['G_universal'])
    cna_offset = n_genes
    rna_offset = 2 * n_genes

    gene_to_pathway_indices = {}
    for p_idx, p_name in enumerate(global_data['nodes']):
        for gene_name in global_data['pathway_dict'].get(p_name, set()):
            if gene_name not in gene_to_pathway_indices:
                gene_to_pathway_indices[gene_name] = []
            gene_to_pathway_indices[gene_name].append(p_idx)

    importance_scores = {}

    for pathway_idx, pathway_name in enumerate(global_data['nodes']):
        genes_in_pathway = global_data['pathway_dict'].get(pathway_name, set())

        if not genes_in_pathway:
            importance_scores[pathway_name] = 0.0
            continue

        counterfactual_pafe = original_pafe_tensor.clone()
        genes_knocked_out = 0

        for gene_name in genes_in_pathway:
            if gene_name not in gene_to_idx:
                continue

            gene_idx = gene_to_idx[gene_name]
            min_rna = gene_minimum_rna_map.get(gene_name, 0)

            pathways_with_this_gene = gene_to_pathway_indices.get(gene_name, [])

            for p_idx in pathways_with_this_gene:
                counterfactual_pafe[p_idx, gene_idx] = 0
                counterfactual_pafe[p_idx, cna_offset + gene_idx] = 1
                counterfactual_pafe[p_idx, rna_offset + gene_idx] = min_rna

            genes_knocked_out += 1

        if genes_knocked_out == 0:
            importance_scores[pathway_name] = 0.0
            continue

        with torch.no_grad():
            counterfactual_prediction = model(
                counterfactual_pafe,
                global_data['edge_index'],
                drug_fp_tensor_batch
            ).item()

        importance_scores[pathway_name] = baseline_prediction - counterfactual_prediction

    return (pd.Series(importance_scores, name=cell_id), baseline_prediction)


def run_input_space_pathway_knockout_for_drug(
    drug_name: str,
    dataset_name: str,
    config: Dict,
    model: torch.nn.Module,
    global_data: Dict,
    master_smiles_map: Dict[str, str],
    npvae_embeddings_map: Dict,
    device: torch.device,
    genomic_lookups: Dict,
    gene_minimum_rna_map: Dict[str, float],
    disable_tqdm: bool = False
) -> Optional[Dict]:
    """Run pathway knockout for all samples of a drug.

    Args:
        drug_name: Drug to process.
        dataset_name: Dataset name.
        config: Configuration dictionary.
        model: SCM model instance.
        global_data: Global pathway graph data.
        master_smiles_map: Drug SMILES mapping.
        npvae_embeddings_map: Drug embeddings.
        device: Torch device.
        genomic_lookups: Gene mappings.
        gene_minimum_rna_map: Min RNA per gene.
        disable_tqdm: Whether to disable progress bar.

    Returns:
        Dictionary with results and statistics.
    """
    print(f"\nProcessing {drug_name} - {dataset_name}")
    print("Method: Input-Space Pathway Knockout")

    feature_generator = FeatureGenerator(
        cell_mutation_map=genomic_lookups['cell_mutation_map'],
        cell_cna_map=genomic_lookups['cell_cna_map'],
        cell_rna_map=genomic_lookups['cell_rna_map'],
        pathway_dict=global_data['pathway_dict'],
        ordered_pathway_names=global_data['nodes'],
        smiles_map=master_smiles_map,
        npvae_embeddings_map=npvae_embeddings_map,
        g_universal=genomic_lookups['G_universal']
    )

    rna_expression_file = config['rna_expression_file']
    rna_df = pd.read_csv(rna_expression_file, index_col=0)

    if dataset_name == 'gdsc0_true_test':
        response_dir = Path(config['gdsc0_drug_response_dir'])
    elif dataset_name == 'ctrpv2':
        response_dir = Path(config['ctrpv2_drug_response_dir'])
    else:
        print(f"Unknown dataset: {dataset_name}")
        return None

    response_file = response_dir / f"{drug_name}.csv"
    if not response_file.exists():
        print(f"Response file not found: {response_file}")
        return None

    response_df = pd.read_csv(response_file, index_col=0)

    if dataset_name.endswith('_test'):
        cell_lines = response_df[response_df['split'] == 'test'].index.tolist()
    else:
        cell_lines = response_df.index.tolist()

    cell_lines = [c for c in cell_lines if c in rna_df.index]
    print(f"Processing {len(cell_lines)} cell lines")

    all_importance_scores = []
    cell_to_predicted_aac = {}
    cell_to_actual_aac = {}

    iterator = tqdm(cell_lines, desc=f"{drug_name}", disable=disable_tqdm)

    for cell_id in iterator:
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
            importance_series, baseline_pred = result
            all_importance_scores.append(importance_series)
            cell_to_predicted_aac[cell_id] = baseline_pred

            if cell_id in response_df.index:
                cell_to_actual_aac[cell_id] = response_df.loc[cell_id, 'aac']

    if not all_importance_scores:
        print(f"No results for {drug_name}")
        return None

    all_scores_df = pd.concat(all_importance_scores, axis=1)

    print(f"\nResults:")
    print(f"  Pathways: {len(all_scores_df)}")
    print(f"  Samples: {len(all_scores_df.columns)}")

    summary_df = pd.DataFrame(index=all_scores_df.index)
    summary_df['pathway_name'] = all_scores_df.index
    summary_df['mean_importance'] = all_scores_df.mean(axis=1)
    summary_df['mean_abs_importance'] = all_scores_df.abs().mean(axis=1)
    summary_df['std_importance'] = all_scores_df.std(axis=1)
    summary_df['median_importance'] = all_scores_df.median(axis=1)
    summary_df['num_genes_in_pathway'] = [
        len(global_data['pathway_dict'].get(p, set()))
        for p in all_scores_df.index
    ]

    summary_df = summary_df.sort_values('mean_abs_importance', ascending=False)

    print(f"\nTop 10 pathways for {drug_name}:")
    for i, (idx, row) in enumerate(summary_df.head(10).iterrows(), 1):
        pathway_name = row['pathway_name']
        display_name = pathway_name if len(pathway_name) <= 50 else pathway_name[:47] + "..."
        print(f"  {i:2d}. {display_name:50s}  {row['mean_importance']:8.4f}  ({row['num_genes_in_pathway']} genes)")

    return {
        'drug_name': drug_name,
        'dataset_name': dataset_name,
        'summary_df': summary_df,
        'raw_scores_df': all_scores_df,
        'cell_to_predicted_aac': cell_to_predicted_aac,
        'cell_to_actual_aac': cell_to_actual_aac
    }
