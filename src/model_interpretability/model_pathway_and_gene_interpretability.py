"""Virtual knockout analysis for pathway and gene importance.

This module implements virtual knockout workflows to determine the causal
importance of biological pathways and genes to drug efficacy predictions.
"""

import os
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.scm_wrapper import load_scm_from_checkpoint
from src.data_processing.feature_generator import FeatureGenerator
from src.model_interpretability.plotting_interpretability import (
    create_heatmap,
    create_scatter_plot,
    create_quartile_box_plot,
    create_top_bottom_responder_box_plot,
    create_raincloud_plot,
    generate_final_summary_heatmaps
)
from src.model_interpretability.virtual_knockout import (
    run_virtual_gene_knockout_for_sample,
    run_virtual_double_knockout_for_sample,
    run_synergistic_knockout_for_gene_sample,
    run_synergistic_knockout_for_pathway_sample,
    generate_synthetic_baseline_state,
    run_scm_edge_importance_for_sample,
    run_virtual_knockout_for_sample_scm_pathway,
    identify_genes_in_enriched_pathways
)

config_path = "/home/charif/PIGE/PIGE/src/model_interpretability/configs/single_drug_model_pathway_knockout.yaml"

def get_drugs_to_process(config: Dict[str, Any]) -> List[str]:
    """Get list of drugs to process from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        List of drug names to process.
    """
    main_drugs = config.get('drugs_to_process')
    if main_drugs:
        return main_drugs

    drugs_file = config.get('smiles_file')
    if not drugs_file or not Path(drugs_file).exists():
        return []

    drugs = []
    with open(drugs_file, 'r') as f:
        next(f)
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                drugs.append(parts[0])

    return drugs[-1:]


def get_knockout_output_base_dir(config: Dict) -> Path:
    """Determine output directory for knockout analysis results.

    Args:
        config: Configuration dictionary containing output_dir and knockout settings.

    Returns:
        Path object pointing to the output directory.
    """
    output_dir = Path(config['output_dir'])
    knockout_mode = config.get('knockout_target', 'pathway')

    if 'primary_knockout_gene' in config or 'primary_knockout_pathway' in config:
        primary_target = config.get('primary_knockout_gene') or config.get('primary_knockout_pathway')
        return output_dir / f"Directed_{primary_target}_KO"

    subfolder_map = {
        'pathway': 'SinglePathwayKO',
        'double_pathway': 'DoublePathwayKO',
        'gene': 'SingleGeneKO',
        'double_gene': 'DoubleGeneKO',
    }

    subfolder = subfolder_map.get(knockout_mode, f'unsupported_knockout_mode_{knockout_mode}')
    return output_dir / subfolder


def load_go_term_names(gmt_file_path: str) -> Dict[str, str]:
    """Load GO term ID to name mappings from GMT file.

    Args:
        gmt_file_path: Path to GMT format file.

    Returns:
        Dictionary mapping GO IDs to GO names.
    """
    print(f"Loading GO term names from: {gmt_file_path}")

    go_map = {}
    with open(gmt_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                go_id = parts[0]
                go_name = parts[1]
                go_map[go_id] = go_name

    print(f"Loaded {len(go_map)} GO term mappings")
    return go_map


def _get_dataset_paths(dataset_name: str, drug_name: str, paths_config: Dict) -> Tuple[str, str]:
    """Get file paths for genomic lookups and drug response data.

    Args:
        dataset_name: Name of the dataset (e.g., 'ctrpv2', 'gdsc0_main').
        drug_name: Name of the drug.
        paths_config: Configuration dictionary with path information.

    Returns:
        Tuple of (genomic_lookups_path, response_file_path).
    """
    lookup_map = {
        'ctrpv2': ('ctrp', 'ctrpv2_drug_response_dir'),
        'gdsc0_main': ('gdsc0', 'gdsc0_drug_response_dir'),
        'gdsc0_true_test': ('gdsc0', 'gdsc0_drug_response_dir'),
    }

    lookup_prefix_base, response_dir_key = lookup_map[dataset_name]
    drug_name_safe = drug_name.replace('-', '_').replace(' ', '_')

    genomic_lookups_path = paths_config.get(
        'global_genomic_lookup_file',
        os.path.join(paths_config['genomic_lookups_dir'], lookup_prefix_base + f'_{drug_name_safe}.pkl')
    )

    response_path = os.path.join(paths_config[response_dir_key], f"{drug_name_safe}.csv")

    if 'true_test' in dataset_name:
        response_path = os.path.join(paths_config[response_dir_key], 'true_test', f"{drug_name_safe}.csv")

    return genomic_lookups_path, response_path


def process_drug_for_knockout(
    drug_name: str,
    config: Dict,
    go_map: Dict[str, str],
    model: torch.nn.Module,
    global_data: Dict,
    master_smiles_map: Dict[str, str],
    npvae_embeddings_map: Dict,
    device: torch.device,
    gene_minimum_rna_map: Dict[str, float],
    disable_tqdm: bool = False,
    summary_file_path: Optional[str] = None
) -> Optional[Dict]:
    """Run virtual knockout analysis for a single drug across datasets.

    Args:
        drug_name: Name of drug to analyze.
        config: Configuration dictionary.
        go_map: GO ID to name mapping.
        model: Trained PyTorch model.
        global_data: Dictionary with pathway graph data.
        master_smiles_map: Drug name to SMILES mapping.
        npvae_embeddings_map: Drug embeddings.
        device: PyTorch device.
        gene_minimum_rna_map: Gene minimum RNA expression values.
        disable_tqdm: Whether to disable progress bars.
        summary_file_path: Path for summary output file.

    Returns:
        Dictionary of summary results by dataset, or None if no results.
    """
    model.to(device)
    print(f"\n{'='*20} Processing Drug: {drug_name} {'='*20}")

    summaries_by_dataset = {}
    name_to_go_map = {v: k for k, v in go_map.items()}

    knockout_mode = config.get('knockout_target', 'pathway')
    is_double_knockout = knockout_mode.startswith('double_')
    is_gene_knockout = 'gene' in knockout_mode

    print(f"Mode: {knockout_mode.upper()}")

    if config.get('use_all_selected_cell_lines_with_zero_aac'):
        print("Processing in 'all' mode for selected cell lines")
        datasets_to_process = ['all_selected_cell_lines']
    else:
        datasets_to_process = config['datasets_to_process']

    for dataset_name in datasets_to_process:
        print(f"\nProcessing dataset: {dataset_name}")

        all_samples_importance_scores = []
        all_response_dfs = []
        cell_to_actual_aac = {}
        cell_to_predicted_aac = {}

        pairs_to_process = []
        df_single_ko_raw_T = pd.DataFrame()

        if config.get('load_raw_scores_from_dir'):
            raw_scores_dir = Path(config['load_raw_scores_from_dir'])
            if is_double_knockout and is_gene_knockout:
                raw_scores_path = raw_scores_dir / 'DoubleGeneKO' / dataset_name / 'raw_scores' / f"{drug_name}_{dataset_name}_knockout_raw_scores.csv"
            elif is_double_knockout and not is_gene_knockout:
                raw_scores_path = raw_scores_dir / 'DoublePathwayKO' / dataset_name / 'raw_scores' / f"{drug_name}_{dataset_name}_knockout_raw_scores.csv"
            elif is_gene_knockout and not is_double_knockout:
                raw_scores_path = raw_scores_dir / 'SingleGeneKO' / dataset_name / 'raw_scores' / f"{drug_name}_{dataset_name}_knockout_raw_scores.csv"
            elif not is_double_knockout and not is_gene_knockout:
                raw_scores_path = raw_scores_dir / 'SinglePathwayKO' / dataset_name / 'raw_scores' / f"{drug_name}_{dataset_name}_knockout_raw_scores.csv"

            if not raw_scores_path.exists():
                print(f"Raw scores file not found: {raw_scores_path}")
                continue

            print(f"Loading pre-computed scores from: {raw_scores_path}")
            transposed_df = pd.read_csv(raw_scores_path, index_col=0)
            transposed_df.index = transposed_df.index.map(str)
            transposed_df.index.name = 'cell_id'

            cell_to_actual_aac = transposed_df['actual_aac'].to_dict()
            cell_to_predicted_aac = transposed_df['predicted_aac'].to_dict()

            df_response = pd.DataFrame({
                'ModelID': list(cell_to_actual_aac.keys()),
                'aac': list(cell_to_actual_aac.values())
            })
            all_response_dfs.append(df_response)

            all_scores_df = transposed_df.drop(columns=['actual_aac', 'predicted_aac'], errors='ignore').T
            all_samples_importance_scores = [all_scores_df[col] for col in all_scores_df.columns]

        else:
            if config.get('use_all_selected_cell_lines_with_zero_aac'):
                pre_selected_cells = config.get('cell_lines_to_process', [])
                if not pre_selected_cells:
                    print("No cell lines provided for 'all' mode")
                    continue

                df_response = pd.DataFrame({'ModelID': pre_selected_cells, 'aac': 0.0})
                genomic_lookups_path = config.get('global_genomic_lookup_file')
            else:
                genomic_lookups_path, response_file = _get_dataset_paths(dataset_name, drug_name, config)

                if not all([os.path.exists(genomic_lookups_path), os.path.exists(response_file)]):
                    print(f"Missing lookup or response file for {drug_name} in {dataset_name}")
                    print(f"  Expected genomic lookups: {genomic_lookups_path} (exists: {os.path.exists(genomic_lookups_path)})")
                    print(f"  Expected response file: {response_file} (exists: {os.path.exists(response_file)})")
                    continue
                df_response = pd.read_csv(response_file)

            if is_double_knockout and 'primary_knockout_gene' not in config and 'primary_knockout_pathway' not in config:
                single_ko_run_config = config.copy()
                single_ko_run_config['knockout_target'] = 'gene' if is_gene_knockout else 'pathway'
                if 'primary_knockout_gene' in single_ko_run_config:
                    del single_ko_run_config['primary_knockout_gene']

                single_ko_base_dir = get_knockout_output_base_dir(single_ko_run_config)
                summary_path = single_ko_base_dir / dataset_name / 'summary' / f"{drug_name}_{dataset_name}_knockout_summary.csv"
                raw_scores_path = single_ko_base_dir / dataset_name / 'raw_scores' / f"{drug_name}_{dataset_name}_knockout_raw_scores.csv"

                if not summary_path.exists() or not raw_scores_path.exists():
                    print(f"Running single KO first for {drug_name}")
                    process_drug_for_knockout(
                        drug_name, single_ko_run_config, go_map, model, global_data,
                        master_smiles_map, npvae_embeddings_map, device,
                        gene_minimum_rna_map, summary_file_path=summary_file_path
                    )
                    print("Single KO complete, proceeding with double KO")

                df_single_ko_summary = pd.read_csv(summary_path, index_col=0)
                df_single_ko_raw = pd.read_csv(raw_scores_path, index_col=0)

            with open(genomic_lookups_path, 'rb') as f:
                genomic_lookups = pickle.load(f)

            feature_generator = FeatureGenerator(
                pathway_dict=global_data['pathway_dict'],
                ordered_pathway_names=global_data['nodes'],
                g_universal=genomic_lookups['G_universal'],
                cell_mutation_map=genomic_lookups['cell_mutation_map'],
                cell_cna_map=genomic_lookups['cell_cna_map'],
                cell_rna_map=genomic_lookups['cell_rna_map'],
                smiles_map=master_smiles_map,
                npvae_embeddings_map=npvae_embeddings_map
            )

            all_response_dfs.append(df_response)
            cell_to_actual_aac.update(pd.Series(df_response.aac.values, index=df_response.ModelID).to_dict())

            all_cell_lines_in_dataset_for_baseline = df_response['ModelID'].tolist()
            dataset_predicted_aac = {}
            print(f"Pre-calculating predicted AAC for {len(all_cell_lines_in_dataset_for_baseline)} samples")

            iterator = tqdm(all_cell_lines_in_dataset_for_baseline, desc=f"Preds for {dataset_name}", leave=False, disable=disable_tqdm)
            for cell_id in iterator:
                original_pafe_tensor, drug_fp_tensor = feature_generator.generate_features(cell_id, drug_name)
                if original_pafe_tensor is None or drug_fp_tensor is None:
                    continue

                original_pafe_tensor = original_pafe_tensor.to(device)
                drug_fp_tensor_batch = drug_fp_tensor.unsqueeze(0).to(device)

                with torch.no_grad():
                    prediction = model(original_pafe_tensor, global_data['edge_index'], drug_fp_tensor_batch).item()
                dataset_predicted_aac[cell_id] = prediction

            df_response['predicted_aac'] = df_response['ModelID'].map(dataset_predicted_aac)
            df_response.dropna(subset=['predicted_aac'], inplace=True)

            baseline_pathway_states = generate_synthetic_baseline_state(feature_generator, device, gene_minimum_rna_map)

            all_cell_lines_in_dataset = df_response['ModelID'].tolist()
            pre_selected_cells = config.get('cell_lines_to_process')

            if pre_selected_cells:
                print(f"Using {len(pre_selected_cells)} pre-selected cell lines")
                cell_lines_to_process = [cell for cell in all_cell_lines_in_dataset if cell in pre_selected_cells]
                print(f"Found {len(cell_lines_to_process)} in dataset '{dataset_name}'")
            elif is_double_knockout and 'primary_knockout_gene' not in config and 'primary_knockout_pathway' not in config:
                print("Double KO: selecting from single KO results")
                available_cells_df = df_single_ko_raw.copy()
                all_cell_lines_from_single_ko = available_cells_df.index.tolist()

                top_bottom_n = config.get('use_top_bottom_n_cell_lines')
                num_to_sample = config.get('sample_n_cell_lines')

                if top_bottom_n and isinstance(top_bottom_n, int):
                    print(f"Selecting top/bottom {top_bottom_n} cell lines")
                    if len(available_cells_df) < 2 * top_bottom_n:
                        cell_lines_to_process = all_cell_lines_from_single_ko
                    else:
                        df_sorted = available_cells_df.sort_values(by='actual_aac', ascending=False)
                        top_n_cells = df_sorted.head(top_bottom_n).index.tolist()
                        bottom_n_cells = df_sorted.tail(top_bottom_n).index.tolist()
                        cell_lines_to_process = list(set(top_n_cells + bottom_n_cells))
                elif num_to_sample and num_to_sample < len(all_cell_lines_from_single_ko):
                    print(f"Sampling {num_to_sample} random cell lines")
                    import random
                    random.seed(42)
                    cell_lines_to_process = random.sample(all_cell_lines_from_single_ko, num_to_sample)
                else:
                    cell_lines_to_process = all_cell_lines_from_single_ko
            else:
                top_bottom_n = config.get('use_top_bottom_n_cell_lines')
                num_to_sample = config.get('sample_n_cell_lines')

                if top_bottom_n and isinstance(top_bottom_n, int):
                    print(f"Selecting top/bottom {top_bottom_n} cell lines by AAC")
                    if len(df_response) < 2 * top_bottom_n:
                        cell_lines_to_process = df_response['ModelID'].tolist()
                    else:
                        df_sorted = df_response.sort_values(by='aac', ascending=False)
                        top_n_cells = df_sorted.head(top_bottom_n)['ModelID'].tolist()
                        bottom_n_cells = df_sorted.tail(top_bottom_n)['ModelID'].tolist()
                        cell_lines_to_process = list(set(top_n_cells + bottom_n_cells))
                elif num_to_sample and num_to_sample < len(all_cell_lines_in_dataset):
                    print(f"Sampling {num_to_sample} random cell lines")
                    import random
                    random.seed(42)
                    cell_lines_to_process = random.sample(all_cell_lines_in_dataset, num_to_sample)
                else:
                    cell_lines_to_process = all_cell_lines_in_dataset

            print(f"Running knockout on {len(cell_lines_to_process)} samples")

            cell_baseline_predictions = {cell_id: dataset_predicted_aac[cell_id] for cell_id in cell_lines_to_process if cell_id in dataset_predicted_aac}

            if is_double_knockout and 'primary_knockout_gene' not in config and 'primary_knockout_pathway' not in config:
                top_25_entities = df_single_ko_summary.sort_values(by='mean_importance', ascending=False).head(25).index.tolist()
                bottom_25_entities = df_single_ko_summary.sort_values(by='mean_importance', ascending=True).head(25).index.tolist()
                all_pair_entities = top_25_entities + bottom_25_entities

                from itertools import product
                pairs_to_process = list(product(all_pair_entities, all_pair_entities))
                pairs_to_process = [tuple(sorted(p)) for p in pairs_to_process if p[0] != p[1]]
                pairs_to_process = sorted(list(set(pairs_to_process)))

                print(f"Generated {len(pairs_to_process)} pairs for double knockout")

                df_single_ko_raw_T = df_single_ko_raw.drop(columns=['predicted_aac', 'actual_aac'], errors='ignore').T
                if not is_gene_knockout:
                    name_to_go = {v: k for k, v in go_map.items()}
                    df_single_ko_raw_T.rename(index=name_to_go, inplace=True)

            iterator = tqdm(cell_lines_to_process, desc=f"Knockout for {dataset_name}", leave=False, disable=disable_tqdm)
            for cell_id in iterator:
                cell_baseline_pred = cell_baseline_predictions.get(cell_id)
                if cell_baseline_pred is None:
                    continue

                if is_double_knockout:
                    if not is_gene_knockout:
                        knockout_results = run_scm_edge_importance_for_sample(
                            cell_id, drug_name, feature_generator, model, global_data, device,
                            go_map=go_map,
                            baseline_prediction=cell_baseline_pred
                        )
                    elif 'primary_knockout_gene' in config:
                        knockout_results = run_synergistic_knockout_for_gene_sample(
                            cell_id, drug_name, config['primary_knockout_gene'],
                            feature_generator, model, global_data, device,
                            genomic_lookups, gene_minimum_rna_map,
                            baseline_prediction=cell_baseline_pred
                        )
                    elif 'primary_knockout_pathway' in config:
                        primary_pathway_name = config['primary_knockout_pathway']
                        primary_pathway_id = name_to_go_map.get(primary_pathway_name)

                        if not primary_pathway_id:
                            print(f"Could not find GO ID for '{primary_pathway_name}'")
                            continue

                        knockout_results = run_synergistic_knockout_for_pathway_sample(
                            cell_id, drug_name, primary_pathway_id,
                            feature_generator, model, global_data, device,
                            baseline_pathway_states,
                            baseline_prediction=cell_baseline_pred
                        )
                    else:
                        if cell_id not in df_single_ko_raw_T.columns:
                            continue

                        knockout_results = run_virtual_double_knockout_for_sample(
                            cell_id, drug_name, feature_generator, model, global_data, device,
                            pairs_to_process=pairs_to_process,
                            single_ko_scores_for_sample=df_single_ko_raw_T[cell_id],
                            baseline_pathway_states=baseline_pathway_states if not is_gene_knockout else None,
                            genomic_lookups=genomic_lookups if is_gene_knockout else None,
                            gene_minimum_rna_map=gene_minimum_rna_map if is_gene_knockout else None,
                            go_map=go_map,
                            baseline_prediction=cell_baseline_pred
                        )
                elif is_gene_knockout:
                    genes_to_knockout = None
                    if config.get('use_pathway_enrichment_filtering') and config.get('pathway_ko_results_dir'):
                        genes_to_knockout = identify_genes_in_enriched_pathways(
                            cell_id=cell_id,
                            drug_name=drug_name,
                            pathway_ko_results_dir=config['pathway_ko_results_dir'],
                            global_data=global_data,
                            go_map=go_map,
                            top_n_pathways=config.get('top_n_pathways_for_filtering', 50),
                            min_pathways_per_gene=config.get('min_pathways_per_gene', 2),
                        )
                        if genes_to_knockout is None:
                            print(f"  Warning: Filtering failed for {cell_id}, using all genes")

                    knockout_results = run_virtual_gene_knockout_for_sample(
                        cell_id, drug_name, feature_generator, model, global_data, device,
                        genomic_lookups=genomic_lookups,
                        gene_minimum_rna_map=gene_minimum_rna_map,
                        baseline_prediction=cell_baseline_pred,
                        genes_to_knockout=genes_to_knockout
                    )
                else:
                    knockout_results_scm = run_virtual_knockout_for_sample_scm_pathway(
                        cell_id, drug_name, feature_generator, model, global_data, device,
                        genomic_lookups=genomic_lookups,
                        gene_minimum_rna_map=gene_minimum_rna_map,
                    )

                    if knockout_results_scm is not None:
                        importance_series, direct_series, indirect_series, baseline_pred = knockout_results_scm
                        knockout_results = (importance_series, baseline_pred)
                    else:
                        knockout_results = None

                if knockout_results is not None:
                    importance_series, baseline_pred = knockout_results
                    all_samples_importance_scores.append(importance_series)
                    cell_to_predicted_aac[cell_id] = baseline_pred

        if not all_samples_importance_scores:
            print(f"No scores calculated for {drug_name} in {dataset_name}")
            continue

        all_scores_flat = pd.concat(all_samples_importance_scores, axis=1).values.flatten()
        drug_empirical_threshold_95 = np.percentile(np.abs(all_scores_flat), 95)
        print(f"Drug-specific 95th percentile threshold: {drug_empirical_threshold_95:.4f}")

        base_output_dir = get_knockout_output_base_dir(config)
        dataset_output_dir = base_output_dir / dataset_name

        summary_dir = dataset_output_dir / 'summary'
        raw_scores_dir = dataset_output_dir / 'raw_scores'
        plots_dir = dataset_output_dir / 'plots'
        summary_dir.mkdir(parents=True, exist_ok=True)
        raw_scores_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        all_scores_df = pd.concat(all_samples_importance_scores, axis=1)

        summary_df = pd.DataFrame(index=all_scores_df.index)
        summary_df['mean_importance'] = all_scores_df.mean(axis=1)
        summary_df['mean_abs_importance'] = all_scores_df.abs().mean(axis=1)

        actual_aac_for_processed_cells = pd.Series(cell_to_actual_aac).reindex(all_scores_df.columns)
        sorted_cells = actual_aac_for_processed_cells.sort_values(ascending=False).index
        num_cells = len(sorted_cells)
        split_size = num_cells // 2

        if split_size > 0:
            top_cells = sorted_cells[:split_size]
            bottom_cells = sorted_cells[-split_size:]

            mean_top_scores = all_scores_df[top_cells].mean(axis=1)
            mean_bottom_scores = all_scores_df[bottom_cells].mean(axis=1)

            summary_df['differential_importance'] = mean_top_scores - mean_bottom_scores
        else:
            summary_df['differential_importance'] = np.nan

        actual_aac_series = pd.Series(cell_to_actual_aac).reindex(all_scores_df.columns)

        spearman_correlations_actual = {}

        for entity_id, scores in all_scores_df.iterrows():
            combined = pd.concat([scores.rename('scores'), actual_aac_series.rename('actual_aac')], axis=1).dropna()

            if len(combined) > 2:
                corr, _ = spearmanr(combined['scores'], combined['actual_aac'])
                spearman_correlations_actual[entity_id] = corr
            else:
                spearman_correlations_actual[entity_id] = np.nan

        summary_df['spearman_corr_vs_actual_aac'] = pd.Series(spearman_correlations_actual)

        summary_by_mean_importance = summary_df.sort_values(by='mean_importance', ascending=False)
        pathways_by_mean_importance = summary_by_mean_importance.index.tolist()

        summary_by_abs_importance = summary_df.sort_values(by='mean_abs_importance', ascending=False)
        pathways_by_abs_importance = summary_by_abs_importance.index.tolist()

        summary_by_diff_importance = summary_df.sort_values(by='differential_importance', ascending=False, na_position='last')
        pathways_by_diff_importance = summary_by_diff_importance.index.tolist()

        summary_by_spearman_corr_actual = summary_df.sort_values(by='spearman_corr_vs_actual_aac', ascending=False, na_position='last')
        pathways_by_spearman_corr_actual = summary_by_spearman_corr_actual.index.tolist()

        summaries_by_dataset[f"{drug_name}_{dataset_name}"] = {
            'mean_importance': summary_df.get('mean_importance'),
            'mean_abs_importance': summary_df.get('mean_abs_importance'),
            'differential_importance': summary_df.get('differential_importance'),
            'spearman_corr_vs_actual_aac': summary_df.get('spearman_corr_vs_actual_aac')
        }
        summary_to_save = summary_by_diff_importance.copy()
        if ('primary_knockout_pathway' in config) or (not is_gene_knockout and not is_double_knockout):
            summary_to_save['pathway_name'] = summary_to_save.index.map(go_map)
        elif is_double_knockout and not is_gene_knockout:
            summary_to_save['pathway_name'] = summary_to_save.index
        else:
            summary_to_save['pathway_name'] = summary_to_save.index

        summary_cols = [
            'pathway_name', 'mean_importance', 'mean_abs_importance',
            'spearman_corr_vs_actual_aac',
            'differential_importance'
        ]

        final_summary_cols = [col for col in summary_cols if col in summary_to_save.columns]
        summary_to_save = summary_to_save[final_summary_cols]

        summary_output_file = summary_dir / f"{drug_name}_{dataset_name}_knockout_summary.csv"
        summary_to_save.to_csv(summary_output_file, index=True)
        print(f"Summary saved: {summary_output_file}")

        raw_scores_df = pd.concat(all_samples_importance_scores, axis=1).fillna(0)

        raw_scores_df.loc['actual_aac'] = pd.Series(cell_to_actual_aac)
        raw_scores_df.loc['predicted_aac'] = pd.Series(cell_to_predicted_aac)

        transposed_df = raw_scores_df.T

        if ('primary_knockout_pathway' in config) or (not is_gene_knockout and not is_double_knockout):
            transposed_df.rename(columns=go_map, inplace=True)
        elif not is_gene_knockout and is_double_knockout:
            mapper = {
                col: f"{go_map.get(col.split(' <-> ')[0], col.split(' <-> ')[0])} <-> {go_map.get(col.split(' <-> ')[1], col.split(' <-> ')[1])}"
                for col in transposed_df.columns if isinstance(col, str) and ' <-> ' in col
            }
            transposed_df.rename(columns=mapper, inplace=True)

        lead_cols = ['predicted_aac', 'actual_aac']
        existing_lead_cols = [col for col in lead_cols if col in transposed_df.columns]

        entity_cols = sorted([col for col in transposed_df.columns if col not in existing_lead_cols])
        new_column_order = existing_lead_cols + entity_cols
        transposed_df = transposed_df[new_column_order]

        if 'actual_aac' in transposed_df.columns:
            transposed_df = transposed_df.sort_values(by='actual_aac', ascending=False)

        raw_output_file = raw_scores_dir / f"{drug_name}_{dataset_name}_knockout_raw_scores.csv"
        transposed_df.to_csv(raw_output_file, index_label='cell_id')
        print(f"Raw scores saved: {raw_output_file}")

        print(f"\nGenerating heatmaps for {drug_name} in {dataset_name}")
        master_response_df = pd.concat(all_response_dfs).drop_duplicates(subset=['ModelID'])

        cell_line_name_map = None
        if config.get('model_file_path'):
            df_model = pd.read_csv(config['model_file_path'])
            cell_line_name_map = pd.Series(df_model.CellLineName.values, index=df_model.ModelID).to_dict()
            print("Loaded cell line name map")

        sort_by_col = 'actual_aac'
        sort_by_label = 'Measured AAC'
        print("Using Measured AAC for sorting")

        transposed_df = transposed_df.sort_values(by=sort_by_col, ascending=False)

        aac_series = master_response_df.set_index('ModelID')['aac']
        predicted_aac_series = pd.Series(cell_to_predicted_aac)

        color_bars_to_concat = []

        if (aac_series != 0).any():
            print("Including Measured AAC in heatmap color bar")
            aac_norm = plt.Normalize(vmin=aac_series.min(), vmax=aac_series.max())
            aac_cmap = sns.color_palette('Greens', as_cmap=True)
            actual_aac_colors = aac_series.map(lambda x: aac_cmap(aac_norm(x)))
            actual_aac_colors.name = 'Measured AAC'
            color_bars_to_concat.append(actual_aac_colors)
        else:
            print("Skipping Measured AAC in color bar")

        if not predicted_aac_series.empty:
            pred_aac_norm = plt.Normalize(vmin=predicted_aac_series.min(), vmax=predicted_aac_series.max())
            pred_aac_cmap = sns.color_palette('Greens', as_cmap=True)
            predicted_aac_colors = predicted_aac_series.map(lambda x: pred_aac_cmap(pred_aac_norm(x)))
            predicted_aac_colors.name = 'Predicted AAC'
            color_bars_to_concat.append(predicted_aac_colors)

        if color_bars_to_concat:
            combined_colors = pd.concat(color_bars_to_concat, axis=1)
        else:
            combined_colors = None

        sorting_strategies = {
            'by_mean_importance': {
                'pathways': pathways_by_mean_importance[:50],
                'output_dir': 'heatmaps_by_mean_importance',
                'y_label': f"Top 50 {'Pathways' if not is_gene_knockout else 'Genes'} (Sorted by Mean Importance)"
            },
            'by_bottom_mean_importance': {
                'pathways': list(reversed(pathways_by_mean_importance[-50:])),
                'output_dir': 'heatmaps_by_bottom_mean_importance',
                'y_label': f"Bottom 50 {'Pathways' if not is_gene_knockout else 'Genes'} (Sorted by Mean Importance)"
            },
            'by_mean_abs_importance': {
                'pathways': pathways_by_abs_importance[:50],
                'output_dir': 'heatmaps_by_mean_abs_importance',
                'y_label': f"Top 50 {'Pathways' if not is_gene_knockout else 'Genes'} (Sorted by Mean Absolute Importance)"
            },
            'by_differential_importance': {
                'pathways': pathways_by_diff_importance[:50],
                'output_dir': 'heatmaps_by_differential_importance',
                'y_label': f"Top 50 {'Pathways' if not is_gene_knockout else 'Genes'} (Sorted by Differential Importance)"
            },
            'by_spearman_corr_actual': {
                'pathways': pathways_by_spearman_corr_actual[:50],
                'output_dir': 'heatmaps_by_spearman_corr_actual',
                'y_label': f"Top 50 {'Pathways' if not is_gene_knockout else 'Genes'} (Sorted by Spearman Corr w/ Actual AAC)"
            }
        }

        masking_strategies = {
            'no_mask': {'subdir': '01_no_mask'}
        }

        for sort_key, sort_config in sorting_strategies.items():
            print(f"Generating heatmaps sorted {sort_key}")
            sort_output_dir = plots_dir / sort_config['output_dir']
            top_pathways = sort_config['pathways']
            y_label = sort_config['y_label']

            if not top_pathways:
                print(f"Skipping {sort_key} - no pathways")
                continue

            all_scores_top = all_scores_df.loc[top_pathways]
            sorter_series = transposed_df[sort_by_col]

            sorted_all_cells = sorter_series.index.tolist()

            sample_groups = {
                'all_samples': {'ids': sorted_all_cells, 'title': 'All Samples'},
            }

            for mask_key, mask_config in masking_strategies.items():
                mask_output_dir = sort_output_dir / mask_config['subdir']
                mask_output_dir.mkdir(parents=True, exist_ok=True)

                for group_name, group_info in sample_groups.items():
                    cell_ids = group_info['ids']
                    group_title = group_info['title']

                    valid_cell_ids = [c for c in cell_ids if c in all_scores_top.columns]
                    if not valid_cell_ids:
                        continue

                    sorted_cell_ids = sorter_series.reindex(valid_cell_ids).sort_values(ascending=False).index

                    group_data = all_scores_top[sorted_cell_ids]
                    col_colors_group = combined_colors.reindex(group_data.columns) if combined_colors is not None else None

                    create_heatmap(
                        group_data,
                        f"Knockout Importance for {drug_name} ({dataset_name} - {group_title})",
                        mask_output_dir / f"{drug_name}_{dataset_name}_{group_name}_heatmap.png",
                        go_map if not is_gene_knockout else None,
                        col_colors=col_colors_group,
                        y_label=y_label,
                        cell_line_name_map=cell_line_name_map,
                        x_axis_label=f"Cell Lines (Sorted by {sort_by_label})",
                        actual_aac=aac_series,
                        predicted_aac=predicted_aac_series,
                        summary_file_path=summary_file_path
                    )

        is_gene_knockout = config.get('knockout_target') == 'gene'
        name_to_go_map = {v: k for k, v in go_map.items()} if not is_gene_knockout else None

        plot_entity_groups = {}

        relevant_entities = config.get('target_pathways_by_drug', {}).get(drug_name)
        if relevant_entities:
            plot_entity_groups['relevant'] = {
                'entities': relevant_entities,
                'subdir': 'relevant'
            }

        top_entities_ids = summary_by_diff_importance.head(3).index.tolist()
        if top_entities_ids:
            top_entities_names = [go_map.get(eid, eid) if not is_gene_knockout else eid for eid in top_entities_ids]
            plot_entity_groups['predicted'] = {
                'entities': top_entities_names,
                'subdir': 'predicted'
            }

        if plot_entity_groups:
            print(f"\nGenerating detailed plots for {drug_name} in {dataset_name}")
            original_aac_series = pd.Series(cell_to_predicted_aac)

            if 'actual_aac' in transposed_df.columns and transposed_df['actual_aac'].nunique() > 1:
                entity_cols = all_scores_df.index.tolist()

                correlations = {}
                for entity_id in entity_cols:
                    entity_name = go_map.get(entity_id, entity_id) if not is_gene_knockout else entity_id

                    if entity_name in transposed_df.columns:
                        temp_df = transposed_df[[entity_name, 'actual_aac']].dropna()
                        if len(temp_df) > 2:
                            corr, _ = spearmanr(temp_df[entity_name], temp_df['actual_aac'])
                            if not np.isnan(corr):
                                correlations[entity_name] = corr

                if correlations:
                    top_3_entities = sorted(correlations, key=lambda k: abs(correlations[k]), reverse=True)[:3]

                    top_corr_dir = plots_dir / 'scatter_plots' / 'top_corr'
                    top_corr_dir.mkdir(parents=True, exist_ok=True)

                    for entity_name in top_3_entities:
                        safe_entity_name = entity_name.replace(' ', '_').replace('/', '_')
                        safe_drug_name = drug_name.replace(' ', '_').replace('/', '_')

                        scatter_output_path = top_corr_dir / f"{safe_drug_name}_{safe_entity_name}_scatter.png"

                        create_scatter_plot(
                            data_df=transposed_df,
                            drug_name=drug_name,
                            target_pathway_name=entity_name,
                            output_path=scatter_output_path,
                            x_col='actual_aac',
                            x_label='Measured AAC',
                            summary_file_path=summary_file_path
                        )
                else:
                    print("No valid correlations for top correlated plots")
            else:
                print("No actual AAC data for top correlated plots")

            for group_name, group_info in plot_entity_groups.items():
                print(f"Generating plots for '{group_name}' entities")
                subdir = group_info['subdir']

                scatter_dir = plots_dir / 'scatter_plots' / subdir
                scatter_dir.mkdir(parents=True, exist_ok=True)

                box_plot_dir_20 = plots_dir / 'box_plots_stratified_aac_20' / subdir
                box_plot_dir_20.mkdir(parents=True, exist_ok=True)

                box_plot_dir_10 = plots_dir / 'box_plots_stratified_aac_10' / subdir
                box_plot_dir_10.mkdir(parents=True, exist_ok=True)

                quartile_box_plot_dir = plots_dir / 'box_plots_quartile' / subdir
                quartile_box_plot_dir.mkdir(parents=True, exist_ok=True)

                raincloud_dir = plots_dir / 'raincloud_plots' / subdir
                raincloud_dir.mkdir(parents=True, exist_ok=True)

                top_bottom_box_plot_dir = plots_dir / 'box_plots_top_bottom_10' / subdir
                top_bottom_box_plot_dir.mkdir(parents=True, exist_ok=True)

                for entity_name in group_info['entities']:
                    safe_entity_name = entity_name.replace(' ', '_').replace('/', '_')
                    safe_drug_name = drug_name.replace(' ', '_').replace('/', '_')

                    create_scatter_plot(
                        transposed_df, drug_name, entity_name,
                        scatter_dir / f"{safe_drug_name}_{safe_entity_name}_scatter.png",
                        x_col=sort_by_col, x_label=sort_by_label,
                        summary_file_path=summary_file_path
                    )

                    create_quartile_box_plot(
                        transposed_df, drug_name, entity_name,
                        quartile_box_plot_dir / f"{safe_drug_name}_{safe_entity_name}_quartile_boxplot.png",
                        sensitivity_col=sort_by_col, sensitivity_label=sort_by_label,
                        summary_file_path=summary_file_path
                    )

                    create_top_bottom_responder_box_plot(
                        transposed_df, drug_name, entity_name,
                        top_bottom_box_plot_dir / f"{safe_drug_name}_{safe_entity_name}_top_bottom_10_boxplot.png",
                        sensitivity_col=sort_by_col, sensitivity_label=sort_by_label,
                        summary_file_path=summary_file_path
                    )

                    is_double_ko = config.get('knockout_target', '').startswith('double_')

                    if is_double_ko:
                        entity_id = entity_name
                    else:
                        entity_id = entity_name if is_gene_knockout else name_to_go_map.get(entity_name)

                    if entity_id and entity_id in all_scores_df.index:
                        importance_scores_for_entity = all_scores_df.loc[entity_id]
                        knockout_aac_series = original_aac_series - importance_scores_for_entity

                        raincloud_output_path = raincloud_dir / f"{safe_drug_name}_{safe_entity_name}_raincloud.png"
                        create_raincloud_plot(
                            drug_name,
                            entity_name,
                            original_aac_series,
                            knockout_aac_series,
                            raincloud_output_path,
                            summary_file_path,
                        )
                    else:
                        print(f"Could not find scores for '{entity_name}' to generate raincloud plot")

    if not summaries_by_dataset:
        print(f"No scores calculated for {drug_name} across any dataset")
        return None

    return summaries_by_dataset


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_model_path_for_drug(model_path: str, drug_name: str) -> Optional[str]:
    """Get the model path for a specific drug.

    Args:
        model_path: Path to model (file for multi-drug, directory for per-drug).
        drug_name: Name of the drug.

    Returns:
        Path to the model file, or None if not found.
    """
    model_path = Path(model_path)

    if model_path.is_file():
        return str(model_path)

    if model_path.is_dir():
        drug_dir = model_path / drug_name

        if not drug_dir.exists():
            print(f"Drug directory not found: {drug_dir}")
            return None

        search_paths = []
        search_paths.append(drug_dir)

        for subdir in drug_dir.iterdir():
            if subdir.is_dir() and subdir.name.startswith(drug_name):
                search_paths.append(subdir)

        for search_dir in search_paths:
            if search_dir.exists() and search_dir.is_dir():
                for file in search_dir.iterdir():
                    if file.suffix == '.pth' and 'selected' in file.name:
                        return str(file)

        print(f"No model found for drug '{drug_name}' in {drug_dir}")
        return None

    print(f"Model path does not exist: {model_path}")
    return None


def main(config: Dict, drugs_to_process: List[str]) -> None:
    """Run virtual knockout analysis pipeline.

    Args:
        config: Configuration dictionary with all settings and paths.
        drugs_to_process: List of drug names to analyze.
    """
    knockout_target = config.get('knockout_target', 'pathway')
    summary_file_path = str(get_knockout_output_base_dir(config) / 'run_summary.txt')

    if 'primary_knockout_gene' in config:
        print(f"Mode: SYNERGISTIC {knockout_target.upper()} KNOCKOUT (Primary: {config['primary_knockout_gene']})")
    elif 'primary_knockout_pathway' in config:
        print(f"Mode: SYNERGISTIC {knockout_target.upper()} KNOCKOUT (Primary: {config['primary_knockout_pathway']})")
    elif knockout_target == 'double_gene':
        print("Mode: DOUBLE GENE KNOCKOUT")
    elif knockout_target == 'double_pathway':
        print("Mode: DOUBLE PATHWAY KNOCKOUT")
    else:
        print(f"Mode: {knockout_target.upper()} KNOCKOUT")

    with open(config['relevant_entities_file'], 'r') as f:
        import json
        config['target_pathways_by_drug'] = json.load(f)
    print(f"Loaded relevant entities from: {config['relevant_entities_file']}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    go_map = load_go_term_names(config['gmt_file'])

    with open(config['pathway_interaction_graph_file'], 'rb') as f:
        graph_data = pickle.load(f)

    import ast
    edge_list = ast.literal_eval(graph_data['edge_list']) if isinstance(graph_data['edge_list'], str) else graph_data['edge_list']

    nodes = graph_data['nodes']
    node_to_idx = {node_id: i for i, node_id in enumerate(nodes)}
    edge_index = torch.tensor([[node_to_idx.get(u), node_to_idx.get(v)] for u, v in edge_list if u in node_to_idx and v in node_to_idx], dtype=torch.long).t().contiguous().to(device)

    global_data = {
        'nodes': nodes,
        'edge_index': edge_index,
        'pathway_dict': graph_data['node_to_genes']
    }

    df_drugs = pd.read_csv(config['smiles_file'])
    master_smiles_map = {}
    for _, row in df_drugs.iterrows():
        drug_name = str(row['drug_name'])
        smiles = str(row['smiles'])
        master_smiles_map[drug_name] = smiles

    drug_name_mapping = config.get('drug_name_mapping', {})
    for response_name, smiles_name in drug_name_mapping.items():
        if smiles_name in master_smiles_map and response_name not in master_smiles_map:
            master_smiles_map[response_name] = master_smiles_map[smiles_name]

    with open(config['npvae_embeddings_file'], 'rb') as f:
        npvae_embeddings_map = pickle.load(f)

    is_per_drug_mode = Path(config['model_path']).is_dir()
    if is_per_drug_mode:
        print(f"Per-drug mode: Model path is a directory - {config['model_path']}")
    else:
        print(f"Multi-drug mode: Model path is a file - {config['model_path']}")
        checkpoint = torch.load(config['model_path'], map_location=device, weights_only=False)
        model_hps = checkpoint['hyperparameters']
        model = None

    print("Global data loaded")

    rna_map_cache_file = config.get('rna_map_cache_file', None)
    if rna_map_cache_file is None:
        rna_map_cache_file = os.path.splitext(config['rna_expression_file'])[0] + '.min_map.pkl'

    if os.path.isfile(rna_map_cache_file):
        with open(rna_map_cache_file, 'rb') as cache_f:
            gene_minimum_rna_map = pickle.load(cache_f)
        print(f"Loaded gene minimum RNA map from cache with {len(gene_minimum_rna_map)} entries")
    else:
        print("Computing gene minimum RNA map from expression data")
        df_rna = pd.read_csv(config['rna_expression_file'])
        gene_cols = [col for col in df_rna.columns if col != df_rna.columns[0]]

        if any('(' in col and ')' in col for col in gene_cols):
            parsed_gene_map = {re.match(r'(.+?) \(', col).group(1): col for col in gene_cols if re.match(r'(.+?) \(', col)}
        else:
            parsed_gene_map = {col: col for col in gene_cols}

        min_rna_series = df_rna[gene_cols].min()
        gene_minimum_rna_map = {parsed_name: min_rna_series[orig_name] for parsed_name, orig_name in parsed_gene_map.items()}

        with open(rna_map_cache_file, 'wb') as cache_f:
            pickle.dump(gene_minimum_rna_map, cache_f)

        print(f"Calculated and cached minimum expression for {len(gene_minimum_rna_map)} genes")

    all_drug_summaries = {}
    for drug_name in drugs_to_process:
        if is_per_drug_mode:
            drug_model_path = get_model_path_for_drug(config['model_path'], drug_name)
            if drug_model_path is None:
                print(f"Skipping {drug_name} - no model found")
                continue

            print(f"\nLoading per-drug model for {drug_name}: {drug_model_path}")
            checkpoint = torch.load(drug_model_path, map_location=device, weights_only=False)
            model_hps = checkpoint['hyperparameters']

            model = load_scm_from_checkpoint(
                checkpoint_path=drug_model_path,
                pathway_dict=global_data['pathway_dict'],
                ordered_pathway_names=global_data['nodes'],
                pathway_graph_pickle_path=config['pathway_interaction_graph_file'],
                pafe_feature_dim=model_hps['PAFE_FEATURE_DIM'],
                fp_dim=model_hps['FP_NBITS'],
                pathway_embedding_dim=model_hps['GNN_EMBEDDING_DIM'],
                drug_embedding_dim=model_hps['DRUG_EMBEDDING_DIM'],
                gnn_hidden_dim=model_hps['GNN_HIDDEN_DIM_1'],
                gnn_heads=model_hps['GNN_HEADS_L1'],
                gnn_dropout=model_hps.get('gnn_dropout', 0.1),
                ann_hidden_dim1=model_hps['ANN_HIDDEN_DIM_1'],
                ann_hidden_dim2=model_hps['ANN_HIDDEN_DIM_2'],
                ann_dropout=model_hps.get('ann_dropout', 0.1),
                scm_hidden_dim=model_hps.get('SCM_HIDDEN_DIM', 128),
                num_message_passing_steps=model_hps.get('NUM_MESSAGE_PASSING_STEPS', 3),
                scm_dropout=model_hps.get('transformer_dropout', 0.1)
            )
            model.to(device).eval()
        else:
            if model is None:
                checkpoint = torch.load(config['model_path'], map_location=device, weights_only=False)
                model_hps = checkpoint['hyperparameters']

                print(f"Loading multi-drug model from: {config['model_path']}")
                model = load_scm_from_checkpoint(
                    checkpoint_path=config['model_path'],
                    pathway_dict=global_data['pathway_dict'],
                    ordered_pathway_names=global_data['nodes'],
                    pathway_graph_pickle_path=config['pathway_interaction_graph_file'],
                    pafe_feature_dim=model_hps['PAFE_FEATURE_DIM'],
                    fp_dim=model_hps['FP_NBITS'],
                    pathway_embedding_dim=model_hps['GNN_EMBEDDING_DIM'],
                    drug_embedding_dim=model_hps['DRUG_EMBEDDING_DIM'],
                    gnn_hidden_dim=model_hps['GNN_HIDDEN_DIM_1'],
                    gnn_heads=model_hps['GNN_HEADS_L1'],
                    gnn_dropout=model_hps.get('gnn_dropout', 0.1),
                    ann_hidden_dim1=model_hps['ANN_HIDDEN_DIM_1'],
                    ann_hidden_dim2=model_hps['ANN_HIDDEN_DIM_2'],
                    ann_dropout=model_hps.get('ann_dropout', 0.1),
                    scm_hidden_dim=model_hps.get('SCM_HIDDEN_DIM', 128),
                    num_message_passing_steps=model_hps.get('NUM_MESSAGE_PASSING_STEPS', 3),
                    scm_dropout=model_hps.get('transformer_dropout', 0.1)
                )
                model.to(device).eval()

        summaries_by_dataset = process_drug_for_knockout(
            drug_name, config, go_map, model, global_data,
            master_smiles_map, npvae_embeddings_map, device,
            gene_minimum_rna_map=gene_minimum_rna_map,
            summary_file_path=summary_file_path
        )
        if summaries_by_dataset is not None:
            all_drug_summaries.update(summaries_by_dataset)

    print("\n\nAll drugs processed")

    if all_drug_summaries:
        generate_final_summary_heatmaps(
            all_drug_summaries,
            config,
            go_map,
            summary_file_path,
            output_dir=get_knockout_output_base_dir(config)
        )


if __name__ == "__main__":
    print(f"Loading configuration from: {config_path}")

    config = load_config(config_path)
    drugs = get_drugs_to_process(config)
    main(config, drugs)
