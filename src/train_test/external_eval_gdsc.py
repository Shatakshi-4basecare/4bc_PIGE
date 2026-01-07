"""External GDSC dataset evaluation.

Provides utilities to evaluate trained models on external GDSC datasets.
"""

import os
import pickle
import functools
from typing import Dict, List, Tuple, Optional

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

from src.data_processing.cancer_drug_dataset import CancerDrugDataset
from src.data_processing.collate_fold_data import collate_gpu_fold
from src.models.scm_wrapper import create_scm_model


def get_dataset_configurations(
    paths_config: Dict[str, str],
    include_gdsc2_datasets: bool = True
) -> List[Dict[str, str]]:
    """Get GDSC dataset configurations for evaluation.

    Args:
        paths_config: Dictionary with 'intermediate_data_dir' key.
        include_gdsc2_datasets: Whether to include GDSC2 datasets.

    Returns:
        List of dataset configuration dictionaries.
    """
    intermediate_data_dir = paths_config['intermediate_data_dir']
    precomputed_base = os.path.join(intermediate_data_dir, 'precomputed_features')

    configurations = [{
        "name": "GDSC0_true_test",
        "response_file_subdir": "GDSC0_drug_response_data",
        "precomputed_dir": os.path.join(precomputed_base, "GDSC0_true_test"),
        "is_true_test": True
    }]

    if include_gdsc2_datasets:
        configurations.extend([
            {
                "name": "GDSC2_main",
                "response_file_subdir": "GDSC2_drug_response_data",
                "precomputed_dir": os.path.join(precomputed_base, "GDSC2"),
                "is_true_test": False
            },
            {
                "name": "GDSC2_true_test",
                "response_file_subdir": "GDSC2_drug_response_data",
                "precomputed_dir": os.path.join(precomputed_base, "GDSC2_true_test"),
                "is_true_test": True
            }
        ])

    return configurations


def prepare_eval_samples(
    response_file_path: str,
    precomputed_features_dir: str,
    dataset_version_name: str,
    drug_name: str
) -> List[Tuple[str, str, float]]:
    """Prepare evaluation samples for a drug from a response file.

    Args:
        response_file_path: Path to response CSV file.
        precomputed_features_dir: Directory with precomputed features.
        dataset_version_name: Name of dataset for logging.
        drug_name: Drug name.

    Returns:
        List of (cell_id, drug_name, response) tuples.
    """
    df_resp_data = pd.read_csv(response_file_path)
    df_drug_data = df_resp_data.dropna(subset=['aac']).copy()
    df_drug_data['aac'] = pd.to_numeric(df_drug_data['aac'], errors='coerce')
    df_drug_data = df_drug_data.dropna(subset=['aac'])
    df_current_eval_data = df_drug_data[['ModelID', 'aac']].drop_duplicates(
        subset=['ModelID'], keep='first'
    ).copy()

    sample_list = []
    drug_specific_features_root = os.path.join(precomputed_features_dir, drug_name)

    for _, row in df_current_eval_data.iterrows():
        cell_id = row['ModelID']
        expected_filename = f"pafe_fp_{cell_id}_{drug_name}.pt"
        expected_filepath = os.path.join(drug_specific_features_root, expected_filename)

        try:
            feature_data = torch.load(expected_filepath, map_location=torch.device('cpu'))
            pafe_features = feature_data.get('pafe_features')

            if pafe_features is not None and torch.any(pafe_features):
                response = float(row['aac'])
                sample_list.append((cell_id, drug_name, response))
        except:
            pass

    print(f"Prepared {len(sample_list)} samples for {dataset_version_name} ({drug_name})")
    return sample_list


def prepare_evaluation_data_for_all_drugs(
    drug_names_to_process: List[str],
    paths_config: Dict[str, str],
    dataset_configs: List[Dict[str, str]]
) -> Tuple[Dict, Dict]:
    """Prepare evaluation data for all drugs across all dataset configurations.

    Args:
        drug_names_to_process: List of drug names.
        paths_config: Path configuration dictionary.
        dataset_configs: List of dataset configurations.

    Returns:
        Tuple of (all_prepared_data, graph_info) where all_prepared_data is
        {drug_name: {config_name: {'sample_list': list, 'precomputed_dir': str}}}
        and graph_info contains N_PATHWAYS, EDGE_INDEX, MAX_SEQ_LEN.
    """
    intermediate_data_dir = paths_config['intermediate_data_dir']
    graph_info_file_path = paths_config['pathway_interaction_graph_file']

    with open(graph_info_file_path, 'rb') as f:
        pathway_graph = pickle.load(f)

    all_pathway_names = pathway_graph.get('nodes')
    if not all_pathway_names:
        raise ValueError("'nodes' key not found in pathway interaction graph")

    pathway_to_idx = {name: i for i, name in enumerate(all_pathway_names)}
    n_pathways = len(all_pathway_names)
    raw_edge_list = pathway_graph.get('edge_list', [])

    edge_list = []
    for source_id, target_id in raw_edge_list:
        if source_id in pathway_to_idx and target_id in pathway_to_idx:
            edge_list.append([pathway_to_idx[source_id], pathway_to_idx[target_id]])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    graph_info = {
        'N_PATHWAYS': n_pathways,
        'EDGE_INDEX': edge_index,
        'MAX_SEQ_LEN': 1 + 1 + n_pathways
    }

    all_prepared_data = {}

    for drug_name_iter in drug_names_to_process:
        all_prepared_data[drug_name_iter] = {}

        for config in dataset_configs:
            current_config_name = config["name"]
            current_precomputed_dir = config["precomputed_dir"]
            response_file_name = f'{drug_name_iter}.csv'

            if config["is_true_test"]:
                response_path = os.path.join(
                    intermediate_data_dir,
                    config["response_file_subdir"],
                    'true_test',
                    response_file_name
                )
            else:
                response_path = os.path.join(
                    intermediate_data_dir,
                    config["response_file_subdir"],
                    response_file_name
                )

            if not os.path.exists(response_path):
                all_prepared_data[drug_name_iter][current_config_name] = {
                    "sample_list": [],
                    "precomputed_dir": current_precomputed_dir
                }
                continue

            sample_list = prepare_eval_samples(
                response_file_path=response_path,
                precomputed_features_dir=current_precomputed_dir,
                dataset_version_name=current_config_name,
                drug_name=drug_name_iter
            )
            all_prepared_data[drug_name_iter][current_config_name] = {
                "sample_list": sample_list,
                "precomputed_dir": current_precomputed_dir
            }

    return all_prepared_data, graph_info


def test_model(
    current_eval_sample_list: List[Tuple[str, str, float]],
    current_precomputed_features_dir: str,
    dataset_name_tag: str,
    drug_name_for_dataset: str,
    graph_info: Dict,
    model_state_dict_input: Optional[Dict] = None,
    saved_hps_input: Optional[Dict] = None,
    model_path_str: Optional[str] = None,
    tqdm_disable: bool = False,
    pathway_interaction_graph_path: Optional[str] = None
) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, List[float]]]]:
    """Test model on evaluation dataset and compute metrics.

    Args:
        current_eval_sample_list: List of (cell_id, drug_name, response) tuples.
        current_precomputed_features_dir: Directory with precomputed features.
        dataset_name_tag: Dataset name for logging.
        drug_name_for_dataset: Drug name.
        graph_info: Graph information dict with N_PATHWAYS, EDGE_INDEX.
        model_state_dict_input: Model state dict (alternative to model_path_str).
        saved_hps_input: Saved hyperparameters (alternative to model_path_str).
        model_path_str: Path to model checkpoint (alternative to state dict).
        tqdm_disable: Whether to disable progress bar.
        pathway_interaction_graph_path: Path to pathway interaction graph.

    Returns:
        Tuple of (metrics_dict, predictions_dict) where:
        - metrics_dict contains 'spearman' and 'pearson'
        - predictions_dict contains 'actual_aac' and 'predicted_aac' lists
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_state_dict_input is not None and saved_hps_input is not None:
        model_state_dict = model_state_dict_input
        saved_hps = saved_hps_input
    elif model_path_str is not None:
        if not os.path.exists(model_path_str):
            print(f"Model file not found: {model_path_str}")
            return None, None

        state_dict_data = torch.load(model_path_str, map_location=device, weights_only=False)
        model_state_dict = state_dict_data['model_state_dict']
        saved_hps = state_dict_data['hyperparameters']
    else:
        print("test_model requires either model_path_str or (model_state_dict_input and saved_hps_input)")
        return None, None

    with open(pathway_interaction_graph_path, 'rb') as f:
        gdata = pickle.load(f)

    ordered_names = gdata.get('nodes', [])
    node_to_genes = gdata.get('node_to_genes', {}) or gdata.get('node_to_genes_map', {})
    p_dict = {
        k: set(v) if isinstance(v, (list, tuple)) else (v if isinstance(v, set) else set())
        for k, v in node_to_genes.items()
    }

    selected_omics_type = saved_hps.get(
        'selected_omics_type',
        saved_hps.get('SELECTED_OMICS_TYPE', 'all')
    )

    eval_model = create_scm_model(
        pathway_graph_pickle_path=pathway_interaction_graph_path,
        pathway_dict=p_dict,
        ordered_pathway_names=ordered_names,
        pafe_feature_dim=saved_hps['PAFE_FEATURE_DIM'],
        fp_dim=saved_hps['FP_NBITS'],
        pathway_embedding_dim=saved_hps['GNN_EMBEDDING_DIM'],
        drug_embedding_dim=saved_hps['DRUG_EMBEDDING_DIM'],
        gnn_hidden_dim=saved_hps['GNN_HIDDEN_DIM_1'],
        gnn_heads=saved_hps['GNN_HEADS_L1'],
        gnn_dropout=saved_hps['gnn_dropout'],
        ann_hidden_dim1=saved_hps['ANN_HIDDEN_DIM_1'],
        ann_hidden_dim2=saved_hps['ANN_HIDDEN_DIM_2'],
        ann_dropout=saved_hps['ann_dropout'],
        scm_hidden_dim=saved_hps['scm_hidden_dim'],
        num_message_passing_steps=saved_hps['num_message_passing_steps'],
        scm_dropout=saved_hps['transformer_dropout'],
        selected_omics_type=selected_omics_type
    ).to(device)

    eval_model.load_state_dict(model_state_dict, strict=False)
    eval_model.eval()

    batch_size = saved_hps.get('BATCH_SIZE', 16)
    eval_dataset = CancerDrugDataset(
        samples=current_eval_sample_list,
        precomputed_dir=current_precomputed_features_dir,
        device=device
    )
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=functools.partial(
            collate_gpu_fold,
            num_nodes=graph_info['N_PATHWAYS'],
            edge_index_for_batch=graph_info['EDGE_INDEX']
        )
    )

    all_preds_eval = []
    all_labels_eval = []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f"Evaluating {dataset_name_tag}", leave=False, disable=tqdm_disable):
            pafe_flat = batch['pafe_features_flat'].to(device, non_blocking=True)
            edge_idx = batch['edge_index_batch'].to(device, non_blocking=True)
            fp = batch['drug_fingerprints'].to(device, non_blocking=True)
            labels = batch['labels'].float()
            if labels.ndim == 1:
                labels = labels.unsqueeze(1)

            predictions = eval_model(pafe_flat, edge_idx, fp)
            all_preds_eval.append(predictions)
            all_labels_eval.append(labels)

    if not all_preds_eval or not all_labels_eval:
        print("No predictions collected")
        return {'spearman': np.nan, 'pearson': np.nan}, {'actual_aac': [], 'predicted_aac': []}

    predictions_cat = torch.cat(all_preds_eval).squeeze().cpu().numpy()
    all_aac_values = torch.cat(all_labels_eval).squeeze().cpu().numpy()

    spearman_result = spearmanr(predictions_cat, all_aac_values)
    pearson_result = pearsonr(predictions_cat, all_aac_values)

    metrics_dict = {
        'spearman': abs(spearman_result.correlation) if not np.isnan(spearman_result.correlation) else 0.0,
        'pearson': abs(pearson_result.statistic) if not np.isnan(pearson_result.statistic) else 0.0
    }

    predictions_vs_actual_dict = {
        'actual_aac': all_aac_values.tolist(),
        'predicted_aac': predictions_cat.tolist()
    }

    return metrics_dict, predictions_vs_actual_dict
