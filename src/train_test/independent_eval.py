"""Evaluate SCM models on CTRPv2 or GDSC0 datasets.

Supports both single model and per-drug model evaluation with separate outputs for each dataset.
"""

from __future__ import annotations

import os
import sys
import pickle
import ast
import functools
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_processing.cancer_drug_dataset import CancerDrugDataset
from src.data_processing.collate_fold_data import collate_gpu_fold
from src.models.scm_wrapper import load_scm_from_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    'dataset': 'gdsc0',
    'model_path': 'data/output_data/pige_pipeline_scm1/For_Real_ALL_DRUGS/run_fixed_params_seed_10663/all_epoch_checkpoints/fold_1/fold1_epoch27_model.pth',
    'per_drug_models_dir': None,
    'base_data_dir': 'data',
    'drug_list_file': 'data/input_data/drugs_to_process.csv',
    'output_dir': 'PIGE_Figures/Fig1/Fig1b/output_data/independent_eval',
    'random_state': 10663,
}


def load_drug_names(drug_list_file: str) -> List[str]:
    """Load drug names from CSV.

    Args:
        drug_list_file: Path to drugs CSV

    Returns:
        List of drug names
    """
    drug_names = []
    df_drugs = pd.read_csv(drug_list_file)
    for _, row in df_drugs.iterrows():
        drug_name = str(row['drug_name']).replace("-", "_").replace(" ", "_")
        drug_names.append(drug_name)
    print(f"Loaded {len(drug_names)} drugs")
    return drug_names


def load_graph_info(graph_file_path: str) -> Dict:
    """Load pathway graph information.

    Args:
        graph_file_path: Path to graph pickle

    Returns:
        Dictionary with graph structure
    """
    with open(graph_file_path, 'rb') as f:
        pathway_graph = pickle.load(f)

    all_pathway_names = pathway_graph.get('nodes')
    pathway_to_idx = {name: i for i, name in enumerate(all_pathway_names)}
    n_pathways = len(all_pathway_names)

    raw_edge_list = pathway_graph.get('edge_list', [])
    edge_list = []
    for source_id, target_id in raw_edge_list:
        if source_id in pathway_to_idx and target_id in pathway_to_idx:
            edge_list.append([pathway_to_idx[source_id], pathway_to_idx[target_id]])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(DEVICE)

    return {
        'N_PATHWAYS': n_pathways,
        'EDGE_INDEX': edge_index
    }


def filter_samples_by_features(samples: List[Tuple[str, str, float]], precomputed_dir: str) -> List[Tuple[str, str, float]]:
    """Filter samples to only include those with valid features.

    Args:
        samples: List of (cell_id, drug_name, response) tuples
        precomputed_dir: Directory with precomputed features

    Returns:
        Filtered list of samples
    """
    filtered = []
    samples_by_drug = {}
    for cell_id, drug_name, response in samples:
        if drug_name not in samples_by_drug:
            samples_by_drug[drug_name] = []
        samples_by_drug[drug_name].append((cell_id, response))

    for drug_name, drug_samples in samples_by_drug.items():
        drug_features_dir = os.path.join(precomputed_dir, drug_name)

        for cell_id, response in drug_samples:
            feature_file = os.path.join(drug_features_dir, f"pafe_fp_{cell_id}_{drug_name}.pt")

            if not os.path.exists(feature_file):
                continue

            feature_data = torch.load(feature_file, map_location=torch.device('cpu'))
            pafe_features = feature_data.get('pafe_features')

            if pafe_features is None or not torch.any(pafe_features):
                continue

            filtered.append((cell_id, drug_name, response))

    return filtered


def prepare_eval_samples(response_file: str, precomputed_dir: str, drug_name: str) -> List[Tuple[str, str, float]]:
    """Prepare evaluation samples for a drug.

    Args:
        response_file: Path to response CSV
        precomputed_dir: Precomputed features directory
        drug_name: Drug name

    Returns:
        List of samples
    """
    if not os.path.exists(response_file):
        print(f"Response file not found: {response_file}")
        return []

    df = pd.read_csv(response_file)
    df.dropna(subset=['aac'], inplace=True)
    df['aac'] = pd.to_numeric(df['aac'], errors='coerce')
    df.dropna(subset=['aac'], inplace=True)
    df = df[['ModelID', 'aac']].drop_duplicates(subset=['ModelID'], keep='first').copy()

    samples = []
    drug_features_dir = os.path.join(precomputed_dir, drug_name)

    for _, row in df.iterrows():
        cell_id = row['ModelID']
        feature_file = os.path.join(drug_features_dir, f"pafe_fp_{cell_id}_{drug_name}.pt")

        if not os.path.exists(feature_file):
            continue

        feature_data = torch.load(feature_file, map_location=torch.device('cpu'))
        pafe_features = feature_data.get('pafe_features')

        if pafe_features is None or not torch.any(pafe_features):
            continue

        samples.append((cell_id, drug_name, float(row['aac'])))

    print(f"Prepared {len(samples)} samples for {drug_name}")
    return samples


def find_per_drug_model(drug_name: str, base_dir: str) -> Optional[str]:
    """Find model for a specific drug.

    Args:
        drug_name: Drug name
        base_dir: Base models directory

    Returns:
        Model path or None
    """
    drug_model_dir = os.path.join(base_dir, drug_name, "run_fixed_params_seed_10663")

    if not os.path.exists(drug_model_dir):
        return None

    for file in os.listdir(drug_model_dir):
        if file.endswith(".pth"):
            model_path = os.path.join(drug_model_dir, file)
            print(f"Found model for {drug_name}: {model_path}")
            return model_path

    return None


def test_model(model_path: str, eval_samples: List[Tuple[str, str, float]], precomputed_dir: str,
               graph_info: Dict, graph_info_file: str, eval_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Run model evaluation.

    Args:
        model_path: Path to model checkpoint
        eval_samples: List of evaluation samples
        precomputed_dir: Precomputed features directory
        graph_info: Graph structure info
        graph_info_file: Path to graph pickle
        eval_name: Name for progress display

    Returns:
        Tuple of (predictions, labels) or (None, None)
    """
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None, None

    state_dict_data = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model_state_dict = state_dict_data['model_state_dict']
    saved_hps = state_dict_data['hyperparameters']

    with open(graph_info_file, 'rb') as f:
        graph_data = pickle.load(f)

    edge_list = ast.literal_eval(graph_data['edge_list']) if isinstance(graph_data['edge_list'], str) else graph_data['edge_list']
    nodes = graph_data['nodes']
    node_to_idx = {node_id: i for i, node_id in enumerate(nodes)}
    edge_index = torch.tensor(
        [[node_to_idx.get(u), node_to_idx.get(v)] for u, v in edge_list if u in node_to_idx and v in node_to_idx],
        dtype=torch.long
    ).t().contiguous().to(DEVICE)

    global_data = {
        "nodes": nodes,
        "edge_index": edge_index,
        "pathway_dict": graph_data['node_to_genes']
    }

    eval_model = load_scm_from_checkpoint(
        checkpoint_path=model_path,
        pathway_dict=global_data['pathway_dict'],
        ordered_pathway_names=global_data['nodes'],
        pathway_graph_pickle_path=graph_info_file,
        pafe_feature_dim=saved_hps['PAFE_FEATURE_DIM'],
        fp_dim=saved_hps['FP_NBITS'],
        pathway_embedding_dim=saved_hps['GNN_EMBEDDING_DIM'],
        drug_embedding_dim=saved_hps['DRUG_EMBEDDING_DIM'],
        gnn_hidden_dim=saved_hps['GNN_HIDDEN_DIM_1'],
        gnn_heads=saved_hps['GNN_HEADS_L1'],
        gnn_dropout=saved_hps.get('gnn_dropout', 0.1),
        ann_hidden_dim1=saved_hps['ANN_HIDDEN_DIM_1'],
        ann_hidden_dim2=saved_hps['ANN_HIDDEN_DIM_2'],
        ann_dropout=saved_hps.get('ann_dropout', 0.1),
        scm_hidden_dim=saved_hps.get('SCM_HIDDEN_DIM', 128),
        scm_dropout=saved_hps.get('transformer_dropout', 0.1)
    )

    eval_model.load_state_dict(model_state_dict, strict=False)
    eval_model.to(DEVICE)
    eval_model.eval()

    batch_size = saved_hps['BATCH_SIZE']
    eval_dataset = CancerDrugDataset(samples=eval_samples, precomputed_dir=precomputed_dir, device=DEVICE)
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

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f"Evaluating {eval_name}", leave=False):
            pafe_flat = batch['pafe_features_flat'].to(DEVICE, non_blocking=True)
            edge_idx = batch['edge_index_batch'].to(DEVICE, non_blocking=True)
            fp = batch['drug_fingerprints'].to(DEVICE, non_blocking=True)
            labels = batch['labels'].float()
            if labels.ndim == 1:
                labels = labels.unsqueeze(1)

            predictions = eval_model(pafe_flat, edge_idx, fp)

            all_preds.append(predictions)
            all_labels.append(labels)

    if not all_preds or not all_labels:
        print(f"No predictions collected for {eval_name}")
        return None, None

    predictions = torch.cat(all_preds).squeeze().cpu().numpy()
    labels = torch.cat(all_labels).squeeze().cpu().numpy()

    return predictions, labels


def evaluate_ctrpv2(config: Dict) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Run CTRPv2 evaluation with ShuffleSplit.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (spearman, pearson, rmse)
    """
    base_data_dir = config['base_data_dir']
    random_state = config['random_state']
    intermediate_data_dir = os.path.join(base_data_dir, 'intermediate_data')
    graph_info_file = os.path.join(intermediate_data_dir, 'all_pathway_interaction_graph.pkl')
    drug_response_dir = os.path.join(intermediate_data_dir, 'CTRPv2_drug_response_data')
    precomputed_dir = os.path.join(intermediate_data_dir, 'precomputed_features', 'CTRPv2')

    drug_names = load_drug_names(config['drug_list_file'])
    graph_info = load_graph_info(graph_info_file)

    print("Starting CTRPv2 Evaluation")

    all_response_dfs = []
    for drug in tqdm(drug_names, desc="Loading drug response CSVs"):
        response_file = os.path.join(drug_response_dir, f"{drug}.csv")
        if os.path.exists(response_file):
            df = pd.read_csv(response_file)
            df['drug_name'] = drug
            all_response_dfs.append(df)

    if not all_response_dfs:
        print("No drug response files found")
        return None, None, None

    df_full = pd.concat(all_response_dfs, ignore_index=True)
    df_full.dropna(subset=['aac'], inplace=True)
    df_full['aac'] = pd.to_numeric(df_full['aac'], errors='coerce')
    df_full.dropna(subset=['aac'], inplace=True)

    initial_samples = []
    for _, r in df_full.iterrows():
        if 'ModelID' in r and 'drug_name' in r:
            initial_samples.append((r['ModelID'], r['drug_name'], float(r['aac'])))

    print(f"Total samples: {len(initial_samples)}")

    indices = np.arange(len(initial_samples))
    ss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state) # To replicate the same split as in the original code, we need to use the same random state.
    _, test_indices = next(ss.split(indices))

    test_samples = [initial_samples[i] for i in test_indices]
    eval_samples = filter_samples_by_features(test_samples, precomputed_dir)
    print(f"Test set size after filtering: {len(eval_samples)}")

    if not eval_samples:
        return None, None, None

    use_per_drug = config['per_drug_models_dir'] is not None
    model_name = 'per_drug_models_ctrpv2' if use_per_drug else os.path.splitext(os.path.basename(config['model_path']))[0]
    output_dir = os.path.join(config['output_dir'], model_name, 'ctrpv2', 'raw_data_ctrpv2_internal_validation')
    os.makedirs(output_dir, exist_ok=True)

    if use_per_drug:
        print("Per-drug model evaluation not implemented for CTRPv2 ShuffleSplit")
        return None, None, None

    predictions, labels = test_model(
        config['model_path'], eval_samples, precomputed_dir,
        graph_info, graph_info_file, "CTRPv2 Test"
    )

    if predictions is None or labels is None:
        return None, None, None

    cell_ids = [s[0] for s in eval_samples]
    drug_names_eval = [s[1] for s in eval_samples]
    results_df = pd.DataFrame({
        'cell_id': cell_ids,
        'drug': drug_names_eval,
        'prediction': predictions,
        'label': labels
    })

    per_drug_metrics = {}
    for drug_name, group in results_df.groupby('drug'):
        df_drug = group[['cell_id', 'prediction', 'label']].copy()
        df_drug.rename(columns={'cell_id': 'sample_name', 'prediction': 'predicted_aac', 'label': 'actual_aac'}, inplace=True)
        output_file = os.path.join(output_dir, f"{drug_name}_predictions.csv")
        df_drug.to_csv(output_file, index=False, float_format='%.4g')

        if len(group) < 2 or group['prediction'].nunique() < 2 or group['label'].nunique() < 2:
            continue

        sr, sp = spearmanr(group['prediction'], group['label'])
        pr, pp = pearsonr(group['prediction'], group['label'])
        rmse = np.sqrt(mean_squared_error(group['label'], group['prediction']))

        per_drug_metrics[drug_name] = {
            'spearman': sr,
            'spearman_p': sp,
            'pearson': pr,
            'pearson_p': pp,
            'rmse': rmse,
            'count': len(group)
        }

    print("\n\nPer-Drug Results")
    for drug, metrics in sorted(per_drug_metrics.items(), key=lambda x: x[1]['count'], reverse=True):
        print(f"\n{drug} ({metrics['count']} samples)")
        print(f"  Spearman: {metrics['spearman']:.4f}")
        print(f"  Pearson:  {metrics['pearson']:.4f}")
        print(f"  RMSE:     {metrics['rmse']:.4f}")

    if per_drug_metrics:
        median_spearman = np.median([m['spearman'] for m in per_drug_metrics.values() if pd.notna(m['spearman'])])
        median_pearson = np.median([m['pearson'] for m in per_drug_metrics.values() if pd.notna(m['pearson'])])
        print(f"\nMedian Spearman: {median_spearman:.4f}")
        print(f"Median Pearson: {median_pearson:.4f}")

    print("\nCombined Results")
    if len(predictions) >= 2:
        comb_sr, comb_sp = spearmanr(predictions, labels)
        comb_pr, comb_pp = pearsonr(predictions, labels)
        comb_rmse = np.sqrt(mean_squared_error(labels, predictions))

        print(f"Total samples: {len(predictions)}")
        print(f"  Spearman: {comb_sr:.4f}")
        print(f"  Pearson:  {comb_pr:.4f}")
        print(f"  RMSE:     {comb_rmse:.4f}")

        return comb_sr, comb_pr, comb_rmse

    return None, None, None


def evaluate_gdsc0(config: Dict) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Run GDSC0 true test evaluation.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (spearman, pearson, rmse)
    """
    base_data_dir = config['base_data_dir']
    intermediate_data_dir = os.path.join(base_data_dir, 'intermediate_data')
    graph_info_file = os.path.join(intermediate_data_dir, 'all_pathway_interaction_graph.pkl')
    drug_response_dir = os.path.join(intermediate_data_dir, 'GDSC0_drug_response_data', 'true_test')
    precomputed_dir = os.path.join(intermediate_data_dir, 'precomputed_features', 'GDSC0_true_test')

    drug_names = load_drug_names(config['drug_list_file'])
    graph_info = load_graph_info(graph_info_file)

    print("Starting GDSC0 Evaluation")

    all_predictions = []
    all_labels = []
    per_drug_metrics = {}

    use_per_drug = config['per_drug_models_dir'] is not None
    model_name = 'per_drug_models_gdsc0' if use_per_drug else os.path.splitext(os.path.basename(config['model_path']))[0]
    output_dir = os.path.join(config['output_dir'], model_name, 'gdsc0', 'raw_data_scm')
    os.makedirs(output_dir, exist_ok=True)

    for drug in drug_names:
        print(f"\nProcessing {drug}")

        if use_per_drug:
            model_path = find_per_drug_model(drug, config['per_drug_models_dir'])
            if model_path is None:
                print(f"No model found for {drug}")
                continue
        else:
            model_path = config['model_path']

        response_file = os.path.join(drug_response_dir, f"{drug}.csv")
        eval_samples = prepare_eval_samples(response_file, precomputed_dir, drug)

        if not eval_samples:
            continue

        predictions, labels = test_model(
            model_path, eval_samples, precomputed_dir,
            graph_info, graph_info_file, drug
        )

        if predictions is None or labels is None:
            continue

        cell_ids = [s[0] for s in eval_samples]
        df_drug = pd.DataFrame({
            'sample_name': cell_ids,
            'predicted_aac': predictions,
            'actual_aac': labels
        })
        output_file = os.path.join(output_dir, f"{drug}_predictions.csv")
        df_drug.to_csv(output_file, index=False, float_format='%.4g')

        sr, sp = spearmanr(predictions, labels)
        pr, pp = pearsonr(predictions, labels)
        rmse = np.sqrt(mean_squared_error(labels, predictions))

        per_drug_metrics[drug] = {
            'spearman': sr,
            'spearman_p': sp,
            'pearson': pr,
            'pearson_p': pp,
            'rmse': rmse,
            'count': len(predictions)
        }

        all_predictions.extend(predictions)
        all_labels.extend(labels)

    print("\n\nPer-Drug Results")
    for drug, metrics in per_drug_metrics.items():
        print(f"\n{drug} ({metrics['count']} samples)")
        print(f"  Spearman: {metrics['spearman']:.4f}")
        print(f"  Pearson:  {metrics['pearson']:.4f}")
        print(f"  RMSE:     {metrics['rmse']:.4f}")

    if per_drug_metrics:
        median_spearman = np.median([m['spearman'] for m in per_drug_metrics.values() if pd.notna(m['spearman'])])
        median_pearson = np.median([m['pearson'] for m in per_drug_metrics.values() if pd.notna(m['pearson'])])
        print(f"\nMedian Spearman: {median_spearman:.4f}")
        print(f"Median Pearson: {median_pearson:.4f}")

    print("\nCombined Results")
    if len(all_predictions) >= 2:
        comb_sr, comb_sp = spearmanr(all_predictions, all_labels)
        comb_pr, comb_pp = pearsonr(all_predictions, all_labels)
        comb_rmse = np.sqrt(mean_squared_error(all_labels, all_predictions))

        print(f"Total samples: {len(all_predictions)}")
        print(f"  Spearman: {comb_sr:.4f}")
        print(f"  Pearson:  {comb_pr:.4f}")
        print(f"  RMSE:     {comb_rmse:.4f}")

        return comb_sr, comb_pr, comb_rmse

    return None, None, None


def main(config: Dict) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Run evaluation based on dataset in config.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (spearman, pearson, rmse)
    """
    dataset = config.get('dataset', 'ctrpv2').lower()

    if dataset == 'ctrpv2':
        return evaluate_ctrpv2(config)
    elif dataset == 'gdsc0':
        return evaluate_gdsc0(config)
    else:
        print(f"Unknown dataset: {dataset}")
        return None, None, None


if __name__ == "__main__":
    main(CONFIG)
