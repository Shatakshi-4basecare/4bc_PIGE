"""Data loading and preprocessing for training."""

import os
import pickle
from typing import Tuple, Set, List

import pandas as pd
import torch


def load_prerequisite_data(
    all_drugs_to_train: List[str],
    pathway_interaction_graph_file: str,
    ctrpv2_drug_response_dir: str,
    response_val_col: str,
    device: torch.device
) -> Tuple[int, torch.Tensor, int, Set[str], pd.DataFrame]:
    """Load all prerequisite data for training.

    Args:
        all_drugs_to_train: List of drug names to train on.
        pathway_interaction_graph_file: Path to pathway interaction graph pickle.
        ctrpv2_drug_response_dir: Directory containing drug response CSV files.
        response_val_col: Column name for response values.
        device: Device to load tensors onto.

    Returns:
        Tuple of (n_pathways, edge_index, max_seq_len, target_cell_lines_set, df_response).
    """
    with open(pathway_interaction_graph_file, 'rb') as f:
        pathway_graph = pickle.load(f)

    all_pathway_names = pathway_graph.get('nodes')
    if not all_pathway_names:
        raise ValueError("'nodes' key not found or empty in pathway interaction graph")

    pathway_to_idx = {name: i for i, name in enumerate(all_pathway_names)}
    n_pathways = len(all_pathway_names)

    raw_edge_list = pathway_graph.get('edge_list')
    if raw_edge_list is None:
        print("WARNING: 'edge_list' key not found. Assuming no edges.")
        raw_edge_list = []

    edge_list = []
    for source_id, target_id in raw_edge_list:
        if source_id in pathway_to_idx and target_id in pathway_to_idx:
            edge_list.append([pathway_to_idx[source_id], pathway_to_idx[target_id]])
        else:
            print(f"WARNING: Edge ({source_id}, {target_id}) contains node not in 'nodes' list. Skipping.")

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(device)
    max_seq_len = 1 + 1 + n_pathways
    print(f"Loaded N_PATHWAYS={n_pathways}, MAX_SEQ_LEN={max_seq_len}, EDGE_INDEX (on {device}) from {os.path.basename(pathway_interaction_graph_file)}")

    all_drug_response_dfs = []
    for drug_name in all_drugs_to_train:
        current_drug_response_file = os.path.join(ctrpv2_drug_response_dir, f'{drug_name}.csv')
        if os.path.exists(current_drug_response_file):
            df_drug = pd.read_csv(current_drug_response_file)
            df_drug['drug_name'] = drug_name
            all_drug_response_dfs.append(df_drug)
        else:
            print(f"  WARNING: Response data file not found for {drug_name} at {current_drug_response_file}. Skipping this drug.")

    if not all_drug_response_dfs:
        raise FileNotFoundError("No drug response data could be loaded")

    df_response = pd.concat(all_drug_response_dfs, ignore_index=True)
    target_cell_lines_set = df_response['ModelID'].unique()

    print(f"Loaded response data for {len(all_drugs_to_train)} drugs. Total samples: {len(df_response)}.")
    print(f"Total unique target cell line IDs: {len(target_cell_lines_set)}.")
    print(f"Response data loaded using column '{response_val_col}'.")

    return n_pathways, edge_index, max_seq_len, target_cell_lines_set, df_response
