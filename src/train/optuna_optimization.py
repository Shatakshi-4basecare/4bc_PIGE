"""Optuna optimization for SCM hyperparameter tuning."""

import os
import sys
import time
import json
import functools
import pickle
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import optuna
import pandas as pd
import numpy as np
import sklearn

from .data_loading import load_prerequisite_data
from .experiment import run_experiment


def objective(
    trial: optuna.trial.Trial,
    all_drugs_to_train: List[str],
    base_checkpoint_dir_trial_specific: str,
    n_pathways: int,
    edge_index: torch.Tensor,
    max_seq_len: int,
    target_cell_lines_set: set,
    df_response: pd.DataFrame,
    paths_config: Dict,
    training_settings: Dict,
    model_arch_config: Dict,
    optuna_settings: Dict,
    enable_plotting: bool,
    include_gdsc2_datasets: bool,
    pathway_dict: Optional[Dict] = None,
    ordered_pathway_names: Optional[List[str]] = None
) -> float:
    """Optuna objective function for hyperparameter optimization.

    Args:
        trial: Optuna trial object for hyperparameter suggestions.
        all_drugs_to_train: List of drug names to train models for.
        base_checkpoint_dir_trial_specific: Directory for saving trial checkpoints.
        n_pathways: Number of pathways in the interaction graph.
        edge_index: Graph edge indices.
        max_seq_len: Maximum sequence length for cell lines.
        target_cell_lines_set: Set of target cell line IDs.
        df_response: Drug response dataframe.
        paths_config: Configuration dictionary for file paths.
        training_settings: Configuration dictionary for training parameters.
        model_arch_config: Configuration dictionary for model architecture.
        optuna_settings: Configuration dictionary for Optuna settings.
        enable_plotting: Whether to enable plotting during training.
        include_gdsc2_datasets: Whether to include GDSC2 datasets in evaluation.
        pathway_dict: Pathway dictionary mapping.
        ordered_pathway_names: Ordered list of pathway names.

    Returns:
        Objective value to be maximized by Optuna.
    """
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    lr = round(lr, 8)

    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.25)
    weight_decay = round(weight_decay, 6)

    clip_grad_norm_val = trial.suggest_float("clip_grad_norm", 0.5, 3.0)
    clip_grad_norm_val = round(clip_grad_norm_val, 2)

    pathway_embedding_dim = trial.suggest_categorical("pathway_embedding_dim", [96, 128, 160, 192, 224, 256])
    drug_embedding_dim = trial.suggest_categorical("drug_embedding_dim", [96, 128, 160, 192, 224, 256])

    possible_hidden_dims = [384, 512, 640, 768, 896, 1024]
    possible_heads = [4, 6, 8, 10, 12]
    valid_combinations = []
    for dim in possible_hidden_dims:
        for heads in possible_heads:
            if dim % heads == 0:
                valid_combinations.append((dim, heads))

    print(f"Valid GNN combinations: {valid_combinations}")
    print(f"# 6: {valid_combinations[6]}")
    selected_combo_idx = trial.suggest_int("gnn_dim_heads_combo_idx", 0, len(valid_combinations) - 1)
    gnn_hidden_dim1, gnn_heads_l1 = valid_combinations[selected_combo_idx]

    gnn_dropout = trial.suggest_float("gnn_dropout", 0.05, 0.60)
    gnn_dropout = round(gnn_dropout, 4)

    ann_hidden_dim1 = trial.suggest_categorical("ann_hidden_dim1", [320, 384, 448, 512, 576, 640])
    ann_hidden_dim2 = trial.suggest_categorical("ann_hidden_dim2", [160, 192, 224, 256, 288, 320])

    ann_dropout = trial.suggest_float("ann_dropout", 0.05, 0.50)
    ann_dropout = round(ann_dropout, 4)

    scm_hidden_dim = trial.suggest_categorical("scm_hidden_dim", [160, 192, 224, 256, 288, 320])

    scm_dropout = trial.suggest_float("scm_dropout", 0.05, 0.40)
    scm_dropout = round(scm_dropout, 4)

    num_message_passing_steps = trial.suggest_int("num_message_passing_steps", 3, 6)

    batch_size_trial = trial.suggest_categorical("batch_size", [16, 24, 32, 48, 64])

    pos_emb_dropout = 0.0
    use_scm = True

    trial_start_time = time.time()

    trial_model_arch_config = model_arch_config.copy()
    trial_model_arch_config['pathway_embedding_dim'] = pathway_embedding_dim
    trial_model_arch_config['gnn_embedding_dim'] = pathway_embedding_dim
    trial_model_arch_config['drug_embedding_dim'] = drug_embedding_dim
    trial_model_arch_config['gnn_hidden_dim1'] = gnn_hidden_dim1
    trial_model_arch_config['gnn_heads_l1'] = gnn_heads_l1
    trial_model_arch_config['ann_hidden_dim1'] = ann_hidden_dim1
    trial_model_arch_config['ann_hidden_dim2'] = ann_hidden_dim2
    trial_model_arch_config['scm_hidden_dim'] = scm_hidden_dim
    trial_model_arch_config['num_message_passing_steps'] = num_message_passing_steps

    trial_training_settings = training_settings.copy()
    trial_training_settings['batch_size'] = batch_size_trial
    trial_training_settings['clip_grad_norm'] = clip_grad_norm_val

    objective_value, best_model_paths_for_this_trial = run_experiment(
        run_identifier=trial.number,
        current_seed=optuna_settings['predetermined_seed'],
        all_drugs_to_train=all_drugs_to_train,
        base_checkpoint_dir=base_checkpoint_dir_trial_specific,
        n_pathways_gs=n_pathways,
        edge_index_gs=edge_index,
        max_seq_len_gs=max_seq_len,
        target_cell_lines_gs=target_cell_lines_set,
        df_response_gs=df_response,
        lr=lr,
        weight_decay=weight_decay,
        ann_dropout=ann_dropout,
        gnn_dropout=gnn_dropout,
        pos_emb_dropout=pos_emb_dropout,
        transformer_dropout=scm_dropout,
        paths_config=paths_config,
        training_settings=trial_training_settings,
        model_arch_config=trial_model_arch_config,
        optuna_settings=optuna_settings,
        enable_plotting=enable_plotting,
        include_gdsc2_datasets=include_gdsc2_datasets,
        pathway_dict=pathway_dict,
        ordered_pathway_names=ordered_pathway_names
    )

    if best_model_paths_for_this_trial:
        trial.set_user_attr("best_model_paths_for_trial", best_model_paths_for_this_trial)

    trial_duration_mins = (time.time() - trial_start_time) / 60
    print(f"Optuna Trial {trial.number} duration: {trial_duration_mins:.2f} mins. Objective Value: {objective_value:.4f}")

    return objective_value


def main(
    all_drugs_to_train: List[str],
    base_checkpoint_dir_root: str,
    paths_config: Dict,
    training_settings: Dict,
    model_arch_config: Dict,
    optuna_settings: Dict,
    enable_plotting: bool = True,
    include_gdsc2_datasets: bool = True
) -> Dict[str, str]:
    """Main orchestration function for training and optimization.

    Args:
        all_drugs_to_train: List of drug names to train models for.
        base_checkpoint_dir_root: Root directory for saving checkpoints.
        paths_config: Configuration dictionary for file paths.
        training_settings: Configuration dictionary for training parameters.
        model_arch_config: Configuration dictionary for model architecture.
        optuna_settings: Configuration dictionary for Optuna settings.
        enable_plotting: Whether to enable plotting during training.
        include_gdsc2_datasets: Whether to include GDSC2 datasets in evaluation.

    Returns:
        Dictionary mapping drug names to best model paths.
    """
    device = training_settings['device']
    precomputed_features_dir = paths_config['precomputed_features_dir']
    pathway_interaction_graph_file = paths_config['pathway_interaction_graph_file']
    ctrpv2_drug_response_dir = paths_config['ctrpv2_drug_response_dir']
    response_val_col = training_settings['response_val_col']
    use_optuna = optuna_settings['use_optuna']
    fixed_hyperparameters = optuna_settings['fixed_hyperparameters']
    n_optuna_trials = optuna_settings['n_optuna_trials']
    predetermined_seed = optuna_settings['predetermined_seed']
    enable_plotting_main = enable_plotting
    include_gdsc2_main = include_gdsc2_datasets

    print(f"Device: {device}")
    print(f"Using Precomputed Features from: {precomputed_features_dir}")
    print("-" * 30)

    n_pathways, edge_index, max_seq_len, target_cell_lines_set, df_response = load_prerequisite_data(
        all_drugs_to_train, pathway_interaction_graph_file, ctrpv2_drug_response_dir, response_val_col, device
    )
    overall_start_time = time.time()

    os.makedirs(base_checkpoint_dir_root, exist_ok=True)

    final_drug_to_model_path_map = {}
    best_model_paths_overall = None

    print("\n--- GDSC Evaluation Data Preparation is now handled by external_eval_gdsc.main ---")

    env_info = {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "optuna_version": optuna.__version__ if use_optuna else "N/A",
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "script_execution_time": time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(overall_start_time)),
        "using_optuna": use_optuna,
        "fixed_hyperparameters": fixed_hyperparameters if not use_optuna else "N/A",
        "paths_config": paths_config,
        "training_settings": {k: str(v) if isinstance(v, (torch.device, nn.modules.loss._Loss)) else v for k, v in training_settings.items()},
        "model_arch_config": model_arch_config,
        "optuna_settings_main": {k: str(v) if isinstance(v, dict) else v for k, v in optuna_settings.items()}
    }
    env_info_path = os.path.join(base_checkpoint_dir_root, "environment_info.json")
    try:
        with open(env_info_path, 'w') as f:
            json.dump(env_info, f, indent=4)
        print(f"Environment information saved to: {env_info_path}")
    except Exception as e:
        print(f"Error saving environment information: {e}")
        print(f"Problematic env_info: {env_info}")

    if use_optuna:
        study_name = f"PathInteractTransformer_Optuna_{time.strftime('%Y%m%d-%H%M%S')}"

        print(f"\n--- Starting Optuna Optimization: {n_optuna_trials} Trials ---")
        print(f"Study Name: {study_name}")

        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=None,
        )

        pathway_graph_pickle = paths_config.get('pathway_interaction_graph_file')
        ordered_names = None
        p_dict = None
        if pathway_graph_pickle and os.path.exists(pathway_graph_pickle):
            with open(pathway_graph_pickle, 'rb') as f:
                gdata = pickle.load(f)
            ordered_names = gdata.get('nodes', [])
            node_to_genes = gdata.get('node_to_genes', {}) or gdata.get('node_to_genes_map', {})
            p_dict = {}
            for k, v in node_to_genes.items():
                if isinstance(v, (list, tuple)):
                    p_dict[k] = set(v)
                elif isinstance(v, set):
                    p_dict[k] = v
                else:
                    p_dict[k] = set()
            print(f"[optuna] Loaded pathway graph for SCM: {len(ordered_names)} nodes, {len(gdata.get('edge_list', []))} edges")

        objective_with_args = functools.partial(
            objective,
            all_drugs_to_train=all_drugs_to_train,
            base_checkpoint_dir_trial_specific=base_checkpoint_dir_root,
            n_pathways=n_pathways,
            edge_index=edge_index,
            max_seq_len=max_seq_len,
            target_cell_lines_set=target_cell_lines_set,
            df_response=df_response,
            paths_config=paths_config,
            training_settings=training_settings,
            model_arch_config=model_arch_config,
            optuna_settings=optuna_settings,
            enable_plotting=enable_plotting_main,
            include_gdsc2_datasets=include_gdsc2_main,
            pathway_dict=p_dict,
            ordered_pathway_names=ordered_names
        )

        study.optimize(
            lambda trial: objective_with_args(
                trial,
                base_checkpoint_dir_trial_specific=base_checkpoint_dir_root,
                enable_plotting=enable_plotting_main,
                include_gdsc2_datasets=include_gdsc2_main
            ),
            n_trials=n_optuna_trials,
            timeout=None
        )

        total_script_duration_mins = (time.time() - overall_start_time) / 60
        print(f"\n--- Optuna Optimization Finished (Total Time: {total_script_duration_mins:.2f} mins) ---")
        print(f"Number of finished trials: {len(study.trials)}")
        print(f"Best trial: {study.best_trial.number}")
        best_model_paths_overall = study.best_trial.user_attrs.get("best_model_paths_for_trial")
        if best_model_paths_overall:
            print(f"Best model paths from Optuna (Trial {study.best_trial.number}): {best_model_paths_overall}")
        else:
            print(f"WARNING: Best model paths not found in Optuna best trial user attributes.")

    else:
        print(f"\n--- Running with Fixed Hyperparameters (Optuna Disabled) ---")
        print(f"Fixed Hyperparameters: {fixed_hyperparameters}")
        fixed_run_start_time = time.time()

        pathway_graph_pickle = paths_config.get('pathway_interaction_graph_file')
        ordered_names = None
        p_dict = None
        if pathway_graph_pickle and os.path.exists(pathway_graph_pickle):
            with open(pathway_graph_pickle, 'rb') as f:
                gdata = pickle.load(f)
            ordered_names = gdata.get('nodes', [])
            node_to_genes = gdata.get('node_to_genes', {}) or gdata.get('node_to_genes_map', {})
            p_dict = {}
            for k, v in node_to_genes.items():
                if isinstance(v, (list, tuple)):
                    p_dict[k] = set(v)
                elif isinstance(v, set):
                    p_dict[k] = v
                else:
                    p_dict[k] = set()
            print(f"[fixed] Loaded pathway graph for SCM: {len(ordered_names)} nodes, {len(gdata.get('edge_list', []))} edges")

        fixed_model_arch_config = model_arch_config.copy()
        if 'pathway_embedding_dim' in fixed_hyperparameters:
            fixed_model_arch_config['pathway_embedding_dim'] = fixed_hyperparameters['pathway_embedding_dim']
            fixed_model_arch_config['gnn_embedding_dim'] = fixed_hyperparameters['pathway_embedding_dim']
        if 'drug_embedding_dim' in fixed_hyperparameters:
            fixed_model_arch_config['drug_embedding_dim'] = fixed_hyperparameters['drug_embedding_dim']
        if 'gnn_hidden_dim1' in fixed_hyperparameters:
            fixed_model_arch_config['gnn_hidden_dim1'] = fixed_hyperparameters['gnn_hidden_dim1']
        if 'gnn_heads_l1' in fixed_hyperparameters:
            fixed_model_arch_config['gnn_heads_l1'] = fixed_hyperparameters['gnn_heads_l1']
        if 'ann_hidden_dim1' in fixed_hyperparameters:
            fixed_model_arch_config['ann_hidden_dim1'] = fixed_hyperparameters['ann_hidden_dim1']
        if 'ann_hidden_dim2' in fixed_hyperparameters:
            fixed_model_arch_config['ann_hidden_dim2'] = fixed_hyperparameters['ann_hidden_dim2']
        if 'scm_hidden_dim' in fixed_hyperparameters:
            fixed_model_arch_config['scm_hidden_dim'] = fixed_hyperparameters['scm_hidden_dim']
        if 'num_message_passing_steps' in fixed_hyperparameters:
            fixed_model_arch_config['num_message_passing_steps'] = fixed_hyperparameters['num_message_passing_steps']

        fixed_training_settings = training_settings.copy()
        if 'batch_size' in fixed_hyperparameters:
            fixed_training_settings['batch_size'] = fixed_hyperparameters['batch_size']
        if 'clip_grad_norm' in fixed_hyperparameters:
            fixed_training_settings['clip_grad_norm'] = fixed_hyperparameters['clip_grad_norm']

        objective_val_fixed, model_paths_fixed = run_experiment(
            run_identifier="fixed_params",
            current_seed=predetermined_seed,
            all_drugs_to_train=all_drugs_to_train,
            base_checkpoint_dir=base_checkpoint_dir_root,
            n_pathways_gs=n_pathways,
            edge_index_gs=edge_index,
            max_seq_len_gs=max_seq_len,
            target_cell_lines_gs=target_cell_lines_set,
            df_response_gs=df_response,
            lr=float(fixed_hyperparameters["lr"]),
            weight_decay=float(fixed_hyperparameters["weight_decay"]),
            ann_dropout=float(fixed_hyperparameters["ann_dropout"]),
            gnn_dropout=float(fixed_hyperparameters["gnn_dropout"]),
            pos_emb_dropout=float(fixed_hyperparameters.get("pos_emb_dropout", 0.0)),
            transformer_dropout=float(fixed_hyperparameters.get("scm_dropout", fixed_hyperparameters.get("transformer_dropout", 0.25))),
            paths_config=paths_config,
            training_settings=fixed_training_settings,
            model_arch_config=fixed_model_arch_config,
            optuna_settings=optuna_settings,
            enable_plotting=enable_plotting_main,
            include_gdsc2_datasets=include_gdsc2_main,
            pathway_dict=p_dict,
            ordered_pathway_names=ordered_names
        )
        fixed_run_duration_mins = (time.time() - fixed_run_start_time) / 60
        print(f"\n--- Fixed Hyperparameter Run Finished (Total Time: {fixed_run_duration_mins:.2f} mins) ---")
        best_model_paths_overall = model_paths_fixed
        if best_model_paths_overall:
            print(f"Model paths from fixed hyperparameter run: {best_model_paths_overall}")
        else:
            print(f"WARNING: Model paths not found from fixed hyperparameter run.")

    if best_model_paths_overall:
        for drug_name in all_drugs_to_train:
            final_drug_to_model_path_map[drug_name] = best_model_paths_overall
    else:
        print("WARNING: No overall best model path was determined. SHAP explanations might not run.")

    print("\n--- Script Finished ---")
    return final_drug_to_model_path_map
