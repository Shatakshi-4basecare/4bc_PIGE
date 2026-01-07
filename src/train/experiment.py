"""Main experiment orchestration for SCM training."""

import os
import copy
import random
import traceback
import functools
from typing import Dict, List, Optional, Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
import tqdm
import shutil
from sklearn.model_selection import KFold, ShuffleSplit

from src.models.scm_wrapper import create_scm_model
from src.data_processing.cancer_drug_dataset import CancerDrugDataset
from src.data_processing.collate_fold_data import GPUDatasetFold, collate_gpu_fold
from src.train_test.external_eval_gdsc import prepare_evaluation_data_for_all_drugs as external_prepare_eval_data
from src.train_test.external_eval_gdsc import test_model as external_test_model
from src.train_test.external_eval_gdsc import get_dataset_configurations as external_get_dataset_configurations

from .training import train_epoch, validate_epoch
from .plotting import _generate_and_save_epoch_plots
from .metrics import select_best_epoch_from_metrics


def run_experiment(
    run_identifier: str,
    current_seed: int,
    all_drugs_to_train: List[str],
    base_checkpoint_dir: str,
    n_pathways_gs: int,
    edge_index_gs: torch.Tensor,
    max_seq_len_gs: int,
    target_cell_lines_gs: set,
    df_response_gs: pd.DataFrame,
    lr: float,
    weight_decay: float,
    ann_dropout: float,
    gnn_dropout: float,
    pos_emb_dropout: float,
    transformer_dropout: float,
    paths_config: Dict,
    training_settings: Dict,
    model_arch_config: Dict,
    optuna_settings: Dict,
    enable_plotting: bool,
    include_gdsc2_datasets: bool,
    pretrained_state_dict_path: Optional[str] = None,
    pathway_dict: Optional[Dict] = None,
    ordered_pathway_names: Optional[List[str]] = None
) -> Tuple[float, List[str]]:
    """Run cross-validation experiment for drug response prediction.

    Args:
        run_identifier: Unique identifier for the run.
        current_seed: Random seed for reproducibility.
        all_drugs_to_train: List of drug names to train on.
        base_checkpoint_dir: Directory for saving checkpoints.
        n_pathways_gs: Number of pathways.
        edge_index_gs: Graph edge indices.
        max_seq_len_gs: Maximum sequence length.
        target_cell_lines_gs: Set of target cell line IDs.
        df_response_gs: Drug response dataframe.
        lr: Learning rate.
        weight_decay: Weight decay for optimizer.
        ann_dropout: Dropout for ANN encoder.
        gnn_dropout: Dropout for GNN encoder.
        pos_emb_dropout: Dropout for positional embeddings.
        transformer_dropout: Dropout for transformer/SCM.
        paths_config: Configuration for file paths.
        training_settings: Training configuration.
        model_arch_config: Model architecture configuration.
        optuna_settings: Optuna settings.
        enable_plotting: Whether to generate plots.
        include_gdsc2_datasets: Whether to include GDSC2 datasets.
        pretrained_state_dict_path: Path to pretrained weights.
        pathway_dict: Pathway to gene mapping.
        ordered_pathway_names: Ordered pathway names.

    Returns:
        Tuple of (objective_value, best_model_paths).
    """
    n_folds = training_settings['n_folds']
    num_epochs_per_drug = training_settings['num_epochs_per_drug']
    batch_size = training_settings['batch_size']
    early_stopping_patience = training_settings['early_stopping_patience']
    clip_grad_norm = training_settings['clip_grad_norm']
    loss_fn = training_settings['loss_fn']
    scheduler_factor = training_settings['scheduler_factor']
    scheduler_patience = training_settings['scheduler_patience']
    device = training_settings['device']
    selected_omics_type = training_settings.get('selected_omics_type', 'all')
    precomputed_features_dir = paths_config['precomputed_features_dir']
    response_col_gs = training_settings['response_val_col']

    num_epochs = int(num_epochs_per_drug * len(all_drugs_to_train))

    pafe_feature_dim = model_arch_config['pafe_feature_dim']
    fp_nbits = model_arch_config['fp_nbits']
    gnn_embedding_dim = model_arch_config.get('pathway_embedding_dim', model_arch_config.get('gnn_embedding_dim', 128))
    drug_embedding_dim = model_arch_config.get('drug_embedding_dim', 128)
    transformer_input_dim = model_arch_config.get('transformer_input_dim', 256)
    transformer_nhead = model_arch_config.get('transformer_nhead', 8)
    transformer_dim_ff = model_arch_config.get('transformer_dim_ff', 1024)
    transformer_num_layers = model_arch_config.get('transformer_num_layers', 2)
    gnn_hidden_dim1 = model_arch_config['gnn_hidden_dim1']
    gnn_heads_l1 = model_arch_config['gnn_heads_l1']
    ann_hidden_dim1 = model_arch_config['ann_hidden_dim1']
    ann_hidden_dim2 = model_arch_config['ann_hidden_dim2']
    scm_hidden_dim = model_arch_config.get('scm_hidden_dim', 128)
    num_message_passing_steps = model_arch_config.get('num_message_passing_steps', 3)

    print(f"\n--- Starting Run {run_identifier}, Seed: {current_seed} ---")
    print(f"  Hyperparameters: lr={lr:.2e}, wd={weight_decay:.2e}, ann_do={ann_dropout:.3f}, gnn_do={gnn_dropout:.3f}, pos_do={pos_emb_dropout:.3f}, tr_do={transformer_dropout:.3f}")
    print(f"  SCM params: scm_hidden_dim={scm_hidden_dim}, num_message_passing_steps={num_message_passing_steps}")
    if selected_omics_type and selected_omics_type.lower() != 'all':
        print(f"  Omics selection: {selected_omics_type}")

    random.seed(current_seed)
    np.random.seed(current_seed)
    torch.manual_seed(current_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(current_seed)

    df_target_response = df_response_gs[df_response_gs['ModelID'].isin(target_cell_lines_gs)].copy()
    initial_sample_list = []
    for _, r in df_target_response.iterrows():
        if pd.notna(r[response_col_gs]):
            initial_sample_list.append((r['ModelID'], r['drug_name'], float(r[response_col_gs])))

    if n_folds > 1:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=current_seed)
    elif n_folds == 1:
        print(f"  n_folds=1, using a single 80/20 train/validation split.")
        kf = ShuffleSplit(n_splits=1, test_size=0.2, random_state=current_seed)
    else:
        raise ValueError(f"n_folds must be a positive integer, got {n_folds}")

    fold_best_val_spearmans = []
    fold_best_val_pearsons = []
    fold_best_val_per_drug_spearmans = []
    fold_best_val_per_drug_pearsons = []
    fold_best_val_combined_spearmans_all_samples = []
    fold_best_val_combined_pearsons_all_samples = []
    all_folds_epoch_metrics = []
    trial_saved_checkpoint_paths = []

    # For single-drug training, save directly to base_checkpoint_dir to match pre-trained model structure
    # For multi-drug training, create run subdirectory for better organization
    if len(all_drugs_to_train) == 1:
        run_checkpoint_dir = base_checkpoint_dir
    else:
        run_checkpoint_dir = os.path.join(base_checkpoint_dir, f"run_{run_identifier}_seed_{current_seed}")
    os.makedirs(run_checkpoint_dir, exist_ok=True)

    print(f"\n--- Run {run_identifier}-S{current_seed}: Preparing all GDSC evaluation data upfront... ---")

    dataset_configs_for_gdsc = external_get_dataset_configurations(paths_config, include_gdsc2_datasets=include_gdsc2_datasets)

    all_gdsc_eval_data_prepared, gdsc_graph_info_prepared = external_prepare_eval_data(
        all_drugs_to_train, paths_config, dataset_configs_for_gdsc
    )
    if not all_gdsc_eval_data_prepared:
        print(f"  WARNING: Run {run_identifier}-S{current_seed}: No GDSC evaluation data could be prepared. GDSC plots will be skipped.")
    print(f"--- Run {run_identifier}-S{current_seed}: Finished preparing GDSC evaluation data. ---")

    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(initial_sample_list)):
        fold_num_display = fold_idx + 1
        print(f"  Training on fold {fold_num_display} of {n_folds}...")

        train_samples_fold = [initial_sample_list[i] for i in train_indices]
        val_samples_fold = [initial_sample_list[i] for i in val_indices]

        current_fold_epoch_train_losses = []
        current_fold_epoch_val_losses = []
        current_fold_epoch_avg_val_spearmans = []
        current_fold_epoch_per_drug_val_spearmans = []
        current_fold_epoch_gdsc_spearmans_all_datasets = []
        current_fold_epoch_avg_val_pearsons = []
        current_fold_epoch_per_drug_val_pearsons = []
        current_fold_epoch_gdsc_pearsons_all_datasets = []

        try:
            train_dataset_fold = CancerDrugDataset(samples=train_samples_fold, precomputed_dir=precomputed_features_dir)
            val_dataset_fold = CancerDrugDataset(samples=val_samples_fold, precomputed_dir=precomputed_features_dir)

            gpu_data_fold = {'train': [], 'val': []}
            for subset_name, subset_dataset in [('train', train_dataset_fold), ('val', val_dataset_fold)]:
                for i in range(len(subset_dataset)):
                    sample = subset_dataset[i]
                    gpu_data_fold[subset_name].append(sample)

            edge_idx_fold_device = edge_index_gs
            train_loader_fold = DataLoader(
                GPUDatasetFold(gpu_data_fold['train']),
                batch_size=batch_size,
                shuffle=True,
                collate_fn=functools.partial(collate_gpu_fold, num_nodes=n_pathways_gs, edge_index_for_batch=edge_idx_fold_device)
            )
            val_loader_fold = DataLoader(
                GPUDatasetFold(gpu_data_fold['val']),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=functools.partial(collate_gpu_fold, num_nodes=n_pathways_gs, edge_index_for_batch=edge_idx_fold_device)
            )
        except Exception as e_data_prep:
            print(f"Run {run_identifier}-S{current_seed}-F{fold_num_display} Error during data prep: {e_data_prep}")
            traceback.print_exc()
            fold_best_val_spearmans.append(0.0)
            fold_best_val_pearsons.append(0.0)
            fold_best_val_per_drug_spearmans.append({})
            fold_best_val_per_drug_pearsons.append({})
            fold_best_val_combined_spearmans_all_samples.append(0.0)
            fold_best_val_combined_pearsons_all_samples.append(0.0)
            continue

        print(f"  Creating Structural Causal Model (SCM) for fold {fold_num_display}...")
        pathway_graph_pickle_path = paths_config.get('pathway_interaction_graph_file')
        if pathway_graph_pickle_path is None:
            raise ValueError("pathway_interaction_graph_file must be specified in paths_config for SCM")

        model_fold = create_scm_model(
            pathway_graph_pickle_path=pathway_graph_pickle_path,
            pathway_dict=pathway_dict,
            ordered_pathway_names=ordered_pathway_names,
            pafe_feature_dim=pafe_feature_dim,
            fp_dim=fp_nbits,
            pathway_embedding_dim=gnn_embedding_dim,
            drug_embedding_dim=drug_embedding_dim,
            gnn_hidden_dim=gnn_hidden_dim1,
            gnn_heads=gnn_heads_l1,
            gnn_dropout=gnn_dropout,
            ann_hidden_dim1=ann_hidden_dim1,
            ann_hidden_dim2=ann_hidden_dim2,
            ann_dropout=ann_dropout,
            scm_hidden_dim=scm_hidden_dim,
            scm_dropout=transformer_dropout,
            num_message_passing_steps=num_message_passing_steps,
            selected_omics_type=selected_omics_type
        ).to(device)

        if pretrained_state_dict_path:
            checkpoint_pre = torch.load(pretrained_state_dict_path, map_location=device, weights_only=False)
            if 'model_state_dict' in checkpoint_pre:
                missing, unexpected = model_fold.load_state_dict(checkpoint_pre['model_state_dict'], strict=False)
            else:
                missing, unexpected = model_fold.load_state_dict(checkpoint_pre, strict=False)

            print(f"Loaded pre-trained weights from: {pretrained_state_dict_path}")

        optimizer_fold = optim.AdamW(filter(lambda p: p.requires_grad, model_fold.parameters()), lr=lr, weight_decay=weight_decay)
        scheduler_fold = ReduceLROnPlateau(optimizer_fold, mode='max', factor=scheduler_factor, patience=scheduler_patience, threshold=0.005)

        best_val_spearman_this_fold = -np.inf
        best_val_pearson_this_fold = -np.inf
        best_per_drug_spearman_this_fold = {}
        best_per_drug_pearson_this_fold = {}
        best_combined_spearman_all_samples_this_fold = -np.inf
        best_combined_pearson_all_samples_this_fold = -np.inf
        epochs_no_improve_fold = 0

        for epoch_f_idx in tqdm.tqdm(range(num_epochs), desc=f"Run {run_identifier}-S{current_seed}-F{fold_num_display} Training", leave=False, disable=False):
            epoch_num_display = epoch_f_idx + 1
            log_prefix_epoch = f"Run {run_identifier}-S{current_seed}-F{fold_num_display}-E{epoch_num_display}"

            avg_train_loss_f, train_issues_f = train_epoch(model_fold, train_loader_fold, optimizer_fold, loss_fn, device, clip_grad_norm, log_prefix_epoch)
            if train_issues_f or np.isnan(avg_train_loss_f):
                current_fold_epoch_train_losses.append(avg_train_loss_f if pd.notna(avg_train_loss_f) else np.nan)
                current_fold_epoch_val_losses.append(np.nan)
                current_fold_epoch_avg_val_spearmans.append(np.nan)
                current_fold_epoch_per_drug_val_spearmans.append({})
                current_fold_epoch_avg_val_pearsons.append(np.nan)
                current_fold_epoch_per_drug_val_pearsons.append({})
                current_fold_epoch_gdsc_spearmans_all_datasets.append({})
                current_fold_epoch_gdsc_pearsons_all_datasets.append({})
                break

            avg_val_loss_f, val_spearman_avg_per_drug_f, val_pearson_avg_per_drug_f, per_drug_spearman_f, per_drug_pearson_f, combined_sp_all_samples_f, combined_p_all_samples_f, val_issues_f = validate_epoch(
                model_fold, val_loader_fold, loss_fn, log_prefix_epoch
            )

            current_fold_epoch_train_losses.append(avg_train_loss_f)
            current_fold_epoch_val_losses.append(avg_val_loss_f if pd.notna(avg_val_loss_f) else np.nan)
            current_fold_epoch_avg_val_spearmans.append(val_spearman_avg_per_drug_f if pd.notna(val_spearman_avg_per_drug_f) else np.nan)
            current_fold_epoch_per_drug_val_spearmans.append(per_drug_spearman_f if per_drug_spearman_f else {})
            current_fold_epoch_avg_val_pearsons.append(val_pearson_avg_per_drug_f if pd.notna(val_pearson_avg_per_drug_f) else np.nan)
            current_fold_epoch_per_drug_val_pearsons.append(per_drug_pearson_f if per_drug_pearson_f else {})

            epoch_gdsc_metrics_all_datasets_spearman = {cfg['name']: {} for cfg in dataset_configs_for_gdsc}
            epoch_gdsc_metrics_all_datasets_pearson = {cfg['name']: {} for cfg in dataset_configs_for_gdsc}

            if all_gdsc_eval_data_prepared and model_fold is not None:
                current_model_state_dict_for_gdsc = copy.deepcopy(model_fold.state_dict())
                current_hps_for_gdsc = {
                    "N_PATHWAYS": n_pathways_gs,
                    "PAFE_FEATURE_DIM": pafe_feature_dim,
                    "FP_NBITS": fp_nbits,
                    "GNN_EMBEDDING_DIM": gnn_embedding_dim,
                    "DRUG_EMBEDDING_DIM": drug_embedding_dim,
                    "TRANSFORMER_INPUT_DIM": transformer_input_dim,
                    "TRANSFORMER_NHEAD": transformer_nhead,
                    "TRANSFORMER_DIM_FEEDFORWARD": transformer_dim_ff,
                    "TRANSFORMER_NUM_LAYERS": transformer_num_layers,
                    "transformer_dropout": transformer_dropout,
                    "GNN_HIDDEN_DIM_1": gnn_hidden_dim1,
                    "GNN_HEADS_L1": gnn_heads_l1,
                    "gnn_dropout": gnn_dropout,
                    "ANN_HIDDEN_DIM_1": ann_hidden_dim1,
                    "ANN_HIDDEN_DIM_2": ann_hidden_dim2,
                    "ann_dropout": ann_dropout,
                    "MAX_SEQ_LEN": max_seq_len_gs,
                    "pos_emb_dropout": pos_emb_dropout,
                    "BATCH_SIZE": batch_size,
                    "scm_hidden_dim": scm_hidden_dim,
                    "num_message_passing_steps": num_message_passing_steps,
                    "selected_omics_type": selected_omics_type,
                }

                for drug_name_gdsc_eval in all_drugs_to_train:
                    if drug_name_gdsc_eval not in all_gdsc_eval_data_prepared:
                        for cfg_item in dataset_configs_for_gdsc:
                            epoch_gdsc_metrics_all_datasets_spearman[cfg_item['name']][drug_name_gdsc_eval] = np.nan
                            epoch_gdsc_metrics_all_datasets_pearson[cfg_item['name']][drug_name_gdsc_eval] = np.nan
                        continue

                    for gdsc_config in dataset_configs_for_gdsc:
                        config_name = gdsc_config["name"]
                        data_package = all_gdsc_eval_data_prepared[drug_name_gdsc_eval].get(config_name)

                        if data_package and data_package["sample_list"]:
                            gdsc_metrics, _ = external_test_model(
                                model_state_dict_input=current_model_state_dict_for_gdsc,
                                saved_hps_input=current_hps_for_gdsc,
                                current_eval_sample_list=data_package["sample_list"],
                                current_precomputed_features_dir=data_package["precomputed_dir"],
                                dataset_name_tag=f"{config_name}_{drug_name_gdsc_eval}_E{epoch_num_display}",
                                drug_name_for_dataset=drug_name_gdsc_eval,
                                tqdm_disable=True,
                                graph_info=gdsc_graph_info_prepared,
                                pathway_interaction_graph_path=pathway_graph_pickle_path
                            )
                            if gdsc_metrics and 'spearman' in gdsc_metrics:
                                epoch_gdsc_metrics_all_datasets_spearman[config_name][drug_name_gdsc_eval] = gdsc_metrics['spearman']
                                epoch_gdsc_metrics_all_datasets_pearson[config_name][drug_name_gdsc_eval] = gdsc_metrics.get('pearson', np.nan)
                            else:
                                epoch_gdsc_metrics_all_datasets_spearman[config_name][drug_name_gdsc_eval] = np.nan
                                epoch_gdsc_metrics_all_datasets_pearson[config_name][drug_name_gdsc_eval] = np.nan
                        else:
                            epoch_gdsc_metrics_all_datasets_spearman[config_name][drug_name_gdsc_eval] = np.nan
                            epoch_gdsc_metrics_all_datasets_pearson[config_name][drug_name_gdsc_eval] = np.nan

            current_fold_epoch_gdsc_spearmans_all_datasets.append(epoch_gdsc_metrics_all_datasets_spearman)
            current_fold_epoch_gdsc_pearsons_all_datasets.append(epoch_gdsc_metrics_all_datasets_pearson)

            _generate_and_save_epoch_plots(
                run_identifier=run_identifier,
                current_seed=current_seed,
                fold_num_display=fold_num_display,
                current_epoch_num_display=epoch_num_display,
                run_checkpoint_dir=run_checkpoint_dir,
                all_drugs_to_train=all_drugs_to_train,
                external_dataset_configs=dataset_configs_for_gdsc,
                epochs_range_to_plot=range(1, epoch_num_display + 1),
                train_losses_current_fold=current_fold_epoch_train_losses,
                val_losses_current_fold=current_fold_epoch_val_losses,
                avg_val_spearmans_current_fold=current_fold_epoch_avg_val_spearmans,
                per_drug_val_spearmans_current_fold=current_fold_epoch_per_drug_val_spearmans,
                gdsc_epoch_spearmans_current_fold=current_fold_epoch_gdsc_spearmans_all_datasets,
                avg_val_pearsons_current_fold=current_fold_epoch_avg_val_pearsons,
                per_drug_val_pearsons_current_fold=current_fold_epoch_per_drug_val_pearsons,
                gdsc_epoch_pearsons_current_fold=current_fold_epoch_gdsc_pearsons_all_datasets,
                enable_plotting=enable_plotting
            )

            all_epochs_checkpoints_base_dir = os.path.join(run_checkpoint_dir, "all_epoch_checkpoints")
            os.makedirs(all_epochs_checkpoints_base_dir, exist_ok=True)
            all_epochs_checkpoints_fold_dir = os.path.join(all_epochs_checkpoints_base_dir, f"fold_{fold_num_display}")
            os.makedirs(all_epochs_checkpoints_fold_dir, exist_ok=True)

            epoch_model_hps = {
                "lr": lr,
                "weight_decay": weight_decay,
                "ann_dropout": ann_dropout,
                "gnn_dropout": gnn_dropout,
                "pos_emb_dropout": pos_emb_dropout,
                "transformer_dropout": transformer_dropout,
                "PAFE_FEATURE_DIM": model_arch_config['pafe_feature_dim'],
                "FP_NBITS": model_arch_config['fp_nbits'],
                "GNN_EMBEDDING_DIM": model_arch_config['gnn_embedding_dim'],
                "DRUG_EMBEDDING_DIM": model_arch_config['drug_embedding_dim'],
                "TRANSFORMER_INPUT_DIM": model_arch_config['transformer_input_dim'],
                "TRANSFORMER_NHEAD": model_arch_config['transformer_nhead'],
                "TRANSFORMER_DIM_FEEDFORWARD": model_arch_config['transformer_dim_ff'],
                "TRANSFORMER_NUM_LAYERS": model_arch_config['transformer_num_layers'],
                "GNN_HIDDEN_DIM_1": model_arch_config['gnn_hidden_dim1'],
                "GNN_HEADS_L1": model_arch_config['gnn_heads_l1'],
                "ANN_HIDDEN_DIM_1": model_arch_config['ann_hidden_dim1'],
                "ANN_HIDDEN_DIM_2": model_arch_config['ann_hidden_dim2'],
                "MAX_SEQ_LEN": max_seq_len_gs,
                "N_PATHWAYS": n_pathways_gs,
                "BATCH_SIZE": batch_size,
                "SCM_HIDDEN_DIM": scm_hidden_dim,
                "NUM_MESSAGE_PASSING_STEPS": num_message_passing_steps,
                "SCM_DROPOUT": transformer_dropout,
                "selected_omics_type": selected_omics_type,
                "SELECTED_OMICS_TYPE": selected_omics_type,
            }

            epoch_checkpoint_content = {
                'model_state_dict': copy.deepcopy(model_fold.state_dict()),
                'hyperparameters': epoch_model_hps,
                'run_info': {
                    'run_identifier': run_identifier,
                    'current_seed': current_seed,
                    'fold': fold_num_display,
                    'epoch': epoch_num_display,
                },
                'metrics': {
                    'avg_train_loss': avg_train_loss_f if pd.notna(avg_train_loss_f) else None,
                    'avg_val_loss': avg_val_loss_f if pd.notna(avg_val_loss_f) else None,
                    'internal_val_spearman_avg_per_drug': val_spearman_avg_per_drug_f if pd.notna(val_spearman_avg_per_drug_f) else None,
                    'internal_val_pearson_avg_per_drug': val_pearson_avg_per_drug_f if pd.notna(val_pearson_avg_per_drug_f) else None,
                    'internal_val_combined_spearman_all_samples': combined_sp_all_samples_f if pd.notna(combined_sp_all_samples_f) else None,
                    'internal_val_combined_pearson_all_samples': combined_p_all_samples_f if pd.notna(combined_p_all_samples_f) else None,
                }
            }
            epoch_checkpoint_filename = f"fold{fold_num_display}_epoch{epoch_num_display}_model.pth"
            path_to_epoch_checkpoint = os.path.join(all_epochs_checkpoints_fold_dir, epoch_checkpoint_filename)
            torch.save(epoch_checkpoint_content, path_to_epoch_checkpoint)
            print(f"  {log_prefix_epoch}: Saved epoch checkpoint to {path_to_epoch_checkpoint}")

            if val_issues_f or np.isnan(avg_val_loss_f):
                break

            val_spearman_clean_f = val_spearman_avg_per_drug_f if pd.notna(val_spearman_avg_per_drug_f) else 0.0
            val_pearson_clean_f = val_pearson_avg_per_drug_f if pd.notna(val_pearson_avg_per_drug_f) else 0.0
            per_drug_spearman_clean_f = per_drug_spearman_f if per_drug_spearman_f else {}
            per_drug_pearson_clean_f = per_drug_pearson_f if per_drug_pearson_f else {}
            combined_sp_all_samples_clean_f = combined_sp_all_samples_f if pd.notna(combined_sp_all_samples_f) else 0.0
            combined_p_all_samples_clean_f = combined_p_all_samples_f if pd.notna(combined_p_all_samples_f) else 0.0

            if val_spearman_clean_f > best_val_spearman_this_fold:
                best_val_spearman_this_fold = val_spearman_clean_f
                best_val_pearson_this_fold = val_pearson_clean_f
                best_per_drug_spearman_this_fold = per_drug_spearman_clean_f
                best_per_drug_pearson_this_fold = per_drug_pearson_clean_f
                best_combined_spearman_all_samples_this_fold = combined_sp_all_samples_clean_f
                best_combined_pearson_all_samples_this_fold = combined_p_all_samples_clean_f
                epochs_no_improve_fold = 0
            else:
                epochs_no_improve_fold += 1

            scheduler_fold.step(val_spearman_clean_f)

            if epochs_no_improve_fold >= early_stopping_patience:
                print(f"  Run {run_identifier}-S{current_seed}-F{fold_num_display}: Early stopping at epoch {epoch_num_display}.")
                break

        if np.isfinite(best_val_spearman_this_fold):
            fold_best_val_spearmans.append(best_val_spearman_this_fold)
        else:
            fold_best_val_spearmans.append(0.0)

        if np.isfinite(best_val_pearson_this_fold):
            fold_best_val_pearsons.append(best_val_pearson_this_fold)
        else:
            fold_best_val_pearsons.append(0.0)

        fold_best_val_per_drug_spearmans.append(best_per_drug_spearman_this_fold if best_per_drug_spearman_this_fold else {})
        fold_best_val_per_drug_pearsons.append(best_per_drug_pearson_this_fold if best_per_drug_pearson_this_fold else {})
        fold_best_val_combined_spearmans_all_samples.append(best_combined_spearman_all_samples_this_fold if np.isfinite(best_combined_spearman_all_samples_this_fold) else 0.0)
        fold_best_val_combined_pearsons_all_samples.append(best_combined_pearson_all_samples_this_fold if np.isfinite(best_combined_pearson_all_samples_this_fold) else 0.0)

        current_fold_metrics = {
            "fold_num": fold_num_display,
            "train_losses": current_fold_epoch_train_losses,
            "val_losses": current_fold_epoch_val_losses,
            "avg_val_spearmans": current_fold_epoch_avg_val_spearmans,
            "per_drug_val_spearmans": current_fold_epoch_per_drug_val_spearmans,
            "gdsc_epoch_spearmans": current_fold_epoch_gdsc_spearmans_all_datasets,
            "avg_val_pearsons": current_fold_epoch_avg_val_pearsons,
            "per_drug_val_pearsons": current_fold_epoch_per_drug_val_pearsons,
            "gdsc_epoch_pearsons": current_fold_epoch_gdsc_pearsons_all_datasets
        }
        all_folds_epoch_metrics.append(current_fold_metrics)

        print(f"  Run {run_identifier}-S{current_seed}-F{fold_num_display} Finished. ")
        if best_per_drug_spearman_this_fold:
            print(f"    Best Per-Drug Spearman for this fold:")
            for drug_name, score in best_per_drug_spearman_this_fold.items():
                print(f"      {drug_name}: {score:.4f}")
        if best_per_drug_pearson_this_fold:
            print(f"    Best Per-Drug Pearson for this fold:")
            for drug_name, score in best_per_drug_pearson_this_fold.items():
                print(f"      {drug_name}: {score:.4f}")
        print(f"    Best Combined (All Samples) Spearman for this fold: {best_combined_spearman_all_samples_this_fold:.4f}")
        print(f"    Best Combined (All Samples) Pearson for this fold: {best_combined_pearson_all_samples_this_fold:.4f}")
        print(f"    Best Val Spearman (Avg Per-Drug) for this fold: {best_val_spearman_this_fold:.4f}")
        print(f"    Best Val Pearson (Avg Per-Drug) for this fold: {best_val_pearson_this_fold:.4f}")

        primary_metric_name = optuna_settings.get('primary_metric_name_for_selection', 'GDSC0_true_test')
        smoothing_window = optuna_settings.get('num_past_epochs_for_smoothing', 4) + 1

        best_epoch_this_fold_new = select_best_epoch_from_metrics(
            fold_epoch_metrics=current_fold_metrics,
            all_drugs_to_train=all_drugs_to_train,
            primary_metric_name=primary_metric_name,
            smoothing_window_size=smoothing_window
        )

        if best_epoch_this_fold_new != -1:
            all_epochs_checkpoints_fold_dir = os.path.join(run_checkpoint_dir, "all_epoch_checkpoints", f"fold_{fold_num_display}")
            epoch_checkpoint_filename = f"fold{fold_num_display}_epoch{best_epoch_this_fold_new}_model.pth"
            path_to_best_model_ckpt = os.path.join(all_epochs_checkpoints_fold_dir, epoch_checkpoint_filename)

            if os.path.exists(path_to_best_model_ckpt):
                best_epoch_idx = best_epoch_this_fold_new - 1
                val_spearman_at_best_epoch = 0.0
                if best_epoch_idx < len(current_fold_epoch_avg_val_spearmans):
                    val_spearman_at_best_epoch = current_fold_epoch_avg_val_spearmans[best_epoch_idx]

                best_model_final_filename = f"model_run{run_identifier}_seed{current_seed}_best_fold{fold_num_display}_e{best_epoch_this_fold_new}_selected.pth"
                final_path_to_model_ckpt = os.path.join(run_checkpoint_dir, best_model_final_filename)

                shutil.copy(path_to_best_model_ckpt, final_path_to_model_ckpt)
                trial_saved_checkpoint_paths.append((final_path_to_model_ckpt, val_spearman_at_best_epoch))
                print(f"  Selected best model for fold {fold_num_display} from epoch {best_epoch_this_fold_new}. Copied to: {final_path_to_model_ckpt}")
            else:
                print(f"  WARNING: Best model checkpoint not found at {path_to_best_model_ckpt}.")
        else:
            print(f"  WARNING: Could not determine best epoch for fold {fold_num_display}.")

        del model_fold, optimizer_fold, scheduler_fold, train_loader_fold, val_loader_fold, gpu_data_fold
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_val_spearman = np.mean(fold_best_val_spearmans) if fold_best_val_spearmans else 0.0
    std_val_spearman = np.std(fold_best_val_spearmans) if fold_best_val_spearmans and len(fold_best_val_spearmans) > 1 else 0.0
    avg_val_pearson = np.mean(fold_best_val_pearsons) if fold_best_val_pearsons else 0.0
    std_val_pearson = np.std(fold_best_val_pearsons) if fold_best_val_pearsons and len(fold_best_val_pearsons) > 1 else 0.0

    if trial_saved_checkpoint_paths:
        trial_saved_checkpoint_paths.sort(key=lambda x: x[1], reverse=True)
        best_model_paths_this_run = [p for p, s in trial_saved_checkpoint_paths]
    else:
        best_model_paths_this_run = []

    print(f"\n--- Finished Run {run_identifier} ---")
    print(f"  Validation Spearman: {avg_val_spearman:.4f} (Std: {std_val_spearman:.4f})")
    print(f"  Validation Pearson: {avg_val_pearson:.4f} (Std: {std_val_pearson:.4f})")

    objective_value = avg_val_spearman

    return objective_value, best_model_paths_this_run
