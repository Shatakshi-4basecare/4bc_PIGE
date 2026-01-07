"""Pipeline orchestrator for drug response prediction.

Coordinates data processing, model training, and evaluation stages.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn

from src.data_processing import prep_omics_data, generate_pafe, precompute_features
from src.train.optuna_optimization import main as train_transformer_optuna_main
from src.pathway_interaction_setup import pathway_setup_main


STAGE_PATHWAY_SETUP = "pathway_setup"
STAGE_PREP_OMICS = "prep_omics_data"
STAGE_GENERATE_PAFE = "generate_pafe"
STAGE_PRECOMPUTE_FEATURES = "precompute_features"
STAGE_TRAIN_MODELS = "train_models"

ALL_STAGES = [
    STAGE_PATHWAY_SETUP,
    STAGE_PREP_OMICS,
    STAGE_GENERATE_PAFE,
    STAGE_PRECOMPUTE_FEATURES,
    STAGE_TRAIN_MODELS,
]


class PipelineOrchestrator:
    """Orchestrates the drug response prediction pipeline.

    Args:
        config: Configuration dictionary containing paths, training settings,
            model architecture, and pipeline stage settings.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.device = self._setup_device()
        self.run_id = self._generate_run_id()
        self.smiles_map: Dict[str, str] = {}
        self.drug_names: List[str] = []
        self.trained_model_paths: Dict[str, List[str]] = {}
        self.stage_timings: Dict[str, float] = {}

        self._setup_directories()
        self._load_drug_configuration()

    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        device_str = self.config.get('runtime', {}).get('device', 'cpu')
        if device_str == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    def _generate_run_id(self) -> str:
        """Generate unique run identifier."""
        run_name = self.config.get('experiment', {}).get('run_name')
        if run_name:
            return run_name
        experiment_name = self.config.get('experiment', {}).get('name', 'pipeline')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{experiment_name}_{timestamp}"

    def _setup_directories(self) -> None:
        """Create necessary directories."""
        paths = self.config.get('paths', {})
        for key in ['input_data_dir', 'intermediate_data_dir', 'output_data_dir']:
            dir_path = paths.get(key)
            if dir_path:
                Path(dir_path).mkdir(parents=True, exist_ok=True)

    def _load_drug_configuration(self) -> None:
        """Load drug names and SMILES from configuration."""
        self._load_smiles_mapping()

        main_drugs = self.config.get('drugs', {}).get('main_drugs', [])
        if not main_drugs:
            main_drugs = list(self.smiles_map.keys())

        self.drug_names = list(main_drugs)
        print(f"Configured pipeline for {len(self.drug_names)} drugs")

    def _load_smiles_mapping(self) -> None:
        """Load SMILES mapping from drugs file."""
        drugs_file = self.config.get('paths', {}).get('drugs_to_process_file')
        print(f"Drugs file: {drugs_file}")
        df_drugs = pd.read_csv(drugs_file)

        smiles_map = {}
        for _, row in df_drugs.iterrows():
            drug_name = str(row['drug_name']).replace("-", "_").replace(" ", "_")
            smiles = str(row['smiles'])
            smiles_map[drug_name] = smiles

        self.smiles_map = smiles_map
        print(f"Loaded SMILES for {len(self.smiles_map)} drugs")

    def _get_pafe_feature_dim(self) -> int:
        """Determine PAFE feature dimension from pathway graph."""
        graph_file = self.config.get('paths', {}).get('pathway_interaction_graph_file')
        if not graph_file or not Path(graph_file).exists():
            return self.config.get('model', {}).get('pafe_feature_dim', 24822)

        import pickle
        with open(graph_file, 'rb') as f:
            data = pickle.load(f)

        pathway_map = data.get('node_to_genes', {})
        all_genes = set()
        for genes in pathway_map.values():
            all_genes.update(genes)

        pafe_dim = len(all_genes) * 3
        print(f"PAFE feature dim: {pafe_dim} ({len(all_genes)} genes * 3)")
        return pafe_dim

    def _build_legacy_config_dicts(self) -> tuple:
        """Build configuration dictionaries for pipeline stages.

        Returns:
            Tuple of (paths_config, training_settings, model_arch_config, optuna_settings).
        """
        paths_config = {
            'input_data_dir': self.config['paths']['input_data_dir'],
            'intermediate_data_dir': self.config['paths']['intermediate_data_dir'],
            'output_data_dir': self.config['paths']['output_data_dir'],
            'ctrpv2_drug_response_dir': self.config['paths']['ctrpv2_drug_response_dir'],
            'gdsc0_drug_response_dir': self.config['paths']['gdsc0_drug_response_dir'],
            'gdsc2_drug_response_dir': self.config['paths']['gdsc2_drug_response_dir'],
            'genomic_lookups_dir': self.config['paths']['genomic_lookups_dir'],
            'processed_omics_data_file': self.config['paths']['processed_omics_data_file'],
            'pathway_interaction_graph_file': self.config['paths']['pathway_interaction_graph_file'],
            'precomputed_features_dir': self.config['paths']['precomputed_features_dir'],
            'shap_output_base_dir': self.config['paths'].get('shap_output_base_dir', ''),
            'pathway_genesets_filename': self.config['paths'].get('pathway_genesets_file', ''),
            'pathway_genesets_file': self.config['paths'].get('pathway_genesets_file', ''),
        }

        loss_fn_map = {
            'mse': nn.MSELoss(),
            'mae': nn.L1Loss(),
            'huber': nn.SmoothL1Loss(),
        }

        training_config = self.config.get('training', {})
        training_settings = {
            'n_folds': training_config.get('n_folds', 5),
            'clip_grad_norm': training_config.get('clip_grad_norm', 1.0),
            'num_epochs_per_drug': training_config.get('num_epochs_per_drug', 100),
            'batch_size': training_config.get('batch_size', 32),
            'scheduler_patience': training_config.get('scheduler_patience', 10),
            'scheduler_factor': training_config.get('scheduler_factor', 0.5),
            'early_stopping_patience': training_config.get('early_stopping_patience', 20),
            'response_val_col': training_config.get('response_val_col', 'ln_IC50'),
            'loss_fn': loss_fn_map.get(training_config.get('loss_function', 'mse'), nn.MSELoss()),
            'device': self.device,
            'enable_plotting': training_config.get('enable_plotting', True),
            'include_gdsc2_datasets': training_config.get('include_gdsc2_datasets', True),
            'selected_omics_type': training_config.get('selected_omics_type', 'all'),
        }

        pafe_dim = self._get_pafe_feature_dim()
        model_config = self.config.get('model', {})
        model_arch_config = {
            'pafe_feature_dim': pafe_dim,
            'fp_nbits': model_config.get('fp_nbits', 2048),
            'gnn_embedding_dim': model_config.get('gnn_embedding_dim', 128),
            'drug_embedding_dim': model_config.get('drug_embedding_dim', 128),
            'transformer_input_dim': model_config.get('transformer_input_dim', 256),
            'transformer_nhead': model_config.get('transformer_nhead', 8),
            'transformer_dim_ff': model_config.get('transformer_dim_ff', 512),
            'transformer_num_layers': model_config.get('transformer_num_layers', 4),
            'gnn_hidden_dim1': model_config.get('gnn_hidden_dim1', 512),
            'gnn_heads_l1': model_config.get('gnn_heads_l1', 8),
            'ann_hidden_dim1': model_config.get('ann_hidden_dim1', 512),
            'ann_hidden_dim2': model_config.get('ann_hidden_dim2', 256),
            'scm_hidden_dim': model_config.get('scm_hidden_dim', 128),
            'num_message_passing_steps': model_config.get('num_message_passing_steps', 3),
        }

        optuna_config = self.config.get('optuna', {})
        optuna_settings = {
            'use_optuna': optuna_config.get('use_optuna', False),
            'predetermined_seed': optuna_config.get('predetermined_seed', 42),
            'n_optuna_trials': optuna_config.get('n_optuna_trials', 50),
            'top_k_models_to_evaluate': optuna_config.get('top_k_models_to_evaluate', 3),
            'primary_metric_name_for_selection': optuna_config.get('primary_metric_name_for_selection', 'spearman'),
            'num_past_epochs_for_smoothing': optuna_config.get('num_past_epochs_for_smoothing', 5),
            'overfitting_penalty_factor': optuna_config.get('overfitting_penalty_factor', 0.1),
            'fixed_hyperparameters': optuna_config.get('fixed_hyperparameters', {}),
        }

        return paths_config, training_settings, model_arch_config, optuna_settings

    def run_pipeline(self, resume_from_stage: Optional[str] = None) -> bool:
        """Run the complete pipeline.

        Args:
            resume_from_stage: Optional stage name to resume from.

        Returns:
            True if pipeline completed successfully.
        """
        print(f"Starting pipeline run: {self.run_id}")

        start_stage = resume_from_stage or self.config.get('pipeline', {}).get('resume_from_stage')
        if start_stage:
            print(f"Resuming from stage: {start_stage}")

        success = self._execute_stages(start_stage)

        total_duration = sum(self.stage_timings.values())
        print(f"\nPipeline {'completed' if success else 'failed'}")
        print(f"Total duration: {total_duration:.2f}s")

        return success

    def _execute_stages(self, start_stage: Optional[str] = None) -> bool:
        """Execute pipeline stages.

        Args:
            start_stage: Optional stage to start from.

        Returns:
            True if all stages completed successfully.
        """
        stages_to_run = self._get_stages_to_run(start_stage)
        print(f"Executing {len(stages_to_run)} pipeline stages")

        for stage_name in stages_to_run:
            if not self.config.get('pipeline', {}).get('stages', {}).get(stage_name, True):
                print(f"Skipping disabled stage: {stage_name}")
                continue

            if not self._execute_single_stage(stage_name):
                return False

        return True

    def _get_stages_to_run(self, start_stage: Optional[str] = None) -> List[str]:
        """Get list of stages to run.

        Args:
            start_stage: Optional stage to start from.

        Returns:
            List of stage names to execute.
        """
        if start_stage is None:
            return ALL_STAGES

        if start_stage in ALL_STAGES:
            start_idx = ALL_STAGES.index(start_stage)
            return ALL_STAGES[start_idx:]

        print(f"Unknown start stage: {start_stage}, starting from beginning")
        return ALL_STAGES

    def _execute_single_stage(self, stage_name: str) -> bool:
        """Execute a single pipeline stage.

        Args:
            stage_name: Name of the stage to execute.

        Returns:
            True if stage completed successfully.
        """
        print(f"\nStarting stage: {stage_name}")
        start_time = time.time()

        success = self._run_stage_function(stage_name)

        duration = time.time() - start_time
        self.stage_timings[stage_name] = duration

        status = "completed" if success else "failed"
        print(f"Stage {stage_name} {status} ({duration:.2f}s)")

        return success

    def _run_stage_function(self, stage_name: str) -> bool:
        """Run the specific function for a pipeline stage.

        Args:
            stage_name: Name of the stage to run.

        Returns:
            True if stage completed successfully.
        """
        if stage_name == STAGE_PATHWAY_SETUP:
            return self._run_pathway_setup()

        paths_config, training_settings, model_arch_config, optuna_settings = self._build_legacy_config_dicts()

        if stage_name == STAGE_PREP_OMICS:
            return self._run_prep_omics_data(paths_config, training_settings)
        elif stage_name == STAGE_GENERATE_PAFE:
            return self._run_generate_pafe(paths_config, training_settings)
        elif stage_name == STAGE_PRECOMPUTE_FEATURES:
            return self._run_precompute_features(paths_config, training_settings)
        elif stage_name == STAGE_TRAIN_MODELS:
            return self._run_train_models(paths_config, training_settings, model_arch_config, optuna_settings)
        else:
            print(f"Unknown stage: {stage_name}")
            return False

    def _run_pathway_setup(self) -> bool:
        """Execute pathway interaction setup stage.

        Returns:
            True if successful.
        """
        pathway_config = self.config.get('pathway_setup', {})

        if not pathway_config:
            print("Warning: pathway_setup configuration not found. Using standalone config.")
            success = pathway_setup_main.run_leaf_up_pipeline()
        else:
            success = pathway_setup_main.run_leaf_up_pipeline(pathway_config)

        if success:
            print("Completed pathway interaction setup")
        else:
            print("Failed to complete pathway interaction setup")

        return success

    def _run_prep_omics_data(self, paths_config: Dict, training_settings: Dict) -> bool:
        """Execute omics data preparation stage.

        Args:
            paths_config: Paths configuration dictionary.
            training_settings: Training settings dictionary.

        Returns:
            True if successful.
        """
        all_drugs = self.drug_names + (self.config.get('drugs', {}).get('leave_out_drugs', []) or [])
        include_gdsc2 = training_settings.get('include_gdsc2_datasets', False)

        prep_omics_data.main(all_drugs, paths_config, include_gdsc2_datasets=include_gdsc2)
        print("Completed omics data preparation")
        return True

    def _run_generate_pafe(self, paths_config: Dict, training_settings: Dict) -> bool:
        """Execute PAFE generation stage.

        Args:
            paths_config: Paths configuration dictionary.
            training_settings: Training settings dictionary.

        Returns:
            True if successful.
        """
        all_drugs = self.drug_names + (self.config.get('drugs', {}).get('leave_out_drugs', []) or [])
        include_gdsc2 = training_settings.get('include_gdsc2_datasets', False)

        generate_pafe.main(all_drugs, paths_config, include_gdsc2_datasets=include_gdsc2)
        print("Completed PAFE generation")
        return True

    def _run_precompute_features(self, paths_config: Dict, training_settings: Dict) -> bool:
        """Execute feature precomputation stage.

        Args:
            paths_config: Paths configuration dictionary.
            training_settings: Training settings dictionary.

        Returns:
            True if successful.
        """
        all_drugs = self.drug_names + (self.config.get('drugs', {}).get('leave_out_drugs', []) or [])
        include_gdsc2 = training_settings.get('include_gdsc2_datasets', False)

        precompute_features.main(all_drugs, self.smiles_map, paths_config, include_gdsc2_datasets=include_gdsc2)
        print("Completed feature precomputation")
        return True

    def _run_train_models(
        self,
        paths_config: Dict,
        training_settings: Dict,
        model_arch_config: Dict,
        optuna_settings: Dict
    ) -> bool:
        """Execute model training stage.

        Args:
            paths_config: Paths configuration dictionary.
            training_settings: Training settings dictionary.
            model_arch_config: Model architecture configuration dictionary.
            optuna_settings: Optuna optimization settings dictionary.

        Returns:
            True if successful.
        """
        experiment_name = self.config.get('experiment', {}).get('name', 'experiment')
        base_checkpoint_dir = Path(paths_config['output_data_dir']) / experiment_name / self.run_id
        base_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_model_paths_map = train_transformer_optuna_main(
            all_drugs_to_train=self.drug_names,
            base_checkpoint_dir_root=str(base_checkpoint_dir),
            paths_config=paths_config,
            training_settings=training_settings,
            model_arch_config=model_arch_config,
            optuna_settings=optuna_settings,
            enable_plotting=training_settings.get('enable_plotting', True),
            include_gdsc2_datasets=training_settings.get('include_gdsc2_datasets', True)
        )

        if not best_model_paths_map:
            print("Training failed - no best model paths returned")
            return False

        print(f"Training completed. Best model paths: {best_model_paths_map}")

        output_dir = Path(paths_config['output_data_dir'])
        best_model_info_file = output_dir / "best_model_paths.json"
        with open(best_model_info_file, 'w') as f:
            json.dump(best_model_paths_map, f, indent=2)
        print(f"Saved best model paths to: {best_model_info_file}")

        self._store_trained_model_paths(best_model_paths_map)
        return True

    def _store_trained_model_paths(self, best_model_paths_map: Dict) -> None:
        """Store trained model paths.

        Args:
            best_model_paths_map: Dictionary mapping drug names to model paths.
        """
        for drug_name, model_paths in best_model_paths_map.items():
            if drug_name in self.drug_names:
                if drug_name not in self.trained_model_paths:
                    self.trained_model_paths[drug_name] = []

                if isinstance(model_paths, list):
                    for path in model_paths:
                        if path not in self.trained_model_paths[drug_name]:
                            self.trained_model_paths[drug_name].append(path)
                else:
                    if model_paths not in self.trained_model_paths[drug_name]:
                        self.trained_model_paths[drug_name].append(model_paths)

        print(f"Stored model paths for {len(self.trained_model_paths)} drugs")
    