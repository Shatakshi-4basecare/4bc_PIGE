"""Precompute feature matrices for drug response prediction.

Combines genomic data (mutations, CNA, RNA) with pathway features and drug fingerprints.
Precomputes and caches feature tensors for efficient model training.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union

import pandas as pd
import torch
from tqdm import tqdm

from .feature_generator import FeatureGenerator
from .cancer_drug_dataset import DataConstants


def load_pathway_graph_and_get_gene_sets(file_path: Union[str, Path]) -> Dict[str, Set[str]]:
    """Load pathway interaction graph and extract gene sets.

    Args:
        file_path: Path to pathway graph pickle file.

    Returns:
        Dictionary mapping pathway names to gene sets.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    node_to_genes = data['node_to_genes']

    for pathway_name, genes in node_to_genes.items():
        if not isinstance(genes, set):
            node_to_genes[pathway_name] = set(genes)

    print(f"Loaded {len(node_to_genes)} pathways")
    return node_to_genes


def load_genomic_data(genomic_lookups_path: Union[str, Path]) -> Dict[str, Any]:
    """Load genomic lookup data from pickle file.

    Args:
        genomic_lookups_path: Path to genomic lookups pickle file.

    Returns:
        Dictionary containing genomic data maps.
    """
    with open(genomic_lookups_path, 'rb') as f:
        genomic_lookups = pickle.load(f)

    loaded_data = {
        'cell_mutation_map': genomic_lookups['cell_mutation_map'],
        'cell_cna_map': genomic_lookups['cell_cna_map'],
        'G_universal': genomic_lookups['G_universal'],
        'cell_rna_map': genomic_lookups.get('cell_rna_map', {})
    }

    return loaded_data


def load_response_data(response_data_path: Union[str, Path]) -> pd.DataFrame:
    """Load drug response data from CSV file.

    Args:
        response_data_path: Path to response data CSV file.

    Returns:
        DataFrame containing response data.
    """
    df_response = pd.read_csv(response_data_path)
    print(f"Loaded response data with {len(df_response)} samples")
    return df_response


def load_data_sources(
    genomic_lookups_path: Union[str, Path],
    response_data_path: Union[str, Path]
) -> Dict[str, Any]:
    """Load all required data sources for feature generation.

    Args:
        genomic_lookups_path: Path to genomic lookups pickle file.
        response_data_path: Path to response data CSV file.

    Returns:
        Dictionary containing all loaded data.
    """
    genomic_data = load_genomic_data(genomic_lookups_path)
    response_data = load_response_data(response_data_path)

    return {
        **genomic_data,
        'df_response': response_data
    }


def prepare_sample_list(
    df_response: pd.DataFrame,
    drug_name: str
) -> List[Tuple[str, str, float]]:
    """Prepare list of samples from response DataFrame.

    Args:
        df_response: DataFrame containing response data.
        drug_name: Name of the drug being processed.

    Returns:
        List of tuples (cell_id, drug_name, response_value).
    """
    samples = []

    for _, row in df_response.iterrows():
        cell_id = row[DataConstants.MODEL_ID_COLUMN]
        response_value = float(row[DataConstants.RESPONSE_COLUMN])
        samples.append((cell_id, drug_name, response_value))

    return samples


def save_precomputed_features(
    pafe_tensor: torch.Tensor,
    fp_tensor: torch.Tensor,
    label_tensor: torch.Tensor,
    output_filepath: Path
) -> None:
    """Save precomputed features to file.

    Args:
        pafe_tensor: PAFE feature tensor.
        fp_tensor: Drug fingerprint tensor.
        label_tensor: Response label tensor.
        output_filepath: Path to save the features.
    """
    data_to_save = {
        'pafe_features': pafe_tensor.to_sparse(),
        'drug_fingerprint': fp_tensor,
        'label': label_tensor
    }
    torch.save(data_to_save, output_filepath)


def process_single_sample(
    cell_id: str,
    drug_name: str,
    response_value: float,
    feature_generator: FeatureGenerator,
    output_dir: Path
) -> str:
    """Process a single sample and save precomputed features.

    Args:
        cell_id: Cell line identifier.
        drug_name: Drug name.
        response_value: Drug response value.
        feature_generator: FeatureGenerator instance.
        output_dir: Output directory for saved features.

    Returns:
        Status string indicating result.
    """
    output_filename = f"pafe_fp_{cell_id}_{drug_name}.pt"
    output_filepath = output_dir / output_filename

    if output_filepath.exists():
        return "skipped_existing"

    pafe_tensor, fp_tensor = feature_generator.generate_features(cell_id, drug_name)

    if pafe_tensor is None or fp_tensor is None:
        return "error_generating_features"

    # Skip if all features are zero
    if not torch.any(pafe_tensor):
        return "skipped_zero_pafe"

    label_tensor = torch.tensor([response_value], dtype=DataConstants.TENSOR_DTYPE)

    save_precomputed_features(pafe_tensor, fp_tensor, label_tensor, output_filepath)

    return "processed"


def precompute_and_save_all_samples(
    samples_to_process: List[Tuple[str, str, float]],
    precomputed_dir: Path,
    pathway_dict: Dict[str, Set[str]],
    ordered_pathway_names: List[str],
    g_universal_list: List[str],
    cell_mutation_map: Dict[str, Set[str]],
    cell_cna_map: Dict[Tuple[str, str], float],
    cell_rna_map: Dict[Tuple[str, str], float],
    smiles_map: Dict[str, str],
    npvae_embeddings_map: Dict[str, List[float]]
) -> Dict[str, int]:
    """Precompute and save features for all samples.

    Args:
        samples_to_process: List of (cell_id, drug_name, response_value) tuples.
        precomputed_dir: Directory to save precomputed features.
        pathway_dict: Dictionary mapping pathway names to gene sets.
        ordered_pathway_names: Ordered list of pathway names.
        g_universal_list: List of all genes in feature space.
        cell_mutation_map: Cell mutation data.
        cell_cna_map: Cell CNA data.
        cell_rna_map: Cell RNA data.
        smiles_map: Drug SMILES mapping.
        npvae_embeddings_map: Drug embedding mapping.

    Returns:
        Dictionary with processing statistics.
    """
    precomputed_dir.mkdir(parents=True, exist_ok=True)

    feature_generator = FeatureGenerator(
        pathway_dict,
        ordered_pathway_names,
        g_universal_list,
        cell_mutation_map,
        cell_cna_map,
        cell_rna_map,
        smiles_map,
        npvae_embeddings_map
    )

    print(f"Feature dimensions: {feature_generator.n_genes_in_features} genes, "
          f"{feature_generator.total_features_per_pathway} features per pathway")

    counts = {
        "processed": 0,
        "skipped_existing": 0,
        "skipped_zero_pafe": 0,
        "errors": 0,
        "error_generating_features": 0
    }

    for cell_id, drug_name, response_value in tqdm(samples_to_process, desc="Precomputing Features"):
        try:
            result = process_single_sample(
                cell_id, drug_name, response_value,
                feature_generator,
                precomputed_dir
            )
            counts[result] += 1

        except Exception as e:
            print(f"Error processing {cell_id} for {drug_name}: {e}")
            counts["errors"] += 1

    print(f"Processed: {counts['processed']}, Skipped existing: {counts['skipped_existing']}, "
          f"Skipped zero: {counts['skipped_zero_pafe']}, Errors: {counts['errors'] + counts['error_generating_features']}")

    return counts


def create_dataset_config(name: str, subdir: str, response_dir: str,
                         genomic_template: str, is_true_test: bool = False) -> Dict[str, Any]:
    """Create dataset processing configuration.

    Args:
        name: Dataset name.
        subdir: Subdirectory name for output.
        response_dir: Response data directory name.
        genomic_template: Genomic lookup path template.
        is_true_test: Whether this is a true test dataset.

    Returns:
        Dataset configuration dictionary.
    """
    return {
        "name": name,
        "subdir_name": subdir,
        "response_dir_name": response_dir,
        "genomic_lookup_path_template": genomic_template,
        "is_true_test": is_true_test
    }


def get_dataset_processing_configs(
    intermediate_data_dir: Path,
    include_gdsc2_datasets: bool = True
) -> List[Dict[str, Any]]:
    """Get dataset processing configurations.

    Args:
        intermediate_data_dir: Path to intermediate data directory.
        include_gdsc2_datasets: Whether to include GDSC2 datasets.

    Returns:
        List of dataset configuration dictionaries.
    """
    genomic_lookups_dir = intermediate_data_dir / DataConstants.GENOMIC_LOOKUPS_SUBDIR

    configs = [
        create_dataset_config(
            "CTRPv2",
            DataConstants.PRECOMPUTED_SUBDIR_CTRP,
            DataConstants.CTRP_DRUG_RESPONSE_DIR,
            str(genomic_lookups_dir / 'ctrp_{drug_name}.pkl')
        ),
        create_dataset_config(
            "GDSC0",
            DataConstants.PRECOMPUTED_SUBDIR_GDSC0,
            DataConstants.GDSC0_DRUG_RESPONSE_DIR,
            str(genomic_lookups_dir / 'gdsc0_{drug_name}.pkl')
        ),
        create_dataset_config(
            "GDSC0_true_test",
            DataConstants.PRECOMPUTED_SUBDIR_GDSC0_TRUE_TEST,
            DataConstants.GDSC0_DRUG_RESPONSE_DIR,
            str(genomic_lookups_dir / 'gdsc0_{drug_name}.pkl'),
            is_true_test=True
        )
    ]

    if include_gdsc2_datasets:
        configs.extend([
            create_dataset_config(
                "GDSC2",
                DataConstants.PRECOMPUTED_SUBDIR_GDSC2,
                DataConstants.GDSC2_DRUG_RESPONSE_DIR,
                str(genomic_lookups_dir / 'gdsc2_{drug_name}.pkl')
            ),
            create_dataset_config(
                "GDSC2_true_test",
                DataConstants.PRECOMPUTED_SUBDIR_GDSC2_TRUE_TEST,
                DataConstants.GDSC2_DRUG_RESPONSE_DIR,
                str(genomic_lookups_dir / 'gdsc2_{drug_name}.pkl'),
                is_true_test=True
            )
        ])

    return configs


def load_npvae_embeddings(npvae_embeddings_path: Path) -> Dict[str, List[float]]:
    """Load NpVae drug embeddings from pickle file.

    Args:
        npvae_embeddings_path: Path to NpVae embeddings file.

    Returns:
        Dictionary mapping SMILES strings to embedding vectors.
    """
    with open(npvae_embeddings_path, 'rb') as f:
        npvae_embeddings_map = pickle.load(f)
    print(f"Loaded NpVae embeddings for {len(npvae_embeddings_map)} compounds")
    return npvae_embeddings_map


def process_drug_dataset_combination(
    drug_name: str,
    config: Dict[str, Any],
    intermediate_data_dir: Path,
    pathway_dict: Dict[str, Set[str]],
    ordered_pathway_names: List[str],
    smiles_map: Dict[str, str],
    npvae_embeddings_map: Dict[str, List[float]],
    precomputed_features_base_dir: Path
) -> None:
    """Process a single drug-dataset combination.

    Args:
        drug_name: Name of the drug to process.
        config: Dataset configuration.
        intermediate_data_dir: Path to intermediate data directory.
        pathway_dict: Pathway gene mapping.
        ordered_pathway_names: Ordered pathway names.
        smiles_map: Drug SMILES mapping.
        npvae_embeddings_map: Drug embedding mapping.
        precomputed_features_base_dir: Base directory for precomputed features.
    """
    version_name = config["name"]
    response_dir_name = config["response_dir_name"]
    genomic_lookup_template = config["genomic_lookup_path_template"]
    dataset_subdir = config["subdir_name"]
    is_true_test = config["is_true_test"]

    response_data_dir = intermediate_data_dir / response_dir_name
    if is_true_test:
        current_drug_response_path = response_data_dir / DataConstants.TRUE_TEST_SUBDIR / f'{drug_name}.csv'
    else:
        current_drug_response_path = response_data_dir / f'{drug_name}.csv'

    current_genomic_lookups_path = Path(genomic_lookup_template.format(drug_name=drug_name))

    precomputed_output_dir = precomputed_features_base_dir / dataset_subdir / drug_name
    precomputed_output_dir.mkdir(parents=True, exist_ok=True)

    dataset_type = 'True-Test' if is_true_test else 'Standard'
    print(f"Loading {version_name} data for {drug_name} ({dataset_type})")

    if not current_drug_response_path.exists() or not current_genomic_lookups_path.exists():
        print(f"Skipping {drug_name} ({version_name}): missing files")
        return

    current_data_sources = load_data_sources(
        current_genomic_lookups_path,
        current_drug_response_path
    )
    current_samples = prepare_sample_list(
        current_data_sources['df_response'],
        drug_name
    )

    print(f"Processing {len(current_samples)} {version_name} samples for {drug_name} ({dataset_type})")

    precompute_and_save_all_samples(
        current_samples,
        precomputed_output_dir,
        pathway_dict,
        ordered_pathway_names,
        current_data_sources['G_universal'],
        current_data_sources['cell_mutation_map'],
        current_data_sources['cell_cna_map'],
        current_data_sources['cell_rna_map'],
        smiles_map,
        npvae_embeddings_map
    )


def main(
    drug_names_list: List[str],
    smiles_master_map: Dict[str, str],
    paths_config: Dict[str, Union[str, Path]],
    include_gdsc2_datasets: bool = True
) -> None:
    """Precompute features for all drugs and datasets.

    Args:
        drug_names_list: List of drug names to process.
        smiles_master_map: Dictionary mapping drug names to SMILES strings.
        paths_config: Configuration containing required file paths.
        include_gdsc2_datasets: Whether to include GDSC2 datasets.
    """
    input_data_dir = Path(paths_config['input_data_dir'])
    intermediate_data_dir = Path(paths_config['intermediate_data_dir'])
    pathway_interaction_graph_file = Path(paths_config['pathway_interaction_graph_file'])

    npvae_embeddings_path = input_data_dir / DataConstants.NPVAE_EMBEDDINGS_FILE
    precomputed_features_base_dir = intermediate_data_dir / DataConstants.PRECOMPUTED_FEATURES_SUBDIR

    print("Loading core data components.")

    npvae_embeddings_map = load_npvae_embeddings(npvae_embeddings_path)
    pathway_dict = load_pathway_graph_and_get_gene_sets(pathway_interaction_graph_file)
    ordered_pathway_names = sorted(pathway_dict.keys())

    print(f"Loaded {len(ordered_pathway_names)} ordered pathways")

    dataset_configs = get_dataset_processing_configs(
        intermediate_data_dir,
        include_gdsc2_datasets
    )

    for drug_name in drug_names_list:
        print(f"Processing drug: {drug_name}")

        for config in dataset_configs:
            try:
                process_drug_dataset_combination(
                    drug_name,
                    config,
                    intermediate_data_dir,
                    pathway_dict,
                    ordered_pathway_names,
                    smiles_master_map,
                    npvae_embeddings_map,
                    precomputed_features_base_dir
                )
            except Exception as e:
                print(f"Failed to process {drug_name} with {config['name']}: {e}")

    print("Finished precomputing features for all drugs.")
