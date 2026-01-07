"""Pathway Activity Feature Engineering (PAFE) module.

Generates pathway-based features for multi-omics drug response prediction.
Processes genomic data (mutations, CNA, RNA) and creates pathway-level features.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union, Any

import numpy as np
import pandas as pd


class DataConstants:
    """File and column name constants."""

    OMICS_DATA_FILENAME = 'omics_data.pkl'
    GENOMIC_LOOKUPS_SUBDIR = 'genomic_lookups'

    CTRP_DRUG_RESPONSE_DIR = 'CTRPv2_drug_response_data'
    GDSC0_DRUG_RESPONSE_DIR = 'GDSC0_drug_response_data'
    GDSC2_DRUG_RESPONSE_DIR = 'GDSC2_drug_response_data'

    MODEL_ID_COLUMN = 'ModelID'
    GENE_COLUMN = 'Gene'
    CNA_VALUE_COLUMN = 'CNA_value'
    RNA_VALUE_COLUMN = 'RNA_value'

    MUTATION_PREFIX = 'Mut_'
    CNA_PREFIX = 'CNA_'
    RNA_PREFIX = 'RNA_'

    LOOKUP_FILE_TEMPLATE_CTRP = 'ctrp_{drug_name}.pkl'
    LOOKUP_FILE_TEMPLATE_GDSC0 = 'gdsc0_{drug_name}.pkl'
    LOOKUP_FILE_TEMPLATE_GDSC2 = 'gdsc2_{drug_name}.pkl'

    DEFAULT_MISSING_VALUE = 0
    DEFAULT_DTYPE = np.float32


def load_pathway_graph_and_get_gene_sets(file_path: Union[str, Path]) -> Dict[str, List[str]]:
    """Load pathway interaction graph and extract gene sets.

    Args:
        file_path: Path to pathway graph pickle file.

    Returns:
        Dictionary mapping pathway names to gene lists.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    pathway_map = data.get('node_to_genes')
    if pathway_map is None:
        raise ValueError("Could not find 'node_to_genes' in pathway graph file")

    print(f"Loaded {len(pathway_map)} pathways from {Path(file_path).name}")
    return pathway_map


def load_processed_omics_data(omics_data_filepath: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load processed omics data from pickle file.

    Args:
        omics_data_filepath: Path to omics data pickle file.

    Returns:
        Tuple of (mutation_df, cna_df, rna_df).
    """
    with open(omics_data_filepath, 'rb') as f:
        omics_data = pickle.load(f)

    df_mut_filtered = omics_data['df_mut_filtered']
    df_cna_long = omics_data['df_cna_long']
    df_rna_long = omics_data['df_rna_long']

    print("Loaded mutation, CNA, and RNA data")
    return df_mut_filtered, df_cna_long, df_rna_long


def load_target_cell_lines(drug_response_file: Union[str, Path]) -> np.ndarray:
    """Load target cell lines from drug response file.

    Args:
        drug_response_file: Path to drug response CSV file.

    Returns:
        Array of unique cell line IDs.
    """
    df_response = pd.read_csv(drug_response_file)
    target_cell_lines = df_response[DataConstants.MODEL_ID_COLUMN].unique()
    print(f"Loaded {len(target_cell_lines)} target cell lines")
    return target_cell_lines


def define_universal_genes_and_features(pathway_dict: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    """Define universal gene set and feature columns.

    Args:
        pathway_dict: Dictionary mapping pathway names to gene lists.

    Returns:
        Tuple of (universal_genes_list, feature_columns_list).
    """
    all_genes = []
    for genes in pathway_dict.values():
        all_genes.extend(genes)

    G_universal = sorted(list(set(all_genes)))
    print(f"Found {len(G_universal)} unique genes in {len(pathway_dict)} pathways")

    mut_cols = [DataConstants.MUTATION_PREFIX + gene for gene in G_universal]
    cna_cols = [DataConstants.CNA_PREFIX + gene for gene in G_universal]
    rna_cols = [DataConstants.RNA_PREFIX + gene for gene in G_universal]

    feature_columns = mut_cols + cna_cols + rna_cols
    print(f"Total feature columns: {len(feature_columns)}")

    return G_universal, feature_columns


def create_mutation_lookup(df_mut: pd.DataFrame, target_cell_lines: np.ndarray) -> Dict[str, Set[str]]:
    """Create lookup dictionary for mutations.

    Args:
        df_mut: Mutation DataFrame.
        target_cell_lines: Array of target cell line IDs.

    Returns:
        Dictionary mapping cell line IDs to sets of mutated genes.
    """
    is_target = df_mut[DataConstants.MODEL_ID_COLUMN].isin(target_cell_lines)
    df_mut_target = df_mut[is_target].copy()

    cell_mutation_map = {}
    for model_id in df_mut_target[DataConstants.MODEL_ID_COLUMN].unique():
        mutations_for_cell = df_mut_target[df_mut_target[DataConstants.MODEL_ID_COLUMN] == model_id]
        genes_mutated = set(mutations_for_cell[DataConstants.GENE_COLUMN])
        cell_mutation_map[model_id] = genes_mutated

    print(f"Created mutation lookup for {len(cell_mutation_map)} cell lines")
    return cell_mutation_map


def create_cna_lookup(df_cna: pd.DataFrame, target_cell_lines: np.ndarray) -> Dict[Tuple[str, str], float]:
    """Create lookup dictionary for CNA values.

    Args:
        df_cna: CNA DataFrame.
        target_cell_lines: Array of target cell line IDs.

    Returns:
        Dictionary mapping (cell_line_id, gene) tuples to CNA values.
    """
    is_target = df_cna[DataConstants.MODEL_ID_COLUMN].isin(target_cell_lines)
    df_cna_target = df_cna[is_target].copy()

    if df_cna_target.empty:
        print("No CNA data found for target cell lines")
        return {}

    cell_cna_map = df_cna_target.set_index([DataConstants.MODEL_ID_COLUMN, DataConstants.GENE_COLUMN])[DataConstants.CNA_VALUE_COLUMN].to_dict()
    print(f"Created CNA lookup with {len(cell_cna_map)} entries")
    return cell_cna_map


def create_rna_lookup(df_rna: pd.DataFrame, target_cell_lines: np.ndarray) -> Dict[Tuple[str, str], float]:
    """Create lookup dictionary for RNA expression values.

    Args:
        df_rna: RNA expression DataFrame.
        target_cell_lines: Array of target cell line IDs.

    Returns:
        Dictionary mapping (cell_line_id, gene) tuples to RNA values.
    """
    is_target = df_rna[DataConstants.MODEL_ID_COLUMN].isin(target_cell_lines)
    df_rna_target = df_rna[is_target].copy()

    if df_rna_target.empty:
        print("No RNA data found for target cell lines")
        return {}

    cell_rna_map = df_rna_target.set_index([DataConstants.MODEL_ID_COLUMN, DataConstants.GENE_COLUMN])[DataConstants.RNA_VALUE_COLUMN].to_dict()
    print(f"Created RNA lookup with {len(cell_rna_map)} entries")
    return cell_rna_map


def preprocess_genomic_data_to_lookups(
    df_mut_filtered: pd.DataFrame,
    df_cna_long: pd.DataFrame,
    df_rna_long: pd.DataFrame,
    target_cell_lines: np.ndarray
) -> Tuple[Dict[str, Set[str]], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    """Preprocess genomic data into lookup dictionaries.

    Args:
        df_mut_filtered: Filtered mutation DataFrame.
        df_cna_long: CNA DataFrame in long format.
        df_rna_long: RNA expression DataFrame in long format.
        target_cell_lines: Array of target cell line IDs.

    Returns:
        Tuple of (mutation_lookup, cna_lookup, rna_lookup).
    """
    print("Creating genomic data lookups...")

    cell_mutation_map = create_mutation_lookup(df_mut_filtered, target_cell_lines)
    cell_cna_map = create_cna_lookup(df_cna_long, target_cell_lines)
    cell_rna_map = create_rna_lookup(df_rna_long, target_cell_lines)

    print("Completed genomic data lookup creation")
    return cell_mutation_map, cell_cna_map, cell_rna_map


def create_pafe_features_for_cell_line(
    cell_line_id: str,
    pathway_dict: Dict[str, List[str]],
    G_universal: List[str],
    feature_columns: List[str],
    cell_mutation_map: Dict[str, Set[str]],
    cell_cna_map: Dict[Tuple[str, str], float],
    cell_rna_map: Dict[Tuple[str, str], float]
) -> pd.DataFrame:
    """Create PAFE feature matrix for a single cell line.

    Args:
        cell_line_id: ID of the cell line.
        pathway_dict: Dictionary mapping pathway names to gene lists.
        G_universal: List of all genes in feature space.
        feature_columns: List of all feature column names.
        cell_mutation_map: Mutation lookup dictionary.
        cell_cna_map: CNA lookup dictionary.
        cell_rna_map: RNA lookup dictionary.

    Returns:
        DataFrame with pathways as rows and features as columns.
    """
    pathway_names = list(pathway_dict.keys())

    output_df = pd.DataFrame(
        DataConstants.DEFAULT_MISSING_VALUE,
        index=pathway_names,
        columns=feature_columns,
        dtype=DataConstants.DEFAULT_DTYPE
    )

    mutated_genes = cell_mutation_map.get(cell_line_id, set())

    for pathway_name in pathway_names:
        member_genes = pathway_dict[pathway_name]

        for gene in member_genes:
            if gene in G_universal:
                mut_col_name = DataConstants.MUTATION_PREFIX + gene
                if gene in mutated_genes:
                    output_df.loc[pathway_name, mut_col_name] = 1

                cna_col_name = DataConstants.CNA_PREFIX + gene
                cna_value = cell_cna_map.get((cell_line_id, gene), DataConstants.DEFAULT_MISSING_VALUE)
                output_df.loc[pathway_name, cna_col_name] = DataConstants.DEFAULT_DTYPE(cna_value)

                rna_col_name = DataConstants.RNA_PREFIX + gene
                rna_value = cell_rna_map.get((cell_line_id, gene), DataConstants.DEFAULT_MISSING_VALUE)
                output_df.loc[pathway_name, rna_col_name] = DataConstants.DEFAULT_DTYPE(rna_value)

    return output_df


def save_genomic_lookups(
    cell_mutation_map: Dict[str, Set[str]],
    cell_cna_map: Dict[Tuple[str, str], float],
    cell_rna_map: Dict[Tuple[str, str], float],
    G_universal: List[str],
    save_filepath: Union[str, Path]
) -> None:
    """Save genomic lookup dictionaries to pickle file.

    Args:
        cell_mutation_map: Mutation lookup dictionary.
        cell_cna_map: CNA lookup dictionary.
        cell_rna_map: RNA lookup dictionary.
        G_universal: List of universal genes.
        save_filepath: Path to save the lookup data.
    """
    genomic_lookups = {
        'cell_mutation_map': cell_mutation_map,
        'cell_cna_map': cell_cna_map,
        'cell_rna_map': cell_rna_map,
        'G_universal': G_universal
    }

    save_path = Path(save_filepath)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'wb') as f:
        pickle.dump(genomic_lookups, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved genomic lookup data to {save_path}")


def setup_dataset_configurations(
    intermediate_data_dir: Path,
    include_gdsc2_datasets: bool = True
) -> List[Dict[str, Union[str, Path]]]:
    """Setup configuration for different drug response datasets.

    Args:
        intermediate_data_dir: Path to intermediate data directory.
        include_gdsc2_datasets: Whether to include GDSC2 datasets.

    Returns:
        List of dataset configuration dictionaries.
    """
    configs = [
        {
            "version": "ctrpv2",
            "response_base_path": intermediate_data_dir / DataConstants.CTRP_DRUG_RESPONSE_DIR,
            "lookup_file_template": DataConstants.LOOKUP_FILE_TEMPLATE_CTRP
        },
        {
            "version": "GDSC0",
            "response_base_path": intermediate_data_dir / DataConstants.GDSC0_DRUG_RESPONSE_DIR,
            "lookup_file_template": DataConstants.LOOKUP_FILE_TEMPLATE_GDSC0
        }
    ]

    if include_gdsc2_datasets:
        configs.append({
            "version": "GDSC2",
            "response_base_path": intermediate_data_dir / DataConstants.GDSC2_DRUG_RESPONSE_DIR,
            "lookup_file_template": DataConstants.LOOKUP_FILE_TEMPLATE_GDSC2
        })

    return configs


def process_single_drug_dataset(
    drug_name: str,
    config: Dict[str, Union[str, Path]],
    genomic_lookups_dir: Path,
    df_mut_filtered: pd.DataFrame,
    df_cna_long: pd.DataFrame,
    df_rna_long: pd.DataFrame,
    G_universal: List[str]
) -> None:
    """Process a single drug dataset and save genomic lookups.

    Args:
        drug_name: Name of the drug.
        config: Dataset configuration dictionary.
        genomic_lookups_dir: Directory to save lookup files.
        df_mut_filtered: Filtered mutation DataFrame.
        df_cna_long: CNA DataFrame in long format.
        df_rna_long: RNA expression DataFrame in long format.
        G_universal: List of universal genes.
    """
    version = config["version"]
    response_base_path = Path(config["response_base_path"])
    lookup_file_template = config["lookup_file_template"]

    print(f"Processing {version} for {drug_name}")

    drug_response_file = response_base_path / f'{drug_name}.csv'
    if not drug_response_file.exists():
        print(f"Response file not found at {drug_response_file}, skipping")
        return

    lookup_filename = lookup_file_template.format(drug_name=drug_name)
    lookup_file_path = genomic_lookups_dir / lookup_filename

    target_cell_lines = load_target_cell_lines(drug_response_file)
    cell_mutation_map, cell_cna_map, cell_rna_map = preprocess_genomic_data_to_lookups(
        df_mut_filtered, df_cna_long, df_rna_long, target_cell_lines
    )

    save_genomic_lookups(
        cell_mutation_map, cell_cna_map, cell_rna_map, G_universal, lookup_file_path
    )


def main(drug_names_list: List[str], paths_config: Dict[str, Any], include_gdsc2_datasets: bool = True) -> None:
    """Generate PAFE features for specified drugs.

    Args:
        drug_names_list: List of drug names to process.
        paths_config: Configuration dictionary with required paths.
        include_gdsc2_datasets: Whether to include GDSC2 datasets.
    """
    print("Starting PAFE generation pipeline")

    intermediate_data_dir = Path(paths_config['intermediate_data_dir'])
    pathway_interaction_graph_file = Path(paths_config['pathway_interaction_graph_file'])

    genomic_lookups_dir = intermediate_data_dir / DataConstants.GENOMIC_LOOKUPS_SUBDIR
    genomic_lookups_dir.mkdir(parents=True, exist_ok=True)
    omics_data_file = intermediate_data_dir / DataConstants.OMICS_DATA_FILENAME

    print("Loading omics and pathway data")

    df_mut_filtered, df_cna_long, df_rna_long = load_processed_omics_data(omics_data_file)
    pathway_dict = load_pathway_graph_and_get_gene_sets(pathway_interaction_graph_file)
    G_universal, feature_columns = define_universal_genes_and_features(pathway_dict)

    print("Finished loading common data")

    dataset_configs = setup_dataset_configurations(intermediate_data_dir, include_gdsc2_datasets)

    for drug_name in drug_names_list:
        print(f"Processing drug: {drug_name}")

        for config in dataset_configs:
            try:
                process_single_drug_dataset(
                    drug_name, config, genomic_lookups_dir,
                    df_mut_filtered, df_cna_long, df_rna_long, G_universal
                )
            except Exception as e:
                print(f"Failed to process {config['version']} for {drug_name}: {e}")
                continue

    print("PAFE generation pipeline completed")
