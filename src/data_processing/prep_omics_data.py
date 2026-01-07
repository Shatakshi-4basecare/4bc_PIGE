"""Omics data preparation for drug response prediction.

Handles preprocessing of mutation, CNA, RNA expression, and drug response data
from CTRPv2, GDSC0, and GDSC2 datasets. Note that the GDSC0 is the GDSC2 dataset but in a different format.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union, Any

import pandas as pd
from tqdm import tqdm

class DataConstants:
    """File and column name constants."""

    CTRP_DRUG_RESPONSE_SUBDIR = 'CTRPv2_drug_response_data'
    GDSC0_DRUG_RESPONSE_SUBDIR = 'GDSC0_drug_response_data'
    GDSC2_DRUG_RESPONSE_SUBDIR = 'GDSC2_drug_response_data'
    TRUE_TEST_SUBDIR = 'true_test'

    MUTATION_FILENAME = 'OmicsSomaticMutations.csv'
    CNA_FILENAME = 'OmicsCNGene.csv'
    METADATA_FILENAME = 'Model.csv'
    RNA_FILENAME = 'OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv'
    OMICS_DATA_FILENAME = 'omics_data.pkl'
    GDSC2_FITTED_DOSE_RESPONSE_FILENAME = 'GDSC2_fitted_dose_response_27Oct23.csv'
    SUMMARY_FILENAME = 'drug_cell_line_counts_summary.csv'

    MODEL_ID_COL = 'ModelID'
    CELL_LINE_NAME_COL = 'CellLineName'
    SANGER_MODEL_ID_COL = 'Sanger_Model_ID'
    GENE_COL = 'Gene'
    AAC_COL = 'aac'

    METADATA_COLS = [MODEL_ID_COL, CELL_LINE_NAME_COL, SANGER_MODEL_ID_COL]


def load_pathway_graph_and_get_gene_sets(file_path: Union[str, Path]) -> Dict[str, Set[str]]:
    """Load pathway graph from pickle file.

    Args:
        file_path: Path to pathway graph pickle file.

    Returns:
        Dictionary mapping pathway IDs to gene sets.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    pathway_map = data.get('node_to_genes')
    if pathway_map is None:
        raise ValueError("Could not find 'node_to_genes' in pathway graph file")

    print(f"Loaded {len(pathway_map)} pathways from {Path(file_path).name}")
    return pathway_map


def load_and_prepare_metadata(metadata_file_path: Union[str, Path]) -> pd.DataFrame:
    """Load and filter metadata file.

    Args:
        metadata_file_path: Path to metadata CSV file.

    Returns:
        Filtered metadata DataFrame.
    """
    df_meta = pd.read_csv(metadata_file_path)
    print(f"Loaded metadata: {df_meta.shape[0]} rows")

    df_meta_filtered = df_meta[DataConstants.METADATA_COLS].copy()
    df_meta_filtered.dropna(subset=[DataConstants.MODEL_ID_COL], inplace=True)

    print(f"Filtered metadata: {df_meta_filtered.shape[0]} rows")
    return df_meta_filtered


def load_and_filter_mutations(mutation_file_path: Union[str, Path]) -> Tuple[pd.DataFrame, List[str]]:
    """Load and filter mutation data.

    Args:
        mutation_file_path: Path to mutation CSV file.

    Returns:
        Tuple of (filtered mutation DataFrame, list of cell line IDs).
    """
    df_mut = pd.read_csv(mutation_file_path)
    print(f"Loaded mutations: {df_mut.shape[0]} rows")

    required_cols = [DataConstants.MODEL_ID_COL, 'HugoSymbol']
    df_mut_filtered = df_mut[required_cols].rename(columns={'HugoSymbol': DataConstants.GENE_COL})

    mut_cell_lines = df_mut_filtered[DataConstants.MODEL_ID_COL].unique().tolist()
    print(f"Filtered mutations: {df_mut_filtered.shape[0]} rows for {len(mut_cell_lines)} cell lines")

    return df_mut_filtered, mut_cell_lines


def extract_gene_name(gene_raw: str) -> str:
    """Extract gene name from raw gene string.

    Args:
        gene_raw: Raw gene string.

    Returns:
        Cleaned gene name.
    """
    if gene_raw:
        return str(gene_raw).split()[0]
    else:
        return ""


def load_and_process_cna(cna_file_path: Union[str, Path]) -> Tuple[pd.DataFrame, Set[str]]:
    """Load and process CNA data.

    Args:
        cna_file_path: Path to CNA CSV file.

    Returns:
        Tuple of (processed CNA DataFrame in long format, set of cell line IDs).
    """
    df_cna = pd.read_csv(cna_file_path)
    print(f"Loaded CNA data: {df_cna.shape[0]} cell lines")

    if DataConstants.MODEL_ID_COL not in df_cna.columns:
        df_cna.rename(columns={df_cna.columns[0]: DataConstants.MODEL_ID_COL}, inplace=True)

    df_cna_long = pd.melt(
        df_cna,
        id_vars=[DataConstants.MODEL_ID_COL],
        var_name='Gene_Raw',
        value_name='CNA_value'
    )

    tqdm.pandas(desc="Extracting gene names from CNA data")
    df_cna_long[DataConstants.GENE_COL] = df_cna_long['Gene_Raw'].progress_apply(extract_gene_name)
    df_cna_long = df_cna_long[[DataConstants.MODEL_ID_COL, DataConstants.GENE_COL, 'CNA_value']]
    df_cna_long.dropna(subset=[DataConstants.GENE_COL, 'CNA_value'], inplace=True)

    tqdm.pandas(desc="Converting CNA values to numeric")
    df_cna_long['CNA_value'] = df_cna_long['CNA_value'].progress_apply(pd.to_numeric, errors='coerce')
    df_cna_long.dropna(subset=['CNA_value'], inplace=True)

    cna_cell_lines = set(df_cna_long[DataConstants.MODEL_ID_COL].unique())
    print(f"Processed CNA: {df_cna_long.shape[0]} alterations for {len(cna_cell_lines)} cell lines")

    return df_cna_long, cna_cell_lines


def load_and_process_rna(rna_file_path: Union[str, Path]) -> Tuple[pd.DataFrame, Set[str]]:
    """Load and process RNA expression data.

    Args:
        rna_file_path: Path to RNA expression CSV file.

    Returns:
        Tuple of (processed RNA DataFrame in long format, set of cell line IDs).
    """
    df_rna = pd.read_csv(rna_file_path)
    print(f"Loaded RNA data: {df_rna.shape[0]} cell lines")

    df_rna.rename(columns={df_rna.columns[0]: DataConstants.MODEL_ID_COL}, inplace=True)

    df_rna_long = pd.melt(
        df_rna,
        id_vars=[DataConstants.MODEL_ID_COL],
        var_name='Gene_Raw',
        value_name='RNA_value'
    )

    tqdm.pandas(desc="Extracting gene names from RNA data")
    df_rna_long[DataConstants.GENE_COL] = df_rna_long['Gene_Raw'].progress_apply(extract_gene_name)
    df_rna_long = df_rna_long[[DataConstants.MODEL_ID_COL, DataConstants.GENE_COL, 'RNA_value']]
    df_rna_long.dropna(subset=[DataConstants.GENE_COL, 'RNA_value'], inplace=True)

    tqdm.pandas(desc="Converting RNA values to numeric")
    df_rna_long['RNA_value'] = df_rna_long['RNA_value'].progress_apply(pd.to_numeric, errors='coerce')
    df_rna_long.dropna(subset=['RNA_value'], inplace=True)

    rna_cell_lines = set(df_rna_long[DataConstants.MODEL_ID_COL].unique())
    print(f"Processed RNA: {df_rna_long.shape[0]} expression values for {len(rna_cell_lines)} cell lines")

    return df_rna_long, rna_cell_lines


def load_and_process_response(
    response_file_path: Union[str, Path],
    cell_lines_depmap: List[str],
    df_meta: pd.DataFrame
) -> pd.DataFrame:
    """Load and process drug response data.

    Args:
        response_file_path: Path to drug response CSV file.
        cell_lines_depmap: List of DepMap cell line IDs to filter for.
        df_meta: Metadata DataFrame for mapping.

    Returns:
        Processed drug response DataFrame.
    """
    df_resp = pd.read_csv(response_file_path)
    print(f"Loaded {df_resp.shape[0]} drug responses")

    df_meta_data = df_meta[df_meta[DataConstants.MODEL_ID_COL].isin(cell_lines_depmap)]

    df_merged = pd.merge(
        df_resp,
        df_meta_data,
        left_on=df_resp.columns[0],
        right_on=DataConstants.CELL_LINE_NAME_COL,
        how='inner'
    )
    print(f"Matched {df_merged.shape[0]} responses to ModelIDs")

    df_final = df_merged[[DataConstants.MODEL_ID_COL, 'aac_recomputed']].rename(
        columns={'aac_recomputed': DataConstants.AAC_COL}
    )
    df_final[DataConstants.AAC_COL] = pd.to_numeric(df_final[DataConstants.AAC_COL], errors='coerce')
    df_final.dropna(subset=[DataConstants.AAC_COL], inplace=True)

    unique_cell_lines = len(df_final[DataConstants.MODEL_ID_COL].unique())
    print(f"Final dataset: {unique_cell_lines} cell lines with response data")

    return df_final


def load_and_process_gdsc2_response(
    gdsc2_data_file_path: Union[str, Path],
    drug_name: str,
    cell_lines_depmap: List[str],
    df_meta: pd.DataFrame
) -> pd.DataFrame:
    """Load and process GDSC2 drug response data.

    Args:
        gdsc2_data_file_path: Path to GDSC2 fitted dose response CSV file.
        drug_name: Name of drug to extract data for.
        cell_lines_depmap: List of DepMap cell line IDs to filter for.
        df_meta: Metadata DataFrame for mapping.

    Returns:
        Processed GDSC2 drug response DataFrame.
    """
    drug_name_normalized = drug_name.replace("_", "-")
    print(f"Processing GDSC2 data for drug: {drug_name_normalized}")

    df_gdsc2_raw = pd.read_csv(gdsc2_data_file_path)
    print(f"Loaded raw GDSC2 data: {df_gdsc2_raw.shape[0]} rows")

    df_drug_gdsc2 = df_gdsc2_raw[
        df_gdsc2_raw['DRUG_NAME'].str.upper() == drug_name_normalized.upper()
    ].copy()
    print(f"Found {df_drug_gdsc2.shape[0]} GDSC2 entries for {drug_name_normalized}")

    if df_drug_gdsc2.empty:
        print(f"No GDSC2 data found for drug: {drug_name_normalized}")
        return pd.DataFrame()

    df_drug_gdsc2.dropna(subset=['SANGER_MODEL_ID', 'AUC'], inplace=True)
    df_drug_gdsc2['SANGER_MODEL_ID'] = df_drug_gdsc2['SANGER_MODEL_ID'].astype(str).str.strip()

    df_meta_for_gdsc_map = df_meta[[DataConstants.MODEL_ID_COL, DataConstants.SANGER_MODEL_ID_COL]].copy()
    df_meta_for_gdsc_map.dropna(subset=[DataConstants.SANGER_MODEL_ID_COL], inplace=True)
    df_meta_for_gdsc_map[DataConstants.SANGER_MODEL_ID_COL] = (
        df_meta_for_gdsc_map[DataConstants.SANGER_MODEL_ID_COL].astype(str).str.strip()
    )

    df_merged_gdsc2 = pd.merge(
        df_drug_gdsc2,
        df_meta_for_gdsc_map.drop_duplicates(subset=[DataConstants.SANGER_MODEL_ID_COL]),
        left_on='SANGER_MODEL_ID',
        right_on=DataConstants.SANGER_MODEL_ID_COL,
        how='inner'
    )
    print(f"Matched {df_merged_gdsc2.shape[0]} GDSC2 responses to DepMap ModelIDs")

    df_gdsc2_filtered = df_merged_gdsc2[
        df_merged_gdsc2[DataConstants.MODEL_ID_COL].isin(cell_lines_depmap)
    ]
    print(f"{df_gdsc2_filtered.shape[0]} GDSC2 responses after filtering")

    df_gdsc2_final = df_gdsc2_filtered[[DataConstants.MODEL_ID_COL, 'AUC']].rename(
        columns={'AUC': DataConstants.AAC_COL}
    )
    df_gdsc2_final[DataConstants.AAC_COL] = pd.to_numeric(df_gdsc2_final[DataConstants.AAC_COL], errors='coerce')

    # Convert AUC to AAC
    df_gdsc2_final[DataConstants.AAC_COL] = 1.0 - df_gdsc2_final[DataConstants.AAC_COL]

    df_gdsc2_final.dropna(subset=[DataConstants.AAC_COL], inplace=True)

    unique_cell_lines = len(df_gdsc2_final[DataConstants.MODEL_ID_COL].unique())
    print(f"Final GDSC2 dataset for {drug_name_normalized}: {unique_cell_lines} cell lines")

    return df_gdsc2_final


def find_common_cell_lines(
    df_meta: pd.DataFrame,
    df_mut: pd.DataFrame,
    df_cna: pd.DataFrame,
    df_rna: pd.DataFrame
) -> List[str]:
    """Find cell lines common across required omics datasets.

    Mutations are optional - cell lines without mutations are included.

    Args:
        df_meta: Metadata DataFrame.
        df_mut: Mutation DataFrame.
        df_cna: CNA DataFrame.
        df_rna: RNA expression DataFrame.

    Returns:
        List of common cell line IDs.
    """
    common_cell_lines = (
        set(df_meta[DataConstants.MODEL_ID_COL]) &
        set(df_cna[DataConstants.MODEL_ID_COL]) &
        set(df_rna[DataConstants.MODEL_ID_COL])
    )

    print(f"Found {len(common_cell_lines)} common cell lines")
    return list(common_cell_lines)


def find_true_test_cell_lines(train_response_data: pd.DataFrame, test_response_data: pd.DataFrame) -> List[str]:
    """Find cell lines in test dataset but not in training dataset.

    Args:
        train_response_data: Training response DataFrame.
        test_response_data: Test response DataFrame.

    Returns:
        List of cell line IDs exclusive to test dataset.
    """
    train_cell_lines = set(train_response_data[DataConstants.MODEL_ID_COL])
    test_cell_lines = set(test_response_data[DataConstants.MODEL_ID_COL])

    true_test_cell_lines = list(test_cell_lines - train_cell_lines)
    print(f"Found {len(true_test_cell_lines)} cell lines exclusive to test dataset")

    return true_test_cell_lines


def save_response_data(df: pd.DataFrame, file_path: Union[str, Path], data_type: str) -> None:
    """Save response data to CSV file.

    Args:
        df: DataFrame to save.
        file_path: Output file path.
        data_type: Description for logging.
    """
    columns_to_save = [DataConstants.MODEL_ID_COL, DataConstants.AAC_COL]
    df[columns_to_save].to_csv(file_path, index=False)
    print(f"{data_type} data saved to {file_path} ({len(df)} cell lines)")


def save_omics_data(
    mut_data: pd.DataFrame,
    cna_data: pd.DataFrame,
    rna_data: pd.DataFrame,
    ctrpv2_response_data: pd.DataFrame,
    pkl_output_filepath: Union[str, Path],
    drug_name: str
) -> None:
    """Save omics data to pickle file.

    Args:
        mut_data: Mutation DataFrame.
        cna_data: CNA DataFrame.
        rna_data: RNA DataFrame.
        ctrpv2_response_data: CTRPv2 response DataFrame.
        pkl_output_filepath: Output pickle file path.
        drug_name: Drug name for logging.
    """
    save_data = {
        'df_mut_filtered': mut_data,
        'df_cna_long': cna_data,
        'df_rna_long': rna_data,
        'df_erl_final': ctrpv2_response_data
    }

    with open(pkl_output_filepath, 'wb') as f:
        pd.to_pickle(save_data, f)
    print(f"Saved omics and {drug_name} CTRPv2 data to {pkl_output_filepath}")


def save_outputs(
    drug_name: str,
    pkl_output_filepath: Union[str, Path],
    ctrpv2_response_save_file: Union[str, Path],
    gdsc0_response_save_file: Union[str, Path],
    gdsc2_response_save_file: Union[str, Path],
    true_test_gdsc0_response_save_file: Union[str, Path],
    true_test_gdsc2_response_save_file: Union[str, Path],
    mut_data: pd.DataFrame,
    cna_data: pd.DataFrame,
    rna_data: pd.DataFrame,
    ctrpv2_response_data: pd.DataFrame,
    gdsc0_response_data: pd.DataFrame,
    gdsc2_response_data: pd.DataFrame,
    true_test_lines_gdsc0: List[str],
    true_test_lines_gdsc2: List[str],
    include_gdsc2_datasets: bool = True
) -> Dict[str, Any]:
    """Save all output files and return summary information.

    Args:
        drug_name: Name of drug being processed.
        pkl_output_filepath: Path for omics pickle file.
        ctrpv2_response_save_file: Path for CTRPv2 response CSV.
        gdsc0_response_save_file: Path for GDSC0 response CSV.
        gdsc2_response_save_file: Path for GDSC2 response CSV.
        true_test_gdsc0_response_save_file: Path for true test GDSC0 CSV.
        true_test_gdsc2_response_save_file: Path for true test GDSC2 CSV.
        mut_data: Mutation DataFrame.
        cna_data: CNA DataFrame.
        rna_data: RNA DataFrame.
        ctrpv2_response_data: CTRPv2 response DataFrame.
        gdsc0_response_data: GDSC0 response DataFrame.
        gdsc2_response_data: GDSC2 response DataFrame.
        true_test_lines_gdsc0: List of GDSC0 true test cell lines.
        true_test_lines_gdsc2: List of GDSC2 true test cell lines.
        include_gdsc2_datasets: Whether to include GDSC2 datasets.

    Returns:
        Dictionary with summary information.
    """
    save_omics_data(mut_data, cna_data, rna_data, ctrpv2_response_data, pkl_output_filepath, drug_name)

    save_response_data(ctrpv2_response_data, ctrpv2_response_save_file, f"CTRPv2 response for {drug_name}")
    save_response_data(gdsc0_response_data, gdsc0_response_save_file, f"GDSC0 response for {drug_name}")

    true_test_gdsc0_df = gdsc0_response_data[
        gdsc0_response_data[DataConstants.MODEL_ID_COL].isin(true_test_lines_gdsc0)
    ]
    save_response_data(
        true_test_gdsc0_df,
        true_test_gdsc0_response_save_file,
        f"True-test GDSC0 response for {drug_name}"
    )

    summary_info = {
        'drug_name': drug_name,
        'ctrpv2_cell_lines': len(ctrpv2_response_data),
        'gdsc0_cell_lines': len(gdsc0_response_data),
        'gdsc0_true_test_cell_lines': len(true_test_gdsc0_df)
    }

    if include_gdsc2_datasets:
        if gdsc2_response_data is not None and not gdsc2_response_data.empty:
            save_response_data(
                gdsc2_response_data,
                gdsc2_response_save_file,
                f"GDSC2 response for {drug_name}"
            )
            summary_info['gdsc2_cell_lines'] = len(gdsc2_response_data)

            if true_test_lines_gdsc2:
                true_test_gdsc2_df = gdsc2_response_data[
                    gdsc2_response_data[DataConstants.MODEL_ID_COL].isin(true_test_lines_gdsc2)
                ]
                save_response_data(
                    true_test_gdsc2_df,
                    true_test_gdsc2_response_save_file,
                    f"True-test GDSC2 response for {drug_name}"
                )
                summary_info['gdsc2_true_test_cell_lines'] = len(true_test_gdsc2_df)
            else:
                print(f"No true-test GDSC2 cell lines for {drug_name}")
                summary_info['gdsc2_true_test_cell_lines'] = 0
        else:
            print(f"No GDSC2 response data available for {drug_name}")
            summary_info['gdsc2_cell_lines'] = 0
            summary_info['gdsc2_true_test_cell_lines'] = 0

    return summary_info


def setup_directories(
    input_data_dir: Union[str, Path],
    intermediate_data_dir: Union[str, Path],
    include_gdsc2_datasets: bool = True
) -> Dict[str, Path]:
    """Setup and create necessary directories.

    Args:
        input_data_dir: Base input data directory.
        intermediate_data_dir: Base intermediate data directory.
        include_gdsc2_datasets: Whether to create GDSC2 directories.

    Returns:
        Dictionary of directory paths.
    """
    input_data_dir = Path(input_data_dir)
    intermediate_data_dir = Path(intermediate_data_dir)

    ctrp_input_dir = input_data_dir / DataConstants.CTRP_DRUG_RESPONSE_SUBDIR
    gdsc0_input_dir = input_data_dir / DataConstants.GDSC0_DRUG_RESPONSE_SUBDIR

    ctrp_intermediate_dir = intermediate_data_dir / DataConstants.CTRP_DRUG_RESPONSE_SUBDIR
    gdsc0_intermediate_dir = intermediate_data_dir / DataConstants.GDSC0_DRUG_RESPONSE_SUBDIR
    gdsc2_intermediate_dir = intermediate_data_dir / DataConstants.GDSC2_DRUG_RESPONSE_SUBDIR

    true_test_gdsc0_dir = gdsc0_intermediate_dir / DataConstants.TRUE_TEST_SUBDIR
    true_test_gdsc2_dir = gdsc2_intermediate_dir / DataConstants.TRUE_TEST_SUBDIR

    dirs_to_create = [
        ctrp_intermediate_dir,
        gdsc0_intermediate_dir,
        true_test_gdsc0_dir
    ]

    if include_gdsc2_datasets:
        dirs_to_create.extend([gdsc2_intermediate_dir, true_test_gdsc2_dir])

    for path in dirs_to_create:
        Path(path).mkdir(parents=True, exist_ok=True)

    return {
        'ctrp_input_dir': ctrp_input_dir,
        'gdsc0_input_dir': gdsc0_input_dir,
        'ctrp_intermediate_dir': ctrp_intermediate_dir,
        'gdsc0_intermediate_dir': gdsc0_intermediate_dir,
        'gdsc2_intermediate_dir': gdsc2_intermediate_dir,
        'true_test_gdsc0_dir': true_test_gdsc0_dir,
        'true_test_gdsc2_dir': true_test_gdsc2_dir
    }


def process_single_drug(
    drug_name: str,
    directories: Dict[str, Path],
    input_data_dir: Path,
    intermediate_data_dir: Path,
    cell_lines_depmap: List[str],
    df_meta: pd.DataFrame,
    mut_data: pd.DataFrame,
    cna_data: pd.DataFrame,
    rna_data: pd.DataFrame,
    include_gdsc2_datasets: bool = True
) -> Dict[str, Any]:
    """Process data for a single drug.

    Args:
        drug_name: Name of drug to process.
        directories: Dictionary of directory paths.
        input_data_dir: Base input data directory.
        intermediate_data_dir: Base intermediate data directory.
        cell_lines_depmap: List of common cell line IDs.
        df_meta: Metadata DataFrame.
        mut_data: Mutation DataFrame.
        cna_data: CNA DataFrame.
        rna_data: RNA DataFrame.
        include_gdsc2_datasets: Whether to include GDSC2 datasets.

    Returns:
        Dictionary with summary information for the drug.
    """
    print(f"Processing data for drug: {drug_name}")

    ctrpv2_input_file = directories['ctrp_input_dir'] / f'{drug_name}.csv'
    gdsc0_input_file = directories['gdsc0_input_dir'] / f'{drug_name}.csv'
    gdsc2_input_file = input_data_dir / DataConstants.GDSC2_FITTED_DOSE_RESPONSE_FILENAME

    ctrpv2_output_file = directories['ctrp_intermediate_dir'] / f'{drug_name}.csv'
    gdsc0_output_file = directories['gdsc0_intermediate_dir'] / f'{drug_name}.csv'
    gdsc2_output_file = directories['gdsc2_intermediate_dir'] / f'{drug_name}.csv'

    true_test_gdsc0_file = directories['true_test_gdsc0_dir'] / f'{drug_name}.csv'
    true_test_gdsc2_file = directories['true_test_gdsc2_dir'] / f'{drug_name}.csv'

    omics_output_file = intermediate_data_dir / DataConstants.OMICS_DATA_FILENAME

    ctrpv2_response_data = load_and_process_response(ctrpv2_input_file, cell_lines_depmap, df_meta)
    gdsc0_response_data = load_and_process_response(gdsc0_input_file, cell_lines_depmap, df_meta)

    gdsc2_response_data = pd.DataFrame()
    true_test_cell_lines_gdsc2 = []

    if include_gdsc2_datasets:
        gdsc2_response_data = load_and_process_gdsc2_response(
            gdsc2_input_file, drug_name, cell_lines_depmap, df_meta
        )
        true_test_cell_lines_gdsc2 = find_true_test_cell_lines(ctrpv2_response_data, gdsc2_response_data)

    true_test_cell_lines_gdsc0 = find_true_test_cell_lines(ctrpv2_response_data, gdsc0_response_data)

    summary_data = save_outputs(
        drug_name,
        omics_output_file,
        ctrpv2_output_file,
        gdsc0_output_file,
        gdsc2_output_file,
        true_test_gdsc0_file,
        true_test_gdsc2_file,
        mut_data,
        cna_data,
        rna_data,
        ctrpv2_response_data,
        gdsc0_response_data,
        gdsc2_response_data,
        true_test_cell_lines_gdsc0,
        true_test_cell_lines_gdsc2,
        include_gdsc2_datasets
    )

    return summary_data


def main(
    drug_names_list: List[str],
    paths_config: Dict[str, Union[str, Path]],
    include_gdsc2_datasets: bool = True
) -> None:
    """Process omics data for multiple drugs.

    Args:
        drug_names_list: List of drug names to process.
        paths_config: Dictionary containing paths configuration.
        include_gdsc2_datasets: Whether to include GDSC2 datasets.
    """
    print("Starting omics data preparation")

    input_data_dir = Path(paths_config['input_data_dir'])
    intermediate_data_dir = Path(paths_config['intermediate_data_dir'])
    pathway_graph_file = Path(paths_config['pathway_interaction_graph_file'])

    directories = setup_directories(input_data_dir, intermediate_data_dir, include_gdsc2_datasets)

    mutation_file = input_data_dir / DataConstants.MUTATION_FILENAME
    cna_file = input_data_dir / DataConstants.CNA_FILENAME
    metadata_file = input_data_dir / DataConstants.METADATA_FILENAME
    rna_file = input_data_dir / DataConstants.RNA_FILENAME

    print("Loading omics datasets...")
    df_meta = load_and_prepare_metadata(metadata_file)
    df_mut_filtered, _ = load_and_filter_mutations(mutation_file)
    df_cna_long, _ = load_and_process_cna(cna_file)
    pathway_dict = load_pathway_graph_and_get_gene_sets(pathway_graph_file)
    df_rna_long, _ = load_and_process_rna(rna_file)

    cell_lines_depmap = find_common_cell_lines(df_meta, df_mut_filtered, df_cna_long, df_rna_long)

    all_drugs_summary_data = []
    failed_drugs = []

    for drug_name in drug_names_list:
        try:
            summary_data = process_single_drug(
                drug_name,
                directories,
                input_data_dir,
                intermediate_data_dir,
                cell_lines_depmap,
                df_meta,
                df_mut_filtered,
                df_cna_long,
                df_rna_long,
                include_gdsc2_datasets
            )
            all_drugs_summary_data.append(summary_data)
        except Exception as e:
            print(f"Error processing drug {drug_name}: {e}")
            failed_drugs.append(drug_name)
            continue

    if all_drugs_summary_data:
        summary_df = pd.DataFrame(all_drugs_summary_data)

        columns = ['drug_name', 'ctrpv2_cell_lines', 'gdsc0_cell_lines', 'gdsc0_true_test_cell_lines']
        if include_gdsc2_datasets:
            columns.extend(['gdsc2_cell_lines', 'gdsc2_true_test_cell_lines'])

        summary_df = summary_df[columns]
        summary_output_path = intermediate_data_dir / DataConstants.SUMMARY_FILENAME
        summary_df.to_csv(summary_output_path, index=False)

        print(f"Saved summary to {summary_output_path}")
        print(f"Summary:\n{summary_df}")

        if failed_drugs:
            print(f"Failed to process {len(failed_drugs)} drugs: {failed_drugs}")

        print(f"Successfully processed {len(all_drugs_summary_data)} out of {len(drug_names_list)} drugs")
    else:
        print("No drugs were successfully processed")
