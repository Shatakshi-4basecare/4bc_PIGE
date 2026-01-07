"""Example script for running knockout analysis with YAML configs.

This script demonstrates how to use the knockout analysis module with
YAML configuration files.
"""

from pathlib import Path
from model_pathway_and_gene_interpretability import load_config, main


def run_single_drug_analysis():
    """Run single drug pathway knockout analysis."""
    config_path = Path(__file__).parent / 'configs' / 'single_drug_pathway_knockout.yaml'
    config = load_config(str(config_path))

    drugs = ['Paclitaxel']
    main(config, drugs)


def run_multi_drug_analysis():
    """Run multi-drug gene knockout analysis."""
    config_path = Path(__file__).parent / 'configs' / 'multi_drug_gene_knockout.yaml'
    config = load_config(str(config_path))

    drugs = ['Erlotinib', 'Paclitaxel', 'Sunitinib']
    main(config, drugs)


def run_beataml_analysis():
    """Run BeatAML pathway knockout analysis."""
    config_path = Path(__file__).parent / 'configs' / 'beataml_pathway_knockout.yaml'
    config = load_config(str(config_path))

    drugs = ['Nutlin_3', 'Sorafenib']
    main(config, drugs)


def run_custom_analysis(config_file: str, drugs: list):
    """Run analysis with custom config.

    Args:
        config_file: Path to YAML config file.
        drugs: List of drug names to process.
    """
    config = load_config(config_file)
    main(config, drugs)


if __name__ == '__main__':
    print("Single drug analysis")
    run_single_drug_analysis()

    print("\nMulti-drug analysis")
    run_multi_drug_analysis()
