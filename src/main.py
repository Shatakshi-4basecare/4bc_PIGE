"""Main entry point for the PIGE drug response prediction pipeline.

Loads configuration from YAML and runs the pipeline orchestrator.
Supports both single-drug mode (process each drug separately) and
multi-drug mode (process all drugs together).
"""

import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.pipeline_orchestrator import PipelineOrchestrator

config_name = "config.yaml"

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def resolve_config_variables(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve variable references in config recursively (e.g., ${paths.input_data_dir}).

    Handles nested variable references by performing multiple passes until
    all variables are resolved or no more changes occur.

    Args:
        config: Configuration dictionary.

    Returns:
        Resolved configuration dictionary.
    """
    import re

    def get_nested_value(obj: Any, path: str) -> Any:
        """Get a value from nested dict using dot notation path."""
        keys = path.split('.')
        current = obj
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    def resolve_string(value: str, context: Dict[str, Any]) -> str:
        """Replace all ${variable} references in a string."""
        def replace_var(match):
            var_path = match.group(1)
            resolved = get_nested_value(context, var_path)
            if resolved is not None:
                return str(resolved)
            return match.group(0)

        return re.sub(r'\$\{([^}]+)\}', replace_var, value)

    def resolve_value(value: Any, context: Dict[str, Any]) -> Any:
        """Recursively resolve values in the config."""
        if isinstance(value, str):
            return resolve_string(value, context)
        elif isinstance(value, dict):
            return {k: resolve_value(v, context) for k, v in value.items()}
        elif isinstance(value, list):
            return [resolve_value(item, context) for item in value]
        return value

    max_iterations = 10
    previous = None
    current = config

    for _ in range(max_iterations):
        current = resolve_value(current, current)
        if current == previous:
            break
        previous = current

    return current


def get_drugs_to_process(config: Dict[str, Any]) -> List[str]:
    """Get list of drugs to process from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        List of drug names to process.
    """
    main_drugs = config.get('drugs', {}).get('main_drugs', [])
    if main_drugs:
        return main_drugs

    drugs_file = config.get('paths', {}).get('drugs_to_process_file')
    if not drugs_file or not Path(drugs_file).exists():
        return []

    drugs = []
    with open(drugs_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                drugs.append(parts[0])

    return drugs


def run_single_drug_mode(config: Dict[str, Any], drugs: List[str]) -> None:
    """Run pipeline in single-drug mode (process each drug separately).

    Args:
        config: Configuration dictionary.
        drugs: List of drug names to process.
    """
    print(f"\nSingle-drug mode: Processing {len(drugs)} drugs sequentially")

    for drug_name in drugs:
        print(f"\n{'='*80}")
        print(f"Processing drug: {drug_name}")
        print(f"{'='*80}")

        drug_config = config.copy()
        drug_config['drugs'] = config['drugs'].copy()
        drug_config['drugs']['main_drugs'] = [drug_name]
        drug_config['experiment'] = config.get('experiment', {}).copy()
        drug_config['experiment']['run_name'] = drug_name

        orchestrator = PipelineOrchestrator(drug_config)
        resume_from = config.get('pipeline', {}).get('resume_from_stage')
        success = orchestrator.run_pipeline(resume_from_stage=resume_from)

        status = "completed" if success else "failed"
        print(f"\nPipeline {status} for {drug_name}")


    print(f"\n{'='*80}")
    print(f"All drugs processed in single-drug mode")
    print(f"{'='*80}")


def run_multi_drug_mode(config: Dict[str, Any], drugs: List[str]) -> bool:
    """Run pipeline in multi-drug mode (process all drugs together).

    Args:
        config: Configuration dictionary.
        drugs: List of drug names to process.

    Returns:
        True if successful.
    """
    print(f"\nMulti-drug mode: Processing {len(drugs)} drugs together")
    print(f"Drugs: {drugs}")

    config['drugs']['main_drugs'] = drugs

    orchestrator = PipelineOrchestrator(config)
    resume_from = config.get('pipeline', {}).get('resume_from_stage')
    success = orchestrator.run_pipeline(resume_from_stage=resume_from)

    if success:
        print(f"Pipeline completed for all drugs")
    else:
        print(f"Pipeline failed for all drugs")

    return success

def main(config_name: str) -> None:
    """Main entry point."""
    config_path = Path(__file__).parent / "configs" / config_name
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    print("Loading configuration")
    config = load_config(str(config_path))
    config = resolve_config_variables(config)

    drugs = get_drugs_to_process(config)

    if not drugs:
        print("No drugs to process. Check configuration.")
        sys.exit(1)

    single_drug_mode = config.get('pipeline', {}).get('single_drug_mode', False)

    if single_drug_mode:
        run_single_drug_mode(config, drugs)
    else:
        run_multi_drug_mode(config, drugs)


if __name__ == "__main__":
    main(config_name)
