"""Master script to build the complete PIGE Graph Atlas.

Runs all steps in order:
1. Generate PIGE graphs for all drugs across multiple datasets
2. Create HTML visualizations for all drugs across multiple datasets
3. Extract key pathways and genes
4. Build graph index data with drug classes and disease types
5. Generate HTML index with enhanced search and filtering
"""

import shutil
from pathlib import Path
from typing import Dict

from new_graphsv2 import run_graph_generation
from visualize_graphs import visualize_graphs
from extract_key_entities import extract_key_entities
from build_graph_index_data import build_graph_index
from generate_graphs_index_v2 import generate_html_index

config = {
    'base_dir': "23-12-2025_pige_graph_final",
    'dataset': ["gdsc0_true_test", "ctrpv2", "BeatAML"],
    'drug_classes': "src/model_interpretability/relevant_entities/drug_classes.json",
    'model_csv': "data/input_data/Model.csv",
    "about_md": "src/pige_atlas/about_documentation/about.md",
    "documentation_md": "src/pige_atlas/about_documentation/documentation.md",
    "drugs_to_process": None,
    "validated_pathways_json": "src/model_interpretability/relevant_entities/relevant_pathways.json",
    "validated_genes_json": "src/model_interpretability/relevant_entities/relevant_genes.json",
}


def build_pige_atlas(config: Dict) -> None:
    """Build the complete PIGE Graph Atlas."""
    base_dir = Path(config['base_dir']).resolve()
    dataset = config['dataset']
    drug_classes = config['drug_classes']
    model_csv = config['model_csv']
    graphs_base_dir = base_dir / "PIGE_graphs"
    output_dir = graphs_base_dir

    graphs_index_json = output_dir / "graphs_index.json"
    key_entities_json = output_dir / "key_entities.json"
    validated_pathways_json = config['validated_pathways_json']
    validated_genes_json = config['validated_genes_json']
    about_md = config['about_md']
    documentation_md = config['documentation_md']
    drugs_to_process = config.get('drugs_to_process')

    print("=" * 60)
    print("PIGE Graph Atlas Builder")
    print("=" * 60)
    print(f"Base directory: {base_dir}")
    print(f"Datasets: {dataset}")
    print(f"Output directory: {output_dir}")
    print(f"Drug classes: {drug_classes}")
    print(f"Model CSV: {model_csv}")
    if drugs_to_process:
        print(f"Requested drugs: {len(drugs_to_process)} ({', '.join(drugs_to_process)})")
    else:
        print(f"Requested drugs: All available (auto-discover per dataset)")

    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(about_md, output_dir / "about.md")
    shutil.copy2(documentation_md, output_dir / "documentation.md")
    print(f"Copied documentation to {output_dir}")

    print(f"\n{'=' * 60}")
    print("Step: Generate PIGE graphs for all drugs")
    print(f"{'=' * 60}")
    run_graph_generation({
        'base_dir': str(base_dir),
        'dataset': dataset,
        'drug_names': drugs_to_process,
    })
    print("\nGenerate PIGE graphs for all drugs completed successfully")

    print(f"\n{'=' * 60}")
    print("Step: Create HTML visualizations for all drugs")
    print(f"{'=' * 60}")
    visualize_graphs({
        'base_dir': str(base_dir),
        'dataset': dataset,
        'drug_names': drugs_to_process,
    })
    print("\nCreate HTML visualizations for all drugs completed successfully")

    print(f"\n{'=' * 60}")
    print("Step: Extract key pathways and genes")
    print(f"{'=' * 60}")
    extract_key_entities({
        'graphs_dir': str(graphs_base_dir),
        'output': str(key_entities_json)
    })
    print("\nExtract key pathways and genes completed successfully")

    print(f"\n{'=' * 60}")
    print("Step: Build graph index data")
    print(f"{'=' * 60}")
    build_graph_index(
        graphs_dir=graphs_base_dir,
        drug_classes_path=drug_classes,
        model_csv_path=model_csv,
        key_entities_path=key_entities_json,
        validated_pathways_path=validated_pathways_json,
        validated_genes_path=validated_genes_json,
    )
    print("\nBuild graph index data completed successfully")

    print(f"\n{'=' * 60}")
    print("Step: Generate HTML index")
    print(f"{'=' * 60}")
    generate_html_index({
        'output_dir': str(output_dir),
        'data_json': 'graphs_index.json',
        'key_entities_json': 'key_entities.json'
    })
    print("\nGenerate HTML index completed successfully")

    print("\n" + "=" * 60)
    print("PIGE Graph Atlas built successfully")
    print("=" * 60)
    print(f"\nGenerated files:")
    print(f"  - {output_dir / 'index.html'}")
    print(f"  - {graphs_index_json}")
    print(f"  - {key_entities_json}")
    print(f"\nOpen {output_dir / 'index.html'} in a web browser to view the atlas.")

if __name__ == "__main__":
    build_pige_atlas(config)