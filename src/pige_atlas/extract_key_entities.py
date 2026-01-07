"""Extract key pathways and genes from nodes CSV files for search functionality."""

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def process_nodes_csv(csv_path: Path) -> Tuple[List[str], List[str]]:
    """Process a single nodes CSV file and extract key pathways and genes."""
    pathways = []
    all_genes = set()

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_type = row.get('node_type', '')
            label = row.get('label', '')

            if node_type == 'drug':
                continue

            if label and node_type in ('resistance', 'sensitivity'):
                pathways.append(label)

            if label and node_type in ('gene_resistance', 'gene_sensitivity'):
                all_genes.add(label)

    return pathways, sorted(list(all_genes))


def build_key_entities_index(graphs_root: Path) -> Dict:
    """Build a complete index of key pathways and genes for each drug+cellline combination."""
    drug_cellline_map = {}
    all_pathways_set = set()
    all_genes_set = set()

    for nodes_csv in graphs_root.rglob("debug_csv/*_nodes.csv"):
        parts = nodes_csv.relative_to(graphs_root).parts

        if len(parts) < 4:
            continue

        drug = parts[1]
        filename = parts[3]

        cell_line = filename.replace("_nodes.csv", "")

        pathways, genes = process_nodes_csv(nodes_csv)

        key = f"{drug}|{cell_line}"
        drug_cellline_map[key] = {
            "pathways": pathways,
            "genes": genes
        }

        all_pathways_set.update(pathways)
        all_genes_set.update(genes)

    return {
        "drug_cellline_map": drug_cellline_map,
        "all_pathways": sorted(list(all_pathways_set)),
        "all_genes": sorted(list(all_genes_set))
    }


def extract_key_entities(config: Dict) -> None:
    """Extract key entities from graphs and save to JSON."""
    graphs_root = Path(config['graphs_dir']).resolve()
    output_path = Path(config['output']).resolve()

    print(f"Extracting key entities from {graphs_root}")
    key_entities = build_key_entities_index(graphs_root)

    print(f"Found {len(key_entities['all_pathways'])} unique pathways")
    print(f"Found {len(key_entities['all_genes'])} unique genes")
    print(f"Indexed {len(key_entities['drug_cellline_map'])} drug+cellline combinations")

    print(f"Writing to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(key_entities, f, ensure_ascii=False, indent=2)

    print("Done")
