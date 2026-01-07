"""Build optimized graph index data for PIGE Atlas."""

import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

from clean_disease_types import clean_disease_type


random.seed(42)


def load_drug_classes(path: Path) -> Dict[str, str]:
    """Load drug to class mapping.

    Args:
        path: Path to drug_classes.json file.

    Returns:
        Dictionary mapping drug name to drug class.
    """
    with open(path) as f:
        data = json.load(f)

    mapping = {}
    for drug_class, drugs in data.items():
        for drug in drugs:
            mapping[drug] = drug_class
    return mapping


def load_cell_line_diseases(path: Path) -> Dict[str, str]:
    """Load cell line to disease type mapping from Model.csv.

    Args:
        path: Path to Model.csv file.

    Returns:
        Dictionary mapping cell line name to disease type.
    """
    mapping = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            cell_line = row.get('CellLineName', '')
            disease = row.get('OncotreeSubtype', '')
            if cell_line and disease:
                mapping[cell_line] = disease
    return mapping


def load_key_entities(path: Path) -> Dict:
    """Load pathway and gene data for drug-cell line combinations.

    Args:
        path: Path to key_entities.json file.

    Returns:
        Dictionary with drug_cellline_map containing pathways and genes.
    """
    with open(path) as f:
        return json.load(f)


def load_validated_entities(pathways_path: Path, genes_path: Path) -> Tuple[Dict, Dict]:
    """Load validated pathways and genes for each drug.

    Args:
        pathways_path: Path to relevant_pathways.json.
        genes_path: Path to relevant_genes.json.

    Returns:
        Tuple of (validated_pathways, validated_genes) dictionaries.
    """
    with open(pathways_path) as f:
        pathways = json.load(f)
    with open(genes_path) as f:
        genes = json.load(f)
    return pathways, genes


def discover_graphs(
    graphs_dir: Path,
    drug_to_class: Dict[str, str],
    cell_line_to_disease: Dict[str, str],
    key_entities: Dict
) -> List[Dict]:
    """Discover all PIGE graph HTML files and extract metadata.

    Args:
        graphs_dir: Root directory containing graph HTML files.
        drug_to_class: Mapping of drug to drug class.
        cell_line_to_disease: Mapping of cell line to disease type.
        key_entities: Dictionary containing pathway and gene data.

    Returns:
        List of graph entry dictionaries with metadata.
    """
    entries = []

    for html_path in graphs_dir.rglob('visualizations/*.html'):
        if not html_path.name.endswith('_PIGE_graph.html'):
            continue

        parts = html_path.relative_to(graphs_dir).parts
        if len(parts) < 4:
            continue

        dataset_raw = parts[0]
        drug = parts[1]
        filename = parts[3]

        is_average = '_AVERAGE_' in filename
        cell_line = 'AVERAGE' if is_average else filename.replace('_PIGE_graph.html', '')
        display_name = 'Average across cell lines' if is_average else cell_line.replace('_', ' ')

        dataset = {'gdsc0_true_test': 'GDSC2', 'ctrpv2': 'CTRPv2'}.get(dataset_raw, dataset_raw)
        drug_class = drug_to_class.get(drug, 'Other')

        disease_type = ''
        if not is_average:
            raw_disease = cell_line_to_disease.get(cell_line, 'Unknown')
            disease_type = clean_disease_type(raw_disease)

        with open(html_path, errors='ignore') as f:
            content = f.read(1500000)

        aac_val = None
        if match := re.search(r'Predicted AAC:\s*([0-9]*\.?[0-9]+)', content):
            aac_val = float(match.group(1))
        elif match := re.search(r'Mean AAC:\s*([0-9]*\.?[0-9]+)', content):
            aac_val = float(match.group(1))

        pathway_text = ''
        gene_text = ''
        search_text_base = ' '.join([drug, display_name, cell_line, disease_type, drug_class, dataset]).lower()

        key = f'{drug}|{cell_line}'
        entities = key_entities.get('drug_cellline_map', {}).get(key)

        if entities:
            pathways = entities.get('pathways', [])
            genes = entities.get('genes', [])
            pathway_text = ' '.join(pathways).lower()
            gene_text = ' '.join(genes).lower()
            search_text = f'{search_text_base} {pathway_text} {gene_text}'
        else:
            search_text = search_text_base

        entries.append({
            'href': str(html_path.relative_to(graphs_dir)).replace('\\', '/'),
            'dataset': dataset,
            'drug': drug,
            'drug_class': drug_class,
            'cell_line': cell_line,
            'name': display_name,
            'disease_type': disease_type,
            'average': is_average,
            'aac': aac_val,
            '_searchText': search_text,
            '_pathwayText': pathway_text,
            '_geneText': gene_text,
        })

    entries.sort(key=lambda e: (e['drug'], not e['average'], e['name'].lower()))
    return entries


def is_validated(entry: Dict, validated_pathways: Dict, validated_genes: Dict) -> bool:
    """Check if entry has 2+ validated pathways and 2+ validated genes.

    Args:
        entry: Graph entry dictionary.
        validated_pathways: Dictionary of validated pathways per drug.
        validated_genes: Dictionary of validated genes per drug.

    Returns:
        True if entry meets validation criteria.
    """
    drug = entry.get('drug', '')
    drug_pathways = set(validated_pathways.get(drug, []))
    drug_genes = set(validated_genes.get(drug, []))

    pathway_text = entry.get('_pathwayText', '').lower()
    gene_text = entry.get('_geneText', '').upper()

    pathway_matches = sum(1 for p in drug_pathways if p.lower() in pathway_text)
    gene_matches = sum(1 for g in drug_genes if g.upper() in gene_text)

    return pathway_matches >= 2 and gene_matches >= 2


def select_preview(
    entries: List[Dict],
    validated_pathways: Dict,
    validated_genes: Dict,
    count: int = 5000
) -> List[Dict]:
    """Select preview entries with validated graphs first.

    Randomizes all entries, then puts validated entries first.

    Args:
        entries: List of all graph entries.
        validated_pathways: Dictionary of validated pathways per drug.
        validated_genes: Dictionary of validated genes per drug.
        count: Number of entries to include in preview.

    Returns:
        List of selected entries with validated ones first.
    """
    randomized = entries.copy()
    random.shuffle(randomized)

    validated = [e for e in randomized if is_validated(e, validated_pathways, validated_genes)]
    non_validated = [e for e in randomized if e not in validated]

    preview = validated + non_validated
    print(f'Selected {len(validated)} validated entries (first in preview)')

    return preview[:count]


def strip_pathway_gene_data(entry: Dict) -> Dict:
    """Remove pathway and gene text from entry to reduce size.

    Args:
        entry: Graph entry dictionary.

    Returns:
        Clean entry without pathway/gene text.
    """
    clean = entry.copy()
    clean.pop('_pathwayText', None)
    clean.pop('_geneText', None)
    clean['_searchText'] = ' '.join([
        clean.get('drug', ''),
        clean.get('name', ''),
        clean.get('cell_line', ''),
        clean.get('disease_type', ''),
        clean.get('drug_class', ''),
        clean.get('dataset', '')
    ]).lower()
    return clean


def build_metadata(entries: List[Dict]) -> Dict:
    """Extract metadata for filters.

    Args:
        entries: List of all graph entries.

    Returns:
        Dictionary containing filter options and stats.
    """
    drugs = sorted(set(e['drug'] for e in entries))
    cell_lines = sorted(set(e['cell_line'] for e in entries if not e.get('average')))
    drug_classes = sorted(set(e['drug_class'] for e in entries if e.get('drug_class')))
    disease_types = sorted(set(e['disease_type'] for e in entries if e.get('disease_type')))
    datasets = sorted(set(e['dataset'] for e in entries))

    return {
        'drugs': drugs,
        'cell_lines': cell_lines,
        'drug_classes': drug_classes,
        'disease_types': disease_types,
        'datasets': datasets,
        'stats': {
            'total_graphs': len(entries),
            'unique_drugs': len(drugs),
            'unique_cell_lines': len(cell_lines)
        }
    }


def build_pathway_gene_map(entries: List[Dict]) -> Dict:
    """Build pathway/gene lookup map.

    Args:
        entries: List of all graph entries.

    Returns:
        Dictionary mapping drug|cell_line to pathways and genes.
    """
    pathway_gene_map = {}

    for entry in entries:
        key = f"{entry['drug']}|{entry['cell_line']}"
        pathway_text = entry.get('_pathwayText', '')
        gene_text = entry.get('_geneText', '')

        if pathway_text or gene_text:
            pathway_gene_map[key] = {
                'pathways': pathway_text,
                'genes': gene_text
            }

    return pathway_gene_map


def build_graph_index(
    graphs_dir: Path,
    drug_classes_path: Path,
    model_csv_path: Path,
    key_entities_path: Path,
    validated_pathways_path: Path,
    validated_genes_path: Path
) -> None:
    """Build optimized graph index files.

    Creates three JSON files:
    - index_preview.json: First 5000 entries (validated first)
    - index_remaining.json: Remaining entries
    - pathway_gene_data.json: Pathway and gene data for advanced search

    Args:
        graphs_dir: Directory containing graph HTML files and where to write output.
        drug_classes_path: Path to drug_classes.json.
        model_csv_path: Path to Model.csv.
        key_entities_path: Path to key_entities.json.
        validated_pathways_path: Path to relevant_pathways.json.
        validated_genes_path: Path to relevant_genes.json.
    """
    drug_to_class = load_drug_classes(drug_classes_path)
    cell_line_to_disease = load_cell_line_diseases(model_csv_path)
    key_entities = load_key_entities(key_entities_path)
    validated_pathways, validated_genes = load_validated_entities(validated_pathways_path, validated_genes_path)

    print(f'Discovering graphs in {graphs_dir}')
    entries = discover_graphs(graphs_dir, drug_to_class, cell_line_to_disease, key_entities)
    print(f'Found {len(entries)} graphs')

    preview_entries = select_preview(entries, validated_pathways, validated_genes, 5000)
    preview_ids = {id(e) for e in preview_entries}
    remaining_entries = [e for e in entries if id(e) not in preview_ids]

    metadata = build_metadata(entries)

    preview_clean = [strip_pathway_gene_data(e) for e in preview_entries]
    remaining_clean = [strip_pathway_gene_data(e) for e in remaining_entries]

    pathway_gene_map = build_pathway_gene_map(entries)

    preview_path = graphs_dir / 'index_preview.json'
    with open(preview_path, 'w') as f:
        json.dump({'entries': preview_clean, 'metadata': metadata}, f, separators=(',', ':'))

    remaining_path = graphs_dir / 'index_remaining.json'
    with open(remaining_path, 'w') as f:
        json.dump(remaining_clean, f, separators=(',', ':'))

    pathway_gene_path = graphs_dir / 'pathway_gene_data.json'
    with open(pathway_gene_path, 'w') as f:
        json.dump(pathway_gene_map, f, separators=(',', ':'))

    print(f'Generated: {preview_path.name}, {remaining_path.name}, {pathway_gene_path.name}')
