"""Functions for filtering GO graph nodes based on various criteria."""

import os
import pandas as pd


def filter_by_namespace(graph, namespace='biological_process'):
    """Filter graph to keep only nodes of a specific namespace.

    Args:
        graph: NetworkX DiGraph to filter.
        namespace: GO namespace to keep (default: 'biological_process').

    Returns:
        Filtered NetworkX DiGraph containing only nodes from specified namespace.
    """
    print(f"Filtering graph to keep only '{namespace}' namespace")

    nodes_to_keep = {n for n, d in graph.nodes(data=True) if d.get('namespace') == namespace}
    subgraph = graph.subgraph(nodes_to_keep).copy()

    print(f'Graph filtered from {graph.number_of_nodes()} to {subgraph.number_of_nodes()} nodes')
    return subgraph


def filter_by_direct_annotation_size(graph, min_size, max_size, exceptions):
    """Filter graph nodes by direct gene annotation size.

    Args:
        graph: NetworkX DiGraph to filter.
        min_size: Minimum number of direct gene annotations.
        max_size: Maximum number of direct gene annotations.
        exceptions: Set of node IDs to exclude from filtering.

    Returns:
        Filtered NetworkX DiGraph.
    """
    print(f'Filtering nodes by direct annotation size (min: {min_size}, max: {max_size})')

    nodes_to_remove = []
    for node_id, data in graph.nodes(data=True):
        if node_id in exceptions:
            continue

        gene_set_size = len(data.get('genes', set()))
        if not (min_size <= gene_set_size <= max_size):
            nodes_to_remove.append(node_id)

    print(f'Identified {len(nodes_to_remove)} nodes to remove based on direct size constraints')

    nodes_to_keep = [n for n in graph.nodes() if n not in nodes_to_remove]
    filtered_graph = graph.subgraph(nodes_to_keep).copy()

    print(f'Graph size after direct annotation filtering: {filtered_graph.number_of_nodes()} nodes')
    return filtered_graph


def filter_by_manual_curation(graph, curation_csv_path):
    """Filter graph to keep only manually curated pathways.

    Args:
        graph: NetworkX DiGraph to filter.
        curation_csv_path: Path to CSV with 'Pathway Name' and 'Final Decision' columns.

    Returns:
        Filtered NetworkX DiGraph containing only pathways marked as 'Keep'.
    """
    if not os.path.exists(curation_csv_path):
        print(f'Manual curation CSV not found at {curation_csv_path}. Skipping filter')
        return graph

    print(f'Applying manual curation filter from: {curation_csv_path}')

    curation_df = pd.read_csv(curation_csv_path)

    pathway_name_column = curation_df.columns[0]
    decision_column = 'Final Decision'

    if decision_column not in curation_df.columns:
        print(f"Column '{decision_column}' not found in CSV. Skipping filter")
        return graph

    pathways_to_keep = curation_df[
        curation_df[decision_column] == 'Keep'
    ][pathway_name_column].tolist()

    print(f"Found {len(pathways_to_keep)} pathways marked as 'Keep' in curation CSV")

    name_to_id_map = {}
    for node_id, node_data in graph.nodes(data=True):
        node_name = node_data.get('name', '')
        if node_name:
            name_to_id_map[node_name] = node_id

    kept_node_ids = set()
    for pathway_name in pathways_to_keep:
        if pathway_name in name_to_id_map:
            kept_node_ids.add(name_to_id_map[pathway_name])
        else:
            print(f"Pathway '{pathway_name}' from CSV not found in graph")

    print(f'Matching nodes to keep: {len(kept_node_ids)} out of {graph.number_of_nodes()} original nodes')

    filtered_graph = graph.subgraph(kept_node_ids).copy()

    print(f'Graph filtered from {graph.number_of_nodes()} to {filtered_graph.number_of_nodes()} nodes')
    print(f'Graph filtered from {graph.number_of_edges()} to {filtered_graph.number_of_edges()} edges')

    return filtered_graph
