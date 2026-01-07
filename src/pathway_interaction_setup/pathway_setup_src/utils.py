"""Utility functions for pathway setup pipeline."""

import os
import networkx as nx
import yaml


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration YAML file.

    Returns:
        Dictionary containing configuration parameters.
    """
    config_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_config_path = os.path.join(config_dir, config_path)

    with open(full_config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets.

    Args:
        set1: First set.
        set2: Second set.

    Returns:
        Float between 0 and 1 representing similarity.
    """
    if not set1 and not set2:
        return 1.0
    intersection_size = len(set1.intersection(set2))
    union_size = len(set1.union(set2))
    if union_size == 0:
        return 0.0
    return intersection_size / union_size


def clean_graph(graph):
    """Remove isolated nodes and empty components from graph.

    Args:
        graph: NetworkX DiGraph to clean.

    Returns:
        Cleaned NetworkX DiGraph.
    """
    isolated_nodes = list(nx.isolates(graph))
    # Check that 'G protein-coupled receptor signaling pathway, coupled to cyclic nucleotide second messenger' pathway is in the graph
    if 'GO:0007187' in isolated_nodes:
        print('GO:0007187 pathway found in the isolated nodes')
        isolated_nodes.remove('GO:0007187')

    if isolated_nodes:
        print(f'Removing {len(isolated_nodes)} isolated nodes')
        graph.remove_nodes_from(isolated_nodes)

    for component in list(nx.weakly_connected_components(graph)):
        if len(component) <= 1:
            node_to_remove = list(component)[0]
            if graph.degree(node_to_remove) == 0:
                print(f'Removing single-node component: {node_to_remove}')
                graph.remove_node(node_to_remove)

    return graph


def save_gmt_file(graph_data, output_path):
    """Save gene sets from graph data to GMT file format.

    GMT format: GO_ID<tab>GO_NAME<tab>GENE1<tab>GENE2...

    Args:
        graph_data: Dictionary containing graph information with keys 'nodes',
                    'node_names', and 'node_to_genes'.
        output_path: Path for output GMT file.
    """
    print(f'Generating GMT file at: {output_path}')
    with open(output_path, 'w') as f:
        for node_id in sorted(graph_data['nodes']):
            node_name = graph_data['node_names'].get(node_id, 'N/A')
            genes = sorted(list(graph_data['node_to_genes'].get(node_id, set())))

            if genes:
                line = f"{node_id}\t{node_name}\t" + "\t".join(genes) + "\n"
                f.write(line)
    print('Successfully saved GMT file')
