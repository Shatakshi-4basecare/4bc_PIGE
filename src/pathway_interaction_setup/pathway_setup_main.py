"""Main pipeline for pathway interaction setup using leaf-up OmniPath-linked approach.

This module executes the full pathway curation pipeline:
1. Load GO graph and gene annotations
2. Filter GO terms by namespace and annotation size
3. Identify leaf nodes in the GO hierarchy
4. Build crosstalk edges using OmniPath PPI interactions
5. Apply optional manual curation filter
6. Build gene interaction graph
7. Validate and export results
"""

import pickle
import os
import networkx as nx
from typing import Dict, Any, Optional

from src.pathway_interaction_setup.pathway_setup_src import utils
from src.pathway_interaction_setup.pathway_setup_src import go_parser
from src.pathway_interaction_setup.pathway_setup_src import filtering
from src.pathway_interaction_setup.pathway_setup_src import graph_builder
from src.pathway_interaction_setup.pathway_setup_src import visualizer


def run_leaf_up_pipeline(config: Optional[Dict[str, Any]] = None) -> bool:
    """Execute the leaf-up OmniPath-linked curation pipeline.

    Args:
        config: Optional configuration dictionary. If None, loads from config.yaml.

    Returns:
        True if pipeline completed successfully.
    """
    if config is None:
        config = utils.load_config()

    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    gene_universe_path = config['gene_universe_file']
    go_obo_path = config['go_obo_file']
    go_annotation_path = config['go_annotation_file']
    manual_curation_csv = config['manual_curation_file']

    gene_universe = go_parser.load_gene_universe(gene_universe_path)
    full_go_graph = go_parser.load_go_graph(go_obo_path)
    high_confidence_annotations = go_parser.load_go_annotations(
        go_annotation_path,
        config['evidence_codes'],
        gene_universe
    )

    annotated_graph = go_parser.map_annotations_to_graph(full_go_graph, high_confidence_annotations)
    bp_graph = filtering.filter_by_namespace(annotated_graph, namespace='biological_process')

    candidate_graph = filtering.filter_by_direct_annotation_size(
        bp_graph,
        config['min_direct_annotation_size'],
        config['max_direct_annotation_size'],
        set(config['size_filter_exceptions'])
    )
    print(f'Identified {candidate_graph.number_of_nodes()} candidate pathways after initial filtering')

    leaf_node_set = set()
    for node in candidate_graph.nodes():
        successors = set(candidate_graph.successors(node))
        if not successors:
            leaf_node_set.add(node)

    print(f'Identified {len(leaf_node_set)} leaf nodes to form the basis of the new graph')

    if not leaf_node_set:
        print('No leaf nodes found after filtering. Cannot proceed')
        return False

    final_graph = nx.DiGraph()
    final_graph.add_nodes_from(
        (n, candidate_graph.nodes[n]) for n in leaf_node_set
    )

    omnipath_df = graph_builder.get_omnipath_interactions(
        config['omnipath_datasets'],
        config['omnipath_min_literature_refs'],
        cache_path=config['omnipath_cache_path']
    )

    leaf_node_to_genes_map = {n: d.get('genes', set()) for n, d in final_graph.nodes(data=True)}

    crosstalk_edges, crosstalk_evidence = graph_builder.build_crosstalk_edges(
        leaf_node_to_genes_map,
        omnipath_df
    )

    print(f'Pruning crosstalk edges with fewer than {config["min_crosstalk_evidence"]} supporting gene interactions')

    pruned_edges = []
    pruned_evidence = {}
    for edge, evidence in crosstalk_evidence.items():
        if len(evidence) >= config['min_crosstalk_evidence']:
            pruned_edges.append(edge)
            pruned_evidence[edge] = evidence

    print(f'Reduced crosstalk edges from {len(crosstalk_edges)} to {len(pruned_edges)}')

    final_graph.add_edges_from(pruned_edges)
    nx.set_edge_attributes(final_graph, 'crosstalk', 'type')
    nx.set_edge_attributes(final_graph, pruned_evidence, 'evidence')

    final_graph = utils.clean_graph(final_graph)

    final_graph = filtering.filter_by_manual_curation(
        final_graph,
        manual_curation_csv
    )

    gene_interaction_graph = graph_builder.build_gene_interaction_graph(final_graph, omnipath_df)

    gene_graph_output_path = os.path.join(output_dir, 'all_gene_interaction_graph.pkl')
    with open(gene_graph_output_path, 'wb') as f:
        pickle.dump(gene_interaction_graph, f)
    print(f'Successfully saved gene interaction graph to: {gene_graph_output_path}')

    num_nodes = final_graph.number_of_nodes()
    num_edges = final_graph.number_of_edges()
    print(f'Final Graph Stats: {num_nodes} nodes, {num_edges} edges')

    edge_types_dict = dict(nx.get_edge_attributes(final_graph, 'type'))
    edge_evidence_dict = dict(nx.get_edge_attributes(final_graph, 'evidence'))

    final_output_data = {
        'nodes': sorted(list(final_graph.nodes())),
        'node_names': {n: d.get('name', 'N/A') for n, d in final_graph.nodes(data=True)},
        'edge_list': [[u, v] for u, v in final_graph.edges()],
        'node_to_genes': {n: d.get('genes', set()) for n, d in final_graph.nodes(data=True)},
        'edge_types': edge_types_dict,
        'edge_evidence': edge_evidence_dict,
    }

    pathway_interaction_output_path = os.path.join(output_dir, 'all_pathway_interaction_graph.pkl')
    with open(pathway_interaction_output_path, 'wb') as f:
        pickle.dump(final_output_data, f)
    print(f'Successfully saved pathway interaction graph to: {pathway_interaction_output_path}')

    num_nodes = len(final_output_data['nodes'])
    num_edges = len(final_output_data['edge_list'])
    print(f'Pathway interaction graph statistics: {num_nodes} pathways, {num_edges} edges')

    gmt_output_path = os.path.join(output_dir, 'all_pathway_genesets.gmt')
    utils.save_gmt_file(final_output_data, gmt_output_path)

    viz_path = os.path.join(output_dir, 'all_pathway_interaction_graph_visualization.html')
    visualizer.create_interactive_visualization(
        graph=final_graph,
        output_filename=viz_path,
        graph_title="PIGE Pathway Interaction Graph"
    )

    return True


if __name__ == '__main__':
    run_leaf_up_pipeline()
