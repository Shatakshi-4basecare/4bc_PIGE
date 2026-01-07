"""Functions for building pathway interaction graphs using OmniPath data.

This module provides functionality to:
- Retrieve high-confidence molecular interactions from OmniPath
- Build crosstalk edges between pathways based on gene interactions
- Construct gene-level interaction graphs
"""

import os
import omnipath as op
import pandas as pd
import networkx as nx
from tqdm import tqdm


def get_omnipath_interactions(datasets, min_refs, cache_path=None):
    """Download and filter high-confidence molecular interactions from OmniPath.

    Args:
        datasets: List of OmniPath dataset names to query.
        min_refs: Minimum number of literature references required.
        cache_path: Optional path to cached OmniPath data csv file.
                   If provided and file exists, loads from cache instead of querying.

    Returns:
        DataFrame containing filtered high-confidence interactions.
    """
    if cache_path and os.path.exists(cache_path):
        print(f'Loading cached OmniPath data from {cache_path}')
        with open(cache_path, 'r') as f:
            cached_data = pd.read_csv(f)
            print(f'Loaded {len(cached_data)} cached interactions')
            cached_data['n_references'] = cached_data['references'].apply(
                lambda x: len(str(x).split(';')) if pd.notna(x) else 0
            )
            filtered_interactions = cached_data[cached_data['n_references'] >= min_refs].copy()
            print(f'Filtered to {len(filtered_interactions)} interactions with at least {min_refs} references')
            return filtered_interactions

    print(f'Querying OmniPath (live database) for interactions from {len(datasets)} datasets (min refs: {min_refs})')

    interactions = op.interactions.OmniPath.get(
        organisms='human',
        datasets=datasets,
        fields=['sources', 'references', 'type'],
        genesymbols=True
    )

    print(f'Downloaded {len(interactions)} total interactions from OmniPath')

    interactions['n_references'] = interactions['references'].apply(
        lambda x: len(str(x).split(';')) if pd.notna(x) else 0
    )
    filtered_interactions = interactions[interactions['n_references'] >= min_refs].copy()

    print(f'Filtered to {len(filtered_interactions)} interactions with at least {min_refs} references')
    return filtered_interactions


def build_crosstalk_edges(node_to_genes_map, omnipath_df):
    """Build directed crosstalk edges between pathways based on OmniPath interactions.

    Args:
        node_to_genes_map: Dictionary mapping pathway IDs to sets of gene symbols.
        omnipath_df: DataFrame of OmniPath interactions.

    Returns:
        Tuple of (edges_list, evidence_dict) where edges_list contains pathway pairs
        and evidence_dict maps edges to supporting gene interaction evidence.
    """
    print('Building crosstalk edges between pathways using OmniPath data')

    gene_to_pathway_map = {}
    for pathway_id, genes in node_to_genes_map.items():
        for gene in genes:
            if gene not in gene_to_pathway_map:
                gene_to_pathway_map[gene] = []
            gene_to_pathway_map[gene].append(pathway_id)

    crosstalk_edges_with_evidence = {}

    for _, row in tqdm(omnipath_df.iterrows(), total=len(omnipath_df),
                       desc='Mapping OmniPath to Pathways'):
        source_gene = row['source_genesymbol']
        target_gene = row['target_genesymbol']

        source_pathways = gene_to_pathway_map.get(source_gene, [])
        target_pathways = gene_to_pathway_map.get(target_gene, [])

        if source_pathways and target_pathways:
            for sp in source_pathways:
                for tp in target_pathways:
                    if sp != tp:
                        edge = (sp, tp)
                        if edge not in crosstalk_edges_with_evidence:
                            crosstalk_edges_with_evidence[edge] = []
                        crosstalk_edges_with_evidence[edge].append(f'{source_gene}->{target_gene}')

    crosstalk_edges = list(crosstalk_edges_with_evidence.keys())

    for edge, evidence in crosstalk_edges_with_evidence.items():
        if len(evidence) > 10:
            crosstalk_edges_with_evidence[edge] = evidence[:10] + [f'... and {len(evidence) - 10} more']

    print(f'Generated {len(crosstalk_edges)} unique directed crosstalk edges')
    return crosstalk_edges, crosstalk_edges_with_evidence


def build_gene_interaction_graph(pathway_graph, omnipath_df):
    """Build a directed graph of gene interactions for genes in final pathways.

    Args:
        pathway_graph: NetworkX graph with pathway nodes containing 'genes' attribute.
        omnipath_df: DataFrame containing OmniPath interactions.

    Returns:
        NetworkX DiGraph with genes as nodes and interactions as edges.
    """
    print('Building gene interaction graph from final pathways')

    pathway_genes = set()
    for node_id, node_data in pathway_graph.nodes(data=True):
        genes = node_data.get('genes', set())
        pathway_genes.update(genes)

    print(f'Found {len(pathway_genes)} unique genes across all final pathways')

    gene_graph = nx.DiGraph()
    gene_graph.add_nodes_from(pathway_genes)

    filtered_interactions = omnipath_df[
        (omnipath_df['source_genesymbol'].isin(pathway_genes)) &
        (omnipath_df['target_genesymbol'].isin(pathway_genes))
    ].copy()

    print(f'Filtered to {len(filtered_interactions)} gene interactions involving pathway genes')

    for _, row in tqdm(filtered_interactions.iterrows(), total=len(filtered_interactions),
                       desc='Building gene graph'):
        source_gene = row['source_genesymbol']
        target_gene = row['target_genesymbol']

        edge_data = {
            'interaction_type': row.get('type', 'unknown'),
            'references': row.get('references', ''),
            'sources': row.get('sources', ''),
            'n_references': row.get('n_references', 0)
        }

        if gene_graph.has_edge(source_gene, target_gene):
            existing_refs = gene_graph[source_gene][target_gene].get('n_references', 0)
            if edge_data['n_references'] > existing_refs:
                gene_graph[source_gene][target_gene].update(edge_data)
        else:
            gene_graph.add_edge(source_gene, target_gene, **edge_data)

    print(f'Gene interaction graph: {gene_graph.number_of_nodes()} nodes, {gene_graph.number_of_edges()} edges')

    return gene_graph
