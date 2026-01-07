"""Functions for parsing and processing Gene Ontology data.

This module provides functionality to:
- Load GO graphs from OBO files
- Load and filter GO annotations from GAF files
- Map gene annotations to GO graph nodes
- Load gene universe definitions
"""

import pandas as pd
import networkx as nx
from goatools.obo_parser import GODag
from tqdm import tqdm
from collections import defaultdict
import gzip
import os


def load_go_graph(obo_file_path):
    """Parse GO OBO file into NetworkX DiGraph using GODag.

    Args:
        obo_file_path: Path to GO OBO file.

    Returns:
        NetworkX DiGraph with GO terms as nodes and parent-child relationships as edges.
    """
    print(f'Parsing GO OBO file from: {obo_file_path} using goatools.GODag')
        
    go_dag = GODag(obo_file_path, optional_attrs={'relationship'})
    graph = nx.DiGraph()

    for term_id, term_obj in tqdm(go_dag.items(), desc='Converting GODag to NetworkX'):
        if term_obj.is_obsolete:
            continue

        graph.add_node(term_obj.id, name=term_obj.name, namespace=term_obj.namespace)

        for parent_obj in term_obj.parents:
            if not parent_obj.is_obsolete:
                graph.add_edge(parent_obj.id, term_obj.id)

    print(f'Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges')
    return graph


def load_go_annotations(gaf_file_path, evidence_codes_to_keep, gene_universe_set):
    """Load and filter GO annotations from GAF file.

    Args:
        gaf_file_path: Path to GO annotation file (GAF format).
        evidence_codes_to_keep: List of evidence codes to retain.
        gene_universe_set: Set of gene symbols to filter for.

    Returns:
        DataFrame with filtered annotations containing 'DB_Object_Symbol' and 'GO_ID' columns.
    """
    print(f'Loading and filtering annotations from: {gaf_file_path}')

    gaf_columns = [
        'DB', 'DB_Object_ID', 'DB_Object_Symbol', 'Qualifier', 'GO_ID',
        'DB_Reference', 'Evidence_Code', 'With_From', 'Aspect', 'DB_Object_Name',
        'DB_Object_Synonym', 'DB_Object_Type', 'Taxon', 'Date', 'Assigned_By',
        'Annotation_Extension', 'Gene_Product_Form_ID'
    ]

    if not os.path.exists(gaf_file_path) and os.path.exists(gaf_file_path + '.gz'):
        with gzip.open(gaf_file_path + '.gz', 'rt') as f:
            df = pd.read_csv(f, sep='\t', comment='!', header=None,
                             names=gaf_columns, low_memory=False)
    else:
        df = pd.read_csv(gaf_file_path, sep='\t', comment='!', header=None,
                        names=gaf_columns, low_memory=False)

    initial_count = len(df)
    print(f'Loaded {initial_count} total annotations')

    df = df[df['DB_Object_Symbol'].isin(gene_universe_set)]
    print(f'Annotations after gene universe filter ({len(gene_universe_set)} genes): {len(df)}')

    df = df[df['Evidence_Code'].isin(evidence_codes_to_keep)]
    print(f'Annotations after evidence code filter: {len(df)}')

    df = df[['DB_Object_Symbol', 'GO_ID']].drop_duplicates()
    print(f'Final unique high-confidence annotations: {len(df)}')

    return df


def map_annotations_to_graph(graph, annotations_df):
    """Map gene annotations to GO graph nodes.

    Args:
        graph: NetworkX DiGraph of GO terms.
        annotations_df: DataFrame with 'DB_Object_Symbol' and 'GO_ID' columns.

    Returns:
        NetworkX DiGraph with 'genes' attribute added to nodes containing gene sets.
    """
    print('Mapping gene annotations to graph nodes')

    nx.set_node_attributes(graph, {node: set() for node in graph.nodes()}, 'genes')

    go_id_to_genes = defaultdict(set)
    for _, row in tqdm(annotations_df.iterrows(), total=len(annotations_df),
                       desc='Aggregating genes per GO term'):
        go_id_to_genes[row['GO_ID']].add(row['DB_Object_Symbol'])

    for go_id, genes in tqdm(go_id_to_genes.items(), desc='Updating graph with gene sets'):
        if go_id in graph:
            graph.nodes[go_id]['genes'] = genes

    return graph


def load_gene_universe(file_path):
    """Load gene universe from CSV file.

    Args:
        file_path: Path to gene universe CSV file.

    Returns:
        Set of gene symbols.
    """
    print(f'Loading gene universe from {file_path}')

    gene_universe = set(pd.read_csv(file_path, header=None)[0].unique())
    print(f'Loaded {len(gene_universe)} unique genes into universe')
    return gene_universe
