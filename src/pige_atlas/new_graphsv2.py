"""Generate sparse, interpretable drug response graphs from pathway knockout data.

Multi-phase algorithm:
1. Primary pathway selection based on AAC and importance
2. Secondary pathway selection via recursive edge expansion
3. Edge selection for pathway-pathway interactions
4. Gene annotations for each pathway
5. Pruning of unreachable nodes
"""

import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional


def load_cell_line_names(model_file: str = "data/input_data/Model.csv") -> Dict[str, str]:
    """Load cell line name mappings from Model.csv."""
    df = pd.read_csv(model_file)
    return dict(zip(df['ModelID'], df['CellLineName']))


class GraphGeneratorV2:
    """Generates sparse interpretable graphs from knockout data."""

    def __init__(
        self,
        single_pathway_file: str,
        single_gene_file: str,
        double_pathway_file: str,
        drug_name: str = "Drug",
        gmt_file: Optional[str] = None,
        min_primary_pathways: int = 4,
        min_secondary_pathways: int = 4,
        min_edges: int = 10,
        primary_range: Tuple[int, int] = (4, 7),
        max_total_pathways: int = 15,
        max_edges_per_pathway: int = 15,
        top_genes_per_pathway: int = 3,
        global_edge_percentile: float = 95.0,
        global_pathway_percentile: float = 95.0,
        selection_strategy: str = "importance_based",
    ) -> None:
        """Initialize the graph generator."""
        self.drug_name = drug_name
        self.min_primary_pathways = min_primary_pathways
        self.min_secondary_pathways = min_secondary_pathways
        self.min_edges = min_edges
        self.primary_range = primary_range
        self.max_total_pathways = max_total_pathways
        self.max_edges_per_pathway = max_edges_per_pathway
        self.top_genes_per_pathway = top_genes_per_pathway
        self.global_edge_percentile = global_edge_percentile
        self.global_pathway_percentile = global_pathway_percentile
        self.selection_strategy = selection_strategy

        self.pathway_genes = {}
        self.pathway_name_map = {}

        if gmt_file:
            self._load_gmt_file(gmt_file)

        print(f"Loading data for {drug_name}")
        self.df_single_pathway = pd.read_csv(single_pathway_file, index_col=0)
        self.df_single_gene = pd.read_csv(single_gene_file, index_col=0)
        self.df_double_pathway = pd.read_csv(double_pathway_file, index_col=0)

        self.pathway_cols = [c for c in self.df_single_pathway.columns
                            if c not in ['predicted_aac', 'actual_aac']]
        self.gene_cols = [c for c in self.df_single_gene.columns
                         if c not in ['predicted_aac', 'actual_aac']]
        self.edge_cols = [c for c in self.df_double_pathway.columns
                         if c not in ['predicted_aac', 'actual_aac']]

        print(f"Loaded: {len(self.pathway_cols)} pathways, {len(self.gene_cols)} genes, {len(self.edge_cols)} edges")

        self._compute_global_thresholds()
        self._build_edge_lookup()
        self.cell_line_names = load_cell_line_names()

    def _load_gmt_file(self, gmt_file: str) -> None:
        """Load GMT file and create pathway-gene mappings."""
        print(f"Loading GMT file: {gmt_file}")
        with open(gmt_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    go_id = parts[0]
                    pathway_name = parts[1]
                    genes = parts[2:]
                    self.pathway_genes[go_id] = genes
                    self.pathway_genes[pathway_name] = genes
                    self.pathway_name_map[go_id] = pathway_name
        print(f"Loaded {len(self.pathway_name_map)} pathway-gene mappings")

    def _compute_global_thresholds(self) -> None:
        """Compute global significance thresholds across all cell lines."""
        pathway_scores = self.df_single_pathway[self.pathway_cols].values.flatten()
        pathway_scores = pathway_scores[~np.isnan(pathway_scores)]
        self.global_pathway_threshold = np.percentile(
            np.abs(pathway_scores),
            self.global_pathway_percentile
        )

        edge_scores = self.df_double_pathway[self.edge_cols].values.flatten()
        edge_scores = edge_scores[~np.isnan(edge_scores)]
        self.global_edge_threshold = np.percentile(
            np.abs(edge_scores),
            self.global_edge_percentile
        )

        print(f"Global thresholds - Pathway: {self.global_pathway_threshold:.6f}, Edge: {self.global_edge_threshold:.6f}")

    def _build_edge_lookup(self) -> None:
        """Build lookup dictionary for edges: (source, target) -> edge_col_name."""
        self.edge_lookup = {}
        self.edge_reverse_lookup = {}

        for edge_col in self.edge_cols:
            if ' → ' in edge_col:
                parts = edge_col.split(' → ')
                if len(parts) == 2:
                    source, target = parts[0].strip(), parts[1].strip()
                    self.edge_lookup[(source, target)] = edge_col
                    if target not in self.edge_reverse_lookup:
                        self.edge_reverse_lookup[target] = []
                    if source not in self.edge_reverse_lookup:
                        self.edge_reverse_lookup[source] = []
                    self.edge_reverse_lookup[target].append(source)
                    self.edge_reverse_lookup[source].append(target)

    def _get_aac_percentile(self, predicted_aac: float) -> float:
        """Get the percentile rank of a given AAC value."""
        all_aacs = self.df_single_pathway['predicted_aac'].values
        percentile = (all_aacs < predicted_aac).sum() / len(all_aacs) * 100
        return percentile

    def _find_elbow_point(self, values: np.ndarray, min_n: int = 3, max_n: int = 7) -> int:
        """Find the elbow point in a sorted array using maximum dropoff method."""
        if len(values) < min_n:
            return len(values)

        values = values[:max_n]

        if len(values) <= min_n:
            return len(values)

        diffs = np.abs(np.diff(values))

        if len(diffs) > 0:
            max_dropoff_idx = np.argmax(diffs) + 1
            return max(min_n, min(max_dropoff_idx + 1, max_n))
        else:
            return min_n

    def _select_primary_pathways_for_cell(
        self,
        cell_id: str,
        predicted_aac: float
    ) -> List[Tuple[str, float]]:
        """Select primary pathways based on AAC thresholds and elbow detection."""
        pathway_scores = self.df_single_pathway.loc[cell_id, self.pathway_cols]

        resistance_pathways = pathway_scores[pathway_scores < 0].sort_values()
        sensitivity_pathways = pathway_scores[pathway_scores > 0].sort_values(ascending=False)

        if predicted_aac >= 0.2:
            sensitivity_count = max(
                self.min_primary_pathways,
                self._find_elbow_point(
                    sensitivity_pathways.values,
                    self.primary_range[0],
                    self.primary_range[1]
                )
            )
            resistance_count = 0
            quintile_desc = "Top (≥0.2)"
        elif predicted_aac >= 0.15:
            min_sensitivity = max(2, self.min_primary_pathways - 1)
            max_sensitivity = min(6, self.primary_range[1] - 1)
            sensitivity_count = self._find_elbow_point(
                sensitivity_pathways.values,
                min_sensitivity,
                max_sensitivity
            )
            resistance_count = max(1, self.min_primary_pathways - sensitivity_count)
            resistance_count = min(resistance_count, len(resistance_pathways))
            quintile_desc = "Second (0.15-0.2)"
        elif predicted_aac >= 0.1:
            r_elbow = self._find_elbow_point(np.abs(resistance_pathways.values), 1, 3)
            s_elbow = self._find_elbow_point(sensitivity_pathways.values, 1, self.primary_range[1] - r_elbow)
            total = r_elbow + s_elbow
            if total < self.min_primary_pathways:
                needed = self.min_primary_pathways - total
                if abs(np.mean(resistance_pathways.values[:3])) > abs(np.mean(sensitivity_pathways.values[:3])):
                    r_elbow = min(r_elbow + needed, len(resistance_pathways))
                else:
                    s_elbow = min(s_elbow + needed, len(sensitivity_pathways))
            resistance_count = r_elbow
            sensitivity_count = s_elbow
            quintile_desc = "Middle (0.1-0.15)"
        elif predicted_aac >= 0.05:
            min_resistance = max(2, self.min_primary_pathways - 1)
            max_resistance = min(6, self.primary_range[1] - 1)
            resistance_count = self._find_elbow_point(
                np.abs(resistance_pathways.values),
                min_resistance,
                max_resistance
            )
            sensitivity_count = max(1, self.min_primary_pathways - resistance_count)
            sensitivity_count = min(sensitivity_count, len(sensitivity_pathways))
            quintile_desc = "Fourth (0.05-0.1)"
        else:
            resistance_count = max(
                self.min_primary_pathways,
                self._find_elbow_point(
                    np.abs(resistance_pathways.values),
                    self.primary_range[0],
                    self.primary_range[1]
                )
            )
            sensitivity_count = 0
            quintile_desc = "Bottom (<0.05)"

        selected = []

        for i, (pathway, score) in enumerate(resistance_pathways.items()):
            if i >= resistance_count:
                break
            selected.append((pathway, score))

        for i, (pathway, score) in enumerate(sensitivity_pathways.items()):
            if i >= sensitivity_count:
                break
            selected.append((pathway, score))

        print(f"Cell {cell_id} (AAC={predicted_aac:.3f}, quintile={quintile_desc}): "
              f"Selected {len(selected)} primary pathways ({resistance_count} resistance, {sensitivity_count} sensitivity)")

        return selected

    def _select_primary_pathways_importance_based(self, cell_id: str) -> List[Tuple[str, float]]:
        """Select pathways purely by absolute importance magnitude, ignoring AAC predictions."""
        pathway_scores = self.df_single_pathway.loc[cell_id, self.pathway_cols]

        resistance_pathways = pathway_scores[pathway_scores < 0].sort_values()
        sensitivity_pathways = pathway_scores[pathway_scores > 0].sort_values(ascending=False)

        avg_resistance_importance = np.abs(resistance_pathways.values[:10]).mean() if len(resistance_pathways) > 0 else 0
        avg_sensitivity_importance = sensitivity_pathways.values[:10].mean() if len(sensitivity_pathways) > 0 else 0

        all_pathways_abs = pathway_scores.abs().sort_values(ascending=False)

        n_pathways = self._find_elbow_point(
            all_pathways_abs.values,
            min_n=self.primary_range[0],
            max_n=self.primary_range[1]
        )
        n_pathways = max(n_pathways, self.min_primary_pathways)

        if avg_resistance_importance > avg_sensitivity_importance:
            ratio = avg_resistance_importance / (avg_resistance_importance + avg_sensitivity_importance)
            resistance_count = int(n_pathways * ratio)
            sensitivity_count = n_pathways - resistance_count
        else:
            ratio = avg_sensitivity_importance / (avg_resistance_importance + avg_sensitivity_importance)
            sensitivity_count = int(n_pathways * ratio)
            resistance_count = n_pathways - sensitivity_count

        if len(resistance_pathways) > 0 and resistance_count == 0:
            resistance_count = 1
            sensitivity_count = n_pathways - 1
        if len(sensitivity_pathways) > 0 and sensitivity_count == 0:
            sensitivity_count = 1
            resistance_count = n_pathways - 1

        selected = []

        for i, (pathway, score) in enumerate(resistance_pathways.items()):
            if i >= resistance_count:
                break
            selected.append((pathway, score))

        for i, (pathway, score) in enumerate(sensitivity_pathways.items()):
            if i >= sensitivity_count:
                break
            selected.append((pathway, score))

        print(f"Cell {cell_id} (Importance-Based): Selected {len(selected)} primary pathways "
              f"({resistance_count} resistance, {sensitivity_count} sensitivity)")

        return selected

    def _get_outgoing_edges(self, pathway: str, cell_id: str) -> List[Tuple[str, float, float]]:
        """Get all outgoing edges from a pathway with their scores."""
        outgoing = []
        for (source, target), edge_col in self.edge_lookup.items():
            if source == pathway:
                edge_score = self.df_double_pathway.loc[cell_id, edge_col]
                if target in self.pathway_cols:
                    target_score = self.df_single_pathway.loc[cell_id, target]
                    outgoing.append((target, edge_score, target_score))
        return outgoing

    def _select_secondary_pathways_for_cell(
        self,
        cell_id: str,
        primary_pathways: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """Recursively add significant downstream pathways."""
        current_pathways = set([p[0] for p in primary_pathways])
        secondary_pathways = []

        to_explore = list(current_pathways)
        explored = set()

        while to_explore and len(current_pathways) + len(secondary_pathways) < self.max_total_pathways:
            pathway = to_explore.pop(0)
            if pathway in explored:
                continue
            explored.add(pathway)

            outgoing = self._get_outgoing_edges(pathway, cell_id)

            if not outgoing:
                continue

            outgoing.sort(key=lambda x: abs(x[1]), reverse=True)

            for target, edge_score, target_score in outgoing:
                if abs(edge_score) < self.global_edge_threshold:
                    continue

                if abs(target_score) < self.global_pathway_threshold:
                    continue

                if target in current_pathways or any(p[0] == target for p in secondary_pathways):
                    continue

                secondary_pathways.append((target, target_score))
                to_explore.append(target)
                break

        if len(secondary_pathways) < self.min_secondary_pathways:
            print(f"Adding more pathways to reach minimum of {self.min_secondary_pathways}")

            all_candidates = []
            for pathway in current_pathways:
                outgoing = self._get_outgoing_edges(pathway, cell_id)
                for target, edge_score, target_score in outgoing:
                    if target not in current_pathways and not any(p[0] == target for p in secondary_pathways):
                        all_candidates.append((target, target_score, edge_score))

            all_candidates.sort(key=lambda x: abs(x[1]), reverse=True)

            for target, target_score, edge_score in all_candidates:
                if len(secondary_pathways) >= self.min_secondary_pathways:
                    break
                if target not in current_pathways and not any(p[0] == target for p in secondary_pathways):
                    secondary_pathways.append((target, target_score))

        print(f"Cell {cell_id}: Added {len(secondary_pathways)} secondary pathways")

        return secondary_pathways

    def _select_edges_for_cell(self, cell_id: str, all_pathways: List[str]) -> List[Tuple[str, str, float]]:
        """Select significant edges between pathways."""
        edges = []
        pathway_set = set(all_pathways)

        for pathway in all_pathways:
            pathway_edges = []

            for (source, target), edge_col in self.edge_lookup.items():
                if source == pathway and target in pathway_set:
                    edge_score = self.df_double_pathway.loc[cell_id, edge_col]

                    if abs(edge_score) >= self.global_edge_threshold:
                        pathway_edges.append((source, target, edge_score))

            pathway_edges.sort(key=lambda x: abs(x[2]), reverse=True)
            edges.extend(pathway_edges[:self.max_edges_per_pathway])

        if len(edges) < self.min_edges:
            print(f"Adding more edges to reach minimum of {self.min_edges}")

            all_edges = []
            for (source, target), edge_col in self.edge_lookup.items():
                if source in pathway_set and target in pathway_set:
                    edge_score = self.df_double_pathway.loc[cell_id, edge_col]
                    if not any(e[0] == source and e[1] == target for e in edges):
                        all_edges.append((source, target, edge_score))

            all_edges.sort(key=lambda x: abs(x[2]), reverse=True)

            for edge in all_edges:
                if len(edges) >= self.min_edges:
                    break
                edges.append(edge)

        print(f"Cell {cell_id}: Selected {len(edges)} edges")

        return edges

    def _annotate_genes_for_pathways(self, cell_id: str, pathways: List[str]) -> Dict[str, List[Tuple[str, int]]]:
        """Annotate each pathway with top genes using GMT mappings."""
        if cell_id not in self.df_single_gene.index:
            return {pathway: [] for pathway in pathways}

        gene_annotations = {}
        gene_scores = self.df_single_gene.loc[cell_id, self.gene_cols]
        gene_ranks = gene_scores.abs().rank(ascending=False, method='min')

        for pathway in pathways:
            pathway_genes_list = self.pathway_genes.get(pathway, [])

            if not pathway_genes_list:
                top_genes = gene_scores.abs().nlargest(5)
                gene_annotations[pathway] = [
                    (gene, int(gene_ranks[gene]))
                    for gene in top_genes.index
                ]
            else:
                available_genes = [g for g in pathway_genes_list if g in self.gene_cols]

                if not available_genes:
                    gene_annotations[pathway] = []
                else:
                    pathway_gene_scores = gene_scores[available_genes]
                    top_pathway_genes = pathway_gene_scores.abs().nlargest(min(5, len(available_genes)))

                    gene_annotations[pathway] = [
                        (gene, int(gene_ranks[gene]))
                        for gene in top_pathway_genes.index
                    ]

        return gene_annotations

    def _identify_shared_genes(
        self,
        cell_id: str,
        gene_annotations: Dict[str, List[Tuple[str, int]]]
    ) -> Dict[str, Dict]:
        """Identify genes that appear in top-5 of multiple pathways."""
        if cell_id not in self.df_single_gene.index:
            return {}

        gene_scores = self.df_single_gene.loc[cell_id, self.gene_cols]
        gene_ranks = gene_scores.abs().rank(ascending=False, method='min')

        gene_to_pathways = {}
        for pathway, genes in gene_annotations.items():
            for gene, rank in genes:
                if gene not in gene_to_pathways:
                    gene_to_pathways[gene] = []
                gene_to_pathways[gene].append(pathway)

        shared_genes = {}
        for gene, pathways in gene_to_pathways.items():
            if len(pathways) >= 2:
                shared_genes[gene] = {
                    'score': gene_scores[gene],
                    'pathways': pathways,
                    'rank': int(gene_ranks[gene])
                }

        print(f"Cell {cell_id}: Found {len(shared_genes)} shared genes across pathways")

        return shared_genes

    def _prune_unreachable_nodes(
        self,
        pathways: List[Tuple[str, float]],
        edges: List[Tuple[str, str, float]],
        primary_pathway_names: set
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[str, str, float]]]:
        """Remove nodes not reachable from the drug node."""
        G = nx.DiGraph()
        G.add_node("DRUG")

        for pathway, score in pathways:
            G.add_node(pathway)

        for pathway, score in pathways:
            if pathway in primary_pathway_names:
                G.add_edge("DRUG", pathway)

        for source, target, score in edges:
            G.add_edge(source, target)

        reachable = set(nx.descendants(G, "DRUG"))
        reachable.add("DRUG")

        pruned_pathways = [(p, s) for p, s in pathways if p in reachable]
        pruned_edges = [(s, t, score) for s, t, score in edges if s in reachable and t in reachable]

        removed = len(pathways) - len(pruned_pathways)
        if removed > 0:
            print(f"Pruned {removed} unreachable pathways")

        return pruned_pathways, pruned_edges

    def generate_graph_for_cell(self, cell_id: str) -> Dict:
        """Generate a complete graph for a single cell line."""
        print(f"\n{'='*60}")
        print(f"Generating graph for cell: {cell_id}")
        print(f"{'='*60}")

        predicted_aac = self.df_single_pathway.loc[cell_id, 'predicted_aac']

        if self.selection_strategy == "importance_based":
            primary_pathways = self._select_primary_pathways_importance_based(cell_id)
        else:
            primary_pathways = self._select_primary_pathways_for_cell(cell_id, predicted_aac)

        secondary_pathways = self._select_secondary_pathways_for_cell(cell_id, primary_pathways)

        primary_pathway_names = set([p[0] for p in primary_pathways])

        all_pathways = primary_pathways + secondary_pathways
        pathway_names = [p[0] for p in all_pathways]

        edges = self._select_edges_for_cell(cell_id, pathway_names)
        gene_annotations = self._annotate_genes_for_pathways(cell_id, pathway_names)
        shared_genes = self._identify_shared_genes(cell_id, gene_annotations)

        all_pathways, edges = self._prune_unreachable_nodes(all_pathways, edges, primary_pathway_names)

        result = {
            'cell_id': cell_id,
            'predicted_aac': predicted_aac,
            'pathways': all_pathways,
            'primary_pathways': primary_pathway_names,
            'edges': edges,
            'gene_annotations': gene_annotations,
            'shared_genes': shared_genes,
        }

        print(f"Final graph: {len(all_pathways)} pathways, {len(edges)} edges, {len(shared_genes)} shared genes")

        return result

    def generate_average_graph(self) -> Dict:
        """Generate an average graph across all cell lines."""
        print(f"\n{'='*60}")
        print(f"Generating AVERAGE graph across all cell lines")
        print(f"{'='*60}")

        pathway_cols = [c for c in self.df_single_pathway.columns if c not in ['predicted_aac', 'actual_aac']]
        avg_pathway_scores = self.df_single_pathway[pathway_cols].mean(axis=0)

        edge_cols = [c for c in self.df_double_pathway.columns if c not in ['predicted_aac', 'actual_aac']]
        avg_edge_scores = self.df_double_pathway[edge_cols].mean(axis=0)

        gene_cols = [c for c in self.df_single_gene.columns if c not in ['predicted_aac', 'actual_aac']]
        avg_gene_scores = self.df_single_gene[gene_cols].mean(axis=0)

        avg_predicted_aac = self.df_single_pathway['predicted_aac'].mean()

        avg_single_df = pd.DataFrame([avg_pathway_scores], columns=pathway_cols)
        avg_single_df['predicted_aac'] = avg_predicted_aac
        avg_single_df.index = ['AVERAGE']

        avg_double_df = pd.DataFrame([avg_edge_scores], columns=edge_cols)
        avg_double_df['predicted_aac'] = avg_predicted_aac
        avg_double_df.index = ['AVERAGE']

        avg_gene_df = pd.DataFrame([avg_gene_scores], columns=gene_cols)
        avg_gene_df['predicted_aac'] = avg_predicted_aac
        avg_gene_df.index = ['AVERAGE']

        original_single = self.df_single_pathway
        original_double = self.df_double_pathway
        original_gene = self.df_single_gene

        self.df_single_pathway = avg_single_df
        self.df_double_pathway = avg_double_df
        self.df_single_gene = avg_gene_df

        avg_graph = self.generate_graph_for_cell('AVERAGE')

        self.df_single_pathway = original_single
        self.df_double_pathway = original_double
        self.df_single_gene = original_gene

        print(f"Average graph: {len(avg_graph['pathways'])} pathways, {len(avg_graph['edges'])} edges")

        return avg_graph

    def generate_graphs_all_cells(self, include_average: bool = True) -> Dict[str, Dict]:
        """Generate graphs for all cell lines."""
        results = {}

        for cell_id in self.df_single_pathway.index:
            results[cell_id] = self.generate_graph_for_cell(cell_id)

        if include_average:
            results['AVERAGE'] = self.generate_average_graph()

        return results

    def save_graph_to_csv(self, graph_data: Dict, output_dir: Path) -> None:
        """Save graph data to CSV files (nodes and edges)."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cell_id = graph_data['cell_id']
        cell_line_name = self.cell_line_names.get(cell_id, cell_id).replace(' ', '_').replace('/', '_').replace('\\', '_')

        nodes_data = []

        nodes_data.append({
            'node_id': 'DRUG',
            'node_type': 'drug',
            'label': self.drug_name,
            'causal_importance': graph_data['predicted_aac'],
            'top_genes': '',
        })

        for pathway, score in graph_data['pathways']:
            genes = graph_data['gene_annotations'].get(pathway, [])
            genes_str = ', '.join([f"{gene} ({rank})" for gene, rank in genes[:3]])
            node_type = 'resistance' if score < 0 else 'sensitivity'

            nodes_data.append({
                'node_id': pathway,
                'node_type': node_type,
                'label': pathway,
                'causal_importance': score,
                'top_genes': genes_str,
            })

        shared_genes = graph_data.get('shared_genes', {})
        for gene, gene_data in shared_genes.items():
            node_type = 'gene_resistance' if gene_data['score'] < 0 else 'gene_sensitivity'
            connected_pathways = ', '.join(gene_data['pathways'])

            nodes_data.append({
                'node_id': gene,
                'node_type': node_type,
                'label': gene,
                'causal_importance': gene_data['score'],
                'top_genes': f"Rank: {gene_data['rank']}, Pathways: {connected_pathways}",
            })

        nodes_df = pd.DataFrame(nodes_data)
        nodes_file = output_dir / f"{cell_line_name}_nodes.csv"
        print(nodes_file)
        nodes_df.to_csv(nodes_file, index=False)
        print(f"Saved nodes to {nodes_file}")

        edges_data = []

        pathway_pairs_with_genes = set()
        for gene, gene_data in shared_genes.items():
            pathways = gene_data['pathways']
            for i in range(len(pathways)):
                for j in range(i + 1, len(pathways)):
                    pathway_pairs_with_genes.add((pathways[i], pathways[j]))
                    pathway_pairs_with_genes.add((pathways[j], pathways[i]))

        primary_pathways = graph_data.get('primary_pathways', set())
        for pathway, score in graph_data['pathways']:
            if pathway in primary_pathways:
                edges_data.append({
                    'source': 'DRUG',
                    'target': pathway,
                    'edge_type': 'drug_pathway',
                    'causal_importance': score,
                    'in_pathway_mode': True,
                    'in_gene_mode': True,
                })

        pathway_types = {p: ('resistance' if s < 0 else 'sensitivity') for p, s in graph_data['pathways']}

        for source, target, score in graph_data['edges']:
            edge_type = pathway_types.get(source, 'sensitivity')
            in_gene_mode = (source, target) not in pathway_pairs_with_genes

            edges_data.append({
                'source': source,
                'target': target,
                'edge_type': edge_type,
                'causal_importance': score,
                'in_pathway_mode': True,
                'in_gene_mode': in_gene_mode,
            })

        for gene, gene_data in shared_genes.items():
            for pathway in gene_data['pathways']:
                edge_type = pathway_types.get(pathway, 'sensitivity')

                edges_data.append({
                    'source': pathway,
                    'target': gene,
                    'edge_type': edge_type,
                    'causal_importance': gene_data['score'],
                    'in_pathway_mode': False,
                    'in_gene_mode': True,
                })

        edges_df = pd.DataFrame(edges_data)
        edges_file = output_dir / f"{cell_line_name}_edges.csv"
        edges_df.to_csv(edges_file, index=False)
        print(f"Saved edges to {edges_file}")


def discover_available_drugs(base_dir: Path, dataset: str) -> List[str]:
    """Discover available drugs for a given dataset by scanning the raw_scores directory.

    Args:
        base_dir: Base directory containing the knockout data
        dataset: Dataset name

    Returns:
        List of drug names that have data files for this dataset
    """
    drugs = set()
    raw_scores_dir = base_dir / "SinglePathwayKO" / dataset / "raw_scores"

    if not raw_scores_dir.exists():
        print(f"Warning: Directory not found: {raw_scores_dir}")
        return []

    # Look for files matching pattern: {drug}_{dataset}_knockout_raw_scores.csv
    for file_path in raw_scores_dir.glob(f"*_{dataset}_knockout_raw_scores.csv"):
        drug_name = file_path.stem.replace(f"_{dataset}_knockout_raw_scores", "")
        drugs.add(drug_name)

    return sorted(list(drugs))


def run_graph_generation(config: Dict) -> None:
    """Main entry point for graph generation using config dictionary."""
    base_dir = Path(config['base_dir'])
    datasets = config['dataset']
    requested_drug_names = config.get('drug_names')

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset}")
        print(f"{'='*60}")

        # Discover available drugs for this dataset
        available_drugs = discover_available_drugs(base_dir, dataset)

        if not available_drugs:
            print(f"No drugs found for dataset {dataset}, skipping...")
            continue

        # Filter to requested drugs if specified, otherwise use all available
        if requested_drug_names:
            drug_names = [d for d in requested_drug_names if d in available_drugs]
            skipped = [d for d in requested_drug_names if d not in available_drugs]
            if skipped:
                print(f"Skipping drugs not available for {dataset}: {', '.join(skipped)}")
        else:
            drug_names = available_drugs

        if not drug_names:
            print(f"No matching drugs to process for dataset {dataset}, skipping...")
            continue

        print(f"Processing {len(drug_names)} drugs for {dataset}: {', '.join(drug_names)}")

        for drug_name in drug_names:
            print(f"\nProcessing drug: {drug_name}")

            generator = GraphGeneratorV2(
                single_pathway_file=f"{base_dir}/SinglePathwayKO/{dataset}/raw_scores/{drug_name}_{dataset}_knockout_raw_scores.csv",
                single_gene_file=f"{base_dir}/SingleGeneKO/{dataset}/raw_scores/{drug_name}_{dataset}_knockout_raw_scores.csv",
                double_pathway_file=f"{base_dir}/DoublePathwayKO/{dataset}/raw_scores/{drug_name}_{dataset}_knockout_raw_scores.csv",
                drug_name=drug_name,
                gmt_file="data/input_data/all_pathway_genesets.gmt",
                min_primary_pathways=5,
                min_secondary_pathways=3,
                min_edges=15,
                primary_range=(5, 8),
            )

            csv_output_dir = base_dir / "PIGE_graphs" / dataset / drug_name / "debug_csv"
            csv_output_dir.mkdir(parents=True, exist_ok=True)

            all_graphs = generator.generate_graphs_all_cells()

            for cell_id, graph in all_graphs.items():
                generator.save_graph_to_csv(graph, csv_output_dir)

            summary_stats = []
            for cell_id, graph in all_graphs.items():
                summary_stats.append({
                    'cell_id': cell_id,
                    'predicted_aac': graph['predicted_aac'],
                    'num_pathways': len(graph['pathways']),
                    'num_edges': len(graph['edges']),
                    'num_resistance': sum(1 for _, score in graph['pathways'] if score < 0),
                    'num_sensitivity': sum(1 for _, score in graph['pathways'] if score > 0),
                })

            summary_df = pd.DataFrame(summary_stats)
            summary_file = csv_output_dir / "summary_statistics.csv"
            summary_df.to_csv(summary_file, index=False)
            print(f"Saved summary statistics to {summary_file}")

            print(f"\nFinished processing drug: {drug_name}")

        print(f"\nFinished processing dataset: {dataset}")
