"""PIGE graph visualization.

Creates interactive graph visualizations.
"""

import base64
from pathlib import Path
from typing import Dict, List, Optional
from pyvis.network import Network

import pandas as pd
import networkx as nx
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def load_drug_smiles(drugs_file: str = "data/input_data/drugs_to_process.csv") -> Dict[str, str]:
    """Load drug SMILES from CSV file.

    Args:
        drugs_file: Path to drugs CSV file.

    Returns:
        Dictionary mapping drug_name to smiles string.
    """
    df = pd.read_csv(drugs_file)
    return dict(zip(df['drug_name'], df['smiles']))


def load_cell_line_names(model_file: str = "data/input_data/Model.csv") -> Dict[str, str]:
    """Load cell line names from Model.csv.

    Args:
        model_file: Path to model CSV file.

    Returns:
        Dictionary mapping ModelID to CellLineName.
    """
    df = pd.read_csv(model_file)
    return dict(zip(df['ModelID'], df['CellLineName']))


def render_drug_node_svg(drug_name: str, smiles: str) -> Optional[str]:
    """Render drug structure as SVG data URI.

    Args:
        drug_name: Drug name.
        smiles: SMILES string.

    Returns:
        Base64-encoded SVG data URI or None.
    """
    if not RDKIT_AVAILABLE:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    rdDepictor.Compute2DCoords(mol)

    struct_size = 400
    box_height = 100
    total_height = struct_size + box_height

    drawer = rdMolDraw2D.MolDraw2DSVG(struct_size, struct_size)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    struct_svg = drawer.GetDrawingText()

    struct_svg_lines = struct_svg.split('\n')
    struct_content = '\n'.join([
        line for line in struct_svg_lines
        if not line.strip().startswith('<?xml')
        and not line.strip().startswith('<svg')
        and not line.strip() == '</svg>'
    ])

    composite_svg = f'''<svg width="{struct_size}" height="{total_height}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="warmGradient" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#FFFEF8;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#FFF9E6;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="goldGradient" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#FFE066;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#FFD700;stop-opacity:1" />
    </linearGradient>
  </defs>
  <rect width="{struct_size}" height="{struct_size}" fill="url(#warmGradient)" />
  <g transform="translate(0, 0)">
    {struct_content}
  </g>
  <rect y="{struct_size}" width="{struct_size}" height="{box_height}" fill="url(#goldGradient)" />
</svg>'''

    svg_b64 = base64.b64encode(composite_svg.encode('utf-8')).decode()
    return f"data:image/svg+xml;base64,{svg_b64}"


class PIGEBrandColors:
    """PIGE brand color specifications."""

    PIGE_BLUE = '#1E5AA8'
    PIGE_NAVY = '#0D2137'
    PIGE_WHITE = '#FFFFFF'

    DEEP_BLUE = '#4393C3'
    MEDIUM_BLUE = '#92C5DE'
    LIGHT_BLUE = '#D1E5F0'

    DEEP_RED = '#F4A582'
    MEDIUM_RED = '#FDDBC7'
    LIGHT_RED = '#FEF0E8'

    WARM_GREY = '#E8E8E8'
    DRUG_EDGE = '#7A8DA3'
    DRUG_NODE_GOLD = '#FFD700'
    DRUG_NODE_BORDER = '#B8860B'
    OFF_WHITE = '#FAFAFA'
    GRID_GREY = '#E0E0E0'


class PIGEGraphVisualizer:
    """Visualizes PIGE graphs.

    Args:
        nodes_file: Path to nodes CSV.
        edges_file: Path to edges CSV.
        output_dir: Output directory.
        drug_name: Optional drug name.
    """

    def __init__(
        self,
        nodes_file: str,
        edges_file: str,
        output_dir: str = "visualizations",
        drug_name: Optional[str] = None,
    ) -> None:
        self.nodes_df = pd.read_csv(nodes_file)
        self.edges_df = pd.read_csv(edges_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.drug_name = drug_name

        self.drug_smiles_map = load_drug_smiles() if drug_name else {}
        self.drug_node_svg = None
        if drug_name and drug_name in self.drug_smiles_map:
            self.drug_node_svg = render_drug_node_svg(drug_name, self.drug_smiles_map[drug_name])

        self.G = self._build_graph()
        self._compute_color_thresholds()

    def _build_graph(self) -> nx.DiGraph:
        """Build NetworkX directed graph from nodes and edges.

        Returns:
            Directed graph.
        """
        G = nx.DiGraph()

        for _, row in self.nodes_df.iterrows():
            G.add_node(
                row['node_id'],
                label=row['label'],
                node_type=row['node_type'],
                causal_importance=row['causal_importance'],
                top_genes=row.get('top_genes', ''),
            )

        for _, row in self.edges_df.iterrows():
            G.add_edge(
                row['source'],
                row['target'],
                edge_type=row['edge_type'],
                causal_importance=row['causal_importance'],
            )

        return G

    def _compute_color_thresholds(self) -> None:
        """Compute dynamic color thresholds based on pathway importance distribution."""
        pathway_nodes = self.nodes_df[self.nodes_df['node_type'] != 'drug']

        if len(pathway_nodes) == 0:
            self.sensitivity_threshold_deep = 0.03
            self.sensitivity_threshold_medium = 0.015
            self.resistance_threshold_deep = 0.03
            self.resistance_threshold_medium = 0.015
            return

        sensitivity_nodes = pathway_nodes[pathway_nodes['node_type'] == 'sensitivity']
        resistance_nodes = pathway_nodes[pathway_nodes['node_type'] == 'resistance']

        if len(sensitivity_nodes) > 0:
            sens_values = sensitivity_nodes['causal_importance'].abs().values
            self.sensitivity_threshold_deep = np.percentile(sens_values, 70) if len(sens_values) > 1 else sens_values[0]
            self.sensitivity_threshold_medium = np.percentile(sens_values, 40) if len(sens_values) > 1 else sens_values[0] * 0.5
        else:
            self.sensitivity_threshold_deep = 0.03
            self.sensitivity_threshold_medium = 0.015

        if len(resistance_nodes) > 0:
            res_values = resistance_nodes['causal_importance'].abs().values
            self.resistance_threshold_deep = np.percentile(res_values, 70) if len(res_values) > 1 else res_values[0]
            self.resistance_threshold_medium = np.percentile(res_values, 40) if len(res_values) > 1 else res_values[0] * 0.5
        else:
            self.resistance_threshold_deep = 0.03
            self.resistance_threshold_medium = 0.015

    def _get_node_color(self, node_type: str, importance: float) -> str:
        """Get node color based on type and importance.

        Args:
            node_type: Node type.
            importance: Causal importance score.

        Returns:
            Hex color code.
        """
        if node_type == 'drug':
            return PIGEBrandColors.DRUG_NODE_GOLD

        abs_imp = abs(importance)

        if node_type in ['sensitivity', 'gene_sensitivity']:
            if abs_imp >= self.sensitivity_threshold_deep:
                return PIGEBrandColors.DEEP_BLUE
            elif abs_imp >= self.sensitivity_threshold_medium:
                return PIGEBrandColors.MEDIUM_BLUE
            else:
                return PIGEBrandColors.LIGHT_BLUE
        elif node_type in ['resistance', 'gene_resistance']:
            if abs_imp >= self.resistance_threshold_deep:
                return PIGEBrandColors.DEEP_RED
            elif abs_imp >= self.resistance_threshold_medium:
                return PIGEBrandColors.MEDIUM_RED
            else:
                return PIGEBrandColors.LIGHT_RED

    def _get_node_size(self, node_type: str, importance: float) -> float:
        """Get node size based on importance.

        Args:
            node_type: Node type.
            importance: Causal importance.

        Returns:
            Node size.
        """
        if node_type == 'drug':
            return 4500

        if node_type in ['gene_sensitivity', 'gene_resistance']:
            return 1200

        abs_imp = abs(importance)
        base_size = 1500
        scaled_size = base_size + (abs_imp * 20000)
        return min(scaled_size, 2500)

    def _get_edge_color(self, edge_type: str) -> str:
        """Get edge color based on type.

        Args:
            edge_type: Edge type.

        Returns:
            Hex color code.
        """
        if edge_type == 'drug_pathway':
            return PIGEBrandColors.DRUG_EDGE
        elif edge_type == 'sensitivity':
            return PIGEBrandColors.MEDIUM_BLUE
        elif edge_type == 'resistance':
            return PIGEBrandColors.MEDIUM_RED
        else:
            return PIGEBrandColors.DRUG_EDGE

    def _get_edge_width(
        self,
        importance: float,
        source_node_type: Optional[str] = None,
        target_node_type: Optional[str] = None
    ) -> float:
        """Get edge width based on importance.

        Args:
            importance: Edge importance value.
            source_node_type: Source node type.
            target_node_type: Target node type.

        Returns:
            Edge width.
        """
        if source_node_type in ['gene_sensitivity', 'gene_resistance'] or target_node_type in ['gene_sensitivity', 'gene_resistance']:
            return 1.0

        abs_imp = abs(importance)
        min_width, max_width = 1.0, 4.0
        normalized = min(abs_imp / 0.04, 1.0)
        return min_width + (normalized * (max_width - min_width))

    def _get_text_color(self, bg_color: str) -> str:
        """Determine text color based on background.

        Args:
            bg_color: Background hex color.

        Returns:
            Text color (white or dark).
        """
        rgb = tuple(int(bg_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
        return PIGEBrandColors.PIGE_WHITE if luminance < 0.5 else PIGEBrandColors.PIGE_NAVY

    def _truncate_pathway_label(self, label: str, max_len: int) -> str:
        """Truncate pathway names for display.

        Args:
            label: Original label.
            max_len: Maximum length.

        Returns:
            Truncated label.
        """
        lowered = label.lower()
        if lowered.startswith("positive regulation of "):
            label = "pos. reg. of " + label[len("positive regulation of "):]
        elif lowered.startswith("negative regulation of "):
            label = "neg. reg. of " + label[len("negative regulation of "):]

        if len(label) <= max_len:
            return label
        return label[:max_len - 3] + "..."

    def create_interactive_visualization(
        self,
        output_filename: str,
        title: Optional[str] = None,
    ) -> None:
        """Create interactive HTML visualization.

        Args:
            output_filename: Output HTML file path.
            title: Optional title.
        """
        from pyvis.network import Network

        pathway_mode_data = self._prepare_mode_data('pathway')
        gene_mode_data = self._prepare_mode_data('gene')

        net = self._create_network_from_data(pathway_mode_data)

        output_path = self.output_dir / output_filename
        net.save_graph(str(output_path))

        self._add_gene_mode_to_html(output_path, title, pathway_mode_data, gene_mode_data)

        print(f"Saved {output_path}")

    def _prepare_mode_data(self, mode: str) -> Dict:
        """Prepare nodes and edges for a specific mode.

        Args:
            mode: 'pathway' or 'gene'.

        Returns:
            Dictionary with filtered nodes and edges.
        """
        mode_column = 'in_pathway_mode' if mode == 'pathway' else 'in_gene_mode'

        if mode == 'pathway':
            nodes = self.nodes_df[~self.nodes_df['node_type'].isin(['gene_sensitivity', 'gene_resistance'])]
        else:
            nodes = self.nodes_df

        edges = self.edges_df[self.edges_df[mode_column] == True]

        return {'nodes': nodes, 'edges': edges}

    def _create_network_from_data(self, data: Dict) -> Network:
        """Create a pyvis Network from filtered data.

        Args:
            data: Dictionary with 'nodes' and 'edges' DataFrames.

        Returns:
            Configured Network object.
        """

        net = Network(
            height='calc(100vh - 140px)',
            width='100%',
            bgcolor=PIGEBrandColors.OFF_WHITE,
            font_color=PIGEBrandColors.PIGE_NAVY,
            directed=True,
            heading='',
        )

        for _, row in data['nodes'].iterrows():
            node_id = row['node_id']
            node_type = row['node_type']
            importance = row['causal_importance']
            label = row['label']
            genes = row.get('top_genes', '')

            if node_type == 'drug':
                title_text = f"{label}\nAAC: {importance:.4f}\nThe predicted response of the cell line to the drug"
            elif node_type in ['gene_sensitivity', 'gene_resistance']:
                node_type_name = 'Sensitivity Gene' if node_type == 'gene_sensitivity' else 'Resistance Gene'
                title_text = f"{label}\nType: {node_type_name}\nImportance: {importance:.5f}\n{genes}\n"
                if node_type == 'gene_sensitivity':
                    title_text += "Knocking out this gene increases drug sensitivity"
                else:
                    title_text += "Knocking out this gene decreases drug sensitivity (resistance)"
            else:
                node_type_name = 'Sensitivity' if node_type == 'sensitivity' else 'Resistance'
                title_text = f"{label}\nType: {node_type_name}\nImportance: {importance:.5f}\n"
                if genes and str(genes).lower() not in ['nan', 'none', '']:
                    title_text += f"Top Genes: {genes}\n"
                else:
                    title_text += "Top Genes: N/A\n"
                if node_type == 'sensitivity':
                    title_text += "Knocking out this pathway increases drug sensitivity"
                else:
                    title_text += "Knocking out this pathway decreases drug sensitivity (resistance)"

            color = self._get_node_color(node_type, importance)
            size = self._get_node_size(node_type, importance) / 100
            text_color = self._get_text_color(color)

            if node_type == 'drug' and self.drug_node_svg:
                display_name = label if len(label) <= 16 else label[:13] + "..."
                net.add_node(
                    node_id,
                    label=display_name,
                    title=title_text,
                    image=self.drug_node_svg,
                    shape='image',
                    size=50,
                    shapeProperties={'useBorderWithImage': True},
                    font={'color': PIGEBrandColors.PIGE_NAVY, 'size': 16, 'face': 'Arial, sans-serif', 'bold': True, 'vadjust': -30},
                )
            else:
                shape = 'box' if node_type == 'drug' else 'ellipse'
                net.add_node(
                    node_id,
                    label=self._truncate_pathway_label(label, 50),
                    title=title_text,
                    color=color,
                    size=size,
                    shape=shape,
                    font={'color': text_color, 'size': 14 if node_type == 'drug' else 12},
                )

        for _, row in data['edges'].iterrows():
            u = row['source']
            v = row['target']
            edge_type = row['edge_type']
            importance = row['causal_importance']

            source_node = data['nodes'][data['nodes']['node_id'] == u]
            target_node = data['nodes'][data['nodes']['node_id'] == v]
            source_node_type = source_node['node_type'].values[0] if len(source_node) > 0 else None
            target_node_type = target_node['node_type'].values[0] if len(target_node) > 0 else None

            color = self._get_edge_color(edge_type)
            width = self._get_edge_width(importance, source_node_type, target_node_type)

            title_text = f"{u} → {v}\nImportance: {importance:.5e}\n"
            if edge_type == 'drug_pathway':
                title_text += "Drug directly affects this pathway"
            elif edge_type == 'sensitivity':
                title_text += f"Sensitivity signal: {u} propagates pro-sensitive information to {v}"
            else:
                title_text += f"Resistance signal: {u} propagates pro-resistance information to {v}"

            smooth = {'type': 'curvedCW', 'roundness': 0.15} if edge_type == 'drug_pathway' else {'type': 'continuous'}

            net.add_edge(u, v, color=color, width=width, title=title_text, arrows='to', smooth=smooth)

        net.barnes_hut(
            gravity=-80000,
            central_gravity=0.01,
            spring_length=100,
            spring_strength=0.005,
            damping=0.4,
        )

        return net

    def _build_node_configs(self, nodes_df: pd.DataFrame) -> List[Dict]:
        """Build complete node configurations for JavaScript.

        Args:
            nodes_df: DataFrame of nodes.

        Returns:
            List of node configuration dicts.
        """
        configs = []

        for _, row in nodes_df.iterrows():
            node_id = row['node_id']
            node_type = row['node_type']
            importance = row['causal_importance']
            label = row['label']
            genes = row.get('top_genes', '')

            if node_type == 'drug':
                title = f"{label}\nAAC: {importance:.4f}\nThe predicted response of the cell line to the drug"
            elif node_type in ['gene_sensitivity', 'gene_resistance']:
                type_name = 'Sensitivity Gene' if node_type == 'gene_sensitivity' else 'Resistance Gene'
                title = f"{label}\nType: {type_name}\nImportance: {importance:.5f}\n{genes}\n"
                if node_type == 'gene_sensitivity':
                    title += "Knocking out this gene increases drug sensitivity"
                else:
                    title += "Knocking out this gene decreases drug sensitivity (resistance)"
            else:
                type_name = 'Sensitivity' if node_type == 'sensitivity' else 'Resistance'
                title = f"{label}\nType: {type_name}\nImportance: {importance:.5f}\n"
                if genes and str(genes).lower() not in ['nan', 'none', '']:
                    title += f"Top Genes: {genes}\n"
                else:
                    title += "Top Genes: N/A\n"
                if node_type == 'sensitivity':
                    title += "Knocking out this pathway increases drug sensitivity"
                else:
                    title += "Knocking out this pathway decreases drug sensitivity (resistance)"

            color = self._get_node_color(node_type, importance)
            size = self._get_node_size(node_type, importance) / 100
            text_color = self._get_text_color(color)

            if node_type == 'drug':
                if self.drug_node_svg:
                    display_name = label if len(label) <= 16 else label[:13] + "..."
                    config = {
                        'id': node_id,
                        'label': display_name,
                        'title': title,
                        'image': self.drug_node_svg,
                        'shape': 'image',
                        'size': 50,
                        'shapeProperties': {'useBorderWithImage': True},
                        'font': {'color': PIGEBrandColors.PIGE_NAVY, 'size': 16, 'face': 'Arial, sans-serif', 'bold': True, 'vadjust': -30}
                    }
                else:
                    config = {
                        'id': node_id,
                        'label': self._truncate_pathway_label(label, 50),
                        'title': title,
                        'color': color,
                        'size': size,
                        'shape': 'box',
                        'font': {'color': text_color, 'size': 14}
                    }
            else:
                shape = 'ellipse'
                config = {
                    'id': node_id,
                    'label': self._truncate_pathway_label(label, 50),
                    'title': title,
                    'color': color,
                    'size': size,
                    'shape': shape,
                    'font': {'color': text_color, 'size': 12}
                }

            configs.append(config)

        return configs

    def _build_edge_configs(self, edges_df: pd.DataFrame, nodes_df: pd.DataFrame) -> List[Dict]:
        """Build complete edge configurations for JavaScript.

        Args:
            edges_df: DataFrame of edges.
            nodes_df: DataFrame of nodes.

        Returns:
            List of edge configuration dicts.
        """
        configs = []

        for _, row in edges_df.iterrows():
            u = row['source']
            v = row['target']
            edge_type = row['edge_type']
            importance = row['causal_importance']

            source_node = nodes_df[nodes_df['node_id'] == u]
            target_node = nodes_df[nodes_df['node_id'] == v]
            source_node_type = source_node['node_type'].values[0] if len(source_node) > 0 else None
            target_node_type = target_node['node_type'].values[0] if len(target_node) > 0 else None

            color = self._get_edge_color(edge_type)
            width = self._get_edge_width(importance, source_node_type, target_node_type)

            title = f"{u} → {v}\nImportance: {importance:.5e}\n"
            if edge_type == 'drug_pathway':
                title += "Drug directly affects this pathway"
            elif edge_type == 'sensitivity':
                title += f"Sensitivity signal: {u} propagates pro-sensitive information to {v}"
            else:
                title += f"Resistance signal: {u} propagates pro-resistance information to {v}"

            smooth = {'type': 'curvedCW', 'roundness': 0.15} if edge_type == 'drug_pathway' else {'type': 'continuous'}

            config = {
                'from': u,
                'to': v,
                'color': color,
                'width': width,
                'title': title,
                'arrows': 'to',
                'smooth': smooth
            }

            configs.append(config)

        return configs

    def _add_gene_mode_to_html(
        self,
        output_path: Path,
        title: Optional[str],
        pathway_mode_data: Dict,
        gene_mode_data: Dict
    ) -> None:
        """Add gene mode toggle to HTML file.

        Args:
            output_path: Path to HTML file.
            title: Optional title text.
            pathway_mode_data: Data for pathway mode.
            gene_mode_data: Data for gene mode.
        """
        import json
        import re

        with open(output_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        pathway_nodes_config = self._build_node_configs(pathway_mode_data['nodes'])
        pathway_edges_config = self._build_edge_configs(pathway_mode_data['edges'], pathway_mode_data['nodes'])

        gene_nodes_config = self._build_node_configs(gene_mode_data['nodes'])
        gene_edges_config = self._build_edge_configs(gene_mode_data['edges'], gene_mode_data['nodes'])

        if title:
            header_html = f'''
<div style="background-color: {PIGEBrandColors.OFF_WHITE}; padding: clamp(12px, 2vw, 20px); border-bottom: 1px solid {PIGEBrandColors.GRID_GREY};">
    <style>
        @media (max-width: 600px) {{
            .header-container {{
                flex-direction: column !important;
                gap: 12px !important;
            }}
            .header-title {{
                text-align: center !important;
                padding: 0 !important;
            }}
            .header-button {{
                width: 100% !important;
                max-width: 200px !important;
                margin: 0 auto !important;
            }}
        }}
    </style>
    <div class="header-container" style="max-width: 1200px; margin: 0 auto; display: flex; justify-content: space-between; align-items: center; gap: clamp(12px, 2vw, 20px); flex-wrap: wrap;">
        <h2 class="header-title" style="color: {PIGEBrandColors.PIGE_NAVY}; font-family: Arial, sans-serif; margin: 0; flex: 1; min-width: 0; font-size: clamp(1.1rem, 3vw, 1.8rem); line-height: 1.3; padding-right: clamp(8px, 2vw, 12px);">{title}</h2>
        <button id="geneModeToggle" class="header-button" style="
            padding: clamp(4px, 0.8vw, 6px) clamp(16px, 3vw, 22px);
            font-size: clamp(14px, 2.5vw, 16px);
            font-weight: 500;
            background: {PIGEBrandColors.PIGE_WHITE};
            border: 1px solid #CCCCCC;
            border-radius: 4px;
            color: {PIGEBrandColors.PIGE_BLUE};
            cursor: pointer;
            white-space: nowrap;
            transition: all 0.2s;
            font-family: Arial, sans-serif;
            flex-shrink: 0;
        ">
            Show Genes
        </button>
    </div>
</div>
'''
        else:
            header_html = f'''
<div style="background-color: {PIGEBrandColors.OFF_WHITE}; padding: clamp(12px, 2vw, 20px); border-bottom: 1px solid {PIGEBrandColors.GRID_GREY};">
    <div style="max-width: 1200px; margin: 0 auto; display: flex; justify-content: flex-end;">
        <button id="geneModeToggle" style="
            padding: clamp(4px, 0.8vw, 6px) clamp(16px, 3vw, 22px);
            font-size: clamp(14px, 2.5vw, 16px);
            font-weight: 500;
            background: {PIGEBrandColors.PIGE_WHITE};
            border: 1px solid #CCCCCC;
            border-radius: 4px;
            color: {PIGEBrandColors.PIGE_BLUE};
            cursor: pointer;
            white-space: nowrap;
            transition: all 0.2s;
            font-family: Arial, sans-serif;
        ">
            Show Genes
        </button>
    </div>
</div>
'''

        mode_script = f'''
<script type="text/javascript">
    const PATHWAY_MODE_DATA = {{
        nodes: {json.dumps(pathway_nodes_config)},
        edges: {json.dumps(pathway_edges_config)}
    }};

    const GENE_MODE_DATA = {{
        nodes: {json.dumps(gene_nodes_config)},
        edges: {json.dumps(gene_edges_config)}
    }};

    let currentMode = 'pathway';

    window.addEventListener('load', function() {{
        setTimeout(function() {{
            const toggleBtn = document.getElementById('geneModeToggle');
            if (toggleBtn) {{
                toggleBtn.addEventListener('mouseenter', function() {{
                    if (currentMode === 'pathway') {{
                        this.style.background = '#F8F9FA';
                        this.style.borderColor = '{PIGEBrandColors.PIGE_BLUE}';
                    }} else {{
                        this.style.background = '#174a8f';
                    }}
                }});

                toggleBtn.addEventListener('mouseleave', function() {{
                    if (currentMode === 'pathway') {{
                        this.style.background = '{PIGEBrandColors.PIGE_WHITE}';
                        this.style.borderColor = '#CCCCCC';
                    }} else {{
                        this.style.background = '{PIGEBrandColors.PIGE_BLUE}';
                    }}
                }});

                toggleBtn.addEventListener('click', function() {{
                    if (currentMode === 'pathway') {{
                        currentMode = 'gene';
                        this.textContent = 'Hide Genes';
                        this.style.background = '{PIGEBrandColors.PIGE_BLUE}';
                        this.style.color = '{PIGEBrandColors.PIGE_WHITE}';
                        this.style.borderColor = '{PIGEBrandColors.PIGE_BLUE}';
                        rebuildNetwork(GENE_MODE_DATA);
                    }} else {{
                        currentMode = 'pathway';
                        this.textContent = 'Show Genes';
                        this.style.background = '{PIGEBrandColors.PIGE_WHITE}';
                        this.style.color = '{PIGEBrandColors.PIGE_BLUE}';
                        this.style.borderColor = '#CCCCCC';
                        rebuildNetwork(PATHWAY_MODE_DATA);
                    }}
                }});
            }}
        }}, 500);
    }});

    function rebuildNetwork(data) {{
        if (typeof network === 'undefined' || !network) {{
            return;
        }}

        const nodes = network.body.data.nodes;
        const edges = network.body.data.edges;

        nodes.clear();
        edges.clear();

        nodes.add(data.nodes);
        edges.add(data.edges);

        network.redraw();
        network.stabilize(100);

        setTimeout(function() {{
            network.fit({{
                animation: {{
                    duration: 300,
                    easingFunction: 'easeInOutQuad'
                }}
            }});
        }}, 200);
    }}
</script>
'''

        html_content = re.sub(r'("font"\s*:\s*\{[^}]*?)(\})(?=[^}]*"shape"\s*:\s*"image")', r'\1,"vadjust":-30\2', html_content, flags=re.DOTALL)
        html_content = re.sub(r'("shape"\s*:\s*"image"[^}]*?"font"\s*:\s*\{[^}]*?)(\})', r'\1,"vadjust":-30\2', html_content, flags=re.DOTALL)

        html_content = html_content.replace('<body>', f'<body>{header_html}', 1)
        html_content = html_content.replace('</body>', f'{mode_script}</body>', 1)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)


def create_visualizations_for_cell(
    cell_id: str,
    nodes_dir: str,
    edges_dir: str,
    output_dir: str,
    drug_name: str,
    cell_line_names: Optional[Dict[str, str]] = None,
) -> None:
    """Create HTML visualization for a single cell line.

    Args:
        cell_id: Cell line ID or 'AVERAGE'.
        nodes_dir: Directory containing nodes CSV files.
        edges_dir: Directory containing edges CSV files.
        output_dir: Output directory.
        drug_name: Drug name for title.
        cell_line_names: Optional mapping of ModelID to CellLineName.
    """
    cell_line_name = cell_line_names.get(cell_id, cell_id).replace(' ', '_').replace('/', '_').replace('\\', '_') if cell_line_names else cell_id
    nodes_file = Path(nodes_dir) / f"{cell_line_name}_nodes.csv"
    edges_file = Path(edges_dir) / f"{cell_line_name}_edges.csv"

    if not nodes_file.exists() or not edges_file.exists():
        return

    viz = PIGEGraphVisualizer(
        nodes_file=str(nodes_file),
        edges_file=str(edges_file),
        output_dir=output_dir,
        drug_name=drug_name,
    )

    cell_name = cell_line_names.get(cell_id, cell_id).replace(' ', '_').replace('/', '_').replace('\\', '_') if cell_line_names else cell_id

    if cell_id == 'AVERAGE':
        title = f"PIGE Graph: {drug_name} (Average across all cell lines)"
        output_filename = f"{drug_name}_AVERAGE_PIGE_graph.html"
    else:
        title = f"PIGE Graph: {drug_name} in {cell_name}"
        output_filename = f"{cell_name}_PIGE_graph.html"

    viz.create_interactive_visualization(output_filename=output_filename, title=title)


def create_all_visualizations(input_dir: str, output_dir: str, drug_name: str) -> None:
    """Batch process all cell line graphs.

    Args:
        input_dir: Directory containing CSV files.
        output_dir: Directory to save HTML visualizations.
        drug_name: Drug name for titles.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cell_line_names = load_cell_line_names()
    cell_line_to_id = {name.replace(' ', '_').replace('/', '_').replace('\\', '_'): id for id, name in cell_line_names.items()}

    node_files = list(input_path.glob("*_nodes.csv"))
    cell_line_names_from_files = [f.stem.replace('_nodes', '') for f in node_files]
    cell_ids = [cell_line_to_id.get(cell_line_name, cell_line_name) for cell_line_name in cell_line_names_from_files]

    for cell_id in sorted(cell_ids):
        create_visualizations_for_cell(
            cell_id=cell_id,
            nodes_dir=str(input_path),
            edges_dir=str(input_path),
            output_dir=str(output_path),
            drug_name=drug_name,
            cell_line_names=cell_line_names,
        )


def discover_available_drugs(base_dir: Path, dataset: str) -> List[str]:
    """Discover available drugs for a given dataset by scanning the debug_csv directory.

    Args:
        base_dir: Base directory containing the PIGE_graphs data
        dataset: Dataset name

    Returns:
        List of drug names that have graph data for this dataset
    """
    drugs = set()
    graphs_dir = base_dir / "PIGE_graphs" / dataset

    if not graphs_dir.exists():
        print(f"Warning: Directory not found: {graphs_dir}")
        return []

    # Look for drug directories that contain debug_csv subdirectories
    for drug_dir in graphs_dir.iterdir():
        if drug_dir.is_dir() and (drug_dir / "debug_csv").exists():
            drugs.add(drug_dir.name)

    return sorted(list(drugs))


def visualize_graphs(config: Dict) -> None:
    """Create visualizations for all drugs from configuration.

    Args:
        config: Configuration dictionary with keys:
            - base_dir: Base directory path
            - dataset: List of dataset names
            - drug_names: Optional list of drug names to process
    """
    base_dir = Path(config['base_dir'])
    datasets = config['dataset']
    requested_drug_names = config.get('drug_names')

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Visualizing graphs for dataset: {dataset}")
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
            print(f"No matching drugs to visualize for dataset {dataset}, skipping...")
            continue

        print(f"Visualizing {len(drug_names)} drugs for {dataset}: {', '.join(drug_names)}")

        for drug_name in drug_names:
            input_dir = base_dir / "PIGE_graphs" / dataset / drug_name / "debug_csv"
            output_dir = base_dir / "PIGE_graphs" / dataset / drug_name / "visualizations"

            create_all_visualizations(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                drug_name=drug_name,
            )

        print(f"\nFinished visualizing graphs for dataset: {dataset}")
