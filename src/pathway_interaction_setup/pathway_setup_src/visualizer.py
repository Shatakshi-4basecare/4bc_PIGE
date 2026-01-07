"""Functions for creating interactive visualizations of pathway graphs."""

import networkx as nx
import os
import webbrowser
from pyvis.network import Network


def create_interactive_visualization(graph, output_filename, notebook=False,
                                     graph_title='PIGE Pathway Interaction Graph'):
    """Create self-contained interactive HTML visualization using Pyvis.

    Args:
        graph: NetworkX DiGraph to visualize.
        output_filename: Path to save output HTML file.
        notebook: True if running in Jupyter Notebook environment.
        graph_title: Title to display on HTML page.
    """
    print(f'Creating interactive visualization for graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges')

    if graph.number_of_nodes() == 0:
        print('Input graph is empty. An empty HTML file will be generated')

    net = Network(
        notebook=notebook,
        height='950px',
        width='100%',
        heading=graph_title,
        directed=True,
        bgcolor='#222222',
        font_color='#FFFFFF'
    )

    for node_id, attrs in graph.nodes(data=True):
        label = attrs.get('name', str(node_id))
        title = f'ID: {node_id}\nName: {label}'
        color = attrs.get('color', '#007bff')
        size = attrs.get('viz_size', 15)

        net.add_node(
            node_id,
            label=label,
            title=title,
            color=color,
            size=size,
            shape='ellipse'
        )

    net.add_edges(graph.edges())

    print('Configuring physics layout and interaction options')

    net.show_buttons(filter_=['physics', 'nodes', 'edges', 'interaction'])

    net.force_atlas_2based(
        gravity=-50,
        central_gravity=0.01,
        spring_length=150,
        spring_strength=0.08,
        damping=0.4,
        overlap=0
    )

    net.save_graph(output_filename)
    print(f'Successfully saved interactive visualization to: {output_filename}')

    abs_path = os.path.abspath(output_filename)
    print(f'Opening {abs_path} in default web browser')
    webbrowser.open(f'file://{abs_path}', new=2)
