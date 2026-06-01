import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.pige_atlas.new_graphsv2 import run_graph_generation
from src.pige_atlas.visualize_graphs import visualize_graphs

raw_input = sys.argv[1]
drugs_list = [d.strip() for d in raw_input.split(',')]

config = {
    'base_dir': str(PROJECT_ROOT / 'data/output_data/quickstart/interpretability'),
    'dataset': ['gdsc0_true_test'],
    'drug_names': drugs_list, # Tracks all specified batch drugs dynamically
}

print(f"--- Generating Graphs for Multi-Drug Batch ---")
run_graph_generation(config)
visualize_graphs(config)