import sys
import yaml
from pathlib import Path

# Absolute Pathing & Module Setup
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.pipeline_orchestrator import PipelineOrchestrator
from src.main import resolve_config_variables

#Parse incoming comma-separated string from Bash into Python list
raw_input = sys.argv[1]
drugs_list = [d.strip() for d in raw_input.split(',')]

config_path = PROJECT_ROOT / "notebooks/configs/config_train.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

def fix_paths(d):
    for k, v in d.items():
        if isinstance(v, dict):
            fix_paths(v)
        elif isinstance(v, str) and v.startswith('../'):
            d[k] = str(PROJECT_ROOT / v[3:]) # Strips the '../' and glues it to your root

fix_paths(config)

# Feed the fully dynamic list straight to the core config
config['drugs']['main_drugs'] = drugs_list

#DYNAMIC FOLDER NAMING LOGIC
if len(drugs_list) == 1:
    folder_name = drugs_list[0]
else:
    folder_name = "Multiple_Drugs_Batch"

# Overwrite the cosmetic 'Erlotinib' string in the YAML config
config['experiment']['name'] = folder_name

config = resolve_config_variables(config)

#EXPERIMENTAL PIPELINE METADATA PRINTING (Saved to Logs)
print("="*60)
print("EXPERIMENTAL PIPELINE METADATA")
print("="*60)
print(f"TRAINING MULTI-DRUG MODEL FOR: {len(drugs_list)} drug(s)")
print(f"Drugs in this batch: {config['drugs']['main_drugs']}")

#Safely extract nested parameters using .get() to prevent KeyErrors
training_cfg = config.get('training', {})
runtime_cfg = config.get('runtime', {})
paths_cfg = config.get('paths', {})

print(f"Number of epochs: {training_cfg.get('num_epochs_per_drug', 'Default/Early Stopping')}")
print(f"Device: {runtime_cfg.get('device', 'cuda (auto)')}")
print(f"Output directory: {paths_cfg.get('output_data_dir', 'data/output_data')}")
print("="*60 + "\n")

print(f"--- Running Training for: {config['drugs']['main_drugs']} ---")

#Execute Pipeline
orchestrator = PipelineOrchestrator(config)
orchestrator.run_pipeline()