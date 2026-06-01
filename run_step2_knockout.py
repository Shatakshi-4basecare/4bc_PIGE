import sys
import json
import shutil
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.model_interpretability.model_pathway_and_gene_interpretability import load_config, main as run_knockout_analysis

# Parse incoming comma-separated string from Bash into a native Python list
raw_input = sys.argv[1]
drugs_list = [d.strip() for d in raw_input.split(',')]

config_path = PROJECT_ROOT / "notebooks/configs/config_knockout.yaml"
config = load_config(str(config_path))

def fix_paths(d):
    for k, v in d.items():
        if isinstance(v, dict):
            fix_paths(v)
        elif isinstance(v, str) and v.startswith('../'):
            d[k] = str(PROJECT_ROOT / v[3:])

fix_paths(config)

# ==========================================
# 1. READ OFFICIAL MODEL PATH
# ==========================================
json_path = PROJECT_ROOT / "data/output_data/quickstart/best_model_paths.json"
with open(json_path, 'r') as f:
    model_registry = json.load(f)

first_value = list(model_registry.values())[0]

# Defensive check: if it's a list, extract the string inside it; otherwise, keep it as is
latest_model_rel_path = first_value[0] if isinstance(first_value, list) else first_value

latest_model_full_path = PROJECT_ROOT / latest_model_rel_path

config['model_path'] = str(latest_model_full_path)
if 'paths' not in config:
    config['paths'] = {}
config['paths']['model_dir'] = str(latest_model_full_path)

print(f"Loaded official model: {latest_model_full_path}")

# ==========================================
# 2. VAULT BACKUP
# ==========================================
vault_dir = PROJECT_ROOT / "clinical_models"
vault_dir.mkdir(exist_ok=True)
shutil.copy(latest_model_full_path, vault_dir / "MultiDrug_master_model.pth")

# ==========================================
# 3. INTERPRETABILITY & METADATA LOGGING
# ==========================================
top_n = 20
config['top_n_cell_lines'] = top_n
config['drugs_to_process'] = drugs_list

print("\n" + "="*60)
print(f"STARTING KNOCKOUT ANALYSIS FOR MULTI-DRUG BATCH")
print(f"Drugs in batch: {drugs_list}")
print("="*60)

# Print top/bottom cell lines to the bash log file for EACH drug
for drug in drugs_list:
    response_file = PROJECT_ROOT / "data/intermediate_data/CTRPv2_drug_response_data" / f"{drug}.csv"
    if response_file.exists():
        df = pd.read_csv(response_file)
        df_sorted = df.sort_values('aac', ascending=False)
        top_sensitive = df_sorted.head(top_n)['ModelID'].tolist()
        top_resistant = df_sorted.tail(top_n)['ModelID'].tolist()
        print(f"\n[{drug}] Top {top_n} sensitive lines: {top_sensitive[:4]}...")
        print(f"[{drug}] Top {top_n} resistant lines: {top_resistant[:4]}...")
    else:
        print(f"\n[WARNING] Response CSV missing for {drug}. PIGE may skip this drug.")

print("\n" + "-"*60)

# ==========================================
# 4. EXECUTE VIRTUAL KNOCKOUT
# ==========================================
for target in ['pathway', 'gene', 'double_pathway', 'double_gene']:
    print(f"\nRunning {target} knockout for all drugs...")
    config['knockout_target'] = target
    
    # Passes the entire list to natively preserve the single all_drugs_summary file
    run_knockout_analysis(config, drugs_list)
    
    print(f"{target} analysis completed & native multi-drug summary generated!")

print("\nKNOCKOUT ANALYSIS COMPLETE.")