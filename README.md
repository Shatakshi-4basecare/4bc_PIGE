# PIGE: Pathway-Informed Graph Explanation Framework for Mechanistic Interpretability


[![PIGE Atlas](https://img.shields.io/badge/Interactive-PIGE_Atlas-blue?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiIGZpbGw9IiMwMDdhZmYiLz48L3N2Zz4=)](https://www.pigeatlas.com)

## Overview

**PIGE** (Pathway-Informed Graph Explanation) is a deep learning framework that predicts cancer drug response by understanding the underlying biological mechanisms. PIGE allows you to answer the following questions:
- *Why* is this cell line sensitive or resistant to this drug?
- *Which pathways/genes/interactions* drive the drug response or lack thereof?
- *What happens* to drug response if we block a specific pathway/gene/interaction?

### Our Approach

PIGE integrates multi-omics data (mutations, copy number alterations, gene expression) with drug chemical structures through a **biologically informed pathway interaction network**. The model architecture processes genomic features through this pathway graph, creating interpretable pathway-level representations. By systematically knocking out pathways and genes and measuring changes in model predictions, PIGE generates importance scores that identify influential biological mechanisms.

## Key Results

- **Superior Prediction Performance and Generalizability**
  - Spearman ρ=0.84 in 5-fold cross-validation on CTRPv2 training data
  - Spearman ρ=0.56 on external GDSC2 validation (excluding all overlapping cell line-drug pairs)
  - Outperforms DrugCell (ρ=0.52), DRPreter (ρ=0.51), and other state-of-the-art methods on external validation

- **Mechanistic Interpretability**
  - 74% pathway hit rate at K=25 vs 50% for CRISPR differential essentiality, 60% for GSEA, and 29% for DrugCell
  - Over 60% gene hit rate at K=100 (representing 4.8% of all genes) vs 41% for CRISPR screens
  - Recovers specific mechanisms of action including non-obvious targets (e.g. CX3CL1 as a resistance hub to chemotherapy in triple-negative breast cancer, only discovered experimentally in 2017)

- **Clinical Translation**
  - Validated on BeatAML ex vivo patient samples
  - Generalizes to unseen drugs and drug classes (e.g. MDM2 inhibitors not present in training data)
  - Identifies patient-specific resistance mechanisms that suggest rational combination therapies

## Architecture

PIGE is based on a **biologically informed pathway interaction network** constructed from Gene Ontology (GO) biological processes and OmniPath protein-protein interactions. The architecture consists of three integrated components:

![PIGE Architecture](docs/pige_architecture.png)

This design **mirrors biology**: each pathway's influence on a cell's response is determined by its interactions with upstream pathways, the cell's genomics, and the drug's properties.

---

## Project Structure

The project consists of the following folders:

- **`data/`** - Contains all data on which models were built
  - `input_data/` - Raw input data (omics, drug response, pathway data)
  - `intermediate_data/` - Processed intermediate files (pathway graphs, precomputed features)
  - `output_data/` - Model outputs and trained model checkpoints
- **`src/`** - Source code for PIGE
  - `data_processing/` - Data preprocessing and feature generation
  - `models/` - Model architectures (Structural Causal Model, GAT layers)
  - `pathway_interaction_setup/` - Pathway graph construction from GO and OmniPath
  - `model_interpretability/` - Virtual knockout analysis and interpretability tools
  - `pige_atlas/` - Interactive graph visualization and atlas generation
  - `train/` - Training scripts and experiment management
  - `train_test/` - External validation and evaluation scripts
- **`notebooks/`** - Jupyter notebook tutorials and examples
  - `01_train_erlotinib.ipynb` - Train PIGE on a single drug
  - `02_knockout_analysis.ipynb` - Virtual knockout analysis
  - `03_generate_pige_graphs.ipynb` - Generate interactive pathway graphs

## Quick Start

### Virtual Environment Setup

We recommend using a virtual environment to manage dependencies:

```bash
# Create virtual environment (using conda)
conda create -n pige python=3.10
conda activate pige

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Installation

```bash
# Install PyTorch (adjust CUDA version as needed)
# For CUDA 11.0+
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 -f https://download.pytorch.org/whl/torch_stable.html

# Install PyTorch Geometric and dependencies
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
pip install torch-geometric

# Install other dependencies
pip install -r requirements.txt
```

### Basic Usage

Start with the tutorial notebooks in `notebooks/`:

1. **Train a model**: `notebooks/01_train_erlotinib.ipynb` (~15 minutes)
2. **Run interpretability**: `notebooks/02_knockout_analysis.ipynb` (~5 minutes)
3. **Generate graphs**: `notebooks/03_generate_pige_graphs.ipynb` (~5 minutes)

For programmatic usage, use the main pipeline:

```python
from src.pipeline.pipeline_orchestrator import PipelineOrchestrator
from src.main import resolve_config_variables
import yaml

# Load and resolve config
with open('src/configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config = resolve_config_variables(config)

# Run pipeline
orchestrator = PipelineOrchestrator(config)
orchestrator.run()
```

## Notebooks

Interactive tutorials demonstrating PIGE's capabilities:

| Tutorial | Description | Runtime | Link |
|----------|-------------|---------|------|
| **Train PIGE** | Train PIGE on a single drug (Erlotinib) using the full 5-stage pipeline | ~15 min | [`notebooks/01_train_erlotinib.ipynb`](notebooks/01_train_erlotinib.ipynb) |
| **Virtual Knockout Analysis** | Identify key pathways and genes driving drug sensitivity vs resistance | ~5 min | [`notebooks/02_knockout_analysis.ipynb`](notebooks/02_knockout_analysis.ipynb) |
| **Generate PIGE Graphs** | Create interactive visualizations of pathway crosstalk and mechanisms | ~5 min | [`notebooks/03_generate_pige_graphs.ipynb`](notebooks/03_generate_pige_graphs.ipynb) |

## Data

### Data Sources

PIGE uses the following publicly available datasets:

- **CTRPv2** - Cancer Therapeutics Response Portal v2 (drug response data)
- **GDSC** - Genomics of Drug Sensitivity in Cancer (external validation)
- **DepMap** - Dependency Map (omics data: mutations, CNAs, expression)
- **Gene Ontology** - Biological process annotations (pathway definitions)
- **OmniPath** - Protein-protein interaction network (pathway crosstalk)

## Citation

If you use PIGE in your research, please cite:

[Will be added when paper is published]

## Links

- **Interactive PIGE Atlas**: [https://www.pigeatlas.com](https://www.pigeatlas.com) - Explore 45,000+ pathway graphs across 70 drugs and 800+ cell lines
- Research Paper: [Will be added when paper is published]

## Requirements

### System Requirements

- **Python**: 3.10+
- **PyTorch**: 1.13+ (with CUDA support recommended)
- **CUDA**: 11.0+ (GPU recommended, 8GB+ VRAM)
  - Training and interpretability will work on CPU but will be significantly slower
- **Disk Space**: ~30GB for data and intermediate files (if training on all 68 drugs)
- **RAM**: 16GB+ needed for interpretability analysis

## Contact

For questions, issues, or collaboration inquiries:

- **Email**: cbahl076@uottawa.ca
- **Issues**: Please open an issue on GitHub for bug reports or feature requests
