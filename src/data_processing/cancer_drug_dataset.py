"""Cancer drug dataset for precomputed features.

PyTorch Dataset for loading precomputed PAFE features, drug fingerprints,
and response labels from disk.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class DataConstants:
    """Constants for data processing pipeline."""

    CTRP_DRUG_RESPONSE_DIR = 'CTRPv2_drug_response_data'
    GDSC0_DRUG_RESPONSE_DIR = 'GDSC0_drug_response_data'
    GDSC2_DRUG_RESPONSE_DIR = 'GDSC2_drug_response_data'

    PRECOMPUTED_SUBDIR_CTRP = "CTRPv2"
    PRECOMPUTED_SUBDIR_GDSC0 = "GDSC0"
    PRECOMPUTED_SUBDIR_GDSC0_TRUE_TEST = "GDSC0_true_test"
    PRECOMPUTED_SUBDIR_GDSC2 = "GDSC2"
    PRECOMPUTED_SUBDIR_GDSC2_TRUE_TEST = "GDSC2_true_test"

    NPVAE_EMBEDDINGS_FILE = 'npvae_drug_embeddings.pkl'
    GENOMIC_LOOKUPS_SUBDIR = 'genomic_lookups'
    PRECOMPUTED_FEATURES_SUBDIR = 'precomputed_features'
    TRUE_TEST_SUBDIR = 'true_test'

    RESPONSE_COLUMN = 'aac'
    MODEL_ID_COLUMN = 'ModelID'

    N_FEATURE_TYPES = 3 # MUT, CNA, RNA

    TENSOR_DTYPE = torch.float32
    NUMPY_DTYPE = np.float32


class DatasetConstants:
    """Constants for cancer drug dataset."""

    PRECOMPUTED_FILE_PATTERN = "pafe_fp_{cell_id}_{drug_name}.pt"
    REQUIRED_DATA_KEYS = {'pafe_features', 'drug_fingerprint', 'label'}
    LABEL_EXPECTED_SHAPE = (1,)
    DEFAULT_DEVICE = 'cpu'


class CancerDrugDataset(Dataset):
    """PyTorch Dataset for cancer drug response data.

    Loads precomputed PAFE features, drug fingerprints, and labels
    from disk for each cell-drug combination.

    Args:
        samples: List of (cell_id, drug_name, label) tuples.
        precomputed_dir: Directory with precomputed data files.
        device: Device for tensor operations (default: cpu).
    """

    def __init__(
        self,
        samples: List[Tuple[str, str, Any]],
        precomputed_dir: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None
    ) -> None:
        """Initialize dataset.

        Args:
            samples: List of (cell_id, drug_name, label) tuples.
            precomputed_dir: Path to precomputed data directory.
            device: Device for tensors (defaults to cpu).
        """
        super().__init__()

        self.samples = samples
        self.precomputed_dir = Path(precomputed_dir)

        if device is None:
            self.device = torch.device(DatasetConstants.DEFAULT_DEVICE)
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        print(f"Initialized CancerDrugDataset: {len(self.samples)} samples")

    def __len__(self) -> int:
        """Return number of samples.

        Returns:
            Dataset size.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """Get sample by index.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with pafe_features, drug_fingerprint, label, cell_id, drug_name.
        """
        cell_id, drug_name, _ = self.samples[idx] # _ is the label from samples

        filename = DatasetConstants.PRECOMPUTED_FILE_PATTERN.format(
            cell_id=cell_id,
            drug_name=drug_name
        )
        filepath = self.precomputed_dir / drug_name / filename

        loaded_data = torch.load(filepath, map_location=self.device)

        label_tensor = loaded_data['label'].float()
        if label_tensor.ndim == 0:
            # Scalar tensor -> [1] shape
            label_tensor = label_tensor.unsqueeze(0)
        elif label_tensor.shape != DatasetConstants.LABEL_EXPECTED_SHAPE:
            # Non-standard shape -> reshape to [1]
            label_tensor = label_tensor.view(1)

        return {
            'pafe_features': loaded_data['pafe_features'].float().to_sparse(),
            'drug_fingerprint': loaded_data['drug_fingerprint'].float().to(self.device),
            'label': label_tensor.to(self.device),
            'cell_id': cell_id,
            'drug_name': drug_name
        }

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset summary information.

        Returns:
            Dictionary with dataset statistics.
        """
        unique_cells = set(sample[0] for sample in self.samples)
        unique_drugs = set(sample[1] for sample in self.samples)

        return {
            'total_samples': len(self.samples),
            'unique_cell_lines': len(unique_cells),
            'unique_drugs': len(unique_drugs),
            'precomputed_dir': str(self.precomputed_dir),
            'device': str(self.device),
            'cell_line_list': sorted(unique_cells),
            'drug_list': sorted(unique_drugs)
        }
