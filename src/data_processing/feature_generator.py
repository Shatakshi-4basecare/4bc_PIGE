"""Feature generator for drug response prediction.

Generates pathway-based (PAFE) matrices and drug fingerprints from genomic data.
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch

from .cancer_drug_dataset import DataConstants


class FeatureGenerator:
    """Generates PAFE matrices and drug fingerprints for cell-drug pairs."""

    def __init__(
        self,
        pathway_dict: Dict[str, Set[str]],
        ordered_pathway_names: List[str],
        g_universal: List[str],
        cell_mutation_map: Dict[str, Set[str]],
        cell_cna_map: Dict[Tuple[str, str], float],
        cell_rna_map: Dict[Tuple[str, str], float],
        smiles_map: Dict[str, str],
        npvae_embeddings_map: Dict[str, List[float]],
        selected_omics_type: str = "all",
    ) -> None:
        """Initialize the FeatureGenerator.

        Args:
            pathway_dict: Mapping from pathway names to gene sets.
            ordered_pathway_names: Ordered list of pathway names.
            g_universal: List of all genes used in features.
            cell_mutation_map: Mapping of cell lines to mutated genes.
            cell_cna_map: Mapping of (cell, gene) to CNA values.
            cell_rna_map: Mapping of (cell, gene) to RNA values.
            smiles_map: Mapping of drug names to SMILES strings.
            npvae_embeddings_map: Mapping of SMILES to embedding vectors.
            selected_omics_type: Which omics to use ('all', 'mutation', 'cna', 'rna').
        """
        self.pathway_dict = pathway_dict
        self.ordered_pathway_names = ordered_pathway_names
        self.g_universal = g_universal
        self.cell_mutation_map = cell_mutation_map
        self.cell_cna_map = cell_cna_map
        self.cell_rna_map = cell_rna_map
        self.smiles_map = smiles_map
        self.npvae_embeddings_map = npvae_embeddings_map

        self.gene_to_index_map = {gene: idx for idx, gene in enumerate(g_universal)}
        self.n_genes_in_features = len(g_universal)
        self.total_features_per_pathway = DataConstants.N_FEATURE_TYPES * self.n_genes_in_features
        self.n_pathways = len(ordered_pathway_names)
        self.selected_omics_type = self._normalize_selected_omics_type(selected_omics_type)
        self._selected_omics_block_idx = self._determine_block_index(self.selected_omics_type)

        print(f"FeatureGenerator initialized: {self.n_pathways} pathways, {self.n_genes_in_features} genes")
        if self.selected_omics_type != "all":
            print(f"Using only {self.selected_omics_type.upper()} omics features")

    def generate_features(
        self, cell_id: str, drug_name: str, return_full_vectors: bool = False
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[Dict[str, np.ndarray]]]:
        """Generate PAFE and fingerprint tensors for a cell-drug pair.

        Args:
            cell_id: Cell line identifier (ModelID).
            drug_name: Drug name.
            return_full_vectors: If True, also returns raw omic vectors.

        Returns:
            Tuple of (pafe_tensor, fp_tensor, omic_vectors).
            Returns (None, None, None) if generation fails.
        """
        try:
            if return_full_vectors:
                pafe_tensor, omic_vectors = self._create_pafe_features_from_cell_id(cell_id, return_full_vectors=True)
            else:
                pafe_tensor = self._create_pafe_features_from_cell_id(cell_id)
                omic_vectors = None

            fp_tensor = self._generate_drug_fingerprint(drug_name)

            if pafe_tensor is not None and not torch.any(pafe_tensor):
                print(f"Generated PAFE for {cell_id} is all zeros.")
                if return_full_vectors:
                    return None, None, None
                else:
                    return None, None

            if return_full_vectors:
                return pafe_tensor, fp_tensor, omic_vectors
            else:
                return pafe_tensor, fp_tensor

        except Exception as e:
            print(f"Failed to generate features for {cell_id}-{drug_name}: {e}")
            if return_full_vectors:
                return None, None, None
            else:
                return None, None

    def _get_omic_vectors(self, cell_id: str) -> Dict[str, np.ndarray]:
        """Get raw omic feature vectors for a cell line.

        Args:
            cell_id: Cell line identifier.

        Returns:
            Dictionary with 'mutation', 'cna', and 'rna' vectors.
        """
        mutation_vector = np.zeros(self.n_genes_in_features, dtype=DataConstants.NUMPY_DTYPE)
        cna_vector = np.zeros(self.n_genes_in_features, dtype=DataConstants.NUMPY_DTYPE)
        rna_vector = np.zeros(self.n_genes_in_features, dtype=DataConstants.NUMPY_DTYPE)

        mutated_genes = self.cell_mutation_map.get(cell_id, set())
        for gene in mutated_genes:
            if gene in self.gene_to_index_map:
                mutation_vector[self.gene_to_index_map[gene]] = 1.0

        for gene, idx in self.gene_to_index_map.items():
            cna_vector[idx] = self.cell_cna_map.get((cell_id, gene), 0.0)
            rna_vector[idx] = self.cell_rna_map.get((cell_id, gene), 0.0)

        return {"mutation": mutation_vector, "cna": cna_vector, "rna": rna_vector}

    def _create_pafe_features_from_vectors(
        self, mut_vector: np.ndarray, cna_vector: np.ndarray, rna_vector: np.ndarray
    ) -> torch.Tensor:
        """Generate PAFE tensor from omic vectors.

        Args:
            mut_vector: Mutation vector.
            cna_vector: CNA vector.
            rna_vector: RNA vector.

        Returns:
            PAFE tensor.
        """
        pafe_matrix_np = np.zeros(
            (self.n_pathways, self.total_features_per_pathway),
            dtype=DataConstants.NUMPY_DTYPE
        )

        for p_idx, pathway_name in enumerate(self.ordered_pathway_names):
            pathway_genes = self.pathway_dict.get(pathway_name, set())

            # Create a mask for genes in the current pathway
            pathway_gene_indices = [self.gene_to_index_map[g] for g in pathway_genes if g in self.gene_to_index_map]
            if not pathway_gene_indices:
                continue

            mask = np.zeros(self.n_genes_in_features, dtype=bool)
            mask[pathway_gene_indices] = True

            # Apply mask to get features for the current pathway
            offset_cna = self.n_genes_in_features
            offset_rna = 2 * self.n_genes_in_features

            pafe_matrix_np[p_idx, :offset_cna] = np.where(mask, mut_vector, 0)
            pafe_matrix_np[p_idx, offset_cna:offset_rna] = np.where(mask, cna_vector, 0)
            pafe_matrix_np[p_idx, offset_rna:] = np.where(mask, rna_vector, 0)

        return self._apply_omics_selection(torch.from_numpy(pafe_matrix_np))

    def _create_pafe_features_from_cell_id(self, cell_id: str, return_full_vectors: bool = False):
        """Generate PAFE matrix for a cell line.

        Args:
            cell_id: Cell line identifier.
            return_full_vectors: If True, also return raw omic vectors.

        Returns:
            PAFE tensor, or (PAFE tensor, omic vectors) if return_full_vectors=True.
        """
        omic_vectors = self._get_omic_vectors(cell_id)
        pafe_tensor = self._create_pafe_features_from_vectors(
            omic_vectors["mutation"], omic_vectors["cna"], omic_vectors["rna"]
        )

        if return_full_vectors:
            return pafe_tensor, omic_vectors
        return pafe_tensor

    def _generate_drug_fingerprint(self, drug_name: str) -> Optional[torch.Tensor]:
        """Generate drug fingerprint tensor from NpVae embeddings.

        Args:
            drug_name: Drug name.

        Returns:
            Drug fingerprint tensor.
        """
        if drug_name == 'ZERO_DRUG':
            embedding_vector = self.npvae_embeddings_map.get('ZERO_DRUG')
            if embedding_vector is None:
                raise ValueError("'ZERO_DRUG' embedding not found")
            return torch.tensor(embedding_vector, dtype=DataConstants.TENSOR_DTYPE)

        smiles_str = self.smiles_map.get(drug_name)
        if smiles_str is None:
            raise ValueError(f"SMILES not found for drug: {drug_name}")

        embedding_vector = self.npvae_embeddings_map.get(smiles_str)
        if embedding_vector is None:
            raise ValueError(f"NpVae embedding not found for SMILES: {smiles_str}")

        return torch.tensor(embedding_vector, dtype=DataConstants.TENSOR_DTYPE)

    @staticmethod
    def _determine_block_index(normalized_type: str) -> Optional[int]:
        """Return block index (0-based) for selected omics type.

        Args:
            normalized_type: Normalized omics type string.

        Returns:
            Block index (0-2) or None for 'all'.
        """
        if normalized_type == "all":
            return None
        block_map = {"mutation": 0, "cna": 1, "rna": 2}
        return block_map[normalized_type]

    def _apply_omics_selection(self, pafe_tensor: torch.Tensor) -> torch.Tensor:
        """Zero out unused omics feature blocks.

        Args:
            pafe_tensor: Full PAFE tensor.

        Returns:
            PAFE tensor with only selected omics features.
        """
        if self._selected_omics_block_idx is None:
            return pafe_tensor

        feature_dim = pafe_tensor.shape[-1]
        if feature_dim % DataConstants.N_FEATURE_TYPES != 0:
            return pafe_tensor

        block_width = feature_dim // DataConstants.N_FEATURE_TYPES
        start = self._selected_omics_block_idx * block_width
        end = start + block_width

        filtered = torch.zeros_like(pafe_tensor)
        filtered[..., start:end] = pafe_tensor[..., start:end]
        return filtered

    def create_pafe_df_for_cell_line(self, cell_id: str) -> Optional[pd.DataFrame]:
        """Create pandas DataFrame of PAFE features for a cell line.

        Args:
            cell_id: Cell line identifier.

        Returns:
            DataFrame with PAFE features, or None on failure.
        """
        try:
            pafe_tensor = self._create_pafe_features_from_cell_id(cell_id)

            mut_cols = [f"Mut_{gene}" for gene in self.g_universal]
            cna_cols = [f"CNA_{gene}" for gene in self.g_universal]
            rna_cols = [f"RNA_{gene}" for gene in self.g_universal]
            feature_columns = mut_cols + cna_cols + rna_cols

            pafe_df = pd.DataFrame(
                pafe_tensor.numpy(),
                index=self.ordered_pathway_names,
                columns=feature_columns
            )
            return pafe_df

        except Exception as e:
            print(f"Failed to create PAFE DataFrame for {cell_id}: {e}")
            return None

    @staticmethod
    def _normalize_selected_omics_type(selected_omics_type: Optional[str]) -> str:
        """Normalize user-provided omics selection.

        Args:
            selected_omics_type: User input string.

        Returns:
            Normalized omics type ('all', 'mutation', 'cna', or 'rna').
        """
        if not selected_omics_type:
            return "all"

        normalized = selected_omics_type.strip().lower()

        allowed = {"all", "mutation", "cna", "rna"}
        if normalized not in allowed:
            raise ValueError(
                f"Invalid selected_omics_type '{selected_omics_type}'. "
                "Valid options: all, mutation, cna, rna."
            )
        return normalized
