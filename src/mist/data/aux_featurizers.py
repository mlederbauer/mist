"""aux_featurizers.py

Auxiliary molecular conditioning sources for fingerprint prediction: given a
variable-count set of known SMILES associated with a spectrum (e.g. reaction
starting materials, or other confirmed related structures), produce a single
fixed-width vector to condition the model on. Registry-shaped so new source
types can be added without touching the model, dataset key structure, or CLI.
"""
from typing import List

import numpy as np

from mist.data import data
from mist.data.featurizers import FingerprintFeaturizer


class RelatedStructureFeaturizer:
    """Featurize a variable-count set of related-structure SMILES via mean
    Morgan fingerprint pooling."""

    def __init__(self, fp_names: List[str] = ["morgan2048"], **kwargs):
        self._fp_featurizer = FingerprintFeaturizer(fp_names=fp_names)
        self._dim = FingerprintFeaturizer.get_fingerprint_size(fp_names=fp_names)

    @property
    def dim(self) -> int:
        return self._dim

    def featurize(self, smiles_list: List[str]) -> np.ndarray:
        """Mean-pool the fingerprints of every parseable SMILES in
        smiles_list. Unparseable SMILES are dropped (matches the filtering
        get_paired_spectra already does for the primary molecule). An empty
        or all-unparseable list returns a zero vector."""
        fps = []
        for smiles in smiles_list:
            mol = data.Mol.MolFromSmiles(smiles)
            if mol is None:
                continue
            fps.append(self._fp_featurizer._featurize(mol))

        if not fps:
            return np.zeros(self._dim, dtype=np.float32)
        return np.mean(fps, axis=0).astype(np.float32)


AUX_REGISTRY = {"related_structures": RelatedStructureFeaturizer}
