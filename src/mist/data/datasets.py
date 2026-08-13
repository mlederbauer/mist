""" datasets.py """
from pathlib import Path
import copy
import pickle
import h5py
import logging
from functools import partial
from typing import Optional, List, Tuple, Set, Callable, Union
import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from mist import utils
from mist.data import featurizers, aux_featurizers
from mist.data.data import Spectra, Mol
from mist.utils.hdf5_utils import Hdf5Store, is_hdf5_path


def get_paired_spectra(
    labels_file: str,
    spec_folder: str = None, 
    max_count: Optional[int] = None,
    allow_none_smiles: bool = False,
    prog_bars: bool = True,
    **kwargs,
) -> Tuple[List[Spectra], List[Mol]]:
    """_summary_

    Args:
        labels_file (str): _description_
        spec_folder (str): _description_
        max_count (Optional[int], optional): _description_. Defaults to None.
        allow_none_smiles (bool, optional): _description_. Defaults to False.
        prog_bars (bool, optional): _description_. Defaults to True.

    Returns:
        Tuple[List[Spectra], List[Mol]]: _description_
    """
    # First get labels
    compound_id_file = pd.read_csv(labels_file, sep="\t").astype(str)
    name_to_formula = dict(compound_id_file[["spec", "formula"]].values)

    name_to_smiles = {}
    if "smiles" in compound_id_file.keys():
        name_to_smiles = dict(compound_id_file[["spec", "smiles"]].values)

    name_to_inchikey = {}
    if "inchikey" in compound_id_file.keys():
        name_to_inchikey = dict(compound_id_file[["spec", "inchikey"]].values)

    name_to_instrument = {}
    if "instrument" in compound_id_file.keys():
        name_to_instrument = dict(compound_id_file[["spec", "instrument"]].values)

    # Note, loading has moved to the dataloader itself
    logging.info(f"Loading paired specs")

    spectra_hdf5 = None
    if spec_folder is not None and is_hdf5_path(spec_folder):
        # Packed format: a single .hdf5 keyed by "{spec}.ms" instead of a
        # directory of individual .ms files. Drive candidates from the
        # labels file (already in hand) and do single-key membership
        # checks, rather than listing every key in the store -- for
        # large hdf5 files, a full listing is a slow B-tree walk while a
        # per-key `in` check is a cheap hash lookup.
        spectra_hdf5 = Hdf5Store(spec_folder)
        candidate_names = list(name_to_formula) if max_count is None else list(name_to_formula)[:max_count]
        spectra_files = [
            Path(f"{name}.ms") for name in candidate_names if f"{name}.ms" in spectra_hdf5
        ]
    else:
        spec_folder = Path(spec_folder) if spec_folder is not None else None
        # Resolve for full path
        if spec_folder is not None and spec_folder.exists():
            spectra_files = [Path(i).resolve() for i in spec_folder.glob("*.ms")]
        else:
            logging.info(
                f"Unable to find spec folder {str(spec_folder)}, adding placeholders"
            )
            spectra_files = [Path(f"{i}.ms") for i in name_to_formula]

    if max_count is not None:
        spectra_files = spectra_files[:max_count]

    # Get file name from Path obj
    get_name = lambda x: x.name.split(".")[0]

    # Just added!
    spectra_files = [i for i in spectra_files if i.stem in name_to_formula]


    spectra_names = [get_name(spectra_file) for spectra_file in spectra_files]
    spectra_formulas = [name_to_formula[spectra_name] for spectra_name in spectra_names]
    spectra_instruments = [
        name_to_instrument.get(spectra_name, "") for spectra_name in spectra_names
    ]

    logging.info(f"Converting paired samples into Spectra objects")

    tq = tqdm if prog_bars else lambda x: x

    spectra_list = [
        Spectra(
            spectra_name=spectra_name,
            spectra_file=str(spectra_file),
            spectra_formula=spectra_formula,
            instrument=instrument,
            spectra_hdf5=spectra_hdf5,
            spectra_hdf5_key=spectra_file.name if spectra_hdf5 is not None else None,
            **kwargs,
        )
        for spectra_name, spectra_file, spectra_formula, instrument in tq(
            zip(
                spectra_names,
                spectra_files,
                spectra_formulas,
                spectra_instruments,
            )
        )
    ]

    # Create molecules
    spectra_smiles = [name_to_smiles.get(j, None) for j in spectra_names]
    spectra_inchikey = [name_to_inchikey.get(j, None) for j in spectra_names]
    if not allow_none_smiles:
        paired = [
            (spec, Mol.MolFromSmiles(smiles, inchikey=inchikey))
            for spec, smiles, inchikey in tq(
                zip(spectra_list, spectra_smiles, spectra_inchikey)
            )
            if smiles is not None
        ]
        # Drop entries whose smiles failed to parse (Mol.MolFromSmiles
        # returns None), rather than keeping a None misaligned into mol_list.
        num_dropped = sum(1 for _, mol in paired if mol is None)
        if num_dropped:
            logging.info(f"Dropping {num_dropped} spectra with unparseable smiles")
        paired = [(spec, mol) for spec, mol in paired if mol is not None]
        spectra_list = [spec for spec, _ in paired]
        mol_list = [mol for _, mol in paired]
    else:
        mol_list = [
            Mol.MolFromSmiles(smiles, inchikey=inchikey)
            if smiles is not None
            else Mol.MolFromSmiles("")
            for smiles, inchikey in tq(zip(spectra_smiles, spectra_inchikey))
        ]
        spectra_list = [spec for spec, smi in tq(zip(spectra_list, spectra_smiles))]

    logging.info("Done creating spectra objects")
    return (spectra_list, mol_list)


def attach_reactions(
    spectra_mol_pairs: List[Tuple[Spectra, Mol]],
    reaction_metadata_files: Union[str, List[str]],
    max_reactions_per_compound: Optional[int] = 10,
    seed: Optional[int] = None,
) -> List[Tuple[Spectra, Mol]]:
    """Join spectra_mol_pairs to one or more reaction_metadata_<source>.tsv
    files (concatenated) by inchikey and attach ALL matched reactions (up to
    max_reactions_per_compound) to each Spectra's aux_data (see
    mist.data.aux_featurizers, Spectra.get_aux_data/get_aux_records) --
    no row duplication. Which reaction (if any) is used for a given forward
    pass is decided later, at __getitem__ time: randomly per-epoch during
    training, or by explicit reaction_id for steered eval/inference.

    A compound with zero matched reactions passes through unchanged (no aux
    data attached), so nothing is dropped from training for lack of reaction
    data. A compound with N matched reactions keeps min(N,
    max_reactions_per_compound) of them, chosen deterministically (seeded) to
    bound the influence of a few promiscuous compounds (e.g. common
    salt-forming counter-ions matching thousands of incidental reactions)
    without dropping reaction diversity from every other compound. Pass
    max_reactions_per_compound=None for no cap.

    Leakage guard: candidates is filtered to exclude the pair's own compound
    by inchikey (not string/SMILES equality), regardless of what the source
    reaction_metadata_file row contains -- this makes leakage structurally
    impossible here rather than relying on upstream data being clean.
    """
    if isinstance(reaction_metadata_files, str):
        reaction_metadata_files = [reaction_metadata_files]
    reaction_df = pd.concat(
        [
            pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False)
            for f in reaction_metadata_files
        ],
        ignore_index=True,
    )

    reactions_by_inchikey = {}
    for inchikey, group in reaction_df.groupby("inchikey"):
        reactions_by_inchikey[inchikey] = group.to_dict("records")

    rng = np.random.default_rng(seed)
    n_attached = 0
    n_capped = 0
    for spec, mol in spectra_mol_pairs:
        inchikey = mol.get_inchikey()
        reaction_rows = reactions_by_inchikey.get(inchikey, [])
        if not reaction_rows:
            continue

        if max_reactions_per_compound is not None and len(
            reaction_rows
        ) > max_reactions_per_compound:
            keep_inds = rng.choice(
                len(reaction_rows), size=max_reactions_per_compound, replace=False
            )
            reaction_rows = [reaction_rows[i] for i in keep_inds]
            n_capped += 1

        starting_material_records = []
        candidate_records = []
        for reaction_row in reaction_rows:
            reaction_id = reaction_row.get("reaction_id")
            starting_materials = [
                s for s in reaction_row.get("starting_materials", "").split(";") if s
            ]
            candidate_smis = [
                s for s in reaction_row.get("candidates", "").split(";") if s
            ]
            # Leakage guard: never let the training target appear in its own
            # candidate set, regardless of what the source data contains.
            candidates = [
                s
                for s in candidate_smis
                if Mol.MolFromSmiles(s) is None
                or Mol.MolFromSmiles(s).get_inchikey() != inchikey
            ]
            starting_material_records.append(
                {"reaction_id": reaction_id, "smiles": starting_materials}
            )
            candidate_records.append({"reaction_id": reaction_id, "smiles": candidates})

        spec.aux_data = {
            "starting_materials": starting_material_records,
            "candidates": candidate_records,
        }
        n_attached += 1

    logging.info(
        f"attach_reactions: {n_attached}/{len(spectra_mol_pairs)} pairs matched "
        f"a reaction ({n_capped} capped to {max_reactions_per_compound})"
    )
    return spectra_mol_pairs


class SpectraMolDataset(Dataset):
    """SpectraMolDataset.

    Dataset to hold paired spectra and molecules

    """

    def __init__(
        self,
        spectra_mol_list: List[Tuple[Spectra, Mol]],
        featurizer: featurizers.PairedFeaturizer,
        **kwargs,
    ):
        """_summary_

        Args:
            spectra_mol_list (List[Tuple[Spectra, Mol]]): _description_
            featurizer (featurizers.PairedFeaturizer): _description_
        """
        super().__init__()
        spectra_list, mol_list = list(zip(*spectra_mol_list))
        self.spectra_list = np.array(spectra_list)
        self.mol_list = np.array(mol_list)
        self.smi_list = np.array([mol.get_smiles() for mol in self.mol_list])
        self.inchikey_list = np.array([mol.get_inchikey() for mol in self.mol_list])
        self.orig_len = len(self.mol_list)
        self.len = len(self.mol_list)

        # Extract all chem formulas
        self.chem_formulas = set()
        for spec in spectra_list:
            formula = spec.get_spectra_formula()
            self.chem_formulas.add(formula)

        # Verify same length
        # For examples where we have mismatches, we should use None
        assert len(self.spectra_list) == len(self.mol_list)

        # Store paired featurizer
        self.featurizer = featurizer
        self.train_mode = False

        # Save for subsetting
        self.kwargs = kwargs
        self.extract_kwargs(**self.kwargs)

        # List of specs and mols to use for aug
        self.aux_specs, self.aux_mols = None, None

    def extract_kwargs(
        self,
        forward_labels: str = None,
        fp_names: list = [],
        frac_orig: float = 0.4,
        forward_aug_folder=None,
        aux_dim: int = 0,
        aux_gate: bool = False,
        aux_dropout: float = 0.2,
        aux_sources: Optional[List[str]] = None,
        aux_reaction_id_by_spec: Optional[dict] = None,
        aux_seed: Optional[int] = None,
        aux_use_preset_data: bool = False,
        **kwargs,
    ):
        self.forward_labels = forward_labels
        self.fp_names = fp_names
        self.frac_orig = frac_orig
        self.forward_aug_folder = forward_aug_folder

        # Aux molecular conditioning (see mist.data.aux_featurizers). Off by
        # default (aux_dim=0, aux_gate=False) -- a strict no-op, no aux keys
        # added to batches, so this never affects anyone not using the
        # feature. Either aux-conditioning mechanism (--aux-dim concat or
        # --aux-gate blend) needs the same underlying aux_vec_*/aux_mask_*
        # batch keys, so both flags activate data loading the same way.
        # aux_sources (CLI --aux-sources) restricts which AUX_REGISTRY
        # entries are actually loaded/gated -- e.g. for an ablation that
        # isolates starting_materials from candidates. None (default) means
        # "all registered sources", same as before this parameter existed.
        self.aux_dim = aux_dim
        self.aux_gate = aux_gate
        self.aux_dropout = aux_dropout
        all_sources = list(aux_featurizers.AUX_REGISTRY)
        if aux_sources is not None:
            unknown = set(aux_sources) - set(all_sources)
            if unknown:
                raise ValueError(
                    f"Unknown aux source(s) {unknown} -- must be a subset of "
                    f"{all_sources}"
                )
        selected_sources = aux_sources if aux_sources is not None else all_sources
        self.aux_sources = selected_sources if (aux_dim > 0 or aux_gate) else []
        self.aux_featurizers = {
            source: aux_featurizers.AUX_REGISTRY[source](fp_names=fp_names)
            for source in self.aux_sources
        }
        # Compound/reaction steering at inference time: {spec_name:
        # reaction_id} to pin a specific matched reaction per spectrum
        # instead of the training-time random pick. Unset (None) means "no
        # steering" -- eval/inference then use no reaction (zero vector) for
        # every example, same as a compound with no matches, since there's no
        # principled way to auto-pick "the best" reaction for an example that
        # may match several. See mist.pred_fp --reaction-id-file.
        self.aux_reaction_id_by_spec = aux_reaction_id_by_spec or {}
        # For callers (e.g. analyze_reaction_sensitivity) that build one
        # dataset ROW per (compound, single chosen reaction) themselves --
        # each Spectra copy's aux_data already holds exactly the one record
        # that row should use, keyed by object identity rather than name (so
        # duplicated compounds with the same spec_name don't collide via
        # aux_reaction_id_by_spec). Skips both the random-pick (train) and
        # by-name-steering (eval) paths and just uses aux_data as-is.
        self.aux_use_preset_data = aux_use_preset_data
        self.aux_rng = np.random.default_rng(aux_seed)

    def upsample_forward(self):
        """add new forward entries"""

        logging.info(
            f"Resampling training, keeping ratio at {self.frac_orig} true examples"
        )

        if self.aux_specs is None or self.aux_mols is None:
            # data/paired_spectra/csi2022/magma_sub_trees/magma_tree_summary.tsv
            aug_labels = self.forward_labels
            df = pd.read_csv(aug_labels, sep="\t")

            # Construct molecules
            spectra_smiles = df["smiles"].values
            spectra_inchikey = df["inchikey"].values
            spectra_formulas = df["formula"].values
            spectra_names = df["spec"].values

            # All mol list
            logging.info("Creating auxilary mol objs")
            self.aux_mols = [
                Mol.MolFromSmiles(smiles, inchikey=inchikey)
                for smiles, inchikey in tqdm(zip(spectra_smiles, spectra_inchikey))
                if smiles is not None
            ]

            logging.info("Creating auxilary spec objs")
            # Create spectra list
            self.aux_specs = [
                Spectra(
                    spectra_name=spectra_name,
                    spectra_file="",
                    spectra_formula=spectra_formula,
                    **self.kwargs,
                )
                for spectra_name, spectra_smi, spectra_formula in tqdm(
                    zip(spectra_names, spectra_smiles, spectra_formulas)
                )
                if spectra_smi is not None
            ]

            self.aux_sample_weights = np.ones(len(self.aux_specs))
            self.aux_sample_weights = np.array(self.aux_sample_weights) / np.sum(
                self.aux_sample_weights
            )

            self.aux_specs = np.array(self.aux_specs)
            self.aux_mols = np.array(self.aux_mols)

        # Use replace to avoid dataloader problems
        aux_inds = np.random.choice(
            len(self.aux_specs),
            self.num_to_sample,
            p=self.aux_sample_weights,
            replace=True,
        )

        new_specs = self.aux_specs[aux_inds].tolist()
        new_mols = self.aux_mols[aux_inds].tolist()

        orig_specs = self.spectra_list[: self.orig_len].tolist()
        orig_mols = self.mol_list[: self.orig_len].tolist()

        self.mol_list = np.array(orig_mols + new_mols)
        self.spectra_list = np.array(orig_specs + new_specs)

        # Reset the self statistics
        self.smi_list = np.array([mol.get_smiles() for mol in self.mol_list])
        self.inchikey_list = np.array([mol.get_inchikey() for mol in self.mol_list])
        # Extract all chem formulas
        self.chem_formulas = set()
        for spec in self.spectra_list:
            formula = spec.get_spectra_formula()
            self.chem_formulas.add(formula)

    def set_train_mode(self, train_mode: bool):
        self.train_mode = train_mode

        # Only if in train mode
        if self.forward_labels is not None and self.train_mode:
            # Save these to the obj to avoid recalculation
            self.num_to_sample = int(
                ((1 - self.frac_orig) / self.frac_orig) * self.orig_len
            )
            self.len = self.num_to_sample + self.orig_len

    def get_spectra_list(self) -> List[Spectra]:
        """get_spectra_list."""
        return self.spectra_list

    def get_featurizer(self) -> featurizers.PairedFeaturizer:
        """get_spectra_list."""
        return self.featurizer

    def set_featurizer(self, featurizer: featurizers.PairedFeaturizer):
        """get_spectra_list."""
        self.featurizer = featurizer

    def get_spectra_names(self) -> List[str]:
        """get_spectra_names."""
        return [i.spectra_name for i in self.spectra_list]

    def get_mol_list(self) -> List[Mol]:
        """get_mol_list."""
        return self.mol_list

    def get_smi_list(self) -> List[str]:
        """Get the smiles list associated with data"""
        return self.smi_list

    def get_inchikey_list(self) -> List[str]:
        """Get the inchikey list associated with data"""
        return self.inchikey_list

    def get_all_formulas(self) -> Set[str]:
        """get_all_formulas.

        Return the set of all chemical formulas contained in this
        SpectraMolDataset

        """
        return self.chem_formulas

    def __len__(self) -> int:
        """__len__."""
        return self.len

    def __getitem__(self, idx: int) -> dict:
        """_summary_

        Args:
            idx (int): _description_

        Returns:
            dict: _description_
        """
        mol = self.mol_list[idx]
        spec = self.spectra_list[idx]

        mol_features = self.featurizer.featurize_mol(mol, train_mode=self.train_mode)
        spec_features = self.featurizer.featurize_spec(spec, train_mode=self.train_mode)

        out = {
            "spec": [spec_features],
            "mol": [mol_features],
            "spec_indices": [0],
            "mol_indices": [0],
            "matched": [True],
        }

        for source in self.aux_sources:
            featurizer = self.aux_featurizers[source]
            if self.aux_use_preset_data:
                # Caller already set this exact row's aux_data -- use it
                # as-is (a compound with 0 or 1 records here is the normal
                # case; get_aux_records handles both).
                records = spec.get_aux_records(source)
                items = records[0].get("smiles", []) if records else []
            elif self.train_mode:
                # Random pick per call -- a compound with N>1 matched
                # reactions sees a different one each epoch, turning
                # multiplicity into stochastic augmentation instead of
                # static duplicate rows.
                items = spec.get_aux_data(source, rng=self.aux_rng)
            else:
                # Eval/inference: no implicit random pick. Default is "no
                # reaction" (zero vector) unless the caller explicitly steers
                # this spectrum toward one via aux_reaction_id_by_spec.
                reaction_id = self.aux_reaction_id_by_spec.get(spec.get_spec_name())
                items = (
                    spec.get_aux_data(source, reaction_id=reaction_id)
                    if reaction_id is not None
                    else []
                )
            aux_present = len(items) > 0
            if aux_present and self.train_mode and np.random.random() < self.aux_dropout:
                aux_present = False
            aux_vec = (
                featurizer.featurize(items)
                if aux_present
                else np.zeros(featurizer.dim, dtype=np.float32)
            )
            out[f"aux_vec_{source}"] = [aux_vec]
            out[f"aux_mask_{source}"] = [len(items) > 0]

        return out


class SpectraMolMismatchHDFDataset(SpectraMolDataset):
    """SpectraMolMismatchDataset.

    Dataset to hold paired spectra and molecules
    This is enabled for training to mismatch datasets

    """

    def __init__(
        self,
        spectra_mol_list: List[Tuple[Spectra, Mol]],
        featurizer: featurizers.PairedFeaturizer,
        **kwargs,
    ):

        """__init__.

        Args:
            spectra_mol_list (List[Tuple[Spectra, Mol]]): List of spectra for the dataset
            mol_list (List[Mol]): List of molecules to store in the dataset
            featurizer (featurizer.PairedFeaturizer): Paired featurizer
            **kwargs

        """
        super().__init__(spectra_mol_list, featurizer, **kwargs)
        self.kwargs = kwargs
        self._extract_self_props(**self.kwargs)

    def _extract_self_props(
        self,
        hdf_file: str = "",
        contrastive_loss: str = "binary",
        num_decoys: int = 5,
        fp_names: list = ["morgan2048"],
        num_workers: int = 16,
        negative_strategy: str = "random",
        max_db_decoys: int = 512,
        decoy_norm_exp: float = None,
        forward_labels: str = None,
        frac_orig: float = 0.4,
        **kwargs,
    ):
        """Extract properties"""

        self.contrastive_loss = contrastive_loss
        self.num_decoys = num_decoys
        self.fp_names = fp_names if isinstance(fp_names, list) else [fp_names]
        self.fp_bits = self.featurizer.mol_featurizer.get_fingerprint_size(
            self.fp_names
        )
        self.negative_strategy = negative_strategy
        self.max_db_decoys = max_db_decoys
        self.decoy_norm_exp = decoy_norm_exp
        self.mol_to_weights = {}

        if self.forward_labels is not None:
            pass

        # Only load if we have a loss that requires it
        if self.contrastive_loss in ["softmax", "nce", "triplet", "triplet_rand"]:
            self.hdf_obj = h5py.File(hdf_file, "r")
            self.hdf_fps = self.hdf_obj["fingerprints"]
            self.base_ikeys = self.hdf_obj["base_ikeys"]
            self.dists = self.hdf_obj["dists"]
            self.ikey_lengths = self.hdf_obj["ikey_lengths"]
            self.ikey_offset = self.hdf_obj["ikey_offset"]
            self.ikey_to_ind = dict(
                zip(
                    [i.decode() for i in self.base_ikeys],
                    np.arange(len(self.base_ikeys)),
                )
            )
            self.num_bits = self.hdf_obj.attrs["num_bits"]

        self.num_workers = max(num_workers, 1)

    def set_train_mode(self, train_mode: bool):
        self.train_mode = train_mode

        # Only if in train mode
        if self.forward_labels is not None and self.train_mode:
            # Save these to the obj to avoid recalculation
            self.num_to_sample = int(
                ((1 - self.frac_orig) / self.frac_orig) * self.orig_len
            )
            self.len = self.num_to_sample + self.orig_len

    def _norm_weights(self, weights: np.ndarray):
        """_norm_weights.

        Given a vector of unnormalized weights, compute probabilities
        """
        # Num stability
        if self.decoy_norm_exp is None:
            weights += 1e-12
            weights = weights / weights.sum()
        else:
            # Scale + softmax
            exp_vals = np.exp(self.decoy_norm_exp * weights)
            weights = exp_vals / exp_vals.sum()

        return weights

    def _get_decoys(
        self, sample_num: int, exclude_idx: int, mol: Mol, mol_features: np.ndarray
    ):
        """Get decoys"""
        fps = self.hdf_fps
        if self.negative_strategy == "random":
            # Randomly draw fps from hdf
            start, end = 0, fps.shape[0]
            rand_inds = np.random.randint(start, end, sample_num)
            rand_inds.sort()
            rand_inds = np.unique(rand_inds)
            out_fps = fps[rand_inds]
            out_fps = utils.unpack_bits(out_fps, self.num_bits)

        elif self.negative_strategy == "hardisomer_tani_pickled":
            # Draw FPs from isomers weighted by tani similarity to targ
            # Check for forward upsample
            if exclude_idx >= self.orig_len and self.forward_labels is not None:

                # Randomly sample from all fps (avoid redoing sampling)
                # Deviates from original implementation that used hard sampling
                # for contrastive decoys as well. Changed for simplicity
                start, end = 0, fps.shape[0]
                rand_inds = np.random.randint(start, end, sample_num)
                rand_inds.sort()
                rand_inds = np.unique(rand_inds)
                out_fps = fps[rand_inds]
                out_fps = utils.unpack_bits(out_fps, self.num_bits)

            else:
                true_mol_ikey = mol.get_inchikey()
                lookup_ind = self.ikey_to_ind.get(true_mol_ikey)

                if lookup_ind is None:
                    return []

                length = self.ikey_lengths[lookup_ind]
                offset = self.ikey_offset[lookup_ind]
                length = min(length, self.max_db_decoys)
                start, end = offset, offset + length

                if end <= start:
                    out_fps = []
                else:
                    options = np.arange(start, end)
                    sample_num = min(len(options), sample_num)
                    option_weights = self.dists[options]
                    option_weights = self._norm_weights(option_weights)
                    rand_inds = np.random.choice(
                        options, sample_num, p=option_weights, replace=False
                    )
                    rand_inds.sort()
                    rnd_min, rnd_max = rand_inds.min(), rand_inds.max()
                    out_fps = fps[rnd_min : rnd_max + 1][rand_inds - rnd_min]
                    out_fps = utils.unpack_bits(out_fps, self.num_bits)
        else:
            raise NotImplementedError()

        return out_fps

    def _get_softmax_batch(
        self, idx: int, mol, spec, mol_features, spec_features
    ) -> dict:
        """_get_softmax_batch."""

        mol_feats = [mol_features]

        # Draw mismatched
        num_to_draw = self.num_decoys

        # Sample isomers
        mismatched_mols = self._get_decoys(
            exclude_idx=idx,
            sample_num=num_to_draw,
            mol=mol,
            mol_features=mol_features,
        )
        mol_feats.extend(mismatched_mols)
        return {
            "spec": [spec_features],
            "mol": mol_feats,
            "spec_indices": [0] * len(mol_feats),
            "mol_indices": np.arange(len(mol_feats)),
            "matched": [True] + [False] * len(mismatched_mols),
        }

    def _get_clip_batch(self, idx: int, mol, spec, mol_features, spec_features) -> dict:
        """_get_clip_batch."""
        return {
            "spec": [spec_features],
            "mol": [mol_features],
            "spec_indices": [0],
            "mol_indices": [0],
            "matched": [True],
        }

    def __getitem__(self, idx: int) -> dict:
        """__get_item__.

        Args:
            idx (int): Int index

        Return:
            dict
        """
        # If not train mode, don't bother with decoys
        # Remove this for now
        # if not self.train_mode:
        #    return super().__getitem__(idx)

        mol = self.mol_list[idx]
        spec = self.spectra_list[idx]

        mol_features = self.featurizer.featurize_mol(mol, train_mode=self.train_mode)
        spec_features = self.featurizer.featurize_spec(spec, train_mode=self.train_mode)

        # Contrastive types:
        if self.contrastive_loss in ["softmax", "nce", "triplet", "triplet_rand"]:
            batch = self._get_softmax_batch(idx, mol, spec, mol_features, spec_features)
        elif self.contrastive_loss == "clip":
            batch = self._get_clip_batch(idx, mol, spec, mol_features, spec_features)
        else:
            raise NotImplementedError()

        return batch


def _collate_pairs(
    input_batch: List[dict], mol_collate_fn: Callable, spec_collate_fn: Callable
) -> dict:
    """_collate_pairs

    Flexible function to collate pairs given a mol and spec collate fn

    Args:
        input_batch (List[dict]): List containing each entry from the dataset to be
            batched
        mol_collate_fn (Callable): Callable function that is applied to the
            molecules
        spec_collate_fn (Callable): Callable function that is applied to the
            spectra
    Return:
        dict: output containing batched entries
    """

    #### Spectra loading
    spec_dict = spec_collate_fn([j for jj in input_batch for j in jj["spec"]])

    #### Mol loading
    mol_dict = mol_collate_fn([j for jj in input_batch for j in jj["mol"]])

    # Get the paired molecule and paired spectra indices
    mol_indices = torch.tensor([j for jj in input_batch for j in jj["mol_indices"]])
    spec_indices = torch.tensor([j for jj in input_batch for j in jj["spec_indices"]])

    # Calculate number of unique molecules and spec in each entry
    len_mols = torch.tensor([len(j["mol"]) for j in input_batch])

    # Length of torch.sum(num_pairs) with max = len(spec_dict['spec'])
    len_specs = torch.tensor([len(j["spec"]) for j in input_batch])

    # Number of pairs
    num_pairs = torch.tensor([len(j["mol_indices"]) for j in input_batch])

    # Modify mol_indices pairs s.t. they are consistent with batch
    expanded_indices = torch.arange(len(len_mols)).repeat_interleave(num_pairs)
    addition_factor = torch.cumsum(len_mols, 0)
    addition_factor = torch.nn.functional.pad(addition_factor[:-1], (1, 0))

    mol_indices = mol_indices + addition_factor[expanded_indices]

    # Modify spec_indices pairs s.t. they are consistent with batch
    expanded_indices = torch.arange(len(len_specs)).repeat_interleave(num_pairs)
    addition_factor = torch.cumsum(len_specs, 0)
    addition_factor = torch.nn.functional.pad(addition_factor[:-1], (1, 0))

    # Length of torch.sum(num_pairs) with max = len(spec_dict['spec'])
    spec_indices = spec_indices + addition_factor[expanded_indices]

    # Matched loading
    matched = torch.tensor([j for jj in input_batch for j in jj["matched"]])

    # Create loading
    base_dict = {
        "matched": matched,
        "spec_indices": spec_indices,
        "mol_indices": mol_indices,
    }
    base_dict.update(spec_dict)
    base_dict.update(mol_dict)

    # Aux molecular conditioning (see mist.data.aux_featurizers), one vector
    # per dataset item per configured source -- absent entirely when aux
    # featurization is off, so this is a strict no-op for anyone not using
    # the feature. Each source's key is independent (aux_vec_<source>) since
    # sources are independently dropped out and projected.
    aux_vec_keys = [k for k in input_batch[0] if k.startswith("aux_vec_")]
    for vec_key in aux_vec_keys:
        source = vec_key[len("aux_vec_") :]
        mask_key = f"aux_mask_{source}"
        base_dict[vec_key] = torch.tensor(
            np.array([j for jj in input_batch for j in jj[vec_key]]),
            dtype=torch.float32,
        )
        base_dict[mask_key] = torch.tensor(
            [j for jj in input_batch for j in jj[mask_key]]
        )

    return base_dict


class SpecDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train=None,
        val=None,
        test=None,
        batch_size=32,
        num_workers=0,
        persistent_workers=False,
        forward_labels=None,
        debug: str = None,
        **kwargs,
    ):
        """_summary_

        Args:
            train (_type_, optional): _description_. Defaults to None.
            val (_type_, optional): _description_. Defaults to None.
            test (_type_, optional): _description_. Defaults to None.
            batch_size (int, optional): _description_. Defaults to 32.
            num_workers (int, optional): _description_. Defaults to 0.
            persistent_workers (bool, optional): _description_. Defaults to False.
            forward_labels (str, optional): _description_. Defaults to None.
            debug (str, optional): _description_. Defaults to None.
        """
        super().__init__()
        self.train = train
        self.val = val
        self.test = test

        self.debug = debug

        # Set in train mode
        self.train.set_train_mode(True)
        self.val.set_train_mode(False)
        if self.test is not None:
            self.test.set_train_mode(False)

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.forward_labels = forward_labels

        self.persistent_workers = False
        if num_workers > 0 and persistent_workers:
            self.persistent_workers = True

    @staticmethod
    def get_paired_loader(
        dataset: SpectraMolDataset,
        shuffle: bool = False,
        batch_size: int = 32,
        num_workers: int = 0,
        persistent_workers: bool = False,
        **kwargs,
    ):
        """_summary_

        Function to return the proper paired dataloader

        Args:
            dataset (SpectraMolDataset): _description_
            shuffle (bool, optional): _description_. Defaults to False.
            batch_size (int, optional): _description_. Defaults to 32.
            num_workers (int, optional): _description_. Defaults to 0.
            persistent_workers (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        mol_collate_fn = dataset.get_featurizer().get_mol_collate()
        spec_collate_fn = dataset.get_featurizer().get_spec_collate()
        collate_pairs = partial(
            _collate_pairs,
            mol_collate_fn=mol_collate_fn,
            spec_collate_fn=spec_collate_fn,
        )

        _persistent_workers = False
        if num_workers > 0 and persistent_workers:
            _persistent_workers = True

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=None,
            batch_sampler=None,
            num_workers=num_workers,
            collate_fn=collate_pairs,
            persistent_workers=_persistent_workers,
        )

    @staticmethod
    def get_mol_loader(
        dataset: SpectraMolDataset,
        shuffle: bool = False,
        batch_size: int = 32,
        num_workers: int = 0,
        persistent_workers: bool = False,
        **kwargs,
    ):
        """_summary_

        Function to return the proper paired dataloader

        Args:
            dataset (SpectraMolDataset): _description_
            shuffle (bool, optional): _description_. Defaults to False.
            batch_size (int, optional): _description_. Defaults to 32.
            num_workers (int, optional): _description_. Defaults to 0.
            persistent_workers (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """

        mol_collate_fn = dataset.get_featurizer().get_mol_collate()
        spec_collate_fn = lambda x: {}
        collate_pairs = partial(
            _collate_pairs,
            mol_collate_fn=mol_collate_fn,
            spec_collate_fn=spec_collate_fn,
        )

        _persistent_workers = False
        if num_workers > 0 and persistent_workers:
            _persistent_workers = True

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=None,
            batch_sampler=None,
            num_workers=num_workers,
            collate_fn=collate_pairs,
            persistent_workers=_persistent_workers,
        )

    def train_dataloader(self):
        """train_dataloader."""
        if self.forward_labels is not None:
            self.train.upsample_forward()
        return SpecDataModule.get_paired_loader(
            self.train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        """val_dataloader."""
        if self.debug == "test_val":
            # val_loader = SpecDataModule.get_paired_loader(
            #    self.val, shuffle=False, batch_size=self.batch_size,
            #    num_workers=self.num_workers,
            #    persistent_workers=self.persistent_workers
            # )
            test_loader = SpecDataModule.get_paired_loader(
                self.test,
                shuffle=False,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
            )
            return test_loader
        else:
            return SpecDataModule.get_paired_loader(
                self.val,
                shuffle=False,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
            )

    def test_dataloader(self):
        """test_dataloader."""
        if len(self.test) == 0:
            return None
        return SpecDataModule.get_paired_loader(
            self.test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )
