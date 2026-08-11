""" analyze_candidate_discrimination.py

Tests whether the candidates aux pathway has learned genuine discrimination
(responds differently to a REAL candidate vs. a plausible-but-wrong DECOY
candidate for the same compound) or just "presence of any candidate helps a
little regardless of content" (no real discrimination -- in which case more
candidates coverage in training data, or a better candidate source than
smartreact's naive pairwise combinations, is the actual lever to pull).

For each test compound with a real candidate available, build three rows:
  - none:  no candidate supplied (zero vector)
  - real:  that compound's own real candidate (positive control)
  - decoy: a randomly-chosen OTHER compound's real candidate (same aux
           machinery exercised, but a candidate that has no reason to be
           structurally informative for THIS compound)

If real clearly beats decoy on average, the model discriminates candidate
CONTENT, not just candidate presence -- evidence that the sparse training
signal already generalizes, and scaling up an expensive/inaccurate source
like ASKCOS could plausibly help further. If real and decoy score similarly,
the model is not yet using candidate content meaningfully, and more
data alone is unlikely to fix that without also improving how candidates are
integrated/trained on.

Usage:
    pixi run python -m mist.analyze_candidate_discrimination \
        --model-ckpt results/nist23_fp_mist_aux32_allrxn/split_1/split_1/last.ckpt \
        --labels-file $DATA/labels.tsv \
        --spec-folder $DATA/spec_files.hdf5 \
        --subform-folder $DATA/subformulae/magma_subform_50.hdf5 \
        --split-file $DATA/splits/split_1.tsv \
        --reaction-metadata-file data/nist23/reaction_metadata_uspto.tsv \
        --subset-datasets test_only \
        --save-dir results/nist23_fp_mist_aux32_allrxn/split_1/split_1/preds_candidates
"""
from pathlib import Path
import argparse
import copy as _copy
import logging

import numpy as np
import pandas as pd
import torch

from mist.models import base
from mist.data import datasets, featurizers
from mist import utils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ckpt", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--labels-file", required=True)
    parser.add_argument("--spec-folder", required=True)
    parser.add_argument("--subform-folder", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument(
        "--reaction-metadata-file", required=True, nargs="+", action="store"
    )
    parser.add_argument("--max-reactions-per-compound", type=int, default=10)
    parser.add_argument(
        "--subset-datasets",
        default="test_only",
        choices=["none", "test_only", "val_only", "train_only"],
    )
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _subset_to_fold(spectra_mol_pairs, split_file, fold_name):
    split_df = pd.read_csv(split_file, sep="\t")
    name_col = "name" if "name" in split_df.columns else split_df.columns[0]
    split_col = "split" if "split" in split_df.columns else split_df.columns[1]
    valid_names = set(split_df[name_col][split_df[split_col] == fold_name].values)
    return [(i, j) for i, j in spectra_mol_pairs if i.get_spec_name() in valid_names]


def _tanimoto(pred_fp: np.ndarray, target_fp: np.ndarray, thresh: float) -> np.ndarray:
    pred_bool = pred_fp > thresh
    target_bool = target_fp.astype(bool)
    intersection = (pred_bool & target_bool).sum(-1)
    union = (pred_bool | target_bool).sum(-1)
    return intersection / np.clip(union, 1, None)


def run_analysis():
    args = get_args()
    kwargs = args.__dict__.copy()
    save_dir = Path(kwargs["save_dir"])
    save_dir.mkdir(exist_ok=True, parents=True)
    utils.setup_logger(save_dir, log_name="analyze_candidate_discrimination.log")
    device = torch.device("cuda:0") if kwargs.get("gpu") else torch.device("cpu")

    pretrain_ckpt = torch.load(
        kwargs["model_ckpt"], map_location=torch.device("cpu"), weights_only=False
    )
    main_hparams = pretrain_ckpt["hyper_parameters"]
    if main_hparams.get("aux_dim", 0) <= 0:
        raise ValueError(
            "This checkpoint was trained with aux_dim=0 -- candidate-discrimination "
            "analysis requires a checkpoint trained with --aux-dim > 0."
        )
    main_hparams.update(kwargs)
    kwargs = main_hparams
    kwargs["max_count"] = None

    model = base.build_model(**kwargs)
    model.load_state_dict(pretrain_ckpt["state_dict"])
    model = model.to(device).eval()

    kwargs["spec_features"] = model.spec_features(mode="test")
    kwargs["mol_features"] = model.mol_features()
    kwargs["allow_none_smiles"] = False
    paired_featurizer = featurizers.get_paired_featurizer(**kwargs)

    spectra_mol_pairs = datasets.get_paired_spectra(**kwargs)
    spectra_mol_pairs = list(zip(*spectra_mol_pairs))

    subset_datasets = kwargs.get("subset_datasets")
    if subset_datasets != "none":
        fold_name = subset_datasets.removesuffix("_only")
        logging.info(f"Subset to {fold_name} of split {kwargs['split_file']}")
        spectra_mol_pairs = _subset_to_fold(
            spectra_mol_pairs, kwargs["split_file"], fold_name
        )

    spectra_mol_pairs = datasets.attach_reactions(
        spectra_mol_pairs,
        kwargs["reaction_metadata_file"],
        max_reactions_per_compound=kwargs.get("max_reactions_per_compound") or None,
    )

    # Compounds with at least one usable (non-empty) candidates record.
    with_candidates = [
        (spec, mol)
        for spec, mol in spectra_mol_pairs
        if any(r.get("smiles") for r in spec.get_aux_records("candidates"))
    ]
    logging.info(
        f"{len(with_candidates)}/{len(spectra_mol_pairs)} pairs have a real "
        "candidates record -- analyzing these"
    )
    if len(with_candidates) < 2:
        logging.info("Not enough compounds with candidates to build decoys -- exiting")
        return

    rng = np.random.default_rng(kwargs["seed"])
    real_records = [
        next(r for r in spec.get_aux_records("candidates") if r.get("smiles"))
        for spec, _mol in with_candidates
    ]

    expanded_pairs = []
    row_meta = []  # (spec_name, condition) where condition in {"none","real","decoy"}
    for i, (spec, mol) in enumerate(with_candidates):
        spec_name = spec.get_spec_name()

        none_copy = _copy.copy(spec)
        none_copy.aux_data = {"starting_materials": [], "candidates": []}
        expanded_pairs.append((none_copy, mol))
        row_meta.append((spec_name, "none"))

        real_copy = _copy.copy(spec)
        real_copy.aux_data = {"starting_materials": [], "candidates": [real_records[i]]}
        expanded_pairs.append((real_copy, mol))
        row_meta.append((spec_name, "real"))

        # Decoy: a different compound's real candidate record, picked
        # uniformly at random from all OTHER compounds in this set.
        decoy_idx = rng.integers(len(with_candidates) - 1)
        if decoy_idx >= i:
            decoy_idx += 1
        decoy_copy = _copy.copy(spec)
        decoy_copy.aux_data = {
            "starting_materials": [],
            "candidates": [real_records[decoy_idx]],
        }
        expanded_pairs.append((decoy_copy, mol))
        row_meta.append((spec_name, "decoy"))

    logging.info(f"Built {len(expanded_pairs)} rows (3 per compound) for a single pass")

    thresh = model.thresh
    kwargs["aux_use_preset_data"] = True
    dataset = datasets.SpectraMolDataset(
        spectra_mol_list=expanded_pairs, featurizer=paired_featurizer, **kwargs
    )
    preds = model.encode_all_spectras(dataset, no_grad=True, **kwargs).cpu().numpy()
    targs = model.encode_all_mols(dataset, no_grad=True, **kwargs).cpu().numpy()
    sims = _tanimoto(preds, targs, thresh)

    per_compound = {}
    for (spec_name, condition), sim in zip(row_meta, sims):
        per_compound.setdefault(spec_name, {})[condition] = float(sim)

    rows = [
        {
            "spec_name": name,
            "none_tanimoto": d.get("none"),
            "real_tanimoto": d.get("real"),
            "decoy_tanimoto": d.get("decoy"),
        }
        for name, d in per_compound.items()
    ]
    result_df = pd.DataFrame(rows)
    out_path = save_dir / "candidate_discrimination.tsv"
    result_df.to_csv(out_path, sep="\t", index=False)

    real_vs_decoy = result_df["real_tanimoto"] - result_df["decoy_tanimoto"]
    real_vs_none = result_df["real_tanimoto"] - result_df["none_tanimoto"]
    decoy_vs_none = result_df["decoy_tanimoto"] - result_df["none_tanimoto"]
    logging.info(f"Wrote {len(result_df)} compounds to {out_path}")
    logging.info(f"Mean real_tanimoto: {result_df['real_tanimoto'].mean():.4f}")
    logging.info(f"Mean decoy_tanimoto: {result_df['decoy_tanimoto'].mean():.4f}")
    logging.info(f"Mean none_tanimoto: {result_df['none_tanimoto'].mean():.4f}")
    logging.info(
        f"Mean (real - decoy): {real_vs_decoy.mean():.4f} -- "
        "positive and large means genuine content discrimination"
    )
    logging.info(
        f"Mean (real - none): {real_vs_none.mean():.4f}, "
        f"mean (decoy - none): {decoy_vs_none.mean():.4f} -- "
        "if these two are similar, candidates help mainly via PRESENCE, not "
        "CONTENT -- decoy wins as often as real would suggest no real content signal"
    )
    logging.info(
        f"Fraction where real beats decoy: {(real_vs_decoy > 0).mean():.4f}"
    )


if __name__ == "__main__":
    run_analysis()
