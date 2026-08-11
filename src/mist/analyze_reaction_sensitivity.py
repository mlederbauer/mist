""" analyze_reaction_sensitivity.py

For an aux-conditioned MIST checkpoint, measure whether reaction context acts
as guidance or memorization: for each val/test compound matching >=2
reactions, run inference once per matched reaction (steered via
aux_reaction_id_by_spec) plus once with no reaction, and report the spread of
resulting Tanimoto similarity to the true fingerprint per compound.

Low spread (all reactions + no-reaction give similar, similarly-good Tanimoto)
means the model treats reaction context as a soft prior -- guidance. High
spread (some reactions give great predictions, others terrible, wildly
different from the no-reaction baseline) means the model learned
reaction-specific shortcuts -- memorization.

Usage:
    pixi run python -m mist.analyze_reaction_sensitivity \
        --model-ckpt results/nist23_fp_mist_aux32_allrxn/split_1/split_1/last.ckpt \
        --labels-file $DATA/labels.tsv \
        --spec-folder $DATA/spec_files.hdf5 \
        --subform-folder $DATA/subformulae/magma_subform_50.hdf5 \
        --split-file $DATA/splits/split_1.tsv \
        --reaction-metadata-file data/nist23/reaction_metadata_uspto.tsv \
            data/nist23/reaction_metadata_cas.tsv data/nist23/reaction_metadata_pistachio.tsv \
        --subset-datasets test_only \
        --save-dir results/nist23_fp_mist_aux32_allrxn/split_1/split_1/preds
"""
from pathlib import Path
import argparse
import logging
import pickle

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
        "--aux-source",
        default="starting_materials",
        choices=["starting_materials", "candidates"],
        help=(
            "Which aux source to steer with -- starting_materials (reaction "
            "context) or candidates (only populated for a subset of USPTO "
            "rows; a checkpoint trained without candidates data present will "
            "have an untrained-but-live aux_projections['candidates'] "
            "pathway, so this tests what an unlearned projection produces vs. "
            "one that saw real candidates during training)."
        ),
    )
    parser.add_argument(
        "--subset-datasets",
        default="test_only",
        choices=["none", "test_only", "val_only", "train_only"],
    )
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--debug", action="store_true")
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
    utils.setup_logger(save_dir, log_name="analyze_reaction_sensitivity.log")
    device = torch.device("cuda:0") if kwargs.get("gpu") else torch.device("cpu")

    pretrain_ckpt = torch.load(kwargs["model_ckpt"], map_location=torch.device("cpu"))
    main_hparams = pretrain_ckpt["hyper_parameters"]
    if main_hparams.get("aux_dim", 0) <= 0:
        raise ValueError(
            "This checkpoint was trained with aux_dim=0 -- reaction-sensitivity "
            "analysis requires a checkpoint trained with --aux-dim > 0."
        )
    main_hparams.update(kwargs)
    kwargs = main_hparams
    kwargs["max_count"] = 10 if args.debug else None

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

    aux_source = kwargs["aux_source"]

    # Keep only compounds with >=2 matched reactions that ALSO have non-empty
    # data for this source -- candidates is only populated for a subset of
    # USPTO rows, so a reaction match doesn't guarantee usable candidates.
    multi_reaction_pairs = [
        (spec, mol)
        for spec, mol in spectra_mol_pairs
        if sum(1 for r in spec.get_aux_records(aux_source) if r.get("smiles")) >= 2
    ]
    logging.info(
        f"{len(multi_reaction_pairs)}/{len(spectra_mol_pairs)} pairs have "
        ">=2 matched reactions -- analyzing these"
    )
    if not multi_reaction_pairs:
        logging.info("Nothing to analyze -- exiting")
        return

    # Build ONE expanded pair list covering every (compound, condition) combo
    # -- one row per matched reaction, plus one "no reaction" row per
    # compound -- and run it through the model in a SINGLE pass. The naive
    # alternative (one full-dataset pass per distinct reaction_id, steered
    # globally via aux_reaction_id_by_spec) does not scale: tens of thousands
    # of distinct reaction_ids across a realistic multi-match test set would
    # mean tens of thousands of full passes. Each duplicated Spectra copy
    # here gets its OWN aux_data preset to exactly one reaction (or none),
    # so a plain single forward pass through this list already reflects the
    # right per-row condition -- no dataset-level steering needed.
    import copy as _copy

    expanded_pairs = []
    # (unique_row_id, spec_name, condition) so results can be re-grouped by
    # compound afterward; unique_row_id disambiguates duplicated spec names.
    row_meta = []
    for spec, mol in multi_reaction_pairs:
        spec_name = spec.get_spec_name()

        none_copy = _copy.copy(spec)
        none_copy.aux_data = {}
        expanded_pairs.append((none_copy, mol))
        row_meta.append((spec_name, "none"))

        for record in spec.get_aux_records(aux_source):
            if not record.get("smiles"):
                continue
            rid = record["reaction_id"]
            reaction_copy = _copy.copy(spec)
            reaction_copy.aux_data = {"starting_materials": [], "candidates": []}
            reaction_copy.aux_data[aux_source] = [record]
            expanded_pairs.append((reaction_copy, mol))
            row_meta.append((spec_name, rid))

    logging.info(
        f"Expanded {len(multi_reaction_pairs)} compounds into "
        f"{len(expanded_pairs)} (compound, condition) rows for a single pass"
    )

    thresh = model.thresh
    kwargs["aux_use_preset_data"] = True
    dataset = datasets.SpectraMolDataset(
        spectra_mol_list=expanded_pairs, featurizer=paired_featurizer, **kwargs
    )
    preds = model.encode_all_spectras(dataset, no_grad=True, **kwargs).cpu().numpy()
    targs = model.encode_all_mols(dataset, no_grad=True, **kwargs).cpu().numpy()
    sims = _tanimoto(preds, targs, thresh)

    # Debug dump for manual row-alignment spot-checking: raw (spec_name,
    # condition, sim, target_bit_count) per row -- target_bit_count should be
    # IDENTICAL across all rows of the same spec_name (same compound, same
    # true fingerprint, only aux input varies), which is the cheapest
    # possible proof the per-row alignment isn't shuffled/off-by-one.
    debug_rows = [
        {
            "spec_name": spec_name,
            "condition": condition,
            "tanimoto": float(sim),
            "target_bit_count": int(targ.sum()),
        }
        for (spec_name, condition), sim, targ in zip(row_meta, sims, targs)
    ]
    pd.DataFrame(debug_rows).to_csv(
        save_dir / "reaction_sensitivity_debug_rows.tsv", sep="\t", index=False
    )

    per_compound_tanimoto = {}
    for (spec_name, condition), sim in zip(row_meta, sims):
        per_compound_tanimoto.setdefault(spec_name, {})[condition] = float(sim)

    rows = []
    for spec_name, cond_sims in per_compound_tanimoto.items():
        reaction_sims = [v for k, v in cond_sims.items() if k != "none"]
        if len(reaction_sims) < 2:
            continue
        rows.append(
            {
                "spec_name": spec_name,
                "n_reactions": len(reaction_sims),
                "no_reaction_tanimoto": cond_sims.get("none"),
                "reaction_tanimoto_mean": float(np.mean(reaction_sims)),
                "reaction_tanimoto_std": float(np.std(reaction_sims)),
                "reaction_tanimoto_min": float(np.min(reaction_sims)),
                "reaction_tanimoto_max": float(np.max(reaction_sims)),
            }
        )

    result_df = pd.DataFrame(rows)
    out_path = save_dir / "reaction_sensitivity.tsv"
    result_df.to_csv(out_path, sep="\t", index=False)
    logging.info(f"Wrote {len(result_df)} compounds' reaction-sensitivity to {out_path}")
    logging.info(
        "Mean within-compound reaction std (lower = more guidance-like, "
        f"higher = more memorization-like): {result_df['reaction_tanimoto_std'].mean():.4f}"
    )
    logging.info(
        "Mean |reaction_mean - no_reaction| gap: "
        f"{(result_df['reaction_tanimoto_mean'] - result_df['no_reaction_tanimoto']).abs().mean():.4f}"
    )


if __name__ == "__main__":
    run_analysis()
