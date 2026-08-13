"""eval_realistic_mixed_aux.py

Score a --aux-gate (or --aux-dim) checkpoint on the FULL test set, using
each compound's actual available aux data -- not one artificial condition
forced on everyone. Concretely, per compound:
  - if it has a real starting_materials match, supply one (first match, for
    determinism)
  - if it ALSO has a real candidates match, supply that too
  - if it has neither, aux is absent (zero vector), same as today's default
    eval behavior

This is the number that answers "what would we actually see running this
checkpoint against a real, mixed population" -- as opposed to
analyze_reaction_sensitivity.py's per-condition breakdowns (which force
one uniform condition -- all-present or all-absent -- across compounds to
isolate the guidance-vs-memorization question), or the standard eval loop
(which forces aux absent for every compound, always).

Threshold is swept (same motivation as eval_checkpoint_threshold_sweep.py:
sparse fingerprints make the default 0.5 threshold a poor, misleading fit).

Usage:
    pixi run python analysis/eval_realistic_mixed_aux.py \
        --model-ckpt results/nist23_fp_mist_mhplus_aux_gate/split_1/split_1/best.ckpt \
        --labels-file data/nist23/labels_mh_only.tsv \
        --spec-folder $DATA/spec_files.hdf5 \
        --subform-folder data/nist23/subformulae/subform_50_repaired \
        --split-file $DATA/splits/split_1.tsv \
        --reaction-metadata-file data/nist23/reaction_metadata_uspto.tsv \
            data/nist23/reaction_metadata_cas.tsv data/nist23/reaction_metadata_pistachio.tsv
"""
import argparse
import copy as _copy

import torch
import numpy as np
import pandas as pd

from mist.models import base
from mist.data import datasets, featurizers


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ckpt", required=True)
    parser.add_argument("--labels-file", required=True)
    parser.add_argument("--spec-folder", required=True)
    parser.add_argument("--subform-folder", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument(
        "--reaction-metadata-file", required=True, nargs="+", action="store"
    )
    parser.add_argument("--max-reactions-per-compound", type=int, default=10)
    parser.add_argument("--subset-datasets", default="test_only",
                         choices=["none", "test_only", "val_only", "train_only"])
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
    )
    return parser.parse_args()


def _subset_to_fold(spectra_mol_pairs, split_file, fold_name):
    split_df = pd.read_csv(split_file, sep="\t")
    name_col = "name" if "name" in split_df.columns else split_df.columns[0]
    split_col = "split" if "split" in split_df.columns else split_df.columns[1]
    valid_names = set(split_df[name_col][split_df[split_col] == fold_name].values)
    return [(i, j) for i, j in spectra_mol_pairs if i.get_spec_name() in valid_names]


def tanimoto_at(preds: np.ndarray, targs: np.ndarray, thresh: float) -> float:
    pred_bool = preds > thresh
    targ_bool = targs.astype(bool)
    intersection = (pred_bool & targ_bool).sum(-1).astype(float)
    union = (pred_bool | targ_bool).sum(-1).astype(float)
    union = np.clip(union, 1, None)
    return float((intersection / union).mean())


def cosine_sim(preds: np.ndarray, targs: np.ndarray) -> float:
    num = (preds * targs).sum(-1)
    den = np.linalg.norm(preds, axis=-1) * np.linalg.norm(targs.astype(float), axis=-1)
    den = np.clip(den, 1e-8, None)
    return float((num / den).mean())


def main():
    args = get_args()
    device = torch.device("cuda:0") if args.gpu else torch.device("cpu")
    kwargs = vars(args).copy()

    ckpt = torch.load(args.model_ckpt, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    hp.update(kwargs)
    hp["max_count"] = None

    model = base.build_model(**hp)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()

    hp["spec_features"] = model.spec_features(mode="test")
    hp["mol_features"] = model.mol_features()
    hp["allow_none_smiles"] = False
    paired_featurizer = featurizers.get_paired_featurizer(**hp)

    spectra_mol_pairs = datasets.get_paired_spectra(**hp)
    spectra_mol_pairs = list(zip(*spectra_mol_pairs))

    subset = hp.get("subset_datasets")
    if subset != "none":
        fold_name = subset.removesuffix("_only")
        spectra_mol_pairs = _subset_to_fold(spectra_mol_pairs, hp["split_file"], fold_name)
    print(f"{len(spectra_mol_pairs)} pairs in {subset}")

    spectra_mol_pairs = datasets.attach_reactions(
        spectra_mol_pairs, hp["reaction_metadata_file"],
        max_reactions_per_compound=hp.get("max_reactions_per_compound") or None,
    )

    # Build the REALISTIC per-compound condition: whatever real aux data
    # that compound actually has (first match per source, for determinism),
    # nothing invented. Every compound appears exactly once here, unlike
    # analyze_reaction_sensitivity.py's expanded (compound, condition) rows.
    realistic_pairs = []
    has_sm_flags, has_cand_flags = [], []
    for spec, mol in spectra_mol_pairs:
        spec_copy = _copy.copy(spec)
        aux_data = {}
        sm_records = [r for r in spec.get_aux_records("starting_materials") if r.get("smiles")]
        cand_records = [r for r in spec.get_aux_records("candidates") if r.get("smiles")]
        has_sm = len(sm_records) > 0
        has_cand = len(cand_records) > 0
        if has_sm:
            aux_data["starting_materials"] = [sm_records[0]]
        if has_cand:
            aux_data["candidates"] = [cand_records[0]]
        spec_copy.aux_data = aux_data
        realistic_pairs.append((spec_copy, mol))
        has_sm_flags.append(has_sm)
        has_cand_flags.append(has_cand)

    has_sm_flags = np.array(has_sm_flags)
    has_cand_flags = np.array(has_cand_flags)
    has_either = has_sm_flags | has_cand_flags
    print(f"has starting_materials: {has_sm_flags.sum()}/{len(has_sm_flags)} "
          f"({100 * has_sm_flags.mean():.1f}%)")
    print(f"has candidates: {has_cand_flags.sum()}/{len(has_cand_flags)} "
          f"({100 * has_cand_flags.mean():.1f}%)")
    print(f"has either: {has_either.sum()}/{len(has_either)} "
          f"({100 * has_either.mean():.1f}%)")
    print(f"has neither (fully cold-start): {(~has_either).sum()}/{len(has_either)} "
          f"({100 * (~has_either).mean():.1f}%)")

    hp["aux_use_preset_data"] = True
    dataset = datasets.SpectraMolDataset(
        spectra_mol_list=realistic_pairs, featurizer=paired_featurizer, **hp
    )
    preds = model.encode_all_spectras(dataset, no_grad=True, **hp).cpu().numpy()
    targs = model.encode_all_mols(dataset, no_grad=True, **hp).cpu().numpy()

    print(f"\ncosine similarity (realistic mixed aux, whole set): {cosine_sim(preds, targs):.4f}")
    print(f"cosine similarity (has_either subset):   {cosine_sim(preds[has_either], targs[has_either]):.4f}")
    print(f"cosine similarity (neither subset):       {cosine_sim(preds[~has_either], targs[~has_either]):.4f}")

    best_t, best_tan = None, -1.0
    for t in args.thresholds:
        tan = tanimoto_at(preds, targs, t)
        marker = ""
        if tan > best_tan:
            best_tan, best_t = tan, t
            marker = "  <-- best so far"
        print(f"  thresh={t:.2f}  tanimoto(ALL)={tan:.4f}  "
              f"tanimoto(has_either)={tanimoto_at(preds[has_either], targs[has_either], t):.4f}  "
              f"tanimoto(neither)={tanimoto_at(preds[~has_either], targs[~has_either], t):.4f}{marker}")

    print(f"\n>>> REALISTIC MIXED-AUX WHOLE-SET BEST: thresh={best_t}, tanimoto={best_tan:.4f}")


if __name__ == "__main__":
    main()
