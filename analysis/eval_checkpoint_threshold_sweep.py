"""eval_checkpoint_threshold_sweep.py

Re-score a MIST checkpoint's val/test predictions with a swept binarization
threshold, instead of trusting a single hardcoded value. Motivation: the
default threshold (0.5, or whatever --binarization-thresh a checkpoint was
trained with) is a poor fit for sparse fingerprints (~1% bit density) --
most checkpoints' true optimal threshold sits well below 0.5, and for
--aux-gate checkpoints specifically the softmax blend can push predicted
probabilities systematically lower as training improves, making the
default-threshold Tanimoto look like it's getting WORSE even as cosine
similarity (threshold-free) improves. Cosine similarity is reported
alongside as a sanity check that isn't affected by this issue.

Usage:
    pixi run python -m analysis.eval_checkpoint_threshold_sweep \
        --model-ckpt results/nist23_fp_mist_mhplus_aux_gate/split_1/split_1/last.ckpt \
        --labels-file data/nist23/labels_mh_only.tsv \
        --spec-folder $DATA/spec_files.hdf5 \
        --subform-folder data/nist23/subformulae/subform_50_repaired \
        --split-file $DATA/splits/split_1.tsv
"""
import argparse

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
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7],
    )
    return parser.parse_args()


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

    ckpt = torch.load(args.model_ckpt, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    hp["spec_folder"] = args.spec_folder
    hp["subform_folder"] = args.subform_folder
    hp["labels_file"] = args.labels_file
    hp["split_file"] = args.split_file
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
    print(f"{len(spectra_mol_pairs)} total pairs loaded")

    split_df = pd.read_csv(hp["split_file"], sep="\t")
    name_col = "name" if "name" in split_df.columns else split_df.columns[0]
    split_col = "split" if "split" in split_df.columns else split_df.columns[1]

    for fold_name in ["val", "test"]:
        valid_names = set(split_df[name_col][split_df[split_col] == fold_name].values)
        fold_pairs = [
            (s, m) for s, m in spectra_mol_pairs if s.get_spec_name() in valid_names
        ]
        print(f"\n=== {fold_name}: {len(fold_pairs)} pairs ===")
        if not fold_pairs:
            continue

        dataset = datasets.SpectraMolDataset(
            spectra_mol_list=fold_pairs, featurizer=paired_featurizer, **hp
        )
        preds = model.encode_all_spectras(dataset, no_grad=True, **hp).cpu().numpy()
        targs = model.encode_all_mols(dataset, no_grad=True, **hp).cpu().numpy()

        print(f"preds min/max/mean: {preds.min():.4f} {preds.max():.4f} {preds.mean():.4f}")
        print(f"cosine similarity: {cosine_sim(preds, targs):.4f}")
        print(f"target mean bit density: {targs.mean():.4f}")

        best_t, best_tan = None, -1.0
        for t in args.thresholds:
            tan = tanimoto_at(preds, targs, t)
            marker = ""
            if tan > best_tan:
                best_tan, best_t = tan, t
                marker = "  <-- best so far"
            print(
                f"  thresh={t:.2f}  tanimoto={tan:.4f}  "
                f"pred_density={(preds > t).mean():.4f}{marker}"
            )

        print(f">>> {fold_name} BEST: thresh={best_t}, tanimoto={best_tan:.4f}")
        print(f">>> {fold_name} at thresh=0.5: tanimoto={tanimoto_at(preds, targs, 0.5):.4f}")


if __name__ == "__main__":
    main()
