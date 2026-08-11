"""Compute avg Tanimoto and plot a histogram from one pred_fp.py pickle.

Usage: python tanimoto_from_preds.py <out_dir> name=pickle.p
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def tanimoto(preds: np.ndarray, targs: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """Per-row Tanimoto similarity between thresholded predicted and true fingerprints."""
    pred_bits = preds >= thresh
    targ_bits = targs >= thresh
    intersection = np.logical_and(pred_bits, targ_bits).sum(-1)
    union = np.logical_or(pred_bits, targ_bits).sum(-1)
    return np.where(union == 0, 1.0, intersection / np.where(union == 0, 1, union))


def sims_from_pickle(pred_pickle: str) -> np.ndarray:
    with open(pred_pickle, "rb") as fp:
        result = pickle.load(fp)
    preds, targs = np.asarray(result["preds"]), np.asarray(result["targs"])
    if targs.ndim == 0 or targs[0] is None:
        raise ValueError(f"No targets in {pred_pickle} — rerun pred_fp.py with --output-targs")
    return tanimoto(preds, targs)


def main(out_dir: str, name: str, pickle_path: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    sims = sims_from_pickle(pickle_path)
    stats = f"n={len(sims)}  avg={sims.mean():.4f}  median={np.median(sims):.4f}"
    print(f"{name}: {stats}")
    np.save(out_dir / f"tanimoto_{name}.npy", sims)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(sims, bins=50, range=(0, 1))
    ax.set_title(f"{name}\n{stats}")
    ax.set_xlabel("Tanimoto similarity")
    ax.set_ylabel("count")
    fig.tight_layout()
    out_path = out_dir / f"tanimoto_hist_{name}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved histogram to {out_path}")


def demo():
    preds = np.array([[0.9, 0.1, 0.8], [0.1, 0.9, 0.1]])
    targs = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    sims = tanimoto(preds, targs)
    assert np.allclose(sims, [1.0, 1.0]), sims
    preds2 = np.array([[1.0, 1.0, 0.0]])
    targs2 = np.array([[1.0, 0.0, 0.0]])
    assert np.isclose(tanimoto(preds2, targs2)[0], 0.5), tanimoto(preds2, targs2)
    print("demo ok")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        demo()
    else:
        out_dir = sys.argv[1]
        name, pickle_path = sys.argv[2].split("=", 1)
        main(out_dir, name, pickle_path)
