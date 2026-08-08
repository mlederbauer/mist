""" build_hdf5_index.py

Precompute a spec-name -> hdf5 key(s) index for a packed hdf5 file (e.g.
spec_files.hdf5, magma_outputs/*.hdf5, subformulae/*.hdf5) and cache it as
JSON next to the file. Listing all keys in a large hdf5 file is a slow
B-tree walk over a network filesystem; featurizers.py and datasets.py load
this cached index instead of re-listing the hdf5 on every job launch.

Usage:
    python -m mist.build_hdf5_index /path/to/magma_tsv.hdf5
"""
import argparse
import json
import time
from pathlib import Path

import h5py


def build_index(hdf5_file: str) -> dict:
    """Build a spec_name -> list-of-keys map.

    Keys may be packed one-per-spectrum ("{spec}.ext") or one-per-(spectrum,
    collision energy) ("{spec}_collision {energy}.ext"); either way, every
    spec_name maps to a list of one or more keys to be read and pooled.
    """
    f = h5py.File(hdf5_file, "r")
    name_map = {}
    count = 0
    t0 = time.time()
    for k in f.keys():
        spec_name = Path(k).stem.split("_collision")[0]
        name_map.setdefault(spec_name, []).append(k)
        count += 1
        if count % 50000 == 0:
            print(f"{count} keys, {time.time() - t0:.1f}s elapsed", flush=True)
    print(f"done: {count} keys, {len(name_map)} spectra, {time.time() - t0:.1f}s")
    return name_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5_file", help="Path to the packed hdf5 file to index")
    args = parser.parse_args()

    hdf5_file = Path(args.hdf5_file)
    out_path = hdf5_file.with_name(f"{hdf5_file.stem}_index.json")

    name_map = build_index(str(hdf5_file))

    with open(out_path, "w") as fp:
        json.dump(name_map, fp)
    print(f"wrote index to {out_path}")


if __name__ == "__main__":
    main()
