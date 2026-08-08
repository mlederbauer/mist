""" hdf5_utils.py

Lazy readers for datasets that pack many small per-spectrum files (.ms,
.magma, subformula .json) into a single hdf5 file keyed by filename, as an
alternative to a directory of individual files. Used when a --spec-folder /
--magma-folder / --subform-folder CLI arg points at a .hdf5 file instead of
a directory.
"""
from pathlib import Path
from typing import Dict, List

import h5py


def is_hdf5_path(path) -> bool:
    """True if path points at an .hdf5 file (rather than a directory)."""
    if path is None:
        return False
    return Path(path).suffix == ".hdf5"


class Hdf5Store:
    """Hdf5Store.

    Opens an hdf5 file once and decodes values (raw bytes stored as
    shape-(1,) object arrays) to str on lookup.
    """

    def __init__(self, hdf5_file: str):
        self.hdf5_file = str(hdf5_file)
        self._handle = None

    @property
    def handle(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.hdf5_file, "r")
        return self._handle

    def keys(self) -> List[str]:
        return list(self.handle.keys())

    def __contains__(self, key: str) -> bool:
        return key in self.handle

    def __getitem__(self, key: str) -> str:
        val = self.handle[key][()]
        item = val[0] if hasattr(val, "__len__") else val
        return item.decode() if isinstance(item, bytes) else str(item)
