# NIST23 Training — Changelog

Context: training a MIST fingerprint model on NIST23 (`/orcd/data/ccoley/001/msms_data/nist23/`),
a dataset prepared by someone else using an hdf5-packed storage format that diverged from this
repo's original directory-of-files convention in several ways. Each divergence surfaced as a
training crash; this log tracks what broke, why, and how it was fixed, in the order discovered.

## Environment

- No conda/mamba was available on this system for this user. Installed [pixi](https://pixi.sh)
  and ran `pixi init --import environment.yml` to translate the existing conda env spec — see
  `pixi.toml`. Use `pixi run <command>` (e.g. `pixi run python src/mist/train_mist.py ...`)
  instead of `conda activate`.
- Added `wandb` to the pixi env (`pixi add wandb`) for training visibility (see below).

## Bugs found and fixed, in the order hit

### 1. Spectra stored as a single `spec_files.hdf5` instead of a directory of `.ms` files

The repo's loader (`get_paired_spectra` in `src/mist/data/datasets.py`) only knew how to `glob`
a directory of `.ms` files. NIST23's `spec_files.hdf5` is a flat `h5py.File` keyed by
`"{spec}.ms"`, each value the raw `.ms` file text as bytes.

**Fix:** new `src/mist/utils/hdf5_utils.py` (`Hdf5Store`, `is_hdf5_path`) plus an hdf5-aware
branch in `get_paired_spectra`, `Spectra` (`src/mist/data/data.py`), and
`utils.parse_spectra_str` (new function in `src/mist/utils/parse_utils.py`, factored out of
`parse_spectra` so it can parse in-memory text instead of only opening a file path). The
directory-based path is untouched — hdf5 support is a new alternative branch, not a replacement.

**Perf note:** the initial version called `Hdf5Store.keys()` (a full listing) before filtering by
`max_count` — a full listing of a 176K-key hdf5 file is a slow B-tree walk over this networked
filesystem (took minutes, sometimes didn't finish in the timeouts tried). Fixed by driving
candidate spec names from the already-in-hand `labels.tsv` and doing single-key membership
checks (`key in Hdf5Store`) instead of listing everything.

### 2. MAGMa and subformula outputs also hdf5-packed, and split **per collision energy**

`magma_outputs/magma_tsv.hdf5` and `subformulae/magma_subform_50.hdf5` are similarly packed, but
keyed as `"{spec}_collision {energy}.{ext}"` — one entry per (spectrum, collision-energy) pair,
not one per whole spectrum as this repo's own `assign_subformulae.py` / `run_magma.py` produce.

**Fix:** `src/mist/data/featurizers.py`'s `PeakFormula.__init__` now groups hdf5 keys by spec
name (stripping the `_collision N` suffix) into a list, for both the subform and magma branches.
`_get_peak_dict` was rewritten to load every tree/table for a spectrum and pool them: for magma,
concatenate the per-collision-energy dataframes (safe — the consumer already does a flat
nearest-mz match against the pooled peak list); for subform JSON trees, take `cand_form`/
`cand_ion` from the first tree (verified identical across collision energies for a given
spectrum) and concatenate `output_tbl`'s `formula`/`ms2_inten`/`ions` lists across all of them.
This mirrors what the repo's own per-spectrum generation code already does internally (it parses
all collision blocks of a `.ms` file at once before producing one pooled tree).

**Perf:** listing all keys in these ~8-9GB hdf5 files (over 1.1M keys each) is slow (5–16 min,
one-off). Added `src/mist/build_hdf5_index.py` (`python -m mist.build_hdf5_index <path>`) to
build a `{stem}_index.json` sidecar once; `PeakFormula.__init__` loads that cached index if
present (near-instant) instead of re-listing on every job launch.

### 3. Split files use different column names

`splitter.PresetSpectraSplitter` hardcoded columns `name`/`split`. NIST23's split files
(`splits/split_1.tsv`, `splits/scaffold_1.tsv`) use `spec`/`Fold_0` with the same train/val/test
string values.

**Fix:** `src/mist/data/splitter.py` now falls back to the first two columns positionally if
`name`/`split` aren't present.

### 4. Unparseable SMILES silently corrupted `mol_list`

`get_paired_spectra` filtered `spectra_list` and `mol_list` independently by the same weak
condition (`smiles string is not None`), which doesn't account for `Mol.MolFromSmiles` itself
returning `None` on an RDKit parse failure. NIST23 includes SMILES with stereo-descriptors
(`[S@SP3]`, `[P@SP2]`, `[P@SP3]` — trigonal-bipyramidal/square-planar chirality) that this
project's pinned RDKit 2021.03 can't parse. The resulting `None`s slipped into `mol_list` and
crashed `SpectraMolDataset.__init__` with `AttributeError: 'NoneType' object has no attribute
'get_smiles'`.

**Fix:** `get_paired_spectra` now zips spectra+smiles+inchikey, calls `MolFromSmiles` once, and
filters the *paired* list by whether the mol actually parsed — logging how many were dropped.
On NIST23 split_1: **3,355 spectra dropped** (out of 176,851) for unparseable SMILES.

### 5. Ion/adduct vocabulary too narrow for NIST23

`utils.chem_utils.ION_LST` only had 7 hardcoded H+ positive-mode adducts (matching what
canopus_train/csi2022 needed). NIST23 has 13 distinct ionization types, including negative-mode
adducts — `[M-H]-` alone is ~19% of the dataset (34,267 / 176,851 spectra). Crashed with
`KeyError` in `utils.get_ion_idx`.

Also found: `get_ion_idx` called `ion_to_idx[ionization]` directly, **never applying**
`ion_remap` (a dict meant to normalize raw string spellings) — a separate pre-existing bug.

**Fix:** extended `ION_LST` from 7 to 14 entries (see `chem_utils.py`) with chemically-derived
mass deltas (`ion_to_mass`) and element-count vectors (`ion_to_add_vec`) for each new adduct,
and fixed `get_ion_idx` to apply `ion_remap` first. `num_adducts` (which sizes the model's ion
embedding table) derives from `len(ION_LST)` in one place, so this propagates automatically.

Full adduct distribution in NIST23 (`labels.tsv`):

| Adduct | Count |
|---|---|
| `[M+H]+` | 80,372 |
| `[M-H]-` | 34,267 |
| `[M-H2O+H]+` | 19,497 |
| `[M+Na]+` | 17,907 |
| `[M+H-NH3]+` | 4,111 |
| `[M-H-CO2]-` | 4,018 |
| `[M-H4O2+H]+` | 3,929 |
| `[M+H3N+H]+` | 3,254 |
| `[M-H-H2O]-` | 2,841 |
| `[M+CHO2]-` | 2,377 |
| `[M+Cl]-` | 2,123 |
| `[M+H-CH2O2]+` | 1,818 |
| `[M+K]+` | 337 |

### 6. MAGMa TSVs missing the `frag_fp` column (`--magma-aux-loss` incompatible)

This repo's own `run_magma.py` writes columns `mz_observed, mz_corrected, inten, ppm_diff,
frag_inds, frag_mass, frag_h_shift, frag_base_form, frag_hashes, frag_fp`. NIST23's magma TSVs
have all of these **except `mz_corrected` and `frag_fp`**.

- `mz_corrected` is derivable (`mz_observed - ion_to_mass[adduct]`) but wasn't worth patching in
  isolation, because:
- `frag_fp` (the fragment fingerprint bits used as the auxiliary-loss target) is **not**
  derivable from the other columns — it requires re-running MAGMa's fragment-fingerprinting
  step (`frag_fp.fp_from_frag`) against the original molecule fragment graphs, which NIST23's
  precomputed data doesn't retain.

**Decision:** disabled `--magma-aux-loss` / `--magma-folder` for NIST23 training rather than
regenerate the missing data. This drops one auxiliary supervision signal from the original
paper's recipe; the core fingerprint-prediction objective is unaffected. Revisit if the aux loss
turns out to matter for NIST-scale performance.

### 7. CUDA OOM: `--max-peaks` was uncapped

Once magma-aux-loss was disabled and everything else fixed, training reached real forward passes
but crashed with `RuntimeError: CUDA out of memory. Tried to allocate 40.08 GiB`. Root cause:
`modules.py`'s `form_diffs = orig_form_vec[:, :, None, :] - orig_form_vec[:, None, :, :]` is an
all-pairs peak-vs-peak formula-difference tensor, O(num_peaks²). `--max-peaks` defaults to
`None` (uncapped), and NIST23 spectra — which pool many collision energies per spectrum (some
spectra have 20+ collision blocks) — have far more peaks per spectrum than canopus_train ever
did.

**Fix:** added `--max-peaks 50` to the training command (matching a sibling featurizer class's
own default).

## Feature additions (not bugs)

### `--embed-instrument`

NIST23's `labels.tsv` has a real `instrument` column (values like `Orbitrap`, `QTOF`, etc).
`--embed-instrument` (existing flag, off by default) enables the model's instrument embedding.
Turned on for this run.

### wandb logging

The repo only had TensorBoard + a console logger, no wandb integration at all. Added:
- `--wandb-project` / `--wandb-entity` CLI flags (`src/mist/parsing.py`)
- A conditional `WandbLogger` alongside the existing `TensorBoardLogger` in
  `src/mist/models/base.py`'s `train_model` (only instantiated when `--wandb-project` is set,
  and skipped during `tune` / hyperopt runs)
- `wandb` added to the pixi environment

Runs log to `wandb.ai/<entity>/<project>`; this NIST23 run uses project `mist-nist23`, entity
defaulting to the logged-in user (`mlederbauer`, verified via `~/.netrc`).

### Checkpoint + auto-resume for preemption

Training runs on `mit_preemptable` (see the Slurm script), which can preempt jobs at any time.
Previously, the only checkpoint saved was `best.ckpt` (`save_weights_only=True` — model weights
only, no optimizer/scheduler/epoch state), and there was no mechanism to resume mid-training:
`--requeue` would resubmit the exact same `sbatch` script from scratch with no checkpoint
argument, silently restarting at epoch 0 every time a job was preempted.

**Fix** (`src/mist/models/base.py`, `train_model`):
- Added a second `ModelCheckpoint` callback with `save_last=True, save_weights_only=False` —
  writes a full-state `last.ckpt` (optimizer, LR scheduler, epoch, global step) every epoch,
  overwriting in place.
- `trainer.fit(...)` now checks whether `last.ckpt` already exists in the run's checkpoint
  directory and passes it as `ckpt_path=...` if so, logging `"Resuming training from ..."`.
- This works because `TensorBoardLogger`'s `version` is set to the fixed split name (e.g.
  `split_1`), not auto-incremented — so every resubmission of the same job lands in the same
  checkpoint directory and finds the previous `last.ckpt`.

Not yet exercised against a real preemption event (hard to trigger deliberately), but confirmed
`last.ckpt` gets written during normal training.

## Data quality notes (not fixed, just observed)

- **NIST26** (`/orcd/data/ccoley/001/msms_data/nist26/`, ~402K spectra) has no `splits/` or
  `retrieval/` directories yet — everything else (labels, subformulae, magma_outputs, mgf) is
  pre-processed. Splits need to be created before NIST26 can be trained on with this pipeline.
- NIST23's `nist20` sibling dataset has the older `name`/`split` column convention and no
  `instrument` column — NIST23 added `instrument` and switched to `spec`/`Fold_N` at some point
  in this data-generation pipeline's history.

## Final training command

See `run_scripts/submit_nist23_fp_mist.sh`. Key flags relative to the README's canopus_train
example: `--spec-folder`/`--subform-folder` point at `.hdf5` files, no `--magma-folder`/
`--magma-aux-loss`, no `--forward-labels`/`--forward-aug-folder` (no ICEBERG-augmented data
exists for NIST23 yet), added `--embed-instrument`, `--max-peaks 50`, `--wandb-project
mist-nist23`. Hyperparameters (hidden-size, learning-rate, etc.) are copied from the README's
canopus_train example and **have not been separately tuned for NIST23**.
