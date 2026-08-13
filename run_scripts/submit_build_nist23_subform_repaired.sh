#!/bin/bash
#SBATCH --job-name=nist23_subform_repaired
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=mit_preemptable,mit_normal,pi_ccoley
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue

cd /home/magled/mist
source "run_scripts/mail_notify.sh"
export PATH="$HOME/.pixi/bin:$PATH"

DATA=/orcd/data/ccoley/001/msms_data/nist23
OUT=data/nist23/subformulae/subform_50_repaired

# Rebuilds NIST23's subformula assignments the way canopus_train/csi2022 were
# originally processed: merge every spectrum's collision-energy blocks (dedup
# by rounded m/z, keep max intensity) BEFORE formula assignment, producing one
# assignment per spectrum. This replaces the current magma_subform_50.hdf5,
# whose per-collision-energy keys mean formula assignment ran independently
# per CE block, then got naively concatenated (no dedup) downstream in
# PeakFormula -- see NIST23_TRAINING_CHANGELOG.md bug #2 for why that's a
# materially different, less faithful pipeline than the original. ~56 min
# extrapolated from a 300-spectrum timing probe at --num-workers 8;
# budgeting more workers/time here for margin.
pixi run python -m mist.subformulae.assign_subformulae \
    --spec-files "$DATA/spec_files.hdf5" \
    --labels-file "$DATA/labels.tsv" \
    --output-dir "$OUT" \
    --num-workers 16 &

CHILD_PID=$!
wait $CHILD_PID
