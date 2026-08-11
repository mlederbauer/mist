#!/bin/bash
#SBATCH --job-name=eval_test_tanimoto
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=mit_normal,mit_preemptable,pi_ccoley
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00

cd /home/magled/mist
export PATH="$HOME/.pixi/bin:$PATH"

CKPT=results/nist23_fp_mist/split_1/split_1/best.ckpt
SAVE_DIR="$(dirname "$CKPT")/preds"

for FOLD in test val train; do
    pixi run python -m mist.pred_fp \
        --model-ckpt "$CKPT" \
        --save-dir "$SAVE_DIR" \
        --dataset-name "$FOLD" \
        --subset-datasets "${FOLD}_only" \
        --output-targs \
        --labels-file /orcd/data/ccoley/001/msms_data/nist23/labels.tsv \
        --spec-folder /orcd/data/ccoley/001/msms_data/nist23/spec_files.hdf5 \
        --subform-folder /orcd/data/ccoley/001/msms_data/nist23/subformulae/magma_subform_50.hdf5

    pixi run python run_scripts/tanimoto_from_preds.py "$SAVE_DIR" "$FOLD=$SAVE_DIR/fp_preds_$FOLD.p"
done
