#!/bin/bash
#SBATCH --job-name=intermediate_gate_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=mit_preemptable,mit_normal_gpu,pi_ccoley,ou_cheme
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --requeue

cd /home/magled/mist
source "run_scripts/mail_notify.sh"
export PATH="$HOME/.pixi/bin:$PATH"

DATA=/orcd/data/ccoley/001/msms_data/nist23

# Intermediate check on the in-progress nist23_fp_mist_mhplus_aux_gate run
# (job 20238793): re-scores its current last.ckpt on the real val+test
# splits with a swept binarization threshold, instead of trusting the
# training-time logged val_tanimoto (hardcoded 0.5 threshold, which the
# --aux-gate softmax blend drives well below its optimal range -- see
# analysis/eval_checkpoint_threshold_sweep.py docstring; cosine similarity
# is a more reliable signal while this is unresolved).
pixi run python analysis/eval_checkpoint_threshold_sweep.py \
    --model-ckpt results/nist23_fp_mist_mhplus_aux_gate/split_1/split_1/last.ckpt \
    --labels-file data/nist23/labels_mh_only.tsv \
    --spec-folder "$DATA/spec_files.hdf5" \
    --subform-folder data/nist23/subformulae/subform_50_repaired \
    --split-file "$DATA/splits/split_1.tsv" \
    --gpu \
    2>&1 | tee results/intermediate_gate_eval_output.log &

CHILD_PID=$!
wait $CHILD_PID
