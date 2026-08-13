#!/bin/bash
#SBATCH --job-name=intermediate_gate_eval_cpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=mit_normal,mit_preemptable,pi_ccoley
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --requeue

cd /home/magled/mist
source "run_scripts/mail_notify.sh"
export PATH="$HOME/.pixi/bin:$PATH"

DATA=/orcd/data/ccoley/001/msms_data/nist23

# CPU-only twin of submit_intermediate_gate_eval.sh -- same threshold-sweep
# check, no --gpu flag (model.eval() on CPU). Submitted alongside the GPU
# version since mit_normal has idle CPU capacity and GPU jobs can sit
# pending on priority/resources for a while; whichever lands first gives
# the same result (deterministic eval on a fixed checkpoint), the other is
# redundant but harmless once one finishes.
pixi run python analysis/eval_checkpoint_threshold_sweep.py \
    --model-ckpt results/nist23_fp_mist_mhplus_aux_gate/split_1/split_1/last.ckpt \
    --labels-file data/nist23/labels_mh_only.tsv \
    --spec-folder "$DATA/spec_files.hdf5" \
    --subform-folder data/nist23/subformulae/subform_50_repaired \
    --split-file "$DATA/splits/split_1.tsv" \
    2>&1 | tee results/intermediate_gate_eval_cpu_output.log &

CHILD_PID=$!
wait $CHILD_PID
