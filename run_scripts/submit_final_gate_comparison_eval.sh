#!/bin/bash
#SBATCH --job-name=final_gate_comparison_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=mit_normal,mit_preemptable,pi_ccoley
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue

cd /home/magled/mist
source "run_scripts/mail_notify.sh"
export PATH="$HOME/.pixi/bin:$PATH"

DATA=/orcd/data/ccoley/001/msms_data/nist23

# Final comparison, threshold-corrected: all 4 finished --aux-gate variants
# plus the plain [M+H]+-only baseline (no aux at all), on their best.ckpt
# (not last.ckpt -- these runs have converged/early-stopped, best.ckpt is
# the actual model to compare, not whatever epoch happened to be last).
# Raw logged test_tanimoto (default 0.5 threshold) is NOT a fair comparison
# across these -- see analysis/eval_checkpoint_threshold_sweep.py docstring.
for name in \
    nist23_fp_mist_mhplus_only \
    nist23_fp_mist_mhplus_aux_gate \
    nist23_fp_mist_mhplus_aux_gate_sm_only \
    nist23_fp_mist_mhplus_aux_gate_cand_only \
    nist23_fp_mist_mhplus_aux_gate_no_dropout \
; do
    echo "################ $name ################"
    pixi run python analysis/eval_checkpoint_threshold_sweep.py \
        --model-ckpt "results/$name/split_1/split_1/best.ckpt" \
        --labels-file data/nist23/labels_mh_only.tsv \
        --spec-folder "$DATA/spec_files.hdf5" \
        --subform-folder data/nist23/subformulae/subform_50_repaired \
        --split-file "$DATA/splits/split_1.tsv"
    echo
done 2>&1 | tee results/final_gate_comparison_eval_output.log &

CHILD_PID=$!
wait $CHILD_PID
