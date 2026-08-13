#!/bin/bash
#SBATCH --job-name=realistic_mixed_aux_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue

cd /home/magled/mist
source "run_scripts/mail_notify.sh"
export PATH="$HOME/.pixi/bin:$PATH"

DATA=/orcd/data/ccoley/001/msms_data/nist23
REACTION_FILES="data/nist23/reaction_metadata_uspto.tsv data/nist23/reaction_metadata_cas.tsv data/nist23/reaction_metadata_pistachio.tsv"

# The population-weighted, realistic number: score each --aux-gate variant
# on the FULL test set using each compound's ACTUAL available aux data
# (real starting_materials/candidates where a match exists, absent
# otherwise) -- not one artificial condition forced on every compound like
# analyze_reaction_sensitivity.py's per-condition breakdown. This is what
# you'd actually see deploying the checkpoint against a real, mixed
# population where most compounds have no reaction match at all.
for name in \
    nist23_fp_mist_mhplus_only \
    nist23_fp_mist_mhplus_aux_gate \
    nist23_fp_mist_mhplus_aux_gate_sm_only \
    nist23_fp_mist_mhplus_aux_gate_cand_only \
    nist23_fp_mist_mhplus_aux_gate_no_dropout \
; do
    echo "################ $name ################"
    pixi run python analysis/eval_realistic_mixed_aux.py \
        --model-ckpt "results/$name/split_1/split_1/best.ckpt" \
        --labels-file data/nist23/labels_mh_only.tsv \
        --spec-folder "$DATA/spec_files.hdf5" \
        --subform-folder data/nist23/subformulae/subform_50_repaired \
        --split-file "$DATA/splits/split_1.tsv" \
        --reaction-metadata-file $REACTION_FILES
    echo
done 2>&1 | tee results/realistic_mixed_aux_eval_output.log &

CHILD_PID=$!
wait $CHILD_PID
