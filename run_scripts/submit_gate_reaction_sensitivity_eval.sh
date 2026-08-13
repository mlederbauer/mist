#!/bin/bash
#SBATCH --job-name=gate_reaction_sensitivity_eval
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
REACTION_FILES="data/nist23/reaction_metadata_uspto.tsv data/nist23/reaction_metadata_cas.tsv data/nist23/reaction_metadata_pistachio.tsv"

# Real present-vs-absent comparison (per src/mist/analyze_reaction_sensitivity
# .py, generalized this session from >=2 to >=1 matched reaction): for every
# test compound with a real reaction match, run inference BOTH with that
# reaction steered in AND with no reaction, on the same compound -- this is
# the number that answers "does the gate actually benefit from a reaction
# prior when it has one," which the training logs never show (eval always
# defaults to no-aux unless explicitly steered like this).
run_analysis() {
    local ckpt_name=$1
    local aux_source=$2
    local save_dir="results/${ckpt_name}/split_1/split_1/reaction_sensitivity_${aux_source}"
    echo "################ $ckpt_name / $aux_source ################"
    pixi run python -m mist.analyze_reaction_sensitivity \
        --model-ckpt "results/${ckpt_name}/split_1/split_1/best.ckpt" \
        --labels-file data/nist23/labels_mh_only.tsv \
        --spec-folder "$DATA/spec_files.hdf5" \
        --subform-folder data/nist23/subformulae/subform_50_repaired \
        --split-file "$DATA/splits/split_1.tsv" \
        --reaction-metadata-file $REACTION_FILES \
        --aux-source "$aux_source" \
        --subset-datasets test_only \
        --save-dir "$save_dir"
    echo "--- summary for $ckpt_name / $aux_source ---"
    tail -5 "$save_dir/analyze_reaction_sensitivity.log"
    echo
}

(
    run_analysis nist23_fp_mist_mhplus_aux_gate starting_materials
    run_analysis nist23_fp_mist_mhplus_aux_gate candidates
    run_analysis nist23_fp_mist_mhplus_aux_gate_sm_only starting_materials
    run_analysis nist23_fp_mist_mhplus_aux_gate_cand_only candidates
    run_analysis nist23_fp_mist_mhplus_aux_gate_no_dropout starting_materials
    run_analysis nist23_fp_mist_mhplus_aux_gate_no_dropout candidates
) 2>&1 | tee results/gate_reaction_sensitivity_eval_output.log &

CHILD_PID=$!
wait $CHILD_PID
