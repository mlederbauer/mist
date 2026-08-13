#!/bin/bash
#SBATCH --job-name=all_metadata_eval
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
REACTION_FILES="data/nist23/reaction_metadata_uspto.tsv data/nist23/reaction_metadata_cas.tsv data/nist23/reaction_metadata_pistachio.tsv data/nist23/reaction_metadata_suong.tsv"

# Threshold sweep + realistic-mixed-aux eval for the two new "all 4 reaction
# sources" runs (adds reaction_metadata_suong.tsv to the existing
# uspto/cas/pistachio aux_gate lineup):
#   - nist23_fp_mist_mhplus_aux_gate_all  ([M+H]+-only, labels_mh_only.tsv)
#   - nist23_fp_mist_embed_adduct_aux_gate_all (full 13-adduct mix, labels.tsv)
# Same two-stage eval as every other aux_gate checkpoint in
# notebooks/gate_results_summary.ipynb Sections 6a/6c.
run_threshold_sweep() {
    local name=$1
    local labels=$2
    echo "################ threshold sweep: $name ################"
    pixi run python analysis/eval_checkpoint_threshold_sweep.py \
        --model-ckpt "results/$name/split_1/split_1/best.ckpt" \
        --labels-file "$labels" \
        --spec-folder "$DATA/spec_files.hdf5" \
        --subform-folder data/nist23/subformulae/subform_50_repaired \
        --split-file "$DATA/splits/split_1.tsv"
    echo
}

run_realistic_mixed_aux() {
    local name=$1
    local labels=$2
    echo "################ realistic mixed-aux: $name ################"
    pixi run python analysis/eval_realistic_mixed_aux.py \
        --model-ckpt "results/$name/split_1/split_1/best.ckpt" \
        --labels-file "$labels" \
        --spec-folder "$DATA/spec_files.hdf5" \
        --subform-folder data/nist23/subformulae/subform_50_repaired \
        --split-file "$DATA/splits/split_1.tsv" \
        --reaction-metadata-file $REACTION_FILES
    echo
}

(
    run_threshold_sweep nist23_fp_mist_mhplus_aux_gate_all data/nist23/labels_mh_only.tsv
    run_threshold_sweep nist23_fp_mist_embed_adduct_aux_gate_all "$DATA/labels.tsv"
    run_realistic_mixed_aux nist23_fp_mist_mhplus_aux_gate_all data/nist23/labels_mh_only.tsv
    run_realistic_mixed_aux nist23_fp_mist_embed_adduct_aux_gate_all "$DATA/labels.tsv"
) 2>&1 | tee results/all_metadata_eval_output.log &

CHILD_PID=$!
wait $CHILD_PID
