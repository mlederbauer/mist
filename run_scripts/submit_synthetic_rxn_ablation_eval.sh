#!/bin/bash
#SBATCH --job-name=synthetic_rxn_ablation_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=mit_normal,mit_preemptable,pi_ccoley
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --requeue

cd /home/magled/mist
source "run_scripts/mail_notify.sh"
export PATH="$HOME/.pixi/bin:$PATH"

DATA=/orcd/data/ccoley/001/msms_data/nist23
LABELS=data/nist23/labels_mh_only.tsv
SUBFORM=data/nist23/subformulae/subform_50_repaired
SPLIT="$DATA/splits/split_1.tsv"

# Eval MUST use only REAL experimental reactions (uspto/cas/pistachio) --
# never suong -- regardless of what a given checkpoint trained with. suong
# is simulated/name-reaction-derived data; the test question is "does
# training on it transfer to real reaction-steered predictions," which
# would be circular if suong reactions were also used to steer eval.
REAL_REACTION_FILES="data/nist23/reaction_metadata_uspto.tsv data/nist23/reaction_metadata_cas.tsv data/nist23/reaction_metadata_pistachio.tsv"

# Synthetic-vs-real reaction data ablation (all [M+H]+-only, aux-gate, same
# base hyperparameters): can the gate learn a useful reaction prior from
# reaction_metadata_suong.tsv's simulated/name-reaction data, transferable to
# real reaction-steered eval? Six arms:
#   baseline            -- nist23_fp_mist_mhplus_only          (no aux at all)
#   real_only           -- nist23_fp_mist_mhplus_aux_gate       (uspto+cas+pistachio)
#   uspto_only          -- nist23_fp_mist_mhplus_aux_gate_uspto_only
#   suong_only_cap10    -- nist23_fp_mist_mhplus_aux_gate_suong_only_cap10
#   suong_only_cap100   -- nist23_fp_mist_mhplus_aux_gate_suong_only_cap100
#   all_sources         -- nist23_fp_mist_mhplus_aux_gate_all   (uspto+cas+pistachio+suong)
# Three eval angles per checkpoint:
#   (a) cold-start threshold sweep (eval_checkpoint_threshold_sweep.py) --
#       no reaction data at all, comparable to the vanilla-MIST baseline.
#   (b) reaction-sensitivity (analyze_reaction_sensitivity.py) -- for every
#       REAL-matched test compound, tanimoto WITH that real reaction steered
#       in vs WITHOUT (same compound/spectrum/target) -- isolates whether
#       the checkpoint's gate actually uses a real reaction prior when given
#       one, regardless of what it trained on.
#   (c) realistic mixed-aux (eval_realistic_mixed_aux.py) -- population-
#       weighted whole-test-set number using each compound's actual REAL
#       aux data where matched, absent otherwise.
run_threshold_sweep() {
    local name=$1
    echo "################ (a) cold-start threshold sweep: $name ################"
    pixi run python analysis/eval_checkpoint_threshold_sweep.py \
        --model-ckpt "results/$name/split_1/split_1/best.ckpt" \
        --labels-file "$LABELS" \
        --spec-folder "$DATA/spec_files.hdf5" \
        --subform-folder "$SUBFORM" \
        --split-file "$SPLIT"
    echo
}

run_reaction_sensitivity() {
    local name=$1
    local save_dir="results/${name}/split_1/split_1/reaction_sensitivity_real"
    echo "################ (b) reaction sensitivity (real rxns only): $name ################"
    pixi run python -m mist.analyze_reaction_sensitivity \
        --model-ckpt "results/$name/split_1/split_1/best.ckpt" \
        --labels-file "$LABELS" \
        --spec-folder "$DATA/spec_files.hdf5" \
        --subform-folder "$SUBFORM" \
        --split-file "$SPLIT" \
        --reaction-metadata-file $REAL_REACTION_FILES \
        --aux-source starting_materials \
        --subset-datasets test_only \
        --save-dir "$save_dir"
    echo "--- summary for $name ---"
    tail -5 "$save_dir/analyze_reaction_sensitivity.log"
    echo
}

run_realistic_mixed_aux() {
    local name=$1
    echo "################ (c) realistic mixed-aux (real rxns only): $name ################"
    pixi run python analysis/eval_realistic_mixed_aux.py \
        --model-ckpt "results/$name/split_1/split_1/best.ckpt" \
        --labels-file "$LABELS" \
        --spec-folder "$DATA/spec_files.hdf5" \
        --subform-folder "$SUBFORM" \
        --split-file "$SPLIT" \
        --reaction-metadata-file $REAL_REACTION_FILES
    echo
}

(
    for name in \
        nist23_fp_mist_mhplus_only \
        nist23_fp_mist_mhplus_aux_gate \
        nist23_fp_mist_mhplus_aux_gate_uspto_only \
        nist23_fp_mist_mhplus_aux_gate_suong_only_cap10 \
        nist23_fp_mist_mhplus_aux_gate_suong_only_cap100 \
        nist23_fp_mist_mhplus_aux_gate_all \
    ; do
        run_threshold_sweep "$name"
    done
    # Reaction-sensitivity/realistic-mixed-aux need real aux_data attached --
    # meaningless for the no-aux baseline (nist23_fp_mist_mhplus_only has no
    # aux_projections at all), so skip it for those two eval angles.
    for name in \
        nist23_fp_mist_mhplus_aux_gate \
        nist23_fp_mist_mhplus_aux_gate_uspto_only \
        nist23_fp_mist_mhplus_aux_gate_suong_only_cap10 \
        nist23_fp_mist_mhplus_aux_gate_suong_only_cap100 \
        nist23_fp_mist_mhplus_aux_gate_all \
    ; do
        run_reaction_sensitivity "$name"
        run_realistic_mixed_aux "$name"
    done
) 2>&1 | tee results/synthetic_rxn_ablation_eval_output.log &

CHILD_PID=$!
wait $CHILD_PID
