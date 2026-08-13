#!/bin/bash
#SBATCH --job-name=nist23_fp_mist_mhplus_aux_gate_no_dropout
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=mit_preemptable,mit_normal_gpu,pi_ccoley,ou_cheme
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

cd /home/magled/mist
source "run_scripts/mail_notify.sh"
export PATH="$HOME/.pixi/bin:$PATH"
export TORCH_CPP_LOG_LEVEL="ERROR"

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    for var in SLURM_STEP_GPUS SLURM_JOB_GPUS GPU_DEVICE_ORDINAL; do
        val=$(eval echo \$$var)
        if [ -n "$val" ]; then
            export CUDA_VISIBLE_DEVICES=$val
            break
        fi
    done
fi

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
nvidia-smi -L

handle_preemption() {
    echo "Received preemption signal (SIGUSR1) at $(date)"
    echo "Job will be terminated soon, saving checkpoint..."
    if [ ! -z "$CHILD_PID" ]; then
        kill -TERM $CHILD_PID
        wait $CHILD_PID
    fi
    exit 0
}
trap handle_preemption SIGUSR1

DATA=/orcd/data/ccoley/001/msms_data/nist23

# Hparam variant of submit_nist23_fp_mist_mhplus_aux_gate.sh (the "both
# sources" run): --aux-dropout 0.0 instead of the default 0.2. Tests
# whether randomly zeroing present aux data during training (so the model
# doesn't learn to depend on it always being there) helps or hurts the
# GATE specifically -- --aux-dropout was tuned/defaulted for the older
# concatenation-based (--aux-dim) conditioning path; the gate's presence
# flag already gives it an explicit, always-visible signal about whether
# aux data is real for this example, so it's not obvious the same dropout
# rate (or dropout at all) is still the right regularization here.
pixi run python src/mist/train_mist.py \
    --cache-featurizers \
    --labels-file data/nist23/labels_mh_only.tsv \
    --spec-folder "$DATA/spec_files.hdf5" \
    --subform-folder data/nist23/subformulae/subform_50_repaired \
    --split-file "$DATA/splits/split_1.tsv" \
    --embed-instrument \
    --fp-names morgan4096 \
    --num-workers 16 \
    --seed 1 \
    --gpus 1 \
    --augment-data \
    --batch-size 128 \
    --iterative-preds 'growing' \
    --iterative-loss-weight 0.4 \
    --learning-rate 0.00077 \
    --weight-decay 1e-07 \
    --lr-decay-frac 0.9 \
    --hidden-size 256 \
    --pairwise-featurization \
    --peak-attn-layers 2 \
    --refine-layers 4 \
    --spectra-dropout 0.1 \
    --max-peaks 50 \
    --magma-loss-lambda 8 \
    --magma-modulo 512 \
    --form-embedder 'pos-cos' \
    --no-diffs \
    --aux-gate \
    --aux-dropout 0.0 \
    --checkpoint-every-n-train-steps 500 \
    --reaction-metadata-file \
        /home/magled/mist/data/nist23/reaction_metadata_uspto.tsv \
        /home/magled/mist/data/nist23/reaction_metadata_cas.tsv \
        /home/magled/mist/data/nist23/reaction_metadata_pistachio.tsv \
    --wandb-project mist-nist23 \
    --save-dir results/nist23_fp_mist_mhplus_aux_gate_no_dropout/split_1 &

CHILD_PID=$!
wait $CHILD_PID
