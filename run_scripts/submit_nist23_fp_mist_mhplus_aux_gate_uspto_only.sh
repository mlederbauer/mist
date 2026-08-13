#!/bin/bash
#SBATCH --job-name=nist23_fp_mist_mhplus_aux_gate_uspto_only
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=mit_preemptable,mit_normal_gpu,pi_ccoley,ou_cheme
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=256G
#SBATCH --time=06:00:00
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

# Single-source ablation floor: reaction_metadata_uspto.tsv only (9,792
# compounds matched -- the smallest of the 3 real sources). Compare against
# submit_nist23_fp_mist_mhplus_aux_gate.sh (uspto+cas+pistachio) to isolate
# how much of that run's gain comes from uspto alone vs. the combined real
# sources, and against the suong-only runs to compare real vs. simulated
# single-source reaction conditioning at matched (single-source) coverage.
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
    --aux-dropout 0.2 \
    --checkpoint-every-n-train-steps 500 \
    --reaction-metadata-file \
        /home/magled/mist/data/nist23/reaction_metadata_uspto.tsv \
    --wandb-project mist-nist23 \
    --save-dir results/nist23_fp_mist_mhplus_aux_gate_uspto_only/split_1 &

CHILD_PID=$!
wait $CHILD_PID
