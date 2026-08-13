#!/bin/bash
#SBATCH --job-name=nist23_fp_mist_mhplus_only
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

# Same as submit_nist23_fp_mist.sh (repaired subformula data), but restricted
# to [M+H]+ spectra only via data/nist23/labels_mh_only.tsv (80,372/176,851
# rows) -- the single adduct the original MIST paper trained on. Comparison
# point for whether NIST23's broader adduct mix (13 types, 45.5% [M+H]+)
# helps or hurts relative to the paper's narrower, single-adduct setup.
# split_1.tsv is left as the full split file; PresetSpectraSplitter only
# assigns folds to specs actually present in the loaded (filtered) dataset,
# so no separate filtered split file is needed.
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
    --wandb-project mist-nist23 \
    --save-dir results/nist23_fp_mist_mhplus_only/split_1 &

CHILD_PID=$!
wait $CHILD_PID
