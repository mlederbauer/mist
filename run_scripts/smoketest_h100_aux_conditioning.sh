#!/bin/bash
#SBATCH --job-name=h100_aux_smoketest
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=mit_preemptable,mit_normal_gpu,pi_ccoley
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00

cd /home/magled/mist
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
nvidia-smi -L

DATA=/orcd/data/ccoley/001/msms_data/nist23
AUX_LABELS=/home/magled/mist/results/labels_aux_test.tsv

COMMON_ARGS=(
    --debug test
    --spec-folder "$DATA/spec_files.hdf5"
    --subform-folder "$DATA/subformulae/magma_subform_50.hdf5"
    --split-file "$DATA/splits/split_1.tsv"
    --embed-instrument
    --fp-names morgan4096
    --num-workers 4
    --seed 1
    --gpus 1
    --batch-size 16
    --iterative-preds growing
    --iterative-loss-weight 0.4
    --learning-rate 0.00077
    --hidden-size 256
    --pairwise-featurization
    --peak-attn-layers 2
    --refine-layers 4
    --max-peaks 50
    --magma-loss-lambda 8
    --magma-modulo 512
    --form-embedder pos-cos
    --no-diffs
)

echo "=== Regression check: --aux-dim 0 (default), real labels.tsv ==="
pixi run python src/mist/train_mist.py \
    "${COMMON_ARGS[@]}" \
    --labels-file "$DATA/labels.tsv" \
    --save-dir results/h100_aux_smoketest/no_aux

echo "=== New path: --aux-dim 16, synthetic labels with related_structures ==="
pixi run python src/mist/train_mist.py \
    "${COMMON_ARGS[@]}" \
    --labels-file "$AUX_LABELS" \
    --aux-dim 16 \
    --aux-dropout 0.2 \
    --save-dir results/h100_aux_smoketest/with_aux
