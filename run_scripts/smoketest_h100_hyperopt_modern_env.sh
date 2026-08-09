#!/bin/bash
#SBATCH --job-name=h100_hyperopt_smoketest
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

pixi run -e modern python src/mist/hyperopt_mist.py \
    --debug test \
    --labels-file "$DATA/labels.tsv" \
    --spec-folder "$DATA/spec_files.hdf5" \
    --subform-folder "$DATA/subformulae/magma_subform_50.hdf5" \
    --split-file "$DATA/splits/split_1.tsv" \
    --embed-instrument \
    --fp-names morgan4096 \
    --seed 1 \
    --gpus 1 \
    --batch-size 16 \
    --pairwise-featurization \
    --set-pooling cls \
    --cls-type ms1 \
    --patience 20 \
    --max-peaks 50 \
    --iterative-preds growing \
    --loss-fn cosine \
    --num-h-samples 2 \
    --max-concurrent 1 \
    --num-workers 2 \
    --no-diffs \
    --save-dir results/h100_hyperopt_smoketest/
