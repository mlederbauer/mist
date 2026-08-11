#!/bin/bash
#SBATCH --job-name=analyze_reaction_sensitivity_allrxn_candidates
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=mit_preemptable,mit_normal_gpu,pi_ccoley
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
# mail-type/mail-user removed: mail_notify.sh below emails log tails instead

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
nvidia-smi -L

DATA=/orcd/data/ccoley/001/msms_data/nist23

pixi run python -m mist.analyze_reaction_sensitivity \
    --model-ckpt results/nist23_fp_mist_aux32_allrxn/split_1/split_1/last.ckpt \
    --labels-file "$DATA/labels.tsv" \
    --spec-folder "$DATA/spec_files.hdf5" \
    --subform-folder "$DATA/subformulae/magma_subform_50.hdf5" \
    --split-file "$DATA/splits/split_1.tsv" \
    --reaction-metadata-file \
        /home/magled/mist/data/nist23/reaction_metadata_uspto.tsv \
        /home/magled/mist/data/nist23/reaction_metadata_cas.tsv \
        /home/magled/mist/data/nist23/reaction_metadata_pistachio.tsv \
    --aux-source candidates \
    --subset-datasets test_only \
    --num-workers 16 \
    --gpu \
    --save-dir results/nist23_fp_mist_aux32_allrxn/split_1/split_1/preds_candidates
