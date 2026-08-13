#!/bin/bash
#SBATCH --job-name=nist23_fp_mist_mhplus_aux_gate_suong_only_cap10
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

# Ablation: can the aux-gate mechanism learn from SIMULATED reaction data
# alone? reaction_metadata_suong.tsv only (post-fix: 47,421 unique compounds,
# ~100 reactions/compound, synthetic/name-reaction-derived, not real USPTO/
# CAS/Pistachio literature reactions). --max-reactions-per-compound 10
# (default, same cap as every other aux_gate run) -- attach_reactions seeds a
# random 10-of-~100 subsample per compound, and SpectraMolDataset.__getitem__
# re-picks one of those 10 at random every forward pass, so training still
# sees rotating diversity within the cap. Compare against
# _suong_only_cap100.sh (no cap) to see if 10 is a meaningfully lossy subsample
# of suong's reaction diversity. [M+H]+-only, same hyperparameters as every
# other aux_gate variant. Eval MUST use only real reactions (uspto/cas/
# pistachio) -- never suong -- per the deliberate train/eval split.
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
    --max-reactions-per-compound 10 \
    --reaction-metadata-file \
        /home/magled/mist/data/nist23/reaction_metadata_suong.tsv \
    --wandb-project mist-nist23 \
    --save-dir results/nist23_fp_mist_mhplus_aux_gate_suong_only_cap10/split_1 &

CHILD_PID=$!
wait $CHILD_PID
