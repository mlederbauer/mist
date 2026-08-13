#!/bin/bash
#SBATCH --job-name=nist23_hyperopt_mist_mhplus_repaired
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=mit_normal_gpu,mit_preemptable,pi_ccoley
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:l40s:3
#SBATCH --mem=192G
#SBATCH --time=48:00:00
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

# Rerun of submit_nist23_hyperopt_mist.sh, motivated by a confirmed
# overfitting diagnosis (train cosine sim ~0.79 vs. val ~0.66-0.72 on the
# plain [M+H]+-only repaired-data run) -- this sweep specifically targets
# regularization/capacity tradeoffs (weight_decay widened to include
# actually-meaningful strengths, batch_size newly added to the search
# space; see hyperopt_mist.py's get_param_space).
#
# Differences from the old run:
# - repaired subformula data (subform_50_repaired, not magma_subform_50
#   .hdf5) and [M+H]+-only labels (labels_mh_only.tsv) -- the CE-pooling
#   fix and adduct-mix control this session found, so hparams are tuned on
#   the data/task the rest of this session's work is standardized on.
# - NO --train-subsample-frac/--val-subsample-frac: the old run subsampled
#   to 10%/15% specifically because the pre-fix pipeline was too slow
#   (~25 min/epoch) to search over full data. Post-fix, full-dataset
#   epochs run in ~1 min, so subsampling is unnecessary AND risks tuning
#   hyperparameters (esp. capacity/regularization strength) that don't
#   transfer to the full-data regime -- dropout/weight-decay needs
#   genuinely can shift with dataset size.
# - --gres=gpu:l40s:3 (was 1 GPU with --max-concurrent 3, i.e. 3 trials
#   TIME-SLICING one GPU) + --max-concurrent 3 -- score_function in
#   hyperopt_mist.py already round-robins trial_number % num_cuda_devices,
#   so requesting 3 real GPUs makes those 3 concurrent trials actually
#   parallel instead of contending for one device.
pixi run python src/mist/hyperopt_mist.py \
    --cache-featurizers \
    --labels-file data/nist23/labels_mh_only.tsv \
    --spec-folder "$DATA/spec_files.hdf5" \
    --subform-folder data/nist23/subformulae/subform_50_repaired \
    --split-file "$DATA/splits/split_1.tsv" \
    --embed-instrument \
    --fp-names morgan4096 \
    --seed 1 \
    --gpus 1 \
    --batch-size 128 \
    --max-epochs 100 \
    --pairwise-featurization \
    --set-pooling cls \
    --cls-type ms1 \
    --patience 20 \
    --max-peaks 50 \
    --iterative-preds growing \
    --loss-fn cosine \
    --num-h-samples 30 \
    --max-concurrent 3 \
    --num-workers 8 \
    --augment-prob 0.5 \
    --inten-prob 0.12 \
    --remove-prob 0.5 \
    --remove-weights exp \
    --no-diffs \
    --tune-save \
    --save-dir results/nist23_hyperopt_mist_mhplus_repaired/ &

CHILD_PID=$!
wait $CHILD_PID
