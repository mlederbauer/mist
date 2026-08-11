#!/bin/bash
#SBATCH --job-name=nist23_hyperopt_mist
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=mit_preemptable,mit_normal_gpu,pi_ccoley
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

cd /home/magled/mist
source "$(dirname "$0")/mail_notify.sh"
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

# Signal handler for preemptable node termination -- avoids an abrupt kill
# so already-running trials get a chance to finish their current step and
# write their (--tune-save) checkpoint before the node is reclaimed. The
# STUDY (which trials have run, best-so-far) auto-resumes on --requeue since
# it's backed by a SQLite file at --save-dir/study.db (optuna.create_study
# with load_if_exists=True in base_hyperopt.py). Any ONE trial that was
# mid-epoch at the moment of preemption is not resumed mid-epoch -- it's
# left incomplete in the study and retried fresh on resume.
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

pixi run python src/mist/hyperopt_mist.py \
    --cache-featurizers \
    --labels-file "$DATA/labels.tsv" \
    --spec-folder "$DATA/spec_files.hdf5" \
    --subform-folder "$DATA/subformulae/magma_subform_50.hdf5" \
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
    --train-subsample-frac 0.1 \
    --val-subsample-frac 0.15 \
    --num-h-samples 30 \
    --max-concurrent 3 \
    --num-workers 5 \
    --augment-prob 0.5 \
    --inten-prob 0.12 \
    --remove-prob 0.5 \
    --remove-weights exp \
    --no-diffs \
    --tune-save \
    --save-dir results/nist23_hyperopt_mist/ &

CHILD_PID=$!
wait $CHILD_PID
