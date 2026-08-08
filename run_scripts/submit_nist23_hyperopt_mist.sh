#!/bin/bash
#SBATCH --job-name=nist23_hyperopt_mist
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=mit_preemptable,mit_normal_gpu,pi_ccoley
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=magled@mit.edu

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

# Signal handler for preemptable node termination. Ray Tune itself receives
# and handles the termination; this just avoids an abrupt kill so the
# already-running trials get a chance to finish their current step and
# write their (--tune-save) checkpoint before the node is reclaimed. The
# STUDY (which trials have run, best-so-far) auto-resumes on --requeue via
# base_hyperopt.py's fixed experiment_dir + Tuner.restore. Any ONE trial
# that was mid-epoch at the moment of preemption is not resumed mid-epoch
# -- Ray marks it errored and it is retried fresh on resume.
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
    --max-epochs 600 \
    --pairwise-featurization \
    --set-pooling cls \
    --cls-type ms1 \
    --patience 20 \
    --max-peaks 50 \
    --iterative-preds growing \
    --loss-fn cosine \
    --train-subsample-frac 0.1 \
    --cpus-per-trial 5 \
    --gpus-per-trial 0.33 \
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
