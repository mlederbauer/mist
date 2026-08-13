#!/bin/bash
#SBATCH --job-name=nist23_fp_mist_mhplus_aux_gate
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

# First iteration of gated-residual reaction conditioning (--aux-gate,
# src/mist/models/mist_model.py): instead of concatenating a compressed
# projection of the aux fingerprint onto the pooled spectrum representation
# (--aux-dim), the aux source's OWN fingerprint is blended directly into the
# final prediction via a learned, per-bit, softmax-normalized gate. When a
# source is absent for an example, its gate weight is architecturally
# exactly 0 (verified: masked to -inf pre-softmax, not just multiplied by a
# near-zero learned weight) -- so this is a strict superset of plain MIST's
# behavior, never a dependency on aux data being present. See the reaction
# aux fingerprint correlation notebook (notebooks/reaction_aux_fp_correlation
# .ipynb) for why this is motivated: starting_materials/candidates already
# correlate with the true product fingerprint at ~4x random-pairing Tanimoto
# similarity, a signal plain concatenation makes the model re-derive
# indirectly through a lossy projection rather than exploiting directly.
#
# Scoped to [M+H]+ only (data/nist23/labels_mh_only.tsv) per the deliberate
# decision to develop/validate the reaction-conditioning architecture on the
# cleaner single-adduct subset first (closer to the original paper's
# training regime, less confounded by NIST23's adduct-diversity issue,
# which [M+H]+-only vs. full-mix experiments this session showed is a
# LARGER, separate driver of the Tanimoto gap than reaction conditioning).
# Generalizing back to the full adduct mix is a deliberately separate,
# later step -- this script and the --aux-gate architecture make no
# adduct-specific assumptions, so nothing here needs to change to do that.
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
        /home/magled/mist/data/nist23/reaction_metadata_cas.tsv \
        /home/magled/mist/data/nist23/reaction_metadata_pistachio.tsv \
    --wandb-project mist-nist23 \
    --save-dir results/nist23_fp_mist_mhplus_aux_gate/split_1 &

CHILD_PID=$!
wait $CHILD_PID
