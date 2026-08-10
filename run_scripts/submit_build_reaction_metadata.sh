#!/bin/bash
#SBATCH --job-name=build_reaction_metadata
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=mit_normal,mit_preemptable,pi_ccoley
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

cd /home/magled/mist
export PATH="$HOME/.pixi/bin:$PATH"

pixi run python -m mist.build_reaction_metadata \
    --reaction-source /orcd/data/ccoley/001/uspto_data/USPTO_FULL.csv \
    --source-name USPTO_FULL \
    --labels-file /orcd/data/ccoley/001/msms_data/nist26/labels.tsv \
    --out /home/magled/mist/data/nist26/reaction_metadata.tsv
