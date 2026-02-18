#!/bin/bash
# Example:
#   sbatch --array=1-100%20 slurm/de_array.sbatch

#SBATCH --job-name=de_array
#SBATCH --output=logs/de_array_%A_%a.out
#SBATCH --error=logs/de_array_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

# Adjust to your environment.
PROJECT_DIR="${PROJECT_DIR:-$HOME/Github/DifferentialEvolution}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_DIR/results}"

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$RESULTS_DIR"
cd "$PROJECT_DIR"

export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export SEED="${SLURM_ARRAY_TASK_ID}"
export RESULTS_DIR

# Optional runtime knobs:
# export OBJECTIVE=rastrigin
# export DIM=10
# export ALGORITHM=shade
# export MAXEVALS=200000
# export LOCAL_REFINE=true
# export LOCAL_METHOD=nelder_mead

julia --project=. scripts/run_de.jl

# After all array jobs finish, run once on a login node or a dependent job:
# julia --project=. scripts/summarize_runs.jl --results_dir "$RESULTS_DIR" --top_k 10
