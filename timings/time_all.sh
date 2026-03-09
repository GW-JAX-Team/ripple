#!/bin/bash -l
#Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu_h100
#SBATCH -t 01:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=40G
#SBATCH --output="run_timings.out"
#SBATCH --job-name="ripple-timings"

now=$(date)
echo "$now"

# Loading modules
# module load 2024
# module load Python/3.10.4-GCCcore-11.3.0
# source /path/to/.venv/bin/activate

# Display GPU name
nvidia-smi --query-gpu=name --format=csv,noheader

DEVICE="gpu"
N_WAVEFORMS="10000"

PRECISIONS=("float32" "float64")
MODELS=("TaylorF2" "IMRPhenomD" "IMRPhenomXAS" "IMRPhenomPv2" "IMRPhenomXPHM")

for PRECISION in "${PRECISIONS[@]}"; do
    echo "=============================="
    echo "Running with precision = $PRECISION"
    echo "=============================="

    for MODEL in "${MODELS[@]}"; do
        echo "Running $MODEL with precision $PRECISION"
        ripple_time "$MODEL" \
            --device "$DEVICE" \
            --n-waveforms "$N_WAVEFORMS" \
            --precision "$PRECISION"
    done
done
