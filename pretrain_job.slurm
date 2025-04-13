#!/bin/bash

#SBATCH --job-name=hindu-pretrain  # Job name

#SBATCH --output=pretrain_%j.out   # Standard output log (%j expands to job ID)

#SBATCH --error=pretrain_%j.err    # Standard error log (%j expands to job ID)

#SBATCH --nodes=1                  # Number of nodes to use

#SBATCH --ntasks-per-node=1        # Number of tasks (usually 1 for single Python script)

#SBATCH --cpus-per-task=14         # 14 CPUs required per GPU on ai partition

#SBATCH --mem=32G                 # Memory per node (adjust if needed)

#SBATCH --time=01:50:00            # MAXIMUM time - adjusted for deadline! VERY UNLIKELY TO FINISH.

#SBATCH --partition=ai             # Run on GPU partition

#SBATCH --gres=gpu:1               # Request 1 GPU

#SBATCH --account=mlp              # Your account name



# --- End of Slurm Directives ---



echo "Starting job $SLURM_JOB_ID on $SLURM_NODELIST"

echo "Working directory: $(pwd)"



# Load the necessary modules in correct order

echo "Loading modules..."

module purge

module load gcc/14.1.0

module load python/3.11.9

module load cuda/12.6             # Or your chosen CUDA version



# Activate your Python virtual environment

echo "Activating environment..."

# Assuming venv is in the submission directory (e.g., ~/my_scripture_project/venv)

source venv/bin/activate



# Run the pre-training script

echo "Running pretrain.py..."

# Assuming pretrain.py is in the submission directory

python pretrain.py



echo "Job finished with exit code $?"
