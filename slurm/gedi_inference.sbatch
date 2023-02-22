#!/usr/bin/env bash

#SBATCH --output=jobs/paragedi/inference/%J_slurm.out
#SBATCH --error=jobs/paragedi/inference/%J_slurm.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=martinig@kth.se
#SBATCH --constrain="rivendell"
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
#SBATCH -t 0-6:00  # time limit: (D-HH:MM) 


# Check job environment
echo "JOB:  ${SLURM_JOB_ID}"
echo "TASK: ${SLURM_ARRAY_TASK_ID}"
echo "HOST: $(hostname)"
echo ""
nvidia-smi

# Activate conda
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate detox


JOB_PATH=/Midgard/home/martinig/detox/jobs/paragedi/inference/${SLURM_JOB_ID}
DATA_PATH=/Midgard/home/martinig/detox/emnlp2021/data/test_10k_toxic

mkdir -p $JOB_PATH

export TRANSFORMERS_OFFLINE=1
python emnlp2021/style_transfer/paraGeDi/inference.py --data_dir $DATA_PATH --output_dir $JOB_PATH