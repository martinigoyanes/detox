#!/usr/bin/env bash

#SBATCH --output=jobs/paragedi/%J_slurm.out
#SBATCH --error=jobs/paragedi/%J_slurm.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=martinig@kth.se
#SBATCH --constrain="rivendell"
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=10GB
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


JOB_PATH=/Midgard/home/martinig/detox/jobs/paragedi/${SLURM_JOB_ID}
DATA_PATH=/Midgard/home/martinig/detox/emnlp2021/data

N_EPOCHS=2 # default
TOKENIZER_NAME=ceshine/t5-paraphrase-paws-msrp-opinosis # default
MAX_SEQUENCE_LEN=128 # default
BATCH_SIZE=4 # default=8
LOGGING_STEPS=100
SAVING_STEPS=2000

mkdir -p "${JOB_PATH}/model"
mkdir -p "${JOB_PATH}/logs"

export TRANSFORMERS_OFFLINE=1
python emnlp2021/style_transfer/paraGeDi/gedi_training/gedi_training.py \
--tokenizer_name $TOKENIZER_NAME \
--data_dir $DATA_PATH \
--working_dir "${JOB_PATH}/model" \
--log_dir "${JOB_PATH}/logs" \
--n_epochs $N_EPOCHS \
--max_seq_length $MAX_SEQUENCE_LEN \
--train_batch_size $BATCH_SIZE \
--eval_batch_size $BATCH_SIZE \
--logging_steps $LOGGING_STEPS \
--saving_steps $SAVING_STEPS \
--tensorboard