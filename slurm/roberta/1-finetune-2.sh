#!/bin/bash -l
#SBATCH --job-name=tun-2
#SBATCH --time=72:00:00
#SBATCH --output tun-2-%a.out
#SBATCH -D /blue/anshumanc.usf/nn-infl/roberta
#SBATCH -p hpg-ai # run on partition general
#SBATCH --gpus=1 # 1 GPU
#SBATCH --mem=8G # default 4GB
#SBATCH --array=0-4

module load conda/24.7.1
conda activate /home/dvitel.usf/torch-env

tasks=("cola" "mnli" "rte" "wnli" "stsb")
learning_rates=(1e-4 1e-4 3e-4 3e-4 1e-4)  # Define learning rates for each task
lr=${learning_rates[$SLURM_ARRAY_TASK_ID]}  # Get the learning rate for the current task

task=${tasks[$SLURM_ARRAY_TASK_ID]}

task_cwd=/blue/anshumanc.usf/nn-infl/roberta/$task

for run_id in {0..4}; do
    echo "Starting finetuning $task $run_id with learning rate $lr"
    INFL_SEED=$run_id INFL_CWD=$task_cwd python \
        /home/dvitel.usf/nn-infl/src/exp.py finetune --task=$task --model=roberta-large \
        --lr=$lr --unfreeze-regex=.\*\\.word_embeddings\\..\* --lora-targets=query,value --num-epochs=15
    echo "Done finetuning $task $run_id"
done