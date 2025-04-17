#!/bin/bash -l
#SBATCH --job-name=r-tun
#SBATCH --time=72:00:00
#SBATCH --output 2-tun-%a.out
#SBATCH -D /blue/anshumanc.usf/nn-infl/roberta
#SBATCH -p hpg-ai # run on partition general
#SBATCH --gpus=1 # 1 GPU
#SBATCH --mem=16G # default 4GB
#SBATCH --array=0-7

module load conda/24.7.1
conda activate /home/dvitel.usf/torch-env

tasks=("qnli" "mrpc" "sst2" "qqp" "cola" "mnli" "rte" "stsb")
learning_rates=(3e-4 3e-4 3e-4 3e-4 3e-4 3e-4 3e-4 3e-4 3e-4)  # Define learning rates for each task

lr=${learning_rates[$SLURM_ARRAY_TASK_ID]}  # Get the learning rate for the current task

task=${tasks[$SLURM_ARRAY_TASK_ID]}

task_cwd=/blue/anshumanc.usf/nn-infl/roberta/$task

for run_id in {5..9}; do
    echo "Starting finetuning $task $run_id with learning rate $lr"
    HF_TOKEN=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 INFL_SEED=$run_id INFL_CWD=$task_cwd python \
        /home/dvitel.usf/nn-infl/src/exp.py finetune --task=$task --num-epochs=10 --lr=$lr
    echo "Done finetuning $task $run_id with learning rate $lr"
    echo "----------------------------------"
done