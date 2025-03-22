#!/bin/bash -l
#SBATCH --job-name=r-tune
#SBATCH --time=72:00:00
#SBATCH --output roberta-tune-%a.out
#SBATCH -D /blue/anshumanc.usf/nn-infl/roberta
#SBATCH -p hpg-ai # run on partition general
#SBATCH --open-mode=append
#SBATCH --gpus=1 # 1 GPU
#SBATCH --mem=8G # default 4GB
#SBATCH --array=0-3

module load conda/24.7.1
conda activate /home/dvitel.usf/torch-env

tasks=("qnli" "mrpc" "sst2" "qqp")

task=${tasks[$SLURM_ARRAY_TASK_ID]}

task_cwd=/blue/anshumanc.usf/nn-infl/roberta/$task

for run_id in {0..4}; do
    echo "Starting finetuning $task $run_id"
    INFL_SEED=$run_id INFL_CWD=$task_cwd python \
        /home/dvitel.usf/nn-infl/src/exp.py finetune --task=$task --model=roberta-large --num-epochs=15 --unfreeze-regex=.\*\\.word_embeddings\\..\*
    echo "Done finetuning $task $run_id"
done