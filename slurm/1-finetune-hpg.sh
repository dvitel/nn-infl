#!/bin/bash -l
#SBATCH --job-name=roberta-tune
#SBATCH --time=72:00:00
#SBATCH --output roberta-tune-%j.out
#SBATCH -D /blue/anshumanc.usf/nn-infl
#SBATCH -p hpg-ai # run on partition general
#SBATCH --open-mode=append
#SBATCH --gpus=1 # 1 GPU
#SBATCH --array=0-3

module load conda/24.7.1
conda activate /home/dvitel.usf/torch-env

tasks=("qnli" "mrpc" "sst2" "qqp")

task=${tasks[$SLURM_ARRAY_TASK_ID]}

task_cwd=/blue/anshumanc.usf/nn-infl/$task

for run_id in {0..9}; do
    echo "Starting finetuning $task $run_id"
    srun --export=ALL,INFL_SEED=$run_id,INFL_CWD=$task_cwd python /home/dvitel.usf/nn-infl/src/exp.py finetune --task=$task --model=roberta-large --num-epochs=10 --unfreeze-regex=.\*\\.word_embeddings\\..\*
    echo "Done finetuning $task $run_id"
done