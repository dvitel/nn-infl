#!/bin/bash -l
#SBATCH --job-name=infl-tune
#SBATCH --output i-t-%j.out
#SBATCH -D /data/dvitel/infl
#SBATCH -p Quick # run on partition general
#SBATCH --open-mode=append
#SBATCH --gpus=1 # 1 GPU
#SBATCH --exclude GPU41
#SBATCH --array=0-3

conda activate torch-env

tasks=("qnli" "mrpc" "sst2" "qqp")

task=${tasks[$SLURM_ARRAY_TASK_ID]}

task_cwd=/data/dvitel/infl/$task

for run_id in {0..10}; do
    echo "Starting finetuning $task $run_id"
    srun --export=ALL,INFL_SEED=$run_id,INFL_CWD=$task_cwd python ~/infl/src/exp.py finetune --task=$task --model=roberta-large --num-epochs=10
    echo "Done finetuning $task $run_id"
done