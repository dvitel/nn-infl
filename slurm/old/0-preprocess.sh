#!/bin/bash -l
#SBATCH --job-name=infl-prep
#SBATCH --output i-p-%j.out
#SBATCH -D /data/dvitel/infl
#SBATCH -p Quick # run on partition general
#SBATCH --open-mode=append
#SBATCH --gpus=1 # 1 GPU
#SBATCH --exclude GPU49,GPU50,GPU51,GPU52
#SBATCH --array=0-3

conda activate torch-env

tasks=("qnli" "mrpc" "sst2" "qqp")

task=${tasks[$SLURM_ARRAY_TASK_ID]}

task_cwd=/data/dvitel/infl/$task

mkdir -p $task_cwd

for run_id in {0..10}; do
    echo "Starting preprocess $task $run_id"
    srun --export=ALL,INFL_SEED=$run_id,INFL_CWD=$task_cwd python ~/infl/src/exp.py preprocess --task=$task --tokenizer-name=roberta-large
    echo "Done preprocess $task $run_id"
done