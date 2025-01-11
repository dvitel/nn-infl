#!/bin/bash -l
#SBATCH --job-name=infl-infl
#SBATCH --output i-i-%j.out
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
    echo "Starting influences $task $run_id"
    srun --export=ALL,INFL_SEED=$run_id,INFL_CWD=$task_cwd python ~/infl/src/exp.py infl --task=$task --compute-accurate --self-influence
    echo "Done influences $task $run_id"
done