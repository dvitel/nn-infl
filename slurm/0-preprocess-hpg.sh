#!/bin/bash -l
#SBATCH --job-name=glue-noise-dataset
#SBATCH --output glue-noise-dataset-%j.out
#SBATCH -D /blue/anshumanc.usf/nn-infl
#SBATCH -p hpg-ai # run on partition general
#SBATCH --open-mode=append
#SBATCH --gpus=1 # 1 GPU
#SBATCH --array=0-3

module load conda/24.7.1
conda activate /home/dvitel.usf/torch-env

# one job per task 
tasks=("qnli" "mrpc" "sst2" "qqp")

task=${tasks[$SLURM_ARRAY_TASK_ID]}

task_cwd=/blue/anshumanc.usf/nn-infl/$task

mkdir -p $task_cwd

for run_id in {0..9}; do
    echo "Starting preprocess $task $run_id"
    srun --export=ALL,INFL_SEED=$run_id,INFL_CWD=$task_cwd python /home/dvitel.usf/nn-infl/src/exp.py preprocess --task=$task --tokenizer-name=roberta-large
    echo "Done preprocess $task $run_id"
done