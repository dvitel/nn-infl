#!/bin/bash -l
#SBATCH --job-name=l-pre
#SBATCH --time=72:00:00
#SBATCH --output 0-pre-%a.out
#SBATCH -D /blue/anshumanc.usf/nn-infl/llama
#SBATCH -p hpg-ai # run on partition general
#SBATCH --gpus=1 # 1 GPU
#SBATCH --array=0-7

module load conda/24.7.1
conda activate /home/dvitel.usf/torch-env

# one job per task 
tasks=("qnli" "mrpc" "sst2" "qqp" "cola" "mnli" "rte" "stsb")

task=${tasks[$SLURM_ARRAY_TASK_ID]}

task_cwd=/blue/anshumanc.usf/nn-infl/llama/$task

mkdir -p $task_cwd

for run_id in {0..4}; do
    echo "Starting preprocess $task $run_id"
    HF_TOKEN=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 INFL_SEED=$run_id INFL_CWD=$task_cwd python \
        /home/dvitel.usf/nn-infl/src/exp.py preprocess --task=$task --tokenizer-name=meta-llama/Llama-3.2-1B
    echo "Done preprocess $task $run_id"
    echo "----------------------------------"
done