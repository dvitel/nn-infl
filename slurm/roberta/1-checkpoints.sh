#!/bin/bash -l
#SBATCH --job-name=r-init
#SBATCH --time=72:00:00
#SBATCH --output 1-init-%a.out
#SBATCH -D /blue/anshumanc.usf/nn-infl/roberta
#SBATCH -p hpg-ai # run on partition general
#SBATCH --gpus=1 # 1 GPU
#SBATCH --array=0-8

module load conda/24.7.1
conda activate /home/dvitel.usf/torch-env

# one job per task 
tasks=("qnli" "mrpc" "sst2" "qqp" "cola" "mnli" "rte" "wnli" "stsb")

task=${tasks[$SLURM_ARRAY_TASK_ID]}

task_cwd=/blue/anshumanc.usf/nn-infl/roberta/$task

mkdir -p $task_cwd

for run_id in {0..4}; do
    echo "Init start checkpoint $task $run_id"
    HF_TOKEN=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 INFL_SEED=$run_id INFL_CWD=$task_cwd python \
        /home/dvitel.usf/nn-infl/src/exp.py init-checkpoint --task=$task --model=roberta-large \
        --unfreeze-regex=.*\\.word_embeddings\\..* --lora-targets=query,value
    echo "Done init checkpoint $task $run_id"
    echo "----------------------------------"
done