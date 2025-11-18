#!/bin/bash -l
#SBATCH --job-name=q-init
#SBATCH --time=72:00:00
#SBATCH --output 1-init-%a.out
#SBATCH -D /blue/anshumanc.usf/nn-infl/qwen
#SBATCH -p hpg-b200 # run on partition general
#SBATCH --gpus=1 # 1 GPU
#SBATCH --array=0-7

module load conda
conda activate /blue/anshumanc.usf/nn-infl/nn-infl-env

# one job per task 
tasks=("qnli" "mrpc" "sst2" "qqp" "cola" "mnli" "rte" "stsb")

task=${tasks[$SLURM_ARRAY_TASK_ID]}

task_cwd=/blue/anshumanc.usf/nn-infl/qwen/$task

mkdir -p $task_cwd

for run_id in {0..9}; do
    echo "Init start checkpoint $task $run_id"
    HF_TOKEN=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 INFL_SEED=$run_id INFL_CWD=$task_cwd python \
        /home/dvitel.usf/nn-infl/src/exp.py init-checkpoint --task=$task --model=Qwen/Qwen2.5-1.5B \
        --unfreeze-regex=.\*\\.embed_tokens\\..\* --lora-targets=q_proj,v_proj
    echo "Done init checkpoint $task $run_id"
    echo "----------------------------------"
done