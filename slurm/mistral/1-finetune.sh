#!/bin/bash -l
#SBATCH --job-name=m-tune
#SBATCH --time=72:00:00
#SBATCH --output mistral-tune-%a.out
#SBATCH -D /blue/anshumanc.usf/nn-infl/mistral
#SBATCH -p hpg-ai # run on partition general
#SBATCH --open-mode=append
#SBATCH --gpus=1 # 1 GPU
#SBATCH --mem=16G # default 4GB
#SBATCH --array=0-3

module load conda/24.7.1
conda activate /home/dvitel.usf/torch-env

tasks=("qnli" "mrpc" "sst2" "qqp")

task=${tasks[$SLURM_ARRAY_TASK_ID]}

task_cwd=/blue/anshumanc.usf/nn-infl/mistral/$task

for run_id in {0..4}; do
    echo "Starting finetuning $task $run_id"
    HF_TOKEN=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 INFL_SEED=$run_id INFL_CWD=$task_cwd python \
        /home/dvitel.usf/nn-infl/src/exp.py finetune --task=$task --model=mistralai/Mistral-7B-v0.3 \
        --unfreeze-regex=.\*\\.embed_tokens\\..\* --lora-targets=q_proj,v_proj
    echo "Done finetuning $task $run_id"
done