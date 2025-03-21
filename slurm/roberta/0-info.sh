#!/bin/bash -l
#SBATCH --job-name=0-info
#SBATCH --time=72:00:00
#SBATCH --output 0-info-%j.out
#SBATCH -D /blue/anshumanc.usf/nn-infl/roberta
#SBATCH -p hpg-ai # run on partition general
#SBATCH --open-mode=append
#SBATCH --gpus=1 # 1 GPU
#SBATCH --array=0-0

module load conda/24.7.1
conda activate /home/dvitel.usf/torch-env

# one job per task 
# tasks=("qnli" "mrpc" "sst2" "qqp")

# task=${tasks[$SLURM_ARRAY_TASK_ID]}

task_cwd=/blue/anshumanc.usf/nn-infl/roberta/

mkdir -p $task_cwd

echo "Model info"
HF_TOKEN=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 INFL_CWD=$task_cwd python \
    /home/dvitel.usf/nn-infl/src/exp.py info --model=roberta-large --unfreeze-regex=.*\\.embed_tokens\\..* --lora-targets=v_proj --out-file=model-info.txt
echo "Done Model info"