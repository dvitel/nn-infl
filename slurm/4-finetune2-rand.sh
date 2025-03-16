#!/bin/bash -l
#SBATCH --job-name=tun2-hf
#SBATCH --time=72:00:00
#SBATCH --output tun2-hf-%j.out
#SBATCH -D /blue/anshumanc.usf/nn-infl
#SBATCH -p hpg-ai
#SBATCH --open-mode=append
#SBATCH --gpus=1 # 1 GPU
#SBATCH --array=0-3

module load conda/24.7.1
conda activate /home/dvitel.usf/torch-env

tasks=("qnli" "mrpc" "sst2" "qqp")

task=${tasks[$SLURM_ARRAY_TASK_ID]}

task_cwd=/blue/anshumanc.usf/nn-infl/$task

for run_id in {0..4}; do
    echo "Starting finetune2 WE $task $run_id"
    INFL_SEED=$run_id INFL_CWD=$task_cwd python /home/dvitel.usf/nn-infl/src/exp.py finetune2 --task=$task --filter-method=rand --unfreeze-regex=.\*\\.word_embeddings\\..\* --tag='rand'
    echo "Done finetune2 WE $task $run_id"
done