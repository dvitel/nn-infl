#!/bin/bash -l
#SBATCH --job-name=TUN2
#SBATCH --time=96:00:00
#SBATCH --output TUN2-%j.out
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

seeds=(117 191)

base_method_names=('rand' 'denoise')

for method_name in "${base_method_names[@]}"; do
    for seed2 in "${seeds[@]}"; do 
        for run_id in {0..4}; do
            echo "Starting finetune2 WE $task $run_id $seed2 $method_name"
            INFL_SEED=$run_id INFL_CWD=$task_cwd python /home/dvitel.usf/nn-infl/src/exp.py finetune2 --task=$task --filter-method=$method_name --unfreeze-regex=.\*\\.word_embeddings\\..\* --tag=$method_name --i-prefix=i_b --seed2=$seed2
            echo "Done finetune2 WE $task $run_id $seed2 $method_name"
        done    
    done
done