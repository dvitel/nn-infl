#!/bin/bash -l
#SBATCH --job-name=r2-infl-hf
#SBATCH --time=72:00:00
#SBATCH --output r2-infl-hf-%j.out
#SBATCH -D /blue/anshumanc.usf/nn-infl
#SBATCH -p hpg-ai 
#SBATCH --open-mode=append
#SBATCH --gpus=1 # 1 GPU
#SBATCH --array=0-3

module load conda/24.7.1
conda activate /home/dvitel.usf/torch-env

tasks=("qnli")

task=${tasks[$SLURM_ARRAY_TASK_ID]}

task_cwd=/blue/anshumanc.usf/nn-infl/$task

method_name=hf
mem_koef=1.1

for run_id in {0..0}; do

    echo "Infl matrix $task $run_id $method_name"
    srun --export=ALL,INFL_SEED=$run_id,INFL_CWD=$task_cwd python /home/dvitel.usf/nn-infl/src/exp.py infl-matrix --task=$task --methods=$method_name --mem-koef=$mem_koef --ignore-metrics --i-prefix=i2 --m-prefix=m2
    echo "----- Done $task $run_id $method_name"

done