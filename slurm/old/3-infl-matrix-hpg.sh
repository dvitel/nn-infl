#!/bin/bash -l
#SBATCH --job-name=r-infl-m
#SBATCH --time=72:00:00
#SBATCH --output r-infl-m-%j.out
#SBATCH -D /blue/anshumanc.usf/nn-infl
#SBATCH -p hpg-ai 
#SBATCH --open-mode=append
#SBATCH --gpus=1 # 1 GPU
#SBATCH --array=0-3

module load conda/24.7.1
conda activate /home/dvitel.usf/torch-env

tasks=("qnli" "mrpc" "sst2" "qqp")

methods=("hf" "hf_we_" "hf_we_topk_10" "cos" "cov" "datainf_one" "datainf")
mem_koefs=("1.1" "2" "2" "1.1" "2" "1.1" "2")

task=${tasks[$SLURM_ARRAY_TASK_ID]}

task_cwd=/data/dvitel/infl/$task

for run_id in {0..9}; do

    for ((i=0; i<${#methods[@]}; i++)); do
        method_name=${methods[$i]}
        mem_koef=${mem_koefs[$i]}
        echo "Infl matrix $task $run_id $method_name"
        srun --export=ALL,INFL_SEED=$run_id,INFL_CWD=$task_cwd python /home/dvitel.usf/nn-infl/src/exp.py infl-matrix --task=$task --methods=$method_name --mem-koef=$mem_koef
        echo "----- Done $task $run_id $method_name"
    done

done