#!/bin/bash -l
#SBATCH --job-name=r-i-cos
#SBATCH --time=72:00:00
#SBATCH --output i-cos-%a.out
#SBATCH -D /blue/anshumanc.usf/nn-infl/roberta
#SBATCH -p hpg-ai
#SBATCH --open-mode=append
#SBATCH --gpus=1 # 1 GPU
#SBATCH --mem=4G
#SBATCH --array=0-3

module load conda/24.7.1
conda activate /home/dvitel.usf/torch-env

tasks=("qnli" "mrpc" "sst2" "qqp")

task=${tasks[$SLURM_ARRAY_TASK_ID]}

task_cwd=/blue/anshumanc.usf/nn-infl/roberta/$task

method_name=cos
mem_koef=1.1

for run_id in {0..4}; do

    echo "Infl matrix $task $run_id $method_name"
    HF_TOKEN=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 INFL_SEED=$run_id INFL_CWD=$task_cwd python \
        /home/dvitel.usf/nn-infl/src/exp.py infl-matrix --task=$task --methods=$method_name --mem-koef=$mem_koef --m-prefix=m_l --i-prefix=i_l
    echo "----- Done $task $run_id $method_name"

done