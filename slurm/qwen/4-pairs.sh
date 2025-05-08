#!/bin/bash -l
#SBATCH --job-name=q-pair
#SBATCH --time=96:00:00
#SBATCH --output 4-pair-%a.out
#SBATCH -D /blue/anshumanc.usf/nn-infl/qwen
#SBATCH -p hpg-ai
#SBATCH --gpus=1 # 1 GPU
#SBATCH --mem=16G # default 4GB
#SBATCH --array=0-7%1

module load conda/24.7.1
conda activate /home/dvitel.usf/torch-env

tasks=("qnli" "mrpc" "sst2" "qqp" "cola" "mnli" "rte" "stsb")

task=${tasks[$SLURM_ARRAY_TASK_ID]}

task_cwd=/blue/anshumanc.usf/nn-infl/qwen/$task
run_id=0

echo "INFL PAIRS $task $run_id"
HF_TOKEN=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 INFL_SEED=$run_id INFL_CWD=$task_cwd python \
    /home/dvitel.usf/nn-infl/src/exp.py infl-noise --task=$task \
    --infl-methods=datainf,cos,hf,hf_we_,hf_we_topk_10 \
    --m-prefix=m_bl --i-prefix=i_bl --topk=5

echo "Done infl pairs $task $run_id"
echo "----------------------------------"