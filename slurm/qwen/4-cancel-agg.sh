#!/bin/bash -l
#SBATCH --job-name=q-cagg
#SBATCH --time=96:00:00
#SBATCH --output 4-cagg-%a.out
#SBATCH -D /blue/anshumanc.usf/nn-infl/qwen
#SBATCH -p hpg-ai
#SBATCH --gpus=1 # 1 GPU
#SBATCH --mem=16G # default 4GB
#SBATCH --array=0-0

module load conda/24.7.1
conda activate /home/dvitel.usf/torch-env

tasks="qnli,mrpc,sst2,qqp,cola,mnli,rte,stsb"

task_cwd=/blue/anshumanc.usf/nn-infl/qwen

run_ids="0,1,2,3,4,5,6,7,8,9"

echo "Cancel agg $tasks $run_ids"
HF_TOKEN=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 INFL_SEED=0 INFL_CWD=$task_cwd python \
    /home/dvitel.usf/nn-infl/src/exp.py combine-cancel --tasks=$tasks --m-prefix=m_bl \
    --run-ids=$run_ids --group-file=./groups.json

echo "Cancel agg $tasks $run_ids"
echo "----------------------------------"