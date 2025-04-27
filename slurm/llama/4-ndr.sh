#!/bin/bash -l
#SBATCH --job-name=l-ndr
#SBATCH --time=96:00:00
#SBATCH --output 4-ndr-%a.out
#SBATCH -D /blue/anshumanc.usf/nn-infl/llama
#SBATCH -p hpg-ai
#SBATCH --gpus=1 # 1 GPU
#SBATCH --mem=16G # default 4GB
#SBATCH --array=0-7

module load conda/24.7.1
conda activate /home/dvitel.usf/torch-env

tasks=("qnli" "mrpc" "sst2" "qqp" "cola" "mnli" "rte" "stsb")

task=${tasks[$SLURM_ARRAY_TASK_ID]}

task_cwd=/blue/anshumanc.usf/nn-infl/llama

echo "NDR $task"
HF_TOKEN=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 INFL_SEED=0 INFL_CWD=$task_cwd python \
    /home/dvitel.usf/nn-infl/src/exp.py ndr --task=$task \
    --infl-methods=datainf,cos,hf,hf_we_,hf_we_topk_10 \
    --agg-methods=mean,mean-c,rank,rank-c,cset,cset-c,vote2,vote2-c \
    --m-prefix=m_bl --i-prefix=i_bl --ndr-prefix=ndr_bl \
    --group-file=./groups.json

echo "Done ndr $task"
echo "----------------------------------"