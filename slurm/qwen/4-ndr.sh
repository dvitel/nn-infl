#!/bin/bash -l
#SBATCH --job-name=q-ndr
#SBATCH --time=96:00:00
#SBATCH --output 4-ndr-%a.out
#SBATCH -D /blue/anshumanc.usf/nn-infl/qwen
#SBATCH -p hpg-b200
#SBATCH --gpus=1 # 1 GPU
#SBATCH --mem=16G # default 4GB
#SBATCH --array=0-7

module load conda
conda activate /blue/anshumanc.usf/nn-infl/nn-infl-env

tasks=("qnli" "mrpc" "sst2" "qqp" "cola" "mnli" "rte" "stsb")

task=${tasks[$SLURM_ARRAY_TASK_ID]}

task_cwd=/blue/anshumanc.usf/nn-infl/qwen

echo "NDR $task"
HF_TOKEN=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 INFL_SEED=0 INFL_CWD=$task_cwd python \
    /home/dvitel.usf/nn-infl/src/exp.py ndr --task=$task \
    --infl-methods=datainf,cos,hf,hf_we_,hf_we_topk_10 \
    --agg-methods=mean,mean-c,rank,rank-c,vote2,vote2-c \
    --m-prefix=m_bl --i-prefix=i_bl --ndr-prefix=ndr_bl \
    --group-file=./groups.json

echo "Done ndr $task"
echo "----------------------------------"