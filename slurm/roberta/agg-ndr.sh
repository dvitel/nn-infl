#!/bin/bash -l
#SBATCH --job-name=r-agg
#SBATCH --time=96:00:00
#SBATCH --output 4-ndr-agg-%a.out
#SBATCH -D /blue/anshumanc.usf/nn-infl/roberta
#SBATCH -p hpg-b200
#SBATCH --gpus=1 # 1 GPU
#SBATCH --mem=8G # default 4GB
#SBATCH --array=0-7%3

module load conda
conda activate /blue/anshumanc.usf/nn-infl/nn-infl-env

tasks=("qnli" "mrpc" "sst2" "qqp" "cola" "mnli" "rte" "stsb")

task=${tasks[$SLURM_ARRAY_TASK_ID]}

task_cwd=/blue/anshumanc.usf/nn-infl/roberta

echo "NDR $task"
HF_TOKEN=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 INFL_SEED=0 INFL_CWD=$task_cwd python \
    /home/dvitel.usf/nn-infl/src/exp.py ndr --task=$task \
    --infl-methods=datainf,cos,hf,hf_we_,hf_we_topk_10 \
    --agg-methods=mean,mean-c,rank,rank-c,vote2,vote2-c,rmin,median,maj,cmean,mean_10,mean_50,commonset-10,commonset-30,commonset-40,commonset-80,commonsubset-10,commonsubset-30,commonsubset-40,commonsubset-80,commonsubset-40r,commonsubset-40rr \
    --m-prefix=m_bl --i-prefix=i_bl --ndr-prefix=ndr_agga \
    --levels 30 --hist-bins 0 \
    --group-file=./groups.json

echo "Done ndr $task"
echo "----------------------------------"