#!/bin/bash -l
#SBATCH --job-name=m-score
#SBATCH --time=96:00:00
#SBATCH --output 4-score-%a.out
#SBATCH -D /blue/anshumanc.usf/nn-infl/mistral
#SBATCH -p hpg-ai
#SBATCH --gpus=1 # 1 GPU
#SBATCH --mem=4G # default 4GB
#SBATCH --array=0-7

module load conda/24.7.1
conda activate /home/dvitel.usf/torch-env

tasks=("qnli" "mrpc" "sst2" "qqp" "cola" "mnli" "rte" "stsb")

task=${tasks[$SLURM_ARRAY_TASK_ID]}

task_cwd=/blue/anshumanc.usf/nn-infl/mistral/$task

for run_id in {0..4}; do
    echo "Scores $task $run_id"
    HF_TOKEN=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 INFL_SEED=$run_id INFL_CWD=$task_cwd python \
        /home/dvitel.usf/nn-infl/src/exp.py scores --task=$task \
        --infl-methods=datainf,cos,hf,hf_we_,hf_we_topk_10,rand,denoise \
        --agg-methods=mean,mean-c,rank,rank-c,vote,vote-c,rmin,rmin-c \
        --m-prefix=m_bl --i-prefix=i_bl --s-prefix=s_bl \
        --group-file=../groups.json

    cat ../groups.json
    echo "Done scores $task $run_id"
    echo "----------------------------------"
done