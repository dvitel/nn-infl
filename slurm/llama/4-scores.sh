#!/bin/bash -l
#SBATCH --job-name=l-score
#SBATCH --time=96:00:00
#SBATCH --output 4-score-%a.out
#SBATCH -D /blue/anshumanc.usf/nn-infl/llama
#SBATCH -p hpg-b200
#SBATCH --gpus=1 # 1 GPU
#SBATCH --mem=4G # default 4GB
#SBATCH --array=0-7%4

module load conda
conda activate /blue/anshumanc.usf/nn-infl/nn-infl-env

tasks=("qnli" "mrpc" "sst2" "qqp" "cola" "mnli" "rte" "stsb")

task=${tasks[$SLURM_ARRAY_TASK_ID]}

task_cwd=/blue/anshumanc.usf/nn-infl/llama/$task

for run_id in {0..9}; do
    echo "Scores $task $run_id"
    HF_TOKEN=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 INFL_SEED=$run_id INFL_CWD=$task_cwd python \
        /home/dvitel.usf/nn-infl/src/exp.py scores --task=$task \
        --infl-methods=hf,cos,datainf,outlier \
        --agg-methods=mean,rank-c,vote2-c,rand,denoise \
        --m-prefix=m_bl --i-prefix=i_bl --s-prefix=s_bl \
        --group-file=../groups.json

    cat ../groups.json
    echo "Done scores $task $run_id"
    echo "----------------------------------"
done