#!/bin/bash -l
#SBATCH --job-name=q-vk-score
#SBATCH --time=96:00:00
#SBATCH --output 4-score-vk-%a.out
#SBATCH -D /blue/anshumanc.usf/nn-infl/qwen
#SBATCH -p hpg-b200
#SBATCH --gpus=1 # 1 GPU
#SBATCH --mem=4G # default 4GB
#SBATCH --array=0-7%4

module load conda
conda activate /home/dvitel.usf/torch-env

tasks=("qnli" "mrpc" "sst2" "qqp" "cola" "mnli" "rte" "stsb")

task=${tasks[$SLURM_ARRAY_TASK_ID]}

task_cwd=/blue/anshumanc.usf/nn-infl/qwen/$task

for run_id in {0..9}; do
    echo "Scores $task $run_id"
    HF_TOKEN=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 INFL_SEED=$run_id INFL_CWD=$task_cwd python \
        /home/dvitel.usf/nn-infl/src/exp.py scores --task=$task \
        --infl-methods=datainf,cos,hf,hf_we_,hf_we_topk_10,rand,denoise \
        --agg-methods=vote2-c-10,vote2-c-20,vote2-c-30,vote2-c-40,vote2-c-50,vote2-c-60,vote2-c-70,vote2-c-80,vote2-c-90,vote2-c-100 \
        --m-prefix=m_bl --i-prefix=i_bl --s-prefix=s_vote_k \
        --group-file=../groups.json

    echo "Done scores $task $run_id"
    echo "----------------------------------"
done