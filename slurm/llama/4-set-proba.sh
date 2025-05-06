#!/bin/bash -l
#SBATCH --job-name=l-proba
#SBATCH --time=96:00:00
#SBATCH --output 4-proba-%a.out
#SBATCH -D /blue/anshumanc.usf/nn-infl/llama
#SBATCH -p hpg-ai
#SBATCH --gpus=1 # 1 GPU
#SBATCH --mem=16G # default 4GB
#SBATCH --array=0-7%1

module load conda/24.7.1
conda activate /home/dvitel.usf/torch-env

tasks=("qnli" "mrpc" "sst2" "qqp" "cola" "mnli" "rte" "stsb")

task=${tasks[$SLURM_ARRAY_TASK_ID]}

task_cwd=/blue/anshumanc.usf/nn-infl/llama/$task

for run_id in {0..9}; do
    echo "Calc checkpoint train logits $task $run_id"
    HF_TOKEN=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 INFL_SEED=$run_id INFL_CWD=$task_cwd python \
        /home/dvitel.usf/nn-infl/src/exp.py set-logits --task=$task --m-prefix=m_bl \
        --batch-size=64 --set-name=test --softmaxed

    echo "Done calc train logits $task $run_id"
    echo "----------------------------------"
done