#!/bin/bash -l
#SBATCH --job-name=m-ds-score
#SBATCH --time=72:00:00
#SBATCH --output ds-score-%a.out
#SBATCH -D /blue/anshumanc.usf/nn-infl/mistral
#SBATCH -p hpg-b200
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --array=0-2

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

echo 'Activate environment'
module load conda
conda activate /blue/anshumanc.usf/nn-infl/nn-infl-env

tasks=("sentense" "math" "mathR")
task=${tasks[$SLURM_ARRAY_TASK_ID]}
dataset=${datasets[$SLURM_ARRAY_TASK_ID]}

echo 'Running AUC Recall...'
task_cwd=/blue/anshumanc.usf/nn-infl/mistral/$task

HF_HOME=/blue/anshumanc.usf/nn-infl/.cache \
HF_TOKEN=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 \
INFL_SEED=0 \
INFL_CWD=$task_cwd \
python /home/dvitel.usf/nn-infl/src/exp.py auc-recall \
    --task $task \
    --infl-methods hf,cos,datainf,outlier \
    --i-prefix i_ds \
    --seeds 0,1,2,3,4 \
    --s-prefix metrics

echo 'Done AUC Recall'
