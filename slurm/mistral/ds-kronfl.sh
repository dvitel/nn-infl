#!/bin/bash -l
#SBATCH --job-name=m-ds-krfl
#SBATCH --time=72:00:00
#SBATCH --output ds-infl-%a.out
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
datasets=("grammars" "math_without_reason" "math_with_reason")
task=${tasks[$SLURM_ARRAY_TASK_ID]}
dataset=${datasets[$SLURM_ARRAY_TASK_ID]}

task_cwd=/blue/anshumanc.usf/nn-infl/mistral/$task

echo 'Computing kronfluence...'

for run_id in {0..4}; do
    echo "Seed $run_id"
    HF_HOME=/blue/anshumanc.usf/nn-infl/.cache \
    HF_TOKEN=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 \
    INFL_SEED=$run_id \
    INFL_CWD=$task_cwd \
    python /home/dvitel.usf/nn-infl/src/exp.py kronfl \
        --task $task \
        --method ekfac \
        --checkpoint m_ds \
        --dataset $dataset \
        --dataset-path /home/dvitel.usf/nn-infl/datasets \
        --i-prefix i_ds
done


echo 'Done kronfluence'
