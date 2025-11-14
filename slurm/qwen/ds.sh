#!/bin/bash -l
#SBATCH --job-name=q-ds
#SBATCH --time=72:00:00
#SBATCH --output ds-%a.out
#SBATCH -D /blue/anshumanc.usf/nn-infl/qwen
#SBATCH -p hpg-b200
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --array=0-2

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

echo 'Activate environment'
module load conda
conda activate /blue/anshumanc.usf/nn-infl/nn-infl-env

echo 'Running SFT...'
tasks=("sentense" "math" "mathR")
datasets=("grammars_train" "math_without_reason_train" "math_with_reason_train")
task=${tasks[$SLURM_ARRAY_TASK_ID]}
dataset=${datasets[$SLURM_ARRAY_TASK_ID]}

task_cwd=/blue/anshumanc.usf/nn-infl/qwen/$task

HF_HOME=/blue/anshumanc.usf/nn-infl/.cache \
HF_TOKEN=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 \
INFL_SEED=0 \
python /home/dvitel.usf/nn-infl/src/sft_trainer.py \
    --model-name Qwen/Qwen2.5-1.5B \
    --dataset-name /home/dvitel.usf/nn-infl/datasets/$dataset.hf \
    --output-dir $task_cwd/checkpoint \
    --dataset-text-field text \
    --learning-rate 1e-3 \
    --batch-size 128 \
    --num-train-epochs 10

echo 'Done SFT'
