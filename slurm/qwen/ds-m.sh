#!/bin/bash -l
#SBATCH --job-name=q-ds-m
#SBATCH --time=72:00:00
#SBATCH --output ds-s-%a.out
#SBATCH -D /blue/anshumanc.usf/nn-infl/qwen
#SBATCH -p hpg-b200
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --array=0-0

module load conda
conda activate /home/dvitel.usf/torch-env

task=math
task_cwd=/blue/anshumanc.usf/nn-infl/qwen/$task

HF_TOKEN=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 INFL_SEED=0 python \
    /home/dvitel.usf/nn-infl/src/sft_trainer.py \
    --model-name Qwen/Qwen2.5-1.5B \
    --dataset-name /home/dvitel.usf/nn-infl/datasets/math_without_reason_train.hf \
    --output-dir $task_cwd/checkpoint \
    --dataset-text-field text \
    --learning-rate 3e-4 \
    --batch-size 128 \
    --num-train-epochs 10
