#!/bin/bash -l
#SBATCH --job-name=m-ds-s
#SBATCH --time=72:00:00
#SBATCH --output ds-s-%a.out
#SBATCH -D /blue/anshumanc.usf/nn-infl/mistral
#SBATCH -p hpg-b200
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --array=0-0

echo 'Activate environment'
module load conda
conda activate /blue/anshumanc.usf/nn-infl/nn-infl-env

echo 'Running SFT...'
task=sentense
task_cwd=/blue/anshumanc.usf/nn-infl/mistral/$task

HF_HOME=/blue/anshumanc.usf/nn-infl/.cache \
HF_TOKEN=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 \
INFL_SEED=0 \
python /home/dvitel.usf/nn-infl/src/sft_trainer.py \
    --model-name mistralai/Mistral-7B-v0.3 \
    --dataset-name /home/dvitel.usf/nn-infl/datasets/grammars_train.hf \
    --output-dir $task_cwd/checkpoint \
    --dataset-text-field text \
    --learning-rate 5e-5 \
    --batch-size 128 \
    --num-train-epochs 10

echo 'Done SFT'
