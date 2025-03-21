#!/bin/bash -l
#SBATCH --job-name=resnet-tune
#SBATCH --output resnet-%j.out
#SBATCH -D /data/dvitel/infl
#SBATCH -p Quick # run on partition general
#SBATCH --open-mode=append
#SBATCH --gpus=1 # 1 GPU
#SBATCH --exclude GPU41,GPU49,GPU50,GPU51,GPU52
#SBATCH --array=0-9

conda activate torch-env

dataset=cifar10

run_id=$SLURM_ARRAY_TASK_ID

cwd=/data/dvitel/infl/$dataset

dataset_path=${cwd}/d_${dataset}_0

echo "Starting finetuning $dataset $run_id"
srun --export=ALL,INFL_SEED=$run_id,INFL_CWD=$cwd python ~/infl/src/exp_resnet.py finetune --dataset-path=$dataset_path --model-name=resnet34 --noise-type=worst
echo "Done finetuning $dataset $run_id"