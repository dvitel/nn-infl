#!/bin/bash -l
#SBATCH --job-name=resnet-train
#SBATCH --output resnet-%j.out
#SBATCH -D /blue/anshumanc.usf/nn-infl
#SBATCH -p hpg-ai # run on partition general
#SBATCH --open-mode=append
#SBATCH --gpus=1 # 1 GPU
#SBATCH --array=0-9

module load conda/24.7.1
conda activate /home/dvitel.usf/torch-env

dataset=cifar10

run_id=$SLURM_ARRAY_TASK_ID

cwd=/blue/anshumanc.usf/nn-infl/$dataset

dataset_path=${cwd}/d_${dataset}_0

echo "Starting finetuning $dataset $run_id"
srun --export=ALL,INFL_SEED=$run_id,INFL_CWD=$cwd python /home/dvitel.usf/nn-infl/src/exp_resnet.py finetune --dataset-path=$dataset_path --model-name=resnet34 --noise-type=worst
echo "Done finetuning $dataset $run_id"