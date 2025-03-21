#!/bin/bash -l
#SBATCH --job-name=cifar-prep
#SBATCH --output cifar-%j.out
#SBATCH -D /data/dvitel/infl
#SBATCH -p Quick # run on partition general
#SBATCH --open-mode=append
#SBATCH --gpus=1 # 1 GPU
#SBATCH --exclude GPU49,GPU50,GPU51,GPU52
#SBATCH --array=0-0

conda activate torch-env

#datasets=("cifar10" "cifar100")
datasets=("cifar10")

dataset=${datasets[$SLURM_ARRAY_TASK_ID]}

cwd=/data/dvitel/infl/$dataset

mkdir -p $cwd

echo "Starting preprocess $dataset"
srun --export=ALL,INFL_SEED=0,INFL_CWD=$cwd python ~/infl/src/exp_resnet.py preprocess --dataset=$dataset --noise-path='/home/d/dvitel/infl/datasets/cifar-noise/CIFAR-10_human.pt' --cache-dir=$cwd
echo "Done preprocess $dataset"
done