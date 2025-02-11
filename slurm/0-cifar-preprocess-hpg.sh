#!/bin/bash -l
#SBATCH --job-name=cifar-prep
#SBATCH --output cifar-%j.out
#SBATCH -D /blue/anshumanc.usf/nn-infl
#SBATCH -p hpg-ai # run on partition general
#SBATCH --open-mode=append
#SBATCH --gpus=1 # 1 GPU
#SBATCH --array=0-0

conda activate torch-env

#datasets=("cifar10" "cifar100")
datasets=("cifar10")

dataset=${datasets[$SLURM_ARRAY_TASK_ID]}

cwd=/blue/anshumanc.usf/nn-infl/$dataset

mkdir -p $cwd

echo "Starting preprocess $dataset"
srun --export=ALL,INFL_SEED=0,INFL_CWD=$cwd python ~/nn-infl/src/exp_resnet.py preprocess --dataset=$dataset --noise-path='/home/dvitel.usf/nn-infl/data/CIFAR-10_human.pt' --cache-dir=$cwd
echo "Done preprocess $dataset"
done