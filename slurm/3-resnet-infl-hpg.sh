#!/bin/bash -l
#SBATCH --job-name=resnet-infl
#SBATCH --output resnet-i-%j.out
#SBATCH -D /blue/anshumanc.usf/nn-infl
#SBATCH -p hpg-ai # run on partition general
#SBATCH --open-mode=append
#SBATCH --gpus=1 # 1 GPU
#SBATCH --array=0-9

module load conda/24.7.1
conda activate /home/dvitel.usf/torch-env

dataset="cifar10"

run_id=$SLURM_ARRAY_TASK_ID

cwd=/blue/anshumanc.usf/nn-infl/$dataset

dataset_path=${cwd}/d_${dataset}_0

echo "Starting hessian free infl $dataset $run_id"
srun --export=ALL,INFL_SEED=$run_id,INFL_CWD=$cwd python /home/dvitel.usf/nn-infl/src/exp_resnet.py infl --dataset-name=$dataset --model-name=resnet34 --method=hf --size-koef=0.9
echo "Done hessian free infl $dataset $run_id"

echo "Starting hessian free infl $dataset $run_id"
srun --export=ALL,INFL_SEED=$run_id,INFL_CWD=$cwd python /home/dvitel.usf/nn-infl/src/exp_resnet.py infl --dataset-name=$dataset --model-name=resnet34 --method=datainf --size-koef=0.5
echo "Done hessian free infl $dataset $run_id"

echo "Starting hessian free infl $dataset $run_id"
srun --export=ALL,INFL_SEED=$run_id,INFL_CWD=$cwd python /home/dvitel.usf/nn-infl/src/exp_resnet.py infl --dataset-name=$dataset --model-name=resnet34 --method=lissa --size-koef=0.5
echo "Done hessian free infl $dataset $run_id"