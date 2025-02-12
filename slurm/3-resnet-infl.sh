#!/bin/bash -l
#SBATCH --job-name=resnet-infl
#SBATCH --output resnet-i-%j.out
#SBATCH -D /data/dvitel/infl
#SBATCH -p Quick # run on partition general
#SBATCH --open-mode=append
#SBATCH --gpus=1 # 1 GPU
#SBATCH --exclude GPU49,GPU50,GPU51,GPU52
#SBATCH --array=0-9

conda activate torch-env

dataset="cifar10"

run_id=$SLURM_ARRAY_TASK_ID

cwd=/data/dvitel/infl/$dataset

dataset_path=${cwd}/d_${dataset}_0

echo "Starting hessian free infl $dataset $run_id"
srun --export=ALL,INFL_SEED=$run_id,INFL_CWD=$cwd python ~/infl/src/exp_resnet.py infl --dataset-name=$dataset --model-name=resnet34 --method=hf --size-koef=0.9
echo "Done hessian free infl $dataset $run_id"

echo "Starting hessian free infl $dataset $run_id"
srun --export=ALL,INFL_SEED=$run_id,INFL_CWD=$cwd python ~/infl/src/exp_resnet.py infl --dataset-name=$dataset --model-name=resnet34 --method=datainf --size-koef=0.5
echo "Done hessian free infl $dataset $run_id"

echo "Starting hessian free infl $dataset $run_id"
srun --export=ALL,INFL_SEED=$run_id,INFL_CWD=$cwd python ~/infl/src/exp_resnet.py infl --dataset-name=$dataset --model-name=resnet34 --method=lissa --size-koef=0.5
echo "Done hessian free infl $dataset $run_id"