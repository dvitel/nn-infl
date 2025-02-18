#!/bin/bash -l
#SBATCH --job-name=infl-matrix
#SBATCH --time=72:00:00
#SBATCH --output i-matrix-%j.out
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

echo "Starting cos infl matrix $dataset $run_id"
srun --export=ALL,INFL_SEED=$run_id,INFL_CWD=$cwd python /home/dvitel.usf/nn-infl/src/exp_resnet.py infl-matrix --dataset-name=$dataset --model-name=resnet34 --method=cos --size-koef=0.7
echo "Done hessian free infl $dataset $run_id"

echo "Starting cov infl matrix $dataset $run_id"
srun --export=ALL,INFL_SEED=$run_id,INFL_CWD=$cwd python /home/dvitel.usf/nn-infl/src/exp_resnet.py infl-matrix --dataset-name=$dataset --model-name=resnet34 --method=cov --size-koef=0.7
echo "Done datainf infl $dataset $run_id"