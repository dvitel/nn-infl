#!/bin/bash -l
#SBATCH --job-name=influence
#SBATCH -o infl-exp.out
#SBATCH -e infl-exp.err
#SBATCH -D /data/dvitel/infl
#SBATCH -p Quick # run on partition general
#SBATCH --open-mode=append
#SBATCH --gpus=1 # 1 GPU

conda activate torch-env

for run_id in {0..29}; do
    echo "Starting run $run_id"
    srun python ~/infl/exp.py infl --run-id=$run_id --task=qnli --model=roberta-large --num-epochs=10 --cwd=/data/dvitel/infl
    echo "Done run $run_id"
done

echo "Done infl"