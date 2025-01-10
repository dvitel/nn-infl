#!/bin/bash -l
#SBATCH --job-name=influence
#SBATCH -o infl-exp.out
#SBATCH -e infl-exp.err
#SBATCH -D /data/dvitel/infl
#SBATCH -p Quick # run on partition general
#SBATCH --open-mode=append
#SBATCH --gpus=1 # 1 GPU
#SBATCH --array=0-3

conda activate torch-env

tasks=("qnli" "mrpc" "sst2" "qqp")

task=${tasks[$SLURM_ARRAY_TASK_ID]}

for run_id in {0..19}; do
    echo "Starting run $task $run_id"
    srun python ~/infl/exp.py infl --run-id=$run_id --task=$task --model=roberta-large --num-epochs=10 --cwd=/data/dvitel/infl
    echo "Done run $task $run_id"
done

echo "Done infl"