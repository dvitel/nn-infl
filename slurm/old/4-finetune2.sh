#!/bin/bash -l
#SBATCH --job-name=infl-tune2
#SBATCH --output i-t2-%j.out
#SBATCH -D /data/dvitel/infl
#SBATCH -p Quick # run on partition general
#SBATCH --open-mode=append
#SBATCH --gpus=1 # 1 GPU
#SBATCH --exclude GPU41,GPU49,GPU50,GPU51,GPU52
#SBATCH --array=0-3

conda activate torch-env

tasks=("qnli" "mrpc" "sst2" "qqp")

task=${tasks[$SLURM_ARRAY_TASK_ID]}

task_cwd=/data/dvitel/infl/$task

for run_id in {0..10}; do
    echo "Starting finetune2 $task $run_id"
    # example of module-pattern --module-pattern=.\*\\.layer\\.0\\..\*\\.lora_\(A\|B\)\\..\*   --  picks LORA modules from first layer
    srun --export=ALL,INFL_SEED=$run_id,INFL_CWD=$task_cwd python ~/infl/src/exp.py finetune2 --task=$task --filter-method=infl --infl-method=datainf --module-pattern=
    echo "Done finetune2 $task $run_id"
done