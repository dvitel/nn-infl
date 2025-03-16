#!/bin/bash -l
#SBATCH --job-name=tun2-cos
#SBATCH --time=72:00:00
#SBATCH --output tun2-cos-%j.out
#SBATCH -D /blue/anshumanc.usf/nn-infl
#SBATCH -p hpg-ai
#SBATCH --open-mode=append
#SBATCH --gpus=1 # 1 GPU
#SBATCH --array=0-3

module load conda/24.7.1
conda activate /home/dvitel.usf/torch-env

tasks=("qnli" "mrpc" "sst2" "qqp")

task=${tasks[$SLURM_ARRAY_TASK_ID]}

task_cwd=/blue/anshumanc.usf/nn-infl/$task

method_name=cos

# WE layer
for seed2 in {1..2}; do 
    for run_id in {0..4}; do
        echo "Starting finetune2 WE $task $run_id"
        INFL_SEED=$run_id INFL_CWD=$task_cwd python /home/dvitel.usf/nn-infl/src/exp.py finetune2 --task=$task --filter-method=infl --infl-method=$method_name --i-prefix=i_b --unfreeze-regex=.\*\\.word_embeddings\\..\* --module-pattern=.\*\\.word_embeddings\\..\* --tag='WE' --seed2=$seed2
        echo "Done finetune2 WE $task $run_id"
    done
done

# First 16 layers of LORA
for seed2 in {1..2}; do 
    for run_id in {0..4}; do
        echo "Starting finetune2 LORA 0-15 $task $run_id"
        INFL_SEED=$run_id INFL_CWD=$task_cwd python /home/dvitel.usf/nn-infl/src/exp.py finetune2 --task=$task --filter-method=infl --infl-method=$method_name --i-prefix=i_b --unfreeze-regex=.\*\\.word_embeddings\\..\* --module-pattern=.\*\\.layer\\.\(\[0-9\]\|1\[0-5\]\)\\..\*\\.lora_\(A\|B\)\\..\* --tag='0-15' --seed2=$seed2
        echo "Done finetune2 LORA 0-15 $task $run_id"
    done
done

# Last 8 layers of LORA
for seed2 in {1..2}; do 
    for run_id in {0..4}; do
        echo "Starting finetune2 LORA 16-23 $task $run_id"
        INFL_SEED=$run_id INFL_CWD=$task_cwd python /home/dvitel.usf/nn-infl/src/exp.py finetune2 --task=$task --filter-method=infl --infl-method=$method_name --i-prefix=i_b --unfreeze-regex=.\*\\.word_embeddings\\..\* --module-pattern=.\*\\.layer\\.\(1\[6-9\]\|2\[0-3\]\)\\..\*\\.lora_\(A\|B\)\\..\* --tag='16-23' --seed2=$seed2
        echo "Done finetune2 LORA 16-23 $task $run_id"
    done
done

# Classifier
for seed2 in {1..2}; do 
    for run_id in {0..4}; do
        echo "Starting finetune2 CL $task $run_id"
        INFL_SEED=$run_id INFL_CWD=$task_cwd python /home/dvitel.usf/nn-infl/src/exp.py finetune2 --task=$task --filter-method=infl --infl-method=$method_name --i-prefix=i_b --unfreeze-regex=.\*\\.word_embeddings\\..\* --module-pattern=.\*\\.classifier\\..\* --tag='CL' --seed2=$seed2
        echo "Done finetune2 CL $task $run_id"
    done
done