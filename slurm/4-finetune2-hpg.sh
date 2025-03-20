#!/bin/bash -l
#SBATCH --job-name=TUN2
#SBATCH --time=96:00:00
#SBATCH --output TUN2-%j.out
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

method_names=('hf' 'datainf' 'cos')
seeds=(117 191)

for method_name in "${method_names[@]}"; do
    for seed2 in "${seeds[@]}"; do 

        # WE
        for run_id in {0..4}; do
            echo "Starting finetune2 WE $task $run_id $seed2 $method_name"
            INFL_SEED=$run_id INFL_CWD=$task_cwd python /home/dvitel.usf/nn-infl/src/exp.py finetune2 --task=$task --filter-method=infl --infl-method=$method_name --i-prefix=i_b --unfreeze-regex=.\*\\.word_embeddings\\..\* --module-pattern=.\*\\.word_embeddings\\..\* --tag='WE' --seed2=$seed2
            echo "Done finetune2 WE $task $run_id $seed2 $method_name"
        done

        # Total
        for run_id in {0..4}; do
            echo "Starting finetune2 Total $task $run_id $seed2 $method_name"
            INFL_SEED=$run_id INFL_CWD=$task_cwd python /home/dvitel.usf/nn-infl/src/exp.py finetune2 --task=$task --filter-method=infl --infl-method=$method_name --i-prefix=i_b --unfreeze-regex=.\*\\.word_embeddings\\..\* --module-pattern= --tag='total' --seed2=$seed2
            echo "Done finetune2 Total $task $run_id $seed2 $method_name"
        done

        # LORA 0-5
        for run_id in {0..4}; do
            echo "Starting finetune2 LORA 0-5 $task $run_id $seed2 $method_name"
            INFL_SEED=$run_id INFL_CWD=$task_cwd python /home/dvitel.usf/nn-infl/src/exp.py finetune2 --task=$task --filter-method=infl --infl-method=$method_name --i-prefix=i_b --unfreeze-regex=.\*\\.word_embeddings\\..\* --module-pattern=.\*\\.layer\\.\(\[0-5\]\)\\..\*\\.lora_\(A\|B\)\\..\* --tag='0-5' --seed2=$seed2
            echo "Done finetune2 LORA 0-5 $task $run_id $seed2 $method_name"
        done

        # LORA 6-11
        for run_id in {0..4}; do
            echo "Starting finetune2 LORA 6-11 $task $run_id $seed2 $method_name"
            INFL_SEED=$run_id INFL_CWD=$task_cwd python /home/dvitel.usf/nn-infl/src/exp.py finetune2 --task=$task --filter-method=infl --infl-method=$method_name --i-prefix=i_b --unfreeze-regex=.\*\\.word_embeddings\\..\* --module-pattern=.\*\\.layer\\.\(\[6-9\]\|1\[0-1\]\)\\..\*\\.lora_\(A\|B\)\\..\* --tag='6-11' --seed2=$seed2
            echo "Done finetune2 LORA 6-11 $task $run_id $seed2 $method_name"
        done

        # LORA 12-17
        for run_id in {0..4}; do
            echo "Starting finetune2 LORA 12-17 $task $run_id $seed2 $method_name"
            INFL_SEED=$run_id INFL_CWD=$task_cwd python /home/dvitel.usf/nn-infl/src/exp.py finetune2 --task=$task --filter-method=infl --infl-method=$method_name --i-prefix=i_b --unfreeze-regex=.\*\\.word_embeddings\\..\* --module-pattern=.\*\\.layer\\.\(1\[2-7\]\)\\..\*\\.lora_\(A\|B\)\\..\* --tag='12-17' --seed2=$seed2
            echo "Done finetune2 LORA 12-17 $task $run_id $seed2 $method_name"
        done

        # LORA 12-17 B
        for run_id in {0..4}; do
            echo "Starting finetune2 LORA 12-17 B $task $run_id $seed2 $method_name"
            INFL_SEED=$run_id INFL_CWD=$task_cwd python /home/dvitel.usf/nn-infl/src/exp.py finetune2 --task=$task --filter-method=infl --infl-method=$method_name --i-prefix=i_b --unfreeze-regex=.\*\\.word_embeddings\\..\* --module-pattern=.\*\\.layer\\.\(1\[2-7\]\)\\..\*\\.lora_B\\..\* --tag='12-17 B' --seed2=$seed2
            echo "Done finetune2 LORA 12-17 B $task $run_id $seed2 $method_name"
        done        

        # LORA 12-17 A
        for run_id in {0..4}; do
            echo "Starting finetune2 LORA 12-17 A $task $run_id $seed2 $method_name"
            INFL_SEED=$run_id INFL_CWD=$task_cwd python /home/dvitel.usf/nn-infl/src/exp.py finetune2 --task=$task --filter-method=infl --infl-method=$method_name --i-prefix=i_b --unfreeze-regex=.\*\\.word_embeddings\\..\* --module-pattern=.\*\\.layer\\.\(1\[2-7\]\)\\..\*\\.lora_A\\..\* --tag='12-17 A' --seed2=$seed2
            echo "Done finetune2 LORA 12-17 A $task $run_id $seed2 $method_name"
        done        

        # LORA 18-23
        for run_id in {0..4}; do
            echo "Starting finetune2 LORA 18-23 $task $run_id $seed2 $method_name"
            INFL_SEED=$run_id INFL_CWD=$task_cwd python /home/dvitel.usf/nn-infl/src/exp.py finetune2 --task=$task --filter-method=infl --infl-method=$method_name --i-prefix=i_b --unfreeze-regex=.\*\\.word_embeddings\\..\* --module-pattern=.\*\\.layer\\.\(1\[8-9\]\|2\[0-3\]\)\\..\*\\.lora_\(A\|B\)\\..\* --tag='18-23' --seed2=$seed2
            echo "Done finetune2 LORA 18-23 $task $run_id $seed2 $method_name"
        done

        # LORA 18-23 B
        for run_id in {0..4}; do
            echo "Starting finetune2 LORA 18-23 B $task $run_id $seed2 $method_name"
            INFL_SEED=$run_id INFL_CWD=$task_cwd python /home/dvitel.usf/nn-infl/src/exp.py finetune2 --task=$task --filter-method=infl --infl-method=$method_name --i-prefix=i_b --unfreeze-regex=.\*\\.word_embeddings\\..\* --module-pattern=.\*\\.layer\\.\(1\[8-9\]\|2\[0-3\]\)\\..\*\\.lora_B\\..\* --tag='18-23 B' --seed2=$seed2
            echo "Done finetune2 LORA 18-23 B $task $run_id $seed2 $method_name"
        done        

        # LORA 18-23 A
        for run_id in {0..4}; do
            echo "Starting finetune2 LORA 18-23 A $task $run_id $seed2 $method_name"
            INFL_SEED=$run_id INFL_CWD=$task_cwd python /home/dvitel.usf/nn-infl/src/exp.py finetune2 --task=$task --filter-method=infl --infl-method=$method_name --i-prefix=i_b --unfreeze-regex=.\*\\.word_embeddings\\..\* --module-pattern=.\*\\.layer\\.\(1\[8-9\]\|2\[0-3\]\)\\..\*\\.lora_A\\..\* --tag='18-23 A' --seed2=$seed2
            echo "Done finetune2 LORA 18-23 A $task $run_id $seed2 $method_name"
        done        

        # CL
        for run_id in {0..4}; do
            echo "Starting finetune2 CL $task $run_id $seed2 $method_name"
            INFL_SEED=$run_id INFL_CWD=$task_cwd python /home/dvitel.usf/nn-infl/src/exp.py finetune2 --task=$task --filter-method=infl --infl-method=$method_name --i-prefix=i_b --unfreeze-regex=.\*\\.word_embeddings\\..\* --module-pattern=.\*\\.classifier\\..\* --tag='CL' --seed2=$seed2
            echo "Done finetune2 CL $task $run_id $seed2 $method_name"
        done
    done
done

new_method_names=('hf_we_' 'hf_we_topk_10')

for method_name in "${new_method_names[@]}"; do
    for seed2 in "${seeds[@]}"; do 

        # WE
        for run_id in {0..4}; do
            echo "Starting finetune2 WE $task $run_id $seed2 $method_name"
            INFL_SEED=$run_id INFL_CWD=$task_cwd python /home/dvitel.usf/nn-infl/src/exp.py finetune2 --task=$task --filter-method=infl --infl-method=$method_name --i-prefix=i_b --unfreeze-regex=.\*\\.word_embeddings\\..\* --module-pattern=.\*\\.word_embeddings\\..\* --tag='WE' --seed2=$seed2
            echo "Done finetune2 WE $task $run_id $seed2 $method_name"
        done

    done 
done

base_method_names=('rand' 'denoise')
for method_name in "${base_method_names[@]}"; do
    for seed2 in "${seeds[@]}"; do 
        for run_id in {0..4}; do
            echo "Starting finetune2 WE $task $run_id $seed2 $method_name"
            INFL_SEED=$run_id INFL_CWD=$task_cwd python /home/dvitel.usf/nn-infl/src/exp.py finetune2 --task=$task --filter-method=$method_name --unfreeze-regex=.\*\\.word_embeddings\\..\* --tag=$method_name --i-prefix=i_b --seed2=$seed2
            echo "Done finetune2 WE $task $run_id $seed2 $method_name"
        done    
    done
done