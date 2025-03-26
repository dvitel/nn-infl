#!/bin/bash -l
#SBATCH --job-name=l-tun2
#SBATCH --time=96:00:00
#SBATCH --output tun2-%a.out
#SBATCH -D /blue/anshumanc.usf/nn-infl/llama
#SBATCH -p hpg-ai
#SBATCH --open-mode=append
#SBATCH --gpus=1 # 1 GPU
#SBATCH --mem=16G # default 4GB
#SBATCH --array=0-3

module load conda/24.7.1
conda activate /home/dvitel.usf/torch-env

tasks=("qnli" "mrpc" "sst2" "qqp")

task=${tasks[$SLURM_ARRAY_TASK_ID]}

task_cwd=/blue/anshumanc.usf/nn-infl/llama/$task

agg_method='mean'
method_names=('hf' 'datainf' 'cos')
seeds=(117 191)
module_names=('00-03 A' '04-07 A' '08-11 A' '12-15 A' '00-03 B' '04-07 B' '08-11 B' '12-15 B')

for method_name in "${method_names[@]}"; do
    for seed2 in "${seeds[@]}"; do 
        for module_name in "${module_names[@]}"; do
            for run_id in {0..4}; do
                echo "Finetune2 $task $run_id $seed2 $method_name $agg_method $module_name"
                HF_TOKEN=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 INFL_SEED=$run_id INFL_CWD=$task_cwd python \
                    /home/dvitel.usf/nn-infl/src/exp.py finetune2 --task=$task \
                    --infl-method=$method_name --agg-method=$agg_method --module-name="$module_name" \
                    --s-prefix=s_b --unfreeze-regex=.\*\\.embed_tokens\\..\* --seed2=$seed2
                echo "Done finetune2 $task $run_id $seed2 $method_name $agg_method $module_name"
            done
        done
    done
done