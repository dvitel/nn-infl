#!/bin/bash -l
#SBATCH --job-name=l-tun2
#SBATCH --time=96:00:00
#SBATCH --output 5-tun2-%a.out
#SBATCH -D /blue/anshumanc.usf/nn-infl/llama
#SBATCH -p hpg-b200
#SBATCH --gpus=1 # 1 GPU
#SBATCH --mem=16G # default 4GB
#SBATCH --array=0-7%4

module load conda
conda activate /blue/anshumanc.usf/nn-infl/nn-infl-env

tasks=("qnli" "mrpc" "sst2" "qqp" "cola" "mnli" "rte" "stsb")

task=${tasks[$SLURM_ARRAY_TASK_ID]}

task_cwd=/blue/anshumanc.usf/nn-infl/llama/$task

agg_method='mean'
method_names=('hf' 'datainf' 'cos')
module_names=('v-b-8' 'v-b-9')

for method_name in "${method_names[@]}"; do
    for module_name in "${module_names[@]}"; do
        for run_id in {0..9}; do
            echo "Finetune2 $task $run_id $method_name $agg_method $module_name"
            HF_TOKEN=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 INFL_SEED=$run_id INFL_CWD=$task_cwd python \
                /home/dvitel.usf/nn-infl/src/exp.py finetune2 --task=$task \
                --infl-method=$method_name --agg-method=$agg_method --module-name=$module_name \
                --s-prefix=s_bl --metrics-file=$task-bl.jsonlist --filter-perc=0.3
            echo "Done finetune2 $task $run_id $method_name $agg_method $module_name"
            echo "----------------------------------"
        done
    done
done

base_method_names=('rand' 'denoise')
agg_method=''
module_name=''

for method_name in "${base_method_names[@]}"; do
    for run_id in {0..9}; do
        echo "Finetune2 $task $run_id $method_name $agg_method $module_name"
        HF_TOKEN=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 INFL_SEED=$run_id INFL_CWD=$task_cwd python \
            /home/dvitel.usf/nn-infl/src/exp.py finetune2 --task=$task \
            --infl-method=$method_name --agg-method=$agg_method --module-name=$module_name \
            --s-prefix=s_bl --metrics-file=$task-bl.jsonlist --filter-perc=0.3
        echo "Done finetune2 $task $run_id $method_name $agg_method $module_name"
        echo "----------------------------------"
    done
done