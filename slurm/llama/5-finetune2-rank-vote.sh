#!/bin/bash -l
#SBATCH --job-name=l-tun2
#SBATCH --time=96:00:00
#SBATCH --output 5-tun2-%a.out
#SBATCH -D /blue/anshumanc.usf/nn-infl/llama
#SBATCH -p hpg-ai
#SBATCH --gpus=1 # 1 GPU
#SBATCH --mem=16G # default 4GB
#SBATCH --array=0-7%6

module load conda/24.7.1
conda activate /home/dvitel.usf/torch-env

tasks=("qnli" "mrpc" "sst2" "qqp" "cola" "mnli" "rte" "stsb")

task=${tasks[$SLURM_ARRAY_TASK_ID]}

task_cwd=/blue/anshumanc.usf/nn-infl/llama/$task

agg_method='rank-c'
method_names=('hf' 'datainf' 'cos')
# module_names=('WE' '' '00-03' '04-07' '08-11' '12-15' '00-03 A' '04-07 A' '08-11 A' '12-15 A' '00-03 B' '04-07 B' '08-11 B' '12-15 B' 'CL')
module_names=('WE' '' '00-03' '04-07' '08-11' '12-15' 'CL')

for method_name in "${method_names[@]}"; do
    for module_name in "${module_names[@]}"; do
        for run_id in {0..4}; do
            echo "Finetune2 $task $run_id $method_name $agg_method $module_name"
            HF_TOKEN=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 INFL_SEED=$run_id INFL_CWD=$task_cwd python \
                /home/dvitel.usf/nn-infl/src/exp.py finetune2 --task=$task \
                --infl-method=$method_name --agg-method=$agg_method --module-name=$module_name \
                --s-prefix=s_bl --metrics-file=$task-rank.jsonlist --filter-perc=0.3
            echo "Done finetune2 $task $run_id $method_name $agg_method $module_name"
            echo "----------------------------------"
        done
    done
done

new_method_names=('hf_we_' 'hf_we_topk_10')
module_name='WE'

for method_name in "${new_method_names[@]}"; do
    for run_id in {0..4}; do
        echo "Finetune2 $task $run_id $method_name $agg_method $module_name"
        HF_TOKEN=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 INFL_SEED=$run_id INFL_CWD=$task_cwd python \
            /home/dvitel.usf/nn-infl/src/exp.py finetune2 --task=$task \
            --infl-method=$method_name --agg-method=$agg_method --module-name=$module_name \
            --s-prefix=s_bl --metrics-file=$task-rank.jsonlist --filter-perc=0.3
        echo "Done finetune2 $task $run_id $method_name $agg_method $module_name"
        echo "----------------------------------"
    done
done

agg_method='vote2-c'
method_names=('hf' 'datainf' 'cos')
# module_names=('WE' '' '00-03' '04-07' '08-11' '12-15' '00-03 A' '04-07 A' '08-11 A' '12-15 A' '00-03 B' '04-07 B' '08-11 B' '12-15 B' 'CL')
module_names=('WE' '' '00-03' '04-07' '08-11' '12-15' 'CL')

for method_name in "${method_names[@]}"; do
    for module_name in "${module_names[@]}"; do
        for run_id in {0..4}; do
            echo "Finetune2 $task $run_id $method_name $agg_method $module_name"
            HF_TOKEN=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 INFL_SEED=$run_id INFL_CWD=$task_cwd python \
                /home/dvitel.usf/nn-infl/src/exp.py finetune2 --task=$task \
                --infl-method=$method_name --agg-method=$agg_method --module-name=$module_name \
                --s-prefix=s_bl --metrics-file=$task-vote.jsonlist --filter-perc=0.3
            echo "Done finetune2 $task $run_id $method_name $agg_method $module_name"
            echo "----------------------------------"
        done
    done
done

new_method_names=('hf_we_' 'hf_we_topk_10')
module_name='WE'

for method_name in "${new_method_names[@]}"; do
    for run_id in {0..4}; do
        echo "Finetune2 $task $run_id $method_name $agg_method $module_name"
        HF_TOKEN=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 INFL_SEED=$run_id INFL_CWD=$task_cwd python \
            /home/dvitel.usf/nn-infl/src/exp.py finetune2 --task=$task \
            --infl-method=$method_name --agg-method=$agg_method --module-name=$module_name \
            --s-prefix=s_bl --metrics-file=$task-vote.jsonlist --filter-perc=0.3
        echo "Done finetune2 $task $run_id $method_name $agg_method $module_name"
        echo "----------------------------------"
    done
done