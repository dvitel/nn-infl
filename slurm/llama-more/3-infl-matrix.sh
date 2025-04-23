#!/bin/bash -l
#SBATCH --job-name=l-infl
#SBATCH --time=72:00:00
#SBATCH --output 3-infl-%a.out
#SBATCH -D /blue/anshumanc.usf/nn-infl/llama-more
#SBATCH -p hpg-ai
#SBATCH --gpus=1 # 1 GPU
#SBATCH --mem=16G
#SBATCH --array=0-7

module load conda/24.7.1
conda activate /home/dvitel.usf/torch-env

tasks=("qnli" "mrpc" "sst2" "qqp" "cola" "mnli" "rte" "stsb")

task=${tasks[$SLURM_ARRAY_TASK_ID]}

task_cwd=/blue/anshumanc.usf/nn-infl/llama-more/$task

method_name=datainf
mem_koef=2.0

for run_id in {5..9}; do

    echo "Infl matrix $task $run_id $method_name"
    HF_TOKEN=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 INFL_SEED=$run_id INFL_CWD=$task_cwd python \
        /home/dvitel.usf/nn-infl/src/exp.py infl-matrix --task=$task --methods=$method_name \
        --mem-koef=$mem_koef --m-prefix=m_bl --i-prefix=i_bl
    echo "----- Done $task $run_id $method_name"
    echo "----------------------------------"

done

method_name=cos
mem_koef=1.1

for run_id in {5..9}; do

    echo "Infl matrix $task $run_id $method_name"
    HF_TOKEN=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 INFL_SEED=$run_id INFL_CWD=$task_cwd python \
        /home/dvitel.usf/nn-infl/src/exp.py infl-matrix --task=$task --methods=$method_name --mem-koef=$mem_koef \
        --m-prefix=m_bl --i-prefix=i_bl
    echo "----- Done $task $run_id $method_name"
    echo "----------------------------------"

done

method_name=hf
mem_koef=1.1

for run_id in {5..9}; do

    echo "Infl matrix $task $run_id $method_name"
    HF_TOKEN=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 INFL_SEED=$run_id INFL_CWD=$task_cwd python \
        /home/dvitel.usf/nn-infl/src/exp.py infl-matrix --task=$task --methods=$method_name \
        --mem-koef=$mem_koef --m-prefix=m_bl --i-prefix=i_bl
    echo "----- Done $task $run_id $method_name"
    echo "----------------------------------"

done


method_name='hf_we_topk_10'
mem_koef=2.2

for run_id in {5..9}; do

    echo "Infl matrix $task $run_id $method_name"
    HF_TOKEN=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 INFL_SEED=$run_id INFL_CWD=$task_cwd python \
        /home/dvitel.usf/nn-infl/src/exp.py infl-matrix --task=$task --methods=$method_name \
        --mem-koef=$mem_koef --m-prefix=m_bl --i-prefix=i_bl
    echo "----- Done $task $run_id $method_name"
    echo "----------------------------------"
done

method_name='hf_we_'
mem_koef=2.2

for run_id in {5..9}; do

    echo "Infl matrix $task $run_id $method_name"
    HF_TOKEN=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 INFL_SEED=$run_id INFL_CWD=$task_cwd python \
        /home/dvitel.usf/nn-infl/src/exp.py infl-matrix --task=$task --methods=$method_name \
        --mem-koef=$mem_koef --m-prefix=m_bl --i-prefix=i_bl
    echo "----- Done $task $run_id $method_name"
    echo "----------------------------------"

done