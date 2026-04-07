# First is Not Really Better Than Last: Evaluating Layer Choice and Aggregation Strategies in Language Model Data Influence Estimation

**Paper:** [https://openreview.net/forum?id=Dkgw08Z4sj](https://openreview.net/forum?id=Dkgw08Z4sj)

Experiments on measuring and applying influence estimation of detrimental samples across many layers of LLMs.
The repository also contains the implementation of different influence aggregation methods, including reported Ranking and Voring.

## Quickstart

File ```exp.py``` contains five-stage pipeline for the experimentation. Each stage is executed separatelly and the results 
are preserved in the intermediate tensors on the disk. File ```influence.py``` contains influence functions implementation,
computed on GPU as 3D tensors. File ```lora_model.py``` attaches LoRA modules, load weights, performs embedding compression. 
Folder ```slurm``` contains scripts for different experimental pipline stages to execute on SLURM enabled cluster. 

Consider ```environment.yaml``` for setting up conda environment and installing dependencies. 
File ```.vscode/launch.json``` contains examples of how to start different stages of the experiment. 

### Stage 1. Starting checkpoint initialization 

The following script would create initial dataset with detrimental samples and initialize the cheeckpoint with compressed embeddings according to the dataset. 

```
HF_TOKEN=XXXX INFL_SEED=0 INFL_CWD=. python exp.py preprocess --task=$task --tokenizer-name=mistralai/Mistral-7B-v0.3
HF_TOKEN=XXXX INFL_SEED=0 INFL_CWD=. python exp.py init-checkpoint --task=$task --model=mistralai/Mistral-7B-v0.3 \
    --unfreeze-regex=.\*\\.embed_tokens\\..\* --lora-targets=q_proj,v_proj
```

### Stage 2. Finetuning 

```
HF_TOKEN=XXXX INFL_SEED=0 INFL_CWD=. python exp.py finetune --task=$task --num-epochs=10 --lr=$lr
```

### Stage 3. Computing influence tensors 

Finetuning stage preserves best checkpoints in the file system. From them, we compute the influence tensors. 

```
HF_TOKEN=XXXX INFL_SEED=0 INFL_CWD=. python exp.py infl-matrix --task=$task --methods=$method_name \
    --mem-koef=$mem_koef --m-prefix=m_bl --i-prefix=i_bl
```

### Stage 4. Reduction to sample scores and sample distribution analysis. 

```
HF_TOKEN=XXXX INFL_SEED=0 INFL_CWD=. python exp.py scores --task=$task \
    --infl-methods=datainf,cos,hf,hf_we_,hf_we_topk_10,rand,denoise \
    --agg-methods=mean,mean-c,rank,rank-c,vote2,vote2-c \
    --m-prefix=m_bl --i-prefix=i_bl --s-prefix=s_bl \
    --group-file=../groups.json
```

### Stage 5. Second finetuning after filtering according to scores.

```
agg_method='vote-c'
method_names=('hf' 'datainf' 'cos')
module_names=('WE' '' '00-07' '08-15' '16-23' '24-31' 'CL')

for method_name in "${method_names[@]}"; do
    for module_name in "${module_names[@]}"; do
        for run_id in {0..4}; do
            echo "Finetune2 $task $run_id $method_name $agg_method $module_name"
            HF_TOKEN=XXXX INFL_SEED=0 INFL_CWD=. python exp.py finetune2 --task=$task \
                --infl-method=$method_name --agg-method=$agg_method --module-name=$module_name \
                --s-prefix=s_bl --metrics-file=$task-rank.jsonlist --filter-perc=0.3
            echo "Done finetune2 $task $run_id $method_name $agg_method $module_name"
            echo "----------------------------------"
        done
    done
done
```

## Citation 

The methods and code in this repository are described in the following publication:

```bibtex
@inproceedings{
vitel2026first,
title={First is Not Really Better Than Last: Evaluating Layer Choice and Aggregation Strategies in Language Model Data Influence Estimation},
author={Dmytro Vitel and Anshuman Chhabra},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=Dkgw08Z4sj}
}
```