from typing import Any, Dict, List, Optional, Union
from kronfluence import FactorArguments, ScoreArguments
import torch
from torch import nn

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.task import Task

from transformers import DataCollatorWithPadding
from exp import KronfluenceTask
from lora_model import causal_tokenize, load_causal_LORA_model, load_causal_tokenizer
from datasets import Dataset
from kronfluence.utils.common.factor_arguments import all_low_precision_factor_arguments
from kronfluence.utils.common.score_arguments import all_low_precision_score_arguments
from kronfluence.utils.dataset import DataLoaderKwargs


model_path = "./data/dev/qwen/sentense/checkpoint"
dataset_path = "./datasets"
dataset = 'grammars'
factor_strategy = "ekfac"
autoregressive = True
device = "cuda"

tokenizer = load_causal_tokenizer(model_path)
model = load_causal_LORA_model(model_name_or_path=model_path)
model.to(device)
model.eval()  

train_dataset = Dataset.load_from_disk(f"{dataset_path}/{dataset}_train.hf")
eval_dataset = Dataset.load_from_disk(f"{dataset_path}/{dataset}_test.hf")
datasets = causal_tokenize(tokenizer, device=device, dataset=dataset, train=train_dataset, validation=eval_dataset)
train_dataset = datasets["train"]
eval_dataset = datasets["validation"]

task = KronfluenceTask(model, autoregressive = autoregressive)

# Define the task. See the Technical Documentation page for details.
# task = MnistTask()

# Prepare the model for influence computation.
model = prepare_model(model=model, task=task)
analyzer = Analyzer(analysis_name=dataset, model=model, task=task)

collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", return_tensors="pt")  
dataloader_kwargs = DataLoaderKwargs(collate_fn=collator)
analyzer.set_dataloader_kwargs(dataloader_kwargs)

# Fit all EKFAC factors for the given model.
factor_args = FactorArguments(strategy=factor_strategy)
# factor_args = all_low_precision_factor_arguments(strategy=factor_strategy, dtype=torch.float16)

analyzer.fit_all_factors(factors_name=factor_strategy, 
                         dataset=train_dataset,
                         factor_args=factor_args,
                         overwrite_output_dir=True,
                         per_device_batch_size=None,
                         initial_per_device_batch_size_attempt=512,)

# score_args = ScoreArguments()
score_args = all_low_precision_score_arguments(dtype=torch.bfloat16)

# Compute all pairwise influence scores with the computed factors.
analyzer.compute_pairwise_scores(
    score_args = score_args,
    scores_name=factor_strategy,
    factors_name=factor_strategy,
    query_dataset=eval_dataset,
    train_dataset=train_dataset,
    per_device_query_batch_size=512,
    per_device_train_batch_size=512,
    overwrite_output_dir=True,
)

# Load the scores with dimension `len(eval_dataset) x len(train_dataset)`.
scores = analyzer.load_pairwise_scores(scores_name=factor_strategy)
pass