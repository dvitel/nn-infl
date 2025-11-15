# import warnings
# import traceback

# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
#     print("⚠️ WARNING CAUGHT")
#     print(message)
#     traceback.print_stack()

# warnings.showwarning = warn_with_traceback

from collections import defaultdict
import fcntl
from functools import partial
import json
import os
import re
from typing import Any, Dict, List, Optional, Union

import datasets
from kronfluence import Analyzer, FactorArguments, ScoreArguments, Task, prepare_model
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from postprocess import compute_ndr_metrics_table, cset_matrix_score, mean_matrix_score, min_matrix_score, rank_matrix_score, vote2_matrix_score, vote_matrix_score
os.environ['HF_HOME'] = os.path.join(os.getcwd(), '.cache')
import pickle
import random

import argh
import numpy as np
from lora_model import build_LORA_model, causal_tokenize, load_causal_tokenizer, load_tokenizer, save_checkpoint, train_LORA_model, load_pretrained_LORA_model, compute_grads, load_causal_LORA_model
from influence import compute_hessian_free_influences, compute_datainf_influences, compute_lissa_influences, compute_accurate_influences
import torch
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset, load_from_disk, Dataset, ClassLabel
from torch.utils.data import DataLoader
from kronfluence.utils.common.factor_arguments import all_low_precision_factor_arguments
from kronfluence.utils.common.score_arguments import all_low_precision_score_arguments
from kronfluence.utils.dataset import DataLoaderKwargs

if torch.cuda.is_available():
    torch.cuda.init()

seed = int(os.environ.get("INFL_SEED", 0))
cwd = os.environ.get("INFL_CWD", "./data/dev")

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

task_to_keys = {
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "sst2": ("sentence", None),

    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"), # 3 classes --> convert to 2 classes (entailment, not entailment)
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
}

def flip_label(example, ind, noise_index):
    if ind in noise_index:
        example["label"] = 1 - example["label"]
        example["noise"] = True
    else:
        example["noise"] = False
    return example

# def select_shuffled_balanced(dataset, total_size, num_groups = 1):
#     possible_labels = len(dataset.features['label'].names)
#     num_samples_per_class_per_group = total_size // possible_labels
#     num_samples_per_class = num_groups * num_samples_per_class_per_group
#     labels = np.array(dataset['label'])
#     all_indices = [[] for _ in range(num_groups)]
#     for i in range(possible_labels):
#         class_0_indices = np.where(labels == i)[0]
#         sampled_class_0_indices = np.random.choice(class_0_indices, num_samples_per_class, replace=False)
#         for i, g in enumerate(all_indices):
#             g.append(sampled_class_0_indices[i * num_samples_per_class_per_group:(i + 1) * num_samples_per_class_per_group])
#     datasets = []
#     for group in all_indices:
#         sampled_indices = np.concatenate(group)
#         np.random.shuffle(sampled_indices)
#         sampled_dataset = dataset.select(sampled_indices)
#         datasets.append(sampled_dataset)
#     return datasets

def load_noisy_dataset_by_task(task, infl_ratio = 0.5, max_val_size = None, max_train_size = None, noise_ratio=0.2):
    glue_datasets = load_dataset("glue", task) 
    if task == 'mnli':
        glue_datasets['train'] = glue_datasets['train'].filter(lambda x: x['label'] != 2)
        glue_datasets['validation'] = glue_datasets['validation_matched'].filter(lambda x: x['label'] != 2)
        new_features = glue_datasets['train'].features.copy()
        new_features['label'] = ClassLabel(names=["entailment", "neutral"])      
        glue_datasets = glue_datasets.cast(new_features)  
    if task == 'stsb':
        def remap_label(row):
            ''' remaps label 0-5 to two classes 0 and 1 based on threshold 3 '''
            if row['label'] < 3:
                row['label'] = 0
            else:
                row['label'] = 1
            return row
        glue_datasets['train'] = glue_datasets['train'].map(remap_label)
        glue_datasets['validation'] = glue_datasets['validation'].map(remap_label)
        new_features = glue_datasets['train'].features.copy()
        new_features['label'] = ClassLabel(names=["not similar", "similar"])
        glue_datasets = glue_datasets.cast(new_features)

    if max_train_size is not None and max_train_size < len(glue_datasets['train']):
        tmpsets = glue_datasets['train'].train_test_split(train_size = max_train_size, shuffle=True, seed=seed, stratify_by_column='label')
        glue_datasets['train'] = tmpsets['train']
    if max_val_size is not None and max_val_size < len(glue_datasets['validation']):
        tmpsets = glue_datasets['validation'].train_test_split(train_size = max_val_size, shuffle=True, seed=seed, stratify_by_column='label')  
        glue_datasets['validation'] = tmpsets['train']

    infl_size = int(infl_ratio * len(glue_datasets['validation']))
    if task == 'wnli' or task == 'rte': # very small - we use infl and validation as same set 
        glue_datasets['infl'] = glue_datasets['validation']
    else:
        tmpsets = glue_datasets['validation'].train_test_split(train_size = infl_size, shuffle=True, seed=seed, stratify_by_column='label') 
        glue_datasets['infl'] = tmpsets['train']
        glue_datasets['validation'] = tmpsets['test']

    ds_names = list(glue_datasets.keys())
    for key in ds_names:
        if key not in ['train', 'validation', 'infl']:
            del glue_datasets[key]

    train_size = len(glue_datasets['train'])

    if noise_ratio > 0.0:
        noise_index = set(np.random.choice(train_size, size=int(noise_ratio*train_size), replace=False))
    else:
        noise_index = []

    glue_datasets['train'] = glue_datasets['train'].map(flip_label, with_indices=True, fn_kwargs={'noise_index':noise_index})
    
    # glue_datasets.pop('test')
    
    return glue_datasets

def preprocess(task = 'qnli', noise_ratio = 0.2, tokenizer_name='roberta-large'):
    ''' Preprocoess GLUE dataset of specific task '''
    config = dict(seed = seed, task=task, noise_ratio=noise_ratio, 
                  tokenizer_name=tokenizer_name)

    sentence1_key, sentence2_key = task_to_keys[task]
    tokenizer = load_tokenizer(tokenizer_name)
    def tokenize_function(examples, max_length=128):
        # max_length=None => use the model max length (it's actually the default)
        if sentence2_key is None:
            outputs = tokenizer(examples[sentence1_key], truncation=True, max_length=max_length)
        else:
            outputs = tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, max_length=max_length)
        return outputs

    noisy_datasets = load_noisy_dataset_by_task(task, max_train_size = 4500, max_val_size = 1000, infl_ratio=0.5, noise_ratio=noise_ratio)
    if sentence2_key is None:
        tokenized_datasets = noisy_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["idx", sentence1_key],
        )
    else:
        tokenized_datasets = noisy_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["idx", sentence1_key, sentence2_key],
        )

    # We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
    # transformers library
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    config_path = os.path.join(cwd, f'c_{task}_{seed}.json')
    with open(config_path, 'w') as file:
        json.dump(config, file)

    dataset_path = os.path.join(cwd, f'd_{task}_{seed}')
    tokenized_datasets.save_to_disk(dataset_path)

def present_token_ids(*dataloaders: DataLoader):
    all_token_id_set = set()
    for dl in dataloaders:
        for batch in dl:
            batch_ids = batch['input_ids']            
            all_token_id_set.update(torch.unique(torch.flatten(batch_ids)).tolist())
    all_token_id_list = list(all_token_id_set)
    all_token_ids = torch.tensor(all_token_id_list)
    old_to_new_token_id = {old_id: new_id for new_id, old_id in enumerate(all_token_id_list)}
    mapping_tensor = torch.tensor([old_to_new_token_id.get(old_id, old_id) for old_id in range(torch.max(all_token_ids).item() + 1)])
    return all_token_ids, mapping_tensor

def build_loaders(dataset_path, tokenizer_name, batch_size = 32, shuffle_train = True, 
                    filter_fn = None):
    datasets = load_from_disk(dataset_path)
    trainset = datasets['train']#.select(range(100))
    valset = datasets['validation']#.select(range(100))
    inflset = datasets['infl']#.select(range(100))
    if filter_fn is not None:
        trainset = filter_fn(trainset)
    trainset = trainset.remove_columns(['noise'])
    tokenizer = load_tokenizer(tokenizer_name)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", return_tensors="pt")  
        
    train_dataloader = DataLoader(trainset, #.select(range(100)),
                                  shuffle=shuffle_train, 
                                  collate_fn=collator,
                                  batch_size=batch_size)
    eval_dataloader = DataLoader(valset, #.select(range(100)),
                                 shuffle=False, 
                                 collate_fn=collator, 
                                 batch_size=batch_size)
    infl_dataloader = DataLoader(inflset, #.select(range(100)),
                                 shuffle=False, 
                                 collate_fn=collator, 
                                 batch_size=batch_size)  

    all_token_ids, mapping_tensor = present_token_ids(train_dataloader, eval_dataloader, infl_dataloader)

    return train_dataloader, eval_dataloader, infl_dataloader, tokenizer, all_token_ids, mapping_tensor

def init_checkpoint(task = 'mrpc', model: str = 'roberta-large', unfreeze_regex = None, 
                    low_rank = 4, lora_targets: str = 'value'):
    ''' Loads and Prints information about the model '''

    lora_targets = lora_targets.split(',')
    config_path = os.path.join(cwd, f'c_{task}_{seed}.json')
    with open(config_path, 'r') as file:
        config = json.load(file)    

    start_checkpoint_path = os.path.join(cwd, f'm_00_{task}_{seed}')
    dataset_path = os.path.join(cwd, f'd_{task}_{seed}')

    config.update(low_rank=low_rank, task_name = task, model_name = model,
                  target_modules=lora_targets, unfreeze_regex = unfreeze_regex,
                  start_checkpoint_path = start_checkpoint_path,
                  dataset_path = dataset_path)

    _, _, _, tokenizer, all_token_ids, mapping_tensor = build_loaders(dataset_path, model)
        
    lora_model, lora_model_info = build_LORA_model(model_name_or_path=model, pad_token_id=tokenizer.pad_token_id,
                                                    target_modules=lora_targets,
                                                    low_rank=low_rank, unfreeze_modules_regex=None,
                                                    all_token_ids = all_token_ids, mapping_tensor = mapping_tensor)

    save_checkpoint(lora_model, start_checkpoint_path)
    tokenizer.save_pretrained(start_checkpoint_path)

    info_path = os.path.join(start_checkpoint_path, 'model-layers-info.txt')
    with open(info_path, 'w') as file:
        file.write(str(lora_model_info))

    with open(config_path, 'w') as file:
        json.dump(config, file)

# def remap_vocab(tokenizer, all_token_ids: torch.Tensor):
#     old_vocab = tokenizer.vocab
#     new_vocab = { token_id: idx for idx, token_id in enumerate(all_token_ids.tolist()) }
#     tokenizer.vocab = new_vocab
#     tokenizer.ids_to_tokens = {idx: token for token, idx in new_vocab.items()}
#     tokenizer.tokens_to_ids = new_vocab

    
def finetune(task = 'mrpc', device = 'cuda', lr = 3e-4, batch_size = 32, num_epochs = 10,
             fast = False):
    ''' Fine tune specific model on specific task and save it to disk for later postprocessing'''
    config_path = os.path.join(cwd, f'c_{task}_{seed}.json')
    with open(config_path, 'r') as file:
        config = json.load(file)
        
    if 'finetune' in config:
        del config['finetune']

    config.update(device=device, lr=lr, batch_size=batch_size, num_epochs=num_epochs)

    dataset_path = os.path.join(cwd, f'd_{task}_{seed}')
    train_dataloader, eval_dataloader, infl_dataloader, tokenizer, _, _ = \
        build_loaders(dataset_path, config['tokenizer_name'], batch_size)    

    # collected checkpoints     
    start_checkpoint_path = config['start_checkpoint_path']
    if fast:
        best_model_path = None 
        best_loss_model_path = None
        last_model_path = None 
    else:
        best_model_path = os.path.join(cwd, f'm_b_{task}_{seed}')
        best_loss_model_path = os.path.join(cwd, f'm_bl_{task}_{seed}')
        last_model_path = os.path.join(cwd, f'm_l_{task}_{seed}')

    # compute_cancellation = not fast
    # compute_gold_val_predictions = not fast

    # unfreeze_regex = config.get('unfreeze_regex', None)

    # if os.path.exists(base_model_path):
    print(f"Loading {start_checkpoint_path}")
    # todo check if reequires_grad is attached to embedding
    lora_model = load_pretrained_LORA_model(model_name_or_path=start_checkpoint_path, unfreeze_modules_regex=None)
    # else:
    #     print(f"Creating base checkpoint")
    #     lora_model = build_LORA_model(model_name_or_path=model, pad_token_id=tokenizer.pad_token_id,
    #                                 target_modules=lora_targets, 
    #                                 low_rank=low_rank, unfreeze_modules_regex=unfreeze_regex,
    #                                 all_token_ids = all_token_ids, mapping_tensor = mapping_tensor)

    #     save_checkpoint(lora_model, base_model_path)
    #     tokenizer.save_pretrained(base_model_path)

        # lora_model2 = load_pretrained_LORA_model(model_name_or_path=base_model_path, unfreeze_modules_regex=unfreeze_regex)

        # for (name1, param1), (name2, param2) in zip(lora_model.named_parameters(), lora_model2.named_parameters()):
        #     if "original_module" not in name1:
        #         assert torch.allclose(param1, param2, rtol=1e-05, atol=1e-08), f'Parameters are not equal: {name1} {name2}'

    
    eval_metrics = train_LORA_model(lora_model, train_dataloader, eval_dataloader, infl_dataloader, device, num_epochs, lr,
                                    # compute_cancellation=compute_cancellation, 
                                    # compute_gold_val_predictions=compute_gold_val_predictions, 
                                    best_checkpoint_path = best_model_path,
                                    last_checkpoint_path = last_model_path,
                                    best_loss_model_path = best_loss_model_path)

    config['finetune'] = eval_metrics
    
    if not fast:
        with open(config_path, 'w') as file:
            json.dump(config, file)       


        tokenizer.save_pretrained(best_model_path)
        tokenizer.save_pretrained(best_loss_model_path)
        tokenizer.save_pretrained(last_model_path)

    ## next code is for testing weights preservation 
    # lora_model.to('cpu')
    # another_lora_model = load_pretrained_LORA_model(model_name_or_path=model_path)
    # for (name1, param1), (name2, param2) in zip(lora_model.named_parameters(), another_lora_model.named_parameters()):
    #     if "original_module" not in name1:
    #         assert torch.allclose(param1, param2, rtol=1e-05, atol=1e-08), f'Parameters are not equal: {name1} {name2}'

    # lora_model.save_pretrained(model_path)
    
    del lora_model, train_dataloader, eval_dataloader

    with torch.no_grad():
        torch.cuda.empty_cache()

def cancel_eff(task = 'qnli',  
               m_prefix = 'm_bl', device = 'cuda',
                group_file: str = './groups.json'):
    ''' Computes cancelation effect metric for provided checkpoint 
        Passess train dataset once (1 epoch) one by one through the model and aggregates the grads of modules (including WE).
        Cancel = sum(abs(grad)) / norm(sum(grad))
    '''
    config_path = os.path.join(cwd, f'c_{task}_{seed}.json')
    with open(config_path, 'r') as file:
        config = json.load(file)    

    with open(os.path.join(cwd, group_file), 'r') as file:
        module_groups_regex = json.load(file)
    module_groups_patterns = {name: re.compile(pattern) for name, pattern in module_groups_regex.items()}

    unfreeze_regex = config.get('unfreeze_regex', None)
    dataset_path = os.path.join(cwd, f'd_{task}_{seed}')
    model_path = os.path.join(cwd, f'{m_prefix}_{task}_{seed}')
    train_dataloader, _, _, _, _, _ = \
        build_loaders(dataset_path, model_path, batch_size=1, shuffle_train=False)
    
    lora_model = load_pretrained_LORA_model(model_name_or_path=model_path, unfreeze_modules_regex=unfreeze_regex)
    lora_model.to(device)
    lora_model.eval() #no norm and dropout

    module_names = [name for name, p in lora_model.named_parameters() if p.requires_grad]

    module_groups = defaultdict(list)
    for group_name, pattern in module_groups_patterns.items():
        for module_name in module_names:
            if pattern.match(module_name):
                module_groups[module_name].append(group_name)        


    module_grads_l1 = {}
    module_grads_sum = {}
    
    for step, batch in enumerate(tqdm(train_dataloader)):
        lora_model.zero_grad()
        batch.to(device)
        outputs = lora_model(**batch)
        loss = outputs.loss
        loss.backward()
        for name, param in lora_model.named_parameters():
            if param.requires_grad:
                if name in module_grads_l1:
                    module_grads_l1[name] += param.grad.view(-1).abs()
                else:
                    module_grads_l1[name] = param.grad.view(-1).abs()
                if name in module_grads_sum:
                    module_grads_sum[name] += param.grad.view(-1)
                else:
                    module_grads_sum[name] = param.grad.view(-1).clone()
    module_every_cancellation = {}
    module_cancellation = {}
    group_cancellation = {}
    group_every_cancellation = {}
    
    for module_name in module_names:
        g_l1 = module_grads_l1[module_name]
        g_sum = module_grads_sum[module_name]
        g_l1_nonzero_ids = torch.nonzero(g_l1, as_tuple=False)
        g_l1_nz = g_l1[g_l1_nonzero_ids]
        g_sum_nz = g_sum[g_l1_nonzero_ids]
        module_every_cancellation[module_name] = g_l1_nz / g_sum_nz.abs()
        mc = np.array([g_l1.sum().item(), g_sum.abs().sum().item(), len(g_l1_nonzero_ids), torch.norm(g_sum).item()])
        module_cancellation[module_name] = mc
        for group_name in module_groups.get(module_name, []):
            if group_name in group_cancellation:
                group_cancellation[group_name] += mc[:-1]
            else:
                group_cancellation[group_name] = mc[:-1].copy()
            group_every_cancellation.setdefault(group_name, []).append(module_every_cancellation[module_name])
        del g_l1, g_sum

    all_results = {}
    for module_name in module_names:
        module_results = all_results.setdefault(module_name, {})
        mc = module_cancellation[module_name]
        module_results['cancellation'] = mc[0] / mc[1]
        module_results['num_params'] = mc[2]
        module_results['cancellation_l2'] = mc[0] / mc[3]
        module_results["median_cancellation"] = module_every_cancellation[module_name].median().item()
        module_results["min_cancellation"] = module_every_cancellation[module_name].min().item()
        module_results["max_cancellation"] = module_every_cancellation[module_name].max().item()

    for group_name, group_c in group_cancellation.items():
        group_results = all_results.setdefault(group_name, {})
        group_results['cancellation'] = group_c[0] / group_c[1]
        group_results['num_params'] = group_c[2]
        cls = torch.cat(group_every_cancellation[group_name])
        group_results["median_cancellation"] = cls.median().item()
        group_results["min_cancellation"] = cls.min().item()
        group_results["max_cancellation"] = cls.max().item()
        del cls

    out_folder = os.path.join(model_path, f'cancellation.json')
    with open(out_folder, 'w') as file:
        json.dump(all_results, file)
    pass 

def combine_cancel(tasks = "qnli", m_prefix = "m_bl", run_ids = "0,1,2,3,4,5,6,7,8,9", 
                   group_file = './groups.json'):
    ''' Combines cancellation results into one dataframe saved in cwd'''
    import pandas as pd
    if group_file != '':
        with open(os.path.join(cwd, group_file), 'r') as file:
            module_groups_regex = json.load(file)
        selected_layers = list(module_groups_regex.keys())
    else:
        selected_layers = []
    tasks = tasks.split(',')
    run_ids = [int(x) for x in run_ids.split(',')]
    if len(selected_layers) == 0:
        selected_layers = None
    all_data = []
    for task in tasks:
        for seed in run_ids:
            cancel_path = os.path.join(cwd, task, f'{m_prefix}_{task}_{seed}', f'cancellation.json')
            with open(cancel_path, 'r') as f:
                cancellations = json.load(f)
            if selected_layers is None:
                selected_layers = cancellations.keys()
            for layer_name in selected_layers:
                if layer_name not in cancellations:
                    continue
                metrics = cancellations[layer_name]
                all_data.append({'task': task, 'layer': layer_name, 'run_id': seed, **metrics})
    df = pd.DataFrame(all_data).set_index(['task', 'layer', 'run_id'])
    df.to_pickle(os.path.join(cwd, f'cancellation_{m_prefix}.pkl'))
    pass


def grads(task = 'mrpc', no_val = False, return_grads = False, config = None, m_prefix = 'm_b'):
    ''' Computes gradients for modules of the model'''
    if config is None:
        config_path = os.path.join(cwd, f'c_{task}_{seed}.json')
        with open(config_path, 'r') as file:
            config = json.load(file)
    device = config['device']
    model_path = os.path.join(cwd, f'{m_prefix}_{task}_{seed}')
    lora_model = load_pretrained_LORA_model(model_name_or_path=model_path)
    dataset_path = os.path.join(cwd, f'd_{task}_{seed}')
    train_dataloader, _, infl_dataloader, _ = \
        build_loaders(dataset_path, config['tokenizer_name'], batch_size=1, 
                        shuffle_train=False)
    train_grads = compute_grads(lora_model, train_dataloader, device=device, bring_to_cpu=not return_grads)
    if no_val:
        val_grads = {}
    else:
        val_grads = compute_grads(lora_model, infl_dataloader, device=device, bring_to_cpu=not return_grads)

    if return_grads:
        return {'train': train_grads, 'validation': val_grads}
    else:
        grad_path = os.path.join(cwd, f'g_{task}_{seed}.pt')
        torch.save({'train': train_grads, 'validation': val_grads}, grad_path)

# add new methods here
influence_methods = \
    {
        "hf": compute_hessian_free_influences,
        "datainf": compute_datainf_influences,
        "lissa": compute_lissa_influences,
        "exact": compute_accurate_influences
    }

def infl(task = 'mrpc', methods = "hf", self_influence = False, with_grads = False, i_prefix='i_bl', m_prefix='m_bl', ignore_metrics = False):
    config_path = os.path.join(cwd, f'c_{task}_{seed}.json')
    with open(config_path, 'r') as file:
        config = json.load(file)

    device = config['device']

    if with_grads:
        gradients = grads(task = task, return_grads=True, config=config, no_val=self_influence, m_prefix = m_prefix)
    else:
        gradients = torch.load(os.path.join(cwd, f'g_{task}_{seed}.pt'))
    
    train_grads = {k:v.to(device) for k, v in gradients['train'].items()}

    if self_influence:
        val_grads = train_grads
    else:        
        val_grads = {k:v.to(device) for k, v in gradients['validation'].items()}

    runtimes = {}
    influences = {}
    for infl_method in methods.split(','):
        if infl_method not in influence_methods:
            continue
        # NOTE: not all methods require mean gradients
        avg_val_grads = {module_name: torch.mean(module_grads, dim=0) for module_name, module_grads in val_grads.items()}
        method_fn = influence_methods[infl_method]
        runtine, inf_tensors = method_fn(train_grads, val_grads, avg_val_grads, bring_to_cpu=True)
                
        # model_path = os.path.join(cwd, f'm_{task}_{seed}')
        # lora_model = load_pretrained_LORA_model(model_name_or_path=model_path)
        # dataset_path = os.path.join(cwd, f'd_{task}_{seed}')
        # train_dataloader, eval_dataloader, _ = \
        #     build_loaders(dataset_path, config['tokenizer_name'], batch_size=1, 
        #                     shuffle_train=False, val_size=500)

        # inf_tensors2 = compute_infl_from_model(lora_model, train_dataloader, eval_dataloader, device=device, infl_fn=lissa_fn)
        # pass
        
        
        influences[infl_method] = inf_tensors
        runtimes[infl_method] = runtine

    config["infl_runtimes"] = runtimes

    for infl_method, infls in influences.items():
        infl_path = os.path.join(cwd, f'{i_prefix}_{infl_method}_{task}_{seed}.pt')
        torch.save(infls, infl_path)

    if not ignore_metrics:
        with open(config_path, 'w') as file:
            json.dump(config, file)


def get_dataset_splits(dataset: Dataset, numel: int): 
    ''' Generator that returns split flow, datasets of max of numel size from dataset '''    
    cur_dataset = dataset
    # cur_dataset.cache_files = []
    total_shift = 0
    while len(cur_dataset) > numel:        
        new_dict = cur_dataset.train_test_split(train_size = numel, shuffle=False)  #keep_in_memory = True, load_from_cache_file=False, test_indices_cache_file_name = None, train_indices_cache_file_name= None)
        cur_dataset = new_dict['test']
        yield (total_shift, new_dict['train'])
        total_shift += len(new_dict['train'])
    if len(cur_dataset) > 0:
        yield (total_shift, cur_dataset)

class DatasetSplits:
    def __init__(self, trainset: Dataset, train_size: int, valset: Dataset, val_size: int):
        self.trainset = trainset
        self.train_size = train_size
        self.valset = valset
        self.val_size = val_size
        self.tran_batches = len(trainset) // train_size + (1 if len(trainset) % train_size > 0 else 0)
        self.val_batches = len(valset) // val_size + (1 if len(valset) % val_size > 0 else 0)
        self.total_batches = self.tran_batches * self.val_batches

    def __iter__(self):
        for val_shift, val_ds in get_dataset_splits(self.valset, self.val_size):
            for train_shift, train_ds in get_dataset_splits(self.trainset, self.train_size):
                yield (val_shift, val_ds, train_shift, train_ds)
    
    def __len__(self):
        return self.total_batches
    
# numel_levels = [*[ 100*(i + 1) for i in range(10)], *[ 1000*(i + 1) for i in range(10)], *[ 10000*(i + 1) for i in range(10)]]        
# numel_levels_with_ids = list(enumerate(numel_levels))

def pick_modules_and_split_size(active_modules_sorted: list[tuple[str, int, torch.nn.Parameter]], train_num_samples: int, val_num_samples: int, 
                            method_memory_koef: float = 1.0,
                            memory_delta: float = 0.5,
                            device = "cuda", 
                            force_val_size: bool = False):
    ''' Computes necessary memory for the grads and outputs suggestion for the adjusted dataset size
        Param active_modules - list of module_name and its size in bytes
        Param method_memory_koef allows to scale memory requirement, for instance hessian-free influence would not require more memory than grads,
        but exact influence computation would require. 
        Param memory_delta - in GB, memory to leave free 
    '''

    # the atomic unit of grad computation is one parameter tensor of the model - therefore we pick the biggest module
    # max_size_module_name, max_module_byte_size = max([(param_name, param.numel() * (torch.finfo(param.dtype).bits // 8)) for param_name, param in model.named_parameters()], key=lambda x: x[1])
    # estimated_memory_size = max_module_byte_size * (train_size + val_size) * method_memory_koef
    allocated_memory = torch.cuda.memory_allocated(device)
    total_memory_bytes = torch.cuda.get_device_properties(device).total_memory - allocated_memory - (memory_delta * 1024 ** 3)
    total_memory_GB = total_memory_bytes / 1024 ** 3
    total_num_samples = train_num_samples + val_num_samples

    selected_byte_size = 0 
    selected_numel = 0
    selected_index = 0
    estimated_num_samples = total_num_samples

    while selected_index < len(active_modules_sorted):
        cur_mod_byte_size = active_modules_sorted[selected_index][1]
        cur_numel = active_modules_sorted[selected_index][2].numel()
        selected_byte_size += cur_mod_byte_size
        selected_numel += cur_numel
        estimated_num_samples = round(total_memory_bytes / (selected_byte_size * method_memory_koef))
        selected_index += 1
        if estimated_num_samples <= total_num_samples:
            if estimated_num_samples == 0:
                if selected_byte_size == 0:
                    estimated_num_samples = total_num_samples
                else:
                    estimated_num_samples = round(total_memory_bytes / (selected_byte_size * method_memory_koef))
                selected_byte_size -= cur_mod_byte_size
                selected_numel -= cur_numel
                selected_index -= 1        
            break
        
    # too_big_WARN = False
    if selected_index == 0: # module at all do noto fit this GPU - just allow to crash 
        selected_byte_size = active_modules_sorted[0][1]
        selected_numel = active_modules_sorted[0][2].numel()
        selected_index = 1     
        # too_big_WARN = True 

    if force_val_size:
        estimated_val_size = val_num_samples
        estimated_train_size = max(1, estimated_num_samples - estimated_val_size)
    else:
        if estimated_num_samples >= total_num_samples:
            estimated_num_samples = total_num_samples   
            estimated_train_size = train_num_samples
            estimated_val_size = val_num_samples
        else:
            estimated_train_size = round(train_num_samples * estimated_num_samples / total_num_samples)
            estimated_val_size = estimated_num_samples - estimated_train_size

        if estimated_val_size == 0:
            estimated_train_size -= 1
            estimated_val_size = 1

        if estimated_train_size == 0:
            # not enough memory to process
            estimated_train_size = 1

    selected_module_names = [module[0] for module in active_modules_sorted[:selected_index]]
    selected_module_names_str = ','.join(selected_module_names)

    selected_GB_size = round(selected_byte_size * (estimated_train_size + estimated_val_size) / 1024.0 ** 3)
    # if too_big_WARN:
    #     print(f"WARNING: Module {selected_module_names_str} is too big {selected_GB_size}GB for device {device} with memory {total_memory_GB}GB (delta {memory_delta}GB)")

    train_num_batches = train_num_samples // estimated_train_size + (1 if train_num_samples % estimated_train_size > 0 else 0)
    val_num_batches = val_num_samples // estimated_val_size + (1 if val_num_samples % estimated_val_size > 0 else 0)
    print(f"Modules: {selected_module_names_str} {selected_numel}\nSize: {selected_GB_size:.2f}GB/{total_memory_GB:.2f}GB\nTrain-val split: {estimated_train_size}/{train_num_samples} ({train_num_batches} splits), {estimated_val_size}/{val_num_samples} ({val_num_batches} splits). Total splits {train_num_batches * val_num_batches}")
    return (selected_index, estimated_train_size, estimated_val_size)

class CurrentActiveModules:
    def __init__(self, cur_active_modules: list[tuple[str, int, torch.nn.Parameter]], active_modules: list[tuple[str, int, torch.nn.Parameter]]):
        self.cur_active_modules = cur_active_modules
        self.all_active_modules = active_modules
        self.cur_params = [param for _, _, param in self.cur_active_modules]
        self.enumerated_names = [name for name, _, _ in self.cur_active_modules]
        pass

    def __enter__(self):
        for _, _, param in self.all_active_modules:
            param.requires_grad = False
        for _, _, param in self.cur_active_modules:
            param.requires_grad = True

    def __exit__(self, exc_type, exc_value, traceback):
        for _, _, param in self.all_active_modules:
            param.requires_grad = True

    def numel(self):
        return sum([param.numel() for _, _, param in self.cur_active_modules])
    
    def alloc_grads(self, num_samples):
        ''' Preserves the shape of params but adds dimension for samples '''
        new_grads = [torch.zeros((num_samples, *param.shape), dtype = param.dtype, device=param.device) for _, _, param in self.cur_active_modules ]
        return new_grads
    
    def get_cur_params(self):
        return self.cur_params
    
    def enumerate_names(self):
        return enumerate(self.enumerated_names)
    
def get_infl_loader(dataset: Dataset, collator):
    dataloader = DataLoader(dataset,
                    shuffle=False, 
                    collate_fn=collator,
                    batch_size=1)
    return dataloader

def set_grad_values(all_grads: list[torch.Tensor], model: torch.nn.Module, active_params: list, dataloader: DataLoader, device = "cuda",
                        autoregressive: bool = False):     
    for sampleId, batch in enumerate(dataloader):
        model.zero_grad()
        if autoregressive: # labels are inputs - teacher forcing
            batch["labels"] = batch["input_ids"]
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        for g, p in zip(all_grads, active_params):
            g[sampleId] = p.grad
    model.zero_grad()

# here we assume that positive 'infl' is good and negative is 'bad'

def matrix_hf_fn(int_view: torch.Tensor, val_grad: torch.Tensor, train_grad: torch.Tensor, **_) -> None:
    val_grad_flat = val_grad.view(val_grad.shape[0], -1)
    train_grad_flat = train_grad.view(train_grad.shape[0], -1)
    tmp_prods = torch.einsum('ik,jk->ij', val_grad_flat, train_grad_flat)
    # https://github.com/pytorch/pytorch/issues/75701
    # NOTE: out param is not supported yet
    int_view[:] = tmp_prods
    del tmp_prods, val_grad_flat, train_grad_flat
    pass 

def matrix_cos_fn(int_view: torch.Tensor, val_grad: torch.Tensor, train_grad: torch.Tensor, **_) -> None:
    val_grad_flat = val_grad.view(val_grad.shape[0], -1)
    train_grad_flat = train_grad.view(train_grad.shape[0], -1)
    tmp_prods = torch.einsum('ik,jk->ij', val_grad_flat, train_grad_flat)
    int_view[:] = tmp_prods / torch.norm(train_grad_flat, dim=-1) / torch.norm(val_grad_flat, dim=-1).view(-1, 1)
    del tmp_prods
    pass

def matrix_cov_fn(int_view: torch.Tensor, val_grad: torch.Tensor, train_grad: torch.Tensor, **_) -> None:
    val_grad_flat = val_grad.view(val_grad.shape[0], -1)
    train_grad_flat = train_grad.view(train_grad.shape[0], -1)
    train_means = torch.mean(train_grad_flat, dim=-1)
    val_means = torch.mean(val_grad_flat, dim=-1)
    train_centered = train_grad_flat - train_means.view(-1, 1)
    val_centered = val_grad_flat - val_means.view(-1, 1)
    del train_means, val_means
    tmp_prods = torch.einsum('ik,jk->ij', val_centered, train_centered) 
    int_view[:] = tmp_prods / (train_centered.shape[-1] - 1)
    del train_centered, val_centered, tmp_prods
    pass 

def matrix_datainf_one_sample_fn(int_view: torch.Tensor, val_grad: torch.Tensor, train_grad: torch.Tensor, *, lambda_const_param = 10, **_) -> None:
    val_grad_flat = val_grad.view(val_grad.shape[0], -1)
    train_grad_flat = train_grad.view(train_grad.shape[0], -1)
    module_numel = train_grad_flat.shape[-1]
    denom = torch.einsum('ki,ki->k', train_grad_flat, train_grad_flat)
    tmp_prod = torch.einsum('ik,jk->ij', val_grad_flat, train_grad_flat)
    tmp_prod /= denom
    tmp_prod *= (module_numel / ((1 / lambda_const_param) + module_numel)) 
    del denom 
    int_view[:] = tmp_prod
    del tmp_prod
    pass

def matrix_datainf_continuation(module_int_matrices: dict[str, torch.Tensor], module_val_train_products: dict[str, torch.Tensor], 
                                    module_train_train_products: dict[str, torch.Tensor], module_num_params: dict[str, int], 
                                    lambda_const_param, use_orig_def, **_) -> None:
    for module_name, int_matrix in module_int_matrices.items():
        num_train = int_matrix.shape[1]
        num_params = module_num_params[module_name]
        val_train_prods = module_val_train_products[module_name]
        train_train_prods = module_train_train_products[module_name]
        train_train_diag = torch.diag(train_train_prods)
        lambda_const = torch.sum(train_train_diag) / (num_train * lambda_const_param * num_params)
        lambda_const_n = lambda_const * num_train
        train_train_diag += lambda_const
        train_train_prods /= train_train_diag
        tmp_prods = torch.einsum("ji,ki -> jk", val_train_prods, train_train_prods) 
        tmp_prods /= lambda_const_n
        if use_orig_def:
            val_train_prods /= lambda_const_n
        else:
            val_train_prods /= lambda_const #NOTE: here we have difference from original impl which divides by n in addition
        int_matrix[:] = val_train_prods - tmp_prods # we ignore negation 
        del tmp_prods, train_train_diag
        del val_train_prods, train_train_prods
        pass

# NOTE: it only works when train_grad contains all training samples
# NOTE: we need to iterate through training samples twice if we want to implement datainf
def matrix_datainf_fn(__: torch.Tensor, val_grad: torch.Tensor, train_grad: torch.Tensor, lambda_const_param = 10,
                        *, use_orig_def = False, module_name: str, infl_context: dict, train_shift: int, val_shift: int, full_train_size: int, full_val_size: int,
                        **_) -> None:
    ''' Here we just prepare necessary vector products, which will be used in matrix_datainf_continuation '''
    if "continuation" not in infl_context:
        infl_context["continuation"] = matrix_datainf_continuation
        infl_context["module_val_train_products"] = {}
        infl_context["module_train_train_products"] = {}
        infl_context["module_num_params"] = {}
        infl_context["lambda_const_param"] = lambda_const_param
        infl_context["use_orig_def"] = use_orig_def

    val_grad_flat = val_grad.view(val_grad.shape[0], -1)
    train_grad_flat = train_grad.view(train_grad.shape[0], -1)

    if module_name not in infl_context["module_val_train_products"]:
        infl_context["module_val_train_products"][module_name] = torch.zeros((full_val_size, full_train_size), device=train_grad.device, dtype = train_grad.dtype)
        infl_context["module_train_train_products"][module_name] = torch.zeros((full_train_size, full_train_size), device=train_grad.device, dtype = train_grad.dtype)
    infl_context["module_num_params"].setdefault(module_name, train_grad_flat.shape[1])
    val_batch_size = val_grad_flat.shape[0]
    train_batch_size = train_grad_flat.shape[0]
    tmp_prod1 = torch.einsum('ik,jk->ij', val_grad_flat, train_grad_flat)
    infl_context["module_val_train_products"][module_name][val_shift:val_shift + val_batch_size, train_shift:train_shift + train_batch_size] = tmp_prod1
    del tmp_prod1
    # NOTE: to compute the following we need additional loop over training samples --> TODO: finish if necessary, but for now
    tmp_prod2 = torch.einsum('ik,jk->ij', train_grad_flat, train_grad_flat)
    infl_context["module_train_train_products"][module_name][train_shift:train_shift + train_batch_size, train_shift:train_shift + train_batch_size] = tmp_prod2
    del tmp_prod2
    pass

from sklearn.svm import OneClassSVM
def outlier_fn(int_view: torch.Tensor, val_grad: torch.Tensor, train_grad: torch.Tensor, *, 
                detector_fn = OneClassSVM, full_val_size, autoregressive=True, **_) -> None:

    assert full_val_size == val_grad.shape[0], f"To fit grad region we need all validation gradients. Given {val_grad.shape[0]}, total {full_val_size}"
    val_grad_flat = val_grad.view(val_grad.shape[0], -1)
    val_grad_flat_tmp = val_grad_flat.float()
    val_grad_np = val_grad_flat_tmp.cpu().numpy()
    train_grad_flat = train_grad.view(train_grad.shape[0], -1)
    train_grad_flat_tmp = train_grad_flat.float()
    train_grad_np = train_grad_flat_tmp.cpu().numpy()
    # quick hack - in rereality we need to pass to influence the groups of validation samples that we consider to be coaligned in grad space intuitivelly same label
    if autoregressive: # 10 groups of 
        n_class = 10
        n_val_per_class = 10
        for i_class in range(n_class):
            class_val_grad_np = val_grad_np[i_class * n_val_per_class:(i_class + 1) * n_val_per_class]
            detector = detector_fn()
            detector.fit(class_val_grad_np)
            scores = detector.decision_function(train_grad_np)
            scores_tmp = torch.tensor(scores, device=int_view.device, dtype=int_view.dtype)
            int_view[i_class * n_val_per_class:(i_class + 1) * n_val_per_class] = scores_tmp            
            pass
    else:
        detector = detector_fn()
        detector.fit(val_grad_np)
        scores = detector.decision_function(train_grad_np)
        scores_tmp = torch.tensor(scores, device=int_view.device, dtype=int_view.dtype)
        int_view[:] = scores_tmp
    del val_grad_flat, train_grad_flat, val_grad_flat_tmp, train_grad_flat_tmp, scores_tmp
    pass 

# def matrix_lissa_fn(int_view: torch.Tensor, val_grad: torch.Tensor, train_grad: torch.Tensor, lambda_const_param = 10, n_iteration=10, alpha_const=1.) -> None:
#     n_train = train_grad.shape[0]
#     module_train_grad_squares = train_grad ** 2
#     lambda_const = torch.mean(module_train_grad_squares) / lambda_const_param

#     # hvp computation
#     running_hvp = val_grad
#     for _ in range(n_iteration):
#         new_running_hvp = val_grad + running_hvp - alpha_const * torch.sum((torch.sum(train_grad * running_hvp, dim=-1).view(-1, 1) * train_grad - lambda_const * running_hvp) / n_train, dim=0)
#         del running_hvp
#         running_hvp = new_running_hvp

#     module_infl_values = running_hvp * train_grad
#     int_view[:] = module_infl_values.sum(dim=-1)
#     del module_train_grad_squares, running_hvp, module_infl_values

def common_we(int_view: torch.Tensor, val_grad: torch.Tensor, train_grad: torch.Tensor, *, base_method_fn, train_shift, val_shift, common_tokens: dict[tuple[int, int], dict[int, tuple[int, int]]] = {}, **kwargs) -> None:
    ''' 
        Works only on embedding layer grads: i.e. 50K * 768 dense representation for BERT models 
        Modifies base_method_fn by zeroing out gradients for tokens that are not common between train and val samples
        Param common_tokens is dict of (train_id, val_id) to dict of token id to counts in (train_occur, val_occur), common to train and val sample
    '''
    # NOTE: it is infeasible to allocate mask tensor = 50K * 1K * 5K * 8 ~ 2K GB of memory 
    # grad_mask = torch.zeros((train_grad.shape[0], val_grad.shape[0], train_grad.shape[1]), dtype=train_grad.device, device=train_grad.device)
    uncommon_mask = torch.ones(train_grad.shape[1], device=train_grad.device, dtype=torch.bool)
    train_grad_clone = train_grad.clone()
    val_grad_clone = val_grad.clone()
    for train_id in range(train_grad.shape[0]):
        for val_id in range(val_grad.shape[0]):
            common_token_ids_list = list(common_tokens.get((train_shift + train_id, val_shift + val_id), {}).keys())
            common_token_ids = torch.tensor(common_token_ids_list, device=train_grad.device, dtype=torch.int)
            uncommon_mask[:] = True
            uncommon_mask[common_token_ids] = False 
            train_grad_clone[train_id, uncommon_mask] = 0
            val_grad_clone[val_id, uncommon_mask] = 0
            del common_token_ids
    base_method_fn(int_view, val_grad_clone, train_grad_clone, train_shift=train_shift, val_shift=val_shift, common_tokens = common_tokens, **kwargs)
    del uncommon_mask, train_grad_clone, val_grad_clone
    pass

def common_we_topk(int_view: torch.Tensor, val_grad: torch.Tensor, train_grad: torch.Tensor, *, base_method_fn, train_shift, val_shift, common_tokens: dict[tuple[int, int], dict[int, tuple[int, int]]] = {}, topk = 10, **kwargs) -> None:
    ''' 
        Similar to common_we, but selects topk grads by norm.
        https://proceedings.neurips.cc/paper_files/paper/2022/file/d07022783ff6f7bf7a288c207b7dcbd1-Paper-Conference.pdf
        topk = {10,20,...,100}
    '''
    # NOTE: it is infeasible to allocate mask tensor = 50K * 1K * 5K * 8 ~ 2K GB of memory 
    # grad_mask = torch.zeros((train_grad.shape[0], val_grad.shape[0], train_grad.shape[1]), dtype=train_grad.device, device=train_grad.device)
    train_grad_repr = torch.zeros((train_grad.shape[0], topk, train_grad.shape[-1]), device=train_grad.device, dtype=train_grad.dtype)
    val_grad_repr = torch.zeros((val_grad.shape[0], topk, val_grad.shape[-1]), device=val_grad.device, dtype=val_grad.dtype)
    for train_id in range(train_grad.shape[0]):
        for val_id in range(val_grad.shape[0]):
            common_token_ids_dict = common_tokens.get((train_shift + train_id, val_shift + val_id), {})
            common_token_ids = torch.tensor(list(common_token_ids_dict.keys()), device=train_grad.device, dtype=torch.int)
            train_denom = torch.tensor([common_token_ids_dict[token_id][0] for token_id in common_token_ids_dict.keys()], device=train_grad.device, dtype=train_grad.dtype)
            train_token_scores = torch.norm(train_grad[train_id, common_token_ids], dim = -1) / train_denom            
            train_token_ids_ids = torch.argsort(train_token_scores, descending=True)[:topk]
            train_token_ids = common_token_ids[train_token_ids_ids]

            val_denom = torch.tensor([common_token_ids_dict[token_id][1] for token_id in common_token_ids_dict.keys()], device=train_grad.device, dtype=val_grad.dtype)
            val_token_scores = torch.norm(val_grad[val_id, common_token_ids], dim = -1) / val_denom
            val_token_ids_ids = torch.argsort(val_token_scores, descending=True)[:topk]
            val_token_ids = common_token_ids[val_token_ids_ids]
            train_grad_repr[train_id, :len(train_token_ids)] = train_grad[train_id, train_token_ids]
            val_grad_repr[val_id, :len(val_token_ids)] = val_grad[val_id, val_token_ids]
            del common_token_ids, train_denom, train_token_scores, train_token_ids_ids, val_denom, val_token_scores, val_token_ids_ids
    base_method_fn(int_view, val_grad_repr, train_grad_repr, train_shift=train_shift, val_shift=val_shift, common_tokens = common_tokens, **kwargs)
    del train_grad_repr, val_grad_repr
    pass

# def matrix_tf_idf_head():
#     ''' 
#         Executes on head, tf_idf of train sample on val sample tokens 
#     '''
#     pass 

matrix_infl_methods = {
    "hf": matrix_hf_fn,
    "hf_we_": partial(common_we, base_method_fn=matrix_hf_fn),
    "hf_we_topk_10": partial(common_we_topk, base_method_fn=matrix_hf_fn, topk=10),
    # "hw_we_topk_20": partial(common_we_topk, base_method_fn=matrix_hf_fn, topk=20),
    "cos": matrix_cos_fn,
    # "cos_we": partial(common_we, base_method_fn=matrix_cos_fn),
    "cov": matrix_cov_fn,
    # "cov_we": partial(common_we, base_method_fn=matrix_cov_fn),
    "datainf_one": matrix_datainf_one_sample_fn,
    'datainf': matrix_datainf_fn,
    'datainf0': partial(matrix_datainf_fn, use_orig_def=True),
    'outlier': outlier_fn,
    # "datainf_we": partial(common_we, base_method_fn=matrix_datainf_fn),
    # "lissa": matrix_lissa_fn,
    # "lissa_we": partial(common_we, base_method_fn=matrix_lissa_fn),
}

def is_embedding_module(module_name: str):
    return ('.word_embeddings.' in module_name) or ('.embed_tokens.' in module_name)

def set_logits(task = "qnli", m_prefix: str = 'm_bl', batch_size=32, device = "cuda",
                set_name = 'train', softmaxed = False):
    ''' Routine is added post-factum to compute and save train set logits for given checkpoint '''
    model_path = os.path.join(cwd, f'{m_prefix}_{task}_{seed}')

    # todo check if reequires_grad is attached to embedding
    lora_model = load_pretrained_LORA_model(model_name_or_path=model_path)
    lora_model.to(device)
    lora_model.eval()  

    dataset_path = os.path.join(cwd, f'd_{task}_{seed}')

    datasets = load_from_disk(dataset_path) # add validation dataset 

    trainset = datasets[set_name]#.select(range(100))
    train_labels = trainset['labels']
    trainset = trainset.remove_columns(['noise'])
    tokenizer = load_tokenizer(model_path)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", return_tensors="pt")  

    train_dataloader = DataLoader(trainset,
                    shuffle=False, 
                    collate_fn=collator,
                    batch_size=batch_size)
    
    # predict loop 
    logits = None
    train_shift = 0
    for batch in tqdm(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = lora_model(**batch)
            if logits is None:
                logits = torch.zeros((len(train_dataloader.dataset), lora_model.config.num_labels), device=outputs.logits.device, dtype = outputs.logits.dtype)
            batch_size = outputs.logits.shape[0]
            logits[train_shift:train_shift+batch_size] = outputs.logits   
            train_shift += batch_size         

    # save logits tensor to model folder 
    train_preds = torch.argmax(logits, dim=-1)
    accuracy = sum([1 if train_labels[i] == train_preds[i] else 0 for i in range(len(train_labels))]) / len(train_labels)
    print(f"Accuracy: {accuracy:.4f}")
    if softmaxed:
        logits = torch.nn.functional.softmax(logits, dim=-1)
        prefix = "probas"
    else:
        prefix = "logits"
    logits_path = os.path.join(model_path, f'{set_name}_{prefix}.pt')
    torch.save(logits, logits_path)
    pass

# Mem koef NOTE: hf (1.1, 0.3), hf_we_ (2, 0.3), hf_we_topk (2, 0.3), cos (1.1, 0.3), cov (2, 0.3), datainf_one (1.1, 0.3), datainf (2, 0.3), 
def infl_matrix(task = 'mrpc', methods = "hf,hf_we_,hw_we_topk_10,cos,cov,datainf_one,datainf", mem_koef: float = 2.0, mem_delta: float = 0.3,
                i_prefix='i_bl', m_prefix='m_bl'):
    config_path = os.path.join(cwd, f'c_{task}_{seed}.json')
    with open(config_path, 'r') as file:
        config = json.load(file)

    # method_list = methods.split(',')
    device = config['device']
    unfreeze_regex = config.get('unfreeze_regex', None)

    # if with_grads:
    #     gradients = grads(task = task, return_grads=True, config=config, no_val=self_influence)
    # else:
    #     gradients = torch.load(os.path.join(cwd, f'g_{task}_{seed}.pt'))

    model_path = os.path.join(cwd, f'{m_prefix}_{task}_{seed}')

    # todo check if reequires_grad is attached to embedding
    lora_model = load_pretrained_LORA_model(model_name_or_path=model_path, unfreeze_modules_regex=unfreeze_regex)
    lora_model.to(device)
    lora_model.eval()  

    # module_filter = ['lora_A', 'lora_B', 'modules_to_save.default.out_proj.weight']
    # # module_filter = ['modules_to_save.default.out_proj.weight']
    # for param_name, param_param in lora_model.named_parameters():
    #     if any([module in param_name for module in module_filter]):
    #         param_param.requires_grad = True
    #     else:
    #         param_param.requires_grad = False

    active_modules = [(name, param.numel() * (torch.finfo(param.dtype).bits // 8), param) for name, param in lora_model.named_parameters() if param.requires_grad]
    active_modules.sort(key=lambda x: x[1], reverse=True)

    dataset_path = os.path.join(cwd, f'd_{task}_{seed}')

    datasets = load_from_disk(dataset_path) # add validation dataset 

    trainset = datasets['train']#.select(range(100))
    trainset = trainset.remove_columns(['noise'])
    tokenizer = load_tokenizer(config['tokenizer_name'])
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", return_tensors="pt")  

    # method_fn = matrix_infl_methods[method]
    method_names = [name for name in methods.split(',') if name in matrix_infl_methods]

    valset = datasets['infl']#.select(range(100))

    common_tokens = {}

    if any('_we_' in method_name for method_name in method_names):
        common_tokens_ds = os.path.join(cwd, f'common_tokens_{task}_{seed}.pkl')
        should_recompute = False
        try:
            with open(common_tokens_ds, 'rb') as file:
                common_tokens = pickle.load(file)
        except:
            should_recompute = True
        if should_recompute:
            # need to compute common tokens         
            vocab_remap = lora_model.get_input_embeddings().vocab_mapping.tolist()
            train_token_counts = {}
            for train_id in range(len(trainset)):
                cur_train_token_counts = train_token_counts.setdefault(train_id, {})
                for token_id in trainset[train_id]['input_ids']:
                    remapped_token_id = vocab_remap[token_id]
                    cur_train_token_counts[remapped_token_id] = cur_train_token_counts.get(remapped_token_id, 0) + 1
            val_token_counts = {}
            for val_id in range(len(valset)):
                cur_val_token_counts = val_token_counts.setdefault(val_id, {})
                for token_id in valset[val_id]['input_ids']:
                    remapped_token_id = vocab_remap[token_id]
                    cur_val_token_counts[remapped_token_id] = cur_val_token_counts.get(remapped_token_id, 0) + 1
            for train_id, train_token_count in train_token_counts.items():
                for val_id, val_token_count in val_token_counts.items():
                    common_token_ids = set(train_token_count.keys()).intersection(val_token_count.keys())
                    common_tokens[(train_id, val_id)] = {token_id: (train_token_count[token_id], val_token_count[token_id]) for token_id in common_token_ids}
            with open(common_tokens_ds, 'wb') as file:
                pickle.dump(common_tokens, file)

    methods_without_we = ['datainf', 'datainf0', 'outlier' ]
    interaction_matrices = {method_name: {module_name: torch.zeros((len(valset), len(trainset)), dtype=torch.bfloat16, device=device)                                             
                                            for module_name, _, module_params in active_modules
                                            if ((method_name in methods_without_we) and (module_params.numel() < 100000)) or
                                                ((method_name not in methods_without_we) and ('_we_' not in method_name)) or 
                                                ((method_name not in methods_without_we) and ('_we_' in method_name) and is_embedding_module(module_name))} 
                                for method_name in method_names}
    
    interaction_modules = set([module_name for int_matrices in interaction_matrices.values() for module_name in int_matrices.keys()])

    all_active_modules = [(name, size, params) for name, size, params in active_modules if name in interaction_modules]

    force_val_size_for_methods = ['outlier']
    force_val_size = any(method_name in force_val_size_for_methods for method_name in method_names)

    while len(all_active_modules) > 0:
        selected_module_count, train_size, test_size = pick_modules_and_split_size(
            all_active_modules, len(trainset), len(valset), method_memory_koef=mem_koef, 
            memory_delta=mem_delta, device=device, force_val_size=force_val_size)
        cur_active_modules = all_active_modules[:selected_module_count]
        all_active_modules = all_active_modules[selected_module_count:]
        cur_params = CurrentActiveModules(cur_active_modules, active_modules)
        infl_contexts = {method_name: {} for method_name in method_names} # methods could store arbitrary data here between calls to method_fn
        with cur_params:
            train_val_splits = DatasetSplits(trainset, train_size, valset, test_size)

            val_grads = None

            for val_shift, cur_val_dataset, train_shift, cur_train_dataset in tqdm(train_val_splits): 
                if train_shift == 0:
                    if val_grads is not None:
                        del val_grads
                    val_grads = cur_params.alloc_grads(len(cur_val_dataset))
                    val_dataloader = get_infl_loader(cur_val_dataset, collator)
                    set_grad_values(val_grads, lora_model, cur_params.get_cur_params(), val_dataloader, device=device)
                train_grads = cur_params.alloc_grads(len(cur_train_dataset))
                train_dataloader = get_infl_loader(cur_train_dataset, collator)
                set_grad_values(train_grads, lora_model, cur_params.get_cur_params(), train_dataloader, device=device)
                for method_name, module_int_matrices in interaction_matrices.items():
                    for cur_module_id, module_name in cur_params.enumerate_names():
                        if module_name in module_int_matrices:
                            cur_int_matrix_view = module_int_matrices[module_name][val_shift:val_shift + len(cur_val_dataset), train_shift:train_shift + len(cur_train_dataset)]
                            method_fn = matrix_infl_methods[method_name]
                            method_fn(cur_int_matrix_view, val_grads[cur_module_id], train_grads[cur_module_id],
                                    module_name = module_name, infl_context = infl_contexts[method_name],
                                    train_shift=train_shift, val_shift=val_shift, 
                                    full_train_size=len(trainset), full_val_size=len(valset),
                                    common_tokens=common_tokens)
                del train_grads
                torch.cuda.empty_cache() 
            if val_grads is not None:
                del val_grads
            for method_name, method_infl_context in infl_contexts.items():
                if 'continuation' in method_infl_context:
                    cur_int_matrices = interaction_matrices[method_name]
                    method_infl_context['continuation'](cur_int_matrices, **method_infl_context)
        pass 


    for method_name, infl_matrices in interaction_matrices.items():
        infl_path = os.path.join(cwd, f'{i_prefix}_{method_name}_{task}_{seed}.pt')
        torch.save(infl_matrices, infl_path)

    # with open(config_path, 'w') as file:
    #     fcntl.flock(file, fcntl.LOCK_EX)
    #     json.dump(config, file)
    #     fcntl.flock(file, fcntl.LOCK_UN)    


def infl_matrix_causal(task='sentense', 
                       methods="hf,hf_we_,hw_we_topk_10,cos,cov,datainf_one,datainf", 
                       mem_koef: float = 2.0, 
                       mem_delta: float = 0.3,
                       i_prefix='i_ds', 
                       checkpoint='m_bl',
                       dataset:str="",
                       dataset_path:str=".",
                       device='cuda'):

    # todo check if reequires_grad is attached to embedding

    model_path = f"{cwd}/{checkpoint}_{seed}"
    print(f"Loading model {model_path}...")
    lora_model = load_causal_LORA_model(model_name_or_path=model_path)
    lora_model.to(device)
    lora_model.eval()  

    print(f"Loading datasets {dataset} from {dataset_path}...")
    trainset = Dataset.load_from_disk(f"{dataset_path}/{dataset}_train.hf")
    valset = Dataset.load_from_disk(f"{dataset_path}/{dataset}_test.hf")

    tokenizer = load_causal_tokenizer(model_path)   
    
    datasets = causal_tokenize(tokenizer, device=device, dataset=dataset, train=trainset, validation=valset)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", return_tensors="pt")  

    # torch.backends.cuda.enable_mem_efficient_sdp(False)
    # torch.backends.cuda.enable_flash_sdp(False)

    # module_filter = ['lora_A', 'lora_B', 'modules_to_save.default.out_proj.weight']
    # # module_filter = ['modules_to_save.default.out_proj.weight']
    # for param_name, param_param in lora_model.named_parameters():
    #     if any([module in param_name for module in module_filter]):
    #         param_param.requires_grad = True
    #     else:
    #         param_param.requires_grad = False

    active_modules = [(name, param.numel() * (torch.finfo(param.dtype).bits // 8), param) for name, param in lora_model.named_parameters() if param.requires_grad]
    active_modules.sort(key=lambda x: x[1], reverse=True)

    # dataset_path = os.path.join(cwd, f'd_{task}_{seed}')

    # tokenizer = load_tokenizer(config['tokenizer_name'])
    # collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", return_tensors="pt")  

    # method_fn = matrix_infl_methods[method]
    method_names = [name for name in methods.split(',') if name in matrix_infl_methods]

    trainset = datasets['train']#.select(range(100))
    valset = datasets['validation']#.select(range(100))

    common_tokens = {}

    # if any('_we_' in method_name for method_name in method_names):
    #     common_tokens_ds = os.path.join(cwd, f'common_tokens_{task}_{seed}.pkl')
    #     should_recompute = False
    #     try:
    #         with open(common_tokens_ds, 'rb') as file:
    #             common_tokens = pickle.load(file)
    #     except:
    #         should_recompute = True
    #     if should_recompute:
    #         # need to compute common tokens         
    #         vocab_remap = lora_model.get_input_embeddings().vocab_mapping.tolist()
    #         train_token_counts = {}
    #         for train_id in range(len(trainset)):
    #             cur_train_token_counts = train_token_counts.setdefault(train_id, {})
    #             for token_id in trainset[train_id]['input_ids']:
    #                 remapped_token_id = vocab_remap[token_id]
    #                 cur_train_token_counts[remapped_token_id] = cur_train_token_counts.get(remapped_token_id, 0) + 1
    #         val_token_counts = {}
    #         for val_id in range(len(valset)):
    #             cur_val_token_counts = val_token_counts.setdefault(val_id, {})
    #             for token_id in valset[val_id]['input_ids']:
    #                 remapped_token_id = vocab_remap[token_id]
    #                 cur_val_token_counts[remapped_token_id] = cur_val_token_counts.get(remapped_token_id, 0) + 1
    #         for train_id, train_token_count in train_token_counts.items():
    #             for val_id, val_token_count in val_token_counts.items():
    #                 common_token_ids = set(train_token_count.keys()).intersection(val_token_count.keys())
    #                 common_tokens[(train_id, val_id)] = {token_id: (train_token_count[token_id], val_token_count[token_id]) for token_id in common_token_ids}
    #         with open(common_tokens_ds, 'wb') as file:
    #             pickle.dump(common_tokens, file)

    methods_without_we = ['datainf', 'datainf0', 'outlier' ] #'kronfl' ]
    interaction_matrices = {method_name: {module_name: torch.zeros((len(valset), len(trainset)), dtype=torch.bfloat16, device=device)                                             
                                            for module_name, _, module_params in active_modules
                                            if ((method_name in methods_without_we) and (module_params.numel() < 100000)) or
                                                ((method_name not in methods_without_we) and ('_we_' not in method_name)) or 
                                                ((method_name not in methods_without_we) and ('_we_' in method_name) and is_embedding_module(module_name))} 
                                for method_name in method_names}
    
    interaction_modules = set([module_name for int_matrices in interaction_matrices.values() for module_name in int_matrices.keys()])

    all_active_modules = [(name, size, params) for name, size, params in active_modules if name in interaction_modules]

    force_val_size_for_methods = ['outlier']
    force_val_size = any(method_name in force_val_size_for_methods for method_name in method_names)

    print(f"Running influences...")

    while len(all_active_modules) > 0:
        selected_module_count, train_size, test_size = pick_modules_and_split_size(
            all_active_modules, len(trainset), len(valset), method_memory_koef=mem_koef, 
            memory_delta=mem_delta, device=device, force_val_size=force_val_size)
        cur_active_modules = all_active_modules[:selected_module_count]
        all_active_modules = all_active_modules[selected_module_count:]
        cur_params = CurrentActiveModules(cur_active_modules, active_modules)
        infl_contexts = {method_name: {} for method_name in method_names} # methods could store arbitrary data here between calls to method_fn
        with cur_params:
            train_val_splits = DatasetSplits(trainset, train_size, valset, test_size)

            val_grads = None

            for val_shift, cur_val_dataset, train_shift, cur_train_dataset in tqdm(train_val_splits): 
                if train_shift == 0:
                    if val_grads is not None:
                        del val_grads
                    val_grads = cur_params.alloc_grads(len(cur_val_dataset))
                    val_dataloader = get_infl_loader(cur_val_dataset, collator)
                    set_grad_values(val_grads, lora_model, cur_params.get_cur_params(), val_dataloader, device=device,
                        autoregressive=True)
                train_grads = cur_params.alloc_grads(len(cur_train_dataset))
                train_dataloader = get_infl_loader(cur_train_dataset, collator)
                set_grad_values(train_grads, lora_model, cur_params.get_cur_params(), train_dataloader, device=device,
                    autoregressive=True)
                for method_name, module_int_matrices in interaction_matrices.items():
                    for cur_module_id, module_name in cur_params.enumerate_names():
                        if module_name in module_int_matrices:
                            cur_int_matrix_view = module_int_matrices[module_name][val_shift:val_shift + len(cur_val_dataset), train_shift:train_shift + len(cur_train_dataset)]
                            method_fn = matrix_infl_methods[method_name]
                            method_fn(cur_int_matrix_view, val_grads[cur_module_id], train_grads[cur_module_id],
                                    module_name = module_name, infl_context = infl_contexts[method_name],
                                    train_shift=train_shift, val_shift=val_shift, 
                                    full_train_size=len(trainset), full_val_size=len(valset),
                                    common_tokens=common_tokens, autoregressive=True)
                del train_grads
                torch.cuda.empty_cache() 
            if val_grads is not None:
                del val_grads
            for method_name, method_infl_context in infl_contexts.items():
                if 'continuation' in method_infl_context:
                    cur_int_matrices = interaction_matrices[method_name]
                    method_infl_context['continuation'](cur_int_matrices, **method_infl_context)
        pass 


    for method_name, infl_matrices in interaction_matrices.items():
        infl_path = os.path.join(cwd, f'{i_prefix}_{method_name}_{task}_{seed}.pt')
        print(f"Saving {infl_path}...")
        torch.save(infl_matrices, infl_path)

    # with open(config_path, 'w') as file:
    #     fcntl.flock(file, fcntl.LOCK_EX)
    #     json.dump(config, file)
    #     fcntl.flock(file, fcntl.LOCK_UN)    

class KronfluenceTask(Task):
    def __init__(self, model: torch.nn.Module, autoregressive: bool = False, device: str = "cuda",
                 target_modules: list[str] = ["q_proj", "v_proj", "modules_to_save.default.out_proj"]):
        super().__init__()
        self.autoregressive = autoregressive
        self.device = device
        self.model = model
        self.target_modules = target_modules

    def compute_train_loss(
        self,
        batch: Any,
        model: torch.nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        # TODO: Complete this method.
        if self.autoregressive: # labels are inputs - teacher forcing
            batch["labels"] = batch["input_ids"]
        batch.to(self.device)
        outputs = model(**batch)
        loss = outputs.loss
        return loss

    def compute_measurement(
        self,
        batch: Any,
        model: torch.nn.Module,
    ) -> torch.Tensor:
        return self.compute_train_loss(batch, model)

    def get_influence_tracked_modules(self) -> Optional[List[str]]:
        collected_names = []
        for name, module in self.model.named_modules():
            if any(target_module in name for target_module in self.target_modules) \
                and isinstance(module, torch.nn.Linear) and "base_layer" not in name:
                collected_names.append(name)
        return collected_names

    def get_attention_mask(self, batch: Any) -> Optional[Union[Dict[str, torch.Tensor], torch.Tensor]]:
        # TODO: [Optional] Complete this method.
        if "attention_mask" in batch:
            return batch["attention_mask"]
        return None  # Attention mask not used.

def kronfl(task='sentense', 
            method="ekfac", 
            checkpoint='m_bl',
            dataset:str="",
            dataset_path:str=".",
            device='cuda',
            i_prefix='i_ds'):

    model_path = f"{cwd}/{checkpoint}_{seed}"
    print(f"Loading model {model_path}...")
    tokenizer = load_causal_tokenizer(model_path)
    model = load_causal_LORA_model(model_name_or_path=model_path)
    model.to(device)
    model.eval()  

    print(f"Loading datasets {dataset} from {dataset_path}...")
    train_dataset = Dataset.load_from_disk(f"{dataset_path}/{dataset}_train.hf")
    eval_dataset = Dataset.load_from_disk(f"{dataset_path}/{dataset}_test.hf")
    datasets = causal_tokenize(tokenizer, device=device, dataset=dataset, train=train_dataset, validation=eval_dataset)
    train_dataset = datasets["train"]
    eval_dataset = datasets["validation"]

    fl_task = KronfluenceTask(model, autoregressive = True)

    # Prepare the model for influence computation.
    model = prepare_model(model=model, task=fl_task)
    analyzer = Analyzer(
                    analysis_name=dataset, 
                    model=model, 
                    task=fl_task,
                    output_dir=f"{cwd}/kronfl_{method}_{seed}"
                )

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", return_tensors="pt")  
    dataloader_kwargs = DataLoaderKwargs(collate_fn=collator)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    # Fit all EKFAC factors for the given model.
    factor_args = FactorArguments(strategy=method)
    # factor_args = all_low_precision_factor_arguments(strategy=factor_strategy, dtype=torch.float16)

    print(f"Fitting all factors...")

    analyzer.fit_all_factors(factors_name=method, 
                            dataset=train_dataset,
                            factor_args=factor_args,
                            overwrite_output_dir=False,
                            per_device_batch_size=None,
                            # initial_per_device_batch_size_attempt=512,
                            )

    score_args = ScoreArguments(
        compute_per_module_scores=True
    )
    # score_args = all_low_precision_score_arguments(dtype=torch.bfloat16)

    print(f"Scoring...")

    # Compute all pairwise influence scores with the computed factors.
    analyzer.compute_pairwise_scores(
        score_args = score_args,
        scores_name=method,
        factors_name=method,
        query_dataset=eval_dataset,
        train_dataset=train_dataset,
        # per_device_query_batch_size=16,
        # per_device_train_batch_size=16,
        per_device_query_batch_size=1,
        per_device_train_batch_size=1,
        overwrite_output_dir=True,
    )

    # Load the scores with dimension `len(eval_dataset) x len(train_dataset)`.
    scores = analyzer.load_pairwise_scores(scores_name=method)

    infl_path = os.path.join(cwd, f'{i_prefix}_kr-{method}_{task}_{seed}.pt')
    print(f"Saving {infl_path}...")
    torch.save(scores, infl_path)
    pass 


def load_ds_info(task_in_dir: str, task: str, with_input_ids = False):
    ds_path = os.path.join(task_in_dir, f'd_{task}_{seed}')
    ds = datasets.load_from_disk(ds_path)
    trainset = ds['train']
    noise_list = trainset['noise']
    trainset_labels = trainset['labels']
    inflset = ds['infl']
    inflset_labels = inflset['labels']
    if with_input_ids:
        train_input_ids = trainset['input_ids']
        infl_input_ids = inflset['input_ids']
        return noise_list, trainset_labels, inflset_labels, (train_input_ids, infl_input_ids)
    return noise_list, trainset_labels, inflset_labels

def load_m_info(task_in_dir: str, task:str, m_prefix: str, with_train_logits = False):
    ''' Loads infl samples logits on the specified chehckpoint '''
    logits_m_path = os.path.join(task_in_dir, f'{m_prefix}_{task}_{seed}', "infl_logits.pt")
    logits = torch.load(logits_m_path)
    if with_train_logits:
        train_logits_path = os.path.join(task_in_dir, f'{m_prefix}_{task}_{seed}', "train_logits.pt")
        train_logits = torch.load(train_logits_path)
        return logits, train_logits
    return logits

def ndr(task = "qnli", infl_methods: str = 'datainf', agg_methods: str = 'mean',
            i_prefix: str = 'i_bl', m_prefix: str = 'm_bl', group_file: str = './groups.json',
            include_total = True, ndr_prefix: str = 'ndr_bl', device = "cuda",
            levels:str = "5,10,15,20,25,30,35,40,45,50,60,70,80,90",
            hist_bins = 10):
    levels = levels.split(',')
    levels = [int(level) for level in levels]
    infl_methods = infl_methods.split(',')
    agg_method_names = agg_methods.split(',')
    compute_ndr_metrics_table(cwd, task, group_file,
                                    infl_methods = infl_methods,
                                    agg_method_names = agg_method_names,
                                    include_total = include_total, levels = levels,
                                    m_prefix = m_prefix, i_prefix=i_prefix,
                                    ndr_prefix=ndr_prefix, device = device,
                                    noise_hist_bins = hist_bins)
    pass

from postprocess import agg_methods as agg_method_fns

def scores(task = "qnli", infl_methods: str = 'datainf', agg_methods: str = 'mean', 
                 i_prefix: str = 'i_bl', m_prefix: str = 'm_bl', group_file: str = '',
                 include_total = True, s_prefix: str = 's_bl', device = "cuda"):
    infl_methods = infl_methods.split(',')
    agg_method_names = agg_methods.split(',')
    if group_file != "":
        with open(os.path.join(cwd, group_file), 'r') as file:
            module_groups_regex = json.load(file)
        module_groups_patterns = {name: re.compile(pattern) for name, pattern in module_groups_regex.items()}
    else:
        module_groups_patterns = {}

    noise_list, trainset_labels, inflset_labels = load_ds_info(cwd, task)
    noise_mask = torch.tensor(noise_list, device = device)
    trainset_labels = torch.tensor(trainset_labels, device = device)
    inflset_labels = torch.tensor(inflset_labels, device = device)

    inflset_logits = load_m_info(cwd, task, m_prefix).to(device)
    inflset_preds = torch.argmax(inflset_logits, dim = -1)
    correct_infl_preds = inflset_preds == inflset_labels

    scores_dict = {"noise_mask": noise_mask} #preserve mask to find most infl noise samples

    for infl_method in infl_methods:

        if infl_method == 'rand': # random shuffle 
            scores = torch.rand(trainset_labels.shape[0], dtype = torch.bfloat16, device = device)
            scores_dict[(infl_method, '', '')] = scores
            continue
        if infl_method == 'denoise': # first noise, then shuffled clean
            scores = torch.rand(trainset_labels.shape[0], dtype = torch.bfloat16, device = device)
            scores[noise_mask] = -1.0
            scores_dict[(infl_method, '', '')] = scores
            continue

        file_path = os.path.join(cwd, f'{i_prefix}_{infl_method}_{task}_{seed}.pt')
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist")
            continue
        matrix_dict = torch.load(file_path)        
        module_names = list(matrix_dict.keys())
        module_and_group_names = list(module_names)
        # transpose all matrices 
        for module_name in module_names:
            matrix_dict[module_name] = matrix_dict[module_name].t().to(device) # first dim is train_sample now and second is infl val sample
        if len(module_names) == 0:
            continue
        all_interactions = torch.stack([matrix_dict[module_name] for module_name in module_names], dim = 1)
        for int_matrix in matrix_dict.values():
            del int_matrix
        # now the dimensions is train_sample * module * infl_val_sample

        # create module views
        module_interactions = [all_interactions[:, module_id:module_id+1] for module_id in range(len(module_names))]
        group_modules = defaultdict(list)
        for group_name, pattern in module_groups_patterns.items():
            for module_id, module_name in enumerate(module_names):
                if pattern.match(module_name):
                    group_modules[group_name].append(module_id)

        group_names = list(group_modules.keys())
        module_and_group_names.extend(group_names)
        for group_name in group_names:
            group_module_ids = torch.tensor(group_modules[group_name], device = device)
            module_interactions.append(all_interactions[:, group_module_ids])
            del group_module_ids

        if include_total:
            module_interactions.append(all_interactions)
            module_and_group_names.append('')
                
        scores = torch.zeros((len(agg_method_names), len(module_interactions), all_interactions.shape[0]), dtype=all_interactions.dtype, device = device)

        for agg_method_id, agg_method_name in enumerate(agg_method_names):
            for matrix_id, inf_matrix in enumerate(module_interactions):
                agg_method_fn = agg_method_fns[agg_method_name] 
                new_scores = agg_method_fn(inf_matrix, noise_mask = noise_mask, 
                                       trainset_labels = trainset_labels, inflset_labels = inflset_labels, 
                                       correct_infl_preds = correct_infl_preds, inflset_logits = inflset_logits, run_id = seed)
                scores[agg_method_id, matrix_id] = new_scores
                del new_scores
                torch.cuda.empty_cache()                 

        for agg_method_id, agg_method_name in enumerate(agg_method_names):
            for module_id, module_name in enumerate(module_and_group_names):
                scores_dict[(infl_method, agg_method_name, module_name)] = scores[agg_method_id, module_id]
        torch.cuda.empty_cache() 

    scores_path = os.path.join(cwd, f'{s_prefix}_{seed}.pt')
    torch.save(scores_dict, scores_path)

# def infl_noise(task = "qnli", infl_methods: str = 'datainf',
#                  s_prefix: str = 's_bl', m_prefix: str = 'm_bl',
#                  topk = 5, device = "cuda"):
#     ''' For a given dataset, infl methods finds topk infl scores and corrersponding pairs of infl-noisy-train vs validation sample.
#         It is done for each module in infl_matrix dict. 
#     '''
#     infl_methods = infl_methods.split(',')

#     # load dataset --> load noise tensor 
#     # load model and tokenizer  --> token compression
#     # load infl scores 
#     # from infl scores, find train sample ids that both belong to noise and has highest scores, top5 per method per layer
#     # find input_ids by obtained train sample ids
#     # transform them back to text - using additional mapper that compresses tokens

#     noise_list, trainset_labels, inflset_labels, input_ids = load_ds_info(cwd, task, with_input_ids=True)
#     # next 4 lines for testing
#     noise_list = noise_list[:100]
#     trainset_labels = trainset_labels[:100]
#     inflset_labels = inflset_labels[:100]
#     input_ids = input_ids[:100]

#     noise_mask = torch.tensor(noise_list, device = device)
#     trainset_labels = torch.tensor(trainset_labels, device = device)
#     noise_labels = trainset_labels[noise_mask]
#     noise_input_ids = [input_ids[i] for i in range(len(noise_mask)) if noise_mask[i]]
#     clean_labels = trainset_labels[~noise_mask]
#     clean_input_ids = [input_ids[i] for i in range(len(noise_mask)) if not noise_mask[i]]
#     inflset_labels = torch.tensor(inflset_labels, device = device)

#     scores_path = os.path.join(cwd, f'{s_prefix}_{seed}.pt')
#     scores_dict = torch.load(scores_path)

#     model_path = os.path.join(cwd, f'{m_prefix}_{task}_{seed}')

#     tokenizer = load_tokenizer(model_path)
#     # we need 'rank' as agg_method and module_name == '' (total) - total ranking of samples across network
#     method_ids = [(m, 'rank', '') for m in infl_methods]
#     for method_id in method_ids:
#         scores = scores_dict[method_id]
#         noise_scores = scores[noise_mask]
#         clean_scores = scores[~noise_mask]
#         noisy_train_idxs = torch.argsort(noise_scores)
#         noisy_topk_train_idxs = noisy_train_idxs[-topk:] 
#         infl_noise_input_ids = [noise_input_ids[i] for i in noisy_topk_train_idxs]
#         clean_train_idxs = torch.argsort(clean_scores)
#         clean_topk_train_idxs = clean_train_idxs[-topk:]
#         infl_clean_input_ids = [clean_input_ids[i] for i in clean_topk_train_idxs]
#         pass
#     pass

def auc_recall(task = "sentense", infl_methods: str = 'hf,cos,datainf,outlier,kr-ekfac',
                 i_prefix: str = 'i_ds', seeds: str = '0', 
                 s_prefix: str = 'metrics'):
    ''' For causal datasets compute these metrics for each layer and module, similar to layer_ndr '''
    infl_methods = infl_methods.split(',')
    seeds = [int(s) for s in seeds.split(',')]

    n_train, n_val = 900, 100
    n_sample_per_class = 90 
    n_class = 10

    def extract_qv(name: str) -> str:
        if ".q_proj" in name:
            return "q"
        elif ".v_proj" in name:
            return "v"
        raise ValueError(f"Unknown module name {name}")

    def extract_module(name: str) -> str: # either lora A or B
        if "lora_A" in name:
            return "A"
        elif "lora_B" in name:
            return "B"
        raise ValueError(f"Unknown module name {name}")
    
    def extract_layer(name:str) -> int: # numeric
        name_parts = name.split('.')
        loc_of_layer = name_parts.index("layers")
        if loc_of_layer > 0 and loc_of_layer + 1 < len(name_parts):
            layer = name_parts[loc_of_layer + 1]
            return int(layer)
        raise ValueError(f"Unknown layer {name}")

    metrics = []
    for seed in seeds:
        for method in infl_methods:

            infl_score_path = os.path.join(cwd, f"{i_prefix}_{method}_{task}_{seed}.pt")
            infl_scores = torch.load(infl_score_path)

            for module_name, module_scores in infl_scores.items():
                module_type = f"{extract_module(module_name)} {extract_qv(module_name)}"
                layer = extract_layer(module_name)
                aucs = []
                recalls = []
                for i in range(n_val):
                    gt_array=np.zeros(n_train)
                    gt_array[(i//n_class)*n_sample_per_class:((i//n_class)+1)*n_sample_per_class]=1
                    cur_scores = module_scores[i].float().cpu().numpy()
                    auc = roc_auc_score(gt_array, cur_scores)
                    aucs.append(auc)

                    sorted_ids = np.argsort(cur_scores)
                    pick_most_infl = sorted_ids[-n_sample_per_class:] // n_sample_per_class
                    correct_label = i // (n_val / n_class)
                    recall = np.count_nonzero(pick_most_infl == correct_label) / float(n_sample_per_class)
                    recalls.append(recall)

                auc_mean = np.mean(aucs)
                auc_std = np.std(aucs)

                recall_mean = np.mean(recalls)
                recall_std = np.std(recalls)

                row = {"method": method, "module": module_type, "layer": layer, "seed": seed, "auc_mean": auc_mean, "auc_std": auc_std, "recall_mean": recall_mean, "recall_std": recall_std, "full_module": module_name}
                metrics.append(row)

    metrics_df = pd.DataFrame(metrics)
    output_dir = os.path.join(cwd, f"{s_prefix}_{task}.csv")
    metrics_df.to_csv(output_dir)
    pass

def infl_noise(task = "qnli", infl_methods: str = 'datainf',
                 i_prefix: str = 'i_bl', m_prefix: str = 'm_bl',
                 topk = 5, device = "cuda"):
    ''' For a given dataset, infl methods finds topk infl scores and corrersponding pairs of infl-noisy-train vs validation sample.
        It is done for each module in infl_matrix dict. 
    '''
    infl_methods = infl_methods.split(',')
    topk = int(topk)

    noise_list, trainset_labels, inflset_labels, (train_input_ids, infl_input_ids) = load_ds_info(cwd, task, with_input_ids=True)
    
    # next lines for testing
    # noise_list = noise_list[:100]
    # trainset_labels = trainset_labels[:100]
    # inflset_labels = inflset_labels[:100]
    # train_input_ids = train_input_ids[:100]
    # infl_input_ids = infl_input_ids[:100]
    
    noise_mask = torch.tensor(noise_list, device = device)
    trainset_labels = torch.tensor(trainset_labels, device = device)
    noise_labels = trainset_labels[noise_mask]
    noise_input_ids = [train_input_ids[i] for i in range(len(noise_mask)) if noise_mask[i]]
    clean_labels = trainset_labels[~noise_mask]
    clean_input_ids = [train_input_ids[i] for i in range(len(noise_mask)) if not noise_mask[i]]
    inflset_labels = torch.tensor(inflset_labels, device = device)
    model_path = os.path.join(cwd, f'{m_prefix}_{task}_{seed}')
    tokenizer = load_tokenizer(model_path)
    inflset_logits, train_logits = load_m_info(cwd, task, m_prefix, with_train_logits=True)
    inflset_logits = inflset_logits.to(device)
    # train_logits = train_logits[:100]
    train_logits = train_logits.to(device)
    inflset_preds = torch.argmax(inflset_logits, dim = -1)   
    train_preds = torch.argmax(train_logits, dim = -1)

    noise_preds = train_preds[noise_mask]
    clean_preds = train_preds[~noise_mask]

    results = {}

    for infl_method in infl_methods:

        file_path = os.path.join(cwd, f'{i_prefix}_{infl_method}_{task}_{seed}.pt')
        matrix_dict = torch.load(file_path)        
        module_names = list(matrix_dict.keys())
        # transpose all matrices 
        for module_name in module_names:
            matrix_dict[module_name] = matrix_dict[module_name].t().to(device) # first dim is train_sample now and second is infl val sample
        if len(module_names) == 0:
            continue
        all_interactions = torch.stack([matrix_dict[module_name] for module_name in module_names], dim = 2)
        for int_matrix in matrix_dict.values():
            del int_matrix

        avg_scores_over_modules = all_interactions.mean(dim = 2)
        del all_interactions

        noise_avg_scores = avg_scores_over_modules[noise_mask]
        clean_avg_scores = avg_scores_over_modules[~noise_mask]
        del avg_scores_over_modules

        noise_flat_scores_1d = noise_avg_scores.view(-1)
        noise_sort_pair_ids = torch.argsort(noise_flat_scores_1d, descending=True) # from high to low score 
        noise_topk_pair_ids = noise_sort_pair_ids[:topk]
        noise_topk_train_ids = noise_topk_pair_ids // noise_avg_scores.shape[1]
        noise_topk_val_ids = noise_topk_pair_ids % noise_avg_scores.shape[1]
        noise_topk_scores = noise_flat_scores_1d[noise_topk_pair_ids]
        pass 

        clean_flat_scores_1d = clean_avg_scores.view(-1)
        clean_sort_pair_ids = torch.argsort(clean_flat_scores_1d, descending=True) # from high to low score 
        clean_topk_pair_ids = clean_sort_pair_ids[:topk]
        clean_topk_train_ids = clean_topk_pair_ids // clean_avg_scores.shape[1]
        clean_topk_val_ids = clean_topk_pair_ids % clean_avg_scores.shape[1]
        clean_topk_scores = clean_flat_scores_1d[clean_topk_pair_ids]
        pass 

        noise_topk_train_input_ids = [noise_input_ids[i] for i in noise_topk_train_ids]
        noise_top_val_input_ids = [infl_input_ids[i] for i in noise_topk_val_ids]
        noise_topk_train_strs = tokenizer.batch_decode(noise_topk_train_input_ids, skip_special_tokens=False)
        noise_topk_train_labels = [noise_labels[i] for i in noise_topk_train_ids]
        noise_topk_val_strs = tokenizer.batch_decode(noise_top_val_input_ids, skip_special_tokens=False)
        noise_topk_val_labels = [inflset_labels[i] for i in noise_topk_val_ids]
        noise_topk_train_preds = noise_preds[noise_topk_train_ids]
        noise_topk_val_preds = inflset_preds[noise_topk_val_ids]
        pass

        clean_topk_train_input_ids = [clean_input_ids[i] for i in clean_topk_train_ids]
        clean_top_val_input_ids = [infl_input_ids[i] for i in clean_topk_val_ids]
        clean_topk_train_strs = tokenizer.batch_decode(clean_topk_train_input_ids, skip_special_tokens=False)
        clean_topk_train_labels = [clean_labels[i] for i in clean_topk_train_ids]
        clean_topk_val_strs = tokenizer.batch_decode(clean_top_val_input_ids, skip_special_tokens=False)
        clean_topk_val_labels = [inflset_labels[i] for i in clean_topk_val_ids]
        clean_topk_train_preds = clean_preds[clean_topk_train_ids]
        clean_topk_val_preds = inflset_preds[clean_topk_val_ids]
        pass 

        all_data = [(noise_topk_train_strs, noise_topk_train_labels, noise_topk_val_strs, noise_topk_val_labels, noise_topk_train_preds, noise_topk_val_preds, noise_topk_scores), 
                    (clean_topk_train_strs, clean_topk_train_labels, clean_topk_val_strs, clean_topk_val_labels, clean_topk_train_preds, clean_topk_val_preds, clean_topk_scores)]
        all_pairs = []
        for one_data_i, one_data in enumerate(all_data):
            for train_str, train_label, val_str, val_label, train_pred, val_pred, score in zip(*one_data):
                all_pairs.append({ \
                    "train_str": train_str, 
                    "train_label": train_label.item(), 
                    "train_pred": train_pred.item(),
                    "val_str": val_str, 
                    "val_label": val_label.item(), 
                    "val_pred": val_pred.item(),
                    "infl_score": score.item(),
                    "is_noise": one_data_i == 0,
                })
        all_pairs.sort(key=lambda x: x["infl_score"], reverse=True)
        pass
        
        results[infl_method] = all_pairs
        
    # save results to file

    out_path = os.path.join(cwd, f'pairs_{task}_{seed}.json')
    with open(out_path, 'w') as file:
        json.dump(results, file)

    pass

def finetune2(task = 'qnli',
         infl_method='datainf', agg_method='', module_name = '',
         filter_perc = 0.3, s_prefix='s_bl', fix_labels = False,
         metrics_file = 'qnli-bl.jsonlist'):
    
    module_name = module_name.strip("\"")   
    config_path = os.path.join(cwd, f'c_{task}_{seed}.json')
    with open(config_path, 'r') as file:
        config = json.load(file)    
    scores_path = os.path.join(cwd, f'{s_prefix}_{seed}.pt')
    base_model_path = os.path.join(cwd, f'm_00_{task}_{seed}')
    best_model_path = os.path.join(cwd, f'm_2_b_{task}_{seed}')
    best_loss_model_path = os.path.join(cwd, f'm_2_bl_{task}_{seed}')
    last_model_path = os.path.join(cwd, f'm_2_l_{task}_{seed}')
    # unfreeze_regex = config.get('unfreeze_regex', None)
    finetune2_config=dict(task = task, seed = seed,
                                 infl_method=infl_method, agg_method=agg_method, module_name=module_name,
                                 filter_perc=filter_perc, source_scores_file=scores_path, target_lmodel_file=last_model_path,
                                 target_bmodel_file=best_model_path, best_loss_model_path = best_loss_model_path)
    print(f"Finetune with modules: {(infl_method, agg_method, module_name)}")
    num_epochs = config['num_epochs']
    device = config['device']
    lr = config['lr']
    # model = config['model']
    batch_size = config['batch_size']
    # target_modules = config['target_modules']
    # cancel_abs = []
    # cancel_norm = []
    scores_dict = torch.load(scores_path)
    scores = scores_dict[(infl_method, agg_method, module_name)]
    train_idxs = torch.argsort(scores)
    filter_len = int(filter_perc*len(train_idxs))
    train_idxs_left = train_idxs[filter_len:].cpu().numpy()
    noise_list = []
    def denoise_filter(train_dataset, train_idxs_left = train_idxs_left):        
        nonlocal noise_list
        noise_list = train_dataset['noise']
        return train_dataset.select(train_idxs_left)
    def fix_label_filter(train_dataset, train_idxs_left = train_idxs_left):        
        nonlocal noise_list
        noise_list = train_dataset['noise']
        return train_dataset.select(train_idxs_left)
    # if fix_labels:
    filter_fn = denoise_filter    
    dataset_path = os.path.join(cwd, f'd_{task}_{seed}')
    train_dataloader, eval_dataloader, infl_dataloader, tokenizer, _, _ = \
        build_loaders(dataset_path, config['tokenizer_name'], batch_size, filter_fn=filter_fn)

    # if os.path.exists(base_model_path):
    print(f"Loading {base_model_path}")
    # todo check if reequires_grad is attached to embedding
    lora_model = load_pretrained_LORA_model(model_name_or_path=base_model_path, unfreeze_modules_regex=None)

    # lora_model = build_LORA_model(model_name_or_path=model, pad_token_id=tokenizer.pad_token_id,
    #                             target_modules=target_modules, 
    #                             low_rank=config['low_rank'],
    #                             unfreeze_modules_regex=unfreeze_regex,
    #                             all_token_ids = all_token_ids, mapping_tensor = mapping_tensor)
            
    eval_metrics = train_LORA_model(lora_model, train_dataloader, eval_dataloader, infl_dataloader, device, num_epochs, lr,
                                    best_checkpoint_path=best_model_path,
                                    best_loss_model_path = best_loss_model_path,
                                    last_checkpoint_path=last_model_path)                                    

    del lora_model, train_dataloader, eval_dataloader

    tokenizer.save_pretrained(best_model_path)
    tokenizer.save_pretrained(best_loss_model_path)
    tokenizer.save_pretrained(last_model_path)

    metric_file = os.path.join(cwd, metrics_file)
    # old_gold_val_predictions = np.array(config['finetune']['gold_val_predictions'])
    # new_gold_val_predictions = np.array(eval_metrics['gold_val_predictions'])
    # logits_change = (new_gold_val_predictions - old_gold_val_predictions).mean()
    # eval_metrics['logits_change'] = logits_change

    # noise detection rate metrics
    noise_mask = torch.tensor(noise_list, device = train_idxs.device)
    num_noise = noise_mask.sum()
    noise_tensor = noise_mask[train_idxs]
    ideal_area = noise_mask.shape[0] - (num_noise / 2)

    del noise_mask

    noise_detection_curve = torch.cumsum(noise_tensor, dim = -1, dtype = torch.float)
    noise_detection_curve /= num_noise

    auc_ndr = noise_detection_curve.sum(dim = -1) 
    auc_ndr /= ideal_area

    filtered = noise_detection_curve[filter_len]

    eval_metrics["auc_ndr"] = auc_ndr.cpu().item()
    eval_metrics["filtered"] = filtered.cpu().item()
    eval_metrics["ndr_curve"] = noise_detection_curve.cpu().tolist()
    eval_metrics["first_finetune"] = config['finetune']

    # if len(cancel_abs) > 0:
    #     eval_metrics['cancel_abs'] = np.mean(cancel_abs)
    # if len(cancel_norm) > 0:
    #     eval_metrics['cancel_norm'] = np.mean(cancel_norm)
    # del eval_metrics['gold_val_predictions']
    with open(metric_file, "a") as f:                
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps({**eval_metrics, 'config': finetune2_config}) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)    

    with torch.no_grad():
        torch.cuda.empty_cache()

parser = argh.ArghParser()
parser.add_commands([preprocess, init_checkpoint, finetune, grads, infl, infl_matrix, scores, 
                     ndr, finetune2, infl_noise, set_logits, cancel_eff, combine_cancel,
                     infl_matrix_causal, kronfl, auc_recall])

def test_infl_vs_infl_matrix(file1: str, file2: str):
    infl = torch.load(file1)
    infl_matrices = torch.load(file2)

    common_dict_keys = set(infl.keys()).intersection(infl_matrices.keys())
    for key in common_dict_keys:
        one_infl = infl[key]
        one_infl_matrix = infl_matrices[key]
        one_infl2 = torch.mean(one_infl_matrix, dim=0).cpu()
        one_infl2.neg_()
        assert torch.allclose(one_infl, one_infl2, atol=1e-5), f"Failed for {key}"
        pass
    print("All tests passed")


if __name__ == '__main__':
    parser.dispatch()
    # test_infl_vs_infl_matrix("data/dev/i2_hf_qnli_0.pt", "data/dev/i3_hf_qnli_0.pt")


# import torch

# a = torch.zeros((1, 100), dtype=torch.float)
# b = torch.zeros((1, 100), dtype=torch.float)
# a[0, 0] = 1
# b[0, 0] = 1
# b[0, 1] = 1
# tmp_prods = torch.einsum('ik,jk->ij', a, b)
# res = tmp_prods / torch.norm(a, dim=-1) / torch.norm(b, dim=-1).view(-1, 1)
# print(res)


# a = torch.rand((10, 100), dtype=torch.float)
# b = torch.rand((11, 100), dtype=torch.float)

# prods = torch.einsum('ik,jk->ij', a, b)

# prods2 = torch.zeros((10, 11), dtype=torch.float)
# for i in range(10):
#     for j in range(11):
#         prods2[i, j] = torch.dot(a[i], b[j])

## Checked - it is true
# assert torch.allclose(prods, prods2)  



# a = torch.zeros(10, dtype=torch.int)

# i = torch.tensor([0, 1, 1, 1, 2, 2, 5])
# i_u, i_c = torch.unique(i, return_counts=True)

# a[i_u] += i_c
# n = 100
# a = torch.rand(n, dtype = torch.float)
# b = torch.rand(n, dtype = torch.float)
# ab = torch.outer(a, b)

# d = torch.rand(n, dtype = torch.float)
# l = torch.mm(d.view(1, -1), ab)
# e = torch.rand(n, dtype = torch.float)
# f1 = torch.dot(l.view(-1), e)

# l2 = torch.dot(d, a)
# l3 = torch.dot(b, e)
# l2 * l3

# torch.mm(d.view(1, -1), torch.eye(n, dtype = torch.float))